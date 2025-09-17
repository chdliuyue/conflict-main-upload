import argparse
import json
import os
import random
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch import amp
from torch.utils.data import DataLoader, Dataset, Subset, random_split

import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from module.latent_factors import SharedLatentFactors
from module.metrics import evaluate_multitask_predictions, format_results_table
from module.mtop_probit import (
    MOCE_Loss,
    MultiTaskOrderedProbitHead,
    class_balanced_weights_from_labels,
)
from module.ordinal_penalties import ordinal_margin_penalty, ppo_consistency_penalty, tau_spacing_penalty
from module.prior_coef import apply_direction_init_to_head, make_prior_from_coef_orderonly
from module.shape_1d import Shape1DMonotone, make_quantile_knots, plot_shape_grid_by_feature


TASK_NAMES = ["TTC", "DRAC", "PSD"]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class CustomDataset(Dataset):
    """Light-weight dataset that keeps the entire table in memory."""

    def __init__(self, data_path: str | os.PathLike[str]):
        data = pd.read_csv(data_path)
        self.X = torch.tensor(data.iloc[:, :-3].values, dtype=torch.float32)
        self.y = torch.tensor(data.iloc[:, -3:].values, dtype=torch.long)

    def __len__(self) -> int:  # type: ignore[override]
        return self.X.shape[0]

    def __getitem__(self, idx: int):  # type: ignore[override]
        return self.X[idx], self.y[idx]


class MT_MAON(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        task_names: Sequence[str],
        coef_matrix: pd.DataFrame,
        *,
        shared_k: int = 8,
        shared_hidden: Sequence[int] = (64,),
        shared_act: str = "gelu",
        shared_dropout: float = 0.0,
        cat_original: bool = True,
        use_shape: bool = False,
        knots: Optional[torch.Tensor] = None,
        sign_prior: Optional[torch.Tensor] = None,
        smooth_lambda: float = 1e-3,
        head_mode: str = "POM",
    ) -> None:
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.task_names = list(task_names)
        self.coef_matrix = coef_matrix

        self.shared = SharedLatentFactors(
            in_dim=self.input_dim,
            k=shared_k,
            hidden=tuple(shared_hidden),
            act=shared_act,
            use_layernorm=True,
            dropout=shared_dropout,
        )

        head_in_dim = self.input_dim + shared_k if cat_original else shared_k
        self.head = MultiTaskOrderedProbitHead(
            head_in_dim, len(self.task_names), self.output_dim, mode=head_mode
        )
        self.cat_original = cat_original
        self.use_shape = bool(use_shape)

        if self.use_shape:
            if knots is None or sign_prior is None:
                raise ValueError("use_shape=True 时需提供 knots 与 sign_prior")
            self.shape1d = Shape1DMonotone(
                knots=knots,
                sign_prior=sign_prior,
                init_scale=1e-2,
                smooth_lambda=smooth_lambda,
            )
        self._last_reg: Optional[torch.Tensor] = None
        self._last_eta_k: Optional[torch.Tensor] = None
        self._last_tau: Optional[torch.Tensor] = None

    def regularization(self) -> torch.Tensor:
        if self._last_reg is None:
            return torch.tensor(0.0, device=next(self.parameters()).device)
        return self._last_reg

    def penalties_from_latents(
        self,
        y: torch.Tensor,
        *,
        delta_margin: float = 0.25,
        lam_spacing: float = 1e-3,
        lam_ppo: float = 0.0,
    ) -> torch.Tensor:
        if self._last_eta_k is None or self._last_tau is None:
            raise RuntimeError("penalties_from_latents() must follow a forward pass")

        eta_k, tau = self._last_eta_k, self._last_tau
        penalty = ordinal_margin_penalty(eta_k, tau, y, delta=delta_margin)
        penalty = penalty + lam_spacing * tau_spacing_penalty(tau)
        if self.head.mode == "PPO" and lam_ppo > 0.0:
            penalty = penalty + lam_ppo * ppo_consistency_penalty(eta_k)
        return penalty

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        q = self.shared(x)
        h = torch.cat([x, q], dim=-1) if self.cat_original else q

        eta_add = self.shape1d(x) if self.use_shape else None
        probs, eta_k, tau = self.head(h, eta_add=eta_add)

        self._last_eta_k = eta_k
        self._last_tau = tau
        if self.use_shape:
            self._last_reg = self.shape1d.smooth_penalty()
        else:
            self._last_reg = torch.tensor(0.0, device=x.device)
        return probs


def _loss_components(
    model: MT_MAON,
    criterion: MOCE_Loss,
    outputs: torch.Tensor,
    targets: torch.Tensor,
    *,
    delta_margin: float,
    lam_spacing: float,
    lam_ppo: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    loss_main = criterion(outputs, targets)
    loss_shape = model.regularization()
    loss_latent = model.penalties_from_latents(
        targets, delta_margin=delta_margin, lam_spacing=lam_spacing, lam_ppo=lam_ppo
    )
    total = loss_main + loss_shape + loss_latent
    return total, loss_main, loss_shape, loss_latent


def evaluate_objective(
    model: MT_MAON,
    criterion: MOCE_Loss,
    data_loader: DataLoader,
    *,
    device: torch.device,
    use_amp: bool,
    delta_margin: float,
    lam_spacing: float,
    lam_ppo: float,
) -> float:
    model.eval()
    total = 0.0
    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            with amp.autocast(device_type=device.type, enabled=use_amp):
                outputs = model(inputs)
                loss, _, _, _ = _loss_components(
                    model,
                    criterion,
                    outputs,
                    targets,
                    delta_margin=delta_margin,
                    lam_spacing=lam_spacing,
                    lam_ppo=lam_ppo,
                )
            total += loss.item()
    return total / max(1, len(data_loader))


def train_level_2_model(
    model: MT_MAON,
    criterion: MOCE_Loss,
    optimizer: optim.Optimizer,
    train_loader: DataLoader,
    *,
    device: torch.device,
    epochs: int,
    grad_clip: Optional[float],
    use_amp: bool,
    delta_margin: float,
    lam_spacing: float,
    lam_ppo: float,
    val_loader: Optional[DataLoader] = None,
    scheduler: Optional[optim.lr_scheduler._LRScheduler] = None,
    patience: Optional[int] = None,
) -> MT_MAON:
    scaler: amp.GradScaler | None = None
    if use_amp:
        scaler = amp.GradScaler(device_type=device.type)
    best_state: Optional[dict[str, torch.Tensor]] = None
    best_val = float("inf")
    epochs_without_improvement = 0

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad(set_to_none=True)

            with amp.autocast(device_type=device.type, enabled=use_amp):
                outputs = model(inputs)
                loss, loss_main, loss_shape, loss_latent = _loss_components(
                    model,
                    criterion,
                    outputs,
                    targets,
                    delta_margin=delta_margin,
                    lam_spacing=lam_spacing,
                    lam_ppo=lam_ppo,
                )

            if use_amp and scaler is not None:
                scaler.scale(loss).backward()
                if grad_clip is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

            running_loss += loss.item()

        train_loss = running_loss / max(1, len(train_loader))

        val_loss = None
        if val_loader is not None:
            val_loss = evaluate_objective(
                model,
                criterion,
                val_loader,
                device=device,
                use_amp=use_amp,
                delta_margin=delta_margin,
                lam_spacing=lam_spacing,
                lam_ppo=lam_ppo,
            )
            if scheduler is not None:
                scheduler.step(val_loss)
        elif scheduler is not None:
            scheduler.step()

        current_lr = None
        if scheduler is not None:
            current_lr = scheduler.get_last_lr()[0]

        msg = f"Epoch {epoch}/{epochs} | train_loss={train_loss:.4f}"
        if val_loss is not None:
            msg += f" | val_loss={val_loss:.4f}"
        if current_lr is not None:
            msg += f" | lr={current_lr:.6g}"
        print(msg)

        if val_loss is not None:
            if val_loss < best_val - 1e-6:
                best_val = val_loss
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if patience is not None and epochs_without_improvement >= patience:
                    print(f"Early stopping triggered after {epoch} epochs without improvement.")
                    break
        else:
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def evaluate_model(
    model: MT_MAON,
    data_loader: DataLoader,
    *,
    device: torch.device,
    task_names: Sequence[str],
    ece_bins: int,
    use_amp: bool,
):
    model.eval()
    all_targets: list[np.ndarray] = []
    all_probas: list[np.ndarray] = []
    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            with amp.autocast(device_type=device.type, enabled=use_amp):
                outputs = model(inputs)
            all_targets.append(targets.cpu().numpy())
            all_probas.append(outputs.cpu().numpy())

    y_true = np.concatenate(all_targets, axis=0)
    probas = np.concatenate(all_probas, axis=0)
    metrics = evaluate_multitask_predictions(y_true, probas, task_names, ece_bins=ece_bins)
    print(format_results_table(metrics))
    return metrics, probas


def _labels_from_subset(subset: Dataset | Subset) -> torch.Tensor:
    if isinstance(subset, Subset):
        base = subset.dataset
        if not hasattr(base, "y"):
            raise AttributeError("Subset dataset must expose attribute 'y'")
        return base.y[subset.indices]
    if not hasattr(subset, "y"):
        raise AttributeError("Dataset must expose attribute 'y'")
    return subset.y  # type: ignore[return-value]


def build_dataloaders(
    dataset: CustomDataset,
    *,
    batch_size: int,
    val_split: float,
    seed: int,
    num_workers: int,
    pin_memory: bool,
):
    if val_split <= 0.0 or len(dataset) < 2:
        train_subset = dataset
        val_subset = None
    else:
        val_size = max(1, int(round(len(dataset) * val_split)))
        val_size = min(val_size, len(dataset) - 1)
        train_size = len(dataset) - val_size
        train_subset, val_subset = random_split(
            dataset, [train_size, val_size], generator=torch.Generator().manual_seed(seed)
        )

    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

    val_loader = None
    if val_subset is not None:
        val_loader = DataLoader(
            val_subset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

    return train_subset, train_loader, val_subset, val_loader


def save_metrics(metrics, output_path: Path) -> None:
    rows = [
        {
            "task": m.task,
            "accuracy": m.accuracy,
            "f1_score": m.f1_score,
            "qwk": m.qwk,
            "ordmae": m.ordmae,
            "nll": m.nll,
            "brier": m.brier,
            "auroc": m.auroc,
            "brdece": m.brdece,
        }
        for m in metrics
    ]
    output_path.write_text(json.dumps(rows, indent=2, ensure_ascii=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    ratio_name = "highD_ratio_20"
    parser.add_argument("--train", default=f"../data/{ratio_name}/train.csv")
    parser.add_argument("--test", default=f"../data/{ratio_name}/test.csv")
    parser.add_argument("--out_dir", default=f"../output/{ratio_name}/results_level_2")
    parser.add_argument(
        "--coef_file",
        default=f"../output/{ratio_name}/results_level_0/level_0_coefficients.csv",
        help="Path to Level 0 coefficients file",
    )
    parser.add_argument("--epochs", type=int, default=80, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--val_split", type=float, default=0.15, help="Validation split ratio")
    parser.add_argument("--patience", type=int, default=20, help="Early stopping patience")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="Gradient clipping value")
    parser.add_argument("--amp", action="store_true", help="Enable mixed precision training")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--pin_memory", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ece_bins", type=int, default=15)
    parser.add_argument("--margin_delta", type=float, default=0.25)
    parser.add_argument("--tau_spacing", type=float, default=1e-3)
    parser.add_argument("--ppo_consistency", type=float, default=0.0)
    parser.add_argument("--class_weight_beta", type=float, default=0.999)
    parser.add_argument("--smooth_lambda", type=float, default=1e-3)
    parser.add_argument("--no_shape", action="store_true", help="Disable shape functions")
    parser.add_argument("--plot_shapes", action="store_true")
    parser.add_argument("--shape_plot_path", default=None)
    parser.add_argument("--save_model", action="store_true")
    parser.add_argument("--model_path", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    set_seed(args.seed)

    device = torch.device(args.device)
    use_amp = bool(args.amp and device.type == "cuda")
    if args.amp and not use_amp:
        print("已请求 AMP 但当前设备不支持，将自动禁用混合精度训练。")

    train_dataset = CustomDataset(args.train)
    test_dataset = CustomDataset(args.test)
    input_dim = train_dataset.X.shape[1]
    num_classes = train_dataset.y.max().item() + 1 if train_dataset.y.numel() > 0 else 4

    train_subset, train_loader, val_subset, val_loader = build_dataloaders(
        train_dataset,
        batch_size=args.batch_size,
        val_split=args.val_split,
        seed=args.seed,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
    )

    coef_df = pd.read_csv(args.coef_file, index_col="feature")
    coef_df = coef_df[TASK_NAMES][:input_dim]
    sign_prior, mag_prior = make_prior_from_coef_orderonly(coef_df, TASK_NAMES, input_dim)

    knots = None
    sign_prior_t = None
    if not args.no_shape:
        knots = make_quantile_knots(train_dataset.X, R=6).to(device)
        sign_prior_t = sign_prior.to(device)

    model = MT_MAON(
        input_dim=input_dim,
        output_dim=num_classes,
        task_names=TASK_NAMES,
        coef_matrix=coef_df,
        shared_k=8,
        shared_hidden=(64,),
        shared_act="gelu",
        shared_dropout=0.0,
        cat_original=True,
        use_shape=not args.no_shape,
        knots=knots,
        sign_prior=sign_prior_t,
        smooth_lambda=args.smooth_lambda,
        head_mode="POM",
    ).to(device)

    apply_direction_init_to_head(
        head=model.head,
        sign_prior=sign_prior,
        mag_prior=mag_prior,
        scale=0.10,
        orig_feat_dim=input_dim,
        orig_offset=0,
    )

    train_labels = _labels_from_subset(train_subset).to(device)
    w_mc = class_balanced_weights_from_labels(
        train_labels, num_classes=num_classes, beta=args.class_weight_beta
    )

    criterion = MOCE_Loss(
        reduction="mean",
        class_weight_mode="provided",
        class_weights=w_mc,
        learn_task_uncertainty=True,
        focal_gamma=None,
        learn_focal_gamma=False,
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = None
    if val_loader is not None:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5
        )

    model = train_level_2_model(
        model,
        criterion,
        optimizer,
        train_loader,
        device=device,
        epochs=args.epochs,
        grad_clip=args.grad_clip,
        use_amp=use_amp,
        delta_margin=args.margin_delta,
        lam_spacing=args.tau_spacing,
        lam_ppo=args.ppo_consistency,
        val_loader=val_loader,
        scheduler=scheduler,
        patience=args.patience,
    )

    if not args.no_shape and (args.plot_shapes or args.shape_plot_path):
        feature_names = [
            "UF",
            "UAS",
            "UD",
            "UAL",
            "DF",
            "DAS",
            "DD",
            "DAL",
            "rq_rel",
            "rk_rel",
            "CV_v",
            "E_BRK",
        ]
        plot_shape_grid_by_feature(
            model.shape1d,
            train_dataset.X,
            feature_names,
            TASK_NAMES,
            qrange=(0.01, 0.99),
            num=200,
            ncols=4,
            figsize_scale=2.6,
            savepath=args.shape_plot_path,
            show=args.plot_shapes,
        )

    metrics, probas = evaluate_model(
        model,
        test_loader,
        device=device,
        task_names=TASK_NAMES,
        ece_bins=args.ece_bins,
        use_amp=use_amp,
    )

    results_file = Path(args.out_dir) / "evaluation_results.txt"
    with results_file.open("w") as f:
        f.write("Task | Accuracy | F1 | QWK | OrdMAE | NLL | Brier | AUROC | BrdECE\n")
        for m in metrics:
            f.write(
                f"{m.task} | {m.accuracy:.4f} | {m.f1_score:.4f} | {m.qwk:.4f} | "
                f"{m.ordmae:.4f} | {m.nll:.4f} | {m.brier:.4f} | {m.auroc:.4f} | {m.brdece:.4f}\n"
            )

    metrics_json = Path(args.out_dir) / "evaluation_results.json"
    save_metrics(metrics, metrics_json)

    np.save(Path(args.out_dir) / "test_probabilities.npy", probas)

    if args.save_model:
        model_path = Path(args.model_path) if args.model_path else Path(args.out_dir) / "model.pt"
        torch.save(model.state_dict(), model_path)
        print(f"Model checkpoint saved to {model_path}")

    print(f"Evaluation results saved to {results_file}")


if __name__ == "__main__":
    main()