import math
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def make_quantile_knots(X: np.ndarray | torch.Tensor,
                        R: int = 6,
                        q_low: float = 0.05,
                        q_high: float = 0.95) -> torch.Tensor:
    """
    生成每个特征的分位数折点（升序），用于分段线性基函数。
    X: [N, D]
    return: knots [D, R] (torch.float32)
    """
    if isinstance(X, torch.Tensor):
        X = X.detach().cpu().numpy()
    N, D = X.shape
    qs = np.linspace(q_low, q_high, R)
    knots = np.stack([np.quantile(X[:, j], qs) for j in range(D)], axis=0).astype(np.float32)
    # 去重与微小扰动，避免折点相等导致零梯度
    for j in range(D):
        k = knots[j]
        for t in range(1, R):
            if k[t] <= k[t-1]:
                k[t] = np.nextafter(k[t-1], np.float32(np.inf))
        knots[j] = k
    return torch.from_numpy(knots)


class Shape1DMonotone(nn.Module):
    """
    单任务/多任务共享的 1D 分段线性形状函数（单调方向由 sign_prior 指定）。
    输入:
      knots      : [D, R]  —— 每个特征的折点(升序)
      sign_prior : [M, D]  —— {-1, 0, +1}；-1: 非增, +1: 非减, 0: 自由
    输出:
      forward(x) -> eta_shape: [N, M] —— 加性项（可直接加到 eta 上）
      smooth_penalty() -> 标量平滑正则
    """
    def __init__(self,
                 knots: torch.Tensor,
                 sign_prior: torch.Tensor,
                 init_scale: float = 1e-2,
                 smooth_lambda: float = 1e-3):
        super().__init__()
        self.register_buffer("knots", knots.float())          # [D,R]
        self.register_buffer("sign",  sign_prior.float())     # [M,D]
        D, R = self.knots.shape
        M, D2 = self.sign.shape
        assert D == D2, "sign_prior 与折点特征数不一致"
        self.D, self.R, self.M = D, R, M

        # 训练参数（未约束），通过 _delta() 映射到受约束的增量
        self.theta = nn.Parameter(torch.zeros(M, D, R))
        nn.init.normal_(self.theta, mean=0.0, std=init_scale)

        self.smooth_lambda = float(smooth_lambda)

    def _delta(self) -> torch.Tensor:
        """
        将 θ 映射为分段线性增量 δ：
          sign=+1 → +softplus(θ)  (非负)
          sign=-1 → -softplus(θ)  (非正)
          sign=0  → θ             (自由)
        返回: δ [M, D, R]
        """
        sp = F.softplus(self.theta)
        s  = self.sign.unsqueeze(-1)         # [M,D,1]
        abs_s = self.sign.abs().unsqueeze(-1)
        constrained = sp * s                 # 按符号约束
        delta = abs_s * constrained + (1.0 - abs_s) * self.theta
        return delta

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [N, D]
        return: eta_shape [N, M]
        """
        # 基函数: ReLU(x - t)
        basis = torch.relu(x.unsqueeze(-1) - self.knots.unsqueeze(0).to(x.device))  # [N,D,R]
        delta = self._delta().to(x.device)                                          # [M,D,R]
        eta_shape = torch.einsum('ndr,mdr->nm', basis, delta)                       # [N,M]
        return eta_shape

    def smooth_penalty(self) -> torch.Tensor:
        """
        二阶差分平滑: Σ_{m,j} Σ_r (δ_{r+1}-δ_r)^2
        """
        delta = self._delta()
        if self.R <= 1:
            return delta.new_tensor(0.0)
        diff = delta[..., 1:] - delta[..., :-1]   # [M,D,R-1]
        return self.smooth_lambda * (diff.pow(2).mean())


@torch.no_grad()
def eval_shape_curve(shape_module, j: int, m: int,
                     x_min: float, x_max: float,
                     num: int = 200, device: torch.device | None = None):
    """
    返回单特征 j、任务 m 的曲线 (xs, ys)。
    其它特征置 0（加性可分）。
    """
    device = device or next(shape_module.parameters()).device
    D = shape_module.D
    xs = torch.linspace(float(x_min), float(x_max), num, device=device)
    X = torch.zeros(num, D, device=device)
    X[:, j] = xs
    ys = shape_module(X)[:, m]     # [num]
    return xs.cpu().numpy(), ys.cpu().numpy()


@torch.no_grad()
def plot_shape_grid_by_feature(shape_module,
                               X_train,
                               feature_names: Sequence[str],
                               task_names: Sequence[str],
                               qrange: tuple[float, float] = (0.01, 0.99),
                               num: int = 200,
                               ncols: int = 3,
                               figsize_scale: float = 2.6,
                               markevery: int | None = None,
                               *,
                               savepath: str | None = None,
                               show: bool = False):
    """
    每个子图=一个特征；子图内画三个任务的形状曲线。
    美化：标记符号、细网格、统一y轴、紧凑图例、latex纵轴。
    """
    X_np = X_train.detach().cpu().numpy() if hasattr(X_train, "detach") else np.asarray(X_train)
    D, M = len(feature_names), len(task_names)
    assert (M, D) == tuple(shape_module.sign.shape), "shape_module.sign 维度与任务/特征数不一致"

    nrows = math.ceil(D / ncols)
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(figsize_scale * ncols, figsize_scale * nrows),
                             squeeze=False)

    # 样式：三条线的 marker / 线宽 / 透明度
    MARKERS = ('o', 's', '^')                   # 圆 / 方 / 三角
    LINEW   = 2.2
    ALPHA   = 0.95
    # 自动计算标记间隔，避免太密
    if markevery is None:
        markevery = max(1, num // 10)

    def arrow_for(sign_val: float) -> str:
        return "↑" if sign_val > 0 else ("↓" if sign_val < 0 else "-")

    for j in range(D):
        r, c = divmod(j, ncols)
        ax = axes[r, c]

        # x 轴范围：按分位数
        lo, hi = np.quantile(X_np[:, j], qrange)
        # 预计算三条曲线
        curves = []
        for m in range(M):
            xs, ys = eval_shape_curve(shape_module, j, m, lo, hi, num=num)
            curves.append((xs, ys, m))

        # 统一该子图 y 范围
        y_min = min(ys.min() for _, ys, _ in curves)
        y_max = max(ys.max() for _, ys, _ in curves)
        pad = 0.06 * (y_max - y_min + 1e-12)
        ax.set_ylim(y_min - pad, y_max + pad)

        # 画三条线（默认颜色循环），加不同的 marker
        for idx, (xs, ys, m) in enumerate(curves):
            s = float(shape_module.sign[m, j].item())
            label = f"{task_names[m]} {arrow_for(s)}"
            ax.plot(xs, ys,
                    linestyle='-',
                    linewidth=LINEW,
                    marker=MARKERS[idx % len(MARKERS)],
                    markersize=5.5,
                    markevery=markevery,
                    alpha=ALPHA,
                    label=label)

        # 美化：网格、零线
        ax.grid(True, which='major', linestyle='--', linewidth=0.8, alpha=0.25)
        ax.axhline(0.0, color='0.5', linewidth=0.8, alpha=0.35)

        # 题注与坐标轴
        # ax.set_title(f"{feature_names[j]}", fontsize=11)
        # if r == nrows - 1:
        ax.set_xlabel(feature_names[j])
        if c == 0:
            ax.set_ylabel(r"$\eta_{\mathrm{add}}$")

        # 紧凑图例
        ax.legend(fontsize=10, frameon=False, loc='best', handlelength=2.5, handletextpad=0.6)

    # 清理空格子
    for k in range(D, nrows * ncols):
        r, c = divmod(k, ncols)
        axes[r, c].axis("off")

    plt.tight_layout()
    if savepath is not None:
        plt.savefig(savepath, dpi=200, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)

