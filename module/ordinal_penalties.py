# ordinal_penalties.py
import torch
import torch.nn.functional as F

@torch.no_grad()
def _check_shapes(eta_k, tau, y):
    N, M, K = eta_k.shape          # K = C-1
    assert tau.shape == (M, K)
    assert y.shape[:2] == (N, M)


def ordinal_margin_penalty(eta_k: torch.Tensor,
                           tau: torch.Tensor,
                           y: torch.Tensor,
                           delta: float = 0.25,
                           reduction: str = "mean") -> torch.Tensor:
    """
    有序“间隔”正则：鼓励样本在其真实区间内留出 margin。
    对于标签 c：
      c=0      :  τ1 - η_{k=0}         ≥ δ
      1..C-2   :  η_{k=c-1} - τ_c ≥ δ  且  τ_{c+1} - η_{k=c} ≥ δ
      c=C-1    :  η_{k=C-2} - τ_{C-1}  ≥ δ
    其中 η_{k} 与 τ_k 的下标 k=0..K-1 对应边界 τ_{k+1}.
    """
    device = eta_k.device
    N, M, K = eta_k.shape          # K = C-1
    C = K + 1
    assert tau.shape == (M, K), f"tau shape {tau.shape} != (M,K)={(M,K)}"
    y = y.to(device).long().clamp(0, C - 1)

    tau_exp = tau.unsqueeze(0).expand(N, -1, -1)          # [N,M,K]

    # 下边界（y>0 时存在）：k_low = max(y-1, 0) —— 即使 y=0 也不会越界
    mask_low = (y > 0)
    k_low = (y - 1).clamp_min(0)                          # [N,M] ∈ [0, K-1]
    eta_low = eta_k.gather(2, k_low.unsqueeze(-1)).squeeze(-1)   # [N,M]
    tau_low = tau_exp.gather(2, k_low.unsqueeze(-1)).squeeze(-1) # [N,M]
    dist_low = eta_low - tau_low                                # 期待 ≥ δ
    pen_low = F.relu(delta - dist_low) * mask_low.float()

    # 上边界（y<C-1 时存在）：k_up = min(y, K-1) —— 对 y=C-1 会被 clamp 到 K-1，随后用 mask_up 屏蔽
    mask_up = (y < C - 1)
    k_up = torch.clamp(y, max=K - 1)                      # [N,M] ∈ [0, K-1]
    eta_up = eta_k.gather(2, k_up.unsqueeze(-1)).squeeze(-1)
    tau_up = tau_exp.gather(2, k_up.unsqueeze(-1)).squeeze(-1)
    dist_up = tau_up - eta_up                              # 期待 ≥ δ
    pen_up = F.relu(delta - dist_up) * mask_up.float()

    # 对存在的边界取平均
    denom = (mask_low.float() + mask_up.float()).clamp_min(1.0)
    pen = (pen_low + pen_up) / denom

    if reduction == "mean":
        return pen.mean()
    elif reduction == "sum":
        return pen.sum()
    else:
        return pen  # [N,M]


def tau_spacing_penalty(tau: torch.Tensor,
                        target: float | None = None,
                        reduction: str = "mean") -> torch.Tensor:
    """
    阈值间距平稳化：让相邻 τ 的间距更均匀（抑制塌缩/过密）。
      若 target=None：最小化各任务内间距的方差 Var(d_k)。
      若 target=常数：最小化 (d_k - target)^2 的均值。
    """
    # tau: [M,K]
    d = tau[:, 1:] - tau[:, :-1]                      # [M,K-1]
    if target is None:
        d_centered = d - d.mean(dim=1, keepdim=True)
        pen = (d_centered ** 2).mean()
    else:
        pen = ((d - float(target)) ** 2).mean()
    if reduction == "sum":
        return pen.sum() if pen.dim() > 0 else pen
    return pen

def ppo_consistency_penalty(eta_k: torch.Tensor,
                            reduction: str = "mean") -> torch.Tensor:
    """
    仅当 PPO（partial=True）时使用：鼓励各阈值的 η_k 不要相差太大，接近平行斜率。
    """
    N, M, K = eta_k.shape
    if K <= 1:
        return eta_k.new_tensor(0.0)
    eta_mean = eta_k.mean(dim=2, keepdim=True)
    pen = ((eta_k - eta_mean) ** 2).mean()
    if reduction == "sum":
        return pen.sum() if pen.dim() > 0 else pen
    return pen
