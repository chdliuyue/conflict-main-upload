import torch
import pandas as pd
import numpy as np


def make_prior_from_coef_orderonly(coef_df_or_ndarray,
                                   task_names: list[str],
                                   D: int):
    """
    coef_df_or_ndarray: 形状 [D, M] 的 DataFrame 或 ndarray
                        行是特征顺序(0..D-1)，列是任务（最好叫 TTC/DRAC/PSD）
    task_names: 任务名顺序（与模型一致）
    D: 原始特征数（你的就是 12）

    返回:
      sign_prior: [M, D] ∈ {-1,0,+1}
      mag_prior : [M, D] ∈ [0,1]
    """
    if isinstance(coef_df_or_ndarray, pd.DataFrame):
        # 如果列名就是任务名，按列名取；否则直接转成数组
        try:
            arr = coef_df_or_ndarray[task_names].to_numpy(dtype=float)
        except Exception:
            arr = coef_df_or_ndarray.to_numpy(dtype=float)
    else:
        arr = np.asarray(coef_df_or_ndarray, dtype=float)

    M = arr.shape[1]
    assert arr.shape == (D, M), f"期望形状为 ({D},{len(task_names)}), 实际 {arr.shape}"

    sign = np.sign(np.nan_to_num(arr, nan=0.0))         # NaN→0
    mag  = np.abs(np.nan_to_num(arr, nan=0.0))
    max_per_task = np.maximum(mag.max(axis=0, keepdims=True), 1e-8)
    mag_norm = mag / max_per_task                       # 每任务归一化到 [0,1]

    # 转成 [M,D]
    sign_prior = torch.from_numpy(sign.T).float()
    mag_prior  = torch.from_numpy(mag_norm.T).float()
    return sign_prior, mag_prior



@torch.no_grad()
def apply_direction_init_to_head(head,
                                 sign_prior: torch.Tensor,   # [M,D]
                                 mag_prior : torch.Tensor|None = None,
                                 scale: float = 0.10,
                                 orig_feat_dim: int = 12,
                                 orig_offset: int = 0):
    """
    仅初始化 head.gamma 中“对应原始特征”的那段权重：
      - POM: gamma 形状 [M, in_dim]
      - PPO: gamma 形状 [M, in_dim, C-1]
    其余（例如共享因子 q(x) 的权重）保持 0。
    """
    device = head.gamma.device
    sign_prior = sign_prior.to(device)
    mag_prior  = mag_prior.to(device) if mag_prior is not None else None

    M, D = sign_prior.shape
    assert D == orig_feat_dim, "orig_feat_dim 与先验特征数不一致"
    lo, hi = orig_offset, orig_offset + D

    if hasattr(head, "partial") and head.partial:  # PPO
        base = scale * sign_prior
        if mag_prior is not None:
            base = base * (0.5 + 0.5 * mag_prior)  # 小幅区分强弱
        base = base.unsqueeze(-1).expand(-1, -1, head.C - 1)  # [M,D,C-1]
        head.gamma.zero_()
        head.gamma[:, lo:hi, :].copy_(base)
    else:                                          # POM
        base = scale * sign_prior
        if mag_prior is not None:
            base = base * (0.5 + 0.5 * mag_prior)
        head.gamma.zero_()
        head.gamma[:, lo:hi].copy_(base)