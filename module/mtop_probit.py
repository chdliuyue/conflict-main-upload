# mtop_probit.py
# 深度学习版多任务有序 Probit（与传统模型同形），支持 POM / PPO。

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np


# ===== 标准正态 CDF：probit 链接 =====
def std_normal_cdf(z: torch.Tensor) -> torch.Tensor:
    # Φ(z) = 0.5 * (1 + erf(z / sqrt(2)))
    return 0.5 * (1.0 + torch.erf(z / math.sqrt(2.0)))


# ===== 从潜变量 + 阈值计算每一类的概率（与有序 Probit 同式）=====
def ordered_probit_probs(eta_k: torch.Tensor, tau: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """
    eta_k: [N, M, C-1]  —— 每个任务的“≤k”对应的线性预测子（POM 时为复制）
    tau  : [M, C-1]     —— 任务阈值（严格单调）
    return probs: [N, M, C]
    """
    # 累积概率 P(Y<=k) = Φ(τ_k - η_k)
    z = tau.unsqueeze(0) - eta_k                     # [N,M,C-1]
    cdf = std_normal_cdf(z).clamp(1e-7, 1-1e-7)      # 数值稳定

    # 区间差得到每一类概率
    N, M, Km1 = cdf.shape
    C = Km1 + 1
    probs = torch.empty((N, M, C), device=cdf.device, dtype=cdf.dtype)
    probs[..., 0]    = cdf[..., 0]
    if C > 2:
        probs[..., 1:-1] = cdf[..., 1:] - cdf[..., :-1]
    probs[..., -1]   = 1.0 - cdf[..., -1]

    probs = probs.clamp_min(eps)
    probs = probs / probs.sum(dim=-1, keepdim=True)  # 保险归一化
    return probs


# ===== 多任务有序 Probit 头（潜变量 + 单调阈值），POM/PPO 可切换 =====
class MultiTaskOrderedProbitHead(nn.Module):
    """
    参数
    ----
    in_dim : 骨干输出维度 H
    tasks  : 任务数 M
    classes: 类别数 C (>=3)
    mode   : 'POM' 或 'PPO'
    min_gap: 阈值之间的最小间隔，确保严格单调
    说明
    ----
    - POM: 对每个任务 m 仅学习一个 η_{i,m}，对所有阈值共享（平行斜率）。
    - PPO: 对每个任务、每个阈值 k 学习 η_{i,m,k}（部分比例优势）。
    - 阈值采用“递增重参数化”保证 τ_{m,1}<...<τ_{m,C-1}。
    前向
    ----
    输入 h: [N, H]  —— 共享骨干的嵌入
    返回 probs: [N,M,C], eta_k: [N,M,C-1], tau: [M,C-1]
    """
    def __init__(self, in_dim: int, tasks: int, classes: int,
                 mode: str = 'POM', min_gap: float = 1e-3):
        super().__init__()
        assert classes >= 3, "C 至少为 3。"
        assert mode in ('POM', 'PPO')
        self.H, self.M, self.C = int(in_dim), int(tasks), int(classes)
        self.mode = mode
        self.min_gap = min_gap

        if mode == 'PPO':
            self.gamma = nn.Parameter(torch.zeros(self.M, self.H, self.C - 1))
        else:
            self.gamma = nn.Parameter(torch.zeros(self.M, self.H))

        # 阈值原始参数；通过 softplus 累加实现严格单调
        init = torch.linspace(-1.0, 1.0, classes - 1).repeat(tasks, 1)
        self.cut_raw = nn.Parameter(init)  # [M, C-1]

        self.reset_parameters()

    def reset_parameters(self):
        # 轻微随机扰动，帮助打破对称；系数从近零开始
        if self.mode == 'PPO':
            nn.init.normal_(self.gamma, mean=0.0, std=0.01)
        else:
            nn.init.normal_(self.gamma, mean=0.0, std=0.01)
        # cut_raw 已按线性初始化，无需额外处理

    def _ordered_cuts(self) -> torch.Tensor:
        base = self.cut_raw[:, :1]                                # [M,1]
        deltas = F.softplus(self.cut_raw[:, 1:]) + self.min_gap   # [M,C-2] (>=min_gap)
        if deltas.numel() == 0:
            return base
        tail = base + torch.cumsum(deltas, dim=1)                 # 依次累加，确保严格递增
        return torch.cat([base, tail], dim=1)                     # [M,C-1]

    def forward(self, h: torch.Tensor, eta_add: torch.Tensor | None = None):
        if self.mode == 'PPO':
            # η_{i,m,k} = Σ_d h_{i,d} * γ_{m,d,k}
            eta_k = torch.einsum('nd,mdk->nmk', h, self.gamma)
        else:
            # η_{i,m}  = Σ_d h_{i,d} * γ_{m,d}，再复制到每个 k
            eta = torch.einsum('nd,md->nm', h, self.gamma)        # [N,M]
            eta_k = eta.unsqueeze(-1).expand(-1, -1, self.C - 1)  # [N,M,C-1]

        if eta_add is not None:
            eta_k = eta_k + eta_add.unsqueeze(-1)

        tau = self._ordered_cuts()                                # [M,C-1]
        probs = ordered_probit_probs(eta_k, tau)                  # [N,M,C]
        return probs, eta_k, tau


# ===== 负对数似然（MOCE）：与传统极大似然同形 =====
# moce_loss.py
import torch
import torch.nn as nn

class MOCE_Loss(nn.Module):
    """
    多任务有序交叉熵（MOCE）——扩展版
    - 仍然基于 p = Φ(τ-η) 的区间差（由模型头计算并传入），这里只做 NLL + 外层加权
    - 内置：类别不平衡加权、任务不确定性加权（可选）
    - 可选 Focal 因子（极端失衡时启用；默认关闭以保持 MLE 语义）
    """
    def __init__(self,
                 reduction: str = 'mean',
                 class_weight_mode: str = 'none',   # 'none' | 'effective' | 'inverse' | 'provided'
                 beta: float = 0.999,               # effective-number 的 β
                 class_weights: torch.Tensor | None = None,  # [M,C]，当 mode='provided' 时使用
                 learn_task_uncertainty: bool = False,       # 学习 s_m（log σ_m^2）
                 focal_gamma: float | None = None,  # None 或 浮点；也可后续 set_focal_gamma()
                 learn_focal_gamma: bool = False,   # 让 γ_m 可学习（谨慎使用）
                 eps: float = 1e-12):
        super().__init__()
        assert reduction in ('mean', 'sum', 'none')
        assert class_weight_mode in ('none', 'effective', 'inverse', 'provided')
        self.reduction = reduction
        self.class_weight_mode = class_weight_mode
        self.beta = beta
        self.eps = eps

        # 类别权重（缓冲/占位，shape 将在首个 forward 校正）
        if class_weights is not None:
            self.register_buffer("w_mc", class_weights.float())  # [M,C]
        else:
            self.w_mc = None

        # 任务不确定性 s_m
        if learn_task_uncertainty:
            # s_m 初值 0 → 等权
            self.s_m = nn.Parameter(torch.zeros(1))  # 先占位，首个 forward 再扩展到 [M]
            self._s_m_ready = False
        else:
            self.s_m = None
            self._s_m_ready = True

        # Focal 因子
        if learn_focal_gamma:
            self.gamma_m = nn.Parameter(torch.zeros(1))  # 先占位，首个 forward 再扩展到 [M]
            self._gamma_ready = False
            self._fixed_gamma = None
        else:
            self.gamma_m = None
            self._gamma_ready = True
            self._fixed_gamma = None if focal_gamma is None else float(focal_gamma)

    @staticmethod
    def _effective_number_weights(y: torch.Tensor, C: int, beta: float) -> torch.Tensor:
        """
        y: [N,M]；返回 [M,C]
        """
        N, M = y.shape
        w = torch.zeros(M, C, dtype=torch.float32, device=y.device)
        for m in range(M):
            counts = torch.bincount(y[:, m].clamp(0, C-1), minlength=C).float()
            numer = 1.0 - beta
            denom = 1.0 - torch.pow(beta, counts)
            denom = torch.where(counts > 0, denom, torch.ones_like(denom))
            w_m = numer / denom
            # 对空类置 0
            w_m = torch.where(counts > 0, w_m, torch.zeros_like(w_m))
            # 归一化到均值≈1（可选，便于与未加权的标度一致）
            if w_m.sum() > 0:
                w_m = w_m * (C / torch.clamp(w_m.sum(), min=1e-8))
            w[m] = w_m
        return w

    @staticmethod
    def _inverse_freq_weights(y: torch.Tensor, C: int, eps: float = 1.0) -> torch.Tensor:
        """
        简单逆频率权重（count+eps 的逆），返回 [M,C]
        """
        N, M = y.shape
        w = torch.zeros(M, C, dtype=torch.float32, device=y.device)
        for m in range(M):
            counts = torch.bincount(y[:, m].clamp(0, C-1), minlength=C).float()
            w_m = 1.0 / (counts + eps)
            if w_m.sum() > 0:
                w_m = w_m * (C / torch.clamp(w_m.sum(), min=1e-8))
            w[m] = w_m
        return w

    def set_class_weights(self, w_mc: torch.Tensor):
        """显式设置类别权重 [M,C]"""
        self.register_buffer("w_mc", w_mc.float())

    def set_focal_gamma(self, gamma: float | torch.Tensor):
        """设置固定的 focal γ（标量或 [M]），仅在 learn_focal_gamma=False 时生效"""
        if self.gamma_m is None:
            if isinstance(gamma, torch.Tensor):
                self._fixed_gamma = gamma.detach().float()
            else:
                self._fixed_gamma = float(gamma)
        # 若 learn_focal_gamma=True，则忽略此设置

    def forward(self, probs: torch.Tensor, y: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        probs: [N,M,C]
        y    : [N,M] （0..C-1）
        mask : [N,M] （缺失标签=0），可选
        """
        N, M, C = probs.shape
        device = probs.device
        y = y.long().clamp(0, C - 1)

        # === 基本 NLL ===
        p = probs.gather(-1, y.unsqueeze(-1)).squeeze(-1).clamp_min(self.eps)  # [N,M]
        loss_nm = -torch.log(p)                                                # [N,M]

        # === Focal 因子（可选，默认关闭）===
        if self.gamma_m is not None and not self._gamma_ready:
            # 首次 forward 时把形状扩展到 [M]
            self.gamma_m.data = self.gamma_m.data.expand(M).contiguous()
            self._gamma_ready = True

        if self.gamma_m is not None:
            # learnable γ_m ≥ 0：用 softplus 保证非负
            gamma_m = torch.nn.functional.softplus(self.gamma_m)              # [M]
            focal = torch.pow(1.0 - p, gamma_m.unsqueeze(0))                  # [N,M]
            loss_nm = focal * loss_nm
        elif self._fixed_gamma is not None:
            if isinstance(self._fixed_gamma, torch.Tensor):
                gamma_m = self._fixed_gamma.to(device).view(1, -1)            # [1,M]
            else:
                gamma_m = torch.tensor(self._fixed_gamma, device=device).view(1, 1)
            loss_nm = torch.pow(1.0 - p, gamma_m) * loss_nm

        # === 类别不平衡权重 ===
        if self.class_weight_mode != 'none':
            if (self.w_mc is None) or (self.w_mc.shape != (M, C)) or (self.w_mc.device != device):
                # 首个 batch（或设备/形状变化）时按当前 y 计算
                if self.class_weight_mode == 'effective':
                    w_mc = self._effective_number_weights(y, C, self.beta)    # [M,C]
                elif self.class_weight_mode == 'inverse':
                    w_mc = self._inverse_freq_weights(y, C, eps=1.0)          # [M,C]
                elif self.class_weight_mode == 'provided':
                    raise ValueError("class_weights 未提供或形状不匹配")
                self.set_class_weights(w_mc.to(device))

            w = self.w_mc.unsqueeze(0).expand(N, -1, -1)                       # [N,M,C]
            w_nm = w.gather(2, y.unsqueeze(-1)).squeeze(-1)                    # [N,M]
            loss_nm = loss_nm * w_nm

        # === 缺失标签掩码（可选）===
        if mask is not None:
            loss_nm = loss_nm * mask
            denom_per_task = torch.clamp(mask.sum(dim=0).float(), min=1.0)     # [M]
        else:
            denom_per_task = torch.full((M,), float(N), device=device)

        per_task = loss_nm.sum(dim=0) / denom_per_task                          # [M]

        # === 任务不确定性加权 ===
        if self.s_m is not None:
            if not self._s_m_ready:
                self.s_m.data = self.s_m.data.expand(M).contiguous()
                self._s_m_ready = True
            weights = torch.exp(-self.s_m)                                      # e^{-s_m}，[M]
            total = (weights * per_task).sum() + self.s_m.sum()
        else:
            total = per_task.mean()

        if self.reduction == 'none':
            # 返回逐任务损失（外面若要更复杂的聚合/监控）
            return per_task
        elif self.reduction == 'sum':
            return total if self.s_m is not None else per_task.sum()
        else:  # 'mean'
            return total


# ====== 类别权重（Class-Balanced / Effective Number）======
import torch

def class_balanced_weights_from_labels(y: torch.Tensor, num_classes: int, beta: float = 0.999) -> torch.Tensor:
    """
    y: [N, M] (long, 取值 0..C-1)
    返回: w_mc [M, C]，每任务每类的权重；对空类权重=0，并做均值≈1的归一化
    """
    assert y.dim() == 2
    N, M = y.shape
    w = torch.zeros(M, num_classes, dtype=torch.float32, device=y.device)
    for m in range(M):
        counts = torch.bincount(y[:, m].clamp(0, num_classes-1), minlength=num_classes).float()
        numer = 1.0 - beta
        denom = 1.0 - torch.pow(beta, counts)
        denom = torch.where(counts > 0, denom, torch.ones_like(denom))
        w_m = numer / denom
        w_m = torch.where(counts > 0, w_m, torch.zeros_like(w_m))  # 空类置 0
        if w_m.sum() > 0:
            w_m = w_m * (num_classes / torch.clamp(w_m.sum(), min=1e-8))  # 归一化到均值≈1
        w[m] = w_m
    return w  # [M, C]

# （可选）简单逆频率权重
def inverse_freq_weights_from_labels(y: torch.Tensor, num_classes: int, eps: float = 1.0) -> torch.Tensor:
    N, M = y.shape
    w = torch.zeros(M, num_classes, dtype=torch.float32, device=y.device)
    for m in range(M):
        counts = torch.bincount(y[:, m].clamp(0, num_classes-1), minlength=num_classes).float()
        w_m = 1.0 / (counts + eps)
        if w_m.sum() > 0:
            w_m = w_m * (num_classes / torch.clamp(w_m.sum(), min=1e-8))
        w[m] = w_m
    return w



