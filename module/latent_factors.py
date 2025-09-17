import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np


# ------------------------------------------------------------
# 1) 共享潜在因子模块：SharedLatentFactors
#    作用：x -> q(x) ∈ R^K （低维共享表征）
#    设计：小 MLP（可选隐藏层）+ LayerNorm（稳尺度）
# ------------------------------------------------------------
class SharedLatentFactors(nn.Module):
    def __init__(self, in_dim: int, k: int,
                 hidden: tuple[int, ...] = (),
                 act: str = "gelu",
                 use_layernorm: bool = True,
                 dropout: float = 0.0):
        super().__init__()
        acts = {
            "relu": nn.ReLU,
            "gelu": nn.GELU,
            "tanh": nn.Tanh,
            "silu": nn.SiLU
        }
        Act = acts.get(act.lower(), nn.GELU)

        layers: list[nn.Module] = []
        d_prev = in_dim
        for d in hidden:
            layers += [nn.Linear(d_prev, d, bias=True), Act()]
            if dropout > 0:
                layers += [nn.Dropout(dropout)]
            d_prev = d
        layers += [nn.Linear(d_prev, k, bias=True)]
        self.net = nn.Sequential(*layers)
        self.norm = nn.LayerNorm(k) if use_layernorm else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q = self.net(x)           # [N, K]
        q = self.norm(q)
        return q
