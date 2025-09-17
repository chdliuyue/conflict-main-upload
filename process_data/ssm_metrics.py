# ssm_metrics.py
# -*- coding: utf-8 -*-
"""
Surrogate Safety Measures (TTC / DRAC / PSD)
— 统一生效条件 + 统一口径 + 节点时间聚合（含 1Hz 合桶）
— TTC 线性口径 + 反应时间距离修正（min{D/dv, (D - v_f*TAU_TTC)/dv}）
— DRAC 含反应时间补偿，PSD 节点同口径

分类规则（基于节点）：
  TTC_node  = 节点内 min( TTC_base, TTC_L, TTC_R )
  DRAC_node = 节点内 max( DRAC_base, DRAC_L, DRAC_R ) 〔各自有效帧〕
  PSD_node  = 节点内 min( PSD_base ) 〔仅接近帧〕
节点值 → 权重(0/1/2/3) → 节点平均 → 四分类；TTC/DRAC输出暴露秒（1/2/3档）。
"""

from __future__ import annotations
from typing import Dict, List, Optional
import numpy as np
import pandas as pd

__all__ = [
    "percentile", "resolve_col",
    "compute_frame_ssm_base", "compute_frame_ssm_union",
    "compute_window_base_quantiles", "compute_nodewise_labels",
    "ttc_weight_from_value", "drac_weight_from_value", "psd_weight_from_value",
    "cls_from_avg_weight", "cls4_from_psd",
]

# ============================ 常量/旋钮 ============================ #
EPS   = 1e-12
EPS_D = 1e-3
EPS_V = 1e-3
MIN_VALID_FRAMES_DEFAULT = 10

# 几何距离“保守收缩”：越大越敏感（拉低TTC、抬高DRAC）
DIST_MARGIN_M   = 1.2
LEN_MARGIN_FRAC = 0.20

# 反应时间补偿
TAU_TTC = 0.5  # 仅用于 TTC 的距离修正 D_ttc = max(D - v_f*TAU_TTC, EPS_D) 值越大高类别越多
TAU_S   = 0.6  # DRAC 的距离修正 D_drac = max(D - v_f*TAU_S, EPS_D) 值越大高类别越多

# PSD 四分类阈
PSD_THRS = (0.60, 0.80, 1.00)

# 节点合桶目标频率（把 node_hz 的节点先合成约 1 Hz 再做权重平均）
NODE_BUCKET_HZ_TARGET = 1.0


# ============================ 小工具 ============================ #
def percentile(s: pd.Series, q_percent: float) -> float:
    s = pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    return float(np.percentile(s, q_percent)) if len(s) else np.nan

def resolve_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    m = {c.lower(): c for c in df.columns}
    for name in candidates:
        c = m.get(name.lower())
        if c is not None:
            return c
    return None


# ============================ 帧级：基础 ============================ #
def compute_frame_ssm_base(df: pd.DataFrame, mu: float, grav: float) -> pd.DataFrame:
    """
    生成帧级 TTC/DRAC/PSD_allen（基于 dhw）；最终分类用的接近掩码与几何距离在 union 中统一。
    依赖列：ttc(可选)、dhw/spaceHeadway、xVelocity_raw、precedingXVelocity_raw、veh_len。
    """
    out = df.copy()

    # 原始 TTC（兜底）
    col_ttc = resolve_col(out, ["ttc"])
    if col_ttc and col_ttc in out.columns:
        ttc_raw = pd.to_numeric(out[col_ttc], errors="coerce").where(lambda s: s > 0.0, np.nan)
    else:
        ttc_raw = pd.Series(np.nan, index=out.index, dtype=float)

    # 数据自带 DHW（保险杠距）
    col_dhw = resolve_col(out, ["dhw", "spaceHeadway", "space_gap"])
    if col_dhw and col_dhw in out.columns:
        D_data = pd.to_numeric(out[col_dhw], errors="coerce").where(lambda s: s > 0.0, np.nan)
    else:
        D_data = pd.Series(np.nan, index=out.index, dtype=float)
    out["DHW"] = D_data

    # 相对速度（后-前）
    vf = out.get("xVelocity_raw", pd.Series(np.nan, index=out.index)).astype(float)
    vl = out.get("precedingXVelocity_raw", pd.Series(np.nan, index=out.index)).astype(float)
    dv = vf - vl

    # 线性 TTC（仅接近；基于 D_data）
    TTC_lin = pd.Series(np.nan, index=out.index, dtype=float)
    m_lin = D_data.notna() & (dv > EPS_V)
    TTC_lin.loc[m_lin] = D_data[m_lin] / (dv[m_lin] + EPS)

    # 帧级 TTC：原始与线性取更小（分类时会被几何口径覆盖）
    out["TTC"] = pd.concat([ttc_raw, TTC_lin], axis=1).min(axis=1, skipna=True)

    # DRAC（基于 D_data；分类时会被几何口径覆盖）
    DRAC_base = pd.Series(np.nan, index=out.index, dtype=float)
    m_dr = D_data.notna() & (dv > EPS_V)
    DRAC_base.loc[m_dr] = (dv[m_dr] ** 2) / (2.0 * D_data[m_dr] + EPS)
    out["DRAC"] = DRAC_base
    out["DRAC_valid_mask"] = m_dr.astype(int)

    # PSD_allen（p95参考，不纳入分类）
    vf_abs = vf.abs()
    PSD = pd.Series(np.nan, index=out.index, dtype=float)
    m_psd = D_data.notna() & (vf_abs > 0.0)
    denom = (vf_abs ** 2) / (2.0 * mu * grav + EPS)
    PSD.loc[m_psd] = D_data[m_psd] / (denom[m_psd] + EPS)
    out["PSD_allen"] = PSD.clip(lower=0.0)
    return out


# ============================ 帧级：几何+左右候选 ============================ #
def _attach_candidate_info_unique(out: pd.DataFrame,
                                  ref: pd.DataFrame,
                                  cand_col: str, tag: str) -> pd.DataFrame:
    """ref 在 (frame,id) 唯一；生成 {tag}_lead_x/v/len。"""
    if cand_col not in out.columns:
        out[f"{tag}_lead_x"] = np.nan
        out[f"{tag}_lead_v"] = np.nan
        out[f"{tag}_lead_len"] = np.nan
        return out
    r = ref.rename(columns={
        "id": "leaderId",
        "x": f"{tag}_lead_x",
        "xVelocity_raw": f"{tag}_lead_v",
        "veh_len": f"{tag}_lead_len",
    })
    out = out.merge(r, left_on=["frame", cand_col],
                    right_on=["frame", "leaderId"], how="left")
    out.drop(columns=["leaderId"], inplace=True)
    return out

def _shrink_distance(D: pd.Series, veh_len_f: pd.Series) -> pd.Series:
    """保守收缩：D_adj = max(D - (DIST_MARGIN_M + LEN_MARGIN_FRAC * L_f), EPS_D)"""
    adj = DIST_MARGIN_M + LEN_MARGIN_FRAC * pd.to_numeric(veh_len_f, errors="coerce").fillna(0.0)
    return pd.Series(np.maximum((pd.to_numeric(D, errors="coerce") - adj).astype(float), EPS_D), index=D.index)

def _ttc_linear_with_reaction(D: pd.Series, dv: pd.Series, v_f: pd.Series) -> pd.Series:
    """
    线性 TTC 两路并取最小：
      TTC_lin = D / dv
      TTC_rt  = max(D - v_f*TAU_TTC, EPS_D) / dv
    仅对 dv>0 生效。
    """
    TTC_lin = pd.Series(np.nan, index=D.index, dtype=float)
    TTC_rt  = pd.Series(np.nan, index=D.index, dtype=float)
    m = (D > EPS_D) & (dv > EPS_V)

    TTC_lin.loc[m] = D[m] / (dv[m] + EPS)
    D_rt = np.maximum(D - v_f*float(TAU_TTC), EPS_D)
    TTC_rt.loc[m]  = D_rt[m] / (dv[m] + EPS)

    return pd.concat([TTC_lin, TTC_rt], axis=1).min(axis=1, skipna=True)

def compute_frame_ssm_union(df_lane: pd.DataFrame, mu: float, grav: float, inc: bool) -> pd.DataFrame:
    """
    统一“接近”与距离口径，并并入左右候选：
      base：D_eff_base_adj = shrink( min(dhw, D_net_base) )
            TTC_base = min( D_eff_base_adj/dv_base , (D_eff_base_adj - v_f*TAU_TTC)/dv_base )
            DRAC_base 用 D_drac_base = max(D_eff_base_adj - v_f*TAU_S, EPS_D)
            PSD_base = D_eff_base_adj / (v_f^2/(2μg))
            接近掩码 clos_mask_base = (D_eff_base_adj>0 & Δv_base>0)
      L/R： D_adj_{L/R} = shrink(D_net_{L/R})
            TTC_{L/R} = min( D_adj_{L/R}/dv_{L/R} , (D_adj_{L/R} - v_f*TAU_TTC)/dv_{L/R} )
            DRAC_{L/R} 用 D_drac_{L/R} = max(D_adj_{L/R} - v_f*TAU_S, EPS_D)
    """
    leaders_ref = (df_lane[["frame", "id", "x", "xVelocity_raw", "veh_len"]]
                   .drop_duplicates(subset=["frame", "id"]))

    out = compute_frame_ssm_base(df_lane, mu=mu, grav=grav).reset_index(drop=True)
    out["x_dir"] = out["x"] if inc else -out["x"]

    vf = out.get("xVelocity_raw").astype(float)
    vf_abs = vf.abs()

    # base
    out = _attach_candidate_info_unique(out, leaders_ref, "precedingId", "B")
    B_x_dir = (out["B_lead_x"] if inc else -out["B_lead_x"])
    D_center = B_x_dir - out["x_dir"]
    D_net_base = D_center - 0.5 * (out["B_lead_len"] + out["veh_len"])
    D_eff_base = pd.concat([out["DHW"], D_net_base], axis=1).min(axis=1, skipna=True)
    D_eff_base_adj = _shrink_distance(D_eff_base, out["veh_len"])
    dv_base = vf - out["B_lead_v"]

    clos_mask_base = (D_eff_base_adj > EPS_D) & (dv_base > EPS_V)

    TTC_base = _ttc_linear_with_reaction(D_eff_base_adj, dv_base, vf)

    D_drac_base = np.maximum(D_eff_base_adj - vf * float(TAU_S), EPS_D)
    DRAC_base = pd.Series(np.nan, index=out.index, dtype=float)
    DRAC_base.loc[clos_mask_base] = (dv_base[clos_mask_base] ** 2) / (2.0 * D_drac_base[clos_mask_base] + EPS)

    PSD_base = pd.Series(np.nan, index=out.index, dtype=float)
    denom = (vf_abs ** 2) / (2.0 * mu * grav + EPS)
    PSD_base.loc[clos_mask_base & (vf_abs > 0.0)] = D_eff_base_adj[clos_mask_base & (vf_abs > 0.0)] / (denom[clos_mask_base & (vf_abs > 0.0)] + EPS)

    out["TTC"] = TTC_base
    out["DRAC"] = DRAC_base
    out["DRAC_valid_mask"] = clos_mask_base.astype(int)
    out["PSD_base"] = PSD_base
    out["PSD_valid_mask"] = clos_mask_base.astype(int)

    # L/R 候选
    for tag, cand in [("L", "leftPrecedingId"), ("R", "rightPrecedingId")]:
        out = _attach_candidate_info_unique(out, leaders_ref, cand, tag)
        lead_x_dir = (out[f"{tag}_lead_x"] if inc else -out[f"{tag}_lead_x"])
        D_center = lead_x_dir - out["x_dir"]
        D_net = D_center - 0.5 * (out[f"{tag}_lead_len"] + out["veh_len"])
        D_adj = _shrink_distance(D_net, out["veh_len"])
        dv_tag = vf - out[f"{tag}_lead_v"]

        m = (D_adj > EPS_D) & (dv_tag > EPS_V)
        TTC_tag = _ttc_linear_with_reaction(D_adj, dv_tag, vf)
        out[f"TTC_{tag}"] = np.nan
        out.loc[m, f"TTC_{tag}"] = TTC_tag[m]

        D_drac_tag = np.maximum(D_adj - vf * float(TAU_S), EPS_D)
        out[f"DRAC_{tag}"] = np.nan
        out[f"DRAC_{tag}_valid_mask"] = 0
        out.loc[m, f"DRAC_{tag}"] = ((dv_tag ** 2) / (2.0 * D_drac_tag + EPS))[m]
        out.loc[m, f"DRAC_{tag}_valid_mask"] = 1

    return out


# ============================ 权重/分类映射 ============================ #
def ttc_weight_from_value(ttc_val: pd.Series | float) -> pd.Series | float:
    """TTC>4→0；(3,4]→1；(2,3]→2；(0,2]→3。"""
    if np.isscalar(ttc_val):
        t = float(ttc_val) if np.isfinite(ttc_val) else np.nan
        if not np.isfinite(t) or t <= 0: return np.nan
        if t > 4.0: return 0.0
        if t > 3.0: return 1.0
        if t > 2.0: return 2.0
        return 3.0
    t = pd.to_numeric(ttc_val, errors="coerce")
    w = pd.Series(np.nan, index=t.index, dtype=float)
    w[t > 4.0] = 0.0
    w[(t > 3.0) & (t <= 4.0)] = 1.0
    w[(t > 2.0) & (t <= 3.0)] = 2.0
    w[(t > 0.0) & (t <= 2.0)] = 3.0
    return w

def drac_weight_from_value(drac_val: pd.Series | float, mu: float, grav: float) -> pd.Series | float:
    """r=DRAC/(μg)：r<0.5→0；[0.5,0.75)→1；[0.75,0.90)→2；≥0.90→3。"""
    amax = float(mu) * float(grav)
    if np.isscalar(drac_val):
        d = float(drac_val) if np.isfinite(drac_val) else np.nan
        if not np.isfinite(d) or d <= 0: return np.nan
        r = d / (amax + EPS)
        if r < 0.5: return 0.0
        if r < 0.75: return 1.0
        if r < 0.90: return 2.0
        return 3.0
    r = pd.to_numeric(drac_val, errors="coerce") / (amax + EPS)
    w = pd.Series(np.nan, index=r.index, dtype=float)
    w[r < 0.5] = 0.0
    w[(r >= 0.5) & (r < 0.75)] = 1.0
    w[(r >= 0.75) & (r < 0.90)] = 2.0
    w[r >= 0.90] = 3.0
    return w

def psd_weight_from_value(psd_val: pd.Series | float) -> pd.Series | float:
    """PSD<0.60→3；[0.60,0.80)→2；[0.80,1.00)→1；≥1.00→0。"""
    t1, t2, t3 = PSD_THRS
    if np.isscalar(psd_val):
        p = float(psd_val) if np.isfinite(psd_val) else np.nan
        if not np.isfinite(p) or p <= 0: return np.nan
        if p < t1: return 3.0
        if p < t2: return 2.0
        if p < t3: return 1.0
        return 0.0
    p = pd.to_numeric(psd_val, errors="coerce")
    w = pd.Series(np.nan, index=p.index, dtype=float)
    w[p < t1] = 3.0
    w[(p >= t1) & (p < t2)] = 2.0
    w[(p >= t2) & (p < t3)] = 1.0
    w[p >= t3] = 0.0
    return w

def cls_from_avg_weight(avg_w: float) -> int:
    """平均权重 → 四分类：0、1、2、3；无效 -1。"""
    if not np.isfinite(avg_w): return -1
    if abs(avg_w) < 1e-12: return 0
    if 0.0 < avg_w <= 1.0: return 1
    if 1.0 < avg_w <= 2.0: return 2
    return 3

def cls4_from_psd(psd_p95: float) -> int:
    """仅用于窗口 p95 的四分类（参考）。"""
    if not np.isfinite(psd_p95): return -1
    t1, t2, t3 = PSD_THRS
    if psd_p95 < t1: return 3
    elif psd_p95 < t2: return 2
    elif psd_p95 < t3: return 1
    else: return 0


# ============================ 窗口分位（参考） ============================ #
def compute_window_base_quantiles(
    sub: pd.DataFrame,
    *,
    min_valid_frames: int = MIN_VALID_FRAMES_DEFAULT,
) -> Dict[str, float]:
    out: Dict[str, float] = {}

    # TTC_p05：按车“帧级最小(含L/R)”再跨车 p05；不足回退帧级
    ttc_min_frame = pd.concat(
        [sub.get("TTC", pd.Series(dtype=float)),
         sub.get("TTC_L", pd.Series(dtype=float)),
         sub.get("TTC_R", pd.Series(dtype=float))],
        axis=1
    ).min(axis=1, skipna=True)
    t_perveh_min = (pd.DataFrame({"id": sub["id"].values, "t": ttc_min_frame})
                      .dropna()["t"].groupby(sub["id"]).min())
    nveh = int(sub["id"].nunique())
    need = max(2, min(5, int(np.ceil(0.10 * max(1, nveh)))))
    if len(t_perveh_min) >= need:
        out["TTC_p05"] = percentile(t_perveh_min, 5.0)
    else:
        t_frames = ttc_min_frame.replace([np.inf, -np.inf], np.nan).dropna()
        out["TTC_p05"] = percentile(t_frames, 5.0) if len(t_frames) >= min_valid_frames else np.nan

    # DRAC_p95：三路有效并集按车最大 → 跨车 p95；不足回退帧级
    def _mask(series: pd.Series, mask_col: str) -> pd.Series:
        return series.where(sub.get(mask_col, pd.Series(0, index=sub.index)) == 1, np.nan)
    drac_max_frame = pd.concat(
        [_mask(sub.get("DRAC",   pd.Series(dtype=float)), "DRAC_valid_mask"),
         _mask(sub.get("DRAC_L", pd.Series(dtype=float)), "DRAC_L_valid_mask"),
         _mask(sub.get("DRAC_R", pd.Series(dtype=float)), "DRAC_R_valid_mask")],
        axis=1
    ).max(axis=1, skipna=True)
    a_perveh_max = (pd.DataFrame({"id": sub["id"].values, "a": drac_max_frame})
                      .dropna()["a"].groupby(sub["id"]).max())
    if len(a_perveh_max) >= need:
        out["DRAC_p95"] = percentile(a_perveh_max, 95.0)
    else:
        a_frames = drac_max_frame.replace([np.inf, -np.inf], np.nan).dropna()
        out["DRAC_p95"] = percentile(a_frames, 95.0) if len(a_frames) >= min_valid_frames else np.nan

    # PSD_p95（仍用 PSD_allen；仅参考）
    psd_series = sub.get("PSD_allen", pd.Series(dtype=float)).replace([np.inf, -np.inf], np.nan).dropna()
    out["PSD_p95"] = percentile(psd_series, 95.0) if len(psd_series) else np.nan
    return out


# ============================ 节点级分类（含 1Hz 合桶） ============================ #
def compute_nodewise_labels(
    sub: pd.DataFrame,
    *,
    t0: float,
    fps: float,
    node_hz: float,
    ttc_q_low: float,    # 未用（保留签名）
    drac_q_high: float,  # 未用（保留签名）
    mu: float,
    grav: float,
) -> Dict[str, float]:
    """
    统一时间聚合（节点） + 合桶：
      1) 原始节点 index = floor((t - t0)*node_hz)
      2) bucket_factor = round(node_hz / NODE_BUCKET_HZ_TARGET)；bucket = node // factor
      3) 节点值：TTC→min；DRAC→max（有效并集）；PSD→min（接近掩码）
      4) 权重映射并对“桶节点”求平均
    """
    out: Dict[str, float] = {}

    node_idx = np.floor((sub["time"].values - float(t0)) * float(node_hz)).astype("int64")
    bucket_factor = max(1, int(round(float(node_hz) / float(NODE_BUCKET_HZ_TARGET))))
    bucket_idx = (node_idx // bucket_factor).astype("int64")
    bucket_dt = float(bucket_factor) / float(node_hz)  # 每桶秒
    min_bucket_frames = 1

    # ----- TTC -----
    ttc_frame_min = pd.concat(
        [sub.get("TTC", pd.Series(dtype=float)),
         sub.get("TTC_L", pd.Series(dtype=float)),
         sub.get("TTC_R", pd.Series(dtype=float))],
        axis=1
    ).min(axis=1, skipna=True)
    g_ttc = (pd.DataFrame({"bucket": bucket_idx, "ttc": ttc_frame_min})
             .replace([np.inf, -np.inf], np.nan).dropna())
    if len(g_ttc):
        ttc_bucket = (g_ttc.groupby("bucket")["ttc"]
                        .apply(lambda s: float(np.min(s)) if len(s) >= min_bucket_frames else np.nan)
                        .dropna())
        w_ttc = ttc_weight_from_value(ttc_bucket).dropna()
        if len(w_ttc):
            TTC_weight_avg = float(w_ttc.mean())
            out.update(
                TTC_weight_avg=TTC_weight_avg,
                TTC_exp_s1=float((w_ttc == 1.0).sum() * bucket_dt),
                TTC_exp_s2=float((w_ttc == 2.0).sum() * bucket_dt),
                TTC_exp_s3=float((w_ttc == 3.0).sum() * bucket_dt),
                TTC_cls4=cls_from_avg_weight(TTC_weight_avg),
                TTC_cls_mask=1,
            )
        else:
            out.update(TTC_weight_avg=np.nan, TTC_exp_s1=np.nan, TTC_exp_s2=np.nan, TTC_exp_s3=np.nan,
                       TTC_cls4=-1, TTC_cls_mask=0)
    else:
        out.update(TTC_weight_avg=np.nan, TTC_exp_s1=np.nan, TTC_exp_s2=np.nan, TTC_exp_s3=np.nan,
                   TTC_cls4=-1, TTC_cls_mask=0)

    # ----- DRAC -----
    def _mask(series: pd.Series, col: str) -> pd.Series:
        return series.where(sub.get(col, pd.Series(0, index=sub.index)) == 1, np.nan)
    drac_frame_max = pd.concat(
        [_mask(sub.get("DRAC",   pd.Series(dtype=float)), "DRAC_valid_mask"),
         _mask(sub.get("DRAC_L", pd.Series(dtype=float)), "DRAC_L_valid_mask"),
         _mask(sub.get("DRAC_R", pd.Series(dtype=float)), "DRAC_R_valid_mask")],
        axis=1
    ).max(axis=1, skipna=True)
    g_drac = (pd.DataFrame({"bucket": bucket_idx, "drac": drac_frame_max})
              .replace([np.inf, -np.inf], np.nan).dropna())
    if len(g_drac):
        drac_bucket = (g_drac.groupby("bucket")["drac"]
                         .apply(lambda s: float(np.max(s)) if len(s) >= min_bucket_frames else np.nan)
                         .dropna())
        w_drac = drac_weight_from_value(drac_bucket, mu=mu, grav=grav).dropna()
        if len(w_drac):
            DRAC_weight_avg = float(w_drac.mean())
            out.update(
                DRAC_weight_avg=DRAC_weight_avg,
                DRAC_exp_s1=float((w_drac == 1.0).sum() * bucket_dt),
                DRAC_exp_s2=float((w_drac == 2.0).sum() * bucket_dt),
                DRAC_exp_s3=float((w_drac == 3.0).sum() * bucket_dt),
                DRAC_cls4=cls_from_avg_weight(DRAC_weight_avg),
                DRAC_cls_mask=1,
            )
        else:
            out.update(DRAC_weight_avg=np.nan, DRAC_exp_s1=np.nan, DRAC_exp_s2=np.nan, DRAC_exp_s3=np.nan,
                       DRAC_cls4=-1, DRAC_cls_mask=0)
    else:
        out.update(DRAC_weight_avg=np.nan, DRAC_exp_s1=np.nan, DRAC_exp_s2=np.nan, DRAC_exp_s3=np.nan,
                   DRAC_cls4=-1, DRAC_cls_mask=0)

    # ----- PSD（接近掩码）-----
    psd_series = sub.get("PSD_base", pd.Series(dtype=float))
    psd_mask   = sub.get("PSD_valid_mask", pd.Series(0, index=sub.index)).astype(bool)
    g_psd = (pd.DataFrame({"bucket": bucket_idx, "psd": psd_series.where(psd_mask, np.nan)})
             .replace([np.inf, -np.inf], np.nan).dropna())
    if len(g_psd):
        psd_bucket = (g_psd.groupby("bucket")["psd"]
                        .apply(lambda s: float(np.min(s)) if len(s) >= min_bucket_frames else np.nan)
                        .dropna())
        w_psd = psd_weight_from_value(psd_bucket).dropna()
        if len(w_psd):
            PSD_weight_avg = float(w_psd.mean())
            out.update(
                PSD_cls4=cls_from_avg_weight(PSD_weight_avg),
                PSD_cls_mask=1,
            )
        else:
            out.update(PSD_cls4=-1, PSD_cls_mask=0)
    else:
        out.update(PSD_cls4=-1, PSD_cls_mask=0)

    return out
