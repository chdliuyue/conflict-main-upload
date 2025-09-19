# -*- coding: utf-8 -*-
"""Surrogate safety metrics for the highD processing pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

__all__ = [
    "percentile",
    "weighted_percentile",
    "resolve_col",
    "compute_frame_ssm_base",
    "compute_frame_ssm_union",
    "compute_window_base_quantiles",
    "compute_nodewise_labels",
    "ttc_weight_from_value",
    "drac_weight_from_value",
    "psd_weight_from_value",
    "cls_from_avg_weight",
    "cls4_from_psd",
    "SSMHyperParams",
]

EPS = 1e-12
EPS_DISTANCE = 1e-3
EPS_SPEED = 1e-3


@dataclass(frozen=True)
class SSMHyperParams:
    """Hyper-parameters controlling TTC/DRAC/PSD labelling."""

    dist_margin_m: float
    len_margin_frac: float
    tau_ttc: float
    tau_drac: float
    ttc_thresholds: Tuple[float, float, float]
    drac_thresholds: Tuple[float, float, float]
    psd_thresholds: Tuple[float, float, float]
    node_bucket_hz_target: float

    def __post_init__(self) -> None:
        if not np.isfinite(self.dist_margin_m) or self.dist_margin_m < 0.0:
            raise ValueError("dist_margin_m must be non-negative")
        if not np.isfinite(self.len_margin_frac) or self.len_margin_frac < 0.0:
            raise ValueError("len_margin_frac must be non-negative")
        if not np.isfinite(self.tau_ttc) or self.tau_ttc < 0.0:
            raise ValueError("tau_ttc must be non-negative")
        if not np.isfinite(self.tau_drac) or self.tau_drac < 0.0:
            raise ValueError("tau_drac must be non-negative")

        ttc = tuple(self.ttc_thresholds)
        drac = tuple(self.drac_thresholds)
        psd = tuple(self.psd_thresholds)

        if len(ttc) != 3 or any(not np.isfinite(v) for v in ttc):
            raise ValueError("ttc_thresholds must contain three finite values")
        if not (ttc[0] > ttc[1] > ttc[2] > 0.0):
            raise ValueError("ttc_thresholds must be strictly decreasing and positive")

        if len(drac) != 3 or any(not np.isfinite(v) for v in drac):
            raise ValueError("drac_thresholds must contain three finite values")
        if not (drac[0] < drac[1] < drac[2]):
            raise ValueError("drac_thresholds must be strictly increasing")

        if len(psd) != 3 or any(not np.isfinite(v) for v in psd):
            raise ValueError("psd_thresholds must contain three finite values")
        if not (psd[0] > psd[1] > psd[2] >= 0.0):
            raise ValueError("psd_thresholds must be strictly decreasing and non-negative")

        if not np.isfinite(self.node_bucket_hz_target) or self.node_bucket_hz_target <= 0.0:
            raise ValueError("node_bucket_hz_target must be positive")


def percentile(s: pd.Series, q_percent: float) -> float:
    s = pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    return float(np.percentile(s, q_percent)) if len(s) else np.nan


def weighted_percentile(values: pd.Series, weights: pd.Series, q: float) -> float:
    """Return the weighted quantile of *values* (``0<=q<=1``)."""

    if values is None or weights is None:
        return np.nan

    v = pd.to_numeric(values, errors="coerce")
    w = pd.to_numeric(weights, errors="coerce")
    mask = v.notna() & w.notna() & (w > 0)
    if not mask.any():
        return np.nan

    v = v.loc[mask].astype(float)
    w = w.loc[mask].astype(float)
    if v.empty or w.empty:
        return np.nan

    order = np.argsort(v.values)
    v = v.iloc[order]
    w = w.iloc[order]

    total = float(w.sum())
    if total <= 0:
        return np.nan

    q = float(np.clip(q, 0.0, 1.0))
    if q <= 0.0:
        return float(v.iloc[0])
    if q >= 1.0:
        return float(v.iloc[-1])

    cumulative = w.cumsum().values
    cutoff = q * total
    idx = int(np.searchsorted(cumulative, cutoff, side="left"))
    idx = min(max(idx, 0), len(v) - 1)
    return float(v.iloc[idx])


def resolve_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    m = {c.lower(): c for c in df.columns}
    for name in candidates:
        c = m.get(name.lower())
        if c is not None:
            return c
    return None


def _attach_candidate_info_unique(
    out: pd.DataFrame,
    ref: pd.DataFrame,
    cand_col: str,
    tag: str,
) -> pd.DataFrame:
    """Attach leader position/velocity/length columns for the given candidate column."""

    if cand_col not in out.columns:
        out[f"{tag}_lead_x"] = np.nan
        out[f"{tag}_lead_v"] = np.nan
        out[f"{tag}_lead_len"] = np.nan
        return out

    r = ref.rename(
        columns={
            "id": "leaderId",
            "x": f"{tag}_lead_x",
            "xVelocity_raw": f"{tag}_lead_v",
            "veh_len": f"{tag}_lead_len",
        }
    )
    out = out.merge(
        r,
        left_on=["frame", cand_col],
        right_on=["frame", "leaderId"],
        how="left",
    )
    out.drop(columns=["leaderId"], inplace=True)
    return out


def _shrink_distance(
    distance: pd.Series,
    follower_len: pd.Series,
    *,
    params: SSMHyperParams,
) -> pd.Series:
    """Apply geometric margins to the bumper-to-bumper spacing."""

    dist = pd.to_numeric(distance, errors="coerce")
    follower = pd.to_numeric(follower_len, errors="coerce").fillna(0.0)
    margin = float(params.dist_margin_m) + float(params.len_margin_frac) * follower
    shrunk = np.maximum((dist - margin).astype(float), EPS_DISTANCE)
    return pd.Series(shrunk, index=dist.index, dtype=float)

def _ttc_linear_with_reaction(
    distance: pd.Series,
    dv: pd.Series,
    follower_speed: pd.Series,
    *,
    params: SSMHyperParams,
) -> pd.Series:
    """Return min(D/dv, max(D - vf*tau_ttc, EPS)/dv) for dv>0."""

    D = pd.to_numeric(distance, errors="coerce")
    closing_speed = pd.to_numeric(dv, errors="coerce")
    vf = pd.to_numeric(follower_speed, errors="coerce").clip(lower=0.0)

    ttc_lin = pd.Series(np.nan, index=D.index, dtype=float)
    ttc_rt = pd.Series(np.nan, index=D.index, dtype=float)

    mask = (D > EPS_DISTANCE) & (closing_speed > EPS_SPEED)
    denom = closing_speed.loc[mask] + EPS

    ttc_lin.loc[mask] = D.loc[mask] / denom

    D_rt = pd.Series(
        np.maximum((D - vf * float(params.tau_ttc)).astype(float), EPS_DISTANCE),
        index=D.index,
        dtype=float,
    )
    ttc_rt.loc[mask] = D_rt.loc[mask] / denom

    return pd.concat([ttc_lin, ttc_rt], axis=1).min(axis=1, skipna=True)


def compute_frame_ssm_union(
    df_lane: pd.DataFrame,
    mu: float,
    grav: float,
    inc: bool,
    *,
    params: SSMHyperParams,
) -> pd.DataFrame:
    """Compute TTC/DRAC/PSD for the main, left and right leaders of a lane."""

    if params is None:
        raise ValueError("params must be provided")

    out = df_lane.copy().reset_index(drop=True)

    dhw_col = resolve_col(out, ["DHW", "dhw", "spaceHeadway", "space_gap"])
    if dhw_col is not None and dhw_col in out.columns:
        dhw_series = pd.to_numeric(out[dhw_col], errors="coerce")
        dhw_series = dhw_series.where(dhw_series > 0.0, np.nan)
    else:
        dhw_series = pd.Series(np.nan, index=out.index, dtype=float)
    out["DHW"] = dhw_series

    leaders_ref = (
        df_lane[["frame", "id", "x", "xVelocity_raw", "veh_len"]]
        .drop_duplicates(subset=["frame", "id"])
    )

    out = _attach_candidate_info_unique(out, leaders_ref, "precedingId", "B")
    out = _attach_candidate_info_unique(out, leaders_ref, "leftPrecedingId", "L")
    out = _attach_candidate_info_unique(out, leaders_ref, "rightPrecedingId", "R")

    x_dir = pd.to_numeric(out.get("x"), errors="coerce")
    x_dir = x_dir if inc else -x_dir
    follower_len = pd.to_numeric(out.get("veh_len"), errors="coerce").fillna(0.0)

    vf_raw = pd.to_numeric(
        out.get("xVelocity_raw", out.get("xVelocity", pd.Series(np.nan, index=out.index))),
        errors = "coerce",
    )
    vf_aligned = vf_raw if inc else -vf_raw
    vf_speed = vf_aligned.clip(lower=0.0)

    base_lead_x = pd.to_numeric(out.get("B_lead_x"), errors="coerce")
    base_lead_x = base_lead_x if inc else -base_lead_x
    base_lead_len = pd.to_numeric(out.get("B_lead_len"), errors="coerce").fillna(0.0)
    dhw_series = out.get("DHW", pd.Series(np.nan, index=out.index, dtype=float))
    D_center = base_lead_x - x_dir
    D_net = pd.to_numeric(D_center - 0.5 * (base_lead_len + follower_len), errors="coerce")
    D_net = D_net.where(D_net > 0.0, np.nan)
    D_eff = pd.to_numeric(dhw_series.combine_first(D_net), errors="coerce")
    D_eff = D_eff.where(D_eff > 0.0, np.nan)
    D_adj = _shrink_distance(D_eff, follower_len, params=params)

    lead_v_base = pd.to_numeric(out.get("B_lead_v"), errors="coerce")
    lead_v_aligned = lead_v_base if inc else -lead_v_base
    dv_base = vf_aligned - lead_v_aligned

    closing_mask = (D_eff > EPS_DISTANCE) & (dv_base > EPS_SPEED)

    ttc_base = _ttc_linear_with_reaction(D_adj, dv_base, vf_speed, params=params)

    D_drac = pd.Series(
        np.maximum((D_adj - vf_speed * float(params.tau_drac)).astype(float), EPS_DISTANCE),
        index=out.index,
        dtype=float,
    )
    drac_base = pd.Series(np.nan, index=out.index, dtype=float)
    drac_base.loc[closing_mask] = (dv_base.loc[closing_mask] ** 2) / (
            2.0 * D_drac.loc[closing_mask] + EPS
    )

    lead_speed = lead_v_aligned.clip(lower=0.0)
    psd_denom = (vf_speed ** 2 - lead_speed ** 2) / (2.0 * float(mu) * float(grav) + EPS)
    psd_valid = closing_mask & (vf_speed > EPS_SPEED) & (psd_denom > EPS_DISTANCE)
    psd_base = pd.Series(np.nan, index=out.index, dtype=float)
    psd_base.loc[psd_valid] = D_adj.loc[psd_valid] / (psd_denom.loc[psd_valid] + EPS)

    out["TTC"] = ttc_base
    out["DRAC"] = drac_base
    out["DRAC_valid_mask"] = closing_mask.astype(int)
    out["PSD_base"] = psd_base
    out["PSD_valid_mask"] = psd_valid.astype(int)

    for tag, cand in (("L", "leftPrecedingId"), ("R", "rightPrecedingId")):
        lead_x = pd.to_numeric(out.get(f"{tag}_lead_x"), errors="coerce")
        lead_x = lead_x if inc else -lead_x
        lead_len = pd.to_numeric(out.get(f"{tag}_lead_len"), errors="coerce").fillna(0.0)
        D_center_tag = lead_x - x_dir
        D_net_tag = pd.to_numeric(D_center_tag - 0.5 * (lead_len + follower_len), errors="coerce")
        D_net_tag = D_net_tag.where(D_net_tag > 0.0, np.nan)
        D_adj_tag = _shrink_distance(D_net_tag, follower_len, params=params)

        lead_v_tag = pd.to_numeric(out.get(f"{tag}_lead_v"), errors="coerce")
        lead_v_tag = lead_v_tag if inc else -lead_v_tag
        dv_tag = vf_aligned - lead_v_tag
        mask_tag = (D_net_tag > EPS_DISTANCE) & (dv_tag > EPS_SPEED)

        ttc_tag = _ttc_linear_with_reaction(D_adj_tag, dv_tag, vf_speed, params=params)
        drac_space = pd.Series(
            np.maximum((D_adj_tag - vf_speed * float(params.tau_drac)).astype(float), EPS_DISTANCE),
            index=out.index,
            dtype=float,
        )
        drac_tag = pd.Series(np.nan, index=out.index, dtype=float)
        drac_tag.loc[mask_tag] = (dv_tag.loc[mask_tag] ** 2) / (
                2.0 * drac_space.loc[mask_tag] + EPS
        )

        mask_series = pd.Series(0, index=out.index, dtype=int)
        mask_series.loc[mask_tag] = 1

        out[f"TTC_{tag}"] = ttc_tag
        out[f"DRAC_{tag}"] = drac_tag
        out[f"DRAC_{tag}_valid_mask"] = mask_series

    return out


def compute_frame_ssm_base(
        df_lane: pd.DataFrame,
        mu: float,
        grav: float,
) -> pd.DataFrame:
    """Compute TTC/DRAC/PSD scaffolding using raw headway information."""

    out = df_lane.copy()

    col_ttc = resolve_col(out, ["ttc"])
    if col_ttc and col_ttc in out.columns:
        ttc_raw = pd.to_numeric(out[col_ttc], errors="coerce").where(lambda s: s > 0.0, np.nan)
    else:
        ttc_raw = pd.Series(np.nan, index=out.index, dtype=float)

    col_dhw = resolve_col(out, ["dhw", "spaceHeadway", "space_gap"])
    if col_dhw and col_dhw in out.columns:
        dhw = pd.to_numeric(out[col_dhw], errors="coerce").where(lambda s: s > 0.0, np.nan)
    else:
        dhw = pd.Series(np.nan, index=out.index, dtype=float)
    out["DHW"] = dhw

    vf = pd.to_numeric(
        out.get("xVelocity_raw", out.get("xVelocity", pd.Series(np.nan, index=out.index))),
        errors="coerce",
    )
    vl = pd.to_numeric(
        out.get(
            "precedingXVelocity_raw",
            out.get("precedingXVelocity", pd.Series(np.nan, index=out.index)),
        ),
        errors="coerce",
    )
    dv = vf - vl

    ttc_lin = pd.Series(np.nan, index=out.index, dtype=float)
    mask_lin = dhw.notna() & (dv > EPS_SPEED)
    ttc_lin.loc[mask_lin] = dhw.loc[mask_lin] / (dv.loc[mask_lin] + EPS)

    out["TTC"] = pd.concat([ttc_raw, ttc_lin], axis=1).min(axis=1, skipna=True)

    drac_base = pd.Series(np.nan, index=out.index, dtype=float)
    mask_drac = dhw.notna() & (dv > EPS_SPEED)
    drac_base.loc[mask_drac] = (dv.loc[mask_drac] ** 2) / (2.0 * dhw.loc[mask_drac] + EPS)
    out["DRAC"] = drac_base
    out["DRAC_valid_mask"] = mask_drac.astype(int)

    vf_abs = vf.abs()
    psd_series = pd.Series(np.nan, index=out.index, dtype=float)
    mask_psd = dhw.notna() & (vf_abs > EPS_SPEED)
    denom = (vf_abs ** 2) / (2.0 * float(mu) * float(grav) + EPS)
    psd_series.loc[mask_psd] = dhw.loc[mask_psd] / (denom.loc[mask_psd] + EPS)
    out["PSD_allen"] = psd_series.clip(lower=0.0)

    return out


def ttc_weight_from_value(
    ttc_val: pd.Series | float,
    *,
    params: SSMHyperParams,
) -> pd.Series | float:
    """Map TTC values to ordinal risk weights according to ``params.ttc_thresholds``."""

    thr_safe, thr_low, thr_mid = params.ttc_thresholds

    def _scalar(ttc: float) -> float:
        if not np.isfinite(ttc) or ttc <= 0:
            return np.nan
        if ttc > thr_safe:
            return 0.0
        if ttc > thr_low:
            return 1.0
        if ttc > thr_mid:
            return 2.0
        return 3.0

    if np.isscalar(ttc_val):
        return _scalar(float(ttc_val))

    t = pd.to_numeric(ttc_val, errors="coerce")
    w = pd.Series(np.nan, index=t.index, dtype=float)
    w[t > thr_safe] = 0.0
    w[(t <= thr_safe) & (t > thr_low)] = 1.0
    w[(t <= thr_low) & (t > thr_mid)] = 2.0
    w[(t > 0.0) & (t <= thr_mid)] = 3.0
    return w


def drac_weight_from_value(
    drac_val: pd.Series | float,
    mu: float,
    grav: float,
    *,
    params: SSMHyperParams,
) -> pd.Series | float:
    """Map DRAC values to ordinal risk weights according to ``params.drac_thresholds``."""

    thr_low, thr_mid, thr_high = params.drac_thresholds

    def _scalar(drac: float) -> float:
        if not np.isfinite(drac) or drac <= 0:
            return np.nan
        if drac < thr_low:
            return 0.0
        if drac < thr_mid:
            return 1.0
        if drac < thr_high:
            return 2.0
        return 3.0

    if np.isscalar(drac_val):
        return _scalar(float(drac_val))

    d = pd.to_numeric(drac_val, errors="coerce")
    w = pd.Series(np.nan, index=d.index, dtype=float)
    w[d < thr_low] = 0.0
    w[(d >= thr_low) & (d < thr_mid)] = 1.0
    w[(d >= thr_mid) & (d < thr_high)] = 2.0
    w[d >= thr_high] = 3.0
    return w


def psd_weight_from_value(
    psd_val: pd.Series | float,
    *,
    params: SSMHyperParams,
) -> pd.Series | float:
    """Map PSD values to ordinal risk weights according to ``params.psd_thresholds``."""

    thr_safe, thr_low, thr_mid = params.psd_thresholds

    def _scalar(psd: float) -> float:
        if not np.isfinite(psd):
            return np.nan
        if psd >= thr_safe:
            return 0.0
        if psd >= thr_low:
            return 1.0
        if psd >= thr_mid:
            return 2.0
        return 3.0

    if np.isscalar(psd_val):
        return _scalar(float(psd_val))

    p = pd.to_numeric(psd_val, errors="coerce")
    w = pd.Series(np.nan, index=p.index, dtype=float)
    w[p >= thr_safe] = 0.0
    w[(p < thr_safe) & (p >= thr_low)] = 1.0
    w[(p < thr_low) & (p >= thr_mid)] = 2.0
    w[p < thr_mid] = 3.0
    return w


def cls_from_avg_weight(avg_w: float) -> int:
    """Convert averaged ordinal weight to a four-class label."""

    if not np.isfinite(avg_w):
        return -1
    if abs(avg_w) < 1e-12:
        return 0
    if 0.0 < avg_w <= 1.0:
        return 1
    if 1.0 < avg_w <= 2.0:
        return 2
    return 3


def cls4_from_psd(psd_p95: float, *, params: SSMHyperParams) -> int:
    if not np.isfinite(psd_p95):
        return -1
    thr_safe, thr_low, thr_mid = params.psd_thresholds
    if psd_p95 >= thr_safe:
        return 0
    if psd_p95 >= thr_low:
        return 1
    if psd_p95 >= thr_mid:
        return 2
    return 3


MIN_VALID_FRAMES_DEFAULT = 10


def compute_window_base_quantiles(
    sub: pd.DataFrame,
    *,
    min_valid_frames: int = MIN_VALID_FRAMES_DEFAULT,
) -> Dict[str, float]:
    out: Dict[str, float] = {}

    ttc_min_frame = pd.concat(
        [
            sub.get("TTC", pd.Series(dtype=float)),
            sub.get("TTC_L", pd.Series(dtype=float)),
            sub.get("TTC_R", pd.Series(dtype=float)),
        ],
        axis=1,
    ).min(axis=1, skipna=True)
    t_perveh_min = (
        pd.DataFrame({"id": sub["id"].values, "t": ttc_min_frame})
        .dropna()["t"].groupby(sub["id"]).min()
    )
    nveh = int(sub["id"].nunique())
    need = max(2, min(5, int(np.ceil(0.10 * max(1, nveh)))))
    if len(t_perveh_min) >= need:
        out["TTC_p05"] = percentile(t_perveh_min, 5.0)
    else:
        t_frames = ttc_min_frame.replace([np.inf, -np.inf], np.nan).dropna()
        out["TTC_p05"] = percentile(t_frames, 5.0) if len(t_frames) >= min_valid_frames else np.nan

    def _mask(series: pd.Series, mask_col: str) -> pd.Series:
        return series.where(sub.get(mask_col, pd.Series(0, index=sub.index)) == 1, np.nan)

    drac_max_frame = pd.concat(
        [
            _mask(sub.get("DRAC", pd.Series(dtype=float)), "DRAC_valid_mask"),
            _mask(sub.get("DRAC_L", pd.Series(dtype=float)), "DRAC_L_valid_mask"),
            _mask(sub.get("DRAC_R", pd.Series(dtype=float)), "DRAC_R_valid_mask"),
        ],
        axis=1,
    ).max(axis=1, skipna=True)
    a_perveh_max = (
        pd.DataFrame({"id": sub["id"].values, "a": drac_max_frame})
        .dropna()["a"].groupby(sub["id"]).max()
    )
    if len(a_perveh_max) >= need:
        out["DRAC_p95"] = percentile(a_perveh_max, 95.0)
    else:
        a_frames = drac_max_frame.replace([np.inf, -np.inf], np.nan).dropna()
        out["DRAC_p95"] = percentile(a_frames, 95.0) if len(a_frames) >= min_valid_frames else np.nan

    psd_col = resolve_col(sub, ["PSD_base", "PSD_allen"])
    if psd_col is not None and psd_col in sub.columns:
        psd_values = pd.to_numeric(sub[psd_col], errors="coerce")
    else:
        psd_values = pd.Series(np.nan, index=sub.index, dtype=float)

    psd_mask_col = resolve_col(sub, ["PSD_valid_mask"])
    if psd_mask_col is not None and psd_mask_col in sub.columns:
        mask_series = pd.to_numeric(sub[psd_mask_col], errors="coerce").fillna(0)
        psd_values = psd_values.where(mask_series.astype(int) == 1, np.nan)

    psd_series = psd_values.replace([np.inf, -np.inf], np.nan).dropna()
    out["PSD_p95"] = percentile(psd_series, 95.0) if len(psd_series) else np.nan
    return out


def compute_nodewise_labels(
    sub: pd.DataFrame,
    *,
    t0: float,
    fps: float,
    node_hz: float,
    ttc_q_low: float,
    drac_q_high: float,
    mu: float,
    grav: float,
    params: SSMHyperParams,
) -> Dict[str, float]:
    """Aggregate per-frame metrics into node-wise ordinal labels."""

    out: Dict[str, float] = {}

    node_idx = np.floor((sub["time"].values - float(t0)) * float(node_hz)).astype("int64")
    node_hz_val = float(node_hz)
    bucket_target = float(params.node_bucket_hz_target)
    if node_hz_val <= 0.0 or bucket_target <= 0.0:
        bucket_factor = 1
    else:
        bucket_factor = max(1, int(round(node_hz_val / bucket_target)))
    bucket_idx = (node_idx // bucket_factor).astype("int64")

    default_dt = 1.0 / float(fps) if fps > 0 else (1.0 / float(node_hz) if node_hz > 0 else 0.0)
    if "dt" in sub.columns:
        dt_series = pd.to_numeric(sub["dt"], errors="coerce").fillna(default_dt)
        if default_dt > 0:
            dt_series = dt_series.where(dt_series > 0, default_dt)
        else:
            dt_series = dt_series.clip(lower=0.0)
    else:
        dt_series = pd.Series(default_dt, index=sub.index, dtype=float)
    dt_series = dt_series.astype(float)

    def _bucket_reduce(values: pd.Series, how: str) -> pd.DataFrame:
        df = pd.DataFrame({
            "bucket": bucket_idx,
            "value": values,
            "dt": dt_series,
        })
        df["value"] = pd.to_numeric(df["value"], errors="coerce").replace([np.inf, -np.inf], np.nan)
        df["dt"] = pd.to_numeric(df["dt"], errors="coerce").fillna(0.0)
        df = df[(df["dt"] > 0.0) & df["value"].notna()]
        if df.empty:
            return pd.DataFrame(columns=["value", "dt"])
        grouped = df.groupby("bucket").agg(value=("value", how), dt=("dt", "sum"))
        grouped = grouped[grouped["dt"] > 0.0]
        return grouped

    def _apply_metric(
        values: pd.Series,
        how: str,
        weight_fn,
        prefix: str,
        *,
        quantile_q: float,
        extreme: str,
        cls_weight_mode: str = "time",
    ) -> None:
        bucket = _bucket_reduce(values, how)
        if bucket.empty:
            return
        data = bucket.copy()
        data["weight"] = pd.to_numeric(weight_fn(data["value"]), errors="coerce")
        data = data.replace([np.inf, -np.inf], np.nan)
        data = data[data["weight"].notna() & (data["dt"] > 0.0)]
        if data.empty:
            return

        total_time = float(data["dt"].sum())
        if total_time <= 0.0:
            return

        weight_time_avg = float(np.average(data["weight"], weights=data["dt"]))
        if not np.isfinite(weight_time_avg):
            return

        weight_equal_avg = float(data["weight"].mean()) if len(data) else np.nan
        if cls_weight_mode == "equal" and np.isfinite(weight_equal_avg):
            cls_weight_avg = weight_equal_avg
        else:
            cls_weight_avg = weight_time_avg
        if not np.isfinite(cls_weight_avg):
            return

        quant_label = f"p{int(round(quantile_q * 100)):02d}"
        exposures = {
            f"{prefix}_exp_s1": float(data.loc[np.isclose(data["weight"], 1.0), "dt"].sum()),
            f"{prefix}_exp_s2": float(data.loc[np.isclose(data["weight"], 2.0), "dt"].sum()),
            f"{prefix}_exp_s3": float(data.loc[np.isclose(data["weight"], 3.0), "dt"].sum()),
        }

        out.update({
            f"{prefix}_weight_avg": cls_weight_avg,
            f"{prefix}_cls4": cls_from_avg_weight(cls_weight_avg),
            f"{prefix}_cls_mask": 1,
            f"{prefix}_time_{extreme}": float(getattr(data["value"], extreme)()),
            f"{prefix}_time_{quant_label}": weighted_percentile(data["value"], data["dt"], quantile_q),
        })
        if cls_weight_mode == "equal":
            out[f"{prefix}_weight_avg_time"] = weight_time_avg
        out.update(exposures)

    ttc_frame_min = pd.concat(
        [
            sub.get("TTC", pd.Series(dtype=float)),
            sub.get("TTC_L", pd.Series(dtype=float)),
            sub.get("TTC_R", pd.Series(dtype=float)),
        ],
        axis=1,
    ).min(axis=1, skipna=True)
    _apply_metric(
        ttc_frame_min,
        "min",
        lambda s: ttc_weight_from_value(s, params=params),
        "TTC",
        quantile_q=float(ttc_q_low),
        extreme="min",
    )

    def _mask(series: pd.Series, col: str) -> pd.Series:
        return series.where(sub.get(col, pd.Series(0, index=sub.index)) == 1, np.nan)

    drac_frame_max = pd.concat(
        [
            _mask(sub.get("DRAC", pd.Series(dtype=float)), "DRAC_valid_mask"),
            _mask(sub.get("DRAC_L", pd.Series(dtype=float)), "DRAC_L_valid_mask"),
            _mask(sub.get("DRAC_R", pd.Series(dtype=float)), "DRAC_R_valid_mask"),
        ],
        axis=1,
    ).max(axis=1, skipna=True)
    _apply_metric(
        drac_frame_max,
        "max",
        lambda s: drac_weight_from_value(s, mu=mu, grav=grav, params=params),
        "DRAC",
        quantile_q=float(drac_q_high),
        extreme="max",
    )

    psd_series = sub.get("PSD_base", pd.Series(dtype=float))
    psd_mask = sub.get("PSD_valid_mask", pd.Series(0, index=sub.index)).astype(bool)
    _apply_metric(
        psd_series.where(psd_mask, np.nan),
        "min",
        lambda s: psd_weight_from_value(s, params=params),
        "PSD",
        quantile_q=float(ttc_q_low),
        extreme="min",
        cls_weight_mode="equal",
    )

    return out
