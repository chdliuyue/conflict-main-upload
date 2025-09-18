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

    ttc_thresholds: Tuple[float, float, float]
    drac_thresholds: Tuple[float, float, float]
    psd_thresholds: Tuple[float, float, float]
    node_bucket_hz_target: float

    def __post_init__(self) -> None:
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
        if not (psd[0] > psd[1] > psd[2]):
            raise ValueError("psd_thresholds must be strictly decreasing")

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


def _compute_candidate_metrics(
    gap: pd.Series,
    follower_speed_aligned: pd.Series,
    lead_speed_aligned: pd.Series,
    follower_speed_abs: pd.Series,
    *,
    mu: float,
    grav: float,
) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series, pd.Series]:
    """Compute TTC/DRAC/PSD for a follower-leader pair given the gap and speeds."""

    gap = pd.to_numeric(gap, errors="coerce")
    follower_speed_aligned = pd.to_numeric(follower_speed_aligned, errors="coerce")
    lead_speed_aligned = pd.to_numeric(lead_speed_aligned, errors="coerce")
    follower_speed_abs = pd.to_numeric(follower_speed_abs, errors="coerce").clip(lower=0.0)

    dv = follower_speed_aligned - lead_speed_aligned
    closing = (gap > EPS_DISTANCE) & (dv > EPS_SPEED)

    ttc = pd.Series(np.nan, index=gap.index, dtype=float)
    ttc.loc[closing] = gap.loc[closing] / (dv.loc[closing] + EPS)

    drac = pd.Series(np.nan, index=gap.index, dtype=float)
    drac.loc[closing] = (dv.loc[closing] ** 2) / (2.0 * gap.loc[closing] + EPS)

    drac_mask = closing.astype(int)

    denom = (follower_speed_abs ** 2) / (2.0 * float(mu) * float(grav) + EPS)
    psd_valid = closing & (denom > EPS)

    psd = pd.Series(np.nan, index=gap.index, dtype=float)
    psd.loc[psd_valid] = gap.loc[psd_valid] / (denom.loc[psd_valid] + EPS)

    psd_mask = psd_valid.astype(int)
    return ttc, drac, drac_mask, psd, psd_mask


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

    dhw_col = resolve_col(out, ["dhw", "spaceHeadway", "space_gap"])
    if dhw_col:
        out["DHW"] = pd.to_numeric(out[dhw_col], errors="coerce")
    else:
        out["DHW"] = np.nan

    leaders_ref = (
        df_lane[["frame", "id", "x", "xVelocity_raw", "veh_len"]]
        .drop_duplicates(subset=["frame", "id"])
    )

    out = _attach_candidate_info_unique(out, leaders_ref, "precedingId", "B")
    out = _attach_candidate_info_unique(out, leaders_ref, "leftPrecedingId", "L")
    out = _attach_candidate_info_unique(out, leaders_ref, "rightPrecedingId", "R")

    x_aligned = pd.to_numeric(out["x"], errors="coerce")
    x_aligned = x_aligned if inc else -x_aligned

    follower_len = pd.to_numeric(out.get("veh_len"), errors="coerce").fillna(0.0)

    vf_raw = pd.to_numeric(
        out.get("xVelocity_raw", out.get("xVelocity", pd.Series(np.nan, index=out.index))),
        errors="coerce",
    )
    vf_aligned = vf_raw if inc else -vf_raw
    vf_abs = vf_aligned.clip(lower=0.0)

    def _compute_gap(tag: str) -> Tuple[pd.Series, pd.Series]:
        lead_x = pd.to_numeric(out.get(f"{tag}_lead_x"), errors="coerce")
        lead_x = lead_x if inc else -lead_x
        lead_len = pd.to_numeric(out.get(f"{tag}_lead_len"), errors="coerce").fillna(0.0)
        gap_center = lead_x - x_aligned
        gap = gap_center - 0.5 * (lead_len + follower_len)
        gap = gap.where(gap > EPS_DISTANCE)

        lead_v = pd.to_numeric(out.get(f"{tag}_lead_v"), errors="coerce")
        lead_v_aligned = lead_v if inc else -lead_v
        return gap, lead_v_aligned

    gap_base, lead_v_base = _compute_gap("B")
    ttc_base, drac_base, drac_mask, psd_base, psd_mask = _compute_candidate_metrics(
        gap_base,
        vf_aligned,
        lead_v_base,
        vf_abs,
        mu=mu,
        grav=grav,
    )

    out["TTC"] = ttc_base
    out["DRAC"] = drac_base
    out["DRAC_valid_mask"] = drac_mask.astype(int)
    out["PSD_base"] = psd_base
    out["PSD_valid_mask"] = psd_mask.astype(int)
    out["PSD_allen"] = psd_base

    for tag in ("L", "R"):
        gap_tag, lead_v_tag = _compute_gap(tag)
        ttc_tag, drac_tag, drac_tag_mask, _, _ = _compute_candidate_metrics(
            gap_tag,
            vf_aligned,
            lead_v_tag,
            vf_abs,
            mu=mu,
            grav=grav,
        )
        out[f"TTC_{tag}"] = ttc_tag
        out[f"DRAC_{tag}"] = drac_tag
        out[f"DRAC_{tag}_valid_mask"] = drac_tag_mask.astype(int)

    return out


def compute_frame_ssm_base(
    df_lane: pd.DataFrame,
    mu: float,
    grav: float,
    inc: bool,
    *,
    params: SSMHyperParams,
) -> pd.DataFrame:
    """Return the base follower-leader metrics without side candidates."""

    union_df = compute_frame_ssm_union(df_lane, mu=mu, grav=grav, inc=inc, params=params)
    drop_cols = [
        "TTC_L",
        "TTC_R",
        "DRAC_L",
        "DRAC_R",
        "DRAC_L_valid_mask",
        "DRAC_R_valid_mask",
        "L_lead_x",
        "L_lead_v",
        "L_lead_len",
        "R_lead_x",
        "R_lead_v",
        "R_lead_len",
        "L_lead_gap",
        "R_lead_gap",
    ]
    return union_df.drop(columns=[c for c in drop_cols if c in union_df.columns])


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

    psd_series = (
        sub.get("PSD_allen", pd.Series(dtype=float))
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )
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
