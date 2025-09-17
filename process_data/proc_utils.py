# proc_utils.py
# -*- coding: utf-8 -*-
"""
辅助数据处理函数（与 TTC / DRAC / PSD 处理配套使用）。
从 process_highD 中抽离的通用工具，尽量保持同名/同参，便于最小改动替换。

单位约定：
- 距离 m、速度 m/s、加速度 m/s²；时间 s。
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

__all__ = [
    "percentile",
    "resolve_col",
    "normalize_vehicle_dims",
    "add_time_and_dt",
    "smooth_series",
    "lane_direction_map",
    "infer_lane_inc",
    "crossing_events_full",
    "precompute_location_anchors",
    "parse_start_time_seconds",
    "natural_sorted_dirs",
    "concat_outputs",
]

# ------------------------- 常量 ------------------------- #
CROSS_HYSTERESIS_M: float = 0.2   # 断面穿越的滞回（米）
EPS: float = 1e-12


# ------------------------- 基础/通用 ------------------------- #
def percentile(s: pd.Series, q_percent: float) -> float:
    """
    仅在有限值上计算分位（q 为百分位 0–100）。
    """
    s = pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    return float(np.percentile(s, q_percent)) if len(s) else np.nan


def resolve_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    """
    在 df.columns 中按候选名（不区分大小写）寻找首个存在的列名，返回其原始大小写。
    """
    m = {c.lower(): c for c in df.columns}
    for name in candidates:
        c = m.get(name.lower())
        if c is not None:
            return c
    return None


def normalize_vehicle_dims(df: pd.DataFrame) -> pd.DataFrame:
    """
    将不同数据源中的长度/宽度字段规范为：
      - veh_len: 车长（m）
      - veh_wid: 车宽（m）
    优先使用 length/height，缺失时回退 width。
    """
    out = df
    if "length" in out.columns:
        out["veh_len"] = pd.to_numeric(out["length"], errors="coerce")
    elif "width" in out.columns:
        out["veh_len"] = pd.to_numeric(out["width"], errors="coerce")
    else:
        out["veh_len"] = np.nan

    if "height" in out.columns:
        out["veh_wid"] = pd.to_numeric(out["height"], errors="coerce")
    elif "width" in out.columns:
        out["veh_wid"] = pd.to_numeric(out["width"], errors="coerce")
    else:
        out["veh_wid"] = np.nan

    return out


def add_time_and_dt(df: pd.DataFrame, fps: float) -> pd.DataFrame:
    """
    由 frame 与 fps 生成绝对时间 time 与时间步长 dt（秒）。
    """
    df["time"] = (df["frame"] - 1) / float(fps)
    df["dt"] = 1.0 / float(fps)
    return df


def smooth_series(x: pd.Series, win_frames: int = 10) -> pd.Series:
    """
    居中滑动均值平滑，窗口强制为奇数；不足窗口长度时返回原序列。
    """
    w = max(3, int(win_frames) | 1)
    if len(x) < w:
        return x
    return x.rolling(window=w, center=True, min_periods=max(1, w // 2)).mean()


# ------------------------- 方向/车道工具 ------------------------- #
def lane_direction_map(tracks: pd.DataFrame, tracks_meta: pd.DataFrame) -> Dict[int, int]:
    """
    估计每条 laneId 的主行驶方向（用 tracksMeta 的 drivingDirection 做众数）。
    返回 {laneId -> drivingDirection}。
    """
    id2dir = dict(zip(tracks_meta["id"].values, tracks_meta["drivingDirection"].values))
    tmp = tracks[["laneId", "id"]].copy()
    tmp["dir"] = tmp["id"].map(id2dir)
    mode_dir = tmp.groupby("laneId")["dir"].agg(lambda s: s.value_counts().idxmax() if len(s) else np.nan)
    return {int(k): int(v) for k, v in mode_dir.dropna().items()}


def infer_lane_inc(df_lane_raw: pd.DataFrame) -> bool:
    """
    依据该车道内车辆的 x 变化中位数推断“下游是否为 x 增大方向”。
    返回 True 表示 x 递增为下游（inc），False 表示相反。
    """
    df_l_sorted = df_lane_raw.sort_values(["id", "frame"])
    try:
        delta_by_id = df_l_sorted.groupby("id")["x"].apply(lambda s: s.iloc[-1] - s.iloc[0]).dropna()
        return bool(float(np.nanmedian(delta_by_id.values)) > 0)
    except Exception:
        return True  # 回退：默认 x 递增为下游


# ------------------------- 穿越事件/锚点 ------------------------- #
def crossing_events_full(
    df_lane: pd.DataFrame,
    X_line: float,
    inc: bool,
    eps_m: float = CROSS_HYSTERESIS_M,
) -> pd.DataFrame:
    """
    断面穿越事件（带滞回保护），用于 UF/DF 与边界速度统计。
    参数：
      - df_lane: 单车道帧级数据，需含列 ["time","id","x_prev","x","xVelocity","v_abs","veh_len"]
      - X_line: 断面 x 坐标
      - inc: True 表示下游在 x 增大方向
      - eps_m: 滞回距离（米），减少噪声反复穿越
    返回：包含穿越时刻的行（按 time 排序）
    """
    prev = df_lane["x_prev"] - X_line
    now = df_lane["x"] - X_line
    if inc:
        crossed = ((prev <= 0) & (now >= 0)) | ((prev < -eps_m) & (now > eps_m))
    else:
        crossed = ((prev >= 0) & (now <= 0)) | ((prev > eps_m) & (now < -eps_m))
    cols = ["time", "id", "xVelocity", "v_abs", "veh_len"]
    cols = [c for c in cols if c in df_lane.columns]
    ev = df_lane.loc[crossed, cols].copy()
    return ev.sort_values("time")


def precompute_location_anchors(rec_dirs: List[Path]) -> Dict[int, Tuple[float, float]]:
    """
    预扫所有录像的 x 取值，用 1% 与 99% 分位的中位数作为每个 location 的“上/下游锚点”。
    返回 {locationId -> (x_lo_loc, x_hi_loc)}。
    """
    anchors_raw: Dict[int, List[Tuple[float, float]]] = {}
    for d in rec_dirs:
        meta_fp = d / f"{d.name}_recordingMeta.csv"
        tracks_fp = d / f"{d.name}_tracks.csv"
        tmeta_fp = d / f"{d.name}_tracksMeta.csv"
        if not (meta_fp.exists() and tracks_fp.exists() and tmeta_fp.exists()):
            root = d.parent
            meta_fp = root / f"{d.name}_recordingMeta.csv"
            tracks_fp = root / f"{d.name}_tracks.csv"
            tmeta_fp = root / f"{d.name}_tracksMeta.csv"
            if not (meta_fp.exists() and tracks_fp.exists() and tmeta_fp.exists()):
                continue

        rec_meta = pd.read_csv(meta_fp)
        locationId = int(rec_meta.loc[0, "locationId"])
        try:
            x = pd.read_csv(tracks_fp, usecols=["x"])["x"].dropna().astype(float)
        except Exception:
            df_t = pd.read_csv(tracks_fp)
            col_x = resolve_col(df_t, ["x", "xCenter"])
            x = pd.to_numeric(df_t[col_x], errors="coerce").dropna().astype(float) if col_x else pd.Series(dtype=float)
        if len(x) == 0:
            continue
        x_lo = float(x.quantile(0.01))
        x_hi = float(x.quantile(0.99))
        anchors_raw.setdefault(locationId, []).append((x_lo, x_hi))

    anchors: Dict[int, Tuple[float, float]] = {}
    for loc, pairs in anchors_raw.items():
        los = np.array([p[0] for p in pairs], float)
        his = np.array([p[1] for p in pairs], float)
        anchors[loc] = (float(np.median(los)), float(np.median(his)))
    return anchors


# ------------------------- 时间/文件工具 ------------------------- #
def parse_start_time_seconds(rec_meta: pd.DataFrame) -> Optional[float]:
    """
    从 recordingMeta 中解析绝对起始时间（秒）。支持时间戳/毫秒/ISO 字符串。
    解析失败返回 None。
    """
    cand = [c for c in rec_meta.columns if ("start" in c.lower() and ("time" in c.lower() or "stamp" in c.lower()))]
    for c in cand:
        v = rec_meta.loc[0, c]
        if pd.isna(v):
            continue
        try:
            f = float(v)
            if f > 1e12:  # ms
                return f / 1000.0
            if f > 1e9:   # s
                return f
        except Exception:
            ts = pd.to_datetime(v, utc=True, errors="coerce")
            if ts is not pd.NaT:
                return float(ts.timestamp())
    return None


def natural_sorted_dirs(root: Path) -> List[Path]:
    """
    返回形如 00..99 的子目录（优先），否则根据 *_recordingMeta.csv 推断。
    """
    if root.exists():
        subdirs = [p for p in root.iterdir() if p.is_dir() and p.name.isdigit() and len(p.name) == 2]
    else:
        subdirs = []
    if subdirs:
        return sorted(subdirs, key=lambda p: int(p.name))
    metas = sorted(root.glob("[0-9][0-9]_recordingMeta.csv"), key=lambda p: int(p.name[:2]))
    ids = [m.name[:2] for m in metas]
    return [root / i for i in ids]


def concat_outputs(out_dir: Path, window_sec: float) -> Path:
    """
    合并各录像输出为一个总 CSV：all_windows_{window}s.csv
    """
    parts = sorted(out_dir.glob(f"[0-9][0-9]_windows_{int(window_sec)}s.csv"),
                   key=lambda p: int(p.name[:2]))
    out_fp = out_dir / f"all_windows_{int(window_sec)}s.csv"
    if not parts:
        return out_fp
    dfs = [pd.read_csv(p) for p in parts]
    all_df = pd.concat(dfs, ignore_index=True)
    all_df.to_csv(out_fp, index=False)
    return out_fp
