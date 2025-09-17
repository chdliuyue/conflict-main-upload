#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations
import json, argparse, os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import freeze_support

try:
    from tqdm import tqdm
except Exception:
    tqdm = lambda x, **k: x  # no-op

# ============================================================
# 三个旋钮（其它口径固定）
# ============================================================
NODE_HZ      = 1.0    # 节点频率（Hz）：1.0(更保守) ↔ 2.0(更敏感)
TTC_Q_LOW    = 0.05   # 节点内 TTC 分位（低分位）推荐 0.08–0.20
DRAC_Q_HIGH  = 0.95   # 节点内 DRAC 分位（高分位）推荐 0.80–0.92
# ============================================================

# ===== 汇总统计（可选）=====
from summarize_windows import summarize_windows_csv

# ===== 冲突替代指标（专用）=====
from ssm_metrics import (
    compute_frame_ssm_union,
    compute_window_base_quantiles,
    compute_nodewise_labels,
    cls4_from_psd,
)

# ===== 通用处理工具（从 process_highD 抽离）=====
from proc_utils import (
    normalize_vehicle_dims,
    add_time_and_dt,
    smooth_series,
    lane_direction_map,
    infer_lane_inc,
    crossing_events_full,
    precompute_location_anchors,
    parse_start_time_seconds,
    natural_sorted_dirs,
    concat_outputs,
)

# ============================= 常量 =============================
DEFAULT_WINDOW_SEC = 10.0
DEFAULT_STRIDE_SEC = 10.0
DEFAULT_ALPHA = 0.10
DEFAULT_BOUNDARY_M = 15.0
DEFAULT_JERK_THR = 1.5
DEFAULT_MU = 0.4
DEFAULT_GRAV = 9.81

SPEED_FLOOR = 0.3
MIN_VALID_FRAMES = 10

RQ_DEN_MIN = 200.0
K_EDIE_MIN_VEH_TIME_S = 0.2
K_EDIE_DENSITY_FLOOR = 0.5

EPS = 1e-12

# ============================= 单窗聚合（定版口径） =============================
def aggregate_window_lane(df_lane: pd.DataFrame,
                          evU_all: pd.DataFrame, evD_all: pd.DataFrame,
                          t0: float, W: float,
                          X_up: float, X_down: float,
                          fps: float,
                          boundary_m: float, jerk_thr: float,
                          mu: float, grav: float,
                          inc: bool,
                          keep_empty: bool) -> Dict[str, float]:
    t1 = t0 + W
    evU = evU_all[(evU_all["time"] >= t0) & (evU_all["time"] < t1)]
    evD = evD_all[(evD_all["time"] >= t0) & (evD_all["time"] < t1)]
    sub = df_lane[(df_lane["time"] >= t0) & (df_lane["time"] < t1)].copy()

    if len(evU) == 0 and len(evD) == 0 and not keep_empty:
        return {}

    # —— Octet+（无 w_hat）——
    UF_h = len(evU) * (3600.0 / W)
    DF_h = len(evD) * (3600.0 / W)

    UAS = float(evU["v_abs"].mean()) if len(evU) else np.nan
    DAS = float(evD["v_abs"].mean()) if len(evD) else np.nan
    UAS_mask = int(len(evU) > 0); DAS_mask = int(len(evD) > 0)

    def mean_len(ev: pd.DataFrame) -> float:
        if len(ev) == 0: return np.nan
        return float(ev["veh_len"].mean()) if "veh_len" in ev.columns else np.nan

    UAL = mean_len(evU); DAL = mean_len(evD)
    UAL_mask = int(len(evU) > 0); DAL_mask = int(len(evD) > 0)

    UAS_eff = UAS if (np.isfinite(UAS) and UAS >= SPEED_FLOOR) else np.nan
    DAS_eff = DAS if (np.isfinite(DAS) and DAS >= SPEED_FLOOR) else np.nan
    UD = (UF_h/3600.0)/(UAS_eff+EPS)*1000.0 if np.isfinite(UAS_eff) else np.nan
    DD = (DF_h/3600.0)/(DAS_eff+EPS)*1000.0 if np.isfinite(DAS_eff) else np.nan
    UD = max(UD, 0.0) if np.isfinite(UD) else np.nan
    DD = max(DD, 0.0) if np.isfinite(DD) else np.nan
    UD_mask = int(np.isfinite(UD)); DD_mask = int(np.isfinite(DD))

    # —— rq/rk 相对梯度（可选）——
    dq = DF_h - UF_h
    dk = (DD - UD) if (np.isfinite(DD) and np.isfinite(UD)) else np.nan
    rq_den = 0.5 * (UF_h + DF_h)
    rk_den = 0.5 * (UD + DD) if (np.isfinite(UD) and np.isfinite(DD)) else np.nan
    rq_rel = abs(dq) / (rq_den + EPS) if rq_den >= RQ_DEN_MIN else np.nan
    rk_rel = abs(dk) / (rk_den + EPS) if (rk_den is not None and np.isfinite(rk_den) and rk_den > 0) else np.nan
    rq_rel_mask = int(np.isfinite(rq_rel)); rk_rel_mask = int(np.isfinite(rk_rel))

    # —— Edie 区间密度与 K-QV 一致性 ——
    x_lo, x_hi = (X_up, X_down) if X_up <= X_down else (X_down, X_up)
    x_lo_e, x_hi_e = x_lo + DEFAULT_BOUNDARY_M, x_hi - DEFAULT_BOUNDARY_M
    if x_hi_e - x_lo_e <= 1e-6:
        K_EDIE = np.nan
    else:
        mask_seg = (sub["x"] >= x_lo_e) & (sub["x"] <= x_hi_e)
        veh_time_s = sub.loc[mask_seg, "dt"].sum()
        K_EDIE = (veh_time_s / (W * (x_hi_e - x_lo_e))) * 1000.0 if veh_time_s >= K_EDIE_MIN_VEH_TIME_S else np.nan
    K_EDIE_mask = int(np.isfinite(K_EDIE))

    # K_QV：取 UD/DD 的均值（veh/km）
    k_vals = []
    if np.isfinite(UD): k_vals.append(UD)
    if np.isfinite(DD): k_vals.append(DD)
    K_QV = float(np.mean(k_vals)) if len(k_vals) else np.nan

    if np.isfinite(K_QV) and np.isfinite(K_EDIE) and (K_EDIE >= K_EDIE_DENSITY_FLOOR):
        QKV_relerr_q = float(abs(K_QV - K_EDIE) / (abs(K_EDIE) + EPS))
        QKV_relerr_q_mask = 1
    else:
        QKV_relerr_q = np.nan
        QKV_relerr_q_mask = 0

    # —— 速度波动 / 制动暴露 / Jerk ——
    vvals = sub.loc[(sub["x"] >= x_lo_e) & (sub["x"] <= x_hi_e), "v_abs"].replace([np.inf,-np.inf],np.nan).dropna()
    if len(vvals) >= 2:
        v_mean = float(np.mean(vvals)); v_std = float(np.std(vvals, ddof=1)) if len(vvals)>1 else 0.0
        CV_v = (v_std / (v_mean + EPS)) if v_mean > EPS else np.nan
    else:
        CV_v = np.nan
    CV_v_mask = int(np.isfinite(CV_v))

    neg_ax = np.maximum(-sub["xAcceleration"].values, 0.0)
    nveh_win = sub["id"].nunique()
    E_BRK = float(np.sum((neg_ax**2) * sub["dt"].values) / (nveh_win + EPS)) if len(sub) else np.nan
    E_BRK_mask = int(np.isfinite(E_BRK))

    if "jerk" in sub.columns:
        j = sub["jerk"].replace([np.inf,-np.inf],np.nan).dropna().values
        JBR = float(np.mean(j < (-jerk_thr))) if len(j) else np.nan
        JBR_mask = int(np.isfinite(JBR))
    else:
        JBR, JBR_mask = np.nan, 0

    # ====== SSM：窗口分位（p05/p95） + 节点级打分 ======
    base_q = compute_window_base_quantiles(sub, min_valid_frames=MIN_VALID_FRAMES)
    node_labels = compute_nodewise_labels(
        sub, t0=t0, fps=fps, node_hz=NODE_HZ,
        ttc_q_low=TTC_Q_LOW, drac_q_high=DRAC_Q_HIGH,
        mu=mu, grav=grav
    )

    # PSD 掩码与四分类
    PSD_series = sub["PSD_allen"].replace([np.inf,-np.inf], np.nan).dropna()
    PSD_p95 = base_q.get("PSD_p95", np.nan)
    PSD_cls_mask = int(len(PSD_series) >= MIN_VALID_FRAMES)
    PSD_cls4 = cls4_from_psd(PSD_p95) if PSD_cls_mask else -1

    return dict(
        t0=t0, t1=t1,
        UF=UF_h/3600.0, UAS=UAS, UD=UD/1000.0, UAL=UAL,
        DF=DF_h/3600.0, DAS=DAS, DD=DD/1000.0, DAL=DAL,
        rq_rel=rq_rel, rk_rel=rk_rel,
        K_EDIE=K_EDIE, K_QV=K_QV, QKV_relerr_q=QKV_relerr_q,
        CV_v=CV_v, E_BRK=E_BRK, JBR=JBR,
        UAS_mask=UAS_mask, DAS_mask=DAS_mask, UAL_mask=UAL_mask, DAL_mask=DAL_mask,
        UD_mask=UD_mask, DD_mask=DD_mask,
        rq_rel_mask=rq_rel_mask, rk_rel_mask=rk_rel_mask,
        K_EDIE_mask=K_EDIE_mask, QKV_relerr_q_mask=QKV_relerr_q_mask,
        CV_v_mask=CV_v_mask, E_BRK_mask=E_BRK_mask, JBR_mask=JBR_mask,

        # 数值分位（base）
        TTC_p05=base_q.get("TTC_p05", np.nan),
        DRAC_p95=base_q.get("DRAC_p95", np.nan),
        PSD_p95=PSD_p95,

        # 主标签（仅由三旋钮控制“多少”）
        TTC_weight_avg=node_labels.get("TTC_weight_avg", np.nan),
        DRAC_weight_avg=node_labels.get("DRAC_weight_avg", np.nan),
        TTC_exp_s1=node_labels.get("TTC_exp_s1", np.nan),
        TTC_exp_s2=node_labels.get("TTC_exp_s2", np.nan),
        TTC_exp_s3=node_labels.get("TTC_exp_s3", np.nan),
        DRAC_exp_s1=node_labels.get("DRAC_exp_s1", np.nan),
        DRAC_exp_s2=node_labels.get("DRAC_exp_s2", np.nan),
        DRAC_exp_s3=node_labels.get("DRAC_exp_s3", np.nan),
        TTC_cls4=node_labels.get("TTC_cls4", -1),
        DRAC_cls4=node_labels.get("DRAC_cls4", -1),
        PSD_cls4=PSD_cls4,
        TTC_cls_mask=node_labels.get("TTC_cls_mask", 0),
        DRAC_cls_mask=node_labels.get("DRAC_cls_mask", 0),
        PSD_cls_mask=PSD_cls_mask,

        Nveh_win_q=nveh_win
    )

# ============================= 单条录像处理（供 worker 调用） =============================
def process_one_recording(rec_dir: Path,
                          out_dir: Path,
                          window_sec: float,
                          stride_sec: float,
                          alpha: float,
                          boundary_m: float,
                          jerk_thr: float,
                          mu: float,
                          grav: float,
                          accel_clip: Optional[float],
                          keep_empty: bool,
                          loc_anchors: Dict[int, Tuple[float,float]],
                          abs_time_enabled: bool) -> Optional[Path]:
    meta_fp   = rec_dir / f"{rec_dir.name}_recordingMeta.csv"
    tracks_fp = rec_dir / f"{rec_dir.name}_tracks.csv"
    tmeta_fp  = rec_dir / f"{rec_dir.name}_tracksMeta.csv"
    if not (meta_fp.exists() and tracks_fp.exists() and tmeta_fp.exists()):
        root = rec_dir.parent
        meta_fp   = root / f"{rec_dir.name}_recordingMeta.csv"
        tracks_fp = root / f"{rec_dir.name}_tracks.csv"
        tmeta_fp  = root / f"{rec_dir.name}_tracksMeta.csv"
        if not (meta_fp.exists() and tracks_fp.exists() and tmeta_fp.exists()):
            print(f"[WARN] Missing CSV for recording {rec_dir.name} under {root}")
            return None

    rec_meta  = pd.read_csv(meta_fp)
    fps = float(rec_meta.loc[0,"frameRate"])
    duration = float(rec_meta.loc[0,"duration"])
    locationId = int(rec_meta.loc[0,"locationId"])

    want_cols = [
        "id","frame","laneId","x","xVelocity","xAcceleration","precedingId",
        "leftPrecedingId","rightPrecedingId",
        "width","length","height","dhw","ttc"
    ]
    head = pd.read_csv(tracks_fp, nrows=0)
    use = [c for c in want_cols if c in head.columns]
    tracks = pd.read_csv(tracks_fp, usecols=use)
    tracks_meta = pd.read_csv(tmeta_fp)

    # 数值化
    for c in ["x","xVelocity","xAcceleration","dhw","ttc","width","length","height"]:
        if c in tracks.columns:
            tracks[c] = pd.to_numeric(tracks[c], errors="coerce").astype("float32")
    for c in ["id","laneId","precedingId","leftPrecedingId","rightPrecedingId"]:
        if c in tracks.columns:
            tracks[c] = pd.to_numeric(tracks[c], errors="coerce").astype("int32")

    tracks = normalize_vehicle_dims(tracks)

    id2dir = dict(zip(tracks_meta["id"].values, tracks_meta["drivingDirection"].values))
    tracks["drivingDirection"] = tracks["id"].map(id2dir).astype("int16", copy=False)

    tracks["xVelocity_raw"]     = tracks["xVelocity"].astype("float32")
    tracks["xAcceleration_raw"] = tracks["xAcceleration"].astype("float32")

    tracks = add_time_and_dt(tracks, fps)
    smooth_win = max(3, int(round(0.4*fps))|1)
    tracks["xVelocity"]     = smooth_series(tracks["xVelocity_raw"],     smooth_win).astype("float32")
    tracks["xAcceleration"] = smooth_series(tracks["xAcceleration_raw"], smooth_win).astype("float32")

    if accel_clip is not None and accel_clip > 0:
        amax = float(accel_clip)
        tracks.loc[tracks["xAcceleration"] >  amax, "xAcceleration"] =  amax
        tracks.loc[tracks["xAcceleration"] < -amax, "xAcceleration"] = -amax

    tracks = tracks.sort_values(["id","frame"])
    tracks["jerk"] = tracks.groupby("id", group_keys=False)["xAcceleration"].diff() * float(fps)

    # 若缺列 precedingXVelocity / precedingXVelocity_raw 则合并补齐
    if "precedingXVelocity" not in tracks.columns:
        prec_vel = tracks[["frame","id","xVelocity"]].rename(
            columns={"id":"precedingId","xVelocity":"precedingXVelocity"})
        tracks = tracks.merge(prec_vel, on=["frame","precedingId"], how="left")
    prec_vel_raw = tracks[["frame","id","xVelocity_raw"]].rename(
        columns={"id":"precedingId","xVelocity_raw":"precedingXVelocity_raw"})
    tracks = tracks.merge(prec_vel_raw, on=["frame","precedingId"], how="left")

    tracks_ssm_base = tracks.copy()
    lane2dir = lane_direction_map(tracks_ssm_base, tracks_meta)

    # 位置锚点
    if locationId in loc_anchors:
        x_lo_loc, x_hi_loc = loc_anchors[locationId]
    else:
        x_lo_loc = float(tracks_ssm_base["x"].quantile(0.01))
        x_hi_loc = float(tracks_ssm_base["x"].quantile(0.99))
    XU_loc = x_lo_loc + alpha * (x_hi_loc - x_lo_loc)
    XD_loc = x_hi_loc - alpha * (x_hi_loc - x_lo_loc)

    start_epoch_s = parse_start_time_seconds(rec_meta) if abs_time_enabled else None

    starts = np.arange(0.0, max(0.0, duration - window_sec + 1e-6) + 1e-9, stride_sec)
    rows: List[dict] = []

    for lane_id, df_lane_raw in tracks_ssm_base.groupby("laneId", sort=False):
        if lane_id not in lane2dir:
            continue

        # 下游方向判断
        inc = infer_lane_inc(df_lane_raw)

        # 方向一致化与准备
        df_lane = df_lane_raw.sort_values(["time","id"]).copy()
        df_lane["v_dir"] = df_lane["xVelocity"] if inc else -df_lane["xVelocity"]
        df_lane["v_abs"] = df_lane["v_dir"].abs()
        df_lane["x_prev"] = df_lane.groupby("id")["x"].shift(1)
        df_lane["x_dir"] = df_lane["x"] if inc else -df_lane["x"]

        X_up, X_down = (XU_loc, XD_loc) if inc else (XD_loc, XU_loc)

        # 帧级 SSM（同/左/右候选并入）
        df_lane_ssm = compute_frame_ssm_union(df_lane, mu=DEFAULT_MU, grav=DEFAULT_GRAV, inc=inc)

        # 断面穿越事件
        evU_all = crossing_events_full(df_lane_ssm, X_up,   inc)
        evD_all = crossing_events_full(df_lane_ssm, X_down, inc)

        # 窗口滚动
        for t0 in starts:
            t1 = t0 + window_sec
            if t1 > duration + 1e-9:
                continue
            row = aggregate_window_lane(
                df_lane=df_lane_ssm, evU_all=evU_all, evD_all=evD_all,
                t0=t0, W=window_sec,
                X_up=X_up, X_down=X_down,
                fps=fps,
                boundary_m=DEFAULT_BOUNDARY_M, jerk_thr=DEFAULT_JERK_THR,
                mu=DEFAULT_MU, grav=DEFAULT_GRAV, inc=inc, keep_empty=keep_empty,
            )
            if not row:
                continue
            row.update(dict(
                recId=int(rec_dir.name), locationId=locationId,
                drivingDirection=int(lane2dir[lane_id]),
                laneId=int(lane_id),
                alpha=alpha, window=window_sec, stride=stride_sec, fps=fps,
                loc_x_up=XU_loc, loc_x_down=XD_loc
            ))
            if start_epoch_s is not None:
                row["t_abs0"] = start_epoch_s + t0
                row["t_abs1"] = start_epoch_s + t1
                row["start_epoch_s"] = start_epoch_s
            rows.append(row)

    if not rows:
        print(f"[WARN] No rows produced for {rec_dir}")
        return None

    out_df = pd.DataFrame(rows)

    params = dict(alpha=alpha, boundary_delta=DEFAULT_BOUNDARY_M, jerk_thr=DEFAULT_JERK_THR,
                  mu=DEFAULT_MU, grav=DEFAULT_GRAV, window=window_sec, stride=stride_sec,
                  accel_clip=accel_clip,
                  speed_floor=SPEED_FLOOR, rq_den_min=RQ_DEN_MIN,
                  k_edie_min_veh_time_s=K_EDIE_MIN_VEH_TIME_S,
                  k_edie_density_floor=K_EDIE_DENSITY_FLOOR,
                  keep_empty=keep_empty,
                  candidate_union="base + leftPreceding + rightPreceding",
                  labels_policy=f"per-node({NODE_HZ} Hz): TTC p{int(TTC_Q_LOW*100)} / DRAC p{int(DRAC_Q_HIGH*100)} → avg-weight",
                  location_anchors_used=True,
                  abs_time_enabled=(start_epoch_s is not None))
    out_df["params_json"] = json.dumps(params, ensure_ascii=False)

    key_cols = ["recId","locationId","drivingDirection","laneId","t0","t1"]
    features = ["UF","UAS","UD","UAL","DF","DAS","DD","DAL",
                "rq_rel","rk_rel","K_EDIE","K_QV","QKV_relerr_q",
                "CV_v","E_BRK","JBR"]
    feature_masks = ["UAS_mask","DAS_mask","UAL_mask","DAL_mask","UD_mask","DD_mask",
                     "rq_rel_mask","rk_rel_mask",
                     "K_EDIE_mask","QKV_relerr_q_mask",
                     "CV_v_mask","E_BRK_mask","JBR_mask"]
    labels_main = ["TTC_p05","DRAC_p95","PSD_p95",
                   "TTC_weight_avg","DRAC_weight_avg",
                   "TTC_exp_s1","TTC_exp_s2","TTC_exp_s3",
                   "DRAC_exp_s1","DRAC_exp_s2","DRAC_exp_s3",
                   "TTC_cls4","DRAC_cls4","PSD_cls4",
                   "TTC_cls_mask","DRAC_cls_mask","PSD_cls_mask"]
    extras = ["alpha","window","stride","fps","loc_x_up","loc_x_down","Nveh_win_q","params_json"]
    time_cols = ["t_abs0","t_abs1","start_epoch_s"] if "t_abs0" in out_df.columns else []

    cols = key_cols + features + feature_masks + labels_main + extras + time_cols
    for c in cols:
        if c not in out_df.columns:
            out_df[c] = np.nan
    out_df = out_df[cols]

    out_dir.mkdir(parents=True, exist_ok=True)
    out_fp = out_dir / f"{rec_dir.name}_windows_{int(window_sec)}s.csv"
    out_df.to_csv(out_fp, index=False)
    return out_fp

# ============================= 并行 worker（顶层可 pickle） =============================
def worker_process_one(args_tuple) -> Optional[Path]:
    return process_one_recording(*args_tuple)

# ============================= CLI/IO =============================
def parse_args():
    ap = argparse.ArgumentParser(
        description="highD → lane×10s-window CSV (TTC/DRAC per-node quantiles → avg-weight；无 w_hat；并行安全)")
    ap.add_argument("--root", type=str, default="../datasets/highD/data/")
    ap.add_argument("--out",  type=str, default="../data/highD/")
    ap.add_argument("--window",   type=float, default=DEFAULT_WINDOW_SEC)
    ap.add_argument("--stride",   type=float, default=DEFAULT_STRIDE_SEC)
    ap.add_argument("--alpha",    type=float, default=DEFAULT_ALPHA)
    ap.add_argument("--boundary", type=float, default=DEFAULT_BOUNDARY_M)
    ap.add_argument("--jerk_thr", type=float, default=DEFAULT_JERK_THR)
    ap.add_argument("--mu",       type=float, default=DEFAULT_MU)
    ap.add_argument("--grav",     type=float, default=DEFAULT_GRAV)
    ap.add_argument("--accel_clip", type=float, default=None)
    ap.add_argument("--keep_empty", action="store_true")
    ap.add_argument("--workers", type=int, default=max(1, os.cpu_count()//2))
    ap.add_argument("--no_loc_align", action="store_true")
    ap.add_argument("--abs_time", action="store_true")
    return ap.parse_args()

# ============================= main =============================
if __name__ == "__main__":
    freeze_support()  # Windows spawn 需要
    args = parse_args()
    ROOT = Path(args.root).expanduser().resolve()
    OUT  = Path(args.out).expanduser().resolve()
    WINDOW_SEC = float(args.window); STRIDE_SEC = float(args.stride)
    alpha = float(args.alpha); boundary_m = float(args.boundary)
    jerk_thr = float(args.jerk_thr); mu = float(args.mu); grav = float(args.grav)
    accel_clip = (float(args.accel_clip) if args.accel_clip is not None else None)
    keep_empty = bool(args.keep_empty)

    rec_dirs = natural_sorted_dirs(ROOT)
    print(f"[INFO] Found {len(rec_dirs)} recordings under {ROOT}")

    loc_anchors = {}
    if not args.no_loc_align:
        loc_anchors = precompute_location_anchors(rec_dirs)
        print(f"[INFO] Location anchors prepared for {len(loc_anchors)} locations")

    produced: List[Path] = []
    job_args_list = [
        (d, OUT, WINDOW_SEC, STRIDE_SEC, alpha, DEFAULT_BOUNDARY_M, DEFAULT_JERK_THR, DEFAULT_MU, DEFAULT_GRAV,
         accel_clip, keep_empty, loc_anchors, bool(args.abs_time))
        for d in rec_dirs
    ]

    if int(args.workers) <= 1:
        for ja in tqdm(job_args_list, desc="Recordings"):
            fp = worker_process_one(ja)
            if fp: produced.append(fp)
    else:
        with ProcessPoolExecutor(max_workers=int(args.workers)) as ex:
            futs = [ex.submit(worker_process_one, ja) for ja in job_args_list]
            for f in as_completed(futs):
                fp = f.result()
                if fp: produced.append(fp)

    if produced:
        all_fp = concat_outputs(OUT, WINDOW_SEC)
        print(f"[OK] Wrote per-recording CSVs to: {OUT}")
        print(f"[OK] Concatenated CSV: {all_fp}")
        try:
            summarize_windows_csv(str(all_fp))
        except Exception as e:
            print(f"[WARN] summarize_windows failed: {e}")
    else:
        print("[WARN] No outputs produced.")
