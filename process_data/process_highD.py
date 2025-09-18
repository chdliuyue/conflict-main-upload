#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Entry-point for the integrated highD data processing pipeline."""

from __future__ import annotations

import argparse
from multiprocessing import freeze_support
from pathlib import Path

from highd_pipeline import (
    DEFAULT_ALPHA,
    DEFAULT_BOUNDARY_M,
    DEFAULT_GRAV,
    DEFAULT_DRAC_THRESHOLDS,
    DEFAULT_JERK_THR,
    DEFAULT_MU,
    DEFAULT_NODE_BUCKET_HZ_TARGET,
    DEFAULT_PSD_THRESHOLDS,
    DEFAULT_STRIDE_SEC,
    DEFAULT_TTC_THRESHOLDS,
    DEFAULT_WINDOW_SEC,
    DEFAULT_WORKERS,
    HighDPipelineConfig,
    run_highd_pipeline,
)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=(
            "Convert highD recordings to per-lane window features, clean the"
            " 12 core features with 3 conflict labels and optionally summarise"
            " the result."
        )
    )
    ap.add_argument("--root", type=str, default="../datasets/highD/data/", help="highD dataset root")
    ap.add_argument("--out", type=str, default="../data/highD/", help="output directory")
    ap.add_argument("--window", type=float, default=DEFAULT_WINDOW_SEC, help="window size in seconds")
    ap.add_argument("--stride", type=float, default=DEFAULT_STRIDE_SEC, help="stride in seconds")
    ap.add_argument("--alpha", type=float, default=DEFAULT_ALPHA, help="location anchor quantile")
    ap.add_argument("--boundary", type=float, default=DEFAULT_BOUNDARY_M, help="boundary shrink distance [m]")
    ap.add_argument("--jerk_thr", type=float, default=DEFAULT_JERK_THR, help="jerk threshold for JBR metric")
    ap.add_argument("--mu", type=float, default=DEFAULT_MU, help="friction coefficient for DRAC/PSD")
    ap.add_argument("--grav", type=float, default=DEFAULT_GRAV, help="gravity constant for DRAC/PSD")
    ap.add_argument(
        "--ttc_thr_safe_low",
        type=float,
        default=DEFAULT_TTC_THRESHOLDS[0],
        help="TTC threshold (s) separating safe (class 0) and low-risk (class 1).",
    )
    ap.add_argument(
        "--ttc_thr_low_medium",
        type=float,
        default=DEFAULT_TTC_THRESHOLDS[1],
        help="TTC threshold (s) separating low-risk (class 1) and medium-risk (class 2).",
    )
    ap.add_argument(
        "--ttc_thr_medium_high",
        type=float,
        default=DEFAULT_TTC_THRESHOLDS[2],
        help="TTC threshold (s) separating medium-risk (class 2) and high-risk (class 3).",
    )
    ap.add_argument(
        "--drac_thr_safe_low",
        type=float,
        default=DEFAULT_DRAC_THRESHOLDS[0],
        help="DRAC threshold (m/s^2) between safe (class 0) and low-risk (class 1).",
    )
    ap.add_argument(
        "--drac_thr_low_medium",
        type=float,
        default=DEFAULT_DRAC_THRESHOLDS[1],
        help="DRAC threshold (m/s^2) between low-risk (class 1) and medium-risk (class 2).",
    )
    ap.add_argument(
        "--drac_thr_medium_high",
        type=float,
        default=DEFAULT_DRAC_THRESHOLDS[2],
        help="DRAC threshold (m/s^2) for medium-risk (class 2) versus high-risk (class 3).",
    )
    ap.add_argument(
        "--psd_thr_safe_low",
        type=float,
        default=DEFAULT_PSD_THRESHOLDS[0],
        help="PSD threshold separating safe (class 0) and low-risk (class 1).",
    )
    ap.add_argument(
        "--psd_thr_low_medium",
        type=float,
        default=DEFAULT_PSD_THRESHOLDS[1],
        help="PSD threshold separating low-risk (class 1) and medium-risk (class 2).",
    )
    ap.add_argument(
        "--psd_thr_medium_high",
        type=float,
        default=DEFAULT_PSD_THRESHOLDS[2],
        help="PSD threshold separating medium-risk (class 2) and high-risk (class 3).",
    )
    ap.add_argument(
        "--node_bucket_hz",
        type=float,
        default=DEFAULT_NODE_BUCKET_HZ_TARGET,
        help="Target Hz for node aggregation when averaging SSM exposure buckets.",
    )
    ap.add_argument("--accel_clip", type=float, default=None, help="optional acceleration clipping value")
    ap.add_argument("--keep_empty", action="store_true", help="keep empty windows without events")
    ap.add_argument("--workers", type=int, default=None, help="number of workers (defaults to half CPUs)")
    ap.add_argument("--no_loc_align", action="store_true", help="disable location anchor alignment")
    ap.add_argument("--abs_time", action="store_true", help="export absolute timestamps when available")
    ap.add_argument("--skip_clean", action="store_true", help="skip the 12-feature/3-label cleaning step")
    ap.add_argument("--clean_output", type=str, default=None, help="path for the cleaned CSV output")
    ap.add_argument("--strict_clean", action="store_true", help="strict mode for cleaning (missing columns error)")
    ap.add_argument("--skip_summary", action="store_true", help="do not print summary tables")
    ap.add_argument("--summary_out", type=str, default=None, help="directory to save summary CSVs")
    return ap.parse_args()


def main() -> None:
    freeze_support()  # Windows compatibility when using multiprocessing
    args = parse_args()

    config = HighDPipelineConfig(
        root=Path(args.root).expanduser().resolve(),
        out=Path(args.out).expanduser().resolve(),
        window_sec=float(args.window),
        stride_sec=float(args.stride),
        alpha=float(args.alpha),
        boundary_m=float(args.boundary),
        jerk_thr=float(args.jerk_thr),
        mu=float(args.mu),
        grav=float(args.grav),
        ttc_thresholds=(
            float(args.ttc_thr_safe_low),
            float(args.ttc_thr_low_medium),
            float(args.ttc_thr_medium_high),
        ),
        drac_thresholds=(
            float(args.drac_thr_safe_low),
            float(args.drac_thr_low_medium),
            float(args.drac_thr_medium_high),
        ),
        psd_thresholds=(
            float(args.psd_thr_safe_low),
            float(args.psd_thr_low_medium),
            float(args.psd_thr_medium_high),
        ),
        node_bucket_hz_target=float(args.node_bucket_hz),
        accel_clip=(float(args.accel_clip) if args.accel_clip is not None else None),
        keep_empty=bool(args.keep_empty),
        workers=int(args.workers) if args.workers is not None else DEFAULT_WORKERS,
        align_locations=not bool(args.no_loc_align),
        abs_time=bool(args.abs_time),
        clean=not bool(args.skip_clean),
        clean_output=(Path(args.clean_output).expanduser().resolve() if args.clean_output else None),
        strict_clean=bool(args.strict_clean),
        summary=not bool(args.skip_summary),
        summary_save_dir=(Path(args.summary_out).expanduser().resolve() if args.summary_out else None),
    )

    result = run_highd_pipeline(config)

    if result.clean_path:
        print(f"[OK] Clean dataset ready at: {result.clean_path}")
    elif result.merged_path:
        print(f"[INFO] Clean step skipped. Raw windows CSV: {result.merged_path}")


if __name__ == "__main__":
    main()

