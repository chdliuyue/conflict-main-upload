#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Summaries for the cleaned highD window dataset.

The statistics focus on the 12 core features and the three four-class conflict
labels.  The helpers operate on dataframes directly which makes it possible to
reuse them both from the CLI and the integrated processing pipeline without
having to re-read intermediate CSV files.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from constants import (
    ALLOWED_LABEL_VALUES,
    CORE_FEATURE_COLUMNS,
    FEATURE_MASK_MAP,
    LABEL_COLUMNS,
    LABEL_MASK_MAP,
    SUMMARY_EXTRA_COLUMNS,
    available_columns,
)

__all__ = ["summarize_windows_df", "summarize_windows_csv"]

pd.set_option("display.max_rows", 200)
pd.set_option("display.width", 160)


def _finite_series(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan)


def _finite_stats(series: Optional[pd.Series], total_rows: int) -> Dict[str, float]:
    if series is None:
        return dict(mean=np.nan, std=np.nan, min=np.nan, max=np.nan, valid=0, total=total_rows)
    vals = _finite_series(series).dropna()
    if vals.empty:
        return dict(mean=np.nan, std=np.nan, min=np.nan, max=np.nan, valid=0, total=total_rows)
    return dict(
        mean=float(vals.mean()),
        std=float(vals.std(ddof=1)) if len(vals) > 1 else 0.0,
        min=float(vals.min()),
        max=float(vals.max()),
        valid=int(vals.shape[0]),
        total=total_rows,
    )


def _mask_valid_counts(
    df: pd.DataFrame,
    feature: str,
    feature_mask_map: Mapping[str, str],
) -> Tuple[Optional[str], Optional[int], Optional[int]]:
    mask_col = feature_mask_map.get(feature)
    if mask_col and mask_col in df.columns:
        m = pd.to_numeric(df[mask_col], errors="coerce").fillna(0).astype(int)
        return mask_col, int((m == 1).sum()), int((m != 1).sum())
    return None, None, None


def _label_valid_mask(
    df: pd.DataFrame,
    lab: str,
    mask_col: Optional[str],
) -> pd.Series:
    if lab not in df.columns:
        return pd.Series(False, index=df.index)
    vals = pd.to_numeric(df[lab], errors="coerce")
    in_set = vals.isin(list(ALLOWED_LABEL_VALUES))
    if mask_col and mask_col in df.columns:
        m = pd.to_numeric(df[mask_col], errors="coerce").fillna(0).astype(int) == 1
        return m & in_set
    return in_set


def _label_stats(
    df: pd.DataFrame,
    lab: str,
    mask_col: Optional[str],
    total_rows: int,
) -> Dict[str, float]:
    if lab not in df.columns:
        return dict(mean=np.nan, std=np.nan, min=np.nan, max=np.nan, valid=0, total=total_rows)
    mask = _label_valid_mask(df, lab, mask_col)
    vals = pd.to_numeric(df.loc[mask, lab], errors="coerce")
    if vals.empty:
        return dict(mean=np.nan, std=np.nan, min=np.nan, max=np.nan, valid=0, total=total_rows)
    return dict(
        mean=float(vals.mean()),
        std=float(vals.std(ddof=1)) if len(vals) > 1 else 0.0,
        min=float(vals.min()),
        max=float(vals.max()),
        valid=int(vals.shape[0]),
        total=total_rows,
    )


def summarize_windows_df(
    df: pd.DataFrame,
    *,
    feature_cols: Optional[Sequence[str]] = None,
    label_cols: Optional[Sequence[str]] = None,
    extra_numeric: Sequence[str] = SUMMARY_EXTRA_COLUMNS,
    feature_mask_map: Mapping[str, str] = FEATURE_MASK_MAP,
    label_mask_map: Mapping[str, str] = LABEL_MASK_MAP,
    print_to_console: bool = True,
) -> Dict[str, pd.DataFrame]:
    """Summarise *df* and return the resulting tables as dataframes."""

    feature_cols = available_columns(df.columns, feature_cols or CORE_FEATURE_COLUMNS)
    label_cols = available_columns(df.columns, label_cols or LABEL_COLUMNS)
    total_rows = len(df)

    # --- (1) Feature + label stats ---
    stats_rows = []
    for feat in feature_cols:
        stats_rows.append(dict(name=feat, **_finite_stats(df.get(feat), total_rows)))
    for lab in label_cols:
        stats_rows.append(dict(name=lab, **_label_stats(df, lab, label_mask_map.get(lab), total_rows)))
    for col in extra_numeric:
        if col in df.columns:
            stats_rows.append(dict(name=col, **_finite_stats(df[col], total_rows)))
    stats_df = pd.DataFrame(stats_rows, columns=["name", "mean", "std", "min", "max", "valid", "total"])

    # --- (2) Feature mask stats ---
    mask_rows = []
    for feat in feature_cols:
        finite_valid = int(_finite_series(df[feat]).notna().sum()) if feat in df.columns else 0
        finite_invalid = total_rows - finite_valid
        mask_col, mask_ones, mask_zeros = _mask_valid_counts(df, feat, feature_mask_map)
        mask_rows.append(
            dict(
                feature=feat,
                mask_col=mask_col,
                valid_by_finite=finite_valid,
                invalid_by_finite=finite_invalid,
                mask_ones=mask_ones,
                mask_zeros=mask_zeros,
            )
        )
    masks_df = pd.DataFrame(
        mask_rows,
        columns=[
            "feature",
            "mask_col",
            "valid_by_finite",
            "invalid_by_finite",
            "mask_ones",
            "mask_zeros",
        ],
    )

    # --- (3) Label counts ---
    label_rows = []
    for lab in label_cols:
        mask_col = label_mask_map.get(lab)
        if lab not in df.columns:
            label_rows.append(
                dict(
                    label=lab,
                    mask_col=mask_col if mask_col in df.columns else None,
                    valid=0,
                    invalid=total_rows,
                    n_class_0=0,
                    n_class_1=0,
                    n_class_2=0,
                    n_class_3=0,
                )
            )
            continue

        valid_mask = _label_valid_mask(df, lab, mask_col)
        vals = pd.to_numeric(df.loc[valid_mask, lab], errors="coerce").astype(int)
        counts = vals.value_counts().to_dict()
        label_rows.append(
            dict(
                label=lab,
                mask_col=mask_col if mask_col in df.columns else None,
                valid=int(valid_mask.sum()),
                invalid=int((~valid_mask).sum()),
                n_class_0=int(counts.get(0, 0)),
                n_class_1=int(counts.get(1, 0)),
                n_class_2=int(counts.get(2, 0)),
                n_class_3=int(counts.get(3, 0)),
            )
        )
    labels_df = pd.DataFrame(
        label_rows,
        columns=[
            "label",
            "mask_col",
            "valid",
            "invalid",
            "n_class_0",
            "n_class_1",
            "n_class_2",
            "n_class_3",
        ],
    )

    if print_to_console:
        print("\n=== 特征与标签：均值 / Std / 最小 / 最大（仅按有限值或有效标签计算） ===")
        print(stats_df.to_string(index=False))
        print("\n=== 特征的 mask 统计（优先 *_mask；无则按有限值推断） ===")
        print(masks_df.to_string(index=False))
        print("\n=== 三个分类标签：有效/无效计数 + 各类别个数（仅在有效样本内统计；无 -1 类） ===")
        print(labels_df.to_string(index=False))
        print("\n================ END OF SUMMARY ================\n")

    return {"stats": stats_df, "masks": masks_df, "labels": labels_df}


def summarize_windows_csv(
    csv_path: str,
    *,
    save_dir: Optional[str] = None,
    print_to_console: bool = True,
    feature_cols: Optional[Sequence[str]] = None,
    label_cols: Optional[Sequence[str]] = None,
) -> Dict[str, pd.DataFrame]:
    df = pd.read_csv(csv_path)
    summaries = summarize_windows_df(
        df,
        feature_cols=feature_cols,
        label_cols=label_cols,
        print_to_console=print_to_console,
    )
    if save_dir is not None:
        outdir = Path(save_dir)
        outdir.mkdir(parents=True, exist_ok=True)
        summaries["stats"].to_csv(outdir / "summary_stats.csv", index=False)
        summaries["masks"].to_csv(outdir / "feature_masks.csv", index=False)
        summaries["labels"].to_csv(outdir / "label_counts.csv", index=False)
    return summaries


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Summarize features and labels from concatenated CSV")
    ap.add_argument("csv", type=str, help="path to all_windows_XXs.csv")
    ap.add_argument("--outdir", type=str, default=None, help="directory to save CSV summaries")
    ap.add_argument("--no-print", action="store_true", help="do not print tables to console")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    summarize_windows_csv(
        csv_path=args.csv,
        save_dir=args.outdir,
        print_to_console=not args.no_print,
    )


if __name__ == "__main__":
    main()

