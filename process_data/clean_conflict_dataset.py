#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Clean processed window CSVs to the 12 core features and 3 conflict labels.

The original repository exposed the cleaning rules only via a CLI script.  To
make the data pipeline composable the logic has been refactored into reusable
functions while keeping the behaviour identical: rows are kept only if all
selected feature and label columns are valid, their associated mask columns (if
present) equal one and the label values are not ``-1``.  The functions return a
``CleanResult`` object describing the outcome so that other modules (such as the
integrated ``process_highD`` pipeline) can access the cleaned dataframe without
re-reading it from disk.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import pandas as pd

from constants import (
    ALLOWED_LABEL_VALUES,
    CORE_FEATURE_COLUMNS,
    FEATURE_MASK_MAP,
    LABEL_COLUMNS,
    LABEL_MASK_MAP,
    available_columns,
)

__all__ = ["CleanResult", "clean_conflict_dataframe", "clean_conflict_csv"]


@dataclass
class CleanResult:
    """Container describing the outcome of :func:`clean_conflict_dataframe`."""

    data: pd.DataFrame
    feature_columns: Tuple[str, ...]
    label_columns: Tuple[str, ...]
    missing_features: Tuple[str, ...]
    missing_labels: Tuple[str, ...]
    dropped_rows: int
    invalid_label_counts: Dict[str, int]


def _candidate_mask_columns(col: str) -> Tuple[str, ...]:
    """Possible mask column names associated with *col* (ordered)."""

    candidates = []
    if col in FEATURE_MASK_MAP:
        candidates.append(FEATURE_MASK_MAP[col])
    if col in LABEL_MASK_MAP:
        candidates.append(LABEL_MASK_MAP[col])
    candidates.append(f"{col}_mask")
    if col.endswith("_cls4"):
        candidates.append(col.replace("_cls4", "_cls_mask"))
    # Remove duplicates while preserving order (dict keys keep insertion order).
    return tuple(dict.fromkeys(candidates))


def _mask_series(df: pd.DataFrame, col: str) -> pd.Series:
    """Return a boolean Series describing whether each row is valid for *col*."""

    for candidate in _candidate_mask_columns(col):
        if candidate in df.columns:
            m = pd.to_numeric(df[candidate], errors="coerce").fillna(0.0)
            return m.astype(int) == 1
    return pd.Series(True, index=df.index)


def clean_conflict_dataframe(
    df: pd.DataFrame,
    *,
    feature_cols: Optional[Sequence[str]] = None,
    label_cols: Optional[Sequence[str]] = None,
    strict: bool = False,
) -> CleanResult:
    """Apply the highD cleaning rules to *df*.

    Parameters
    ----------
    df:
        Dataframe produced by ``process_highD``.
    feature_cols, label_cols:
        Optional custom column lists.  When omitted the defaults (12 features and
        3 labels) are used.  The order of the resulting dataframe follows the
        provided sequence.
    strict:
        If ``True`` missing feature/label columns raise an error instead of being
        silently skipped.
    """

    features = tuple(feature_cols) if feature_cols is not None else CORE_FEATURE_COLUMNS
    labels = tuple(label_cols) if label_cols is not None else LABEL_COLUMNS

    available_feats = available_columns(df.columns, features)
    available_labs = available_columns(df.columns, labels)
    missing_feats = tuple(col for col in features if col not in available_feats)
    missing_labs = tuple(col for col in labels if col not in available_labs)

    if strict and (missing_feats or missing_labs):
        raise ValueError(f"[STRICT] 缺少列：features={missing_feats}, labels={missing_labs}")
    if not available_feats or not available_labs:
        raise ValueError("[ERROR] 可用的特征或标签列为空，请检查输入文件。")

    # Convert once to numeric for validity checks.
    converted = {
        col: pd.to_numeric(df[col], errors="coerce") for col in available_feats + available_labs
    }

    valid = pd.Series(True, index=df.index)
    for col, series in converted.items():
        valid &= series.notna()
        valid &= _mask_series(df, col)

    if available_labs:
        label_frame = pd.concat([converted[col] for col in available_labs], axis=1)
        valid &= ~(label_frame.eq(-1).any(axis=1))

    n_before = len(df)
    clean_df = pd.DataFrame({col: converted[col] for col in available_feats + available_labs})
    clean_df = clean_df.loc[valid].copy()
    n_after = len(clean_df)

    for col in available_feats:
        clean_df[col] = clean_df[col].astype(float)

    invalid_label_counts: Dict[str, int] = {}
    for col in available_labs:
        clean_df[col] = clean_df[col].astype(int)
        invalid_mask = ~clean_df[col].isin(ALLOWED_LABEL_VALUES)
        if invalid_mask.any():
            invalid_label_counts[col] = int(invalid_mask.sum())

    return CleanResult(
        data=clean_df.reset_index(drop=True),
        feature_columns=available_feats,
        label_columns=available_labs,
        missing_features=missing_feats,
        missing_labels=missing_labs,
        dropped_rows=int(n_before - n_after),
        invalid_label_counts=invalid_label_counts,
    )


def clean_conflict_csv(
    src: Path | str,
    dst: Path | str,
    *,
    sep: str = ",",
    encoding: str = "utf-8",
    strict: bool = False,
    feature_cols: Optional[Sequence[str]] = None,
    label_cols: Optional[Sequence[str]] = None,
    summarize: bool = True,
) -> CleanResult:
    """Read *src*, clean it and write the resulting dataframe to *dst*."""

    df = pd.read_csv(src, sep=sep, encoding=encoding, low_memory=False)
    print(f"[INFO] 读取完成：{src}  形状={df.shape}")

    result = clean_conflict_dataframe(
        df,
        feature_cols=feature_cols,
        label_cols=label_cols,
        strict=strict,
    )

    if result.missing_features:
        print(f"[WARN] 缺少特征列（将跳过）：{result.missing_features}")
    if result.missing_labels:
        print(f"[WARN] 缺少标签列（将跳过）：{result.missing_labels}")

    print(
        f"[INFO] 清洗完成：删除行数={result.dropped_rows}，保留行数="
        f"{len(result.data)}；保留列={len(result.data.columns)}"
    )

    for label, count in result.invalid_label_counts.items():
        print(
            f"[WARN] 标签 {label} 存在 {count} 个越界取值（非 0/1/2/3），"
            "已保留到输出，请后续处理。"
        )

    dst_path = Path(dst)
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    result.data.to_csv(dst_path, index=False, sep=sep, encoding=encoding)
    print(f"[OK] 已保存至：{dst_path}  形状={result.data.shape}")

    if summarize:
        from summarize_windows import summarize_windows_df

        summarize_windows_df(
            result.data,
            feature_cols=result.feature_columns,
            label_cols=result.label_columns,
        )

    return result


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="../data/highD/all_windows_10s.csv", help="输入 CSV 路径")
    ap.add_argument("--output", default="../data/highD/all_windows_clean.csv", help="输出 CSV 路径")
    ap.add_argument("--strict", action="store_true", help="缺列时报错（默认宽松跳过）")
    ap.add_argument("--sep", default=",", help="CSV 分隔符（默认 ,）")
    ap.add_argument("--encoding", default="utf-8", help="文件编码（默认 utf-8）")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    clean_conflict_csv(
        src=Path(args.input),
        dst=Path(args.output),
        sep=args.sep,
        encoding=args.encoding,
        strict=args.strict,
    )


if __name__ == "__main__":
    main()

