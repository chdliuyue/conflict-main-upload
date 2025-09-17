#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
clean_conflict_dataset_v2.py

功能：
- 仅保留 16 个特征 + 3 个四分类标签；
- 行过滤规则：任一保留列为 null → 删除；任一对应 *_mask==0 → 删除；
- 进一步：任一标签值为 -1 → 删除；
- 输出干净 CSV，并调用 summarize_windows() 打印数据摘要（特征 + 标签一起做统计）。

用法：
  python clean_conflict_dataset_v2.py --input 01_windows_10s.csv --output 01_windows_10s_clean.csv
  # 严格模式（缺列即报错）
  python clean_conflict_dataset_v2.py --input ... --output ... --strict
"""

import argparse
from pathlib import Path
import pandas as pd
import numpy as np

FEATURES_16 = [
    "UF","UAS","UD","UAL","DF","DAS","DD","DAL",
    "rq_rel","rk_rel","K_EDIE","K_QV","QKV_relerr_q","CV_v","E_BRK","JBR",
]
FEATURES_12 = [
    "UF","UAS","UD","UAL","DF","DAS","DD","DAL",
    "rq_rel","rk_rel","CV_v","E_BRK",
]

LABELS_3 = ["TTC_cls4","DRAC_cls4","PSD_cls4"]

def find_mask_series(df: pd.DataFrame, col: str) -> pd.Series:
    """
    返回与列 col 对应的掩码列（若存在），不存在则返回全 True。
    对标签做了命名兼容：TTC_cls4 -> TTC_cls_mask 等。
    """
    candidates = [f"{col}_mask"]
    if col.endswith("_cls4"):
        candidates.append(col.replace("_cls4", "_cls_mask"))
    for m in candidates:
        if m in df.columns:
            return (df[m] == 1)
    return pd.Series(True, index=df.index)

def summarize_windows(df: pd.DataFrame, feature_cols, label_cols):
    """
    轻量摘要：把“特征 + 三个标签”统一统计 mean/std/min/max/valid，
    并单独列出三标签的类别频数。
    """
    print("\n=== 统计摘要：特征 + 标签（有效样本） ===")
    cols = list(feature_cols) + list(label_cols)
    rows = []
    for c in cols:
        s = pd.to_numeric(df[c], errors="coerce")  # 防御性转换
        rows.append({
            "name":  c,
            "mean":  s.mean(),
            "std":   s.std(ddof=1),
            "min":   s.min(),
            "max":   s.max(),
            "valid": int(s.notna().sum()),
        })
    summary = pd.DataFrame(rows, columns=["name","mean","std","min","max","valid"])
    print(summary.to_string(index=False, float_format=lambda x: f"{x:0.6f}"))

    print("\n=== 标签分布（四分类；已剔除 -1） ===")
    for y in label_cols:
        vc = df[y].value_counts(dropna=False).sort_index()
        print(f"{y}:")
        print(vc.to_string())
        print()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="../data/highD/all_windows_10s.csv", help="输入 CSV 路径")
    ap.add_argument("--output", default="../data/highD/all_windows_clean.csv", help="输出 CSV 路径")
    ap.add_argument("--strict", action="store_true", help="缺列时报错（默认宽松跳过）")
    ap.add_argument("--sep", default=",", help="CSV 分隔符（默认 ,）")
    ap.add_argument("--encoding", default="utf-8", help="文件编码（默认 utf-8）")
    args = ap.parse_args()

    src = Path(args.input)
    dst = Path(args.output)

    df = pd.read_csv(src, sep=args.sep, encoding=args.encoding, low_memory=False)
    print(f"[INFO] 读取完成：{src}  形状={df.shape}")

    # ——列存在性检查——
    missing_feats = [c for c in FEATURES_12 if c not in df.columns]
    missing_labs  = [c for c in LABELS_3   if c not in df.columns]
    if (missing_feats or missing_labs) and args.strict:
        raise ValueError(f"[STRICT] 缺少列：features={missing_feats}, labels={missing_labs}")

    feat_cols = [c for c in FEATURES_12 if c in df.columns]
    lab_cols  = [c for c in LABELS_3   if c in df.columns]
    if missing_feats:
        print(f"[WARN] 缺少特征列（将跳过）：{missing_feats}")
    if missing_labs:
        print(f"[WARN] 缺少标签列（将跳过）：{missing_labs}")
    if not feat_cols or not lab_cols:
        raise ValueError("[ERROR] 可用的特征或标签列为空，请检查输入文件。")

    # ——构造有效性掩码：notna 且 mask==1 且 标签 != -1——
    valid = pd.Series(True, index=df.index)
    for col in feat_cols + lab_cols:
        valid &= df[col].notna()
        valid &= find_mask_series(df, col)

    # 额外规则：任一标签为 -1 → 删除
    if set(lab_cols):
        lab_mat = pd.concat([df[y] for y in lab_cols], axis=1)
        valid &= ~(lab_mat.eq(-1).any(axis=1))

    n_before = len(df)
    df_clean = df.loc[valid, feat_cols + lab_cols].copy()
    n_after = len(df_clean)

    print(f"[INFO] 清洗完成：删除行数={n_before - n_after}，保留行数={n_after}；保留列={len(df_clean.columns)}")
    # 简单标签值检查
    for y in lab_cols:
        bad = ~df_clean[y].isin([0,1,2,3])
        if bad.any():
            print(f"[WARN] 标签 {y} 存在 {int(bad.sum())} 个越界取值（非 0/1/2/3），已保留到输出，请后续处理。")

    # ——保存——
    dst.parent.mkdir(parents=True, exist_ok=True)
    df_clean.to_csv(dst, index=False, sep=args.sep, encoding=args.encoding)
    print(f"[OK] 已保存至：{dst}  形状={df_clean.shape}")

    # ——调用一次 summarize_windows ——
    summarize_windows(df_clean, feature_cols=feat_cols, label_cols=lab_cols)

if __name__ == "__main__":
    main()
