#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Summarize Octet(+15 after removing w_hat) features and 3 label columns from concatenated windows CSV.

输出三块内容：
1) 特征 + 标签的数值描述：名称、均值、std、最小、最大、有效数量(valid)、总行数(total)
   - 特征：按“有限值（非 NaN/±Inf）”计算
   - 标签：仅统计 mask==1 且标签 ∈ {0,1,2,3} 的样本；不再出现 -1 类别
2) 特征的 mask 统计：优先 *_mask；无则按“是否为有限值”推断
3) 三个标签：有效/无效计数 + 各类别（0,1,2,3）个数（仅在有效样本内统计）

用法：
  作为脚本：python summarize_windows.py path/to/all_windows_10s.csv --outdir path/to/save (可选)
  作为库  ：from summarize_windows import summarize_windows_csv
"""

import argparse
from typing import List, Dict, Optional, Tuple
import numpy as np
import pandas as pd
from pathlib import Path

# -------------------------- 统计列配置（去掉 w_hat） --------------------------
FEATURES: List[str] = [
    "UF","UAS","UD","UAL","DF","DAS","DD","DAL",
    "rq_rel","rk_rel","K_EDIE","QKV_relerr_q","CV_v","E_BRK","JBR"  # 共15个
]

EXTRA_NUMERIC_FOR_STATS = ["TTC_weight_avg", "DRAC_weight_avg",
                           "TTC_exp_s1","TTC_exp_s2","TTC_exp_s3",
                           "DRAC_exp_s1","DRAC_exp_s2","DRAC_exp_s3"]

# 若存在这些 *_mask 列，优先使用；否则按有限值推断有效性
FEATURE_MASKS_MAP: Dict[str, str] = {
    "UAS":"UAS_mask", "DAS":"DAS_mask", "UAL":"UAL_mask", "DAL":"DAL_mask",
    "UD":"UD_mask", "DD":"DD_mask",
    "rq_rel":"rq_rel_mask", "rk_rel":"rk_rel_mask",
    "K_EDIE":"K_EDIE_mask", "QKV_relerr_q":"QKV_relerr_q_mask",
    "CV_v":"CV_v_mask", "E_BRK":"E_BRK_mask", "JBR":"JBR_mask",
    # UF/DF 通常没有专门 mask → 按有限值推断
}

# 三个分类标签及其 mask（只统计 0/1/2/3）
LABELS: List[Tuple[str, str]] = [
    ("TTC_cls4","TTC_cls_mask"),
    ("DRAC_cls4","DRAC_cls_mask"),
    ("PSD_cls4","PSD_cls_mask"),
]

ALLOWED_CLASSES = set([0,1,2,3])

# 打印格式
pd.set_option("display.max_rows", 200)
pd.set_option("display.width", 160)

# -------------------------- 工具函数 --------------------------
def _finite_series(s: pd.Series) -> pd.Series:
    """将列转为数值并把 ±Inf → NaN，返回便于统计的 Series。"""
    return pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan)

def _finite_stats(s: Optional[pd.Series], total_rows: int) -> Dict[str, float]:
    """对单列做均值/Std/最小/最大；仅计算有限值（去除 NaN/±Inf）。"""
    if s is None:
        return dict(mean=np.nan, std=np.nan, min=np.nan, max=np.nan, valid=0, total=total_rows)
    x = _finite_series(s).dropna()
    if x.empty:
        return dict(mean=np.nan, std=np.nan, min=np.nan, max=np.nan, valid=0, total=total_rows)
    return dict(
        mean=float(x.mean()),
        std=float(x.std(ddof=1)) if len(x) > 1 else 0.0,
        min=float(x.min()),
        max=float(x.max()),
        valid=int(x.shape[0]),
        total=total_rows,
    )

def _mask_valid_counts(df: pd.DataFrame, feat: str) -> Tuple[Optional[str], Optional[int], Optional[int]]:
    """返回(mask列名, mask==1计数, mask!=1计数)。若无mask列则返回(None, None, None)。"""
    mcol = FEATURE_MASKS_MAP.get(feat)
    if (mcol is not None) and (mcol in df.columns):
        m = pd.to_numeric(df[mcol], errors="coerce").fillna(0).astype(int)
        return mcol, int((m == 1).sum()), int((m != 1).sum())
    return None, None, None

def _label_valid_mask(df: pd.DataFrame, lab: str, mcol: Optional[str]) -> pd.Series:
    """
    标签有效性：mask==1（若有）且 lab ∈ {0,1,2,3}。
    若无 mask 列，则按 lab 是否在 {0,1,2,3} 且为有限值来判断。
    """
    lab_vals = pd.to_numeric(df[lab], errors="coerce") if (lab in df.columns) else pd.Series([], dtype=float)
    in_set = lab_vals.isin(list(ALLOWED_CLASSES))
    if (mcol is not None) and (mcol in df.columns):
        m = pd.to_numeric(df[mcol], errors="coerce").fillna(0).astype(int) == 1
        return m & in_set
    else:
        return in_set

def _label_stats(df: pd.DataFrame, lab: str, mcol: Optional[str], total_rows: int) -> Dict[str, float]:
    """
    只在有效样本（mask==1 且 lab∈{0,1,2,3}）上计算标签的均值/Std/最小/最大。
    注意：这里的均值/Std 是数值化标签的描述统计，满足你的“统一输出数值描述”的需求。
    """
    if lab not in df.columns:
        return dict(mean=np.nan, std=np.nan, min=np.nan, max=np.nan, valid=0, total=total_rows)
    vm = _label_valid_mask(df, lab, mcol)
    x = pd.to_numeric(df.loc[vm, lab], errors="coerce")
    if x.empty:
        return dict(mean=np.nan, std=np.nan, min=np.nan, max=np.nan, valid=0, total=total_rows)
    return dict(
        mean=float(x.mean()),
        std=float(x.std(ddof=1)) if len(x) > 1 else 0.0,
        min=float(x.min()),
        max=float(x.max()),
        valid=int(x.shape[0]),
        total=total_rows,
    )

# -------------------------- 主函数 --------------------------
def summarize_windows_csv(csv_path: str,
                          save_dir: Optional[str] = None,
                          print_to_console: bool = True) -> Dict[str, pd.DataFrame]:
    """
    读取拼接后的 all_windows_XXs.csv，返回三个 DataFrame，并可选打印/保存：
      - stats_df:  特征 + 标签 的 均值/Std/最小/最大/有效数/总数
      - masks_df:  特征的 mask 统计（若无 *_mask 列，则以有限值推断）
      - labels_df: 三个标签的有效/无效计数 + 各类别（0,1,2,3）计数（仅在有效样本中）
    """
    df = pd.read_csv(csv_path)
    n_total = len(df)

    # ------- (1) 特征 + 标签 数值描述 -------
    rows_stats = []
    # 特征
    for feat in FEATURES:
        s = df[feat] if feat in df.columns else None
        st = _finite_stats(s, n_total)
        rows_stats.append(dict(name=feat, **st))
    # 标签（在有效样本内计算数值描述；不包含 -1）
    for lab, mcol in LABELS:
        st = _label_stats(df, lab, mcol, n_total)
        rows_stats.append(dict(name=lab, **st))
    stats_df = pd.DataFrame(rows_stats, columns=["name","mean","std","min","max","valid","total"])
    # 追加：权重均值与暴露秒（若存在）
    for col in EXTRA_NUMERIC_FOR_STATS:
        if col in df.columns:
            st = _finite_stats(df[col], n_total)
            rows_stats.append(dict(name=col, **st))

    # ------- (2) 特征的 mask 统计 -------
    mask_rows = []
    for feat in FEATURES:
        if feat in df.columns:
            finite_valid = int(_finite_series(df[feat]).notna().sum())
            finite_invalid = n_total - finite_valid
        else:
            finite_valid, finite_invalid = 0, n_total
        mcol, m1, m0 = _mask_valid_counts(df, feat)
        mask_rows.append(dict(
            feature=feat, mask_col=mcol,
            valid_by_finite=finite_valid, invalid_by_finite=finite_invalid,
            mask_ones=(int(m1) if m1 is not None else None),
            mask_zeros=(int(m0) if m0 is not None else None)
        ))
    masks_df = pd.DataFrame(mask_rows, columns=[
        "feature","mask_col","valid_by_finite","invalid_by_finite","mask_ones","mask_zeros"
    ])

    # ------- (3) 三个标签：有效/无效 + 各类别（0,1,2,3） -------
    label_rows = []
    for lab, mcol in LABELS:
        if lab not in df.columns:
            label_rows.append({
                "label": lab, "mask_col": (mcol if mcol in df.columns else None),
                "valid": 0, "invalid": n_total,
                "n_class_0": 0, "n_class_1": 0, "n_class_2": 0, "n_class_3": 0,
            })
            continue

        valid_mask = _label_valid_mask(df, lab, mcol)
        v = int(valid_mask.sum())
        inv = int((~valid_mask).sum())

        vals = pd.to_numeric(df.loc[valid_mask, lab], errors="coerce").astype(int)
        counts = vals.value_counts().to_dict()

        label_rows.append({
            "label": lab,
            "mask_col": (mcol if mcol in df.columns else None),
            "valid": v, "invalid": inv,
            "n_class_0": int(counts.get(0, 0)),
            "n_class_1": int(counts.get(1, 0)),
            "n_class_2": int(counts.get(2, 0)),
            "n_class_3": int(counts.get(3, 0)),
        })
    labels_df = pd.DataFrame(label_rows, columns=[
        "label","mask_col","valid","invalid",
        "n_class_0","n_class_1","n_class_2","n_class_3"
    ])

    # ------- 打印/保存 -------
    if print_to_console:
        print("\n=== 特征与标签：均值 / Std / 最小 / 最大（仅按有限值或有效标签计算） ===")
        print(stats_df.to_string(index=False))
        print("\n=== 特征的 mask 统计（优先 *_mask；无则按有限值推断） ===")
        print(masks_df.to_string(index=False))
        print("\n=== 三个分类标签：有效/无效计数 + 各类别个数（仅在有效样本内统计；无 -1 类） ===")
        print(labels_df.to_string(index=False))
        print("\n================ END OF SUMMARY ================\n")

    out = {"stats": stats_df, "masks": masks_df, "labels": labels_df}
    if save_dir is not None:
        outdir = Path(save_dir); outdir.mkdir(parents=True, exist_ok=True)
        stats_df.to_csv(outdir / "summary_stats.csv", index=False)
        masks_df.to_csv(outdir / "feature_masks.csv", index=False)
        labels_df.to_csv(outdir / "label_counts.csv", index=False)
    return out

# -------------------------- CLI --------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Summarize features (no w_hat) and 3 labels from concatenated CSV")
    ap.add_argument("csv", type=str, help="path to all_windows_XXs.csv")
    ap.add_argument("--outdir", type=str, default=None, help="directory to save CSV summaries")
    ap.add_argument("--no-print", action="store_true", help="do not print tables to console")
    args = ap.parse_args()
    summarize_windows_csv(csv_path=args.csv, save_dir=args.outdir, print_to_console=not args.no_print)
