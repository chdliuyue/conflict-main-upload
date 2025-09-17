# -*- coding: utf-8 -*-
"""
1) 读取 CSV（前12列=特征，最后3列=标签0/1/2/3）
2) 按三标签联合分布做分层划分（test_size=0.2），失败回退到单标签分层
3) 仅在训练集上拟合 StandardScaler，对训练/测试做 transform
4) 保存 train.csv / test.csv（标准化后的12特征 + 原始3标签）
5) 打印每个标签的类别“计数 + 占比”（整体/训练/测试）
"""
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from joblib import dump

def make_joint_code(y_mat: np.ndarray) -> np.ndarray:
    a = y_mat[:, 0].astype(int)
    b = y_mat[:, 1].astype(int)
    c = y_mat[:, 2].astype(int)
    return a * 16 + b * 4 + c  # 0..63

def show_count_ratio(tag: str, y: np.ndarray):
    vals, cnt = np.unique(y, return_counts=True)
    ratio = cnt / len(y) if len(y) > 0 else np.zeros_like(cnt, dtype=float)
    cnt_dict   = {int(v): int(c)   for v, c in zip(vals, cnt)}
    ratio_dict = {int(v): float(r) for v, r in zip(vals, ratio)}
    print(f"[{tag}] 类别计数: {cnt_dict}")
    print(f"[{tag}] 类别占比: {ratio_dict}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="../data/highD/all_windows_clean.csv",
                    help="输入CSV路径（前12列=特征，后三列=标签）")
    ap.add_argument("--train_out", default="../data/highD_ratio_40/train.csv", help="训练集输出CSV")
    ap.add_argument("--test_out", default="../data/highD_ratio_40/test.csv", help="测试集输出CSV")
    ap.add_argument("--test_size", type=float, default=0.40, help="测试集比例")
    ap.add_argument("--seed", type=int, default=42, help="随机种子")
    args = ap.parse_args()

    # 1) 读取与列定位
    df = pd.read_csv(args.input)
    if df.shape[1] < 15:
        raise ValueError(f"列数不足：检测到 {df.shape[1]} 列，但需要至少 15 列（12特征+3标签）")

    feat_cols  = list(df.columns[:12])
    label_cols = list(df.columns[-3:])
    print(f"[INFO] 特征列(12): {feat_cols}")
    print(f"[INFO] 标签列(3): {label_cols}")

    # 2) 清洗：仅保留必要列，去 NaN/Inf，标签限定 {0,1,2,3}
    df = df[feat_cols + label_cols].copy()
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=feat_cols + label_cols).reset_index(drop=True)
    for lc in label_cols:
        df[lc] = df[lc].astype(int)
        df = df[df[lc].isin([0, 1, 2, 3])]
    df = df.reset_index(drop=True)
    print(f"[INFO] 清洗后样本数: {len(df)}")

    # —— 整体数据的标签计数/占比 —— #
    for lc in label_cols:
        show_count_ratio(f"All-{lc}", df[lc].to_numpy())

    X = df[feat_cols].to_numpy(dtype=float)
    Y = df[label_cols].to_numpy(dtype=int)

    # 3) 分层划分（优先联合分层）
    y_joint = make_joint_code(Y)
    try:
        sss = StratifiedShuffleSplit(n_splits=1, test_size=args.test_size, random_state=args.seed)
        tr_idx, te_idx = next(sss.split(X, y_joint))
        how = "joint(3-label) stratified split"
    except ValueError as e:
        print(f"[WARN] 联合分层失败：{e} → 回退到单标签分层（{label_cols[0]}）")
        sss = StratifiedShuffleSplit(n_splits=1, test_size=args.test_size, random_state=args.seed)
        tr_idx, te_idx = next(sss.split(X, Y[:, 0]))
        how = f"fallback: single-label stratified ({label_cols[0]})"
    print(f"[INFO] 分层方式: {how} | 测试集比例: {args.test_size}")

    X_train, X_test = X[tr_idx], X[te_idx]
    Y_train, Y_test = Y[tr_idx], Y[te_idx]

    # —— 训练/测试标签计数/占比 —— #
    for j, lc in enumerate(label_cols):
        show_count_ratio(f"Train-{lc}", Y_train[:, j])
        show_count_ratio(f" Test-{lc}", Y_test[:, j])

    # 4) 标准化：仅训练集 fit，再同时 transform 训练/测试
    scaler = StandardScaler()
    X_train_std = scaler.fit_transform(X_train)
    X_test_std  = scaler.transform(X_test)

    # 5) 组装并写出 CSV
    train_df = pd.DataFrame(X_train_std, columns=feat_cols)
    test_df  = pd.DataFrame(X_test_std,  columns=feat_cols)
    for j, lc in enumerate(label_cols):
        train_df[lc] = Y_train[:, j]
        test_df[lc]  = Y_test[:, j]

    train_df.to_csv(args.train_out, index=False)
    test_df.to_csv(args.test_out, index=False)
    print(f"[DONE] 训练集写入: {args.train_out} | 测试集写入: {args.test_out}")


if __name__ == "__main__":
    main()
