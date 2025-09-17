import os
import argparse
import numpy as np
import pandas as pd
from statsmodels.miscmodels.ordinal_model import OrderedModel

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from module.metrics import evaluate_multitask_predictions, format_results_table


# 读取和准备数据
def load_data(train_path, test_path):
    # 读取训练数据和测试数据
    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)

    # 提取特征和标签
    X_train = train_data.iloc[:, :-3].values  # 特征（前12列）
    y_train = train_data.iloc[:, -3:].values  # 标签（后三列）
    X_test = test_data.iloc[:, :-3].values    # 特征（前12列）
    y_test = test_data.iloc[:, -3:].values    # 标签（后三列）

    return X_train, y_train, X_test, y_test

# 构建并训练PPO有序Logit模型
def train_ppo_model(X, y, output_dir, significance_threshold=0.05):
    """
    使用Partial Proportional Odds（PPO）方法训练有序Logit模型，并保存系数表
    """

    models = []
    coef_matrix = []
    task_names = ['TTC', 'DRAC', 'PSD']  # 显示任务对应的标签名
    for task in range(y.shape[1]):  # 针对每个任务（TTC, DRAC, PSD）训练模型
        print(f"Training model for task {task_names[task]}...")

        # 为PPO方法准备数据
        y_task = y[:, task]

        # 训练OrderedModel模型
        model = OrderedModel(y_task, X, distr='probit')
        result = model.fit(method='bfgs')
        models.append(result)

        # 获取系数表，并添加到系数列表中
        coef_df = pd.DataFrame({
            'feature': model.exog_names,  # 特征名称
            'coef': result.params,        # 对应系数值
            'p_value': result.pvalues     # 对应p值
        })

        # 将系数值为NaN的行设置为P值不显著的系数
        coef_df['coef'] = np.where(coef_df['p_value'] > significance_threshold, np.nan, coef_df['coef'])
        coef_matrix.append(coef_df['coef'].values)

        print(result.summary())

    # 将所有任务的系数表保存为一个 CSV 文件
    coef_matrix = np.array(coef_matrix).T
    feature_names = coef_df['feature'].values
    coef_matrix_df = pd.DataFrame(coef_matrix, columns=task_names, index=feature_names)

    coef_matrix_df.to_csv(
        os.path.join(output_dir, 'level_0_coefficients.csv'), index_label='feature'
    )
    print(f"Level 0 coefficients saved to {output_dir}/level_0_coefficients.csv")

    return models

# 评估模型
def evaluate_model(models, X_test, y_test):
    """
    评估每个任务的PPO模型性能
    """
    task_names = ['TTC', 'DRAC', 'PSD']
    all_probas = []
    for task, model in enumerate(models):
        print(f"Evaluating model for task {task_names[task]}...")
        probas = np.asarray(model.predict(X_test))
        if probas.ndim != 2:
            raise ValueError(f"Expected 2-D probability array, got shape {probas.shape}")
        all_probas.append(probas)

    all_probas = np.stack(all_probas, axis=1)  # [N, M, C]
    metrics = evaluate_multitask_predictions(y_true=y_test, probas=all_probas, task_names=task_names)

    print(format_results_table(metrics))

    return metrics

# 主函数
def main():
    # 设置输入路径、输出路径等参数
    ratio_name = "highD_ratio_20"
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="../data/" + ratio_name + "/train.csv")
    ap.add_argument("--test", default="../data/" + ratio_name + "/test.csv")
    ap.add_argument("--out_dir", default="../output/" + ratio_name + "/results_level_0")
    args = ap.parse_args()

    # 确保输出目录存在
    os.makedirs(args.out_dir, exist_ok=True)

    # 载入数据
    X_train, y_train, X_test, y_test = load_data(args.train, args.test)

    # 训练PPO模型，并保存系数表
    models = train_ppo_model(X_train, y_train, args.out_dir)

    # 评估模型性能
    metrics = evaluate_model(models, X_test, y_test)

    # 保存评估结果
    results_file = os.path.join(args.out_dir, "evaluation_results.txt")
    with open(results_file, 'w') as f:
        # 输出表头
        f.write("Task | Accuracy | F1 | QWK | OrdMAE | NLL | Brier | AUROC | BrdECE\n")
        # 输出每个任务的评价指标
        for metric in metrics:
            f.write(
                f"{metric.task} | {metric.accuracy:.4f} | {metric.f1_score:.4f} | "
                f"{metric.qwk:.4f} | {metric.ordmae:.4f} | {metric.nll:.4f} | "
                f"{metric.brier:.4f} | {metric.auroc:.4f} | {metric.brdece:.4f}\n"
            )

    print(f"Evaluation results saved to {results_file}")

if __name__ == "__main__":
    main()
