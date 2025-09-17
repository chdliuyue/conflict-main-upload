import argparse
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.miscmodels.ordinal_model import OrderedModel
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, roc_auc_score
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import brier_score_loss
from sklearn.metrics import cohen_kappa_score
from copulas.multivariate import GaussianMultivariate  # 使用copulas库的GaussianMultivariate
import os


# 读取和准备数据
def load_data(train_path, test_path):
    # 读取训练数据和测试数据
    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)

    # 提取特征和标签
    X_train = train_data.iloc[:, :-3].values  # 特征（前12列）
    y_train = train_data.iloc[:, -3:].values  # 标签（后三列）
    X_test = test_data.iloc[:, :-3].values  # 特征（前12列）
    y_test = test_data.iloc[:, -3:].values  # 标签（后三列）

    return X_train, y_train, X_test, y_test


# 构建并训练联合有序Logit模型（Level 1）
def train_joint_ordered_model_with_copula(X, y):
    """
    使用联合有序模型（带 Copula）来训练模型
    """
    models = []
    # 使用 copulas 库来建模任务间的相关性（例如，高斯 Copula）
    copula = GaussianMultivariate()
    copula.fit(y)  # 拟合任务间的相关性

    for task in range(y.shape[1]):  # 针对每个任务（TTC, DRAC, PSD）训练模型
        print(f"Training model for task {task}...")

        y_task = y[:, task]

        # 训练OrderedModel模型
        model = OrderedModel(y_task, X, distr='probit')
        result = model.fit(method='bfgs')
        models.append(result)
        print(result.summary())

    return models, copula


# 评估模型
def evaluate_model(models, X_test, y_test, copula):
    """
    评估每个任务的联合有序模型性能
    """
    predictions = []
    results = []
    task_names = ['TTC', 'DRAC', 'PSD']  # 显示任务对应的标签名
    for task, model in enumerate(models):
        print(f"Evaluating model for task {task_names[task]}...")
        y_task = y_test[:, task]
        probas = model.predict(X_test)

        # 获取预测的类别
        predicted_labels = np.argmax(probas, axis=1)

        # 计算混淆矩阵、准确率、F1分数等指标
        cm = confusion_matrix(y_task, predicted_labels)
        accuracy = accuracy_score(y_task, predicted_labels)
        f1 = f1_score(y_task, predicted_labels, average='macro')
        qwk = cohen_kappa_score(y_task, predicted_labels, weights='quadratic')
        ordmae = np.mean(np.abs(y_task - predicted_labels))  # OrdMAE
        nll = -np.sum(np.log(probas[np.arange(len(y_task)), y_task])) / len(y_task)  # NLL
        brier = brier_score_loss(y_task, probas)  # Brier Score (fixed)
        auroc = roc_auc_score(y_task, probas, multi_class='ovr', average='macro')  # AUROC

        # BrdECE (Binned Calibration Error)
        ece = np.abs(probas.max(axis=1) - y_task)
        binned_ece = np.mean(ece)  # Simplified BrdECE calculation

        # 收集每个任务的结果
        predictions.append({
            'task': task_names[task],
            'confusion_matrix': cm,
            'accuracy': accuracy,
            'f1_score': f1,
            'qwk': qwk,
            'ordmae': ordmae,
            'nll': nll,
            'brier': brier,
            'auroc': auroc,
            'brdece': binned_ece
        })

        # 将每个任务的结果按一行打印
        results.append([accuracy, f1, qwk, ordmae, nll, brier, auroc, binned_ece])

    # 打印所有任务的结果，每个任务一行
    # 控制台输出
    print(
        f"{'Task':<8} | {'Accuracy':>10} | {'F1':>10} | {'QWK':>10} | {'OrdMAE':>10} | {'NLL':>10} | {'Brier':>10} | {'AUROC':>10} | {'BrdECE':>10}")
    for i, result in enumerate(results):
        print(f"{task_names[i]:<8} | " + " | ".join([f"{x:>10.4f}" for x in result]))

    return predictions


# 主函数
def main():
    # 设置输入路径、输出路径等参数
    ratio_name = "highD_ratio_20"
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="../data/" + ratio_name + "/train.csv")
    ap.add_argument("--test", default="../data/" + ratio_name + "/test.csv")
    ap.add_argument("--out_dir", default="../output/" + ratio_name + "/results_level_1")
    args = ap.parse_args()

    # 确保输出目录存在
    os.makedirs(args.out_dir, exist_ok=True)

    # 载入数据
    X_train, y_train, X_test, y_test = load_data(args.train, args.test)

    # 训练联合有序模型（Level 1）
    models, copula = train_joint_ordered_model_with_copula(X_train, y_train)

    # 评估模型性能
    predictions = evaluate_model(models, X_test, y_test, copula)

    # 保存评估结果
    results_file = os.path.join(args.out_dir, "evaluation_results.txt")
    with open(results_file, 'w') as f:
        # 输出表头
        f.write("Task | Accuracy | F1 | QWK | OrdMAE | NLL | Brier | AUROC | BrdECE\n")
        # 输出每个任务的评价指标
        for prediction in predictions:
            f.write(f"{prediction['task']} | {prediction['accuracy']:.4f} | {prediction['f1_score']:.4f} | "
                    f"{prediction['qwk']:.4f} | {prediction['ordmae']:.4f} | {prediction['nll']:.4f} | "
                    f"{prediction['brier']:.4f} | {prediction['auroc']:.4f} | {prediction['brdece']:.4f}\n")

    print(f"Evaluation results saved to {results_file}")


if __name__ == "__main__":
    main()
