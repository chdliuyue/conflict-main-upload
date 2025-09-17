import os
import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, roc_auc_score
from sklearn.metrics import brier_score_loss
from sklearn.metrics import cohen_kappa_score
from torch.utils.data import Dataset, DataLoader

from module.mtop_probit import MOCE_Loss, MultiTaskOrderedProbitHead

# 自定义 Dataset 类
class CustomDataset(Dataset):
    def __init__(self, data_path):
        # 读取数据
        data = pd.read_csv(data_path)
        # 提取特征和标签
        self.X = data.iloc[:, :-3].values  # 特征（前12列）
        self.y = data.iloc[:, -3:].values  # 标签（后三列）
        # 转换为 Tensor
        self.X = torch.tensor(self.X, dtype=torch.float32)
        self.y = torch.tensor(self.y, dtype=torch.long)

    def __len__(self):
        # 返回数据集的样本数量
        return len(self.X)

    def __getitem__(self, idx):
        # 返回指定索引的数据样本（特征，标签）
        return self.X[idx], self.y[idx]


class MT_MAON(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, task_names, coef_matrix):
        super(MT_MAON, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.task_names = task_names
        self.coef_matrix = coef_matrix

        # 定义网络结构
        self.head = MultiTaskOrderedProbitHead(self.input_dim, len(self.task_names), self.output_dim, mode='POM')

    def forward(self, x):
        probs, eta_k, tau = self.head(x)

        return probs  # [N, 3, 4]


def train_level_2_model(train_loader, model, criterion, optimizer, num_epochs=20, device='cpu'):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            optimizer.zero_grad()
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        # 打印每个 epoch 的平均损失
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss / len(train_loader):.4f}')

    return model


# 评估模型性能
def evaluate_model(model, test_loader, device='cpu'):
    model.eval()
    with torch.no_grad():
        all_predictions = []
        all_targets = []
        all_prod = []
        for inputs, targets in test_loader:
            # 将数据移到GPU
            inputs, targets = inputs.to(device), targets.to(device)

            # 获取模型的预测
            outputs = model(inputs) # torch.Size([N, 3, 4])
            _, predicted = torch.max(outputs, 2)  # torch.Size([N, 3])

            # 收集所有预测和真实标签
            all_predictions.append(predicted.cpu().numpy())
            all_targets.append(targets.cpu().numpy())
            all_prod.append(outputs.cpu().numpy())

        # 拼接所有批次的预测和真实标签
        all_predictions = np.concatenate(all_predictions)
        all_targets = np.concatenate(all_targets)
        all_prod = np.concatenate(all_prod)
    predictions = []
    results = []
    task_names = ['TTC', 'DRAC', 'PSD']  # 显示任务对应的标签名
    for task in range(len(task_names)):
        print(f"Evaluating model for task {task_names[task]}...")
        y_task = all_targets[:, task]
        predicted_labels = all_predictions[:, task]
        probas = all_prod[:, task]

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
    ap.add_argument("--out_dir", default="../output/" + ratio_name + "/results_level_2")
    ap.add_argument("--coef_file", default="../output/" + ratio_name + "/results_level_0/level_0_coefficients.csv",
                    help="Path to Level 0 coefficients file")
    ap.add_argument("--epochs", type=int, default=300, help="Number of training epochs")
    ap.add_argument("-hidden_dim", type=int, default=128, help="hidden_dim")
    ap.add_argument("--lr", type=float, default=0.01, help="Learning rate")
    ap.add_argument("--device", default='cuda' if torch.cuda.is_available() else 'cpu', help="Device to run the model on")
    args = ap.parse_args()

    # 确保输出目录存在
    os.makedirs(args.out_dir, exist_ok=True)
    # 载入数据
    train_dataset = CustomDataset(args.train)
    test_dataset = CustomDataset(args.test)
    input_dim = train_dataset.X.shape[1]  # 12
    num_classes = 4

    train_loader = DataLoader(train_dataset, batch_size=train_dataset.X.shape[0], shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=test_dataset.X.shape[0], shuffle=False)

    coef_df = pd.read_csv(args.coef_file)[:input_dim]  # 12*3
    task_names = ['TTC', 'DRAC', 'PSD']

    model = MT_MAON(input_dim=input_dim, output_dim=num_classes, hidden_dim=args.hidden_dim, task_names=task_names, coef_matrix=coef_df)
    model.to(args.device)

    # 定义损失函数和优化器
    criterion = MOCE_Loss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # 训练模型
    model = train_level_2_model(train_loader, model, criterion, optimizer, num_epochs=args.epochs, device=args.device)

    # 评估模型
    predictions = evaluate_model(model, test_loader, device=args.device)

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
