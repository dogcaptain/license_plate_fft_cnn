"""
模型评估脚本
功能：在测试集上评估训练好的CNN模型，计算准确率、混淆矩阵、分类报告

输出：
  - results/confusion_matrix_{mode}.png
  - 终端打印分类报告
"""
import os
import sys
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import RESULTS_DIR, CHAR_LIST, BATCH_SIZE
from src.dataset import CharDataset
from src.model import build_model
from sklearn.metrics import classification_report
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 黑体
matplotlib.rcParams['axes.unicode_minus'] = False

def per_class_accuracy(y_true, y_pred, num_classes):
    acc = []
    for i in range(num_classes):
        idx = (y_true == i)
        if idx.sum() == 0:
            acc.append(0)
        else:
            acc.append((y_pred[idx] == i).mean())
    return acc

def top_k_accuracy(model, dataloader, device, k=3):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)

            _, topk = outputs.topk(k, dim=1)
            correct += (topk == labels.unsqueeze(1)).sum().item()
            total += labels.size(0)

    return correct / total

def find_confusion_pairs(cm, top_n=10):
    pairs = []
    for i in range(len(cm)):
        for j in range(len(cm)):
            if i != j and cm[i][j] > 0:
                pairs.append((i, j, cm[i][j]))

    pairs.sort(key=lambda x: x[2], reverse=True)
    return pairs[:top_n]

def plot_confusion_matrix(cm, save_path, labels):
    """
    绘制并保存混淆矩阵热力图

    Args:
        cm: 混淆矩阵 (numpy array)
        save_path: 保存路径
        labels: 类别标签列表
    """
    plt.figure(figsize=(20, 18))
    sns.heatmap(
        cm,
        annot=False,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        cbar=True,
    )
    plt.xlabel("预测标签", fontsize=12)
    plt.ylabel("真实标签", fontsize=12)
    plt.title("混淆矩阵", fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"混淆矩阵已保存: {save_path}")


@torch.no_grad()
def evaluate(model, dataloader, device):
    """
    在测试集上评估模型

    Returns:
        y_true: 真实标签列表
        y_pred: 预测标签列表
        accuracy: 准确率
    """
    model.eval()
    y_true = []
    y_pred = []

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        _, predicted = outputs.max(1)

        y_true.extend(labels.cpu().numpy().tolist())
        y_pred.extend(predicted.cpu().numpy().tolist())

    accuracy = accuracy_score(y_true, y_pred)
    return np.array(y_true), np.array(y_pred), accuracy


def main():
    parser = argparse.ArgumentParser(description="车牌字符识别模型评估")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="模型权重路径，如 results/best_model_fft.pth",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="spatial",
        choices=["spatial", "fft"],
        help="模型输入模式: spatial 或 fft",
    )
    parser.add_argument("--batch_size", type=int, default=64, help="批次大小")
    parser.add_argument("--device", type=str, default=None, help="评估设备 (cuda/cpu)")

    args = parser.parse_args()

    # 解析设备
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 60)
    print("模型评估")
    print(f"  模型路径: {args.model_path}")
    print(f"  模式: {args.mode}")
    print(f"  设备: {device}")
    print("=" * 60)

    # 检查模型文件是否存在
    if not os.path.exists(args.model_path):
        print(f"[错误] 模型文件不存在: {args.model_path}")
        return

    # 确定输入通道数
    in_channels = 1 if args.mode == "spatial" else 2

    # 创建模型
    model = build_model(mode=args.mode).to(device)

    # 加载权重
    checkpoint = torch.load(args.model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"模型已加载 (训练epoch: {checkpoint.get('epoch', 'N/A')}, val_acc: {checkpoint.get('val_acc', 'N/A'):.4f})")

    # 创建测试数据集
    test_dataset = CharDataset(split="test", mode=args.mode, augmentation=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    print(f"测试集样本数: {len(test_dataset)}")

    # 评估
    y_true, y_pred, accuracy = evaluate(model, test_loader, device)

    print(f"\n{'=' * 60}")
    print(f"测试集准确率: {accuracy:.4f} ({accuracy * 100:.2f}%)")
    print(f"{'=' * 60}")

    # 计算混淆矩阵
    cm = confusion_matrix(y_true, y_pred, normalize='true')

    # 绘制混淆矩阵
    model_name = "baseline" if args.mode == "spatial" else "fft"
    cm_path = os.path.join(RESULTS_DIR, f"confusion_matrix_{model_name}.png")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    plot_confusion_matrix(cm, cm_path, labels=CHAR_LIST)

    top3 = top_k_accuracy(model, test_loader, device, k=3)
    top5 = top_k_accuracy(model, test_loader, device, k=5)

    print(f"Top-3 Acc: {top3:.4f}")
    print(f"Top-5 Acc: {top5:.4f}")

    acc_list = per_class_accuracy(y_true, y_pred, len(CHAR_LIST))

    worst = sorted(enumerate(acc_list), key=lambda x: x[1])[:10]

    for i, acc in worst:
        print(f"{CHAR_LIST[i]}: {acc:.3f}")

    pairs = find_confusion_pairs(cm)

    for i, j, cnt in pairs:
        print(f"{CHAR_LIST[i]} → {CHAR_LIST[j]}: {cnt}")

    # 分类报告
    print("\n分类报告:")
    #print(classification_report(y_true, y_pred, target_names=CHAR_LIST, zero_division=0))
    labels = list(range(len(CHAR_LIST)))

    print(classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=CHAR_LIST,
        zero_division=0
    ))

    # 保存分类报告到文件
    report_path = os.path.join(RESULTS_DIR, f"classification_report_{model_name}.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"模型: {args.model_path}\n")
        f.write(f"模式: {args.mode}\n")
        f.write(f"测试集样本数: {len(test_dataset)}\n")
        f.write(f"准确率: {accuracy:.4f}\n\n")
        f.write("分类报告:\n")
        #f.write(classification_report(y_true, y_pred, target_names=CHAR_LIST, zero_division=0))
    print(f"\n分类报告已保存: {report_path}")


if __name__ == "__main__":
    main()
