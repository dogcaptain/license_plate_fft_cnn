"""
对比实验脚本
功能：一键运行完整的对比实验，评估Baseline CNN与FFT+CNN的性能差异

实验内容：
  实验1：Baseline CNN vs FFT+CNN 准确率对比
  实验2：不同噪声水平下的识别率
  实验3（可选）：不同滤波器效果对比
"""
import os
import sys
import json
import subprocess
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import seaborn as sns

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import RESULTS_DIR, NOISE_LEVELS, BATCH_SIZE, EPOCHS
from src.dataset import CharDataset
from src.model import build_model
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score


# ============================================================
#  实验1：Baseline CNN vs FFT+CNN 准确率对比
# ============================================================

def experiment_accuracy_comparison():
    """
    实验1：对比 Baseline CNN (spatial) 和 FFT+CNN (fft) 的准确率
    """
    print("\n" + "=" * 70)
    print("实验1: Baseline CNN vs FFT+CNN 准确率对比")
    print("=" * 70)

    results = {}
    best_baseline = os.path.join(RESULTS_DIR, "best_model_baseline.pth")
    best_fft = os.path.join(RESULTS_DIR, "best_model_fft.pth")

    # 检查是否已有训练好的模型
    for mode, model_path, model_name in [
        ("spatial", best_baseline, "baseline"),
        ("fft", best_fft, "fft"),
    ]:
        if not os.path.exists(model_path):
            print(f"\n[{model_name}] 模型未找到，正在训练...")
            print(f"  运行: python src/train.py --mode {mode} --epochs {EPOCHS}")
            result = subprocess.run(
                [sys.executable, "src/train.py", "--mode", mode, "--epochs", str(EPOCHS)],
                cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            )
            if result.returncode != 0:
                print(f"[错误] {mode} 模式训练失败")
                continue

        # 评估模型
        print(f"\n评估 [{model_name}] 模型...")
        result = subprocess.run(
            [sys.executable, "src/evaluate.py",
             "--model_path", model_path, "--mode", mode, "--batch_size", str(BATCH_SIZE)],
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        )
        if result.returncode != 0:
            print(f"[错误] {mode} 模式评估失败")

        # 从训练日志中读取准确率
        log_path = os.path.join(RESULTS_DIR, f"train_log_{model_name}.json")
        if os.path.exists(log_path):
            with open(log_path, "r") as f:
                log = json.load(f)
            results[model_name] = {
                "best_val_acc": log.get("best_val_acc", 0),
                "final_val_acc": log["history"][-1]["val_acc"] if log["history"] else 0,
                "final_val_loss": log["history"][-1]["val_loss"] if log["history"] else 0,
            }

    # 绘制准确率对比柱状图
    if results:
        plot_accuracy_comparison(results)
        save_results(results, "accuracy_comparison.json")

    return results


def plot_accuracy_comparison(results, save_path=None):
    """绘制准确率对比柱状图"""
    if not results:
        return

    fig, ax = plt.subplots(figsize=(8, 5))

    names = list(results.keys())
    val_accs = [results[n]["best_val_acc"] for n in names]
    colors = ["#4C72B0", "#DD8452"]

    bars = ax.bar(names, val_accs, color=colors, width=0.5)

    # 添加数值标签
    for bar, acc in zip(bars, val_accs):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.005,
            f"{acc:.4f}",
            ha="center", va="bottom", fontsize=12,
        )

    ax.set_ylabel("验证集准确率", fontsize=12)
    ax.set_xlabel("模型", fontsize=12)
    ax.set_title("Baseline CNN vs FFT+CNN 准确率对比", fontsize=14)
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "accuracy_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"准确率对比图已保存: {path}")


# ============================================================
#  实验2：不同噪声水平下的识别率
# ============================================================

@torch.no_grad()
def evaluate_with_noise(model, test_dataset, noise_sigma, device, batch_size=64):
    """在测试集上添加指定噪声水平并评估"""
    model.eval()
    dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    all_preds = []
    all_labels = []

    for images, labels in dataloader:
        # 添加噪声
        if noise_sigma > 0:
            noise = torch.randn_like(images) * (noise_sigma / 255.0)
            images = torch.clamp(images + noise, 0, 1)

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        _, predicted = outputs.max(1)

        all_preds.extend(predicted.cpu().numpy().tolist())
        all_labels.extend(labels.cpu().numpy().tolist())

    return accuracy_score(all_labels, all_preds)


def experiment_noise_robustness():
    """
    实验2：测试 Baseline CNN 和 FFT+CNN 在不同噪声水平下的识别率
    """
    print("\n" + "=" * 70)
    print("实验2: 噪声鲁棒性测试")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    best_baseline = os.path.join(RESULTS_DIR, "best_model_baseline.pth")
    best_fft = os.path.join(RESULTS_DIR, "best_model_fft.pth")

    results = {name: {"noise_levels": [], "accuracies": []} for name in ["baseline", "fft"]}

    for mode, model_path, model_name in [
        ("spatial", best_baseline, "baseline"),
        ("fft", best_fft, "fft"),
    ]:
        if not os.path.exists(model_path):
            print(f"[跳过] {model_name} 模型不存在，跳过噪声实验")
            continue

        print(f"\n[{model_name}] 噪声鲁棒性测试...")
        model = build_model(mode=mode).to(device)
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])

        # 使用无噪声的测试集，在评估时动态添加噪声
        test_dataset = CharDataset(split="test", mode=mode, augmentation=False)

        for noise_sigma in NOISE_LEVELS:
            acc = evaluate_with_noise(model, test_dataset, noise_sigma, device)
            results[model_name]["noise_levels"].append(noise_sigma)
            results[model_name]["accuracies"].append(acc)
            print(f"  σ={noise_sigma:3d}: {acc:.4f}")

    # 绘制噪声鲁棒性曲线
    if any(results[name]["accuracies"] for name in results):
        plot_noise_robustness(results)
        save_results(results, "noise_experiment.json")

    return results


def plot_noise_robustness(results, save_path=None):
    """绘制噪声鲁棒性曲线"""
    if not any(results[name]["accuracies"] for name in results):
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    colors = {"baseline": "#4C72B0", "fft": "#DD8452"}
    markers = {"baseline": "o", "fft": "s"}

    for name in ["baseline", "fft"]:
        if results[name]["accuracies"]:
            ax.plot(
                results[name]["noise_levels"],
                results[name]["accuracies"],
                color=colors[name],
                marker=markers[name],
                linewidth=2,
                markersize=8,
                label=f"FFT+CNN" if name == "fft" else "Baseline CNN",
            )

    ax.set_xlabel("噪声标准差 σ", fontsize=12)
    ax.set_ylabel("识别准确率", fontsize=12)
    ax.set_title("不同噪声水平下的识别率对比", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    ax.set_xticks(NOISE_LEVELS)

    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "noise_robustness.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"噪声鲁棒性曲线已保存: {path}")


# ============================================================
#  实验3：训练曲线对比
# ============================================================

def experiment_training_curves():
    """实验3：绘制训练曲线（损失和准确率）"""
    print("\n" + "=" * 70)
    print("实验3: 训练曲线对比")
    print("=" * 70)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for model_name, color in [("baseline", "#4C72B0"), ("fft", "#DD8452")]:
        log_path = os.path.join(RESULTS_DIR, f"train_log_{model_name}.json")
        if not os.path.exists(log_path):
            continue

        with open(log_path, "r") as f:
            log = json.load(f)

        history = log["history"]
        epochs = [h["epoch"] for h in history]
        train_loss = [h["train_loss"] for h in history]
        val_loss = [h["val_loss"] for h in history]
        train_acc = [h["train_acc"] for h in history]
        val_acc = [h["val_acc"] for h in history]

        label = "FFT+CNN" if model_name == "fft" else "Baseline CNN"

        # 损失曲线
        axes[0].plot(epochs, train_loss, color=color, linestyle="--", linewidth=1.5, alpha=0.7, label=f"{label} (train)")
        axes[0].plot(epochs, val_loss, color=color, linewidth=2, label=f"{label} (val)")

        # 准确率曲线
        axes[1].plot(epochs, train_acc, color=color, linestyle="--", linewidth=1.5, alpha=0.7, label=f"{label} (train)")
        axes[1].plot(epochs, val_acc, color=color, linewidth=2, label=f"{label} (val)")

    axes[0].set_xlabel("Epoch", fontsize=12)
    axes[0].set_ylabel("Loss", fontsize=12)
    axes[0].set_title("训练与验证损失曲线", fontsize=14)
    axes[0].legend(fontsize=9)
    axes[0].grid(alpha=0.3)

    axes[1].set_xlabel("Epoch", fontsize=12)
    axes[1].set_ylabel("Accuracy", fontsize=12)
    axes[1].set_title("训练与验证准确率曲线", fontsize=14)
    axes[1].legend(fontsize=9)
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "training_curves.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"训练曲线已保存: {path}")


# ============================================================
#  工具函数
# ============================================================

def save_results(results, filename):
    """保存实验结果为JSON"""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, filename)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"实验结果已保存: {path}")


# ============================================================
#  主程序
# ============================================================

def main():
    print("=" * 70)
    print("开始运行对比实验")
    print(f"噪声水平: {NOISE_LEVELS}")
    print(f"训练轮数: {EPOCHS}")
    print("=" * 70)

    os.makedirs(RESULTS_DIR, exist_ok=True)

    # 确保训练两个模型
    print("\n[第一步] 确保两个模型都已训练...")
    for mode, model_name in [("spatial", "baseline"), ("fft", "fft")]:
        model_path = os.path.join(RESULTS_DIR, f"best_model_{model_name}.pth")
        if not os.path.exists(model_path):
            print(f"\n  训练 {model_name} 模型...")
            result = subprocess.run(
                [sys.executable, "src/train.py", "--mode", mode, "--epochs", str(EPOCHS)],
                cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            )
            if result.returncode != 0:
                print(f"[错误] {model_name} 模式训练失败")
        else:
            print(f"  {model_name} 模型已存在，跳过训练")

    # 运行实验
    experiment_accuracy_comparison()
    experiment_noise_robustness()
    experiment_training_curves()

    print("\n" + "=" * 70)
    print("所有实验完成！结果保存在 results/ 目录")
    print("=" * 70)

    # 打印结果汇总
    print("\n实验结果汇总:")
    print("-" * 50)
    for name, fname in [("准确率对比", "accuracy_comparison.json"), ("噪声实验", "noise_experiment.json")]:
        fpath = os.path.join(RESULTS_DIR, fname)
        if os.path.exists(fpath):
            print(f"\n{name}:")
            with open(fpath, "r") as f:
                data = json.load(f)
            if name == "准确率对比":
                for model, vals in data.items():
                    print(f"  {model}: best_val_acc={vals.get('best_val_acc', 'N/A')}")
            elif name == "噪声实验":
                print(f"  {json.dumps(data, indent=4)}")


if __name__ == "__main__":
    main()
