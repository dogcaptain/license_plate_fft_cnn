"""
模型训练脚本
功能：训练车牌字符识别CNN模型，支持空间域和频域两种模式

支持两种模式：
  - spatial: 纯空间域，仅使用灰度图
  - fft: 空间+频域，灰度图 + FFT高通滤波特征图

输出：
  - results/best_model_baseline.pth (spatial模式)
  - results/best_model_fft.pth (fft模式)
  - results/train_log_baseline.json / train_log_fft.json
"""
import os
import sys
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import CHAR_DIR, RESULTS_DIR, BATCH_SIZE, EPOCHS, LEARNING_RATE, NUM_CLASSES
from src.dataset import CharDataset
from src.model import build_model


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """训练一个epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(dataloader, desc="Training", leave=False)
    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


@torch.no_grad()
def validate(model, dataloader, criterion, device):
    """在验证集上评估"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(dataloader, desc="Validating", leave=False):
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def train(mode="spatial", epochs=None, batch_size=None, lr=None, device=None):
    """
    训练模型

    Args:
        mode: 'spatial' 或 'fft'
        epochs: 训练轮数
        batch_size: 批大小
        lr: 学习率
        device: 训练设备
    """
    # 使用默认配置
    if epochs is None:
        epochs = EPOCHS
    if batch_size is None:
        batch_size = BATCH_SIZE
    if lr is None:
        lr = LEARNING_RATE
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 确定输入通道数
    in_channels = 1 if mode == "spatial" else 2
    model_name = "baseline" if mode == "spatial" else "fft"

    print("=" * 60)
    print(f"开始训练 [{mode.upper()}] 模式")
    print(f"  输入通道: {in_channels}")
    print(f"  批次大小: {batch_size}")
    print(f"  训练轮数: {epochs}")
    print(f"  学习率: {lr}")
    print(f"  设备: {device}")
    print("=" * 60)

    # 创建数据集
    train_dataset = CharDataset(split="train", mode=mode, augmentation=True)
    val_dataset = CharDataset(split="val", mode=mode, augmentation=False)

    # DataLoader设置：Windows下num_workers=0避免多进程问题
    # pin_memory=True加速GPU数据传输（当使用CUDA时）
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True if device.type == "cuda" else False
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True if device.type == "cuda" else False
    )

    print(f"训练集样本数: {len(train_dataset)}")
    print(f"验证集样本数: {len(val_dataset)}")

    # 创建模型
    model = build_model(mode=mode).to(device)
    info = model.get_model_info()
    print(f"模型参数量: {info['total_params']:,}")

    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # 学习率调度器：余弦退火
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # 训练日志
    train_log = {
        "mode": mode,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": lr,
        "in_channels": in_channels,
        "num_classes": NUM_CLASSES,
        "train_samples": len(train_dataset),
        "val_samples": len(val_dataset),
        "total_params": info["total_params"],
        "history": [],
    }

    # 最佳模型跟踪
    best_val_acc = 0.0
    best_epoch = 0

    # 开始训练
    for epoch in range(1, epochs + 1):
        print(f"\nEpoch {epoch}/{epochs}")

        # 训练
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)

        # 验证
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        # 更新学习率
        scheduler.step()

        # 记录日志
        epoch_log = {
            "epoch": epoch,
            "train_loss": float(train_loss),
            "train_acc": float(train_acc),
            "val_loss": float(val_loss),
            "val_acc": float(val_acc),
            "lr": float(scheduler.get_last_lr()[0]),
        }
        train_log["history"].append(epoch_log)

        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"  Val Loss:   {val_loss:.4f}, Val Acc:   {val_acc:.4f}")
        print(f"  LR: {scheduler.get_last_lr()[0]:.6f}")

        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            best_model_path = os.path.join(RESULTS_DIR, f"best_model_{model_name}.pth")
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_acc": val_acc,
                    "val_loss": val_loss,
                    "mode": mode,
                },
                best_model_path,
            )
            print(f"  [保存最佳模型] val_acc={val_acc:.4f}")

    # 保存训练日志
    train_log["best_val_acc"] = float(best_val_acc)
    train_log["best_epoch"] = best_epoch

    log_path = os.path.join(RESULTS_DIR, f"train_log_{model_name}.json")
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(train_log, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 60)
    print(f"训练完成！")
    print(f"  最佳验证准确率: {best_val_acc:.4f} (Epoch {best_epoch})")
    print(f"  模型保存位置: {best_model_path}")
    print(f"  日志保存位置: {log_path}")
    print("=" * 60)

    return model, train_log


def main():
    # 检查CUDA是否可用
    if torch.cuda.is_available():
        print(f"CUDA可用: {torch.cuda.get_device_name(0)}")
        print(f"CUDA版本: {torch.version.cuda}")
        print(f"PyTorch版本: {torch.__version__}")
    else:
        print("警告: CUDA不可用，将使用CPU训练")

    parser = argparse.ArgumentParser(description="车牌字符识别CNN训练")
    parser.add_argument(
        "--mode",
        type=str,
        default="spatial",
        choices=["spatial", "fft"],
        help="输入模式: spatial(纯灰度图) 或 fft(灰度图+FFT特征图)",
    )
    parser.add_argument("--epochs", type=int, default=None, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=None, help="批次大小")
    parser.add_argument("--lr", type=float, default=None, help="学习率")
    parser.add_argument("--device", type=str, default=None, help="训练设备 (cuda/cpu)")

    args = parser.parse_args()

    # 解析设备
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 确保结果目录存在
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # 开始训练
    train(
        mode=args.mode,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=device,
    )


if __name__ == "__main__":
    main()
