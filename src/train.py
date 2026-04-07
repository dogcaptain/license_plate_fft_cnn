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
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
try:
    from torch.cuda.amp import autocast, GradScaler
    HAS_CUDA = True
except ImportError:
    HAS_CUDA = False
    autocast = None
    GradScaler = None
from tqdm import tqdm

# 尝试导入可选的日志库
try:
    from torch.utils.tensorboard import SummaryWriter
    HAS_TENSORBOARD = True
except ImportError:
    HAS_TENSORBOARD = False

try:
    import swanlab
    HAS_SWANLAB = True
except ImportError:
    HAS_SWANLAB = False

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import CHAR_DIR, RESULTS_DIR, BATCH_SIZE, EPOCHS, LEARNING_RATE, NUM_CLASSES, NUM_WORKERS, PIN_MEMORY, USE_AMP
from src.dataset import CharDataset
from src.model import build_model


def train_one_epoch(model, dataloader, criterion, optimizer, device, scaler=None):
    """训练一个epoch，支持混合精度训练"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(dataloader, desc="Training", leave=False)
    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        # 混合精度训练
        if scaler is not None and HAS_CUDA:
            with autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
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


class Logger:
    """统一的日志记录器，支持TensorBoard和SwanLab"""

    def __init__(self, log_tool="tensorboard", log_dir=None, project_name=None, experiment_name=None):
        """
        初始化日志记录器

        Args:
            log_tool: 'tensorboard' 或 'swanlab'
            log_dir: TensorBoard日志目录
            project_name: SwanLab项目名称
            experiment_name: SwanLab实验名称
        """
        self.log_tool = log_tool
        self.writer = None
        self.swanlab_run = None

        if log_tool == "tensorboard":
            if not HAS_TENSORBOARD:
                raise ImportError("TensorBoard未安装，请运行: pip install tensorboard")
            self.writer = SummaryWriter(log_dir=log_dir)
            print(f"  使用 TensorBoard 记录日志: {log_dir}")

        elif log_tool == "swanlab":
            if not HAS_SWANLAB:
                raise ImportError("SwanLab未安装，请运行: pip install swanlab")
            self.swanlab_run = swanlab.init(
                project=project_name or "license-plate-fft-cnn",
                experiment_name=experiment_name or f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                description="车牌字符识别CNN训练",
            )
            print(f"  使用 SwanLab 记录日志")

        else:
            raise ValueError(f"未知的日志工具: {log_tool}，可选: 'tensorboard', 'swanlab'")

    def add_scalar(self, tag, value, step):
        """记录标量值"""
        if self.log_tool == "tensorboard":
            self.writer.add_scalar(tag, value, step)
        elif self.log_tool == "swanlab":
            # SwanLab使用不同的tag格式，转换为字典
            self.swanlab_run.log({tag: value}, step=step)

    def close(self):
        """关闭日志记录器"""
        if self.log_tool == "tensorboard":
            self.writer.close()
        elif self.log_tool == "swanlab":
            self.swanlab_run.finish()


def train(mode="spatial", epochs=None, batch_size=None, lr=None, device=None,
          num_workers=None, pin_memory=None, use_amp=None, log_tool="tensorboard",
          val_interval=5, weight_decay=1e-4, dropout=0.5, label_smoothing=0.1,
          use_multi_gpu=False, gpu_ids=None):
    """
    训练模型

    Args:
        mode: 'spatial' 或 'fft'
        epochs: 训练轮数
        batch_size: 批大小
        lr: 学习率
        device: 训练设备
        num_workers: DataLoader工作进程数
        pin_memory: 是否使用pin_memory加速GPU传输
        use_amp: 是否使用自动混合精度训练
        log_tool: 日志工具 'tensorboard' 或 'swanlab'
        val_interval: 每隔多少个epoch验证一次 (默认1，即每轮都验证)
        weight_decay: 权重衰减
        dropout: Dropout比率
        label_smoothing: 标签平滑系数
        use_multi_gpu: 是否使用多GPU训练
        gpu_ids: 使用的GPU ID列表，如 [0, 1]
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
    if num_workers is None:
        num_workers = NUM_WORKERS
    if pin_memory is None:
        pin_memory = PIN_MEMORY
    if use_amp is None:
        use_amp = USE_AMP

    # 多GPU设置
    num_gpus = torch.cuda.device_count()
    if use_multi_gpu and num_gpus > 1:
        if gpu_ids is None:
            gpu_ids = list(range(num_gpus))
        device = torch.device(f"cuda:{gpu_ids[0]}")
        print(f"  使用多GPU训练: {gpu_ids} ({num_gpus} 张显卡)")
    elif use_multi_gpu and num_gpus <= 1:
        print(f"  [警告] 请求多GPU训练但只有 {num_gpus} 张显卡可用，使用单卡")
        use_multi_gpu = False

    # 确定输入通道数
    in_channels = 1 if mode == "spatial" else 2
    model_name = "baseline" if mode == "spatial" else "fft"

    # 混合精度训练设置
    use_amp = use_amp and HAS_CUDA and torch.cuda.is_available()
    if use_amp:
        try:
            # PyTorch 2.0+ 新API
            scaler = torch.amp.GradScaler('cuda')
        except (AttributeError, TypeError):
            # 旧API兼容
            scaler = GradScaler()
    else:
        scaler = None

    print("=" * 60)
    print(f"开始训练 [{mode.upper()}] 模式")
    print(f"  输入通道: {in_channels}")
    print(f"  批次大小: {batch_size}")
    print(f"  训练轮数: {epochs}")
    print(f"  学习率: {lr}")
    print(f"  权重衰减: {weight_decay}")
    print(f"  Dropout: {dropout}")
    print(f"  标签平滑: {label_smoothing}")
    print(f"  设备: {device}")
    print(f"  多GPU训练: {use_multi_gpu}")
    print(f"  混合精度训练: {use_amp}")
    print(f"  DataLoader workers: {num_workers}")
    print(f"  日志工具: {log_tool}")
    print(f"  验证间隔: 每 {val_interval} 个 epoch")
    print("=" * 60)

    # 创建日志记录器
    log_dir = os.path.join(RESULTS_DIR, "logs", f"{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    logger = Logger(
        log_tool=log_tool,
        log_dir=log_dir,
        project_name="license-plate-fft-cnn",
        experiment_name=f"{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
    )

    # 创建数据集（FFT模式预计算缓存）
    train_dataset = CharDataset(split="train", mode=mode, augmentation=True,
                                 cache_fft=(mode=="fft"), num_workers=num_workers)
    val_dataset = CharDataset(split="val", mode=mode, augmentation=False,
                               cache_fft=(mode=="fft"), num_workers=num_workers)

    # DataLoader配置：Windows下多进程可能有问题，使用单进程更稳定
    # 如果num_workers>0在Windows上出问题，设为0使用主进程加载数据
    if os.name == 'nt' and num_workers > 0:  # Windows
        print("  [警告] Windows系统检测到，建议使用 num_workers=0 避免多进程问题")
        print("  如需多进程加速，请在Linux/Mac上运行")
        # 保持用户设置，但如果出问题可以手动设为0

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory,
        persistent_workers=(num_workers > 0 and os.name != 'nt')
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
        persistent_workers=(num_workers > 0 and os.name != 'nt')
    )

    print(f"训练集样本数: {len(train_dataset)}")
    print(f"验证集样本数: {len(val_dataset)}")

    # 创建模型（传入dropout参数）
    model = build_model(mode=mode, dropout=dropout).to(device)
    info = model.get_model_info()
    print(f"模型参数量: {info['total_params']:,}")

    # 多GPU包装
    if use_multi_gpu and num_gpus > 1:
        model = nn.DataParallel(model, device_ids=gpu_ids)
        print(f"  DataParallel包装完成，使用GPU: {gpu_ids}")

    # 损失函数（带标签平滑）和优化器（带权重衰减）
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # 学习率调度器：OneCycleLR（更快收敛）
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=lr,
        epochs=epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.3,  # 30%时间用于warmup
        anneal_strategy='cos',
        div_factor=25,  # 初始lr = max_lr/25
        final_div_factor=10000,  # 最终lr = max_lr/10000
    )

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
    best_model_path = os.path.join(RESULTS_DIR, f"best_model_{model_name}.pth")

    # 开始训练
    for epoch in range(1, epochs + 1):
        print(f"\nEpoch {epoch}/{epochs}")

        # 训练
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, scaler)

        # 验证（每隔val_interval个epoch验证一次，最后一轮必定验证）
        if epoch % val_interval == 0 or epoch == epochs:
            val_loss, val_acc = validate(model, val_loader, criterion, device)
            print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            print(f"  Val Loss:   {val_loss:.4f}, Val Acc:   {val_acc:.4f}")
        else:
            # 不验证时，使用上一次的验证结果（或0）
            val_loss = 0.0
            val_acc = 0.0
            print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}  (跳过验证)")

        # 更新学习率（OneCycleLR每个step更新）
        scheduler.step()

        # 记录当前学习率
        current_lr = optimizer.param_groups[0]['lr']

        # 记录日志
        epoch_log = {
            "epoch": epoch,
            "train_loss": float(train_loss),
            "train_acc": float(train_acc),
            "val_loss": float(val_loss),
            "val_acc": float(val_acc),
            "lr": float(current_lr),
        }
        train_log["history"].append(epoch_log)

        print(f"  LR: {current_lr:.6f}")

        # 写入日志
        logger.add_scalar("Loss/train", train_loss, epoch)
        logger.add_scalar("Accuracy/train", train_acc, epoch)
        logger.add_scalar("LearningRate", current_lr, epoch)
        if epoch % val_interval == 0 or epoch == epochs:
            logger.add_scalar("Loss/val", val_loss, epoch)
            logger.add_scalar("Accuracy/val", val_acc, epoch)

        # 保存最佳模型（仅在验证时）
        if (epoch % val_interval == 0 or epoch == epochs) and val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            best_model_path = os.path.join(RESULTS_DIR, f"best_model_{model_name}.pth")
            # 多GPU情况下保存原始模型
            model_to_save = model.module if use_multi_gpu else model
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model_to_save.state_dict(),
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

    # 关闭日志记录器
    logger.close()

    print("\n" + "=" * 60)
    print(f"训练完成！")
    print(f"  最佳验证准确率: {best_val_acc:.4f} (Epoch {best_epoch})")
    print(f"  模型保存位置: {best_model_path}")
    if log_tool == "tensorboard":
        print(f"  TensorBoard日志: {log_dir}")
        print(f"  查看命令: tensorboard --logdir={log_dir}")
    elif log_tool == "swanlab":
        print(f"  SwanLab实验已完成")
    print("=" * 60)

    return model, train_log


def main():
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
    parser.add_argument("--num_workers", type=int, default=None, help="DataLoader工作进程数")
    parser.add_argument("--no_amp", action="store_true", help="禁用混合精度训练")
    parser.add_argument(
        "--log_tool",
        type=str,
        default="tensorboard",
        choices=["tensorboard", "swanlab"],
        help="日志工具: tensorboard 或 swanlab",
    )
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="权重衰减 (默认1e-4)")
    parser.add_argument("--dropout", type=float, default=0.5, help="Dropout比率 (默认0.5)")
    parser.add_argument("--label_smoothing", type=float, default=0.1, help="标签平滑系数 (默认0.1)")
    parser.add_argument("--multi_gpu", action="store_true", help="使用多GPU训练")
    parser.add_argument("--gpu_ids", type=str, default=None, help="使用的GPU ID，如 '0,1' (默认使用所有)")
    parser.add_argument(
         "--val_interval",
        type = int,
        default = 5,
        help = "每隔多少个epoch验证一次 (默认1，即每轮都验证)",
    )

    args = parser.parse_args()

    # 解析GPU IDs
    gpu_ids = None
    if args.gpu_ids:
        gpu_ids = [int(x) for x in args.gpu_ids.split(',')]

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
        num_workers=args.num_workers,
        use_amp=not args.no_amp,
        log_tool=args.log_tool,
        val_interval=args.val_interval,
        weight_decay=args.weight_decay,
        dropout=args.dropout,
        label_smoothing=args.label_smoothing,
        use_multi_gpu=args.multi_gpu,
        gpu_ids=gpu_ids,
    )


if __name__ == "__main__":
    main()
