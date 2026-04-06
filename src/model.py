"""
CNN模型定义
功能：车牌字符识别CNN模型，支持空间域和频域两种输入模式

网络结构：
  Conv(in_channels, 32, 3) + ReLU + MaxPool(2)
  Conv(32, 64, 3) + ReLU + MaxPool(2)
  Conv(64, 128, 3) + ReLU
  Flatten
  FC(128*4*4, 256) + ReLU + Dropout(0.5)
  FC(256, NUM_CLASSES)
"""
import torch
import torch.nn as nn

from config import NUM_CLASSES, CHAR_IMG_SIZE


class CharCNN(nn.Module):
    """
    车牌字符识别CNN模型

    Args:
        in_channels: 输入通道数
            - 1: spatial模式（仅灰度图）
            - 2: fft模式（灰度图 + FFT高通滤波特征图）
    """

    def __init__(self, in_channels=1):
        super(CharCNN, self).__init__()
        self.in_channels = in_channels

        # 特征提取器
        self.features = nn.Sequential(
            # Conv1 + ReLU + MaxPool
            # 输入: (in_channels, 20, 20) → 输出: (32, 9, 9)
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),  # 20x20 → 10x10

            # Conv2 + ReLU + MaxPool
            # 输入: (32, 10, 10) → 输出: (64, 4, 4)
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),  # 10x10 → 5x5

            # Conv3 + ReLU
            # 输入: (64, 5, 5) → 输出: (128, 5, 5)
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        # 自适应池化，兼容不同输入尺寸
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))

        # 全连接分类器
        self.classifier = nn.Sequential(
            # Flatten: 128 * 4 * 4 = 2048
            nn.Flatten(),
            # FC1
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            # FC2 (输出层)
            nn.Linear(256, NUM_CLASSES),
        )

    def forward(self, x):
        """
        前向传播

        Args:
            x: 输入张量，shape=(batch_size, in_channels, H, W)

        Returns:
            logits: 输出 logits，shape=(batch_size, NUM_CLASSES)
        """
        x = self.features(x)
        x = self.adaptive_pool(x)
        x = self.classifier(x)
        return x

    def get_model_info(self):
        """返回模型信息"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            "in_channels": self.in_channels,
            "num_classes": NUM_CLASSES,
            "total_params": total_params,
            "trainable_params": trainable_params,
        }


def build_model(mode="spatial"):
    """
    工厂函数：根据模式构建模型

    Args:
        mode: 'spatial' 或 'fft'

    Returns:
        model: CharCNN 实例
    """
    if mode == "spatial":
        in_channels = 1
    elif mode == "fft":
        in_channels = 2
    else:
        raise ValueError(f"未知的模式: {mode}，可选: 'spatial', 'fft'")

    return CharCNN(in_channels=in_channels)


if __name__ == "__main__":
    # 测试模型
    print("=" * 50)
    print("Spatial模式 (in_channels=1)")
    model_spatial = build_model("spatial")
    x_spatial = torch.randn(4, 1, 20, 20)
    out_spatial = model_spatial(x_spatial)
    print(f"  输入: {x_spatial.shape}")
    print(f"  输出: {out_spatial.shape}")
    info = model_spatial.get_model_info()
    print(f"  参数量: {info['total_params']:,}")

    print("=" * 50)
    print("FFT模式 (in_channels=2)")
    model_fft = build_model("fft")
    x_fft = torch.randn(4, 2, 20, 20)
    out_fft = model_fft(x_fft)
    print(f"  输入: {x_fft.shape}")
    print(f"  输出: {out_fft.shape}")
    info = model_fft.get_model_info()
    print(f"  参数量: {info['total_params']:,}")
