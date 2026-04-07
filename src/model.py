"""
CNN模型定义
功能：增强版车牌字符识别CNN模型，支持空间域和频域两种输入模式

网络结构（增强版）：
  Conv(in_channels, 64, 3) + BN + ReLU + MaxPool(2)
  Conv(64, 128, 3) + BN + ReLU + MaxPool(2)
  Conv(128, 256, 3) + BN + ReLU
  Conv(256, 512, 3) + BN + ReLU + MaxPool(2)
  GlobalAvgPool
  FC(512, 512) + BN + ReLU + Dropout(0.5)
  FC(512, 256) + BN + ReLU + Dropout(0.3)
  FC(256, NUM_CLASSES)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import NUM_CLASSES, CHAR_IMG_SIZE


class CharCNN(nn.Module):
    """
    增强版车牌字符识别CNN模型

    Args:
        in_channels: 输入通道数
            - 1: spatial模式（仅灰度图）
            - 2: fft模式（灰度图 + FFT高通滤波特征图）
    """

    def __init__(self, in_channels=1, dropout=0.5):
        super(CharCNN, self).__init__()
        self.in_channels = in_channels

        # 特征提取器 - 更深的网络 + BatchNorm
        self.features = nn.Sequential(
            # Block 1: 20x20 -> 10x10
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),

            # Block 2: 10x10 -> 5x5
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),

            # Block 3: 5x5 -> 5x5
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            # Block 4: 5x5 -> 2x2
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )

        # 全局平均池化
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        # 全连接分类器 - 更深的网络 + BatchNorm
        self.classifier = nn.Sequential(
            nn.Flatten(),
            # FC1
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            # FC2
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.6),  # 递减dropout
            # FC3 (输出层)
            nn.Linear(256, NUM_CLASSES),
        )

        # 初始化权重
        self._initialize_weights()

    def _initialize_weights(self):
        """He初始化"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """
        前向传播

        Args:
            x: 输入张量，shape=(batch_size, in_channels, H, W)

        Returns:
            logits: 输出 logits，shape=(batch_size, NUM_CLASSES)
        """
        x = self.features(x)
        x = self.global_pool(x)
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


def build_model(mode="spatial", dropout=0.5):
    """
    工厂函数：根据模式构建模型

    Args:
        mode: 'spatial' 或 'fft'
        dropout: Dropout比率

    Returns:
        model: CharCNN 实例
    """
    if mode == "spatial":
        in_channels = 1
    elif mode == "fft":
        in_channels = 2
    else:
        raise ValueError(f"未知的模式: {mode}，可选: 'spatial', 'fft'")

    return CharCNN(in_channels=in_channels, dropout=dropout)


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
