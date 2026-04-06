# 给同学B的AI提示词

## 背景

我们正在做一个**基于FFT频域特征与卷积神经网络的车牌字符识别**的学术研究项目。

- 同学A已经完成了：数据预处理、FFT频域分析、图像预处理模块
- **你需要完成**：CNN模型、训练脚本、实验对比脚本

项目目录结构（A已完成的部分）：
```
license_plate_fft_cnn/
├── config.py              # 全局配置（路径、超参数）
├── requirements.txt       # 依赖列表
├── src/
│   ├── preprocess.py      # 图像预处理（灰度化、去噪）
│   ├── fft_features.py    # FFT频域特征提取（2D FFT、高通滤波）
│   └── dataset.py         # 需要你创建
│   ├── model.py           # 需要你创建
│   ├── train.py           # 需要你创建
│   └── evaluate.py        # 需要你创建
├── scripts/
│   ├── prepare_data.py    # CCPD数据预处理（A已完成）
│   └── run_experiments.py # 需要你创建
├── notebooks/
│   └── visualization.ipynb # 可视化Notebook（A已完成FFT部分）
└── data/
    └── characters/        # 分割后的字符图片（运行prepare_data.py生成）
```

## 你需要完成的文件

### 1. `src/dataset.py`

功能：PyTorch Dataset，加载字符图像，支持两种输入模式。

**要求**：
- 自定义 `CharDataset` 类继承 `torch.utils.data.Dataset`
- 支持 `mode='spatial'`（仅灰度图）和 `mode='fft'`（灰度图 + FFT高通滤波特征图）
- 返回格式：
  - spatial模式：`image, label`，image形状 `(1, H, W)`
  - fft模式：`image, label`，image形状 `(2, H, W)`，第0通道是灰度图，第1通道是FFT高通滤波特征图
- 数据增强：随机旋转±5°、随机添加高斯噪声
- 使用 `config.py` 中的 `CHAR_IMG_SIZE`, `CHAR_TO_IDX` 等配置

### 2. `src/model.py`

功能：CNN模型定义。

**要求**：
- 创建 `CharCNN` 类继承 `nn.Module`
- 支持 `in_channels=1`（baseline）和 `in_channels=2`（+FFT特征）
- 网络结构建议（简洁有效）：
  ```
  Conv(in_channels, 32, 3) + ReLU + MaxPool(2)
  Conv(32, 64, 3) + ReLU + MaxPool(2)
  Conv(64, 128, 3) + ReLU
  Flatten
  FC(128*4*4, 256) + ReLU + Dropout(0.5)
  FC(256, 65)  # NUM_CLASSES=65
  ```
- 使用 `config.NUM_CLASSES` 作为输出类别数

### 3. `src/train.py`

功能：训练脚本。

**要求**：
- 加载 `CharDataset`，创建 DataLoader
- 支持两种模式：baseline（纯空间域）和 fft（空间+频域）
- 使用 `CrossEntropyLoss` 和 `Adam` 优化器
- 学习率使用 `config.LEARNING_RATE`
- 训练循环：
  - 记录每个epoch的 train_loss, val_loss, val_acc
  - 保存验证准确率最高的模型到 `results/best_model_baseline.pth` 或 `results/best_model_fft.pth`
  - 保存训练日志为JSON：`results/train_log_baseline.json` 或 `results/train_log_fft.json`
- 命令行参数：`--mode {spatial,fft} --epochs 30 --batch_size 64`
- 打印训练进度（tqdm）

### 4. `src/evaluate.py`

功能：模型评估。

**要求**：
- 加载训练好的模型权重
- 在测试集上计算：准确率、混淆矩阵、分类报告
- 绘制混淆矩阵热力图并保存到 `results/confusion_matrix.png`
- 支持命令行参数：`--model_path results/best_model_fft.pth --mode fft`

### 5. `scripts/run_experiments.py`

功能：一键运行对比实验。

**要求**：
- **实验1**：Baseline CNN vs FFT+CNN 准确率对比
  - 分别训练两个模型
  - 对比最终测试准确率
- **实验2**：不同噪声水平下的识别率
  - 噪声水平：`[0, 10, 20, 30, 50]`（使用 `config.NOISE_LEVELS`）
  - 在测试集上添加对应噪声，测试两个模型的准确率
  - 保存结果到 `results/noise_experiment.json`
- **实验3**：不同滤波器效果对比（可选）
  - 对比高斯高通、理想高通、巴特沃斯高通的效果
- 自动生成对比图表保存到 `results/` 目录

## 关键依赖

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import cv2
import matplotlib.pyplot as plt
import json
from tqdm import tqdm
```

## 与A代码的接口

**从 `fft_features.py` 导入**：
```python
from src.fft_features import extract_fft_features

# 使用方式：
fft_feature = extract_fft_features(gray_image, sigma=10)  # 返回 (20, 20) 的float32数组
```

**从 `preprocess.py` 导入**：
```python
from src.preprocess import preprocess_pipeline, add_gaussian_noise

# 使用方式：
processed = preprocess_pipeline(img, denoise_method='gaussian')  # 返回 (20, 20) float32
```

**从 `config.py` 导入**：
```python
from config import (
    CHAR_DIR, RESULTS_DIR, CHAR_IMG_SIZE, NUM_CLASSES,
    BATCH_SIZE, EPOCHS, LEARNING_RATE, NOISE_LEVELS,
    CHAR_TO_IDX, IDX_TO_CHAR
)
```

## 输出要求

1. 所有模型权重保存到 `results/` 目录
2. 所有训练日志保存为JSON格式
3. 所有图表保存为PNG格式（dpi=150）
4. 代码要有详细的中文注释（这是论文代码，需要展示给老师看）

## 运行流程

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 处理数据（A已完成）
python scripts/prepare_data.py --subset ccpd_base --split

# 3. 训练Baseline模型
python src/train.py --mode spatial --epochs 30

# 4. 训练FFT+CNN模型
python src/train.py --mode fft --epochs 30

# 5. 评估
python src/evaluate.py --model_path results/best_model_baseline.pth --mode spatial
python src/evaluate.py --model_path results/best_model_fft.pth --mode fft

# 6. 运行对比实验
python scripts/run_experiments.py
```

## 论文对应章节

你的代码对应论文的：
- **第四章 实验设计**：实验1、实验2、实验3
- **第五章 实验结果与分析**：训练曲线、准确率对比、噪声鲁棒性分析

请确保实验结果可以生成清晰的图表，用于论文插图。

---

**请直接把这个提示词复制给AI，让它帮你生成上述5个文件的代码。**
