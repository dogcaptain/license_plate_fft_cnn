"""全局配置文件"""
import os

# === 路径配置 ===
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")

# CBLPRD-330k 数据集路径
CBLPRD_DIR = os.path.join(DATA_DIR, "CBLPRD-330k")
CBLPRD_TRAIN_TXT = os.path.join(DATA_DIR, "train.txt")
CBLPRD_VAL_TXT = os.path.join(DATA_DIR, "val.txt")

# 字符图片目录（分割后的数据集）
CHAR_DIR = os.path.join(DATA_DIR, "characters")

# 输出目录
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")

# === 图像配置 ===
CHAR_IMG_SIZE = (20, 20)   # 字符图片统一尺寸

# === 字符映射 ===
# 省份简称（31个）
PROVINCES = [
    "皖", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑",
    "苏", "浙", "京", "闽", "赣", "鲁", "豫", "鄂", "湘", "粤",
    "桂", "琼", "川", "贵", "云", "藏", "陕", "甘", "青", "宁", "新"
]

# 字母（24个，不含I、O）
LETTERS = [
    "A", "B", "C", "D", "E", "F", "G", "H", "J", "K",
    "L", "M", "N", "P", "Q", "R", "S", "T", "U", "V",
    "W", "X", "Y", "Z"
]

# 数字（10个）
DIGITS = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]

# 特殊字符（教练车、挂车等）
SPECIAL_CHARS = ["学", "挂", "警", "港", "澳", "使", "领"]

# 完整字符表：省份(31) + 字母(24) + 数字(10) + 特殊(7) = 72类
CHAR_LIST = PROVINCES + LETTERS + DIGITS + SPECIAL_CHARS
NUM_CLASSES = len(CHAR_LIST)   # 72

# 字符 -> 索引
CHAR_TO_IDX = {c: i for i, c in enumerate(CHAR_LIST)}
IDX_TO_CHAR = {i: c for i, c in enumerate(CHAR_LIST)}

# === 训练配置 ===
BATCH_SIZE = 256          # 增大batch size以充分利用GPU
EPOCHS = 30
LEARNING_RATE = 0.001     # 增大学习率配合更大的batch
NUM_WORKERS = 4           # DataLoader默认工作进程数
PIN_MEMORY = True         # 是否使用pin_memory加速GPU传输
USE_AMP = True            # 默认启用混合精度训练
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1

# === FFT配置 ===
HPF_SIGMA = 10             # 高斯高通滤波器截止频率参数
LPF_SIGMA = 10             # 高斯低通滤波器截止频率参数

# === 实验配置 ===
NOISE_LEVELS = [0, 10, 20, 30, 50]     # 高斯噪声标准差
