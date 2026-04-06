"""全局配置文件"""
import os

# === 路径配置 ===
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
CCPD_DIR = os.path.join(DATA_DIR, "CCPD2019")          # CCPD原始数据集目录
CHAR_DIR = os.path.join(DATA_DIR, "characters")          # 分割后的字符图片目录
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")

# === 图像配置 ===
CHAR_IMG_SIZE = (20, 20)   # 字符图片统一尺寸
PLATE_HEIGHT = 140         # 车牌裁切高度
PLATE_WIDTH = 440          # 车牌裁切宽度

# === 字符映射 ===
PROVINCES = [
    "皖", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑",
    "苏", "浙", "京", "闽", "赣", "鲁", "豫", "鄂", "湘", "粤",
    "桂", "琼", "川", "贵", "云", "藏", "陕", "甘", "青", "宁", "新"
]
LETTERS = [
    "A", "B", "C", "D", "E", "F", "G", "H", "J", "K",
    "L", "M", "N", "P", "Q", "R", "S", "T", "U", "V",
    "W", "X", "Y", "Z"
]
DIGITS = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]

# 完整字符表：省份(31) + 字母(24) + 数字(10) = 65类
CHAR_LIST = PROVINCES + LETTERS + DIGITS
NUM_CLASSES = len(CHAR_LIST)   # 65

# 字符 -> 索引
CHAR_TO_IDX = {c: i for i, c in enumerate(CHAR_LIST)}
IDX_TO_CHAR = {i: c for i, c in enumerate(CHAR_LIST)}

# === 训练配置 ===
BATCH_SIZE = 64
EPOCHS = 30
LEARNING_RATE = 0.001
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1

# === FFT配置 ===
HPF_SIGMA = 10             # 高斯高通滤波器截止频率参数
LPF_SIGMA = 10             # 高斯低通滤波器截止频率参数

# === 实验配置 ===
NOISE_LEVELS = [0, 10, 20, 30, 50]     # 高斯噪声标准差
