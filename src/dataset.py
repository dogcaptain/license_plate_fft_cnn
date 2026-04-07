"""
PyTorch Dataset模块
功能：加载车牌字符图像，支持空间域(spatial)和频域(fft)两种输入模式

spatial模式: 仅灰度图，shape=(1, H, W)
fft模式: 灰度图 + FFT高通滤波特征图，shape=(2, H, W)

优化：FFT特征预计算并缓存，避免训练时重复计算
"""
import os
import torch
import numpy as np
import cv2
from torch.utils.data import Dataset
from torchvision import transforms
from functools import lru_cache

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import CHAR_DIR, CHAR_IMG_SIZE, CHAR_TO_IDX, BATCH_SIZE
from src.fft_features import extract_fft_features
from src.preprocess import add_gaussian_noise


class CharDataset(Dataset):
    """
    车牌字符数据集

    支持两种输入模式:
    - mode='spatial': 仅返回灰度图 (1, H, W)
    - mode='fft': 返回灰度图 + FFT高通滤波特征图 (2, H, W)

    数据增强:
    - 随机旋转 ±5°
    - 随机添加高斯噪声
    """

    def __init__(self, split="train", mode="spatial", noise_sigma=0, augmentation=True,
                 cache_fft=True, num_workers=4):
        """
        初始化数据集

        Args:
            split: 数据集划分 ('train', 'val', 'test')
            mode: 输入模式 ('spatial' 或 'fft')
            noise_sigma: 添加的高斯噪声标准差 (0表示不添加噪声)
            augmentation: 是否启用数据增强 (仅在train模式下生效)
            cache_fft: 是否预计算并缓存FFT特征（显著加速训练）
            num_workers: 预计算FFT特征时的并行工作进程数
        """
        self.split = split
        self.mode = mode
        self.noise_sigma = noise_sigma
        self.augmentation = augmentation and (split == "train")
        self.img_size = CHAR_IMG_SIZE  # (20, 20)
        self.samples = []
        self.class_names = sorted(CHAR_TO_IDX.keys())
        self.cache_fft = cache_fft and (mode == "fft")
        self.fft_cache = {}  # 缓存FFT特征

        self._load_samples()

        # 预计算FFT特征
        if self.cache_fft:
            self._precompute_fft_features(num_workers=num_workers)

    def _load_samples(self):
        """扫描字符目录，构建样本列表"""
        if not os.path.exists(CHAR_DIR):
            raise FileNotFoundError(f"字符目录不存在: {CHAR_DIR}，请先运行 prepare_data.py")

        for char_name in self.class_names:
            char_dir = os.path.join(CHAR_DIR, f"{CHAR_TO_IDX[char_name]:02d}_{char_name}")
            if not os.path.exists(char_dir):
                continue

            # 尝试在 split 子目录下查找
            split_dir = os.path.join(char_dir, self.split)
            if os.path.exists(split_dir):
                split_files = [f for f in os.listdir(split_dir)
                               if f.endswith((".jpg", ".png", ".jpeg"))]
                for f in split_files:
                    self.samples.append((os.path.join(split_dir, f), CHAR_TO_IDX[char_name]))
            else:
                # 如果没有 split 子目录，直接在 char_dir 下查找
                all_files = [f for f in os.listdir(char_dir)
                             if f.endswith((".jpg", ".png", ".jpeg")) and
                             os.path.isfile(os.path.join(char_dir, f))]
                for f in all_files:
                    self.samples.append((os.path.join(char_dir, f), CHAR_TO_IDX[char_name]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """
        获取一个样本

        Returns:
            image: Tensor shape=(1, H, W) 或 (2, H, W)
            label: int，字符类别索引
        """
        img_path, label = self.samples[idx]

        # 读取图像（使用imdecode支持Unicode路径）
        try:
            img_bytes = np.fromfile(img_path, dtype=np.uint8)
            img = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)
            if img is None:
                img = np.zeros((*self.img_size, 3), dtype=np.uint8)
            else:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        except Exception:
            img = np.zeros((*self.img_size, 3), dtype=np.uint8)

        # 转为灰度图
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        # 缩放到固定尺寸
        gray = cv2.resize(gray, (self.img_size[1], self.img_size[0]), interpolation=cv2.INTER_AREA)

        # 数据增强
        if self.augmentation:
            gray = self._augment(gray)

        # 添加噪声
        if self.noise_sigma > 0:
            gray = add_gaussian_noise(gray, sigma=self.noise_sigma)

        # 归一化到 [0, 1]
        gray = gray.astype(np.float32) / 255.0

        # 构建输入图像
        if self.mode == "spatial":
            # spatial模式: (1, H, W)
            image = torch.from_numpy(gray).unsqueeze(0).float()
        elif self.mode == "fft":
            # fft模式: (2, H, W) - [灰度图, FFT高通滤波特征图]
            # 优先使用缓存的FFT特征
            if idx in self.fft_cache:
                fft_feature = self.fft_cache[idx]
            else:
                fft_feature = extract_fft_features(gray, sigma=10)
            # fft_feature 已经是 float32 [0, 1]，shape=(H, W)
            image = torch.from_numpy(
                np.stack([gray, fft_feature], axis=0)
            ).float()
        else:
            raise ValueError(f"未知的模式: {self.mode}，可选: 'spatial', 'fft'")

        label = torch.tensor(label, dtype=torch.long)

        return image, label

    def _precompute_fft_features(self, num_workers=4):
        """预计算所有样本的FFT特征并缓存"""
        from concurrent.futures import ThreadPoolExecutor, as_completed
        from tqdm import tqdm

        print(f"[{self.split}] 预计算FFT特征 (共{len(self.samples)}个样本)...")

        def compute_fft(args):
            idx, img_path = args
            try:
                # 使用imdecode支持Unicode路径
                img_bytes = np.fromfile(img_path, dtype=np.uint8)
                img = cv2.imdecode(img_bytes, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    return idx, None
                gray = img
                gray = cv2.resize(gray, (self.img_size[1], self.img_size[0]),
                                  interpolation=cv2.INTER_AREA)
                # 归一化
                gray = gray.astype(np.float32) / 255.0
                # 计算FFT特征
                fft_feature = extract_fft_features(gray, sigma=10)
                return idx, fft_feature
            except Exception:
                return idx, None

        # 使用多线程并行计算FFT
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = {executor.submit(compute_fft, (i, path))
                      for i, (path, _) in enumerate(self.samples)}

            for future in tqdm(as_completed(futures), total=len(futures), desc="FFT缓存"):
                idx, fft_feature = future.result()
                if fft_feature is not None:
                    self.fft_cache[idx] = fft_feature

        print(f"FFT特征缓存完成: {len(self.fft_cache)}/{len(self.samples)}")

    def _augment(self, img):
        """
        数据增强: 随机旋转 ±5° + 随机添加高斯噪声

        Args:
            img: 灰度图 (H, W), uint8

        Returns:
            img: 增强后的灰度图
        """
        h, w = img.shape

        # 随机旋转 ±5°
        if np.random.random() < 0.5:
            angle = np.random.uniform(-5, 5)
            M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
            img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=0)

        # 随机添加微弱高斯噪声
        if np.random.random() < 0.3:
            noise_sigma_aug = np.random.uniform(1, 5)
            img = add_gaussian_noise(img, sigma=noise_sigma_aug)

        return img

    def get_class_counts(self):
        """返回各类别的样本数量"""
        counts = {}
        for _, label in self.samples:
            char_name = self.class_names[label]
            counts[char_name] = counts.get(char_name, 0) + 1
        return counts
