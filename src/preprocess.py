"""
图像预处理模块
功能：灰度化、滤波去噪、二值化、添加噪声
"""
import cv2
import numpy as np


def to_grayscale(img):
    """将图像转换为灰度图。"""
    if len(img.shape) == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def denoise_gaussian(img, ksize=3):
    """高斯滤波去噪。"""
    return cv2.GaussianBlur(img, (ksize, ksize), 0)


def denoise_median(img, ksize=3):
    """中值滤波去噪（对椒盐噪声效果好）。"""
    return cv2.medianBlur(img, ksize)


def denoise_bilateral(img, d=9, sigma_color=75, sigma_space=75):
    """双边滤波去噪（保边去噪）。"""
    return cv2.bilateralFilter(img, d, sigma_color, sigma_space)


def binarize(img, method="otsu"):
    """
    图像二值化。

    Args:
        img: 灰度图像
        method: 'otsu' 或 'adaptive'
    """
    gray = to_grayscale(img)
    if method == "otsu":
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    elif method == "adaptive":
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2
        )
    else:
        raise ValueError(f"未知的二值化方法: {method}")
    return binary


def add_gaussian_noise(img, sigma=25):
    """
    添加高斯噪声（用于实验对比）。

    Args:
        img: 输入图像
        sigma: 噪声标准差
    """
    noise = np.random.normal(0, sigma, img.shape).astype(np.float64)
    noisy = img.astype(np.float64) + noise
    noisy = np.clip(noisy, 0, 255).astype(np.uint8)
    return noisy


def add_salt_pepper_noise(img, amount=0.05):
    """添加椒盐噪声。"""
    noisy = img.copy()
    # 盐噪声
    num_salt = int(amount * img.size / 2)
    coords = [np.random.randint(0, i, num_salt) for i in img.shape[:2]]
    noisy[coords[0], coords[1]] = 255
    # 椒噪声
    coords = [np.random.randint(0, i, num_salt) for i in img.shape[:2]]
    noisy[coords[0], coords[1]] = 0
    return noisy


def normalize(img):
    """归一化到 [0, 1]。"""
    return img.astype(np.float32) / 255.0


def preprocess_pipeline(img, denoise_method="gaussian"):
    """
    完整预处理流水线：灰度化 → 去噪 → 归一化。

    Args:
        img: 输入图像（BGR或灰度）
        denoise_method: 去噪方法 ('gaussian', 'median', 'bilateral')

    Returns:
        processed: 预处理后的灰度图（float32, 0~1）
    """
    gray = to_grayscale(img)

    if denoise_method == "gaussian":
        denoised = denoise_gaussian(gray)
    elif denoise_method == "median":
        denoised = denoise_median(gray)
    elif denoise_method == "bilateral":
        denoised = denoise_bilateral(gray)
    else:
        denoised = gray

    return normalize(denoised)
