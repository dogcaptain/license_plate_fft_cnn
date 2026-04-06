"""
FFT频域特征提取模块
功能：2D FFT变换、高通/低通/带通滤波器、频谱可视化、逆FFT重建

核心理论：
    图像可视为二维信号 f(x,y)，通过傅里叶变换转到频域 F(u,v)：
    F(u,v) = ΣΣ f(x,y) * exp(-j2π(ux/M + vy/N))

    高频分量 → 边缘、轮廓（字符特征）
    低频分量 → 背景、平滑区域
"""
import numpy as np
import cv2
import matplotlib.pyplot as plt


# ============================================================
#  基础FFT操作
# ============================================================

def fft2d(img):
    """
    对灰度图像执行2D FFT变换并中心化。

    Returns:
        f_shift: 中心化后的频谱（复数）
        magnitude: 幅度谱（取log便于显示）
        phase: 相位谱
    """
    # 确保是灰度图
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    img_float = img.astype(np.float64)

    # 2D FFT
    f = np.fft.fft2(img_float)
    # 中心化：将零频移到中心
    f_shift = np.fft.fftshift(f)

    # 幅度谱（取log增强可视性）
    magnitude = np.log1p(np.abs(f_shift))
    # 相位谱
    phase = np.angle(f_shift)

    return f_shift, magnitude, phase


def ifft2d(f_shift):
    """
    逆FFT：从频域恢复到空间域。

    Args:
        f_shift: 中心化的频谱（已滤波）

    Returns:
        img_back: 恢复的图像（uint8）
    """
    f_ishift = np.fft.ifftshift(f_shift)
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.abs(img_back)
    # 归一化到 0-255
    img_back = np.clip(img_back, 0, 255).astype(np.uint8)
    return img_back


# ============================================================
#  滤波器
# ============================================================

def _distance_matrix(shape):
    """计算频域中每个点到中心的距离矩阵 D(u,v)。"""
    rows, cols = shape
    crow, ccol = rows // 2, cols // 2
    u = np.arange(rows).reshape(-1, 1) - crow
    v = np.arange(cols).reshape(1, -1) - ccol
    return np.sqrt(u ** 2 + v ** 2)


def gaussian_highpass_filter(shape, sigma=10):
    """
    高斯高通滤波器。

    H(u,v) = 1 - exp(-D(u,v)² / (2σ²))

    作用：抑制低频（背景），保留高频（字符边缘）。

    Args:
        shape: 图像尺寸 (rows, cols)
        sigma: 截止频率参数，越小滤波越强

    Returns:
        H: 滤波器矩阵
    """
    D = _distance_matrix(shape)
    H = 1.0 - np.exp(-(D ** 2) / (2 * sigma ** 2))
    return H


def gaussian_lowpass_filter(shape, sigma=10):
    """
    高斯低通滤波器。

    H(u,v) = exp(-D(u,v)² / (2σ²))

    作用：保留低频（平滑），去除高频（噪声/边缘）。
    """
    D = _distance_matrix(shape)
    H = np.exp(-(D ** 2) / (2 * sigma ** 2))
    return H


def gaussian_bandpass_filter(shape, sigma_low=5, sigma_high=30):
    """
    高斯带通滤波器（高通 - 更强的高通 = 带通）。

    作用：保留特定频率范围的信息。
    """
    H_low = gaussian_lowpass_filter(shape, sigma_high)
    H_high = gaussian_highpass_filter(shape, sigma_low)
    H = H_low * H_high
    return H


def ideal_highpass_filter(shape, cutoff=10):
    """理想高通滤波器（阶跃函数，会产生振铃效应，用于对比）。"""
    D = _distance_matrix(shape)
    H = np.zeros(shape, dtype=np.float64)
    H[D > cutoff] = 1.0
    return H


def butterworth_highpass_filter(shape, cutoff=10, order=2):
    """巴特沃斯高通滤波器（比理想滤波器更平滑的过渡）。"""
    D = _distance_matrix(shape)
    # 避免除零
    D[D == 0] = 1e-10
    H = 1.0 / (1.0 + (cutoff / D) ** (2 * order))
    return H


# ============================================================
#  滤波操作
# ============================================================

def apply_filter(img, filter_type="gaussian_high", **kwargs):
    """
    对图像应用频域滤波。

    Args:
        img: 灰度图像
        filter_type: 滤波器类型
            - 'gaussian_high': 高斯高通
            - 'gaussian_low': 高斯低通
            - 'gaussian_band': 高斯带通
            - 'ideal_high': 理想高通
            - 'butterworth_high': 巴特沃斯高通
        **kwargs: 滤波器参数

    Returns:
        filtered: 滤波后的图像（uint8）
        H: 使用的滤波器矩阵
        f_shift: 原始频谱
        f_filtered: 滤波后的频谱
    """
    f_shift, _, _ = fft2d(img)

    filter_funcs = {
        "gaussian_high": gaussian_highpass_filter,
        "gaussian_low": gaussian_lowpass_filter,
        "gaussian_band": gaussian_bandpass_filter,
        "ideal_high": ideal_highpass_filter,
        "butterworth_high": butterworth_highpass_filter,
    }

    if filter_type not in filter_funcs:
        raise ValueError(f"未知滤波器: {filter_type}，可选: {list(filter_funcs.keys())}")

    H = filter_funcs[filter_type](img.shape[:2], **kwargs)

    # 应用滤波
    f_filtered = f_shift * H

    # 逆FFT恢复
    filtered = ifft2d(f_filtered)

    return filtered, H, f_shift, f_filtered


def extract_fft_features(img, sigma=10):
    """
    提取FFT高通滤波特征图（用作CNN的额外输入通道）。

    这是本研究的核心创新点：
    将高通滤波后的频域特征作为CNN的第二个输入通道，
    增强字符边缘信息，提高识别鲁棒性。

    Args:
        img: 灰度图像（uint8 或 float）
        sigma: 高通滤波器参数

    Returns:
        feature_map: 高通滤波特征图（float32, 归一化到0~1）
    """
    if img.dtype == np.float32 or img.dtype == np.float64:
        img_uint8 = (img * 255).astype(np.uint8) if img.max() <= 1.0 else img.astype(np.uint8)
    else:
        img_uint8 = img

    filtered, _, _, _ = apply_filter(img_uint8, "gaussian_high", sigma=sigma)

    # 归一化到 0~1
    feature_map = filtered.astype(np.float32)
    if feature_map.max() > 0:
        feature_map = feature_map / feature_map.max()

    return feature_map


# ============================================================
#  可视化函数
# ============================================================

def plot_fft_pipeline(img, sigma=10, save_path=None):
    """
    绘制完整的FFT处理流程图。

    展示：原图 → 幅度谱 → 高通滤波器 → 滤波后频谱 → 恢复图像
    """
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    f_shift, magnitude, phase = fft2d(img)
    H = gaussian_highpass_filter(img.shape, sigma)
    f_filtered = f_shift * H
    magnitude_filtered = np.log1p(np.abs(f_filtered))
    img_filtered = ifft2d(f_filtered)

    fig, axes = plt.subplots(1, 5, figsize=(20, 4))

    titles = ["原始图像", "FFT幅度谱", "高斯高通滤波器", "滤波后频谱", "逆FFT恢复"]
    images = [img, magnitude, H, magnitude_filtered, img_filtered]
    cmaps = ["gray", "hot", "gray", "hot", "gray"]

    for ax, title, image, cmap in zip(axes, titles, images, cmaps):
        ax.imshow(image, cmap=cmap)
        ax.set_title(title, fontsize=12)
        ax.axis("off")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"图片已保存: {save_path}")
    plt.show()


def plot_filter_comparison(img, save_path=None):
    """
    对比不同滤波器的效果。

    展示：原图 | 高斯高通 | 理想高通 | 巴特沃斯高通 | 高斯低通
    """
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    filters = [
        ("原始图像", img),
        ("高斯高通", apply_filter(img, "gaussian_high", sigma=10)[0]),
        ("理想高通", apply_filter(img, "ideal_high", cutoff=10)[0]),
        ("巴特沃斯高通", apply_filter(img, "butterworth_high", cutoff=10, order=2)[0]),
        ("高斯低通", apply_filter(img, "gaussian_low", sigma=10)[0]),
    ]

    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    for ax, (title, image) in zip(axes, filters):
        ax.imshow(image, cmap="gray")
        ax.set_title(title, fontsize=12)
        ax.axis("off")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_noise_fft_comparison(img, noise_levels=(0, 10, 20, 30, 50), save_path=None):
    """
    对比不同噪声水平下的FFT频谱变化。
    """
    from src.preprocess import add_gaussian_noise

    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    n = len(noise_levels)
    fig, axes = plt.subplots(3, n, figsize=(4 * n, 10))

    for i, sigma in enumerate(noise_levels):
        if sigma == 0:
            noisy = img
        else:
            noisy = add_gaussian_noise(img, sigma)

        _, mag, _ = fft2d(noisy)
        filtered = extract_fft_features(noisy, sigma=10)

        axes[0, i].imshow(noisy, cmap="gray")
        axes[0, i].set_title(f"噪声σ={sigma}", fontsize=11)
        axes[0, i].axis("off")

        axes[1, i].imshow(mag, cmap="hot")
        axes[1, i].set_title("FFT频谱", fontsize=11)
        axes[1, i].axis("off")

        axes[2, i].imshow(filtered, cmap="gray")
        axes[2, i].set_title("高通滤波后", fontsize=11)
        axes[2, i].axis("off")

    axes[0, 0].set_ylabel("含噪图像", fontsize=12)
    axes[1, 0].set_ylabel("频谱", fontsize=12)
    axes[2, 0].set_ylabel("滤波结果", fontsize=12)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_magnitude_and_phase(img, save_path=None):
    """绘制幅度谱和相位谱对比。"""
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    _, magnitude, phase = fft2d(img)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    axes[0].imshow(img, cmap="gray")
    axes[0].set_title("原始图像", fontsize=12)
    axes[0].axis("off")

    axes[1].imshow(magnitude, cmap="hot")
    axes[1].set_title("幅度谱 |F(u,v)|", fontsize=12)
    axes[1].axis("off")

    axes[2].imshow(phase, cmap="hsv")
    axes[2].set_title("相位谱 ∠F(u,v)", fontsize=12)
    axes[2].axis("off")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
