"""
CCPD数据集预处理脚本
功能：解析CCPD文件名 → 裁切车牌 → 分割单个字符 → 按类别保存
"""
import os
import sys
import cv2
import numpy as np
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    CCPD_DIR, CHAR_DIR, CHAR_LIST, CHAR_IMG_SIZE,
    PLATE_HEIGHT, PLATE_WIDTH, TRAIN_RATIO, VAL_RATIO
)


def parse_ccpd_filename(filename):
    """
    解析CCPD文件名，提取车牌四顶点坐标和字符标签。

    CCPD文件名格式（以'-'分割，共7段）：
    0: 面积占比
    1: 倾斜度
    2: 边界框坐标 (左上x&y_右下x&y)
    3: 四顶点坐标 (右下x&y_左下x&y_左上x&y_右上x&y)
    4: 车牌号索引 (7个数字用_分割, 第1个是省份索引, 后6个是字母数字索引)
    5: 亮度
    6: 模糊度

    返回：
        vertices: 四个顶点坐标 [(x,y), ...]
        label_indices: 7个字符对应的索引列表
    """
    parts = filename.split("-")
    if len(parts) < 7:
        return None, None

    # 解析四顶点坐标
    try:
        vertices_str = parts[3].split("_")
        vertices = []
        for v in vertices_str:
            x, y = v.split("&")
            vertices.append((int(x), int(y)))
    except (ValueError, IndexError):
        return None, None

    # 解析车牌字符标签
    try:
        label_str = parts[4].split("_")
        label_indices = [int(l) for l in label_str]
        if len(label_indices) != 7:
            return None, None
    except (ValueError, IndexError):
        return None, None

    return vertices, label_indices


def ccpd_index_to_char_index(position, ccpd_idx):
    """
    将CCPD的字符索引转换为本项目的字符索引。

    CCPD编码规则：
    - 位置0（省份）：索引直接对应省份列表 (0-30)
    - 位置1-6（字母/数字）：索引对应字母表+数字表 (0-33)
      其中 0-23 对应字母A-Z(无I、O), 24-33 对应数字0-9

    本项目编码：省份(0-30) + 字母(31-54) + 数字(55-64)
    """
    if position == 0:
        # 省份，直接映射
        return ccpd_idx
    else:
        # 字母或数字：CCPD中 0-23=字母, 24-33=数字
        if ccpd_idx < 24:
            return 31 + ccpd_idx    # 字母区间
        else:
            return 55 + (ccpd_idx - 24)  # 数字区间


def crop_plate(img, vertices):
    """
    根据四顶点坐标透视变换裁切车牌区域。
    """
    # 顶点顺序：右下、左下、左上、右上 → 转为：左上、右上、右下、左下
    pts_src = np.float32([vertices[2], vertices[3], vertices[0], vertices[1]])
    pts_dst = np.float32([
        [0, 0],
        [PLATE_WIDTH - 1, 0],
        [PLATE_WIDTH - 1, PLATE_HEIGHT - 1],
        [0, PLATE_HEIGHT - 1]
    ])

    M = cv2.getPerspectiveTransform(pts_src, pts_dst)
    plate = cv2.warpPerspective(img, M, (PLATE_WIDTH, PLATE_HEIGHT))
    return plate


def split_characters(plate_img):
    """
    将车牌图像均匀分割为7个字符区域。
    中国车牌格式：[省][字母]·[字母数字x5]
    简单均分方式，每个字符宽度约为 plate_width / 7
    """
    h, w = plate_img.shape[:2]
    char_width = w / 7.0
    chars = []
    for i in range(7):
        x_start = int(i * char_width)
        x_end = int((i + 1) * char_width)
        char_img = plate_img[:, x_start:x_end]
        # 统一缩放到目标尺寸
        char_img = cv2.resize(char_img, CHAR_IMG_SIZE, interpolation=cv2.INTER_AREA)
        chars.append(char_img)
    return chars


def process_ccpd_dataset(subset="ccpd_base", max_images=None):
    """
    处理CCPD数据集的一个子集。

    Args:
        subset: 子集名称，如 'ccpd_base'
        max_images: 最大处理图片数（None表示全部）
    """
    subset_dir = os.path.join(CCPD_DIR, subset)
    if not os.path.exists(subset_dir):
        print(f"[错误] 目录不存在: {subset_dir}")
        print(f"请先下载CCPD数据集并放到 {CCPD_DIR} 目录下")
        print(f"下载地址: https://github.com/detectRecog/CCPD")
        return

    filenames = [f for f in os.listdir(subset_dir)
                 if f.endswith((".jpg", ".png", ".jpeg"))]

    if max_images:
        filenames = filenames[:max_images]

    print(f"正在处理 {subset}，共 {len(filenames)} 张图片...")

    success_count = 0
    fail_count = 0

    for fname in tqdm(filenames, desc=f"处理{subset}"):
        vertices, label_indices = parse_ccpd_filename(fname)
        if vertices is None:
            fail_count += 1
            continue

        img_path = os.path.join(subset_dir, fname)
        img = cv2.imread(img_path)
        if img is None:
            fail_count += 1
            continue

        try:
            # 裁切车牌
            plate = crop_plate(img, vertices)
            # 分割字符
            chars = split_characters(plate)

            # 保存每个字符
            for pos, (char_img, ccpd_idx) in enumerate(zip(chars, label_indices)):
                char_index = ccpd_index_to_char_index(pos, ccpd_idx)
                if char_index >= len(CHAR_LIST):
                    continue

                label = CHAR_LIST[char_index]
                # 省份用拼音首字母做目录名（避免中文路径问题）
                if char_index < 31:
                    dir_name = f"{char_index:02d}_{label}"
                else:
                    dir_name = f"{char_index:02d}_{label}"

                save_dir = os.path.join(CHAR_DIR, dir_name)
                os.makedirs(save_dir, exist_ok=True)

                save_name = f"{fname.replace('.jpg', '')}_{pos}.jpg"
                save_path = os.path.join(save_dir, save_name)
                cv2.imwrite(save_path, char_img)

            success_count += 1
        except Exception as e:
            fail_count += 1
            continue

    print(f"处理完成！成功: {success_count}, 失败: {fail_count}")


def split_train_val_test():
    """
    将字符数据集划分为训练集/验证集/测试集。
    在每个类别目录内创建 train/, val/, test/ 子目录。
    """
    import shutil
    import random

    if not os.path.exists(CHAR_DIR):
        print(f"[错误] 字符目录不存在: {CHAR_DIR}")
        return

    for label_dir in sorted(os.listdir(CHAR_DIR)):
        label_path = os.path.join(CHAR_DIR, label_dir)
        if not os.path.isdir(label_path):
            continue

        # 跳过已经划分过的目录
        if os.path.exists(os.path.join(label_path, "train")):
            continue

        files = [f for f in os.listdir(label_path) if f.endswith((".jpg", ".png"))]
        random.shuffle(files)

        n = len(files)
        n_train = int(n * TRAIN_RATIO)
        n_val = int(n * VAL_RATIO)

        splits = {
            "train": files[:n_train],
            "val": files[n_train:n_train + n_val],
            "test": files[n_train + n_val:]
        }

        for split_name, split_files in splits.items():
            split_dir = os.path.join(label_path, split_name)
            os.makedirs(split_dir, exist_ok=True)
            for f in split_files:
                src = os.path.join(label_path, f)
                dst = os.path.join(split_dir, f)
                shutil.move(src, dst)

    print("数据集划分完成（train/val/test）！")


def print_dataset_stats():
    """打印数据集统计信息。"""
    if not os.path.exists(CHAR_DIR):
        print("字符目录不存在")
        return

    total = 0
    for label_dir in sorted(os.listdir(CHAR_DIR)):
        label_path = os.path.join(CHAR_DIR, label_dir)
        if not os.path.isdir(label_path):
            continue

        count = 0
        for split in ["train", "val", "test"]:
            split_dir = os.path.join(label_path, split)
            if os.path.exists(split_dir):
                count += len([f for f in os.listdir(split_dir) if f.endswith((".jpg", ".png"))])

        if count == 0:
            count = len([f for f in os.listdir(label_path) if f.endswith((".jpg", ".png"))])

        print(f"  {label_dir}: {count} 张")
        total += count

    print(f"\n总计: {total} 张字符图片")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="CCPD数据集预处理")
    parser.add_argument("--subset", type=str, default="ccpd_base",
                        help="CCPD子集名称 (默认: ccpd_base)")
    parser.add_argument("--max_images", type=int, default=None,
                        help="最大处理图片数 (默认: 全部)")
    parser.add_argument("--split", action="store_true",
                        help="是否划分训练/验证/测试集")
    parser.add_argument("--stats", action="store_true",
                        help="打印数据集统计")
    args = parser.parse_args()

    # 处理数据
    process_ccpd_dataset(subset=args.subset, max_images=args.max_images)

    # 划分数据集
    if args.split:
        split_train_val_test()

    # 打印统计
    if args.stats:
        print_dataset_stats()

    print("\n使用方式:")
    print("  1. 处理数据:  python scripts/prepare_data.py --subset ccpd_base --max_images 5000")
    print("  2. 划分数据:  python scripts/prepare_data.py --split")
    print("  3. 查看统计:  python scripts/prepare_data.py --stats")
