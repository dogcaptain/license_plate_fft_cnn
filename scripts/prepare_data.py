"""
车牌数据集预处理脚本
支持：CCPD2019、CBLPRD-330k

功能：解析车牌图片 → 分割单个字符 → 按类别保存
"""
import os
import sys
import cv2
import numpy as np
from tqdm import tqdm
import shutil

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    CCPD_DIR, CBLPRD_DIR, CHAR_DIR, CHAR_LIST, CHAR_IMG_SIZE,
    PLATE_HEIGHT, PLATE_WIDTH, TRAIN_RATIO, VAL_RATIO, CHAR_TO_IDX
)


# ============================================================
#  CCPD 数据集处理
# ============================================================

def parse_ccpd_filename(filename):
    """解析CCPD文件名，提取车牌四顶点坐标和字符标签。"""
    parts = filename.split("-")
    if len(parts) < 7:
        return None, None

    try:
        vertices_str = parts[3].split("_")
        vertices = []
        for v in vertices_str:
            x, y = v.split("&")
            vertices.append((int(x), int(y)))
    except (ValueError, IndexError):
        return None, None

    try:
        label_str = parts[4].split("_")
        label_indices = [int(l) for l in label_str]
        if len(label_indices) != 7:
            return None, None
    except (ValueError, IndexError):
        return None, None

    return vertices, label_indices


def ccpd_index_to_char_index(position, ccpd_idx):
    """将CCPD的字符索引转换为本项目的字符索引。"""
    if position == 0:
        return ccpd_idx
    else:
        if ccpd_idx < 24:
            return 31 + ccpd_idx
        else:
            return 55 + (ccpd_idx - 24)


def crop_plate(img, vertices):
    """根据四顶点坐标透视变换裁切车牌区域。"""
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
    """将车牌图像均匀分割为7个字符区域。"""
    h, w = plate_img.shape[:2]
    char_width = w / 7.0
    chars = []
    for i in range(7):
        x_start = int(i * char_width)
        x_end = int((i + 1) * char_width)
        char_img = plate_img[:, x_start:x_end]
        char_img = cv2.resize(char_img, CHAR_IMG_SIZE, interpolation=cv2.INTER_AREA)
        chars.append(char_img)
    return chars


def process_ccpd_dataset(subset="ccpd_base", max_images=None):
    """处理CCPD数据集。"""
    subset_dir = os.path.join(CCPD_DIR, subset)
    if not os.path.exists(subset_dir):
        print(f"[错误] 目录不存在: {subset_dir}")
        print(f"请先下载CCPD数据集并放到 {CCPD_DIR} 目录下")
        return

    filenames = [f for f in os.listdir(subset_dir)
                 if f.endswith((".jpg", ".png", ".jpeg"))]

    if max_images:
        filenames = filenames[:max_images]

    print(f"正在处理 CCPD {subset}，共 {len(filenames)} 张图片...")

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
            plate = crop_plate(img, vertices)
            chars = split_characters(plate)

            for pos, (char_img, ccpd_idx) in enumerate(zip(chars, label_indices)):
                char_index = ccpd_index_to_char_index(pos, ccpd_idx)
                if char_index >= len(CHAR_LIST):
                    continue

                label = CHAR_LIST[char_index]
                dir_name = f"{char_index:02d}_{label}"
                save_dir = os.path.join(CHAR_DIR, dir_name)
                os.makedirs(save_dir, exist_ok=True)

                save_name = f"{fname.replace('.jpg', '').replace('.png', '')}_{pos}.jpg"
                save_path = os.path.join(save_dir, save_name)
                cv2.imwrite(save_path, char_img)

            success_count += 1
        except Exception as e:
            fail_count += 1
            continue

    print(f"CCPD处理完成！成功: {success_count}, 失败: {fail_count}")


# ============================================================
#  CBLPRD-330k 数据集处理
# ============================================================

def parse_cblprd_filename(filename):
    """
    解析CBLPRD文件名，提取车牌字符。

    CBLPRD文件名格式：车牌号.jpg
    例如：京A12345.jpg、沪B67890.jpg

    返回：字符列表（7个字符）
    """
    # 去掉扩展名
    name = os.path.splitext(filename)[0]
    # 有些文件名可能包含其他信息，用_分割取第一部分
    name = name.split('_')[0]

    # 解析7个字符
    chars = []
    i = 0
    for c in name:
        if c in CHAR_LIST and len(chars) < 7:
            chars.append(c)

    if len(chars) != 7:
        return None
    return chars


def process_cblprd_dataset(subset="train", max_images=None):
    """
    处理CBLPRD-330k数据集。

    Args:
        subset: 'train', 'val', 或 'test'
        max_images: 最大处理图片数
    """
    subset_dir = os.path.join(CBLPRD_DIR, subset)
    if not os.path.exists(subset_dir):
        print(f"[错误] 目录不存在: {subset_dir}")
        print(f"请先下载CBLPRD-330k数据集并放到 {CBLPRD_DIR} 目录下")
        print(f"下载地址: https://github.com/SunlifeV/CBLPRD-330k")
        return

    filenames = [f for f in os.listdir(subset_dir)
                 if f.endswith((".jpg", ".png", ".jpeg", ".JPG", ".PNG"))]

    if max_images:
        filenames = filenames[:max_images]

    print(f"正在处理 CBLPRD-330k {subset}，共 {len(filenames)} 张图片...")

    success_count = 0
    fail_count = 0

    for fname in tqdm(filenames, desc=f"处理{subset}"):
        chars = parse_cblprd_filename(fname)
        if chars is None:
            fail_count += 1
            continue

        img_path = os.path.join(subset_dir, fname)
        img = cv2.imread(img_path)
        if img is None:
            fail_count += 1
            continue

        try:
            # CBLPRD的图片已经是车牌区域，直接分割字符
            plate_chars = split_characters(img)

            for pos, (char_img, char) in enumerate(zip(plate_chars, chars)):
                if char not in CHAR_TO_IDX:
                    continue

                char_index = CHAR_TO_IDX[char]
                label = CHAR_LIST[char_index]
                dir_name = f"{char_index:02d}_{label}"

                # 根据subset保存到不同子目录
                save_dir = os.path.join(CHAR_DIR, dir_name, subset)
                os.makedirs(save_dir, exist_ok=True)

                save_name = f"{fname.replace('.jpg', '').replace('.png', '')}_{pos}.jpg"
                save_path = os.path.join(save_dir, save_name)
                cv2.imwrite(save_path, char_img)

            success_count += 1
        except Exception as e:
            fail_count += 1
            continue

    print(f"CBLPRD {subset} 处理完成！成功: {success_count}, 失败: {fail_count}")


def process_cblprd_all(max_images_per_split=None):
    """处理CBLPRD所有划分（train/val/test）。"""
    for subset in ["train", "val", "test"]:
        process_cblprd_dataset(subset, max_images_per_split)


# ============================================================
#  通用工具函数
# ============================================================

def split_train_val_test():
    """
    将字符数据集划分为训练集/验证集/测试集。
    在每个类别目录内创建 train/, val/, test/ 子目录。
    """
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
        if len(files) == 0:
            continue

        import random
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
    split_counts = {"train": 0, "val": 0, "test": 0, "other": 0}

    for label_dir in sorted(os.listdir(CHAR_DIR)):
        label_path = os.path.join(CHAR_DIR, label_dir)
        if not os.path.isdir(label_path):
            continue

        # 检查是否有子目录划分
        has_splits = any(os.path.exists(os.path.join(label_path, s))
                        for s in ["train", "val", "test"])

        if has_splits:
            for split in ["train", "val", "test"]:
                split_dir = os.path.join(label_path, split)
                if os.path.exists(split_dir):
                    count = len([f for f in os.listdir(split_dir)
                               if f.endswith((".jpg", ".png"))])
                    split_counts[split] += count
                    total += count
        else:
            count = len([f for f in os.listdir(label_path)
                       if f.endswith((".jpg", ".png"))])
            split_counts["other"] += count
            total += count

    print("\n数据集统计:")
    print(f"  训练集: {split_counts['train']} 张")
    print(f"  验证集: {split_counts['val']} 张")
    print(f"  测试集: {split_counts['test']} 张")
    if split_counts["other"] > 0:
        print(f"  未划分: {split_counts['other']} 张")
    print(f"  总计: {total} 张字符图片")


# ============================================================
#  主程序
# ============================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="车牌数据集预处理")
    parser.add_argument("--dataset", type=str, default="cblprd",
                        choices=["ccpd", "cblprd"],
                        help="数据集类型: ccpd 或 cblprd (默认: cblprd)")
    parser.add_argument("--subset", type=str, default="train",
                        help="子集名称 (CCPD: ccpd_base等; CBLPRD: train/val/test)")
    parser.add_argument("--max_images", type=int, default=None,
                        help="最大处理图片数 (默认: 全部)")
    parser.add_argument("--all", action="store_true",
                        help="处理所有子集 (仅CBLPRD)")
    parser.add_argument("--split", action="store_true",
                        help="划分训练/验证/测试集 (仅对未划分的数据)")
    parser.add_argument("--stats", action="store_true",
                        help="打印数据集统计")
    args = parser.parse_args()

    if args.dataset == "ccpd":
        process_ccpd_dataset(subset=args.subset, max_images=args.max_images)
    else:  # cblprd
        if args.all:
            process_cblprd_all(max_images_per_split=args.max_images)
        else:
            process_cblprd_dataset(subset=args.subset, max_images=args.max_images)

    if args.split:
        split_train_val_test()

    if args.stats:
        print_dataset_stats()

    print("\n使用方式:")
    print("  CCPD:")
    print("    python scripts/prepare_data.py --dataset ccpd --subset ccpd_base --max_images 5000")
    print("  CBLPRD-330k (推荐):")
    print("    python scripts/prepare_data.py --dataset cblprd --all")
    print("    python scripts/prepare_data.py --dataset cblprd --subset train --max_images 10000")
    print("  查看统计:")
    print("    python scripts/prepare_data.py --stats")
