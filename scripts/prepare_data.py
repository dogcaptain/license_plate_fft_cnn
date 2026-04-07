"""
CBLPRD-330k 车牌数据集预处理脚本
功能：解析标注文件 → 分割单个字符 → 按类别保存

标注格式：CBLPRD-330k/000272981.jpg 粤Z31632D 新能源大型车
"""
import os
import sys
import cv2
import numpy as np
from tqdm import tqdm
import shutil

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    DATA_DIR, CBLPRD_TRAIN_TXT, CBLPRD_VAL_TXT,
    CHAR_DIR, CHAR_LIST, CHAR_IMG_SIZE,
    TRAIN_RATIO, VAL_RATIO, CHAR_TO_IDX
)


def split_characters(plate_img, num_chars=7):
    """将车牌图像均匀分割为指定数量的字符区域。"""
    h, w = plate_img.shape[:2]
    char_width = w / num_chars
    chars = []
    for i in range(num_chars):
        x_start = int(i * char_width)
        x_end = int((i + 1) * char_width)
        char_img = plate_img[:, x_start:x_end]
        char_img = cv2.resize(char_img, CHAR_IMG_SIZE, interpolation=cv2.INTER_AREA)
        chars.append(char_img)
    return chars


def split_characters_8(plate_img):
    """将8位车牌图像均匀分割为8个字符区域（新能源、教练车等）。"""
    return split_characters(plate_img, num_chars=8)


def parse_annotation_line(line):
    """
    解析标注文件的一行。
    格式：CBLPRD-330k/000272981.jpg 粤Z31632D 新能源大型车
    """
    parts = line.strip().split()
    if len(parts) < 2:
        return None, None, None

    img_rel_path = parts[0]
    plate_number = parts[1]
    plate_type = parts[2] if len(parts) > 2 else "未知"

    img_path = os.path.join(DATA_DIR, img_rel_path)
    return img_path, plate_number, plate_type


def extract_chars_from_plate(plate_number):
    """
    从车牌号中提取字符。
    支持7位（普通车牌）和8位（新能源、教练车、挂车等）。
    """
    chars = []
    for c in plate_number:
        if c in CHAR_LIST:
            chars.append(c)

    if len(chars) >= 7:
        return chars
    return None


def process_dataset(subset="train", max_images=None):
    """处理CBLPRD-330k数据集。"""
    if subset == "train":
        anno_file = CBLPRD_TRAIN_TXT
    elif subset == "val":
        anno_file = CBLPRD_VAL_TXT
    else:
        print(f"[错误] 不支持的子集: {subset}，只支持 train 和 val")
        return

    if not os.path.exists(anno_file):
        print(f"[错误] 标注文件不存在: {anno_file}")
        print(f"请先下载CBLPRD-330k数据集并放到 {DATA_DIR} 目录下")
        print(f"下载地址: https://github.com/SunlifeV/CBLPRD-330k")
        return

    with open(anno_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    if max_images:
        lines = lines[:max_images]

    print(f"正在处理 CBLPRD-330k {subset}，共 {len(lines)} 张图片...")

    success_count = 0
    fail_count = 0

    for line in tqdm(lines, desc=f"处理{subset}"):
        img_path, plate_number, plate_type = parse_annotation_line(line)
        if img_path is None:
            fail_count += 1
            continue

        chars = extract_chars_from_plate(plate_number)
        if chars is None:
            fail_count += 1
            continue

        img = cv2.imread(img_path)
        if img is None:
            fail_count += 1
            continue

        try:
            # 根据字符数量决定分割方式
            if len(chars) == 8:
                plate_chars = split_characters_8(img)
            else:
                plate_chars = split_characters(img)

            if len(plate_chars) < 7:
                fail_count += 1
                continue

            # 保存每个字符
            for pos, (char_img, char) in enumerate(zip(plate_chars, chars)):
                if char not in CHAR_TO_IDX:
                    continue

                char_index = CHAR_TO_IDX[char]
                label = CHAR_LIST[char_index]
                dir_name = f"{char_index:02d}_{label}"

                save_dir = os.path.join(CHAR_DIR, dir_name, subset)
                os.makedirs(save_dir, exist_ok=True)

                base_name = os.path.splitext(os.path.basename(img_path))[0]
                save_name = f"{base_name}_{pos}.jpg"
                save_path = os.path.join(save_dir, save_name)
                # 使用Python文件操作保存，避免OpenCV中文路径问题
                is_success, buffer = cv2.imencode(".jpg", char_img)
                if is_success:
                    with open(save_path, 'wb') as f:
                        f.write(buffer)

            success_count += 1
        except Exception as e:
            fail_count += 1
            continue

    print(f"CBLPRD {subset} 处理完成！成功: {success_count}, 失败: {fail_count}")


def process_all(max_images_per_split=None):
    """处理所有划分（train/val）。"""
    for subset in ["train", "val"]:
        process_dataset(subset, max_images_per_split)


def split_train_val_test():
    """将字符数据集划分为训练集/验证集/测试集。"""
    if not os.path.exists(CHAR_DIR):
        print(f"[错误] 字符目录不存在: {CHAR_DIR}")
        return

    for label_dir in sorted(os.listdir(CHAR_DIR)):
        label_path = os.path.join(CHAR_DIR, label_dir)
        if not os.path.isdir(label_path):
            continue

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


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="CBLPRD-330k 车牌数据集预处理")
    parser.add_argument("--subset", type=str, default="train",
                        choices=["train", "val"],
                        help="子集名称 (默认: train)")
    parser.add_argument("--max_images", type=int, default=None,
                        help="最大处理图片数 (默认: 全部)")
    parser.add_argument("--all", action="store_true",
                        help="处理 train 和 val 两个子集")
    parser.add_argument("--split", action="store_true",
                        help="划分训练/验证/测试集")
    parser.add_argument("--stats", action="store_true",
                        help="打印数据集统计")
    args = parser.parse_args()

    if args.stats:
        print_dataset_stats()
    elif args.all:
        process_all(max_images_per_split=args.max_images)
    else:
        process_dataset(subset=args.subset, max_images=args.max_images)

    if args.split:
        split_train_val_test()

    print("\n使用方式:")
    print("  处理训练集:  python scripts/prepare_data.py --subset train")
    print("  处理验证集:  python scripts/prepare_data.py --subset val")
    print("  处理全部:    python scripts/prepare_data.py --all")
    print("  查看统计:    python scripts/prepare_data.py --stats")
