# 数据集说明

## CBLPRD-330k 数据集（推荐）

中国车牌识别数据集，33万张图片，比CCPD更新、更大、更均衡。

### 下载地址
- GitHub: https://github.com/SunlifeV/CBLPRD-330k
- ModelScope: https://www.modelscope.cn/datasets/sunlifev/China-Balanced-License-Plate-Recognition-Dataset-330k

### 数据集特点
- 33万张车牌图片（训练集/验证集/测试集已划分好）
- 包含蓝牌、黄牌、绿牌（新能源）、白牌、黑牌
- 数据更均衡，覆盖更多场景
- 标注格式：图片文件名即车牌号，如 `京A12345.jpg`

### 使用方法

#### 方式1：直接下载处理好的字符图片（推荐，最简单）
作者已提供分割好的单个字符图片，直接下载使用：
```bash
# 从GitHub Releases下载字符数据集
# 链接：https://github.com/SunlifeV/CBLPRD-330k/releases
# 下载后解压到 data/characters/ 目录
```

#### 方式2：从原始车牌图片自己分割
```bash
1. 下载完整数据集（约10GB）
2. 解压到 data/CBLPRD330k/ 目录下
3. 目录结构：
   data/
     CBLPRD330k/
       train/          # 训练集车牌图片
       val/            # 验证集车牌图片
       test/           # 测试集车牌图片
4. 运行 python scripts/prepare_data.py --dataset cblprd330k 进行字符分割
```

---

## CCPD2019 数据集（备选）

如果CBLPRD下载太慢，可以用CCPD作为备选。

### 下载地址
- GitHub: https://github.com/detectRecog/CCPD

### 使用方法
1. 下载 CCPD2019 数据集
2. 解压到 `data/CCPD2019/` 目录下
3. 运行 `python scripts/prepare_data.py --dataset ccpd --subset ccpd_base`
