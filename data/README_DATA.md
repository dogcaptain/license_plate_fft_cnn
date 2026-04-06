# 数据集说明

## CBLPRD-330k 数据集

中国车牌识别数据集，33万张图片，包含蓝牌、黄牌、绿牌（新能源）、白牌、黑牌等。

### 下载地址
- GitHub: https://github.com/SunlifeV/CBLPRD-330k
- ModelScope: https://www.modelscope.cn/datasets/sunlifev/China-Balanced-License-Plate-Recognition-Dataset-330k

### 数据结构
```
data/
  CBLPRD-330k/          # 车牌图片目录
    000000000.jpg
    000000001.jpg
    ...
  train.txt             # 训练集标注
  val.txt               # 验证集标注
```

### 标注格式
```
CBLPRD-330k/000272981.jpg 粤Z31632D 新能源大型车
CBLPRD-330k/000204288.jpg 藏CFF7440 新能源小型车
```

### 使用方法
```bash
1. 下载数据集并解压到 data/ 目录
2. 运行预处理脚本：
   python scripts/prepare_data.py --all
3. 查看统计：
   python scripts/prepare_data.py --stats
```
