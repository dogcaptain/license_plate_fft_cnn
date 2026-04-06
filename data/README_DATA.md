# 数据集说明

## CCPD2019 数据集

### 下载地址
- GitHub: https://github.com/detectRecog/CCPD
- 百度网盘: 见GitHub页面说明

### 使用方法
1. 下载 CCPD2019 数据集
2. 解压到 `data/CCPD2019/` 目录下
3. 确保目录结构为:
```
data/
  CCPD2019/
    ccpd_base/        # 正常车牌（主要使用这个）
    ccpd_blur/        # 模糊车牌
    ccpd_challenge/   # 挑战性车牌
    ccpd_db/          # 光照不均
    ccpd_fn/          # 远距离
    ccpd_rotate/      # 旋转车牌
    ccpd_tilt/        # 倾斜车牌
    ccpd_weather/     # 天气影响
```
4. 运行 `python scripts/prepare_data.py` 进行字符分割
