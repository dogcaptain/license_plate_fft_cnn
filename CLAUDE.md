# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Chinese license plate character recognition using CNN with 2D FFT frequency-domain features. The research hypothesis is that high-pass filtered edge information (via FFT) improves recognition robustness under noise. The model classifies individual characters (72 classes: 31 province abbreviations + 24 letters (no I/O) + 10 digits + 7 special chars) from segmented license plate images.

Two modes are compared: **baseline (spatial)** uses 1-channel grayscale input, **FFT** uses 2-channel input (grayscale + Gaussian high-pass filtered feature map).

## Commands

### Setup
```bash
# PyTorch must be installed separately for CUDA 12.6+ (RTX 5070)
pip install torch>=2.5.0 torchvision>=0.20.0 --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
```

### Data Preparation
```bash
# Download CBLPRD-330k dataset to data/CBLPRD-330k/ first
python scripts/prepare_data.py --all     # segment characters from plate images
python scripts/prepare_data.py --stats   # view dataset statistics
```

### Training & Evaluation
```bash
python scripts/run_experiments.py   # runs both baseline and FFT experiments end-to-end
```

### Visualization
```bash
# TensorBoard
tensorboard --logdir results/tensorboard
# Jupyter notebook
jupyter notebook notebooks/visualization.ipynb
```

## Architecture

### Pipeline
`Raw plate images` → `prepare_data.py` (segment into 20×20 character crops) → `Dataset` (load + optional FFT channel) → `CharCNN` (train/eval) → `results/`

### Key modules

- **`config.py`** — All global constants: paths, 72-class character table (`CHAR_LIST`, `CHAR_TO_IDX`, `IDX_TO_CHAR`), training hyperparameters (batch 256, 100 epochs, LR 0.01, OneCycleLR, AMP), FFT sigma, noise levels. This is the single source of truth for configuration.
- **`src/model.py`** — `CharCNN` class and `build_model(mode)` factory. Mode `"spatial"` → `in_channels=1`, mode `"fft"` → `in_channels=2`. Architecture: 4 conv blocks (BatchNorm+ReLU) → GlobalAvgPool → 3 FC layers (512→256→72). Kaiming initialization.
- **`src/fft_features.py`** — FFT pipeline. Core function is `extract_fft_features(img, sigma)` which applies Gaussian HPF and returns a normalized feature map used as the second input channel. Also contains various filter implementations (Gaussian, Ideal, Butterworth) and visualization helpers.
- **`src/dataset.py`** — PyTorch `Dataset` that loads character images and optionally adds the FFT channel.
- **`src/preprocess.py`** — Resize to 20×20, grayscale conversion, normalization, `add_gaussian_noise()` for robustness experiments.
- **`src/train.py`** — Training loop with OneCycleLR, mixed precision (AMP), label smoothing, TensorBoard + SwanLab logging, checkpoint saving.
- **`src/evaluate.py`** — Inference, confusion matrix, classification report. Outputs saved to `results/`.

## Platform Notes

- `NUM_WORKERS` is forced to 0 on Windows (`os.name == 'nt'`) due to multiprocessing limitations.
- Mixed precision training (AMP) is enabled by default (`USE_AMP = True`).
- Dataset split: 80% train / 10% val / 10% test.
- SwanLab (Chinese experiment tracking platform) is used alongside TensorBoard for logging.
