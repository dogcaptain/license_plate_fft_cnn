"""
车牌字符识别 Web API
FastAPI 后端，提供单字符识别和整车牌识别接口
"""
import os
import sys
import base64
import numpy as np
import cv2
import torch
from fastapi import FastAPI, UploadFile, File, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles

# 添加项目根目录到路径
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from config import IDX_TO_CHAR, CHAR_LIST, NUM_CLASSES, HPF_SIGMA, CHAR_IMG_SIZE, RESULTS_DIR
from src.model import build_model
from src.fft_features import extract_fft_features

app = FastAPI(title="车牌字符识别 API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# === 全局模型 ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = None
checkpoint_info = {}


@app.on_event("startup")
def load_model():
    global model, checkpoint_info
    model = build_model(mode="fft")
    ckpt_path = os.path.join(RESULTS_DIR, "best_model_fft.pth")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    checkpoint_info = {
        "epoch": ckpt.get("epoch", "N/A"),
        "val_acc": round(ckpt.get("val_acc", 0), 4),
    }
    print(f"模型已加载: {ckpt_path}, device={device}")


# === 工具函数 ===

def preprocess_char_image(img_gray: np.ndarray):
    """灰度图 -> (1, 2, 20, 20) tensor"""
    gray = cv2.resize(img_gray, CHAR_IMG_SIZE, interpolation=cv2.INTER_AREA)
    gray_f = gray.astype(np.float32) / 255.0
    fft_feat = extract_fft_features(gray_f, sigma=HPF_SIGMA)
    stacked = np.stack([gray_f, fft_feat], axis=0)
    return torch.from_numpy(stacked).unsqueeze(0).float(), gray_f, fft_feat


def predict_single(img_gray: np.ndarray):
    """对单个灰度字符图推理，返回结果字典"""
    tensor, gray_f, fft_feat = preprocess_char_image(img_gray)
    with torch.no_grad():
        logits = model(tensor.to(device))
        probs = torch.softmax(logits, dim=1)[0]
        pred_idx = probs.argmax().item()

    # FFT特征图转base64 PNG
    fft_uint8 = (fft_feat * 255).astype(np.uint8)
    _, buf = cv2.imencode(".png", fft_uint8)
    fft_b64 = base64.b64encode(buf).decode("utf-8")

    # 原图也转base64
    gray_uint8 = (gray_f * 255).astype(np.uint8)
    _, buf2 = cv2.imencode(".png", gray_uint8)
    gray_b64 = base64.b64encode(buf2).decode("utf-8")

    top5_indices = probs.topk(5).indices.tolist()
    return {
        "character": IDX_TO_CHAR[pred_idx],
        "confidence": round(probs[pred_idx].item(), 4),
        "top5": [{"char": IDX_TO_CHAR[i], "prob": round(probs[i].item(), 4)} for i in top5_indices],
        "fft_image": fft_b64,
        "gray_image": gray_b64,
    }


def split_plate(plate_img, num_chars=7):
    """将车牌图像均匀分割为字符区域，返回灰度字符图列表"""
    if len(plate_img.shape) == 3:
        gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
    else:
        gray = plate_img
    h, w = gray.shape[:2]
    char_width = w / num_chars
    chars = []
    for i in range(num_chars):
        x1 = int(i * char_width)
        x2 = int((i + 1) * char_width)
        chars.append(gray[:, x1:x2])
    return chars


def decode_upload(data: bytes):
    """将上传的文件字节解码为BGR图像"""
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return img


# === API 端点 ===

@app.post("/api/predict")
async def predict_char(file: UploadFile = File(...)):
    """单字符识别"""
    data = await file.read()
    img = decode_upload(data)
    if img is None:
        return JSONResponse({"error": "无法解码图片"}, status_code=400)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    result = predict_single(gray)
    return result


@app.post("/api/predict-plate")
async def predict_plate(
    file: UploadFile = File(...),
    num_chars: int = Query(default=7, ge=7, le=8, description="车牌字符数(7普通/8新能源)"),
):
    """整车牌识别"""
    data = await file.read()
    img = decode_upload(data)
    if img is None:
        return JSONResponse({"error": "无法解码图片"}, status_code=400)

    # 原图base64
    _, buf = cv2.imencode(".png", img)
    plate_b64 = base64.b64encode(buf).decode("utf-8")

    char_imgs = split_plate(img, num_chars)
    characters = []
    plate_number = ""
    for char_img in char_imgs:
        result = predict_single(char_img)
        characters.append(result)
        plate_number += result["character"]

    return {
        "plate_number": plate_number,
        "plate_image": plate_b64,
        "characters": characters,
    }


@app.get("/api/model/info")
def model_info():
    """模型信息"""
    info = model.get_model_info()
    return {
        "mode": "fft",
        "in_channels": info["in_channels"],
        "num_classes": info["num_classes"],
        "total_params": info["total_params"],
        "trainable_params": info["trainable_params"],
        "device": str(device),
        "char_img_size": list(CHAR_IMG_SIZE),
        "hpf_sigma": HPF_SIGMA,
        "checkpoint_epoch": checkpoint_info.get("epoch"),
        "checkpoint_val_acc": checkpoint_info.get("val_acc"),
    }


# === 前端静态文件（npm run build 后） ===
FRONTEND_DIST = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "frontend", "dist")

if os.path.isdir(FRONTEND_DIST):
    # 静态资源（js/css/图片等）
    app.mount("/assets", StaticFiles(directory=os.path.join(FRONTEND_DIST, "assets")), name="assets")

    # SPA fallback：所有非 /api 的请求返回 index.html
    @app.get("/{full_path:path}")
    async def serve_spa(full_path: str):
        file_path = os.path.join(FRONTEND_DIST, full_path)
        if full_path and os.path.isfile(file_path):
            return FileResponse(file_path)
        return FileResponse(os.path.join(FRONTEND_DIST, "index.html"))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
