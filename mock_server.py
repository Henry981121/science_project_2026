"""
Mock API Server — 前端開發用
==============================
在沒有真實模型的情況下，模擬 api_server.py 的回傳格式。
之後換成真實後端只需要把這個檔案換成 api_server.py。

啟動：
    pip install fastapi uvicorn python-multipart pillow numpy
    uvicorn mock_server:app --port 8000 --reload
"""

import io
import base64
import random
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import colorsys

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

app = FastAPI(title="AI Detector Mock API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

STREAMS = ['clip', 'fft', 'dct', 'dire', 'noise']
STREAM_DISPLAY = {'clip': 'CLIP', 'fft': 'FFT', 'dct': 'DCT', 'dire': 'DIRE', 'noise': 'Noise'}


def make_fake_gradcam(img_pil: Image.Image) -> np.ndarray:
    """產生一個假的 Grad-CAM 熱力圖 overlay，視覺上像真的。"""
    img = img_pil.resize((224, 224)).convert("RGB")
    img_np = np.array(img, dtype=np.float32) / 255.0

    # 產生假的熱力圖：幾個高斯 blob
    h, w = 224, 224
    cam = np.zeros((h, w), dtype=np.float32)
    for _ in range(random.randint(2, 4)):
        cx = random.randint(40, w - 40)
        cy = random.randint(40, h - 40)
        sigma = random.randint(25, 55)
        strength = random.uniform(0.5, 1.0)
        Y, X = np.ogrid[:h, :w]
        blob = strength * np.exp(-((X - cx)**2 + (Y - cy)**2) / (2 * sigma**2))
        cam += blob

    cam = np.clip(cam, 0, None)
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

    # 轉 jet colormap
    def jet_color(v):
        r = np.clip(1.5 - abs(4*v - 3), 0, 1)
        g = np.clip(1.5 - abs(4*v - 2), 0, 1)
        b = np.clip(1.5 - abs(4*v - 1), 0, 1)
        return np.stack([r, g, b], axis=-1)

    heatmap = jet_color(cam)
    overlay = np.clip(0.5 * img_np + 0.5 * heatmap, 0, 1)
    return (overlay * 255).astype(np.uint8)


def numpy_to_b64(arr: np.ndarray) -> str:
    pil = Image.fromarray(arr)
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def pil_to_b64(pil: Image.Image) -> str:
    pil = pil.resize((224, 224)).convert("RGB")
    return numpy_to_b64(np.array(pil))


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(400, "Please upload an image file.")

    contents = await file.read()
    try:
        img_pil = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(400, "Cannot decode image.")

    # ── 隨機產生假結果（每次略有不同，測試各種 UI 狀態）─────────────
    seed = random.randint(0, 1)   # 0 = Real, 1 = AI（各半）
    is_ai = seed == 1

    # Fusion 結果
    if is_ai:
        fusion_prob = round(random.uniform(68, 97), 2)
    else:
        fusion_prob = round(random.uniform(8, 38), 2)
    fusion_pred = "AI" if fusion_prob > 50 else "Real"

    def conf_str(prob_fake, pred):
        if pred == "AI":
            return f"{prob_fake:.1f}% AI"
        return f"{100 - prob_fake:.1f}% Real"

    # Stream 結果（部分可能與 fusion 不一致，模擬真實情況）
    streams_out = {}
    raw_weights = {}
    for s in STREAMS:
        # 大部分跟 fusion 一致，偶爾有一兩個串流不同
        if random.random() < 0.80:
            prob = round(random.uniform(60, 95) if is_ai else random.uniform(5, 35), 2)
            pred = "AI" if is_ai else "Real"
        else:
            # 不一致的串流
            prob = round(random.uniform(40, 65), 2)
            pred = "AI" if prob > 50 else "Real"

        weight = random.uniform(0.08, 0.35)
        raw_weights[s] = weight
        streams_out[s] = {
            "display_name": STREAM_DISPLAY[s],
            "prob_fake":    prob,
            "prediction":   pred,
            "weight":       0,   # 填完 normalize 後再設
            "conf_str":     conf_str(prob, pred),
        }

    # Normalize weights to sum to 100
    total_w = sum(raw_weights.values())
    for s in STREAMS:
        streams_out[s]["weight"] = round(raw_weights[s] / total_w * 100, 1)

    # Grad-CAM
    heatmap_np = make_fake_gradcam(img_pil)
    original_np = np.array(img_pil.resize((224, 224)).convert("RGB"))

    return JSONResponse({
        "fusion": {
            "prob_fake":  fusion_prob,
            "prediction": fusion_pred,
            "conf_str":   conf_str(fusion_prob, fusion_pred),
        },
        "streams":      streams_out,
        "heatmap_b64":  numpy_to_b64(heatmap_np),
        "original_b64": numpy_to_b64(original_np),
        "__mock__": True,   # 方便之後確認是否還在用假資料
    })


@app.get("/health")
def health():
    return {"status": "ok", "mode": "mock"}