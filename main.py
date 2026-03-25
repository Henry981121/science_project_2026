from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import torch
import torchvision.transforms as transforms
import io

app = FastAPI()

# 允許前端連線（開發時先開放所有來源）
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# 載入你的模型
model = torch.load("your_model.pth", map_location="cpu")
model.eval()

# 類別名稱（改成你訓練時的類別）
CLASS_NAMES = ["類別A", "類別B", "類別C"]

# 圖片前處理（要跟你訓練時一致）
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # 讀取圖片
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    
    # 前處理
    tensor = transform(image).unsqueeze(0)
    
    # 推論
    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.softmax(outputs, dim=1)[0]
        top_idx = probs.argmax().item()
    
    return {
        "prediction": CLASS_NAMES[top_idx],
        "confidence": round(probs[top_idx].item() * 100, 2),
        "all_probs": {CLASS_NAMES[i]: round(p.item() * 100, 2) for i, p in enumerate(probs)}
    }