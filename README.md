# 組合特徵 AI 圖片辨識平台

整合 CLIP、FFT、DCT、DIRE、Noise 五種分析串流，判斷圖片是否由 AI 生成。
提供網頁介面，支援電腦和手機瀏覽器上傳圖片即時分析。

## 檔案結構

```
east_zone_project/
├── index.html       # 前端網頁介面
├── api_server.py    # 正式後端（接真實模型）
├── detector.py      # 核心偵測邏輯（特徵提取 + 融合推論）
├── mock_server.py   # 開發用假後端（不需要模型即可測試 UI）
├── main.py          # 早期測試用，目前不使用
└── README.md
```

> 依賴的模型權重和特徵提取器來自另一個資料夾（`tech_project`），路徑在 `detector.py` 頂端設定。

---

## 環境需求

```
pip install fastapi uvicorn python-multipart pillow torch torchvision
```

---

## 路徑設定（重要）

**每個人的電腦需要分別設定** `detector.py` 最頂端的路徑：

```python
# ── 路徑設定 ──────────────────────────────────────────
TECH_PROJECT_DIR = Path(r'你的 tech_project 資料夾路徑')
OUTPUTS_DIR      = Path(r'你的 outputs 資料夾路徑')
```

| 路徑變數 | 說明 |
|---|---|
| `TECH_PROJECT_DIR` | 含有 `src/feature_extractors/` 和 `s3_main_grl.py` 的資料夾 |
| `OUTPUTS_DIR` | 含有 `main_grl/best_model.pth` 和 `exp_a/` 的輸出資料夾 |
| `GRADCAM_PATH` | ResNet50 fine-tuned 權重（用於 Grad-CAM 視覺化） |

---

## 啟動方式

### 正式模式（需要模型權重）

```bash
uvicorn api_server:app --host 0.0.0.0 --port 8000 --reload
```

### 開發模式（Mock，不需要模型）

```bash
uvicorn mock_server:app --host 0.0.0.0 --port 8000 --reload
```

> `--host 0.0.0.0` 讓同一個 Wi-Fi 下的手機也能連線。

啟動後開啟：
- 電腦：`http://localhost:8000`
- 手機：`http://電腦的區域網路IP:8000`（例如 `http://192.168.1.5:8000`）

---

## 功能說明

### 選擇分析模組
上傳圖片後，可勾選要啟用的分析串流（預設全選）：

| 模組 | 說明 |
|---|---|
| CLIP | 語義特徵，使用 ViT-B/32 |
| FFT | 頻域分析，偵測頻率異常 |
| DCT | 離散餘弦轉換特徵 |
| DIRE | 擴散模型重建誤差 |
| Noise | SRM 高通濾波器雜訊特徵 |

未勾選的串流特徵會補零後傳入融合模型，確保輸入維度不變。

### 結果說明
- **融合判定**：五個串流經 GRL 融合模型加權後的最終結果
- **各串流分析**：每個串流的獨立預測與信心值
- **注意力權重**：融合時各串流的相對貢獻比例
- **Grad-CAM**：以熱力圖標示圖片中被模型判定為關鍵的區域

---

## API

| 方法 | 路徑 | 說明 |
|---|---|---|
| `POST` | `/predict` | 上傳圖片進行分析 |
| `GET` | `/health` | 確認伺服器狀態 |
| `GET` | `/` | 回傳前端網頁 |

### `/predict` 參數

| 參數 | 類型 | 說明 |
|---|---|---|
| `file` | Form (image) | 圖片檔案（JPG、PNG、WebP） |
| `streams` | Form (string, 可選) | 啟用的串流，逗號分隔，如 `clip,fft,noise`，預設全部 |

---

## 從 GitHub 取得更新

```bash
git pull origin dev
```

只要路徑設定正確、模型權重在位，pull 完即可直接啟動。
