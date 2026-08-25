# v2_fusion

重寫的融合層。跑在**現有的 feature cache** 上，不需要重抽特徵、不需要改資料管線。

每一項改動都對應 2026-08-25 架構稽核的一條發現，編號寫在下面的表格與程式碼註解裡。

---

## 快速開始（隊友端）

```bash
cd <專案根目錄>

# 1. 靜態驗證：不需要 GPU、不需要資料，30 秒
python -m v2_fusion.sanity

# 2. 資料對齊檢查：讀 cache 與 CSV，不訓練
python -m v2_fusion.train --dry-run

# 3. 單一組訓練
python -m v2_fusion.train --preset src0_grl0

# 4. 四組對照一次跑完（每組約 6 分鐘）
python -m v2_fusion.train --preset all
```

路徑預設從專案根目錄的 `config.py` 讀（`FEAT_CACHE_DIR` / `TRAIN_CSV` / `VAL_CSV`）。
要覆蓋就用 `--cache-dir` / `--train-csv` / `--val-csv`。

需要 `torch`、`pandas`、`numpy`、`scikit-learn`。

---

## 四組對照要回答什麼

先前 no-GRL vs with-GRL 的對照（89.01 vs 88.85）被記成「GRL 對泛化無效」。
但稽核發現 **GRL 從來沒有真正開過** —— λ 同時被放進 GRL 的 backward 和 loss 權重，
backbone 實收 λ² = 0.00198，比反方向拉扯的 λ_src=0.1 弱 **50.6 倍**。

所以那個 null result 不能寫成「GRL 對本任務無效」，只能說「在 λ 被平方且
λ_src=0.1 的設定下無可觀測效益」—— 而這句話沒有論文價值。

這四組只差 λ，其餘完全相同：

| preset | λ_src | λ_grl | 回答什麼 |
|---|---|---|---|
| `src0_grl0` | 0 | 0 | 乾淨 baseline，只有 CE_binary |
| `src0_grl0p05` | 0 | 0.05 | **GRL 的真實效益** —— 唯一沒有反向抵消的對抗設定 |
| `src0p1_grl0` | 0.1 | 0 | source head 單獨的效益 |
| `src0p1_grl0p05` | 0.1 | 0.05 | 舊模型的設定，但 λ 只乘一次 |

第二列是重點。國際科展評審會問「你怎麼確定是這個模組造成的」，這是唯一的答案。

---

## 架構

```
每條流 (B, in_dim)
    │
    ├─ StreamAdapter : LayerNorm(in_dim) → Linear(in_dim→d) → LayerNorm(d)
    ▼
(B, 5, d) + stream_embed
    │
    ├─ StreamSelfAttention × 2        Q=K=V，流之間互相溝通
    ▼
(B, 5, d)
    │
    ├─ CrossAttentionReadout          Q 來自可學習的 fusion token、K/V 來自五條流
    ▼
(B, d)
    ├─ head_binary   → (B, 2)
    ├─ head_source   → (B, 19)   × λ_src
    └─ GRL → head_gen → (B, 17)   梯度反轉，強度 λ_grl
```

**命名如實。** 前半段 Q=K=V 就叫 self-attention；後半段 Q≠K/V，在定義上就是
cross-attention，不是硬拗。舊版把 Q=K=V 的那層叫 `CrossAttentionFusionLayer`，
論文寫「Cross-Attention Fusion」會被直接打。

**為什麼不做「CLIP 當 query」的 cross-attention。** 那等於在架構裡寫死
「CLIP 是主、其餘是輔」。支持這個假設的證據（ablation CLIP 0.30 最高）本身是被污染的
—— 其他三條流是隨機投影或隨機 CNN。用「CLIP 最重要」去 justify 把 CLIP 設為 query
是循環論證。可學習的 fusion token 不預設任何一條流為主。

參數量 **8,831,526**（舊版 9,601,574）。舊版 flatten 後的 `Linear(2560→1024)`
單層就佔 2,622,464，是最大的單一層；換成 fusion token readout 之後消失。

---

## 對應的稽核發現

| 發現 | 舊版 | v2 |
|---|---|---|
| 05 | λ 乘兩次 → backbone 實收 λ²；λ_src=0.1 意外開著 | λ 只出現在 `model.grad_reverse()` 一次。`LossConfig` **沒有** gen loss 的權重欄位，所以不可能乘第二次。`sanity.py` 有回歸測試 |
| 07 | 未知 generator 靜默標成真圖 | `data.py` 直接 raise，錯誤訊息列出打錯的名稱與筆數 |
| 08 | `clamp` 當 remap，class 10/11 死輸出、id 17/18 撞號 | `SOURCE_ID_TO_GEN_ID` 明確字典映射，一對一。真圖填 −1 由 `ignore_index` 吃掉 |
| 11 | 同一個 val split 挑 checkpoint 又拿來報告 | 提供 `--test-csv` 就分開報告；沒提供的話 `results.json` 明確標 `selection_biased: true` |
| 12 | token 進 attention 前無 LayerNorm，尺度大的流主導 | 每條流 `LayerNorm(in_dim)` 打頭。測試：輸入尺度差 10⁶ 倍，token 範數 max/min = 1.000 |
| 13 | `CrossAttentionFusionLayer` 其實是 self-attention | 兩段各自命名如實 |
| 14 | `stream_embed` σ=0.02，流身份訊號微弱 | 流身份主要由「每條流各自獨立的 adapter」承擔，比加法式 embedding 強得多 |
| 15 | `flatten` → `Linear(2560→1024)`，2.62M、佔 27.3% | fusion token cross-attention readout。XAI 讀出變成天然的 `(B, 5)` 向量 |
| 18 | `dropout=0.3` 傳進去被 attention 層吃掉 | `dropout_attn` / `dropout_head` 分開，都真的生效 |
| 20 | `final_results.json` 寫死假的 curriculum 字串 | `results.json` 的 config 直接 `asdict(RunConfig)`，不可能記錄到沒發生的事 |
| 02 | 整條路徑 `grep manual_seed` 零命中 | `set_seed()` 涵蓋 random / numpy / torch / cuda，seed 進 config 也進 results |
| 21 | repo 裡三份互相矛盾的 fusion 實作 | 只有這一份 |

**還沒解決的**（需要重抽特徵或改資料管線，不在這個套件的範圍）：
發現 01（投影層隨機未訓練）、03（切分腳本不存在）、04（DCT batch-dependent）、
06（resize 摧毀 FFT/DCT/Noise 前提）、09（augmentation 凍結進 cache）、10（全黑圖）。

---

## 之後接上「原始維度 cache」

發現 01 與 16：目前 cache 存的是投影**後**的 512 維，而那些投影層是隨機初始化、
從未訓練、也沒存檔。CLIP 被丟掉的那一半資訊已經不在檔案裡，模型端救不回來。

`StreamAdapter` 已經預留好了 —— 它的 `Linear(in_dim → d_model)` 就是搬進來的投影層。
隊友下次重抽特徵時只要改三行：

| 檔案 | 現在 | 改成 |
|---|---|---|
| `clip_extractor.py:57` | `return self.proj(raw)` | `return raw` → 存 1024 維 |
| `dire_extractor.py:102` | `feat = self.projection(feat)` | 刪掉 → 存 2048 維 |
| `dct_extractor.py:120` | `feat = self.projection(feat)` | 刪掉 → 存 512 維 |

然後這邊加一個旗標即可，模型會自動接上：

```bash
python -m v2_fusion.train --preset all --raw-dims
```

參數量 8.83M → 9.88M。**同樣的 backbone、同樣的五條流、同樣的抽取時間**，
cache 約大 1.6 倍。這不改變「用哪些特徵」，只改變「特徵在哪裡被壓縮、
以及壓縮器是不是學出來的」。

⚠️ **有時效性**：隊友只要因為特徵選擇實驗動到任何一條流就必須重抽，
那時順手改維度是零成本；抽完才提就要為此再跑一輪 GPU。

---

## 順便建議一起做的

抽特徵時同時存下每一筆的 path（順序與寫入 tensor 的順序相同）：

```python
json.dump(paths, open(FEAT_DIR / f"{split}_paths.json", "w"))
```

那是**唯一**能真正保證 feature/label 對齊的東西。有了它就可以打開嚴格模式：

```bash
python -m v2_fusion.train --require-manifest
```

目前 `s3a_single_stream.py:147` 是 `shuffle=False`，所以順序確實一致 ——
但那是「靠慣例成立」不是「靠檢查保證」。任何人改了 shuffle 或換了 CSV
就會整批錯位，而且不會報錯。

---

## 檔案

| 檔案 | 內容 |
|---|---|
| `config.py` | 所有超參數的唯一來源。generator 標籤表、`ModelConfig` / `LossConfig` / `TrainConfig`、四組 preset |
| `model.py` | `FusionDetectorV2`、`StreamAdapter`、`StreamSelfAttention`、`CrossAttentionReadout`、`grad_reverse` |
| `losses.py` | `FusionLoss`。刻意沒有 gen loss 的權重參數 |
| `data.py` | cache 載入與對齊驗證。對不上就 raise，不猜 |
| `sanity.py` | 靜態驗證，14 項，不需要 GPU 與資料 |
| `train.py` | 訓練 CLI 與 preset sweep |

## 已驗證

`python -m v2_fusion.sanity` — 14/14 通過（torch 2.13，CPU）。
端到端在合成 cache 上跑過 `--preset all`：四組訓練、對照總表、`results.json` 皆正常。
四種對齊錯誤（CSV 筆數不符、generator 打錯字、順序被打亂、某條流中斷）都會被擋下。

**尚未在真實 cache 上跑過** —— 本機沒有資料與 GPU。隊友第一次跑請先 `--dry-run`。
