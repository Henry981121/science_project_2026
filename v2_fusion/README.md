# v2_fusion

融合層。跑在 **2026-08-26 handoff 的 feature cache** 上
（`handoff_dct_clip_dinov2_20260826/feature_data/crop`），不需要重抽特徵。

---

## 快速開始（隊友端）

```powershell
cd <專案根目錄>
$CACHE = "C:\Users\harry\OneDrive\Desktop\science_project_v3\handoff_dct_clip_dinov2_20260826\feature_data\crop"

# 1. 靜態驗證：不需要 GPU、不需要資料，30 秒
python -m v2_fusion.sanity

# 2. 資料對齊檢查：讀 cache 與 index CSV，不訓練
python -m v2_fusion.train --cache-dir $CACHE --dry-run

# 3. 單一組訓練
python -m v2_fusion.train --cache-dir $CACHE --preset src0

# 4. 兩組對照
python -m v2_fusion.train --cache-dir $CACHE --preset all
```

需要 `torch`、`pandas`、`numpy`、`scikit-learn`。

---

## 2026-08-27 的兩項變更

### 1. GRL 已整條移除

不再有 gradient reversal、gen discriminator、`lambda_grl`。整條路徑刪除，
**而不是留著預設 0** —— 留著就還有人會去設它，也還要在論文裡解釋一個沒在用的模組。

`sanity.py` 的 `[3]` 是這件事的**回歸測試**：檢查 `model` 模組沒有 `grad_reverse`、
模型沒有 `head_gen`、`LossConfig` 沒有任何含 `grl` 的欄位、`forward()` 不收 `grl_lambda`。
任何人想加回來都會讓測試失敗，那時要有意識地決定，不能是無意間長回來的。

> `head_source` **不是** GRL。它是正常梯度的 auxiliary head（預測「是哪個 source」），
> 預設 `lambda_src=0` 關閉。兩者不要混為一談。

### 2. 特徵流換成 dct / clip / dinov2

| 流 | 維度 | 來源 |
|---|---|---|
| `dct` | 192 | 64 個頻率位置 × 3 個統計量（手工特徵，無 CNN） |
| `clip` | 768 | CLIP ViT-L/14 image embedding |
| `dinov2` | 1024 | DINOv2 ViT-L/14 CLS token |

這些是各 backbone 的**原生輸出**，不再經過任何隨機初始化的投影層 ——
**audit 發現 01（三條流的編碼器隨機、未訓練、未存檔）因此消失。**
唯一的降維層是 `StreamAdapter` 的 `Linear(in_dim → 512)`，而它是被訓練的。

---

## 架構

```
每條流 (B, in_dim)
    │
    ├─ StreamAdapter : LayerNorm(in_dim) → Linear(in_dim→512) → LayerNorm(512)
    ▼
(B, 3, 512) + stream_embed
    │
    ├─ StreamSelfAttention × 2        Q=K=V，流之間互相溝通
    ▼
(B, 3, 512)
    │
    ├─ CrossAttentionReadout          Q 來自可學習的 fusion token、K/V 來自三條流
    ▼
(B, 512)
    ├─ head_binary   → (B, 2)
    └─ head_source   → (B, 10)   × λ_src（auxiliary，預設關閉）
```

**命名如實。** 前半段 Q=K=V 就叫 self-attention；後半段 Q≠K/V，在定義上就是
cross-attention。舊版把 Q=K=V 的那層叫 `CrossAttentionFusionLayer`，
論文寫「Cross-Attention Fusion」會被直接打。

**為什麼不做「CLIP 當 query」的 cross-attention。** 那等於在架構裡寫死
「CLIP 是主、其餘是輔」。支持這個假設的證據（舊版 ablation CLIP 0.30 最高）本身是
被污染的 —— 當時其他流是隨機投影或隨機 CNN。用「CLIP 最重要」去 justify 把 CLIP
設為 query 是循環論證。可學習的 fusion token 不預設任何一條流為主。

參數量 **8,391,052**（`use_source_head=False` 時 8,385,922）。
舊版 `s3_main_grl.py` 是 9,601,574，其中 flatten 後的 `Linear(2560→1024)`
單層就佔 2,622,464（27.3%），換成 fusion token readout 之後消失。

---

## 標籤空間

**source head 的類別只有 train/val 實際出現的 10 種**：
`real, real_extra, adm, glide, sdv4, sdv5, midjourney, wildfake, stylegan, dcgan`

只出現在 `cross_generator_test` 的 5 種（`wildfake_ddim, wildfake_other,
dcgan_unseen, fursona_gan, waifu_gan`）**刻意不編進類別空間**，source 標籤為 `-1`，
由 `ignore_index` 吃掉。把它們編進來會製造永遠收不到樣本的死輸出 ——
那正是 audit 發現 08 的同一個病。

---

## cache 佈局與對齊保證

```
{cache}/index_{split}.csv          row, path, generator, is_real, split, valid_all
{cache}/{stream}/{split}.npy       (N, dim) float32
{cache}/{stream}/{split}.valid.npy (N,) bool
```

`data.py` 的原則是**對不上就 raise，絕不猜**。已實測會被擋下的 7 種錯誤：

| 錯誤 | 舊版行為 |
|---|---|
| CSV 列數與 npy 不符 | `df.iloc[:n]` 截斷 → 整批 label 錯位且不報錯 |
| 未知 generator 名稱 | `.get(name, REAL_ID)` → 假圖靜默變真圖 |
| `generator` 與 `is_real` 互相矛盾 | 不檢查 |
| CSV 被重排（`row` 不是 0..N-1） | 不檢查 |
| 某條流的維度與 config 不符 | 不檢查 |
| `valid_all` 與各流 `.valid.npy` 交集不一致 | 沒有 valid 概念 |
| 整條流的目錄不見了 | 不檢查 |

`valid` mask 取的是**被選用那幾條流**的交集，不是 CSV 的 `valid_all` ——
只用 3 條裡的 2 條時，第 3 條的 invalid 不該把樣本丟掉。差異會印在 `--dry-run` 的輸出裡。

---

## 對照組

GRL 移除後只剩一個 λ 可調：

| preset | λ_src | 回答什麼 |
|---|---|---|
| `src0` | 0 | 乾淨 baseline，只有 CE_binary |
| `src0p1` | 0.1 | source auxiliary head 有沒有幫助 |

---

## selection bias

`--test-split` 預設 `cross_generator_test` —— 它不參與 checkpoint 選擇，
所以那組數字可以寫進論文；`val` 的數字在 `results.json` 裡一律標
`selection_biased: true`。

handoff 裡真正上鎖的 `test` split **沒有被包進 package**，那是刻意的：
它只能在特徵集與模型都凍結之後才打開。這是目前方法論上最強的一點，不要提早破壞它。

---

## 已驗證 / 未驗證

- ✅ `sanity.py` 17/17 通過（torch 2.13 CPU）
- ✅ 在**與 handoff 同構的合成 cache** 上跑過 `--dry-run` 與 `--preset all`
- ✅ 7 種對齊錯誤都會被擋下
- ❌ **尚未在真實 cache 上跑過**（開發機無資料無 GPU）。第一次跑請先 `--dry-run`

## 待處理（不在這個套件的範圍）

- `train` 與 `cross_generator_test` 有 **704 筆檔名重疊**（真圖側為 0，
  最可能是不同 generator 目錄下的檔名撞號）—— 需確認不是同一張圖
- `wildfake` 在 train，`wildfake_ddim` / `wildfake_other` 在 cross_generator_test；
  `dcgan` 對 `dcgan_unseen` 同理。**「未見生成器」的說法需要證據或改寫敘事**
- XAI（`src/xai/`）仍是針對舊的 5 條流寫的，需重跑
