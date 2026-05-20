# XAI 模組重做實作計畫

## 背景

目前專案的 Grad-CAM 有三個問題：

1. `ai_detector_demo.py` 顯示的熱力圖來自一顆獨立的 ResNet50（行 135-146），與五流融合模型無關
2. `s4_supplementary.py` 的 `step5_gradcam()` 同樣對該 ResNet50 跑 GradCAM++
3. `src/xai/gradcam.py` 的 `MultiStreamExplainer` 整個包在 `@torch.no_grad()` 底下（行 105），完全沒算梯度，回傳的只是輸入域的 FFT/DCT/DIRE/Noise 變換

本計畫重建一套真正對應五流融合模型的 XAI 模組，並加上方法論驗證。

---

## 文獻基礎（先讀完再實作）

| # | 論文 | 用途 |
|---|---|---|
| 1 | Selvaraju et al., 2017, ICCV — Grad-CAM | CNN 四條流的方法基礎 |
| 2 | Chefer et al., 2021, CVPR — Transformer Interpretability Beyond Attention Visualization | CLIP 流的方法基礎 |
| 3 | Jain & Wallace 2019 + Wiegreffe & Pinter 2019 | Attention 作為解釋的辯論，決定 Level 2 怎麼寫 |
| 4 | Adebayo et al., 2018, NeurIPS — Sanity Checks for Saliency Maps | Defense layer |
| 5 | Wang et al., 2020, CVPR — CNN-generated images... | 領域 anchor |
| 6 | Frank et al., 2020, ICML — Leveraging Frequency Analysis | FFT/DCT 流解讀依據 |
| 7 | Wang et al., 2023, ICCV — DIRE | DIRE 流原理 |
| 8 | Petsiuk et al., 2018, BMVC — RISE | Deletion AUC 指標（選讀） |

---

## 整體架構

四個輸出物：

| 層次 | 輸出 | 主要依據文獻 |
|---|---|---|
| Level 1 | 一張六格圖：原圖 + 5 條流的熱力圖 | 1, 2 |
| Level 2 | Attention 分布 + ablation cross-check | 3 |
| Level 3 | FN/FP 失敗案例診斷 | — |
| Defense | Sanity check + deletion curve | 4, 8 |

## 現有 code 處置

- **保留**：`ai_detector_demo.py:54-77` 的 `GradCAM` class（公式正確）
- **保留**：`ai_detector_demo.py:120-125` 已掛好的 hooks
- **刪除**：`ai_detector_demo.py:135-146` 的獨立 ResNet50
- **刪除**：`s4_supplementary.py` 的 `step5_gradcam()`
- **改寫**：`src/xai/gradcam.py` 的 `MultiStreamExplainer`（拿掉 `@torch.no_grad()`，接上真實梯度流）

---

## Pre-flight 結果（已完成）

### Risk A：梯度流檢查 — 全 PASS

5 條 extractor 已加 `xai_mode=False` 參數，預設行為與原本完全相同；**呼叫 Grad-CAM 時要顯式傳 `xai_mode=True`** 才會解 no_grad。

**重要副作用**：Noise extractor 的 4 個 `nn.ReLU(inplace=True)` 改成 `inplace=False`，每層多耗 ~50MB activation memory，跑批次大小要留意。

**確認的 Grad-CAM target layer**（Day 1 不用再花時間找）：

| 流 | Target module | Activation/Gradient shape |
|---|---|---|
| FFT | `BasicBlock` | (1, 512, 7, 7) |
| DCT | `BasicBlock` | (1, 512, 7, 7) |
| DIRE | `Bottleneck` | (1, 2048, 7, 7) |
| Noise | `ReLU` | (1, 512, 28, 28) |
| CLIP | — | 走 Chefer relevance（無 conv target） |

### Risk B：Attention 模式檢查 — 原假設打臉，但發現更強的敘事

**家族層級**（CLIP 在所有家族都 47-57% 主導）：

| 流 | Diffusion | GAN | Real |
|---|---|---|---|
| CLIP | 47.81% | 56.86% | 52.19% |
| Noise | 20.37% | 20.45% | 18.16% |
| DCT | 12.26% | 6.20% | 10.61% |
| FFT | 10.75% | 12.46% | 11.15% |
| DIRE | 8.82% | 4.03% | 7.90% |

**原假設「GAN→FFT/DCT、Diffusion→DIRE」只成立一半**：
- GAN 那邊 ✅（FFT+DCT 18.66% > DIRE 4.03%）
- Diffusion 那邊 ❌（DIRE 才 8.82%，FFT+DCT 反而 23%）

**Per-generator 揭露的真正模式**（比家族二分法更值錢）：

| 生成器 | 主導流 | 啟示 |
|---|---|---|
| dcgan | CLIP 69% + Noise 20%（DIRE ≈ 0） | 簡單 GAN 純語意就能解 |
| adm | CLIP 47% + **DCT 29%** | ADM 靠 DCT 不是 DIRE |
| midjourney | CLIP 43% + **Noise 31%** | MJ 指紋在高頻雜訊 |
| sdv4/v5 | CLIP 42% + Noise 28% + DIRE 13% | DIRE 真正有效是在 SD 上 |
| stylegan | CLIP 51% + Noise 20% + FFT 16% | 高頻 FFT 強訊號 |
| glide | CLIP 61% + Noise 15% | 與其他 Diffusion 不同 |
| wildfake | CLIP 51% + DCT 20% | 又一個靠 DCT 的 Diffusion |

**跨實驗一致性**：
- CLIP 47-57% attention ⟷ Shapley CLIP +30.33%（最大貢獻）
- Noise 18-21% attention ⟷ Shapley Noise +12.77%（被 LOO 低估）
- DIRE 在 SDv4/v5 上 13% ⟷ EXP-A 顯示 DIRE 對 Diffusion 較強（76% ACC）
- 每個生成器啟動不同流 ⟷ EXP-D 的「per-generator FNR 差異很大」

→ 對應 Level 2 敘事改寫，見下方。

---

## Level 1：一流一圖

### 1A. CNN 四條流（FFT / DCT / DIRE / Noise）

- **Target layer**：見 Pre-flight 結果表（FFT/DCT 為 `BasicBlock`、DIRE 為 `Bottleneck`、Noise 為 `ReLU`）
- **Backward target**：該流自己的 EXP-A LinearHead 對 fake 類別的 logit
- **必須傳 `xai_mode=True`** 給 `extract_features()`，否則梯度被 no_grad 切掉
- **流程**：forward(xai_mode=True) → 取該流 head 的 fake logit → `model.zero_grad()` → `logit.backward()` → hook 抓 activation/gradient → 套 Grad-CAM 公式 → ReLU → normalize → resize 到 224

### 1B. CLIP 流（ViT）

**已決定：採用 Chefer relevance propagation。**

主要依據 **Chefer 2021 CVPR**（"Transformer Interpretability Beyond Attention Visualization"）的方法：
- 對 ViT 每層 transformer block 掛 hook，抓 attention map + 對 attention 的 gradient
- 套 layer-wise relevance propagation 規則（gradient-weighted attention averaging across heads）
- 逐層累積得到最終 relevance map

**輔助讀**：**Chefer 2021 ICCV**（"Generic Attention-Model Explainability..."），同作者擴展版，方法寫得更通用，CLIP 是被討論的例子之一。確認實作方向用。

**參考實作**：作者有 open-source repo，可以拿來改不用從零實作。

### 1C. 解讀眉角

- **FFT/DCT 熱區在頻率空間**，不是像素空間
  - FFT：右上 = 高頻
  - DCT：座標 = 係數區段
  - Caption 要寫「frequency region」不是「image region」
- **DIRE / Noise 熱區是真實像素空間**

### 1D. 輸出
單張測試圖 → 1×6 圖：`原圖 | CLIP | FFT | DCT | DIRE | Noise`
所有 heatmap 用同一個 colormap 和 normalization 範圍。

**預期視覺結果（管理期待）**：根據 Pre-flight Risk B，CLIP 在所有家族 47-57% 主導；Level 1 圖預期會是「CLIP 那格訊號最強，其他四格相對淡」。**這不是 bug，是模型實際的分工**。寫作時要說清楚並非所有流貢獻相等，配合 per-generator 結果論述哪條流在哪類圖才會「啟動」。

---

## Level 2：流級 attention 分布

**重要：根據 Pre-flight Risk B，敘事已從「家族二分法」改為「per-generator 差異化」。**

### 2A. 抓 attention（已完成）

Pre-flight Risk B 已產出 `outputs/preflight_attn.csv`，**直接用這個檔案畫圖，不用重跑 inference**。包含每張圖的 5 條流 attention 權重 + generator + family 欄位。

### 2B. Ablation cross-check（關鍵）

W&P 的反駁要求證明「移除某 component 真的會改變預測」。

- 對每張圖，依序 zero out 第 i 條流特徵，重跑 fusion，記錄 fake-prob 的 |Δ|
- 對每組算 ablation importance vector shape `(5,)`
- 比較 attention vs. ablation：
  - **Spearman ρ > 0.7**：可宣稱「attention 反映真實貢獻」
  - **ρ 低**：主敘事改用 ablation，attention 降為「fusion 內部加權」中性描述

### 2C. 輸出（**主附圖角色對調**）

- **主圖**：**按具體 generator 分組**的 attention stacked bar / heatmap — 展示「每個 generator 啟動不同流的差異化模式」
- **附圖**：按家族分組（Real / GAN / Diffusion）— 退到附錄當參考，並明說「家族層級的差異被 per-generator 模式平均掉了」
- attention vs. ablation 相關性散點圖 + Spearman ρ 表
- 一段論述：「在本實驗設定下，attention 與 ablation 一致度 ρ = X」

### 2D. 敘事範本

**不能寫**（舊敘事，會被打臉）：
> 「GAN 圖讓 FFT/DCT 權重升高，Diffusion 圖讓 DIRE 權重升高，證明多流按生成器類型互補」

**應該寫**（基於實際資料，更強）：
> 「Cross-Attention 顯示模型學到了**逐生成器的差異化權重分配**——並非簡單的『GAN→頻域、Diffusion→重建』二分法，而是針對每個生成器啟動不同的特徵組合。例如 ADM 由 DCT 主導抓取（28.6%）、Midjourney 由 Noise 主導（31.2%）、SDv4/v5 由 DIRE+Noise 共同把關（合計 41%）、StyleGAN 則啟動 FFT（16.0%）。這比預期的家族二分法更精細，更能解釋為何五條流缺一不可。」

### 2E. 三個有力論點

1. **CLIP 是「萬能語意基座」**：在所有家族 47-57%，呼應 Shapley +30.33% 最大貢獻
2. **真正的「Diffusion 殺手」不是 DIRE 而是 Noise+DCT**：在 ADM、MJ、SDv4/v5、wildfake 上 Noise+DCT 常合計 > 40%
3. **DCGAN 的特殊性**：CLIP 獨佔 68.86%、DIRE 僅 0.25%（幾乎為零）—— 模型對「簡單 GAN」只用語意流就 100% 正確

---

## Level 3：失敗案例（壓軸）

### 3A. 篩案例（混合做法）

兩步驟：
1. **自動篩出候選池**：FN/FP 各篩出「信心 70-90% 的自信錯誤」（不要邊界 case），得到 20-30 張候選
2. **手挑最終 3-5 張**：從候選池中挑「最有故事」的，**刻意涵蓋不同 generator**

**配合 Level 2 新敘事**：優先挑「per-generator 預期被打破」的案例，例如：
- ADM 圖被誤判時，DCT 是不是真的沒激活？
- MJ 圖被誤判時，Noise heatmap 是不是跑到無關區域？
- DCGAN 圖被誤判時，CLIP 主導為什麼這次失靈？

這樣 Level 2 + Level 3 直接綁在一起講，敘事更緊。

優點：篩選邏輯可重現（可寫進論文），但保留視覺品質。論文裡敘述：「我們從信心區間 [0.70, 0.90] 的失敗案例中，按 generator 多樣性挑選 N 張代表案例」。

### 3B. 對每個 case 產出

1. Level 1 六格圖
2. Level 2 attention bar
3. 人工診斷一行：「Fusion 給 Noise 38% 權重，但 Noise heatmap 在背景雜訊，未抓到主體 → 誤導融合」

### 3C. 論文價值

一般論文只 cherry-pick 成功案例。這節能直接拉高方法可信度。

---

## Defense Layer

### D1. Adebayo Cascading Randomization

- 把 fusion model 的 layer 由上到下逐層權重隨機化，每次重跑 Level 1 Grad-CAM
- **預期**：隨機化越多層，熱力圖與原始 SSIM 越低（最後變雜訊）
- **若不成立**：Grad-CAM 沒抓模型內部資訊，只反映 input bias → 整套解釋失效
- **規模**：50-100 張圖即可
- **輸出**：SSIM vs. 隨機化層數曲線

### D2. Insertion / Deletion AUC（可選）

來源：**RISE**（Petsiuk et al., 2018, BMVC）。Grad-CAM 原論文 section 4.4 有相關的 occlusion analysis，但「曲線 + AUC」的標準形式是 RISE 提出的，後續 Grad-CAM++ / Score-CAM 等都拿這個當 benchmark。

兩條曲線一起做，互相印證：

**Deletion**：拿走重要區域，看 prob 崩多快
- 對一張圖按 Grad-CAM 強度排序 pixel
- 從**強到弱**逐步遮黑 top-k%，餵回模型，記錄 fake-prob
- x = 遮掉比例，y = fake-prob，**曲線越早崩越好**
- Deletion AUC **越低越好**

**Insertion**：把重要區域加回，看 prob 漲多快
- 從一張全黑/全模糊圖開始
- 從**強到弱**逐步把 top-k% pixel 加回去（注意方向同樣是強到弱，不是弱到強）
- x = 加回比例，y = fake-prob，**曲線越早上升越好**
- Insertion AUC **越高越好**

**Baseline 對照**：另外跑一條「隨機順序」的 deletion/insertion 曲線。如果你的 Grad-CAM AUC 跟 random 差不多，代表 saliency 沒抓到模型決策依據。

**輸出**：對整個 test set 平均，得到
> 「Ours: Deletion AUC = X (lower is better), Insertion AUC = Y (higher is better)
> Random baseline: Deletion AUC = X', Insertion AUC = Y'」

**注意事項**（風險 D）：
- DIRE / Noise / CLIP 三條像素空間流可直接套
- FFT / DCT 兩條頻率域流，「遮掉一個頻率係數」的意義需另外定義；保守做法是這兩條跳過，caption 註記原因

---

## 實作優先序

### Phase 1：核心三層（必做）

| Day | 任務 | 對應文獻 |
|---|---|---|
| 1 | Level 1A（CNN 四條流 Grad-CAM）+ 接 demo | 1 |
| 2-3 | Level 1B（CLIP Chefer relevance）| 2 |
| 4 | Level 2 attention + ablation cross-check | 3 |
| 5 | Level 3 失敗案例（用 Level 1+2 結果） | 6 |

完成 Phase 1 後，已經是「真 Grad-CAM + 有方法論的 attention 解釋 + 失敗診斷」。可以先寫論文初稿。

### Phase 2：Defense 強化（時間允許再做）

| Day | 任務 | 對應文獻 |
|---|---|---|
| 6 | D1 sanity check | 4 |
| 7 | D2 deletion/insertion curve | 8 |

**砍掉風險**：Phase 2 不做論文仍然站得住，只是少一層 faithfulness 驗證。如果口委不問就過了，問了就在 limitations 寫「未進行 saliency faithfulness 量化驗證，留作 future work」。

---

## 待決事項

- [x] ~~CLIP 流要走 rollout 還是 Chefer relevance？~~ → **採用 Chefer relevance**（主讀 CVPR 版，輔讀 ICCV 版確認方向）
- [x] ~~Test set 分組要切到多細的 generator？~~ → **兩個都做**：主圖按家族（Real/GAN/Diffusion），附圖按具體 generator
- [x] ~~FN/FP 抽樣是手挑還是依信心區間自動選？~~ → **混合做法**：自動篩信心 [0.70, 0.90] 候選池，再手挑涵蓋不同 generator
- [ ] D1 隨機化要做完整 cascade 還是只做 top-k layer？（**暫緩，Phase 2 再決定**）
