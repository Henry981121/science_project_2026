# 科展實驗結果統整 — 撰寫報告用

> 統整日期：2026-05-24
> 主模型：FusionDetectorGRL (5-stream + Cross-Attention + GRL)
> 結果根目錄：`C:\Users\harry\OneDrive\Desktop\outputs\`

---

## 0. 資料集規模（測試集）

| 集合 | 樣本數 | 說明 |
|---|---|---|
| **G1**（同分佈測試集） | 23,599 | 8 個訓練內生成器 + Real |
| **G2**（跨生成器測試集） | 34,596 | 5 個未見生成器：dcgan_unseen, fursona_gan, waifu_gan, wildfake_ddim, wildfake_other |
| **wildfake_other (wfother)** | 5,000 fake + 17,298 real | G2 中最難子集（unknown generator） |

---

## 1. 主模型最終效能

### 1.1 訓練成果（`outputs/3.22output/main_grl/final_results.json`）

| 指標 | G1 (Val) |
|---|---|
| Best epoch | 28 / 30 |
| **Val ACC** | **98.54%** |
| **Val AUC** | **99.85%** |
| **Val F1** | **0.9837** |
| 訓練時間 | 6 分鐘 |
| 參數量 | 9.6M |

**Config：** 5 stream (CLIP/FFT/DCT/DIRE/Noise) + Cross-Attention + GRL (λ_src=0.1, λ_grl=0.05, γ=10.0) + Curriculum (easy→medium→hard)

### 1.2 跨生成器評估（`outputs/3.22output/exp_d_grl/generalization_results.json`）

| 集合 | ACC | AUC | F1 | Gap vs G1 |
|---|---|---|---|---|
| G1 | 98.44 | 99.81 | 0.9827 | — |
| G2 | 88.06 | 95.25 | 0.8665 | **−10.38%** |

**G3 逐生成器（G1 內 8 個）：** 全部 ≥98.15% ACC（最低 wildfake，最高 stylegan 99.04%）

### 1.3 多 Seed 穩定性（`outputs/3.22output/exp_i/multi_seed_results.json`）

| Seed | Val ACC |
|---|---|
| 42 | 98.43 |
| 123 | 98.40 |
| 456 | 98.46 |
| **Mean ± Std** | **98.43 ± 0.02** |

---

## 2. 消融實驗（5 串流分析）

### 2.1 單流效能（`outputs/3.22output/exp_a/results.json`）

| Stream | ACC | AUC | F1 |
|---|---|---|---|
| **CLIP** | **97.88** | **0.998** | 0.976 |
| DCT | 83.48 | 0.911 | 0.809 |
| DIRE | 81.77 | 0.900 | 0.791 |
| FFT | 76.84 | 0.846 | 0.720 |
| Noise | 64.11 | 0.664 | 0.515 |

### 2.2 5 流組合（part1_combinations）與 LOO（part2_loo）

`outputs/3.22output/exp_b/results.json`

**全部 5 流：ACC = 98.19**

**LOO（拔掉某流的全模型 ACC）：**

| 拔掉 | ACC | 掉幅 |
|---|---|---|
| CLIP | 90.13 | **−8.06** |
| DCT | 97.96 | −0.24 |
| FFT | 98.07 | −0.13 |
| DIRE | 98.10 | −0.09 |
| **Noise** | **98.17** | **−0.025** |

> CLIP 主導；Noise 在 LOO 上看似可拋棄 → 但 Shapley 與 Noise Decisive 分析證明否。

### 2.3 融合方法比較（`outputs/3.22output/exp_c/results.json`）

| 方法 | ACC | AUC |
|---|---|---|
| Concat + MLP | 97.76 | 0.998 |
| Weighted Fusion | 97.84 | 0.998 |
| **Cross-Attention（我們）** | **98.20** | **0.998** |

**跨生成器（G2）gap：** Cross-Attention −8.04（與其他方法相當）→ 主要優勢在 G1，G2 由 GRL 提供

---

## 3. SOTA 對比（`outputs/3.22output/exp_f_v2/sota_comparison_v2.json`）

| 模型 | ACC | AUC | F1 | 參數 (M) |
|---|---|---|---|---|
| ResNet50 (fine-tune) | 95.51 | 99.14 | 0.949 | 23.5 |
| EfficientNet-B4 | 91.24 | 97.19 | 0.902 | 17.6 |
| CLIP Linear Probe | 91.34 | 97.19 | 0.905 | 0.001 |
| **Ours (GRL v2.1)** | **98.44** | **99.81** | **0.983** | 9.6 |

---

## 4. 強健性（degradation）測試

### 4.1 本模型 13 種劣化（`outputs/3.22output/exp_g/robustness_results.json`）

| 劣化 | ACC | 掉幅 |
|---|---|---|
| Original | 98.45 | 0 |
| JPEG q=70/50/30 | 97.8 / 98.6 / 97.85 | < 1% |
| Resize 0.75x / 0.50x / **0.25x** | 97.5 / 96.35 / **85.85** | -1 ~ **−12.6** |
| Blur σ=1/2/**3** | 97.45 / 89.65 / **77.5** | -1 ~ **−20.95** |
| Noise σ=10/25/**50** | 95.35 / 93.45 / **84.35** | -3 ~ **−14.1** |

### 4.2 vs 三個 baseline（`outputs/3.22output/exp_g_v2 all model/robustness_all_models.json`）

完整對照表（每個模型 × 13 種劣化的 ACC/AUC），用於畫對照圖：`robustness_all_models.png`

---

## 5. 跨生成器分解（method × dataset 矩陣）

`outputs/3.22output/method_x_dataset_matrix.json`

每個 stream（CLIP/FFT/DCT/DIRE/Noise/Fusion+GRL）× 13 個生成器（包含 OOD：wildfake_other 等）的 ACC：

**最難 OOD = `wildfake_other`：CLIP 74.6, Fusion 70.5**（這就是後面要救的對象）

---

## 6. 失敗案例分析（`outputs/3.22output/exp_e supplementary/step11_error_analysis.json`）

逐 OOD 生成器的 False Negative Rate：

| OOD 生成器 | N | FN | FNR |
|---|---|---|---|
| dcgan_unseen | 1,500 | 0 | 0.0% |
| wildfake_ddim | 5,000 | 167 | 3.34% |
| fursona_gan | 2,798 | 135 | 4.82% |
| waifu_gan | 3,000 | 705 | 23.5% |
| **wildfake_other** | **5,000** | **2,880** | **57.6%** ← 主要痛點 |

---

## 7. NSS / NSS++ 獨立 OOD 信號（Variants A/B/C/D — 2×2 設計）

`outputs/exp_nss_2x2/2x2_summary_table_final.csv`

| 變體 | 特徵維度 | 形式 | G1 ACC | G2 ACC | wfother ACC | wfother FNR |
|---|---|---|---|---|---|---|
| **A.** Original NSS Fused (5+36) | 36 | Cross-Attention | — | — | 68.79 | 58.50 |
| **B.** Original NSS Independent | 36 | Standalone Mahalanobis | 66.97 | 63.05 | 9.52 | 90.48 |
| **C.** NSS++ Fused (5+79) | 79 | Cross-Attention（重訓） | 98.49 | 88.39 | **85.75** | 58.60 |
| **D.** NSS++ Independent | 79 | Standalone Mahalanobis | 65.42 | 65.00 | 10.70 | 89.30 |

**結論：** NSS 無論原版或 ++，作為「獨立 OOD 信號」皆飽和失效（B/D 兩變體 ≈ 隨機）。融合（A/C）有效但邊際提升有限 → 需要更獨立的信號（CA、Energy）。

**Late Fusion 試驗（`outputs/late_fusion/late_fusion_results.json`）：**
- BNSS / DNSS 在所有切片（G1/G2/wfother）的 best α = 0.0，即融合後 AUC = base AUC，**NSS 無加成**
- α 從 0 一路單調下降到 1.0 全 NSS 時崩壞

---

## 8. 三大突破：CA + Energy 三信號融合 ⭐

### 8.1 Main + CA Late Fusion（`outputs/ca_late_fusion/ca_late_fusion_results.json`）

CA = Chromatic Aberration（色差）作為獨立 OOD 信號

| 切片 | Baseline AUC | +CA Best AUC | Δ |
|---|---|---|---|
| G1 | 99.81 | 99.81 | 0.00 |
| G2 | 95.25 | 95.25 | 0.00 |
| wfother | 84.78 | 84.78 | 0.00 |

→ CA 單獨融合**無提升**（AUC 0.546），但配合 Energy 可解鎖。

### 8.2 Triple Fusion: Main + Energy + CA（`outputs/main_plus_ca/main_plus_ca_results.json`）⭐

| 切片 | Baseline | +Energy (α=0.05) | **+Triple (α=0.05, β=0.05)** | Δ Total |
|---|---|---|---|---|
| G1 AUC | 99.81 | 99.81 | 99.81 | 0.00 |
| G2 AUC | 95.25 | 95.75 | **96.41** | **+1.16** |
| **wfother AUC** | **84.78** | 87.92 | **90.07** | **+5.30** ⭐ |

→ **wfother AUC 從 84.78 → 90.07，提升 +5.30%**（科展核心成果）

### 8.3 Triple Override Strategies（`outputs/triple_override/triple_override_results.json`）

針對 wfother FNR 57.6% 的策略掃描：

| 策略 | wfother FNR | G1 FPR | 備註 |
|---|---|---|---|
| S0 baseline | 57.60% | 1.51% | — |
| S1 Energy override @0.7 | 25.08% | 15.89% | FNR 大降但 FPR 升 |
| S4 AND@E0.6_C0.5 | 31.78% | 12.50% | 平衡 |
| **S5 Tiered E0.7→E0.5+C0.6** | **18.62%** | **21.56%** | **FNR 從 57.60% → 18.62%** ⭐ |
| S6 Vote (soft) | 56.38% | 1.63% | wfother AUC 89.95 |

---

## 9. 五大驗證方法（`outputs/validation_methods/`）

### 9.1 Shapley Value（`shapley/shapley_results.json`）

| Stream | Shapley | LOO Drop | 單流 ACC |
|---|---|---|---|
| **CLIP** | **30.33** | 8.06 | 97.88 |
| DCT | 19.43 | 0.24 | 83.48 |
| DIRE | 18.71 | 0.09 | 81.77 |
| FFT | 16.96 | 0.13 | 76.84 |
| **Noise** | **12.77** | 0.025 | 64.11 |

**Sum = 98.19 = Full 5-stream ACC ✓**（efficiency property 成立）

**Noise 的 LOO=0.025 但 Shapley=12.77**：Noise 在邊際情境下仍有貢獻（見 §9.5）

### 9.2 mCE (mean Corruption Error, vs EfficientNet-B4 baseline)（`mce/mce_results.json`）

| 模型 | mCE（越低越好） |
|---|---|
| EfficientNet-B4 | 1.000（基準） |
| ResNet-50 | 0.97 |
| CLIP Linear Probe | 0.53 |
| **Ours (GRL)** | **0.38** ⭐ |

### 9.3 ECE (Expected Calibration Error)（`ece/ece_results.json`）

| 切片 | ECE | 錯誤樣本平均信心 |
|---|---|---|
| G1 | 0.0139 | 0.951 |
| G2 | 0.1132 | 0.973 |
| wfother (子集) | 0.1353 | 0.982 |
| **wfother ONLY (fake only)** | **0.561** | **0.986** ← OOD 嚴重 overconfident |

### 9.4 Silhouette + MMD（`silhouette_mmd/silhouette_mmd_results.json`）

**Silhouette（embedding 內聚度）：**
- G1 fake by generator: 0.527
- G2 fake by generator: 0.354 ← OOD 分群較鬆
- G1 real vs fake: 0.604 ← 兩類分離良好

**MMD（vs G1 fake 的分佈距離）：**

| OOD 生成器 | MMD_linear | MMD_rbf |
|---|---|---|
| dcgan_unseen | 14.90 | 0.912 ← 距離最遠 |
| wildfake_ddim | 10.48 | 0.702 |
| wildfake_other | 9.98 | 0.566 |
| fursona_gan | 6.03 | 0.428 |
| waifu_gan | 5.69 | 0.383 |
| Real vs Real (control) | 0.07 | 0.011 ← baseline |

### 9.5 DAUC/IAUC (Petsiuk 2018, faithfulness)

腳本：`day_dauc_iauc.py`
- 排除 Noise（SRM residual 非 Grad-CAM）
- 測 4 流：CLIP / FFT / DCT / DIRE
- 23 張 G2 圖像
- 結果：CLIP gap −0.081（最好），DIRE −0.174（最差）
- 解釋：全部負值反映「多流融合下單流 cam 無法獨立完整解釋預測」

---

## 10. Noise Decisive 分析（`outputs/3.22output/level2/`）

腳本：`noise_decisive_analysis.py`, `noise_decisive_plot.py`
資料：`level2/noise_decisive.json`, `level2/attention_ablation.csv`
圖：`level2/noise_decisive_bar.png`

**三種「Noise 是決定因素」定義（N = 23,599）：**

| 定義 | N | % |
|---|---|---|
| D1: Noise 排第一名（top-1 ablation） | 79 | **0.33%** |
| **D2: Ablate Noise 後預測翻轉（過 0.5 邊界）** | **1,975** | **8.37%** ⭐ |
| D3: Noise 排前二名 | 312 | 1.32% |

**逐家族（D2 翻轉率）：**

| Family | N | D2 % |
|---|---|---|
| GAN | 1,570 | 3.38% |
| Real | 12,983 | 8.06% |
| **Diffusion** | **9,046** | **9.68%** ← Noise 對 Diffusion 最關鍵 |

**逐生成器最高（D2）：** sdv4 14.48%, sdv5 14.25%, midjourney 11.53%, glide 9.29%

**結論：** 雖然 Noise LOO 影響極小（0.025），但有 8.37% 的樣本「沒有 Noise 就會翻車」→ 為 Noise 的保留提供統計證據。

---

## 11. NSS 補充驗證實驗

### 11.1 EXP-H v1（早期 concat 形式）`outputs/3.22output/exp_h/nss_results.json`

| 設定 | G1 | G2 | Gap |
|---|---|---|---|
| Baseline 5-stream | 95.47 | 89.02 | −6.45 |
| NSS only (36-dim) | 72.23 | 58.42 | −13.81 |
| +NSS (5-stream + 36) | 95.47 | 87.20 | −8.27 |

### 11.2 EXP-H v2（NSS 投影成 6th stream）`exp_h_v2 補充NSS/nss_6stream_results.json`

6-stream GRL：G1=96.40 / G2=82.50 / Gap=−13.9 → **加 NSS 反而變差**

### 11.3 EXP-H v3（wildfake_other 專項）`exp_h_v3 補充NSS(OTHER)/wildfake_other_results.json`

| 模型 | ACC | AUC | FNR |
|---|---|---|---|
| 5-stream baseline | 71.11 | 0.843 | 54.36 |
| 6-stream + NSS proj | 68.79 | 0.833 | 58.50 ← 更差 |

→ NSS 對 OOD wildfake_other 無幫助。

---

## 12. 早期/補助實驗（重要參考）

| 目錄 | 內容 |
|---|---|
| `outputs/no_grl_baseline/` | 不加 GRL 的對照組（證明 GRL 必要） |
| `outputs/nss_original_phase1/`, `outputs/nss_plus_phase1/` | NSS / NSS++ 初步驗證（Variants 起點） |
| `outputs/dcct/`, `outputs/dcct_late_fusion/` | DCCT 嘗試（未進主結果） |
| `outputs/exp_e_showcase/` | 範例可視化 |
| `outputs/exp_e_gate/` | Energy gate 早期掃描 |
| `outputs/variant_c/` | NSS++ Fused 訓練細節 |
| `outputs/decision_gate/` | 決策閘門校準（linear / energy / dual / conditional） |
| `outputs/3.22output/exp_e supplementary/` | Grad-CAM, attention 視覺化, 頻譜圖, 誤判分析 |
| `outputs/3.22output/exp_TSNE/` | t-SNE 視覺化 |

---

## 13. 主要視覺化檔案位置

| 圖 | 路徑 |
|---|---|
| 訓練曲線 | （需從 `main_grl/final_results.json` history 重繪） |
| 強健性對照 | `outputs/3.22output/exp_g_v2 all model/robustness_all_models.png` |
| SOTA 對照 | `outputs/3.22output/exp_f_v2/sota_comparison_v2.png` |
| Grad-CAM 範例 | `outputs/3.22output/exp_e supplementary/step5_gradcam.png` |
| Real vs Fake Attention | `outputs/3.22output/exp_e supplementary/step6_attention_real_vs_fake.png` |
| 頻譜對比 | `outputs/3.22output/exp_e supplementary/step8_frequency_spectrum.png` |
| 誤判分析 | `outputs/3.22output/exp_e supplementary/step11_error_analysis.png` |
| t-SNE 4 類 | `outputs/3.22output/exp_TSNE/tsne_4cat.png` |
| Midjourney source dist | `outputs/3.22output/exp_TSNE/part1_midjourney_source_dist.png` |
| **Noise Decisive 長條** | `outputs/3.22output/level2/noise_decisive_bar.png` |
| OOD Pareto | `outputs/ood_calibrator/ood_pareto_analysis.png` |

---

## 14. 報告書建議章節對應

| 報告章節 | 對應實驗 / 數據來源 |
|---|---|
| **研究動機** | §6 失敗案例（wfother FNR 57.6%）, §7 NSS 飽和 |
| **方法 — 主模型** | §1 (FusionDetectorGRL) + §2.3 Cross-Attention |
| **方法 — 三信號融合** | §8 CA + Energy + Main |
| **實驗 — 主結果** | §1.1, §1.2, §1.3 |
| **實驗 — 消融** | §2.1, §2.2, §2.3 |
| **實驗 — 比較其他模型** | §3 SOTA, §4 強健性 |
| **實驗 — OOD/跨生成器** | §1.2, §5 矩陣, §6 失敗案例 |
| **核心突破** | §8.2 wfother AUC 84.78 → 90.07 |
| **驗證實驗 (五法)** | §9.1 Shapley, §9.2 mCE, §9.3 ECE, §9.4 Silhouette+MMD, §9.5 DAUC/IAUC |
| **可解釋性分析** | §9.5 + §10 Noise Decisive (8.37%) |
| **NSS 探索（負面結果）** | §7 + §11 |
| **討論 / 限制** | §4 強健性掉點（resize 0.25x, blur σ=3）+ §9.3 OOD overconfident |

---

## 15. 關鍵數字一覽表（報告書中可直接引用）

| 場景 | 數字 |
|---|---|
| 主模型 G1 ACC | **98.54%** |
| 主模型 G1 AUC | **99.85%** |
| 跨生成器 (G2) ACC | 88.06% |
| 跨生成器 (G2) AUC | 95.25% |
| Multi-seed 標準差 | **0.02%** |
| vs ResNet50 ACC 提升 | +2.93% |
| vs EfficientNet-B4 ACC 提升 | +7.20% |
| 強健性 mCE（vs EffNet） | **0.38** |
| **wfother AUC 提升（Triple）** | **84.78 → 90.07 (+5.30)** ⭐ |
| **wfother FNR 改善（Tiered S5）** | **57.60% → 18.62%（−38.98%）** ⭐ |
| **G2 AUC 提升（Triple）** | 95.25 → 96.41 (+1.16) |
| Shapley CLIP 貢獻 | 30.33（最高） |
| Shapley Noise 貢獻 | 12.77（仍非零） |
| **Noise Decisive D2** | **8.37%（1,975/23,599）** |
| 參數量 | 9.6M |
| 訓練時間 | 6 分鐘 |

---

## 16. 還沒做 / 可選未來工作

- DAUC/IAUC 只跑了 23 張，可擴大樣本
- Triple fusion 的 calibration（α/β）只試了 0.05 step，可細掃
- Chefer relevance（CVPR 2021）已實作但未納入主結果
- Cross-Attention 「Attention is not Explanation」討論可帶入 §9 限制
