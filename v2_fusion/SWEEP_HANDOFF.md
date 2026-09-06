# 三軸因子設計 sweep — 執行說明

## 這在回答什麼問題

不是「哪一種 fusion 最準」，是**「fusion 設計的哪一個維度真的重要」**。
所以跑的是一個 2×2×2 因子設計，不是一串有名字的方法。

| 軸 | 兩個值 |
|---|---|
| A. order | `1` = 只有加權和 / `2` = 加權和 + 交互項 |
| B. weighting | `static` = 學到的常數權重 / `gated` = 權重隨樣本而變 |
| C. level | `feat` = d 維表示上融合 / `dec` = 各流 logits 上融合 |

八格 + 參照組 `concat` + 單流參照 `single`。

**為什麼不直接比 TFN / GMU / MBT**：那些方法同時差在交互階數、參數量、
權重來源、正規化位置、初始化。差異出現時無法歸因。這裡每格只放
「具備該性質的最小實現」，其他一切共用。

## 主指標

`gap_auc = val_auc − test_auc`（泛化落差），**越小越好**。

不是準確率 —— val 已經飽和在 0.999，八格會全部擠在小數點後三位。

## 執行

```bash
git pull

# 0. 靜態驗證（不需 GPU、不需資料，約 30 秒）
python -m v2_fusion.sanity            # 應該 27/27

# 1. 先看會跑什麼、各自多大（不訓練）
python -m v2_fusion.sweep --stage pilot --dry-run

# 2. PILOT —— 先跑這個（10 runs）
python -m v2_fusion.sweep --cache-dir <handoff>/feature_data/crop --stage pilot
```

### pilot 怎麼判讀（這一步決定要不要跑滿）

pilot 只跑 A 軸兩端（`o1.static.feat` vs `o2.static.feat`）各 5 seeds。
它回答的**不是「哪個好」，是「這個實驗有沒有解析度」**：

- 兩組 95% 信賴區間**分得開** → 有解析度，繼續
- **完全重疊** → 跑滿八格也只是八條一樣的線。先去處理解析度，不要硬跑

```bash
# 3. pilot 過關後
python -m v2_fusion.sweep --cache-dir <cache> --stage grid     # 45 runs
python -m v2_fusion.sweep --cache-dir <cache> --stage single   # 15 runs
python -m v2_fusion.sweep --cache-dir <cache> --stage full     # 135 runs
```

中斷了直接重跑同一行 —— 已完成的 run 會跳過（`--force` 強制重跑）。

### 要跑多久

**用 pilot 的實測去推，不要用估的。** 每個 run 的耗時會印在畫面上，也存在
`results.json` 的 `minutes` 欄。

參考點：EXP3B 的 legacy hybrid 一個 run 是 **4.32 分鐘**（30 epochs）。這裡的
cell 模型小很多（1.23M vs 8.39M，而且沒有 self-attention 層），應該更快。
以 2-4 分鐘/run 推算：pilot 約 20-40 分鐘、grid 約 1.5-3 小時、
full 約 4.5-9 小時（可以放著跑一晚，斷了會續跑）。

## 輸出

`outputs/v2_sweep/<stage>/sweep_results.csv` —— 一列一個 run，三軸座標是獨立欄位，
可以直接丟進統計軟體做因子分析。`in_factorial=0` 的列是參照組，**不要**放進因子回歸。

每個 run 另有 `outputs/v2_sweep/<stage>/<run_name>/results.json`。

## 三個必看的欄位

| 欄位 | 意思 |
|---|---|
| `gap_auc` | 主指標 |
| `attn_std_mean` | readout 權重的**逐樣本**標準差。趨近 0 → 那組權重其實是常數，gating/attention 相對於固定加權平均沒有增益 |
| `interact_scale` | 交互項學到的權重。趨近 0 → 模型自己認為二階交互沒用 |

後兩個是中介變數，不是效能指標。它們回答「為什麼」，而準確率只回答「有沒有」。

## 三條紀律（由 script 強制，不靠自律）

1. **參數量對齊**：每個 cell 只有一個自由度 `width`，二分搜尋調到同一預算。
   實測落在預算的 95-100%。沒有這步，「二階比較好」與「二階比較大」分不開。
2. **只換那一個模組**：adapter、學習率、epoch、early stopping 全部共用。
3. **多 seed**：預設 5 個，報 t 分布的 95% 信賴區間。**區間重疊就不要宣稱誰贏。**

## 已知的邊界（論文要寫進去）

每條流只有一個全域向量 → 融合層只看到 3 個 token。這個 regime 下
attention 幾乎必然退化成固定加權（`attn_std_mean` 會證實這件事）。

**結論只能宣稱在 descriptor-level fusion 範圍內成立**，不能外推到有
patch token 的情形。要涵蓋後者需要重存 cache（見下）。

## 現有結果不受影響

`fusion_cell=None` 時走的還是原本的 `fusion_mode` 路徑，模型參數量仍是
**8,391,052**，2026-08-31 那批數字完全可重現。sanity 有一條回歸測試守著這件事。

## 前提（不做的話後面全白跑）

- `cross_generator_test` 先去掉重複的 5,732 筆
- 排除 `dcgan_unseen`（70.5% 與 train/val 重疊）
- locked `test` split 繼續不要開
