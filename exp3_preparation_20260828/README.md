# EXP3C/D/F/G 前置準備

此資料夾只存放 EXP3C、EXP3D、EXP3F、EXP3G 的前置模板、檢查結果與執行說明，不會修改既有 EXP2/EXP3 正式結果。

## 目前狀態

| 實驗 | 前置狀態 | 尚缺少的正式條件 |
|---|---|---|
| EXP3C | 已建立 content 標註模板 | 可靠、完整且平衡的人工或資料集 content labels |
| EXP3D | 已有 4 x 4 severity manifest | transformed images、三流特徵 cache、row identity 驗證 |
| EXP3F | 已建立 baseline registry | 可重現的官方權重或相同 protocol 的 retrained baseline |
| EXP3G | 已建立外部資料匯入模板 | 全新且與訓練/validation/cross/holdout 不重疊的外部資料 |

## 執行順序

1. `python prepare_exp3c_template.py --cache-dir ... --out-dir ...`
2. 由人工或可信資料來源填寫 `exp3c_content_labels_template.csv`，再執行 `validate_exp3c.py`。
3. 依 `exp3d_pipeline_spec.md` 生成四種退化與三流 cache，再執行 `validate_exp3d.py`。
4. 填寫 `exp3f_baseline_registry.csv`，只有完成官方程式/權重與資料協議確認的模型才標記 `ready_for_run=true`。
5. 填寫 `exp3g_external_manifest_template.csv`，再執行 `validate_exp3g.py`；確認 zero overlap 後才可建立正式評估輸出。

## 明確限制

- 不從檔名猜測 EXP3C 的 content。
- 不把只有 manifest 的 EXP3D 當成已完成的 robustness 實驗。
- 不把論文數值直接當成 EXP3F 的本研究實驗結果。
- 不把任何既有 split 或 seen-generator holdout 當成 EXP3G 外部資料。
- `locked_test` 在所有設定凍結前維持未使用。

## EXP3D 全量特徵抽取

64 筆 sanity 與全量抽取均已通過；全量抽取命令如下，使用 `--skip-existing` 可從已完成的 split 繼續。正式 EXP3D 評估前仍須保留並引用 `exp3d_cache_summary.csv` 的對齊檢查結果。

```powershell
$manifest = "C:/Users/harry/OneDrive/Desktop/science_project_v3/science_project_2026_exp/exp3_preparation_20260828/exp3d_all_feature_manifest.csv"
$cache = "C:/Users/harry/OneDrive/Desktop/science_project_v3/exp3d_feature_cache_20260828"
python C:/Users/harry/OneDrive/Desktop/science_project_v3/handoff_dct_clip_dinov2_20260826/code/extract_features.py `
  --manifest $manifest --cache $cache --streams dct,clip,dinov2 `
  --device cuda --workers 8 --batch 32 --size 256 --skip-existing
```

依目前實測，DCT 約 230 張/秒；CLIP、DINOv2 受 GPU 與模型推論速度影響，完整 551,376 筆需要長時間執行。這是計算時間限制，不是人工標註工作。

## 全量完成結果

- DCT、CLIP、DINOv2：各 16/16 split 完成。
- 每個 split：34,461 筆；總計 551,376 筆。
- 三流共同 `valid_all=1`：551,376；無效：0。
- 最終 cache：`C:/Users/harry/OneDrive/Desktop/science_project_v3/exp3d_feature_cache_20260828/`。

## 既有資料來源

- EXP3D manifest: `C:/Users/harry/OneDrive/Desktop/science_project_v3/exp3_data/exp3d_robustness/cross_generator_multiseverity.csv`
- seen holdout: `C:/Users/harry/OneDrive/Desktop/science_project_v3/exp3_data/seen_generator_holdout/seen_generator_holdout.csv`
- feature cache: `C:/Users/harry/OneDrive/Desktop/science_project_v3/handoff_dct_clip_dinov2_20260826/feature_data/crop`
- frozen-model checkpoints: `science_project_2026_exp/outputs/exp2a_retrained_excluding_holdout_v2/<model>/best_model.pth`
