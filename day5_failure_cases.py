"""
day5_failure_cases.py
=====================

Level 3 — 失敗案例候選篩選（XAI 計畫 Day 5 / 3A）。

從 attention_ablation.csv 篩出「模型自信卻答錯」的案例，輸出候選 CSV，
供人工挑 3-5 張代表案例做 Level 3 失敗診斷。

【篩選邏輯（可寫進論文，可重現）】
只取「自信錯誤」—— 模型對錯誤答案的信心落在 [conf_lo, conf_hi]
（預設 0.70-0.90）。不取信心 ~50% 的邊界 case（只是模型猶豫，不有趣），
也不取信心 ~100% 的極端 case（可能是標註錯誤）。

  - False Negative (FN)：AI 圖(label=1)被判為真
        模型對「真」的信心 ∈ [conf_lo, conf_hi]
        ⇔ fake_prob ∈ [1-conf_hi, 1-conf_lo]
  - False Positive (FP)：真圖(label=0)被判為 AI
        模型對「假」的信心 ∈ [conf_lo, conf_hi]
        ⇔ fake_prob ∈ [conf_lo, conf_hi]

【輸出】
  <outdir>/failure_candidates.csv
    每列一個候選，含 path / fail_type / generator / family / label /
    fake_prob_base / confidence（模型對錯誤答案的信心）/ difficulty /
    attn_* / ablimp_*
  —— 診斷時 attention 與 ablation 數字都在手邊。

【接下來（人工 + 隊友）】
1. 從 failure_candidates.csv 手挑 3-5 張「最有故事」、刻意涵蓋不同
   generator 的案例
2. 隊友對每張挑中的圖跑 day1_test.py 取 Level 1 六格圖
3. 配 Level 2 attention 圖 + 人工寫一句診斷

【純 CSV 篩選，不需 GPU / checkpoint，任何機器可跑】

【怎麼用】
    python day5_failure_cases.py
    python day5_failure_cases.py <attention_ablation.csv 路徑>
    python day5_failure_cases.py --conf-lo 0.70 --conf-hi 0.90

【相依】pandas
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

STREAMS = ["clip", "fft", "dct", "dire", "noise"]
ATTN_COLS = [f"attn_{s}" for s in STREAMS]
ABL_COLS = [f"ablimp_{s}" for s in STREAMS]


def main() -> None:
    ap = argparse.ArgumentParser(description="Level 3 失敗案例候選篩選")
    ap.add_argument("csv", nargs="?",
                    default="outputs/level2/attention_ablation.csv",
                    help="attention_ablation.csv 路徑")
    ap.add_argument("--outdir", default="outputs/level3", help="輸出資料夾")
    ap.add_argument("--conf-lo", type=float, default=0.70,
                    help="信心區間下界（預設 0.70）")
    ap.add_argument("--conf-hi", type=float, default=0.90,
                    help="信心區間上界（預設 0.90）")
    args = ap.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        sys.exit(f"[day5] 找不到 CSV: {csv_path}")
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)
    need = ["path", "label", "generator", "fake_prob_base"]
    missing = [c for c in need if c not in df.columns]
    if missing:
        sys.exit(f"[day5] CSV 缺欄位: {missing}\n"
                 "     （需要 day4_2b_ablation.py 產出的 attention_ablation.csv）")

    lo, hi = args.conf_lo, args.conf_hi
    if not (0.5 < lo < hi < 1.0):
        sys.exit(f"[day5] 信心區間不合理: [{lo}, {hi}]（需 0.5 < lo < hi < 1.0）")
    p = df["fake_prob_base"]

    # FN：AI 圖(label=1)被判真 —— 對「真」的信心 = 1-fake_prob ∈ [lo,hi]
    fn = df[(df["label"] == 1) & (p >= 1 - hi) & (p <= 1 - lo)].copy()
    fn["fail_type"] = "FN"
    fn["confidence"] = 1 - fn["fake_prob_base"]

    # FP：真圖(label=0)被判 AI —— 對「假」的信心 = fake_prob ∈ [lo,hi]
    fp = df[(df["label"] == 0) & (p >= lo) & (p <= hi)].copy()
    fp["fail_type"] = "FP"
    fp["confidence"] = fp["fake_prob_base"]

    cand = pd.concat([fn, fp], ignore_index=True)
    if cand.empty:
        sys.exit(f"[day5] 信心區間 [{lo}, {hi}] 內沒有任何自信錯誤案例。"
                 "可放寬 --conf-lo / --conf-hi。")

    # 欄位排版：metadata 在前，attn_* / ablimp_* 在後
    keep = ["path", "fail_type", "generator", "label", "fake_prob_base",
            "confidence"]
    for opt in ("family", "difficulty"):
        if opt in cand.columns:
            keep.append(opt)
    keep += [c for c in ATTN_COLS + ABL_COLS if c in cand.columns]
    cand = cand[keep].sort_values(["fail_type", "generator", "confidence"],
                                  ascending=[True, True, False])

    out_path = outdir / "failure_candidates.csv"
    cand.to_csv(out_path, index=False)

    # ── console 摘要 ──
    print("=" * 56)
    print(f"  Level 3 失敗案例候選  (信心區間 [{lo}, {hi}])")
    print("=" * 56)
    print(f"  資料總數 : {len(df)}")
    print(f"  候選總數 : {len(cand)}  (FN={len(fn)}, FP={len(fp)})")
    print("\n  FN（AI 圖被判真）— 按 generator：")
    if len(fn):
        for g, n in fn["generator"].value_counts().items():
            print(f"    {g:14s} {n}")
    else:
        print("    （無）")
    print("\n  FP（真圖被判 AI）— 按 generator：")
    if len(fp):
        for g, n in fp["generator"].value_counts().items():
            print(f"    {g:14s} {n}")
    else:
        print("    （無）")
    print(f"\n[day5] saved {out_path}")
    print("[day5] 下一步：從這份 CSV 手挑 3-5 張（涵蓋不同 generator），"
          "隊友再對每張跑 day1_test.py 取六格圖。")


if __name__ == "__main__":
    main()
