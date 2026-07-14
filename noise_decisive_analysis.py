"""
noise_decisive_analysis.py
============================
雖然 Noise 在「平均 ablation importance」上最低（0.0072），但在某些圖上
反而是模型靠 Noise 抓對的。本腳本量化「Noise 是決定因素」的比例。

使用三種定義（互相獨立）：

  D1. Noise 是 top-1 重要：
        該圖五流 ablation importance 排序中，Noise 排第一名。

  D2. Noise 是「臨界決策因子」：
        |Δ fake-prob (ablate noise)| ≥ τ_flip 且基線預測會被翻轉。
        （ablate Noise 後若 fake_prob 跨過 0.5 邊界 → 預測類別變了）

  D3. Noise 顯著貢獻（top-2）：
        Noise ablation importance 排前兩名。

逐家族 / 逐生成器 / 逐 真實/偽造類別 拆解比例。
"""

import os, sys, json
os.environ.setdefault('PYTHONIOENCODING', 'utf-8')

from pathlib import Path
import numpy as np
import pandas as pd

CSV = Path(r'C:\Users\harry\OneDrive\Desktop\outputs\3.22output\level2\attention_ablation.csv')
OUT = Path(r'C:\Users\harry\OneDrive\Desktop\outputs\3.22output\level2\noise_decisive.json')
STREAMS = ['clip', 'fft', 'dct', 'dire', 'noise']


def main():
    print("=" * 70)
    print("  Noise Decisive Analysis")
    print("=" * 70)

    df = pd.read_csv(CSV)
    print(f"\n[data] {CSV}")
    print(f"  rows: {len(df)}")
    print(f"  cols: {list(df.columns)}")

    abl_cols = [f'ablimp_{s}' for s in STREAMS]
    # ablation importance is |delta fake_prob| (magnitude)
    abl_mat = df[abl_cols].values  # (N, 5)

    # ─── D1: Noise 排第一名 ─────────────────────────────────────────
    rank1_idx = np.argmax(abl_mat, axis=1)  # 0..4
    noise_idx = STREAMS.index('noise')
    is_noise_top1 = (rank1_idx == noise_idx)

    # ─── D3: Noise 排前二名 ─────────────────────────────────────────
    # 排序每張圖的 5 個 ablimp，取最高兩個的 stream index
    ranked = np.argsort(-abl_mat, axis=1)  # 降序
    is_noise_top2 = (ranked[:, 0] == noise_idx) | (ranked[:, 1] == noise_idx)

    # ─── D2: Noise 是「臨界決策因子」(prediction flip) ───────────
    # 邏輯：baseline fake_prob 是 fake_prob_base；
    #       拿掉 noise 後新 prob = base ± abl_noise（方向不可知，因 ablimp 是絕對值）
    # 由於 CSV 只記 |delta|，無法直接判定方向，採近似：
    #   若 |delta noise| > min(base, 1-base) → 「足以翻轉預測」
    base = df['fake_prob_base'].values
    abl_noise = df['ablimp_noise'].values
    margin_to_edge = np.minimum(base, 1 - base)  # 距離 0.5 邊界的距離
    is_noise_flip = abl_noise >= margin_to_edge  # 足以翻過 0.5

    n_total = len(df)
    print(f"\n[total] N = {n_total}")

    def stats(mask, name):
        n = mask.sum()
        pct = n / n_total * 100
        print(f"\n  {name}")
        print(f"    Count: {n} / {n_total}  ({pct:.2f}%)")
        return n, pct

    print()
    print("=" * 70)
    print("  Three definitions of 'Noise is decisive'")
    print("=" * 70)
    n1, p1 = stats(is_noise_top1,  "D1. Noise has TOP-1 ablation importance")
    n2, p2 = stats(is_noise_flip,  "D2. Ablating Noise alone FLIPS prediction (crosses 0.5)")
    n3, p3 = stats(is_noise_top2,  "D3. Noise in TOP-2 ablation importance")
    n_intersect_12 = (is_noise_top1 & is_noise_flip).sum()
    n_union_12 = (is_noise_top1 | is_noise_flip).sum()
    print(f"\n  D1 AND D2 (top-1 + flips):   {n_intersect_12} / {n_total} ({n_intersect_12/n_total*100:.2f}%)")
    print(f"  D1 OR  D2 (either):           {n_union_12} / {n_total} ({n_union_12/n_total*100:.2f}%)")

    # ─── Per-family ─────────────────────────────────────────────
    print()
    print("=" * 70)
    print("  Per-FAMILY breakdown")
    print("=" * 70)
    fam_table = []
    for fam, fam_mask in [('Real', df['family'] == 'Real'),
                           ('GAN', df['family'] == 'GAN'),
                           ('Diffusion', df['family'] == 'Diffusion')]:
        nf = fam_mask.sum()
        if nf == 0:
            continue
        d1 = (fam_mask & is_noise_top1).sum()
        d2 = (fam_mask & is_noise_flip).sum()
        d3 = (fam_mask & is_noise_top2).sum()
        fam_table.append({
            'family': fam, 'n': int(nf),
            'D1_top1_n':  int(d1), 'D1_top1_pct':  d1 / nf * 100,
            'D2_flip_n':  int(d2), 'D2_flip_pct':  d2 / nf * 100,
            'D3_top2_n':  int(d3), 'D3_top2_pct':  d3 / nf * 100,
        })
        print(f"  {fam:<10}  n={nf:>6d}   D1={d1:>5d} ({d1/nf*100:5.2f}%)   "
              f"D2={d2:>5d} ({d2/nf*100:5.2f}%)   D3={d3:>5d} ({d3/nf*100:5.2f}%)")

    # ─── Per-generator ─────────────────────────────────────────
    print()
    print("=" * 70)
    print("  Per-GENERATOR breakdown")
    print("=" * 70)
    gen_table = []
    print(f"  {'Generator':<14}  {'n':>6}  {'D1 top1':>16}  {'D2 flip':>16}  {'D3 top2':>16}")
    print(f"  {'-'*72}")
    for gen, gen_df in df.groupby('generator'):
        ngen = len(gen_df)
        gen_mask = df['generator'] == gen
        d1 = (gen_mask & is_noise_top1).sum()
        d2 = (gen_mask & is_noise_flip).sum()
        d3 = (gen_mask & is_noise_top2).sum()
        gen_table.append({
            'generator': gen, 'n': int(ngen),
            'D1_top1_n':  int(d1), 'D1_top1_pct':  d1 / ngen * 100,
            'D2_flip_n':  int(d2), 'D2_flip_pct':  d2 / ngen * 100,
            'D3_top2_n':  int(d3), 'D3_top2_pct':  d3 / ngen * 100,
        })
        print(f"  {gen:<14}  {ngen:>6}  {d1:>5} ({d1/ngen*100:5.2f}%)  "
              f"{d2:>5} ({d2/ngen*100:5.2f}%)  {d3:>5} ({d3/ngen*100:5.2f}%)")

    # ─── Real vs Fake breakdown ─────────────────────────────
    print()
    print("=" * 70)
    print("  Real vs Fake breakdown (label)")
    print("=" * 70)
    for lab, lab_name in [(0, 'Real'), (1, 'Fake')]:
        m = df['label'] == lab
        n = m.sum()
        if n == 0:
            continue
        d1 = (m & is_noise_top1).sum()
        d2 = (m & is_noise_flip).sum()
        d3 = (m & is_noise_top2).sum()
        print(f"  label={lab} ({lab_name:<4}) n={n:>6d}   "
              f"D1={d1:>4d} ({d1/n*100:5.2f}%)   "
              f"D2={d2:>4d} ({d2/n*100:5.2f}%)   "
              f"D3={d3:>4d} ({d3/n*100:5.2f}%)")

    # ─── Save outputs ───────────────────────────────────────
    OUT.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        'n_total': int(n_total),
        'definitions': {
            'D1_top1': {
                'desc': 'Noise has top-1 ablation importance',
                'n': int(n1), 'pct': float(p1),
            },
            'D2_flip': {
                'desc': 'Ablating Noise alone crosses 0.5 boundary',
                'n': int(n2), 'pct': float(p2),
            },
            'D3_top2': {
                'desc': 'Noise in top-2 ablation importance',
                'n': int(n3), 'pct': float(p3),
            },
            'D1_AND_D2': {
                'desc': 'Noise top-1 AND ablation flips prediction',
                'n': int(n_intersect_12),
                'pct': float(n_intersect_12 / n_total * 100),
            },
            'D1_OR_D2': {
                'desc': 'Either condition',
                'n': int(n_union_12),
                'pct': float(n_union_12 / n_total * 100),
            },
        },
        'per_family': fam_table,
        'per_generator': gen_table,
    }
    with open(OUT, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\n[saved] {OUT}")

    # 範例：列出 Noise 真正關鍵的 5 張圖
    print()
    print("=" * 70)
    print("  Examples: 5 images where Noise is most decisive (D1 + highest abl_noise)")
    print("=" * 70)
    cand_df = df[is_noise_top1].copy()
    cand_df['noise_abl'] = cand_df['ablimp_noise']
    cand_df = cand_df.sort_values('noise_abl', ascending=False).head(5)
    for _, r in cand_df.iterrows():
        print(f"  base_prob={r['fake_prob_base']:.3f}  abl_noise={r['ablimp_noise']:.3f}  "
              f"family={r['family']:<10}  gen={r['generator']:<14}  path=...{r['path'][-50:]}")


if __name__ == '__main__':
    main()
