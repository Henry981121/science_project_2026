"""
驗證方法一：DAUC / IAUC Grad-CAM 忠實度測試 — 結果圖
======================================================
左 panel (a): DAUC vs IAUC 群組長條 + 平均 Gap 標註，4 流依 Gap 排序
右 panel (b): 逐張圖之 Faithfulness Gap (IAUC - DAUC) 箱線分布

資料來源:
  outputs/dauc_iauc.json (彙總 mean/std)
  outputs/dauc_iauc.csv  (23 張逐張數據)
"""

import os, json
os.environ.setdefault('PYTHONIOENCODING', 'utf-8')

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# CJK 字體
for fpath in (r'C:\Windows\Fonts\msjh.ttc',
              r'C:\Windows\Fonts\msyh.ttc',
              r'C:\Windows\Fonts\mingliu.ttc'):
    if os.path.exists(fpath):
        fm.fontManager.addfont(fpath)
        plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'Microsoft YaHei',
                                            'PMingLiU', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        break

ROOT = Path(r'C:\Users\harry\OneDrive\Desktop\science_project_2026_v2')
JSON = ROOT / 'outputs' / 'dauc_iauc.json'
CSV  = ROOT / 'outputs' / 'dauc_iauc.csv'
OUT  = Path(r'C:\Users\harry\OneDrive\Desktop\outputs\fig_method1_dauc_iauc.png')


def main():
    with open(JSON, encoding='utf-8') as f:
        d = json.load(f)
    df = pd.read_csv(CSV)

    # ── 依 Gap 排序（高至低 = 越忠實在左）──
    streams_raw = list(d['per_stream'].keys())
    streams = sorted(streams_raw,
                     key=lambda s: -d['per_stream'][s]['gap_mean'])
    n_img   = d['n_images_evaluated']

    dauc_m  = [d['per_stream'][s]['dauc_mean'] for s in streams]
    dauc_sd = [d['per_stream'][s]['dauc_std']  for s in streams]
    iauc_m  = [d['per_stream'][s]['iauc_mean'] for s in streams]
    iauc_sd = [d['per_stream'][s]['iauc_std']  for s in streams]
    gap_m   = [d['per_stream'][s]['gap_mean']  for s in streams]
    gap_sd  = [d['per_stream'][s]['gap_std']   for s in streams]

    # ── Figure ──
    fig, axes = plt.subplots(1, 2, figsize=(15, 6),
                              gridspec_kw={'width_ratios': [1.1, 1]})

    # ────────────────────────────────────────────
    # Panel (a): DAUC vs IAUC 群組長條
    # ────────────────────────────────────────────
    ax = axes[0]
    x = np.arange(len(streams))
    w = 0.36

    bars_d = ax.bar(x - w / 2, dauc_m, w, yerr=dauc_sd, capsize=4,
                    color='#C44E52', edgecolor='black', linewidth=0.6,
                    label='DAUC  (lower = better, 熱圖忠實)',
                    error_kw=dict(lw=1, ecolor='#5a1c1c'))
    bars_i = ax.bar(x + w / 2, iauc_m, w, yerr=iauc_sd, capsize=4,
                    color='#4C72B0', edgecolor='black', linewidth=0.6,
                    label='IAUC  (higher = better, 熱圖忠實)',
                    error_kw=dict(lw=1, ecolor='#1f3a5f'))

    # 標 DAUC / IAUC 數值
    for b, v in zip(bars_d, dauc_m):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.012,
                f'{v:.3f}', ha='center', va='bottom',
                fontsize=8.5, color='#5a1c1c', fontweight='bold')
    for b, v in zip(bars_i, iauc_m):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.012,
                f'{v:.3f}', ha='center', va='bottom',
                fontsize=8.5, color='#1f3a5f', fontweight='bold')

    # Gap 標註於頂部
    for i, (gm, gs) in enumerate(zip(gap_m, gap_sd)):
        col = '#1f7a1f' if gm > 0 else '#a13030'
        bg  = '#e8f8e8' if gm > 0 else '#fde8e8'
        ax.text(i, 1.18,
                f'Gap = {gm:+.3f}\n(IAUC - DAUC)',
                ha='center', va='top', fontsize=9, fontweight='bold',
                color=col,
                bbox=dict(facecolor=bg, edgecolor=col,
                          boxstyle='round,pad=0.3', lw=1))

    ax.set_xticks(x)
    ax.set_xticklabels([s.upper() for s in streams], fontsize=11, fontweight='bold')
    ax.set_ylabel('AUC value', fontsize=11)
    ax.set_ylim(0, 1.25)
    ax.set_title(f'(a) DAUC vs IAUC per Stream  (N = {n_img} G2 images, sorted by Gap)',
                 fontsize=12, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    ax.set_axisbelow(True)
    ax.legend(loc='lower right', fontsize=9, framealpha=0.95)
    ax.axhline(1.0, color='gray', ls=':', lw=0.8, alpha=0.5)

    # 「越忠實 →」箭頭註解
    ax.annotate('', xy=(-0.45, 1.23), xytext=(len(streams) - 0.55, 1.23),
                 arrowprops=dict(arrowstyle='<-', color='#444', lw=1.5))
    ax.text(-0.45, 1.245, '更忠實', fontsize=9, color='#444',
            ha='left', fontweight='bold')
    ax.text(len(streams) - 0.55, 1.245, '較不忠實', fontsize=9,
            color='#444', ha='right', fontweight='bold')

    # ────────────────────────────────────────────
    # Panel (b): 逐張 Gap 分布 (box + scatter)
    # ────────────────────────────────────────────
    ax = axes[1]
    gap_cols = {s: f'{s}_gap' for s in streams}
    gap_data = [df[gap_cols[s]].dropna().values for s in streams]

    colors = ['#4C72B0', '#55A868', '#CCB974', '#C44E52']  # CLIP→DIRE
    box = ax.boxplot(gap_data, positions=np.arange(len(streams)),
                      widths=0.55, patch_artist=True,
                      medianprops=dict(color='black', lw=2),
                      boxprops=dict(lw=1),
                      whiskerprops=dict(lw=1),
                      capprops=dict(lw=1),
                      flierprops=dict(marker='o', markersize=4,
                                       markerfacecolor='gray', alpha=0.5))
    for patch, c in zip(box['boxes'], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.45)

    # 疊上散點 (jitter)
    rng = np.random.default_rng(0)
    for i, (s, c) in enumerate(zip(streams, colors)):
        y = df[gap_cols[s]].dropna().values
        x_j = rng.normal(i, 0.05, size=len(y))
        ax.scatter(x_j, y, s=24, color=c, edgecolor='black',
                   linewidth=0.4, alpha=0.75, zorder=3)

        # mean marker (diamond)
        ax.scatter([i], [np.mean(y)], marker='D', s=70,
                   color='white', edgecolor=c, linewidth=2, zorder=5)

    ax.axhline(0, color='black', ls='--', lw=1, alpha=0.7,
               label='Gap = 0 (perfectly faithful)')
    ax.set_xticks(np.arange(len(streams)))
    ax.set_xticklabels([s.upper() for s in streams], fontsize=11, fontweight='bold')
    ax.set_ylabel('Faithfulness Gap = IAUC - DAUC', fontsize=11)
    ax.set_title('(b) Per-image Gap Distribution\n(box + jittered points; ◇ = mean)',
                 fontsize=12, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    ax.set_axisbelow(True)
    ax.legend(loc='lower left', fontsize=9, framealpha=0.95)

    # 整體標題
    fig.suptitle(
        '驗證方法一：DAUC / IAUC Grad-CAM 忠實度測試  '
        '(Petsiuk et al. 2018; CLIP via Chefer relevance, others via Grad-CAM; '
        'Noise excluded as SRM residual)',
        fontsize=12, fontweight='bold', y=1.02)

    plt.tight_layout()
    plt.savefig(OUT, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"[saved] {OUT}")


if __name__ == '__main__':
    main()
