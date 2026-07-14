"""
驗證方法一：DAUC / IAUC — 拆成兩張獨立圖片
============================================
fig_method1_a_bars.png  : DAUC vs IAUC per Stream（含 Gap 標註）
fig_method1_b_box.png   : 逐張 Gap 分布箱線圖
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
OUT_A = Path(r'C:\Users\harry\OneDrive\Desktop\outputs\fig_method1_a_bars.png')
OUT_B = Path(r'C:\Users\harry\OneDrive\Desktop\outputs\fig_method1_b_box.png')


def load_data():
    with open(JSON, encoding='utf-8') as f:
        d = json.load(f)
    df = pd.read_csv(CSV)
    streams_raw = list(d['per_stream'].keys())
    streams = sorted(streams_raw, key=lambda s: -d['per_stream'][s]['gap_mean'])
    return d, df, streams


# ────────────────────────────────────────────
# 圖 A：DAUC vs IAUC 群組長條
# ────────────────────────────────────────────
def fig_a():
    d, _, streams = load_data()
    n_img = d['n_images_evaluated']

    dauc_m  = [d['per_stream'][s]['dauc_mean'] for s in streams]
    dauc_sd = [d['per_stream'][s]['dauc_std']  for s in streams]
    iauc_m  = [d['per_stream'][s]['iauc_mean'] for s in streams]
    iauc_sd = [d['per_stream'][s]['iauc_std']  for s in streams]
    gap_m   = [d['per_stream'][s]['gap_mean']  for s in streams]

    fig, ax = plt.subplots(figsize=(10, 6.5))
    x = np.arange(len(streams))
    w = 0.36

    bars_d = ax.bar(x - w / 2, dauc_m, w, yerr=dauc_sd, capsize=5,
                    color='#C44E52', edgecolor='black', linewidth=0.7,
                    label='DAUC  (lower = better, 熱圖忠實)',
                    error_kw=dict(lw=1.2, ecolor='#5a1c1c'))
    bars_i = ax.bar(x + w / 2, iauc_m, w, yerr=iauc_sd, capsize=5,
                    color='#4C72B0', edgecolor='black', linewidth=0.7,
                    label='IAUC  (higher = better, 熱圖忠實)',
                    error_kw=dict(lw=1.2, ecolor='#1f3a5f'))

    for b, v in zip(bars_d, dauc_m):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.015,
                f'{v:.3f}', ha='center', va='bottom',
                fontsize=10, color='#5a1c1c', fontweight='bold')
    for b, v in zip(bars_i, iauc_m):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.015,
                f'{v:.3f}', ha='center', va='bottom',
                fontsize=10, color='#1f3a5f', fontweight='bold')

    # Gap 標籤
    for i, gm in enumerate(gap_m):
        col = '#1f7a1f' if gm > 0 else '#a13030'
        bg  = '#e8f8e8' if gm > 0 else '#fde8e8'
        ax.text(i, 1.18,
                f'Gap = {gm:+.3f}\n(IAUC - DAUC)',
                ha='center', va='top', fontsize=10, fontweight='bold',
                color=col,
                bbox=dict(facecolor=bg, edgecolor=col,
                          boxstyle='round,pad=0.35', lw=1.2))

    ax.set_xticks(x)
    ax.set_xticklabels([s.upper() for s in streams], fontsize=12, fontweight='bold')
    ax.set_ylabel('AUC value', fontsize=12)
    ax.set_ylim(0, 1.28)
    ax.set_title(f'(a) DAUC vs IAUC per Stream  (N = {n_img} G2 images, sorted by Gap)',
                 fontsize=13, fontweight='bold', pad=12)
    ax.grid(axis='y', alpha=0.3)
    ax.set_axisbelow(True)
    ax.legend(loc='lower right', fontsize=10, framealpha=0.95)
    ax.axhline(1.0, color='gray', ls=':', lw=0.8, alpha=0.5)

    # 「越忠實 →」箭頭
    ax.annotate('', xy=(-0.45, 1.25), xytext=(len(streams) - 0.55, 1.25),
                 arrowprops=dict(arrowstyle='<-', color='#444', lw=1.5))
    ax.text(-0.45, 1.265, '更忠實', fontsize=10, color='#444',
            ha='left', fontweight='bold')
    ax.text(len(streams) - 0.55, 1.265, '較不忠實', fontsize=10,
            color='#444', ha='right', fontweight='bold')

    fig.suptitle(
        '驗證方法一：DAUC / IAUC Grad-CAM 忠實度測試 — 平均值對比\n'
        '(Petsiuk et al. 2018; CLIP via Chefer relevance, others via Grad-CAM; '
        'Noise excluded as SRM residual)',
        fontsize=11, fontweight='bold', y=1.02)

    plt.tight_layout()
    plt.savefig(OUT_A, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"[saved] {OUT_A}")


# ────────────────────────────────────────────
# 圖 B：逐張 Gap 分布箱線
# ────────────────────────────────────────────
def fig_b():
    d, df, streams = load_data()
    n_img = d['n_images_evaluated']

    fig, ax = plt.subplots(figsize=(10, 6.5))

    gap_cols = {s: f'{s}_gap' for s in streams}
    gap_data = [df[gap_cols[s]].dropna().values for s in streams]

    colors = ['#4C72B0', '#55A868', '#CCB974', '#C44E52']  # CLIP→DIRE
    box = ax.boxplot(gap_data, positions=np.arange(len(streams)),
                      widths=0.55, patch_artist=True,
                      medianprops=dict(color='black', lw=2),
                      boxprops=dict(lw=1.2),
                      whiskerprops=dict(lw=1.2),
                      capprops=dict(lw=1.2),
                      flierprops=dict(marker='o', markersize=5,
                                       markerfacecolor='gray', alpha=0.5))
    for patch, c in zip(box['boxes'], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.45)

    rng = np.random.default_rng(0)
    for i, (s, c) in enumerate(zip(streams, colors)):
        y = df[gap_cols[s]].dropna().values
        x_j = rng.normal(i, 0.06, size=len(y))
        ax.scatter(x_j, y, s=36, color=c, edgecolor='black',
                   linewidth=0.5, alpha=0.8, zorder=3)

        # mean marker
        m = np.mean(y)
        ax.scatter([i], [m], marker='D', s=110,
                   color='white', edgecolor=c, linewidth=2.2, zorder=5)
        ax.text(i + 0.32, m, f'mean={m:+.3f}',
                fontsize=9, color=c, fontweight='bold', va='center')

    ax.axhline(0, color='black', ls='--', lw=1.2, alpha=0.7,
               label='Gap = 0 (perfectly faithful)')
    ax.set_xticks(np.arange(len(streams)))
    ax.set_xticklabels([s.upper() for s in streams], fontsize=12, fontweight='bold')
    ax.set_ylabel('Faithfulness Gap = IAUC - DAUC', fontsize=12)
    ax.set_title(f'(b) Per-image Gap Distribution  (N = {n_img}; box + jittered points; ◇ = mean)',
                 fontsize=13, fontweight='bold', pad=12)
    ax.grid(axis='y', alpha=0.3)
    ax.set_axisbelow(True)
    ax.legend(loc='lower left', fontsize=10, framealpha=0.95)

    # 註解：右側標「更忠實」「較不忠實」
    ymin, ymax = ax.get_ylim()
    ax.annotate('', xy=(len(streams) - 0.3, ymax * 0.85),
                 xytext=(len(streams) - 0.3, ymin * 0.85),
                 arrowprops=dict(arrowstyle='<-', color='#444', lw=1.5))
    ax.text(len(streams) - 0.22, ymax * 0.85, '更忠實',
            fontsize=10, color='#444', fontweight='bold', va='center')
    ax.text(len(streams) - 0.22, ymin * 0.85, '較不忠實',
            fontsize=10, color='#444', fontweight='bold', va='center')

    fig.suptitle(
        '驗證方法一：DAUC / IAUC Grad-CAM 忠實度測試 — 逐張分布\n'
        '(每張圖一個資料點；箱線顯示穩定性)',
        fontsize=11, fontweight='bold', y=1.02)

    plt.tight_layout()
    plt.savefig(OUT_B, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"[saved] {OUT_B}")


if __name__ == '__main__':
    fig_a()
    fig_b()
