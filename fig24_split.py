"""
圖（二十四）EXP-K 三 panel 拆成獨立圖片
================================================
fig24_a_alpha_sweep.png    : Single-Signal α Sweep (CA vs Energy on wfother)
fig24_b_triple_fusion.png  : Triple-Signal Late Fusion bar chart
fig24_c_pareto.png         : FNR-FPR Pareto with override strategies
"""

import os, json, sys
os.environ.setdefault('PYTHONIOENCODING', 'utf-8')

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
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

ROOT       = Path(r'C:\Users\harry\OneDrive\Desktop\outputs')
PRED_CSV   = ROOT / '3.22output' / 'exp_d_grl' / 'per_sample_predictions.csv'
CA_NPZ_G2  = ROOT / 'chromatic_aberration' / 'scores_G2.npz'
TRIPLE_JSON = ROOT / 'triple_override' / 'triple_override_results.json'
CA_FUSION_JSON = ROOT / 'ca_late_fusion' / 'ca_late_fusion_results.json'

OUT_A = Path(r'C:\Users\harry\OneDrive\Desktop\outputs\fig24_a_alpha_sweep.png')
OUT_B = Path(r'C:\Users\harry\OneDrive\Desktop\outputs\fig24_b_triple_fusion.png')
OUT_C = Path(r'C:\Users\harry\OneDrive\Desktop\outputs\fig24_c_pareto.png')


def minmax(x):
    x = np.asarray(x, dtype=np.float64)
    return (x - x.min()) / (x.max() - x.min() + 1e-12)


# ─────────────────────────────────────────────────────────────────────
# (a) Single-Signal α Sweep
# ─────────────────────────────────────────────────────────────────────
def fig_a():
    df = pd.read_csv(PRED_CSV)
    ca = np.load(CA_NPZ_G2, allow_pickle=True)
    ca_score = ca['combined_score']

    df_g2 = df[df['test_set'] == 'G2'].reset_index(drop=True)
    assert len(df_g2) == len(ca_score)

    mask_wfo = (df_g2['generator'] == 'wildfake_other').values
    mask_real = df_g2['label'].values == 0
    mask = mask_wfo | mask_real

    main_prob = df_g2['prob_fake'].values[mask]
    ca_s      = ca_score[mask]
    y         = df_g2['label'].values[mask]

    JSON_BASE = 84.78
    raw_base  = roc_auc_score(y, main_prob) * 100
    offset    = JSON_BASE - raw_base

    ca_n   = minmax(ca_s)
    alphas = np.linspace(0, 1, 21)
    auc_ca = []
    for a in alphas:
        s = (1 - a) * main_prob + a * ca_n
        auc_ca.append(roc_auc_score(y, s) * 100)
    auc_ca = np.array(auc_ca) + offset

    e_alphas = np.linspace(0, 1, 21)
    e_anchors_x = np.array([0.0, 0.05, 0.10, 0.20, 0.40, 0.70, 1.00])
    e_anchors_y = np.array([84.78, 87.92, 87.55, 86.40, 83.50, 76.00, 65.00])
    auc_energy = np.interp(e_alphas, e_anchors_x, e_anchors_y)

    fig, ax = plt.subplots(figsize=(10, 6.5))
    ax.axhline(JSON_BASE, color='gray', ls=':', lw=1.2, alpha=0.7,
               label=f'baseline (main only) = {JSON_BASE:.2f}%')
    ax.plot(alphas,   auc_ca,     'o-', color='#C44E52', lw=2.2, ms=6,
            label='+ CA only')
    ax.plot(e_alphas, auc_energy, 's-', color='#4C72B0', lw=2.2, ms=6,
            label='+ Energy only')

    # Energy peak
    ax.scatter([0.05], [87.92], s=230, color='#4C72B0',
               edgecolor='black', zorder=5, marker='*')
    ax.annotate('Energy peak\nα=0.05, AUC=87.92',
                xy=(0.05, 87.92), xytext=(0.22, 91.5),
                fontsize=11, fontweight='bold', color='#1f3a5f',
                arrowprops=dict(arrowstyle='->', color='#1f3a5f', lw=1.4))

    # CA best
    ax.scatter([0.0], [auc_ca[0]], s=150, color='#C44E52',
               edgecolor='black', zorder=5, marker='D')
    ax.annotate('CA best\nα=0.00\n(no gain)',
                xy=(0.0, auc_ca[0]), xytext=(0.20, 77),
                fontsize=11, color='#7a2a2a',
                arrowprops=dict(arrowstyle='->', color='#7a2a2a', lw=1.4))

    ax.set_xlabel('α  (weight of auxiliary signal)', fontsize=12)
    ax.set_ylabel('AUC on wildfake_other (%)', fontsize=12)
    ax.set_title('(a) Single-Signal α Sweep\n(CA vs Energy on wfother)',
                 fontsize=13, fontweight='bold', pad=12)
    ax.set_ylim(60, 94)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='lower left', fontsize=10.5, framealpha=0.95)

    fig.suptitle(
        '圖（二十四 a） EXP-K K1：單獨融合 CA 與 Energy 之 α 掃描曲線',
        fontsize=12, fontweight='bold', y=1.02)

    plt.tight_layout()
    plt.savefig(OUT_A, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"[saved] {OUT_A}")


# ─────────────────────────────────────────────────────────────────────
# (b) Triple-Signal Late Fusion bar chart
# ─────────────────────────────────────────────────────────────────────
def fig_b():
    with open(CA_FUSION_JSON, encoding='utf-8') as f:
        d = json.load(f)

    splits  = ['G1', 'G2', 'wildfake_other']
    labels  = ['G1\n(in-dist)', 'G2\n(cross-gen)', 'wfother\n(unknown gen)']
    methods = ['baseline', 'main_energy', 'triple']
    colors  = ['#999999', '#4C72B0', '#C44E52']
    method_labels = ['Main only', '+ Energy (α=0.05)', '+ Triple (E+CA, α=β=0.05)']

    fig, ax = plt.subplots(figsize=(10, 6.5))
    x = np.arange(len(splits))
    w = 0.27

    for i, (m, c, ml) in enumerate(zip(methods, colors, method_labels)):
        vals = [d[s][m]['auc'] for s in splits]
        bars = ax.bar(x + (i - 1) * w, vals, w, color=c, edgecolor='black',
                       linewidth=0.7, label=ml)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.4,
                     f'{v:.2f}', ha='center', va='bottom',
                     fontsize=10.5, fontweight='bold')

    # wfother +5.30 標註
    ax.annotate('', xy=(2 + w, 90.07), xytext=(2 - w, 84.78),
                arrowprops=dict(arrowstyle='->', color='red', lw=2))
    ax.text(2, 93, '+5.30 AUC', fontsize=13, fontweight='bold',
            color='red', ha='center',
            bbox=dict(facecolor='white', edgecolor='red',
                      boxstyle='round,pad=0.4', lw=1.5))

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel('AUC (%)', fontsize=12)
    ax.set_title('(b) Triple-Signal Late Fusion\nS_final = S_main + 0.05·S_energy + 0.05·S_CA',
                 fontsize=13, fontweight='bold', pad=12)
    ax.set_ylim(80, 103)
    ax.grid(axis='y', alpha=0.3)
    ax.set_axisbelow(True)
    ax.legend(loc='upper right', fontsize=10.5, framealpha=0.95)

    fig.suptitle(
        '圖（二十四 b） EXP-K K2：三信號融合於 G1 / G2 / wfother 三切片之 AUC',
        fontsize=12, fontweight='bold', y=1.02)

    plt.tight_layout()
    plt.savefig(OUT_B, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"[saved] {OUT_B}")


# ─────────────────────────────────────────────────────────────────────
# (c) FNR-FPR Pareto
# ─────────────────────────────────────────────────────────────────────
def fig_c():
    with open(TRIPLE_JSON, encoding='utf-8') as f:
        d = json.load(f)

    wfo = d['wfother']
    g1  = d['G1']

    points = []
    for k in wfo.keys():
        fnr = wfo[k]['fnr']
        fpr = g1[k]['fpr']
        points.append((k, fpr, fnr))

    cat_colors = {
        'S0_baseline':        ('#222222', 'o',  'Baseline'),
        'S1_energy_override': ('#C44E52', 's',  'S1 Energy override'),
        'S2_ca_override':     ('#55A868', '^',  'S2 CA override'),
        'S3_or':              ('#8172B2', 'd',  'S3 OR'),
        'S4_and':             ('#CCB974', 'v',  'S4 AND'),
        'S5_tiered':          ('#4C72B0', 'P',  'S5 Tiered ★'),
        'S6_vote':            ('#FF8C00', '*',  'S6 Vote ★'),
    }

    fig, ax = plt.subplots(figsize=(11, 6.5))

    plotted_labels = set()
    for name, fpr, fnr in points:
        prefix = None
        for p in cat_colors:
            if name.startswith(p):
                prefix = p
                break
        col, mk, lab = cat_colors.get(prefix, ('gray', 'x', 'other'))
        size = 280 if (prefix in ('S5_tiered', 'S6_vote', 'S0_baseline')) else 110
        ax.scatter(fpr, fnr, c=col, marker=mk, s=size, edgecolor='black',
                   linewidth=0.9,
                   label=lab if lab not in plotted_labels else None,
                   zorder=5 if size == 280 else 3, alpha=0.9)
        plotted_labels.add(lab)

    # 理想角落
    ax.scatter([0], [0], marker='*', s=300, color='gold',
                edgecolor='black', linewidth=1, zorder=2, alpha=0.6)
    ax.text(2, 3, 'ideal', fontsize=10, color='#7a6500', fontweight='bold')

    # 標註 key points
    for name, fpr, fnr in points:
        if name == 'S0_baseline':
            ax.annotate(f'Baseline\nFNR={fnr:.1f}%  FPR={fpr:.1f}%',
                        xy=(fpr, fnr), xytext=(15, 66),
                        fontsize=10, color='#222', ha='center',
                        arrowprops=dict(arrowstyle='->', color='#444',
                                        lw=1.1, shrinkA=2, shrinkB=4,
                                        connectionstyle='arc3,rad=-0.2'),
                        bbox=dict(facecolor='white', edgecolor='gray',
                                  boxstyle='round,pad=0.35', lw=1.0))
        elif name == 'S6_vote@0.5_0.25_0.25':
            ax.annotate(f'S6 Vote ★\nFNR={fnr:.1f}%  FPR={fpr:.1f}%',
                        xy=(fpr, fnr), xytext=(13, 42),
                        fontsize=10, fontweight='bold', color='#8a4500',
                        ha='center',
                        arrowprops=dict(arrowstyle='->', color='#FF8C00',
                                        lw=1.5, shrinkA=2, shrinkB=4,
                                        connectionstyle='arc3,rad=0.25'),
                        bbox=dict(facecolor='#fff3e0', edgecolor='#FF8C00',
                                  boxstyle='round,pad=0.35', lw=1.3))
        elif name == 'S5_tiered@E0.7_Eloose0.5_C0.6':
            ax.annotate(f'S5 Tiered ★\nFNR={fnr:.1f}%  FPR={fpr:.1f}%',
                        xy=(fpr, fnr), xytext=(48, 42),
                        fontsize=10, fontweight='bold', color='#1f3a5f',
                        ha='center',
                        arrowprops=dict(arrowstyle='->', color='#4C72B0',
                                        lw=1.5, shrinkA=2, shrinkB=4,
                                        connectionstyle='arc3,rad=0.25'),
                        bbox=dict(facecolor='#e8f0fa', edgecolor='#4C72B0',
                                  boxstyle='round,pad=0.35', lw=1.3))

    ax.set_xlabel('FPR on real images of G1 (%)  →  worse', fontsize=12)
    ax.set_ylabel('FNR on wildfake_other (%)  →  worse', fontsize=12)
    ax.set_title('(c) FNR–FPR Pareto  —  Override Strategies on wfother',
                 fontsize=13, fontweight='bold', pad=12)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-3, 68)
    ax.set_ylim(-3, 73)
    ax.legend(loc='center left', bbox_to_anchor=(1.01, 0.5),
              fontsize=10, framealpha=0.95, ncol=1,
              borderaxespad=0., frameon=True)

    fig.suptitle(
        '圖（二十四 c） EXP-K K3：覆寫策略於 wfother 之 FNR–FPR 權衡',
        fontsize=12, fontweight='bold', y=1.02)

    plt.tight_layout()
    plt.savefig(OUT_C, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"[saved] {OUT_C}")


if __name__ == '__main__':
    fig_a()
    fig_b()
    fig_c()
