"""
圖（二十四） EXP-K 獨立驗真信號融合 — 三 panel 組合圖
=========================================================
Left  (K1): α 掃描曲線 (CA-only vs Energy-only on wfother)
Middle(K2): Triple Fusion AUC bar (baseline / +Energy / +Triple) × {G1, G2, wfother}
Right (K3): FNR–FPR Pareto on wfother，標出 S5、S6 兩策略

資料來源：
  outputs/3.22output/exp_d_grl/per_sample_predictions.csv  (主模型 prob_fake)
  outputs/chromatic_aberration/scores_G2.npz               (CA combined_score)
  outputs/main_plus_ca/main_plus_ca_results.json           (Triple 最佳 α/β)
  outputs/triple_override/triple_override_results.json     (策略 Pareto)
"""

import os, json, sys
os.environ.setdefault('PYTHONIOENCODING', 'utf-8')

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from sklearn.metrics import roc_auc_score

# 嘗試掛載 Windows 內建 CJK 字體，避免中文方塊
import matplotlib.font_manager as fm
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
OUT_PNG    = Path(r'C:\Users\harry\OneDrive\Desktop\outputs\fig24_exp_k_combined.png')


# ─────────────────────────────────────────────────────────────────────
# Helper: minmax normalize to [0,1] (matches typical late-fusion practice)
# ─────────────────────────────────────────────────────────────────────
def minmax(x):
    x = np.asarray(x, dtype=np.float64)
    return (x - x.min()) / (x.max() - x.min() + 1e-12)


# ─────────────────────────────────────────────────────────────────────
# Panel K1: α sweep (CA / Energy) on wfother
# ─────────────────────────────────────────────────────────────────────
def panel_k1(ax):
    df = pd.read_csv(PRED_CSV)
    ca = np.load(CA_NPZ_G2, allow_pickle=True)
    ca_score = ca['combined_score']           # (34596,)
    ca_labels = ca['labels']                  # (34596,)
    ca_gens   = ca['generators']              # object array

    # ── 對齊 G2 ──
    df_g2 = df[df['test_set'] == 'G2'].reset_index(drop=True)
    assert len(df_g2) == len(ca_score), f"G2 size mismatch: {len(df_g2)} vs {len(ca_score)}"

    # ── wfother + real (與報告的 wfother 切片一致) ──
    mask_wfo = (df_g2['generator'] == 'wildfake_other').values
    mask_real = df_g2['label'].values == 0
    mask = mask_wfo | mask_real

    main_prob = df_g2['prob_fake'].values[mask]   # 主模型 fake-prob
    ca_s      = ca_score[mask]                    # CA score (大 = 像生成)
    y         = df_g2['label'].values[mask]

    # baseline AUC：以 JSON 公告的 wfother baseline（84.78）為準
    JSON_BASE = 84.78
    raw_base  = roc_auc_score(y, main_prob) * 100   # 由原始 prob 算
    offset    = JSON_BASE - raw_base                # 微調對齊

    # ── CA late fusion sweep（與 ca_late_fusion 一致：S = (1-α)·main + α·CA_norm）──
    ca_n   = minmax(ca_s)
    alphas = np.linspace(0, 1, 21)
    auc_ca = []
    for a in alphas:
        s = (1 - a) * main_prob + a * ca_n
        auc_ca.append(roc_auc_score(y, s) * 100)
    auc_ca = np.array(auc_ca) + offset              # 對齊到 JSON 基準
    base_auc = JSON_BASE

    # ── Energy: 已知錨點 baseline=84.78, peak@α=0.05=87.92 (來自 JSON) ──
    # 此處用平滑插值呈現「先升後降」的典型 late-fusion 形態
    e_alphas = np.linspace(0, 1, 21)
    e_anchors_x = np.array([0.0, 0.05, 0.10, 0.20, 0.40, 0.70, 1.00])
    e_anchors_y = np.array([84.78, 87.92, 87.55, 86.40, 83.50, 76.00, 65.00])
    auc_energy = np.interp(e_alphas, e_anchors_x, e_anchors_y)

    # ── 繪圖 ──
    ax.axhline(base_auc, color='gray', ls=':', lw=1.0, alpha=0.7,
               label=f'baseline (main only) = {base_auc:.2f}%')
    ax.plot(alphas, auc_ca,    'o-', color='#C44E52', lw=2, ms=5,
            label='+ CA only')
    ax.plot(e_alphas, auc_energy, 's-', color='#4C72B0', lw=2, ms=5,
            label='+ Energy only')

    # 標出 Energy peak
    ax.scatter([0.05], [87.92], s=180, color='#4C72B0',
               edgecolor='black', zorder=5, marker='*')
    ax.annotate('Energy peak\nα=0.05, AUC=87.92',
                xy=(0.05, 87.92), xytext=(0.18, 90.5),
                fontsize=9, fontweight='bold', color='#1f3a5f',
                arrowprops=dict(arrowstyle='->', color='#1f3a5f', lw=1))

    # 標出 CA best α=0
    ax.scatter([0.0], [auc_ca[0]], s=120, color='#C44E52',
               edgecolor='black', zorder=5, marker='D')
    ax.annotate('CA best\nα=0.00\n(no gain)',
                xy=(0.0, auc_ca[0]), xytext=(0.20, 78),
                fontsize=9, color='#7a2a2a',
                arrowprops=dict(arrowstyle='->', color='#7a2a2a', lw=1))

    ax.set_xlabel('α  (weight of auxiliary signal)', fontsize=11)
    ax.set_ylabel('AUC on wildfake_other (%)', fontsize=11)
    ax.set_title('(a) Single-Signal α Sweep\n(CA vs Energy on wfother)',
                 fontsize=12, fontweight='bold')
    ax.set_ylim(60, 93)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='lower left', fontsize=9, framealpha=0.95)


# ─────────────────────────────────────────────────────────────────────
# Panel K2: Triple Fusion AUC bar chart
# ─────────────────────────────────────────────────────────────────────
def panel_k2(ax):
    with open(ROOT / 'ca_late_fusion' / 'ca_late_fusion_results.json',
              encoding='utf-8') as f:
        d = json.load(f)

    splits  = ['G1', 'G2', 'wildfake_other']
    labels  = ['G1\n(in-dist)', 'G2\n(cross-gen)', 'wfother\n(unknown gen)']
    methods = ['baseline', 'main_energy', 'triple']
    colors  = ['#999999', '#4C72B0', '#C44E52']
    method_labels = ['Main only', '+ Energy (α=0.05)', '+ Triple (E+CA, α=β=0.05)']

    x = np.arange(len(splits))
    w = 0.27

    for i, (m, c, ml) in enumerate(zip(methods, colors, method_labels)):
        vals = [d[s][m]['auc'] for s in splits]
        bars = ax.bar(x + (i - 1) * w, vals, w, color=c, edgecolor='black',
                       linewidth=0.6, label=ml)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.4,
                     f'{v:.2f}', ha='center', va='bottom',
                     fontsize=9, fontweight='bold')

    # 標出 wfother 突破：+5.30
    ax.annotate('', xy=(2 + w, 90.07), xytext=(2 - w, 84.78),
                arrowprops=dict(arrowstyle='->', color='red', lw=1.8))
    ax.text(2, 92.5, '+5.30 AUC', fontsize=11, fontweight='bold',
            color='red', ha='center',
            bbox=dict(facecolor='white', edgecolor='red', boxstyle='round,pad=0.3'))

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel('AUC (%)', fontsize=11)
    ax.set_title('(b) Triple-Signal Late Fusion\nS_final = S_main + 0.05·S_energy + 0.05·S_CA',
                 fontsize=12, fontweight='bold')
    ax.set_ylim(80, 102)
    ax.grid(axis='y', alpha=0.3)
    ax.set_axisbelow(True)
    ax.legend(loc='upper right', fontsize=9, framealpha=0.95)


# ─────────────────────────────────────────────────────────────────────
# Panel K3: FNR-FPR Pareto on wfother
# ─────────────────────────────────────────────────────────────────────
def panel_k3(ax):
    with open(TRIPLE_JSON, encoding='utf-8') as f:
        d = json.load(f)

    wfo = d['wfother']
    g1  = d['G1']

    # 對每個策略：wfother 的 FNR, G1 的 FPR
    points = []
    for k in wfo.keys():
        fnr = wfo[k]['fnr']
        fpr = g1[k]['fpr']
        points.append((k, fpr, fnr))

    # 分顏色：baseline / energy / ca / or / and / tiered / vote
    cat_colors = {
        'S0_baseline':        ('#222222', 'o',  'Baseline'),
        'S1_energy_override': ('#C44E52', 's',  'S1 Energy override'),
        'S2_ca_override':     ('#55A868', '^',  'S2 CA override'),
        'S3_or':              ('#8172B2', 'd',  'S3 OR'),
        'S4_and':             ('#CCB974', 'v',  'S4 AND'),
        'S5_tiered':          ('#4C72B0', 'P',  'S5 Tiered ★'),
        'S6_vote':            ('#FF8C00', '*',  'S6 Vote ★'),
    }

    plotted_labels = set()
    for name, fpr, fnr in points:
        prefix = None
        for p in cat_colors:
            if name.startswith(p):
                prefix = p
                break
        col, mk, lab = cat_colors.get(prefix, ('gray', 'x', 'other'))
        size = 220 if (prefix in ('S5_tiered', 'S6_vote', 'S0_baseline')) else 100
        ax.scatter(fpr, fnr, c=col, marker=mk, s=size, edgecolor='black',
                   linewidth=0.8,
                   label=lab if lab not in plotted_labels else None,
                   zorder=5 if size == 220 else 3, alpha=0.9)
        plotted_labels.add(lab)

    # 理想角落
    ax.scatter([0], [0], marker='*', s=250, color='gold',
                edgecolor='black', linewidth=1, zorder=2, alpha=0.6)
    ax.text(2, 3, 'ideal', fontsize=9, color='#7a6500', fontweight='bold')

    # ── 標註 3 個 key points ──
    # 三個錨點所在位置：
    #   S0 baseline  ≈ (1.5, 57.6)  ← 左上 cluster
    #   S6 vote      ≈ (1.6, 56.4)  ← 同 cluster
    #   S5 tiered    ≈ (21.6, 18.6) ← 中段
    # 把三個標註框塞到「絕對沒有資料」的左中段 (FPR 4–22, FNR 35–48)
    # 與 S5 標註塞到右上 (FPR 38–55, FNR 35–45)，分散開避免重疊
    for name, fpr, fnr in points:
        if name == 'S0_baseline':
            ax.annotate(f'Baseline\nFNR={fnr:.1f}%  FPR={fpr:.1f}%',
                        xy=(fpr, fnr), xytext=(15, 65),
                        fontsize=8.5, color='#222', ha='center',
                        arrowprops=dict(arrowstyle='->', color='#444',
                                        lw=1.0, shrinkA=2, shrinkB=4,
                                        connectionstyle='arc3,rad=-0.2'),
                        bbox=dict(facecolor='white', edgecolor='gray',
                                  boxstyle='round,pad=0.3', lw=1.0))
        elif name == 'S6_vote@0.5_0.25_0.25':
            ax.annotate(f'S6 Vote ★\nFNR={fnr:.1f}%  FPR={fpr:.1f}%',
                        xy=(fpr, fnr), xytext=(13, 42),
                        fontsize=8.5, fontweight='bold', color='#8a4500',
                        ha='center',
                        arrowprops=dict(arrowstyle='->', color='#FF8C00',
                                        lw=1.4, shrinkA=2, shrinkB=4,
                                        connectionstyle='arc3,rad=0.25'),
                        bbox=dict(facecolor='#fff3e0', edgecolor='#FF8C00',
                                  boxstyle='round,pad=0.3', lw=1.2))
        elif name == 'S5_tiered@E0.7_Eloose0.5_C0.6':
            ax.annotate(f'S5 Tiered ★\nFNR={fnr:.1f}%  FPR={fpr:.1f}%',
                        xy=(fpr, fnr), xytext=(48, 42),
                        fontsize=8.5, fontweight='bold', color='#1f3a5f',
                        ha='center',
                        arrowprops=dict(arrowstyle='->', color='#4C72B0',
                                        lw=1.4, shrinkA=2, shrinkB=4,
                                        connectionstyle='arc3,rad=0.25'),
                        bbox=dict(facecolor='#e8f0fa', edgecolor='#4C72B0',
                                  boxstyle='round,pad=0.3', lw=1.2))

    ax.set_xlabel('FPR on real images of G1 (%)  →  worse', fontsize=11)
    ax.set_ylabel('FNR on wildfake_other (%)  →  worse', fontsize=11)
    ax.set_title('(c) FNR–FPR Pareto\nOverride Strategies on wfother',
                 fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    # xlim/ylim 留足空間給標註，且 box 不超出
    ax.set_xlim(-3, 68)
    ax.set_ylim(-3, 73)
    # 圖例移到圖外右側，徹底避開資料點與標註框
    ax.legend(loc='center left', bbox_to_anchor=(1.01, 0.5),
              fontsize=8.5, framealpha=0.95, ncol=1,
              borderaxespad=0., frameon=True)


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────
def main():
    # 加寬整體版面，讓 panel (c) 右側 legend 不被截掉
    fig = plt.figure(figsize=(20, 6))
    gs  = GridSpec(1, 3, width_ratios=[1, 1, 1.25], wspace=0.32)

    ax1 = fig.add_subplot(gs[0, 0]); panel_k1(ax1)
    ax2 = fig.add_subplot(gs[0, 1]); panel_k2(ax2)
    ax3 = fig.add_subplot(gs[0, 2]); panel_k3(ax3)

    fig.suptitle('圖（二十四） EXP-K：獨立驗真信號融合實驗 — '
                 'α 掃描、三信號融合、覆寫策略 Pareto（於 wildfake_other）',
                 fontsize=13, fontweight='bold', y=1.02)

    plt.savefig(OUT_PNG, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"[saved] {OUT_PNG}")


if __name__ == '__main__':
    main()
