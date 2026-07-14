"""
圖（十三）Feature Space Diagnostics — 加上 (A) (B) (C) (D) 標號
================================================================
(A) Silhouette Score Bar
(B) MMD RBF 長條圖
(C) MMD 線性長條圖
(D) 領域偏移與準確率散佈圖
"""

import os, json
os.environ.setdefault('PYTHONIOENCODING', 'utf-8')

from pathlib import Path
import numpy as np
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

SIL_JSON = Path(r'C:\Users\harry\OneDrive\Desktop\outputs\validation_methods\silhouette_mmd\silhouette_mmd_results.json')
ERR_JSON = Path(r'C:\Users\harry\OneDrive\Desktop\outputs\3.22output\exp_e supplementary\step11_error_analysis.json')
OUT      = Path(r'C:\Users\harry\OneDrive\Desktop\outputs\fig13_silhouette_mmd_ABCD.png')


def main():
    sil = json.load(open(SIL_JSON, encoding='utf-8'))
    err = json.load(open(ERR_JSON, encoding='utf-8'))

    fig, axes = plt.subplots(2, 2, figsize=(15, 11))
    fig.suptitle('Feature Space Diagnostics: Silhouette + MMD',
                  fontsize=16, fontweight='bold', y=0.995)

    # ─────────────────────────────────────────────────────────
    # (A) Silhouette Score
    # ─────────────────────────────────────────────────────────
    ax = axes[0, 0]
    s = sil['silhouette']
    labels = ['G1 fake\nby generator', 'G2 fake\nby generator',
              'G1\nreal vs fake', 'G2\nreal vs fake']
    keys = ['G1_fake_by_generator', 'G2_fake_by_generator',
            'G1_by_real_fake', 'G2_by_real_fake']
    vals = [s[k]['score'] for k in keys]
    colors = ['#c87a7a', '#c87a7a', '#5fa56b', '#5fa56b']

    bars = ax.bar(labels, vals, color=colors, edgecolor='black', linewidth=0.7)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.012,
                f'+{v:.4f}', ha='center', va='bottom',
                fontsize=11, fontweight='bold')

    ax.set_ylabel('Silhouette score', fontsize=11)
    ax.set_title('(a)  Silhouette Score by Group\n'
                 '(red = generator clustering — want LOW; '
                 'green = real/fake — want HIGH)',
                 fontsize=12, fontweight='bold')
    ax.set_ylim(0, max(vals) * 1.18)
    ax.grid(axis='y', alpha=0.3)
    ax.set_axisbelow(True)

    # ─────────────────────────────────────────────────────────
    # (B) MMD RBF
    # ─────────────────────────────────────────────────────────
    ax = axes[0, 1]
    mmd = sil['mmd']
    gens = ['dcgan_unseen', 'wildfake_ddim', 'wildfake_other',
            'fursona_gan', 'waifu_gan']
    rbf_vals = [mmd[g]['mmd_rbf'] for g in gens]
    ctrl_rbf = mmd['_control_real_vs_real']['mmd_rbf']

    cmap = plt.colormaps['Reds_r']
    colors_b = [cmap(0.2 + i * 0.13) for i in range(len(gens))]

    bars = ax.bar(gens, rbf_vals, color=colors_b,
                  edgecolor='black', linewidth=0.7)
    for b, v in zip(bars, rbf_vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.012,
                f'{v:.3f}', ha='center', va='bottom',
                fontsize=11, fontweight='bold')

    ax.axhline(ctrl_rbf, color='#2a8c3a', ls='--', lw=1.2,
               label=f'real vs real control ({ctrl_rbf:.3f})')
    ax.set_ylabel('MMD (RBF kernel)', fontsize=11)
    ax.set_title('(b)  MMD: G1 fake → each G2 unseen generator\n'
                 '(higher = more out-of-distribution)',
                 fontsize=12, fontweight='bold')
    ax.tick_params(axis='x', rotation=20)
    ax.legend(fontsize=9, loc='upper right')
    ax.grid(axis='y', alpha=0.3)
    ax.set_axisbelow(True)

    # ─────────────────────────────────────────────────────────
    # (C) MMD Linear
    # ─────────────────────────────────────────────────────────
    ax = axes[1, 0]
    lin_vals = [mmd[g]['mmd_linear'] for g in gens]
    ctrl_lin = mmd['_control_real_vs_real']['mmd_linear']

    bars = ax.bar(gens, lin_vals, color=colors_b,
                  edgecolor='black', linewidth=0.7)
    for b, v in zip(bars, lin_vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.2,
                f'{v:.3f}', ha='center', va='bottom',
                fontsize=11, fontweight='bold')

    ax.axhline(ctrl_lin, color='#2a8c3a', ls='--', lw=1.2,
               label=f'real vs real control ({ctrl_lin:.3f})')
    ax.set_ylabel('MMD (linear kernel)', fontsize=11)
    ax.set_title('(c)  MMD (linear): G1 fake → G2 unseen',
                 fontsize=12, fontweight='bold')
    ax.tick_params(axis='x', rotation=20)
    ax.legend(fontsize=9, loc='upper right')
    ax.grid(axis='y', alpha=0.3)
    ax.set_axisbelow(True)

    # ─────────────────────────────────────────────────────────
    # (D) Domain shift vs Accuracy
    # ─────────────────────────────────────────────────────────
    ax = axes[1, 1]
    # 從 error_analysis.json 取 FN rate 並換算 ACC
    pg = err['per_generator']
    acc = []
    for g in gens:
        if g in pg:
            fnr = pg[g]['fn_rate']
            acc.append(100 - fnr)
        else:
            acc.append(np.nan)

    cmap_d = plt.colormaps['viridis']
    pt_colors = [cmap_d(i / (len(gens) - 1)) for i in range(len(gens))]

    for i, (g, x, y) in enumerate(zip(gens, rbf_vals, acc)):
        if np.isnan(y):
            continue
        ax.scatter(x, y, s=200, color=pt_colors[i],
                   edgecolor='black', linewidth=0.9, zorder=3)
        ax.annotate(g, (x, y),
                    xytext=(8, 6), textcoords='offset points',
                    fontsize=10)

    # Pearson r
    valid = [(x, y) for x, y in zip(rbf_vals, acc) if not np.isnan(y)]
    if len(valid) >= 2:
        xs, ys = zip(*valid)
        r = np.corrcoef(xs, ys)[0, 1]
        ax.text(0.04, 0.05, f'Pearson r = {r:.3f}',
                transform=ax.transAxes, fontsize=11,
                bbox=dict(facecolor='white', edgecolor='#444',
                          boxstyle='round,pad=0.4'))

    ax.set_xlabel('MMD to G1 fake (RBF)', fontsize=11)
    ax.set_ylabel('Main model ACC on this generator (%)', fontsize=11)
    ax.set_title('(d)  Domain shift (MMD) vs Model accuracy\n'
                 '(higher MMD → lower accuracy expected)',
                 fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_axisbelow(True)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(OUT, dpi=180, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f'[saved] {OUT}')


if __name__ == '__main__':
    main()
