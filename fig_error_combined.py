"""
EXP-J 錯誤類型分析 — 合併圖（左 A 右 B）
==========================================
左 (A)：G2 Unseen 混淆矩陣
右 (B)：各未見生成器之 False Negative Rate
"""

import os, json
os.environ.setdefault('PYTHONIOENCODING', 'utf-8')

from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.colors import LinearSegmentedColormap

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

SRC = Path(r'C:\Users\harry\OneDrive\Desktop\outputs\3.22output\exp_e supplementary\step11_error_analysis.json')
OUT = Path(r'C:\Users\harry\OneDrive\Desktop\outputs\fig_error_combined_AB.png')


def main():
    with open(SRC, encoding='utf-8') as f:
        d = json.load(f)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5),
                              gridspec_kw={'width_ratios': [1, 1.2]})

    # ────────────────────────────────────────────────
    # (A) 混淆矩陣
    # ────────────────────────────────────────────────
    ax = axes[0]
    cm = np.array([[d['overall']['tn'], d['overall']['fp']],
                   [d['overall']['fn'], d['overall']['tp']]])
    blue = LinearSegmentedColormap.from_list('blue', ['#e8f0fa', '#1f3a5f'])
    ax.imshow(cm, cmap=blue, aspect='auto')

    max_v = cm.max()
    for i in range(2):
        for j in range(2):
            v = cm[i, j]
            color = 'white' if v > max_v * 0.5 else '#222'
            ax.text(j, i, f'{v:,}',
                    ha='center', va='center',
                    fontsize=20, fontweight='bold', color=color)

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['Pred Real', 'Pred Fake'], fontsize=12)
    ax.set_yticklabels(['True Real', 'True Fake'], fontsize=12)
    ax.set_title('(a) Confusion Matrix (G2 Unseen)',
                  fontsize=13, fontweight='bold', pad=12)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)

    # ────────────────────────────────────────────────
    # (B) FN Rate per generator
    # ────────────────────────────────────────────────
    ax = axes[1]
    per_gen = d['per_generator']
    order = ['dcgan_unseen', 'fursona_gan', 'waifu_gan',
             'wildfake_ddim', 'wildfake_other']
    gens = [g for g in order if g in per_gen]
    rates = [per_gen[g]['fn_rate'] for g in gens]
    colors = ['#a13030' if r > 30 else '#3a8c5a' for r in rates]

    y = np.arange(len(gens))
    bars = ax.barh(y, rates, color=colors, edgecolor='black',
                   linewidth=0.7, height=0.6)
    for b, r in zip(bars, rates):
        ax.text(b.get_width() + 1.0,
                b.get_y() + b.get_height() / 2,
                f'{r:.1f}%',
                va='center', fontsize=11, fontweight='bold', color='#222')

    ax.set_yticks(y)
    ax.set_yticklabels(gens, fontsize=11)
    ax.invert_yaxis()
    ax.set_xlim(0, 105)
    ax.set_xlabel('False Negative Rate (%)', fontsize=12)
    ax.set_title('(b) Per-Generator: How Often Fake Fools Model',
                  fontsize=13, fontweight='bold', pad=12)
    ax.grid(axis='x', alpha=0.3)
    ax.set_axisbelow(True)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)

    plt.tight_layout()
    plt.savefig(OUT, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f'[saved] {OUT}')


if __name__ == '__main__':
    main()
