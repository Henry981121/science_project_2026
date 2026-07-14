"""
EXP-J 錯誤類型分析 — 拆成兩張獨立圖檔
========================================
圖片A：G2 Unseen 混淆矩陣
圖片B：各未知生成器之 False Negative Rate
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
OUT_A = Path(r'C:\Users\harry\OneDrive\Desktop\outputs\figA_confusion_matrix.png')
OUT_B = Path(r'C:\Users\harry\OneDrive\Desktop\outputs\figB_fn_rate_per_generator.png')


def load_data():
    with open(SRC, encoding='utf-8') as f:
        return json.load(f)


# ────────────────────────────────────────────────
# 圖片A：混淆矩陣
# ────────────────────────────────────────────────
def fig_a(d):
    cm = [[d['overall']['tn'], d['overall']['fp']],
          [d['overall']['fn'], d['overall']['tp']]]
    cm = np.array(cm)

    fig, ax = plt.subplots(figsize=(7, 5.5))
    blue = LinearSegmentedColormap.from_list('blue', ['#e8f0fa', '#1f3a5f'])
    im = ax.imshow(cm, cmap=blue, aspect='auto')

    # 填字
    max_v = cm.max()
    for i in range(2):
        for j in range(2):
            v = cm[i, j]
            color = 'white' if v > max_v * 0.5 else '#222'
            ax.text(j, i, f'{v:,}',
                    ha='center', va='center',
                    fontsize=22, fontweight='bold', color=color)

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['Pred Real', 'Pred Fake'], fontsize=13)
    ax.set_yticklabels(['True Real', 'True Fake'], fontsize=13)
    ax.set_title('圖片A：混淆矩陣（G2 未見生成器測試集）',
                  fontsize=14, fontweight='bold', pad=15)

    # 移除上右框
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)

    plt.tight_layout()
    plt.savefig(OUT_A, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f'[saved] {OUT_A}')


# ────────────────────────────────────────────────
# 圖片B：各生成器 FN 率長條圖
# ────────────────────────────────────────────────
def fig_b(d):
    per_gen = d['per_generator']
    # 保留報告書原圖順序：dcgan_unseen, fursona_gan, waifu_gan, wildfake_ddim, wildfake_other
    order = ['dcgan_unseen', 'fursona_gan', 'waifu_gan',
             'wildfake_ddim', 'wildfake_other']
    gens = [g for g in order if g in per_gen]
    rates = [per_gen[g]['fn_rate'] for g in gens]

    # 顏色：> 30% 用紅，否則用綠
    colors = ['#a13030' if r > 30 else '#3a8c5a' for r in rates]

    fig, ax = plt.subplots(figsize=(8, 5.5))
    y = np.arange(len(gens))
    bars = ax.barh(y, rates, color=colors, edgecolor='black',
                   linewidth=0.7, height=0.6)

    # 標數值
    for b, r in zip(bars, rates):
        ax.text(b.get_width() + 1.0,
                b.get_y() + b.get_height() / 2,
                f'{r:.1f}%',
                va='center', fontsize=12, fontweight='bold', color='#222')

    ax.set_yticks(y)
    ax.set_yticklabels(gens, fontsize=12)
    ax.invert_yaxis()
    ax.set_xlim(0, 105)
    ax.set_xlabel('False Negative Rate (%)', fontsize=12)
    ax.set_title('圖片B：各未見生成器之假陰性率（FN Rate）',
                  fontsize=14, fontweight='bold', pad=15)
    ax.grid(axis='x', alpha=0.3)
    ax.set_axisbelow(True)

    # 移除上右框
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)

    plt.tight_layout()
    plt.savefig(OUT_B, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f'[saved] {OUT_B}')


if __name__ == '__main__':
    d = load_data()
    fig_a(d)
    fig_b(d)
