"""
ECE 信心校準分析：合併圖（左 a 右 b）
======================================
(a) 各資料集 ECE 與錯誤預測平均信心對照
(b) wildfake_other_ONLY 校準曲線（ECE = 0.5611）
"""

import os, json
os.environ.setdefault('PYTHONIOENCODING', 'utf-8')

from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

for fpath in (r'C:\Windows\Fonts\msjh.ttc', r'C:\Windows\Fonts\msyh.ttc'):
    if os.path.exists(fpath):
        fm.fontManager.addfont(fpath)
        plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        break

SRC = Path(r'C:\Users\harry\OneDrive\Desktop\outputs\validation_methods\ece\ece_results.json')
OUT = Path(r'C:\Users\harry\OneDrive\Desktop\outputs\fig_ece_combined_ab.png')


def main():
    d = json.load(open(SRC, encoding='utf-8'))

    fig, axes = plt.subplots(1, 2, figsize=(15, 6),
                              gridspec_kw={'width_ratios': [1.2, 1]})

    # ─────────────────────────────────────────────────
    # (a) ECE + Avg conf on wrong by dataset
    # ─────────────────────────────────────────────────
    ax = axes[0]
    datasets = ['G1', 'G2', 'wildfake_other_subset', 'wildfake_other_ONLY']
    labels   = ['G1', 'G2', 'wfo_subset', 'wfo_ONLY']
    ece_vals  = [d[k]['ece']                   for k in datasets]
    conf_vals = [d[k]['avg_conf_on_wrong']     for k in datasets]

    x = np.arange(len(datasets))
    w = 0.38

    bars_e = ax.bar(x - w/2, ece_vals, w,
                    color='#4C72B0', edgecolor='black', linewidth=0.7,
                    label='ECE')
    bars_c = ax.bar(x + w/2, conf_vals, w,
                    color='#C44E52', edgecolor='black', linewidth=0.7,
                    label='Avg confidence on WRONG predictions')

    for b, v in zip(bars_e, ece_vals):
        ax.text(b.get_x() + b.get_width()/2, v + 0.015,
                f'{v:.4f}', ha='center', va='bottom',
                fontsize=10, fontweight='bold', color='#1f3a5f')
    for b, v in zip(bars_c, conf_vals):
        ax.text(b.get_x() + b.get_width()/2, v + 0.015,
                f'{v:.3f}', ha='center', va='bottom',
                fontsize=10, fontweight='bold', color='#7a2222')

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_ylim(0, 1.15)
    ax.set_title('(a) ECE and Overconfidence (avg conf on wrong) by Dataset',
                  fontsize=12, fontweight='bold', pad=10)
    ax.grid(axis='y', alpha=0.3)
    ax.set_axisbelow(True)
    ax.legend(loc='upper left', fontsize=10, framealpha=0.95)

    # ─────────────────────────────────────────────────
    # (b) Reliability diagram for wildfake_other_ONLY
    # ─────────────────────────────────────────────────
    ax = axes[1]
    bins = d['wildfake_other_ONLY']['per_bin']
    centers = [(b['bin_lo'] + b['bin_hi']) / 2 for b in bins]
    accs    = [b['avg_acc']  for b in bins]
    confs   = [b['avg_conf'] for b in bins]
    widths  = [b['bin_hi'] - b['bin_lo'] for b in bins]

    # 信心區間長條：actual accuracy
    ax.bar(centers, accs, width=widths,
            color='#4C72B0', edgecolor='black', linewidth=0.7, alpha=0.85,
            label='actual accuracy')
    # gap 區（信心 vs 準確度差距）紅色
    for c, a, conf, w_ in zip(centers, accs, confs, widths):
        if conf > a:
            ax.bar(c, conf - a, width=w_, bottom=a,
                    color='#C44E52', alpha=0.55, edgecolor='black', linewidth=0.5)

    # 完美校準對角線
    ax.plot([0.5, 1.0], [0.5, 1.0], '--', color='black', lw=1.4,
             label='perfect calibration')

    ax.set_xlim(0.5, 1.0)
    ax.set_ylim(0.0, 1.05)
    ax.set_xlabel('Confidence', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title(f'(b) wildfake_other_ONLY  Reliability Diagram\n'
                 f'ECE = {d["wildfake_other_ONLY"]["ece"]:.4f}',
                  fontsize=12, fontweight='bold', pad=10)
    ax.grid(True, alpha=0.3)
    ax.set_axisbelow(True)
    ax.legend(loc='upper left', fontsize=10, framealpha=0.95)

    fig.suptitle(
        '圖（十五）模型信心校準分析（ECE）：各資料集對照與 wildfake_other_ONLY 校準曲線',
        fontsize=14, fontweight='bold', y=1.01)

    plt.tight_layout()
    plt.savefig(OUT, dpi=180, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f'[saved] {OUT}')


if __name__ == '__main__':
    main()
