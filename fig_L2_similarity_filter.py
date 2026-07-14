"""
EXP-L2 相似度篩選結果視覺化
================================
(a) Top-K accuracy bar，對照整體 54.07%
(b) Top-16 最相似圖縮圖網格（含相似度 + fusion_prob）
"""

import os, json, csv
os.environ.setdefault('PYTHONIOENCODING', 'utf-8')

from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.patches import Rectangle
from PIL import Image

for fpath in (r'C:\Windows\Fonts\msjh.ttc', r'C:\Windows\Fonts\msyh.ttc'):
    if os.path.exists(fpath):
        fm.fontManager.addfont(fpath)
        plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        break

OUT_DIR  = Path(r'C:\Users\harry\OneDrive\Desktop\outputs\exp_L2_gpt_image2_full')
REF_IMG  = Path(r'C:\Users\harry\OneDrive\Desktop\CHAT-GPT_IMAGE_2.0\ChatGPT Image 2026年5月25日 下午09_08_39.png')
IMG_DIR  = Path(r'C:\Users\harry\Downloads\gpt-image-2-dataset\gpt-image-2-dataset\images')
SUMMARY  = OUT_DIR / 'similarity_filter_summary.json'
TOP500   = OUT_DIR / 'top500_similar_to_ref.csv'

OUT_BAR  = OUT_DIR / 'figL2sim_a_accuracy.png'
OUT_GRID = OUT_DIR / 'figL2sim_b_top16_grid.png'
OUT_BOTH = OUT_DIR / 'figL2sim_combined.png'

OVERALL_ACC = 54.07


def load_data():
    summary = json.load(open(SUMMARY, encoding='utf-8'))
    with open(TOP500, encoding='utf-8-sig') as f:
        top_rows = list(csv.DictReader(f))
    return summary, top_rows


def plot_bar(ax, summary):
    ks = sorted(int(k) for k in summary['top_k_results'].keys())
    accs = [summary['top_k_results'][str(k)]['accuracy_pct'] for k in ks]
    nais = [summary['top_k_results'][str(k)]['n_AI']        for k in ks]

    x = np.arange(len(ks))
    bars = ax.bar(x, accs, color='#4C72B0', edgecolor='black',
                  linewidth=0.7, width=0.6)
    for b, a, k, nai in zip(bars, accs, ks, nais):
        ax.text(b.get_x() + b.get_width() / 2, a + 1.2,
                f'{a:.1f}%\n({nai}/{k})',
                ha='center', va='bottom',
                fontsize=11, fontweight='bold')

    ax.axhline(OVERALL_ACC, color='#C44E52', ls='--', lw=1.8,
                label=f'整體 10,217 張準確率 = {OVERALL_ACC}%')
    ax.axhline(50, color='gray', ls=':', lw=1.2,
                label='隨機猜測 = 50%')

    ax.set_xticks(x)
    ax.set_xticklabels([f'Top-{k}' for k in ks], fontsize=11)
    ax.set_ylabel('子集準確率 (%)', fontsize=12)
    ax.set_ylim(0, 100)
    ax.set_title('(a) CLIP 相似度子集準確率\n'
                 '（與「復古攝影靜物」參考圖最相近的 Top-K 張）',
                  fontsize=12.5, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    ax.set_axisbelow(True)
    ax.legend(loc='lower left', fontsize=10, framealpha=0.95)


def plot_grid(fig, gs, top_rows, n=16):
    sub_gs = gs.subgridspec(4, 4, hspace=0.45, wspace=0.08)
    for i in range(n):
        r = top_rows[i]
        ax = fig.add_subplot(sub_gs[i // 4, i % 4])
        img_path = IMG_DIR / r['filename']
        try:
            img = Image.open(img_path).convert('RGB')
            img.thumbnail((256, 256))
            ax.imshow(img)
        except Exception:
            ax.text(0.5, 0.5, 'N/A', ha='center', va='center')
        ax.set_xticks([]); ax.set_yticks([])

        sim = float(r['similarity'])
        fp  = float(r['fusion_prob'])
        pred = r['fusion_pred']
        # 框線：綠=對  紅=錯
        color = '#2ca02c' if pred == 'AI' else '#d62728'
        for spine in ax.spines.values():
            spine.set_edgecolor(color); spine.set_linewidth(2.2)
        ax.set_title(f'#{i+1}  sim={sim:.3f}\n'
                     f'AI機率={fp:.3f} → {pred}',
                     fontsize=8.5, color=color,
                     pad=2)


def plot_ref(fig, gs):
    """左上角放參考圖（單格）"""
    ax = fig.add_subplot(gs)
    img = Image.open(REF_IMG).convert('RGB')
    img.thumbnail((640, 640))
    ax.imshow(img)
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_edgecolor('black'); spine.set_linewidth(1.5)
    ax.set_title('參考圖（reference）\n復古攝影靜物', fontsize=11, fontweight='bold')


def main():
    summary, top_rows = load_data()

    # ───────── 合併圖 ─────────
    fig = plt.figure(figsize=(18, 9))
    outer = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.4], wspace=0.18)

    # 左：條圖 + 參考圖
    left = outer[0].subgridspec(2, 1, height_ratios=[1.5, 1.0], hspace=0.32)
    ax_bar = fig.add_subplot(left[0])
    plot_bar(ax_bar, summary)
    plot_ref(fig, left[1])

    # 右：縮圖網格
    right = outer[1].subgridspec(1, 1)
    ax_title = fig.add_subplot(right[0])
    ax_title.axis('off')
    ax_title.set_title('(b) Top-16 最相似圖（綠框=正確，紅框=漏判）\n'
                        '左上 #1 為相似度最高，逐列由左至右遞減',
                        fontsize=12.5, fontweight='bold')
    plot_grid(fig, right[0], top_rows, n=16)

    fig.suptitle('圖（Y）EXP-L2 補充：CLIP 相似度子集評估 — '
                 '「復古靜物」風格子集準確率 79–81%，遠高於整體 54.07%',
                  fontsize=14, fontweight='bold', y=1.0)

    plt.tight_layout(rect=[0, 0, 1, 0.965])
    plt.savefig(OUT_BOTH, dpi=170, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f'[saved] {OUT_BOTH}')


if __name__ == '__main__':
    main()
