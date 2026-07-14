"""
EXP-L2 補充：以 EXP-L 25 張為參考集的相似度篩選結果視覺化
=========================================================
左：Top-K 準確率 vs 整體 54.07%
中：累積（>=門檻）的張數與準確率
右：Top-16 最相似圖縮圖網格（centroid 法）
"""
import os, json, csv
os.environ.setdefault('PYTHONIOENCODING', 'utf-8')

from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from PIL import Image

for fpath in (r'C:\Windows\Fonts\msjh.ttc', r'C:\Windows\Fonts\msyh.ttc'):
    if os.path.exists(fpath):
        fm.fontManager.addfont(fpath)
        plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        break

OUT_DIR  = Path(r'C:\Users\harry\OneDrive\Desktop\outputs\exp_L2_gpt_image2_full')
IMG_DIR  = Path(r'C:\Users\harry\Downloads\gpt-image-2-dataset\gpt-image-2-dataset\images')
SUMMARY  = OUT_DIR / 'refset_similarity_summary.json'
TOPCSV   = OUT_DIR / 'refset_top500_centroid.csv'

OUT_FIG  = OUT_DIR / 'figL2_refset_filter.png'
OVERALL  = 54.07
EXPL_ACC = 84.00


def load():
    s = json.load(open(SUMMARY, encoding='utf-8'))
    with open(TOPCSV, encoding='utf-8-sig') as f:
        rows = list(csv.DictReader(f))
    return s, rows


def main():
    summary, top_rows = load()

    fig = plt.figure(figsize=(20, 9.5))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.2],
                          width_ratios=[1, 1, 1.3], hspace=0.34, wspace=0.25)

    # ─── (a) Top-K 準確率長條 ───
    ax = fig.add_subplot(gs[0, 0])
    A = summary['method_A_centroid']['top_k_results']
    ks = sorted(int(k) for k in A.keys())
    accs = [A[str(k)]['accuracy_pct'] for k in ks]
    nais = [A[str(k)]['n_AI'] for k in ks]

    x = np.arange(len(ks))
    bars = ax.bar(x, accs, color='#4C72B0', edgecolor='black',
                  linewidth=0.7, width=0.6)
    for b, a, k, nai in zip(bars, accs, ks, nais):
        ax.text(b.get_x() + b.get_width() / 2, a + 1.2,
                f'{a:.1f}%\n({nai}/{k})',
                ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax.axhline(OVERALL,  color='#C44E52', ls='--', lw=1.8,
                label=f'整體 10,217 = {OVERALL}%')
    ax.axhline(EXPL_ACC, color='green',   ls='--', lw=1.6,
                label=f'EXP-L 25 張 = {EXPL_ACC}%')
    ax.axhline(50, color='gray', ls=':', lw=1.0, label='隨機 = 50%')
    ax.set_xticks(x)
    ax.set_xticklabels([f'Top-{k}' for k in ks], fontsize=10)
    ax.set_ylabel('子集準確率 (%)', fontsize=11)
    ax.set_ylim(0, 100)
    ax.set_title('(a) 攝影風格子集準確率\n(centroid 法：對 25 張平均嵌入的 cosine sim)',
                  fontsize=11.5, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    ax.set_axisbelow(True)
    ax.legend(loc='lower left', fontsize=9, framealpha=0.95)

    # ─── (b) 累積 N + 準確率（雙 y 軸）───
    ax  = fig.add_subplot(gs[0, 1])
    ax2 = ax.twinx()
    cum = summary['method_A_centroid']['cumulative']
    thrs = [c['threshold']    for c in cum]
    ns   = [c['n']            for c in cum]
    accs = [c['accuracy_pct'] for c in cum]
    xs = np.arange(len(thrs))
    bar = ax.bar(xs, ns, color='#4C72B0', alpha=0.6,
                 edgecolor='black', linewidth=0.6, label='累積張數')
    line, = ax2.plot(xs, accs, 'o-', color='#C44E52', lw=2.0, ms=8,
                     label='累積準確率')
    for i, (n, a) in enumerate(zip(ns, accs)):
        ax.text(i, n + max(ns) * 0.015, f'{n}', ha='center', fontsize=8.5)
        ax2.text(i + 0.15, a + 1.0, f'{a:.1f}%', fontsize=8.5, color='#C44E52')
    ax.set_xticks(xs)
    ax.set_xticklabels([f'>={t:.2f}' for t in thrs], fontsize=9, rotation=15)
    ax.set_xlabel('攝影風格相似度門檻', fontsize=11)
    ax.set_ylabel('累積張數', color='#4C72B0', fontsize=11)
    ax2.set_ylabel('準確率 (%)', color='#C44E52', fontsize=11)
    ax2.set_ylim(40, 100)
    ax2.axhline(OVERALL, color='gray', ls=':', lw=1.0)
    ax.set_title('(b) 不同門檻下的張數與準確率', fontsize=11.5, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    ax.set_axisbelow(True)

    # ─── (c) 相似度分布 ───
    sims = np.load(OUT_DIR / 'sims_centroid.npy')
    ax = fig.add_subplot(gs[0, 2])
    ax.hist(sims, bins=80, color='#4C72B0', edgecolor='black',
             linewidth=0.3, alpha=0.85)
    ax.axvline(sims.mean(), color='black', ls='--', lw=1.4,
                label=f'平均 = {sims.mean():.3f}')
    ax.axvline(0.70, color='#C44E52', ls='-',  lw=1.6, label='>=0.70 高度相似')
    ax.axvline(0.65, color='#C44E52', ls='--', lw=1.4, label='>=0.65 中度相似')
    ax.set_xlabel('攝影風格相似度', fontsize=11)
    ax.set_ylabel('圖片數', fontsize=11)
    ax.set_title(f'(c) 10,217 張對攝影風格中心的相似度分布\n'
                  f'min={sims.min():.3f}  max={sims.max():.3f}  median={np.median(sims):.3f}',
                  fontsize=11.5, fontweight='bold')
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(axis='y', alpha=0.3); ax.set_axisbelow(True)

    # ─── (d) Top-16 縮圖網格 ───
    ax_title = fig.add_subplot(gs[1, :])
    ax_title.axis('off')
    ax_title.set_title('(d) Top-16 最相似圖（綠框=正確判 AI，紅框=漏判）',
                        fontsize=12, fontweight='bold', pad=2)
    grid_gs = gs[1, :].subgridspec(2, 8, hspace=0.5, wspace=0.05)
    for i in range(16):
        r = top_rows[i]
        ax = fig.add_subplot(grid_gs[i // 8, i % 8])
        try:
            img = Image.open(IMG_DIR / r['filename']).convert('RGB')
            img.thumbnail((256, 256))
            ax.imshow(img)
        except Exception:
            ax.text(0.5, 0.5, 'N/A', ha='center', va='center')
        ax.set_xticks([]); ax.set_yticks([])
        sim = float(r['similarity']);  fp = float(r['fusion_prob'])
        pred = r['fusion_pred']
        color = '#2ca02c' if pred == 'AI' else '#d62728'
        for s in ax.spines.values():
            s.set_edgecolor(color); s.set_linewidth(2.2)
        ax.set_title(f'#{i+1}  sim={sim:.3f}\nAI機率={fp:.2f}',
                      fontsize=8.5, color=color, pad=2)

    fig.suptitle('圖（Z）EXP-L2 補充：以 EXP-L 25 張「真人攝影風格」為參考集 — '
                 '攝影風格子集準確率 78–88%（整體僅 54.07%）',
                  fontsize=14, fontweight='bold', y=0.995)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(OUT_FIG, dpi=170, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f'[saved] {OUT_FIG}')


if __name__ == '__main__':
    main()
