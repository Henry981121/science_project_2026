"""
GPT-Image-2 完整測試（N=10,217）視覺化
=========================================
(a) 各 CLIP zero-shot 類別的準確率長條圖
(b) fusion_prob 分布直方（含 0.5 決策邊界）
"""

import os, json, csv
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

JSON_PATH = Path(r'C:\Users\harry\OneDrive\Desktop\outputs\exp_L2_gpt_image2_full\summary_full10217.json')
CSV_PATH  = Path(r'C:\Users\harry\OneDrive\Desktop\outputs\exp_L2_gpt_image2_full\predictions_full10217.csv')
OUT_A     = Path(r'C:\Users\harry\OneDrive\Desktop\outputs\exp_L2_gpt_image2_full\figL2_a_category_accuracy.png')
OUT_B     = Path(r'C:\Users\harry\OneDrive\Desktop\outputs\exp_L2_gpt_image2_full\figL2_b_fusion_prob_hist.png')
OUT_AB    = Path(r'C:\Users\harry\OneDrive\Desktop\outputs\exp_L2_gpt_image2_full\figL2_combined.png')

CAT_ORDER = [
    'photo_portrait', 'photo_landscape', 'photo_object', 'photo_scene',
    'anime_illust',   'text_poster',     'fantasy_art',  'ui_screenshot',
]
CAT_DISPLAY = {
    'photo_portrait':  '寫實人像\nportrait',
    'photo_landscape': '寫實風景\nlandscape',
    'photo_object':    '寫實物件\nobject',
    'photo_scene':     '寫實場景\nscene',
    'anime_illust':    '動漫插畫\nanime',
    'text_poster':     '文字海報\ntext',
    'fantasy_art':     '奇幻藝術\nfantasy',
    'ui_screenshot':   'UI 截圖\nUI',
}
PHOTO_KEYS = ['photo_portrait', 'photo_landscape', 'photo_object', 'photo_scene']


def main():
    summary = json.load(open(JSON_PATH, encoding='utf-8'))
    overall_acc = summary['overall']['accuracy_pct']

    # ── 讀 CSV 取所有 fusion_prob ──
    with open(CSV_PATH, encoding='utf-8-sig') as f:
        rows = list(csv.DictReader(f))
    valid = [r for r in rows if r.get('fusion_prob')]
    probs_all = np.array([float(r['fusion_prob']) for r in valid])
    probs_photo = np.array([float(r['fusion_prob']) for r in valid
                            if str(r.get('is_photographic')).lower() == 'true'])
    probs_nonphoto = np.array([float(r['fusion_prob']) for r in valid
                               if str(r.get('is_photographic')).lower() != 'true'])

    # ─────────────────────────────────────────────
    # 合併 (a) + (b)
    # ─────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(17, 6.5),
                              gridspec_kw={'width_ratios': [1.3, 1]})

    # ── (a) 各類別準確率 ──
    ax = axes[0]
    cats = CAT_ORDER
    accs = [summary['per_category'][k]['acc'] for k in cats]
    nvals = [summary['per_category'][k]['n'] for k in cats]
    colors = ['#4C72B0' if k in PHOTO_KEYS else '#C44E52' for k in cats]

    x = np.arange(len(cats))
    bars = ax.bar(x, accs, color=colors, edgecolor='black', linewidth=0.7)
    for b, a, n in zip(bars, accs, nvals):
        ax.text(b.get_x() + b.get_width() / 2, a + 1.0,
                f'{a:.1f}%\n(N={n})',
                ha='center', fontsize=9.5, fontweight='bold')

    # 整體準確率線 + 隨機猜測線
    ax.axhline(overall_acc, color='black', ls='-', lw=1.4,
                label=f'整體準確率 = {overall_acc}%')
    ax.axhline(50, color='gray', ls='--', lw=1.2,
                label='隨機猜測 = 50%')

    ax.set_xticks(x)
    ax.set_xticklabels([CAT_DISPLAY[k] for k in cats], fontsize=10)
    ax.set_ylabel('準確率 (%)', fontsize=11)
    ax.set_ylim(0, 100)
    ax.set_title(f'(a) 各類別準確率（CLIP zero-shot 分類，N = 10,217）\n'
                  f'藍色 = 寫實子集；紅色 = 非寫實子集',
                  fontsize=12, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    ax.set_axisbelow(True)
    ax.legend(loc='upper right', fontsize=10, framealpha=0.95)

    # ── (b) fusion_prob 分布 ──
    ax = axes[1]
    bins = np.linspace(0, 1, 41)
    ax.hist(probs_photo, bins=bins, alpha=0.55, color='#4C72B0',
             label=f'寫實子集 (N={len(probs_photo)})', edgecolor='black', linewidth=0.4)
    ax.hist(probs_nonphoto, bins=bins, alpha=0.55, color='#C44E52',
             label=f'非寫實子集 (N={len(probs_nonphoto)})', edgecolor='black', linewidth=0.4)
    ax.axvline(0.5, color='black', ls='--', lw=1.4, label='決策邊界 = 0.5')

    ax.set_xlabel('fusion_prob (預測為 AI 之機率)', fontsize=11)
    ax.set_ylabel('圖片數', fontsize=11)
    ax.set_title(f'(b) fusion_prob 分布\n'
                  f'寫實 median={np.median(probs_photo):.3f}, '
                  f'非寫實 median={np.median(probs_nonphoto):.3f}',
                  fontsize=12, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    ax.set_axisbelow(True)
    ax.legend(loc='upper center', fontsize=10, framealpha=0.95)

    fig.suptitle(
        '圖（X）EXP-L2：GPT-Image-2 完整資料集測試結果 — 整體準確率 54.07%',
        fontsize=14, fontweight='bold', y=1.0)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(OUT_AB, dpi=170, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f'[saved] {OUT_AB}')


if __name__ == '__main__':
    main()
