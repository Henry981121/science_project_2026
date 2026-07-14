"""
分析 10,217 張資料集中與參考圖的 CLIP 相似度分布
==================================================
回答：「資料集中有多少張與參考圖高度相似？」
"""
import os, sys, json, csv
os.environ.setdefault('PYTHONIOENCODING', 'utf-8')
sys.stdout.reconfigure(line_buffering=True)

from pathlib import Path
import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor
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

REF_IMG    = Path(r'C:\Users\harry\OneDrive\Desktop\CHAT-GPT_IMAGE_2.0\ChatGPT Image 2026年5月25日 下午09_08_39.png')
OUT_DIR    = Path(r'C:\Users\harry\OneDrive\Desktop\outputs\exp_L2_gpt_image2_full')
EMB_CACHE  = OUT_DIR / 'clip_embeddings_10217.npy'
NAME_CACHE = OUT_DIR / 'clip_embeddings_names.json'
PRED_CSV   = OUT_DIR / 'predictions_full10217.csv'
OUT_HIST   = OUT_DIR / 'fig_similarity_distribution.png'
OUT_JSON   = OUT_DIR / 'similarity_distribution.json'

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# 分組門檻
BUCKETS = [
    (0.75, 1.00, '極高 >=0.75'),
    (0.70, 0.75, '高    0.70-0.75'),
    (0.65, 0.70, '中    0.65-0.70'),
    (0.60, 0.65, '低    0.60-0.65'),
    (0.55, 0.60, '較低 0.55-0.60'),
    (0.50, 0.55, '很低 0.50-0.55'),
    (0.00, 0.50, '無關 <0.50'),
]


def main():
    # ─── 載入快取 ───
    embs  = np.load(EMB_CACHE)
    names = json.loads(NAME_CACHE.read_text(encoding='utf-8'))
    print(f'[cache] {embs.shape} ({len(names)} names)')

    # ─── 計算參考圖嵌入 ───
    print('[clip] computing reference embedding ...')
    model = CLIPModel.from_pretrained('openai/clip-vit-large-patch14').to(DEVICE).eval()
    proc  = CLIPProcessor.from_pretrained('openai/clip-vit-large-patch14')
    with torch.no_grad():
        inp  = proc(images=Image.open(REF_IMG).convert('RGB'), return_tensors='pt').to(DEVICE)
        vout = model.vision_model(**inp)
        ref  = model.visual_projection(vout.pooler_output)
        ref  = (ref / ref.norm(dim=-1, keepdim=True))[0].cpu().numpy().astype(np.float32)

    sims = embs @ ref
    print(f'[sim] N={len(sims)}  min={sims.min():.4f}  max={sims.max():.4f}  '
          f'mean={sims.mean():.4f}  median={np.median(sims):.4f}  std={sims.std():.4f}')

    # ─── 載入既有預測，準備按 bucket 算準確率 ───
    pred_map = {}
    with open(PRED_CSV, encoding='utf-8-sig') as f:
        for r in csv.DictReader(f):
            if r.get('filename') and r.get('fusion_pred'):
                pred_map[r['filename']] = r['fusion_pred']

    # ─── 各 bucket 統計 ───
    print()
    print(f'{"區間":<22}{"張數":>8}{"佔比":>10}{"準確率":>12}')
    print('─' * 56)
    bucket_results = []
    for lo, hi, label in BUCKETS:
        idx = np.where((sims >= lo) & (sims < hi))[0] if hi < 1.0 \
              else np.where(sims >= lo)[0]
        n = len(idx)
        pct = n / len(sims) * 100
        n_AI = sum(1 for i in idx if pred_map.get(names[i]) == 'AI')
        n_Real = sum(1 for i in idx if pred_map.get(names[i]) == 'Real')
        acc = n_AI / max(1, n_AI + n_Real) * 100
        print(f'{label:<22}{n:>8d}{pct:>9.2f}%{acc:>11.2f}%')
        bucket_results.append({
            'label': label, 'low': lo, 'high': hi,
            'n': int(n), 'pct': round(pct, 2),
            'n_AI': int(n_AI), 'n_Real': int(n_Real),
            'accuracy_pct': round(acc, 2),
        })

    # ─── 累積（≥threshold）統計 ───
    print()
    print('累積(>=門檻):')
    print(f'{"門檻":<14}{"張數":>8}{"佔比":>10}{"準確率":>12}')
    print('─' * 48)
    cum_results = []
    for thr in [0.80, 0.75, 0.70, 0.68, 0.65, 0.62, 0.60, 0.55, 0.50]:
        idx = np.where(sims >= thr)[0]
        n = len(idx)
        pct = n / len(sims) * 100
        n_AI = sum(1 for i in idx if pred_map.get(names[i]) == 'AI')
        n_Real = sum(1 for i in idx if pred_map.get(names[i]) == 'Real')
        acc = n_AI / max(1, n_AI + n_Real) * 100 if (n_AI + n_Real) else 0
        print(f'>={thr:.2f}       {n:>8d}{pct:>9.2f}%{acc:>11.2f}%')
        cum_results.append({
            'threshold': thr, 'n': int(n), 'pct': round(pct, 2),
            'accuracy_pct': round(acc, 2),
        })

    # ─── 圖：直方圖 + 累積 ───
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))

    # (a) 分布直方
    ax = axes[0]
    ax.hist(sims, bins=80, color='#4C72B0', edgecolor='black', linewidth=0.3, alpha=0.85)
    ax.axvline(sims.mean(), color='black', ls='--', lw=1.4,
               label=f'平均 = {sims.mean():.4f}')
    ax.axvline(0.65, color='#C44E52', ls='--', lw=1.4,
               label='0.65（中度相似門檻）')
    ax.axvline(0.70, color='#C44E52', ls='-',  lw=1.6,
               label='0.70（高度相似門檻）')
    ax.set_xlabel('CLIP 餘弦相似度', fontsize=11)
    ax.set_ylabel('圖片數', fontsize=11)
    ax.set_title(f'(a) 10,217 張對參考圖的相似度分布\n'
                  f'min={sims.min():.3f}  max={sims.max():.3f}  median={np.median(sims):.3f}',
                  fontsize=12, fontweight='bold')
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(axis='y', alpha=0.3); ax.set_axisbelow(True)

    # (b) 累積準確率
    ax = axes[1]
    thrs = [c['threshold'] for c in cum_results]
    ns   = [c['n']         for c in cum_results]
    accs = [c['accuracy_pct'] for c in cum_results]
    ax2 = ax.twinx()
    bar = ax.bar(range(len(thrs)), ns, color='#4C72B0',
                 alpha=0.55, edgecolor='black', linewidth=0.6, label='累積張數')
    line = ax2.plot(range(len(thrs)), accs, 'o-',
                    color='#C44E52', lw=2.0, ms=8, label='累積準確率')
    for i, (n, a) in enumerate(zip(ns, accs)):
        ax.text(i, n + max(ns) * 0.01, f'{n}', ha='center', fontsize=9)
        ax2.text(i + 0.15, a + 1.0, f'{a:.1f}%', fontsize=9, color='#C44E52')
    ax.set_xticks(range(len(thrs)))
    ax.set_xticklabels([f'≥{t:.2f}' for t in thrs], fontsize=10)
    ax.set_xlabel('相似度門檻', fontsize=11)
    ax.set_ylabel('累積張數', color='#4C72B0', fontsize=11)
    ax2.set_ylabel('累積準確率 (%)', color='#C44E52', fontsize=11)
    ax2.set_ylim(40, 100)
    ax2.axhline(54.07, color='gray', ls=':', lw=1.2)
    ax2.text(len(thrs) - 0.5, 55.5, '整體 54.07%', color='gray', fontsize=9, ha='right')
    ax.set_title('(b) 不同門檻下的累積張數與準確率',
                  fontsize=12, fontweight='bold')
    ax.grid(axis='y', alpha=0.3); ax.set_axisbelow(True)

    fig.suptitle('資料集中與「復古攝影靜物」參考圖的相似度分布分析（N=10,217）',
                  fontsize=13.5, fontweight='bold', y=1.0)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(OUT_HIST, dpi=170, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f'\n[saved] {OUT_HIST}')

    # ─── JSON 摘要 ───
    out = {
        'reference_image': REF_IMG.name,
        'n_total': int(len(sims)),
        'similarity_stats': {
            'min':    float(sims.min()),
            'max':    float(sims.max()),
            'mean':   float(sims.mean()),
            'median': float(np.median(sims)),
            'std':    float(sims.std()),
        },
        'buckets':           bucket_results,
        'cumulative':        cum_results,
    }
    OUT_JSON.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding='utf-8')
    print(f'[saved] {OUT_JSON}')


if __name__ == '__main__':
    main()
