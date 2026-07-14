"""
GPT-Image-2 「類似風格」子集篩選 + 重新評估
================================================
給定一張參考圖（復古靜物 / 攝影器材），用 CLIP image embedding
餘弦相似度從 10,217 張中挑 Top-K，再用既有預測表算準確率。

第一次跑會建立 10,217 張的 CLIP 嵌入快取 (.npy)；之後跑只需幾秒。
"""

import os, sys, json, csv, time
os.environ.setdefault('PYTHONIOENCODING', 'utf-8')
os.environ.setdefault('HF_HUB_OFFLINE', '1')
sys.stdout.reconfigure(line_buffering=True)

from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

# ───────── 路徑 ─────────
REF_IMG   = Path(r'C:\Users\harry\OneDrive\Desktop\CHAT-GPT_IMAGE_2.0\ChatGPT Image 2026年5月25日 下午09_08_39.png')
IMG_DIR   = Path(r'C:\Users\harry\Downloads\gpt-image-2-dataset\gpt-image-2-dataset\images')
PRED_CSV  = Path(r'C:\Users\harry\OneDrive\Desktop\outputs\exp_L2_gpt_image2_full\predictions_full10217.csv')
OUT_DIR   = Path(r'C:\Users\harry\OneDrive\Desktop\outputs\exp_L2_gpt_image2_full')
EMB_CACHE = OUT_DIR / 'clip_embeddings_10217.npy'
NAME_CACHE = OUT_DIR / 'clip_embeddings_names.json'

TOPK_LIST = [100, 200, 300, 500]   # 多種 K 都報告
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


# ───────── 載入 CLIP ─────────
def load_clip():
    print('[clip] loading openai/clip-vit-large-patch14 ...')
    model = CLIPModel.from_pretrained('openai/clip-vit-large-patch14').to(DEVICE).eval()
    proc  = CLIPProcessor.from_pretrained('openai/clip-vit-large-patch14')
    return model, proc


@torch.no_grad()
def embed_image(model, proc, pil_img):
    inp = proc(images=pil_img, return_tensors='pt').to(DEVICE)
    vout = model.vision_model(**inp)
    emb  = model.visual_projection(vout.pooler_output)
    emb  = emb / emb.norm(dim=-1, keepdim=True)
    return emb[0].cpu().numpy().astype(np.float32)


# ───────── 建立 / 載入 10K 嵌入快取 ─────────
def build_or_load_cache(model, proc, image_paths):
    if EMB_CACHE.exists() and NAME_CACHE.exists():
        names = json.loads(NAME_CACHE.read_text(encoding='utf-8'))
        embs  = np.load(EMB_CACHE)
        if len(names) == len(image_paths) and embs.shape[0] == len(image_paths):
            print(f'[cache] hit — {embs.shape}')
            return names, embs
        print(f'[cache] mismatch (names {len(names)} vs paths {len(image_paths)}); rebuild')

    print(f'[cache] building embeddings for {len(image_paths)} images ...')
    names, vecs = [], []
    t0, last = time.time(), time.time()
    for i, p in enumerate(image_paths):
        try:
            img = Image.open(p).convert('RGB')
            v   = embed_image(model, proc, img)
            names.append(p.name); vecs.append(v)
        except Exception as e:
            print(f'  [skip] {p.name}: {e}')
        if time.time() - last > 15 or i + 1 == len(image_paths):
            r = (i + 1) / (time.time() - t0)
            eta = (len(image_paths) - (i + 1)) / max(r, 1e-6) / 60
            print(f'  [{i+1:5d}/{len(image_paths)}]  {r:.1f} img/s  ETA={eta:.1f} min')
            last = time.time()

    embs = np.stack(vecs, axis=0)
    np.save(EMB_CACHE, embs)
    NAME_CACHE.write_text(json.dumps(names, ensure_ascii=False), encoding='utf-8')
    print(f'[cache] saved {EMB_CACHE} ({embs.shape})')
    return names, embs


# ───────── 主流程 ─────────
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if not REF_IMG.exists():
        print(f'[err] reference image not found: {REF_IMG}')
        sys.exit(1)
    print(f'[ref] {REF_IMG.name}')

    paths = sorted([p for p in IMG_DIR.iterdir() if p.suffix.lower() == '.jpg'])
    print(f'[dataset] {len(paths)} images')

    model, proc = load_clip()

    ref_pil = Image.open(REF_IMG).convert('RGB')
    ref_emb = embed_image(model, proc, ref_pil)
    print(f'[ref] embedding shape={ref_emb.shape}')

    names, embs = build_or_load_cache(model, proc, paths)

    # cosine sim (embs are already L2-normalized)
    sims = embs @ ref_emb                          # (N,)
    order = np.argsort(-sims)                      # 由高到低

    # ───────── 對照預測 CSV ─────────
    print(f'[csv] reading {PRED_CSV}')
    rows = {}
    with open(PRED_CSV, encoding='utf-8-sig') as f:
        for r in csv.DictReader(f):
            if r.get('filename') and r.get('fusion_prob'):
                rows[r['filename']] = r

    matched = [(names[i], float(sims[i]), rows.get(names[i])) for i in order]
    matched_valid = [(n, s, r) for (n, s, r) in matched if r is not None]
    print(f'[match] {len(matched_valid)}/{len(matched)} have predictions')

    # ───────── 各 K 的子集準確率 ─────────
    results = {}
    for K in TOPK_LIST:
        sub = matched_valid[:K]
        n_AI   = sum(1 for _, _, r in sub if r['fusion_pred'] == 'AI')
        n_Real = sum(1 for _, _, r in sub if r['fusion_pred'] == 'Real')
        acc    = n_AI / max(1, n_AI + n_Real) * 100
        sims_K = [s for _, s, _ in sub]
        probs  = [float(r['fusion_prob']) for _, _, r in sub]
        results[K] = {
            'n': len(sub),
            'sim_min':  round(float(min(sims_K)), 4),
            'sim_max':  round(float(max(sims_K)), 4),
            'sim_mean': round(float(np.mean(sims_K)), 4),
            'n_AI': n_AI,
            'n_Real': n_Real,
            'accuracy_pct': round(acc, 2),
            'fusion_prob_mean':   round(float(np.mean(probs)), 4),
            'fusion_prob_median': round(float(np.median(probs)), 4),
        }
        print()
        print(f'─── Top-{K} 結果 ───────────────────────────────')
        print(f'  similarity range : {results[K]["sim_min"]:.4f} ~ {results[K]["sim_max"]:.4f}'
              f'   (mean={results[K]["sim_mean"]:.4f})')
        print(f'  正確判為 AI       : {n_AI}')
        print(f'  漏判為 Real       : {n_Real}')
        print(f'  *** 子集準確率   : {acc:.2f}% ***')
        print(f'  fusion_prob mean={results[K]["fusion_prob_mean"]:.4f}'
              f'   median={results[K]["fusion_prob_median"]:.4f}')

    # ───────── 儲存子集明細 CSV (Top 500) ─────────
    sub500 = matched_valid[:500]
    out_csv = OUT_DIR / 'top500_similar_to_ref.csv'
    keys = ['rank', 'filename', 'similarity', 'fusion_prob', 'fusion_pred',
            'clip_prob', 'fft_prob', 'dct_prob', 'dire_prob', 'noise_prob']
    with open(out_csv, 'w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for rank, (name, s, r) in enumerate(sub500, 1):
            w.writerow({
                'rank': rank, 'filename': name, 'similarity': round(s, 4),
                'fusion_prob': r['fusion_prob'], 'fusion_pred': r['fusion_pred'],
                'clip_prob': r.get('clip_prob', ''), 'fft_prob': r.get('fft_prob', ''),
                'dct_prob':  r.get('dct_prob', ''), 'dire_prob': r.get('dire_prob', ''),
                'noise_prob': r.get('noise_prob', ''),
            })
    print(f'\n[saved] {out_csv}')

    # ───────── 儲存 summary JSON ─────────
    summary = {
        'reference_image': REF_IMG.name,
        'n_total':   len(paths),
        'n_matched': len(matched_valid),
        'top_k_results': results,
        'note': '子集準確率 = 在與參考圖最相似的 Top-K 張中，主模型判為 AI 的比例',
    }
    out_json = OUT_DIR / 'similarity_filter_summary.json'
    out_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')
    print(f'[saved] {out_json}')


if __name__ == '__main__':
    main()
