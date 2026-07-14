"""
以 EXP-L 之 25 張「真人攝影風格」測試圖為參考集，
從 10,217 張 GPT-Image-2 中篩選「攝影風格」高度相似子集，並重新評估。

方法：
  A. centroid：25 張 CLIP 嵌入平均 → L2 正規化 → 共通「攝影風格」中心
  B. max-sim ：每張 dataset 圖對 25 張取最大相似度

主要採用 A（消除個別主題特化，保留攝影共性）
"""
import os, sys, json, csv, time
os.environ.setdefault('PYTHONIOENCODING', 'utf-8')
os.environ.setdefault('HF_HUB_OFFLINE', '1')
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

REFSET_DIR = Path(r'C:\Users\harry\OneDrive\Desktop\CHAT-GPT_IMAGE_2.0')
IMG_DIR    = Path(r'C:\Users\harry\Downloads\gpt-image-2-dataset\gpt-image-2-dataset\images')
OUT_DIR    = Path(r'C:\Users\harry\OneDrive\Desktop\outputs\exp_L2_gpt_image2_full')
EMB_CACHE  = OUT_DIR / 'clip_embeddings_10217.npy'
NAME_CACHE = OUT_DIR / 'clip_embeddings_names.json'
REF_EMB    = OUT_DIR / 'refset25_embeddings.npy'
PRED_CSV   = OUT_DIR / 'predictions_full10217.csv'
OUT_CSV_A  = OUT_DIR / 'refset_top500_centroid.csv'
OUT_CSV_B  = OUT_DIR / 'refset_top500_maxsim.csv'
OUT_JSON   = OUT_DIR / 'refset_similarity_summary.json'

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
TOPK_LIST = [100, 200, 300, 500, 1000]
OVERALL_ACC = 54.07


@torch.no_grad()
def embed_image(model, proc, pil_img):
    inp = proc(images=pil_img, return_tensors='pt').to(DEVICE)
    vout = model.vision_model(**inp)
    emb  = model.visual_projection(vout.pooler_output)
    emb  = emb / emb.norm(dim=-1, keepdim=True)
    return emb[0].cpu().numpy().astype(np.float32)


def load_clip():
    print('[clip] loading openai/clip-vit-large-patch14 ...')
    model = CLIPModel.from_pretrained('openai/clip-vit-large-patch14').to(DEVICE).eval()
    proc  = CLIPProcessor.from_pretrained('openai/clip-vit-large-patch14')
    return model, proc


def compute_refset(model, proc):
    if REF_EMB.exists():
        ref_embs = np.load(REF_EMB)
        print(f'[refset] cached: {ref_embs.shape}')
        return ref_embs

    ref_paths = sorted([p for p in REFSET_DIR.iterdir()
                        if p.suffix.lower() in ('.png', '.jpg', '.jpeg')])
    print(f'[refset] {len(ref_paths)} images')
    embs = []
    for i, p in enumerate(ref_paths):
        try:
            v = embed_image(model, proc, Image.open(p).convert('RGB'))
            embs.append(v)
            print(f'  [{i+1:2d}/{len(ref_paths)}] {p.name}')
        except Exception as e:
            print(f'  [skip] {p.name}: {e}')
    embs = np.stack(embs, axis=0)
    np.save(REF_EMB, embs)
    print(f'[refset] saved {REF_EMB} ({embs.shape})')
    return embs


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    model, proc = load_clip()

    ref_embs = compute_refset(model, proc)
    n_ref = ref_embs.shape[0]

    # ─── 質心 ───
    centroid = ref_embs.mean(axis=0)
    centroid = centroid / np.linalg.norm(centroid)
    print(f'[centroid] L2-norm = 1.0 (after renormalize)')

    # ─── 載入 10K 快取 ───
    embs  = np.load(EMB_CACHE)
    names = json.loads(NAME_CACHE.read_text(encoding='utf-8'))
    print(f'[10K] {embs.shape}')

    # ─── 方法 A：對質心 ───
    sims_centroid = embs @ centroid

    # ─── 方法 B：對 25 張取最大 ───
    sims_max = (embs @ ref_embs.T).max(axis=1)
    sims_mean = (embs @ ref_embs.T).mean(axis=1)

    print()
    print(f'[A centroid] min={sims_centroid.min():.4f}  max={sims_centroid.max():.4f}  '
          f'mean={sims_centroid.mean():.4f}  median={np.median(sims_centroid):.4f}')
    print(f'[B max-sim ] min={sims_max.min():.4f}  max={sims_max.max():.4f}  '
          f'mean={sims_max.mean():.4f}  median={np.median(sims_max):.4f}')

    # ─── 載入既有預測 ───
    pred_map = {}
    with open(PRED_CSV, encoding='utf-8-sig') as f:
        for r in csv.DictReader(f):
            if r.get('filename') and r.get('fusion_pred'):
                pred_map[r['filename']] = r

    # ─── 各 K 結果（兩種方法）───
    def eval_topk(sims, name):
        order = np.argsort(-sims)
        result = {}
        for K in TOPK_LIST:
            top_idx = order[:K]
            top_names = [names[i] for i in top_idx]
            n_AI = sum(1 for nm in top_names if pred_map.get(nm, {}).get('fusion_pred') == 'AI')
            n_Real = sum(1 for nm in top_names if pred_map.get(nm, {}).get('fusion_pred') == 'Real')
            acc = n_AI / max(1, n_AI + n_Real) * 100
            sims_K = sims[top_idx]
            probs = [float(pred_map[nm]['fusion_prob'])
                     for nm in top_names if nm in pred_map]
            result[K] = {
                'n': K, 'n_AI': n_AI, 'n_Real': n_Real,
                'accuracy_pct': round(acc, 2),
                'sim_min': round(float(sims_K.min()), 4),
                'sim_max': round(float(sims_K.max()), 4),
                'sim_mean': round(float(sims_K.mean()), 4),
                'fusion_prob_mean':   round(float(np.mean(probs)), 4),
                'fusion_prob_median': round(float(np.median(probs)), 4),
            }
            print(f'  [{name:8s}] Top-{K:4d}  sim {sims_K.min():.3f}~{sims_K.max():.3f} '
                  f'(mean={sims_K.mean():.3f})  AI={n_AI}/Real={n_Real}  '
                  f'ACC={acc:.2f}%  prob_med={np.median(probs):.3f}')
        return result, order

    print()
    print('── 方法 A：對 25 張平均質心的相似度 ──')
    result_A, order_A = eval_topk(sims_centroid, 'centroid')
    print()
    print('── 方法 B：對 25 張最大相似度 ──')
    result_B, order_B = eval_topk(sims_max, 'max-sim')

    # ─── 寫 Top-500 CSV（方法 A）───
    def write_csv(path, order, sims, ref_kind):
        keys = ['rank', 'filename', 'similarity', 'fusion_prob', 'fusion_pred',
                'clip_prob', 'fft_prob', 'dct_prob', 'dire_prob', 'noise_prob']
        with open(path, 'w', encoding='utf-8-sig', newline='') as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for rank, i in enumerate(order[:500], 1):
                nm = names[i]
                r = pred_map.get(nm, {})
                w.writerow({
                    'rank': rank, 'filename': nm,
                    'similarity': round(float(sims[i]), 4),
                    'fusion_prob': r.get('fusion_prob', ''),
                    'fusion_pred': r.get('fusion_pred', ''),
                    'clip_prob':  r.get('clip_prob', ''),
                    'fft_prob':   r.get('fft_prob', ''),
                    'dct_prob':   r.get('dct_prob', ''),
                    'dire_prob':  r.get('dire_prob', ''),
                    'noise_prob': r.get('noise_prob', ''),
                })
        print(f'[saved] {path}')

    write_csv(OUT_CSV_A, order_A, sims_centroid, 'centroid')
    write_csv(OUT_CSV_B, order_B, sims_max, 'max-sim')

    # ─── 累積（>=門檻）統計 — 方法 A ───
    print()
    print('── 方法 A 累積（>=門檻）──')
    cum_A = []
    for thr in [0.80, 0.75, 0.70, 0.68, 0.65, 0.62, 0.60, 0.55, 0.50]:
        idx = np.where(sims_centroid >= thr)[0]
        n = len(idx)
        n_AI = sum(1 for i in idx if pred_map.get(names[i], {}).get('fusion_pred') == 'AI')
        n_Real = sum(1 for i in idx if pred_map.get(names[i], {}).get('fusion_pred') == 'Real')
        acc = n_AI / max(1, n_AI + n_Real) * 100 if (n_AI + n_Real) else 0
        pct = n / len(sims_centroid) * 100
        cum_A.append({'threshold': thr, 'n': int(n), 'pct': round(pct, 2),
                      'accuracy_pct': round(acc, 2)})
        print(f'  >={thr:.2f}   {n:>5d}  ({pct:5.2f}%)   ACC={acc:.2f}%')

    # ─── JSON 摘要 ───
    summary = {
        'reference_set':       str(REFSET_DIR),
        'n_reference_images':  int(n_ref),
        'method': {
            'A_centroid': '對 25 張 CLIP 嵌入平均後 L2-normalize，再對 10K 算 cosine',
            'B_maxsim':   '對 10K 中每張，取它與 25 張中最大的 cosine',
        },
        'overall_baseline_acc_pct': OVERALL_ACC,
        'method_A_centroid': {
            'sim_stats': {
                'min':    float(sims_centroid.min()),
                'max':    float(sims_centroid.max()),
                'mean':   float(sims_centroid.mean()),
                'median': float(np.median(sims_centroid)),
                'std':    float(sims_centroid.std()),
            },
            'top_k_results': result_A,
            'cumulative':    cum_A,
        },
        'method_B_maxsim': {
            'sim_stats': {
                'min':    float(sims_max.min()),
                'max':    float(sims_max.max()),
                'mean':   float(sims_max.mean()),
                'median': float(np.median(sims_max)),
                'std':    float(sims_max.std()),
            },
            'top_k_results': result_B,
        },
    }
    OUT_JSON.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')
    print(f'\n[saved] {OUT_JSON}')

    # ─── 也存質心相似度陣列，方便後續視覺化 ───
    np.save(OUT_DIR / 'sims_centroid.npy', sims_centroid)
    np.save(OUT_DIR / 'sims_maxsim.npy',   sims_max)
    print(f'[saved] sims_centroid.npy, sims_maxsim.npy')


if __name__ == '__main__':
    main()
