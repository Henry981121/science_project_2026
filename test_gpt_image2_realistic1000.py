"""
GPT-Image-2 realistic-1000 資料集測試 + CLIP 零樣本分類
==============================================
針對新的「擬真」資料集 (gpt_image2_realistic_1000) 重跑模型效能。
單次掃描每張：
  ① 主模型判決（fusion_prob）
  ② CLIP zero-shot 8 類分類（寫實人像 / 寫實風景 / 寫實物件 / 寫實場景
     / 動漫插畫 / 文字海報 / 奇幻藝術 / UI 截圖）
  ③ 合併計算：整體準確率、寫實子集準確率、非寫實子集準確率
（與 test_gpt_image2_full_with_category.py 相同流程，僅換資料集與輸出路徑）
"""

import os, sys, json, time, csv, argparse
os.environ.setdefault('PYTHONIOENCODING', 'utf-8')
os.environ.setdefault('HF_HUB_OFFLINE', '1')
# 強制無緩衝輸出
sys.stdout.reconfigure(line_buffering=True) if hasattr(sys.stdout, 'reconfigure') else None

from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

PROJECT_DIR = Path(r'C:\Users\harry\Downloads\east_zone_project_v2\east_zone_project')
sys.path.insert(0, str(PROJECT_DIR))

from detector import AIDetector, STREAMS, EVAL_TF, DEVICE, TEMPERATURE

IMG_DIR = Path(r'C:\Users\harry\OneDrive\Desktop\gpt_image2_realistic_1000\images')
OUT_DIR = Path(r'C:\Users\harry\OneDrive\Desktop\outputs\exp_realistic1000_gpt_image2')
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── 8 類分類提示詞 ──────────────────────────────────────────────
CATEGORIES = {
    # 寫實照片類（模型訓練時主要看到的類型）
    'photo_portrait':  'a photographic portrait of a real person',
    'photo_landscape': 'a landscape photograph of natural scenery',
    'photo_object':    'a photographic close-up of a real object',
    'photo_scene':     'a real-world street or indoor scene photograph',
    # 非寫實類（模型訓練分布外）
    'anime_illust':    'an anime or illustrated character drawing',
    'text_poster':     'a text-heavy poster or graphic design',
    'fantasy_art':     'a fantasy or surreal digital artwork',
    'ui_screenshot':   'a screenshot of an app or UI mockup',
}
CAT_KEYS = list(CATEGORIES.keys())
PHOTO_KEYS = ['photo_portrait', 'photo_landscape', 'photo_object', 'photo_scene']


def parse_args():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument('--sample', type=int)
    g.add_argument('--all', action='store_true')
    return ap.parse_args()


def main():
    args = parse_args()

    all_images = sorted([p for p in IMG_DIR.iterdir()
                          if p.suffix.lower() in ('.jpg', '.jpeg', '.png')])
    if args.all:
        images = all_images
        tag = f'full{len(all_images)}'
    else:
        import random
        rng = random.Random(42)
        images = rng.sample(all_images, args.sample)
        tag = f'sample{args.sample}'

    print(f'[load] Will process {len(images)} images')

    # ─── 載入主模型 ───
    print('[load] Loading AIDetector...')
    t0 = time.time()
    det = AIDetector()
    print(f'[load] Done in {time.time()-t0:.1f}s')

    # ─── 載入獨立 CLIP 模型供 zero-shot 分類 ───
    print('[load] Loading independent CLIPModel for zero-shot...')
    clip_zs = CLIPModel.from_pretrained('openai/clip-vit-large-patch14').to(DEVICE).eval()
    processor = CLIPProcessor.from_pretrained('openai/clip-vit-large-patch14',
                                               use_fast=False)
    text_inputs = processor(text=list(CATEGORIES.values()),
                             return_tensors='pt', padding=True)
    text_inputs = {k: v.to(DEVICE) for k, v in text_inputs.items()}
    with torch.no_grad():
        # 直接走 text_model + text_projection（避免某些 transformers 版本 get_text_features API 差異）
        text_out = clip_zs.text_model(**text_inputs)
        text_emb = clip_zs.text_projection(text_out.pooler_output)
        text_emb = text_emb / text_emb.norm(dim=-1, keepdim=True)
    print(f'[load] {len(CAT_KEYS)} category prompts encoded')
    img_proc = processor

    # ─── 批次推論（增量寫 CSV）───
    csv_path = OUT_DIR / f'predictions_{tag}.csv'
    rows = []
    n_AI, n_Real, n_fail = 0, 0, 0
    t0 = time.time()
    last_print = t0
    last_save = t0

    # 續跑支援：讀已存在的 CSV
    done_files = set()
    if csv_path.exists():
        with open(csv_path, encoding='utf-8-sig') as f:
            for r in csv.DictReader(f):
                done_files.add(r['filename'])
                if r.get('fusion_pred') == 'AI':
                    n_AI += 1
                elif r.get('fusion_pred') == 'Real':
                    n_Real += 1
        print(f'[resume] Found {len(done_files)} existing predictions, will skip',
              flush=True)

    fieldnames = None  # 第一次寫入時建立

    def flush_csv():
        """把目前 rows 增量寫入 CSV"""
        nonlocal fieldnames
        if not rows:
            return
        new_file = not csv_path.exists()
        valid = [r for r in rows if 'fusion_pred' in r]
        if not valid:
            return
        if fieldnames is None:
            fieldnames = list(valid[0].keys())
        with open(csv_path, 'a', encoding='utf-8-sig', newline='') as f:
            w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
            if new_file:
                w.writeheader()
            for r in rows:
                w.writerow(r)
        rows.clear()

    for i, img_path in enumerate(images):
        if img_path.name in done_files:
            continue
        try:
            img_pil = Image.open(img_path).convert('RGB')

            # ── 1) 主模型判決 ──
            img_tensor = EVAL_TF(img_pil).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                feats = [det.extractors[s].extract_features(img_tensor) for s in STREAMS]
                fused = torch.cat(feats, dim=1)
                lb, _, _, _ = det.fusion(fused, grl_lambda=0)
                fusion_prob = F.softmax(lb / TEMPERATURE, dim=1)[0, 1].item()
                fusion_pred = 'AI' if fusion_prob > 0.5 else 'Real'

            # ── 2) CLIP zero-shot 分類 ──
            with torch.no_grad():
                img_inp = img_proc(images=img_pil, return_tensors='pt')
                img_inp = {k: v.to(DEVICE) for k, v in img_inp.items()}
                img_out = clip_zs.vision_model(**img_inp)
                img_emb = clip_zs.visual_projection(img_out.pooler_output)
                img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True)
                # cosine similarity
                sims = (img_emb @ text_emb.T)[0]    # (8,)
                probs = sims.softmax(dim=0).cpu().numpy()
                top_idx = int(probs.argmax())
                top_cat = CAT_KEYS[top_idx]
                top_conf = float(probs[top_idx])
                is_photo = top_cat in PHOTO_KEYS

            row = {
                'idx': i + 1,
                'filename': img_path.name,
                'fusion_prob': round(fusion_prob, 4),
                'fusion_pred': fusion_pred,
                'is_correct': fusion_pred == 'AI',
                'top_category': top_cat,
                'category_confidence': round(top_conf, 4),
                'is_photographic': is_photo,
            }
            # 也記每個類別的相似度
            for k, v in zip(CAT_KEYS, probs):
                row[f'p_{k}'] = round(float(v), 4)
            rows.append(row)

            if fusion_pred == 'AI':
                n_AI += 1
            else:
                n_Real += 1
        except Exception as e:
            n_fail += 1
            rows.append({'idx': i + 1, 'filename': img_path.name, 'error': str(e)})

        now = time.time()
        # 進度（每 30 秒）
        if now - last_print >= 30 or (i + 1) == len(images):
            elapsed = now - t0
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            eta = (len(images) - (i + 1)) / rate / 60 if rate > 0 else 0
            acc = n_AI / max(1, n_AI + n_Real) * 100
            print(f'  [{i+1:5d}/{len(images)}]  rate={rate:.2f} img/s  '
                  f'ETA={eta:.1f} min  ACC={acc:.2f}%', flush=True)
            last_print = now
        # 增量存 CSV（每 100 張或每 2 分鐘）
        if len(rows) >= 100 or (now - last_save) >= 120:
            flush_csv()
            last_save = now

    # ─── 最後 flush ───
    flush_csv()
    print(f'\n[saved CSV] {csv_path}', flush=True)

    # ─── 讀回完整 CSV 算統計 ───
    with open(csv_path, encoding='utf-8-sig') as f:
        all_rows = list(csv.DictReader(f))
    rows = all_rows  # 後續統計用

    # ─── 計算分類統計 ───
    valid = [r for r in rows if r.get('fusion_pred')]
    # CSV 讀回後布林/數值都是字串，轉回
    for r in valid:
        r['is_correct'] = (r['fusion_pred'] == 'AI')
        r['is_photographic'] = (str(r.get('is_photographic')).lower() == 'true')
        r['fusion_prob'] = float(r['fusion_prob'])
    cat_stats = {}
    for k in CAT_KEYS:
        sub = [r for r in valid if r['top_category'] == k]
        if not sub:
            cat_stats[k] = {'n': 0, 'acc': None}
            continue
        n_correct = sum(1 for r in sub if r['is_correct'])
        cat_stats[k] = {
            'n': len(sub),
            'acc': round(n_correct / len(sub) * 100, 2),
            'avg_fusion_prob': round(float(np.mean([r['fusion_prob'] for r in sub])), 4),
        }

    # 寫實 vs 非寫實
    photo_rows = [r for r in valid if r['is_photographic']]
    nonphoto_rows = [r for r in valid if not r['is_photographic']]
    n_AI = sum(1 for r in valid if r['is_correct'])
    n_Real = sum(1 for r in valid if not r['is_correct'])
    photo_acc = (sum(1 for r in photo_rows if r['is_correct']) /
                 max(1, len(photo_rows)) * 100)
    nonphoto_acc = (sum(1 for r in nonphoto_rows if r['is_correct']) /
                    max(1, len(nonphoto_rows)) * 100)

    summary = {
        'tag': tag,
        'dataset': str(IMG_DIR),
        'n_total': len(images),
        'n_evaluated': len(valid),
        'n_failed': n_fail,
        'overall': {
            'n_AI': n_AI,
            'n_Real': n_Real,
            'accuracy_pct': round(n_AI / max(1, n_AI + n_Real) * 100, 2),
        },
        'photographic_subset': {
            'n': len(photo_rows),
            'accuracy_pct': round(photo_acc, 2),
        },
        'non_photographic_subset': {
            'n': len(nonphoto_rows),
            'accuracy_pct': round(nonphoto_acc, 2),
        },
        'per_category': cat_stats,
        'wall_time_min': round((time.time() - t0) / 60, 2),
    }
    summary_path = OUT_DIR / f'summary_{tag}.json'
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')
    print(f'[saved summary] {summary_path}')

    # ─── 列印關鍵摘要 ───
    print()
    print('=' * 70)
    print(f'  GPT-Image-2 realistic-1000 測試結果 ({tag})')
    print('=' * 70)
    print(f'  總體準確率              : {summary["overall"]["accuracy_pct"]}%   ({n_AI}/{len(valid)})')
    print(f'  寫實子集準確率           : {summary["photographic_subset"]["accuracy_pct"]}%  '
          f'(N = {len(photo_rows)})')
    print(f'  非寫實子集準確率         : {summary["non_photographic_subset"]["accuracy_pct"]}%  '
          f'(N = {len(nonphoto_rows)})')
    print()
    print(f'  {"類別":<18s} {"數量":>6s} {"準確率":>8s}  fusion_prob 均')
    print(f'  {"-"*60}')
    for k in CAT_KEYS:
        s = cat_stats[k]
        if s['n'] == 0:
            continue
        photo_tag = '寫實' if k in PHOTO_KEYS else '非寫實'
        print(f'  {k:<14s} [{photo_tag:>3s}]  {s["n"]:>5d}  {s["acc"]:>7.2f}%   {s["avg_fusion_prob"]:.4f}')
    print('=' * 70)


if __name__ == '__main__':
    main()
