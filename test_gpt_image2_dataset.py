"""
GPT-Image-2 完整資料集準確率測試
===================================
對 10,217 張 GPT-Image-2 推特採集圖（全部為 AI 生成）跑主模型判別，
計算整體準確率。批次處理，跳過 Grad-CAM 以加速。

用法：
    # 先跑 500 張隨機抽樣
    python test_gpt_image2_dataset.py --sample 500
    # 跑全部
    python test_gpt_image2_dataset.py --all
"""

import os, sys, json, time, argparse, csv, random
os.environ.setdefault('PYTHONIOENCODING', 'utf-8')
os.environ.setdefault('HF_HUB_OFFLINE', '1')

from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

PROJECT_DIR = Path(r'C:\Users\harry\Downloads\east_zone_project_v2\east_zone_project')
sys.path.insert(0, str(PROJECT_DIR))

from detector import AIDetector, STREAMS, EVAL_TF, DEVICE, TEMPERATURE

IMG_DIR = Path(r'C:\Users\harry\Downloads\gpt-image-2-dataset\gpt-image-2-dataset\images')
OUT_DIR = Path(r'C:\Users\harry\OneDrive\Desktop\outputs\exp_L2_gpt_image2_full')
OUT_DIR.mkdir(parents=True, exist_ok=True)


def parse_args():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument('--sample', type=int, help='隨機抽樣 N 張')
    g.add_argument('--all', action='store_true', help='跑全部')
    ap.add_argument('--seed', type=int, default=42)
    return ap.parse_args()


def main():
    args = parse_args()

    # ─── 收集所有圖檔 ───
    all_images = sorted([p for p in IMG_DIR.iterdir()
                          if p.suffix.lower() == '.jpg'])
    print(f'[load] Found {len(all_images)} images in dataset')

    if args.all:
        images = all_images
        tag = 'full'
    else:
        rng = random.Random(args.seed)
        images = rng.sample(all_images, args.sample)
        tag = f'sample{args.sample}'

    print(f'[load] Will process {len(images)} images')

    # ─── 載入主模型 ───
    print('[load] Loading AIDetector (~30s)...')
    t0 = time.time()
    det = AIDetector()
    print(f'[load] Done in {time.time()-t0:.1f}s')

    # ─── 批次推論（單張處理，但跳過 Grad-CAM）───
    csv_path = OUT_DIR / f'predictions_{tag}.csv'
    rows = []
    n_AI, n_Real, n_fail = 0, 0, 0
    t0 = time.time()
    last_print = t0

    for i, img_path in enumerate(images):
        try:
            img_pil = Image.open(img_path).convert('RGB')
            img_tensor = EVAL_TF(img_pil).unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                # 5 流特徵 + 融合
                feats = [det.extractors[s].extract_features(img_tensor) for s in STREAMS]
                fused = torch.cat(feats, dim=1)
                lb, _, _, attn = det.fusion(fused, grl_lambda=0)
                fusion_prob = F.softmax(lb / TEMPERATURE, dim=1)[0, 1].item()
                fusion_pred = 'AI' if fusion_prob > 0.5 else 'Real'

                # 也記 5 流個別預測
                stream_probs = {}
                for s in STREAMS:
                    feat = det.extractors[s].extract_features(img_tensor)
                    logits = det.heads[s](feat)
                    stream_probs[s] = F.softmax(logits / TEMPERATURE, dim=1)[0, 1].item()

            row = {
                'idx': i + 1,
                'filename': img_path.name,
                'fusion_prob': round(fusion_prob, 4),
                'fusion_pred': fusion_pred,
                'is_correct': fusion_pred == 'AI',
                'clip_prob':  round(stream_probs['clip'], 4),
                'fft_prob':   round(stream_probs['fft'], 4),
                'dct_prob':   round(stream_probs['dct'], 4),
                'dire_prob':  round(stream_probs['dire'], 4),
                'noise_prob': round(stream_probs['noise'], 4),
            }
            rows.append(row)
            if fusion_pred == 'AI':
                n_AI += 1
            else:
                n_Real += 1
        except Exception as e:
            n_fail += 1
            rows.append({'idx': i + 1, 'filename': img_path.name, 'error': str(e)})

        # 進度
        now = time.time()
        if now - last_print >= 10 or (i + 1) == len(images):
            elapsed = now - t0
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            eta = (len(images) - (i + 1)) / rate if rate > 0 else 0
            acc_so_far = n_AI / max(1, n_AI + n_Real) * 100
            print(f'  [{i+1:5d}/{len(images)}]  rate={rate:.1f} img/s  '
                  f'ETA={eta/60:.1f} min  current_ACC={acc_so_far:.2f}%')
            last_print = now

    # ─── 儲存 CSV ───
    keys = ['idx', 'filename', 'fusion_prob', 'fusion_pred', 'is_correct',
            'clip_prob', 'fft_prob', 'dct_prob', 'dire_prob', 'noise_prob']
    with open(csv_path, 'w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction='ignore')
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f'\n[saved] {csv_path}')

    # ─── 計算統計 ───
    valid_rows = [r for r in rows if 'fusion_pred' in r]
    fusion_probs = [r['fusion_prob'] for r in valid_rows]

    summary = {
        'tag': tag,
        'n_total': len(images),
        'n_evaluated': len(valid_rows),
        'n_failed': n_fail,
        'n_predicted_AI':   n_AI,
        'n_predicted_Real': n_Real,
        'accuracy_pct':     round(n_AI / max(1, n_AI + n_Real) * 100, 2),
        'fusion_prob_mean': round(float(np.mean(fusion_probs)), 4),
        'fusion_prob_std':  round(float(np.std(fusion_probs)),  4),
        'fusion_prob_median': round(float(np.median(fusion_probs)), 4),
        'wall_time_min': round((time.time() - t0) / 60, 2),
        'note': 'GPT-Image-2 dataset 全部為 AI 生成；準確率 = 預測為 AI 的比例',
    }
    summary_path = OUT_DIR / f'summary_{tag}.json'
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')
    print(f'[saved] {summary_path}')

    # ─── 列印關鍵摘要 ───
    print()
    print('=' * 60)
    print(f'  GPT-Image-2 測試結果 ({tag})')
    print('=' * 60)
    print(f'  總圖片數           : {len(images)}')
    print(f'  成功推論          : {len(valid_rows)}')
    print(f'  判為 AI（正確）    : {n_AI}')
    print(f'  判為 Real（漏判）  : {n_Real}')
    print(f'  失敗              : {n_fail}')
    print(f'  *** 系統準確率    : {summary["accuracy_pct"]:.2f}% ***')
    print(f'  fusion_prob mean   : {summary["fusion_prob_mean"]:.4f}')
    print(f'  fusion_prob median : {summary["fusion_prob_median"]:.4f}')
    print(f'  總耗時            : {summary["wall_time_min"]:.1f} 分鐘')
    print('=' * 60)


if __name__ == '__main__':
    main()
