"""
EXP-L：ChatGPT Image 2.0 延伸驗證測試
========================================
批次跑主模型對 25 張 ChatGPT Image 2.0 生成圖的推論結果。
輸出：
  outputs/exp_L_chatgpt_image2/per_image_results.csv
  outputs/exp_L_chatgpt_image2/summary.json
"""

import os, sys, json
os.environ.setdefault('PYTHONIOENCODING', 'utf-8')
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')

from pathlib import Path
import time
from PIL import Image

# 把 east_zone_project 的 detector 模組引進來
PROJECT_DIR = Path(r'C:\Users\harry\Downloads\east_zone_project_v2\east_zone_project')
sys.path.insert(0, str(PROJECT_DIR))

from detector import AIDetector, STREAMS, STREAM_DISPLAY

IMG_DIR = Path(r'C:\Users\harry\OneDrive\Desktop\CHAT-GPT_IMAGE_2.0')
OUT_DIR = Path(r'C:\Users\harry\OneDrive\Desktop\outputs\exp_L_chatgpt_image2')
OUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    images = sorted([p for p in IMG_DIR.iterdir()
                      if p.suffix.lower() in ('.png', '.jpg', '.jpeg', '.webp')])
    print(f"[load] Found {len(images)} images")

    print("[load] Loading AIDetector (~30s)...")
    t0 = time.time()
    det = AIDetector()
    print(f"[load] Done in {time.time()-t0:.1f}s")

    rows = []
    n_correct_AI = 0   # ChatGPT Image 2.0 = AI 生成 → 正確答案是 AI
    for i, img_path in enumerate(images, 1):
        try:
            img = Image.open(img_path).convert('RGB')
            t0 = time.time()
            result = det.analyze_image(img)
            dt = time.time() - t0

            f = result['fusion']
            sw = result['stream_weights']
            row = {
                'idx': i,
                'filename': img_path.name,
                'fusion_prob_fake': f['prob_fake'],
                'fusion_pred': f['prediction'],
                'is_correct': f['prediction'] == 'AI',
                'inference_sec': round(dt, 2),
            }
            # 五流明細
            for s in STREAMS:
                if s in result['streams']:
                    r = result['streams'][s]
                    row[f'{s}_fake'] = r['prob_fake']
                    row[f'{s}_pred'] = r['prediction']
                    row[f'{s}_weight'] = sw.get(s, 0)
            rows.append(row)
            n_correct_AI += int(f['prediction'] == 'AI')

            print(f"  [{i:2d}/{len(images)}] fusion={f['prob_fake']:5.2f}% {f['prediction']:4s} "
                  f"| clip={result['streams'].get('clip',{}).get('prob_fake','--')} "
                  f"| {img_path.name[:50]}")
        except Exception as e:
            print(f"  [{i:2d}/{len(images)}] FAILED: {e}")
            rows.append({'idx': i, 'filename': img_path.name, 'error': str(e)})

    # ─── 寫 CSV ───
    import csv
    csv_path = OUT_DIR / 'per_image_results.csv'
    keys = ['idx', 'filename',
            'fusion_prob_fake', 'fusion_pred', 'is_correct',
            'clip_fake', 'fft_fake', 'dct_fake', 'dire_fake', 'noise_fake',
            'clip_weight', 'fft_weight', 'dct_weight', 'dire_weight', 'noise_weight',
            'clip_pred', 'fft_pred', 'dct_pred', 'dire_pred', 'noise_pred',
            'inference_sec']
    with open(csv_path, 'w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction='ignore')
        w.writeheader()
        for row in rows:
            w.writerow(row)
    print(f"\n[saved CSV] {csv_path}")

    # ─── 統計摘要 ───
    n_total = len([r for r in rows if 'fusion_pred' in r])
    n_AI = sum(1 for r in rows if r.get('fusion_pred') == 'AI')
    n_Real = sum(1 for r in rows if r.get('fusion_pred') == 'Real')
    n_fail = sum(1 for r in rows if 'error' in r)

    summary = {
        'n_total': len(images),
        'n_evaluated': n_total,
        'n_failed': n_fail,
        'n_predicted_AI': n_AI,
        'n_predicted_Real': n_Real,
        'accuracy_pct': round(n_AI / n_total * 100, 2) if n_total else 0,
        'note': 'ChatGPT Image 2.0 全部為 AI 生成圖，正確判決應為 AI',
    }
    with open(OUT_DIR / 'summary.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"[saved] {OUT_DIR / 'summary.json'}")

    print()
    print("=" * 60)
    print(f"  EXP-L Summary  (N = {n_total})")
    print("=" * 60)
    print(f"  判為 AI（正確）     : {n_AI}  ({n_AI/n_total*100:.1f}%)")
    print(f"  判為 Real（漏判）   : {n_Real}  ({n_Real/n_total*100:.1f}%)")
    print(f"  失敗                : {n_fail}")
    print(f"  系統判別準確率      : {n_AI/n_total*100:.2f}%")
    print("=" * 60)


if __name__ == '__main__':
    main()
