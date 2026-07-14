"""
realistic-1000 失效分析
====================================
回答兩個問題：
  (A) 模型內部：5 個 stream (clip/fft/dct/dire/noise) 各自的偵測率，
      找出是哪一條 stream 把整體 fusion 拖低。
  (B) 資料層面：用 metadata (scene_id/region/subject/weather/格式 jpg-png)
      交叉分析「被抓到 (AI)」與「漏掉 (Real)」的關鍵差異。
輸出：
  - perstream_full496.csv  每張每 stream 的 prob_fake
  - analysis_report.json / 終端報表
"""
import os, sys, json, csv
os.environ.setdefault('PYTHONIOENCODING', 'utf-8')
os.environ.setdefault('HF_HUB_OFFLINE', '1')
sys.stdout.reconfigure(line_buffering=True) if hasattr(sys.stdout, 'reconfigure') else None

from pathlib import Path
from collections import defaultdict
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

PROJECT_DIR = Path(r'C:\Users\harry\Downloads\east_zone_project_v2\east_zone_project')
sys.path.insert(0, str(PROJECT_DIR))
from detector import AIDetector, STREAMS, EVAL_TF, DEVICE, TEMPERATURE

DATA_DIR = Path(r'C:\Users\harry\OneDrive\Desktop\gpt_image2_realistic_1000')
IMG_DIR = DATA_DIR / 'images'
OUT_DIR = Path(r'C:\Users\harry\OneDrive\Desktop\outputs\exp_realistic1000_gpt_image2')
META = DATA_DIR / 'metadata.jsonl'

THRESH = 0.5  # prob_fake > 0.5 判 AI


def load_metadata():
    m = {}
    with open(META, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            spec = d.get('prompt_spec', {}) or {}
            m[d['file_name']] = {
                'scene_id': spec.get('scene_id', '?'),
                'region': spec.get('region', '?'),
                'subject': spec.get('subject', '?'),
                'weather': spec.get('weather', '?'),
                'lighting': spec.get('lighting', '?'),
            }
    return m


def has_person(subject: str) -> str:
    s = subject.lower()
    if 'no main person' in s or 'no posed' in s or 'no person' in s:
        return 'no_person'
    if 'person' in s or 'model' in s or 'people' in s or 'portrait' in s:
        return 'person'
    return 'other'


def main():
    images = sorted([p for p in IMG_DIR.iterdir()
                     if p.suffix.lower() in ('.jpg', '.jpeg', '.png')])
    print(f'[load] {len(images)} images')
    meta = load_metadata()
    det = AIDetector()
    has_heads = [s for s in STREAMS if s in det.heads]
    print(f'[info] per-stream heads available: {has_heads}')

    rows = []
    for i, p in enumerate(images):
        try:
            img = Image.open(p).convert('RGB')
            t = EVAL_TF(img).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                feats = {s: det.extractors[s].extract_features(t) for s in STREAMS}
                # fusion
                fused = torch.cat([feats[s] for s in STREAMS], dim=1)
                lb, _, _, _ = det.fusion(fused, grl_lambda=0)
                fusion_prob = F.softmax(lb / TEMPERATURE, dim=1)[0, 1].item()
                # per-stream heads
                sp = {}
                for s in has_heads:
                    logits = det.heads[s](feats[s])
                    sp[s] = F.softmax(logits / TEMPERATURE, dim=1)[0, 1].item()
            row = {'filename': p.name, 'format': p.suffix.lower().lstrip('.'),
                   'fusion_prob': round(fusion_prob, 4)}
            for s in has_heads:
                row[f'{s}_prob'] = round(sp[s], 4)
            row.update(meta.get(p.name, {'scene_id': '?', 'region': '?',
                                          'subject': '?', 'weather': '?', 'lighting': '?'}))
            row['person'] = has_person(row.get('subject', ''))
            rows.append(row)
        except Exception as e:
            rows.append({'filename': p.name, 'error': str(e)})
        if (i + 1) % 100 == 0 or (i + 1) == len(images):
            print(f'  {i+1}/{len(images)}', flush=True)

    valid = [r for r in rows if 'fusion_prob' in r]

    # 寫 per-stream CSV
    fn = list(valid[0].keys())
    with open(OUT_DIR / 'perstream_full496.csv', 'w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fn, extrasaction='ignore')
        w.writeheader()
        w.writerows(valid)

    # ── (A) 每 stream 偵測率 + fusion ──
    def det_rate(rs, key):
        probs = [r[key] for r in rs]
        n_ai = sum(1 for x in probs if x > THRESH)
        return n_ai, len(rs), round(n_ai / max(1, len(rs)) * 100, 2), round(float(np.mean(probs)), 4)

    print('\n' + '=' * 64)
    print('  (A) 各 stream 偵測率（單獨判 AI 的比例）')
    print('=' * 64)
    stream_summary = {}
    for s in has_heads + ['fusion']:
        key = 'fusion_prob' if s == 'fusion' else f'{s}_prob'
        n_ai, n, rate, mean = det_rate(valid, key)
        stream_summary[s] = {'detect_rate': rate, 'mean_prob': mean, 'n_ai': n_ai, 'n': n}
        bar = '#' * int(rate / 2)
        print(f'  {s:8s}  偵測率 {rate:6.2f}%  平均prob {mean:.3f}  {bar}')

    # ── 把 fusion 分成 命中(AI) / 漏抓(Real) 兩群，比較各 stream 平均 ──
    hit = [r for r in valid if r['fusion_prob'] > THRESH]
    miss = [r for r in valid if r['fusion_prob'] <= THRESH]
    print('\n' + '=' * 64)
    print(f'  命中群 N={len(hit)}  vs  漏抓群 N={len(miss)} 的各 stream 平均 prob')
    print('=' * 64)
    contrast = {}
    for s in has_heads:
        h = float(np.mean([r[f'{s}_prob'] for r in hit])) if hit else 0
        m = float(np.mean([r[f'{s}_prob'] for r in miss])) if miss else 0
        contrast[s] = {'hit_mean': round(h, 4), 'miss_mean': round(m, 4), 'gap': round(h - m, 4)}
        print(f'  {s:8s}  命中 {h:.3f}   漏抓 {m:.3f}   差距 {h-m:+.3f}')

    # ── (B) metadata 交叉分析 ──
    def group_rate(rs, field):
        g = defaultdict(list)
        for r in rs:
            g[r.get(field, '?')].append(r['fusion_prob'])
        out = {}
        for k, ps in g.items():
            n_ai = sum(1 for x in ps if x > THRESH)
            out[k] = {'n': len(ps), 'detect_rate': round(n_ai / len(ps) * 100, 2),
                      'mean_prob': round(float(np.mean(ps)), 4)}
        return dict(sorted(out.items(), key=lambda kv: kv[1]['detect_rate']))

    print('\n' + '=' * 64)
    print('  (B1) 依 格式 (jpg vs png) 的偵測率')
    print('=' * 64)
    fmt = group_rate(valid, 'format')
    for k, v in fmt.items():
        print(f'  {k:6s}  N={v["n"]:4d}  偵測率 {v["detect_rate"]:6.2f}%  平均prob {v["mean_prob"]:.3f}')

    print('\n' + '=' * 64)
    print('  (B2) 依 場景 scene_id 的偵測率（由低到高）')
    print('=' * 64)
    scene = group_rate(valid, 'scene_id')
    for k, v in scene.items():
        print(f'  {k:14s}  N={v["n"]:4d}  偵測率 {v["detect_rate"]:6.2f}%  平均prob {v["mean_prob"]:.3f}')

    print('\n' + '=' * 64)
    print('  (B3) 有人 vs 無人 的偵測率')
    print('=' * 64)
    person = group_rate(valid, 'person')
    for k, v in person.items():
        print(f'  {k:10s}  N={v["n"]:4d}  偵測率 {v["detect_rate"]:6.2f}%  平均prob {v["mean_prob"]:.3f}')

    print('\n' + '=' * 64)
    print('  (B4) 依 region 的偵測率（由低到高, 取頭尾各5）')
    print('=' * 64)
    region = group_rate(valid, 'region')
    items = list(region.items())
    for k, v in items[:5] + items[-5:]:
        print(f'  {k:28s}  N={v["n"]:3d}  偵測率 {v["detect_rate"]:6.2f}%')

    # ── 極端案例 ──
    valid_sorted = sorted(valid, key=lambda r: r['fusion_prob'])
    print('\n' + '=' * 64)
    print('  最容易漏抓（fusion_prob 最低 8 張）')
    print('=' * 64)
    for r in valid_sorted[:8]:
        print(f'  {r["filename"]:26s} prob={r["fusion_prob"]:.3f}  {r["scene_id"]:10s} {r["region"]}')
    print('\n  最自信抓到（fusion_prob 最高 8 張）')
    for r in valid_sorted[-8:][::-1]:
        print(f'  {r["filename"]:26s} prob={r["fusion_prob"]:.3f}  {r["scene_id"]:10s} {r["region"]}')

    report = {
        'n': len(valid),
        'overall_detect_rate': round(len(hit) / len(valid) * 100, 2),
        'per_stream': stream_summary,
        'hit_vs_miss_stream_contrast': contrast,
        'by_format': fmt,
        'by_scene': scene,
        'by_person': person,
        'by_region': region,
        'easiest_to_miss': [{'file': r['filename'], 'prob': r['fusion_prob'],
                             'scene': r['scene_id'], 'region': r['region']} for r in valid_sorted[:15]],
        'most_confident_hits': [{'file': r['filename'], 'prob': r['fusion_prob'],
                                 'scene': r['scene_id'], 'region': r['region']} for r in valid_sorted[-15:][::-1]],
    }
    (OUT_DIR / 'analysis_report.json').write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    print(f'\n[saved] {OUT_DIR / "analysis_report.json"}')
    print(f'[saved] {OUT_DIR / "perstream_full496.csv"}')


if __name__ == '__main__':
    main()
