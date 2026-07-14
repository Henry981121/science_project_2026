"""
day_dauc_iauc.py
=================
Method 1 (Grad-CAM 忠實度): DAUC / IAUC test.

對每張測試圖，從 4 條 CNN-style Grad-CAM 流（CLIP via Chefer relevance,
FFT/DCT/DIRE via Grad-CAM）各自取熱圖，做：

  - Deletion (DAUC):
      從原圖開始，依熱圖最高分逐步塗黑像素，記錄主模型 fake-prob 下降。
      AUC 越「低」 → 熱圖越忠實（被塗黑的真的是模型靠的）。
  - Insertion (IAUC):
      從黑圖開始，依熱圖最高分逐步還原像素，記錄主模型 fake-prob 上升。
      AUC 越「高」 → 熱圖越忠實。

  Faithfulness gap = IAUC − DAUC（越大越好）。

排除 Noise 流：Noise 已改用 SRM 殘差圖，非 Grad-CAM，跳過。

用法:
    python day_dauc_iauc.py --n-images 30 --steps 10 --out outputs/dauc_iauc.csv
"""

import os, sys, argparse, time, json
os.environ.setdefault('PYTHONIOENCODING', 'utf-8')
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
sys.path.insert(0, '.')

from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from config import OUTPUTS_DIR, FEAT_CACHE_DIR, TEST_CSV, CROSS_CSV
from src.feature_extractors import (
    CLIPFeatureExtractor, FFTFeatureExtractor,
    DCTFeatureExtractor, DIREFeatureExtractor, NoisePrintExtractor,
)
from src.xai.per_stream_gradcam import PerStreamExplainer
from s3_main_grl import FusionDetectorGRL


# 排除 Noise（SRM 殘差圖，非 Grad-CAM）
TESTED_STREAMS = ['clip', 'fft', 'dct', 'dire']
EVAL_TF = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


class LinearHead(nn.Module):
    def __init__(self, in_dim=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 2),
        )
    def forward(self, x): return self.net(x)


def load_all(device):
    """Load 5 extractors + 5 heads + fusion model."""
    print("[load] extractors ...")
    ext_classes = {
        'clip':  CLIPFeatureExtractor, 'fft': FFTFeatureExtractor,
        'dct':   DCTFeatureExtractor,  'dire': DIREFeatureExtractor,
        'noise': NoisePrintExtractor,
    }
    extractors = {}
    for s, cls in ext_classes.items():
        ext = cls(device='cpu')
        wp = FEAT_CACHE_DIR / f"{s}_extractor.pth"
        if wp.exists():
            ext.load_state_dict(torch.load(wp, map_location='cpu', weights_only=False),
                                strict=False)
        ext.to(device)
        for p in ext.parameters():
            p.requires_grad_(True)
        extractors[s] = ext

    print("[load] heads ...")
    heads = {}
    for s in ['clip', 'fft', 'dct', 'dire', 'noise']:
        hp = OUTPUTS_DIR / 'exp_a' / s / 'best_model.pth'
        if not hp.exists():
            continue
        h = LinearHead().to(device)
        h.load_state_dict(torch.load(hp, weights_only=False))
        for p in h.parameters():
            p.requires_grad_(True)
        heads[s] = h

    print("[load] fusion model ...")
    ck = torch.load(OUTPUTS_DIR / 'main_grl' / 'best_model.pth',
                    map_location=device, weights_only=False)
    fusion = FusionDetectorGRL(
        n_streams=ck['n_streams'], n_sources=ck['n_sources'], n_gen=ck['n_gen']
    ).to(device).eval()
    fusion.load_state_dict(ck['model_state_dict'])
    return extractors, heads, fusion


@torch.no_grad()
def predict_fake_prob(extractors, fusion, img_tensor, device):
    """End-to-end: image -> 5 stream feats -> fusion -> fake_prob."""
    feats = []
    for s in ['clip', 'fft', 'dct', 'dire', 'noise']:
        f = extractors[s].extract_features(img_tensor)
        feats.append(f)
    cat = torch.cat(feats, dim=1)
    lb, _, _, _ = fusion(cat, grl_lambda=0.0)
    prob = F.softmax(lb, dim=1)[0, 1].item()
    return prob


def upsample_cam(cam_2d, target_size=224):
    """Upsample CAM to image resolution."""
    t = torch.tensor(cam_2d).float().unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
    up = F.interpolate(t, size=(target_size, target_size),
                       mode='bilinear', align_corners=False)
    return up.squeeze().numpy()


def dauc_iauc(extractors, fusion, img_tensor, cam_2d, device, steps=10):
    """
    DAUC: blacken top-K pixels progressively, record fake_prob.
    IAUC: restore top-K pixels progressively from black, record fake_prob.
    Returns (dauc, iauc, deletion_scores, insertion_scores).
    """
    H, W = img_tensor.shape[-2:]
    assert cam_2d.shape == (H, W), f"CAM {cam_2d.shape} vs img {(H, W)}"

    flat = cam_2d.flatten()
    sorted_idx = np.argsort(flat)[::-1]  # 高分在前
    total = H * W
    step_size = total // steps

    # Pre-compute (r, c) for each sorted index
    rs = sorted_idx // W
    cs = sorted_idx % W

    # --- Deletion ---
    temp = img_tensor.clone()
    del_scores = []
    for i in range(steps + 1):
        del_scores.append(predict_fake_prob(extractors, fusion, temp, device))
        if i < steps:
            start, end = i * step_size, (i + 1) * step_size
            temp[0, :, rs[start:end], cs[start:end]] = 0.0

    # --- Insertion ---
    temp = torch.zeros_like(img_tensor)
    ins_scores = []
    for i in range(steps + 1):
        ins_scores.append(predict_fake_prob(extractors, fusion, temp, device))
        if i < steps:
            start, end = i * step_size, (i + 1) * step_size
            temp[0, :, rs[start:end], cs[start:end]] = img_tensor[0, :, rs[start:end], cs[start:end]]

    # Trapezoidal AUC (x in [0, 1])
    dauc = float(np.trapz(del_scores, dx=1.0 / steps))
    iauc = float(np.trapz(ins_scores, dx=1.0 / steps))
    return dauc, iauc, del_scores, ins_scores


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n-images', type=int, default=30,
                    help='Number of fake images to test')
    ap.add_argument('--steps', type=int, default=10,
                    help='Number of perturbation steps')
    ap.add_argument('--out', default='outputs/dauc_iauc.csv',
                    help='Output CSV')
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--min-prob', type=float, default=0.0,
                    help='Only include images with baseline fake_prob >= this (filter out saturated)')
    ap.add_argument('--max-prob', type=float, default=1.0,
                    help='Only include images with baseline fake_prob <= this')
    ap.add_argument('--pool-size', type=int, default=200,
                    help='Initial pool to sample from before filtering by prob')
    ap.add_argument('--csv', default='cross', choices=['test', 'cross'],
                    help='test = G1 (saturated probs), cross = G2 (more variation, default)')
    args = ap.parse_args()

    device = args.device if torch.cuda.is_available() else 'cpu'
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  DAUC / IAUC — Grad-CAM faithfulness test")
    print(f"  Device : {device}")
    print(f"  Images : {args.n_images}  Steps : {args.steps}")
    print(f"  Streams tested: {TESTED_STREAMS}  (Noise excluded — SRM residual)")
    print("=" * 60)

    extractors, heads, fusion = load_all(device)
    explainer = PerStreamExplainer(extractors, heads, device=device)

    csv_path = CROSS_CSV if args.csv == 'cross' else TEST_CSV
    print(f"\n[csv] Using {args.csv.upper()}: {csv_path}")
    df = pd.read_csv(csv_path)
    # Sample large pool of fake images, then filter by baseline prob
    pool_n = min(args.pool_size, (df['label'] == 1).sum())
    pool = df[df['label'] == 1].sample(pool_n, random_state=args.seed).reset_index(drop=True)
    print(f"\n[pool] Sampled {pool_n} fake images, filtering by baseline prob "
          f"[{args.min_prob:.2f}, {args.max_prob:.2f}] ...")

    # Pre-compute baseline probs for the pool, then filter
    rows_kept = []
    for _, row in pool.iterrows():
        try:
            img = Image.open(row['path']).convert('RGB')
            img_t = EVAL_TF(img).unsqueeze(0).to(device)
            p = predict_fake_prob(extractors, fusion, img_t, device)
        except Exception:
            continue
        if args.min_prob <= p <= args.max_prob:
            rows_kept.append({'path': row['path'], 'generator': row['generator'],
                              'label': row['label'], 'base_prob': p})
        if len(rows_kept) >= args.n_images:
            break

    if len(rows_kept) < args.n_images:
        print(f"  WARN: only {len(rows_kept)} images matched filter (wanted {args.n_images})")
    fake_df = pd.DataFrame(rows_kept).reset_index(drop=True)
    if len(fake_df) == 0:
        print("  ERROR: 0 images after filter. Try widening --min-prob / --max-prob.")
        return
    print(f"[sample] {len(fake_df)} fake images kept after filter")
    print(f"         generator distribution: {dict(fake_df['generator'].value_counts())}")
    print(f"         base_prob range: [{fake_df['base_prob'].min():.3f}, {fake_df['base_prob'].max():.3f}]")

    rows = []
    t0 = time.time()
    for idx, row in fake_df.iterrows():
        path = row['path']
        try:
            img_pil = Image.open(path).convert('RGB')
        except Exception as e:
            print(f"  [{idx}] skip: {e}")
            continue
        img_tensor = EVAL_TF(img_pil).unsqueeze(0).to(device)

        # Baseline prob
        base_prob = predict_fake_prob(extractors, fusion, img_tensor, device)

        # Get all 5 stream CAMs (we'll only test 4)
        try:
            exps = explainer.explain_image(img_tensor)
        except Exception as e:
            print(f"  [{idx}] CAM extraction failed: {e}")
            continue

        per_stream = {'image_id': idx, 'path': path, 'generator': row['generator'],
                      'baseline_prob': base_prob}

        for s in TESTED_STREAMS:
            if s not in exps:
                per_stream[f'{s}_dauc'] = None
                per_stream[f'{s}_iauc'] = None
                continue
            cam = exps[s]
            cam_up = upsample_cam(cam, target_size=224)
            dauc, iauc, _, _ = dauc_iauc(extractors, fusion, img_tensor, cam_up,
                                          device, steps=args.steps)
            per_stream[f'{s}_dauc'] = dauc
            per_stream[f'{s}_iauc'] = iauc
            per_stream[f'{s}_gap']  = iauc - dauc

        rows.append(per_stream)
        elapsed = time.time() - t0
        eta = elapsed / (idx + 1) * (len(fake_df) - idx - 1)
        print(f"  [{idx+1:>3}/{len(fake_df)}] base={base_prob:.3f}  "
              f"CLIP gap={per_stream.get('clip_gap', 0):.3f}  "
              f"FFT gap={per_stream.get('fft_gap', 0):.3f}  "
              f"DCT gap={per_stream.get('dct_gap', 0):.3f}  "
              f"DIRE gap={per_stream.get('dire_gap', 0):.3f}  "
              f"(ETA {eta/60:.1f}min)")

    out_df = pd.DataFrame(rows)
    out_df.to_csv(out_path, index=False)
    print(f"\n[saved] {out_path}  ({len(out_df)} rows)")

    # Summary table
    print("\n" + "=" * 60)
    print("  Summary: Mean ± Std across all images")
    print("=" * 60)
    print(f"  {'Stream':<8} {'DAUC (lower=better)':>22} {'IAUC (higher=better)':>22} {'Gap I-D (higher=better)':>26}")
    print(f"  {'-'*60}")
    for s in TESTED_STREAMS:
        dc = out_df[f'{s}_dauc'].dropna()
        ic = out_df[f'{s}_iauc'].dropna()
        gap = ic - dc
        print(f"  {s:<8} {dc.mean():>8.4f} ± {dc.std():.4f}  "
              f"{ic.mean():>8.4f} ± {ic.std():.4f}  "
              f"{gap.mean():>8.4f} ± {gap.std():.4f}")
    print("\n  Interpretation:")
    print("  - Lower DAUC: blackening high-score pixels makes fake_prob drop faster -> faithful")
    print("  - Higher IAUC: restoring high-score pixels lifts fake_prob faster -> faithful")
    print("  - Gap = IAUC - DAUC, larger = more faithful")

    # Save summary JSON
    summary = {
        'config': vars(args),
        'streams_tested': TESTED_STREAMS,
        'streams_excluded': ['noise (SRM residual, not Grad-CAM)'],
        'n_images_evaluated': len(out_df),
        'per_stream': {}
    }
    for s in TESTED_STREAMS:
        dc = out_df[f'{s}_dauc'].dropna()
        ic = out_df[f'{s}_iauc'].dropna()
        gap = ic - dc
        summary['per_stream'][s] = {
            'dauc_mean': float(dc.mean()), 'dauc_std': float(dc.std()),
            'iauc_mean': float(ic.mean()), 'iauc_std': float(ic.std()),
            'gap_mean':  float(gap.mean()), 'gap_std': float(gap.std()),
            'n_valid':   int(len(dc)),
        }
    json_path = out_path.with_suffix('.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\n[saved] {json_path}")


if __name__ == '__main__':
    main()
