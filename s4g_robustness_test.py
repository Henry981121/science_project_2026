"""
EXP-G: Robustness Test
========================
Tests model performance under various image degradations.
Uses saved extractor weights to extract features from degraded images,
then evaluates with the trained GRL model.

Degradation types:
  1. JPEG compression (quality 30, 50, 70)
  2. Resize down+up (scale 0.25, 0.5, 0.75)
  3. Gaussian blur (sigma 1, 2, 3)
  4. Gaussian noise (sigma 10, 25, 50)

Output: outputs/exp_g/robustness_results.json + chart
"""
import sys

import io
import json
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image, ImageFilter
from torchvision import transforms
from sklearn.metrics import accuracy_score, roc_auc_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
from config import OUTPUTS_DIR, FEAT_CACHE_DIR, SPLITS_DIR
MODEL_PATH = OUTPUTS_DIR / 'main_grl' / 'best_model.pth'
FEAT_DIR = FEAT_CACHE_DIR
TEST_CSV = SPLITS_DIR / 'test.csv'
OUTPUT_DIR = OUTPUTS_DIR / 'exp_g'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

STREAMS = ['clip', 'fft', 'dct', 'dire', 'noise']
N_SAMPLE = 2000  # Test on subset for speed

EVAL_TF = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


# ── Degradation functions ──

def degrade_jpeg(img, quality):
    buf = io.BytesIO()
    img.save(buf, format='JPEG', quality=quality)
    buf.seek(0)
    return Image.open(buf).convert('RGB')

def degrade_resize(img, scale):
    w, h = img.size
    small = img.resize((max(int(w*scale), 1), max(int(h*scale), 1)), Image.BILINEAR)
    return small.resize((w, h), Image.BILINEAR)

def degrade_blur(img, sigma):
    return img.filter(ImageFilter.GaussianBlur(radius=sigma))

def degrade_noise(img, sigma):
    arr = np.array(img).astype(np.float32)
    noise = np.random.RandomState(42).randn(*arr.shape) * sigma
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


DEGRADATIONS = {
    'Original': lambda img: img,
    'JPEG q=70': lambda img: degrade_jpeg(img, 70),
    'JPEG q=50': lambda img: degrade_jpeg(img, 50),
    'JPEG q=30': lambda img: degrade_jpeg(img, 30),
    'Resize 0.75x': lambda img: degrade_resize(img, 0.75),
    'Resize 0.50x': lambda img: degrade_resize(img, 0.50),
    'Resize 0.25x': lambda img: degrade_resize(img, 0.25),
    'Blur s=1': lambda img: degrade_blur(img, 1),
    'Blur s=2': lambda img: degrade_blur(img, 2),
    'Blur s=3': lambda img: degrade_blur(img, 3),
    'Noise s=10': lambda img: degrade_noise(img, 10),
    'Noise s=25': lambda img: degrade_noise(img, 25),
    'Noise s=50': lambda img: degrade_noise(img, 50),
}


def main():
    print("=" * 60)
    print("  EXP-G: Robustness Test")
    print("=" * 60)

    # Load extractors with saved weights
    print("\nLoading extractors...")
    from src.feature_extractors import (
        CLIPFeatureExtractor, FFTFeatureExtractor,
        DCTFeatureExtractor, DIREFeatureExtractor, NoisePrintExtractor,
    )
    ext_classes = {
        'clip': CLIPFeatureExtractor, 'fft': FFTFeatureExtractor,
        'dct': DCTFeatureExtractor, 'dire': DIREFeatureExtractor,
        'noise': NoisePrintExtractor,
    }
    extractors = {}
    for s, cls in ext_classes.items():
        ext = cls(device='cpu')
        wp = FEAT_DIR / f"{s}_extractor.pth"
        if wp.exists():
            ext.load_state_dict(torch.load(wp, map_location='cpu', weights_only=False), strict=False)
        ext.to(DEVICE).eval()
        extractors[s] = ext

    # Load GRL model
    print("Loading GRL model...")
    from s3_main_grl import FusionDetectorGRL
    ckpt = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
    model = FusionDetectorGRL(
        n_streams=ckpt['n_streams'], n_sources=ckpt['n_sources'], n_gen=ckpt['n_gen'])
    model.load_state_dict(ckpt['model_state_dict'])
    model.to(DEVICE).eval()

    # Load test images (subset for speed)
    print(f"Loading test images (n={N_SAMPLE})...")
    df = pd.read_csv(TEST_CSV).sample(N_SAMPLE, random_state=42).reset_index(drop=True)
    paths = df['path'].tolist()
    labels = df['label'].astype(int).tolist()

    # Run each degradation
    results = {}
    print(f"\nTesting {len(DEGRADATIONS)} degradation levels...")
    print(f"{'Degradation':<20} {'Acc%':>8} {'AUC%':>8} {'Drop':>8}")
    print("-" * 48)

    baseline_acc = None

    for deg_name, deg_fn in DEGRADATIONS.items():
        all_probs, all_preds = [], []

        with torch.no_grad():
            for i, (path, label) in enumerate(zip(paths, labels)):
                try:
                    img = Image.open(path).convert('RGB')
                    img = deg_fn(img)  # Apply degradation
                except:
                    img = Image.new('RGB', (224, 224), 128)

                img_t = EVAL_TF(img).unsqueeze(0).to(DEVICE)

                # Extract features
                feats = []
                for s in STREAMS:
                    feats.append(extractors[s].extract_features(img_t))
                fused = torch.cat(feats, dim=1)

                # Predict
                lb, _, _, _ = model(fused, grl_lambda=0)
                prob = F.softmax(lb, dim=1)[0, 1].item()
                pred = lb.argmax(1).item()
                all_probs.append(prob)
                all_preds.append(pred)

                if (i+1) % 500 == 0:
                    print(f"  [{deg_name}] {i+1}/{N_SAMPLE}", end='\r')

        labels_np = np.array(labels)
        preds_np = np.array(all_preds)
        probs_np = np.array(all_probs)

        acc = accuracy_score(labels_np, preds_np) * 100
        try:
            auc = roc_auc_score(labels_np, probs_np) * 100
        except:
            auc = 0

        if baseline_acc is None:
            baseline_acc = acc
            drop = 0
        else:
            drop = acc - baseline_acc

        results[deg_name] = {'acc': round(acc, 2), 'auc': round(auc, 2), 'drop': round(drop, 2)}
        print(f"{deg_name:<20} {acc:>7.2f}% {auc:>7.2f}% {drop:>+7.2f}%")

    # Save results
    with open(OUTPUT_DIR / 'robustness_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    # ── Plot ──
    names = list(results.keys())
    accs = [results[n]['acc'] for n in names]

    # Group by type
    colors = []
    for n in names:
        if 'JPEG' in n: colors.append('#4C72B0')
        elif 'Resize' in n: colors.append('#55A868')
        elif 'Blur' in n: colors.append('#C44E52')
        elif 'Noise' in n: colors.append('#8172B2')
        else: colors.append('#E8770E')

    fig, ax = plt.subplots(figsize=(14, 6))
    bars = ax.bar(range(len(names)), accs, color=colors, edgecolor='black', linewidth=0.5)
    for bar, val in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{val:.1f}', ha='center', va='bottom', fontsize=8, fontweight='bold')

    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title('Robustness Test: Model Accuracy Under Image Degradation', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 110)
    ax.axhline(y=baseline_acc, color='orange', linestyle='--', alpha=0.5, label=f'Baseline ({baseline_acc:.1f}%)')
    ax.axhline(y=50, color='red', linestyle='--', alpha=0.3, label='Random (50%)')
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)

    # Legend for colors
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#E8770E', label='Original'),
        Patch(facecolor='#4C72B0', label='JPEG Compression'),
        Patch(facecolor='#55A868', label='Resize'),
        Patch(facecolor='#C44E52', label='Gaussian Blur'),
        Patch(facecolor='#8172B2', label='Gaussian Noise'),
    ]
    ax.legend(handles=legend_elements, loc='lower left', fontsize=9)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'robustness_chart.png', dpi=200)
    plt.close()
    print(f"\nSaved: {OUTPUT_DIR / 'robustness_chart.png'}")
    print(f"Saved: {OUTPUT_DIR / 'robustness_results.json'}")
    print("\n[Done] EXP-G complete.")


if __name__ == '__main__':
    main()
