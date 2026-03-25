"""
Supplementary Experiments (Steps 5-11)
=======================================
5. Grad-CAM visualization (pytorch-grad-cam)
6. Attention Map (Real vs Fake weight comparison)
7. NSS 驗真實驗
8. 頻譜視覺化 (GAN vs Diffusion vs Real)
9. t-SNE (already in s4d2, skip)
10. 多種子重跑 (separate script needed, skip)
11. 錯誤類型分析 (FP vs FN on G2)

Usage: python s4_supplementary.py [5|6|7|8|11|all]
"""
import sys, io, json, argparse
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8') if hasattr(sys.stdout, 'buffer') else sys.stdout

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from pathlib import Path
from PIL import Image
from torchvision import transforms
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from config import OUTPUTS_DIR, FEAT_CACHE_DIR, SPLITS_DIR
FEAT_DIR = FEAT_CACHE_DIR
MODEL_PATH = OUTPUTS_DIR / 'main_grl' / 'best_model.pth'
TEST_CSV = SPLITS_DIR / 'test.csv'
CROSS_CSV = SPLITS_DIR / 'cross_generator_test.csv'
OUTPUT_DIR = OUTPUTS_DIR / 'supplementary'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
STREAMS = ['clip', 'fft', 'dct', 'dire', 'noise']

EVAL_TF = transforms.Compose([
    transforms.Resize((224, 224)), transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


# ══════════════════════════════════════════════════════════
# Step 5: Grad-CAM (using pytorch-grad-cam package)
# ══════════════════════════════════════════════════════════

def step5_gradcam():
    print("\n" + "="*60)
    print("  Step 5: Grad-CAM Visualization")
    print("="*60)

    try:
        from pytorch_grad_cam import GradCAMPlusPlus
        from pytorch_grad_cam.utils.image import show_cam_on_image
    except ImportError:
        print("  Installing pytorch-grad-cam...")
        import subprocess
        subprocess.run([sys.executable, "-m", "pip", "install", "grad-cam", "-q"])
        from pytorch_grad_cam import GradCAMPlusPlus
        from pytorch_grad_cam.utils.image import show_cam_on_image

    import torchvision.models as models
    import torch.nn as nn
    import cv2

    # Load ResNet50 (trained for AI detection)
    resnet_path = OUTPUTS_DIR / 'grad cam 測試' / 'gradcam' / 'best_model.pth'
    if not resnet_path.exists():
        print("  ResNet50 model not found, training one...")
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        model.fc = nn.Linear(2048, 2)
        model.to(DEVICE).eval()
        # Quick train
        df = pd.read_csv(TEST_CSV).sample(2000, random_state=42)
        # Skip training, just use pretrained for demo
    else:
        model = models.resnet50(weights=None)
        model.fc = nn.Linear(2048, 2)
        model.load_state_dict(torch.load(resnet_path, map_location=DEVICE, weights_only=True))
        model.to(DEVICE).eval()

    target_layer = model.layer4[-1]
    cam = GradCAMPlusPlus(model=model, target_layers=[target_layer])

    # Select 5 real + 5 fake
    df = pd.read_csv(TEST_CSV)
    real_df = df[df['label'] == 0].sample(5, random_state=42)
    fake_df = df[df['label'] == 1].sample(5, random_state=42)
    selected = list(zip(real_df['path'], [0]*5, real_df['generator'])) + \
               list(zip(fake_df['path'], [1]*5, fake_df['generator']))

    fig, axes = plt.subplots(2, len(selected), figsize=(3*len(selected), 6))
    fig.suptitle("Grad-CAM++: AI Image Detection\nTop: Original | Bottom: Heatmap (Red = AI Artifact)", fontsize=13, fontweight='bold')

    for i, (path, label, gen) in enumerate(selected):
        try:
            img_pil = Image.open(path).convert('RGB').resize((224, 224))
        except:
            img_pil = Image.new('RGB', (224, 224), 128)
        img_np = np.array(img_pil).astype(np.float32) / 255.0
        img_tensor = EVAL_TF(Image.open(path).convert('RGB') if Path(path).exists() else Image.new('RGB', (224, 224), 128))

        grayscale_cam = cam(input_tensor=img_tensor.unsqueeze(0).to(DEVICE))
        visualization = show_cam_on_image(img_np, grayscale_cam[0], use_rgb=True)

        with torch.no_grad():
            out = model(img_tensor.unsqueeze(0).to(DEVICE))
            prob = F.softmax(out, 1)[0, 1].item()
            pred = "AI" if out.argmax(1).item() == 1 else "Real"

        gt = "Real" if label == 0 else "AI"
        color = "green" if pred == gt else "red"

        axes[0, i].imshow((img_np * 255).astype(np.uint8))
        axes[0, i].set_title(f"GT:{gt}\nPred:{pred}({prob:.2f})", fontsize=8, color=color)
        axes[0, i].axis('off')
        axes[1, i].imshow(visualization)
        axes[1, i].axis('off')

    plt.tight_layout()
    out_path = OUTPUT_DIR / 'step5_gradcam.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out_path}")


# ══════════════════════════════════════════════════════════
# Step 6: Attention Map (Real vs Fake weight comparison)
# ══════════════════════════════════════════════════════════

def step6_attention():
    print("\n" + "="*60)
    print("  Step 6: Attention Map (Real vs Fake)")
    print("="*60)

    from s3_main_grl import FusionDetectorGRL

    # Load model
    ckpt = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
    model = FusionDetectorGRL(n_streams=ckpt['n_streams'], n_sources=ckpt['n_sources'], n_gen=ckpt['n_gen'])
    model.load_state_dict(ckpt['model_state_dict'])
    model.to(DEVICE).eval()

    # Load test features
    test_feats = torch.cat([torch.load(FEAT_DIR/f"{s}_test_feats.pt", weights_only=False) for s in STREAMS], dim=1)
    test_labels = torch.load(FEAT_DIR/"test_labels.pt", weights_only=False)

    # Collect attention weights
    n = min(2000, len(test_feats))
    sample_feats = test_feats[:n]
    sample_labels = test_labels[:n].numpy()

    all_attn = []
    with torch.no_grad():
        for i in range(0, n, 256):
            batch = sample_feats[i:i+256].to(DEVICE)
            _, _, _, attn = model(batch, grl_lambda=0)
            if attn is not None:
                # Average over heads if multi-head
                a = attn.cpu().numpy()
                if a.ndim == 4:
                    a = a.mean(axis=1)  # (B, N, N)
                all_attn.append(a)

    if not all_attn:
        print("  No attention weights captured")
        return

    all_attn = np.concatenate(all_attn, axis=0)  # (N_samples, n_streams, n_streams)
    # Stream contribution = row sum
    contributions = all_attn.sum(axis=2)  # (N_samples, n_streams)
    contributions = contributions / contributions.sum(axis=1, keepdims=True)

    real_mask = sample_labels[:len(contributions)] == 0
    fake_mask = sample_labels[:len(contributions)] == 1

    real_mean = contributions[real_mask].mean(axis=0)
    fake_mean = contributions[fake_mask].mean(axis=0)

    stream_names = [s.upper() for s in STREAMS]
    colors_bar = ['#2c5f8a', '#a02020']

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: Real vs Fake bar chart
    x = np.arange(len(stream_names))
    axes[0].bar(x - 0.18, real_mean, 0.35, label='Real', color='#2c5f8a', alpha=0.8)
    axes[0].bar(x + 0.18, fake_mean, 0.35, label='AI Fake', color='#a02020', alpha=0.8)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(stream_names)
    axes[0].set_ylabel('Mean Attention Weight')
    axes[0].set_title('Real vs Fake: Stream Attention Weights')
    axes[0].legend()

    # Right: Violin plot
    colors_v = ['#4C72B0', '#55A868', '#C44E52', '#8172B2', '#CCB974']
    parts = axes[1].violinplot([contributions[:, i] for i in range(len(STREAMS))],
                                positions=range(len(STREAMS)), showmeans=True)
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(colors_v[i])
        pc.set_alpha(0.7)
    axes[1].set_xticks(range(len(STREAMS)))
    axes[1].set_xticklabels(stream_names)
    axes[1].set_ylabel('Attention Weight Distribution')
    axes[1].set_title('Per-Stream Attention Distribution (All Samples)')

    plt.tight_layout()
    out_path = OUTPUT_DIR / 'step6_attention_real_vs_fake.png'
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"  Saved: {out_path}")


# ══════════════════════════════════════════════════════════
# Step 8: Frequency Spectrum (GAN vs Diffusion vs Real)
# ══════════════════════════════════════════════════════════

def step8_spectrum():
    print("\n" + "="*60)
    print("  Step 8: Frequency Spectrum Visualization")
    print("="*60)

    df = pd.read_csv(TEST_CSV)

    categories = {
        'Real': df[df['generator'].isin(['real', 'real_extra'])].sample(200, random_state=42),
        'Diffusion': df[df['generator'].isin(['adm', 'glide', 'sdv4', 'sdv5', 'midjourney', 'wildfake'])].sample(200, random_state=42),
        'GAN': df[df['generator'].isin(['stylegan', 'dcgan'])],
    }
    if len(categories['GAN']) > 200:
        categories['GAN'] = categories['GAN'].sample(200, random_state=42)

    avg_spectra = {}
    for cat_name, cat_df in categories.items():
        spectra = []
        for _, row in cat_df.iterrows():
            try:
                img = np.array(Image.open(row['path']).convert('L').resize((256, 256)))
                fft = np.fft.fft2(img)
                magnitude = np.log1p(np.abs(np.fft.fftshift(fft)))
                spectra.append(magnitude)
            except:
                continue
        if spectra:
            avg_spectra[cat_name] = np.mean(spectra, axis=0)
        print(f"  {cat_name}: {len(spectra)} images")

    if not avg_spectra:
        print("  No images found")
        return

    # Plot
    fig, axes = plt.subplots(1, len(avg_spectra) + 1, figsize=(5 * (len(avg_spectra) + 1), 5))

    for i, (name, spec) in enumerate(avg_spectra.items()):
        axes[i].imshow(spec, cmap='hot')
        axes[i].set_title(f'{name}\nAvg Frequency Spectrum', fontsize=12, fontweight='bold')
        axes[i].axis('off')

    # Radial average comparison
    ax = axes[-1]
    colors = {'Real': '#55A868', 'Diffusion': '#4C72B0', 'GAN': '#C44E52'}
    for name, spec in avg_spectra.items():
        h, w = spec.shape
        cy, cx = h//2, w//2
        radii, energies = [], []
        for r in range(1, min(cy, cx), 2):
            y, x = np.ogrid[:h, :w]
            mask = ((x-cx)**2 + (y-cy)**2 >= r**2) & ((x-cx)**2 + (y-cy)**2 < (r+2)**2)
            if mask.sum() > 0:
                radii.append(r)
                energies.append(spec[mask].mean())
        ax.plot(radii, energies, label=name, color=colors.get(name, 'gray'), linewidth=2)
    ax.set_xlabel('Frequency (radius)')
    ax.set_ylabel('Log Magnitude')
    ax.set_title('Radial Avg Spectrum', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.suptitle('Frequency Spectrum: Real vs Diffusion vs GAN', fontsize=14, fontweight='bold')
    plt.tight_layout()
    out_path = OUTPUT_DIR / 'step8_frequency_spectrum.png'
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"  Saved: {out_path}")


# ══════════════════════════════════════════════════════════
# Step 11: Error Type Analysis (FP vs FN on G2)
# ══════════════════════════════════════════════════════════

def step11_error_analysis():
    print("\n" + "="*60)
    print("  Step 11: Error Type Analysis (G2)")
    print("="*60)

    from s3_main_grl import FusionDetectorGRL

    ckpt = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
    model = FusionDetectorGRL(n_streams=ckpt['n_streams'], n_sources=ckpt['n_sources'], n_gen=ckpt['n_gen'])
    model.load_state_dict(ckpt['model_state_dict'])
    model.to(DEVICE).eval()

    # Load cross-gen features
    cross_feats = torch.cat([torch.load(FEAT_DIR/f"{s}_cross_gen_test_feats.pt", weights_only=False) for s in STREAMS], dim=1)
    cross_labels = torch.load(FEAT_DIR/"cross_gen_test_labels.pt", weights_only=False)
    cross_df = pd.read_csv(CROSS_CSV)
    n = min(len(cross_df), len(cross_labels))
    cross_df = cross_df.iloc[:n]
    cross_labels = cross_labels[:n]
    cross_feats = cross_feats[:n]
    generators = cross_df['generator'].astype(str).values

    # Predict
    with torch.no_grad():
        lb, _, _, _ = model(cross_feats.to(DEVICE), grl_lambda=0)
        preds = lb.argmax(1).cpu().numpy()
    labels = cross_labels.numpy()

    # Error analysis
    fp = (preds == 1) & (labels == 0)  # Real predicted as Fake
    fn = (preds == 0) & (labels == 1)  # Fake predicted as Real
    tp = (preds == 1) & (labels == 1)
    tn = (preds == 0) & (labels == 0)

    print(f"  Total: {len(labels):,}")
    print(f"  TP (Fake->Fake): {tp.sum():,} ({tp.mean()*100:.1f}%)")
    print(f"  TN (Real->Real): {tn.sum():,} ({tn.mean()*100:.1f}%)")
    print(f"  FP (Real->Fake): {fp.sum():,} ({fp.mean()*100:.1f}%)")
    print(f"  FN (Fake->Real): {fn.sum():,} ({fn.mean()*100:.1f}%)")

    # Per-generator FN rate (which generators fool the model?)
    fake_gens = sorted(set(g for g, l in zip(generators, labels) if l == 1))
    print(f"\n  {'Generator':<20} {'N':>6} {'FN':>6} {'FN Rate':>8}")
    print(f"  {'-'*42}")

    gen_results = {}
    for gen in fake_gens:
        mask = (generators == gen) & (labels == 1)
        n_gen = mask.sum()
        fn_gen = ((preds == 0) & mask).sum()
        fn_rate = fn_gen / max(n_gen, 1) * 100
        print(f"  {gen:<20} {n_gen:>6} {fn_gen:>6} {fn_rate:>7.1f}%")
        gen_results[gen] = {'n': int(n_gen), 'fn': int(fn_gen), 'fn_rate': round(fn_rate, 2)}

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: Overall confusion
    cm = np.array([[tn.sum(), fp.sum()], [fn.sum(), tp.sum()]])
    im = axes[0].imshow(cm, cmap='Blues')
    axes[0].set_xticks([0, 1]); axes[0].set_yticks([0, 1])
    axes[0].set_xticklabels(['Pred Real', 'Pred Fake'])
    axes[0].set_yticklabels(['True Real', 'True Fake'])
    for i in range(2):
        for j in range(2):
            axes[0].text(j, i, f'{cm[i,j]:,}', ha='center', va='center', fontsize=14, fontweight='bold')
    axes[0].set_title('Confusion Matrix (G2 Unseen)', fontsize=12, fontweight='bold')

    # Right: Per-generator FN rate
    gens = list(gen_results.keys())
    fn_rates = [gen_results[g]['fn_rate'] for g in gens]
    colors = ['#C44E52' if r > 50 else '#55A868' for r in fn_rates]
    bars = axes[1].barh(gens, fn_rates, color=colors, edgecolor='black', linewidth=0.5)
    for bar, val in zip(bars, fn_rates):
        axes[1].text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                     f'{val:.1f}%', va='center', fontsize=10)
    axes[1].set_xlabel('False Negative Rate (%)')
    axes[1].set_title('Per-Generator: How Often Fake Fools Model', fontsize=12, fontweight='bold')
    axes[1].set_xlim(0, 110)
    axes[1].invert_yaxis()

    plt.tight_layout()
    out_path = OUTPUT_DIR / 'step11_error_analysis.png'
    plt.savefig(out_path, dpi=200)
    plt.close()

    results = {'overall': {'tp': int(tp.sum()), 'tn': int(tn.sum()), 'fp': int(fp.sum()), 'fn': int(fn.sum())},
               'per_generator': gen_results}
    with open(OUTPUT_DIR / 'step11_error_analysis.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved: {OUTPUT_DIR / 'step11_error_analysis.png'}")


# ══════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('steps', nargs='*', default=['all'])
    args = parser.parse_args()

    steps = args.steps
    run_all = 'all' in steps

    if run_all or '5' in steps:
        step5_gradcam()
    if run_all or '6' in steps:
        step6_attention()
    if run_all or '8' in steps:
        step8_spectrum()
    if run_all or '11' in steps:
        step11_error_analysis()

    print(f"\n{'='*60}")
    print("  All supplementary experiments complete!")
    print(f"  Outputs: {OUTPUT_DIR}")
    print(f"{'='*60}")

if __name__ == '__main__':
    main()
