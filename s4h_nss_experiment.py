"""
EXP-H: NSS (Natural Scene Statistics) Verification Experiment
==============================================================
Tests if adding NSS features improves generalization.
NSS captures physical properties of real photos (1/f spectrum, GGD distribution)
that AI generators cannot replicate regardless of architecture.

Comparison:
  Baseline: 5-stream cached features only
  +NSS:     5-stream + 36-dim NSS features

Key metric: G2 unseen improvement > G1 seen improvement
  → Proves NSS captures generator-agnostic physical rules

Output: outputs/exp_h/
"""
import sys, io, json
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8') if hasattr(sys.stdout, 'buffer') else sys.stdout

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from pathlib import Path
from PIL import Image
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
import pywt
from scipy.stats import gennorm

from config import FEAT_CACHE_DIR, OUTPUTS_DIR, SPLITS_DIR
FEAT_DIR = FEAT_CACHE_DIR
TEST_CSV = SPLITS_DIR / 'test.csv'
CROSS_CSV = SPLITS_DIR / 'cross_generator_test.csv'
OUTPUT_DIR = OUTPUTS_DIR / 'exp_h'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
STREAMS = ['clip', 'fft', 'dct', 'dire', 'noise']
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ── NSS Feature Extraction ──

def compute_mscn(coeff):
    mu = np.mean(coeff)
    sigma = np.std(coeff) + 1e-6
    return (coeff - mu) / sigma

def fit_ggd(data):
    try:
        beta, loc, scale = gennorm.fit(data.flatten(), floc=0)
        return beta, scale
    except:
        return 2.0, 1.0  # fallback to Gaussian

def compute_freq_slope(img):
    fft = np.fft.fft2(img)
    magnitude = np.abs(np.fft.fftshift(fft))
    h, w = magnitude.shape
    cy, cx = h // 2, w // 2
    freqs, energies = [], []
    for r in range(1, min(cy, cx), 5):
        y, x = np.ogrid[:h, :w]
        dist = np.sqrt((x - cx)**2 + (y - cy)**2)
        mask = (dist >= r) & (dist < r + 5)
        if mask.sum() > 0:
            freqs.append(r)
            energies.append(magnitude[mask].mean())
    if len(freqs) < 2:
        return -1.0
    log_f = np.log(np.array(freqs) + 1e-6)
    log_e = np.log(np.array(energies) + 1e-6)
    return np.polyfit(log_f, log_e, 1)[0]

def extract_nss_features(img_gray):
    """Extract ~36-dim NSS features from grayscale image."""
    features = []
    try:
        coeffs = pywt.wavedec2(img_gray, wavelet='db4', level=3)
        for level_coeffs in coeffs[1:]:
            for subband in level_coeffs:
                mscn = compute_mscn(subband)
                alpha, sigma = fit_ggd(mscn)
                features.extend([alpha, sigma])
        # Cross-scale correlation
        for i in range(len(coeffs[1:]) - 1):
            b1 = np.abs(coeffs[i+1][0]).flatten()
            b2 = np.abs(coeffs[i+2][0]).flatten()
            min_len = min(len(b1), len(b2))
            if min_len > 1:
                features.append(np.corrcoef(b1[:min_len], b2[:min_len])[0, 1])
            else:
                features.append(0)
    except:
        features = [0] * 20

    # 1/f slope
    features.append(compute_freq_slope(img_gray))

    # Pad/truncate to fixed size
    target_dim = 36
    if len(features) < target_dim:
        features.extend([0] * (target_dim - len(features)))
    return np.array(features[:target_dim], dtype=np.float32)


def extract_nss_for_split(csv_path, prefix, n_sample=None):
    """Extract NSS features for a CSV split."""
    cache_path = OUTPUT_DIR / f'nss_{prefix}.pt'
    if cache_path.exists():
        print(f"  [Cache] {prefix} NSS features found")
        return torch.load(cache_path, weights_only=False)

    df = pd.read_csv(csv_path)
    if n_sample and len(df) > n_sample:
        df = df.sample(n_sample, random_state=42).reset_index(drop=True)

    print(f"  Extracting NSS for {prefix} ({len(df)} images)...")
    all_feats = []
    for i, row in df.iterrows():
        try:
            img = np.array(Image.open(row['path']).convert('L').resize((256, 256)))
            feat = extract_nss_features(img)
        except:
            feat = np.zeros(36, dtype=np.float32)
        all_feats.append(feat)
        if (i + 1) % 500 == 0:
            print(f"    {i+1}/{len(df)}")

    tensor = torch.tensor(np.stack(all_feats), dtype=torch.float32)
    torch.save(tensor, cache_path)
    return tensor


def main():
    print("=" * 60)
    print("  EXP-H: NSS Verification Experiment")
    print("=" * 60)

    # Load cached stream features
    print("\n[1] Loading cached features...")
    train_feats = torch.cat([torch.load(FEAT_DIR/f"{s}_train_feats.pt", weights_only=False) for s in STREAMS], dim=1)
    train_labels = torch.load(FEAT_DIR/"train_labels.pt", weights_only=False)
    val_feats = torch.cat([torch.load(FEAT_DIR/f"{s}_val_feats.pt", weights_only=False) for s in STREAMS], dim=1)
    val_labels = torch.load(FEAT_DIR/"val_labels.pt", weights_only=False)
    cross_feats = torch.cat([torch.load(FEAT_DIR/f"{s}_cross_gen_test_feats.pt", weights_only=False) for s in STREAMS], dim=1)
    cross_labels = torch.load(FEAT_DIR/"cross_gen_test_labels.pt", weights_only=False)

    # Subsample for speed
    n_train = min(len(train_feats), 10000)
    n_val = min(len(val_feats), 3000)
    n_cross = min(len(cross_feats), 5000)

    torch.manual_seed(42)
    train_idx = torch.randperm(len(train_feats))[:n_train]
    val_idx = torch.randperm(len(val_feats))[:n_val]
    cross_idx = torch.randperm(len(cross_feats))[:n_cross]

    train_feats_sub = train_feats[train_idx]
    train_labels_sub = train_labels[train_idx]
    val_feats_sub = val_feats[val_idx]
    val_labels_sub = val_labels[val_idx]
    cross_feats_sub = cross_feats[cross_idx]
    cross_labels_sub = cross_labels[cross_idx]

    print(f"  Subsampled: train={n_train} val={n_val} cross={n_cross}")

    # Extract NSS features
    print("\n[2] Extracting NSS features...")
    # We need to extract NSS for the same subset - use CSV row indices
    train_df = pd.read_csv(SPLITS_DIR / 'train.csv')
    val_df = pd.read_csv(SPLITS_DIR / 'val.csv')
    cross_df = pd.read_csv(CROSS_CSV)

    def extract_nss_subset(df, indices, name):
        cache = OUTPUT_DIR / f'nss_{name}_sub.pt'
        if cache.exists():
            print(f"  [Cache] {name}")
            return torch.load(cache, weights_only=False)
        feats = []
        subset_df = df.iloc[indices.numpy()] if len(df) > max(indices) else df.iloc[:len(indices)]
        for i, (_, row) in enumerate(subset_df.iterrows()):
            try:
                img = np.array(Image.open(row['path']).convert('L').resize((256, 256)))
                feat = extract_nss_features(img)
            except:
                feat = np.zeros(36, dtype=np.float32)
            feats.append(feat)
            if (i+1) % 500 == 0:
                print(f"    {name}: {i+1}/{len(subset_df)}")
        t = torch.tensor(np.stack(feats), dtype=torch.float32)
        torch.save(t, cache)
        return t

    nss_train = extract_nss_subset(train_df, train_idx, 'train')
    nss_val = extract_nss_subset(val_df, val_idx, 'val')
    nss_cross = extract_nss_subset(cross_df, cross_idx, 'cross')

    # Train and compare
    print("\n[3] Training comparison models...")

    class SimpleClassifier(nn.Module):
        def __init__(self, input_dim):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, 512), nn.ReLU(), nn.Dropout(0.3),
                nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.3),
                nn.Linear(256, 2))
        def forward(self, x):
            return self.net(x)

    def train_and_eval(name, train_x, train_y, val_x, val_y, cross_x, cross_y):
        model = SimpleClassifier(train_x.shape[1]).to(DEVICE)
        opt = optim.AdamW(model.parameters(), lr=1e-3)
        crit = nn.CrossEntropyLoss()
        loader = DataLoader(TensorDataset(train_x, train_y), batch_size=256, shuffle=True)

        best_acc = 0
        for ep in range(20):
            model.train()
            for f, l in loader:
                f, l = f.to(DEVICE), l.to(DEVICE)
                opt.zero_grad(); crit(model(f), l).backward(); opt.step()
            model.eval()
            with torch.no_grad():
                va = accuracy_score(val_y.numpy(), model(val_x.to(DEVICE)).argmax(1).cpu().numpy()) * 100
            if va > best_acc:
                best_acc = va; best_state = {k: v.clone() for k, v in model.state_dict().items()}

        model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            g1_acc = accuracy_score(val_y.numpy(), model(val_x.to(DEVICE)).argmax(1).cpu().numpy()) * 100
            g2_acc = accuracy_score(cross_y.numpy(), model(cross_x.to(DEVICE)).argmax(1).cpu().numpy()) * 100
        print(f"  {name:<25} G1={g1_acc:.2f}%  G2={g2_acc:.2f}%  gap={g2_acc-g1_acc:+.2f}%")
        return {'g1_acc': round(g1_acc, 2), 'g2_acc': round(g2_acc, 2), 'gap': round(g2_acc - g1_acc, 2)}

    results = {}

    # Baseline: 5-stream only
    results['Baseline (5-stream)'] = train_and_eval(
        'Baseline (5-stream)', train_feats_sub, train_labels_sub,
        val_feats_sub, val_labels_sub, cross_feats_sub, cross_labels_sub)

    # NSS only
    results['NSS only (36-dim)'] = train_and_eval(
        'NSS only (36-dim)', nss_train, train_labels_sub,
        nss_val, val_labels_sub, nss_cross, cross_labels_sub)

    # 5-stream + NSS
    train_combined = torch.cat([train_feats_sub, nss_train], dim=1)
    val_combined = torch.cat([val_feats_sub, nss_val], dim=1)
    cross_combined = torch.cat([cross_feats_sub, nss_cross], dim=1)
    results['+NSS (5-stream+36)'] = train_and_eval(
        '+NSS (5-stream+36)', train_combined, train_labels_sub,
        val_combined, val_labels_sub, cross_combined, cross_labels_sub)

    # Save
    with open(OUTPUT_DIR / 'nss_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {OUTPUT_DIR / 'nss_results.json'}")
    print("\n[Done] EXP-H complete.")


if __name__ == '__main__':
    main()
