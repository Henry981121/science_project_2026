"""
EXP-D (GRL): Generalization Test for FusionDetectorGRL
=======================================================
對應 s3_main_grl.py 訓練出的模型進行泛化測試。
架構與 s4d_generalization.py 完全一樣，差別在：
  - MODEL_PATH → outputs/main_grl/best_model.pth
  - FusionDetector 換成 FusionDetectorGRL（含 GRL + gen_discriminator）
  - OUTPUT_DIR  → outputs/exp_d_grl/

Test conditions:
  G1 - Standard test set          (data/splits/test.csv)
  G2 - Cross-generator test set   (data/splits/cross_generator_test.csv)
  G3 - Per-generator breakdown    (from G1)

Outputs:
  outputs/exp_d_grl/generalization_results.json
  outputs/exp_d_grl/per_sample_predictions.csv
"""

import sys

import csv
import json
import warnings
warnings.filterwarnings("ignore")
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torchvision import transforms
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
from config import OUTPUTS_DIR, FEAT_CACHE_DIR, SPLITS_DIR
MODEL_PATH      = OUTPUTS_DIR / 'main_grl' / 'best_model.pth'
DATA_DIR        = SPLITS_DIR.parent
SPLITS_DIR      = SPLITS_DIR
OUTPUT_DIR      = OUTPUTS_DIR / 'exp_d_grl'
FEAT_CACHE_DIR  = FEAT_CACHE_DIR

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

STREAMS = ["clip", "fft", "dct", "dire", "noise"]
device  = "cuda" if torch.cuda.is_available() else "cpu"

GENERATOR_TO_ID = {
    'adm': 0, 'glide': 1, 'sdv4': 2, 'sdv5': 3, 'midjourney': 4,
    'wildfake': 5, 'biggan': 6, 'vqdm': 7, 'wukong': 8, 'firefly': 9, 'real': 10,
    'real_extra': 11, 'wildfake_ddim': 12, 'wildfake_other': 13,
    'stylegan': 14, 'dcgan': 15, 'dcgan_unseen': 16,
    'fursona_gan': 17, 'waifu_gan': 18,
}
REAL_ID = GENERATOR_TO_ID['real']
N_SOURCES = len(GENERATOR_TO_ID)
N_GEN     = N_SOURCES - 2   # 17 個 AI generator（排除 real + real_extra）

EVAL_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


# ---------------------------------------------------------------------------
# Gradient Reversal Layer（inference 時 forward 跟 identity 一樣，不影響結果）
# ---------------------------------------------------------------------------

class _GRL(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.save_for_backward(torch.tensor(lambda_))
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        lambda_ = ctx.saved_tensors[0].item()
        return -lambda_ * grad_output, None


def grad_reverse(x, lambda_: float = 1.0):
    return _GRL.apply(x, lambda_)


# ---------------------------------------------------------------------------
# FusionDetectorGRL（必須與 s3_main_grl.py 完全一致才能載入 weights）
# ---------------------------------------------------------------------------

class CrossAttentionFusionLayer(nn.Module):
    def __init__(self, d_model=512, n_heads=8, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model), nn.Dropout(dropout),
        )

    def forward(self, x):
        attn_out, w = self.self_attn(x, x, x)
        x = self.norm1(x + attn_out)
        x = self.norm2(x + self.ffn(x))
        return x, w


class FusionDetectorGRL(nn.Module):
    def __init__(self, n_streams=5, n_sources=11, n_gen=10,
                 d_model=512, n_heads=8, n_layers=2, dropout=0.3):
        super().__init__()
        self.n_streams = n_streams
        self.layers = nn.ModuleList([
            CrossAttentionFusionLayer(d_model, n_heads, dropout=0.1)
            for _ in range(n_layers)
        ])
        self.stream_embed = nn.Parameter(torch.randn(1, n_streams, d_model) * 0.02)
        self.backbone = nn.Sequential(
            nn.Linear(n_streams * d_model, 1024), nn.LayerNorm(1024), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(1024, 512),                 nn.LayerNorm(512),  nn.GELU(), nn.Dropout(dropout),
        )
        self.head_binary       = nn.Linear(512, 2)
        self.head_source       = nn.Linear(512, n_sources)
        self.gen_discriminator = nn.Sequential(
            nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, n_gen),
        )

    def forward(self, x, grl_lambda: float = 0.0):
        B = x.shape[0]
        tokens = x.view(B, self.n_streams, 512) + self.stream_embed
        attn_weights = None
        for layer in self.layers:
            tokens, attn_weights = layer(tokens)
        shared          = self.backbone(tokens.flatten(1))
        logits_binary   = self.head_binary(shared)
        logits_source   = self.head_source(shared)
        reversed_shared = grad_reverse(shared, grl_lambda)
        logits_gen      = self.gen_discriminator(reversed_shared)
        return logits_binary, logits_source, logits_gen, attn_weights


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model():
    print(f"\n[Model] Loading FusionDetectorGRL from: {MODEL_PATH}")
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}\n"
            "Run s3_main_grl.py first to train the GRL model."
        )
    ckpt = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    state_dict = ckpt.get('model_state_dict', ckpt)
    n_streams  = ckpt.get('n_streams', 5)
    n_sources  = ckpt.get('n_sources', N_SOURCES)
    n_gen      = ckpt.get('n_gen', N_GEN)
    streams    = ckpt.get('streams', STREAMS)
    epoch      = ckpt.get('epoch', '?')
    val_acc    = ckpt.get('val_acc', '?')

    print(f"  Checkpoint : epoch {epoch}, val_acc={val_acc}")
    print(f"  Streams    : {streams} ({n_streams})")
    print(f"  lambda_grl : {ckpt.get('lambda_grl', '?')}")

    model = FusionDetectorGRL(n_streams=n_streams, n_sources=n_sources, n_gen=n_gen)
    model.load_state_dict(state_dict, strict=True)
    model.to(device).eval()
    print(f"  Loaded on {device}.")
    return model, streams, n_streams


# ---------------------------------------------------------------------------
# Feature loading / extraction
# ---------------------------------------------------------------------------

class ImagePathDataset(Dataset):
    def __init__(self, paths, labels, transform=None):
        self.paths, self.labels, self.transform = paths, labels, transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        try:
            img = Image.open(self.paths[idx]).convert('RGB')
        except Exception:
            img = Image.new('RGB', (224, 224), 128)
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx], idx


def extract_features_for_csv(csv_path, cache_prefix):
    cache_dir    = FEAT_CACHE_DIR
    labels_cache = cache_dir / f"{cache_prefix}_labels.pt"
    df           = pd.read_csv(csv_path)
    paths        = df['path'].tolist()
    labels       = df['label'].astype(int).tolist()
    generators   = (df['generator'].astype(str).tolist()
                    if 'generator' in df.columns else ['unknown'] * len(df))

    available = [s for s in STREAMS
                 if (cache_dir / f"{s}_{cache_prefix}_feats.pt").exists()]

    if available and labels_cache.exists():
        print(f"  [Cache] '{cache_prefix}' found ({available}), loading ...")
        feats  = torch.cat([
            torch.load(cache_dir / f"{s}_{cache_prefix}_feats.pt", weights_only=False)
            for s in available
        ], dim=1)
        cached_labels = torch.load(labels_cache, weights_only=False)
        return feats, cached_labels, generators, df, available

    # ── Extract from images ───────────────────────────────────────────
    print(f"  [Extract] '{cache_prefix}' ({len(paths)} images) ...")
    from src.feature_extractors import (
        CLIPFeatureExtractor, FFTFeatureExtractor,
        DCTFeatureExtractor, NoisePrintExtractor,
    )
    extractor_classes = {
        "clip": CLIPFeatureExtractor, "fft": FFTFeatureExtractor,
        "dct":  DCTFeatureExtractor,  "noise": NoisePrintExtractor,
    }
    try:
        from src.feature_extractors import DIREFeatureExtractor
        extractor_classes["dire"] = DIREFeatureExtractor
    except ImportError:
        print("  [WARN] DIRE not available.")

    extractors = {s: extractor_classes[s](device=device).to(device).eval()
                  for s in STREAMS if s in extractor_classes}
    available  = list(extractors.keys())

    ds     = ImagePathDataset(paths, labels, transform=EVAL_TRANSFORM)
    loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=0)
    stream_feats = {s: [] for s in available}
    all_labels   = []

    with torch.no_grad():
        for i, (imgs, lbls, _) in enumerate(loader):
            imgs = imgs.to(device)
            for s in available:
                stream_feats[s].append(extractors[s](imgs).cpu())
            all_labels.extend(lbls.tolist())
            if (i + 1) % 50 == 0:
                print(f"    Batch {i+1}/{len(loader)}")

    labels_tensor = torch.tensor(all_labels, dtype=torch.long)
    torch.save(labels_tensor, labels_cache)
    all_cat = []
    for s in available:
        t = torch.cat(stream_feats[s])
        torch.save(t, cache_dir / f"{s}_{cache_prefix}_feats.pt")
        all_cat.append(t)

    del extractors
    if device == 'cuda':
        torch.cuda.empty_cache()
    return torch.cat(all_cat, dim=1), labels_tensor, generators, df, available


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_inference(model, feats, labels):
    loader = DataLoader(TensorDataset(feats, labels), batch_size=256, shuffle=False)
    all_probs, all_preds, all_labels = [], [], []

    for feat_batch, lbl_batch in loader:
        feat_batch = feat_batch.to(device)
        # grl_lambda=0.0 → GRL 是 identity（inference 時不需要梯度反轉）
        lb, _, _, _ = model(feat_batch, grl_lambda=0.0)
        probs = F.softmax(lb, dim=1)[:, 1]
        preds = lb.argmax(dim=1)
        all_probs.extend(probs.cpu().numpy().tolist())
        all_preds.extend(preds.cpu().numpy().tolist())
        all_labels.extend(lbl_batch.numpy().tolist())

    return np.array(all_probs), np.array(all_preds), np.array(all_labels)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(labels, preds, probs):
    acc = accuracy_score(labels, preds) * 100
    f1  = f1_score(labels, preds, zero_division=0)
    try:
        auc = roc_auc_score(labels, probs) * 100
    except ValueError:
        auc = float('nan')
    return acc, auc, f1


def evaluate_per_generator(probs, preds, labels, generators):
    labels     = np.array(labels)
    probs      = np.array(probs)
    preds      = np.array(preds)
    generators = np.array(generators)
    real_idx   = np.where(labels == 0)[0]
    rng        = np.random.RandomState(42)
    per_gen    = {}

    for gen in sorted(set(g for g, l in zip(generators, labels) if l == 1)):
        fake_idx = np.where((generators == gen) & (labels == 1))[0]
        if len(fake_idx) == 0:
            continue
        real_sampled = rng.choice(real_idx, size=min(len(fake_idx), len(real_idx)), replace=False)
        idx = np.concatenate([fake_idx, real_sampled])
        acc, auc, f1 = compute_metrics(labels[idx], preds[idx], probs[idx])
        per_gen[gen] = {
            'n_fake': int(len(fake_idx)), 'n_real': int(len(real_sampled)),
            'acc': round(float(acc), 2), 'auc': round(float(auc), 2), 'f1': round(float(f1), 4),
        }
    return per_gen


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("  EXP-D (GRL): Generalization Test — FusionDetectorGRL")
    print("=" * 70)

    model, model_streams, n_streams = load_model()
    results      = {}
    all_pred_rows = []

    # ── G1: Standard ─────────────────────────────────────────────────
    print("\n[G1] Standard test set")
    g1_csv = SPLITS_DIR / 'test.csv'
    if not g1_csv.exists():
        print(f"  [ERROR] {g1_csv} not found.")
        sys.exit(1)

    g1_feats, g1_labels, g1_gens, g1_df, _ = extract_features_for_csv(g1_csv, 'test')
    probs, preds, labels = run_inference(model, g1_feats, g1_labels)
    acc, auc, f1 = compute_metrics(labels, preds, probs)
    results['G1'] = {'description': 'Standard', 'n_samples': len(labels),
                     'acc': round(float(acc), 2), 'auc': round(float(auc), 2),
                     'f1': round(float(f1), 4)}
    print(f"  Acc={acc:.1f}%  AUC={auc:.1f}%  F1={f1:.4f}")

    g1_paths = g1_df['path'].tolist()
    for i in range(len(labels)):
        all_pred_rows.append({
            'test_set': 'G1', 'path': g1_paths[i], 'label': int(labels[i]),
            'prob_fake': round(float(probs[i]), 6), 'pred': int(preds[i]),
            'correct': int(labels[i] == preds[i]), 'generator': g1_gens[i],
        })

    # ── G2: Cross-generator (Midjourney) ─────────────────────────────
    print("\n[G2] Cross-generator test set (Midjourney, unseen)")
    g2_csv = SPLITS_DIR / 'cross_generator_test.csv'
    if g2_csv.exists():
        g2_feats, g2_labels, g2_gens, g2_df, _ = extract_features_for_csv(g2_csv, 'cross_gen_test')
        probs, preds, labels = run_inference(model, g2_feats, g2_labels)
        acc, auc, f1 = compute_metrics(labels, preds, probs)
        results['G2'] = {'description': 'Cross-generator (Midjourney)', 'n_samples': len(labels),
                         'acc': round(float(acc), 2), 'auc': round(float(auc), 2),
                         'f1': round(float(f1), 4)}
        print(f"  Acc={acc:.1f}%  AUC={auc:.1f}%  F1={f1:.4f}")

        g2_paths = g2_df['path'].tolist()
        for i in range(len(labels)):
            all_pred_rows.append({
                'test_set': 'G2', 'path': g2_paths[i], 'label': int(labels[i]),
                'prob_fake': round(float(probs[i]), 6), 'pred': int(preds[i]),
                'correct': int(labels[i] == preds[i]), 'generator': g2_gens[i],
            })
    else:
        print(f"  [SKIP] {g2_csv} not found.")

    # ── G3: Per-generator breakdown ───────────────────────────────────
    print("\n[G3] Per-generator breakdown (from G1)")
    g1_probs, g1_preds, g1_labels_arr = run_inference(model, g1_feats, g1_labels)
    per_gen = evaluate_per_generator(g1_probs, g1_preds, g1_labels_arr, g1_gens)
    print(f"  {'Generator':<20} {'N_fake':>8} {'Acc%':>8} {'AUC%':>8} {'F1':>8}")
    print(f"  {'-'*50}")
    for gen, r in sorted(per_gen.items()):
        print(f"  {gen:<20} {r['n_fake']:>8} {r['acc']:>8.1f} {r['auc']:>8.1f} {r['f1']:>8.4f}")
    results['G3_per_generator'] = per_gen

    # ── Gap vs G1 ────────────────────────────────────────────────────
    g1_acc = results['G1']['acc']
    if 'G2' in results:
        results['G2']['gap_acc_vs_g1'] = round(results['G2']['acc'] - g1_acc, 2)

    # ── Summary ──────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  GENERALIZATION RESULTS (GRL Model)")
    print("=" * 70)
    print(f"  {'Test Set':<28} | {'N':>8} | {'Acc%':>7} | {'AUC%':>7} | {'F1':>7} | Gap vs G1")
    print(f"  {'-'*70}")
    for key in ['G1', 'G2']:
        if key not in results:
            continue
        r   = results[key]
        gap = '—' if key == 'G1' else f"{r.get('gap_acc_vs_g1', 0):+.1f}pp"
        print(f"  {r['description']:<28} | {r['n_samples']:>8,} | "
              f"{r['acc']:>7.1f} | {r['auc']:>7.1f} | {r['f1']:>7.4f} | {gap}")
    print(f"  {'─'*70}")

    # ── Save ─────────────────────────────────────────────────────────
    out_json = OUTPUT_DIR / 'generalization_results.json'
    with open(out_json, 'w', encoding='utf-8') as fh:
        json.dump(results, fh, indent=2, ensure_ascii=False)
    print(f"\n[Saved] {out_json}")

    if all_pred_rows:
        out_csv = OUTPUT_DIR / 'per_sample_predictions.csv'
        fieldnames = ['test_set', 'path', 'label', 'prob_fake', 'pred', 'correct', 'generator']
        with open(out_csv, 'w', newline='', encoding='utf-8') as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_pred_rows)
        print(f"[Saved] {out_csv}")

    print("\n[Done] EXP-D (GRL) complete.")


if __name__ == '__main__':
    main()
