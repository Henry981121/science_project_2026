"""
Stage 3-C: Fusion Strategy Comparison (EXP-C) — Cached Features
================================================================
Compare 3 fusion strategies using pre-extracted features from s3a:

  Fusion_1: Concat+MLP       — concat N*512 -> MLP -> 2
  Fusion_2: Weighted Fusion   — learned per-stream weights, weighted sum -> MLP -> 2
  Fusion_3: Cross-Attention   — 2-layer Transformer + binary head

Training: 20 epochs each, same hyperparameters, same data splits.
Results saved to: outputs/exp_c/
"""

import sys


import json
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from sklearn.metrics import roc_auc_score, f1_score
import numpy as np

# ── Paths ──────────────────────────────────────────────────────────────
from config import TRAIN_CSV, VAL_CSV, OUTPUTS_DIR, FEAT_CACHE_DIR
TRAIN_CSV      = str(TRAIN_CSV)
VAL_CSV        = str(VAL_CSV)
OUTPUT_DIR     = OUTPUTS_DIR / 'exp_c'
FEAT_CACHE_DIR = FEAT_CACHE_DIR

# ── Hyperparameters ────────────────────────────────────────────────────
EPOCHS     = 20
BATCH_SIZE = 256
LR         = 1e-3

STREAMS = ["clip", "fft", "dct", "dire", "noise"]

device = "cuda" if torch.cuda.is_available() else "cpu"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ── Load cached features ─────────────────────────────────────────────

def load_cached_features():
    cache = {}
    available = []
    for s in STREAMS:
        train_p = FEAT_CACHE_DIR / f"{s}_train_feats.pt"
        val_p   = FEAT_CACHE_DIR / f"{s}_val_feats.pt"
        if not (train_p.exists() and val_p.exists()):
            print(f"  [WARN] Missing {s} features, skipping.")
            continue
        cache[s] = {
            "train": torch.load(train_p, weights_only=False),
            "val":   torch.load(val_p, weights_only=False),
        }
        available.append(s)

    if not available:
        print("  ERROR: No cached features found. Run s3a first.")
        sys.exit(1)

    train_labels = torch.load(FEAT_CACHE_DIR / "train_labels.pt", weights_only=False)
    val_labels   = torch.load(FEAT_CACHE_DIR / "val_labels.pt", weights_only=False)

    train_feats = torch.cat([cache[s]["train"] for s in available], dim=1)
    val_feats   = torch.cat([cache[s]["val"]   for s in available], dim=1)

    print(f"  [Data] Streams: {available} ({len(available)})")
    print(f"  [Data] Train: {train_feats.shape[0]:,} x {train_feats.shape[1]}d")
    print(f"  [Data] Val:   {val_feats.shape[0]:,} x {val_feats.shape[1]}d")

    return available, train_feats, val_feats, train_labels, val_labels


# ── Fusion Model 1: Concat + MLP ────────────────────────────────────

class ConcatMLPFusion(nn.Module):
    def __init__(self, n_streams):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(n_streams * 512, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 2),
        )

    def forward(self, x):
        return self.mlp(x)


# ── Fusion Model 2: Weighted Fusion ─────────────────────────────────

class WeightedFusion(nn.Module):
    def __init__(self, n_streams):
        super().__init__()
        self.n_streams = n_streams
        self.weights = nn.Parameter(torch.ones(n_streams) / n_streams)
        self.head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 2),
        )

    def forward(self, x):
        B = x.shape[0]
        streams = x.view(B, self.n_streams, 512)
        w = torch.softmax(self.weights, dim=0)
        fused = (streams * w.unsqueeze(0).unsqueeze(-1)).sum(dim=1)
        return self.head(fused)


# ── Fusion Model 3: Cross-Attention ─────────────────────────────────

class CrossAttentionFusionLayer(nn.Module):
    def __init__(self, d_model=512, n_heads=8, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        attn_out, attn_weights = self.self_attn(x, x, x)
        x = self.norm1(x + attn_out)
        x = self.norm2(x + self.ffn(x))
        return x, attn_weights


class CrossAttentionFusion(nn.Module):
    def __init__(self, n_streams, d_model=512, n_heads=8, n_layers=2, dropout=0.3):
        super().__init__()
        self.n_streams = n_streams
        self.layers = nn.ModuleList([
            CrossAttentionFusionLayer(d_model, n_heads, dropout=0.1)
            for _ in range(n_layers)
        ])
        self.stream_embed = nn.Parameter(torch.randn(1, n_streams, d_model) * 0.02)
        self.backbone = nn.Sequential(
            nn.Linear(n_streams * d_model, 1024),
            nn.LayerNorm(1024),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(1024, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.head = nn.Linear(512, 2)

    def forward(self, x):
        B = x.shape[0]
        tokens = x.view(B, self.n_streams, 512)
        tokens = tokens + self.stream_embed
        for layer in self.layers:
            tokens, _ = layer(tokens)
        fused = tokens.flatten(1)
        shared = self.backbone(fused)
        return self.head(shared)


# ── Training / Evaluation ────────────────────────────────────────────

def train_and_eval(name, model, train_feats, train_labels, val_feats, val_labels):
    train_ds = TensorDataset(train_feats, train_labels)
    val_ds   = TensorDataset(val_feats,   val_labels)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False)

    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=LR * 0.01)
    criterion = nn.CrossEntropyLoss()

    best_acc, best_auc, best_f1 = 0.0, 0.0, 0.0
    save_dir = OUTPUT_DIR / name.lower().replace('+', '_').replace(' ', '_')
    save_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    for epoch in range(EPOCHS):
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        for feats, labels in train_loader:
            feats, labels = feats.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(feats)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            correct += logits.argmax(1).eq(labels).sum().item()
            total += labels.size(0)
        train_acc = 100.0 * correct / total
        scheduler.step()

        model.eval()
        all_probs, all_preds, all_lbls = [], [], []
        with torch.no_grad():
            for feats, labels in val_loader:
                feats = feats.to(device)
                logits = model(feats)
                probs = torch.softmax(logits, 1)[:, 1].cpu().numpy()
                preds = logits.argmax(1).cpu().numpy()
                all_probs.extend(probs)
                all_preds.extend(preds)
                all_lbls.extend(labels.numpy())

        all_lbls = np.array(all_lbls)
        all_probs = np.array(all_probs)
        all_preds = np.array(all_preds)
        val_acc = 100.0 * (all_preds == all_lbls).mean()
        val_auc = roc_auc_score(all_lbls, all_probs) if len(set(all_lbls)) > 1 else 0.0
        val_f1 = f1_score(all_lbls, all_preds, zero_division=0)

        print(f"  [{name}] Ep {epoch+1:02d}/{EPOCHS} | "
              f"Loss {total_loss/len(train_loader):.4f} | Train {train_acc:.1f}% | "
              f"Val {val_acc:.2f}% | AUC {val_auc:.4f} | F1 {val_f1:.4f}")

        if val_acc > best_acc:
            best_acc, best_auc, best_f1 = val_acc, val_auc, val_f1
            torch.save(model.state_dict(), save_dir / "best_model.pth")

    elapsed = time.time() - t0
    print(f"  [{name}] Done in {elapsed:.0f}s | Best: Acc={best_acc:.2f}%  AUC={best_auc:.4f}  F1={best_f1:.4f}")
    return best_acc, best_auc, best_f1


# ── Main ─────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("EXP-C: Fusion Strategy Comparison (Cached Features)")
    print(f"Device: {device}  |  Epochs: {EPOCHS}  |  Batch: {BATCH_SIZE}  |  LR: {LR}")
    print("=" * 70)

    available, train_feats, val_feats, train_labels, val_labels = load_cached_features()
    n_streams = len(available)

    results = {}

    # ── Fusion 1: Concat+MLP ────────────────────────────────────────
    print(f"\n{'─'*70}")
    print("Fusion_1: Concat+MLP (baseline)")
    print(f"{'─'*70}")
    model1 = ConcatMLPFusion(n_streams).to(device)
    acc1, auc1, f1_1 = train_and_eval(
        "Concat+MLP", model1, train_feats, train_labels, val_feats, val_labels)
    results["Fusion_1_ConcatMLP"] = {"acc": round(acc1, 4), "auc": round(auc1, 4), "f1": round(f1_1, 4)}

    # ── Fusion 2: Weighted Fusion ───────────────────────────────────
    print(f"\n{'─'*70}")
    print("Fusion_2: Weighted Fusion")
    print(f"{'─'*70}")
    model2 = WeightedFusion(n_streams).to(device)
    acc2, auc2, f1_2 = train_and_eval(
        "Weighted", model2, train_feats, train_labels, val_feats, val_labels)
    results["Fusion_2_WeightedFusion"] = {"acc": round(acc2, 4), "auc": round(auc2, 4), "f1": round(f1_2, 4)}

    learned_w = torch.softmax(model2.weights.detach().cpu(), dim=0).numpy()
    print("  Learned stream weights:")
    for sn, w in zip(available, learned_w):
        print(f"    {sn:<6}: {w:.4f}")

    # ── Fusion 3: Cross-Attention ───────────────────────────────────
    print(f"\n{'─'*70}")
    print("Fusion_3: Cross-Attention")
    print(f"{'─'*70}")
    model3 = CrossAttentionFusion(n_streams).to(device)
    acc3, auc3, f1_3 = train_and_eval(
        "CrossAttention", model3, train_feats, train_labels, val_feats, val_labels)
    results["Fusion_3_CrossAttention"] = {"acc": round(acc3, 4), "auc": round(auc3, 4), "f1": round(f1_3, 4)}

    # ── Print comparison table ──────────────────────────────────────
    print("\n" + "=" * 70)
    print("EXP-C RESULTS — Fusion Strategy Comparison")
    print("=" * 70)
    print(f"{'Fusion Method':<30} {'Acc (%)':>10} {'AUC':>10} {'F1':>10}")
    print("-" * 62)
    for method, r in results.items():
        print(f"{method:<30} {r['acc']:>10.2f} {r['auc']:>10.4f} {r['f1']:>10.4f}")
    print("=" * 70)

    out_path = OUTPUT_DIR / "results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved -> {out_path}")


if __name__ == "__main__":
    main()
