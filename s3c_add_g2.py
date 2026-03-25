"""
S3C Supplement: Run 3 fusion methods on G2 (unseen) cross_gen_test
Uses cached features from S3A.
"""

import json

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, roc_auc_score

from config import OUTPUTS_DIR, FEAT_CACHE_DIR


# ── Config ─────────────────────────────────────────────────────────────

FEAT_DIR   = FEAT_CACHE_DIR
OUTPUT_DIR = OUTPUTS_DIR / "exp_c"
STREAMS    = ["clip", "fft", "dct", "dire", "noise"]
N_STREAMS  = len(STREAMS)
EPOCHS     = 20
BATCH_SIZE = 256
LR         = 1e-3
device     = "cuda" if torch.cuda.is_available() else "cpu"


# ── Data Loading ────────────────────────────────────────────────────────

def load_split(prefix: str):
    feats = torch.cat(
        [torch.load(FEAT_DIR / f"{s}_{prefix}_feats.pt", weights_only=False)
         for s in STREAMS],
        dim=1,
    )
    labels = torch.load(FEAT_DIR / f"{prefix}_labels.pt", weights_only=False)
    return feats, labels


train_feats,  train_labels  = load_split("train")
val_feats,    val_labels    = load_split("val")
cross_feats,  cross_labels  = load_split("cross_gen_test")

print(f"Train: {len(train_feats):,} | Val: {len(val_feats):,} | Cross-gen: {len(cross_feats):,}")


# ── Models ──────────────────────────────────────────────────────────────

class ConcatMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(N_STREAMS * 512, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 2),
        )

    def forward(self, x):
        return self.mlp(x)


class WeightedFusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.w    = nn.Parameter(torch.ones(N_STREAMS) / N_STREAMS)
        self.head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 2),
        )

    def forward(self, x):
        B       = x.shape[0]
        streams = x.view(B, N_STREAMS, 512)
        w       = torch.softmax(self.w, dim=0)          # (N,)
        fused   = (streams * w.unsqueeze(0).unsqueeze(-1)).sum(dim=1)  # (B, 512)
        return self.head(fused)


class CrossAttn(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = nn.MultiheadAttention(512, num_heads=8, dropout=0.1, batch_first=True)
        self.norm1 = nn.LayerNorm(512)
        self.norm2 = nn.LayerNorm(512)
        self.ffn = nn.Sequential(
            nn.Linear(512, 2048),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(2048, 512),
            nn.Dropout(0.1),
        )
        self.stream_embed = nn.Parameter(torch.randn(1, N_STREAMS, 512) * 0.02)
        self.backbone = nn.Sequential(
            nn.Linear(N_STREAMS * 512, 1024),
            nn.LayerNorm(1024),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.3),
        )
        self.head = nn.Linear(512, 2)

    def forward(self, x):
        B      = x.shape[0]
        tokens = x.view(B, N_STREAMS, 512) + self.stream_embed
        attn_out, _ = self.attn(tokens, tokens, tokens)
        tokens = self.norm1(tokens + attn_out)
        tokens = self.norm2(tokens + self.ffn(tokens))
        return self.head(self.backbone(tokens.flatten(1)))


# ── Training & Evaluation ───────────────────────────────────────────────

def train_and_eval(name: str, model: nn.Module) -> dict:
    model.to(device)
    optimizer  = optim.AdamW(model.parameters(), lr=LR)
    criterion  = nn.CrossEntropyLoss()
    loader     = DataLoader(
        TensorDataset(train_feats, train_labels),
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    best_acc, best_state = 0.0, None
    for epoch in range(EPOCHS):
        model.train()
        for feats, labels in loader:
            feats, labels = feats.to(device), labels.to(device)
            optimizer.zero_grad()
            criterion(model(feats), labels).backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            preds = model(val_feats.to(device)).argmax(dim=1).cpu().numpy()
        acc = accuracy_score(val_labels.numpy(), preds) * 100
        if acc > best_acc:
            best_acc   = acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    model.eval()

    # G1 — validation set
    with torch.no_grad():
        logits_g1 = model(val_feats.to(device))
    g1_acc = accuracy_score(val_labels.numpy(), logits_g1.argmax(1).cpu().numpy()) * 100
    g1_auc = roc_auc_score(val_labels.numpy(), F.softmax(logits_g1, dim=1)[:, 1].cpu().numpy())

    # G2 — unseen cross-generator test
    with torch.no_grad():
        logits_g2 = model(cross_feats.to(device))
    g2_acc = accuracy_score(cross_labels.numpy(), logits_g2.argmax(1).cpu().numpy()) * 100
    g2_auc = roc_auc_score(cross_labels.numpy(), F.softmax(logits_g2, dim=1)[:, 1].cpu().numpy())

    gap = g2_acc - g1_acc
    print(f"  {name:<20} G1={g1_acc:.2f}%  G2={g2_acc:.2f}%  (gap={gap:+.2f}%)")

    return {
        "g1_acc": round(g1_acc, 2),
        "g1_auc": round(g1_auc, 4),
        "g2_acc": round(g2_acc, 2),
        "g2_auc": round(g2_auc, 4),
        "gap":    round(gap, 2),
    }


# ── Main ────────────────────────────────────────────────────────────────

def main():
    print(f"\n{'='*60}")
    print("  EXP-C Supplement: G1 vs G2 Comparison")
    print(f"{'='*60}")

    experiments = [
        ("Concat+MLP",     ConcatMLP),
        ("WeightedFusion", WeightedFusion),
        ("CrossAttention", CrossAttn),
    ]

    results = {}
    for name, ModelClass in experiments:
        results[name] = train_and_eval(name, ModelClass())

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / "results_g2.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
