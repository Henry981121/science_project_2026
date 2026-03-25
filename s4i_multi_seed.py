"""
EXP-I: Multi-Seed Validation
==============================
Runs MAIN_GRL 3 times with different seeds, reports mean +/- std.
Proves results are stable, not a lucky random seed.

Output: outputs/exp_i/multi_seed_results.json
"""
import sys, io, json, time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8') if hasattr(sys.stdout, 'buffer') else sys.stdout

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score

from config import FEAT_CACHE_DIR, OUTPUTS_DIR, TRAIN_CSV, VAL_CSV
FEAT_DIR = FEAT_CACHE_DIR
TRAIN_CSV = str(TRAIN_CSV)
VAL_CSV = str(VAL_CSV)
OUTPUT_DIR = OUTPUTS_DIR / 'exp_i'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
STREAMS = ['clip', 'fft', 'dct', 'dire', 'noise']
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

GENERATOR_TO_ID = {
    'adm': 0, 'glide': 1, 'sdv4': 2, 'sdv5': 3, 'midjourney': 4,
    'wildfake': 5, 'biggan': 6, 'vqdm': 7, 'wukong': 8, 'firefly': 9,
    'real': 10, 'real_extra': 11, 'wildfake_ddim': 12, 'wildfake_other': 13,
    'stylegan': 14, 'dcgan': 15, 'dcgan_unseen': 16,
    'fursona_gan': 17, 'waifu_gan': 18,
}
REAL_ID = GENERATOR_TO_ID['real']
N_SOURCES = len(GENERATOR_TO_ID)
N_GEN = N_SOURCES - 2

EPOCHS = 30
BATCH_SIZE = 256
LR = 1e-3
SEEDS = [42, 123, 456]


def load_data():
    cache = {}
    available = []
    for s in STREAMS:
        tp = FEAT_DIR / f"{s}_train_feats.pt"
        vp = FEAT_DIR / f"{s}_val_feats.pt"
        if tp.exists() and vp.exists():
            cache[s] = {"train": torch.load(tp, weights_only=False), "val": torch.load(vp, weights_only=False)}
            available.append(s)

    train_labels = torch.load(FEAT_DIR / "train_labels.pt", weights_only=False)
    val_labels = torch.load(FEAT_DIR / "val_labels.pt", weights_only=False)
    train_feats = torch.cat([cache[s]["train"] for s in available], dim=1)
    val_feats = torch.cat([cache[s]["val"] for s in available], dim=1)

    train_df = pd.read_csv(TRAIN_CSV)
    val_df = pd.read_csv(VAL_CSV)
    n_train = min(len(train_feats), len(train_df))
    n_val = min(len(val_feats), len(val_df))

    train_src = torch.tensor([GENERATOR_TO_ID.get(str(g).lower().strip(), REAL_ID) for g in train_df['generator'][:n_train]], dtype=torch.long)
    val_src = torch.tensor([GENERATOR_TO_ID.get(str(g).lower().strip(), REAL_ID) for g in val_df['generator'][:n_val]], dtype=torch.long)

    return available, train_feats[:n_train], train_labels[:n_train], train_src, val_feats[:n_val], val_labels[:n_val], val_src


def train_one_seed(seed, n_streams, train_feats, train_labels, train_src, val_feats, val_labels, val_src):
    from s3_main_grl import FusionDetectorGRL, GRLLoss, grl_lambda

    torch.manual_seed(seed)
    np.random.seed(seed)

    model = FusionDetectorGRL(n_streams=n_streams, n_sources=N_SOURCES, n_gen=N_GEN).to(DEVICE)
    criterion = GRLLoss(lambda_src=0.1, lambda_grl=0.05)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)

    warmup = optim.lr_scheduler.LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=5)
    cosine = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS-5, eta_min=LR*0.01)
    scheduler = optim.lr_scheduler.SequentialLR(optimizer, [warmup, cosine], milestones=[5])

    train_ds = TensorDataset(train_feats, train_labels, train_src)
    val_ds = TensorDataset(val_feats, val_labels, val_src)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    best_acc = 0
    for epoch in range(EPOCHS):
        lam = grl_lambda(epoch, EPOCHS, 0.05, 10.0)
        model.train()
        for feats, y_bin, y_src in train_loader:
            feats, y_bin, y_src = feats.to(DEVICE), y_bin.to(DEVICE), y_src.to(DEVICE)
            optimizer.zero_grad()
            lb, ls, lg, _ = model(feats, grl_lambda=lam)
            loss, _ = criterion(lb, ls, lg, y_bin, y_src, lam)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()

        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for feats, y_bin, _ in val_loader:
                lb, _, _, _ = model(feats.to(DEVICE), grl_lambda=0)
                all_preds.extend(lb.argmax(1).cpu().numpy())
                all_labels.extend(y_bin.numpy())
        acc = accuracy_score(all_labels, all_preds) * 100
        if acc > best_acc:
            best_acc = acc

    return best_acc


def main():
    print("=" * 60)
    print("  EXP-I: Multi-Seed Validation")
    print(f"  Seeds: {SEEDS}")
    print("=" * 60)

    available, train_feats, train_labels, train_src, val_feats, val_labels, val_src = load_data()
    n_streams = len(available)
    print(f"  Streams: {available} ({n_streams})")

    results = []
    for seed in SEEDS:
        print(f"\n  [Seed {seed}] Training...")
        t0 = time.time()
        acc = train_one_seed(seed, n_streams, train_feats, train_labels, train_src, val_feats, val_labels, val_src)
        elapsed = time.time() - t0
        print(f"  [Seed {seed}] Val Acc = {acc:.2f}% ({elapsed/60:.1f} min)")
        results.append({'seed': seed, 'val_acc': round(acc, 2)})

    accs = [r['val_acc'] for r in results]
    mean_acc = np.mean(accs)
    std_acc = np.std(accs)

    print(f"\n{'='*60}")
    print(f"  RESULTS")
    print(f"{'='*60}")
    for r in results:
        print(f"  Seed {r['seed']}: {r['val_acc']:.2f}%")
    print(f"\n  Mean: {mean_acc:.2f}% +/- {std_acc:.2f}%")
    print(f"{'='*60}")

    output = {'seeds': results, 'mean': round(mean_acc, 2), 'std': round(std_acc, 2)}
    with open(OUTPUT_DIR / 'multi_seed_results.json', 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved: {OUTPUT_DIR / 'multi_seed_results.json'}")


if __name__ == '__main__':
    main()
