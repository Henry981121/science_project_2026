"""
Stage 3-B: Ablation Study (EXP-B)
==================================
Part 1: All 2-to-5 stream combinations (26 total) with Concat+MLP fusion.
Part 2: Leave-One-Out (LOO) — 5 models each trained without one stream.

Fusion: Concat selected stream features -> Linear(N*512, 512) -> ReLU -> Dropout -> Linear(512, 2)
Training: 15 epochs per combination.

Results saved to: outputs/exp_b/
"""

import sys


import json
import time
from itertools import combinations
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

from sklearn.metrics import roc_auc_score, f1_score
import numpy as np

# ── Paths ──────────────────────────────────────────────────────────────
from config import TRAIN_CSV, VAL_CSV, OUTPUTS_DIR, FEAT_CACHE_DIR
TRAIN_CSV  = str(TRAIN_CSV)
VAL_CSV    = str(VAL_CSV)
OUTPUT_DIR = OUTPUTS_DIR / 'exp_b'

# ── Hyperparameters ────────────────────────────────────────────────────
EPOCHS      = 15
BATCH_SIZE  = 32
LR          = 1e-4
NUM_WORKERS = 4

STREAMS = ["clip", "fft", "dct", "dire", "noise"]

# 優先使用 s3a 預先提取並快取的 feature 檔（避免 VRAM OOM）
# 若快取不存在，才 fallback 到即時提取（需要所有 extractor 同時在 GPU）
# FEAT_CACHE_DIR imported from config above

device = "cuda" if torch.cuda.is_available() else "cpu"

# ── Pre-flight checks ──────────────────────────────────────────────────
if not Path(TRAIN_CSV).exists():
    print("ERROR: train.csv not found. Run s2c_prepare_splits.py first.")
    sys.exit(1)
if not Path(VAL_CSV).exists():
    print("ERROR: val.csv not found. Run s2c_prepare_splits.py first.")
    sys.exit(1)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ── Dataset ────────────────────────────────────────────────────────────

class ImageDataset(Dataset):
    def __init__(self, csv_path, transform=None, difficulty_filter=None):
        import pandas as pd
        self.df = pd.read_csv(csv_path)
        if difficulty_filter is not None:
            self.df = self.df[self.df['difficulty'].isin(difficulty_filter)]
        self.df = self.df.reset_index(drop=True)
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.difficulties = self.df['difficulty'].tolist()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row['path']).convert('RGB')
        img = self.transform(img)
        label = int(row['label'])
        source = int(row.get('source_label', 0))
        return img, label, source


# ── Concat+MLP Model ───────────────────────────────────────────────────

class ConcatMLPModel(nn.Module):
    def __init__(self, selected_streams, extractors, device):
        super().__init__()
        self.streams    = selected_streams
        self.extractors = nn.ModuleDict(extractors)  # [FIX] ModuleDict 讓 .to(device) 能移動 extractors
        n = len(selected_streams)
        self.mlp = nn.Sequential(
            nn.Linear(n * 512, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 2),
        )

    def forward(self, x):
        feats = []
        for s in self.streams:
            with torch.no_grad():
                feats.append(self.extractors[s].extract_features(x))
        return self.mlp(torch.cat(feats, dim=1))


# ── Training / Evaluation Utilities ───────────────────────────────────

def train_one_epoch(model, loader, optimizer, scaler, use_amp):
    model.train()
    criterion = nn.CrossEntropyLoss()
    total_loss, correct, total = 0.0, 0, 0

    for imgs, labels, _ in loader:
        imgs   = imgs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        if use_amp:
            with torch.amp.autocast("cuda"):  # [FIX] deprecated API
                logits = model(imgs)
                loss   = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(imgs)
            loss   = criterion(logits, labels)
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        pred        = logits.argmax(dim=1)
        correct    += pred.eq(labels).sum().item()
        total      += labels.size(0)

    return total_loss / len(loader), 100.0 * correct / total


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    all_labels, all_probs, all_preds = [], [], []

    for imgs, labels, _ in loader:
        imgs    = imgs.to(device)
        logits  = model(imgs)
        probs   = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
        preds   = logits.argmax(dim=1).cpu().numpy()
        all_probs.extend(probs.tolist())
        all_preds.extend(preds.tolist())
        all_labels.extend(labels.numpy().tolist())

    all_labels = np.array(all_labels)
    all_probs  = np.array(all_probs)
    all_preds  = np.array(all_preds)

    acc = 100.0 * (all_preds == all_labels).mean()
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        auc = 0.0
    f1 = f1_score(all_labels, all_preds, zero_division=0)

    return acc, auc, f1


def run_combination(combo_label, selected_streams, extractors,
                    train_loader, val_loader, use_amp, scaler):
    """Train and evaluate a single combination. Returns (acc, auc, f1)."""
    model = ConcatMLPModel(selected_streams, extractors, device).to(device)

    # Only MLP parameters are trainable; extractors are frozen inside forward()
    optimizer = optim.AdamW(model.mlp.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=LR * 0.01)

    best_acc, best_auc, best_f1 = 0.0, 0.0, 0.0

    for epoch in range(EPOCHS):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, scaler, use_amp)
        val_acc, val_auc, val_f1 = evaluate(model, val_loader)
        scheduler.step()

        print(
            f"    Ep {epoch+1:02d}/{EPOCHS} | "
            f"Loss {train_loss:.4f} | Train {train_acc:.1f}% | "
            f"Val {val_acc:.2f}% | AUC {val_auc:.4f} | F1 {val_f1:.4f}"
        )

        if val_acc > best_acc:
            best_acc  = val_acc
            best_auc  = val_auc
            best_f1   = val_f1

    return best_acc, best_auc, best_f1


# ── Main ───────────────────────────────────────────────────────────────

def load_cached_features():
    """
    優先讀取 s3a 已快取的 feature .pt 檔（FEAT_CACHE_DIR）。
    回傳 {stream: {'train': Tensor(N,512), 'val': Tensor(M,512)}}
    或 None（快取不存在時）。
    """
    cache = {}
    for s in STREAMS:
        train_p = FEAT_CACHE_DIR / f"{s}_train_feats.pt"
        val_p   = FEAT_CACHE_DIR / f"{s}_val_feats.pt"
        if not (train_p.exists() and val_p.exists()):
            print(f"  [Cache] 找不到 {s} 的快取 -> 改用即時提取模式")
            return None
        cache[s] = {
            "train": torch.load(train_p, weights_only=False),
            "val":   torch.load(val_p, weights_only=False),
        }
    # Load labels
    train_lbl = FEAT_CACHE_DIR / "train_labels.pt"
    val_lbl   = FEAT_CACHE_DIR / "val_labels.pt"
    if not (train_lbl.exists() and val_lbl.exists()):
        return None
    labels = {
        "train": torch.load(train_lbl, weights_only=False),
        "val":   torch.load(val_lbl, weights_only=False),
    }
    print(f"  [Cache] 成功讀取所有 stream features！")
    return cache, labels


def run_combination_cached(combo_label, selected_streams, feat_cache, labels):
    """從快取 features 訓練組合分類頭（無需 GPU extractor）。"""
    from torch.utils.data import TensorDataset
    n = len(selected_streams)

    train_feats = torch.cat([feat_cache[s]["train"] for s in selected_streams], dim=1)
    val_feats   = torch.cat([feat_cache[s]["val"]   for s in selected_streams], dim=1)
    train_lbl   = labels["train"]
    val_lbl     = labels["val"]

    train_ds = TensorDataset(train_feats, train_lbl)
    val_ds   = TensorDataset(val_feats,   val_lbl)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False)

    mlp = nn.Sequential(
        nn.Linear(n * 512, 512), nn.ReLU(), nn.Dropout(0.3), nn.Linear(512, 2)
    ).to(device)
    optimizer = optim.AdamW(mlp.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=LR * 0.01)
    criterion = nn.CrossEntropyLoss()
    best_acc, best_auc, best_f1 = 0.0, 0.0, 0.0

    for epoch in range(EPOCHS):
        mlp.train()
        total_loss, correct, total = 0.0, 0, 0
        for feats, lbls in train_loader:
            feats, lbls = feats.to(device), lbls.to(device)
            optimizer.zero_grad()
            logits = mlp(feats)
            loss = criterion(logits, lbls)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            correct += logits.argmax(1).eq(lbls).sum().item()
            total += lbls.size(0)
        scheduler.step()

        mlp.eval()
        all_probs, all_preds, all_lbls = [], [], []
        with torch.no_grad():
            for feats, lbls in val_loader:
                feats = feats.to(device)
                logits = mlp(feats)
                probs = torch.softmax(logits, 1)[:, 1].cpu().numpy()
                preds = logits.argmax(1).cpu().numpy()
                all_probs.extend(probs); all_preds.extend(preds); all_lbls.extend(lbls.numpy())
        all_lbls = np.array(all_lbls); all_probs = np.array(all_probs); all_preds = np.array(all_preds)
        val_acc = 100.0 * (all_preds == all_lbls).mean()
        try: val_auc = roc_auc_score(all_lbls, all_probs)
        except: val_auc = 0.0
        val_f1 = f1_score(all_lbls, all_preds, zero_division=0)

        print(f"    Ep {epoch+1:02d}/{EPOCHS} | "
              f"Loss {total_loss/len(train_loader):.4f} | Train {100.*correct/total:.1f}% | "
              f"Val {val_acc:.2f}% | AUC {val_auc:.4f} | F1 {val_f1:.4f}")

        if val_acc > best_acc:
            best_acc, best_auc, best_f1 = val_acc, val_auc, val_f1

    return best_acc, best_auc, best_f1


def main():
    print("=" * 70)
    print("EXP-B: Ablation Study (all 2-5 stream combinations + LOO)")
    print(f"Device : {device} | Epochs per combo: {EPOCHS}")
    print("=" * 70)

    # ── 優先使用快取 features（省 VRAM，速度快）────────────────────────
    cache_result = load_cached_features()
    use_cache = cache_result is not None
    if use_cache:
        feat_cache, feat_labels = cache_result
        print("[Mode] 使用 s3a 快取 features（不需要 extractor GPU）")
    else:
        print("[Mode] 快取不存在，改用即時提取（需要所有 extractor）")
        # 即時提取模式才需要 Dataset / DataLoader
        train_dataset = ImageDataset(TRAIN_CSV)
        val_dataset   = ImageDataset(VAL_CSV)
        train_loader = DataLoader(
            train_dataset, batch_size=BATCH_SIZE, shuffle=True,
            num_workers=NUM_WORKERS, pin_memory=(device == "cuda"),
        )
        val_loader = DataLoader(
            val_dataset, batch_size=BATCH_SIZE, shuffle=False,
            num_workers=NUM_WORKERS, pin_memory=(device == "cuda"),
        )

    use_amp = (device == "cuda")
    scaler  = torch.amp.GradScaler("cuda") if use_amp else None  # [FIX] deprecated API

    extractors = None
    if not use_cache:
        from src.feature_extractors import (
            CLIPFeatureExtractor, FFTFeatureExtractor,
            DCTFeatureExtractor, DIREFeatureExtractor, NoisePrintExtractor,
        )
        # 全部先建在 CPU，避免 DIRE OOM（在 forward 內移 GPU）
        extractors = {
            "clip":  CLIPFeatureExtractor(device='cpu'),
            "fft":   FFTFeatureExtractor(device='cpu'),
            "dct":   DCTFeatureExtractor(device='cpu'),
            "dire":  DIREFeatureExtractor(device='cpu'),
            "noise": NoisePrintExtractor(device='cpu'),
        }

    # ── Part 1: All 2-to-5 stream combinations ─────────────────────────
    all_combos = []
    for r in range(2, 6):
        all_combos.extend(combinations(STREAMS, r))
    # all_combos has 10 + 10 + 5 + 1 = 26 entries

    print(f"\nPart 1: {len(all_combos)} combinations (r=2..5)")
    results_part1 = {}

    for idx, combo in enumerate(all_combos, 1):
        combo_key = "+".join(combo)
        print(f"\n[{idx:02d}/{len(all_combos)}] Combo: [{combo_key}]")
        t0 = time.time()

        if use_cache:
            acc, auc, f1 = run_combination_cached(combo_key, list(combo), feat_cache, feat_labels)
        else:
            acc, auc, f1 = run_combination(
                combo_key, list(combo), extractors,
                train_loader, val_loader, use_amp, scaler,
            )

        elapsed = time.time() - t0
        print(f"  -> Best: Acc={acc:.2f}%  AUC={auc:.4f}  F1={f1:.4f}  ({elapsed:.0f}s)")
        results_part1[combo_key] = {"acc": round(acc, 4), "auc": round(auc, 4), "f1": round(f1, 4), "streams": list(combo)}

    # ── Part 2: LOO (Leave-One-Out) ────────────────────────────────────
    print(f"\n{'='*70}")
    print("Part 2: LOO — 5 models, each trained without one stream")
    print(f"{'='*70}")

    results_loo = {}
    # MAIN (all 5 streams) is the all-5 combo already computed in Part 1
    main_key  = "+".join(STREAMS)
    main_acc  = results_part1.get(main_key, {}).get("acc", None)

    for left_out in STREAMS:
        selected   = [s for s in STREAMS if s != left_out]
        combo_key  = "+".join(selected)
        loo_label  = f"LOO_without_{left_out}"
        print(f"\n[LOO] Removing [{left_out}]  |  Training: {combo_key}")
        t0 = time.time()

        # If this combo was already trained in Part 1, skip re-training
        if combo_key in results_part1:
            acc  = results_part1[combo_key]["acc"]
            auc  = results_part1[combo_key]["auc"]
            f1   = results_part1[combo_key]["f1"]
            print(f"  -> Reusing Part 1 result: Acc={acc:.2f}%  AUC={auc:.4f}")
        else:
            if use_cache:
                acc, auc, f1 = run_combination_cached(loo_label, selected, feat_cache, feat_labels)
            else:
                acc, auc, f1 = run_combination(
                    loo_label, selected, extractors,
                    train_loader, val_loader, use_amp, scaler,
                )
            elapsed = time.time() - t0
            print(f"  -> Best: Acc={acc:.2f}%  AUC={auc:.4f}  F1={f1:.4f}  ({elapsed:.0f}s)")

        results_loo[loo_label] = {
            "removed_stream": left_out,
            "streams":        selected,
            "acc":            round(acc, 4),
            "auc":            round(auc, 4),
            "f1":             round(f1, 4),
        }

    # ── Print sorted bar chart (text) ──────────────────────────────────
    print("\n" + "=" * 70)
    print("EXP-B PART 1 — Sorted by Accuracy (worst → best)")
    print("=" * 70)
    sorted_combos = sorted(results_part1.items(), key=lambda x: x[1]["acc"])
    for combo_key, r in sorted_combos:
        bar_len = max(1, int(r["acc"] / 2))   # scale: 1 char = 2%
        bar     = "█" * bar_len
        print(f"  {combo_key:<35} {r['acc']:>6.2f}%  {bar}")

    print("\n" + "=" * 70)
    print("EXP-B PART 2 — LOO Waterfall (accuracy drop vs MAIN all-5)")
    print("=" * 70)
    if main_acc is not None:
        print(f"  MAIN (all 5 streams): {main_acc:.2f}%")
        print()
    for loo_label, r in results_loo.items():
        drop = (main_acc - r["acc"]) if main_acc is not None else float("nan")
        sign = "-" if drop >= 0 else "+"
        print(
            f"  Without [{r['removed_stream']:<5}]: "
            f"Acc={r['acc']:.2f}%  "
            f"Drop={sign}{abs(drop):.2f}%"
        )
    print("=" * 70)

    # ── Save results ───────────────────────────────────────────────────
    all_results = {
        "part1_combinations": results_part1,
        "part2_loo":          results_loo,
        "main_acc_all5":      main_acc,
    }
    out_path = OUTPUT_DIR / "results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved -> {out_path}")


if __name__ == "__main__":
    main()
