"""
Stage 3-A: Single-Stream Baseline (EXP-A) — 兩階段版本
======================================================
Phase 1: 預先提取並儲存所有特徵到硬碟（只跑一次）
Phase 2: 從硬碟讀取特徵，訓練輕量分類頭（很快）

結果儲存到: outputs/exp_a/
"""

import sys


import json
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torchvision import transforms
from PIL import Image
from sklearn.metrics import roc_auc_score, f1_score
import numpy as np

# ── 設定 ──────────────────────────────────────────────────
from config import TRAIN_CSV, VAL_CSV, TEST_CSV, CROSS_CSV, OUTPUTS_DIR, FEAT_CACHE_DIR
TRAIN_CSV  = str(TRAIN_CSV)
VAL_CSV    = str(VAL_CSV)
TEST_CSV   = str(TEST_CSV)
CROSS_CSV  = str(CROSS_CSV)
OUTPUT_DIR = OUTPUTS_DIR / 'exp_a'
FEAT_DIR   = FEAT_CACHE_DIR

EPOCHS      = 20
BATCH_SIZE  = 256   # 特徵訓練可以用大 batch
LR          = 1e-3
NUM_WORKERS = 2     # 圖片讀取 worker 數
IMG_BATCH   = 64    # 提取特徵時的 batch size（DIRE 會自動縮小）
DIRE_BATCH  = 16    # DIRE VAE 記憶體需求大，用小 batch
# ──────────────────────────────────────────────────────────

device = "cuda" if torch.cuda.is_available() else "cpu"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FEAT_DIR.mkdir(parents=True, exist_ok=True)

if not Path(TRAIN_CSV).exists():
    print("ERROR: train.csv not found. Run s2c_prepare_splits.py first.")
    sys.exit(1)


# ── Phase 1：圖片 Dataset ──────────────────────────────────

class RobustAugmentation:
    """Random image degradation to improve robustness."""
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, img):
        import random, io
        if random.random() > self.prob:
            return img

        aug = random.choice(['jpeg', 'resize', 'blur', 'noise'])

        if aug == 'jpeg':
            # JPEG recompression (q=30-70)
            quality = random.randint(30, 70)
            buf = io.BytesIO()
            img.save(buf, format='JPEG', quality=quality)
            buf.seek(0)
            img = Image.open(buf).convert('RGB')

        elif aug == 'resize':
            # Downscale then upscale (simulate social media thumbnail)
            w, h = img.size
            scale = random.uniform(0.3, 0.7)
            small = img.resize((int(w*scale), int(h*scale)), Image.BILINEAR)
            img = small.resize((w, h), Image.BILINEAR)

        elif aug == 'blur':
            # Gaussian blur
            from PIL import ImageFilter
            radius = random.uniform(0.5, 2.0)
            img = img.filter(ImageFilter.GaussianBlur(radius=radius))

        elif aug == 'noise':
            # Add Gaussian noise
            import numpy as np
            arr = np.array(img).astype(np.float32)
            sigma = random.uniform(5, 25)
            noise = np.random.randn(*arr.shape) * sigma
            arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
            img = Image.fromarray(arr)

        return img


class ImageDataset(Dataset):
    def __init__(self, csv_path, augment=False):
        import pandas as pd
        self.df = pd.read_csv(csv_path).reset_index(drop=True)
        self.augment = RobustAugmentation(prob=0.5) if augment else None
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        try:
            img = Image.open(row['path']).convert('RGB')
            if self.augment:
                img = self.augment(img)
        except Exception:
            img = Image.new('RGB', (224, 224), 0)
        return self.transform(img), int(row['label'])

    @property
    def labels(self):
        return self.df['label'].tolist()


def extract_and_save(extractor, name, split, csv_path):
    """
    從圖片提取特徵並存到硬碟。
    如果已存在就跳過。
    回傳 (features tensor, labels tensor)
    """
    feat_path  = FEAT_DIR / f"{name}_{split}_feats.pt"
    label_path = FEAT_DIR / f"{split}_labels.pt"

    if feat_path.exists() and label_path.exists():
        print(f"  [{name}/{split}] 已存在，直接讀取...")
        return torch.load(feat_path, weights_only=False), torch.load(label_path, weights_only=False)

    print(f"  [{name}/{split}] 提取特徵中...")
    use_augment = (split == "train")  # Only augment training data
    dataset = ImageDataset(csv_path, augment=use_augment)
    if use_augment:
        print(f"    [Augmentation] JPEG/Resize/Blur/Noise (prob=50%)")
    batch_sz = DIRE_BATCH if name == "dire" else IMG_BATCH
    loader  = DataLoader(dataset, batch_size=batch_sz, shuffle=False,
                         num_workers=NUM_WORKERS, pin_memory=True)

    all_feats  = []
    all_labels = []
    extractor.eval()

    with torch.no_grad():
        for i, (imgs, labels) in enumerate(loader):
            imgs = imgs.to(device)
            feats = extractor.extract_features(imgs).cpu()
            all_feats.append(feats)
            all_labels.append(labels)
            if (i + 1) % 50 == 0:
                print(f"    batch {i+1}/{len(loader)}")

    feats_tensor  = torch.cat(all_feats,  dim=0)
    labels_tensor = torch.cat(all_labels, dim=0)

    torch.save(feats_tensor,  feat_path)
    torch.save(labels_tensor, label_path)
    print(f"    儲存完畢：{feats_tensor.shape} -> {feat_path}")

    return feats_tensor, labels_tensor


# ── Phase 2：特徵分類頭 ────────────────────────────────────

class LinearHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 2)
        )

    def forward(self, x):
        return self.net(x)


def train_head(train_feats, train_labels, val_feats, val_labels, stream_name):
    """用預先提取的特徵訓練分類頭"""
    save_dir = OUTPUT_DIR / stream_name
    save_dir.mkdir(exist_ok=True)

    train_ds = TensorDataset(train_feats, train_labels)
    val_ds   = TensorDataset(val_feats,   val_labels)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False)

    model     = LinearHead().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_acc, best_auc, best_f1 = 0.0, 0.0, 0.0

    for epoch in range(EPOCHS):
        # Train
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        for feats, labels in train_loader:
            feats, labels = feats.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(feats)
            loss   = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            correct    += logits.argmax(1).eq(labels).sum().item()
            total      += labels.size(0)
        train_acc = 100. * correct / total
        scheduler.step()

        # Val
        model.eval()
        all_probs, all_preds, all_lbls = [], [], []
        with torch.no_grad():
            for feats, labels in val_loader:
                feats = feats.to(device)
                logits = model(feats)
                probs  = torch.softmax(logits, 1)[:, 1].cpu().numpy()
                preds  = logits.argmax(1).cpu().numpy()
                all_probs.extend(probs)
                all_preds.extend(preds)
                all_lbls.extend(labels.numpy())

        all_lbls  = np.array(all_lbls)
        all_probs = np.array(all_probs)
        all_preds = np.array(all_preds)
        val_acc   = 100. * (all_preds == all_lbls).mean()
        val_auc   = roc_auc_score(all_lbls, all_probs) if len(set(all_lbls)) > 1 else 0.0
        val_f1    = f1_score(all_lbls, all_preds, zero_division=0)

        print(f"  Ep {epoch+1:02d}/{EPOCHS} | "
              f"Train {train_acc:.1f}% | Val {val_acc:.1f}% | "
              f"AUC {val_auc:.4f} | F1 {val_f1:.4f}")

        if val_acc > best_acc:
            best_acc, best_auc, best_f1 = val_acc, val_auc, val_f1
            torch.save(model.state_dict(), save_dir / "best_model.pth")

    return best_acc, best_auc, best_f1


# ── Main ──────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("EXP-A: Single-Stream Baseline (兩階段)")
    print(f"Device: {device}  |  Epochs: {EPOCHS}  |  Batch: {BATCH_SIZE}")
    print("=" * 60)

    from src.feature_extractors import (
        CLIPFeatureExtractor, FFTFeatureExtractor,
        DCTFeatureExtractor, DIREFeatureExtractor, NoisePrintExtractor,
    )

    # 全部先建在 CPU，要用時才搬 GPU（避免 5 個同時佔 VRAM）
    EXTRACTORS = {
        "clip":  CLIPFeatureExtractor(device='cpu'),
        "fft":   FFTFeatureExtractor(device='cpu'),
        "dct":   DCTFeatureExtractor(device='cpu'),
        "dire":  DIREFeatureExtractor(device='cpu'),
        "noise": NoisePrintExtractor(device='cpu'),
    }

    results = {}

    for name, extractor in EXTRACTORS.items():
        print(f"\n{'─'*60}")
        print(f"Stream: [{name.upper()}]")
        print(f"{'─'*60}")

        t0 = time.time()

        # 只把當前 extractor 搬上 GPU
        extractor.to(device)
        torch.cuda.empty_cache()

        # Phase 1: 提取特徵（train/val/test/cross_gen 全部用同一個 extractor）
        print("[Phase 1] 提取特徵...")
        train_feats, train_labels = extract_and_save(extractor, name, "train", TRAIN_CSV)
        val_feats,   val_labels   = extract_and_save(extractor, name, "val",   VAL_CSV)
        extract_and_save(extractor, name, "test", TEST_CSV)
        if Path(CROSS_CSV).exists():
            extract_and_save(extractor, name, "cross_gen_test", CROSS_CSV)

        # 保存 extractor 權重（讓 Demo 能用同一個權重提取新圖片的特徵）
        ext_save_path = FEAT_DIR / f"{name}_extractor.pth"
        torch.save(extractor.state_dict(), ext_save_path)
        print(f"  Extractor saved -> {ext_save_path}")

        # 釋放 extractor GPU 記憶體
        extractor.cpu()
        torch.cuda.empty_cache()

        # Phase 2: 訓練分類頭
        print("[Phase 2] 訓練分類頭...")
        best_acc, best_auc, best_f1 = train_head(
            train_feats, train_labels, val_feats, val_labels, name
        )

        elapsed = time.time() - t0
        print(f"\n[{name}] Best: Acc={best_acc:.2f}%  AUC={best_auc:.4f}  F1={best_f1:.4f}  ({elapsed:.0f}s)")

        results[name] = {"acc": round(best_acc, 4),
                         "auc": round(best_auc, 4),
                         "f1":  round(best_f1, 4)}

        # 把 extractor 搬回 GPU 給下一輪用（但其實下一輪會換新的）
        # 不需要搬回，直接讓 GC 清除

    # ── 結果表 ──
    print("\n" + "=" * 60)
    print("EXP-A RESULTS")
    print("=" * 60)
    print(f"{'Stream':<10} {'Acc%':>8} {'AUC':>8} {'F1':>8}")
    print("-" * 38)
    for s, r in results.items():
        print(f"{s:<10} {r['acc']:>8.2f} {r['auc']:>8.4f} {r['f1']:>8.4f}")
    print("=" * 60)

    out = OUTPUT_DIR / "results.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults -> {out}")


if __name__ == "__main__":
    main()
