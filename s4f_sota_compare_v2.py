"""
EXP-F v2: SOTA Comparison (Fair Conditions)
=============================================
All baselines trained on the SAME training set (full train.csv).
All evaluated on the SAME test set (test.csv).

Baselines:
  1. ResNet50 (full fine-tune) — strongest single-model CNN baseline
  2. EfficientNet-B4 (full fine-tune) — efficient CNN baseline
  3. CLIP Linear Probe — CLIP features + linear head (= UnivFD approach)

Our model:
  4. FusionDetectorGRL v2.1 (5-stream + Cross-Attention + GRL)

Output: outputs/exp_f_v2/
"""
import sys, io, json, time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8') if hasattr(sys.stdout, 'buffer') else sys.stdout

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torchvision import transforms, models
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score

from config import FEAT_CACHE_DIR, OUTPUTS_DIR, SPLITS_DIR, TEST_CSV, TRAIN_CSV
FEAT_DIR = FEAT_CACHE_DIR
MODEL_PATH = OUTPUTS_DIR / 'main_grl' / 'best_model.pth'
TEST_CSV = str(TEST_CSV)
TRAIN_CSV = str(TRAIN_CSV)
OUTPUT_DIR = OUTPUTS_DIR / 'exp_f_v2'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
STREAMS = ['clip', 'fft', 'dct', 'dire', 'noise']

# ── Training config (SAME for all baselines) ──
FINETUNE_EPOCHS = 10
FINETUNE_LR = 1e-4
FINETUNE_BATCH = 32
# Use same number of training samples as our model
# Our model uses full train.csv but baselines are slow, so cap at 20,000
TRAIN_CAP = 20000

EVAL_TF = transforms.Compose([
    transforms.Resize((224, 224)), transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
TRAIN_TF = transforms.Compose([
    transforms.Resize((256, 256)), transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


class ImageDataset(Dataset):
    def __init__(self, samples, transform):
        self.samples = samples
        self.transform = transform
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        path, label = self.samples[idx]
        try:
            img = Image.open(path).convert('RGB')
        except:
            img = Image.new('RGB', (224, 224), 128)
        return self.transform(img), label


def csv_to_samples(csv_path, max_n=None):
    df = pd.read_csv(csv_path)
    if max_n and len(df) > max_n:
        df = df.sample(max_n, random_state=42)
    return list(zip(df['path'].astype(str), df['label'].astype(int)))


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6


def finetune_model(model, train_samples, name):
    """Fine-tune a model on training data."""
    print(f"  Fine-tuning {name} ({len(train_samples):,} samples, {FINETUNE_EPOCHS} epochs)...")
    dataset = ImageDataset(train_samples, TRAIN_TF)
    loader = DataLoader(dataset, batch_size=FINETUNE_BATCH, shuffle=True, num_workers=0, pin_memory=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=FINETUNE_LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=FINETUNE_EPOCHS)
    criterion = nn.CrossEntropyLoss()

    model.train()
    t0 = time.time()
    for epoch in range(1, FINETUNE_EPOCHS + 1):
        correct, total, running_loss = 0, 0, 0
        for imgs, labels in loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            out = model(imgs)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()
            correct += (out.argmax(1) == labels).sum().item()
            total += labels.size(0)
            running_loss += loss.item() * labels.size(0)
        scheduler.step()
        print(f"    Epoch {epoch:>2}/{FINETUNE_EPOCHS}  loss={running_loss/total:.4f}  acc={correct/total*100:.1f}%")

    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.0f}s")
    model.eval()
    return model


def evaluate_model(model, test_samples, is_image_model=True):
    """Evaluate on test set. Returns (acc, auc, f1)."""
    dataset = ImageDataset(test_samples, EVAL_TF)
    loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=0, pin_memory=True)

    all_probs, all_preds, all_labels = [], [], []
    model.eval()
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(DEVICE)
            out = model(imgs)
            probs = F.softmax(out, dim=1)[:, 1]
            preds = out.argmax(dim=1)
            all_probs.extend(probs.cpu().numpy().tolist())
            all_preds.extend(preds.cpu().numpy().tolist())
            all_labels.extend(labels.numpy().tolist())

    acc = accuracy_score(all_labels, all_preds) * 100
    auc = roc_auc_score(all_labels, np.array(all_probs)) * 100
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    return acc, auc, f1


def main():
    print("=" * 65)
    print("  EXP-F v2: SOTA Comparison (Fair Conditions)")
    print(f"  Training: {TRAIN_CAP:,} samples (same for all)")
    print(f"  Testing: test.csv (full)")
    print("=" * 65)

    train_samples = csv_to_samples(TRAIN_CSV, max_n=TRAIN_CAP)
    test_samples = csv_to_samples(TEST_CSV)
    print(f"\n  Train: {len(train_samples):,} | Test: {len(test_samples):,}")

    results = {}

    # ══════════════════════════════════════════════
    # 1. ResNet50 (full fine-tune)
    # ══════════════════════════════════════════════
    print(f"\n{'-'*65}")
    print("  BASELINE 1: ResNet50 (full fine-tune)")
    print(f"{'-'*65}")
    resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    resnet.fc = nn.Linear(2048, 2)
    resnet = resnet.to(DEVICE)
    n_resnet = count_params(resnet)
    print(f"  Params: {n_resnet:.1f}M (all trainable)")

    resnet = finetune_model(resnet, train_samples, "ResNet50")
    acc, auc, f1 = evaluate_model(resnet, test_samples)
    results['ResNet50 (fine-tune)'] = {'acc': round(acc, 2), 'auc': round(auc, 2), 'f1': round(f1, 4), 'params_m': round(n_resnet, 1)}
    print(f"  Result: Acc={acc:.1f}% AUC={auc:.1f}% F1={f1:.4f}")
    del resnet; torch.cuda.empty_cache()

    # ══════════════════════════════════════════════
    # 2. EfficientNet-B4 (full fine-tune)
    # ══════════════════════════════════════════════
    print(f"\n{'-'*65}")
    print("  BASELINE 2: EfficientNet-B4 (full fine-tune)")
    print(f"{'-'*65}")
    try:
        import timm
        effnet = timm.create_model('efficientnet_b4', pretrained=True, num_classes=2)
    except ImportError:
        effnet = models.efficientnet_b4(weights=models.EfficientNet_B4_Weights.IMAGENET1K_V1)
        effnet.classifier[-1] = nn.Linear(effnet.classifier[-1].in_features, 2)
    effnet = effnet.to(DEVICE)
    n_eff = count_params(effnet)
    print(f"  Params: {n_eff:.1f}M")

    effnet = finetune_model(effnet, train_samples, "EfficientNet-B4")
    acc, auc, f1 = evaluate_model(effnet, test_samples)
    results['EfficientNet-B4'] = {'acc': round(acc, 2), 'auc': round(auc, 2), 'f1': round(f1, 4), 'params_m': round(n_eff, 1)}
    print(f"  Result: Acc={acc:.1f}% AUC={auc:.1f}% F1={f1:.4f}")
    del effnet; torch.cuda.empty_cache()

    # ══════════════════════════════════════════════
    # 3. CLIP Linear Probe (= UnivFD approach)
    # ══════════════════════════════════════════════
    print(f"\n{'-'*65}")
    print("  BASELINE 3: CLIP Linear Probe (UnivFD approach)")
    print(f"{'-'*65}")

    # Use cached CLIP features from S3A (same features our model uses)
    clip_train = torch.load(FEAT_DIR / "clip_train_feats.pt", weights_only=False)
    clip_test = torch.load(FEAT_DIR / "clip_test_feats.pt", weights_only=False)
    train_labels = torch.load(FEAT_DIR / "train_labels.pt", weights_only=False)
    test_labels = torch.load(FEAT_DIR / "test_labels.pt", weights_only=False)

    # Cap training to same amount
    n_cap = min(TRAIN_CAP, len(clip_train))
    torch.manual_seed(42)
    idx = torch.randperm(len(clip_train))[:n_cap]
    clip_train_sub = clip_train[idx]
    labels_sub = train_labels[idx]

    # Train linear probe
    probe = nn.Linear(512, 2).to(DEVICE)
    optimizer = torch.optim.AdamW(probe.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    loader = DataLoader(TensorDataset(clip_train_sub, labels_sub), batch_size=256, shuffle=True)

    print(f"  Training linear probe ({n_cap:,} samples, 20 epochs)...")
    probe.train()
    for epoch in range(20):
        for f, l in loader:
            f, l = f.to(DEVICE), l.to(DEVICE)
            optimizer.zero_grad()
            criterion(probe(f), l).backward()
            optimizer.step()

    probe.eval()
    with torch.no_grad():
        logits = probe(clip_test.to(DEVICE))
        probs = F.softmax(logits, 1)[:, 1].cpu().numpy()
        preds = logits.argmax(1).cpu().numpy()
    labels_np = test_labels.numpy()
    acc = accuracy_score(labels_np, preds) * 100
    auc = roc_auc_score(labels_np, probs) * 100
    f1 = f1_score(labels_np, preds, zero_division=0)
    n_probe = count_params(probe)
    results['CLIP Linear Probe'] = {'acc': round(acc, 2), 'auc': round(auc, 2), 'f1': round(f1, 4), 'params_m': round(n_probe, 3)}
    print(f"  Result: Acc={acc:.1f}% AUC={auc:.1f}% F1={f1:.4f} Params={n_probe:.3f}M")

    # ══════════════════════════════════════════════
    # 4. Ours (FusionDetectorGRL v2.1)
    # ══════════════════════════════════════════════
    print(f"\n{'-'*65}")
    print("  OUR MODEL: FusionDetectorGRL v2.1")
    print(f"{'-'*65}")

    from s3_main_grl import FusionDetectorGRL
    ckpt = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
    model = FusionDetectorGRL(n_streams=ckpt['n_streams'], n_sources=ckpt['n_sources'], n_gen=ckpt['n_gen'])
    model.load_state_dict(ckpt['model_state_dict'])
    model.to(DEVICE).eval()
    n_ours = count_params(model)

    # Full 5-stream test features
    test_feats = torch.cat([torch.load(FEAT_DIR / f"{s}_test_feats.pt", weights_only=False) for s in STREAMS], dim=1)
    loader = DataLoader(TensorDataset(test_feats, test_labels), batch_size=256, shuffle=False)
    all_probs, all_preds, all_labels = [], [], []
    with torch.no_grad():
        for fb, lb in loader:
            logits, _, _, _ = model(fb.to(DEVICE), grl_lambda=0)
            all_probs.extend(F.softmax(logits, 1)[:, 1].cpu().numpy().tolist())
            all_preds.extend(logits.argmax(1).cpu().numpy().tolist())
            all_labels.extend(lb.numpy().tolist())

    acc = accuracy_score(all_labels, all_preds) * 100
    auc = roc_auc_score(all_labels, np.array(all_probs)) * 100
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    results['Ours (GRL v2.1)'] = {'acc': round(acc, 2), 'auc': round(auc, 2), 'f1': round(f1, 4), 'params_m': round(n_ours, 1)}
    print(f"  Result: Acc={acc:.1f}% AUC={auc:.1f}% F1={f1:.4f} Params={n_ours:.1f}M")

    # ══════════════════════════════════════════════
    # Summary Table
    # ══════════════════════════════════════════════
    print(f"\n{'='*65}")
    print("  SOTA COMPARISON (Fair Conditions)")
    print(f"  Training: {TRAIN_CAP:,} samples (same for all baselines)")
    print(f"  Our model: {len(train_labels):,} samples (full train set)")
    print(f"{'='*65}")
    print(f"  {'Model':<24} | {'Acc%':>6} | {'AUC%':>6} | {'F1':>6} | {'Params':>8}")
    print(f"  {'-'*58}")
    for name in ['ResNet50 (fine-tune)', 'EfficientNet-B4', 'CLIP Linear Probe', 'Ours (GRL v2.1)']:
        r = results[name]
        print(f"  {name:<24} | {r['acc']:>6.1f} | {r['auc']:>6.1f} | {r['f1']:>6.4f} | {r['params_m']:>7.1f}M")
    print(f"  {'-'*58}")

    # Save
    with open(OUTPUT_DIR / 'sota_comparison_v2.json', 'w') as f:
        json.dump(results, f, indent=2)

    # Plot
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    names = list(results.keys())
    accs = [results[n]['acc'] for n in names]
    colors = ['#4C72B0', '#55A868', '#8172B2', '#E8770E']

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.barh(names, accs, color=colors, edgecolor='black', linewidth=0.5, height=0.6)
    for bar, val in zip(bars, accs):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                f'{val:.1f}%', ha='left', va='center', fontsize=11, fontweight='bold')
    ax.set_xlabel('Accuracy (%)', fontsize=12)
    ax.set_title(f'SOTA Comparison (Fair: {TRAIN_CAP:,} training samples for baselines)',
                 fontsize=13, fontweight='bold')
    ax.set_xlim(0, 110)
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'sota_comparison_v2.png', dpi=200)
    plt.close()

    print(f"\n  Saved: {OUTPUT_DIR / 'sota_comparison_v2.json'}")
    print(f"  Saved: {OUTPUT_DIR / 'sota_comparison_v2.png'}")
    print("\n[Done] EXP-F v2 complete.")


if __name__ == '__main__':
    main()
