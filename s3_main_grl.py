"""
Stage 3 — MAIN (GRL): FusionDetector + Domain Adversarial Training
===================================================================
在原本 s3_main_train.py 的基礎上加入：

  Gradient Reversal Layer (GRL) + Generator Discriminator
  ─────────────────────────────────────────────────────────
  獎勵：binary head 分對 Real/Fake          → 正常梯度 ↑
  懲罰：generator discriminator 猜出 generator → 梯度反轉 ↓

  Backbone 被迫學習「能分 Real/Fake」但「無法辨認 generator」的特徵
  → 遇到新 generator (Midjourney) 仍能正確分類

架構：
  cached features (N×512)
    → CrossAttention + Backbone (512d shared)
    ├─ head_binary  → CE_binary  (正常梯度)
    ├─ head_source  → CE_source × λ_src  (lambda_src=0，關掉)
    └─ GRL → gen_discriminator → CE_gen × λ_grl  (梯度反轉)

  Total loss = CE_binary + λ_src×CE_source + λ_grl×CE_gen
  (GRL 使 CE_gen 梯度對 backbone 呈現反向，backbone 學到 generator-agnostic features)

λ_grl 使用 Progressive Schedule (Ganin et al., 2015)：
  λ_grl = λ_max × (2 / (1 + exp(-γ × progress)) - 1)
  訓練初期接近 0，後期穩定在 λ_max，避免早期擾亂 binary head 收斂

Checkpoint saved to: outputs/main_grl/best_model.pth
Results saved to:    outputs/main_grl/final_results.json
"""

import sys

import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Subset
from sklearn.metrics import roc_auc_score, f1_score

# ── Paths ──────────────────────────────────────────────────────────────
from config import TRAIN_CSV, VAL_CSV, FEAT_CACHE_DIR, OUTPUTS_DIR
TRAIN_CSV      = str(TRAIN_CSV)
VAL_CSV        = str(VAL_CSV)
OUTPUT_DIR     = OUTPUTS_DIR / 'main_grl'
FEAT_CACHE_DIR = FEAT_CACHE_DIR

# ── Hyperparameters ────────────────────────────────────────────────────
EPOCHS      = 30
BATCH_SIZE  = 256
LR          = 1e-3
CLIP_GRAD   = 1.0    # gradient clipping（Transformer 必要）
WARMUP_EPOCHS = 5    # Linear warmup: LR 從 1e-5 線性升到 LR

# Loss weights
LAMBDA_SRC  = 0.1    # source loss 保留（幫助 backbone 學有意義的特徵）
LAMBDA_GRL  = 0.05   # GRL adversarial loss（降低，避免 feature collapse）
GRL_GAMMA   = 10.0   # Progressive schedule 的增長速率

STREAMS = ["clip", "fft", "dct", "dire", "noise"]

GENERATOR_TO_ID = {
    'adm':        0, 'glide':    1, 'sdv4':  2,
    'sdv5':       3, 'midjourney': 4, 'wildfake': 5,
    'biggan':     6, 'vqdm':     7, 'wukong': 8,
    'firefly':    9, 'real':     10,
    'real_extra': 11, 'wildfake_ddim': 12, 'wildfake_other': 13,
    'stylegan': 14, 'dcgan': 15, 'dcgan_unseen': 16,
    'fursona_gan': 17, 'waifu_gan': 18,
}
REAL_IDS  = {GENERATOR_TO_ID['real'], GENERATOR_TO_ID['real_extra']}
REAL_ID   = GENERATOR_TO_ID['real']
N_SOURCES = len(GENERATOR_TO_ID)
# generator discriminator 只分辨 fake 的 generator（排除 real 和 real_extra）
N_GEN     = N_SOURCES - 2   # = 12 個 AI generator（排除 real + real_extra）

device = "cuda" if torch.cuda.is_available() else "cpu"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════
# Gradient Reversal Layer
# ══════════════════════════════════════════════════════════════════════

class _GRL(torch.autograd.Function):
    """Forward: 原封不動。Backward: 梯度乘以 -lambda_。"""
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


def grl_lambda(epoch: int, total_epochs: int,
               lambda_max: float, gamma: float) -> float:
    """Progressive GRL schedule（Ganin 2015）。訓練初期接近 0，後期趨近 lambda_max。"""
    progress = epoch / max(total_epochs - 1, 1)
    return lambda_max * (2.0 / (1.0 + math.exp(-gamma * progress)) - 1.0)


# ══════════════════════════════════════════════════════════════════════
# FusionDetector with GRL
# ══════════════════════════════════════════════════════════════════════

class CrossAttentionFusionLayer(nn.Module):
    def __init__(self, d_model=512, n_heads=8, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, n_heads,
                                               dropout=dropout, batch_first=True)
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
    """
    FusionDetector + GRL Generator Discriminator

    forward() 回傳：
      logits_binary:  (B, 2)       — Real/Fake
      logits_source:  (B, N_SOURCES) — generator 分類（原本的 source head）
      logits_gen:     (B, N_GEN)   — GRL generator discriminator（對 fake 樣本）
      attn_weights:   (B, N, N)
    """
    def __init__(self, n_streams, n_sources, n_gen,
                 d_model=512, n_heads=8, n_layers=2, dropout=0.3):
        super().__init__()
        self.n_streams = n_streams

        # Cross-Attention Fusion
        self.layers = nn.ModuleList([
            CrossAttentionFusionLayer(d_model, n_heads, dropout=0.1)
            for _ in range(n_layers)
        ])
        self.stream_embed = nn.Parameter(torch.randn(1, n_streams, d_model) * 0.02)

        # Shared Backbone
        self.backbone = nn.Sequential(
            nn.Linear(n_streams * d_model, 1024),
            nn.LayerNorm(1024), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(1024, 512),
            nn.LayerNorm(512),  nn.GELU(), nn.Dropout(dropout),
        )

        # Head 1: Binary classification (Real / Fake)
        self.head_binary = nn.Linear(512, 2)

        # Head 2: Source classification (原本的 DualHead，可設 λ=0 關掉)
        self.head_source = nn.Linear(512, n_sources)

        # Head 3: GRL Generator Discriminator
        #   注意：這個 head 的梯度傳回 backbone 時會被 GRL 反轉
        #   架構稍深，讓 discriminator 有能力區分 generator
        self.gen_discriminator = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, n_gen),
        )

    def forward(self, x, grl_lambda: float = 1.0):
        """
        x:           (B, N*512) — concatenated stream features
        grl_lambda:  當前的 GRL 強度（從 schedule 取得）
        """
        B = x.shape[0]
        tokens = x.view(B, self.n_streams, 512) + self.stream_embed
        attn_weights = None
        for layer in self.layers:
            tokens, attn_weights = layer(tokens)

        fused  = tokens.flatten(1)       # (B, N*512)
        shared = self.backbone(fused)    # (B, 512)

        logits_binary = self.head_binary(shared)   # (B, 2)
        logits_source = self.head_source(shared)   # (B, N_SOURCES)

        # GRL：梯度在這裡反轉，discriminator 本身正常訓練，
        # 但 backbone 收到的梯度是反向的
        reversed_shared  = grad_reverse(shared, grl_lambda)
        logits_gen       = self.gen_discriminator(reversed_shared)  # (B, N_GEN)

        return logits_binary, logits_source, logits_gen, attn_weights


# ══════════════════════════════════════════════════════════════════════
# Loss Function
# ══════════════════════════════════════════════════════════════════════

class GRLLoss(nn.Module):
    """
    Total = CE_binary
          + lambda_src  × CE_source          (通常設 0)
          + lambda_grl  × CE_gen_disc        (只對 fake 樣本計算)

    GRL 已經在 forward() 內處理梯度反轉，這裡直接加上 loss 即可。
    """
    def __init__(self, lambda_src=0.0, lambda_grl=0.5):
        super().__init__()
        self.lambda_src = lambda_src
        self.lambda_grl = lambda_grl
        self.ce = nn.CrossEntropyLoss()

    def forward(self, logits_binary, logits_source, logits_gen,
                labels_binary, labels_source, current_lambda_grl):
        # ── Binary loss（所有樣本）──────────────────────────────────
        loss_bin = self.ce(logits_binary, labels_binary)

        # ── Source loss（所有樣本，通常 λ=0 關掉）──────────────────
        loss_src = self.ce(logits_source, labels_source)

        # ── GRL Adversarial loss（只對 fake 樣本）──────────────────
        # 只有 fake 圖片（label=1）有 generator 身份
        fake_mask = labels_binary.bool()              # True = fake
        if fake_mask.sum() > 0:
            # generator id 需重新映射到 [0, N_GEN-1]（排除 real=10）
            gen_ids = labels_source[fake_mask]
            gen_ids = torch.clamp(gen_ids, 0, N_GEN - 1)
            loss_gen = self.ce(logits_gen[fake_mask], gen_ids)
        else:
            loss_gen = torch.tensor(0.0, device=logits_binary.device)

        total = (loss_bin
                 + self.lambda_src * loss_src
                 + current_lambda_grl * loss_gen)

        return total, {
            "loss_binary": loss_bin.item(),
            "loss_source": loss_src.item(),
            "loss_gen":    loss_gen.item(),
            "loss_total":  total.item(),
            "lambda_grl":  current_lambda_grl,
        }


# ══════════════════════════════════════════════════════════════════════
# Curriculum Scheduler（與原版相同）
# ══════════════════════════════════════════════════════════════════════

class CurriculumScheduler:
    """
    GRL 版不使用 curriculum（避免 Easy phase 的 real:fake 不平衡）。
    所有 epoch 都使用全部資料。
    """
    def __init__(self, difficulties):
        counts = {}
        for d in difficulties.tolist():
            counts[int(d)] = counts.get(int(d), 0) + 1
        print(f"[Curriculum] DISABLED for GRL (using all data every epoch)")
        print(f"  Difficulty distribution: {counts}")

    def get_subset(self, epoch):
        return None  # 使用全部資料

    def get_phase_name(self, epoch):
        return "all"


# ══════════════════════════════════════════════════════════════════════
# Data Loading（與原版相同）
# ══════════════════════════════════════════════════════════════════════

def load_data():
    cache, available = {}, []
    for s in STREAMS:
        tp = FEAT_CACHE_DIR / f"{s}_train_feats.pt"
        vp = FEAT_CACHE_DIR / f"{s}_val_feats.pt"
        if not (tp.exists() and vp.exists()):
            print(f"  [WARN] Missing {s} features, skipping.")
            continue
        cache[s] = {
            "train": torch.load(tp, weights_only=False),
            "val":   torch.load(vp, weights_only=False),
        }
        available.append(s)

    if not available:
        print("[ERROR] No cached features found. Run s3a first.")
        sys.exit(1)

    train_labels = torch.load(FEAT_CACHE_DIR / "train_labels.pt", weights_only=False)
    val_labels   = torch.load(FEAT_CACHE_DIR / "val_labels.pt",   weights_only=False)
    train_feats  = torch.cat([cache[s]["train"] for s in available], dim=1)
    val_feats    = torch.cat([cache[s]["val"]   for s in available], dim=1)

    # 以 features 的筆數為基準，截齊所有 tensors 和 CSV
    n_train = train_feats.shape[0]
    n_val   = val_feats.shape[0]
    train_labels = train_labels[:n_train]
    val_labels   = val_labels[:n_val]

    train_df = pd.read_csv(TRAIN_CSV)
    val_df   = pd.read_csv(VAL_CSV)
    train_df = train_df.iloc[:n_train].reset_index(drop=True)
    val_df   = val_df.iloc[:n_val].reset_index(drop=True)

    train_src = torch.tensor([
        GENERATOR_TO_ID.get(str(g).lower().strip(), REAL_ID)
        for g in train_df['generator']
    ], dtype=torch.long)
    val_src = torch.tensor([
        GENERATOR_TO_ID.get(str(g).lower().strip(), REAL_ID)
        for g in val_df['generator']
    ], dtype=torch.long)

    train_diff = torch.tensor(train_df['difficulty'].tolist(), dtype=torch.long)
    val_diff   = torch.tensor(val_df['difficulty'].tolist(),   dtype=torch.long)

    print(f"  [Data] Streams : {available} ({len(available)})")
    print(f"  [Data] Train   : {train_feats.shape[0]:,} | Val: {val_feats.shape[0]:,}")
    print(f"  [Data] Feat dim: {train_feats.shape[1]}")

    return (available, train_feats, train_labels, train_src, train_diff,
            val_feats, val_labels, val_src, val_diff)


# ══════════════════════════════════════════════════════════════════════
# Evaluation
# ══════════════════════════════════════════════════════════════════════

@torch.no_grad()
def evaluate(model, loader, criterion, current_lambda_grl):
    model.eval()
    all_labels, all_probs, all_preds = [], [], []
    total_loss, n_batches = 0.0, 0

    for feats, y_bin, y_src in loader:
        feats, y_bin, y_src = feats.to(device), y_bin.to(device), y_src.to(device)
        lb, ls, lg, _ = model(feats, grl_lambda=current_lambda_grl)
        loss, _ = criterion(lb, ls, lg, y_bin, y_src, current_lambda_grl)
        total_loss += loss.item()
        n_batches  += 1
        probs = torch.softmax(lb, dim=1)[:, 1].cpu().numpy()
        preds = lb.argmax(dim=1).cpu().numpy()
        all_probs.extend(probs)
        all_preds.extend(preds)
        all_labels.extend(y_bin.cpu().numpy())

    all_labels = np.array(all_labels)
    all_probs  = np.array(all_probs)
    all_preds  = np.array(all_preds)
    val_loss   = total_loss / max(n_batches, 1)
    acc  = 100.0 * (all_preds == all_labels).mean()
    auc  = roc_auc_score(all_labels, all_probs) if len(set(all_labels)) > 1 else 0.0
    f1   = f1_score(all_labels, all_preds, zero_division=0)
    return val_loss, acc, auc, f1


# ══════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("MAIN TRAINING — FusionDetector + GRL Domain Adversarial (v2)")
    print(f"Device   : {device}")
    print(f"Epochs   : {EPOCHS}  |  Batch: {BATCH_SIZE}  |  LR: {LR}")
    print(f"Warmup   : {WARMUP_EPOCHS} epochs (LR: {LR*0.01:.5f} -> {LR})")
    print(f"Loss     : CE_binary + {LAMBDA_SRC}*CE_source + lambda_grl*CE_gen")
    print(f"GRL      : lambda_max={LAMBDA_GRL}  gamma={GRL_GAMMA}  (progressive)")
    print(f"Curriculum: DISABLED (all data every epoch, avoid label imbalance)")
    print(f"GradClip : {CLIP_GRAD}")
    print("=" * 70)

    (available, train_feats, train_labels, train_src, train_diff,
     val_feats, val_labels, val_src, val_diff) = load_data()

    n_streams = len(available)

    val_ds     = TensorDataset(val_feats, val_labels, val_src)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    train_ds   = TensorDataset(train_feats, train_labels, train_src)

    model = FusionDetectorGRL(
        n_streams=n_streams,
        n_sources=N_SOURCES,
        n_gen=N_GEN,
    ).to(device)

    criterion = GRLLoss(lambda_src=LAMBDA_SRC, lambda_grl=LAMBDA_GRL)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)

    # Warmup + CosineAnnealing
    warmup_scheduler = optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.01, end_factor=1.0, total_iters=WARMUP_EPOCHS)
    cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS - WARMUP_EPOCHS, eta_min=LR * 0.01)
    scheduler = optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[WARMUP_EPOCHS])

    curriculum = CurriculumScheduler(train_diff)

    best_val_acc, best_val_auc, best_val_f1 = 0.0, 0.0, 0.0
    history = {
        "train_loss": [], "train_acc": [],
        "val_loss":   [], "val_acc":   [], "val_auc": [], "val_f1": [],
        "loss_binary": [], "loss_source": [], "loss_gen": [],
        "lambda_grl": [], "lr": [],
    }

    total_t0 = time.time()

    for epoch in range(EPOCHS):
        # ── GRL Lambda（Progressive Schedule）────────────────────────
        current_lambda_grl = grl_lambda(epoch, EPOCHS, LAMBDA_GRL, GRL_GAMMA)

        # ── Curriculum ────────────────────────────────────────────────
        indices    = curriculum.get_subset(epoch)
        phase_name = curriculum.get_phase_name(epoch)
        subset     = Subset(train_ds, indices) if indices is not None else train_ds
        n_samples  = len(subset)
        train_loader = DataLoader(subset, batch_size=BATCH_SIZE, shuffle=True)

        print(f"\n{'='*70}")
        print(f"Epoch {epoch+1}/{EPOCHS}  |  Phase: [{phase_name.upper()}]"
              f"  |  λ_grl: {current_lambda_grl:.4f}"
              f"  |  Samples: {n_samples:,}")
        print(f"{'='*70}")

        # ── Training ──────────────────────────────────────────────────
        model.train()
        epoch_loss = 0.0
        correct, total = 0, 0
        ep_t0 = time.time()
        loss_bin_sum, loss_src_sum, loss_gen_sum = 0.0, 0.0, 0.0

        for batch_idx, (feats, y_bin, y_src) in enumerate(train_loader):
            feats, y_bin, y_src = (feats.to(device),
                                   y_bin.to(device),
                                   y_src.to(device))

            optimizer.zero_grad()
            lb, ls, lg, _ = model(feats, grl_lambda=current_lambda_grl)
            loss, info = criterion(lb, ls, lg, y_bin, y_src, current_lambda_grl)
            loss.backward()

            # Gradient clipping（Transformer + GRL 都需要）
            nn.utils.clip_grad_norm_(model.parameters(), CLIP_GRAD)

            optimizer.step()

            epoch_loss    += loss.item()
            loss_bin_sum  += info["loss_binary"]
            loss_src_sum  += info["loss_source"]
            loss_gen_sum  += info["loss_gen"]
            correct       += lb.argmax(1).eq(y_bin).sum().item()
            total         += y_bin.size(0)

            if (batch_idx + 1) % 100 == 0 or (batch_idx + 1) == len(train_loader):
                n = batch_idx + 1
                print(f"  Batch {batch_idx+1:04d}/{len(train_loader)} | "
                      f"Loss {loss.item():.4f} "
                      f"(bin={info['loss_binary']:.4f} "
                      f"gen={info['loss_gen']:.4f} "
                      f"λ_grl={current_lambda_grl:.3f}) | "
                      f"Acc {100.*correct/total:.2f}%")

        train_loss = epoch_loss / len(train_loader)
        train_acc  = 100.0 * correct / total
        n_batches  = len(train_loader)
        scheduler.step()
        ep_elapsed = time.time() - ep_t0

        # ── Validation ────────────────────────────────────────────────
        val_loss, val_acc, val_auc, val_f1 = evaluate(
            model, val_loader, criterion, current_lambda_grl)
        current_lr = optimizer.param_groups[0]["lr"]

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_auc"].append(val_auc)
        history["val_f1"].append(val_f1)
        history["loss_binary"].append(loss_bin_sum / n_batches)
        history["loss_source"].append(loss_src_sum / n_batches)
        history["loss_gen"].append(loss_gen_sum / n_batches)
        history["lambda_grl"].append(current_lambda_grl)
        history["lr"].append(current_lr)

        print(f"\nEpoch {epoch+1}/{EPOCHS} | "
              f"Train {train_loss:.4f}/{train_acc:.2f}% | "
              f"Val {val_loss:.4f}/{val_acc:.2f}% | "
              f"AUC {val_auc:.4f} | F1 {val_f1:.4f} | "
              f"LR {current_lr:.6f} | {ep_elapsed:.0f}s")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_val_auc = val_auc
            best_val_f1  = val_f1
            torch.save({
                "epoch":              epoch,
                "model_state_dict":   model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc":            val_acc,
                "val_auc":            val_auc,
                "val_f1":             val_f1,
                "n_streams":          n_streams,
                "streams":            available,
                "n_sources":          N_SOURCES,
                "n_gen":              N_GEN,
                "lambda_grl":         LAMBDA_GRL,
                "lambda_src":         LAMBDA_SRC,
            }, OUTPUT_DIR / "best_model.pth")
            print(f"  -> Best checkpoint saved (val_acc={val_acc:.2f}%)")

    total_elapsed = time.time() - total_t0

    print("\n" + "=" * 70)
    print("MAIN TRAINING COMPLETE")
    print(f"Total time   : {total_elapsed/60:.1f} min")
    print(f"Best Val Acc : {best_val_acc:.2f}%")
    print(f"Best AUC     : {best_val_auc:.4f}")
    print(f"Best F1      : {best_val_f1:.4f}")
    print("=" * 70)

    with open(OUTPUT_DIR / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    # ── Final evaluation on best checkpoint ───────────────────────────
    print("\nFinal evaluation (best checkpoint)...")
    best_ckpt = torch.load(OUTPUT_DIR / "best_model.pth",
                           map_location=device, weights_only=False)
    model.load_state_dict(best_ckpt["model_state_dict"])
    _, final_acc, final_auc, final_f1 = evaluate(
        model, val_loader, criterion, LAMBDA_GRL)

    print(f"Final Val Acc : {final_acc:.2f}%")
    print(f"Final Val AUC : {final_auc:.4f}")
    print(f"Final Val F1  : {final_f1:.4f}")

    final_results = {
        "best_epoch":     int(best_ckpt["epoch"]) + 1,
        "val_acc":        round(final_acc, 4),
        "val_auc":        round(final_auc, 4),
        "val_f1":         round(final_f1, 4),
        "total_time_min": round(total_elapsed / 60, 2),
        "config": {
            "epochs":       EPOCHS,
            "batch_size":   BATCH_SIZE,
            "lr":           LR,
            "lambda_src":   LAMBDA_SRC,
            "lambda_grl":   LAMBDA_GRL,
            "grl_gamma":    GRL_GAMMA,
            "clip_grad":    CLIP_GRAD,
            "streams":      available,
            "n_streams":    n_streams,
            "n_sources":    N_SOURCES,
            "n_gen":        N_GEN,
            "curriculum":   "easy(0-9)->medium(10-19)->hard(20+)",
        },
        "history": history,
    }
    with open(OUTPUT_DIR / "final_results.json", "w") as f:
        json.dump(final_results, f, indent=2)
    print(f"Results saved -> {OUTPUT_DIR / 'final_results.json'}")


if __name__ == "__main__":
    main()