"""
Ablation Study v2.1
LOO (Leave-One-Out) per-stream ablation experiment.

Runs 6 conditions:
  MAIN : all 5 streams
  -CLIP: remove clip
  -FFT : remove fft
  -DCT : remove dct
  -DIRE: remove dire
  -Noise: remove noise

For each condition, reports binary val_acc and AUC.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, accuracy_score
import numpy as np
from typing import Dict, List, Optional
from pathlib import Path
import json
from tqdm import tqdm


STREAM_NAMES = ["clip", "fft", "dct", "dire", "noise"]


# ──────────────────────────────────────────────────────────────────────
# LOO-compatible model wrapper
# ──────────────────────────────────────────────────────────────────────

class LOOWrapper(nn.Module):
    """
    Wraps AIImageDetector and zeros out one stream's contribution.
    Used for Leave-One-Out ablation WITHOUT retraining.

    (For proper ablation, retrain with the stream removed.
     This wrapper is used for fast ablation estimation.)
    """

    def __init__(self, model, excluded_stream: Optional[str] = None):
        super().__init__()
        self.model = model
        self.excluded = excluded_stream

    def forward(self, images: torch.Tensor):
        # Extract features from all streams
        stream_features = {}
        for name, ext in self.model.extractors.items():
            if name == self.excluded:
                # Replace with zeros
                feat = ext(images)
                stream_features[name] = torch.zeros_like(feat)
            else:
                stream_features[name] = ext(images)

        fused, attn = self.model.fusion(stream_features)
        shared = self.model.backbone(fused)
        return self.model.head_binary(shared), self.model.head_source(shared), attn


# ──────────────────────────────────────────────────────────────────────
# Evaluation helper
# ──────────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_model(model: nn.Module, loader: DataLoader, device: str) -> Dict:
    """
    Evaluate a model on val/test set.

    Returns dict with: acc, auc, n_samples
    """
    all_probs = []
    all_preds = []
    all_labels = []

    model.eval()
    for batch in tqdm(loader, desc="Evaluating", leave=False):
        if len(batch) >= 2:
            images, y_bin = batch[0], batch[1]
        images  = images.to(device)
        y_bin   = y_bin.to(device)

        outputs = model(images)
        logits_binary = outputs[0]  # (B, 2)

        probs = F.softmax(logits_binary, dim=1)[:, 1].cpu().numpy()  # P(Fake)
        preds = logits_binary.argmax(dim=1).cpu().numpy()
        labels = y_bin.cpu().numpy()

        all_probs.extend(probs.tolist())
        all_preds.extend(preds.tolist())
        all_labels.extend(labels.tolist())

    acc = accuracy_score(all_labels, all_preds) * 100
    try:
        auc = roc_auc_score(all_labels, all_probs) * 100
    except ValueError:
        auc = float("nan")

    return {"acc": acc, "auc": auc, "n_samples": len(all_labels)}


# ──────────────────────────────────────────────────────────────────────
# LOO Ablation
# ──────────────────────────────────────────────────────────────────────

def run_loo_ablation(
    model,
    val_loader: DataLoader,
    device: str = "cuda",
    save_path: Optional[str] = None,
) -> Dict:
    """
    Run Leave-One-Out ablation study.

    Args:
        model: trained AIImageDetector
        val_loader: validation DataLoader
        device: 'cuda' or 'cpu'
        save_path: optional JSON output path

    Returns:
        results dict
    """
    results = {}

    # MAIN: all streams
    print("\n[Ablation] MAIN (all streams)")
    results["MAIN"] = evaluate_model(model, val_loader, device)
    print(f"  ACC={results['MAIN']['acc']:.2f}%  AUC={results['MAIN']['auc']:.2f}%")

    # LOO: remove one stream at a time
    for stream in STREAM_NAMES:
        cond = f"-{stream.upper()}"
        print(f"\n[Ablation] {cond}")
        wrapper = LOOWrapper(model, excluded_stream=stream).to(device)
        results[cond] = evaluate_model(wrapper, val_loader, device)
        delta_acc = results[cond]["acc"] - results["MAIN"]["acc"]
        delta_auc = results[cond]["auc"] - results["MAIN"]["auc"]
        print(f"  ACC={results[cond]['acc']:.2f}% (D{delta_acc:+.2f}%)  "
              f"AUC={results[cond]['auc']:.2f}% (D{delta_auc:+.2f}%)")

    # Print summary table
    print("\n" + "="*60)
    print(f"{'Condition':<12} {'ACC%':>8} {'D ACC':>8} {'AUC%':>8} {'D AUC':>8}")
    print("-"*60)
    main_acc = results["MAIN"]["acc"]
    main_auc = results["MAIN"]["auc"]
    for cond, res in results.items():
        da = res["acc"] - main_acc
        du = res["auc"] - main_auc
        print(f"{cond:<12} {res['acc']:>8.2f} {da:>+8.2f} {res['auc']:>8.2f} {du:>+8.2f}")
    print("="*60)

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n[Ablation] Results saved -> {save_path}")

    return results


# ──────────────────────────────────────────────────────────────────────
# Cross-Generator Generalization (EXP-C)
# ──────────────────────────────────────────────────────────────────────

def run_cross_generator_eval(
    model,
    generator_loaders: Dict[str, DataLoader],
    device: str = "cuda",
    save_path: Optional[str] = None,
) -> Dict:
    """
    Evaluate on each AI generator separately (GenImage EXP-C).

    generator_loaders: {'midjourney': loader, 'stable-diffusion': loader, ...}
    """
    results = {}
    for gen_name, loader in generator_loaders.items():
        print(f"\n[CrossGen] Evaluating on {gen_name} ...")
        results[gen_name] = evaluate_model(model, loader, device)
        print(f"  ACC={results[gen_name]['acc']:.2f}%  AUC={results[gen_name]['auc']:.2f}%")

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n[CrossGen] Results saved -> {save_path}")

    return results


# ──────────────────────────────────────────────────────────────────────
# Quick test
# ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("ablation_study.py — import OK")
    print("Use run_loo_ablation(model, val_loader, device) to run EXP-B.")
    print("Use run_cross_generator_eval(model, gen_loaders, device) for EXP-C.")
