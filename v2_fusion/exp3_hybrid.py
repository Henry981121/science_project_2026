"""EXP3 evaluation for the frozen EXP2A Hybrid checkpoint.

This script evaluates only what the current handoff cache can support:
EXP3A basic detection, EXP3B generator grouping, and EXP3E calibration.
It never trains, changes the checkpoint, or reads a locked test split.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score, average_precision_score, balanced_accuracy_score,
    brier_score_loss, f1_score, roc_auc_score,
)
from torch.utils.data import DataLoader

from .config import EVAL_ONLY_GENERATORS, STREAMS
from .data import FusionFeatureDataset, collate, load_split
from .model import FusionDetectorV2


def metric_dict(y, p, threshold):
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    pred = (p >= threshold).astype(int)
    pos = max(int((y == 1).sum()), 1)
    neg = max(int((y == 0).sum()), 1)
    return {
        'n': int(len(y)),
        'n_real': int((y == 0).sum()),
        'n_fake': int((y == 1).sum()),
        'accuracy': float(accuracy_score(y, pred)),
        'balanced_accuracy': float(balanced_accuracy_score(y, pred)),
        'auroc': float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else None,
        'ap': float(average_precision_score(y, p)) if len(np.unique(y)) == 2 else None,
        'f1': float(f1_score(y, pred, zero_division=0)),
        'fpr': float(((pred == 1) & (y == 0)).sum() / neg),
        'fnr': float(((pred == 0) & (y == 1)).sum() / pos),
    }


def ece(y, p, bins=10):
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = 0.0
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (p >= lo) & ((p < hi) if hi < 1 else (p <= hi))
        if not mask.any():
            continue
        confidence = float(p[mask].mean())
        accuracy = float(y[mask].mean())
        count = int(mask.sum())
        total += count / len(y) * abs(confidence - accuracy)
        rows.append({'lo': float(lo), 'hi': float(hi), 'n': count,
                     'confidence': confidence, 'accuracy': accuracy})
    return float(total), rows


@torch.no_grad()
def predict(model, bundle, batch_size, device):
    loader = DataLoader(FusionFeatureDataset(bundle, list(STREAMS)),
                        batch_size=batch_size, shuffle=False,
                        collate_fn=collate, num_workers=0)
    probs, labels = [], []
    for feats, y, _ in loader:
        feats = {k: v.to(device) for k, v in feats.items()}
        out = model(feats)
        probs.append(torch.softmax(out['logits_binary'], 1)[:, 1].cpu().numpy())
        labels.append(y.numpy())
    return np.concatenate(labels), np.concatenate(probs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cache-dir', required=True)
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--batch-size', type=int, default=256)
    args = ap.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    cache = Path(args.cache_dir)
    checkpoint_path = Path(args.checkpoint)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = checkpoint['run_config']['model']
    from .config import ModelConfig
    model = FusionDetectorV2(ModelConfig(**cfg)).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    bundles = {
        'validation': load_split(cache, 'val', list(STREAMS), drop_invalid=True),
        'cross_generator_test': load_split(
            cache, 'cross_generator_test', list(STREAMS), drop_invalid=True),
    }
    predictions = {}
    for split, bundle in bundles.items():
        y, p = predict(model, bundle, args.batch_size, device)
        predictions[split] = (y, p)

    # The existing Hybrid protocol uses argmax, equivalent to threshold 0.5.
    threshold = 0.5
    result = {
        'experiment': 'EXP3_Hybrid',
        'checkpoint': str(checkpoint_path),
        'device': device,
        'feature_streams': list(STREAMS),
        'threshold': threshold,
        'locked_test_used': False,
        'supported': ['EXP3A', 'EXP3B', 'EXP3E'],
        'not_run': {
            'EXP3C': 'no reliable content labels in current cache',
            'EXP3D': 'no multi-severity transformed-image feature cache',
            'EXP3F': 'no comparison baseline models and cost protocol',
            'EXP3G': 'no post-freeze open-world external dataset',
        },
        'splits': {},
    }
    train_seen = set(['real', 'real_extra', 'adm', 'glide', 'sdv4', 'sdv5',
                      'midjourney', 'wildfake', 'stylegan', 'dcgan'])
    for split, (y, p) in predictions.items():
        metrics = metric_dict(y, p, threshold)
        cal, bins = ece(y, p)
        metrics.update({'ece': cal, 'brier': float(brier_score_loss(y, p)),
                        'reliability_bins': bins})
        df = bundles[split]['df'].copy()
        by_generator = {}
        for gen in sorted(df['generator'].astype(str).str.lower().unique()):
            mask = df['generator'].astype(str).str.lower().to_numpy() == gen
            by_generator[gen] = metric_dict(y[mask], p[mask], threshold)
        entry = {'metrics': metrics, 'by_generator': by_generator}
        if split == 'cross_generator_test':
            names = df['generator'].astype(str).str.lower().to_numpy()
            seen = np.isin(names, list(train_seen))
            unseen = np.isin(names, list(EVAL_ONLY_GENERATORS)) | np.isin(
                names, ['real', 'real_extra'])
            entry['seen'] = metric_dict(y[seen], p[seen], threshold)
            entry['unseen'] = metric_dict(y[unseen], p[unseen], threshold)
            entry['generalization_gap_auroc'] = (
                entry['seen']['auroc'] - entry['unseen']['auroc']
                if entry['seen']['auroc'] is not None and entry['unseen']['auroc'] is not None
                else None)
        result['splits'][split] = entry

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps({
        'out': str(out), 'device': device,
        'validation_n': result['splits']['validation']['metrics']['n'],
        'cross_n': result['splits']['cross_generator_test']['metrics']['n'],
        'cross_auroc': result['splits']['cross_generator_test']['metrics']['auroc'],
        'cross_unseen_auroc': result['splits']['cross_generator_test']['unseen']['auroc'],
        'locked_test_used': False,
    }, indent=2))


if __name__ == '__main__':
    main()
