"""Evaluate all frozen EXP2A fusion models on the 16-condition EXP3D cache."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (accuracy_score, average_precision_score,
                             balanced_accuracy_score, f1_score, roc_auc_score)
from torch.utils.data import DataLoader

from .config import STREAMS
from .data import FusionFeatureDataset, collate, load_split
from .model import FusionDetectorV2
from .config import ModelConfig


def metrics(y, p, threshold=0.5):
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    pred = (p >= threshold).astype(int)
    pos = max(int((y == 1).sum()), 1)
    neg = max(int((y == 0).sum()), 1)
    return {
        'n': int(len(y)),
        'accuracy': float(accuracy_score(y, pred)),
        'balanced_accuracy': float(balanced_accuracy_score(y, pred)),
        'auroc': float(roc_auc_score(y, p)),
        'ap': float(average_precision_score(y, p)),
        'f1': float(f1_score(y, pred, zero_division=0)),
        'fpr': float(((pred == 1) & (y == 0)).sum() / neg),
        'fnr': float(((pred == 0) & (y == 1)).sum() / pos),
        'mean_fake_probability': float(p.mean()),
    }


@torch.no_grad()
def predict(model, bundle, batch_size, device):
    loader = DataLoader(FusionFeatureDataset(bundle, list(STREAMS)),
                        batch_size=batch_size, shuffle=False,
                        collate_fn=collate, num_workers=0)
    ys, ps = [], []
    for feats, y, _ in loader:
        feats = {k: v.to(device) for k, v in feats.items()}
        out = model(feats)
        ys.append(y.numpy())
        ps.append(torch.softmax(out['logits_binary'], dim=1)[:, 1].cpu().numpy())
    return np.concatenate(ys), np.concatenate(ps)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cache-dir', required=True)
    ap.add_argument('--checkpoint-root', required=True)
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--batch-size', type=int, default=512)
    args = ap.parse_args()

    cache = Path(args.cache_dir)
    checkpoint_root = Path(args.checkpoint_root)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    models = ['concat', 'hybrid', 'self', 'cross']
    split_files = sorted(cache.glob('index_exp3d_*.csv'))
    if len(split_files) != 16:
        raise SystemExit(f'expected 16 EXP3D indices, found {len(split_files)}')

    rows = []
    nested = {}
    for model_name in models:
        checkpoint_path = checkpoint_root / model_name / 'best_model.pth'
        checkpoint = torch.load(checkpoint_path, map_location=device,
                                weights_only=False)
        model = FusionDetectorV2(ModelConfig(**checkpoint['run_config']['model'])).to(device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        nested[model_name] = {
            'checkpoint': str(checkpoint_path), 'splits': {}
        }
        for index_path in split_files:
            split = index_path.stem.replace('index_', '')
            bundle = load_split(cache, split, list(STREAMS), drop_invalid=True)
            y, p = predict(model, bundle, args.batch_size, device)
            m = metrics(y, p)
            nested[model_name]['splits'][split] = m
            rows.append({'model': model_name, 'split': split, **m})

    result = {
        'experiment': 'EXP3D_multi_severity_robustness',
        'cache': str(cache),
        'models': models,
        'feature_streams': {'dct': 192, 'clip': 768, 'dinov2': 1024},
        'threshold': 0.5,
        'locked_test_used': False,
        'n_splits': len(split_files),
        'n_model_condition_pairs': len(rows),
        'results': nested,
    }
    (out / 'exp3d_all_models.json').write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding='utf-8')
    with (out / 'exp3d_all_models.csv').open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({
        'out': str(out), 'device': device, 'splits': len(split_files),
        'model_condition_pairs': len(rows), 'locked_test_used': False,
    }, indent=2))


if __name__ == '__main__':
    main()

