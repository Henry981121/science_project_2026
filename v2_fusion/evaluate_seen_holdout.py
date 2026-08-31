"""Evaluate the reserved seen-generator holdout with a retrained checkpoint."""

import argparse
import json
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from .config import ModelConfig, STREAMS
from .data import FusionFeatureDataset, collate, load_split
from .model import FusionDetectorV2
from .train import evaluate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cache-dir', required=True)
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--holdout-manifest', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--batch-size', type=int, default=256)
    args = ap.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = FusionDetectorV2(ModelConfig(**checkpoint['run_config']['model'])).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    holdout = pd.read_csv(args.holdout_manifest)
    full = load_split(Path(args.cache_dir), 'train', list(STREAMS), drop_invalid=True)
    paths = set(holdout['path'].astype(str))
    mask = full['df']['path'].astype(str).isin(paths).to_numpy()
    if int(mask.sum()) != len(holdout):
        raise RuntimeError(f'holdout rows mismatch: manifest={len(holdout)} cache={int(mask.sum())}')
    bundle = dict(full)
    bundle['feats'] = {s: x[mask] for s, x in full['feats'].items()}
    bundle['labels'] = full['labels'][mask]
    bundle['source_ids'] = full['source_ids'][mask]
    bundle['df'] = full['df'].loc[mask].reset_index(drop=True)
    bundle['n'] = int(mask.sum())
    loader = DataLoader(FusionFeatureDataset(bundle, list(STREAMS)),
                        batch_size=args.batch_size, shuffle=False,
                        collate_fn=collate, num_workers=0)
    metrics = evaluate(model, loader, device, list(STREAMS))
    result = {
        'experiment': 'EXP3B_seen_generator_holdout',
        'checkpoint': str(Path(args.checkpoint)),
        'holdout_manifest': str(Path(args.holdout_manifest)),
        'n': len(holdout), 'device': device, 'metrics': metrics,
        'locked_test_used': False, 'train_holdout_overlap': 0,
        'note': 'The checkpoint was retrained using train_excluding_seen_holdout.csv.',
    }
    Path(args.out).write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
