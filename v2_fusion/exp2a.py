"""EXP2A: compare four fusion modes on fixed DCT/CLIP/DINOv2 caches."""

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader

from .config import ModelConfig, RunConfig, STREAMS, STREAM_DIMS
from .data import FusionFeatureDataset, collate, describe_split, load_split
from .losses import FusionLoss
from .model import FusionDetectorV2
from .train import evaluate, set_seed

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')


MODES = {
    'hybrid': 'Self-Attention + Cross-Attention',
    'self': 'Self-Attention only',
    'cross': 'Cross-Attention only',
    'concat': 'Concatenation + MLP control',
}


def train_one(mode, bundles, device, out_root, epochs, batch_size):
    cfg = RunConfig(name=mode, model=ModelConfig(
        streams=list(STREAMS), stream_dims=dict(STREAM_DIMS), fusion_mode=mode))
    set_seed(cfg.train.seed)
    out_dir = out_root / mode
    out_dir.mkdir(parents=True, exist_ok=True)
    loaders = {k: DataLoader(
        FusionFeatureDataset(b, cfg.model.streams), batch_size=batch_size,
        shuffle=(k == 'train'), collate_fn=collate, num_workers=0)
        for k, b in bundles.items()}
    model = FusionDetectorV2(cfg.model).to(device)
    loss_fn = FusionLoss(cfg.loss)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr,
                                  weight_decay=cfg.train.weight_decay)
    warm = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.01, end_factor=1.0,
        total_iters=cfg.train.warmup_epochs)
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(epochs - cfg.train.warmup_epochs, 1),
        eta_min=cfg.train.lr * 0.01)
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, [warm, cosine], milestones=[cfg.train.warmup_epochs])
    history, best = [], {'metric': -1.0}
    started = time.time()
    print(f"\nRUN EXP2A/{mode}: {MODES[mode]}")
    print(model.describe())
    for epoch in range(epochs):
        model.train(); correct = total = batches = 0
        sums = {'loss_total': 0.0, 'loss_binary': 0.0, 'loss_source': 0.0}
        for feats, y_bin, y_src in loaders['train']:
            feats = {k: v.to(device) for k, v in feats.items()}
            y_bin, y_src = y_bin.to(device), y_src.to(device)
            optimizer.zero_grad()
            output = model(feats)
            loss, info = loss_fn(output, y_bin, y_src)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.train.clip_grad)
            optimizer.step()
            for key in sums: sums[key] += info[key]
            correct += int((output['logits_binary'].argmax(1) == y_bin).sum())
            total += y_bin.numel(); batches += 1
        scheduler.step()
        for key in sums: sums[key] /= max(batches, 1)
        val = evaluate(model, loaders['val'], device, cfg.model.streams)
        record = {'epoch': epoch + 1, 'train_acc': 100 * correct / total,
                  **sums, 'val_acc': val['acc'], 'val_auc': val['auc'],
                  'val_f1': val['f1']}
        history.append(record)
        print(f"  ep {epoch + 1:>2}/{epochs} | train {record['train_acc']:.2f}% | "
              f"val {val['acc']:.2f}% AUC {val['auc']:.4f}")
        if val['auc'] > best['metric']:
            best = {'metric': val['auc'], 'epoch': epoch + 1, 'val': val}
            torch.save({'epoch': epoch + 1,
                        'model_state_dict': model.state_dict(),
                        'run_config': cfg.to_dict(), 'val': val},
                       out_dir / 'best_model.pth')
    checkpoint = torch.load(out_dir / 'best_model.pth', map_location=device,
                            weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    test = evaluate(model, loaders['test'], device, cfg.model.streams)
    result = {'name': mode, 'description': MODES[mode], 'config': cfg.to_dict(),
              'best_epoch': best['epoch'],
              'minutes': round((time.time() - started) / 60, 2),
              'val': {**best['val'], 'selection_biased': True},
              'test': {**test, 'selection_biased': False}, 'history': history}
    (out_dir / 'results.json').write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding='utf-8')
    return result


def main():
    parser = argparse.ArgumentParser(description='EXP2A fusion ablation')
    parser.add_argument('--cache-dir', required=True)
    parser.add_argument('--train-manifest', help='Optional path-filtered train manifest')
    parser.add_argument('--out', default='outputs/exp2a')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch-size', type=int, default=256)
    args = parser.parse_args()
    cache = Path(args.cache_dir)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    splits = {'train': 'train', 'val': 'val', 'test': 'cross_generator_test'}
    bundles = {key: load_split(cache, split, list(STREAMS), drop_invalid=True)
               for key, split in splits.items() if key != 'train'}
    if args.train_manifest:
        requested = pd.read_csv(args.train_manifest)
        requested_paths = set(requested['path'].astype(str))
        full_train = load_split(cache, 'train', list(STREAMS), drop_invalid=True)
        keep = full_train['df']['path'].astype(str).isin(requested_paths).to_numpy()
        if int(keep.sum()) != len(requested_paths):
            raise RuntimeError(
                f'train manifest path mismatch: manifest={len(requested_paths)} '
                f'cache={int(keep.sum())}')
        train_bundle = dict(full_train)
        train_bundle['feats'] = {s: x[keep] for s, x in full_train['feats'].items()}
        train_bundle['labels'] = full_train['labels'][keep]
        train_bundle['source_ids'] = full_train['source_ids'][keep]
        train_bundle['df'] = full_train['df'].loc[keep].reset_index(drop=True)
        train_bundle['n'] = int(keep.sum())
        train_bundle['n_raw'] = int(keep.sum())
        train_bundle['n_invalid'] = 0
        bundles['train'] = train_bundle
    else:
        bundles['train'] = load_split(cache, 'train', list(STREAMS), drop_invalid=True)
    print(f'EXP2A device: {device}')
    for key, bundle in bundles.items(): print(describe_split(bundle, splits[key]))
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    results = [train_one(mode, bundles, device, out_root, args.epochs,
                         args.batch_size) for mode in MODES]
    (out_root / 'comparison.json').write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding='utf-8')
    print('\nEXP2A comparison (cross_generator_test)')
    for result in results:
        metrics = result['test']
        print(f"{result['name']:<8} acc={metrics['acc']:.2f} "
              f"auc={metrics['auc']:.4f} fnr={metrics['fnr']:.2f} "
              f"fpr={metrics['fpr']:.2f}")


if __name__ == '__main__':
    main()
