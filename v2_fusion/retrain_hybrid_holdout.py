"""Retrain Hybrid after reserving the EXP3B seen-generator holdout."""

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .config import ModelConfig, RunConfig, STREAMS, STREAM_DIMS
from .data import FusionFeatureDataset, collate, load_split
from .losses import FusionLoss
from .model import FusionDetectorV2
from .train import evaluate, set_seed


class RunLog:
    def __init__(self, path):
        self.fp = Path(path).open('w', encoding='utf-8')

    def write(self, message):
        print(message, flush=True)
        self.fp.write(message + '\n')
        self.fp.flush()

    def close(self):
        self.fp.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cache-dir', required=True)
    ap.add_argument('--train-manifest', required=True)
    ap.add_argument('--holdout-manifest', required=True)
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--epochs', type=int, default=30)
    ap.add_argument('--batch-size', type=int, default=256)
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    log = RunLog(out / 'training.log')
    started = time.time()
    try:
        cache = Path(args.cache_dir)
        train_df = __import__('pandas').read_csv(args.train_manifest)
        holdout_df = __import__('pandas').read_csv(args.holdout_manifest)
        train_paths = set(train_df['path'].astype(str))
        holdout_paths = set(holdout_df['path'].astype(str))
        overlap = train_paths & holdout_paths
        if overlap:
            raise RuntimeError(f'train/holdout path overlap: {len(overlap)}')

        # Read the complete aligned cache first, then subset by path. Never
        # shorten the CSV before loading the feature arrays.
        full_train = load_split(cache, 'train', list(STREAMS), drop_invalid=True)
        keep = ~full_train['df']['path'].astype(str).isin(holdout_paths).to_numpy()
        if int(keep.sum()) != len(train_paths):
            raise RuntimeError(
                f'expected {len(train_paths)} retained paths, got {int(keep.sum())}')
        train_bundle = dict(full_train)
        train_bundle['feats'] = {
            s: t[keep] for s, t in full_train['feats'].items()
        }
        train_bundle['labels'] = full_train['labels'][keep]
        train_bundle['source_ids'] = full_train['source_ids'][keep]
        train_bundle['df'] = full_train['df'].loc[keep].reset_index(drop=True)
        train_bundle['n'] = int(keep.sum())
        train_bundle['n_raw'] = int(keep.sum())
        train_bundle['n_invalid'] = 0
        bundles = {
            'train': train_bundle,
            'val': load_split(cache, 'val', list(STREAMS), drop_invalid=True),
            'test': load_split(cache, 'cross_generator_test', list(STREAMS),
                               drop_invalid=True),
        }
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        cfg = RunConfig(name='hybrid_retrained_excluding_seen_holdout',
                        model=ModelConfig(streams=list(STREAMS),
                                          stream_dims=dict(STREAM_DIMS),
                                          fusion_mode='hybrid'))
        set_seed(cfg.train.seed)
        loaders = {k: DataLoader(
            FusionFeatureDataset(b, list(STREAMS)), batch_size=args.batch_size,
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
            optimizer, T_max=max(args.epochs - cfg.train.warmup_epochs, 1),
            eta_min=cfg.train.lr * 0.01)
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer, [warm, cosine], milestones=[cfg.train.warmup_epochs])
        log.write('EXP3B Hybrid retraining: holdout excluded')
        log.write(f'device={device} epochs={args.epochs} batch_size={args.batch_size}')
        log.write(f'train_n={bundles["train"]["n"]} val_n={bundles["val"]["n"]} '
                  f'cross_n={bundles["test"]["n"]}')
        log.write(f'holdout_n={len(holdout_df)} overlap={len(overlap)}')
        log.write(f'streams={STREAMS} dims={STREAM_DIMS}')
        history, best = [], {'metric': -1.0}
        for epoch in range(args.epochs):
            model.train()
            sums = {'loss_total': 0.0, 'loss_binary': 0.0, 'loss_source': 0.0}
            correct = total = batches = 0
            for feats, y_bin, y_src in loaders['train']:
                feats = {k: v.to(device) for k, v in feats.items()}
                y_bin, y_src = y_bin.to(device), y_src.to(device)
                optimizer.zero_grad()
                output = model(feats)
                loss, info = loss_fn(output, y_bin, y_src)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), cfg.train.clip_grad)
                optimizer.step()
                for key in sums:
                    sums[key] += info[key]
                correct += int((output['logits_binary'].argmax(1) == y_bin).sum())
                total += y_bin.numel(); batches += 1
            scheduler.step()
            for key in sums:
                sums[key] /= max(batches, 1)
            val = evaluate(model, loaders['val'], device, list(STREAMS))
            record = {'epoch': epoch + 1, 'lr': optimizer.param_groups[0]['lr'],
                      'train_acc': 100.0 * correct / total, **sums,
                      'val_acc': val['acc'], 'val_auc': val['auc'], 'val_f1': val['f1']}
            history.append(record)
            log.write('epoch={epoch} train_acc={train_acc:.4f} loss={loss_total:.6f} '
                      'val_acc={val_acc:.4f} val_auc={val_auc:.8f} val_f1={val_f1:.6f}'
                      .format(**record))
            if val['auc'] > best['metric']:
                best = {'metric': val['auc'], 'epoch': epoch + 1, 'val': val}
                torch.save({'epoch': epoch + 1,
                            'model_state_dict': model.state_dict(),
                            'run_config': cfg.to_dict(), 'val': val,
                            'holdout_excluded': True}, out / 'best_model.pth')

        checkpoint = torch.load(out / 'best_model.pth', map_location=device,
                                weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        cross = evaluate(model, loaders['test'], device, list(STREAMS))
        result = {
            'experiment': 'EXP3B_Hybrid_retrain_excluding_seen_holdout',
            'config': cfg.to_dict(), 'device': device,
            'train_n': bundles['train']['n'], 'val_n': bundles['val']['n'],
            'cross_generator_test_n': bundles['test']['n'],
            'seen_holdout_n': len(holdout_df), 'train_holdout_overlap': len(overlap),
            'best_epoch': best['epoch'], 'minutes': round((time.time() - started) / 60, 2),
            'val': {**best['val'], 'selection_biased': True},
            'cross_generator_test': {**cross, 'selection_biased': False},
            'locked_test_used': False, 'history': history,
            'checkpoint': str(out / 'best_model.pth'),
        }
        (out / 'results.json').write_text(
            json.dumps(result, indent=2, ensure_ascii=False), encoding='utf-8')
        log.write(f'best_epoch={best["epoch"]} val_auc={best["val"]["auc"]:.8f}')
        log.write(f'cross_auc={cross["auc"]:.8f} cross_acc={cross["acc"]:.4f}')
        log.write('self_check=PASS locked_test_used=False train_holdout_overlap=0')
    except Exception as exc:
        log.write(f'ERROR {type(exc).__name__}: {exc}')
        raise
    finally:
        log.close()


if __name__ == '__main__':
    main()
