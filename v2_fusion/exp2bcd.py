"""EXP2B/2C/2D analysis for the frozen-feature Hybrid checkpoint."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score, f1_score
from torch.utils.data import DataLoader

from .config import ModelConfig, STREAMS
from .data import FusionFeatureDataset, collate, load_split
from .model import FusionDetectorV2


def metrics(y, p):
    pred = (p >= 0.5).astype(int)
    return {
        'acc': float(100 * (pred == y).mean()),
        'auc': float(roc_auc_score(y, p)),
        'f1': float(f1_score(y, pred, zero_division=0)),
        'fnr': float(100 * ((pred == 0) & (y == 1)).sum() / max((y == 1).sum(), 1)),
        'fpr': float(100 * ((pred == 1) & (y == 0)).sum() / max((y == 0).sum(), 1)),
    }


def load_model(checkpoint, device):
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    cfg = ModelConfig(**ckpt['run_config']['model'])
    model = FusionDetectorV2(cfg).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    return model, cfg


@torch.no_grad()
def collect(model, loader, device, perturb=None):
    ys, ps, groups, readouts, self_attns = [], [], [], [], []
    for feats, y, src in loader:
        feats = {k: v.to(device) for k, v in feats.items()}
        if perturb:
            feats[perturb] = torch.zeros_like(feats[perturb])
        out = model(feats)
        ys.append(y.numpy()); ps.append(torch.softmax(out['logits_binary'], 1)[:, 1].cpu().numpy())
        groups.append(src.numpy()); readouts.append(out['readout_attn'].cpu().numpy())
        if out['self_attn'] is not None: self_attns.append(out['self_attn'].cpu().numpy())
    return (np.concatenate(ys), np.concatenate(ps), np.concatenate(groups),
            np.concatenate(readouts), np.concatenate(self_attns) if self_attns else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cache-dir', required=True)
    ap.add_argument('--checkpoint', default='outputs/exp2a/hybrid/best_model.pth')
    ap.add_argument('--out', default='outputs/exp2bcd')
    ap.add_argument('--batch-size', type=int, default=512)
    args = ap.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, cfg = load_model(Path(args.checkpoint), device)
    cache = Path(args.cache_dir)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    splits = {'val': 'val', 'cross_generator_test': 'cross_generator_test'}
    loaders = {}
    for key, split in splits.items():
        bundle = load_split(cache, split, list(STREAMS), drop_invalid=True)
        loaders[key] = DataLoader(FusionFeatureDataset(bundle, list(STREAMS)),
                                  batch_size=args.batch_size, shuffle=False,
                                  collate_fn=collate, num_workers=0)

    # EXP2B: cache features are fixed; this is the formal frozen-feature stage.
    hybrid_result = json.loads(Path(args.checkpoint).with_name('results.json').read_text(encoding='utf-8'))
    exp2b = {
        'protocol': 'EXP2B_frozen_feature_fusion_isolation',
        'feature_extractors_trainable': False,
        'fusion_trainable': True,
        'stage2_finetune_run': False,
        'reason': 'The project uses frozen cached DCT/CLIP/DINOv2 features; no extractor fine-tuning setting exists.',
        'source_result': str(Path(args.checkpoint).with_name('results.json')),
        'val': hybrid_result['val'], 'cross_generator_test': hybrid_result['test'],
        'locked_test_used': False,
    }
    (out / 'exp2b_frozen_feature.json').write_text(json.dumps(exp2b, indent=2, ensure_ascii=False), encoding='utf-8')

    # EXP2C: attention distributions and group-wise variability.
    behavior = {'protocol': 'EXP2C_attention_behavior', 'checkpoint': str(args.checkpoint), 'splits': {}}
    perturb = {'protocol': 'EXP2D_stream_perturbation', 'checkpoint': str(args.checkpoint), 'splits': {}}
    for split, loader in loaders.items():
        y, p, src, readout, self_attn = collect(model, loader, device)
        section = {
            'n': int(len(y)), 'metrics': metrics(y, p),
            'readout_mean': {s: float(readout[:, i].mean()) for i, s in enumerate(STREAMS)},
            'readout_std': {s: float(readout[:, i].std()) for i, s in enumerate(STREAMS)},
            'groups': {},
        }
        for group in sorted(set(src.tolist())):
            mask = src == group
            section['groups'][str(int(group))] = {
                'n': int(mask.sum()),
                'readout_mean': {s: float(readout[mask, i].mean()) for i, s in enumerate(STREAMS)},
                'readout_std': {s: float(readout[mask, i].std()) for i, s in enumerate(STREAMS)},
            }
        if self_attn is not None:
            section['self_attention_mean'] = self_attn.mean(axis=0).tolist()
            section['self_attention_std'] = self_attn.std(axis=0).tolist()
        behavior['splits'][split] = section

        base = metrics(y, p)
        psec = {'n': int(len(y)), 'baseline': base, 'streams': {}}
        for stream in STREAMS:
            ym, pm, _, _, _ = collect(model, loader, device, perturb=stream)
            mm = metrics(ym, pm)
            psec['streams'][stream] = {
                'masked': mm,
                'delta_auc': float(mm['auc'] - base['auc']),
                'delta_acc': float(mm['acc'] - base['acc']),
                'mean_abs_score_change': float(np.abs(pm - p).mean()),
            }
        perturb['splits'][split] = psec
    (out / 'exp2c_attention_behavior.json').write_text(json.dumps(behavior, indent=2, ensure_ascii=False), encoding='utf-8')
    (out / 'exp2d_stream_perturbation.json').write_text(json.dumps(perturb, indent=2, ensure_ascii=False), encoding='utf-8')
    print('EXP2B/2C/2D complete; locked_test_used=False')
    for stream, value in perturb['splits']['cross_generator_test']['streams'].items():
        print(f"2D mask {stream}: auc={value['masked']['auc']:.6f} delta={value['delta_auc']:.6f}")


if __name__ == '__main__':
    main()
