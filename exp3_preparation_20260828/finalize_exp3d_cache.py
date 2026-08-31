"""Finalize one aligned EXP3D index from the three stream valid masks."""

import argparse
import csv
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--manifest', required=True)
    ap.add_argument('--cache', required=True)
    args = ap.parse_args()
    df = pd.read_csv(args.manifest)
    cache = Path(args.cache)
    streams = {'dct': 192, 'clip': 768, 'dinov2': 1024}
    summaries = []
    for split, group in df.groupby('split', sort=True):
        group = group.reset_index(drop=True)
        masks = []
        for stream, dim in streams.items():
            arr = np.load(cache / stream / f'{split}.npy', mmap_mode='r')
            valid = np.load(cache / stream / f'{split}.valid.npy')
            if arr.shape != (len(group), dim):
                raise SystemExit(f'{stream}/{split} shape={arr.shape}, expected={(len(group), dim)}')
            if valid.shape != (len(group),) or not np.isfinite(arr[valid]).all():
                raise SystemExit(f'{stream}/{split} invalid mask or non-finite values')
            masks.append(valid.astype(bool))
        all_valid = np.logical_and.reduce(masks)
        target = cache / f'index_{split}.csv'
        out = group[['row', 'path', 'generator', 'is_real']].copy()
        out['split'] = split
        out['valid_all'] = all_valid.astype(int)
        out.to_csv(target, index=False, encoding='utf-8-sig')
        summaries.append({
            'split': split, 'n': len(group),
            'valid_all': int(all_valid.sum()),
            'invalid_all': int((~all_valid).sum()),
        })
    pd.DataFrame(summaries).to_csv(cache / 'exp3d_cache_summary.csv', index=False)
    print(pd.DataFrame(summaries).to_string(index=False))
    print(f'cache={cache}')


if __name__ == '__main__':
    main()

