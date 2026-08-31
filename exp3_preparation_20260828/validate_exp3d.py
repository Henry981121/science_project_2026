"""Validate an EXP3D transformed-image/feature index."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--index', required=True)
    ap.add_argument('--feature-root', required=True)
    args = ap.parse_args()
    df = pd.read_csv(args.index)
    required = {'path', 'condition', 'severity', 'severity_value', 'row'}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f'missing columns: {sorted(missing)}')
    expected_conditions = {'jpeg', 'resize', 'blur', 'noise'}
    if set(df['condition']) != expected_conditions:
        raise SystemExit(f'conditions mismatch: {sorted(set(df["condition"]))}')
    if df.duplicated(['path', 'condition', 'severity']).any():
        raise SystemExit('duplicate transformation rows found')
    root = Path(args.feature_root)
    missing_files = []
    bad_shapes = []
    bad_values = []
    for condition, severity in df[['condition', 'severity']].drop_duplicates().itertuples(index=False):
        sub = df[(df['condition'] == condition) & (df['severity'] == severity)]
        for stream, dim in (('dct', 192), ('clip', 768), ('dinov2', 1024)):
            file = root / condition / severity / f'{stream}.npy'
            if not file.exists():
                missing_files.append(str(file))
                continue
            arr = np.load(file, mmap_mode='r')
            if arr.shape != (len(sub), dim):
                bad_shapes.append(f'{file}: got {arr.shape}, expected {(len(sub), dim)}')
            if not np.isfinite(arr).all():
                bad_values.append(str(file))
    if missing_files or bad_shapes or bad_values:
        raise SystemExit(f'missing={len(missing_files)} bad_shapes={len(bad_shapes)} bad_values={len(bad_values)}')
    print(f'PASS rows={len(df)} conditions={len(expected_conditions)} levels=16')


if __name__ == '__main__':
    main()

