"""Build one manifest per EXP3D condition/severity for feature extraction."""

import argparse
from pathlib import Path

import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--manifest', required=True)
    ap.add_argument('--image-root', required=True)
    ap.add_argument('--out-dir', required=True)
    args = ap.parse_args()
    source = pd.read_csv(args.manifest)
    out = Path(args.out_dir)
    image_root = Path(args.image_root)
    out.mkdir(parents=True, exist_ok=True)
    rows_written = 0
    for (condition, severity), group in source.groupby(['condition', 'severity'], sort=True):
        group = group.copy()
        group['path'] = [str(image_root / str(condition) / str(severity) /
                              f"{int(row):06d}.jpg") for row in group['source_row']]
        group['split'] = f'exp3d_{condition}_{severity}'
        target = out / f'{condition}_{severity}.csv'
        group.to_csv(target, index=False)
        rows_written += len(group)
    print(f'manifests=16 rows={rows_written} out={out}')


if __name__ == '__main__':
    main()

