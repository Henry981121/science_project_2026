"""Validate EXP3G external data isolation and required metadata."""

import argparse
from pathlib import Path

import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--manifest', required=True)
    args = ap.parse_args()
    df = pd.read_csv(args.manifest)
    required = {'external_id', 'path', 'label', 'generator_or_source', 'duplicate_hash',
                'overlap_with_train', 'overlap_with_validation', 'overlap_with_cross',
                'overlap_with_holdout', 'review_status'}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f'missing columns: {sorted(missing)}')
    if df['external_id'].duplicated().any() or df['path'].duplicated().any():
        raise SystemExit('duplicate external_id or path')
    if df['label'].isin([0, 1]).all() is False:
        raise SystemExit('label must be 0 or 1')
    overlap_cols = ['overlap_with_train', 'overlap_with_validation',
                    'overlap_with_cross', 'overlap_with_holdout']
    for col in overlap_cols:
        if df[col].astype(str).str.lower().isin({'1', 'true', 'yes'}).any():
            raise SystemExit(f'overlap found in {col}')
    if (df['review_status'].astype(str).str.lower() != 'reviewed').any():
        raise SystemExit('all external rows must be reviewed')
    print(f'PASS rows={len(df)} overlap=0')


if __name__ == '__main__':
    main()

