"""Validate that EXP3C labels are explicit, reviewed, and usable."""

import argparse
from pathlib import Path

import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--labels', required=True)
    args = ap.parse_args()
    df = pd.read_csv(args.labels)
    required = {'path', 'content', 'content_label_source', 'review_status', 'label'}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f'missing columns: {sorted(missing)}')
    allowed = {'portrait', 'landscape', 'building', 'animal', 'object', 'other'}
    labels = set(df['content'].dropna().astype(str).str.lower())
    bad = sorted(labels - allowed - {''})
    if bad:
        raise SystemExit(f'unsupported content labels: {bad}')
    if df['path'].duplicated().any():
        raise SystemExit('duplicate paths found')
    if (df['review_status'].astype(str).str.lower() != 'reviewed').any():
        raise SystemExit('all rows must be reviewed before EXP3C')
    if df['content'].isna().any() or (df['content'].astype(str).str.strip() == '').any():
        raise SystemExit('blank content labels found')
    print(f'PASS rows={len(df)} classes={sorted(labels)}')


if __name__ == '__main__':
    main()

