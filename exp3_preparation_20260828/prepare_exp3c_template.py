"""Create an explicit, blank content-label template for EXP3C."""

import argparse
from pathlib import Path

import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cache-dir', required=True)
    ap.add_argument('--out-dir', required=True)
    args = ap.parse_args()
    cache = Path(args.cache_dir)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    frames = []
    for split in ('val', 'cross_generator_test'):
        path = cache / f'index_{split}.csv'
        df = pd.read_csv(path)
        df = df[['row', 'path', 'generator', 'is_real', 'split']].copy()
        df['label'] = 1 - df['is_real'].astype(int)
        frames.append(df)
    result = pd.concat(frames, ignore_index=True).drop_duplicates('path')
    result['content'] = ''
    result['content_label_source'] = ''
    result['annotator'] = ''
    result['review_status'] = 'unreviewed'
    result.to_csv(out / 'exp3c_content_labels_template.csv', index=False)
    pd.DataFrame([{
        'allowed_content_labels': 'portrait|landscape|building|animal|object|other',
        'required_review_status': 'reviewed',
        'note': 'Do not infer content from filename; fill from reliable labels or manual review.'
    }]).to_csv(out / 'exp3c_label_policy.csv', index=False)
    print(f'rows={len(result)} out={out}')


if __name__ == '__main__':
    main()
