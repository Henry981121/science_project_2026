"""Prepare non-destructive manifests for EXP3B and EXP3D.

The script does not copy or modify source images. It writes CSV manifests with
stable row ids, source paths, labels, and perturbation parameters.
"""

import argparse
import hashlib
from pathlib import Path

import pandas as pd


def stable_holdout(df, fraction, seed):
    """Select a deterministic, generator-stratified holdout."""
    keys = (df['path'].astype(str) + '|' + df['generator'].astype(str)
            + '|' + str(seed)).map(lambda x: int(hashlib.sha256(
                x.encode('utf-8')).hexdigest()[:16], 16) / 16**16)
    chosen = []
    for generator, group in df.assign(_key=keys).groupby('generator', sort=True):
        n = max(1, int(round(len(group) * fraction)))
        chosen.append(group.nsmallest(n, '_key').index)
    holdout_idx = None
    for idx in chosen:
        holdout_idx = idx if holdout_idx is None else holdout_idx.append(idx)
    holdout = df.loc[holdout_idx].sort_values('path').drop(columns=['_key'], errors='ignore')
    remaining = df.drop(index=holdout.index).sort_values('path')
    return remaining, holdout


def build_robustness_manifest(source, out):
    conditions = {
        'jpeg': [('q95', 95), ('q75', 75), ('q50', 50), ('q30', 30)],
        'resize': [('scale0875', 0.875), ('scale075', 0.75),
                   ('scale050', 0.50), ('scale025', 0.25)],
        'blur': [('sigma0_5', 0.5), ('sigma1', 1.0),
                 ('sigma2', 2.0), ('sigma3', 3.0)],
        'noise': [('sigma2', 2.0), ('sigma5', 5.0),
                  ('sigma10', 10.0), ('sigma20', 20.0)],
    }
    rows = []
    for source_row, record in source.reset_index(drop=True).iterrows():
        base = record.to_dict()
        for condition, levels in conditions.items():
            for severity, value in levels:
                row = dict(base)
                row.update({
                    'source_row': int(source_row),
                    'condition': condition,
                    'severity': severity,
                    'severity_value': value,
                })
                rows.append(row)
    result = pd.DataFrame(rows)
    result.to_csv(out, index=False)
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cache-dir', required=True)
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--holdout-fraction', type=float, default=0.10)
    ap.add_argument('--seed', type=int, default=20260828)
    args = ap.parse_args()

    cache = Path(args.cache_dir)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    holdout_dir = out / 'seen_generator_holdout'
    robust_dir = out / 'exp3d_robustness'
    holdout_dir.mkdir(exist_ok=True)
    robust_dir.mkdir(exist_ok=True)

    train = pd.read_csv(cache / 'index_train.csv')
    cross = pd.read_csv(cache / 'index_cross_generator_test.csv')
    retrain, holdout = stable_holdout(train, args.holdout_fraction, args.seed)
    retrain.to_csv(holdout_dir / 'train_excluding_seen_holdout.csv', index=False)
    holdout.to_csv(holdout_dir / 'seen_generator_holdout.csv', index=False)

    robust = build_robustness_manifest(cross, robust_dir / 'cross_generator_multiseverity.csv')
    summary = pd.DataFrame([
        {'artifact': 'train_excluding_seen_holdout.csv', 'rows': len(retrain),
         'purpose': 'retrain Hybrid without EXP3B holdout'},
        {'artifact': 'seen_generator_holdout.csv', 'rows': len(holdout),
         'purpose': 'EXP3B seen-generator evaluation after retraining'},
        {'artifact': 'cross_generator_multiseverity.csv', 'rows': len(robust),
         'purpose': 'EXP3D: 4 conditions x 4 severity levels on cross-generator images'},
    ])
    summary.to_csv(out / 'manifest_summary.csv', index=False)

    print(summary.to_string(index=False))
    print('\nGenerator counts in seen holdout:')
    print(holdout['generator'].value_counts().sort_index().to_string())
    print('\nEXP3D condition counts:')
    print(robust.groupby(['condition', 'severity']).size().to_string())


if __name__ == '__main__':
    main()
