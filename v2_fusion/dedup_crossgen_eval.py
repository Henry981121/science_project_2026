"""
在 cache 自己的座標系內，對 cross_generator_test 去重後重評。

背景
----
`index_cross_generator_test.csv` 有 34,461 列但只有 28,729 個 unique path
（5,732 筆重複）。`outputs/exp3_retrain_hybrid_clean/results.json` 裡的
cross_generator_test 指標是在含重複的全量上算的。

為什麼不用 exp3f 的 cross_generator_test_valid.csv
--------------------------------------------------
那份 manifest 走的是原始 `C:\\datasets\\genimage\\...` 路徑，而 cache index
走 `C:\\datasets\\genimage_norm_crop\\...`，是兩套路徑空間。而且
`prepare_same_data_retrain.py` 只處理 path 字串，不帶 cache 的列序資訊 ——
它是給 EXP3F 那批直接讀圖的 SOTA 模型用的，本來就不能索引 feature cache。

這支腳本全程使用 cache 自己的列序，不做任何 basename 配對。

先驗證，再計算
--------------
`genimage_norm_crop` 的 "crop" 意味著重複列有兩種可能：

  (a) 逐位元相同 → 純粹的 CSV 重複，去重安全
  (b) 不同       → 同一張圖的多個 crop，去重會丟資料

兩者處置相反，所以先驗。(b) 就停下來報告，不產生任何指標 ——
寧可沒有數字，也不要一個說不清來源的數字。

用法
----
    python -m v2_fusion.dedup_crossgen_eval `
      --cache-dir $CACHE `
      --checkpoint outputs\\exp3_retrain_hybrid_clean\\best_model.pth `
      --out outputs\\exp3_retrain_hybrid_clean\\cross_gen_dedup_results.json
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .config import ModelConfig, STREAMS
from .data import FusionFeatureDataset, collate, load_split
from .model import FusionDetectorV2
from .train import evaluate


def survey_splits(cache_dir):
    """掃每個 split 的 index CSV，看各自內部有沒有重複 path。

    cross_generator_test 有 5,732 筆重複是查出來的，但 train / val 有沒有
    同樣的問題從來沒人看過。如果有，val 的指標會被同一張圖重複計分扭曲，
    而 val 正是拿來選 checkpoint 的那個 split。

    只讀 CSV、不載入特徵，幾秒鐘。locked test 一律跳過。
    """
    out = {}
    for csv_path in sorted(Path(cache_dir).glob('index_*.csv')):
        split = csv_path.stem.replace('index_', '')
        if split == 'test':
            continue
        d = pd.read_csv(csv_path)
        key = d['path'].astype(str).str.lower()
        extra = key.duplicated(keep='first').to_numpy()
        is_real = d['is_real'].to_numpy()
        out[split] = {
            'n_rows': int(len(d)),
            'n_unique_path': int(key.nunique()),
            'n_duplicate_rows': int(extra.sum()),
            'duplicate_rows_real_side': int((extra & (is_real == 1)).sum()),
            'duplicate_rows_fake_side': int((extra & (is_real == 0)).sum()),
        }
    return out


def inspect_duplicates(bundle, streams, tol):
    """檢查重複 path 的各列：欄位是否一致、特徵是否數值等價。

    判準不是 bitwise identical。同一張圖跑兩次前向，GPU 的 reduction 順序
    不保證一致，float32 會有 ULP 級別的差異（值域 ±10 時約 1e-6）。那是
    數值噪音，不是不同的圖。真的是不同 crop 的話，CLIP embedding 的差異
    在 1e-2 以上 —— 中間隔了四個數量級，tol=1e-4 兩邊都留足安全邊界。
    """
    df = bundle['df'].reset_index(drop=True)
    key = df['path'].astype(str).str.lower()
    # groupby().indices 一次拿到所有位置；逐 key 過濾是 O(組數 × 列數)，會慢好幾分鐘
    groups = {k: idx for k, idx in df.groupby(key, sort=False).indices.items()
              if len(idx) > 1}

    field_conflicts, noise, exceeding = [], [], []
    n_identical = 0
    worst = 0.0
    for k, idx in groups.items():
        rows = df.iloc[idx]
        if rows['is_real'].nunique() > 1 or rows['generator'].nunique() > 1:
            field_conflicts.append({
                'path': str(rows['path'].iloc[0]), 'rows': idx.tolist(),
                'is_real': rows['is_real'].unique().tolist(),
                'generator': rows['generator'].unique().tolist()})
        gmax, gstream = 0.0, None
        for s in streams:
            block = bundle['feats'][s][idx]
            d = float((block - block[:1]).abs().max())
            if d > gmax:
                gmax, gstream = d, s
        worst = max(worst, gmax)
        if gmax == 0.0:
            n_identical += 1
        else:
            rec = {'path': str(rows['path'].iloc[0]), 'stream': gstream,
                   'rows': idx.tolist(), 'max_abs_diff': gmax}
            (exceeding if gmax > tol else noise).append(rec)

    is_real = df['is_real'].to_numpy()
    dup_extra = (~key.duplicated(keep='first')).to_numpy()
    return {
        'n_rows': int(len(df)),
        'n_unique_path': int(key.nunique()),
        'n_duplicate_rows': int(len(df) - key.nunique()),
        'n_duplicated_paths': int(len(groups)),
        'duplicate_rows_real_side': int(((~dup_extra) & (is_real == 1)).sum()),
        'duplicate_rows_fake_side': int(((~dup_extra) & (is_real == 0)).sum()),
        'field_conflicts': field_conflicts[:20],
        'n_field_conflicts': len(field_conflicts),
        'feature_tol': tol,
        'n_bitwise_identical': n_identical,
        'n_numeric_noise': len(noise),
        'numeric_noise_examples': sorted(
            noise, key=lambda r: -r['max_abs_diff'])[:10],
        'feature_diffs_exceeding_tol': exceeding[:20],
        'n_exceeding_tol': len(exceeding),
        'max_abs_diff_overall': worst,
        'duplicates_are_equivalent': len(exceeding) == 0,
    }


def take(bundle, mask):
    m = torch.from_numpy(mask)
    out = dict(bundle)
    out['feats'] = {s: x[m] for s, x in bundle['feats'].items()}
    out['labels'] = bundle['labels'][m]
    out['source_ids'] = bundle['source_ids'][m]
    out['df'] = bundle['df'].reset_index(drop=True).loc[mask].reset_index(drop=True)
    out['n'] = int(mask.sum())
    return out


def run_eval(model, bundle, streams, device, batch_size):
    loader = DataLoader(FusionFeatureDataset(bundle, streams),
                        batch_size=batch_size, shuffle=False,
                        collate_fn=collate, num_workers=0)
    return evaluate(model, loader, device, streams)


def per_generator(model, bundle, streams, device, batch_size):
    """EXP3B 要求每個 generator 分開報。真圖報 FPR，假圖報 FNR。"""
    df = bundle['df']
    out = {}
    for gen in sorted(df['generator'].astype(str).unique()):
        mask = (df['generator'].astype(str) == gen).to_numpy()
        sub = take(bundle, mask)
        m = run_eval(model, sub, streams, device, batch_size)
        is_real = bool(df.loc[mask, 'is_real'].iloc[0] == 1)
        out[gen] = {
            'n': int(mask.sum()),
            'side': 'real' if is_real else 'fake',
            'error_rate': m['fpr'] if is_real else m['fnr'],
            'acc': m['acc'],
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cache-dir', required=True)
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--split', default='cross_generator_test')
    ap.add_argument('--batch-size', type=int, default=256)
    ap.add_argument('--feature-tol', type=float, default=1e-4,
                    help='重複列的特徵差異容忍上限。ULP 級的浮點噪音（~1e-6）'
                         '視為同一張圖；不同 crop 會落在 1e-2 以上')
    ap.add_argument('--force', action='store_true',
                    help='特徵差異超過 tol 時仍照常去重評分（預設拒絕）')
    args = ap.parse_args()

    streams = list(STREAMS)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    survey = survey_splits(Path(args.cache_dir))
    print('各 split 的內部重複（只讀 CSV）：')
    for sp, s in survey.items():
        print(f"  {sp:24s} rows={s['n_rows']:>7,}  unique={s['n_unique_path']:>7,}"
              f"  dup={s['n_duplicate_rows']:>6,}"
              f"  (real {s['duplicate_rows_real_side']:,} /"
              f" fake {s['duplicate_rows_fake_side']:,})")
    print()

    bundle = load_split(Path(args.cache_dir), args.split, streams, drop_invalid=True)

    diag = inspect_duplicates(bundle, streams, args.feature_tol)
    print(json.dumps(diag, indent=2, ensure_ascii=False))

    if diag['n_field_conflicts']:
        raise SystemExit(
            f"\n同一個 path 的不同列上，is_real 或 generator 互相矛盾"
            f"（{diag['n_field_conflicts']} 組）。這是資料錯誤，不是重複，"
            f"必須先查清楚，不去重也不評分。")

    if not diag['duplicates_are_equivalent'] and not args.force:
        raise SystemExit(
            f"\n有 {diag['n_exceeding_tol']} 組重複 path 的特徵差異超過 "
            f"{args.feature_tol}（最大 {diag['max_abs_diff_overall']:.3e}）。\n"
            f"這個量級不是浮點噪音，比較像同一張圖的不同 crop。\n"
            f"去重會丟掉有效資料，改用 image-level 聚合才對。\n"
            f"沒有產生任何指標 —— 把 feature_diffs_exceeding_tol 貼回來再決定。")

    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = FusionDetectorV2(ModelConfig(**checkpoint['run_config']['model'])).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    key = bundle['df'].reset_index(drop=True)['path'].astype(str).str.lower()
    keep = (~key.duplicated(keep='first')).to_numpy()
    deduped = take(bundle, keep)

    before = run_eval(model, bundle, streams, device, args.batch_size)
    after = run_eval(model, deduped, streams, device, args.batch_size)

    def balance(b):
        y = b['labels'].numpy() if torch.is_tensor(b['labels']) else np.asarray(b['labels'])
        return {'n': int(len(y)), 'real': int((y == 0).sum()), 'fake': int((y == 1).sum())}

    result = {
        'experiment': 'cross_generator_test_dedup_in_cache_coordinates',
        'split': args.split,
        'checkpoint': str(Path(args.checkpoint)),
        'split_duplicate_survey': survey,
        'diagnosis': diag,
        'class_balance': {'with_duplicates': balance(bundle),
                          'deduplicated': balance(deduped)},
        'metrics_with_duplicates': before,
        'metrics_deduplicated': after,
        'per_generator_deduplicated': per_generator(
            model, deduped, streams, device, args.batch_size),
        'selection_biased': False,
        'locked_test_used': False,
        'note': ('去重在 cache 自己的列序上進行，未使用外部 manifest，'
                 '未做 basename 配對。metrics_deduplicated 是應該引用的數字。'),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=2, ensure_ascii=False),
                              encoding='utf-8')

    print(f"\n{'':16s}{'含重複':>12s}{'去重後':>12s}")
    for k in ('acc', 'auc', 'f1', 'fnr', 'fpr'):
        print(f"  {k:14s}{before[k]:>12.4f}{after[k]:>12.4f}")
    print(f"\n真假比例  含重複 {balance(bundle)}  去重後 {balance(deduped)}")
    print(f"\n寫入 {args.out}")


if __name__ == '__main__':
    main()
