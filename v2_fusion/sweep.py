"""
v2_fusion.sweep — 三軸因子設計的批次執行器
===========================================

回答的問題不是「哪一種 fusion 最準」，而是「fusion 設計的哪一個維度真的重要」。
所以這裡跑的是一個 2×2×2 的因子設計，不是一串有名字的方法：

    A. order      1 = 只有加權和            2 = 加權和 + 交互項
    B. weighting  static = 學到的常數權重    gated = 權重隨樣本而變
    C. level      feature = d 維表示上融合   decision = 各流 logits 上融合

外加參照組 concat（樸素做法）與 single（單流，回答「融合到底有沒有用」）。

三條讓比較有效的紀律，都由這支 script 強制執行
------------------------------------------------
1. **參數量對齊**：每個 cell 只有一個自由度 `width`，用二分搜尋調到同一個
   參數預算（實測落在預算的 95-100%）。沒有這一步，「二階比較好」與
   「二階比較大」永遠分不開。
2. **只換那一個模組**：adapter、分類頭以外的一切、學習率、epoch、
   early stopping 準則，所有 cell 共用 —— 全部來自 config.py 的預設值。
3. **多 seed**：一次跑 2 分鐘，沒有理由只跑一次。預設 5 個 seed 並報信賴區間。

主指標
------
`gap_auc = val_auc − test_auc`（泛化落差）。**不是準確率** —— val 已經
飽和在 0.999，八個 cell 會全部擠在小數點後三位，什麼都分不出來。

用法
----
    # 先跑這個。20 分鐘，決定值不值得跑滿。
    python -m v2_fusion.sweep --cache-dir <handoff>/feature_data/crop --stage pilot

    # pilot 過關後
    python -m v2_fusion.sweep --cache-dir <handoff>/feature_data/crop --stage grid
    python -m v2_fusion.sweep --cache-dir <handoff>/feature_data/crop --stage single
    python -m v2_fusion.sweep --cache-dir <handoff>/feature_data/crop --stage full

中斷了直接重跑同一行指令 —— 已完成的 run 會被跳過（--force 可強制重跑）。
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import replace
from pathlib import Path
from typing import Dict, List, Optional

import torch

from .config import (RunConfig, ModelConfig, LossConfig, TrainConfig,
                     STREAMS, STREAM_DIMS)
from .data import load_split, describe_split
from .fusion_cells import all_cell_ids, is_grid_cell, spec_from_cell_id
from .train import run_one, resolve_cache


# ══════════════════════════════════════════════════════════════════════
# 實驗階段
# ══════════════════════════════════════════════════════════════════════
#
# 預算是「融合模組本身」的參數量，不含共用的 adapter（約 1.02M）與 head。
# 三個點跨越 16 倍，足以看出兩條曲線是平行（純粹容量差異）還是交叉
# （機制真的不同）。

BUDGETS = (50_000, 200_000, 800_000)
DEFAULT_BUDGET = 200_000
DEFAULT_SEEDS = (42, 43, 44, 45, 46)

# pilot 只跑 A 軸的兩端，其餘全部固定。它回答的不是「哪個好」，
# 而是「這個實驗有沒有解析度」—— 兩組信賴區間如果完全重疊，
# 跑滿八格也只是八條一樣的線，該先去解決解析度而不是硬跑。
PILOT_CELLS = ('o1.static.feat', 'o2.static.feat')


def build_runs(stage: str, streams: List[str], budgets: List[int],
               seeds: List[int], epochs: Optional[int]) -> List[RunConfig]:
    runs: List[RunConfig] = []

    def add(cell: str, budget: int, seed: int, run_streams: List[str], tag: str):
        cfg = RunConfig(
            name=tag,
            model=ModelConfig(streams=list(run_streams),
                              fusion_cell=cell, cell_budget=budget),
            loss=LossConfig(lambda_src=0.0),
            train=TrainConfig(seed=seed),
        )
        if epochs:
            cfg.train.epochs = epochs
        runs.append(cfg)

    def b_tag(b: int) -> str:
        return f"b{b // 1000}k"

    if stage == 'pilot':
        for cell in PILOT_CELLS:
            for seed in seeds:
                add(cell, DEFAULT_BUDGET, seed, streams,
                    f"{cell}__{b_tag(DEFAULT_BUDGET)}__s{seed}")

    elif stage == 'grid':
        for cell in all_cell_ids():
            for seed in seeds:
                add(cell, DEFAULT_BUDGET, seed, streams,
                    f"{cell}__{b_tag(DEFAULT_BUDGET)}__s{seed}")

    elif stage == 'full':
        for cell in all_cell_ids():
            for budget in budgets:
                for seed in seeds:
                    add(cell, budget, seed, streams,
                        f"{cell}__{b_tag(budget)}__s{seed}")

    elif stage == 'single':
        # 單流參照。order 必須是 1 —— 一條流沒有交互項可言，
        # FusionCell 會在 order=2 且 n_streams<2 時 raise，那是刻意的。
        for s in streams:
            for seed in seeds:
                add('o1.static.feat', DEFAULT_BUDGET, seed, [s],
                    f"single-{s}__{b_tag(DEFAULT_BUDGET)}__s{seed}")

    else:
        raise SystemExit(f"未知的 stage '{stage}'。可用：pilot, grid, full, single")

    return runs


# ══════════════════════════════════════════════════════════════════════
# 整理成一列
# ══════════════════════════════════════════════════════════════════════

CSV_FIELDS = [
    'run', 'cell_id', 'order', 'weighting', 'level', 'in_factorial',
    'streams', 'budget', 'width', 'fusion_params', 'total_params', 'seed',
    'best_epoch', 'minutes',
    'val_acc', 'val_auc', 'test_acc', 'test_auc', 'test_fnr', 'test_fpr',
    'gap_auc', 'gap_acc',
    'interact_scale', 'attn_std_mean', 'attn_learned',
]


def flatten(result: Dict) -> Dict:
    cfg = result['config']
    mcfg = cfg['model']
    cell = mcfg.get('fusion_cell')
    spec = spec_from_cell_id(cell, mcfg.get('cell_width', 0)) if cell else None
    val = result.get('val', {})
    test = result.get('test')

    row = {
        'run': result['name'],
        'cell_id': cell or f"legacy:{mcfg.get('fusion_mode')}",
        'order': spec.order if spec and is_grid_cell(cell) else '',
        'weighting': spec.weighting if spec and is_grid_cell(cell) else '',
        'level': spec.level if spec and is_grid_cell(cell) else '',
        # 參照組不進因子分析 —— concat 沒有三軸座標，硬給它一組會污染回歸。
        'in_factorial': int(bool(cell) and is_grid_cell(cell)),
        'streams': '+'.join(mcfg.get('streams', [])),
        'budget': mcfg.get('cell_budget') or '',
        'width': mcfg.get('cell_width') or '',
        'fusion_params': result.get('params', {}).get('fusion', ''),
        'total_params': result.get('params', {}).get('total', ''),
        'seed': cfg['train']['seed'],
        'best_epoch': result.get('best_epoch', ''),
        'minutes': result.get('minutes', ''),
        'val_acc': val.get('acc', ''),
        'val_auc': val.get('auc', ''),
        'interact_scale': result.get('fusion_diagnostics', {}).get('interact_scale', ''),
    }

    if test:
        row.update({
            'test_acc': test.get('acc', ''), 'test_auc': test.get('auc', ''),
            'test_fnr': test.get('fnr', ''), 'test_fpr': test.get('fpr', ''),
            # 主指標。val 已經飽和，只有落差有區辨力。
            'gap_auc': round(val['auc'] - test['auc'], 6)
                       if val.get('auc') is not None and test.get('auc') is not None else '',
            'gap_acc': round(val['acc'] - test['acc'], 4)
                       if val.get('acc') is not None and test.get('acc') is not None else '',
        })
        sd = test.get('readout_attn_std', {})
        row['attn_std_mean'] = (round(sum(sd.values()) / len(sd), 6) if sd else '')
        row['attn_learned'] = int(bool(test.get('readout_attn_learned', True)))
    else:
        for k in ('test_acc', 'test_auc', 'test_fnr', 'test_fpr', 'gap_auc', 'gap_acc'):
            row[k] = ''
        sd = val.get('readout_attn_std', {})
        row['attn_std_mean'] = (round(sum(sd.values()) / len(sd), 6) if sd else '')
        row['attn_learned'] = int(bool(val.get('readout_attn_learned', True)))

    return row


# ══════════════════════════════════════════════════════════════════════
# 摘要
# ══════════════════════════════════════════════════════════════════════

# n 很小（預設 5），常態近似會低估區間寬度，所以用 t 分布的 95% 臨界值。
T95 = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571,
       7: 2.447, 8: 2.365, 9: 2.306, 10: 2.262}


def mean_ci(xs: List[float]) -> tuple:
    n = len(xs)
    if n == 0:
        return float('nan'), float('nan')
    m = sum(xs) / n
    if n == 1:
        return m, float('nan')
    sd = math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1))
    t = T95.get(n, 1.96)
    return m, t * sd / math.sqrt(n)


def summarise(rows: List[Dict], metric: str = 'gap_auc') -> None:
    groups: Dict[tuple, List[Dict]] = {}
    for r in rows:
        groups.setdefault((r['cell_id'], r['streams'], r['budget']), []).append(r)

    print("\n" + "=" * 96)
    print(f"摘要 —— 主指標 {metric}（val_auc − test_auc，**越小越好**）")
    print("=" * 96)
    print(f"{'cell':<18}{'streams':<22}{'budget':>9}{'n':>4}"
          f"{'mean':>11}{'95% CI':>13}{'test_auc':>11}{'attn_sd':>10}{'i_scale':>9}")
    print("-" * 96)

    ordered = sorted(groups.items(),
                     key=lambda kv: mean_ci([r[metric] for r in kv[1]
                                             if r[metric] != ''])[0])
    for (cell, streams, budget), rs in ordered:
        vals = [r[metric] for r in rs if r[metric] != '']
        auc = [r['test_auc'] for r in rs if r['test_auc'] != '']
        sd = [r['attn_std_mean'] for r in rs if r['attn_std_mean'] != '']
        isc = [r['interact_scale'] for r in rs if r['interact_scale'] != '']
        m, ci = mean_ci(vals)
        am, _ = mean_ci(auc)
        note = '' if rs[0]['attn_learned'] else '  (attn 是硬填的 1/N)'
        print(f"{cell:<18}{streams:<22}{str(budget):>9}{len(rs):>4}"
              f"{m:>11.4f}{'±' + format(ci, '.4f'):>13}"
              f"{am:>11.4f}"
              f"{(mean_ci(sd)[0] if sd else float('nan')):>10.4f}"
              f"{(mean_ci(isc)[0] if isc else float('nan')):>9.3f}{note}")

    print("-" * 96)
    print("attn_sd = readout 權重的逐樣本標準差。趨近 0 → 那組權重其實是常數，")
    print("          gating/attention 相對於固定加權平均沒有增益。")
    print("i_scale = 交互項的學到權重。趨近 0 → 模型自己認為二階交互沒用。")
    print("信賴區間互相重疊 = 這兩格分不出高下，不要在論文裡宣稱誰贏。")


# ══════════════════════════════════════════════════════════════════════

def main() -> int:
    ap = argparse.ArgumentParser(description="v2_fusion 三軸因子設計 sweep")
    ap.add_argument('--stage', default='pilot',
                    choices=['pilot', 'grid', 'full', 'single'])
    ap.add_argument('--cache-dir', help='handoff 的 feature_data/crop 目錄')
    ap.add_argument('--train-split', default='train')
    ap.add_argument('--val-split', default='val')
    ap.add_argument('--test-split', default='cross_generator_test')
    ap.add_argument('--streams', default=','.join(STREAMS))
    ap.add_argument('--keep-invalid', action='store_true')
    ap.add_argument('--seeds', default=','.join(str(s) for s in DEFAULT_SEEDS))
    ap.add_argument('--budgets', default=','.join(str(b) for b in BUDGETS))
    ap.add_argument('--epochs', type=int)
    ap.add_argument('--out', default='outputs/v2_sweep')
    ap.add_argument('--force', action='store_true',
                    help='重跑已完成的 run（預設跳過，方便中斷後續跑）')
    ap.add_argument('--dry-run', action='store_true',
                    help='只列出會跑哪些 run 與各自的參數量，不訓練')
    args = ap.parse_args()

    streams = [s.strip() for s in args.streams.split(',') if s.strip()]
    seeds = [int(s) for s in args.seeds.split(',') if s.strip()]
    budgets = [int(b) for b in args.budgets.split(',') if b.strip()]
    out_root = Path(args.out) / args.stage
    out_root.mkdir(parents=True, exist_ok=True)

    runs = build_runs(args.stage, streams, budgets, seeds, args.epochs)

    print("=" * 72)
    print(f"v2_fusion sweep —— stage={args.stage}")
    print("=" * 72)
    print(f"cells      : {sorted({r.model.fusion_cell for r in runs})}")
    print(f"streams    : {streams}")
    print(f"seeds      : {seeds}")
    print(f"runs       : {len(runs)}")
    print(f"out        : {out_root}")

    if args.dry_run:
        from .model import FusionDetectorV2
        print("\n--dry-run：各 cell 在各預算下的實際參數量")
        print(f"{'run':<44}{'width':>7}{'fusion':>12}{'total':>12}")
        seen = set()
        for cfg in runs:
            key = (cfg.model.fusion_cell, cfg.model.cell_budget,
                   tuple(cfg.model.streams))
            if key in seen:
                continue
            seen.add(key)
            m = FusionDetectorV2(cfg.model)
            p = m.n_parameters()
            print(f"{cfg.name:<44}{m.cell_width:>7}{p['fusion']:>12,}{p['total']:>12,}")
        return 0

    cache = resolve_cache(args)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"device     : {device}"
          + (f"  ({torch.cuda.get_device_name(0)})" if device == 'cuda' else ""))

    split_names = {'train': args.train_split, 'val': args.val_split}
    if args.test_split and args.test_split.lower() != 'none':
        split_names['test'] = args.test_split
    else:
        print("\n[warn] 沒有 test split —— 主指標 gap_auc 算不出來，"
              "整個 sweep 只會有 val 數字，那是選模用的、帶選擇偏誤。")

    bundles = {k: load_split(cache, n, streams, drop_invalid=not args.keep_invalid)
               for k, n in split_names.items()}
    print()
    for k, b in bundles.items():
        print(describe_split(b, split_names[k]))

    dims = dict(bundles['train']['dims'])

    rows: List[Dict] = []
    for i, cfg in enumerate(runs, 1):
        # streams 由 build_runs 指定（單流參照組會不同），dims 一律以實際 cache 為準。
        cfg.model = replace(cfg.model, stream_dims=dims)
        done = out_root / cfg.name / 'results.json'
        print(f"\n[{i}/{len(runs)}] {cfg.name}")
        if done.exists() and not args.force:
            print("  已完成，跳過（--force 可強制重跑）")
            rows.append(flatten(json.loads(done.read_text())))
            continue
        rows.append(flatten(run_one(cfg, bundles, cache, device, out_root)))

        # 每跑完一筆就落地，中途斷掉不會丟失已跑的結果。
        with open(out_root / 'sweep_results.csv', 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
            w.writeheader()
            w.writerows(rows)

    with open(out_root / 'sweep_results.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        w.writerows(rows)
    with open(out_root / 'sweep_results.json', 'w') as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)

    summarise(rows)
    print(f"\n-> {out_root/'sweep_results.csv'}")

    if args.stage == 'pilot':
        print("\n" + "=" * 72)
        print("pilot 怎麼判讀")
        print("=" * 72)
        print("兩格的 95% 信賴區間**分得開** → 實驗有解析度，接著跑 --stage grid。")
        print("**完全重疊**       → 跑滿八格也只是八條一樣的線。先去處理解析度")
        print("                     （更難的評測子集、或先做合成流干預確認實驗")
        print("                     有偵測能力），不要硬跑。")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
