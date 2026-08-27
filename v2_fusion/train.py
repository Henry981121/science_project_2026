"""
v2_fusion.train — 在 cached features 上訓練 fusion
===================================================
用法（隊友端）：

    # 先看資料對不對得起來，不訓練
    python -m v2_fusion.train --cache-dir <handoff>/feature_data/crop --dry-run

    # 單一組
    python -m v2_fusion.train --cache-dir <handoff>/feature_data/crop --preset src0

    # 兩組對照
    python -m v2_fusion.train --cache-dir <handoff>/feature_data/crop --preset all

cache 佈局（2026-08-26 handoff）：
    {cache}/{stream}/{split}.npy        {cache}/index_{split}.csv

關於 --test-split（預設 cross_generator_test）：
  audit 發現 11 —— 舊版用同一個 val split 挑 checkpoint 又拿來報告數字，
  那個 98.54% 帶有選擇偏誤。這裡分開報告：
    val  = 用來挑 checkpoint 的，標記為 selection-biased
    test = 從未參與選擇的，才是可以寫進論文的數字
  handoff 裡真正上鎖的 `test` split 沒有被包進來，那是刻意的 ——
  它只能在特徵集與模型都凍結之後才打開。
"""

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, f1_score

from .config import (
    RunConfig, ModelConfig, LossConfig, TrainConfig,
    build_presets, PRESET_NOTES, STREAMS, STREAM_DIMS,
)
from .data import load_split, FusionFeatureDataset, collate, describe_split
from .model import FusionDetectorV2
from .losses import FusionLoss


# ══════════════════════════════════════════════════════════════════════

def set_seed(seed: int) -> None:
    """audit 發現 02：舊版整條路徑 grep manual_seed 零命中。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_cache(args) -> Path:
    if args.cache_dir:
        return Path(args.cache_dir)
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    try:
        import config as proj
    except ImportError:
        raise SystemExit("找不到專案根目錄的 config.py，請用 --cache-dir 指定路徑")
    return Path(proj.FEAT_CACHE_DIR)


@torch.no_grad()
def evaluate(model, loader, device, streams=None) -> Dict[str, float]:
    model.eval()
    streams = list(streams) if streams else list(model.streams)
    probs, preds, gts, attns = [], [], [], []
    for feats, y_bin, _ in loader:
        feats = {k: v.to(device) for k, v in feats.items()}
        out = model(feats)
        p = torch.softmax(out['logits_binary'], 1)[:, 1]
        probs.append(p.cpu()); preds.append(out['logits_binary'].argmax(1).cpu())
        gts.append(y_bin); attns.append(out['readout_attn'].cpu())
    probs = torch.cat(probs).numpy(); preds = torch.cat(preds).numpy()
    gts = torch.cat(gts).numpy(); attn = torch.cat(attns).mean(0).numpy()
    return {
        'acc': float(100.0 * (preds == gts).mean()),
        'auc': float(roc_auc_score(gts, probs)) if len(set(gts)) > 1 else 0.0,
        'f1':  float(f1_score(gts, preds, zero_division=0)),
        'fnr': float(100.0 * ((preds == 0) & (gts == 1)).sum() / max((gts == 1).sum(), 1)),
        'fpr': float(100.0 * ((preds == 1) & (gts == 0)).sum() / max((gts == 0).sum(), 1)),
        'readout_attn': {s: float(a) for s, a in zip(streams, attn)},
    }


def run_one(cfg: RunConfig, bundles, cache, device, out_root: Path) -> Dict:
    set_seed(cfg.train.seed)
    out_dir = out_root / cfg.name
    out_dir.mkdir(parents=True, exist_ok=True)

    streams = cfg.model.streams
    loaders = {
        k: DataLoader(FusionFeatureDataset(b, streams),
                      batch_size=cfg.train.batch_size,
                      shuffle=(k == 'train'),
                      collate_fn=collate,
                      num_workers=cfg.train.num_workers)
        for k, b in bundles.items()
    }

    model = FusionDetectorV2(cfg.model).to(device)
    crit = FusionLoss(cfg.loss)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr,
                            weight_decay=cfg.train.weight_decay)
    warm = torch.optim.lr_scheduler.LinearLR(
        opt, start_factor=0.01, end_factor=1.0, total_iters=cfg.train.warmup_epochs)
    cos = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=max(cfg.train.epochs - cfg.train.warmup_epochs, 1),
        eta_min=cfg.train.lr * 0.01)
    sched = torch.optim.lr_scheduler.SequentialLR(
        opt, [warm, cos], milestones=[cfg.train.warmup_epochs])

    print("\n" + "=" * 72)
    print(f"RUN  {cfg.name}    {PRESET_NOTES.get(cfg.name, '')}")
    print(f"     lambda_src={cfg.loss.lambda_src}  seed={cfg.train.seed}")
    print("=" * 72)
    print(model.describe())

    history, best = [], {'metric': -1.0}
    t0 = time.time()

    for epoch in range(cfg.train.epochs):
        model.train()
        agg = {'loss_total': 0.0, 'loss_binary': 0.0, 'loss_source': 0.0}
        correct = total = nb = 0

        for feats, y_bin, y_src in loaders['train']:
            feats = {k: v.to(device) for k, v in feats.items()}
            y_bin, y_src = y_bin.to(device), y_src.to(device)
            opt.zero_grad()
            out = model(feats)
            loss, info = crit(out, y_bin, y_src)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.train.clip_grad)
            opt.step()
            for k in agg:
                agg[k] += info[k]
            correct += int((out['logits_binary'].argmax(1) == y_bin).sum())
            total += y_bin.numel(); nb += 1

        sched.step()
        for k in agg:
            agg[k] /= max(nb, 1)
        val = evaluate(model, loaders['val'], device, streams)
        metric = val['auc'] if cfg.train.select_metric == 'val_auc' else val['acc'] / 100.0

        rec = {'epoch': epoch + 1, 'lr': opt.param_groups[0]['lr'],
               'train_acc': 100.0 * correct / total, **agg,
               'val_acc': val['acc'], 'val_auc': val['auc'], 'val_f1': val['f1']}
        history.append(rec)

        print(f"  ep {epoch+1:>2}/{cfg.train.epochs} | "
              f"loss {agg['loss_total']:.4f} (bin {agg['loss_binary']:.4f} "
              f"src {agg['loss_source']:.4f}) | "
              f"train {rec['train_acc']:.2f}% | val {val['acc']:.2f}% "
              f"AUC {val['auc']:.4f}")

        if metric > best['metric']:
            best = {'metric': metric, 'epoch': epoch + 1, 'val': val}
            torch.save({'epoch': epoch + 1,
                        'model_state_dict': model.state_dict(),
                        'run_config': cfg.to_dict(),
                        'val': val},
                       out_dir / 'best_model.pth')

    elapsed = (time.time() - t0) / 60.0
    ckpt = torch.load(out_dir / 'best_model.pth', map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])

    result = {
        'name': cfg.name,
        'config': cfg.to_dict(),                  # 直接來自 dataclass，不是手寫字串
        'best_epoch': best['epoch'],
        'minutes': round(elapsed, 2),
        'val': {**best['val'],
                'selection_biased': True,
                'note': '此 split 同時用於挑 checkpoint，數字帶有選擇偏誤，不應作為效能報告'},
        'history': history,
    }
    if 'test' in loaders:
        result['test'] = {**evaluate(model, loaders['test'], device, streams),
                          'selection_biased': False,
                          'note': '未參與 checkpoint 選擇，可作為效能報告'}

    with open(out_dir / 'results.json', 'w') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"  -> best epoch {best['epoch']} | {elapsed:.1f} min | {out_dir/'results.json'}")
    ra = best['val']['readout_attn']
    print(f"  -> readout attention: " + "  ".join(f"{k} {v:.3f}" for k, v in ra.items()))
    return result


# ══════════════════════════════════════════════════════════════════════

def main() -> int:
    ap = argparse.ArgumentParser(description="v2 fusion training")
    ap.add_argument('--preset', default='src0',
                    help="preset 名稱或 'all'。可用：" + ', '.join(build_presets()))
    ap.add_argument('--cache-dir', help='handoff 的 feature_data/crop 目錄')
    ap.add_argument('--train-split', default='train')
    ap.add_argument('--val-split', default='val')
    ap.add_argument('--test-split', default='cross_generator_test',
                    help="從未參與 checkpoint 選擇的 split；'none' 表示不載入")
    ap.add_argument('--streams', default=','.join(STREAMS))
    ap.add_argument('--keep-invalid', action='store_true',
                    help='不剔除 valid mask 為 False 的列（預設剔除）')
    ap.add_argument('--epochs', type=int); ap.add_argument('--batch-size', type=int)
    ap.add_argument('--lr', type=float);   ap.add_argument('--seed', type=int)
    ap.add_argument('--dry-run', action='store_true', help='只檢查資料對齊，不訓練')
    ap.add_argument('--out', default='outputs/v2_fusion')
    args = ap.parse_args()

    streams = [s.strip() for s in args.streams.split(',') if s.strip()]
    cache = resolve_cache(args)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print("=" * 72)
    print("v2_fusion — 訓練（GRL 已移除）")
    print("=" * 72)
    print(f"device     : {device}"
          + (f"  ({torch.cuda.get_device_name(0)})" if device == 'cuda' else ""))
    print(f"cache_dir  : {cache}")
    print(f"streams    : {streams}")

    split_names = {'train': args.train_split, 'val': args.val_split}
    if args.test_split and args.test_split.lower() != 'none':
        split_names['test'] = args.test_split
    else:
        print("test_split : (未載入 —— val 數字將標記為 selection-biased)")

    bundles = {
        key: load_split(cache, name, streams, drop_invalid=not args.keep_invalid)
        for key, name in split_names.items()
    }
    print()
    for k, b in bundles.items():
        print(describe_split(b, split_names[k]))

    dims = dict(bundles['train']['dims'])
    for s in streams:
        if dims[s] != STREAM_DIMS.get(s):
            print(f"  [note] {s} 的 cache 維度是 {dims[s]}，"
                  f"config.STREAM_DIMS 預期 {STREAM_DIMS.get(s)} —— 以實際 cache 為準")

    if args.dry_run:
        print("\n--dry-run：資料對齊檢查通過，未進行訓練。")
        return 0

    presets = build_presets()
    names = list(presets) if args.preset == 'all' else [args.preset]
    for n in names:
        if n not in presets:
            raise SystemExit(f"未知的 preset '{n}'。可用：{', '.join(presets)}")

    out_root = Path(args.out)
    results = []
    for n in names:
        cfg = presets[n]
        cfg.model = ModelConfig(streams=streams, stream_dims=dims)
        if args.epochs:     cfg.train.epochs = args.epochs
        if args.batch_size: cfg.train.batch_size = args.batch_size
        if args.lr:         cfg.train.lr = args.lr
        if args.seed:       cfg.train.seed = args.seed
        results.append(run_one(cfg, bundles, cache, device, out_root))

    if len(results) > 1:
        print("\n" + "=" * 72)
        print("對照總表")
        print("=" * 72)
        key = 'test' if 'test' in results[0] else 'val'
        print(f"{'preset':<18}{'λ_src':>7}"
              f"{'acc':>9}{'auc':>9}{'fnr':>8}{'fpr':>8}   ({key})")
        for r in results:
            m, c = r[key], r['config']['loss']
            print(f"{r['name']:<18}{c['lambda_src']:>7}"
                  f"{m['acc']:>9.2f}{m['auc']:>9.4f}{m['fnr']:>8.2f}{m['fpr']:>8.2f}")
        with open(out_root / 'comparison.json', 'w') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\n-> {out_root/'comparison.json'}")
        if key == 'val':
            print("\n⚠️  上表是 val split，同時被用來挑 checkpoint，帶有選擇偏誤。")
            print("   要能寫進論文的數字，請用 --test-split 指定一個沒參與選擇的 split。")
    return 0


if __name__ == '__main__':
    sys.exit(main())
