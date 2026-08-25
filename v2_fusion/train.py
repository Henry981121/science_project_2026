"""
v2_fusion.train — 在 cached features 上訓練 fusion
===================================================
用法（隊友端）：

    # 先看資料對不對得起來，不訓練
    python -m v2_fusion.train --dry-run

    # 單一組
    python -m v2_fusion.train --preset src0_grl0

    # 四組對照一次跑完（每組約 6 分鐘）
    python -m v2_fusion.train --preset all

路徑預設從專案根目錄的 config.py 讀（FEAT_CACHE_DIR / TRAIN_CSV / VAL_CSV），
也可以用 --cache-dir / --train-csv / --val-csv 覆蓋。

關於 --test-csv：
  audit 發現 11 —— 舊版用同一個 val split 挑 checkpoint 又拿來報告數字，
  那個 98.54% 帶有選擇偏誤。這裡若提供 test split，會分開報告：
    val  = 用來挑 checkpoint 的，標記為 selection-biased
    test = 從未參與選擇的，才是可以寫進論文的數字
  沒提供的話，results.json 會明確把 val 標成 selection_biased=true。
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
    build_presets, PRESET_NOTES, STREAMS, STREAM_DIMS_LEGACY, STREAM_DIMS_RAW,
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


def resolve_paths(args) -> Dict[str, Optional[Path]]:
    if args.cache_dir and args.train_csv and args.val_csv:
        return {'cache': Path(args.cache_dir),
                'train': Path(args.train_csv),
                'val': Path(args.val_csv),
                'test': Path(args.test_csv) if args.test_csv else None}
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    try:
        import config as proj
    except ImportError:
        raise SystemExit(
            "找不到專案根目錄的 config.py，請改用 "
            "--cache-dir / --train-csv / --val-csv 指定路徑")
    return {'cache': Path(args.cache_dir) if args.cache_dir else Path(proj.FEAT_CACHE_DIR),
            'train': Path(args.train_csv) if args.train_csv else Path(proj.TRAIN_CSV),
            'val':   Path(args.val_csv)   if args.val_csv   else Path(proj.VAL_CSV),
            'test':  Path(args.test_csv)  if args.test_csv  else None}


@torch.no_grad()
def evaluate(model, loader, device) -> Dict[str, float]:
    model.eval()
    probs, preds, gts, attns = [], [], [], []
    for feats, y_bin, _, _ in loader:
        feats = {k: v.to(device) for k, v in feats.items()}
        out = model(feats, grl_lambda=0.0)
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
        'readout_attn': {s: float(a) for s, a in zip(STREAMS, attn)},
    }


def run_one(cfg: RunConfig, bundles, paths, device, out_root: Path) -> Dict:
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
    print(f"     lambda_src={cfg.loss.lambda_src}  lambda_grl={cfg.loss.lambda_grl}"
          f"  ({cfg.loss.grl_schedule})  seed={cfg.train.seed}")
    print("=" * 72)
    print(model.describe())

    history, best = [], {'metric': -1.0}
    t0 = time.time()

    for epoch in range(cfg.train.epochs):
        lam = cfg.loss.grl_lambda_at(epoch, cfg.train.epochs)
        model.train()
        agg = {'loss_total': 0.0, 'loss_binary': 0.0, 'loss_source': 0.0, 'loss_gen': 0.0}
        correct = total = nb = 0

        for feats, y_bin, y_src, y_gen in loaders['train']:
            feats = {k: v.to(device) for k, v in feats.items()}
            y_bin, y_src, y_gen = y_bin.to(device), y_src.to(device), y_gen.to(device)
            opt.zero_grad()
            out = model(feats, grl_lambda=lam)
            loss, info = crit(out, y_bin, y_src, y_gen)
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
        val = evaluate(model, loaders['val'], device)
        metric = val['auc'] if cfg.train.select_metric == 'val_auc' else val['acc'] / 100.0

        rec = {'epoch': epoch + 1, 'lambda_grl': lam, 'lr': opt.param_groups[0]['lr'],
               'train_acc': 100.0 * correct / total, **agg,
               'val_acc': val['acc'], 'val_auc': val['auc'], 'val_f1': val['f1']}
        history.append(rec)

        print(f"  ep {epoch+1:>2}/{cfg.train.epochs} | λ_grl {lam:.4f} | "
              f"loss {agg['loss_total']:.4f} (bin {agg['loss_binary']:.4f} "
              f"src {agg['loss_source']:.4f} gen {agg['loss_gen']:.4f}) | "
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
        result['test'] = {**evaluate(model, loaders['test'], device),
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
    ap.add_argument('--preset', default='src0_grl0',
                    help="preset 名稱或 'all'。可用：" + ', '.join(build_presets()))
    ap.add_argument('--cache-dir'); ap.add_argument('--train-csv')
    ap.add_argument('--val-csv');   ap.add_argument('--test-csv')
    ap.add_argument('--streams', default=','.join(STREAMS))
    ap.add_argument('--raw-dims', action='store_true',
                    help='cache 存的是 projection 之前的原始維度（CLIP 1024 / DIRE 2048）')
    ap.add_argument('--epochs', type=int); ap.add_argument('--batch-size', type=int)
    ap.add_argument('--lr', type=float);   ap.add_argument('--seed', type=int)
    ap.add_argument('--require-manifest', action='store_true',
                    help='沒有 {split}_paths.json 就失敗（重抽特徵後建議打開）')
    ap.add_argument('--dry-run', action='store_true', help='只檢查資料對齊，不訓練')
    ap.add_argument('--out', default='outputs/v2_fusion')
    args = ap.parse_args()

    streams = [s.strip() for s in args.streams.split(',') if s.strip()]
    paths = resolve_paths(args)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print("=" * 72)
    print("v2_fusion — 訓練")
    print("=" * 72)
    print(f"device     : {device}"
          + (f"  ({torch.cuda.get_device_name(0)})" if device == 'cuda' else ""))
    print(f"cache_dir  : {paths['cache']}")
    print(f"train_csv  : {paths['train']}")
    print(f"val_csv    : {paths['val']}")
    print(f"test_csv   : {paths['test'] or '(未提供 —— val 數字將標記為 selection-biased)'}")
    print(f"streams    : {streams}")

    bundles = {'train': load_split(paths['cache'], paths['train'], 'train',
                                   streams, args.require_manifest),
               'val':   load_split(paths['cache'], paths['val'], 'val',
                                   streams, args.require_manifest)}
    if paths['test']:
        bundles['test'] = load_split(paths['cache'], paths['test'], 'test',
                                     streams, args.require_manifest)
    print()
    for k, b in bundles.items():
        print(describe_split(b, k))

    dims = dict(bundles['train']['dims'])
    expected = STREAM_DIMS_RAW if args.raw_dims else STREAM_DIMS_LEGACY
    for s in streams:
        if dims[s] != expected.get(s):
            print(f"  [note] {s} 的 cache 維度是 {dims[s]}，"
                  f"{'RAW' if args.raw_dims else 'LEGACY'} 預期 {expected.get(s)}"
                  f" —— 以實際 cache 為準")

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
        results.append(run_one(cfg, bundles, paths, device, out_root))

    if len(results) > 1:
        print("\n" + "=" * 72)
        print("對照總表")
        print("=" * 72)
        key = 'test' if 'test' in results[0] else 'val'
        print(f"{'preset':<18}{'λ_src':>7}{'λ_grl':>8}"
              f"{'acc':>9}{'auc':>9}{'fnr':>8}{'fpr':>8}   ({key})")
        for r in results:
            m, c = r[key], r['config']['loss']
            print(f"{r['name']:<18}{c['lambda_src']:>7}{c['lambda_grl']:>8}"
                  f"{m['acc']:>9.2f}{m['auc']:>9.4f}{m['fnr']:>8.2f}{m['fpr']:>8.2f}")
        with open(out_root / 'comparison.json', 'w') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\n-> {out_root/'comparison.json'}")
        if key == 'val':
            print("\n⚠️  上表是 val split，同時被用來挑 checkpoint，帶有選擇偏誤。")
            print("   要能寫進論文的數字，請用 --test-csv 指定一個沒參與選擇的 split。")
    return 0


if __name__ == '__main__':
    sys.exit(main())
