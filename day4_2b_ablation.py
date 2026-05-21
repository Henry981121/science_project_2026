"""
day4_2b_ablation.py
===================

Level 2 — 2B：attention vs. ablation cross-check 的「ablation」那一半
（XAI 計畫 Day 4 / 2B）。

【為什麼要這支】
Wiegreffe & Pinter 2019 的反駁要求：證明「移除某 component 真的會改變
預測」，attention 權重才能當解釋。這支對每張測試圖逐流 zero-out，量
fake-prob 的變化幅度，產生「ablation importance」，供後續跟 attention
算 Spearman ρ。

【做什麼】
1. 載入訓好的 FusionDetectorGRL（outputs/main_grl/best_model.pth）
2. 載入快取的 test features（FEAT_CACHE_DIR/{stream}_test_feats.pt）
   —— 5 條流各 512 維，拼成 (N, 2560)
3. baseline：完整特徵跑 fusion → fake-prob
4. 逐流 i：把該流的 512 維 slice 歸零，重跑 → fake-prob_i
       ablation importance = |fake-prob_baseline - fake-prob_i|
5. 輸出 outputs/level2/attention_ablation.csv
       = preflight_attn.csv 全欄（path/generator/family/attn_*）
       + fake_prob_base + ablimp_{clip,fft,dct,dire,noise}

【執行成本】
特徵已快取，這支不抽特徵，只跑「6 趟很小的 fusion forward」（1 baseline
+ 5 ablated）。即使整個 test set（~2.4 萬張）也是秒~分鐘級。

【在哪跑】
要在有 model checkpoint + 快取特徵的機器上跑（= 隊友機器）。

【怎麼用】
從專案 root：
    python day4_2b_ablation.py
    python day4_2b_ablation.py --device cpu --batch 512

【接下來】
產出的 attention_ablation.csv 同時含 attention 與 ablation，後續
attention-vs-ablation 散點圖 + Spearman ρ 是純數字分析，可在任何
機器上接著做（不需 GPU/checkpoint）。

【相依】torch, pandas, numpy
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, '.')
from config import OUTPUTS_DIR, FEAT_CACHE_DIR
from s3_main_grl import FusionDetectorGRL

# 流順序必須與 preflight_risk_b.py 一致 —— 快取特徵就是照這個順序 cat 的
STREAMS = ['clip', 'fft', 'dct', 'dire', 'noise']
D = 512  # 每條流的特徵維度


# ── 載入 ──────────────────────────────────────────────────────────────
def load_model(model_path: Path, device: str):
    """載入 FusionDetectorGRL，回傳 (model, n_streams)。"""
    if not model_path.exists():
        sys.exit(f"[2b] 找不到 model checkpoint: {model_path}")
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    model = FusionDetectorGRL(
        n_streams=ckpt['n_streams'],
        n_sources=ckpt['n_sources'],
        n_gen=ckpt['n_gen'],
    )
    model.load_state_dict(ckpt['model_state_dict'])
    model.to(device).eval()
    return model, ckpt['n_streams']


def load_test_feats() -> torch.Tensor:
    """載入 5 條流的快取 test features，拼成 (N, 2560)。"""
    parts = []
    for s in STREAMS:
        fp = FEAT_CACHE_DIR / f"{s}_test_feats.pt"
        if not fp.exists():
            sys.exit(f"[2b] 找不到快取特徵: {fp}\n"
                     f"     （需先跑過會產生 {{stream}}_test_feats.pt 的流程）")
        parts.append(torch.load(fp, weights_only=False))
    return torch.cat(parts, dim=1)  # (N, n_streams*512)


# ── fusion forward → fake 機率 ────────────────────────────────────────
@torch.no_grad()
def fake_prob(model, feats: torch.Tensor, device: str, batch: int,
              ablate_idx: int = None) -> np.ndarray:
    """
    跑 fusion 取 fake 機率 (N,)。

    ablate_idx 有給時，把該流的 512 維 slice 歸零再跑（per-batch clone，
    不動到原始 feats）。
    """
    out = []
    for i in range(0, len(feats), batch):
        b = feats[i:i+batch].to(device)
        if ablate_idx is not None:
            b = b.clone()
            b[:, ablate_idx * D:(ablate_idx + 1) * D] = 0.0
        logits_binary, _, _, _ = model(b, grl_lambda=0)
        out.append(torch.softmax(logits_binary, dim=1)[:, 1].cpu())
    return torch.cat(out).numpy()


# ── main ──────────────────────────────────────────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser(description="Level 2 / 2B ablation cross-check")
    ap.add_argument("--model", default=str(OUTPUTS_DIR / 'main_grl' / 'best_model.pth'),
                    help="FusionDetectorGRL checkpoint 路徑")
    ap.add_argument("--attn-csv", default=str(OUTPUTS_DIR / 'preflight_attn.csv'),
                    help="preflight_attn.csv 路徑（提供 metadata + 對齊順序）")
    ap.add_argument("--outdir", default=str(OUTPUTS_DIR / 'level2'),
                    help="輸出資料夾")
    ap.add_argument("--device", default=None, help="cuda / cpu（預設自動）")
    ap.add_argument("--batch", type=int, default=256, help="batch size")
    args = ap.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  Level 2 / 2B — ablation cross-check")
    print("=" * 60)
    print(f"  Device : {device}")

    # 1. model
    model, n_streams = load_model(Path(args.model), device)
    print(f"  Model  : loaded (n_streams={n_streams})")

    # 2. cached features
    feats = load_test_feats()
    print(f"  Feats  : {tuple(feats.shape)}")
    if feats.shape[1] != n_streams * D:
        sys.exit(f"[2b] 特徵維度 {feats.shape[1]} != n_streams*512 "
                 f"({n_streams}*{D})。快取特徵與模型不一致。")
    if n_streams != len(STREAMS):
        sys.exit(f"[2b] 模型 n_streams={n_streams} 與 STREAMS({len(STREAMS)}) 不符。")

    # 3. metadata（與快取特徵同一份 test set、同順序）
    attn_path = Path(args.attn_csv)
    if not attn_path.exists():
        sys.exit(f"[2b] 找不到 attention CSV: {attn_path}")
    meta = pd.read_csv(attn_path)
    if len(meta) != len(feats):
        sys.exit(f"[2b] CSV 列數 ({len(meta)}) 與快取特徵 ({len(feats)}) 不一致，"
                 "可能 cache 過期。")
    print(f"  Meta   : {len(meta)} rows from {attn_path.name}")

    # 4. baseline fake-prob
    print("\n[2b] baseline forward ...")
    base = fake_prob(model, feats, device, args.batch)

    # 5. 逐流 ablation
    out = meta.copy()
    out['fake_prob_base'] = base
    for idx, s in enumerate(STREAMS):
        print(f"[2b] ablate {s} ...")
        ablated = fake_prob(model, feats, device, args.batch, ablate_idx=idx)
        out[f'ablimp_{s}'] = np.abs(base - ablated)  # |Δ fake-prob|

    # 6. 存檔
    out_path = outdir / 'attention_ablation.csv'
    out.to_csv(out_path, index=False)
    print(f"\n[2b] saved {out_path}")

    # 7. console sanity check：逐流平均 ablation importance
    print("\n=== Mean ablation importance |Δ fake-prob| ===")
    for s in STREAMS:
        print(f"  {s:6s} {out[f'ablimp_{s}'].mean():.4f}")
    if 'family' in out.columns:
        print("\n=== Mean ablation importance by family ===")
        cols = [f'ablimp_{s}' for s in STREAMS]
        print(out.groupby('family')[cols].mean().round(4).to_string())

    print(f"\n[2b] 完成。attention_ablation.csv 同時含 attn_* 與 ablimp_*，"
          "後續 Spearman ρ + 散點圖在本機接著做即可。")


if __name__ == "__main__":
    main()
