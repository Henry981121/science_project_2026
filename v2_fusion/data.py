"""
v2_fusion.data — 讀 feature cache，且拒絕靜默對錯
==================================================
對應 2026-08-26 的 handoff package（handoff_dct_clip_dinov2_20260826）：

    {cache}/{stream}/{split}.npy          (N, dim)  float32
    {cache}/{stream}/{split}.valid.npy    (N,)      bool
    {cache}/index_{split}.csv             row,path,generator,is_real,split,valid_all

舊版（s3_main_grl.py / s3a_single_stream.py）的三個對齊問題：

  1. train_df = train_df.iloc[:n_train]            ← 直接截斷 CSV 去配 feature 筆數
     只要 cache 的寫入順序跟 CSV 不同，整批 label 錯位，而且不會報錯。

  2. GENERATOR_TO_ID.get(name, REAL_ID)            ← 未知 generator 靜默變真圖
     CSV 打錯字或新增 generator 忘了進 dict，那批假圖就成了真圖。

  3. torch.clamp(gen_ids, 0, N-1)                  ← clamp 當 remap，撞號 + 死輸出

v2 的原則：對不上就 raise，絕不猜。
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .config import (
    GENERATOR_TO_ID, SOURCE_ID_TO_NAME, KNOWN_GENERATORS, EVAL_ONLY_GENERATORS,
    REAL_GENERATOR_NAMES, REAL_IDS, SOURCE_IGNORE_INDEX, STREAM_DIMS,
)


class AlignmentError(RuntimeError):
    """特徵與 CSV 對不起來。這是硬錯誤，不提供 fallback。"""


# ══════════════════════════════════════════════════════════════════════
# 載入
# ══════════════════════════════════════════════════════════════════════

def load_split(
    cache_dir: Path,
    split: str,
    streams: List[str],
    csv_path: Optional[Path] = None,
    drop_invalid: bool = True,
) -> Dict[str, object]:
    """
    讀一個 split 的所有流特徵 + 對應的 index CSV。

    drop_invalid=True 時，任何一條「被選用的流」標為 invalid 的列會被剔除。
    注意這是**選用的流**的交集，不是 CSV 的 valid_all —— 只用 3 條流裡的 2 條時，
    第 3 條的 invalid 不該把樣本丟掉。兩者的差異會印出來。
    """
    cache_dir = Path(cache_dir)
    csv_path = Path(csv_path) if csv_path else cache_dir / f"index_{split}.csv"

    if not csv_path.exists():
        raise FileNotFoundError(
            f"找不到 {csv_path}\n"
            f"  現有 index：{sorted(p.name for p in cache_dir.glob('index_*.csv'))}")
    df = pd.read_csv(csv_path)

    for col in ('row', 'path', 'generator', 'is_real'):
        if col not in df.columns:
            raise AlignmentError(
                f"{csv_path.name} 缺少 '{col}' 欄。實際欄位：{df.columns.tolist()}")

    n_rows = len(df)

    # ── row 欄必須就是 0..N-1，那是 npy 的列序 ─────────────────────────
    expected_rows = np.arange(n_rows)
    if not np.array_equal(df['row'].to_numpy(), expected_rows):
        bad = int(np.argmax(df['row'].to_numpy() != expected_rows))
        raise AlignmentError(
            f"{csv_path.name} 的 'row' 欄不是 0..{n_rows-1} 的連續序列，"
            f"第一個不符在第 {bad} 列（值 {df['row'].iloc[bad]}）。\n"
            f"  npy 的列序就是這一欄，對不上表示 CSV 被重排或過濾過。")

    # ── 特徵 ──────────────────────────────────────────────────────────
    feats: Dict[str, torch.Tensor] = {}
    valid_per_stream: Dict[str, np.ndarray] = {}

    for s in streams:
        fp = cache_dir / s / f"{split}.npy"
        if not fp.exists():
            available = sorted(p.name for p in cache_dir.iterdir() if p.is_dir())
            raise FileNotFoundError(
                f"找不到 {fp}\n"
                f"  這條流沒有這個 split，或 cache_dir 指錯了。\n"
                f"  cache 裡的流目錄：{available}")

        # 先用 mmap 檢查形狀（不讀進記憶體），過了才整份載入。
        arr = np.load(fp, mmap_mode='r')
        if arr.ndim != 2:
            raise AlignmentError(f"{fp} 應該是 2D (N, dim)，實際 {arr.shape}")
        if arr.shape[0] != n_rows:
            raise AlignmentError(
                f"{s}/{split}.npy 有 {arr.shape[0]:,} 列，"
                f"{csv_path.name} 有 {n_rows:,} 列。\n"
                f"  舊版在這裡做 df.iloc[:n] 截斷，那會把 label 整批錯位而不報錯。\n"
                f"  v2 不截斷。")

        expect_dim = STREAM_DIMS.get(s)
        if expect_dim is not None and arr.shape[1] != expect_dim:
            raise AlignmentError(
                f"{s}/{split}.npy 的維度是 {arr.shape[1]}，"
                f"config.STREAM_DIMS 說是 {expect_dim}。\n"
                f"  維度變了就是換了 backbone 或改了抽取方式，config 要一起更新。")

        del arr                       # 放掉 mmap，改成一次讀進來
        feats[s] = torch.from_numpy(np.load(fp)).float()

        vp = cache_dir / s / f"{split}.valid.npy"
        if vp.exists():
            v = np.load(vp).astype(bool).reshape(-1)
            if v.shape[0] != n_rows:
                raise AlignmentError(
                    f"{s}/{split}.valid.npy 有 {v.shape[0]:,} 筆，應為 {n_rows:,}")
            valid_per_stream[s] = v
        else:
            valid_per_stream[s] = np.ones(n_rows, dtype=bool)

    # ── valid mask：選用流的交集 ──────────────────────────────────────
    mask = np.ones(n_rows, dtype=bool)
    for v in valid_per_stream.values():
        mask &= v

    notes: List[str] = []
    if 'valid_all' in df.columns:
        csv_valid = df['valid_all'].to_numpy().astype(bool)
        all_streams_mask = np.ones(n_rows, dtype=bool)
        for s in sorted(STREAM_DIMS):
            if s in valid_per_stream:
                all_streams_mask &= valid_per_stream[s]
        if set(streams) == set(STREAM_DIMS) and not np.array_equal(csv_valid, all_streams_mask):
            n_diff = int((csv_valid != all_streams_mask).sum())
            raise AlignmentError(
                f"CSV 的 valid_all 與各流 .valid.npy 的交集不一致，共 {n_diff:,} 筆。\n"
                f"  這表示 CSV 與特徵不是同一次抽取的產物。")
        if set(streams) != set(STREAM_DIMS):
            notes.append(
                f"只用了 {len(streams)}/{len(STREAM_DIMS)} 條流，"
                f"mask 取這幾條的交集（不是 CSV 的 valid_all）")

    n_invalid = int((~mask).sum())

    # ── 標籤 ──────────────────────────────────────────────────────────
    src_ids, is_fake = _encode_generators(df, csv_path)

    is_real_csv = df['is_real'].to_numpy().astype(int)
    if not np.array_equal(is_fake.numpy().astype(int), 1 - is_real_csv):
        n_diff = int((is_fake.numpy().astype(int) != (1 - is_real_csv)).sum())
        raise AlignmentError(
            f"{csv_path.name} 的 'generator' 欄與 'is_real' 欄互相矛盾，共 {n_diff:,} 筆。\n"
            f"  例如 generator 是真圖名稱但 is_real=0，或反之。")

    labels = torch.from_numpy((1 - is_real_csv)).long()   # 1 = fake

    # ── 套用 mask ─────────────────────────────────────────────────────
    if drop_invalid and n_invalid > 0:
        keep = torch.from_numpy(np.nonzero(mask)[0])
        feats = {s: t[keep] for s, t in feats.items()}
        labels = labels[keep]
        src_ids = src_ids[keep]
        df = df.iloc[mask].reset_index(drop=True)
        n_kept = int(mask.sum())
    else:
        n_kept = n_rows

    return {
        'feats': feats,
        'labels': labels,
        'source_ids': src_ids,
        'df': df,
        'n': n_kept,
        'n_raw': n_rows,
        'n_invalid': n_invalid,
        'dims': {s: int(t.shape[1]) for s, t in feats.items()},
        'notes': notes,
    }


def _encode_generators(
    df: pd.DataFrame, csv_path: Path
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    generator 名稱 → source id / is_fake。

    · train/val 出現的 10 種 → 0..9
    · 只出現在 cross_generator_test 的 5 種 → -1（ignore_index 吃掉）
      它們沒有 source 標籤，硬編進類別空間會製造死輸出（audit 發現 08）。
    · 不認得的名稱 → raise。舊版是 .get(name, REAL_ID)，會靜默標成真圖。
    """
    if 'generator' not in df.columns:
        raise AlignmentError(f"{csv_path.name} 沒有 'generator' 欄")

    names = df['generator'].astype(str).str.lower().str.strip()

    unknown = sorted(set(names) - KNOWN_GENERATORS)
    if unknown:
        counts = names.value_counts()
        detail = ', '.join(f"{u} ({counts[u]:,} 筆)" for u in unknown[:8])
        raise AlignmentError(
            f"{csv_path.name} 出現不認得的 generator：{detail}\n"
            f"  舊版在這裡是 .get(name, REAL_ID)，會把這些假圖靜默標成真圖。\n"
            f"  請確認拼字，或把新的 generator 加進 config.TRAIN_GENERATORS "
            f"／EVAL_ONLY_GENERATORS。")

    src = torch.tensor(
        [GENERATOR_TO_ID.get(n, SOURCE_IGNORE_INDEX) for n in names], dtype=torch.long)
    is_fake = torch.tensor([n not in REAL_GENERATOR_NAMES for n in names])
    return src, is_fake


# ══════════════════════════════════════════════════════════════════════
# Dataset
# ══════════════════════════════════════════════════════════════════════

class FusionFeatureDataset(Dataset):
    """回傳 (feats_dict, y_binary, y_source)。"""

    def __init__(self, bundle: Dict[str, object], streams: List[str]):
        self.streams = list(streams)
        self.feats = bundle['feats']
        self.labels = bundle['labels']
        self.source_ids = bundle['source_ids']
        self.n = bundle['n']

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, i: int):
        f = {s: self.feats[s][i] for s in self.streams}
        return f, self.labels[i], self.source_ids[i]


def collate(batch):
    feats_list, y_bin, y_src = zip(*batch)
    feats = {s: torch.stack([f[s] for f in feats_list]) for s in feats_list[0]}
    return feats, torch.stack(y_bin), torch.stack(y_src)


def describe_split(bundle: Dict[str, object], split: str) -> str:
    labels = bundle['labels']
    df = bundle['df']
    n = bundle['n']
    n_fake = int((labels == 1).sum())
    gens = df['generator'].astype(str).str.lower().value_counts()
    gen_str = ', '.join(f"{k}:{v:,}" for k, v in gens.items())
    dims = ', '.join(f"{s}:{d}" for s, d in bundle['dims'].items())
    eval_only = sorted(set(gens.index) & set(EVAL_ONLY_GENERATORS))

    lines = [
        f"[{split}] n={n:,}  fake={n_fake:,} ({100*n_fake/max(n,1):.1f}%)  real={n-n_fake:,}",
        f"  dims      : {dims}",
        f"  invalid   : {bundle['n_invalid']:,} / {bundle['n_raw']:,} 已剔除",
        f"  generators: {len(gens)} 種 — {gen_str}",
    ]
    if eval_only:
        lines.append(
            f"  ⚠ source  : {', '.join(eval_only)} 不在 source 類別空間，"
            f"source 標籤為 -1（訓練時會被 ignore_index 吃掉）")
    lines += [f"  note      : {t}" for t in bundle['notes']]
    return '\n'.join(lines)
