"""
v2_fusion.data — 讀 feature cache，且拒絕靜默對錯
==================================================
舊版的三個對齊問題（audit 發現 07、以及 s3_main_grl.py:315 的截斷對齊）：

  1. train_df = train_df.iloc[:n_train]            ← 直接截斷 CSV 去配 feature 筆數
     只要 cache 的寫入順序跟 CSV 不同，整批 label 錯位，而且不會報錯。
     實測 s3a_single_stream.py:147 是 shuffle=False，所以順序目前確實一致 ——
     但那是「靠慣例成立」不是「靠檢查保證」。任何人改了 shuffle 或換了 CSV 就全錯。

  2. GENERATOR_TO_ID.get(name, REAL_ID)            ← 未知 generator 靜默變真圖
     CSV 打錯字或新增 generator 忘了進 dict，那批假圖的 source label 就成了真圖。

  3. torch.clamp(gen_ids, 0, N_GEN-1)              ← clamp 當 remap
     見 config.py 的 SOURCE_ID_TO_GEN_ID。

v2 的原則：對不上就 raise，絕不猜。
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .config import (
    GENERATOR_TO_ID, REAL_IDS, SOURCE_ID_TO_GEN_ID, GEN_ID_TO_NAME,
)


class AlignmentError(RuntimeError):
    """特徵與 CSV 對不起來。這是硬錯誤，不提供 fallback。"""


# ══════════════════════════════════════════════════════════════════════
# 載入
# ══════════════════════════════════════════════════════════════════════

def load_split(
    cache_dir: Path,
    csv_path: Path,
    split: str,
    streams: List[str],
    require_manifest: bool = False,
) -> Dict[str, object]:
    """
    讀一個 split 的所有流特徵 + 對應的 CSV 標籤。

    預期檔案（沿用舊版命名，隊友不用改抽特徵的存檔邏輯）：
        {cache_dir}/{stream}_{split}_feats.pt     每條流一個 (N, in_dim) tensor
        {cache_dir}/{split}_labels.pt             (N,) binary label
        {cache_dir}/{split}_paths.json            選用；有的話會逐筆核對

    require_manifest=True 時，沒有 paths manifest 就直接失敗 ——
    重抽特徵之後應該打開這個開關，那才是真正保證對齊。
    """
    cache_dir = Path(cache_dir)
    feats: Dict[str, torch.Tensor] = {}
    n_rows: Optional[int] = None

    for s in streams:
        fp = cache_dir / f"{s}_{split}_feats.pt"
        if not fp.exists():
            raise FileNotFoundError(
                f"找不到 {fp}\n"
                f"  這條流的特徵還沒抽，或 cache_dir 指錯了。\n"
                f"  現有檔案：{sorted(p.name for p in cache_dir.glob(f'*_{split}_feats.pt'))}"
            )
        t = torch.load(fp, weights_only=False)
        if not isinstance(t, torch.Tensor):
            raise AlignmentError(f"{fp} 不是 tensor，讀到 {type(t)}")
        if t.dim() != 2:
            raise AlignmentError(f"{fp} 應該是 2D (N, dim)，實際 {tuple(t.shape)}")
        if n_rows is None:
            n_rows = t.shape[0]
        elif t.shape[0] != n_rows:
            raise AlignmentError(
                f"各流的樣本數不一致：{s} 有 {t.shape[0]} 筆，先前的流是 {n_rows} 筆。\n"
                f"  很可能是某條流抽到一半中斷。"
            )
        feats[s] = t.float()

    lp = cache_dir / f"{split}_labels.pt"
    if not lp.exists():
        raise FileNotFoundError(f"找不到 {lp}")
    labels = torch.load(lp, weights_only=False)
    labels = torch.as_tensor(labels).long().view(-1)
    if labels.shape[0] != n_rows:
        raise AlignmentError(
            f"labels 有 {labels.shape[0]} 筆，features 有 {n_rows} 筆。")

    # ── CSV：長度必須完全相符，不截斷 ──────────────────────────────
    df = pd.read_csv(csv_path)
    if len(df) != n_rows:
        raise AlignmentError(
            f"CSV 與 features 筆數不符：{csv_path.name} 有 {len(df):,} 列，"
            f"features 有 {n_rows:,} 筆。\n"
            f"  舊版在這裡做 df.iloc[:n] 截斷，那會把 label 整批錯位而不報錯。\n"
            f"  v2 不截斷。請確認 cache 是用這份 CSV 抽出來的。"
        )

    # ── 選用的 paths manifest：唯一能真正證明對齊的東西 ────────────
    manifest_path = cache_dir / f"{split}_paths.json"
    manifest_checked = False
    if manifest_path.exists():
        import json
        with open(manifest_path) as f:
            paths = json.load(f)
        if len(paths) != n_rows:
            raise AlignmentError(
                f"{manifest_path.name} 有 {len(paths)} 筆，features 有 {n_rows} 筆。")
        if 'path' in df.columns:
            mism = [i for i in range(n_rows) if str(paths[i]) != str(df['path'].iloc[i])]
            if mism:
                raise AlignmentError(
                    f"features 與 CSV 的順序不一致，共 {len(mism):,} 筆對不上，"
                    f"前幾筆索引：{mism[:5]}\n"
                    f"  cache 的第 {mism[0]} 筆是 {paths[mism[0]]}\n"
                    f"  CSV  的第 {mism[0]} 列是 {df['path'].iloc[mism[0]]}"
                )
            manifest_checked = True
    elif require_manifest:
        raise AlignmentError(
            f"require_manifest=True 但找不到 {manifest_path.name}。\n"
            f"  抽特徵時請一併存下每一筆的 path（順序與寫入 tensor 的順序相同），\n"
            f"  那是唯一能真正保證 feature/label 對齊的東西。"
        )

    src_ids, gen_ids, is_fake = _encode_generators(df, csv_path)

    if not torch.equal(is_fake.long(), labels):
        n_diff = int((is_fake.long() != labels).sum())
        raise AlignmentError(
            f"CSV 的 generator 欄推出的真假，與 {split}_labels.pt 不一致，共 {n_diff:,} 筆。\n"
            f"  這表示 cache 與 CSV 不是同一份資料，或順序不同。"
        )

    return {
        'feats': feats,
        'labels': labels,
        'source_ids': src_ids,
        'gen_ids': gen_ids,
        'df': df,
        'n': n_rows,
        'dims': {s: t.shape[1] for s, t in feats.items()},
        'manifest_checked': manifest_checked,
    }


def _encode_generators(
    df: pd.DataFrame, csv_path: Path
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """generator 名稱 → source id / gen id / is_fake。未知名稱直接 raise。"""
    if 'generator' not in df.columns:
        raise AlignmentError(f"{csv_path.name} 沒有 'generator' 欄")

    names = df['generator'].astype(str).str.lower().str.strip()

    unknown = sorted(set(names) - set(GENERATOR_TO_ID))
    if unknown:
        counts = names.value_counts()
        detail = ', '.join(f"{u} ({counts[u]:,} 筆)" for u in unknown[:8])
        raise AlignmentError(
            f"{csv_path.name} 出現 GENERATOR_TO_ID 裡沒有的 generator：{detail}\n"
            f"  舊版在這裡是 .get(name, REAL_ID)，會把這些假圖靜默標成真圖。\n"
            f"  請確認拼字，或把新的 generator 加進 config.GENERATOR_TO_ID。"
        )

    src = torch.tensor([GENERATOR_TO_ID[n] for n in names], dtype=torch.long)
    is_fake = torch.tensor([GENERATOR_TO_ID[n] not in REAL_IDS for n in names])

    # 真圖沒有 generator 身份，填 -1；CrossEntropyLoss 的 ignore_index 會吃掉它。
    gen = torch.full_like(src, -1)
    fake_idx = is_fake.nonzero(as_tuple=True)[0]
    gen[fake_idx] = torch.tensor(
        [SOURCE_ID_TO_GEN_ID[int(src[i])] for i in fake_idx], dtype=torch.long)

    return src, gen, is_fake


# ══════════════════════════════════════════════════════════════════════
# Dataset
# ══════════════════════════════════════════════════════════════════════

class FusionFeatureDataset(Dataset):
    """回傳 (feats_dict, y_binary, y_source, y_gen)。"""

    def __init__(self, bundle: Dict[str, object], streams: List[str]):
        self.streams = list(streams)
        self.feats = bundle['feats']
        self.labels = bundle['labels']
        self.source_ids = bundle['source_ids']
        self.gen_ids = bundle['gen_ids']
        self.n = bundle['n']

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, i: int):
        f = {s: self.feats[s][i] for s in self.streams}
        return f, self.labels[i], self.source_ids[i], self.gen_ids[i]


def collate(batch):
    feats_list, y_bin, y_src, y_gen = zip(*batch)
    feats = {s: torch.stack([f[s] for f in feats_list]) for s in feats_list[0]}
    return (feats,
            torch.stack(y_bin),
            torch.stack(y_src),
            torch.stack(y_gen))


def describe_split(bundle: Dict[str, object], split: str) -> str:
    labels = bundle['labels']
    gen = bundle['gen_ids']
    n_fake = int((labels == 1).sum())
    present = sorted(set(int(g) for g in gen[gen >= 0].tolist()))
    names = ', '.join(GEN_ID_TO_NAME[g] for g in present)
    dims = ', '.join(f"{s}:{d}" for s, d in bundle['dims'].items())
    return (
        f"[{split}] n={bundle['n']:,}  fake={n_fake:,} ({100*n_fake/bundle['n']:.1f}%)  "
        f"real={bundle['n']-n_fake:,}\n"
        f"  dims      : {dims}\n"
        f"  generators: {len(present)} 種 — {names}\n"
        f"  對齊驗證  : {'paths manifest 已核對' if bundle['manifest_checked'] else 'manifest 不存在，僅核對筆數與真假一致性'}"
    )
