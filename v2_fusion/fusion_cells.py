"""
v2_fusion.fusion_cells — 三軸因子設計的融合模組
================================================

這個檔案存在的理由
------------------
比較「TFN vs GMU vs MBT」得不到因果解釋，因為那些方法**同時**差在交互階數、
參數量、權重來源、正規化位置、初始化。差異出現時無法歸因。

所以這裡不放文獻上的完整方法，只放「具備某個性質的最小實現」，並且讓
三個軸互相正交、一次只變一件事：

    A. order      1 = 只有加權和            2 = 加權和 + 交互項
    B. weighting  static = 學到的常數權重    gated = 權重隨樣本而變
    C. level      feature = 在 d 維表示上融合  decision = 在各流 logits 上融合

2 × 2 × 2 = 8 格。文獻上的名字只出現在論文的 Related Work，用來說明
「我們的第 k 格對應到哪一類」——見下表：

    o1.static.feat      學到的固定加權平均
    o1.gated.feat       GMU / attention pooling 的本質
    o2.static.feat      低秩雙線性（MLB / MFB / LMF 一類）
    o2.gated.feat       MUTAN 一類
    o1.static.dec       late fusion / weighted ensemble
    o1.gated.dec        mixture of experts
    o2.static.dec       決策層的乘積交互（product of experts 一類）
    o2.gated.dec        —

外加兩個參照組（不是格子，是標尺）：

    concat              concat → MLP，最樸素的常見做法
    （單流）             用 train.py --streams clip 跑，回答「融合到底有沒有用」

巢狀性質（刻意的）
------------------
order=2 的模組是 order=1 的**嚴格超集**：把 `interact_scale` 設為 0，
二階格就退化成對應的一階格。所以 A 軸真的只切換一件事——交互項在不在。
`interact_scale` 訓練後的值本身就是一個中介變數：它趨近 0 就代表
模型自己認為交互項沒用，這比從準確率反推可靠得多。

容量控制
--------
八格的天生大小差六個數量級（固定加權 3 個參數 vs 現有 hybrid 7.3M），
所以「哪一格比較好」在沒有容量控制時無法解讀。

每個 cell 只有**一個**自由度 `width`，`fit_width_to_budget()` 用二分搜尋
把它調到指定的參數預算。掃 3 個預算就得到「參數量 vs 泛化落差」曲線；
兩條曲線平行 = 純粹容量差異，交叉 = 機制真的不同。

注意：同一個預算下，二階格必須把容量分給交互項，post-MLP 就會比一階格窄。
那不是不公平，那正是被研究的取捨本身。
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn


ORDERS = (1, 2)
WEIGHTINGS = ('static', 'gated')
LEVELS = ('feature', 'decision')

# 參照組。不屬於因子設計，跑出來當標尺用。
REFERENCE_CELLS = ('concat',)


# ══════════════════════════════════════════════════════════════════════
# Spec
# ══════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class CellSpec:
    """一個格子的完整身分。`width` 是唯一的容量自由度。"""
    order: int = 1
    weighting: str = 'static'
    level: str = 'feature'
    width: int = 64

    def __post_init__(self):
        if self.order not in ORDERS:
            raise ValueError(f"order 必須是 1 或 2，收到 {self.order}")
        if self.weighting not in WEIGHTINGS:
            raise ValueError(f"weighting 必須是 {WEIGHTINGS}，收到 {self.weighting}")
        if self.level not in LEVELS:
            raise ValueError(f"level 必須是 {LEVELS}，收到 {self.level}")
        if self.width < 1:
            raise ValueError(f"width 必須 >= 1，收到 {self.width}")

    @property
    def cell_id(self) -> str:
        lv = 'feat' if self.level == 'feature' else 'dec'
        return f"o{self.order}.{self.weighting}.{lv}"

    def axes(self) -> Dict[str, object]:
        """存進 results.json 的三軸座標。分析時直接拿來當自變數。"""
        return {'order': self.order, 'weighting': self.weighting,
                'level': self.level, 'width': self.width}


def all_cell_specs(width: int = 64) -> List[CellSpec]:
    """八個格子，順序固定（o1 先於 o2、static 先於 gated、feat 先於 dec）。"""
    return [CellSpec(order=o, weighting=w, level=l, width=width)
            for o in ORDERS for w in WEIGHTINGS for l in LEVELS]


# ══════════════════════════════════════════════════════════════════════
# B 軸：權重來源
# ══════════════════════════════════════════════════════════════════════

class StreamWeighting(nn.Module):
    """
    產生 stream 權重 (B, N)，以及 order=2 時的 pair 權重 (B, P)。

    static：一組學到的常數，對整個資料集共用。
    gated ：由 MLP 從當前樣本的所有 token 算出來，逐樣本不同。

    兩種都經過 softmax，所以 stream 權重永遠是凸組合——這讓 B 軸的差異
    純粹是「權重會不會隨樣本變」，而不是「權重能不能是負的」之類的副作用。

    gated 模式下 gate 一律看 tokens（而不是 logits），feature 和 decision
    兩個 level 都一樣。否則 decision-level 的 gate 只有 2N 個數字可看，
    B 軸在兩個 level 下就不是同一件事了。
    """

    def __init__(self, n_streams: int, n_pairs: int, d_model: int,
                 mode: str, width: int, dropout: float = 0.0):
        super().__init__()
        self.mode = mode
        self.n_streams = n_streams
        self.n_pairs = n_pairs
        n_out = n_streams + n_pairs

        if mode == 'static':
            self.logits = nn.Parameter(torch.zeros(n_out))
            self.gate = None
        else:
            self.logits = None
            self.gate = nn.Sequential(
                nn.LayerNorm(n_streams * d_model),
                nn.Linear(n_streams * d_model, width),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(width, n_out),
            )

    def forward(self, tokens: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B = tokens.shape[0]
        if self.mode == 'static':
            raw = self.logits.unsqueeze(0).expand(B, -1)
        else:
            raw = self.gate(tokens.flatten(1))

        sw = torch.softmax(raw[:, :self.n_streams], dim=-1)
        pw = (torch.softmax(raw[:, self.n_streams:], dim=-1)
              if self.n_pairs > 0 else None)
        return sw, pw


# ══════════════════════════════════════════════════════════════════════
# A 軸：交互項
# ══════════════════════════════════════════════════════════════════════

class PairwiseInteraction(nn.Module):
    """
    低秩雙線性交互：(U_s z_s) ⊙ (U_t z_t)，對所有 s<t 加權求和後投回輸出維度。

    Hadamard 積是 z_s^T W z_t 這個雙線性形式的低秩近似（MLB 那條線），
    rank 就是 `width`。用低秩而不是完整外積的理由很實際：
    完整外積是 d² = 262,144 維，光這一項就吃掉整個參數預算，
    「二階比較好」就會變成「二階比較大」。
    """

    def __init__(self, n_streams: int, d_model: int, out_dim: int,
                 rank: int, dropout: float = 0.0):
        super().__init__()
        self.pairs: List[Tuple[int, int]] = list(
            itertools.combinations(range(n_streams), 2))
        self.proj = nn.ModuleList(
            [nn.Linear(d_model, rank, bias=False) for _ in range(n_streams)])
        self.out = nn.Sequential(nn.Dropout(dropout), nn.Linear(rank, out_dim))

    def forward(self, tokens: torch.Tensor, pair_w: torch.Tensor) -> torch.Tensor:
        u = [self.proj[i](tokens[:, i]) for i in range(tokens.shape[1])]  # (B, r)
        acc = 0.0
        for k, (s, t) in enumerate(self.pairs):
            acc = acc + pair_w[:, k:k + 1] * (u[s] * u[t])
        return self.out(acc)


# ══════════════════════════════════════════════════════════════════════
# Cell
# ══════════════════════════════════════════════════════════════════════

class FusionCell(nn.Module):
    """
    輸入 tokens (B, N, d)，輸出 binary logits。

    這個模組**擁有 head_binary**，因為 decision-level 的分類頭在各流上、
    feature-level 的在融合後——把頭放在 cell 外面就沒辦法讓 C 軸只變一件事。
    `shared` (B, d) 只給 auxiliary source head 和診斷用，不在 binary 路徑上；
    decision-level 時它是 token 的加權和，這一點必須記住，不要拿它去解釋
    binary 的決策。

    輸出的 `stream_weight` 是**逐樣本**的 (B, N)，不是 batch 平均。
    per-sample 變異數是判斷「權重有沒有真的隨樣本變」的唯一依據——
    舊版只存 batch mean，看起來像學到東西的常數權重分不出來。
    """

    def __init__(self, spec: CellSpec, n_streams: int, d_model: int,
                 n_binary: int = 2, dropout: float = 0.0,
                 head_dropout: float = 0.0):
        super().__init__()
        self.spec = spec
        self.n_streams = n_streams
        self.d_model = d_model

        n_pairs = n_streams * (n_streams - 1) // 2
        if spec.order == 2 and n_pairs == 0:
            raise ValueError(
                "order=2 需要至少兩條流才有交互項可言；"
                f"目前 n_streams={n_streams}。單流參照組請用 order=1。")
        self.n_pairs = n_pairs if spec.order == 2 else 0

        self.weighting = StreamWeighting(
            n_streams, self.n_pairs, d_model, spec.weighting, spec.width, dropout)

        # order=2 是 order=1 的嚴格超集：scale=0 時完全退化。
        # 初值 1.0（不是 0）以免梯度一開始就是 0；訓練後的值本身是中介變數。
        self.interact_scale = (nn.Parameter(torch.tensor(1.0))
                               if spec.order == 2 else None)

        if spec.level == 'feature':
            self.interaction = (
                PairwiseInteraction(n_streams, d_model, d_model, spec.width, dropout)
                if spec.order == 2 else None)
            self.post = nn.Sequential(
                nn.LayerNorm(d_model),
                nn.Linear(d_model, spec.width),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(spec.width, d_model),
            )
            self.stream_heads = None
            # 位置與 legacy 路徑的 head_drop 對齊：緊接在產生 binary logits 之前。
            self.head_drop = nn.Dropout(head_dropout)
            self.head_binary = nn.Linear(d_model, n_binary)
        else:
            self.interaction = (
                PairwiseInteraction(n_streams, d_model, n_binary, spec.width, dropout)
                if spec.order == 2 else None)
            self.post = None
            self.stream_heads = nn.ModuleList([
                nn.Sequential(
                    nn.LayerNorm(d_model),
                    nn.Linear(d_model, spec.width),
                    nn.GELU(),
                    nn.Dropout(head_dropout),   # 各流的 binary logits 之前
                    nn.Linear(spec.width, n_binary),
                ) for _ in range(n_streams)
            ])
            self.head_drop = None
            self.head_binary = None

    # ── forward ───────────────────────────────────────────────────────
    def forward(self, tokens: torch.Tensor) -> Dict[str, torch.Tensor]:
        sw, pw = self.weighting(tokens)                     # (B,N), (B,P)|None

        if self.spec.level == 'feature':
            fused = (sw.unsqueeze(-1) * tokens).sum(dim=1)  # (B,d)
            if self.interaction is not None:
                fused = fused + self.interact_scale * self.interaction(tokens, pw)
            shared = fused + self.post(fused)
            logits = self.head_binary(self.head_drop(shared))
        else:
            per_stream = torch.stack(
                [h(tokens[:, i]) for i, h in enumerate(self.stream_heads)], dim=1)
            logits = (sw.unsqueeze(-1) * per_stream).sum(dim=1)   # (B,2)
            if self.interaction is not None:
                logits = logits + self.interact_scale * self.interaction(tokens, pw)
            shared = (sw.unsqueeze(-1) * tokens).sum(dim=1)       # 只給 aux head

        out = {'logits_binary': logits, 'shared': shared, 'stream_weight': sw}
        if pw is not None:
            out['pair_weight'] = pw
        return out

    # ── 診斷 ──────────────────────────────────────────────────────────
    def diagnostics(self) -> Dict[str, float]:
        """訓練後可直接讀的中介變數。"""
        d: Dict[str, float] = {}
        if self.interact_scale is not None:
            d['interact_scale'] = float(self.interact_scale.detach())
        if self.weighting.mode == 'static':
            sw = torch.softmax(
                self.weighting.logits[:self.n_streams].detach(), dim=-1)
            d['static_stream_weight'] = [round(float(x), 4) for x in sw]
        return d

    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ══════════════════════════════════════════════════════════════════════
# 參照組：concat → MLP
# ══════════════════════════════════════════════════════════════════════

class ConcatCell(nn.Module):
    """最樸素的做法。不在因子設計裡，是標尺。介面與 FusionCell 相同。"""

    def __init__(self, spec: CellSpec, n_streams: int, d_model: int,
                 n_binary: int = 2, dropout: float = 0.0,
                 head_dropout: float = 0.0):
        super().__init__()
        self.spec = spec
        self.n_streams = n_streams
        self.head_drop = nn.Dropout(head_dropout)
        self.body = nn.Sequential(
            nn.LayerNorm(n_streams * d_model),
            nn.Linear(n_streams * d_model, spec.width),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(spec.width, d_model),
        )
        self.head_binary = nn.Linear(d_model, n_binary)

    def forward(self, tokens: torch.Tensor) -> Dict[str, torch.Tensor]:
        shared = self.body(tokens.flatten(1))
        B, N = tokens.shape[0], tokens.shape[1]
        # concat 沒有「每條流的權重」這個概念。填 1/N 是為了讓下游欄位存在，
        # 但必須標記出來——舊版沒有標記，並排比較時 concat 會顯示完美的
        # 0.333/0.333/0.333，看起來像模型學到的均衡，其實是硬填的常數。
        sw = torch.full((B, N), 1.0 / N, device=tokens.device, dtype=tokens.dtype)
        return {'logits_binary': self.head_binary(self.head_drop(shared)),
                'shared': shared, 'stream_weight': sw}

    def diagnostics(self) -> Dict[str, float]:
        return {}

    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# stream_weight 是不是模型學出來的。False 的話那組數字不可解讀。
STREAM_WEIGHT_IS_LEARNED = {'concat': False}


def build_cell(cell_id: str, spec: CellSpec, n_streams: int, d_model: int,
               n_binary: int = 2, dropout: float = 0.0,
               head_dropout: float = 0.0) -> nn.Module:
    if cell_id == 'concat':
        return ConcatCell(spec, n_streams, d_model, n_binary, dropout, head_dropout)
    return FusionCell(spec, n_streams, d_model, n_binary, dropout, head_dropout)


# ══════════════════════════════════════════════════════════════════════
# 容量對齊
# ══════════════════════════════════════════════════════════════════════

def count_cell_params(cell_id: str, spec: CellSpec, n_streams: int,
                      d_model: int, n_binary: int = 2) -> int:
    """實際建一次來數。比手推公式可靠——公式會跟實作漂移。"""
    with torch.no_grad():
        return build_cell(cell_id, spec, n_streams, d_model, n_binary).n_parameters()


def fit_width_to_budget(cell_id: str, spec: CellSpec, n_streams: int, d_model: int,
                        target: int, n_binary: int = 2,
                        width_max: int = 4096) -> Tuple[int, int]:
    """
    二分搜尋 width，讓 cell 的參數量儘量接近 `target`（取不超過 target 的最大者；
    連 width=1 都超過就回 width=1）。

    回傳 (width, 實際參數量)。實際值一定要跟著結果一起存——
    「我們對齊了參數量」是需要被查核的宣稱，不是可以用講的。
    """
    def n_at(w: int) -> int:
        return count_cell_params(
            cell_id, CellSpec(spec.order, spec.weighting, spec.level, w),
            n_streams, d_model, n_binary)

    if n_at(1) > target:
        return 1, n_at(1)

    lo, hi = 1, width_max
    if n_at(hi) <= target:
        return hi, n_at(hi)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if n_at(mid) <= target:
            lo = mid
        else:
            hi = mid - 1
    return lo, n_at(lo)


# ══════════════════════════════════════════════════════════════════════
# cell_id ↔ CellSpec
# ══════════════════════════════════════════════════════════════════════

_LEVEL_ALIAS = {'feat': 'feature', 'feature': 'feature',
                'dec': 'decision', 'decision': 'decision'}


def spec_from_cell_id(cell_id: str, width: int) -> CellSpec:
    """
    'o2.gated.dec' → CellSpec(order=2, weighting='gated', level='decision')。

    參照組 'concat' 沒有三軸座標，回傳一個佔位 spec（只有 width 有意義），
    分析時要靠 `is_grid_cell()` 把它排除在因子設計之外。
    """
    if cell_id == 'concat':
        return CellSpec(order=1, weighting='static', level='feature', width=width)

    parts = cell_id.split('.')
    if len(parts) != 3:
        raise ValueError(
            f"無法解析 cell_id '{cell_id}'。格式是 'o<1|2>.<static|gated>.<feat|dec>'，"
            f"或參照組 {REFERENCE_CELLS}")
    o, w, l = parts
    if not (o.startswith('o') and o[1:].isdigit()):
        raise ValueError(f"cell_id '{cell_id}' 的 order 欄位無法解析：'{o}'")
    if l not in _LEVEL_ALIAS:
        raise ValueError(f"cell_id '{cell_id}' 的 level 欄位無法解析：'{l}'")
    return CellSpec(order=int(o[1:]), weighting=w, level=_LEVEL_ALIAS[l], width=width)


def is_grid_cell(cell_id: str) -> bool:
    """True = 屬於 2×2×2 因子設計；False = 參照組，不進因子分析。"""
    return cell_id not in REFERENCE_CELLS


def all_cell_ids(include_reference: bool = True) -> List[str]:
    ids = [s.cell_id for s in all_cell_specs()]
    return ids + list(REFERENCE_CELLS) if include_reference else ids
