"""
v2_fusion.model — 融合架構
===========================
資料流：

  每條流 (B, in_dim_s)
      │
      ├─ StreamAdapter_s : LayerNorm(in_dim) → Linear(in_dim → d) → LayerNorm(d)
      │    · LayerNorm 在前：把各流拉到可比的尺度（audit 發現 12）
      │    · Linear 可訓練：這就是從 extractor 搬進來的投影層（audit 發現 01、16）
      │    · 每條流各自一份：流身份由此承擔，比加法式 embedding 強
      ▼
  (B, N, d) stream tokens  + stream_embed
      │
      ├─ StreamSelfAttention × n_layers      ← 名副其實：Q=K=V，流之間互相溝通
      ▼
  (B, N, d)
      │
      ├─ CrossAttentionReadout               ← 名副其實：Q 來自 fusion token，K/V 來自流
      │    可學習的 fusion token 去「問」五條流                    (audit 發現 13、15)
      ▼
  (B, d) shared
      ├─ head_binary       → (B, 2)      真/假，正常梯度
      └─ head_source       → (B, 10)     哪個 source，正常梯度，乘 lambda_src
                                          （auxiliary，預設 lambda_src=0 關閉）

2026-08-27：GRL 已移除
----------------------
原本這裡還有第三條 `GRL → head_gen` 的對抗分支。方向上不採用 domain
adaptation 之後，整條路徑連同 lambda_grl 一起刪除，而不是留著預設 0 ——
留著就還有人會去設它，也還要在論文裡解釋一個沒有使用的模組。
`head_source` 不是 GRL：它是正常梯度的 auxiliary head，兩者不要混為一談。

為什麼不做「CLIP 當 query、其餘當 key/value」的 cross-attention：
那等於在架構裡寫死「CLIP 是主、其餘是輔」。支持這個假設的證據
（舊版 ablation CLIP 0.30 最高）本身是被污染的 —— 當時其他流是隨機投影／隨機 CNN。
用「CLIP 最重要」去 justify 把 CLIP 設為 query，是循環論證。
可學習的 fusion token 不預設任何一條流為主，由資料決定。
"""

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from .config import ModelConfig
from .fusion_cells import (build_cell, spec_from_cell_id, fit_width_to_budget,
                           STREAM_WEIGHT_IS_LEARNED)


# ══════════════════════════════════════════════════════════════════════
# Stream adapter
# ══════════════════════════════════════════════════════════════════════

class StreamAdapter(nn.Module):
    """
    把一條流的 cache 特徵轉成 d_model 維的 token。

    LayerNorm(in_dim) 是關鍵的一步（audit 發現 12）：五條流的尺度天差地遠，
    而 attention 的 softmax 對輸入尺度極敏感，沒有這個正規化的話
    attention 分布會被 feature norm 最大的流主導 —— 「CLIP attention 0.51 最高」
    可能只是因為 CLIP 的 norm 最大，不是模型真的比較看重它。

    中間的 Linear 就是原本寫死在 extractor 裡、隨機初始化且從未訓練的
    那個投影層（audit 發現 01）。搬到這裡之後它跟著任務一起訓練。
    in_dim == d_model 時它是一個可訓練的方陣，仍然有意義。
    """

    def __init__(self, in_dim: int, d_model: int):
        super().__init__()
        self.in_dim = in_dim
        self.d_model = d_model
        self.norm_in = nn.LayerNorm(in_dim)
        self.proj = nn.Linear(in_dim, d_model)
        self.norm_out = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm_out(self.proj(self.norm_in(x)))


# ══════════════════════════════════════════════════════════════════════
# Attention blocks
# ══════════════════════════════════════════════════════════════════════

class StreamSelfAttention(nn.Module):
    """
    流之間互相溝通。Q = K = V = stream tokens，所以這是 self-attention，
    命名如實（audit 發現 13：舊版把這個叫 CrossAttentionFusionLayer）。
    Pre-norm 結構，比舊版的 post-norm 好訓練。
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float, ffn_mult: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * ffn_mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * ffn_mult, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.norm1(x)
        attn_out, attn_w = self.attn(h, h, h, need_weights=True, average_attn_weights=True)
        x = x + attn_out
        x = x + self.ffn(self.norm2(x))
        return x, attn_w          # attn_w: (B, N, N)


class CrossAttentionReadout(nn.Module):
    """
    可學習的 fusion token 對五條流做 cross-attention。
    Q 來自 fusion token、K/V 來自 stream tokens，Q ≠ K/V —— 這在定義上就是
    cross-attention，不是硬拗（audit 發現 13）。

    取代舊版的 flatten(1) → Linear(2560→1024)（audit 發現 15，2.62M 參數、
    佔全模型 27.3%）。除了參數變少，更重要的是 XAI 讀出變乾淨：
    這一層的 attention 天然就是 (B, 1, N) —— 「模型讀出時看了每條流多少」，
    不需要對 5×5 矩陣做任何摺疊假設。
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        self.query = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.query, std=0.02)
        self.norm_q = nn.LayerNorm(d_model)
        self.norm_kv = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm_out = nn.LayerNorm(d_model)

    def forward(self, tokens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B = tokens.shape[0]
        q = self.norm_q(self.query.expand(B, -1, -1))     # (B, 1, d)
        kv = self.norm_kv(tokens)                          # (B, N, d)
        out, attn_w = self.attn(q, kv, kv, need_weights=True, average_attn_weights=True)
        fused = self.norm_out(out.squeeze(1))              # (B, d)
        return fused, attn_w.squeeze(1)                    # attn_w: (B, N)


# ══════════════════════════════════════════════════════════════════════
# Full model
# ══════════════════════════════════════════════════════════════════════

class FusionDetectorV2(nn.Module):

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.streams: List[str] = list(cfg.streams)
        self.n_streams = len(self.streams)
        d = cfg.d_model
        self.use_cell = cfg.fusion_cell is not None
        if not self.use_cell and cfg.fusion_mode not in {'hybrid', 'self', 'cross', 'concat'}:
            raise ValueError(f"unknown fusion_mode: {cfg.fusion_mode}")

        missing = [s for s in self.streams if s not in cfg.stream_dims]
        if missing:
            raise ValueError(f"stream_dims 缺少這些流的維度：{missing}")

        self.adapters = nn.ModuleDict({
            s: StreamAdapter(cfg.stream_dims[s], d) for s in self.streams
        })

        self.stream_embed = nn.Parameter(
            torch.randn(1, self.n_streams, d) * cfg.stream_embed_std)

        # ── 三軸因子設計路徑 ──────────────────────────────────────────
        # cell 自己擁有 binary head（decision-level 的頭在各流上、
        # feature-level 的在融合後 —— 頭放在 cell 外面，C 軸就不只變一件事）。
        if self.use_cell:
            width = cfg.cell_width
            if cfg.cell_budget is not None:
                width, _ = fit_width_to_budget(
                    cfg.fusion_cell,
                    spec_from_cell_id(cfg.fusion_cell, width),
                    self.n_streams, d, cfg.cell_budget, cfg.n_binary)
            self.cell = build_cell(
                cfg.fusion_cell, spec_from_cell_id(cfg.fusion_cell, width),
                self.n_streams, d, cfg.n_binary,
                dropout=cfg.dropout_attn, head_dropout=cfg.dropout_head)
            self.cell_width = width
            # 把實際生效的值寫回 config，results.json 才會記錄到**真正發生的事**。
            # 不寫回的話，budget 反推出來的 width 是 192，config 裡卻永遠是預設的
            # 64 —— 那就是 audit 抓過的「結果檔記錄到沒發生的事」同一個病。
            cfg.cell_width = width
            cfg.cell_params = self.cell.n_parameters()
        else:
            self.cell = None
            self.cell_width = None

        self.layers = nn.ModuleList([
            StreamSelfAttention(d, cfg.n_heads, cfg.dropout_attn, cfg.ffn_mult)
            for _ in range(cfg.n_layers)
        ]) if (not self.use_cell and cfg.fusion_mode in {'hybrid', 'self'}) else nn.ModuleList()

        self.readout = (CrossAttentionReadout(d, cfg.n_heads, cfg.dropout_attn)
                        if (not self.use_cell and cfg.fusion_mode in {'hybrid', 'cross'})
                        else None)
        self.concat_head = (nn.Sequential(
            nn.LayerNorm(self.n_streams * d), nn.Linear(self.n_streams * d, d),
            nn.GELU(), nn.Dropout(cfg.dropout_attn))
            if (not self.use_cell and cfg.fusion_mode == 'concat') else None)

        self.head_drop = nn.Dropout(cfg.dropout_head)
        self.head_binary = (None if self.use_cell else nn.Linear(d, cfg.n_binary))
        # auxiliary，非對抗。use_source_head=False 時連參數都不建立，
        # 這樣 lambda_src 設了也不會有「有頭但沒梯度」的模糊狀態。
        self.head_source = (
            nn.Linear(d, cfg.n_sources) if cfg.use_source_head else None)

    @property
    def readout_attn_learned(self) -> bool:
        """
        readout_attn 那組數字是不是模型學出來的。

        False 代表它是硬填的 1/N，拿去解讀「模型比較看重哪條流」就是編故事。
        legacy 的 'self' / 'concat' 模式與參照組 ConcatCell 都屬於這一類。
        """
        if self.use_cell:
            return STREAM_WEIGHT_IS_LEARNED.get(self.cfg.fusion_cell, True)
        return self.cfg.fusion_mode in {'hybrid', 'cross'}

    # ── 輸入 ──────────────────────────────────────────────────────────
    def _to_tokens(self, feats: Dict[str, torch.Tensor]) -> torch.Tensor:
        missing = [s for s in self.streams if s not in feats]
        if missing:
            raise KeyError(f"forward() 少了這些流：{missing}")
        toks = [self.adapters[s](feats[s]) for s in self.streams]
        return torch.stack(toks, dim=1) + self.stream_embed   # (B, N, d)

    def forward(self, feats: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """feats: {stream_name: (B, in_dim)}"""
        tokens = self._to_tokens(feats)

        if self.cell is not None:
            cell_out = self.cell(tokens)
            out = {
                'logits_binary': cell_out['logits_binary'],
                'shared':        cell_out['shared'],
                # 逐樣本 (B, N)。per-sample 變異數才能判斷權重有沒有真的隨樣本變；
                # 只存 batch mean 的話，退化成常數的權重看起來會像學到了東西。
                'readout_attn':  cell_out['stream_weight'],
                'self_attn':     None,
                'readout_attn_learned': self.readout_attn_learned,
            }
            if 'pair_weight' in cell_out:
                out['pair_weight'] = cell_out['pair_weight']
            if self.head_source is not None:
                out['logits_source'] = self.head_source(
                    self.head_drop(cell_out['shared']))
            return out

        self_attn_w = None
        for layer in self.layers:
            tokens, self_attn_w = layer(tokens)

        if self.cfg.fusion_mode in {'hybrid', 'cross'}:
            shared, readout_attn = self.readout(tokens)
        elif self.cfg.fusion_mode == 'self':
            shared = tokens.mean(dim=1)
            readout_attn = torch.full(
                (tokens.shape[0], tokens.shape[1]), 1.0 / tokens.shape[1],
                device=tokens.device, dtype=tokens.dtype)
        else:
            shared = self.concat_head(tokens.flatten(1))
            readout_attn = torch.full(
                (tokens.shape[0], tokens.shape[1]), 1.0 / tokens.shape[1],
                device=tokens.device, dtype=tokens.dtype)
        h = self.head_drop(shared)

        out = {
            'logits_binary': self.head_binary(h),
            'shared':        shared,
            'readout_attn':  readout_attn,   # (B, N) —— XAI 直接用這個
            'self_attn':     self_attn_w,    # (B, N, N)
            # False = 這組數字是硬填的 1/N，不是模型學的。'self' 和 'concat'
            # 模式下沒有這個標記的話，並排比較時會看到完美的 0.333/0.333/0.333，
            # 看起來像學到的均衡，其實只是常數。
            'readout_attn_learned': self.readout_attn_learned,
        }
        if self.head_source is not None:
            out['logits_source'] = self.head_source(h)
        return out

    # ── 便利方法 ──────────────────────────────────────────────────────
    def n_parameters(self) -> Dict[str, int]:
        def count(m):
            return sum(p.numel() for p in m.parameters() if p.requires_grad)
        return {
            'adapters':  count(self.adapters),
            'self_attn': count(self.layers),
            # cell 路徑下這一欄就是「被比較的那個模組」的大小 ——
            # 容量對齊對齊的正是它，所以它必須單獨可讀。
            'fusion':    count(self.cell) if self.cell is not None else 0,
            'readout':   (count(self.readout) if self.readout is not None else 0)
                         + (count(self.concat_head) if self.concat_head is not None else 0)
                         + self.stream_embed.numel(),
            'heads':     (count(self.head_binary) if self.head_binary is not None else 0)
                         + (count(self.head_source) if self.head_source is not None else 0),
            'total':     sum(p.numel() for p in self.parameters() if p.requires_grad),
        }

    def fusion_diagnostics(self) -> Dict[str, object]:
        """訓練後可直接讀的中介變數（cell 路徑才有）。"""
        return self.cell.diagnostics() if self.cell is not None else {}

    def describe(self) -> str:
        p = self.n_parameters()
        dims = ', '.join(f"{s}:{self.cfg.stream_dims[s]}" for s in self.streams)
        if self.use_cell:
            fusion_desc = (f"cell={self.cfg.fusion_cell} width={self.cell_width}"
                           f" params={p['fusion']:,}"
                           + (f" (budget {self.cfg.cell_budget:,})"
                              if self.cfg.cell_budget else ""))
        else:
            fusion_desc = f"legacy fusion_mode={self.cfg.fusion_mode}"
        return (
            f"FusionDetectorV2\n"
            f"  streams   : {self.n_streams} ({dims}) -> d_model={self.cfg.d_model}\n"
            f"  fusion    : {fusion_desc}\n"
            f"  attn_w    : {'learned' if self.readout_attn_learned else 'CONSTANT 1/N (not learned)'}\n"
            f"  params    : adapters {p['adapters']:,} | self-attn {p['self_attn']:,} | "
            f"fusion {p['fusion']:,} | readout {p['readout']:,} | heads {p['heads']:,}\n"
            f"  total     : {p['total']:,}"
        )
