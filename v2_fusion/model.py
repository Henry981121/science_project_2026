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
      ├─ head_binary       → (B, 2)      正常梯度
      ├─ head_source       → (B, 19)     正常梯度，乘 lambda_src
      └─ GRL → head_gen    → (B, 17)     梯度反轉，強度 lambda_grl（只在這裡出現一次）

為什麼不做「CLIP 當 query、其餘四條當 key/value」的 cross-attention：
那等於在架構裡寫死「CLIP 是主、其餘是輔」。支持這個假設的證據
（ablation CLIP 0.30 最高）本身是被污染的 —— 其他三條流是隨機投影／隨機 CNN。
用「CLIP 最重要」去 justify 把 CLIP 設為 query，是循環論證。
可學習的 fusion token 不預設任何一條流為主，由資料決定。
"""

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from .config import ModelConfig


# ══════════════════════════════════════════════════════════════════════
# Gradient Reversal
# ══════════════════════════════════════════════════════════════════════

class _GradientReversal(torch.autograd.Function):
    """Forward 恆等；backward 乘上 -lambda_。"""

    @staticmethod
    def forward(ctx, x: torch.Tensor, lambda_: float):
        ctx.lambda_ = float(lambda_)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return -ctx.lambda_ * grad_output, None


def grad_reverse(x: torch.Tensor, lambda_: float) -> torch.Tensor:
    """
    ⚠️ lambda_ 只准在這裡出現一次。

    對應 audit 發現 05：舊版 s3_main_grl.py 把同一個 λ 同時給了
      :203  grad_reverse(shared, grl_lambda)      → backward × -λ
      :257  current_lambda_grl * loss_gen         → loss 權重 × λ
    結果 backbone 實收 λ²。標準 DANN（Ganin 2015）是二選一，不是兩個都放。
    v2 選「GRL 帶 -λ、loss 權重固定 1.0」：
      · discriminator 以全強度訓練 → 是個有能力的對手
      · backbone 收到 -λ           → 對抗強度就是 λ 本身
    LossConfig 沒有 gen loss 的權重欄位，所以 λ 不可能被乘第二次。
    """
    return _GradientReversal.apply(x, lambda_)


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

        missing = [s for s in self.streams if s not in cfg.stream_dims]
        if missing:
            raise ValueError(f"stream_dims 缺少這些流的維度：{missing}")

        self.adapters = nn.ModuleDict({
            s: StreamAdapter(cfg.stream_dims[s], d) for s in self.streams
        })

        self.stream_embed = nn.Parameter(
            torch.randn(1, self.n_streams, d) * cfg.stream_embed_std)

        self.layers = nn.ModuleList([
            StreamSelfAttention(d, cfg.n_heads, cfg.dropout_attn, cfg.ffn_mult)
            for _ in range(cfg.n_layers)
        ])

        self.readout = CrossAttentionReadout(d, cfg.n_heads, cfg.dropout_attn)

        self.head_drop = nn.Dropout(cfg.dropout_head)
        self.head_binary = nn.Linear(d, cfg.n_binary)
        self.head_source = nn.Linear(d, cfg.n_sources)
        self.head_gen = nn.Sequential(
            nn.Linear(d, d // 2),
            nn.GELU(),
            nn.Dropout(cfg.dropout_head),
            nn.Linear(d // 2, cfg.n_gen),
        )

    # ── 輸入 ──────────────────────────────────────────────────────────
    def _to_tokens(self, feats: Dict[str, torch.Tensor]) -> torch.Tensor:
        missing = [s for s in self.streams if s not in feats]
        if missing:
            raise KeyError(f"forward() 少了這些流：{missing}")
        toks = [self.adapters[s](feats[s]) for s in self.streams]
        return torch.stack(toks, dim=1) + self.stream_embed   # (B, N, d)

    def forward(
        self,
        feats: Dict[str, torch.Tensor],
        grl_lambda: float = 0.0,
    ) -> Dict[str, torch.Tensor]:
        """
        feats:      {stream_name: (B, in_dim)}
        grl_lambda: 當前 epoch 的 GRL 強度。λ 只在這裡進入模型一次。
        """
        tokens = self._to_tokens(feats)

        self_attn_w = None
        for layer in self.layers:
            tokens, self_attn_w = layer(tokens)

        shared, readout_attn = self.readout(tokens)
        h = self.head_drop(shared)

        out = {
            'logits_binary': self.head_binary(h),
            'logits_source': self.head_source(h),
            'logits_gen':    self.head_gen(grad_reverse(h, grl_lambda)),
            'shared':        shared,
            'readout_attn':  readout_attn,   # (B, N) —— XAI 直接用這個
            'self_attn':     self_attn_w,    # (B, N, N)
        }
        return out

    # ── 便利方法 ──────────────────────────────────────────────────────
    def n_parameters(self) -> Dict[str, int]:
        def count(m):
            return sum(p.numel() for p in m.parameters() if p.requires_grad)
        return {
            'adapters':  count(self.adapters),
            'self_attn': count(self.layers),
            'readout':   count(self.readout) + self.stream_embed.numel(),
            'heads':     count(self.head_binary) + count(self.head_source) + count(self.head_gen),
            'total':     sum(p.numel() for p in self.parameters() if p.requires_grad),
        }

    def describe(self) -> str:
        p = self.n_parameters()
        dims = ', '.join(f"{s}:{self.cfg.stream_dims[s]}" for s in self.streams)
        return (
            f"FusionDetectorV2\n"
            f"  streams   : {self.n_streams} ({dims}) -> d_model={self.cfg.d_model}\n"
            f"  fusion    : {self.cfg.n_layers}x StreamSelfAttention + CrossAttentionReadout\n"
            f"  params    : adapters {p['adapters']:,} | self-attn {p['self_attn']:,} | "
            f"readout {p['readout']:,} | heads {p['heads']:,}\n"
            f"  total     : {p['total']:,}"
        )
