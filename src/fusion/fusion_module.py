"""
Cross-Attention Fusion Module + Full Model (v2.1)

Architecture:
  5 streams (each 512d)
  -> CrossAttentionFusion (2-layer Transformer)
  -> Shared Backbone: Linear 2560->1024->512
  -> Head 1: Binary (Real/Fake)
  -> Head 2: Multi-class (10 AI source categories)

Loss = CE_binary + 0.3 * CE_multiclass
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple

from ..feature_extractors import (
    CLIPFeatureExtractor,
    FFTFeatureExtractor,
    DCTFeatureExtractor,
    NoisePrintExtractor,
)
# DIREFeatureExtractor 暫時排除（太慢且 VRAM 需求大）
# from ..feature_extractors import DIREFeatureExtractor


# ──────────────────────────────────────────────────────────────────────
# Cross-Attention Fusion
# ──────────────────────────────────────────────────────────────────────

class CrossAttentionFusionLayer(nn.Module):
    """Single Cross-Attention layer with FFN and residual."""

    def __init__(self, d_model: int = 512, n_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """x: (B, N, d_model) — N=5 streams"""
        attn_out, attn_weights = self.self_attn(x, x, x)
        x = self.norm1(x + attn_out)
        x = self.norm2(x + self.ffn(x))
        return x, attn_weights  # attn_weights: (B, N, N)


class CrossAttentionFusion(nn.Module):
    """
    2-layer Cross-Attention Transformer for fusing 5 stream tokens.

    Input:  Dict of 5 stream features, each (B, 512)
    Output: (B, 2560) — concatenation of all 5 attended tokens
            Also returns attention weights for XAI.
    """

    DEFAULT_STREAMS = ["clip", "fft", "dct", "noise"]

    def __init__(self, d_model: int = 512, n_heads: int = 8,
                 n_layers: int = 2, dropout: float = 0.1,
                 stream_names: list = None):
        super().__init__()
        self.stream_names = stream_names or self.DEFAULT_STREAMS
        self.layers = nn.ModuleList([
            CrossAttentionFusionLayer(d_model, n_heads, dropout)
            for _ in range(n_layers)
        ])
        # Learnable stream-type embeddings
        self.stream_embed = nn.Parameter(torch.randn(1, len(self.stream_names), d_model) * 0.02)

    def forward(self, features: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        features: {'clip': (B,512), 'fft': (B,512), ...}
        returns:
          fused: (B, 2560)           concat of 5 attended tokens
          attn:  (B, 5, 5)           last layer attention weights
        """
        tokens = torch.stack(
            [features[name] for name in self.stream_names], dim=1
        )  # (B, N, 512)
        tokens = tokens + self.stream_embed  # add positional/type embedding

        attn_weights = None
        for layer in self.layers:
            tokens, attn_weights = layer(tokens)

        fused = tokens.flatten(1)  # (B, N*512)
        return fused, attn_weights


# ──────────────────────────────────────────────────────────────────────
# Full Model
# ──────────────────────────────────────────────────────────────────────

class AIImageDetector(nn.Module):
    """
    Complete AI-generated image detection model v2.1.

    5 streams -> Cross-Attention Fusion -> Shared Backbone -> 2 Heads
    """

    N_BINARY   = 2   # Real / Fake
    N_SOURCES  = 10  # AI source categories (matches GenImage)
    STREAM_NAMES = ["clip", "fft", "dct", "noise"]

    def __init__(self, device: str = "cuda", dropout: float = 0.3):
        super().__init__()
        self.device = device
        n_streams = len(self.STREAM_NAMES)

        # ── Feature extractors ──────────────────────────────────────
        self.extractors = nn.ModuleDict({
            "clip":  CLIPFeatureExtractor(device=device),
            "fft":   FFTFeatureExtractor(device=device),
            "dct":   DCTFeatureExtractor(device=device),
            "noise": NoisePrintExtractor(device=device),
        })

        # ── Cross-Attention Fusion ──────────────────────────────────
        self.fusion = CrossAttentionFusion(
            d_model=512, n_heads=8, n_layers=2,
            stream_names=self.STREAM_NAMES,
        )

        # ── Shared Backbone ─────────────────────────────────────────
        self.backbone = nn.Sequential(
            nn.Linear(n_streams * 512, 1024),
            nn.LayerNorm(1024),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(1024, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # ── Classification Heads ─────────────────────────────────────
        self.head_binary = nn.Linear(512, self.N_BINARY)
        self.head_source = nn.Linear(512, self.N_SOURCES)

        self.to(device)
        print(f"[AIImageDetector] v2.1 ready | device={device}")
        print(f"  Streams: {', '.join(self.STREAM_NAMES)} (each 512d)")
        print(f"  Fusion:  2-layer Cross-Attention")
        print(f"  Backbone: {n_streams*512}->1024->512")
        print(f"  Heads: binary({self.N_BINARY}) + source({self.N_SOURCES})")

    def _extract_all(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        feats = {}
        for name, ext in self.extractors.items():
            feats[name] = ext(images)
        return feats

    def forward(
        self, images: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            images: (B, 3, H, W) ImageNet-normalized

        Returns:
            logits_binary: (B, 2)
            logits_source: (B, 10)
            attn_weights:  (B, 5, 5)
        """
        stream_features = self._extract_all(images)
        fused, attn = self.fusion(stream_features)          # (B, 2560), (B,5,5)
        shared = self.backbone(fused)                        # (B, 512)
        logits_binary = self.head_binary(shared)             # (B, 2)
        logits_source = self.head_source(shared)             # (B, 10)
        return logits_binary, logits_source, attn

    def predict(self, images: torch.Tensor) -> Dict:
        """Inference only — returns probabilities and label."""
        self.eval()
        with torch.no_grad():
            lb, ls, attn = self.forward(images)
        prob_binary = F.softmax(lb, dim=1)   # (B, 2)
        prob_source = F.softmax(ls, dim=1)   # (B, 10)
        labels = prob_binary.argmax(dim=1)   # 0=Real, 1=Fake
        confidence = prob_binary.max(dim=1).values
        return {
            "label": labels,           # (B,) 0=Real,1=Fake
            "confidence": confidence,  # (B,)
            "prob_binary": prob_binary,
            "prob_source": prob_source,
            "attn_weights": attn,
        }

    def get_stream_features(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Return raw stream features for ablation / GradCAM."""
        self.eval()
        with torch.no_grad():
            return self._extract_all(images)


# ──────────────────────────────────────────────────────────────────────
# Loss function
# ──────────────────────────────────────────────────────────────────────

class DualHeadLoss(nn.Module):
    """CE_binary + lambda * CE_multiclass"""

    def __init__(self, lambda_source: float = 0.3):
        super().__init__()
        self.lambda_source = lambda_source
        self.ce = nn.CrossEntropyLoss()

    def forward(
        self,
        logits_binary: torch.Tensor,
        logits_source: torch.Tensor,
        labels_binary: torch.Tensor,
        labels_source: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict]:
        loss_binary = self.ce(logits_binary, labels_binary)

        if labels_source is not None:
            loss_source = self.ce(logits_source, labels_source)
            total = loss_binary + self.lambda_source * loss_source
        else:
            loss_source = torch.tensor(0.0)
            total = loss_binary

        return total, {
            "loss_binary": loss_binary.item(),
            "loss_source": loss_source.item() if labels_source is not None else 0.0,
            "loss_total": total.item(),
        }


# ──────────────────────────────────────────────────────────────────────
# Quick test
# ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AIImageDetector(device=device)
    imgs = torch.randn(2, 3, 224, 224).to(device)
    lb, ls, attn = model(imgs)
    print(f"Binary logits: {lb.shape}")   # (2,2)
    print(f"Source logits: {ls.shape}")   # (2,10)
    print(f"Attn weights:  {attn.shape}") # (2,5,5)

    criterion = DualHeadLoss(lambda_source=0.3)
    y_bin = torch.randint(0, 2, (2,)).to(device)
    y_src = torch.randint(0, 10, (2,)).to(device)
    loss, info = criterion(lb, ls, y_bin, y_src)
    print(f"Loss: {loss.item():.4f} | {info}")
    print("fusion_module OK")
