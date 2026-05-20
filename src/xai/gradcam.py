"""
XAI Module v2.1
- GradCAM on CNN layer4 for FFT / DCT / DIRE / Noise streams
- Attention Rollout for CLIP stream
- Multi-stream aggregated heatmap
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional


# ──────────────────────────────────────────────────────────────────────
# GradCAM (for CNN-based streams: FFT, DCT, DIRE, Noise)
# ──────────────────────────────────────────────────────────────────────

class GradCAM:
    """
    Standard GradCAM.
    Attach to any nn.Module layer (e.g., ResNet layer4).
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self._activations = None
        self._gradients   = None
        self._fwd_handle = target_layer.register_forward_hook(self._fwd_hook)
        self._bwd_handle = target_layer.register_full_backward_hook(self._bwd_hook)

    def _fwd_hook(self, module, inp, out):
        self._activations = out.detach()

    def _bwd_hook(self, module, grad_in, grad_out):
        self._gradients = grad_out[0].detach()

    def remove(self):
        self._fwd_handle.remove()
        self._bwd_handle.remove()

    def compute(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None,
        output_index: int = 0,  # 0=binary head, 1=source head
    ) -> np.ndarray:
        """
        Compute GradCAM heatmap.

        Args:
            input_tensor: (1, 3, H, W)
            target_class:  class index (None = argmax)
            output_index:  which model output tuple index to use

        Returns:
            cam: (H, W) float32 in [0, 1]
        """
        self.model.eval()
        input_tensor = input_tensor.requires_grad_(True)

        outputs = self.model(input_tensor)
        # outputs is a tuple: (logits_binary, logits_source, attn)
        logits = outputs[output_index]

        if target_class is None:
            target_class = logits.argmax(dim=1).item()

        self.model.zero_grad()
        logits[0, target_class].backward()

        # weights: global average pooled gradients
        weights = self._gradients.mean(dim=[2, 3], keepdim=True)  # (1, C, 1, 1)
        cam = (weights * self._activations).sum(dim=1, keepdim=True)  # (1,1,h,w)
        cam = F.relu(cam).squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam


# ──────────────────────────────────────────────────────────────────────
# Multi-Stream Explainer — DEPRECATED
# ──────────────────────────────────────────────────────────────────────
# 舊的 MultiStreamExplainer 整個包在 @torch.no_grad() 底下，回傳的只是
# 輸入域變換（FFT spectrum / DCT map / DIRE error / SRM noise）而不是真
# 對五流融合模型的 Grad-CAM——本質上不是「解釋」。
#
# 真版本見 src.xai.per_stream_gradcam.PerStreamExplainer：
#   - FFT / DCT / DIRE / Noise: 真 Grad-CAM
#   - CLIP: attention rollout（Day 2-3 換成 Chefer relevance）
# ──────────────────────────────────────────────────────────────────────

class MultiStreamExplainer:
    """Deprecated — 立即指向 PerStreamExplainer。"""

    def __init__(self, *args, **kwargs):
        raise RuntimeError(
            "MultiStreamExplainer 已棄用（舊版不算真實梯度，回傳的不是 Grad-CAM）。"
            "請改用 src.xai.per_stream_gradcam.PerStreamExplainer。"
        )
