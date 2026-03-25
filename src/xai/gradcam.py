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
import cv2
from typing import Dict, Optional, Tuple


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
# Multi-Stream Explainer
# ──────────────────────────────────────────────────────────────────────

class MultiStreamExplainer:
    """
    Generates per-stream explanations:
      - CLIP:  Attention Rollout
      - FFT:   FFT spectrum map
      - DCT:   DCT map
      - DIRE:  reconstruction error map
      - Noise: SRM noise map
    Then aggregates into a single heatmap.
    """

    PATCH_GRID = 16  # ViT-L/14 -> 16x16 patches for 224x224

    def __init__(self, model, device: str = "cuda"):
        """
        model: AIImageDetector instance
        """
        self.model  = model
        self.device = device

    @torch.no_grad()
    def explain(self, image: torch.Tensor) -> Dict[str, np.ndarray]:
        """
        Args:
            image: (1, 3, H, W) ImageNet-normalized

        Returns:
            dict of stream_name -> explanation array (H, W) in [0,1]
        """
        exps = {}

        # 1. CLIP — Attention Rollout
        rollout = self.model.extractors["clip"].get_attention_rollout(image)
        # rollout: (1, 256) for 16x16 patches
        r = rollout[0].cpu().numpy()
        grid = int(r.shape[0] ** 0.5)
        r_map = r.reshape(grid, grid)
        exps["clip"] = self._normalize(r_map)

        # 2. FFT — spectrum map
        spec = self.model.extractors["fft"].get_spectrum(image)
        exps["fft"] = self._normalize(spec[0].mean(0).cpu().numpy())

        # 3. DCT — DCT map
        dct_map = self.model.extractors["dct"].get_dct_visualization(image)
        exps["dct"] = self._normalize(dct_map[0].mean(0).cpu().numpy())

        # 4. DIRE — reconstruction error
        dire_map = self.model.extractors["dire"].get_dire_visualization(image)
        exps["dire"] = self._normalize(dire_map[0].mean(0).cpu().numpy())

        # 5. Noise — SRM noise map
        noise_map = self.model.extractors["noise"].get_noise_map(image)
        exps["noise"] = self._normalize(noise_map[0].mean(0).cpu().numpy())

        return exps

    def aggregate(self, exps: Dict[str, np.ndarray], target_hw: Tuple[int,int] = (224, 224)) -> np.ndarray:
        """Resize all maps to target_hw, then average."""
        maps = []
        for name, m in exps.items():
            resized = cv2.resize(m.astype(np.float32), (target_hw[1], target_hw[0]))
            maps.append(resized)
        return self._normalize(np.mean(maps, axis=0))

    def overlay(
        self,
        image_np: np.ndarray,
        heatmap: np.ndarray,
        alpha: float = 0.5,
    ) -> np.ndarray:
        """
        Overlay heatmap (H,W, [0,1]) onto image_np (H,W,3 uint8).
        """
        h, w = image_np.shape[:2]
        heat = cv2.resize(heatmap, (w, h))
        heat_uint8 = (heat * 255).astype(np.uint8)
        colored = cv2.applyColorMap(heat_uint8, cv2.COLORMAP_JET)
        colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
        return (colored * alpha + image_np * (1 - alpha)).astype(np.uint8)

    def visualize_all(
        self,
        image_np: np.ndarray,
        exps: Dict[str, np.ndarray],
        save_dir: Optional[str] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Return overlay images for each stream + aggregated.
        Optionally save to save_dir.
        """
        result = {}
        target_hw = image_np.shape[:2]

        for name, m in exps.items():
            result[name] = self.overlay(image_np, m)

        result["aggregated"] = self.overlay(
            image_np, self.aggregate(exps, target_hw)
        )

        if save_dir:
            import os
            from PIL import Image
            os.makedirs(save_dir, exist_ok=True)
            for name, img in result.items():
                Image.fromarray(img).save(os.path.join(save_dir, f"{name}.png"))
            print(f"[XAI] Saved to {save_dir}")

        return result

    @staticmethod
    def _normalize(arr: np.ndarray) -> np.ndarray:
        mn, mx = arr.min(), arr.max()
        return (arr - mn) / (mx - mn + 1e-8)


# ──────────────────────────────────────────────────────────────────────
# Quick test
# ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
    from src.fusion.fusion_module import AIImageDetector

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AIImageDetector(device=device)
    img = torch.randn(1, 3, 224, 224).to(device)

    explainer = MultiStreamExplainer(model, device=device)
    exps = explainer.explain(img)

    for name, m in exps.items():
        print(f"  {name}: {m.shape}")

    agg = explainer.aggregate(exps)
    print(f"  aggregated: {agg.shape}")
    print("XAI OK")
