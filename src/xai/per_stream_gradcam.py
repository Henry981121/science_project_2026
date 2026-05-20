"""
per_stream_gradcam.py
======================

Day 1 模組 — 取代 src/xai/gradcam.py 裡那個沒在算梯度的 MultiStreamExplainer。

【做什麼】
對一張測試圖，產生 5 條流各自的空間 attribution heatmap：
  - FFT/DCT/DIRE/Noise (CNN)  : 真正的 Grad-CAM (Selvaraju 2017)，
                                target 為各 backbone 的最後 conv block，
                                backward target 為該流自己的 EXP-A head 的 fake logit。
  - CLIP (ViT)                : Day 1 暫時用 attention rollout (Abnar 2020)，
                                Day 2-3 會升級成 Chefer relevance (CVPR 2021)。

最終輸出 1×6 圖：原圖 | CLIP | FFT | DCT | DIRE | Noise

【重要】
- extract_features() 一定要傳 xai_mode=True，否則梯度被 no_grad 切掉。
- 預期視覺結果：CLIP 那格訊號最強（47-57% attention），其他四格相對淡。
  這是模型實際分工，不是 bug。

【替換對象】
舊的 src/xai/gradcam.py:85-199 的 MultiStreamExplainer（已知問題：
  - 整個包在 @torch.no_grad() 底下
  - FFT/DCT/DIRE/Noise 那四條回傳的是輸入域變換，不是 Grad-CAM）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import matplotlib.pyplot as plt
from typing import Dict, Optional, List


STREAMS = ['clip', 'fft', 'dct', 'dire', 'noise']
STREAM_DISPLAY = {
    'clip':  'CLIP',
    'fft':   'FFT',
    'dct':   'DCT',
    'dire':  'DIRE',
    'noise': 'Noise',
}
CNN_STREAMS = ['fft', 'dct', 'dire', 'noise']


# ──────────────────────────────────────────────────────────────────────
# 1. Grad-CAM hook（與 ai_detector_demo.py:54-77 同公式）
# ──────────────────────────────────────────────────────────────────────

class GradCAMHook:
    """Hook 一個 conv 層，抓 forward activation + backward gradient，算 Grad-CAM。"""

    def __init__(self, target_layer: nn.Module):
        self.activations: Optional[torch.Tensor] = None
        self.gradients: Optional[torch.Tensor] = None
        self._h_fwd = target_layer.register_forward_hook(self._fwd_hook)
        self._h_bwd = target_layer.register_full_backward_hook(self._bwd_hook)

    def _fwd_hook(self, module, inp, out):
        self.activations = out

    def _bwd_hook(self, module, grad_in, grad_out):
        self.gradients = grad_out[0]

    def compute(self) -> Optional[np.ndarray]:
        """Grad-CAM 公式：α_k = GAP(∂y/∂A_k)；CAM = ReLU(Σ α_k · A_k)。"""
        if self.activations is None or self.gradients is None:
            return None
        weights = self.gradients.mean(dim=[2, 3], keepdim=True)  # (1, C, 1, 1)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)  # (1, 1, H, W)
        cam = F.relu(cam).squeeze().detach().cpu().numpy()
        if cam.ndim < 2:
            return None
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam

    def remove(self):
        self._h_fwd.remove()
        self._h_bwd.remove()


# ──────────────────────────────────────────────────────────────────────
# 2. Target layer finder（與 ai_detector_demo.py:150-164 一致）
# ──────────────────────────────────────────────────────────────────────

def find_target_layer(ext: nn.Module, name: str) -> Optional[nn.Module]:
    """
    根據 stream 名字找 Grad-CAM target layer。
    Pre-flight Risk A 確認的對應：
        fft   → ext.backbone     的最後 BasicBlock   (1, 512,  7, 7)
        dct   → ext.feature_net  的最後 BasicBlock   (1, 512,  7, 7)
        dire  → ext.feature_net  的最後 Bottleneck   (1, 2048, 7, 7)
        noise → ext.cnn          的最後 ReLU         (1, 512, 28, 28)
    """
    if name == 'fft':
        children = list(ext.backbone.children())
    elif name in ('dct', 'dire'):
        children = list(ext.feature_net.children())
    elif name == 'noise':
        children = list(ext.cnn.children())
    else:
        return None

    # 從後往前找最後一個 Sequential block（ResNet layer4 那種）
    for c in reversed(children):
        if isinstance(c, nn.Sequential) and len(list(c.children())) > 0:
            return c[-1]
    return children[-2] if len(children) > 1 else None


# ──────────────────────────────────────────────────────────────────────
# 3. Per-stream explainer 主類
# ──────────────────────────────────────────────────────────────────────

class PerStreamExplainer:
    """
    對一張圖產生 5 條流各自的 spatial attribution。

    Args:
        extractors: dict {stream_name: feature_extractor module}
                    (已 .to(device)，且支援 extract_features(img, xai_mode=True))
        heads:      dict {stream_name: LinearHead module}
                    (EXP-A 訓出的單流 head，用來提供 backward target)
        device:     'cuda' or 'cpu'

    Example:
        explainer = PerStreamExplainer(extractors, heads, device='cuda')
        exps = explainer.explain_image(img_tensor)   # img_tensor: (1, 3, 224, 224)
        fig = explainer.visualize(img_np, exps, save_path='out.png')
    """

    def __init__(
        self,
        extractors: Dict[str, nn.Module],
        heads: Dict[str, nn.Module],
        device: str = 'cuda',
    ):
        self.extractors = extractors
        self.heads = heads
        self.device = device

    # ── 核心：對一張圖跑 5 條流 ─────────────────────────────────────────
    def explain_image(self, img_tensor: torch.Tensor) -> Dict[str, np.ndarray]:
        """
        Args:
            img_tensor: (1, 3, 224, 224) ImageNet-normalized，已 .to(device)
        Returns:
            dict {stream_name: heatmap (H, W) in [0,1]}
            某條流如果失敗會被 silently 跳過 + print warning，不會 crash
        """
        result: Dict[str, np.ndarray] = {}

        # ─── 1. CNN 四條流：真正的 Grad-CAM ───
        for s in CNN_STREAMS:
            if s not in self.extractors or s not in self.heads:
                print(f"[PerStreamExplainer] {s} missing extractor or head, skip")
                continue
            try:
                cam = self._gradcam_one_stream(s, img_tensor)
                if cam is not None:
                    result[s] = cam
            except Exception as e:
                print(f"[PerStreamExplainer] {s} Grad-CAM failed: {e}")

        # ─── 2. CLIP：attention rollout（Day 2-3 換 Chefer）───
        if 'clip' in self.extractors:
            try:
                clip_map = self._clip_rollout(img_tensor)
                if clip_map is not None:
                    result['clip'] = clip_map
            except Exception as e:
                print(f"[PerStreamExplainer] CLIP rollout failed: {e}")

        return result

    # ── 單流 Grad-CAM 內部實作 ──────────────────────────────────────────
    def _gradcam_one_stream(
        self,
        stream: str,
        img_tensor: torch.Tensor,
    ) -> Optional[np.ndarray]:
        ext = self.extractors[stream]
        head = self.heads[stream]
        target = find_target_layer(ext, stream)
        if target is None:
            print(f"[PerStreamExplainer] {stream} target layer not found, skip")
            return None

        # 確保梯度可以流通
        for p in ext.parameters():
            p.requires_grad_(True)
        for p in head.parameters():
            p.requires_grad_(True)
        ext.eval()
        head.eval()

        hook = GradCAMHook(target)
        try:
            # ⚠️ xai_mode=True 必傳，否則 extractor 內部 no_grad 切斷梯度
            feat = ext.extract_features(img_tensor, xai_mode=True)
            logits = head(feat)  # (1, 2)

            # backward target = fake logit (class index 1)
            ext.zero_grad()
            head.zero_grad()
            fake_logit = logits[0, 1]
            fake_logit.backward()

            cam = hook.compute()
        finally:
            hook.remove()

        return cam

    # ── CLIP attention rollout（Day 1 暫用，Day 2-3 換 Chefer）─────────
    def _clip_rollout(self, img_tensor: torch.Tensor) -> Optional[np.ndarray]:
        clip_ext = self.extractors['clip']
        if not hasattr(clip_ext, 'get_attention_rollout'):
            print("[PerStreamExplainer] CLIP extractor 沒有 get_attention_rollout()")
            return None

        with torch.no_grad():
            rollout = clip_ext.get_attention_rollout(img_tensor)
        # rollout 預期 shape (1, N_patches)
        r = rollout[0].detach().cpu().numpy()
        n = r.shape[0]
        grid = int(np.sqrt(n))
        if grid * grid != n:
            print(f"[PerStreamExplainer] CLIP rollout 不是方形 patch grid (n={n})")
            return None
        r_map = r.reshape(grid, grid)
        r_map = (r_map - r_map.min()) / (r_map.max() - r_map.min() + 1e-8)
        return r_map

    # ── 視覺化：1×6 對照圖 ──────────────────────────────────────────────
    def visualize(
        self,
        img_np: np.ndarray,
        explanations: Dict[str, np.ndarray],
        save_path: Optional[str] = None,
        title: Optional[str] = None,
    ) -> plt.Figure:
        """
        畫 1×6 對照圖：原圖 | CLIP | FFT | DCT | DIRE | Noise。

        Args:
            img_np:       原圖 (H, W, 3) uint8
            explanations: explain_image() 的回傳
            save_path:    指定就存檔
            title:        super-title（例如 'Verdict: AI (87%)'）
        Returns:
            matplotlib Figure
        """
        fig, axes = plt.subplots(1, 6, figsize=(18, 3.5))
        h, w = img_np.shape[:2]

        # Panel 0: 原圖
        axes[0].imshow(img_np)
        axes[0].set_title('Original', fontsize=11, fontweight='bold')
        axes[0].axis('off')

        # Panel 1-5: 5 條流
        for idx, s in enumerate(STREAMS):
            ax = axes[idx + 1]
            if s not in explanations:
                ax.imshow(np.ones((h, w, 3)) * 0.85)
                ax.set_title(f'{STREAM_DISPLAY[s]}\n(unavailable)',
                             fontsize=10, color='gray')
                ax.axis('off')
                continue

            m = cv2.resize(explanations[s].astype(np.float32), (w, h))
            heat = (m * 255).astype(np.uint8)
            colored = cv2.applyColorMap(heat, cv2.COLORMAP_JET)
            colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
            overlay = (0.5 * colored + 0.5 * img_np).astype(np.uint8)
            ax.imshow(overlay)

            # FFT/DCT 是頻域，caption 標清楚
            suffix = ' (freq domain)' if s in ('fft', 'dct') else ''
            ax.set_title(f'{STREAM_DISPLAY[s]}{suffix}', fontsize=10)
            ax.axis('off')

        if title:
            fig.suptitle(title, fontsize=12, fontweight='bold')
        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"[PerStreamExplainer] Saved to {save_path}")

        return fig


# ──────────────────────────────────────────────────────────────────────
# 4. Quick self-test（python -m src.xai.per_stream_gradcam）
# ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("This module is meant to be imported, not run directly.")
    print("To test, run: python day1_test.py <image_path>")
