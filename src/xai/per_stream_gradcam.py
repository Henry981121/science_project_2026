"""
per_stream_gradcam.py
======================

取代 src/xai/gradcam.py 裡那個沒在算梯度的 MultiStreamExplainer。

【做什麼】
對一張測試圖，產生 5 條流各自的空間視覺化：
  - FFT/DCT/DIRE (CNN)        : 真正的 Grad-CAM (Selvaraju 2017)，
                                target 為各 backbone 的最後 conv block，
                                backward target 為該流自己的 EXP-A head 的 fake logit。
  - CLIP (ViT)                : Chefer relevance propagation (Chefer 2021 CVPR)，
                                對 vision_model 每層 transformer block 抓 attention
                                + gradient，套 layer-wise relevance propagation 公式
                                逐層累積（class-specific 對 fake logit）。
  - Noise                     : **不是 Grad-CAM**。SRM 雜訊殘差是全域統計紋理
                                特徵，沒有可空間定位的決策證據，套 Grad-CAM 實測
                                5/5 圖皆退化成空圖。改為視覺化該流的輸入表徵
                                （30 個 SRM 高通濾波器的逐像素能量）。這是輸入域
                                變換、model-agnostic，caption 一律標「SRM residual」。

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
from typing import Dict, Optional


STREAMS = ['clip', 'fft', 'dct', 'dire', 'noise']
STREAM_DISPLAY = {
    'clip':  'CLIP',
    'fft':   'FFT',
    'dct':   'DCT',
    'dire':  'DIRE',
    'noise': 'Noise',
}
# 走真 Grad-CAM 的流。Noise 不在此列 —— 見 PerStreamExplainer._noise_residual_map。
CNN_STREAMS = ['fft', 'dct', 'dire']


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
    Noise 流不走 Grad-CAM（見 PerStreamExplainer._noise_residual_map），不在此處理。
    """
    if name == 'fft':
        children = list(ext.backbone.children())
    elif name in ('dct', 'dire'):
        children = list(ext.feature_net.children())
    else:
        return None

    # 從後往前找最後一個 Sequential block（ResNet layer4 那種）
    for c in reversed(children):
        if isinstance(c, nn.Sequential) and len(list(c.children())) > 0:
            return c[-1]
    return children[-2] if len(children) > 1 else None


# ──────────────────────────────────────────────────────────────────────
# 3. Chefer CLIP relevance (Chefer 2021 CVPR)
# ──────────────────────────────────────────────────────────────────────
#
# Chefer "Transformer Interpretability Beyond Attention Visualization"
# eq. (13)-(14) class-specific relevance for ViT:
#
#   for each transformer block layer l:
#       A_l    = attention weights         shape (B, H, S, S)
#       g_l    = d(y_target) / d A_l       shape (B, H, S, S)
#       cam_l  = ( g_l ⊙ A_l ).clamp(min=0).mean(dim=H)   # (B, S, S)
#       R̄_l    = I + cam_l                                # row-normalize
#       R      = R̄_l @ R                                  # accumulate
#
#   final relevance = R[:, 0, 1:].reshape(grid, grid)     # CLS row → patches
#
# 跟 attention rollout (Abnar 2020) 的差別：rollout 是 class-agnostic
# 直接平均 attention；Chefer 用 gradient 對特定類別加權，且只取正貢獻。
# ──────────────────────────────────────────────────────────────────────

class CheferCLIPRelevance:
    """
    對 CLIP ViT 跑 class-specific relevance propagation。

    用法：
        chefer = CheferCLIPRelevance(clip_ext, clip_head, device='cuda')
        relevance_map = chefer.compute(img_tensor)   # (grid, grid) in [0, 1]
    """

    def __init__(self, clip_ext: nn.Module, clip_head: nn.Module, device: str = 'cuda'):
        self.clip_ext = clip_ext
        self.clip_head = clip_head
        self.device = device

        # 早期結構檢查：確認 vision_model 是預期的 encoder.layers 結構，
        # 不符就在 PerStreamExplainer init 階段直接 fail（fail fast）。
        vision_model = clip_ext.clip_model.vision_model
        if not hasattr(vision_model, 'encoder') or not hasattr(vision_model.encoder, 'layers'):
            raise RuntimeError(
                "CheferCLIPRelevance 預期 clip_ext.clip_model.vision_model.encoder.layers "
                f"但實際結構是 {type(vision_model).__name__}"
            )

    # ── 主流程 ──────────────────────────────────────────────────────
    def compute(self, img_tensor: torch.Tensor) -> Optional[np.ndarray]:
        """
        Args:
            img_tensor: (1, 3, 224, 224) ImageNet-normalized，已 .to(device)
        Returns:
            relevance heatmap (grid, grid) in [0, 1]，失敗回 None
        """
        # CLIP backbone frozen，但我們要對 attention 中間 tensor 取梯度，
        # 所以 head 必須 requires_grad（提供 backward 起點）
        for p in self.clip_head.parameters():
            p.requires_grad_(True)
        self.clip_ext.eval()
        self.clip_head.eval()

        # 直接用 _forward_xai 回傳的官方 outputs.attentions（每層一個
        # (B, H, S, S) tensor）。不再對 self_attn 掛 forward hook —— 後者
        # 依賴 CLIPAttention 的回傳結構，transformers 一升級就可能壞。
        # _forward_xai 已開 output_attentions=True 並讓 pixel_values 帶梯度。
        feats, attentions = self.clip_ext._forward_xai(img_tensor, output_attentions=True)
        if not attentions:
            print("[CheferCLIPRelevance] _forward_xai 沒回傳 attention —— "
                  "確認 CLIP 是以 attn_implementation='eager' 載入。")
            return None

        # 中間 tensor 預設不存 grad，backward 前要逐層 retain_grad
        for attn in attentions:
            if attn.requires_grad:
                attn.retain_grad()

        logits = self.clip_head(feats)
        fake_logit = logits[0, 1]
        self.clip_head.zero_grad()
        # CLIP backbone 凍結沒梯度，但 attention 中間 tensor 仍有 grad
        fake_logit.backward()

        # ── relevance propagation ──
        B, _, S, _ = attentions[0].shape
        R = torch.eye(S, device=self.device).unsqueeze(0).expand(B, -1, -1).clone()

        n_used = 0
        for attn in attentions:
            if attn.grad is None:
                continue
            n_used += 1
            # attn / grad shape: (B, H, S, S)
            cam_layer = (attn.grad * attn).clamp(min=0).mean(dim=1)  # (B, S, S)
            eye = torch.eye(S, device=self.device).unsqueeze(0).expand(B, -1, -1)
            R_bar = eye + cam_layer
            R_bar = R_bar / R_bar.sum(dim=-1, keepdim=True).clamp(min=1e-8)
            R = torch.bmm(R_bar, R)

        # 所有層 grad 都是 None → 梯度沒回流到 attention，R 還是單位矩陣，
        # 此時硬算只會得到一張全 0 的假 heatmap。明確失敗比回傳空圖好。
        if n_used == 0:
            print("[CheferCLIPRelevance] 所有 attention 層的 grad 全為 None — "
                  "梯度沒回流到 attention map。檢查 _forward_xai 是否讓 "
                  "pixel_values.requires_grad_(True)。")
            return None

        # CLS row → patches（跳過 CLS 自己）
        cls_row = R[0, 0, 1:].detach().cpu().numpy()
        n = cls_row.shape[0]
        grid = int(np.sqrt(n))
        if grid * grid != n:
            print(f"[CheferCLIPRelevance] patch 數 {n} 不是完全平方")
            return None
        m = cls_row.reshape(grid, grid)
        return (m - m.min()) / (m.max() - m.min() + 1e-8)


# ──────────────────────────────────────────────────────────────────────
# 4. Per-stream explainer 主類
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

        # Chefer CLIP relevance — 只在 CLIP extractor + head 都齊備時建
        self._chefer: Optional[CheferCLIPRelevance] = None
        if 'clip' in extractors and 'clip' in heads:
            try:
                self._chefer = CheferCLIPRelevance(
                    extractors['clip'], heads['clip'], device=device,
                )
            except Exception as e:
                print(f"[PerStreamExplainer] CheferCLIPRelevance init 失敗: {e}")

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

        # ─── 1. CNN 三條流（FFT/DCT/DIRE）：真正的 Grad-CAM ───
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

        # ─── 2. CLIP：Chefer relevance propagation ───
        if self._chefer is not None:
            try:
                clip_map = self._chefer.compute(img_tensor)
                if clip_map is not None:
                    result['clip'] = clip_map
            except Exception as e:
                print(f"[PerStreamExplainer] CLIP Chefer relevance failed: {e}")

        # ─── 3. Noise：SRM 殘差圖（非 Grad-CAM，見 _noise_residual_map）───
        if 'noise' in self.extractors:
            try:
                noise_map = self._noise_residual_map(img_tensor)
                if noise_map is not None:
                    result['noise'] = noise_map
            except Exception as e:
                print(f"[PerStreamExplainer] noise SRM residual failed: {e}")

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

    # ── Noise 流：SRM 殘差圖（非 Grad-CAM）─────────────────────────────
    def _noise_residual_map(self, img_tensor: torch.Tensor) -> Optional[np.ndarray]:
        """
        Noise 流的面板 —— **不是 Grad-CAM，也不是模型歸因**。

        SRM 雜訊殘差是全域統計紋理特徵，沒有可空間定位的「模型決策證據」。
        對它套 Grad-CAM 會退化成空圖（實測 5/5 圖：mean≈0.001，SDv5/MJ 連
        max 都為 0）。改為直接視覺化 Noise 流的輸入表徵：30 個 SRM 高通
        濾波器的逐像素能量。

        ⚠️ 這是輸入域變換、model-agnostic。caption / 論文必須明確標為
        「SRM residual」，不可當成 attribution —— 否則就是另一個假 Grad-CAM。
        Noise 流的決策層級重要性由 Level 2（attention + ablation）承擔。
        """
        ext = self.extractors['noise']
        if not hasattr(ext, 'get_noise_map'):
            print("[PerStreamExplainer] noise extractor 無 get_noise_map()，skip")
            return None
        with torch.no_grad():
            noise_map = ext.get_noise_map(img_tensor)   # (B, 30, H, W)
        # 跨 30 個高通濾波器取逐像素能量
        energy = noise_map.abs().mean(dim=1)[0]          # (H, W)
        energy = energy.detach().cpu().numpy().astype(np.float32)
        return (energy - energy.min()) / (energy.max() - energy.min() + 1e-8)

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

            # 各流面板方法不同，caption 標清楚：
            #   FFT/DCT — 頻域 Grad-CAM；Noise — SRM 殘差（非歸因）
            if s in ('fft', 'dct'):
                suffix = ' (freq domain)'
            elif s == 'noise':
                suffix = ' (SRM residual)'
            else:
                suffix = ''
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
