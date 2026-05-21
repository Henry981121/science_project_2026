"""
Stream A: CLIP ViT-L/14 語義特徵提取器 (v2.1)
完全凍結，1024d → projection → 512d
"""

import torch
import torch.nn as nn
from transformers import CLIPProcessor, CLIPModel
from PIL import Image


class CLIPFeatureExtractor(nn.Module):
    MODEL_NAME = "openai/clip-vit-large-patch14"
    RAW_DIM = 1024
    OUT_DIM = 512

    def __init__(self, device: str = "cuda"):
        super().__init__()
        self.device = device
        print(f"[CLIP] Loading {self.MODEL_NAME} ...")
        # attn_implementation="eager" 為必要：現代 transformers 預設 SDPA，
        # 在 SDPA 下 output_attentions=True 會靜默回傳 None，Chefer relevance
        # 與 attention rollout 都會整個失效。
        self.clip_model = CLIPModel.from_pretrained(
            self.MODEL_NAME, attn_implementation="eager"
        )
        self.processor  = CLIPProcessor.from_pretrained(self.MODEL_NAME)
        for param in self.clip_model.parameters():
            param.requires_grad = False
        self.clip_model = self.clip_model.to(device).eval()
        self.proj = nn.Linear(self.RAW_DIM, self.OUT_DIM).to(device)
        print(f"[CLIP] Ready | frozen | {self.RAW_DIM}->512d")

    def to(self, *args, **kwargs):
        result = super().to(*args, **kwargs)
        try:
            self.device = next(self.parameters()).device
        except StopIteration:
            pass
        if hasattr(self, 'clip_model'):
            self.clip_model = self.clip_model.to(self.device)
        return result

    def _to_pil(self, images: torch.Tensor):
        mean = torch.tensor([0.485,0.456,0.406], device=images.device).view(1,3,1,1)
        std  = torch.tensor([0.229,0.224,0.225], device=images.device).view(1,3,1,1)
        imgs_01 = torch.clamp(images * std + mean, 0, 1)
        return [Image.fromarray((img.detach().permute(1,2,0).cpu().numpy()*255).astype("uint8")) for img in imgs_01]

    def _preprocess(self, images):
        inputs = self.processor(images=self._to_pil(images), return_tensors="pt", padding=True)
        return {k: v.to(self.device) for k, v in inputs.items()}

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        inputs = self._preprocess(images)
        with torch.no_grad():
            raw = self.clip_model.vision_model(**inputs).pooler_output  # (B,1024)
        return self.proj(raw)  # (B,512)

    def _forward_xai(self, images: torch.Tensor, output_attentions: bool = False):
        """
        XAI 專用 forward。**不包 no_grad**，允許 backward 到 attention map。
        clip_model 本身仍 frozen（params requires_grad=False），但中間 tensor 仍可取梯度。

        Args:
            images:            (B, 3, H, W)
            output_attentions: True 時額外回傳 attention list（給 Chefer relevance 用）

        Returns:
            feats (B, 512)                                      若 output_attentions=False
            (feats, attentions: tuple of (B, H, S, S) per layer) 若 output_attentions=True
        """
        inputs = self._preprocess(images)
        outputs = self.clip_model.vision_model(**inputs, output_attentions=output_attentions)
        feats = self.proj(outputs.pooler_output)  # (B, 512)
        if output_attentions:
            return feats, outputs.attentions
        return feats

    def extract_features(self, images: torch.Tensor, xai_mode: bool = False) -> torch.Tensor:
        """
        xai_mode=False (default): 行為與原本完全相同 — eval + no_grad
        xai_mode=True           : 解 no_grad，允許梯度流（Chefer relevance 必要條件）

        注意：CLIP backbone 仍 frozen，xai_mode=True 只解 no_grad 包裝，
        不會讓 CLIP 參數開始訓練。
        """
        self.eval()
        if xai_mode:
            return self._forward_xai(images, output_attentions=False)
        with torch.no_grad():
            return self.forward(images)

    def get_attention_rollout(self, images: torch.Tensor) -> torch.Tensor:
        """Attention Rollout for XAI. Returns (B, num_patches) CLS attention."""
        inputs = self._preprocess(images)
        with torch.no_grad():
            outputs = self.clip_model.vision_model(**inputs, output_attentions=True)
        result = torch.eye(outputs.attentions[0].shape[-1], device=self.device).unsqueeze(0)
        for attn in outputs.attentions:
            attn_avg = attn.mean(dim=1)  # (B, seq, seq)
            attn_avg = attn_avg + torch.eye(attn_avg.shape[-1], device=self.device).unsqueeze(0)
            attn_avg = attn_avg / attn_avg.sum(dim=-1, keepdim=True)
            result = torch.bmm(result.expand(attn_avg.shape[0],-1,-1), attn_avg)
        return result[:, 0, 1:]  # (B, num_patches)


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ext = CLIPFeatureExtractor(device=device)
    imgs = torch.randn(2, 3, 224, 224).to(device)
    feats = ext.extract_features(imgs)
    print(f"Features: {feats.shape}")  # (2, 512)
    assert feats.shape == (2, 512)
    print("CLIP OK")
