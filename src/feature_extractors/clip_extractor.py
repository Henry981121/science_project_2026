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
        self.clip_model = CLIPModel.from_pretrained(self.MODEL_NAME)
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
        return [Image.fromarray((img.permute(1,2,0).cpu().numpy()*255).astype("uint8")) for img in imgs_01]

    def _preprocess(self, images):
        inputs = self.processor(images=self._to_pil(images), return_tensors="pt", padding=True)
        return {k: v.to(self.device) for k, v in inputs.items()}

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        inputs = self._preprocess(images)
        with torch.no_grad():
            raw = self.clip_model.vision_model(**inputs).pooler_output  # (B,1024)
        return self.proj(raw)  # (B,512)

    def extract_features(self, images: torch.Tensor) -> torch.Tensor:
        self.eval()
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
