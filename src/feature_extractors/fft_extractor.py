"""
Stream B: FFT 頻域特徵提取器 (v2.1)
重要：FFT 在 resize 之前執行，保留原始高頻訊號。
ResNet34 backbone → 512d
"""

import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np


class FFTFeatureExtractor(nn.Module):
    OUT_DIM = 512

    def __init__(self, device: str = "cuda"):
        super().__init__()
        self.device = device
        resnet = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])  # -> (B,512,1,1)
        self.to(device)
        self.eval()
        print(f"[FFT] Ready | ResNet34 | 512d | analyze-before-resize=True")

    def to(self, *args, **kwargs):
        result = super().to(*args, **kwargs)
        try:
            self.device = next(self.parameters()).device
        except StopIteration:
            pass
        return result

    def _compute_fft(self, images: torch.Tensor) -> torch.Tensor:
        """
        Compute log-magnitude FFT spectrum.
        Operates on whatever resolution is passed in (caller should NOT pre-resize).
        images: (B, C, H, W)
        returns: (B, C, H, W) normalized to [0,1]
        """
        B, C, H, W = images.shape
        spectra = []
        for b in range(B):
            channels = []
            for c in range(C):
                ch = images[b, c].cpu().numpy()
                fft = np.fft.fft2(ch)
                fft_shifted = np.fft.fftshift(fft)
                mag = np.log1p(np.abs(fft_shifted))
                channels.append(mag)
            spectra.append(np.stack(channels, axis=0))
        spec = torch.from_numpy(np.stack(spectra, axis=0)).float().to(self.device)
        # per-batch normalization
        mn = spec.flatten(1).min(1)[0].view(B,1,1,1)
        mx = spec.flatten(1).max(1)[0].view(B,1,1,1)
        spec = (spec - mn) / (mx - mn + 1e-8)
        return spec

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        NOTE: Pass original-resolution images; do NOT resize before calling.
        images: (B, 3, H, W)  -> features: (B, 512)
        """
        spec = self._compute_fft(images)
        # ResNet34 expects 224x224; resize the spectrum (not the original image)
        import torch.nn.functional as F
        spec = F.interpolate(spec, size=(224, 224), mode="bilinear", align_corners=False)
        feat = self.backbone(spec).flatten(1)  # (B, 512)
        return feat

    def extract_features(self, images: torch.Tensor) -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            return self.forward(images)

    def get_spectrum(self, images: torch.Tensor) -> torch.Tensor:
        """Return FFT spectrum for visualization."""
        with torch.no_grad():
            return self._compute_fft(images)


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ext = FFTFeatureExtractor(device=device)
    imgs = torch.randn(2, 3, 512, 512).to(device)  # high-res input
    feats = ext.extract_features(imgs)
    print(f"Features: {feats.shape}")  # (2, 512)
    assert feats.shape == (2, 512)
    print("FFT OK")
