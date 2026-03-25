"""
Stream E: SRM Noise 特徵提取器 (v2.1)
使用 SRM (Steganalysis Rich Model) 30-filter high-pass bank
自訂輕量 CNN → 512d
重要：在 resize 之前執行 SRM 濾波
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# SRM 3x3 filter kernels (30 filters, key subset from Fridrich & Kodovsky 2012)
_SRM_KERNELS_3x3 = [
    # Edge detectors
    [[0, 0, 0], [0,-1, 1], [0, 0, 0]],
    [[0, 0, 0], [0,-1, 0], [0, 1, 0]],
    [[0, 0, 0], [1,-1, 0], [0, 0, 0]],
    [[0, 1, 0], [0,-1, 0], [0, 0, 0]],
    # 2nd order
    [[0, 0, 0], [1,-2, 1], [0, 0, 0]],
    [[0, 1, 0], [0,-2, 0], [0, 1, 0]],
    [[1, 0, 0], [0,-2, 0], [0, 0, 1]],
    [[0, 0, 1], [0,-2, 0], [1, 0, 0]],
    # 3rd order
    [[0, 0, 0], [1,-3, 3], [0,-1, 0]],
    [[0, 1, 0], [0,-3, 0], [0, 3, 0]],
    # Laplacian variants
    [[0,-1, 0], [-1, 4,-1], [0,-1, 0]],
    [[-1,-1,-1], [-1, 8,-1], [-1,-1,-1]],
    # Diagonal
    [[-1, 0, 0], [0, 2, 0], [0, 0,-1]],
    [[0, 0,-1], [0, 2, 0], [-1, 0, 0]],
    # More high-pass
    [[1,-2, 1], [-2, 4,-2], [1,-2, 1]],
    [[-1, 2,-1], [2,-4, 2], [-1, 2,-1]],
    [[0, 0, 0], [-1, 2,-1], [0, 0, 0]],
    [[0,-1, 0], [0, 2, 0], [0,-1, 0]],
    [[1,-1, 0], [-1, 1, 0], [0, 0, 0]],
    [[0,-1, 1], [0, 1,-1], [0, 0, 0]],
    [[0, 0, 0], [0, 1,-1], [0,-1, 1]],
    [[0, 0, 0], [-1, 1, 0], [1,-1, 0]],
    [[-1, 0, 1], [0, 0, 0], [1, 0,-1]],
    [[1, 0,-1], [0, 0, 0], [-1, 0, 1]],
    [[0, 1,-1], [0,-1, 1], [0, 0, 0]],
    [[1,-1, 0], [-1, 1, 0], [0, 0, 0]],
    [[0, 0, 1], [0,-2, 0], [1, 0, 0]],
    [[1, 0, 0], [0,-2, 0], [0, 0, 1]],
    [[0,-1, 0], [1, 0,-1], [0, 1, 0]],
    [[0, 1, 0], [-1, 0, 1], [0,-1, 0]],
]


class NoisePrintExtractor(nn.Module):
    """
    SRM-based noise feature extractor.
    30 fixed high-pass filters -> lightweight CNN -> 512d
    """
    OUT_DIM = 512

    def __init__(self, device: str = "cuda"):
        super().__init__()
        self.device = device

        # Build fixed SRM filter bank: 30 filters, applied per channel
        kernels = np.array(_SRM_KERNELS_3x3, dtype=np.float32)  # (30, 3, 3)
        # Normalize each kernel
        for i in range(len(kernels)):
            s = np.abs(kernels[i]).sum()
            if s > 0:
                kernels[i] = kernels[i] / s
        # Shape: (30, 1, 3, 3) — depthwise per channel; applied to grayscale
        srm_weight = torch.from_numpy(kernels).unsqueeze(1)  # (30, 1, 3, 3)
        self.register_buffer("srm_weight", srm_weight)

        # Lightweight CNN for noise features
        self.cnn = nn.Sequential(
            nn.Conv2d(30, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(256, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.fc = nn.Linear(512, self.OUT_DIM)

        self.to(device)
        self.eval()
        print(f"[Noise] Ready | SRM 30-filter bank | 512d")

    def to(self, *args, **kwargs):
        result = super().to(*args, **kwargs)
        try:
            self.device = next(self.parameters()).device
        except StopIteration:
            pass
        return result

    def _apply_srm(self, images: torch.Tensor) -> torch.Tensor:
        """
        Apply SRM filters to grayscale image.
        images: (B, 3, H, W)
        returns: (B, 30, H, W)
        """
        gray = images.mean(dim=1, keepdim=True)  # (B, 1, H, W)
        noise = F.conv2d(gray, self.srm_weight, padding=1)  # (B, 30, H, W)
        return noise

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        NOTE: Pass original-resolution images for best results.
        images: (B, 3, H, W) -> features: (B, 512)
        """
        noise_map = self._apply_srm(images)  # (B, 30, H, W)
        # Resize if needed for CNN efficiency
        if noise_map.shape[-1] > 224:
            noise_map = F.interpolate(noise_map, size=(224, 224), mode="bilinear", align_corners=False)
        feat = self.cnn(noise_map).flatten(1)  # (B, 512)
        feat = self.fc(feat)                   # (B, 512)
        return feat

    def extract_features(self, images: torch.Tensor) -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            return self.forward(images)

    def get_noise_map(self, images: torch.Tensor) -> torch.Tensor:
        """Return SRM noise map for visualization (B, 30, H, W)."""
        with torch.no_grad():
            return self._apply_srm(images)


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ext = NoisePrintExtractor(device=device)
    imgs = torch.randn(2, 3, 512, 512).to(device)
    feats = ext.extract_features(imgs)
    print(f"Features: {feats.shape}")  # (2, 512)
    assert feats.shape == (2, 512)
    print("Noise OK")
