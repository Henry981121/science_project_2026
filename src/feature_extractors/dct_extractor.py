"""
DCT (Discrete Cosine Transform) Feature Extractor
Analyzes JPEG block structure and frequency coefficient patterns.
AI-generated images have statistically different DCT distributions.
"""

import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np
from typing import Optional
import torch.nn.functional as F


class DCTFeatureExtractor(nn.Module):
    """
    DCT-based feature extractor for AI image detection.

    AI-generated images lack natural JPEG block compression artifacts
    and show distinct DCT coefficient distributions.

    Feature dim: 512
    """

    def __init__(self, device: str = 'cuda', feature_dim: int = 512):
        super().__init__()
        self.device = device
        self.feature_dim = feature_dim

        # ResNet18 backbone (lightweight)
        backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.feature_net = nn.Sequential(*list(backbone.children())[:-1])  # remove FC
        self.projection = nn.Linear(512, feature_dim)

        self.to(device)
        self.eval()
        print(f"[DCT] Ready | ResNet18 | 8x8 block DCT | {feature_dim}d")

    def to(self, *args, **kwargs):
        result = super().to(*args, **kwargs)
        try:
            self.device = next(self.parameters()).device
        except StopIteration:
            pass
        return result

    def _compute_dct_map(self, images: torch.Tensor) -> torch.Tensor:
        """
        Compute DCT coefficient map from images.
        images: (B, 3, H, W) normalized [-1,1] or [0,1]
        returns: (B, 3, H, W) DCT magnitude map
        """
        B, C, H, W = images.shape

        # Convert to grayscale for DCT analysis (average channels)
        gray = images.mean(dim=1, keepdim=True)  # (B, 1, H, W)

        # Unfold into 8x8 blocks (standard JPEG block size)
        # Pad if needed
        pad_h = (8 - H % 8) % 8
        pad_w = (8 - W % 8) % 8
        if pad_h > 0 or pad_w > 0:
            gray = F.pad(gray, (0, pad_w, 0, pad_h), mode='reflect')

        _, _, H_pad, W_pad = gray.shape

        # Extract 8x8 blocks
        blocks = gray.unfold(2, 8, 8).unfold(3, 8, 8)
        # blocks: (B, 1, H/8, W/8, 8, 8)
        bh, bw = blocks.shape[2], blocks.shape[3]
        blocks = blocks.contiguous().view(B, bh * bw, 8, 8)

        # Apply 2D DCT using matrix multiplication
        dct_matrix = self._get_dct_matrix(8).to(images.device)
        # DCT: D = M * block * M^T
        dct_blocks = torch.matmul(
            torch.matmul(dct_matrix, blocks),
            dct_matrix.T
        )  # (B, num_blocks, 8, 8)

        # Log magnitude (better contrast)
        dct_mag = torch.log(torch.abs(dct_blocks) + 1e-6)

        # Reshape back to spatial map
        dct_map = dct_mag.view(B, 1, bh, bw, 8, 8)
        dct_map = dct_map.permute(0, 1, 2, 4, 3, 5).contiguous()
        dct_map = dct_map.view(B, 1, bh * 8, bw * 8)

        # Crop back to original size
        dct_map = dct_map[:, :, :H, :W]

        # Repeat to 3 channels for ResNet
        dct_map = dct_map.expand(-1, 3, -1, -1)

        # Normalize
        dct_map = (dct_map - dct_map.mean()) / (dct_map.std() + 1e-6)

        return dct_map

    def _get_dct_matrix(self, n: int) -> torch.Tensor:
        """Generate n×n DCT-II transform matrix."""
        matrix = torch.zeros(n, n)
        for k in range(n):
            for i in range(n):
                if k == 0:
                    matrix[k, i] = np.sqrt(1.0 / n)
                else:
                    matrix[k, i] = np.sqrt(2.0 / n) * np.cos(
                        np.pi * k * (2 * i + 1) / (2 * n)
                    )
        return matrix

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.extract_features(images)

    def extract_features(self, images: torch.Tensor) -> torch.Tensor:
        """
        Extract DCT features.

        Args:
            images: (B, 3, H, W) tensor, normalized [0,1] or standard ImageNet norm

        Returns:
            features: (B, feature_dim)
        """
        with torch.no_grad():
            dct_map = self._compute_dct_map(images)
            feat = self.feature_net(dct_map)  # (B, 512, 1, 1)
            feat = feat.flatten(1)             # (B, 512)
            feat = self.projection(feat)       # (B, feature_dim)
        return feat

    def get_dct_visualization(self, images: torch.Tensor) -> torch.Tensor:
        """Return DCT map for visualization."""
        with torch.no_grad():
            dct_map = self._compute_dct_map(images)
        return dct_map
