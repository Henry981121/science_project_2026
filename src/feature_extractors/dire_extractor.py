"""
DIRE (Diffusion Reconstruction Error) Feature Extractor - v2
Uses Stable Diffusion VAE for reconstruction instead of full DDIM.
Much faster and more reliable than the original DDIM-based approach.

Reference: Wang et al., DIRE for Diffusion-Generated Image Detection, ICCV 2023
https://arxiv.org/abs/2303.09295
"""

import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F


class DIREFeatureExtractor(nn.Module):
    """
    DIRE using VAE reconstruction error.

    AI-generated images from diffusion models can be reconstructed
    with low error by a VAE, while real photos cannot.

    Uses stabilityai/sd-vae-ft-mse (320MB, still available).
    Feature dim: 512
    """

    def __init__(self, device: str = 'cuda', feature_dim: int = 512):
        super().__init__()
        self.device = device
        self.feature_dim = feature_dim
        self._vae = None
        self._vae_loaded = False

        backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        self.feature_net = nn.Sequential(*list(backbone.children())[:-1])
        self.projection = nn.Linear(2048, feature_dim)

        self.to(device)
        self.eval()

    def to(self, *args, **kwargs):
        result = super().to(*args, **kwargs)
        # 同步更新 self.device
        try:
            self.device = next(self.parameters()).device
        except StopIteration:
            pass
        # 如果 VAE 已載入，一起搬移
        if self._vae is not None:
            self._vae = self._vae.to(self.device)
        return result

    def _load_vae(self):
        if self._vae_loaded:
            return self._vae is not None
        try:
            from diffusers import AutoencoderKL
            dtype = torch.float16 if 'cuda' in str(self.device) else torch.float32
            self._vae = AutoencoderKL.from_pretrained(
                "stabilityai/sd-vae-ft-mse", torch_dtype=dtype
            )
            self._vae = self._vae.to(self.device)
            self._vae.eval()
            self._vae_loaded = True
            print("[DIRE] VAE loaded successfully")
            return True
        except Exception as e:
            print(f"[DIRE] VAE load failed: {e}. Using fallback.")
            self._vae_loaded = True
            return False

    def _reconstruct_with_vae(self, images: torch.Tensor) -> torch.Tensor:
        mean = torch.tensor([0.485, 0.456, 0.406], device=images.device).view(1,3,1,1)
        std  = torch.tensor([0.229, 0.224, 0.225], device=images.device).view(1,3,1,1)
        images_01  = images * std + mean
        vae_dtype = next(self._vae.parameters()).dtype
        images_vae = (images_01 * 2 - 1).to(dtype=vae_dtype, device=images.device)
        with torch.no_grad():
            latents = self._vae.encode(images_vae).latent_dist.sample()
            recon   = self._vae.decode(latents).sample.float()
        recon_01   = (recon + 1) / 2
        recon_norm = (recon_01 - mean.float()) / std.float()
        return recon_norm

    def _compute_dire_map(self, images: torch.Tensor) -> torch.Tensor:
        vae_ok = self._load_vae()
        if vae_ok and self._vae is not None:
            reconstructed = self._reconstruct_with_vae(images)
        else:
            reconstructed = F.avg_pool2d(images, kernel_size=5, stride=1, padding=2)
        error = torch.abs(images - reconstructed)
        error = torch.log(error + 1e-6)
        error = (error - error.mean()) / (error.std() + 1e-6)
        return error

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.extract_features(images)

    def extract_features(self, images: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            error_map = self._compute_dire_map(images)
            feat = self.feature_net(error_map).flatten(1)
            feat = self.projection(feat)
        return feat

    def get_dire_visualization(self, images: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self._compute_dire_map(images)
