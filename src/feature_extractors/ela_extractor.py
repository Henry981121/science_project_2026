"""
Stream B: ELA (Error Level Analysis) 特徵提取器
檢測 JPEG 壓縮不一致性
"""

import torch
import torch.nn as nn
import torchvision.models as models
import cv2
import numpy as np
from PIL import Image
import io
from typing import Optional

class ELAFeatureExtractor(nn.Module):
    """
    ELA (Error Level Analysis) 特徵提取器
    
    流程：
    1. 輸入圖像 → JPEG 壓縮 (quality=90)
    2. 計算壓縮前後差異（ELA map）
    3. 使用 ResNet50 提取 ELA 特徵
    
    輸出維度：2048
    """
    
    def __init__(
        self,
        feature_dim: int = 2048,
        jpeg_quality: int = 90,
        backbone: str = "resnet50",
        pretrained: bool = True,
        device: str = "cuda"
    ):
        """
        初始化 ELA 特徵提取器
        
        Args:
            feature_dim: 特徵維度
            jpeg_quality: JPEG 壓縮品質 (1-100)
            backbone: 骨幹網路 (resnet50, resnet34)
            pretrained: 是否使用預訓練權重
            device: 運算裝置
        """
        super().__init__()
        
        self.feature_dim = feature_dim
        self.jpeg_quality = jpeg_quality
        self.device = device
        
        # 載入 ResNet 骨幹網路
        if backbone == "resnet50":
            resnet = models.resnet50(pretrained=pretrained)
            self.feature_dim = 2048
        elif backbone == "resnet34":
            resnet = models.resnet34(pretrained=pretrained)
            self.feature_dim = 512
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        
        # 移除最後的全連接層
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1])
        
        self.feature_extractor = self.feature_extractor.to(device)
        
        print(f"✅ ELA Feature Extractor initialized")
        print(f"   Backbone: {backbone}")
        print(f"   Feature Dim: {self.feature_dim}")
        print(f"   JPEG Quality: {jpeg_quality}")
        print(f"   Device: {device}")
    
    def compute_ela(self, images: torch.Tensor) -> torch.Tensor:
        """
        計算 ELA (Error Level Analysis) 地圖
        
        Args:
            images: 輸入圖像 (B, C, H, W)，值域 [-1, 1] 或 [0, 1]
            
        Returns:
            ELA 地圖 (B, C, H, W)
        """
        batch_size = images.shape[0]
        ela_maps = []
        
        for i in range(batch_size):
            # 單張圖像處理
            img = images[i]  # (C, H, W)
            
            # 正規化到 [0, 1]
            if img.min() < 0:
                img = (img + 1) / 2
            
            # 轉換為 numpy (H, W, C)
            img_np = img.permute(1, 2, 0).cpu().numpy()
            img_np = (img_np * 255).astype(np.uint8)
            
            # 轉為 PIL Image
            pil_img = Image.fromarray(img_np)
            
            # JPEG 壓縮
            buffer = io.BytesIO()
            pil_img.save(buffer, format='JPEG', quality=self.jpeg_quality)
            buffer.seek(0)
            
            # 重新讀取壓縮後的圖像
            compressed_img = Image.open(buffer)
            compressed_np = np.array(compressed_img)
            
            # 計算誤差（ELA）
            ela = np.abs(img_np.astype(float) - compressed_np.astype(float))
            
            # 增強對比（scale）
            ela = ela / 255.0
            ela = np.clip(ela * 10, 0, 1)  # 放大 10 倍
            
            # 轉回 torch tensor (C, H, W)
            ela_tensor = torch.from_numpy(ela).permute(2, 0, 1).float()
            ela_maps.append(ela_tensor)
        
        # Stack batch
        ela_batch = torch.stack(ela_maps).to(self.device)
        
        return ela_batch
    
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        前向傳播
        
        Args:
            images: 輸入圖像 (B, C, H, W)
            
        Returns:
            ELA 特徵 (B, 2048)
        """
        # 計算 ELA 地圖
        ela_maps = self.compute_ela(images)
        
        # 通過 ResNet 提取特徵
        features = self.feature_extractor(ela_maps)
        
        # Flatten
        features = features.view(features.size(0), -1)  # (B, 2048)
        
        return features
    
    def extract_features(self, images: torch.Tensor) -> torch.Tensor:
        """
        提取特徵（推論模式）
        
        Args:
            images: 輸入圖像 (B, C, H, W)
            
        Returns:
            特徵向量 (B, 2048)
        """
        self.eval()
        with torch.no_grad():
            return self.forward(images)
    
    def get_ela_visualization(self, images: torch.Tensor) -> torch.Tensor:
        """
        獲取 ELA 視覺化（用於可解釋性）
        
        Args:
            images: 輸入圖像 (B, C, H, W)
            
        Returns:
            ELA 地圖 (B, C, H, W)
        """
        with torch.no_grad():
            ela_maps = self.compute_ela(images)
        return ela_maps

def test_ela_extractor():
    """測試 ELA 特徵提取器"""
    print("🧪 Testing ELA Feature Extractor...")
    
    # 建立隨機測試圖像
    batch_size = 4
    test_images = torch.randn(batch_size, 3, 224, 224)
    test_images = (test_images + 1) / 2  # 正規化到 [0, 1]
    
    # 初始化提取器
    device = "cuda" if torch.cuda.is_available() else "cpu"
    extractor = ELAFeatureExtractor(device=device)
    
    # 提取特徵
    features = extractor.extract_features(test_images.to(device))
    
    print(f"\n✅ Test passed!")
    print(f"   Input shape: {test_images.shape}")
    print(f"   Feature shape: {features.shape}")
    print(f"   Expected: (4, 2048)")
    
    assert features.shape == (batch_size, 2048), "Feature shape mismatch!"
    
    # 測試 ELA 視覺化
    ela_maps = extractor.get_ela_visualization(test_images.to(device))
    print(f"   ELA map shape: {ela_maps.shape}")
    
    return extractor

if __name__ == "__main__":
    test_ela_extractor()
