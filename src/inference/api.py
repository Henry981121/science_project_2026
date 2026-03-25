"""
推論 API
用於生產環境的圖像檢測
"""

import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
from pathlib import Path
from typing import Dict, Union, Optional, List
import json

class DeepfakeDetectorAPI:
    """
    Deepfake 檢測 API
    
    提供簡單的推論介面
    """
    
    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        enable_streams: Optional[List[str]] = None
    ):
        """
        初始化 API
        
        Args:
            model_path: 模型檢查點路徑
            device: 裝置
            enable_streams: 啟用的 Streams
        """
        self.device = device
        self.enable_streams = enable_streams or ['clip', 'ela', 'fft', 'noise']
        
        # 載入模型
        self.model = self._load_model(model_path)
        self.model.eval()
        
        print(f"✅ Deepfake Detector API initialized")
        print(f"   Device: {device}")
        print(f"   Streams: {self.enable_streams}")
    
    def _load_model(self, model_path: str):
        """載入模型"""
        from feature_extractors import MultiModalExtractor
        from fusion import DeepfakeDetector
        
        # 建立模型架構
        feature_extractor = MultiModalExtractor(
            enable_streams=self.enable_streams,
            device=self.device
        )
        
        model = DeepfakeDetector(
            feature_extractor=feature_extractor,
            fusion_type='attention',
            num_classes=2,
            hidden_dim=512,
            dropout=0.3
        )
        
        # 載入權重
        checkpoint = torch.load(model_path, map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(self.device)
        
        return model
    
    def preprocess_image(self, image: Union[str, Path, Image.Image, np.ndarray]) -> torch.Tensor:
        """
        預處理圖像
        
        Args:
            image: 圖像（路徑、PIL Image 或 numpy array）
            
        Returns:
            預處理後的張量 (1, 3, 224, 224)
        """
        from torchvision import transforms
        
        # 轉換為 PIL Image
        if isinstance(image, (str, Path)):
            image = Image.open(image).convert('RGB')
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        # 定義轉換
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        # 轉換並增加 batch 維度
        tensor = transform(image).unsqueeze(0)
        
        return tensor
    
    def predict(
        self,
        image: Union[str, Path, Image.Image, np.ndarray],
        return_explanations: bool = False
    ) -> Dict:
        """
        預測圖像真假
        
        Args:
            image: 輸入圖像
            return_explanations: 是否返回解釋
            
        Returns:
            預測結果字典
        """
        # 預處理
        image_tensor = self.preprocess_image(image).to(self.device)
        
        # 推論
        with torch.no_grad():
            logits = self.model(image_tensor)
            probs = F.softmax(logits, dim=1)[0]
            
            pred_class = probs.argmax().item()
            confidence = probs[pred_class].item()
        
        # 結果
        result = {
            'prediction': 'FAKE' if pred_class == 1 else 'REAL',
            'confidence': float(confidence),
            'probabilities': {
                'real': float(probs[0]),
                'fake': float(probs[1])
            }
        }
        
        # 生成解釋（可選）
        if return_explanations:
            from xai import MultiStreamExplainer
            
            explainer = MultiStreamExplainer(
                model=self.model,
                feature_extractor=self.model.feature_extractor,
                device=self.device
            )
            
            explanations = explainer.explain(image_tensor)
            result['explanations'] = {
                name: explanation.tolist()
                for name, explanation in explanations.items()
            }
        
        return result
    
    def batch_predict(
        self,
        images: List[Union[str, Path, Image.Image, np.ndarray]],
        batch_size: int = 8
    ) -> List[Dict]:
        """
        批次預測
        
        Args:
            images: 圖像列表
            batch_size: 批次大小
            
        Returns:
            預測結果列表
        """
        results = []
        
        for i in range(0, len(images), batch_size):
            batch_images = images[i:i + batch_size]
            
            # 預處理批次
            batch_tensors = torch.cat([
                self.preprocess_image(img) for img in batch_images
            ]).to(self.device)
            
            # 推論
            with torch.no_grad():
                logits = self.model(batch_tensors)
                probs = F.softmax(logits, dim=1)
            
            # 處理結果
            for j, prob in enumerate(probs):
                pred_class = prob.argmax().item()
                confidence = prob[pred_class].item()
                
                results.append({
                    'prediction': 'FAKE' if pred_class == 1 else 'REAL',
                    'confidence': float(confidence),
                    'probabilities': {
                        'real': float(prob[0]),
                        'fake': float(prob[1])
                    }
                })
        
        return results

def create_api(
    model_path: str = "outputs/checkpoints/best_model.pth",
    device: str = "cuda"
) -> DeepfakeDetectorAPI:
    """
    創建 API 實例
    
    Args:
        model_path: 模型路徑
        device: 裝置
        
    Returns:
        API 實例
    """
    return DeepfakeDetectorAPI(model_path, device)

if __name__ == "__main__":
    # 測試範例
    print("🚀 Deepfake Detector API")
    print("\nUsage:")
    print("  from inference import create_api")
    print("  api = create_api('path/to/model.pth')")
    print("  result = api.predict('path/to/image.jpg')")
    print("  print(result)")
