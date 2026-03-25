"""
PyTorch Dataset 與 DataLoader
支援多模態圖像載入、資料增強、課程學習
"""

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, Dict, List
import json

class DeepfakeDataset(Dataset):
    """
    Deepfake 檢測資料集
    
    支援功能：
    - 真假圖像載入
    - 難度等級過濾（課程學習）
    - 資料增強
    - 多模態輸出（RGB, ELA, FFT 等）
    """
    
    def __init__(
        self,
        data_dir: str,
        split: str = 'train',
        difficulty_levels: Optional[List[str]] = None,
        image_size: int = 224,
        augmentation: bool = True,
        return_metadata: bool = False
    ):
        """
        初始化資料集
        
        Args:
            data_dir: 資料目錄
            split: 'train', 'val', 或 'test'
            difficulty_levels: 難度等級過濾 ['easy', 'medium', 'hard']
            image_size: 圖像大小
            augmentation: 是否使用資料增強
            return_metadata: 是否返回元資料
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.difficulty_levels = difficulty_levels or ['easy', 'medium', 'hard']
        self.image_size = image_size
        self.augmentation = augmentation and (split == 'train')
        self.return_metadata = return_metadata
        
        # 建立轉換
        self.transform = self._build_transform()
        
        # 載入資料
        self.samples = self._load_samples()
        
        print(f"✅ {split.upper()} 資料集已載入: {len(self.samples)} 張圖像")
    
    def _build_transform(self):
        """建立圖像轉換 pipeline"""
        if self.augmentation:
            # 訓練集：資料增強
            return transforms.Compose([
                transforms.Resize((self.image_size, self.image_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=15),
                transforms.ColorJitter(
                    brightness=0.2,
                    contrast=0.2,
                    saturation=0.2,
                    hue=0.1
                ),
                transforms.RandomResizedCrop(
                    self.image_size,
                    scale=(0.8, 1.0),
                    ratio=(0.9, 1.1)
                ),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
        else:
            # 驗證/測試集：僅基本轉換
            return transforms.Compose([
                transforms.Resize((self.image_size, self.image_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
    
    def _load_samples(self) -> List[Dict]:
        """載入樣本列表"""
        samples = []
        
        # 掃描真實圖像
        real_dirs = [
            self.data_dir / 'wildfake' / 'real',
            self.data_dir / 'processed' / 'real'
        ]
        
        for real_dir in real_dirs:
            if real_dir.exists():
                for img_path in real_dir.glob('*.jpg'):
                    samples.append({
                        'path': str(img_path),
                        'label': 0,  # 0 = Real
                        'difficulty': 'medium',  # 預設
                        'source': 'real'
                    })
                for img_path in real_dir.glob('*.png'):
                    samples.append({
                        'path': str(img_path),
                        'label': 0,
                        'difficulty': 'medium',
                        'source': 'real'
                    })
        
        # 掃描假圖像
        fake_dirs = [
            self.data_dir / 'wildfake' / 'fake',
            self.data_dir / 'synthetic',
            self.data_dir / 'processed' / 'fake'
        ]
        
        for fake_dir in fake_dirs:
            if fake_dir.exists():
                for img_path in fake_dir.glob('*.jpg'):
                    samples.append({
                        'path': str(img_path),
                        'label': 1,  # 1 = Fake
                        'difficulty': 'medium',
                        'source': 'fake'
                    })
                for img_path in fake_dir.glob('*.png'):
                    samples.append({
                        'path': str(img_path),
                        'label': 1,
                        'difficulty': 'medium',
                        'source': 'fake'
                    })
        
        # 載入難度分類（如果有）
        metadata_path = self.data_dir / 'difficulty_classified' / 'difficulty_metadata.json'
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                difficulty_data = json.load(f)
                
                # 建立路徑到難度的映射
                path_to_difficulty = {
                    item['original_path']: item['difficulty']
                    for item in difficulty_data
                }
                
                # 更新樣本難度
                for sample in samples:
                    if sample['path'] in path_to_difficulty:
                        sample['difficulty'] = path_to_difficulty[sample['path']]
        
        # 根據難度過濾
        if self.difficulty_levels:
            samples = [
                s for s in samples
                if s['difficulty'] in self.difficulty_levels
            ]
        
        # 資料分割（簡單的按比例分割）
        np.random.seed(42)
        np.random.shuffle(samples)
        
        n = len(samples)
        if self.split == 'train':
            samples = samples[:int(0.7 * n)]
        elif self.split == 'val':
            samples = samples[int(0.7 * n):int(0.85 * n)]
        else:  # test
            samples = samples[int(0.85 * n):]
        
        return samples
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple:
        """
        獲取單個樣本
        
        Returns:
            (image_tensor, label) 或包含 metadata 的 tuple
        """
        sample = self.samples[idx]
        
        # 載入圖像
        try:
            image = Image.open(sample['path']).convert('RGB')
        except Exception as e:
            print(f"⚠️  載入圖像失敗: {sample['path']}, 錯誤: {e}")
            # 返回黑色圖像作為備用
            image = Image.new('RGB', (self.image_size, self.image_size), (0, 0, 0))
        
        # 應用轉換
        image_tensor = self.transform(image)
        
        # 標籤
        label = torch.tensor(sample['label'], dtype=torch.long)
        
        if self.return_metadata:
            return image_tensor, label, sample
        else:
            return image_tensor, label

def create_dataloaders(
    data_dir: str,
    batch_size: int = 32,
    num_workers: int = 4,
    image_size: int = 224,
    difficulty_levels: Optional[List[str]] = None,
    pin_memory: bool = True
) -> Dict[str, DataLoader]:
    """
    建立訓練、驗證、測試 DataLoader
    
    Args:
        data_dir: 資料目錄
        batch_size: 批次大小
        num_workers: 工作執行緒數
        image_size: 圖像大小
        difficulty_levels: 難度等級過濾
        pin_memory: 是否固定記憶體（GPU 加速）
        
    Returns:
        {'train': DataLoader, 'val': DataLoader, 'test': DataLoader}
    """
    dataloaders = {}
    
    for split in ['train', 'val', 'test']:
        dataset = DeepfakeDataset(
            data_dir=data_dir,
            split=split,
            difficulty_levels=difficulty_levels,
            image_size=image_size,
            augmentation=(split == 'train')
        )
        
        dataloaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split == 'train'),
            num_workers=num_workers,
            pin_memory=pin_memory and torch.cuda.is_available(),
            drop_last=(split == 'train')
        )
    
    print(f"\n✅ DataLoaders 已建立:")
    print(f"   Train: {len(dataloaders['train'].dataset)} 張")
    print(f"   Val:   {len(dataloaders['val'].dataset)} 張")
    print(f"   Test:  {len(dataloaders['test'].dataset)} 張")
    
    return dataloaders

if __name__ == "__main__":
    # 測試 Dataset
    print("🧪 測試 Dataset...")
    
    dataset = DeepfakeDataset(
        data_dir='data',
        split='train',
        image_size=224,
        augmentation=True
    )
    
    if len(dataset) > 0:
        image, label = dataset[0]
        print(f"\n✅ 測試成功！")
        print(f"   圖像形狀: {image.shape}")
        print(f"   標籤: {label.item()} ({'Real' if label == 0 else 'Fake'})")
    else:
        print("\n⚠️  資料集為空，請先準備資料")
