"""
ModelScope 資料集下載工具
用於從 ModelScope 下載 WildFake 資料集
"""

import os
from pathlib import Path
from typing import Optional
import shutil

def download_wildfake_dataset(
    save_dir: str,
    dataset_id: str = "hy2628982280/WildFake",
    subset: Optional[str] = None
):
    """
    從 ModelScope 下載 WildFake 資料集
    
    Args:
        save_dir: 儲存目錄
        dataset_id: ModelScope 資料集 ID
        subset: 子集名稱（如果有）
    """
    try:
        from modelscope.msdatasets import MsDataset
        from tqdm import tqdm
        
        print(f"🔽 開始下載 WildFake 資料集...")
        print(f"   資料集 ID: {dataset_id}")
        print(f"   儲存路徑: {save_dir}")
        
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # 下載資料集
        dataset = MsDataset.load(
            dataset_id,
            subset_name=subset,
            split='train'  # 或根據實際情況調整
        )
        
        print(f"✅ 資料集載入成功")
        print(f"   資料集大小: {len(dataset)} 筆")
        
        # 組織資料到 real 和 fake 資料夾
        real_dir = save_dir / 'real'
        fake_dir = save_dir / 'fake'
        real_dir.mkdir(exist_ok=True)
        fake_dir.mkdir(exist_ok=True)
        
        real_count = 0
        fake_count = 0
        
        print("\\n📦 開始處理資料...")
        for idx, sample in enumerate(tqdm(dataset, desc="Processing")):
            # 根據實際資料集結構調整
            # 假設資料集有 'image' 和 'label' 欄位
            image = sample.get('image')
            label = sample.get('label', 0)  # 0=real, 1=fake
            
            if image is not None:
                # 儲存圖像
                if label == 0:  # Real
                    image_path = real_dir / f"real_{real_count:06d}.jpg"
                    real_count += 1
                else:  # Fake
                    image_path = fake_dir / f"fake_{fake_count:06d}.jpg"
                    fake_count += 1
                
                # 儲存圖像（需根據實際格式調整）
                if hasattr(image, 'save'):
                    image.save(image_path)
        
        print(f"\\n✅ 資料集下載完成！")
        print(f"   真實圖像: {real_count} 張")
        print(f"   AI 生成圖像: {fake_count} 張")
        print(f"   總計: {real_count + fake_count} 張")
        
        # 生成元資料
        metadata = {
            'dataset_id': dataset_id,
            'total_images': real_count + fake_count,
            'real_images': real_count,
            'fake_images': fake_count,
            'save_dir': str(save_dir)
        }
        
        import json
        metadata_path = save_dir / 'metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"   元資料已儲存: {metadata_path}")
        
        return metadata
        
    except Exception as e:
        print(f"❌ 下載資料集時發生錯誤: {e}")
        print(f"   錯誤類型: {type(e).__name__}")
        import traceback
        traceback.print_exc()
        return None

def verify_dataset(dataset_dir: str):
    """
    驗證資料集完整性
    
    Args:
        dataset_dir: 資料集目錄
    """
    dataset_dir = Path(dataset_dir)
    
    print("🔍 驗證資料集...")
    
    real_dir = dataset_dir / 'real'
    fake_dir = dataset_dir / 'fake'
    
    if not real_dir.exists() or not fake_dir.exists():
        print("❌ 資料集目錄結構不完整")
        return False
    
    real_images = list(real_dir.glob('*.jpg')) + list(real_dir.glob('*.png'))
    fake_images = list(fake_dir.glob('*.jpg')) + list(fake_dir.glob('*.png'))
    
    print(f"✅ 資料集驗證通過")
    print(f"   真實圖像: {len(real_images)} 張")
    print(f"   AI 圖像: {len(fake_images)} 張")
    print(f"   總計: {len(real_images) + len(fake_images)} 張")
    
    return True

if __name__ == "__main__":
    # 測試下載
    import argparse
    
    parser = argparse.ArgumentParser(description='Download WildFake dataset')
    parser.add_argument('--save_dir', type=str, default='data/wildfake',
                       help='Directory to save dataset')
    parser.add_argument('--dataset_id', type=str, default='hy2628982280/WildFake',
                       help='ModelScope dataset ID')
    
    args = parser.parse_args()
    
    download_wildfake_dataset(args.save_dir, args.dataset_id)
