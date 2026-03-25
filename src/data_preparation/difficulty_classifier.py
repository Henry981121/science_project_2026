"""
圖像難度分類器
根據 ELA、FFT、CLIP 等特徵自動分類圖像難度（Easy/Medium/Hard）
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, List
import json
from tqdm import tqdm
from PIL import Image

def calculate_ela_variance(image_path: str, quality: int = 90) -> float:
    """
    計算 ELA (Error Level Analysis) 方差
    用於偵測 JPEG 壓縮不一致性
    
    Args:
        image_path: 圖像路徑
        quality: JPEG 壓縮品質
        
    Returns:
        ELA 方差值（越高表示壓縮痕跡越明顯）
    """
    try:
        # 讀取原始圖像
        img = Image.open(image_path)
        
        # 臨時儲存為 JPEG
        import io
        temp_io = io.BytesIO()
        img.save(temp_io, format='JPEG', quality=quality)
        temp_io.seek(0)
        
        # 重新讀取
        compressed_img = Image.open(temp_io)
        
        # 轉換為 numpy 陣列
        original = np.array(img.convert('RGB'))
        compressed = np.array(compressed_img.convert('RGB'))
        
        # 計算誤差
        ela = np.abs(original.astype(float) - compressed.astype(float))
        
        # 計算方差（標準化到 0-100）
        variance = np.var(ela)
        
        return float(variance)
        
    except Exception as e:
        print(f"⚠️  計算 ELA 時發生錯誤: {e}")
        return 0.0

def calculate_fft_energy(image_path: str) -> float:
    """
    計算 FFT 高頻能量
    真實圖像通常有更多高頻成分
    
    Args:
        image_path: 圖像路徑
        
    Returns:
        高頻能量比例
    """
    try:
        # 讀取圖像（灰階）
        img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        
        if img is None:
            return 0.0
        
        # FFT 轉換
        f = np.fft.fft2(img)
        fshift = np.fft.fftshift(f)
        magnitude = np.abs(fshift)
        
        # 計算高頻區域能量（外圍區域）
        rows, cols = img.shape
        crow, ccol = rows // 2, cols // 2
        
        # 建立高頻遮罩（外圍 30%）
        mask = np.ones((rows, cols), dtype=np.uint8)
        r = int(min(rows, cols) * 0.35)
        cv2.circle(mask, (ccol, crow), r, 0, -1)
        
        # 計算高頻與總能量比
        high_freq_energy = np.sum(magnitude * mask)
        total_energy = np.sum(magnitude)
        
        ratio = high_freq_energy / (total_energy + 1e-8)
        
        return float(ratio)
        
    except Exception as e:
        print(f"⚠️  計算 FFT 時發生錯誤: {e}")
        return 0.0

def calculate_clip_confidence(image_path: str, model=None, processor=None) -> float:
    """
    計算 CLIP 對圖像內容的信心分數
    （需要預先載入 CLIP 模型）
    
    Args:
        image_path: 圖像路徑
        model: CLIP 模型（可選）
        processor: CLIP 處理器（可選）
        
    Returns:
        信心分數（0-1）
    """
    # 簡化版：不使用 CLIP，返回預設值
    # 完整版需要載入 CLIP 模型（較耗記憶體）
    return 0.5

def classify_difficulty(
    image_path: str,
    ela_threshold_easy: float = 50.0,
    ela_threshold_hard: float = 10.0,
    fft_threshold_low: float = 0.3,
    fft_threshold_high: float = 0.5
) -> Tuple[str, Dict[str, float]]:
    """
    根據多種特徵分類圖像難度
    
    分類準則：
    - Easy: 高 ELA 方差（明顯壓縮痕跡）或低 FFT 高頻能量
    - Hard: 低 ELA 方差且高 FFT 高頻能量（接近真實圖像）
    - Medium: 介於之間
    
    Args:
        image_path: 圖像路徑
        ela_threshold_easy: ELA 簡單閾值
        ela_threshold_hard: ELA 困難閾值
        fft_threshold_low: FFT 低閾值
        fft_threshold_high: FFT 高閾值
        
    Returns:
        (難度等級, 特徵字典)
    """
    # 計算特徵
    ela_var = calculate_ela_variance(image_path)
    fft_energy = calculate_fft_energy(image_path)
    
    features = {
        'ela_variance': ela_var,
        'fft_high_freq_energy': fft_energy
    }
    
    # 難度分類邏輯
    if ela_var > ela_threshold_easy or fft_energy < fft_threshold_low:
        difficulty = 'easy'
    elif ela_var < ela_threshold_hard and fft_energy > fft_threshold_high:
        difficulty = 'hard'
    else:
        difficulty = 'medium'
    
    return difficulty, features

def classify_dataset(
    input_dir: str,
    output_base_dir: str,
    recursive: bool = False
):
    """
    批次分類整個資料集
    
    Args:
        input_dir: 輸入目錄
        output_base_dir: 輸出基礎目錄
        recursive: 是否遞迴搜尋
    """
    input_dir = Path(input_dir)
    output_base_dir = Path(output_base_dir)
    
    # 建立輸出目錄
    easy_dir = output_base_dir / 'easy'
    medium_dir = output_base_dir / 'medium'
    hard_dir = output_base_dir / 'hard'
    
    for dir in [easy_dir, medium_dir, hard_dir]:
        dir.mkdir(parents=True, exist_ok=True)
    
    # 取得所有圖像
    if recursive:
        image_files = list(input_dir.rglob('*.jpg')) + list(input_dir.rglob('*.png'))
    else:
        image_files = list(input_dir.glob('*.jpg')) + list(input_dir.glob('*.png'))
    
    print(f"🔍 找到 {len(image_files)} 張圖像")
    
    # 分類統計
    stats = {
        'easy': 0,
        'medium': 0,
        'hard': 0
    }
    
    metadata = []
    
    print("\n📊 開始分類...")
    for img_path in tqdm(image_files, desc="Classifying"):
        try:
            # 分類
            difficulty, features = classify_difficulty(str(img_path))
            
            # 複製到對應目錄
            if difficulty == 'easy':
                dest = easy_dir / img_path.name
            elif difficulty == 'medium':
                dest = medium_dir / img_path.name
            else:  # hard
                dest = hard_dir / img_path.name
            
            # 複製檔案
            import shutil
            shutil.copy2(img_path, dest)
            
            # 更新統計
            stats[difficulty] += 1
            
            # 記錄元資料
            metadata.append({
                'original_path': str(img_path),
                'difficulty': difficulty,
                'features': features,
                'new_path': str(dest)
            })
            
        except Exception as e:
            print(f"\n⚠️  處理 {img_path} 時發生錯誤: {e}")
    
    # 儲存元資料
    metadata_path = output_base_dir / 'difficulty_metadata.json'
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    # 顯示統計
    print(f"\n✅ 分類完成！")
    print(f"\n📊 難度分佈:")
    print(f"   Easy:   {stats['easy']:4d} ({stats['easy']/len(image_files)*100:.1f}%)")
    print(f"   Medium: {stats['medium']:4d} ({stats['medium']/len(image_files)*100:.1f}%)")
    print(f"   Hard:   {stats['hard']:4d} ({stats['hard']/len(image_files)*100:.1f}%)")
    print(f"\n   總計: {len(image_files)}")
    print(f"\n💾 元資料已儲存: {metadata_path}")
    
    return stats

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Classify images by difficulty')
    parser.add_argument('--input_dir', type=str, required=True,
                       help='Input directory containing images')
    parser.add_argument('--output_dir', type=str, default='data/difficulty_classified',
                       help='Output base directory')
    parser.add_argument('--recursive', action='store_true',
                       help='Search recursively')
    
    args = parser.parse_args()
    
    classify_dataset(args.input_dir, args.output_dir, args.recursive)
