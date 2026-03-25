"""
Google Colab 專用工具函數
提供 Drive 掛載、GPU 檢測、進度顯示等功能
"""

import os
import sys
from pathlib import Path
from typing import Optional, Dict, Any
import json

def check_colab_environment() -> bool:
    """
    檢查是否在 Google Colab 環境中執行
    
    Returns:
        bool: True if running in Colab, False otherwise
    """
    try:
        import google.colab
        return True
    except ImportError:
        return False

def setup_gpu() -> Dict[str, Any]:
    """
    檢測並設置 GPU
    
    Returns:
        dict: GPU 資訊 (名稱、記憶體、可用性)
    """
    import torch
    
    gpu_info = {
        'available': torch.cuda.is_available(),
        'device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
        'device_name': None,
        'memory_total': None,
        'memory_allocated': None,
        'cuda_version': torch.version.cuda
    }
    
    if gpu_info['available']:
        gpu_info['device_name'] = torch.cuda.get_device_name(0)
        gpu_info['memory_total'] = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
        gpu_info['memory_allocated'] = torch.cuda.memory_allocated(0) / 1024**3  # GB
        
        # 驗證是否為 T4
        if 'T4' in gpu_info['device_name']:
            print(f"✅ Tesla T4 GPU detected!")
            print(f"   Total Memory: {gpu_info['memory_total']:.2f} GB")
        else:
            print(f"⚠️  GPU detected: {gpu_info['device_name']}")
            print(f"   (Expected T4, but can still proceed)")
    else:
        print("❌ No GPU detected! Training will be slow.")
        print("   Please enable GPU: Runtime -> Change runtime type -> GPU")
    
    return gpu_info

def mount_google_drive(mount_point: str = '/content/drive') -> bool:
    """
    掛載 Google Drive
    
    Args:
        mount_point: Drive 掛載路徑
        
    Returns:
        bool: 是否成功掛載
    """
    if not check_colab_environment():
        print("⚠️  Not in Colab environment. Skipping Drive mount.")
        return False
    
    try:
        from google.colab import drive
        drive.mount(mount_point)
        print(f"✅ Google Drive mounted at {mount_point}")
        return True
    except Exception as e:
        print(f"❌ Failed to mount Google Drive: {e}")
        return False

def setup_project_directories(base_path: str, create_on_drive: bool = True) -> Dict[str, Path]:
    """
    建立專案目錄結構
    
    Args:
        base_path: 基礎路徑（通常在 Google Drive 上）
        create_on_drive: 是否在 Drive 上建立
        
    Returns:
        dict: 各目錄的 Path 物件
    """
    base_path = Path(base_path)
    
    directories = {
        'base': base_path,
        'data': base_path / 'data',
        'data_raw': base_path / 'data' / 'raw',
        'data_processed': base_path / 'data' / 'processed',
        'data_synthetic': base_path / 'data' / 'synthetic',
        'data_wildfake': base_path / 'data' / 'wildfake',
        'data_difficulty': base_path / 'data' / 'difficulty_classified',
        'models': base_path / 'models',
        'checkpoints': base_path / 'outputs' / 'checkpoints',
        'visualizations': base_path / 'outputs' / 'visualizations',
        'heatmaps': base_path / 'outputs' / 'heatmaps',
        'logs': base_path / 'outputs' / 'logs',
    }
    
    # 建立所有目錄
    for name, path in directories.items():
        path.mkdir(parents=True, exist_ok=True)
        
    print(f"✅ Project directories created at {base_path}")
    return directories

def install_dependencies(requirements_file: Optional[str] = None, 
                        packages: Optional[list] = None,
                        quiet: bool = True) -> bool:
    """
    安裝 Python 套件
    
    Args:
        requirements_file: requirements.txt 路徑
        packages: 套件列表（如果不使用 requirements.txt）
        quiet: 是否安靜模式（減少輸出）
        
    Returns:
        bool: 是否成功安裝
    """
    import subprocess
    
    try:
        if requirements_file:
            cmd = ['pip', 'install', '-r', requirements_file]
            if quiet:
                cmd.append('-q')
            subprocess.run(cmd, check=True)
            print(f"✅ Installed packages from {requirements_file}")
            
        elif packages:
            for package in packages:
                cmd = ['pip', 'install', package]
                if quiet:
                    cmd.append('-q')
                subprocess.run(cmd, check=True)
            print(f"✅ Installed {len(packages)} packages")
            
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        return False

def save_checkpoint(model, optimizer, epoch, loss, filepath, 
                   additional_info: Optional[Dict] = None):
    """
    保存訓練檢查點
    
    Args:
        model: PyTorch 模型
        optimizer: 優化器
        epoch: 當前 epoch
        loss: 當前損失
        filepath: 保存路徑
        additional_info: 額外資訊
    """
    import torch
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    
    if additional_info:
        checkpoint.update(additional_info)
    
    torch.save(checkpoint, filepath)
    print(f"💾 Checkpoint saved: {filepath}")

def load_checkpoint(filepath, model, optimizer=None, device='cuda'):
    """
    載入訓練檢查點
    
    Args:
        filepath: 檢查點路徑
        model: PyTorch 模型
        optimizer: 優化器（可選）
        device: 運算裝置
        
    Returns:
        dict: 檢查點資訊
    """
    import torch
    
    checkpoint = torch.load(filepath, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    print(f"✅ Checkpoint loaded from: {filepath}")
    print(f"   Epoch: {checkpoint.get('epoch', 'N/A')}")
    print(f"   Loss: {checkpoint.get('loss', 'N/A')}")
    
    return checkpoint

def get_free_memory_mb() -> float:
    """
    獲取可用 GPU 記憶體（MB）
    
    Returns:
        float: 可用記憶體（MB）
    """
    import torch
    if torch.cuda.is_available():
        return (torch.cuda.get_device_properties(0).total_memory - 
                torch.cuda.memory_allocated(0)) / 1024**2
    return 0.0

def clear_gpu_memory():
    """
    清理 GPU 記憶體
    """
    import torch
    import gc
    
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(f"🧹 GPU memory cleared. Free: {get_free_memory_mb():.2f} MB")

def create_progress_bar(total, desc='Processing'):
    """
    建立進度條（使用 tqdm）
    
    Args:
        total: 總數
        desc: 描述
        
    Returns:
        tqdm object
    """
    from tqdm import tqdm
    return tqdm(total=total, desc=desc, ncols=100)

def print_system_info():
    """
    打印系統資訊
    """
    import torch
    import platform
    
    print("=" * 70)
    print("📊 System Information")
    print("=" * 70)
    print(f"Python Version: {sys.version.split()[0]}")
    print(f"PyTorch Version: {torch.__version__}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    
    print(f"Platform: {platform.platform()}")
    print("=" * 70)

if __name__ == "__main__":
    # 測試工具函數
    print_system_info()
    print("\n🔍 Checking Colab environment...")
    is_colab = check_colab_environment()
    print(f"Running in Colab: {is_colab}")
    
    print("\n🎮 Checking GPU...")
    gpu_info = setup_gpu()
