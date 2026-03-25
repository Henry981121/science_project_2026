"""
配置檔載入與管理工具
支援 YAML 配置與命令列參數覆寫
"""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional
import argparse

class Config:
    """配置管理類別"""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        初始化配置
        
        Args:
            config_path: YAML 配置檔路徑
        """
        self.config = {}
        if config_path:
            self.load_from_yaml(config_path)
    
    def load_from_yaml(self, config_path: str):
        """
        從 YAML 檔案載入配置
        
        Args:
            config_path: YAML 檔案路徑
        """
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        print(f"✅ Configuration loaded from {config_path}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        獲取配置值（支援嵌套鍵，例如 'model.clip.feature_dim'）
        
        Args:
            key: 配置鍵（支援點分隔嵌套）
            default: 預設值
            
        Returns:
            配置值
        """
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
            else:
                return default
            
            if value is None:
                return default
        
        return value
    
    def set(self, key: str, value: Any):
        """
        設置配置值（支援嵌套鍵）
        
        Args:
            key: 配置鍵
            value: 配置值
        """
        keys = key.split('.')
        config = self.config
        
        # 遍歷到倒數第二個鍵
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        # 設置最後一個鍵的值
        config[keys[-1]] = value
    
    def update(self, updates: Dict[str, Any]):
        """
        批量更新配置
        
        Args:
            updates: 更新字典
        """
        for key, value in updates.items():
            self.set(key, value)
    
    def save(self, save_path: str):
        """
        保存配置到 YAML 檔案
        
        Args:
            save_path: 保存路徑
        """
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(save_path, 'w', encoding='utf-8') as f:
            yaml.dump(self.config, f, default_flow_style=False, allow_unicode=True)
        
        print(f"💾 Configuration saved to {save_path}")
    
    def __repr__(self) -> str:
        return f"Config({yaml.dump(self.config, default_flow_style=False)})"

def load_config(config_path: str = None, 
                default_config_path: str = 'configs/config.yaml') -> Config:
    """
    載入配置（便利函數）
    
    Args:
        config_path: 配置檔路徑
        default_config_path: 預設配置檔路徑
        
    Returns:
        Config 物件
    """
    path = config_path or default_config_path
    return Config(path)

def create_arg_parser() -> argparse.ArgumentParser:
    """
    建立命令列參數解析器
    
    Returns:
        ArgumentParser
    """
    parser = argparse.ArgumentParser(
        description='Explainable AI Image Detection System'
    )
    
    # 配置檔案
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                       help='Path to config file')
    
    # 訓練參數
    parser.add_argument('--batch_size', type=int, help='Batch size')
    parser.add_argument('--epochs', type=int, help='Number of epochs')
    parser.add_argument('--lr', type=float, help='Learning rate')
    
    # 資料參數
    parser.add_argument('--data_dir', type=str, help='Data directory')
    parser.add_argument('--image_size', type=int, help='Image size')
    
    # 模型參數
    parser.add_argument('--streams', nargs='+', 
                       choices=['clip', 'ela', 'fft', 'dire', 'noise'],
                       help='Enable specific streams')
    
    # 實驗參數
    parser.add_argument('--exp_name', type=str, help='Experiment name')
    parser.add_argument('--save_dir', type=str, help='Save directory')
    
    # 其他
    parser.add_argument('--device', type=str, choices=['cuda', 'cpu', 'mps'],
                       help='Computing device')
    parser.add_argument('--num_workers', type=int, help='Number of data loading workers')
    
    return parser

def merge_args_with_config(config: Config, args: argparse.Namespace) -> Config:
    """
    將命令列參數合併到配置中（覆寫）
    
    Args:
        config: Config 物件
        args: 命令列參數
        
    Returns:
        更新後的 Config
    """
    # 映射命令列參數到配置鍵
    arg_mapping = {
        'batch_size': 'training.batch_size',
        'epochs': 'training.num_epochs',
        'lr': 'training.learning_rate',
        'data_dir': 'data.processed_data_dir',
        'image_size': 'data.image_size',
        'device': 'hardware.device',
        'num_workers': 'hardware.num_workers',
    }
    
    # 更新配置
    for arg_name, config_key in arg_mapping.items():
        arg_value = getattr(args, arg_name, None)
        if arg_value is not None:
            config.set(config_key, arg_value)
            print(f"🔧 Override: {config_key} = {arg_value}")
    
    return config

if __name__ == "__main__":
    # 測試配置載入
    print("Testing configuration loader...")
    
    # 假設配置檔案存在
    try:
        config = load_config('configs/config.yaml')
        print("\n📋 Sample Configuration Values:")
        print(f"Batch Size: {config.get('training.batch_size')}")
        print(f"Learning Rate: {config.get('training.learning_rate')}")
        print(f"CLIP Model: {config.get('model.clip.model_name')}")
        print(f"Device: {config.get('hardware.device')}")
    except FileNotFoundError:
        print("⚠️  Config file not found. This is expected during testing.")
