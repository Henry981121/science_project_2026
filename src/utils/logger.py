"""
日誌系統
支援檔案輸出、控制台輸出、彩色日誌、TensorBoard 整合
"""

import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional

# ANSI 顏色代碼
class Colors:
    """控制台顏色"""
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

class ColoredFormatter(logging.Formatter):
    """彩色日誌格式化器"""
    
    COLORS = {
        'DEBUG': Colors.OKCYAN,
        'INFO': Colors.OKGREEN,
        'WARNING': Colors.WARNING,
        'ERROR': Colors.FAIL,
        'CRITICAL': Colors.FAIL + Colors.BOLD,
    }
    
    def format(self, record):
        # 備份原始訊息
        original_msg = record.msg
        
        # 添加顏色
        if record.levelname in self.COLORS:
            record.msg = f"{self.COLORS[record.levelname]}{record.msg}{Colors.ENDC}"
        
        # 格式化
        result = super().format(record)
        
        # 恢復原始訊息
        record.msg = original_msg
        
        return result

def setup_logger(name: str = 'AI_Detection',
                log_dir: Optional[str] = None,
                level: int = logging.INFO,
                console: bool = True,
                file_logging: bool = True,
                colored: bool = True) -> logging.Logger:
    """
    設置日誌系統
    
    Args:
        name: Logger 名稱
        log_dir: 日誌檔案目錄
        level: 日誌等級
        console: 是否輸出到控制台
        file_logging: 是否輸出到檔案
        colored: 是否使用彩色輸出
        
    Returns:
        Logger 物件
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # 清除現有 handlers
    logger.handlers.clear()
    
    # 日誌格式
    if colored:
        formatter = ColoredFormatter(
            '[%(asctime)s] [%(levelname)s] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
    else:
        formatter = logging.Formatter(
            '[%(asctime)s] [%(levelname)s] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
    
    # 控制台輸出
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # 檔案輸出
    if file_logging and log_dir:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # 建立日誌檔案（以時間命名）
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = log_dir / f'{name}_{timestamp}.log'
        
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(level)
        
        # 檔案不使用顏色
        plain_formatter = logging.Formatter(
            '[%(asctime)s] [%(levelname)s] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(plain_formatter)
        logger.addHandler(file_handler)
        
        logger.info(f"📝 Logging to file: {log_file}")
    
    return logger

def log_experiment_config(logger: logging.Logger, config: dict):
    """
    記錄實驗配置
    
    Args:
        logger: Logger 物件
        config: 配置字典
    """
    logger.info("=" * 70)
    logger.info("📋 Experiment Configuration")
    logger.info("=" * 70)
    
    def log_dict(d, indent=0):
        for key, value in d.items():
            if isinstance(value, dict):
                logger.info("  " * indent + f"{key}:")
                log_dict(value, indent + 1)
            else:
                logger.info("  " * indent + f"{key}: {value}")
    
    log_dict(config)
    logger.info("=" * 70)

def log_model_summary(logger: logging.Logger, model, input_size: tuple):
    """
    記錄模型摘要
    
    Args:
        logger: Logger 物件
        model: PyTorch 模型
        input_size: 輸入尺寸
    """
    try:
        from torchinfo import summary
        logger.info("=" * 70)
        logger.info("🏗️  Model Architecture")
        logger.info("=" * 70)
        model_summary = summary(model, input_size=input_size, verbose=0)
        logger.info(str(model_summary))
        logger.info("=" * 70)
    except ImportError:
        logger.warning("torchinfo not installed. Skipping model summary.")
        logger.info(f"Model: {model.__class__.__name__}")
        
        # 計算參數量
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Total Parameters: {total_params:,}")
        logger.info(f"Trainable Parameters: {trainable_params:,}")

class TensorBoardLogger:
    """TensorBoard 日誌包裝器"""
    
    def __init__(self, log_dir: str, enabled: bool = True):
        """
        初始化 TensorBoard Logger
        
        Args:
            log_dir: TensorBoard 日誌目錄
            enabled: 是否啟用
        """
        self.enabled = enabled
        self.writer = None
        
        if enabled:
            try:
                from torch.utils.tensorboard import SummaryWriter
                self.writer = SummaryWriter(log_dir)
                print(f"📊 TensorBoard logging enabled: {log_dir}")
                print(f"   Run: tensorboard --logdir {log_dir}")
            except ImportError:
                print("⚠️  TensorBoard not available. Install with: pip install tensorboard")
                self.enabled = False
    
    def log_scalar(self, tag: str, value: float, step: int):
        """記錄標量值"""
        if self.enabled and self.writer:
            self.writer.add_scalar(tag, value, step)
    
    def log_scalars(self, main_tag: str, tag_scalar_dict: dict, step: int):
        """記錄多個標量值"""
        if self.enabled and self.writer:
            self.writer.add_scalars(main_tag, tag_scalar_dict, step)
    
    def log_image(self, tag: str, image, step: int):
        """記錄圖像"""
        if self.enabled and self.writer:
            self.writer.add_image(tag, image, step)
    
    def log_histogram(self, tag: str, values, step: int):
        """記錄直方圖"""
        if self.enabled and self.writer:
            self.writer.add_histogram(tag, values, step)
    
    def close(self):
        """關閉 writer"""
        if self.writer:
            self.writer.close()

if __name__ == "__main__":
    # 測試日誌系統
    logger = setup_logger(
        name='TestLogger',
        log_dir='outputs/logs',
        colored=True
    )
    
    logger.debug("This is a DEBUG message")
    logger.info("This is an INFO message")
    logger.warning("This is a WARNING message")
    logger.error("This is an ERROR message")
    logger.critical("This is a CRITICAL message")
    
    # 測試配置記錄
    test_config = {
        'training': {
            'batch_size': 32,
            'learning_rate': 0.0001
        },
        'model': {
            'name': 'MultiStreamDetector'
        }
    }
    log_experiment_config(logger, test_config)
