"""
視覺化工具
用於繪製訓練曲線、資料分佈、特徵圖、熱力圖等
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from typing import List, Optional, Tuple, Dict
import torch

# 設置繪圖風格
sns.set_style("whitegrid")
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'DejaVu Sans']  # 支援中文
plt.rcParams['axes.unicode_minus'] = False  # 正常顯示負號

def plot_training_curves(train_losses: List[float],
                         val_losses: List[float],
                         train_accs: Optional[List[float]] = None,
                         val_accs: Optional[List[float]] = None,
                         save_path: Optional[str] = None):
    """
    繪製訓練曲線
    
    Args:
        train_losses: 訓練損失列表
        val_losses: 驗證損失列表
        train_accs: 訓練準確率列表
        val_accs: 驗證準確率列表
        save_path: 保存路徑
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 損失曲線
    epochs = range(1, len(train_losses) + 1)
    axes[0].plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
    axes[0].plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # 準確率曲線
    if train_accs and val_accs:
        axes[1].plot(epochs, train_accs, 'b-', label='Training Accuracy', linewidth=2)
        axes[1].plot(epochs, val_accs, 'r-', label='Validation Accuracy', linewidth=2)
        axes[1].set_xlabel('Epoch', fontsize=12)
        axes[1].set_ylabel('Accuracy (%)', fontsize=12)
        axes[1].set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 Training curves saved to {save_path}")
    
    plt.show()

def plot_confusion_matrix(cm: np.ndarray,
                         class_names: List[str] = ['Real', 'Fake'],
                         save_path: Optional[str] = None):
    """
    繪製混淆矩陣
    
    Args:
        cm: 混淆矩陣 (2x2 numpy array)
        class_names: 類別名稱
        save_path: 保存路徑
    """
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Count'})
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 Confusion matrix saved to {save_path}")
    
    plt.show()

def plot_dataset_distribution(labels: List[int],
                             difficulty_levels: Optional[List[str]] = None,
                             save_path: Optional[str] = None):
    """
    繪製資料集分佈
    
    Args:
        labels: 標籤列表 (0=Real, 1=Fake)
        difficulty_levels: 難度等級列表
        save_path: 保存路徑
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 真假分佈
    unique, counts = np.unique(labels, return_counts=True)
    axes[0].bar(['Real', 'Fake'], counts, color=['green', 'red'], alpha=0.7)
    axes[0].set_ylabel('Count', fontsize=12)
    axes[0].set_title('Real vs Fake Distribution', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # 在柱狀圖上顯示數值
    for i, count in enumerate(counts):
        axes[0].text(i, count, str(count), ha='center', va='bottom', fontsize=12)
    
    # 難度分佈
    if difficulty_levels:
        unique_diff, counts_diff = np.unique(difficulty_levels, return_counts=True)
        axes[1].bar(unique_diff, counts_diff, color=['lightgreen', 'orange', 'red'], alpha=0.7)
        axes[1].set_ylabel('Count', fontsize=12)
        axes[1].set_title('Difficulty Distribution', fontsize=14, fontweight='bold')
        axes[1].grid(True, alpha=0.3, axis='y')
        
        for i, count in enumerate(counts_diff):
            axes[1].text(i, count, str(count), ha='center', va='bottom', fontsize=12)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 Dataset distribution saved to {save_path}")
    
    plt.show()

def visualize_augmentation(images: torch.Tensor,
                          titles: Optional[List[str]] = None,
                          save_path: Optional[str] = None):
    """
    視覺化資料增強效果
    
    Args:
        images: 圖像張量 (B, C, H, W)
        titles: 每張圖的標題
        save_path: 保存路徑
    """
    batch_size = images.shape[0]
    grid_size = int(np.ceil(np.sqrt(batch_size)))
    
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(12, 12))
    axes = axes.flatten()
    
    for i in range(batch_size):
        img = images[i].permute(1, 2, 0).cpu().numpy()
        # 正規化到 0-1
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)
        
        axes[i].imshow(img)
        axes[i].axis('off')
        if titles and i < len(titles):
            axes[i].set_title(titles[i], fontsize=10)
    
    # 隱藏多餘的子圖
    for i in range(batch_size, len(axes)):
        axes[i].axis('off')
    
    plt.suptitle('Data Augmentation Examples', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 Augmentation visualization saved to {save_path}")
    
    plt.show()

def plot_ablation_results(results: Dict[str, Dict[str, float]],
                         save_path: Optional[str] = None):
    """
    繪製消融實驗結果
    
    Args:
        results: 結果字典 {'exp_name': {'accuracy': 0.95, 'f1': 0.94, ...}}
        save_path: 保存路徑
    """
    exp_names = list(results.keys())
    metrics = ['accuracy', 'precision', 'recall', 'f1']
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(exp_names))
    width = 0.2
    
    for i, metric in enumerate(metrics):
        values = [results[exp][metric] * 100 for exp in exp_names]
        ax.bar(x + i * width, values, width, label=metric.capitalize(), alpha=0.8)
    
    ax.set_xlabel('Experiment', fontsize=12)
    ax.set_ylabel('Score (%)', fontsize=12)
    ax.set_title('Ablation Study Results', fontsize=14, fontweight='bold')
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(exp_names, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim([0, 105])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 Ablation results saved to {save_path}")
    
    plt.show()

def visualize_feature_maps(feature_maps: torch.Tensor,
                          title: str = 'Feature Maps',
                          save_path: Optional[str] = None,
                          max_channels: int = 16):
    """
    視覺化特徵圖
    
    Args:
        feature_maps: 特徵圖張量 (C, H, W)
        title: 標題
        save_path: 保存路徑
        max_channels: 最多顯示的通道數
    """
    num_channels = min(feature_maps.shape[0], max_channels)
    grid_size = int(np.ceil(np.sqrt(num_channels)))
    
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(12, 12))
    axes = axes.flatten()
    
    for i in range(num_channels):
        fm = feature_maps[i].cpu().detach().numpy()
        axes[i].imshow(fm, cmap='viridis')
        axes[i].axis('off')
        axes[i].set_title(f'Ch {i}', fontsize=8)
    
    # 隱藏多餘的子圖
    for i in range(num_channels, len(axes)):
        axes[i].axis('off')
    
    plt.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 Feature maps saved to {save_path}")
    
    plt.show()

if __name__ == "__main__":
    # 測試視覺化函數
    print("Testing visualization tools...")
    
    # 測試訓練曲線
    train_losses = [0.8, 0.6, 0.4, 0.3, 0.25, 0.2]
    val_losses = [0.85, 0.65, 0.5, 0.4, 0.35, 0.3]
    train_accs = [60, 70, 80, 85, 88, 90]
    val_accs = [58, 68, 78, 83, 86, 88]
    
    plot_training_curves(train_losses, val_losses, train_accs, val_accs)
    
    # 測試混淆矩陣
    cm = np.array([[450, 50], [30, 470]])
    plot_confusion_matrix(cm)
