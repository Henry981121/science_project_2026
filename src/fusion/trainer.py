"""
Trainer v2.1
支援：雙分類頭損失、課程學習（Easy→Medium→Hard）、混合精度訓練
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from typing import Dict, Optional, Tuple, List
from pathlib import Path
import json
from tqdm import tqdm

from .fusion_module import DualHeadLoss


# ──────────────────────────────────────────────────────────────────────
# Curriculum Learning Scheduler
# ──────────────────────────────────────────────────────────────────────

class CurriculumScheduler:
    """
    Easy → Medium → Hard 課程學習調度器

    - Phase 1 (easy):   只用「簡單」樣本訓練，讓模型快速學到基礎判別能力
    - Phase 2 (medium): 加入「中等」樣本
    - Phase 3 (hard):   加入全部樣本（包含困難樣本）

    dataset 中每個樣本需要有 'difficulty' 屬性（0=easy, 1=medium, 2=hard）
    如果 dataset 沒有難度標籤，自動降級為標準訓練。
    """

    PHASES = [
        {"name": "easy",   "difficulties": [0],      "epochs": 10},
        {"name": "medium", "difficulties": [0, 1],   "epochs": 10},
        {"name": "hard",   "difficulties": [0, 1, 2], "epochs": 0},  # 0=remaining
    ]

    def __init__(self, dataset, total_epochs: int):
        self.dataset = dataset
        self.total_epochs = total_epochs
        self.has_difficulty = hasattr(dataset, "difficulties")

        if self.has_difficulty:
            # Pre-compute indices per difficulty
            self.indices_by_diff: Dict[int, List[int]] = {0: [], 1: [], 2: []}
            for i, d in enumerate(dataset.difficulties):
                self.indices_by_diff[int(d)].append(i)
            print(f"[Curriculum] easy={len(self.indices_by_diff[0])} "
                  f"medium={len(self.indices_by_diff[1])} "
                  f"hard={len(self.indices_by_diff[2])}")
        else:
            print("[Curriculum] No difficulty labels found, using all data each phase")

    def get_subset(self, epoch: int) -> Optional[List[int]]:
        """
        Return list of sample indices for this epoch.
        Returns None if should use full dataset.
        """
        if not self.has_difficulty:
            return None

        phase_end = 0
        for phase in self.PHASES:
            e = phase["epochs"]
            if e == 0:
                return None  # hard phase: all data
            phase_end += e
            if epoch < phase_end:
                indices = []
                for d in phase["difficulties"]:
                    indices.extend(self.indices_by_diff[d])
                return indices

        return None  # beyond schedule: all data

    def get_phase_name(self, epoch: int) -> str:
        phase_end = 0
        for phase in self.PHASES:
            e = phase["epochs"]
            if e == 0:
                return phase["name"]
            phase_end += e
            if epoch < phase_end:
                return phase["name"]
        return "hard"


# ──────────────────────────────────────────────────────────────────────
# Trainer
# ──────────────────────────────────────────────────────────────────────

class Trainer:
    """
    v2.1 Trainer:
    - Dual head loss (binary + source classification)
    - Curriculum learning (Easy -> Medium -> Hard)
    - Mixed precision (AMP)
    - Early stopping & checkpoint
    """

    def __init__(
        self,
        model: nn.Module,
        train_dataset,
        val_loader: DataLoader,
        optimizer: optim.Optimizer,
        scheduler=None,
        device: str = "cuda",
        save_dir: str = "outputs/checkpoints",
        lambda_source: float = 0.3,
        use_amp: bool = True,
        batch_size: int = 32,
        num_workers: int = 4,
    ):
        self.model = model.to(device)
        self.train_dataset = train_dataset
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.criterion = DualHeadLoss(lambda_source=lambda_source)
        self.use_amp = use_amp and device == "cuda"
        if self.use_amp:
            self.scaler = torch.amp.GradScaler("cuda")

        self.current_epoch = 0
        self.best_val_acc = 0.0
        self.history: Dict = {
            "train_loss": [], "train_acc": [],
            "val_loss": [], "val_acc": [], "lr": []
        }

        print(f"[Trainer] device={device} | AMP={self.use_amp} | lambda_source={lambda_source}")

    def _make_loader(self, indices=None) -> DataLoader:
        dataset = self.train_dataset
        if indices is not None:
            dataset = Subset(self.train_dataset, indices)
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def _train_epoch(self, loader: DataLoader) -> Tuple[float, float]:
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(loader, desc=f"Epoch {self.current_epoch+1} [Train]")
        for batch in pbar:
            # Support both (img, label_bin) and (img, label_bin, label_src)
            if len(batch) == 3:
                images, y_bin, y_src = batch
                y_src = y_src.to(self.device)
            else:
                images, y_bin = batch
                y_src = None

            images = images.to(self.device)
            y_bin  = y_bin.to(self.device)

            if self.use_amp:
                with torch.cuda.amp.autocast():
                    lb, ls, _ = self.model(images)
                    loss, _ = self.criterion(lb, ls, y_bin, y_src)
                self.optimizer.zero_grad()
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                lb, ls, _ = self.model(images)
                loss, _ = self.criterion(lb, ls, y_bin, y_src)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            total_loss += loss.item()
            pred = lb.argmax(dim=1)
            correct += pred.eq(y_bin).sum().item()
            total += y_bin.size(0)
            pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{100.*correct/total:.1f}%")

        return total_loss / len(loader), 100. * correct / total

    @torch.no_grad()
    def _validate(self) -> Tuple[float, float]:
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        for batch in tqdm(self.val_loader, desc=f"Epoch {self.current_epoch+1} [Val]"):
            if len(batch) == 3:
                images, y_bin, y_src = batch
                y_src = y_src.to(self.device)
            else:
                images, y_bin = batch
                y_src = None

            images = images.to(self.device)
            y_bin  = y_bin.to(self.device)

            lb, ls, _ = self.model(images)
            loss, _ = self.criterion(lb, ls, y_bin, y_src)

            total_loss += loss.item()
            pred = lb.argmax(dim=1)
            correct += pred.eq(y_bin).sum().item()
            total += y_bin.size(0)

        return total_loss / len(self.val_loader), 100. * correct / total

    def train(
        self,
        num_epochs: int = 30,
        early_stop_patience: int = 10,
        use_curriculum: bool = True,
    ):
        curriculum = CurriculumScheduler(self.train_dataset, num_epochs) if use_curriculum else None
        patience_counter = 0

        for epoch in range(num_epochs):
            self.current_epoch = epoch

            # Curriculum subset
            if curriculum is not None:
                indices = curriculum.get_subset(epoch)
                phase = curriculum.get_phase_name(epoch)
                print(f"\n[Curriculum] Phase: {phase} | "
                      f"samples={len(indices) if indices else len(self.train_dataset)}")
            else:
                indices = None

            loader = self._make_loader(indices)
            train_loss, train_acc = self._train_epoch(loader)
            val_loss, val_acc     = self._validate()

            self.history["train_loss"].append(train_loss)
            self.history["train_acc"].append(train_acc)
            self.history["val_loss"].append(val_loss)
            self.history["val_acc"].append(val_acc)

            lr = self.optimizer.param_groups[0]["lr"]
            self.history["lr"].append(lr)

            if self.scheduler:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()

            print(f"Epoch {epoch+1}/{num_epochs} | "
                  f"Train {train_loss:.4f}/{train_acc:.1f}% | "
                  f"Val {val_loss:.4f}/{val_acc:.1f}% | lr={lr:.6f}")

            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                patience_counter = 0
                self._save_checkpoint(is_best=True)
                print(f"  Best model saved (val_acc={val_acc:.2f}%)")
            else:
                patience_counter += 1

            if patience_counter >= early_stop_patience:
                print(f"\n[EarlyStop] No improvement for {early_stop_patience} epochs. Stopping.")
                break

        self._save_history()
        print(f"\nTraining done. Best val_acc={self.best_val_acc:.2f}%")

    def _save_checkpoint(self, is_best: bool = False):
        ckpt = {
            "epoch": self.current_epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_val_acc": self.best_val_acc,
            "history": self.history,
        }
        if self.scheduler:
            ckpt["scheduler_state_dict"] = self.scheduler.state_dict()
        path = self.save_dir / ("best_model.pth" if is_best else f"ckpt_ep{self.current_epoch+1}.pth")
        torch.save(ckpt, path)

    def load_checkpoint(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if self.scheduler and "scheduler_state_dict" in ckpt:
            self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        self.current_epoch = ckpt["epoch"]
        self.best_val_acc  = ckpt["best_val_acc"]
        self.history       = ckpt["history"]
        print(f"[Trainer] Loaded checkpoint from {path} (epoch={self.current_epoch})")

    def _save_history(self):
        p = self.save_dir / "training_history.json"
        with open(p, "w") as f:
            json.dump(self.history, f, indent=2)
        print(f"[Trainer] History saved -> {p}")


# ──────────────────────────────────────────────────────────────────────
# Helper: build optimizer + scheduler
# ──────────────────────────────────────────────────────────────────────

def build_optimizer_scheduler(
    model: nn.Module,
    lr: float = 1e-4,
    weight_decay: float = 1e-4,
    num_epochs: int = 30,
    scheduler_type: str = "cosine",
):
    """
    Two-group lr: CLIP projection (lower lr) vs rest.
    """
    # CLIP projection 用較低的 lr，其他用正常 lr
    clip_params = []
    if hasattr(model, 'extractors') and "clip" in model.extractors:
        clip_params = list(model.extractors["clip"].proj.parameters())
    clip_ids = set(id(p) for p in clip_params)
    other_params = [p for p in model.parameters() if id(p) not in clip_ids]

    param_groups = [{"params": other_params, "lr": lr}]
    if clip_params:
        param_groups.append({"params": clip_params, "lr": lr * 0.1})

    optimizer = optim.AdamW(param_groups, weight_decay=weight_decay)

    if scheduler_type == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=lr*0.01)
    elif scheduler_type == "plateau":
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)
    else:
        scheduler = None

    return optimizer, scheduler
