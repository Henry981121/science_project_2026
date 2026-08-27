"""
v2_fusion.losses — 唯一的 loss 定義
====================================
Total = CE_binary + lambda_src * CE_source

2026-08-27：GRL 移除後，這裡只剩兩項。
沒有 gen loss、沒有 lambda_grl —— 也就不存在 audit 發現 05 那種
「同一個 λ 被乘兩次」的可能性了。

lambda_src 作用在 head_source 上，是**正常梯度**的 auxiliary supervision，
不是對抗。預設 0（關閉）；舊版 s3_main_grl.py 是「意外」開在 0.1 的。
"""

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from .config import LossConfig, SOURCE_IGNORE_INDEX


class FusionLoss(nn.Module):

    # cross_generator_test 專屬的 generator 沒有 source 標籤，data.py 填 -1。
    # train 裡不會出現 -1，但 ignore_index 讓這件事不必靠慣例保證。
    SOURCE_IGNORE_INDEX = SOURCE_IGNORE_INDEX

    def __init__(self, cfg: LossConfig):
        super().__init__()
        self.cfg = cfg
        self.ce_binary = nn.CrossEntropyLoss()
        self.ce_source = nn.CrossEntropyLoss(ignore_index=self.SOURCE_IGNORE_INDEX)

    def forward(
        self,
        out: Dict[str, torch.Tensor],
        y_bin: torch.Tensor,
        y_src: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:

        loss_bin = self.ce_binary(out['logits_binary'], y_bin)
        total = loss_bin

        if self.cfg.lambda_src > 0.0:
            if 'logits_source' not in out:
                raise RuntimeError(
                    "lambda_src > 0 但模型沒有 source head。\n"
                    "  請設 ModelConfig.use_source_head=True，或把 lambda_src 設回 0。")
            if y_src is None:
                raise RuntimeError("lambda_src > 0 但沒有傳入 y_src。")
            loss_src = self.ce_source(out['logits_source'], y_src)
            total = total + self.cfg.lambda_src * loss_src
        else:
            # λ_src = 0 時完全不建圖，source head 不會收到任何梯度。
            # 舊版即使 λ=0 也照算 CE，白花時間。
            loss_src = torch.zeros((), device=loss_bin.device)

        return total, {
            'loss_total':  float(total.detach()),
            'loss_binary': float(loss_bin.detach()),
            'loss_source': float(loss_src.detach()),
        }
