"""
v2_fusion.losses — 唯一的 loss 定義
====================================
Total = CE_binary
      + lambda_src * CE_source
      + 1.0        * CE_gen        ← 權重固定，不可設定

lambda_grl 不在這裡。它只出現在 model.grad_reverse()，一次。
這個檔案刻意沒有 gen loss 的權重參數 —— 沒有欄位可以設，就不可能把 λ 乘第二次。
（audit 發現 05：舊版 backbone 實收 λ² = 0.00198，比反方向的 λ_src=0.1 弱 50.6 倍。）
"""

from typing import Dict, Tuple

import torch
import torch.nn as nn

from .config import LossConfig


class FusionLoss(nn.Module):

    GEN_IGNORE_INDEX = -1     # 真圖沒有 generator 身份，data.py 填 -1

    def __init__(self, cfg: LossConfig):
        super().__init__()
        self.cfg = cfg
        self.ce_binary = nn.CrossEntropyLoss()
        self.ce_source = nn.CrossEntropyLoss()
        self.ce_gen = nn.CrossEntropyLoss(ignore_index=self.GEN_IGNORE_INDEX)

    def forward(
        self,
        out: Dict[str, torch.Tensor],
        y_bin: torch.Tensor,
        y_src: torch.Tensor,
        y_gen: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:

        loss_bin = self.ce_binary(out['logits_binary'], y_bin)
        total = loss_bin

        if self.cfg.lambda_src > 0.0:
            loss_src = self.ce_source(out['logits_source'], y_src)
            total = total + self.cfg.lambda_src * loss_src
        else:
            # λ_src = 0 時完全不建圖，source head 不會收到任何梯度。
            # 舊版即使 λ=0 也照算 CE，白花時間。
            loss_src = torch.zeros((), device=loss_bin.device)

        if self.cfg.lambda_grl > 0.0:
            n_fake = int((y_gen != self.GEN_IGNORE_INDEX).sum())
            if n_fake > 0:
                loss_gen = self.ce_gen(out['logits_gen'], y_gen)
                total = total + loss_gen        # 權重固定 1.0，λ 在 GRL 裡
            else:
                loss_gen = torch.zeros((), device=loss_bin.device)
        else:
            loss_gen = torch.zeros((), device=loss_bin.device)

        return total, {
            'loss_total':  float(total.detach()),
            'loss_binary': float(loss_bin.detach()),
            'loss_source': float(loss_src.detach()),
            'loss_gen':    float(loss_gen.detach()),
        }
