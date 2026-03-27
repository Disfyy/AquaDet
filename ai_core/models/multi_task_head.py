from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn as nn


class MultiTaskHead(nn.Module):
    def __init__(self, in_channels: int = 128, num_classes: int = 4):
        super().__init__()
        self.num_classes = num_classes

        self.cls_head = nn.Conv2d(in_channels, num_classes, kernel_size=1)
        self.box_head = nn.Conv2d(in_channels, 4, kernel_size=1)
        self.mask_head = nn.Conv2d(in_channels, 1, kernel_size=1)

    def forward(self, features: List[torch.Tensor]) -> Dict[str, torch.Tensor]:
        p3 = features[0]
        return {
            "logits": self.cls_head(p3),
            "boxes": self.box_head(p3),
            "masks": self.mask_head(p3),
        }
