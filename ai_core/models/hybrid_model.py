from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn

from .bifpn import BiFPN
from .depth_head import DepthHead
from .hybrid_backbone import HybridBackbone
from .multi_task_head import MultiTaskHead
from .pi_ge import PIGE


class AquaDetHybridModel(nn.Module):
    def __init__(self, num_classes: int = 4, pi_ge_enabled: bool = True):
        super().__init__()
        self.pi_ge = PIGE(enabled=pi_ge_enabled)
        self.backbone = HybridBackbone()
        self.neck = BiFPN(channels=128)
        self.head = MultiTaskHead(in_channels=128, num_classes=num_classes)
        self.depth_head = DepthHead(in_channels=128)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        x = self.pi_ge(x)
        features = self.backbone(x)
        fused = self.neck(features)
        outputs = self.head(fused)
        outputs["depth"] = self.depth_head(fused[0])
        return outputs
