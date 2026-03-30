from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .bifpn import BiFPN
from .depth_head import DepthHead
from .hybrid_backbone import HybridBackbone
from .multi_task_head import MultiTaskHead
from .pi_ge import PIGE


class AquaDetHybridModel(nn.Module):
    """AquaDet multi-task underwater perception model.

    Architecture:
        Input → PI-GE (physics restoration) → HybridBackbone → BiFPN (×3) →
        ├── MultiTaskHead → logits, boxes, masks, obj
        └── DepthHead (multi-scale input) → depth
    """

    def __init__(self, num_classes: int = 4, pi_ge_enabled: bool = True):
        super().__init__()
        self.pi_ge = PIGE(enabled=pi_ge_enabled)
        self.backbone = HybridBackbone()
        self.neck = BiFPN(channels=128, num_repeats=3)
        self.head = MultiTaskHead(in_channels=128, num_classes=num_classes)
        # Depth head receives concatenated P3+P4+P5 = 128*3 = 384 channels
        self.depth_head = DepthHead(in_channels=128 * 3)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        x = self.pi_ge(x)
        features = self.backbone(x)
        fused = self.neck(features)

        outputs = self.head(fused)

        # Feed all 3 FPN levels to depth head for global context
        p3, p4, p5 = fused
        p4_up = F.interpolate(p4, size=p3.shape[-2:], mode="bilinear", align_corners=False)
        p5_up = F.interpolate(p5, size=p3.shape[-2:], mode="bilinear", align_corners=False)
        depth_input = torch.cat([p3, p4_up, p5_up], dim=1)
        outputs["depth"] = self.depth_head(depth_input)

        return outputs
