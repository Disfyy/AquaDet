from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiTaskHead(nn.Module):
    """Multi-task prediction head with explicit objectness output.

    Produces four outputs from concatenated multi-scale features:
    - logits: per-class scores (B, num_classes, H, W)
    - boxes: bounding-box offsets (B, 4, H, W)
    - masks: instance segmentation logits (B, 1, H, W)
    - obj: objectness logits (B, 1, H, W)
    """

    def __init__(self, in_channels: int = 128, num_classes: int = 4):
        super().__init__()
        self.num_classes = num_classes
        fused_channels = in_channels * 3  # concatenation of P3, P4, P5

        # Shared feature refinement before task-specific heads
        self.shared = nn.Sequential(
            nn.Conv2d(fused_channels, in_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.SiLU(inplace=True),
        )

        self.cls_head = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(in_channels, num_classes, 1),
        )
        self.box_head = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(in_channels, 4, 1),
        )
        self.mask_head = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(in_channels, 1, 1),
        )
        self.obj_head = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(in_channels, 1, 1),
        )

    def forward(self, features: List[torch.Tensor]) -> Dict[str, torch.Tensor]:
        p3, p4, p5 = features

        # Upsample p4 and p5 to p3 resolution for true multi-scale fusion
        p4_up = F.interpolate(p4, size=p3.shape[-2:], mode="bilinear", align_corners=False)
        p5_up = F.interpolate(p5, size=p3.shape[-2:], mode="bilinear", align_corners=False)

        fused = torch.cat([p3, p4_up, p5_up], dim=1)
        shared = self.shared(fused)

        return {
            "logits": self.cls_head(shared),
            "boxes": self.box_head(shared),
            "masks": self.mask_head(shared),
            "obj": self.obj_head(shared),
        }
