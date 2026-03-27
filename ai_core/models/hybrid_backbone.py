from __future__ import annotations

from typing import List

import torch
import torch.nn as nn


class ConvBranch(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Sequential(nn.Conv2d(3, 32, 3, 2, 1), nn.BatchNorm2d(32), nn.SiLU()),
            nn.Sequential(nn.Conv2d(32, 64, 3, 2, 1), nn.BatchNorm2d(64), nn.SiLU()),
            nn.Sequential(nn.Conv2d(64, 128, 3, 2, 1), nn.BatchNorm2d(128), nn.SiLU()),
        ])

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        feats = []
        for layer in self.layers:
            x = layer(x)
            feats.append(x)
        return feats


class AttentionBranch(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=4),
            nn.GELU(),
            nn.Conv2d(64, 128, kernel_size=2, stride=2),
            nn.GELU(),
        )
        self.refine = nn.Conv2d(128, 128, kernel_size=1)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        x = self.proj(x)
        p3 = self.refine(x)
        p4 = nn.functional.avg_pool2d(p3, kernel_size=2, stride=2)
        p5 = nn.functional.avg_pool2d(p4, kernel_size=2, stride=2)
        return [p3, p4, p5]


class HybridBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = ConvBranch()
        self.attn = AttentionBranch()
        self.fuse = nn.ModuleList([
            nn.Conv2d(32 + 128, 128, 1),
            nn.Conv2d(64 + 128, 128, 1),
            nn.Conv2d(128 + 128, 128, 1),
        ])

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        conv_feats = self.conv(x)
        attn_feats = self.attn(x)
        fused = []
        for i in range(3):
            attn_resized = nn.functional.interpolate(
                attn_feats[i], size=conv_feats[i].shape[-2:], mode="bilinear", align_corners=False
            )
            fused.append(self.fuse[i](torch.cat([conv_feats[i], attn_resized], dim=1)))
        return fused
