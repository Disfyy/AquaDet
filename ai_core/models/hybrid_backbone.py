from __future__ import annotations

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBranch(nn.Module):
    """Standard convolutional feature extractor producing 3 scale levels."""

    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Sequential(nn.Conv2d(3, 32, 3, 2, 1), nn.BatchNorm2d(32), nn.SiLU()),
            nn.Sequential(nn.Conv2d(32, 64, 3, 2, 1), nn.BatchNorm2d(64), nn.SiLU()),
            nn.Sequential(nn.Conv2d(64, 128, 3, 2, 1), nn.BatchNorm2d(128), nn.SiLU()),
        ])

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        feats: List[torch.Tensor] = []
        for layer in self.layers:
            x = layer(x)
            feats.append(x)
        return feats


class ChannelAttention(nn.Module):
    """Squeeze-and-Excitation channel attention."""

    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.fc(self.pool(x))


class AttentionBranch(nn.Module):
    """Hierarchical attention branch producing genuinely distinct multi-scale features.

    Each stage has its own convolution + activation + SE attention, producing
    features at three stride levels (4×, 8×, 16× relative to input).
    This replaces the old approach of computing one feature map and
    faking others via avg_pool2d.
    """

    def __init__(self):
        super().__init__()
        # Stage 1: stride-4, 64 channels
        self.stage1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=4),
            nn.BatchNorm2d(64),
            nn.GELU(),
            ChannelAttention(64),
        )
        # Stage 2: stride-8 (2× from stage1), 128 channels
        self.stage2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=2, stride=2),
            nn.BatchNorm2d(128),
            nn.GELU(),
            ChannelAttention(128),
        )
        # Stage 3: stride-16 (2× from stage2), 128 channels
        self.stage3 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=2, stride=2),
            nn.BatchNorm2d(128),
            nn.GELU(),
            ChannelAttention(128),
        )
        # Project stage1 from 64ch → 128ch to match the fusion layer
        self.proj = nn.Conv2d(64, 128, kernel_size=1)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        s1 = self.stage1(x)   # (B, 64, H/4, W/4)
        s2 = self.stage2(s1)  # (B, 128, H/8, W/8)
        s3 = self.stage3(s2)  # (B, 128, H/16, W/16)
        return [self.proj(s1), s2, s3]  # all 128ch, 3 genuine feature levels


class HybridBackbone(nn.Module):
    """Dual-branch backbone fusing convolutional and attention pathways."""

    def __init__(self):
        super().__init__()
        self.conv = ConvBranch()
        self.attn = AttentionBranch()
        self.fuse = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(32 + 128, 128, 1, bias=False),
                nn.BatchNorm2d(128),
                nn.SiLU(inplace=True),
            ),
            nn.Sequential(
                nn.Conv2d(64 + 128, 128, 1, bias=False),
                nn.BatchNorm2d(128),
                nn.SiLU(inplace=True),
            ),
            nn.Sequential(
                nn.Conv2d(128 + 128, 128, 1, bias=False),
                nn.BatchNorm2d(128),
                nn.SiLU(inplace=True),
            ),
        ])

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        conv_feats = self.conv(x)
        attn_feats = self.attn(x)
        fused: List[torch.Tensor] = []
        for i in range(3):
            attn_resized = F.interpolate(
                attn_feats[i], size=conv_feats[i].shape[-2:],
                mode="bilinear", align_corners=False,
            )
            fused.append(self.fuse[i](torch.cat([conv_feats[i], attn_resized], dim=1)))
        return fused
