from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DepthHead(nn.Module):
    """Monocular Depth Estimation Head.

    Uses Atrous Spatial Pyramid Pooling (ASPP) to capture multi-scale
    context for robust depth perception in underwater environments.
    Accepts concatenated multi-scale features (P3+P4+P5) for global context.
    """

    def __init__(self, in_channels: int = 384):
        """Initialise depth head.

        Args:
            in_channels: Number of input channels.  When fed the concatenation
                of P3+P4+P5 (each 128ch), this should be 384.
        """
        super().__init__()

        # Reduce channel count from the concatenated input
        self.reduce = nn.Sequential(
            nn.Conv2d(in_channels, 128, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

        # Multi-scale ASPP branches (operate on 128ch)
        self.branch1 = nn.Sequential(
            nn.Conv2d(128, 32, kernel_size=3, padding=1, dilation=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(128, 32, kernel_size=3, padding=6, dilation=6),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(128, 32, kernel_size=3, padding=12, dilation=12),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

        # Global context branch
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.global_conv = nn.Sequential(
            nn.Conv2d(128, 32, kernel_size=1),
            nn.ReLU(inplace=True),
        )

        # Fusion and final depth prediction
        self.fusion = nn.Sequential(
            nn.Conv2d(32 * 4, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=1),
            nn.Softplus(),  # Ensures depth is strictly positive (> 0)
        )

    def forward(self, feature_map: torch.Tensor) -> torch.Tensor:
        x = self.reduce(feature_map)
        h, w = x.shape[2:]

        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)

        bg = self.global_pool(x)
        bg = self.global_conv(bg)
        bg = F.interpolate(bg, size=(h, w), mode='bilinear', align_corners=False)

        fused = torch.cat([b1, b2, b3, bg], dim=1)

        # Depth in meters with 0.1m safety floor
        return self.fusion(fused) + 0.1
