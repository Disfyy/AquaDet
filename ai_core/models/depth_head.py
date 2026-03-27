from __future__ import annotations

import torch
import torch.nn as nn


class DepthHead(nn.Module):
    def __init__(self, in_channels: int = 128):
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=1),
            nn.Softplus(),
        )

    def forward(self, feature_map: torch.Tensor) -> torch.Tensor:
        return self.head(feature_map) + 0.1
