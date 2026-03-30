from __future__ import annotations

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


class BiFPNLayer(nn.Module):
    """Single BiFPN round: top-down + bottom-up with fast normalised fusion.

    Each fusion node has its own learnable weight vector whose size matches the
    exact number of inputs to that node, fixing the original dimension mismatch.
    """

    def __init__(self, channels: int = 128):
        super().__init__()
        self.epsilon = 1e-4

        # --- Top-Down convolutions ---
        self.td_conv_p4 = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True),
        )
        self.td_conv_p3 = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True),
        )

        # --- Bottom-Up convolutions ---
        self.bu_conv_p4 = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True),
        )
        self.bu_conv_p5 = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True),
        )

        # Learnable fusion weights – sizes match actual input count per node.
        self.w_td_p4 = nn.Parameter(torch.ones(2, dtype=torch.float32))  # p4 + up(p5)
        self.w_td_p3 = nn.Parameter(torch.ones(2, dtype=torch.float32))  # p3 + up(p4_td)
        self.w_bu_p4 = nn.Parameter(torch.ones(3, dtype=torch.float32))  # p4_td + down(p3_td) + p4_skip
        self.w_bu_p5 = nn.Parameter(torch.ones(2, dtype=torch.float32))  # p5 + down(p4_out)

    @staticmethod
    def _norm_weights(w: torch.Tensor, eps: float) -> torch.Tensor:
        w = F.relu(w)
        return w / (w.sum() + eps)

    def forward(self, features: List[torch.Tensor]) -> List[torch.Tensor]:
        p3, p4, p5 = features

        # --- Top-Down ---
        w = self._norm_weights(self.w_td_p4, self.epsilon)
        p4_td = self.td_conv_p4(
            w[0] * p4 + w[1] * F.interpolate(p5, size=p4.shape[-2:], mode="nearest")
        )
        w = self._norm_weights(self.w_td_p3, self.epsilon)
        p3_td = self.td_conv_p3(
            w[0] * p3 + w[1] * F.interpolate(p4_td, size=p3.shape[-2:], mode="nearest")
        )

        # --- Bottom-Up ---
        w = self._norm_weights(self.w_bu_p4, self.epsilon)
        p4_out = self.bu_conv_p4(
            w[0] * p4_td
            + w[1] * F.avg_pool2d(p3_td, kernel_size=2, stride=2)
            + w[2] * p4  # skip connection from the original p4
        )
        w = self._norm_weights(self.w_bu_p5, self.epsilon)
        p5_out = self.bu_conv_p5(
            w[0] * p5 + w[1] * F.avg_pool2d(p4_out, kernel_size=2, stride=2)
        )

        return [p3_td, p4_out, p5_out]


class BiFPN(nn.Module):
    """Stacked Bidirectional Feature Pyramid Network.

    Args:
        channels: Feature channel count (must match backbone/neck output).
        num_repeats: Number of BiFPN rounds (EfficientDet uses 3-7).
    """

    def __init__(self, channels: int = 128, num_repeats: int = 3):
        super().__init__()
        self.layers = nn.ModuleList(
            [BiFPNLayer(channels) for _ in range(num_repeats)]
        )

    def forward(self, features: List[torch.Tensor]) -> List[torch.Tensor]:
        for layer in self.layers:
            features = layer(features)
        return features
