from __future__ import annotations

from typing import List

import torch
import torch.nn as nn


class BiFPN(nn.Module):
    def __init__(self, channels: int = 128):
        super().__init__()
        self.top_down = nn.ModuleList([nn.Conv2d(channels, channels, 3, padding=1) for _ in range(2)])
        self.bottom_up = nn.ModuleList([nn.Conv2d(channels, channels, 3, padding=1) for _ in range(2)])

    def forward(self, features: List[torch.Tensor]) -> List[torch.Tensor]:
        p3, p4, p5 = features

        p4_td = self.top_down[0](p4 + nn.functional.interpolate(p5, size=p4.shape[-2:], mode="nearest"))
        p3_td = self.top_down[1](p3 + nn.functional.interpolate(p4_td, size=p3.shape[-2:], mode="nearest"))

        p4_out = self.bottom_up[0](p4_td + nn.functional.avg_pool2d(p3_td, kernel_size=2, stride=2))
        p5_out = self.bottom_up[1](p5 + nn.functional.avg_pool2d(p4_out, kernel_size=2, stride=2))

        return [p3_td, p4_out, p5_out]
