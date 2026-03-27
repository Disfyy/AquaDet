from __future__ import annotations

import torch
import torch.nn as nn


class PIGE(nn.Module):
    """Lightweight physics-informed enhancement gate.

    If enabled, applies shallow restoration; otherwise acts as identity.
    """

    def __init__(self, channels: int = 3, enabled: bool = True):
        super().__init__()
        self.enabled = enabled
        self.restore = nn.Sequential(
            nn.Conv2d(channels, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, channels, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.enabled:
            return x
        return torch.clamp(x + self.restore(x), 0.0, 1.0)
