from __future__ import annotations

import torch
import torch.nn as nn


class PIGE(nn.Module):
    """Physical-Image-Geometry Enhancement (PI-GE) Module.

    Implements the underwater optical scattering model:
        I(x) = J(x) * t(x) + A * (1 - t(x))
    where:
        I(x) is the input degraded image
        J(x) is the restored clear image
        t(x) is the transmission map (depth-dependent scattering)
        A is the global background light (backscatter/turbidity)

    Uses a residual connection so the module can learn identity when
    restoration would hurt downstream performance.
    """

    def __init__(self, channels: int = 3, enabled: bool = True):
        super().__init__()
        self.enabled = enabled

        # Network to estimate Transmission Map t(x)
        self.t_estimator = nn.Sequential(
            nn.Conv2d(channels, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=3, padding=1),
            nn.Sigmoid(),  # t(x) in (0, 1)
        )

        # Network to estimate Global Background Light A
        self.a_estimator = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(channels, 16, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, channels, kernel_size=1),
            nn.Sigmoid(),  # A in (0, 1)
        )

        # Learnable gate for residual blending: output = gate * restored + (1 - gate) * input
        # Initialized at 0.5 so the module starts as a 50/50 blend.
        self.gate = nn.Parameter(torch.tensor(0.5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.enabled:
            return x

        # 1. Estimate global backscatter light A
        a_light = self.a_estimator(x)  # (B, C, 1, 1)

        # 2. Estimate spatial transmission map t(x)
        t_map = self.t_estimator(x)    # (B, 1, H, W)
        t_map = torch.clamp(t_map, min=0.1, max=1.0)

        # 3. Restore clean image: J(x) = (I(x) - A * (1 - t(x))) / t(x)
        restored = (x - a_light * (1.0 - t_map)) / t_map

        # 4. Clamp to valid range (safe after division by t >= 0.1)
        restored = restored.clamp(0.0, 1.0)

        # 5. Gated residual: let the network learn how much to restore
        alpha = torch.sigmoid(self.gate)
        return alpha * restored + (1.0 - alpha) * x
