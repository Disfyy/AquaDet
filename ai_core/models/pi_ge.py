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
            nn.Sigmoid()  # t(x) converges to (0, 1) bounds
        )

        # Network to estimate Global Background Light A
        # Learns the "color" and density of underwater turbidity per batch
        self.a_estimator = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(channels, 16, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, channels, kernel_size=1),
            nn.Sigmoid()  # A converges to (0, 1) bounds
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.enabled:
            return x
        
        # 1. Estimate global backscatter light A dynamically from the frame
        a_light = self.a_estimator(x)  # Shape: (B, C, 1, 1)

        # 2. Estimate spatial transmission map (t)
        # Bounding limits division by zero and simulates min visibility limit
        t_map = self.t_estimator(x)    # Shape: (B, 1, H, W)
        t_map = torch.clamp(t_map, min=0.1, max=1.0)

        # 3. Restore the clean image via inverted physical formulation:
        # J(x) = (I(x) - A * (1 - t(x))) / t(x)
        restored_j = (x - a_light * (1.0 - t_map)) / t_map

        # 4. Return safely bounded standardized tensor
        return torch.clamp(restored_j, 0.0, 1.0)
