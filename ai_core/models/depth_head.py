from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DepthHead(nn.Module):
    """Monocular Depth Estimation Head.
    
    Uses Atrous Spatial Pyramid Pooling (ASPP) to capture multi-scale 
    context for robust depth perception in underwater environments, 
    crucial for accurate real-world size estimation.
    """
    def __init__(self, in_channels: int = 128):
        super().__init__()
        
        # Multi-scale receptive fields to understand global vs local context
        self.branch1 = nn.Conv2d(in_channels, 32, kernel_size=1)
        self.branch2 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=6, dilation=6)
        self.branch3 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=12, dilation=12)
        
        # Global context branch
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.global_conv = nn.Conv2d(in_channels, 32, kernel_size=1)
        
        # Fusion and final depth prediction
        self.fusion = nn.Sequential(
            nn.Conv2d(32 * 4, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=1),
            nn.Softplus(),  # Ensures depth is strictly positive (> 0)
        )

    def forward(self, feature_map: torch.Tensor) -> torch.Tensor:
        h, w = feature_map.shape[2:]
        
        # Extract features at different receptive scales
        b1 = self.branch1(feature_map)
        b2 = self.branch2(feature_map)
        b3 = self.branch3(feature_map)
        
        # Global context
        bg = self.global_pool(feature_map)
        bg = self.global_conv(bg)
        bg = F.interpolate(bg, size=(h, w), mode='bilinear', align_corners=False)
        
        # Concat and fuse
        fused = torch.cat([b1, b2, b3, bg], dim=1)
        
        # Absolute depth in meters + 0.1m safety buffer
        return self.fusion(fused) + 0.1
