"""Custom loss functions for AquaDet multi-task training."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Focal Loss for dense classification with class imbalance.

    Lin et al., "Focal Loss for Dense Object Detection", ICCV 2017.
    Focuses training on hard-to-classify examples by down-weighting
    well-classified ones via a modulating factor (1 - p_t)^gamma.

    Args:
        num_classes: Number of output classes.
        alpha: Per-class balancing factor. If float, used for foreground.
        gamma: Focusing parameter (default 2.0).
        ignore_index: Index to ignore in target (e.g., -100 for background).
    """

    def __init__(
        self,
        num_classes: int = 4,
        alpha: float = 0.25,
        gamma: float = 2.0,
        ignore_index: int = -100,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute focal loss.

        Args:
            logits: (B, C, H, W) raw class logits.
            targets: (B, H, W) integer class labels. Positions with
                ``ignore_index`` are excluded from the loss.
        """
        B, C, H, W = logits.shape
        # Flatten to (N, C) and (N,)
        logits_flat = logits.permute(0, 2, 3, 1).reshape(-1, C)
        targets_flat = targets.reshape(-1)

        # Mask out ignored positions
        valid = targets_flat != self.ignore_index
        if not valid.any():
            return torch.tensor(0.0, device=logits.device, dtype=logits.dtype)

        logits_valid = logits_flat[valid]
        targets_valid = targets_flat[valid]

        # Standard cross-entropy (per-element)
        ce = F.cross_entropy(logits_valid, targets_valid, reduction="none")
        p_t = torch.exp(-ce)  # probability of correct class

        # Focal modulation
        focal_weight = self.alpha * (1.0 - p_t) ** self.gamma
        loss = focal_weight * ce

        return loss.mean()


class CIoULoss(nn.Module):
    """Complete-IoU loss for bounding box regression.

    Zheng et al., "Distance-IoU Loss", AAAI 2020.
    Directly optimises the IoU metric plus distance and aspect ratio penalties.

    Expects boxes in center-format (cx, cy, w, h), normalised to [0, 1].
    """

    def __init__(self):
        super().__init__()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute CIoU loss.

        Args:
            pred: (N, 4) predicted boxes [cx, cy, w, h].
            target: (N, 4) target boxes [cx, cy, w, h].

        Returns:
            Scalar CIoU loss (1 - CIoU), averaged over N.
        """
        if pred.numel() == 0:
            return torch.tensor(0.0, device=pred.device, dtype=pred.dtype)

        # Convert center format to corner format
        pred_x1 = pred[:, 0] - pred[:, 2] / 2
        pred_y1 = pred[:, 1] - pred[:, 3] / 2
        pred_x2 = pred[:, 0] + pred[:, 2] / 2
        pred_y2 = pred[:, 1] + pred[:, 3] / 2

        tgt_x1 = target[:, 0] - target[:, 2] / 2
        tgt_y1 = target[:, 1] - target[:, 3] / 2
        tgt_x2 = target[:, 0] + target[:, 2] / 2
        tgt_y2 = target[:, 1] + target[:, 3] / 2

        # Intersection
        inter_x1 = torch.max(pred_x1, tgt_x1)
        inter_y1 = torch.max(pred_y1, tgt_y1)
        inter_x2 = torch.min(pred_x2, tgt_x2)
        inter_y2 = torch.min(pred_y2, tgt_y2)
        inter_area = (inter_x2 - inter_x1).clamp(min=0) * (inter_y2 - inter_y1).clamp(min=0)

        # Union
        pred_area = (pred_x2 - pred_x1).clamp(min=0) * (pred_y2 - pred_y1).clamp(min=0)
        tgt_area = (tgt_x2 - tgt_x1).clamp(min=0) * (tgt_y2 - tgt_y1).clamp(min=0)
        union_area = pred_area + tgt_area - inter_area + 1e-7

        iou = inter_area / union_area

        # Enclosing box
        enc_x1 = torch.min(pred_x1, tgt_x1)
        enc_y1 = torch.min(pred_y1, tgt_y1)
        enc_x2 = torch.max(pred_x2, tgt_x2)
        enc_y2 = torch.max(pred_y2, tgt_y2)

        # Distance between centres
        center_dist_sq = (pred[:, 0] - target[:, 0]) ** 2 + (pred[:, 1] - target[:, 1]) ** 2
        # Diagonal of enclosing box
        diag_sq = (enc_x2 - enc_x1) ** 2 + (enc_y2 - enc_y1) ** 2 + 1e-7

        # Aspect ratio penalty
        import math
        v = (4.0 / (math.pi ** 2)) * (
            torch.atan(target[:, 2] / (target[:, 3] + 1e-7))
            - torch.atan(pred[:, 2] / (pred[:, 3] + 1e-7))
        ) ** 2
        with torch.no_grad():
            alpha_ar = v / (1.0 - iou + v + 1e-7)

        ciou = iou - center_dist_sq / diag_sq - alpha_ar * v

        return (1.0 - ciou).mean()


class UncertaintyWeightedLoss(nn.Module):
    """Homoscedastic uncertainty-based multi-task loss weighting.

    Kendall et al., "Multi-Task Learning Using Uncertainty to Weigh
    Losses for Scene Geometry and Semantics", CVPR 2018.

    Learns a log-variance parameter per task. The effective weight for
    each loss is ``exp(-log_var)`` plus a regularisation term ``log_var``.
    """

    def __init__(self, num_tasks: int = 5):
        super().__init__()
        self.log_vars = nn.Parameter(torch.zeros(num_tasks))

    def forward(self, *losses: torch.Tensor) -> torch.Tensor:
        """Combine task losses with learned weights.

        Args:
            *losses: One scalar loss per task, in consistent order.
        """
        total = torch.tensor(0.0, device=losses[0].device, dtype=losses[0].dtype)
        for i, loss in enumerate(losses):
            precision = torch.exp(-self.log_vars[i])
            total = total + precision * loss + self.log_vars[i]
        return total
