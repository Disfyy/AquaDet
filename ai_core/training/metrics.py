"""COCO-style mAP evaluation for AquaDet."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import torch


@dataclass
class APResult:
    """Holds AP evaluation results."""
    ap_per_class: Dict[int, float] = field(default_factory=dict)
    precision_per_class: Dict[int, float] = field(default_factory=dict)
    recall_per_class: Dict[int, float] = field(default_factory=dict)
    mAP: float = 0.0
    mAP_50: float = 0.0
    mAP_75: float = 0.0


def _box_iou(
    pred: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """Compute IoU between two sets of boxes in xyxy format.

    Args:
        pred: (N, 4) predicted boxes.
        target: (M, 4) target boxes.

    Returns:
        (N, M) IoU matrix.
    """
    x1 = torch.max(pred[:, 0].unsqueeze(1), target[:, 0].unsqueeze(0))
    y1 = torch.max(pred[:, 1].unsqueeze(1), target[:, 1].unsqueeze(0))
    x2 = torch.min(pred[:, 2].unsqueeze(1), target[:, 2].unsqueeze(0))
    y2 = torch.min(pred[:, 3].unsqueeze(1), target[:, 3].unsqueeze(0))

    inter = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)

    area_p = (pred[:, 2] - pred[:, 0]).clamp(min=0) * (pred[:, 3] - pred[:, 1]).clamp(min=0)
    area_t = (target[:, 2] - target[:, 0]).clamp(min=0) * (target[:, 3] - target[:, 1]).clamp(min=0)

    union = area_p.unsqueeze(1) + area_t.unsqueeze(0) - inter + 1e-7
    return inter / union


def _ap_from_pr(precisions: List[float], recalls: List[float]) -> float:
    """Compute AP from precision-recall curve using 101-point interpolation."""
    import numpy as np
    if not precisions or not recalls:
        return 0.0

    prec = np.array(precisions, dtype=np.float64)
    rec = np.array(recalls, dtype=np.float64)

    # Sort by recall
    order = np.argsort(rec)
    rec = rec[order]
    prec = prec[order]

    # Prepend / append sentinel values
    rec = np.concatenate(([0.0], rec, [1.0]))
    prec = np.concatenate(([1.0], prec, [0.0]))

    # Make precision monotonically decreasing
    for i in range(len(prec) - 2, -1, -1):
        prec[i] = max(prec[i], prec[i + 1])

    # 101-point interpolation
    recall_levels = np.linspace(0.0, 1.0, 101)
    ap = 0.0
    for r in recall_levels:
        idx = np.searchsorted(rec, r, side="left")
        if idx < len(prec):
            ap += prec[idx]
    return float(ap / 101.0)


def compute_ap_at_iou(
    all_preds: List[List[Tuple[int, float, Tuple[float, float, float, float]]]],
    all_targets: List[List[Tuple[int, Tuple[float, float, float, float]]]],
    iou_threshold: float = 0.5,
    num_classes: int = 4,
) -> Dict[int, float]:
    """Compute per-class AP at a single IoU threshold.

    Args:
        all_preds: Per-image list of (class_idx, confidence, (x1, y1, x2, y2)).
        all_targets: Per-image list of (class_idx, (x1, y1, x2, y2)).
        iou_threshold: IoU threshold for a true positive match.
        num_classes: Total number of classes.

    Returns:
        Dict mapping class_idx → AP.
    """
    ap_per_class: Dict[int, float] = {}

    for cls in range(num_classes):
        # Gather all predictions for this class across all images
        dets: List[Tuple[int, float, Tuple[float, float, float, float]]] = []  # (img_idx, conf, bbox)
        n_gt = 0

        for img_idx, (preds, targets) in enumerate(zip(all_preds, all_targets)):
            for p_cls, p_conf, p_bbox in preds:
                if p_cls == cls:
                    dets.append((img_idx, p_conf, p_bbox))
            for t_cls, _ in targets:
                if t_cls == cls:
                    n_gt += 1

        if n_gt == 0:
            ap_per_class[cls] = 0.0
            continue

        # Sort by confidence (descending)
        dets.sort(key=lambda x: x[1], reverse=True)

        # Track which ground truths have been matched
        matched: Dict[int, set] = {}  # img_idx → set of matched gt indices
        precisions: List[float] = []
        recalls: List[float] = []
        tp_cum = 0
        fp_cum = 0

        for img_idx, conf, p_bbox in dets:
            p_box = torch.tensor([p_bbox], dtype=torch.float32)

            # Get ground truths for this image and class
            gt_boxes = []
            gt_indices = []
            for gi, (t_cls, t_bbox) in enumerate(all_targets[img_idx]):
                if t_cls == cls:
                    gt_boxes.append(t_bbox)
                    gt_indices.append(gi)

            if not gt_boxes:
                fp_cum += 1
            else:
                gt_tensor = torch.tensor(gt_boxes, dtype=torch.float32)
                ious = _box_iou(p_box, gt_tensor)[0]

                best_iou, best_idx = ious.max(0)
                best_gt_idx = gt_indices[best_idx.item()]

                if img_idx not in matched:
                    matched[img_idx] = set()

                if best_iou.item() >= iou_threshold and best_gt_idx not in matched[img_idx]:
                    tp_cum += 1
                    matched[img_idx].add(best_gt_idx)
                else:
                    fp_cum += 1

            precisions.append(tp_cum / (tp_cum + fp_cum))
            recalls.append(tp_cum / n_gt)

        ap_per_class[cls] = _ap_from_pr(precisions, recalls)

    return ap_per_class


def compute_map(
    all_preds: List[List[Tuple[int, float, Tuple[float, float, float, float]]]],
    all_targets: List[List[Tuple[int, Tuple[float, float, float, float]]]],
    num_classes: int = 4,
) -> APResult:
    """Compute mAP@50, mAP@75, and mAP@[50:95:5].

    Args:
        all_preds: Per-image predictions.
        all_targets: Per-image ground truth.
        num_classes: Number of classes.

    Returns:
        APResult with per-class AP and aggregated mAP values.
    """
    iou_thresholds_50_95 = [0.5 + 0.05 * i for i in range(10)]

    ap_50 = compute_ap_at_iou(all_preds, all_targets, iou_threshold=0.5, num_classes=num_classes)
    ap_75 = compute_ap_at_iou(all_preds, all_targets, iou_threshold=0.75, num_classes=num_classes)

    # mAP@[50:95]
    all_aps: Dict[int, List[float]] = {c: [] for c in range(num_classes)}
    for iou_thresh in iou_thresholds_50_95:
        ap_at_t = compute_ap_at_iou(all_preds, all_targets, iou_threshold=iou_thresh, num_classes=num_classes)
        for c in range(num_classes):
            all_aps[c].append(ap_at_t.get(c, 0.0))

    ap_per_class = {c: sum(vals) / max(1, len(vals)) for c, vals in all_aps.items()}
    valid_aps = [v for v in ap_per_class.values() if v > 0 or True]
    mAP = sum(valid_aps) / max(1, len(valid_aps))

    valid_50 = list(ap_50.values())
    valid_75 = list(ap_75.values())

    return APResult(
        ap_per_class=ap_per_class,
        mAP=mAP,
        mAP_50=sum(valid_50) / max(1, len(valid_50)),
        mAP_75=sum(valid_75) / max(1, len(valid_75)),
    )
