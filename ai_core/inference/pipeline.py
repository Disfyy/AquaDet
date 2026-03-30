from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torchvision
import yaml

from ai_core.models.hybrid_model import AquaDetHybridModel
from ai_core.utils.geometry import estimate_real_size_mm, focal_length_mm_to_px
from ai_core.utils.types import Detection, FrameResult
from .tracker import SimpleIoUTracker


DEFAULT_CLASS_NAMES = ["plastic", "metal", "organic", "microplastic"]


class AquaDetPipeline:
    """End-to-end inference pipeline for AquaDet.

    Handles preprocessing, model inference, NMS, tracking, depth-based
    size estimation, and result packaging.
    """

    def __init__(
        self,
        focal_length_mm: float = 4.25,
        sensor_width_mm: float = 3.68,
        enable_pi_ge: bool = True,
        conf_threshold: float = 0.3,
        nms_iou_threshold: float = 0.5,
        max_detections: int = 50,
        config_path: Path | None = None,
        weights_path: Path | None = None,
        strict_weights: bool = True,
        model_size: int = 640,
        class_names: List[str] | None = None,
    ):
        self.model_size = model_size
        self.sensor_width_mm = sensor_width_mm

        if config_path is not None and Path(config_path).exists():
            with Path(config_path).open("r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
            camera_cfg = cfg.get("camera", {})
            runtime_cfg = cfg.get("runtime", {})
            focal_length_mm = float(camera_cfg.get("focal_length_mm", focal_length_mm))
            self.sensor_width_mm = float(camera_cfg.get("sensor_width_mm", sensor_width_mm))
            enable_pi_ge = bool(runtime_cfg.get("enable_pi_ge", enable_pi_ge))
            conf_threshold = float(runtime_cfg.get("conf_threshold", conf_threshold))
            max_detections = int(runtime_cfg.get("max_detections", max_detections))

            # Load class names from config if available
            if "classes" in cfg:
                class_names = cfg["classes"]

        self.class_names = class_names or DEFAULT_CLASS_NAMES
        num_classes = len(self.class_names)

        self.model = AquaDetHybridModel(num_classes=num_classes, pi_ge_enabled=enable_pi_ge)
        if weights_path is not None and Path(weights_path).exists():
            state = torch.load(weights_path, map_location="cpu")
            self.model.load_state_dict(state, strict=strict_weights)
        elif weights_path is not None:
            raise FileNotFoundError(f"weights not found: {weights_path}")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
        self.tracker = SimpleIoUTracker(iou_threshold=0.3, max_missed=10)
        self.focal_length_mm = focal_length_mm
        self.conf_threshold = conf_threshold
        self.nms_iou_threshold = nms_iou_threshold
        self.max_detections = max_detections

    def _preprocess(self, frame_bgr: np.ndarray) -> Tuple[torch.Tensor, Tuple[int, int]]:
        h, w = frame_bgr.shape[:2]
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, (self.model_size, self.model_size))
        x = frame_resized.astype(np.float32) / 255.0
        x = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0)
        return x.to(self.device), (h, w)

    def _decode_predictions(
        self,
        logits_map: torch.Tensor,
        box_map: torch.Tensor,
        obj_map: torch.Tensor,
        h: int,
        w: int,
    ) -> List[Dict[str, object]]:
        """Decode grid-based predictions into bounding boxes with GPU-accelerated NMS."""
        # Combined confidence = sigmoid(obj) * max(sigmoid(cls))
        obj_scores = torch.sigmoid(obj_map[0, 0])  # (H, W)
        cls_scores = torch.sigmoid(logits_map[0])   # (C, H, W)
        cls_max_scores, cls_indices = torch.max(cls_scores, dim=0)  # (H, W)
        conf_map = obj_scores * cls_max_scores  # (H, W)

        box_pred = box_map[0]  # (4, H, W)
        hh, ww = conf_map.shape

        # Flatten and filter by confidence
        flat_scores = conf_map.reshape(-1)
        mask = flat_scores >= self.conf_threshold
        if not mask.any():
            return []

        indices = mask.nonzero(as_tuple=True)[0]
        scores = flat_scores[indices]

        # Limit candidates
        if scores.numel() > self.max_detections * 4:
            topk = min(self.max_detections * 4, scores.numel())
            scores, top_indices = torch.topk(scores, k=topk, largest=True, sorted=True)
            indices = indices[top_indices]

        # Decode boxes
        gy = indices // ww
        gx = indices % ww

        tx = box_pred[0][gy, gx]
        ty = box_pred[1][gy, gx]
        tw = box_pred[2][gy, gx]
        th = box_pred[3][gy, gx]

        cx = (gx.float() + torch.sigmoid(tx)) / ww * w
        cy = (gy.float() + torch.sigmoid(ty)) / hh * h
        bw = torch.clamp(torch.sigmoid(tw) * w * 0.5, min=2.0)
        bh = torch.clamp(torch.sigmoid(th) * h * 0.5, min=2.0)

        x1 = torch.clamp(cx - bw / 2, min=0)
        y1 = torch.clamp(cy - bh / 2, min=0)
        x2 = torch.clamp(cx + bw / 2, max=w - 1)
        y2 = torch.clamp(cy + bh / 2, max=h - 1)

        boxes_xyxy = torch.stack([x1, y1, x2, y2], dim=1)  # (N, 4)
        cls_ids = cls_indices[gy, gx]

        # GPU-accelerated NMS (class-agnostic first to prevent cross-class duplicates)
        keep = torchvision.ops.nms(boxes_xyxy, scores, self.nms_iou_threshold)
        keep = keep[:self.max_detections]

        results: List[Dict[str, object]] = []
        for idx in keep.tolist():
            results.append({
                "bbox": (
                    int(boxes_xyxy[idx, 0].item()),
                    int(boxes_xyxy[idx, 1].item()),
                    int(boxes_xyxy[idx, 2].item()),
                    int(boxes_xyxy[idx, 3].item()),
                ),
                "confidence": float(scores[idx].item()),
                "class_idx": int(cls_ids[idx].item()),
            })

        return results

    def infer_frame(self, frame_bgr: np.ndarray, frame_index: int = 0) -> FrameResult:
        x, (h, w) = self._preprocess(frame_bgr)

        with torch.no_grad():
            outputs: Dict[str, torch.Tensor] = self.model(x)
            outputs = {k: v.detach().cpu() for k, v in outputs.items()}

        logits = outputs["logits"]
        boxes = outputs["boxes"]
        masks = outputs["masks"]
        obj = outputs["obj"]
        depth_map = outputs["depth"]

        decoded = self._decode_predictions(logits, boxes, obj, h, w)
        bboxes = [d["bbox"] for d in decoded]
        tracks = self.tracker.update(bboxes)

        mask_map = torch.sigmoid(masks[0, 0]).numpy()
        mask_full_res = cv2.resize(mask_map, (w, h), interpolation=cv2.INTER_LINEAR)
        mask_binary = (mask_full_res >= 0.5).astype(np.uint8)

        # Compute focal length in pixels for this frame size
        focal_length_px = focal_length_mm_to_px(
            self.focal_length_mm, w, self.sensor_width_mm,
        )

        detections: List[Detection] = []
        for track, pred in zip(tracks, decoded):
            x1, y1, x2, y2 = track.bbox
            cls_idx = int(pred["class_idx"])
            conf = float(pred["confidence"])

            # Clamp class index to valid range
            cls_idx = min(cls_idx, len(self.class_names) - 1)

            depth_crop = depth_map[0, 0]
            dh, dw = depth_crop.shape
            sx1 = min(dw - 1, max(0, int(x1 / max(1, w) * dw)))
            sy1 = min(dh - 1, max(0, int(y1 / max(1, h) * dh)))
            sx2 = min(dw, max(sx1 + 1, int(x2 / max(1, w) * dw)))
            sy2 = min(dh, max(sy1 + 1, int(y2 / max(1, h) * dh)))
            depth_value = float(depth_crop[sy1:sy2, sx1:sx2].mean().item())
            pixel_size = float(max(x2 - x1, y2 - y1))
            real_size_mm = estimate_real_size_mm(
                pixel_size=pixel_size,
                focal_length_px=focal_length_px,
                depth_m=depth_value,
            )

            obj_mask = np.zeros_like(mask_binary)
            obj_mask[y1:y2, x1:x2] = mask_binary[y1:y2, x1:x2]

            detections.append(
                Detection(
                    class_name=self.class_names[cls_idx],
                    confidence=conf,
                    bbox_xyxy=(x1, y1, x2, y2),
                    mask=obj_mask,
                    depth_m=depth_value,
                    real_size_mm=real_size_mm,
                    track_id=track.track_id,
                )
            )

        return FrameResult(
            detections=detections,
            frame_index=frame_index,
            timestamp_ms=int(time.time() * 1000),
        )
