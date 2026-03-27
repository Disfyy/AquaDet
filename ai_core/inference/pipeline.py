from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
import yaml

from ai_core.models.hybrid_model import AquaDetHybridModel
from ai_core.utils.geometry import estimate_real_size_mm
from ai_core.utils.types import Detection, FrameResult
from .tracker import SimpleIoUTracker


CLASS_NAMES = ["plastic", "metal", "organic", "microplastic"]


class AquaDetPipeline:
    def __init__(
        self,
        focal_length_mm: float = 4.25,
        enable_pi_ge: bool = True,
        conf_threshold: float = 0.3,
        nms_iou_threshold: float = 0.5,
        max_detections: int = 50,
        config_path: Path | None = None,
        weights_path: Path | None = None,
    ):
        if config_path is not None and Path(config_path).exists():
            with Path(config_path).open("r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
            camera_cfg = cfg.get("camera", {})
            runtime_cfg = cfg.get("runtime", {})
            focal_length_mm = float(camera_cfg.get("focal_length_mm", focal_length_mm))
            enable_pi_ge = bool(runtime_cfg.get("enable_pi_ge", enable_pi_ge))
            conf_threshold = float(runtime_cfg.get("conf_threshold", conf_threshold))
            max_detections = int(runtime_cfg.get("max_detections", max_detections))

        self.model = AquaDetHybridModel(num_classes=len(CLASS_NAMES), pi_ge_enabled=enable_pi_ge)
        if weights_path is not None and Path(weights_path).exists():
            state = torch.load(weights_path, map_location="cpu")
            self.model.load_state_dict(state, strict=False)
        self.model.eval()
        self.tracker = SimpleIoUTracker(iou_threshold=0.3)
        self.focal_length_mm = focal_length_mm
        self.conf_threshold = conf_threshold
        self.nms_iou_threshold = nms_iou_threshold
        self.max_detections = max_detections

    def _preprocess(self, frame_bgr: np.ndarray) -> torch.Tensor:
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        x = frame_rgb.astype(np.float32) / 255.0
        x = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0)
        return x

    @staticmethod
    def _bbox_iou(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
        x1 = max(a[0], b[0])
        y1 = max(a[1], b[1])
        x2 = min(a[2], b[2])
        y2 = min(a[3], b[3])
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area_a = max(1, (a[2] - a[0]) * (a[3] - a[1]))
        area_b = max(1, (b[2] - b[0]) * (b[3] - b[1]))
        union = area_a + area_b - inter
        return inter / union if union > 0 else 0.0

    def _decode_predictions(
        self,
        logits_map: torch.Tensor,
        box_map: torch.Tensor,
        h: int,
        w: int,
    ) -> List[Dict[str, object]]:
        cls_scores = torch.sigmoid(logits_map[0])
        conf_map, cls_map = torch.max(cls_scores, dim=0)
        box_pred = box_map[0]

        hh, ww = conf_map.shape
        flat_scores = conf_map.reshape(-1)
        if flat_scores.numel() == 0:
            return []

        topk = min(self.max_detections * 4, flat_scores.numel())
        scores_topk, indices_topk = torch.topk(flat_scores, k=max(1, topk), largest=True, sorted=True)

        candidates: List[Dict[str, object]] = []
        for score, idx in zip(scores_topk.tolist(), indices_topk.tolist()):
            if score < self.conf_threshold:
                break

            gy = idx // ww
            gx = idx % ww

            tx, ty, tw, th = box_pred[:, gy, gx]
            cx = (gx + torch.sigmoid(tx).item()) / ww * w
            cy = (gy + torch.sigmoid(ty).item()) / hh * h
            bw = max(2.0, torch.sigmoid(tw).item() * w * 0.5)
            bh = max(2.0, torch.sigmoid(th).item() * h * 0.5)

            x1 = max(0, cx - bw // 2)
            y1 = max(0, cy - bh // 2)
            x2 = min(w - 1, cx + bw // 2)
            y2 = min(h - 1, cy + bh // 2)

            cls_idx = int(cls_map[gy, gx].item())
            candidates.append(
                {
                    "bbox": (int(x1), int(y1), int(x2), int(y2)),
                    "confidence": float(score),
                    "class_idx": cls_idx,
                }
            )

        kept: List[Dict[str, object]] = []
        for cand in candidates:
            keep = True
            for picked in kept:
                if self._bbox_iou(cand["bbox"], picked["bbox"]) > self.nms_iou_threshold:
                    keep = False
                    break
            if keep:
                kept.append(cand)
            if len(kept) >= self.max_detections:
                break

        return kept

    def infer_frame(self, frame_bgr: np.ndarray, frame_index: int = 0) -> FrameResult:
        h, w = frame_bgr.shape[:2]
        x = self._preprocess(frame_bgr)

        with torch.no_grad():
            outputs: Dict[str, torch.Tensor] = self.model(x)

        logits = outputs["logits"]
        boxes = outputs["boxes"]
        masks = outputs["masks"]
        depth_map = outputs["depth"]

        decoded = self._decode_predictions(logits, boxes, h, w)
        bboxes = [d["bbox"] for d in decoded]
        tracks = self.tracker.update(bboxes)

        mask_map = masks[0, 0].detach().cpu().numpy()
        mask_full_res = cv2.resize(mask_map, (w, h), interpolation=cv2.INTER_LINEAR)
        mask_binary = (mask_full_res >= 0.5).astype(np.uint8)

        detections: List[Detection] = []
        for track, pred in zip(tracks, decoded):
            x1, y1, x2, y2 = track.bbox
            cls_idx = int(pred["class_idx"])
            conf = float(pred["confidence"])

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
                focal_length_mm=self.focal_length_mm,
                depth_m=depth_value,
            )

            obj_mask = np.zeros_like(mask_binary)
            obj_mask[y1:y2, x1:x2] = mask_binary[y1:y2, x1:x2]

            detections.append(
                Detection(
                    class_name=CLASS_NAMES[cls_idx],
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
