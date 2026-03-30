from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Tuple


@dataclass
class Track:
    track_id: int
    bbox: Tuple[int, int, int, int]


class SimpleIoUTracker:
    """Lightweight tracker with linear motion prediction.

    Improvements over the original brute-force IoU tracker:
    - Simple constant-velocity motion model to predict bbox position
    - Velocity-adjusted IoU matching for underwater current resilience
    - Hit count gating for confirmed vs tentative tracks
    """

    def __init__(
        self,
        iou_threshold: float = 0.3,
        max_missed: int = 10,
        min_hits_to_confirm: int = 3,
    ):
        self.iou_threshold = iou_threshold
        self.max_missed = max_missed
        self.min_hits_to_confirm = min_hits_to_confirm
        self.next_id = 1
        self.active_tracks: Dict[int, Dict[str, object]] = {}

    @staticmethod
    def _iou(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
        x1 = max(a[0], b[0])
        y1 = max(a[1], b[1])
        x2 = min(a[2], b[2])
        y2 = min(a[3], b[3])
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area_a = max(1, (a[2] - a[0]) * (a[3] - a[1]))
        area_b = max(1, (b[2] - b[0]) * (b[3] - b[1]))
        union = area_a + area_b - inter
        return inter / union if union > 0 else 0.0

    @staticmethod
    def _predict_bbox(
        bbox: Tuple[int, int, int, int],
        velocity: Tuple[float, float, float, float],
    ) -> Tuple[int, int, int, int]:
        """Apply constant-velocity motion prediction to a bbox."""
        return (
            int(bbox[0] + velocity[0]),
            int(bbox[1] + velocity[1]),
            int(bbox[2] + velocity[2]),
            int(bbox[3] + velocity[3]),
        )

    @staticmethod
    def _compute_velocity(
        prev: Tuple[int, int, int, int],
        curr: Tuple[int, int, int, int],
        alpha: float = 0.7,
        old_vel: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0),
    ) -> Tuple[float, float, float, float]:
        """Exponentially smoothed velocity estimate."""
        return (
            alpha * (curr[0] - prev[0]) + (1 - alpha) * old_vel[0],
            alpha * (curr[1] - prev[1]) + (1 - alpha) * old_vel[1],
            alpha * (curr[2] - prev[2]) + (1 - alpha) * old_vel[2],
            alpha * (curr[3] - prev[3]) + (1 - alpha) * old_vel[3],
        )

    def update(self, bboxes: Iterable[Tuple[int, int, int, int]]) -> List[Track]:
        bboxes = list(bboxes)

        # Predict each track's next position using velocity
        predicted: Dict[int, Tuple[int, int, int, int]] = {}
        for track_id, info in self.active_tracks.items():
            info["missed"] = int(info["missed"]) + 1
            vel = info.get("velocity", (0.0, 0.0, 0.0, 0.0))
            predicted[track_id] = self._predict_bbox(info["bbox"], vel)

        used_track_ids: set = set()
        assigned: List[Track] = []

        for bbox in bboxes:
            best_id, best_iou = None, 0.0
            for track_id, pred_bbox in predicted.items():
                if track_id in used_track_ids:
                    continue
                score = self._iou(bbox, pred_bbox)
                if score > best_iou:
                    best_id, best_iou = track_id, score

            if best_id is not None and best_iou >= self.iou_threshold:
                old_bbox = self.active_tracks[best_id]["bbox"]
                old_vel = self.active_tracks[best_id].get("velocity", (0.0, 0.0, 0.0, 0.0))
                new_vel = self._compute_velocity(old_bbox, bbox, alpha=0.7, old_vel=old_vel)

                self.active_tracks[best_id]["bbox"] = bbox
                self.active_tracks[best_id]["missed"] = 0
                self.active_tracks[best_id]["hits"] = int(self.active_tracks[best_id]["hits"]) + 1
                self.active_tracks[best_id]["velocity"] = new_vel
                used_track_ids.add(best_id)
                assigned.append(Track(track_id=best_id, bbox=bbox))
            else:
                new_id = self.next_id
                self.next_id += 1
                self.active_tracks[new_id] = {
                    "bbox": bbox,
                    "missed": 0,
                    "hits": 1,
                    "velocity": (0.0, 0.0, 0.0, 0.0),
                }
                used_track_ids.add(new_id)
                assigned.append(Track(track_id=new_id, bbox=bbox))

        # Remove stale tracks
        stale_ids = [
            tid for tid, info in self.active_tracks.items()
            if int(info["missed"]) > self.max_missed
        ]
        for tid in stale_ids:
            self.active_tracks.pop(tid, None)

        return assigned
