from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple


@dataclass
class Track:
    track_id: int
    bbox: Tuple[int, int, int, int]


class SimpleIoUTracker:
    """Lightweight fallback tracker with ByteTrack-like ID persistence behavior."""

    def __init__(self, iou_threshold: float = 0.3):
        self.iou_threshold = iou_threshold
        self.next_id = 1
        self.active_tracks: Dict[int, Tuple[int, int, int, int]] = {}

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

    def update(self, bboxes: Iterable[Tuple[int, int, int, int]]) -> List[Track]:
        assigned: List[Track] = []
        for bbox in bboxes:
            best_id, best_iou = None, 0.0
            for track_id, track_bbox in self.active_tracks.items():
                score = self._iou(bbox, track_bbox)
                if score > best_iou:
                    best_id, best_iou = track_id, score

            if best_id is not None and best_iou >= self.iou_threshold:
                self.active_tracks[best_id] = bbox
                assigned.append(Track(track_id=best_id, bbox=bbox))
            else:
                new_id = self.next_id
                self.next_id += 1
                self.active_tracks[new_id] = bbox
                assigned.append(Track(track_id=new_id, bbox=bbox))

        return assigned
