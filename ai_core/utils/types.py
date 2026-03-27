from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass
class Detection:
    class_name: str
    confidence: float
    bbox_xyxy: Tuple[int, int, int, int]
    mask: Optional[object]
    depth_m: float
    real_size_mm: float
    track_id: Optional[int] = None


@dataclass
class FrameResult:
    detections: List[Detection]
    frame_index: int
    timestamp_ms: int
