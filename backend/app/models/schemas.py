from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class DetectionIn(BaseModel):
    class_name: str
    confidence: float = Field(ge=0.0, le=1.0)
    bbox_xyxy: List[int]
    depth_m: float
    real_size_mm: float
    track_id: Optional[int] = None


class FrameIn(BaseModel):
    camera_id: str
    timestamp_ms: int
    detections: List[DetectionIn]


class TelemetryIn(BaseModel):
    device_id: str
    timestamp_ms: int
    latitude: float
    longitude: float
    ph: float
    turbidity_ntu: float


class SummaryOut(BaseModel):
    total_frames: int
    total_detections: int
    by_class: dict
    latest_telemetry: Optional[TelemetryIn] = None
