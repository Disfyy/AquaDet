from __future__ import annotations

from collections import Counter
from typing import Dict, List, Optional

from fastapi import APIRouter

from backend.app.models.schemas import FrameIn, SummaryOut, TelemetryIn

router = APIRouter(prefix="/api/v1", tags=["aquadet"])

_FRAMES: List[FrameIn] = []
_TELEMETRY: List[TelemetryIn] = []


@router.post("/detections")
def ingest_detections(frame: FrameIn) -> Dict[str, int]:
    _FRAMES.append(frame)
    return {"stored_frames": len(_FRAMES), "stored_detections": sum(len(f.detections) for f in _FRAMES)}


@router.post("/telemetry")
def ingest_telemetry(payload: TelemetryIn) -> Dict[str, int]:
    _TELEMETRY.append(payload)
    return {"stored_telemetry": len(_TELEMETRY)}


@router.get("/summary", response_model=SummaryOut)
def summary() -> SummaryOut:
    classes = Counter()
    for frame in _FRAMES:
        classes.update(det.class_name for det in frame.detections)

    latest: Optional[TelemetryIn] = _TELEMETRY[-1] if _TELEMETRY else None

    return SummaryOut(
        total_frames=len(_FRAMES),
        total_detections=sum(len(f.detections) for f in _FRAMES),
        by_class=dict(classes),
        latest_telemetry=latest,
    )
