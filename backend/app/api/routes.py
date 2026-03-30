from __future__ import annotations

from collections import Counter, deque
from typing import Dict, List, Optional

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from backend.app.models.schemas import FrameIn, SummaryOut, TelemetryIn

router = APIRouter(prefix="/api/v1", tags=["aquadet"])

# Bounded storage to prevent OOM during long streaming sessions
_FRAMES: deque[FrameIn] = deque(maxlen=10_000)
_TELEMETRY: deque[TelemetryIn] = deque(maxlen=10_000)

# Active WebSocket connections for real-time streaming
_WS_CLIENTS: List[WebSocket] = []


@router.post("/detections")
async def ingest_detections(frame: FrameIn) -> Dict[str, int]:
    _FRAMES.append(frame)

    # Broadcast to WebSocket clients
    payload = frame.model_dump()
    disconnected: List[WebSocket] = []
    for ws in _WS_CLIENTS:
        try:
            await ws.send_json({"type": "detection", "data": payload})
        except Exception:
            disconnected.append(ws)
    for ws in disconnected:
        _WS_CLIENTS.remove(ws)

    return {
        "stored_frames": len(_FRAMES),
        "stored_detections": sum(len(f.detections) for f in _FRAMES),
    }


@router.post("/telemetry")
async def ingest_telemetry(payload: TelemetryIn) -> Dict[str, int]:
    _TELEMETRY.append(payload)

    # Broadcast to WebSocket clients
    telem_data = payload.model_dump()
    disconnected: List[WebSocket] = []
    for ws in _WS_CLIENTS:
        try:
            await ws.send_json({"type": "telemetry", "data": telem_data})
        except Exception:
            disconnected.append(ws)
    for ws in disconnected:
        _WS_CLIENTS.remove(ws)

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


@router.websocket("/ws/stream")
async def websocket_stream(websocket: WebSocket) -> None:
    """WebSocket endpoint for real-time detection and telemetry streaming.

    Clients connect here to receive live updates as JSON messages:
    - {"type": "detection", "data": {...}}
    - {"type": "telemetry", "data": {...}}
    """
    await websocket.accept()
    _WS_CLIENTS.append(websocket)
    try:
        # Keep connection open, listen for client messages (e.g., ping/pong)
        while True:
            data = await websocket.receive_text()
            # Echo back pings or ignore
            if data == "ping":
                await websocket.send_text("pong")
    except WebSocketDisconnect:
        pass
    finally:
        if websocket in _WS_CLIENTS:
            _WS_CLIENTS.remove(websocket)
