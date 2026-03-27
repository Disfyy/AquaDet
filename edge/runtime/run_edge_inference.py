from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from ai_core.inference.pipeline import AquaDetPipeline


def draw(frame, detections):
    for d in detections:
        x1, y1, x2, y2 = d.bbox_xyxy
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        text = f"{d.class_name} id={d.track_id} {d.real_size_mm:.1f}mm"
        cv2.putText(frame, text, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description="Run AquaDet edge inference")
    parser.add_argument("--source", type=str, default="0")
    parser.add_argument("--config", type=Path, default=Path("configs/inference.yaml"))
    parser.add_argument("--weights", type=Path, default=None)
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    source = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open source: {source}")

    pipeline = AquaDetPipeline(config_path=args.config, weights_path=args.weights)
    frame_index = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        result = pipeline.infer_frame(frame, frame_index=frame_index)
        frame = draw(frame, result.detections)
        frame_index += 1

        if args.show:
            cv2.imshow("AquaDet Edge", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
