from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from ai_core.inference.pipeline import AquaDetPipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Run AquaDet inference demo")
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
        for det in result.detections:
            x1, y1, x2, y2 = det.bbox_xyxy
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 140, 0), 2)
            label = f"{det.class_name} id={det.track_id} {det.real_size_mm:.1f}mm"
            cv2.putText(frame, label, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 140, 0), 1)

        frame_index += 1
        if args.show:
            cv2.imshow("AquaDet Demo", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
