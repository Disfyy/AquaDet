from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Phase-1 baseline (YOLO).")
    parser.add_argument("--data", type=Path, default=Path("configs/dataset_baseline.yaml"))
    parser.add_argument("--model", type=str, default="yolov8n-seg.pt")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=960)
    args = parser.parse_args()

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit("Install training deps: pip install -r requirements-train.txt") from exc

    model = YOLO(args.model)
    model.train(data=str(args.data), epochs=args.epochs, imgsz=args.imgsz, project="artifacts", name="phase1_baseline")


if __name__ == "__main__":
    main()
