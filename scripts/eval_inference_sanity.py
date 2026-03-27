from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ai_core.inference.pipeline import AquaDetPipeline


def iter_images(root: Path):
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
        yield from root.rglob(ext)


def main() -> None:
    parser = argparse.ArgumentParser(description="Quick inference sanity check on prepared dataset images")
    parser.add_argument("--images", type=Path, default=Path("datasets/processed/unified/images/val"))
    parser.add_argument("--config", type=Path, default=Path("configs/inference.yaml"))
    parser.add_argument("--weights", type=Path, default=None)
    parser.add_argument("--max-images", type=int, default=30)
    parser.add_argument("--min-avg-detections", type=float, default=0.2)
    parser.add_argument("--min-avg-confidence", type=float, default=0.2)
    parser.add_argument("--min-unique-classes", type=int, default=1)
    args = parser.parse_args()

    image_paths = sorted(iter_images(args.images))[: args.max_images]
    if not image_paths:
        raise SystemExit(f"No images found in {args.images}")

    pipeline = AquaDetPipeline(config_path=args.config, weights_path=args.weights)

    total_dets = 0
    conf_sum = 0.0
    class_counts: dict[str, int] = {}

    for idx, image_path in enumerate(image_paths):
        frame = cv2.imread(str(image_path))
        if frame is None:
            continue
        result = pipeline.infer_frame(frame, frame_index=idx)
        total_dets += len(result.detections)
        for det in result.detections:
            conf_sum += det.confidence
            class_counts[det.class_name] = class_counts.get(det.class_name, 0) + 1

    avg_det = total_dets / max(1, len(image_paths))
    avg_conf = conf_sum / max(1, total_dets)

    print(f"Images evaluated: {len(image_paths)}")
    print(f"Total detections: {total_dets}")
    print(f"Average detections/image: {avg_det:.2f}")
    print(f"Average confidence: {avg_conf:.3f}")
    print(f"By class: {class_counts}")

    unique_classes = len([c for c, n in class_counts.items() if n > 0])
    ok = True
    if avg_det < args.min_avg_detections:
        print(f"[WARN] avg_det={avg_det:.3f} < min_avg_detections={args.min_avg_detections}")
        ok = False
    if avg_conf < args.min_avg_confidence:
        print(f"[WARN] avg_conf={avg_conf:.3f} < min_avg_confidence={args.min_avg_confidence}")
        ok = False
    if unique_classes < args.min_unique_classes:
        print(f"[WARN] unique_classes={unique_classes} < min_unique_classes={args.min_unique_classes}")
        ok = False

    if ok:
        print("[OK] Sanity check passed.")
    else:
        raise SystemExit("Sanity check failed. Tune thresholds/model/dataset.")


if __name__ == "__main__":
    main()
