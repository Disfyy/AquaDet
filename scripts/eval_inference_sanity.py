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


def resolve_label_path(image_path: Path) -> Path:
    path_str = str(image_path)
    if "/images/" in path_str:
        return Path(path_str.replace("/images/", "/labels/")).with_suffix(".txt")
    return image_path.with_suffix(".txt")


def load_gt_boxes(image_path: Path) -> list[tuple[int, tuple[int, int, int, int]]]:
    label_path = resolve_label_path(image_path)
    boxes: list[tuple[int, tuple[int, int, int, int]]] = []
    if not label_path.exists():
        return boxes

    frame = cv2.imread(str(image_path))
    if frame is None:
        return boxes
    h, w = frame.shape[:2]

    with label_path.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            try:
                cls = int(float(parts[0]))
                cx, cy, bw, bh = map(float, parts[1:5])
            except ValueError:
                continue

            x1 = max(0, int((cx - bw / 2.0) * w))
            y1 = max(0, int((cy - bh / 2.0) * h))
            x2 = min(w - 1, int((cx + bw / 2.0) * w))
            y2 = min(h - 1, int((cy + bh / 2.0) * h))
            boxes.append((cls, (x1, y1, x2, y2)))
    return boxes


def iou_xyxy(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = max(1, (a[2] - a[0]) * (a[3] - a[1]))
    area_b = max(1, (b[2] - b[0]) * (b[3] - b[1]))
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Quick inference sanity check on prepared dataset images")
    parser.add_argument("--images", type=Path, default=Path("datasets/processed/unified/images/val"))
    parser.add_argument("--config", type=Path, default=Path("configs/inference.yaml"))
    parser.add_argument("--weights", type=Path, default=None)
    parser.add_argument("--max-images", type=int, default=30)
    parser.add_argument("--min-avg-detections", type=float, default=0.2)
    parser.add_argument("--min-avg-confidence", type=float, default=0.2)
    parser.add_argument("--min-unique-classes", type=int, default=1)
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--min-precision", type=float, default=0.05)
    parser.add_argument("--min-recall", type=float, default=0.05)
    args = parser.parse_args()

    image_paths = sorted(iter_images(args.images))[: args.max_images]
    if not image_paths:
        raise SystemExit(f"No images found in {args.images}")

    pipeline = AquaDetPipeline(config_path=args.config, weights_path=args.weights)

    total_dets = 0
    conf_sum = 0.0
    class_counts: dict[str, int] = {}
    tp = 0
    fp = 0
    fn = 0

    for idx, image_path in enumerate(image_paths):
        frame = cv2.imread(str(image_path))
        if frame is None:
            continue
        result = pipeline.infer_frame(frame, frame_index=idx)
        total_dets += len(result.detections)
        for det in result.detections:
            conf_sum += det.confidence
            class_counts[det.class_name] = class_counts.get(det.class_name, 0) + 1

        gt = load_gt_boxes(image_path)
        pred = []
        class_name_to_idx = {"plastic": 0, "metal": 1, "organic": 2, "microplastic": 3}
        for det in result.detections:
            pred.append((class_name_to_idx.get(det.class_name, -1), det.bbox_xyxy))

        matched_gt = set()
        for pcls, pb in pred:
            best_idx = -1
            best_iou = 0.0
            for gi, (gcls, gb) in enumerate(gt):
                if gi in matched_gt or gcls != pcls:
                    continue
                iou = iou_xyxy(pb, gb)
                if iou > best_iou:
                    best_iou = iou
                    best_idx = gi
            if best_idx >= 0 and best_iou >= args.iou_threshold:
                tp += 1
                matched_gt.add(best_idx)
            else:
                fp += 1
        fn += max(0, len(gt) - len(matched_gt))

    avg_det = total_dets / max(1, len(image_paths))
    avg_conf = conf_sum / max(1, total_dets)

    print(f"Images evaluated: {len(image_paths)}")
    print(f"Total detections: {total_dets}")
    print(f"Average detections/image: {avg_det:.2f}")
    print(f"Average confidence: {avg_conf:.3f}")
    print(f"By class: {class_counts}")

    precision = float(tp / max(1, tp + fp))
    recall = float(tp / max(1, tp + fn))
    f1 = float(2 * precision * recall / max(1e-8, precision + recall))
    print(f"Precision@IoU{args.iou_threshold:.2f}: {precision:.3f}")
    print(f"Recall@IoU{args.iou_threshold:.2f}: {recall:.3f}")
    print(f"F1@IoU{args.iou_threshold:.2f}: {f1:.3f}")

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
    if precision < args.min_precision:
        print(f"[WARN] precision={precision:.3f} < min_precision={args.min_precision}")
        ok = False
    if recall < args.min_recall:
        print(f"[WARN] recall={recall:.3f} < min_recall={args.min_recall}")
        ok = False

    if ok:
        print("[OK] Sanity check passed.")
    else:
        raise SystemExit("Sanity check failed. Tune thresholds/model/dataset.")


if __name__ == "__main__":
    main()
