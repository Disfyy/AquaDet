from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Check unified dataset readiness and quality signals")
    parser.add_argument("--dataset", type=Path, default=Path("datasets/processed/unified"))
    parser.add_argument("--min-images", type=int, default=500)
    parser.add_argument("--min-classes", type=int, default=3)
    args = parser.parse_args()

    stats_path = args.dataset / "stats.json"
    if not stats_path.exists():
        raise SystemExit(
            f"Missing {stats_path}. Run: python scripts/prepare_hybrid_dataset.py --dataset-root datasets"
        )

    with stats_path.open("r", encoding="utf-8") as f:
        stats = json.load(f)

    total = int(stats.get("total_images", 0))
    by_split = stats.get("by_split", {})
    class_hist = stats.get("class_histogram", {})
    non_empty_classes = sum(1 for _, count in class_hist.items() if int(count) > 0)

    print(f"Dataset: {args.dataset}")
    print(f"Total images: {total}")
    print(f"Split sizes: train={by_split.get('train', 0)} val={by_split.get('val', 0)} test={by_split.get('test', 0)}")
    print(f"Non-empty classes: {non_empty_classes}")

    quality_ok = True
    if total < args.min_images:
        print(f"[WARN] Total images {total} < min-images {args.min_images}")
        quality_ok = False

    if non_empty_classes < args.min_classes:
        print(f"[WARN] Non-empty classes {non_empty_classes} < min-classes {args.min_classes}")
        quality_ok = False

    if int(by_split.get("train", 0)) == 0 or int(by_split.get("val", 0)) == 0:
        print("[WARN] Train/Val split is empty")
        quality_ok = False

    if quality_ok:
        print("[OK] Dataset quality gate passed. Ready for hybrid training.")
    else:
        raise SystemExit("Dataset quality gate failed. Add more balanced labeled data.")


if __name__ == "__main__":
    main()
