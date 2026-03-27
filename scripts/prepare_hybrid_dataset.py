from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

import cv2
import numpy as np

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


@dataclass(frozen=True)
class Sample:
    image: Path
    label: Path
    source: str


def iter_images(root: Path, skip_root: Path) -> Iterable[Path]:
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTS:
            continue
        if skip_root in path.parents:
            continue
        yield path


def resolve_label(image_path: Path) -> Path | None:
    candidates = [image_path.with_suffix(".txt")]

    path_str = str(image_path)
    if "/images/" in path_str:
        candidates.append(Path(path_str.replace("/images/", "/labels/")).with_suffix(".txt"))
    if "\\images\\" in path_str:
        candidates.append(Path(path_str.replace("\\images\\", "\\labels\\")).with_suffix(".txt"))

    candidates.append(image_path.parent / "labels" / f"{image_path.stem}.txt")
    candidates.append(image_path.parent.parent / "labels" / f"{image_path.stem}.txt")

    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def source_name(root: Path, image_path: Path) -> str:
    rel = image_path.relative_to(root)
    if not rel.parts:
        return "unknown"
    first = rel.parts[0]
    if first in {"raw", "external", "processed"} and len(rel.parts) > 1:
        return rel.parts[1]
    return first


def build_samples(dataset_root: Path, unified_root: Path) -> List[Sample]:
    samples: List[Sample] = []
    for image_path in iter_images(dataset_root, unified_root):
        label_path = resolve_label(image_path)
        if label_path is None:
            continue
        samples.append(Sample(image=image_path, label=label_path, source=source_name(dataset_root, image_path)))
    return samples


def ensure_dirs(out_root: Path) -> None:
    for split in ("train", "val", "test"):
        (out_root / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_root / "labels" / split).mkdir(parents=True, exist_ok=True)


def clear_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)


def link_or_copy(src: Path, dst: Path, copy_mode: bool) -> None:
    if dst.exists():
        dst.unlink()
    if copy_mode:
        shutil.copy2(src, dst)
    else:
        dst.symlink_to(src.resolve())


def split_samples(samples: List[Sample], train_ratio: float, val_ratio: float, seed: int) -> dict[str, List[Sample]]:
    random.Random(seed).shuffle(samples)
    n = len(samples)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    n_train = min(n_train, n)
    n_val = min(n_val, max(0, n - n_train))
    return {
        "train": samples[:n_train],
        "val": samples[n_train : n_train + n_val],
        "test": samples[n_train + n_val :],
    }


def write_split(out_root: Path, split: str, samples: List[Sample], copy_mode: bool) -> None:
    for idx, sample in enumerate(samples):
        name = f"{sample.source}_{idx:06d}"
        image_dst = out_root / "images" / split / f"{name}{sample.image.suffix.lower()}"
        label_dst = out_root / "labels" / split / f"{name}.txt"
        link_or_copy(sample.image, image_dst, copy_mode=copy_mode)
        link_or_copy(sample.label, label_dst, copy_mode=copy_mode)


def class_histogram(samples: List[Sample]) -> dict[int, int]:
    histogram: dict[int, int] = {}
    for sample in samples:
        try:
            with sample.label.open("r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split()
                    if not parts:
                        continue
                    cls = int(float(parts[0]))
                    histogram[cls] = histogram.get(cls, 0) + 1
        except (OSError, ValueError):
            continue
    return dict(sorted(histogram.items(), key=lambda x: x[0]))


def write_manifest(out_root: Path, split_map: dict[str, List[Sample]]) -> None:
    manifest_path = out_root / "manifest.csv"
    with manifest_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["split", "source", "image", "label"])
        for split, samples in split_map.items():
            for sample in samples:
                writer.writerow([split, sample.source, str(sample.image), str(sample.label)])


def write_stats(out_root: Path, split_map: dict[str, List[Sample]]) -> None:
    by_split = {split: len(samples) for split, samples in split_map.items()}
    all_samples = [s for items in split_map.values() for s in items]
    by_source: dict[str, int] = {}
    for sample in all_samples:
        by_source[sample.source] = by_source.get(sample.source, 0) + 1

    payload = {
        "total_images": len(all_samples),
        "by_split": by_split,
        "by_source": dict(sorted(by_source.items(), key=lambda x: x[0])),
        "class_histogram": class_histogram(all_samples),
    }
    with (out_root / "stats.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def bootstrap_synthetic_data(dataset_root: Path, count: int, image_size: int, seed: int) -> None:
    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)

    image_dir = dataset_root / "raw" / "bootstrap" / "images"
    label_dir = dataset_root / "raw" / "bootstrap" / "labels"
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)

    for idx in range(count):
        canvas = np.zeros((image_size, image_size, 3), dtype=np.uint8)
        canvas[..., 2] = np_rng.integers(20, 60)
        canvas[..., 1] = np_rng.integers(60, 120)
        canvas[..., 0] = np_rng.integers(80, 160)

        cls = idx % 4
        w = rng.randint(image_size // 12, image_size // 4)
        h = rng.randint(image_size // 12, image_size // 4)
        cx = rng.randint(w // 2 + 2, image_size - w // 2 - 2)
        cy = rng.randint(h // 2 + 2, image_size - h // 2 - 2)
        x1, y1 = cx - w // 2, cy - h // 2
        x2, y2 = cx + w // 2, cy + h // 2

        colors = [(0, 120, 255), (190, 190, 190), (60, 180, 75), (0, 255, 255)]
        cv2.rectangle(canvas, (x1, y1), (x2, y2), colors[cls], thickness=-1)

        image_path = image_dir / f"bootstrap_{idx:05d}.jpg"
        label_path = label_dir / f"bootstrap_{idx:05d}.txt"
        cv2.imwrite(str(image_path), canvas)

        x_c = cx / image_size
        y_c = cy / image_size
        w_n = w / image_size
        h_n = h / image_size
        with label_path.open("w", encoding="utf-8") as f:
            f.write(f"{cls} {x_c:.6f} {y_c:.6f} {w_n:.6f} {h_n:.6f}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare unified AquaDet dataset from all local dataset folders")
    parser.add_argument("--dataset-root", type=Path, default=Path("datasets"))
    parser.add_argument("--out", type=Path, default=Path("datasets/processed/unified"))
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--copy", action="store_true", help="Copy files instead of symlink")
    parser.add_argument("--bootstrap-synthetic-count", type=int, default=0)
    parser.add_argument("--bootstrap-image-size", type=int, default=640)
    args = parser.parse_args()

    if not 0.0 < args.train_ratio < 1.0:
        raise SystemExit("--train-ratio must be between 0 and 1")
    if not 0.0 <= args.val_ratio < 1.0:
        raise SystemExit("--val-ratio must be between 0 and 1")
    if args.train_ratio + args.val_ratio >= 1.0:
        raise SystemExit("train_ratio + val_ratio must be < 1.0")

    dataset_root = args.dataset_root.resolve()
    out_root = args.out.resolve()

    clear_dir(out_root)
    ensure_dirs(out_root)

    samples = build_samples(dataset_root, out_root)
    if not samples:
        if args.bootstrap_synthetic_count > 0:
            bootstrap_synthetic_data(
                dataset_root=dataset_root,
                count=args.bootstrap_synthetic_count,
                image_size=args.bootstrap_image_size,
                seed=args.seed,
            )
            samples = build_samples(dataset_root, out_root)

        if not samples:
            raise SystemExit(
                "No labeled samples found. Put images+YOLO labels under datasets/raw, datasets/external or datasets/processed"
            )

    split_map = split_samples(samples, train_ratio=args.train_ratio, val_ratio=args.val_ratio, seed=args.seed)
    for split, split_samples_list in split_map.items():
        write_split(out_root, split, split_samples_list, copy_mode=args.copy)

    write_manifest(out_root, split_map)
    write_stats(out_root, split_map)

    print(f"Prepared unified dataset: {out_root}")
    print(f"Total images: {len(samples)}")
    print(
        f"Train/Val/Test: {len(split_map['train'])}/{len(split_map['val'])}/{len(split_map['test'])}"
    )


if __name__ == "__main__":
    main()
