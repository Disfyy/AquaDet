from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path

import yaml


DEFAULT_INPUT_DATASET = "/kaggle/input/datasets/disfyy/aquadet-unified"
DEFAULT_WORKING_ROOT = "/kaggle/working/AquaDet"


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def save_yaml(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False)


def ensure_path_exists(path: Path, label: str) -> None:
    if not path.exists():
        raise SystemExit(f"Missing {label}: {path}")


def run_cmd(cmd: list[str], cwd: Path) -> None:
    print("[RUN]", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(cwd), check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Kaggle runner for AquaDet hybrid training")
    parser.add_argument("--project-root", type=Path, default=Path(DEFAULT_WORKING_ROOT))
    parser.add_argument("--input-dataset", type=Path, default=Path(DEFAULT_INPUT_DATASET))
    parser.add_argument("--base-config", type=Path, default=Path("configs/hybrid_train.yaml"))
    parser.add_argument("--kaggle-config", type=Path, default=Path("configs/hybrid_train_kaggle.yaml"))
    parser.add_argument("--out", type=Path, default=Path("/kaggle/working/artifacts/hybrid_model.pt"))
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--imgsz", type=int, default=None)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--log-interval", type=int, default=50)
    parser.add_argument("--save-every", type=int, default=1)
    parser.add_argument("--run-quality-check", action="store_true")
    args = parser.parse_args()

    project_root = args.project_root.resolve()
    input_root = args.input_dataset.resolve()

    # The dataset zip might extract into a nested 'unified' folder
    if not (input_root / "images" / "train").exists() and (input_root / "unified" / "images" / "train").exists():
        input_root = input_root / "unified"

    train_images = input_root / "images" / "train"
    val_images = input_root / "images" / "val"

    ensure_path_exists(project_root, "project-root")
    ensure_path_exists(input_root, "input-dataset")
    ensure_path_exists(train_images, "train images dir")
    ensure_path_exists(val_images, "val images dir")

    base_config = (project_root / args.base_config).resolve()
    ensure_path_exists(base_config, "base config")

    cfg = load_yaml(base_config)
    data_cfg = cfg.setdefault("data", {})
    data_cfg["train_image_dirs"] = [str(train_images)]
    data_cfg["val_image_dirs"] = [str(val_images)]

    if args.epochs is not None:
        cfg.setdefault("train", {})["epochs"] = int(args.epochs)
    if args.batch is not None:
        cfg.setdefault("train", {})["batch_size"] = int(args.batch)
    if args.imgsz is not None:
        cfg.setdefault("train", {})["image_size"] = int(args.imgsz)

    kaggle_cfg_path = (project_root / args.kaggle_config).resolve()
    save_yaml(kaggle_cfg_path, cfg)
    print(f"Saved Kaggle config: {kaggle_cfg_path}", flush=True)

    if args.run_quality_check:
        run_cmd(
            [
                "python",
                "scripts/check_dataset_quality.py",
                "--dataset",
                str(input_root),
                "--min-images",
                "1000",
                "--min-classes",
                "2",
            ],
            cwd=project_root,
        )

    out_path = args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)

    train_cmd = [
        "python",
        "-m",
        "ai_core.training.train_hybrid",
        "--config",
        str(kaggle_cfg_path),
        "--out",
        str(out_path),
        "--workers",
        str(args.workers),
        "--log-interval",
        str(args.log_interval),
        "--save-every",
        str(args.save_every),
    ]

    if args.epochs is not None:
        train_cmd += ["--epochs", str(args.epochs)]
    if args.batch is not None:
        train_cmd += ["--batch", str(args.batch)]
    if args.imgsz is not None:
        train_cmd += ["--imgsz", str(args.imgsz)]

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    print("[INFO] Starting Kaggle training...", flush=True)
    subprocess.run(train_cmd, cwd=str(project_root), check=True, env=env)


if __name__ == "__main__":
    main()
