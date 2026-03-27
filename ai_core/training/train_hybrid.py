from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import cv2
import torch
import yaml
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset

from ai_core.models.hybrid_model import AquaDetHybridModel


class DummyAquaDataset(Dataset):
    def __init__(self, image_size: int = 640, num_classes: int = 4):
        self.image_size = image_size
        self.num_classes = num_classes

    def __len__(self) -> int:
        return 128

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        image = torch.rand(3, self.image_size, self.image_size)
        p3_size = self.image_size // 8
        bbox = torch.tensor([0.5, 0.5, 0.2, 0.2], dtype=torch.float32)
        mask = torch.zeros(p3_size, p3_size, dtype=torch.float32)
        cx = int(bbox[0].item() * p3_size)
        cy = int(bbox[1].item() * p3_size)
        bw = max(1, int(bbox[2].item() * p3_size * 0.5))
        bh = max(1, int(bbox[3].item() * p3_size * 0.5))
        x1, x2 = max(0, cx - bw), min(p3_size, cx + bw)
        y1, y2 = max(0, cy - bh), min(p3_size, cy + bh)
        mask[y1:y2, x1:x2] = 1.0

        target = {
            "cls": torch.randint(0, self.num_classes, (1,)).squeeze(0),
            "bbox": bbox,
            "mask": mask,
            "depth": torch.ones(1, p3_size, p3_size, dtype=torch.float32),
        }
        return image, target


class MultiSourceAquaDataset(Dataset):
    def __init__(self, image_dirs: List[Path], image_size: int = 640, num_classes: int = 4):
        self.image_size = image_size
        self.num_classes = num_classes
        self.images: List[Path] = []
        for image_dir in image_dirs:
            if not image_dir.exists():
                continue
            for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
                self.images.extend(image_dir.rglob(ext))

        self.images = sorted(set(self.images))
        if not self.images:
            raise ValueError("No training images found in configured data.image_dirs")

    def __len__(self) -> int:
        return len(self.images)

    @staticmethod
    def _label_file_path(image_path: Path) -> Path:
        path_str = str(image_path)
        if "/images/" in path_str:
            return Path(path_str.replace("/images/", "/labels/")).with_suffix(".txt")
        if "\\images\\" in path_str:
            return Path(path_str.replace("\\images\\", "\\labels\\")).with_suffix(".txt")
        return image_path.with_suffix(".txt")

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        image_path = self.images[idx]
        image_bgr = cv2.imread(str(image_path))
        if image_bgr is None:
            raise RuntimeError(f"Failed to read image: {image_path}")

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        image_rgb = cv2.resize(image_rgb, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR)
        image_tensor = torch.from_numpy(image_rgb).permute(2, 0, 1).float() / 255.0

        cls = 0
        bbox = torch.tensor([0.5, 0.5, 0.2, 0.2], dtype=torch.float32)

        label_path = self._label_file_path(image_path)
        if label_path.exists():
            with label_path.open("r", encoding="utf-8") as f:
                first_line = f.readline().strip()
            if first_line:
                parts = first_line.split()
                if len(parts) >= 5:
                    try:
                        cls = int(float(parts[0])) % self.num_classes
                        bbox = torch.tensor(
                            [float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])],
                            dtype=torch.float32,
                        )
                    except ValueError:
                        cls = 0

        p3_size = self.image_size // 8
        cx = int(torch.clamp(bbox[0], 0.0, 1.0).item() * p3_size)
        cy = int(torch.clamp(bbox[1], 0.0, 1.0).item() * p3_size)
        bw = max(1, int(torch.clamp(bbox[2], 0.02, 1.0).item() * p3_size * 0.5))
        bh = max(1, int(torch.clamp(bbox[3], 0.02, 1.0).item() * p3_size * 0.5))
        x1, x2 = max(0, cx - bw), min(p3_size, cx + bw)
        y1, y2 = max(0, cy - bh), min(p3_size, cy + bh)

        mask = torch.zeros(p3_size, p3_size, dtype=torch.float32)
        mask[y1:y2, x1:x2] = 1.0

        target = {
            "cls": torch.tensor(cls, dtype=torch.long),
            "bbox": torch.clamp(bbox, 0.0, 1.0),
            "mask": mask,
            "depth": torch.ones(1, p3_size, p3_size, dtype=torch.float32),
        }
        return image_tensor, target


def load_config(path: Path) -> Dict[str, Dict]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train AquaDet hybrid model (Phase-2/3 skeleton).")
    parser.add_argument("--config", type=Path, default=Path("configs/hybrid_train.yaml"))
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--imgsz", type=int, default=None)
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument("--out", type=Path, default=Path("artifacts/hybrid_model.pt"))
    args = parser.parse_args()

    cfg = load_config(args.config)
    model_cfg = cfg.get("model", {})
    train_cfg = cfg.get("train", {})
    loss_cfg = cfg.get("loss_weights", {})
    data_cfg = cfg.get("data", {})

    num_classes = int(model_cfg.get("num_classes", 4))
    pi_ge_enabled = bool(model_cfg.get("pi_ge_enabled", True))

    epochs = int(args.epochs if args.epochs is not None else train_cfg.get("epochs", 5))
    batch = int(args.batch if args.batch is not None else train_cfg.get("batch_size", 4))
    lr = float(args.lr if args.lr is not None else train_cfg.get("lr", 1e-4))
    image_size = int(args.imgsz if args.imgsz is not None else train_cfg.get("image_size", 640))

    w_cls = float(loss_cfg.get("cls", 1.0))
    w_bbox = float(loss_cfg.get("bbox", 2.0))
    w_mask = float(loss_cfg.get("mask", 1.0))
    w_depth = float(loss_cfg.get("depth", 0.2))

    image_dirs = [Path(p) for p in data_cfg.get("image_dirs", [])]
    if image_dirs:
        try:
            dataset = MultiSourceAquaDataset(image_dirs=image_dirs, image_size=image_size, num_classes=num_classes)
        except ValueError as exc:
            print(f"{exc}. Falling back to synthetic dataset. Update data.image_dirs in {args.config}")
            dataset = DummyAquaDataset(image_size=image_size, num_classes=num_classes)
    else:
        print("No data.image_dirs set; using synthetic dataset fallback")
        dataset = DummyAquaDataset(image_size=image_size, num_classes=num_classes)

    loader = DataLoader(dataset, batch_size=batch, shuffle=True)
    print(f"dataset_size={len(dataset)} batch={batch} steps_per_epoch={len(loader)}", flush=True)

    model = AquaDetHybridModel(num_classes=num_classes, pi_ge_enabled=pi_ge_enabled)
    optimizer = AdamW(model.parameters(), lr=lr)
    cls_loss_fn = nn.CrossEntropyLoss()
    bbox_loss_fn = nn.SmoothL1Loss()
    mask_loss_fn = nn.BCELoss()
    depth_loss_fn = nn.L1Loss()

    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for step, (images, targets) in enumerate(loader, start=1):
            outputs = model(images)
            logits = outputs["logits"].mean(dim=[2, 3])
            box_pred = torch.sigmoid(outputs["boxes"].mean(dim=[2, 3]))
            mask_pred = outputs["masks"].squeeze(1)
            depth = outputs["depth"]

            cls_loss = cls_loss_fn(logits, targets["cls"])
            bbox_loss = bbox_loss_fn(box_pred, targets["bbox"])
            mask_target = torch.nn.functional.interpolate(
                targets["mask"].unsqueeze(1), size=mask_pred.shape[-2:], mode="nearest"
            ).squeeze(1)
            mask_loss = mask_loss_fn(mask_pred, mask_target)
            depth_loss = depth_loss_fn(
                depth,
                torch.nn.functional.interpolate(targets["depth"], size=depth.shape[-2:], mode="nearest"),
            )

            loss = w_cls * cls_loss + w_bbox * bbox_loss + w_mask * mask_loss + w_depth * depth_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())

            if args.log_interval > 0 and step % args.log_interval == 0:
                print(
                    f"epoch={epoch+1} step={step}/{len(loader)} loss={float(loss.item()):.4f}",
                    flush=True,
                )

        print(f"epoch={epoch+1} loss={total_loss/len(loader):.4f}", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), args.out)
    print(f"Saved: {args.out}", flush=True)


if __name__ == "__main__":
    main()
