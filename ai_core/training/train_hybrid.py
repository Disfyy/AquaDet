from __future__ import annotations

import argparse
import random
from contextlib import nullcontext
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import torch
import yaml
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset, random_split

from ai_core.models.hybrid_model import AquaDetHybridModel
from ai_core.training.losses import FocalLoss, CIoULoss, UncertaintyWeightedLoss
from ai_core.training.ema import ModelEMA


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------


class DummyAquaDataset(Dataset):
    def __init__(self, image_size: int = 640, num_classes: int = 4):
        self.image_size = image_size
        self.num_classes = num_classes

    def __len__(self) -> int:
        return 128

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        image = torch.rand(3, self.image_size, self.image_size)
        p3 = self.image_size // 2
        cls_map = torch.full((p3, p3), fill_value=-100, dtype=torch.long)
        obj_map = torch.zeros((p3, p3), dtype=torch.float32)
        box_map = torch.zeros((4, p3, p3), dtype=torch.float32)
        seg_map = torch.zeros((p3, p3), dtype=torch.float32)
        depth_map = torch.ones((1, p3, p3), dtype=torch.float32)

        for _ in range(random.randint(1, 3)):
            cls = random.randint(0, self.num_classes - 1)
            cx = random.random() * 0.8 + 0.1
            cy = random.random() * 0.8 + 0.1
            bw = random.random() * 0.25 + 0.05
            bh = random.random() * 0.25 + 0.05
            gx = min(p3 - 1, max(0, int(cx * p3)))
            gy = min(p3 - 1, max(0, int(cy * p3)))
            cls_map[gy, gx] = cls
            obj_map[gy, gx] = 1.0
            box_map[:, gy, gx] = torch.tensor([cx * p3 - gx, cy * p3 - gy, bw, bh], dtype=torch.float32)

            x1 = max(0, int((cx - bw / 2.0) * p3))
            y1 = max(0, int((cy - bh / 2.0) * p3))
            x2 = min(p3, int((cx + bw / 2.0) * p3))
            y2 = min(p3, int((cy + bh / 2.0) * p3))
            seg_map[y1:y2, x1:x2] = 1.0
            depth_map[0, y1:y2, x1:x2] = max(0.2, min(5.0, 1.0 / max(1e-4, bw * bh * 10.0)))

        return image, {
            "cls": cls_map,
            "obj": obj_map,
            "box": box_map,
            "mask": seg_map,
            "depth": depth_map,
        }


class MultiSourceAquaDataset(Dataset):
    def __init__(self, image_dirs: List[Path], image_size: int = 640, num_classes: int = 4, is_train: bool = True):
        self.image_size = image_size
        self.num_classes = num_classes
        self.is_train = is_train
        self.images: List[Path] = []
        for image_dir in image_dirs:
            if not image_dir.exists():
                continue
            for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
                self.images.extend(image_dir.rglob(ext))

        self.images = sorted(set(self.images))
        if not self.images:
            raise ValueError("No training images found in configured data.image_dirs")

        # Set up Albumentations pipeline
        if self.is_train:
            self.transform = A.Compose([
                A.Resize(height=image_size, width=image_size),
                A.HorizontalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.7),
                A.GaussNoise(p=0.3),
                A.ToFloat(max_value=255.0),
                ToTensorV2(),
            ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'], min_visibility=0.3))
        else:
            self.transform = A.Compose([
                A.Resize(height=image_size, width=image_size),
                A.ToFloat(max_value=255.0),
                ToTensorV2(),
            ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

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

        raw_bboxes = []
        raw_labels = []

        label_path = self._label_file_path(image_path)
        if label_path.exists():
            with label_path.open("r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue
                    try:
                        cls_idx = int(float(parts[0])) % self.num_classes
                        cx, cy, bw, bh = (float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4]))
                        
                        bw = min(1.0, max(0.001, bw))
                        bh = min(1.0, max(0.001, bh))
                        cx = min(1.0, max(0.0, cx))
                        cy = min(1.0, max(0.0, cy))
                        
                        raw_bboxes.append([cx, cy, bw, bh])
                        raw_labels.append(cls_idx)
                    except ValueError:
                        continue

        if hasattr(self, 'transform') and self.transform:
            augmented = self.transform(image=image_rgb, bboxes=raw_bboxes, class_labels=raw_labels)
            image_tensor = augmented['image']
            aug_bboxes = augmented['bboxes']
            aug_labels = augmented['class_labels']
            labels = [(cls, *bbox) for cls, bbox in zip(aug_labels, aug_bboxes)]
        else:
            image_rgb = cv2.resize(image_rgb, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR)
            image_tensor = torch.from_numpy(image_rgb).permute(2, 0, 1).float() / 255.0
            labels = [(cls, *bbox) for cls, bbox in zip(raw_labels, raw_bboxes)]

        p3 = self.image_size // 2
        cls_map = torch.full((p3, p3), fill_value=-100, dtype=torch.long)
        obj_map = torch.zeros((p3, p3), dtype=torch.float32)
        box_map = torch.zeros((4, p3, p3), dtype=torch.float32)
        seg_map = torch.zeros((p3, p3), dtype=torch.float32)
        depth_map = torch.ones((1, p3, p3), dtype=torch.float32)

        for cls_idx, cx, cy, bw, bh in labels:
            cx = min(1.0, max(0.0, cx))
            cy = min(1.0, max(0.0, cy))
            bw = min(1.0, max(0.01, bw))
            bh = min(1.0, max(0.01, bh))

            gx = min(p3 - 1, max(0, int(cx * p3)))
            gy = min(p3 - 1, max(0, int(cy * p3)))
            cls_map[gy, gx] = cls_idx
            obj_map[gy, gx] = 1.0
            box_map[:, gy, gx] = torch.tensor([cx * p3 - gx, cy * p3 - gy, bw, bh], dtype=torch.float32)

            x1 = max(0, int((cx - bw / 2.0) * p3))
            y1 = max(0, int((cy - bh / 2.0) * p3))
            x2 = min(p3, int((cx + bw / 2.0) * p3))
            y2 = min(p3, int((cy + bh / 2.0) * p3))
            seg_map[y1:y2, x1:x2] = 1.0
            depth_map[0, y1:y2, x1:x2] = max(0.2, min(5.0, 1.0 / max(1e-4, bw * bh * 10.0)))

        return image_tensor, {
            "cls": cls_map,
            "obj": obj_map,
            "box": box_map,
            "mask": seg_map,
            "depth": depth_map,
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def load_config(path: Path) -> Dict[str, Dict]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def to_device(batch_targets: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {k: v.to(device, non_blocking=True) for k, v in batch_targets.items()}


def compute_mask_iou(mask_logits: torch.Tensor, mask_target: torch.Tensor) -> float:
    pred = (torch.sigmoid(mask_logits) > 0.5).float()
    target = (mask_target > 0.5).float()
    inter = (pred * target).sum().item()
    union = ((pred + target) > 0).float().sum().item()
    return float(inter / union) if union > 0 else 0.0


def build_grad_scaler(use_cuda: bool):
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        try:
            return torch.amp.GradScaler(device="cuda", enabled=use_cuda)
        except TypeError:
            return torch.amp.GradScaler(enabled=use_cuda)
    return torch.cuda.amp.GradScaler(enabled=use_cuda)


def autocast_context(use_cuda: bool):
    if use_cuda:
        if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
            return torch.amp.autocast(device_type="cuda", enabled=True)
        return torch.cuda.amp.autocast(enabled=True)
    return nullcontext()


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Train AquaDet hybrid model.")
    parser.add_argument("--config", type=Path, default=Path("configs/hybrid_train.yaml"))
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--imgsz", type=int, default=None)
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--save-every", type=int, default=1)
    parser.add_argument("--max-train-steps", type=int, default=0)
    parser.add_argument("--max-val-steps", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path("artifacts/hybrid_model.pt"))
    args = parser.parse_args()

    cfg = load_config(args.config)
    model_cfg = cfg.get("model", {})
    train_cfg = cfg.get("train", {})
    data_cfg = cfg.get("data", {})

    num_classes = int(model_cfg.get("num_classes", 4))
    pi_ge_enabled = bool(model_cfg.get("pi_ge_enabled", True))

    epochs = int(args.epochs if args.epochs is not None else train_cfg.get("epochs", 5))
    batch = int(args.batch if args.batch is not None else train_cfg.get("batch_size", 4))
    lr = float(args.lr if args.lr is not None else train_cfg.get("lr", 1e-4))
    image_size = int(args.imgsz if args.imgsz is not None else train_cfg.get("image_size", 640))

    train_image_dirs = [Path(p) for p in data_cfg.get("train_image_dirs", [])]
    val_image_dirs = [Path(p) for p in data_cfg.get("val_image_dirs", [])]
    image_dirs = [Path(p) for p in data_cfg.get("image_dirs", [])]

    # ---- Build datasets ----
    if train_image_dirs:
        try:
            train_dataset = MultiSourceAquaDataset(
                image_dirs=train_image_dirs,
                image_size=image_size,
                num_classes=num_classes,
            )
        except ValueError as exc:
            print(f"{exc}. Falling back to synthetic dataset. Update data.train_image_dirs in {args.config}")
            train_dataset = DummyAquaDataset(image_size=image_size, num_classes=num_classes)
    elif image_dirs:
        try:
            import copy
            full_dataset = MultiSourceAquaDataset(image_dirs=image_dirs, image_size=image_size, num_classes=num_classes)
            val_len = max(1, int(0.1 * len(full_dataset)))
            train_len = max(1, len(full_dataset) - val_len)
            train_dataset, val_dataset_split = random_split(full_dataset, [train_len, val_len])
            
            val_dataset_base = copy.deepcopy(full_dataset)
            val_dataset_base.is_train = False
            val_dataset_base.transform = A.Compose([
                A.Resize(height=image_size, width=image_size),
                A.ToFloat(max_value=255.0),
                ToTensorV2(),
            ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))
            val_dataset_split.dataset = val_dataset_base
            
            val_dataset = val_dataset_split
        except ValueError as exc:
            print(f"{exc}. Falling back to synthetic dataset. Update data.image_dirs in {args.config}")
            train_dataset = DummyAquaDataset(image_size=image_size, num_classes=num_classes)
            val_dataset = DummyAquaDataset(image_size=image_size, num_classes=num_classes)
    else:
        print("No data.train_image_dirs or data.image_dirs set; using synthetic dataset fallback")
        train_dataset = DummyAquaDataset(image_size=image_size, num_classes=num_classes)
        val_dataset = DummyAquaDataset(image_size=image_size, num_classes=num_classes)

    if "val_dataset" not in locals():
        if val_image_dirs:
            try:
                val_dataset = MultiSourceAquaDataset(
                    image_dirs=val_image_dirs,
                    image_size=image_size,
                    num_classes=num_classes,
                    is_train=False
                )
            except ValueError:
                val_dataset = DummyAquaDataset(image_size=image_size, num_classes=num_classes)
        else:
            val_dataset = DummyAquaDataset(image_size=image_size, num_classes=num_classes)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=use_cuda,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=use_cuda,
    )

    print(
        f"train_size={len(train_dataset)} val_size={len(val_dataset)} batch={batch} "
        f"train_steps={len(train_loader)} val_steps={len(val_loader)} device={device}",
        flush=True,
    )

    # ---- Model ----
    model = AquaDetHybridModel(num_classes=num_classes, pi_ge_enabled=pi_ge_enabled).to(device)

    # ---- Optimizer ----
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    # ---- Loss functions ----
    cls_loss_fn = FocalLoss(num_classes=num_classes, alpha=0.25, gamma=2.0, ignore_index=-100)
    bbox_loss_fn = CIoULoss()
    obj_loss_fn = nn.BCEWithLogitsLoss()
    mask_loss_fn = nn.BCEWithLogitsLoss()
    depth_loss_fn = nn.L1Loss()

    # Learnable multi-task loss weighting (5 tasks: cls, bbox, obj, mask, depth)
    mtl_weighter = UncertaintyWeightedLoss(num_tasks=5).to(device)
    # Add weighter params to optimiser
    optimizer.add_param_group({"params": mtl_weighter.parameters(), "lr": lr})

    # ---- Scheduler ----
    from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
    warmup_epochs = min(5, max(1, epochs // 10))
    warmup = LinearLR(optimizer, start_factor=0.01, total_iters=warmup_epochs)
    cosine = CosineAnnealingLR(optimizer, T_max=max(1, epochs - warmup_epochs), eta_min=1e-6)
    scheduler = SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs])

    scaler = build_grad_scaler(use_cuda=use_cuda)

    # ---- EMA ----
    ema = ModelEMA(model, decay=0.9999, device=device)

    # ---- Checkpointing ----
    checkpoint_dir = args.out.parent / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_path = args.out.parent / "hybrid_model_best.pt"
    best_score = -1e9

    # ==================================================================
    # Training loop
    # ==================================================================
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for step, (images, targets) in enumerate(train_loader, start=1):
            images = images.to(device, non_blocking=True)
            targets = to_device(targets, device)

            optimizer.zero_grad(set_to_none=True)
            with autocast_context(use_cuda=use_cuda):
                outputs = model(images)

                cls_logits = outputs["logits"]
                box_pred = torch.sigmoid(outputs["boxes"])
                mask_logits = outputs["masks"].squeeze(1)
                obj_logits = outputs["obj"].squeeze(1)
                depth_pred = outputs["depth"]

                # Classification (Focal Loss)
                cls_loss = cls_loss_fn(cls_logits, targets["cls"])

                # Objectness (BCE)
                obj_loss = obj_loss_fn(obj_logits, targets["obj"])

                # Bounding box (CIoU on positive cells only)
                obj_bool = targets["obj"] > 0.5
                pred_box_flat = box_pred.permute(0, 2, 3, 1)[obj_bool]
                target_box_flat = targets["box"].permute(0, 2, 3, 1)[obj_bool]
                bbox_loss = bbox_loss_fn(pred_box_flat, target_box_flat)

                # Mask (BCE)
                mask_loss = mask_loss_fn(mask_logits, targets["mask"])

                # Depth (L1)
                depth_loss = depth_loss_fn(depth_pred, targets["depth"])

                # Combined with learned uncertainty weighting
                loss = mtl_weighter(cls_loss, bbox_loss, obj_loss, mask_loss, depth_loss)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            scaler.step(optimizer)
            scaler.update()

            ema.update(model)
            total_loss += float(loss.item())

            if args.log_interval > 0 and step % args.log_interval == 0:
                print(
                    f"epoch={epoch+1} step={step}/{len(train_loader)} "
                    f"loss={float(loss.item()):.4f} cls={float(cls_loss.item()):.4f} "
                    f"obj={float(obj_loss.item()):.4f} bbox={float(bbox_loss.item()):.4f} "
                    f"mask={float(mask_loss.item()):.4f} depth={float(depth_loss.item()):.4f}",
                    flush=True,
                )
            if args.max_train_steps > 0 and step >= args.max_train_steps:
                break

        train_loss = total_loss / max(1, len(train_loader))

        # ==============================================================
        # Validation
        # ==============================================================
        model.eval()
        val_loss_total = 0.0
        val_cls_correct = 0
        val_cls_total = 0
        val_mask_iou = 0.0

        with torch.no_grad():
            for val_step, (images, targets) in enumerate(val_loader, start=1):
                images = images.to(device, non_blocking=True)
                targets = to_device(targets, device)

                outputs = ema.module(images)
                cls_logits = outputs["logits"]
                box_pred = torch.sigmoid(outputs["boxes"])
                mask_logits = outputs["masks"].squeeze(1)
                obj_logits = outputs["obj"].squeeze(1)
                depth_pred = outputs["depth"]

                cls_loss = cls_loss_fn(cls_logits, targets["cls"])
                obj_loss = obj_loss_fn(obj_logits, targets["obj"])
                obj_bool = targets["obj"] > 0.5
                pred_box_flat = box_pred.permute(0, 2, 3, 1)[obj_bool]
                target_box_flat = targets["box"].permute(0, 2, 3, 1)[obj_bool]
                bbox_loss = bbox_loss_fn(pred_box_flat, target_box_flat)
                mask_loss = mask_loss_fn(mask_logits, targets["mask"])
                depth_loss = depth_loss_fn(depth_pred, targets["depth"])

                val_loss = mtl_weighter(cls_loss, bbox_loss, obj_loss, mask_loss, depth_loss)
                val_loss_total += float(val_loss.item())

                pred_cls = cls_logits.argmax(dim=1)
                val_cls_correct += int(((pred_cls == targets["cls"]) & obj_bool).sum().item())
                val_cls_total += int(obj_bool.sum().item())
                val_mask_iou += compute_mask_iou(mask_logits, targets["mask"])
                if args.max_val_steps > 0 and val_step >= args.max_val_steps:
                    break

        val_loss_avg = val_loss_total / max(1, len(val_loader))
        val_cls_acc = float(val_cls_correct / max(1, val_cls_total))
        val_mask_iou_avg = float(val_mask_iou / max(1, len(val_loader)))
        score = val_cls_acc + val_mask_iou_avg - 0.1 * val_loss_avg

        # Print learned task weights for monitoring
        task_names = ["cls", "bbox", "obj", "mask", "depth"]
        weights_str = " ".join(
            f"w_{t}={torch.exp(-mtl_weighter.log_vars[i]).item():.3f}"
            for i, t in enumerate(task_names)
        )

        print(
            f"epoch={epoch+1} train_loss={train_loss:.4f} val_loss={val_loss_avg:.4f} "
            f"val_cls_acc={val_cls_acc:.4f} val_mask_iou={val_mask_iou_avg:.4f} "
            f"score={score:.4f} lr={scheduler.get_last_lr()[0]:.6f} "
            f"ema_decay={ema.decay:.6f} {weights_str}",
            flush=True,
        )

        scheduler.step()

        if (epoch + 1) % max(1, args.save_every) == 0:
            epoch_path = checkpoint_dir / f"epoch_{epoch+1:03d}.pt"
            torch.save(ema.module.state_dict(), epoch_path)

        if score > best_score:
            best_score = score
            torch.save(ema.module.state_dict(), best_path)
            print(f"Saved best checkpoint: {best_path}", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ema.module.state_dict(), args.out)
    print(f"Saved: {args.out}", flush=True)


if __name__ == "__main__":
    main()
