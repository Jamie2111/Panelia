from __future__ import annotations

import argparse
from datetime import UTC, datetime
import json
import logging
import os
import random
from pathlib import Path
import signal
import sys
from typing import Any, Callable

from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset, random_split
import torchvision.transforms.v2 as T
from torchvision.tv_tensors import BoundingBoxes, BoundingBoxFormat

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.core.config import get_settings
from app.utils.files import write_json
from training.panel_detector_model import build_panel_detector_model, resolve_torch_device


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("train-panel-detector")


def _status_path_from_env() -> Path | None:
    raw = os.environ.get("PANELIA_TRAINING_STATUS_PATH")
    return Path(raw) if raw else None


def write_training_status(
    *,
    status_path: Path | None,
    training_status: str,
    message: str,
    progress_percent: float,
    current_epoch: int,
    total_epochs: int,
    train_loss: float | None = None,
    val_loss: float | None = None,
    pid: int | None = None,
    is_training: bool = True,
    started_at: str | None = None,
) -> None:
    if status_path is None:
        return
    write_json(
        status_path,
        {
            "training_status": training_status,
            "is_training": is_training,
            "pid": pid,
            "started_at": started_at or datetime.now(tz=UTC).isoformat(),
            "heartbeat_at": datetime.now(tz=UTC).isoformat(),
            "progress_percent": round(progress_percent, 2),
            "current_epoch": int(current_epoch),
            "total_epochs": int(total_epochs),
            "train_loss": train_loss,
            "val_loss": val_loss,
            "message": message,
        },
    )


_TRAIN_AUGMENTATIONS = T.Compose([
    T.RandomHorizontalFlip(p=0.5),
    T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
    T.ToDtype(torch.float32, scale=True),
])

_EVAL_TRANSFORM = T.Compose([
    T.ToDtype(torch.float32, scale=True),
])


class PanelAnnotationDataset(Dataset):
    def __init__(self, dataset_root: str | Path, training: bool = True) -> None:
        self.root = Path(dataset_root)
        self.images_dir = self.root / "images"
        self.annotations_dir = self.root / "annotations"
        self.annotation_paths = sorted(self.annotations_dir.glob("*.json"))
        if not self.annotation_paths:
            raise FileNotFoundError(f"No annotations found in {self.annotations_dir}")
        self.transform = _TRAIN_AUGMENTATIONS if training else _EVAL_TRANSFORM

    def __len__(self) -> int:
        return len(self.annotation_paths)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, dict[str, Any]]:
        annotation_path = self.annotation_paths[index]
        annotation = json.loads(annotation_path.read_text(encoding="utf-8"))
        image_path = self.images_dir / str(annotation["image"])
        with Image.open(image_path) as image:
            rgb = image.convert("RGB")
            image_width, image_height = rgb.size

        boxes_xyxy: list[list[float]] = []
        for panel in annotation.get("panels", []):
            x = float(panel["x"])
            y = float(panel["y"])
            width = float(panel["w"])
            height = float(panel["h"])
            boxes_xyxy.append([x, y, x + width, y + height])

        # Use torchvision v2 transforms so bboxes are correctly transformed with the image
        tv_image = T.functional.to_image(rgb)
        if boxes_xyxy:
            tv_boxes = BoundingBoxes(
                torch.as_tensor(boxes_xyxy, dtype=torch.float32),
                format=BoundingBoxFormat.XYXY,
                canvas_size=(image_height, image_width),
            )
            tv_image, tv_boxes = self.transform(tv_image, tv_boxes)
            boxes = tv_boxes.data if hasattr(tv_boxes, "data") else torch.as_tensor(tv_boxes, dtype=torch.float32)
        else:
            tv_image = self.transform(tv_image)
            boxes = torch.zeros((0, 4), dtype=torch.float32)

        image_tensor = tv_image if isinstance(tv_image, torch.Tensor) else T.functional.to_tensor(tv_image)
        labels = torch.ones((boxes.shape[0],), dtype=torch.int64)
        areas = (
            (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
            if boxes.numel()
            else torch.zeros((0,), dtype=torch.float32)
        )
        target = {
            "boxes": boxes.reshape(-1, 4),
            "labels": labels,
            "image_id": torch.tensor([index]),
            "area": areas,
            "iscrowd": torch.zeros((labels.shape[0],), dtype=torch.int64),
            "orig_size": torch.tensor([image_height, image_width]),
            "size": torch.tensor([image_height, image_width]),
        }
        return image_tensor, target


def detection_collate_fn(batch: list[tuple[torch.Tensor, dict[str, Any]]]) -> tuple[list[torch.Tensor], list[dict[str, Any]]]:
    images, targets = zip(*batch)
    return list(images), list(targets)


def build_data_loaders(
    dataset_root: Path,
    batch_size: int,
    workers: int,
    val_split: float,
    seed: int,
) -> tuple[DataLoader, DataLoader | None, int]:
    # Use separate dataset instances so train gets augmentations and val gets clean images
    train_full = PanelAnnotationDataset(dataset_root, training=True)
    dataset_size = len(train_full)
    if dataset_size < 2 or val_split <= 0:
        train_dataset = train_full
        val_dataset = None
    else:
        val_size = max(1, int(round(dataset_size * val_split)))
        train_size = max(dataset_size - val_size, 1)
        if train_size + val_size > dataset_size:
            val_size = dataset_size - train_size
        generator = torch.Generator().manual_seed(seed)
        train_indices, val_indices = random_split(range(dataset_size), [train_size, val_size], generator=generator)
        train_dataset = torch.utils.data.Subset(train_full, list(train_indices))
        val_full = PanelAnnotationDataset(dataset_root, training=False)
        val_dataset = torch.utils.data.Subset(val_full, list(val_indices))

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        collate_fn=detection_collate_fn,
    )
    val_loader = None
    if val_dataset is not None and len(val_dataset) > 0:
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=workers,
            collate_fn=detection_collate_fn,
        )
    return train_loader, val_loader, dataset_size


def train_one_epoch(
    model: Any,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: str,
    progress_callback: Callable[[int, int, float | None], None] | None = None,
) -> float:
    model.train()
    running_loss = 0.0
    steps = 0
    total_steps = max(len(loader), 1)
    for images, targets in loader:
        images = [image.to(device) for image in images]
        moved_targets = [{key: value.to(device) for key, value in target.items()} for target in targets]
        loss_dict = model(images, moved_targets)
        loss = sum(loss_dict.values())
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        running_loss += float(loss.detach().cpu().item())
        steps += 1
        if progress_callback:
            progress_callback(steps, total_steps, running_loss / max(steps, 1))
    return running_loss / max(steps, 1)


def evaluate_loss(
    model: Any,
    loader: DataLoader,
    device: str,
    progress_callback: Callable[[int, int, float | None], None] | None = None,
) -> float:
    model.train()
    running_loss = 0.0
    steps = 0
    total_steps = max(len(loader), 1)
    with torch.no_grad():
        for images, targets in loader:
            images = [image.to(device) for image in images]
            moved_targets = [{key: value.to(device) for key, value in target.items()} for target in targets]
            loss_dict = model(images, moved_targets)
            loss = sum(loss_dict.values())
            running_loss += float(loss.detach().cpu().item())
            steps += 1
            if progress_callback:
                progress_callback(steps, total_steps, running_loss / max(steps, 1))
    return running_loss / max(steps, 1)


def save_checkpoint(
    model: Any,
    output_path: Path,
    *,
    architecture: str,
    dataset_size: int,
    epochs: int,
    best_val_loss: float | None,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "architecture": architecture,
        "num_classes": 2,
        "state_dict": model.state_dict(),
        "classes": {"panel": 1},
        "dataset_size": dataset_size,
        "epochs_trained": epochs,
        "best_val_loss": best_val_loss,
    }
    torch.save(checkpoint, output_path)


def save_training_metadata(
    *,
    output_path: Path,
    dataset_root: Path,
    dataset_size: int,
    architecture: str,
    epochs: int,
    best_val_loss: float | None,
) -> None:
    metadata_path = output_path.with_suffix(".metadata.json")
    write_json(
        metadata_path,
        {
            "trained_at": datetime.now(tz=UTC).isoformat(),
            "checkpoint_path": str(output_path),
            "dataset_root": str(dataset_root),
            "dataset_size": int(dataset_size),
            "architecture": architecture,
            "epochs_trained": int(epochs),
            "best_val_loss": best_val_loss,
        },
    )


def parse_args() -> argparse.Namespace:
    settings = get_settings()
    parser = argparse.ArgumentParser(description="Fine-tune the panel detector on human-corrected panel boxes.")
    parser.add_argument("--dataset-root", type=Path, default=settings.training_data_dir)
    parser.add_argument("--output", type=Path, default=settings.panel_detector_models_dir / "panel_detector_v2.pt")
    parser.add_argument("--architecture", type=str, default="fasterrcnn_mobilenet_v3_large_fpn")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--patience", type=int, default=3, help="Early stopping: stop after this many epochs with no val_loss improvement")
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--resume", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    status_path = _status_path_from_env()
    started_at = datetime.now(tz=UTC).isoformat()
    cancel_requested = False

    def handle_termination(_signum: int, _frame: Any) -> None:
        nonlocal cancel_requested
        cancel_requested = True
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, handle_termination)
    signal.signal(signal.SIGINT, handle_termination)

    resolved_device = resolve_torch_device() if args.device == "auto" else args.device
    device = resolved_device
    if args.device == "auto" and resolved_device == "mps":
        logger.warning("Auto-selected MPS for detector training, but torchvision detection training is more reliable on CPU. Falling back to CPU.")
        device = "cpu"
    logger.info("Using device: %s", device)
    write_training_status(
        status_path=status_path,
        training_status="running",
        is_training=True,
        pid=os.getpid(),
        progress_percent=0.0,
        current_epoch=0,
        total_epochs=args.epochs,
        message="Scanning saved corrected pages for training data.",
        started_at=started_at,
    )

    try:
        train_loader, val_loader, dataset_size = build_data_loaders(
            args.dataset_root,
            batch_size=args.batch_size,
            workers=args.workers,
            val_split=args.val_split,
            seed=args.seed,
        )
        logger.info("Loaded %s annotated pages from %s", dataset_size, args.dataset_root)
        write_training_status(
            status_path=status_path,
            training_status="running",
            is_training=True,
            pid=os.getpid(),
            progress_percent=4.0,
            current_epoch=0,
            total_epochs=args.epochs,
            message=f"Loaded {dataset_size} corrected pages. Preparing detector model.",
            started_at=started_at,
        )

        write_training_status(
            status_path=status_path,
            training_status="running",
            is_training=True,
            pid=os.getpid(),
            progress_percent=8.0,
            current_epoch=0,
            total_epochs=args.epochs,
            message="Loading pretrained detector weights.",
            started_at=started_at,
        )
        model = build_panel_detector_model(args.architecture, pretrained=True, num_classes=2)
        if args.resume is not None and args.resume.exists():
            checkpoint = torch.load(args.resume, map_location="cpu")
            model.load_state_dict(checkpoint["state_dict"])
            logger.info("Resumed weights from %s", args.resume)

        write_training_status(
            status_path=status_path,
            training_status="running",
            is_training=True,
            pid=os.getpid(),
            progress_percent=10.0,
            current_epoch=0,
            total_epochs=args.epochs,
            message=f"Model ready on {device}. Starting epoch 1/{args.epochs}.",
            started_at=started_at,
        )

        model = model.to(device)
        optimizer = torch.optim.AdamW(
            [parameter for parameter in model.parameters() if parameter.requires_grad],
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))

        best_val_loss = None
        best_state_dict = None
        train_loss: float | None = None
        val_loss: float | None = None
        training_progress_span = 88.0
        preamble_progress = 10.0
        no_improve_epochs = 0

        for epoch in range(1, args.epochs + 1):
            epoch_start_progress = preamble_progress + ((epoch - 1) / max(args.epochs, 1)) * training_progress_span
            epoch_end_progress = preamble_progress + (epoch / max(args.epochs, 1)) * training_progress_span
            train_end_progress = epoch_start_progress + (epoch_end_progress - epoch_start_progress) * 0.86

            def on_train_progress(step: int, total_steps: int, running_loss: float | None) -> None:
                batch_progress = step / max(total_steps, 1)
                progress = epoch_start_progress + (train_end_progress - epoch_start_progress) * batch_progress
                write_training_status(
                    status_path=status_path,
                    training_status="running",
                    is_training=True,
                    pid=os.getpid(),
                    progress_percent=progress,
                    current_epoch=epoch,
                    total_epochs=args.epochs,
                    train_loss=running_loss,
                    val_loss=None,
                    message=f"Epoch {epoch}/{args.epochs} • training batch {step}/{total_steps}",
                    started_at=started_at,
                )

            train_loss = train_one_epoch(model, train_loader, optimizer, device, progress_callback=on_train_progress)

            if val_loader is not None:
                def on_val_progress(step: int, total_steps: int, running_val_loss: float | None) -> None:
                    batch_progress = step / max(total_steps, 1)
                    progress = train_end_progress + (epoch_end_progress - train_end_progress) * batch_progress
                    write_training_status(
                        status_path=status_path,
                        training_status="running",
                        is_training=True,
                        pid=os.getpid(),
                        progress_percent=progress,
                        current_epoch=epoch,
                        total_epochs=args.epochs,
                        train_loss=train_loss,
                        val_loss=running_val_loss,
                        message=f"Epoch {epoch}/{args.epochs} • validating batch {step}/{total_steps}",
                        started_at=started_at,
                    )

                val_loss = evaluate_loss(model, val_loader, device, progress_callback=on_val_progress)
            else:
                val_loss = None

            scheduler.step()

            if val_loss is None:
                logger.info("Epoch %s/%s train_loss=%.4f", epoch, args.epochs, train_loss)
                progress_message = f"Finished epoch {epoch}/{args.epochs}."
            else:
                logger.info("Epoch %s/%s train_loss=%.4f val_loss=%.4f", epoch, args.epochs, train_loss, val_loss)
                progress_message = f"Finished epoch {epoch}/{args.epochs}. val_loss={val_loss:.4f}"
                if best_val_loss is None or val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_state_dict = {
                        key: value.detach().cpu().clone()
                        for key, value in model.state_dict().items()
                    }
                    no_improve_epochs = 0
                else:
                    no_improve_epochs += 1
            write_training_status(
                status_path=status_path,
                training_status="running",
                is_training=True,
                pid=os.getpid(),
                progress_percent=epoch_end_progress,
                current_epoch=epoch,
                total_epochs=args.epochs,
                train_loss=train_loss,
                val_loss=val_loss,
                message=progress_message,
                started_at=started_at,
            )

            # Early stopping: quit if val_loss hasn't improved for `patience` epochs
            if val_loss is not None and no_improve_epochs >= args.patience:
                logger.info(
                    "Early stopping at epoch %d/%d (no val_loss improvement for %d epochs)",
                    epoch, args.epochs, args.patience,
                )
                break

        if best_state_dict is not None:
            model.load_state_dict(best_state_dict)

        write_training_status(
            status_path=status_path,
            training_status="running",
            is_training=True,
            pid=os.getpid(),
            progress_percent=99.0,
            current_epoch=args.epochs,
            total_epochs=args.epochs,
            train_loss=train_loss,
            val_loss=val_loss,
            message="Saving trained detector checkpoint.",
            started_at=started_at,
        )
        save_checkpoint(
            model,
            args.output,
            architecture=args.architecture,
            dataset_size=dataset_size,
            epochs=args.epochs,
            best_val_loss=best_val_loss,
        )
        save_training_metadata(
            output_path=args.output,
            dataset_root=args.dataset_root,
            dataset_size=dataset_size,
            architecture=args.architecture,
            epochs=args.epochs,
            best_val_loss=best_val_loss,
        )
        write_training_status(
            status_path=status_path,
            training_status="completed",
            is_training=False,
            pid=None,
            progress_percent=100.0,
            current_epoch=args.epochs,
            total_epochs=args.epochs,
            train_loss=train_loss,
            val_loss=val_loss,
            message="Latest detector checkpoint is ready to use.",
            started_at=started_at,
        )
        logger.info("Saved trained panel detector checkpoint to %s", args.output)
    except KeyboardInterrupt:
        status_message = "Detector training cancelled."
        if not cancel_requested:
            status_message = "Detector training stopped before finishing."
        write_training_status(
            status_path=status_path,
            training_status="cancelled" if cancel_requested else "failed",
            is_training=False,
            pid=None,
            progress_percent=0.0,
            current_epoch=0,
            total_epochs=args.epochs,
            message=status_message,
            started_at=started_at,
        )
        logger.info(status_message)
        return
    except Exception:
        write_training_status(
            status_path=status_path,
            training_status="failed",
            is_training=False,
            pid=None,
            progress_percent=0.0,
            current_epoch=0,
            total_epochs=args.epochs,
            message="Detector training failed. Check the training log for details.",
            started_at=started_at,
        )
        raise


if __name__ == "__main__":
    main()
