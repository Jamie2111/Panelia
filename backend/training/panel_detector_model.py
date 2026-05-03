from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path
from threading import Lock
from typing import Any

import numpy as np

from app.core.config import get_settings


logger = logging.getLogger(__name__)
_MODEL_LOCK = Lock()
_CACHED_RUNTIME: "LoadedPanelDetector | None" = None
_CACHED_CHECKPOINT: str | None = None


def resolve_torch_device(prefer_mps: bool = True) -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda"
    mps_backend = getattr(torch.backends, "mps", None)
    if prefer_mps and mps_backend is not None and mps_backend.is_available():
        return "mps"
    return "cpu"


def build_panel_detector_model(
    architecture: str = "fasterrcnn_mobilenet_v3_large_fpn",
    *,
    pretrained: bool = False,
    num_classes: int = 2,
) -> Any:
    if architecture != "fasterrcnn_mobilenet_v3_large_fpn":
        raise ValueError(f"Unsupported panel detector architecture: {architecture}")

    from torchvision.models.detection import (
        FasterRCNN_MobileNet_V3_Large_FPN_Weights,
        fasterrcnn_mobilenet_v3_large_fpn,
    )
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

    if pretrained:
        model = fasterrcnn_mobilenet_v3_large_fpn(
            weights=FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT
        )
    else:
        model = fasterrcnn_mobilenet_v3_large_fpn(weights=None, weights_backbone=None)

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def locate_latest_panel_detector_checkpoint(models_dir: str | Path | None = None) -> Path | None:
    settings = get_settings()
    root = Path(models_dir) if models_dir is not None else settings.panel_detector_models_dir
    checkpoints = sorted(
        root.glob(settings.panel_detector_checkpoint_glob),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    return checkpoints[0] if checkpoints else None


@dataclass
class LoadedPanelDetector:
    model: Any
    device: str
    checkpoint_path: Path
    architecture: str
    score_threshold: float

    def predict(self, image: np.ndarray) -> list[tuple[int, int, int, int, float]]:
        from PIL import Image
        from torchvision.transforms.functional import pil_to_tensor
        import torch

        self.model.eval()
        tensor = pil_to_tensor(Image.fromarray(image).convert("RGB")).float() / 255.0
        with torch.inference_mode():
            prediction = self.model([tensor.to(self.device)])[0]

        boxes = prediction.get("boxes")
        scores = prediction.get("scores")
        labels = prediction.get("labels")
        if boxes is None or scores is None or labels is None:
            return []

        results: list[tuple[int, int, int, int, float]] = []
        for box, score, label in zip(boxes, scores, labels, strict=False):
            if int(label) != 1:
                continue
            score_value = float(score.detach().cpu().item())
            if score_value < self.score_threshold:
                continue
            x1, y1, x2, y2 = [int(round(float(value))) for value in box.detach().cpu().tolist()]
            width = max(1, x2 - x1)
            height = max(1, y2 - y1)
            results.append((x1, y1, width, height, score_value))
        return results


def load_panel_detector_checkpoint(
    checkpoint_path: str | Path,
    *,
    score_threshold: float | None = None,
) -> LoadedPanelDetector:
    import torch

    settings = get_settings()
    checkpoint_path = Path(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    architecture = str(checkpoint.get("architecture", "fasterrcnn_mobilenet_v3_large_fpn"))
    num_classes = int(checkpoint.get("num_classes", 2))
    model = build_panel_detector_model(architecture, pretrained=False, num_classes=num_classes)
    model.load_state_dict(checkpoint["state_dict"])

    device = resolve_torch_device()
    if hasattr(model, "to"):
        model = model.to(device)
    if hasattr(model, "eval"):
        model.eval()

    return LoadedPanelDetector(
        model=model,
        device=device,
        checkpoint_path=checkpoint_path,
        architecture=architecture,
        score_threshold=float(score_threshold if score_threshold is not None else settings.panel_detector_score_threshold),
    )


def load_latest_panel_detector_runtime() -> LoadedPanelDetector | None:
    global _CACHED_CHECKPOINT, _CACHED_RUNTIME

    latest = locate_latest_panel_detector_checkpoint()
    if latest is None:
        return None

    latest_key = f"{latest.resolve()}:{latest.stat().st_mtime_ns}"
    with _MODEL_LOCK:
        if _CACHED_RUNTIME is not None and _CACHED_CHECKPOINT == latest_key:
            return _CACHED_RUNTIME
        try:
            _CACHED_RUNTIME = load_panel_detector_checkpoint(latest)
            _CACHED_CHECKPOINT = latest_key
            logger.info("Loaded trained panel detector checkpoint from %s", latest)
        except Exception:
            logger.exception("Unable to load trained panel detector checkpoint from %s", latest)
            _CACHED_RUNTIME = None
            _CACHED_CHECKPOINT = None
        return _CACHED_RUNTIME
