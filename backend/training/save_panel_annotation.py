from __future__ import annotations

from datetime import datetime, timezone
import hashlib
from pathlib import Path
from typing import Any, Iterable

from PIL import Image

from app.core.config import get_settings
from app.schemas.project import PanelBox
from app.utils.files import ensure_dir, write_json


def _normalize_box(box: PanelBox | dict[str, Any]) -> dict[str, int]:
    if isinstance(box, PanelBox):
        x = int(box.x)
        y = int(box.y)
        width = int(box.width)
        height = int(box.height)
    else:
        x = int(box.get("x", 0))
        y = int(box.get("y", 0))
        width = int(box.get("width", box.get("w", 0)))
        height = int(box.get("height", box.get("h", 0)))

    return {
        "x": max(x, 0),
        "y": max(y, 0),
        "w": max(width, 1),
        "h": max(height, 1),
    }


def _stable_image_name(image_path: Path) -> tuple[str, str]:
    image_bytes = image_path.read_bytes()
    image_hash = hashlib.sha256(image_bytes).hexdigest()
    return f"page_{image_hash[:16]}.png", image_hash


def save_panel_annotation(
    image_path: str | Path,
    panel_boxes: Iterable[PanelBox | dict[str, Any]],
    *,
    dataset_root: str | Path | None = None,
    image_name: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, str]:
    """
    Save a human-corrected page image plus its panel annotations.

    The saved dataset is page-level and intentionally simple so it can be
    reused later for real supervised training.
    """
    settings = get_settings()
    source_path = Path(image_path)
    root = Path(dataset_root) if dataset_root is not None else settings.training_data_dir
    images_dir = ensure_dir(root / "images")
    annotations_dir = ensure_dir(root / "annotations")

    resolved_image_name, image_hash = _stable_image_name(source_path)
    final_image_name = image_name or resolved_image_name
    if not final_image_name.lower().endswith(".png"):
        final_image_name = f"{Path(final_image_name).stem}.png"

    image_output_path = images_dir / final_image_name
    annotation_output_path = annotations_dir / f"{Path(final_image_name).stem}.json"

    with Image.open(source_path) as image:
        rgb = image.convert("RGB")
        rgb.save(image_output_path, format="PNG", compress_level=1)
        image_width, image_height = rgb.size

    normalized_boxes = [_normalize_box(box) for box in panel_boxes]
    annotation_payload: dict[str, Any] = {
        "image": final_image_name,
        "panels": normalized_boxes,
        "image_width": image_width,
        "image_height": image_height,
        "image_hash": image_hash,
        "saved_at": datetime.now(timezone.utc).isoformat(),
    }
    if metadata:
        annotation_payload["metadata"] = metadata

    write_json(annotation_output_path, annotation_payload)
    return {
        "image_path": str(image_output_path),
        "annotation_path": str(annotation_output_path),
        "image_name": final_image_name,
    }

