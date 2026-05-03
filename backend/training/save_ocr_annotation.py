from __future__ import annotations

from hashlib import sha256
from pathlib import Path
from typing import Any

from PIL import Image

from app.core.config import get_settings
from app.schemas.project import PanelBox
from app.utils.files import ensure_dir, write_json


def _panel_dict(panel_box: PanelBox | dict[str, Any]) -> dict[str, int]:
    if isinstance(panel_box, PanelBox):
        return {
            "x": int(panel_box.x),
            "y": int(panel_box.y),
            "w": int(panel_box.width),
            "h": int(panel_box.height),
        }
    return {
        "x": int(panel_box["x"]),
        "y": int(panel_box["y"]),
        "w": int(panel_box["w"]),
        "h": int(panel_box["h"]),
    }


def save_ocr_annotation(
    image_path: str | Path,
    panel_box: PanelBox | dict[str, Any],
    *,
    corrected_text: str,
    original_text: str | None = None,
    dataset_root: str | Path | None = None,
    image_name: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    settings = get_settings()
    root = ensure_dir(Path(dataset_root) if dataset_root is not None else settings.ocr_training_data_dir)
    images_dir = ensure_dir(root / "images")
    annotations_dir = ensure_dir(root / "annotations")

    image_path = Path(image_path)
    panel = _panel_dict(panel_box)
    corrected = str(corrected_text or "").strip()
    if not corrected:
        raise ValueError("corrected_text must not be empty")

    with Image.open(image_path) as image:
        rgb = image.convert("RGB")
        x = max(0, int(panel["x"]))
        y = max(0, int(panel["y"]))
        w = max(1, int(panel["w"]))
        h = max(1, int(panel["h"]))
        right = min(rgb.width, x + w)
        bottom = min(rgb.height, y + h)
        if right <= x or bottom <= y:
            raise ValueError("panel_box does not describe a valid crop")
        crop = rgb.crop((x, y, right, bottom))

        output_name = image_name or f"{image_path.stem}_ocr_{sha256(f'{image_path}:{x}:{y}:{w}:{h}'.encode('utf-8')).hexdigest()[:12]}.png"
        output_image = images_dir / output_name
        crop.save(output_image, format="PNG", compress_level=1)

        digest = sha256(output_image.read_bytes()).hexdigest()
        annotation = {
            "image": output_image.name,
            "source_image": image_path.name,
            "panel_box": {
                "x": x,
                "y": y,
                "w": int(right - x),
                "h": int(bottom - y),
            },
            "text_original": str(original_text or "").strip(),
            "text_corrected": corrected,
            "crop_width": crop.width,
            "crop_height": crop.height,
            "image_hash": digest,
            "metadata": metadata or {},
        }

    annotation_path = annotations_dir / f"{Path(output_name).stem}.json"
    write_json(annotation_path, annotation)
    return annotation
