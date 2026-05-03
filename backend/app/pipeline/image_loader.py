from __future__ import annotations

import hashlib
import io
from pathlib import Path
from typing import Iterable

from PIL import Image, ImageOps

from app.schemas.project import PanelBox
from app.utils.files import ensure_dir


class ImageLoader:
    """Shared image loading utilities for panel and page vision passes."""

    def __init__(self, *, project_dir: Path, page_paths: list[Path], max_edge: int = 1024) -> None:
        self.project_dir = project_dir
        self.page_paths = list(page_paths)
        self.max_edge = max(256, int(max_edge))
        self._thumbnail_dir = ensure_dir(project_dir / "temp" / "vision_thumbnails")

    def page_path(self, page_number: int) -> Path | None:
        index = int(page_number) - 1
        if index < 0 or index >= len(self.page_paths):
            return None
        return self.page_paths[index]

    def panel_image_path(self, panel: PanelBox) -> Path | None:
        candidates = [
            self.project_dir / "panels" / f"panel_{int(panel.order):03d}.png",
            self.project_dir / "panels" / f"{panel.id}.png",
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate

        page_path = self.page_path(int(panel.page))
        if page_path is None or not page_path.exists():
            return None

        crop_name = f"{panel.id}.png"
        crop_path = self._thumbnail_dir / "panel_crops" / crop_name
        if crop_path.exists():
            return crop_path

        ensure_dir(crop_path.parent)
        with Image.open(page_path) as image:
            image = ImageOps.exif_transpose(image).convert("RGB")
            x = max(int(panel.x), 0)
            y = max(int(panel.y), 0)
            width = max(int(panel.width), 1)
            height = max(int(panel.height), 1)
            crop = image.crop((x, y, x + width, y + height))
            crop.save(crop_path, format="PNG")
        return crop_path

    def page_thumbnail_path(self, page_number: int, *, max_edge: int | None = None) -> Path | None:
        page_path = self.page_path(page_number)
        if page_path is None or not page_path.exists():
            return None
        raw = page_path.read_bytes()
        thumb_hash = self.sha256_bytes(raw)
        ext = ".png" if page_path.suffix.lower() == ".png" else ".jpg"
        thumb_path = self._thumbnail_dir / f"page_{int(page_number):04d}_{thumb_hash[:12]}{ext}"
        if thumb_path.exists():
            return thumb_path
        self._write_resized_copy(page_path, thumb_path, max_edge=max_edge or self.max_edge)
        return thumb_path

    def image_payload(self, image_path: Path, *, max_edge: int | None = None) -> tuple[bytes, str, str]:
        resized = self._resized_bytes(image_path, max_edge=max_edge or self.max_edge)
        suffix = image_path.suffix.lower()
        mime = "image/png" if suffix == ".png" else "image/jpeg"
        return resized, mime, self.sha256_bytes(resized)

    def composite_hash(self, values: Iterable[str]) -> str:
        joined = "|".join(str(value) for value in values)
        return hashlib.sha256(joined.encode("utf-8")).hexdigest()

    @staticmethod
    def sha256_bytes(raw: bytes) -> str:
        return hashlib.sha256(raw).hexdigest()

    def _write_resized_copy(self, source_path: Path, target_path: Path, *, max_edge: int) -> None:
        ensure_dir(target_path.parent)
        data = self._resized_bytes(source_path, max_edge=max_edge)
        target_path.write_bytes(data)

    def _resized_bytes(self, image_path: Path, *, max_edge: int) -> bytes:
        with Image.open(image_path) as image:
            image = ImageOps.exif_transpose(image).convert("RGB")
            image.thumbnail((max_edge, max_edge), Image.Resampling.LANCZOS)
            output = io.BytesIO()
            image.save(output, format="PNG", optimize=True)
            return output.getvalue()
