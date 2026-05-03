from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from app.core.config import get_settings
from app.utils.files import read_json, write_json

logger = logging.getLogger(__name__)


class AnimeFaceDetectionService:
    """Lightweight manga/anime face detector built around Nagadomi's LBP cascade.

    This is intentionally additive. MAGI still provides page structure and broad
    character boxes; these face crops give the clustering/review step sharper,
    cheaper anchors when MAGI misses close-ups or only finds full bodies.
    """

    provider = "animeface-lbp-v1"

    def __init__(self) -> None:
        self.settings = get_settings()
        self._cascade: Any | None = None

    def is_available(self) -> bool:
        if not self.settings.anime_face_detection_enabled:
            return False
        if not Path(self.settings.anime_face_cascade_path).exists():
            logger.warning("Anime face cascade not found at %s", self.settings.anime_face_cascade_path)
            return False
        try:
            import cv2  # noqa: F401
        except Exception as exc:
            logger.warning("OpenCV is unavailable; anime face detection disabled: %s", exc)
            return False
        return self._load_cascade() is not None

    def detect_page_payloads(
        self,
        page_paths: list[Path],
        *,
        page_numbers: list[int] | None = None,
        cache_path: Path | None = None,
        cancel_callback: callable | None = None,
        progress_callback: callable | None = None,
        progress_label: str = "Finding manga face crops",
    ) -> dict[int, dict[str, Any]]:
        if not self.is_available():
            return {}

        selected_pages = sorted(
            {
                int(page_number)
                for page_number in (page_numbers or range(1, len(page_paths) + 1))
                if 1 <= int(page_number) <= len(page_paths)
            }
        )
        if not selected_pages:
            return {}

        cached = read_json(cache_path, default={}) if cache_path else {}
        if not isinstance(cached, dict):
            cached = {}

        payloads: dict[int, dict[str, Any]] = {}
        updated_cache = dict(cached)
        total = max(len(selected_pages), 1)
        for index, page_number in enumerate(selected_pages, start=1):
            if cancel_callback:
                cancel_callback()
            page_path = page_paths[page_number - 1]
            signature = self._page_signature(page_path)
            cached_payload = cached.get(str(page_number)) or cached.get(page_number)
            if (
                isinstance(cached_payload, dict)
                and cached_payload.get("provider") == self.provider
                and cached_payload.get("page_signature") == signature
                and isinstance(cached_payload.get("characters"), list)
            ):
                payloads[int(page_number)] = cached_payload
            else:
                payload = self._detect_page(page_path, page_number)
                payload["page_signature"] = signature
                payloads[int(page_number)] = payload
                updated_cache[str(int(page_number))] = payload

            if progress_callback:
                progress_callback(
                    (index / total) * 100.0,
                    f"{progress_label} ({index}/{total} pages scanned)",
                )

        if cache_path:
            write_json(cache_path, updated_cache)
        return payloads

    def _load_cascade(self) -> Any | None:
        if self._cascade is not None:
            return self._cascade
        try:
            import cv2
        except Exception:
            return None
        cascade = cv2.CascadeClassifier(str(self.settings.anime_face_cascade_path))
        if cascade.empty():
            logger.warning("OpenCV could not load anime face cascade at %s", self.settings.anime_face_cascade_path)
            return None
        self._cascade = cascade
        return self._cascade

    def _detect_page(self, page_path: Path, page_number: int) -> dict[str, Any]:
        cascade = self._load_cascade()
        if cascade is None:
            return self._empty_payload(page_number)
        try:
            import cv2

            with Image.open(page_path) as source:
                image = source.convert("RGB")
                original_width, original_height = image.size
                resized, scale = self._resize_for_detection(image)
                gray = cv2.cvtColor(np.array(resized), cv2.COLOR_RGB2GRAY)
                gray = cv2.equalizeHist(gray)
                min_size = max(12, int(round(float(self.settings.anime_face_min_size or 24) * scale)))
                raw_boxes = cascade.detectMultiScale(
                    gray,
                    scaleFactor=max(float(self.settings.anime_face_scale_factor or 1.08), 1.01),
                    minNeighbors=max(int(self.settings.anime_face_min_neighbors or 4), 1),
                    minSize=(min_size, min_size),
                )
        except Exception as exc:
            logger.warning("Anime face detection failed for %s: %s", page_path, exc)
            return self._empty_payload(page_number)

        boxes = []
        for raw_box in raw_boxes:
            x, y, width, height = [int(round(float(value) / scale)) for value in raw_box[:4]]
            clipped = self._clip_box([x, y, width, height], original_width, original_height)
            if clipped is not None:
                boxes.append(clipped)
        boxes = self._dedupe_boxes(boxes)

        characters = [
            {
                "character_index": index - 1,
                "character_id": f"animeface-p{page_number:04d}-face-{index:03d}",
                "bbox": box,
                "source": "animeface-lbp",
                "confidence": None,
            }
            for index, box in enumerate(boxes, start=1)
        ]
        return {
            "page": int(page_number),
            "provider": self.provider,
            "panels": [],
            "texts": [],
            "characters": characters,
        }

    def _resize_for_detection(self, image: Image.Image) -> tuple[Image.Image, float]:
        max_edge = max(int(self.settings.anime_face_max_image_edge or 1400), 320)
        width, height = image.size
        edge = max(width, height)
        if edge <= max_edge:
            return image, 1.0
        scale = max_edge / float(edge)
        resized = image.resize(
            (max(1, int(round(width * scale))), max(1, int(round(height * scale)))),
            Image.Resampling.LANCZOS,
        )
        return resized, scale

    def _page_signature(self, page_path: Path) -> dict[str, Any]:
        try:
            stat = page_path.stat()
        except OSError:
            return {"path": str(page_path), "size": 0, "mtime_ns": 0}
        return {
            "path": page_path.name,
            "size": int(stat.st_size),
            "mtime_ns": int(stat.st_mtime_ns),
        }

    def _empty_payload(self, page_number: int) -> dict[str, Any]:
        return {
            "page": int(page_number),
            "provider": self.provider,
            "panels": [],
            "texts": [],
            "characters": [],
        }

    def _clip_box(self, box: list[int], image_width: int, image_height: int) -> list[int] | None:
        x, y, width, height = box[:4]
        if width <= 0 or height <= 0:
            return None
        left = min(max(0, int(x)), image_width - 1)
        top = min(max(0, int(y)), image_height - 1)
        right = min(max(left + 1, int(x + width)), image_width)
        bottom = min(max(top + 1, int(y + height)), image_height)
        clipped_width = right - left
        clipped_height = bottom - top
        min_side = max(int(self.settings.anime_face_min_size or 24), 12)
        if clipped_width < min_side or clipped_height < min_side:
            return None
        image_area = max(image_width * image_height, 1)
        box_area = clipped_width * clipped_height
        if box_area > image_area * 0.18:
            return None
        return [left, top, clipped_width, clipped_height]

    def _dedupe_boxes(self, boxes: list[list[int]]) -> list[list[int]]:
        kept: list[list[int]] = []
        for box in sorted(boxes, key=lambda item: (item[1], item[0], -item[2] * item[3])):
            if any(self._iou(box, existing) >= 0.35 for existing in kept):
                continue
            kept.append(box)
        return kept

    def _iou(self, left: list[int], right: list[int]) -> float:
        lx, ly, lw, lh = left[:4]
        rx, ry, rw, rh = right[:4]
        left_x2 = lx + lw
        left_y2 = ly + lh
        right_x2 = rx + rw
        right_y2 = ry + rh
        inter_w = max(0, min(left_x2, right_x2) - max(lx, rx))
        inter_h = max(0, min(left_y2, right_y2) - max(ly, ry))
        inter_area = inter_w * inter_h
        if inter_area <= 0:
            return 0.0
        union = (lw * lh) + (rw * rh) - inter_area
        return inter_area / float(max(union, 1))
