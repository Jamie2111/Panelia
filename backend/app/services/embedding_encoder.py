from __future__ import annotations

import logging
import math
from pathlib import Path
from threading import Lock
from typing import Any

import numpy as np
from PIL import Image

from app.core.config import get_settings

logger = logging.getLogger(__name__)


class EmbeddingEncoder:
    _CLIP_MODEL: Any | None = None
    _CLIP_PROCESSOR: Any | None = None
    _CLIP_DEVICE = "cpu"
    _LOAD_LOCK = Lock()

    def __init__(self) -> None:
        self.settings = get_settings()

    def encode(
        self,
        page_paths: list[Path],
        detections: list[dict[str, Any]],
    ) -> list[np.ndarray]:
        if not detections:
            return []
        image_cache: dict[int, Image.Image] = {}
        samples: list[Image.Image] = []
        sample_indexes: list[int] = []
        for index, detection in enumerate(detections):
            page = int(detection.get("page") or 0)
            if page <= 0 or page > len(page_paths):
                continue
            image = image_cache.get(page)
            if image is None:
                try:
                    image = Image.open(page_paths[page - 1]).convert("RGB")
                except Exception:
                    continue
                image_cache[page] = image
            bbox = detection.get("bbox")
            crop = self._crop_character(image, bbox)
            if crop is None:
                continue
            samples.append(crop)
            sample_indexes.append(index)

        embeddings = [self._fallback_embedding(detection.get("bbox")) for detection in detections]
        if not samples:
            return embeddings

        model_bundle = self._load_clip_bundle()
        if model_bundle is None:
            return embeddings

        processor, model, device = model_bundle
        import torch

        batch_size = 8
        encoded_vectors: list[np.ndarray] = []
        with torch.inference_mode():
            for start in range(0, len(samples), batch_size):
                batch = samples[start : start + batch_size]
                inputs = processor(images=batch, return_tensors="pt", padding=True)
                inputs = {key: value.to(device) if hasattr(value, "to") else value for key, value in inputs.items()}
                outputs = model.get_image_features(**inputs)
                normalized = outputs / outputs.norm(p=2, dim=-1, keepdim=True).clamp(min=1e-6)
                encoded_vectors.extend(normalized.detach().cpu().numpy())

        for index, vector in zip(sample_indexes, encoded_vectors, strict=False):
            embeddings[index] = self._normalize_vector(vector.astype(np.float32, copy=False))
        return embeddings

    def _load_clip_bundle(self) -> tuple[Any, Any, str] | None:
        if self.__class__._CLIP_MODEL is not None and self.__class__._CLIP_PROCESSOR is not None:
            return self.__class__._CLIP_PROCESSOR, self.__class__._CLIP_MODEL, self.__class__._CLIP_DEVICE
        with self.__class__._LOAD_LOCK:
            if self.__class__._CLIP_MODEL is not None and self.__class__._CLIP_PROCESSOR is not None:
                return self.__class__._CLIP_PROCESSOR, self.__class__._CLIP_MODEL, self.__class__._CLIP_DEVICE
            try:
                from transformers import CLIPModel, CLIPProcessor
                import torch

                self.__class__._CLIP_PROCESSOR = CLIPProcessor.from_pretrained(self.settings.clip_model_id)
                self.__class__._CLIP_MODEL = CLIPModel.from_pretrained(self.settings.clip_model_id)
                self.__class__._CLIP_DEVICE = self._resolve_torch_device(torch)
                if hasattr(self.__class__._CLIP_MODEL, "to"):
                    self.__class__._CLIP_MODEL = self.__class__._CLIP_MODEL.to(self.__class__._CLIP_DEVICE)
                self.__class__._CLIP_MODEL.eval()
            except Exception as exc:
                logger.warning("Embedding encoder fell back to geometric vectors because CLIP could not load: %s", exc)
                self.__class__._CLIP_MODEL = None
                self.__class__._CLIP_PROCESSOR = None
                self.__class__._CLIP_DEVICE = "cpu"
                return None
        return self.__class__._CLIP_PROCESSOR, self.__class__._CLIP_MODEL, self.__class__._CLIP_DEVICE

    def _resolve_torch_device(self, torch_module: Any) -> str:
        if torch_module.cuda.is_available():
            return "cuda"
        mps_backend = getattr(torch_module.backends, "mps", None)
        if mps_backend is not None and mps_backend.is_available():
            return "mps"
        return "cpu"

    def _crop_character(self, image: Image.Image, bbox: Any) -> Image.Image | None:
        if not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
            return None
        try:
            x, y, width, height = [int(round(float(item))) for item in bbox[:4]]
        except (TypeError, ValueError):
            return None
        page_width, page_height = image.size
        x0 = max(0, min(x, page_width - 1))
        y0 = max(0, min(y, page_height - 1))
        x1 = max(x0 + 1, min(x + max(width, 1), page_width))
        y1 = max(y0 + 1, min(y + max(height, 1), page_height))
        if x1 <= x0 or y1 <= y0:
            return None
        return image.crop((x0, y0, x1, y1)).resize((224, 224))

    def _fallback_embedding(self, bbox: Any) -> np.ndarray:
        if not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
            return np.zeros(6, dtype=np.float32)
        try:
            x, y, width, height = [float(item) for item in bbox[:4]]
        except (TypeError, ValueError):
            return np.zeros(6, dtype=np.float32)
        vector = np.array([x, y, width, height, x + width / 2, y + height / 2], dtype=np.float32)
        return self._normalize_vector(vector)

    def _normalize_vector(self, vector: np.ndarray) -> np.ndarray:
        norm = float(np.linalg.norm(vector))
        if not math.isfinite(norm) or norm <= 0.0:
            return vector.astype(np.float32, copy=False)
        return (vector / norm).astype(np.float32, copy=False)
