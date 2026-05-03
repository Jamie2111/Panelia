from __future__ import annotations

import platform
from dataclasses import dataclass

import numpy as np
from PIL import Image

from app.core.config import get_settings
from app.services.dialogue_cleaner import DialogueCleaner


@dataclass(slots=True)
class AppleVisionOCRFragment:
    bbox: list[int]
    text: str
    confidence: float | None


class AppleVisionOCRService:
    def __init__(self, cleaner: DialogueCleaner | None = None) -> None:
        self.cleaner = cleaner or DialogueCleaner()
        self.settings = get_settings()

    def is_available(self) -> bool:
        if not self.settings.apple_vision_ocr_enabled:
            return False
        if platform.system() != "Darwin":
            return False
        try:
            import ocrmac  # noqa: F401

            return True
        except Exception:
            return False

    def extract(self, image: np.ndarray, language_hint: str) -> list[AppleVisionOCRFragment]:
        if image.size == 0 or not self.is_available():
            return []

        annotations = self._recognize_annotations(image, language_hint, framework="livetext")
        if not annotations:
            annotations = self._recognize_annotations(image, language_hint, framework="vision")
        if not annotations:
            return []

        image_height, image_width = image.shape[:2]
        fragments: list[AppleVisionOCRFragment] = []
        for item in annotations:
            if not isinstance(item, (list, tuple)) or len(item) < 3:
                continue
            text = self.cleaner.clean_text(str(item[0] or ""))
            if not self._looks_like_meaningful_text(text):
                continue
            bbox = self._normalized_bbox_to_xywh(item[2], image_width, image_height)
            if bbox is None:
                continue
            confidence = float(item[1]) if isinstance(item[1], (int, float)) else None
            fragments.append(AppleVisionOCRFragment(bbox=bbox, text=text, confidence=confidence))
        return self._dedupe_and_sort(fragments)

    def recognize(self, image: np.ndarray, language_hint: str) -> tuple[str, float | None]:
        fragments = self.extract(image, language_hint)
        if not fragments:
            return "", None
        text = self.cleaner.clean_text(" ".join(fragment.text for fragment in fragments))
        confidences = [
            float(fragment.confidence)
            for fragment in fragments
            if isinstance(fragment.confidence, (int, float))
        ]
        confidence = sum(confidences) / len(confidences) if confidences else None
        return text, confidence

    def _recognize_annotations(
        self,
        image: np.ndarray,
        language_hint: str,
        *,
        framework: str,
    ) -> list[tuple[str, float, tuple[float, float, float, float]]]:
        try:
            from ocrmac import ocrmac
        except Exception:
            return []

        pil_image = Image.fromarray(image)
        kwargs = {
            "framework": framework,
            "recognition_level": "accurate",
            "detail": True,
            "unit": "line",
        }
        language_preference = self._language_preference(language_hint)
        if language_preference:
            kwargs["language_preference"] = language_preference

        try:
            return ocrmac.OCR(pil_image, **kwargs).recognize()
        except Exception:
            if "language_preference" in kwargs:
                kwargs.pop("language_preference", None)
                try:
                    return ocrmac.OCR(pil_image, **kwargs).recognize()
                except Exception:
                    return []
            return []

    def _language_preference(self, language_hint: str) -> list[str] | None:
        normalized = str(language_hint or "").strip().casefold()
        mapping = {
            "en": ["en-US"],
            "ja": ["ja-JP", "en-US"],
            "ko": ["ko-KR", "en-US"],
            "zh": ["zh-Hans", "en-US"],
            "es": ["es-ES", "en-US"],
            "pt": ["pt-BR", "en-US"],
            "de": ["de-DE", "en-US"],
            "fr": ["fr-FR", "en-US"],
            "it": ["it-IT", "en-US"],
            "tr": ["tr-TR", "en-US"],
            "id": ["id-ID", "en-US"],
        }
        return mapping.get(normalized)

    def _normalized_bbox_to_xywh(
        self,
        bbox: tuple[float, float, float, float] | list[float],
        image_width: int,
        image_height: int,
    ) -> list[int] | None:
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            return None
        try:
            x_norm, y_norm, width_norm, height_norm = [float(value) for value in bbox]
        except Exception:
            return None
        x = max(int(round(x_norm * image_width)), 0)
        width = max(int(round(width_norm * image_width)), 1)
        height = max(int(round(height_norm * image_height)), 1)
        # Vision/LiveText coordinates use a bottom-left origin.
        y = max(int(round((1.0 - (y_norm + height_norm)) * image_height)), 0)
        width = min(width, max(image_width - x, 1))
        height = min(height, max(image_height - y, 1))
        if width < 12 or height < 12:
            return None
        if width * height > image_width * image_height * 0.85:
            return None
        return [x, y, width, height]

    def _dedupe_and_sort(self, fragments: list[AppleVisionOCRFragment]) -> list[AppleVisionOCRFragment]:
        deduped: list[AppleVisionOCRFragment] = []
        for fragment in fragments:
            duplicate_index: int | None = None
            for index, other in enumerate(deduped):
                if fragment.text.casefold() == other.text.casefold():
                    duplicate_index = index
                    break
            if duplicate_index is None:
                deduped.append(fragment)
                continue
            other = deduped[duplicate_index]
            fragment_score = fragment.confidence if fragment.confidence is not None else -1.0
            other_score = other.confidence if other.confidence is not None else -1.0
            if fragment_score > other_score or len(fragment.text) > len(other.text):
                deduped[duplicate_index] = fragment
        return sorted(deduped, key=lambda item: (item.bbox[1], item.bbox[0]))

    def _looks_like_meaningful_text(self, text: str) -> bool:
        if self.cleaner.is_usable(text):
            return True
        tokens = [
            token
            for token in text.split()
            if any(character.isalpha() for character in token)
        ]
        if len(tokens) >= 2 and sum(len(token) for token in tokens) >= 6:
            return True
        return len(tokens) == 1 and len(tokens[0]) >= 4
