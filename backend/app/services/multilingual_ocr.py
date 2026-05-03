from __future__ import annotations

from dataclasses import dataclass
import os
import re
from threading import Lock
from typing import Any

import numpy as np

from app.services.apple_vision_ocr import AppleVisionOCRService
from app.services.dialogue_cleaner import DialogueCleaner

os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")


@dataclass(slots=True)
class OCRFragment:
    bbox: list[int]
    text: str
    confidence: float | None


class MultilingualOCRService:
    _OCR: Any | None = None
    _LOAD_LOCK = Lock()

    def __init__(self, cleaner: DialogueCleaner | None = None) -> None:
        self.cleaner = cleaner or DialogueCleaner()
        self.apple_vision = AppleVisionOCRService(self.cleaner)

    def extract(self, image: np.ndarray, language_hint: str = "en") -> list[OCRFragment]:
        fragments: list[OCRFragment] = []
        if self._has_paddleocr():
            try:
                results = self._get_ocr().ocr(image)
            except Exception:
                results = None
            if results is not None:
                for entry in self._extract_entries(results):
                    bbox = self._normalise_bbox(entry["bbox"], image.shape[1], image.shape[0])
                    if bbox is None:
                        continue
                    cleaned = self.cleaner.clean_text(entry["text"])
                    if not self._is_usable_fragment(cleaned):
                        continue
                    fragments.append(
                        OCRFragment(
                            bbox=bbox,
                            text=cleaned,
                            confidence=entry["confidence"],
                        )
                    )

        should_try_apple_vision = (
            not fragments
            or len(fragments) < 3
            or sum(len(fragment.text) for fragment in fragments) < 40
        )
        if should_try_apple_vision and self.apple_vision.is_available():
            for fragment in self.apple_vision.extract(image, language_hint):
                bbox = self._normalise_bbox(fragment.bbox, image.shape[1], image.shape[0])
                if bbox is None:
                    continue
                cleaned = self.cleaner.clean_text(fragment.text)
                if not self._is_usable_fragment(cleaned):
                    continue
                fragments.append(
                    OCRFragment(
                        bbox=bbox,
                        text=cleaned,
                        confidence=fragment.confidence,
                    )
                )
        return self._dedupe(fragments)

    def recognize(self, image: np.ndarray, language_hint: str = "en") -> tuple[str, float | None]:
        fragments = self.extract(image, language_hint=language_hint)
        if not fragments:
            return "", None
        text = self.cleaner.clean_text(" ".join(fragment.text for fragment in fragments))
        scores = [float(fragment.confidence) for fragment in fragments if isinstance(fragment.confidence, (int, float))]
        return text, (sum(scores) / len(scores) if scores else None)

    def _get_ocr(self):
        cached = self.__class__._OCR
        if cached is not None:
            return cached
        with self.__class__._LOAD_LOCK:
            cached = self.__class__._OCR
            if cached is not None:
                return cached
            from paddleocr import PaddleOCR

            try:
                instance = PaddleOCR(
                    lang="ch",
                    text_detection_model_name="PP-OCRv5_mobile_det",
                    text_recognition_model_name="PP-OCRv5_mobile_rec",
                    use_doc_orientation_classify=False,
                    use_doc_unwarping=False,
                    use_textline_orientation=False,
                    textline_orientation_batch_size=1,
                    text_recognition_batch_size=4,
                    text_det_limit_side_len=960,
                    text_det_limit_type="max",
                    text_rec_score_thresh=0.42,
                )
            except Exception:
                instance = PaddleOCR(lang="ch")
            self.__class__._OCR = instance
            return instance

    def _extract_entries(self, results: Any) -> list[dict[str, Any]]:
        entries: list[dict[str, Any]] = []

        def add_entry(points: Any, text: Any, score: Any) -> None:
            if points is None:
                return
            try:
                xs = [int(point[0]) for point in points]
                ys = [int(point[1]) for point in points]
            except Exception:
                return
            entries.append(
                {
                    "bbox": [min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys)],
                    "text": str(text or ""),
                    "confidence": float(score) if isinstance(score, (int, float)) else None,
                }
            )

        def walk(node: Any) -> None:
            if isinstance(node, dict):
                polys = node.get("dt_polys", []) or []
                rec_texts = node.get("rec_texts", []) or []
                rec_scores = node.get("rec_scores", []) or []
                for index, poly in enumerate(polys):
                    text = rec_texts[index] if index < len(rec_texts) else ""
                    score = rec_scores[index] if index < len(rec_scores) else None
                    add_entry(poly, text, score)
                for value in node.values():
                    walk(value)
                return
            if isinstance(node, (list, tuple)):
                if len(node) == 2 and isinstance(node[1], (list, tuple)) and len(node[1]) == 2 and isinstance(node[1][0], str):
                    add_entry(node[0], node[1][0], node[1][1])
                    return
                for item in node:
                    walk(item)

        walk(results)
        return entries

    def _normalise_bbox(self, bbox: list[int], image_width: int, image_height: int) -> list[int] | None:
        x, y, width, height = [int(value) for value in bbox]
        x = max(x, 0)
        y = max(y, 0)
        width = min(width, image_width - x)
        height = min(height, image_height - y)
        if width < 18 or height < 18:
            return None
        if width * height > image_width * image_height * 0.7:
            return None
        return [x, y, width, height]

    def _dedupe(self, fragments: list[OCRFragment]) -> list[OCRFragment]:
        deduped: list[OCRFragment] = []
        for fragment in fragments:
            duplicate_index: int | None = None
            for index, other in enumerate(deduped):
                same_text = fragment.text.casefold() == other.text.casefold()
                overlaps = self._iou(tuple(fragment.bbox), tuple(other.bbox)) >= 0.55
                contains = self._bbox_contains(tuple(fragment.bbox), tuple(other.bbox)) or self._bbox_contains(
                    tuple(other.bbox), tuple(fragment.bbox)
                )
                text_nested = self._texts_are_near_duplicates(fragment.text, other.text)
                if same_text or ((overlaps or contains) and text_nested):
                    duplicate_index = index
                    break
            if duplicate_index is None:
                deduped.append(fragment)
                continue
            other = deduped[duplicate_index]
            fragment_score = self._quality_score(fragment.text, fragment.confidence)
            other_score = self._quality_score(other.text, other.confidence)
            if fragment_score > other_score + 0.05:
                deduped[duplicate_index] = fragment
        return deduped

    def _has_paddleocr(self) -> bool:
        try:
            import paddleocr  # noqa: F401

            return True
        except Exception:
            return False

    def _is_usable_fragment(self, text: str) -> bool:
        cleaned = self.cleaner.clean_text(text)
        if self.cleaner.is_usable(cleaned):
            return True
        tokens = [
            token
            for token in cleaned.split()
            if any(character.isalpha() for character in token)
        ]
        if len(tokens) >= 2 and sum(len(token) for token in tokens) >= 6:
            return True
        return len(tokens) == 1 and len(tokens[0]) >= 4

    def _quality_score(self, text: str, confidence: float | None) -> float:
        cleaned = self.cleaner.clean_text(text)
        if not cleaned or not self._is_usable_fragment(cleaned):
            return -999.0
        tokens = re.findall(r"[A-Za-zÀ-ÖØ-öø-ÿĀ-žƀ-ɏḀ-ỿ\u3040-\u30ff\u4e00-\u9fff\uac00-\ud7af']+", cleaned)
        latin_tokens = re.findall(r"[A-Za-zÀ-ÖØ-öø-ÿĀ-žƀ-ɏḀ-ỿ']+", cleaned)
        letters = sum(char.isalpha() for char in cleaned)
        score = 0.0
        score += min(len(cleaned), 48) * 0.03
        score += min(len(tokens), 5) * 0.22
        score += 0.45 if len(tokens) >= 2 else 0.0
        score += 0.18 if any(len(token) >= 4 for token in tokens) else 0.0
        score += 0.08 if cleaned[-1:] in {".", "!", "?"} else 0.0
        if confidence is not None:
            score += max(min(float(confidence), 1.0), 0.0) * 1.6
        else:
            score += 0.2
        if len(tokens) == 1 and len(tokens[0]) <= 3:
            score -= 0.35
        if letters >= 5 and latin_tokens and not re.search(r"[aeiouyAEIOUY]", " ".join(latin_tokens)):
            score -= 0.3
        return score

    def _texts_are_near_duplicates(self, first: str, second: str) -> bool:
        left = re.sub(r"[^a-z0-9\u3040-\u30ff\u4e00-\u9fff\uac00-\ud7af]+", "", first.casefold())
        right = re.sub(r"[^a-z0-9\u3040-\u30ff\u4e00-\u9fff\uac00-\ud7af]+", "", second.casefold())
        if not left or not right:
            return False
        if left == right:
            return True
        shorter, longer = (left, right) if len(left) <= len(right) else (right, left)
        return len(shorter) >= 4 and shorter in longer

    def _iou(self, first: tuple[int, int, int, int], second: tuple[int, int, int, int]) -> float:
        fx, fy, fw, fh = first
        sx, sy, sw, sh = second
        x1 = max(fx, sx)
        y1 = max(fy, sy)
        x2 = min(fx + fw, sx + sw)
        y2 = min(fy + fh, sy + sh)
        if x2 <= x1 or y2 <= y1:
            return 0.0
        intersection = (x2 - x1) * (y2 - y1)
        union = (fw * fh) + (sw * sh) - intersection
        return intersection / union if union else 0.0

    def _bbox_contains(self, outer: tuple[int, int, int, int], inner: tuple[int, int, int, int]) -> bool:
        ox, oy, ow, oh = outer
        ix, iy, iw, ih = inner
        return ox <= ix and oy <= iy and ox + ow >= ix + iw and oy + oh >= iy + ih
