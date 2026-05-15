from __future__ import annotations

from dataclasses import dataclass
import os
import re
from threading import Lock
from typing import Any

import numpy as np
from PIL import Image

from app.core.config import get_settings
from app.services.apple_vision_ocr import AppleVisionOCRService
from app.services.dialogue_cleaner import DialogueCleaner
from app.services.language_detector import LanguageDetector
from app.services.multilingual_ocr import MultilingualOCRService

os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")


@dataclass(slots=True)
class BubbleOCRResult:
    language: str
    original_text: str
    translated_text: str
    confidence: float | None
    ocr_engine: str


class ComicOCRService:
    _PADDLE_CACHE: dict[str, Any] = {}
    _EASYOCR_CACHE: dict[str, Any] = {}
    _MANGA_OCR: Any | None = None
    _LOAD_LOCK = Lock()

    def __init__(self, cleaner: DialogueCleaner | None = None, language_detector: LanguageDetector | None = None) -> None:
        self.settings = get_settings()
        self.cleaner = cleaner or DialogueCleaner()
        self.language_detector = language_detector or LanguageDetector(self.cleaner)
        self.multilingual_ocr = MultilingualOCRService(self.cleaner)
        self.apple_vision_ocr = AppleVisionOCRService(self.cleaner)

    def detect_candidates(self, panel_image: np.ndarray, language_hint: str) -> list[dict[str, Any]]:
        candidates: list[dict[str, Any]] = []
        hint = self.language_detector.normalize_language_code(language_hint)
        fragments = self._extract_paddle_fragments(panel_image, hint)
        if not fragments:
            fragments = [
                {
                    "bbox": fragment.bbox,
                    "text": fragment.text,
                    "confidence": fragment.confidence,
                    "ocr_engine": "paddleocr-multilingual",
                }
                for fragment in self.multilingual_ocr.extract(panel_image)
            ]

        fallback_attempts = 0
        max_fallback_attempts = 12
        for fragment in fragments:
            bbox = self._normalise_bbox(fragment["bbox"], panel_image.shape[1], panel_image.shape[0])
            if bbox is None:
                continue
            text = self.cleaner.clean_text(str(fragment.get("text") or ""))
            confidence = fragment.get("confidence")
            engine = str(fragment.get("ocr_engine") or "paddleocr").strip() or "paddleocr"
            if self._needs_fallback(text, confidence) and fallback_attempts < max_fallback_attempts:
                fallback_attempts += 1
                crop = self._crop_box(
                    panel_image,
                    self._expand_bbox(bbox, panel_image.shape[1], panel_image.shape[0]),
                )
                fallback = self.recognize_region(crop, language_hint, fast_only=True)
                if self._is_better_candidate(fallback.original_text, fallback.confidence, text, confidence):
                    text = fallback.original_text
                    confidence = fallback.confidence
                    engine = fallback.ocr_engine
            if not self._acceptable_ocr_fragment(text):
                continue
            candidates.append(
                {
                    "bbox": bbox,
                    "text": text,
                    "confidence": confidence,
                    "ocr_engine": engine,
                }
            )
        return self._dedupe_candidates(candidates, panel_image.shape[1], panel_image.shape[0])

    def _acceptable_ocr_fragment(self, text: str) -> bool:
        cleaned = self.cleaner.clean_text(text)
        if not cleaned:
            return False
        if self.cleaner.is_usable(cleaned):
            return True
        tokens = re.findall(r"[A-Za-zÀ-ÖØ-öø-ÿĀ-žƀ-ɏḀ-ỿ\u3040-\u30ff\u4e00-\u9fff\uac00-\ud7af']+", cleaned)
        if len(tokens) >= 2 and sum(len(token) for token in tokens) >= 6:
            return True
        return len(tokens) == 1 and len(tokens[0]) >= 4

    def recognize_region(
        self,
        region_image: np.ndarray,
        language_hint: str,
        *,
        use_apple_vision: bool = True,
        fast_only: bool = False,
    ) -> BubbleOCRResult:
        hint = self.language_detector.normalize_language_code(language_hint)
        best_text = ""
        best_confidence: float | None = None
        best_engine = "none"
        variants = self._ocr_variants(region_image)
        primary_variant = variants[0]
        secondary_variants = variants[1:2] if fast_only else variants[1:]

        engine_order = (
            ["manga-ocr", "paddleocr", "apple-vision", "easyocr"]
            if hint == "ja"
            else ["apple-vision", "paddleocr", "easyocr", "manga-ocr"]
        )
        if not use_apple_vision:
            engine_order = [engine for engine in engine_order if engine != "apple-vision"]
        if fast_only:
            engine_order = [engine for engine in engine_order if engine in {"apple-vision", "paddleocr", "manga-ocr"}]
        for engine in engine_order:
            if engine == "manga-ocr" and hint == "ja" and self._has_manga_ocr():
                for candidate in [primary_variant]:
                    try:
                        text = self.cleaner.clean_text(self._manga_ocr_read(candidate))
                    except Exception:
                        continue
                    if self._is_better_candidate(text, None, best_text, best_confidence):
                        best_text = text
                        best_confidence = None
                        best_engine = "manga-ocr"
                if self._needs_fallback(best_text, best_confidence):
                    for candidate in secondary_variants:
                        try:
                            text = self.cleaner.clean_text(self._manga_ocr_read(candidate))
                        except Exception:
                            continue
                        if self._is_better_candidate(text, None, best_text, best_confidence):
                            best_text = text
                            best_confidence = None
                            best_engine = "manga-ocr"
            elif engine == "paddleocr" and self._has_paddleocr():
                for candidate in [primary_variant]:
                    try:
                        text, confidence = self._paddle_recognise(candidate, hint)
                    except Exception:
                        continue
                    text = self.cleaner.clean_text(text)
                    if self._is_better_candidate(text, confidence, best_text, best_confidence):
                        best_text = text
                        best_confidence = confidence
                        best_engine = "paddleocr"
                if self._needs_fallback(best_text, best_confidence):
                    for candidate in secondary_variants:
                        try:
                            text, confidence = self._paddle_recognise(candidate, hint)
                        except Exception:
                            continue
                        text = self.cleaner.clean_text(text)
                        if self._is_better_candidate(text, confidence, best_text, best_confidence):
                            best_text = text
                            best_confidence = confidence
                            best_engine = "paddleocr"
            elif engine == "easyocr" and self._has_easyocr():
                for candidate in [primary_variant]:
                    try:
                        text, confidence = self._easyocr_recognise(candidate, hint)
                    except Exception:
                        continue
                    text = self.cleaner.clean_text(text)
                    if self._is_better_candidate(text, confidence, best_text, best_confidence):
                        best_text = text
                        best_confidence = confidence
                        best_engine = "easyocr"
                if self._needs_fallback(best_text, best_confidence):
                    for candidate in secondary_variants:
                        try:
                            text, confidence = self._easyocr_recognise(candidate, hint)
                        except Exception:
                            continue
                        text = self.cleaner.clean_text(text)
                        if self._is_better_candidate(text, confidence, best_text, best_confidence):
                            best_text = text
                            best_confidence = confidence
                            best_engine = "easyocr"
            elif engine == "apple-vision" and self._has_apple_vision():
                for candidate in [primary_variant]:
                    try:
                        text, confidence = self._apple_vision_recognise(candidate, hint)
                    except Exception:
                        continue
                    text = self.cleaner.clean_text(text)
                    if self._is_better_candidate(text, confidence, best_text, best_confidence):
                        best_text = text
                        best_confidence = confidence
                        best_engine = "apple-vision"
                if self._needs_fallback(best_text, best_confidence):
                    for candidate in secondary_variants:
                        try:
                            text, confidence = self._apple_vision_recognise(candidate, hint)
                        except Exception:
                            continue
                        text = self.cleaner.clean_text(text)
                        if self._is_better_candidate(text, confidence, best_text, best_confidence):
                            best_text = text
                            best_confidence = confidence
                            best_engine = "apple-vision"
            if best_text and not self._needs_fallback(best_text, best_confidence):
                break

        language = self.language_detector.detect(best_text, hint)
        translated = self.language_detector.translate_to_english(best_text, language)
        return BubbleOCRResult(
            language=language,
            original_text=best_text,
            translated_text=translated,
            confidence=best_confidence,
            ocr_engine=best_engine,
        )

    def recognize_panel_text(self, panel_image: np.ndarray, language_hint: str) -> tuple[str, float | None, str]:
        fragments: list[str] = []
        confidences: list[float] = []
        engines: list[str] = []
        for candidate in self.detect_candidates(panel_image, language_hint):
            text = self.cleaner.clean_text(str(candidate.get("text") or ""))
            if not text or (fragments and text.casefold() == fragments[-1].casefold()):
                continue
            fragments.append(text)
            confidence = candidate.get("confidence")
            if isinstance(confidence, (int, float)):
                confidences.append(float(confidence))
            engine = str(candidate.get("ocr_engine") or "").strip()
            if engine:
                engines.append(engine)
        if not fragments:
            fallback = self.recognize_region(panel_image, language_hint, use_apple_vision=False, fast_only=True)
            return fallback.original_text, fallback.confidence, fallback.ocr_engine
        text = self.cleaner.clean_text(" ".join(fragments))
        confidence = sum(confidences) / len(confidences) if confidences else None
        engine = engines[0] if engines else "paddleocr"
        return text, confidence, engine

    def _needs_fallback(self, text: str, confidence: float | None) -> bool:
        return not self._acceptable_ocr_fragment(text) or confidence is None or confidence < 0.72 or len(text) < 6

    def _is_better_candidate(self, candidate_text: str, candidate_confidence: float | None, best_text: str, best_confidence: float | None) -> bool:
        if not candidate_text or not self._acceptable_ocr_fragment(candidate_text):
            return False
        if not best_text:
            return True
        candidate_score = self._candidate_quality_score(candidate_text, candidate_confidence)
        best_score = self._candidate_quality_score(best_text, best_confidence)
        if candidate_score > best_score + 0.08:
            return True
        if best_score > candidate_score + 0.08:
            return False
        if candidate_confidence is not None and best_confidence is not None:
            if candidate_confidence > best_confidence + 0.04:
                return True
            if best_confidence > candidate_confidence + 0.04:
                return False
        return len(candidate_text) > len(best_text)

    def _crop_box(self, image: np.ndarray, bbox: list[int] | tuple[int, int, int, int]) -> np.ndarray:
        x, y, width, height = [int(value) for value in bbox]
        return image[y : y + height, x : x + width]

    def _expand_bbox(
        self,
        bbox: list[int] | tuple[int, int, int, int],
        image_width: int,
        image_height: int,
    ) -> list[int]:
        x, y, width, height = [int(value) for value in bbox]
        pad_x = min(max(8, int(width * 0.16)), max(image_width // 8, 8))
        pad_y = min(max(8, int(height * 0.16)), max(image_height // 8, 8))
        x1 = max(x - pad_x, 0)
        y1 = max(y - pad_y, 0)
        x2 = min(x + width + pad_x, image_width)
        y2 = min(y + height + pad_y, image_height)
        return [x1, y1, max(x2 - x1, 1), max(y2 - y1, 1)]

    def _ocr_variants(self, region_image: np.ndarray) -> list[np.ndarray]:
        variants: list[np.ndarray] = [region_image]
        scaled = self._scale_for_ocr(region_image)
        if scaled is not None:
            variants.append(scaled)
        thresholded = self._threshold_for_ocr(region_image)
        if thresholded is not None:
            variants.append(thresholded)
        inverted = self._threshold_for_ocr(region_image, invert=True)
        if inverted is not None:
            variants.append(inverted)
        contrast_boosted = self._contrast_for_ocr(region_image)
        if contrast_boosted is not None:
            variants.append(contrast_boosted)
        sharpened = self._sharpen_for_ocr(region_image)
        if sharpened is not None:
            variants.append(sharpened)
        return variants

    def _scale_for_ocr(self, image: np.ndarray) -> np.ndarray | None:
        try:
            import cv2

            height, width = image.shape[:2]
            scale = 2.0 if min(height, width) < 96 else 1.6 if min(height, width) < 160 else 1.4
            return cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        except Exception:
            return None

    def _threshold_for_ocr(self, image: np.ndarray, *, invert: bool = False) -> np.ndarray | None:
        try:
            import cv2

            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            height, width = gray.shape[:2]
            scale = 2.1 if min(height, width) < 96 else 1.8 if min(height, width) < 160 else 1.7
            upscaled = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
            thresholded = cv2.adaptiveThreshold(
                upscaled,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY,
                31,
                9,
            )
            return cv2.cvtColor(thresholded, cv2.COLOR_GRAY2RGB)
        except Exception:
            return None

    def _contrast_for_ocr(self, image: np.ndarray) -> np.ndarray | None:
        try:
            import cv2

            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
            enhanced = clahe.apply(gray)
            scaled = cv2.resize(enhanced, None, fx=1.7, fy=1.7, interpolation=cv2.INTER_CUBIC)
            return cv2.cvtColor(scaled, cv2.COLOR_GRAY2RGB)
        except Exception:
            return None

    def _sharpen_for_ocr(self, image: np.ndarray) -> np.ndarray | None:
        try:
            import cv2

            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            denoised = cv2.bilateralFilter(gray, 5, 40, 40)
            blurred = cv2.GaussianBlur(denoised, (0, 0), 2.4)
            sharpened = cv2.addWeighted(denoised, 1.8, blurred, -0.8, 0)
            scaled = cv2.resize(sharpened, None, fx=1.6, fy=1.6, interpolation=cv2.INTER_CUBIC)
            return cv2.cvtColor(scaled, cv2.COLOR_GRAY2RGB)
        except Exception:
            return None

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

    def _dedupe_candidates(self, candidates: list[dict[str, Any]], image_width: int, image_height: int) -> list[dict[str, Any]]:
        cleaned: list[dict[str, Any]] = []
        for candidate in candidates:
            bbox = self._normalise_bbox(candidate["bbox"], image_width, image_height)
            if bbox is None:
                continue
            candidate["bbox"] = bbox
            duplicate_index: int | None = None
            for index, other in enumerate(cleaned):
                same_text = str(candidate["text"]).casefold() == str(other["text"]).casefold()
                overlaps = self._iou(tuple(candidate["bbox"]), tuple(other["bbox"])) >= 0.55
                contains = self._bbox_contains(tuple(candidate["bbox"]), tuple(other["bbox"])) or self._bbox_contains(tuple(other["bbox"]), tuple(candidate["bbox"]))
                text_nested = self._texts_are_near_duplicates(str(candidate["text"]), str(other["text"]))
                if same_text or ((overlaps or contains) and text_nested):
                    duplicate_index = index
                    break
            if duplicate_index is None:
                cleaned.append(candidate)
                continue
            other = cleaned[duplicate_index]
            other_score = self._candidate_quality_score(
                str(other.get("text") or ""),
                float(other["confidence"]) if isinstance(other.get("confidence"), (int, float)) else None,
            )
            candidate_score = self._candidate_quality_score(
                str(candidate.get("text") or ""),
                float(candidate["confidence"]) if isinstance(candidate.get("confidence"), (int, float)) else None,
            )
            if candidate_score > other_score + 0.05:
                cleaned[duplicate_index] = candidate
        return cleaned

    def _texts_are_near_duplicates(self, first: str, second: str) -> bool:
        left = re.sub(r"[^a-z0-9\u3040-\u30ff\u4e00-\u9fff\uac00-\ud7af]+", "", first.casefold())
        right = re.sub(r"[^a-z0-9\u3040-\u30ff\u4e00-\u9fff\uac00-\ud7af]+", "", second.casefold())
        if not left or not right:
            return False
        if left == right:
            return True
        shorter, longer = (left, right) if len(left) <= len(right) else (right, left)
        return len(shorter) >= 4 and shorter in longer

    def _candidate_quality_score(self, text: str, confidence: float | None) -> float:
        cleaned = self.cleaner.clean_text(text)
        if not cleaned or not self._acceptable_ocr_fragment(cleaned):
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

    def _extract_paddle_entries(self, results: Any) -> list[dict[str, Any]]:
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

    def _paddle_recognise(self, region_image: np.ndarray, language_code: str) -> tuple[str, float | None]:
        try:
            results = self._get_paddle_ocr(language_code).ocr(region_image)
        except Exception:
            return "", None
        entries = self._extract_paddle_entries(results)
        fragments = [self.cleaner.clean_text(str(entry.get("text") or "")) for entry in entries]
        fragments = [fragment for fragment in fragments if fragment]
        scores = [float(entry["confidence"]) for entry in entries if isinstance(entry.get("confidence"), (int, float))]
        text = self.cleaner.clean_text(" ".join(fragments))
        return text, (sum(scores) / len(scores) if scores else None)

    def _extract_paddle_fragments(self, image: np.ndarray, language_code: str) -> list[dict[str, Any]]:
        if not self._has_paddleocr():
            return []
        try:
            results = self._get_paddle_ocr(language_code).ocr(image)
        except Exception:
            return []
        fragments: list[dict[str, Any]] = []
        for entry in self._extract_paddle_entries(results):
            fragments.append(
                {
                    "bbox": entry["bbox"],
                    "text": entry["text"],
                    "confidence": entry["confidence"],
                    "ocr_engine": "paddleocr",
                }
            )
        return fragments

    def _extract_apple_fragments(self, image: np.ndarray, language_code: str) -> list[dict[str, Any]]:
        if not self._has_apple_vision():
            return []
        fragments: list[dict[str, Any]] = []
        for fragment in self.apple_vision_ocr.extract(image, language_code):
            fragments.append(
                {
                    "bbox": fragment.bbox,
                    "text": fragment.text,
                    "confidence": fragment.confidence,
                    "ocr_engine": "apple-vision",
                }
            )
        return fragments

    def _easyocr_recognise(self, region_image: np.ndarray, language_code: str) -> tuple[str, float | None]:
        reader = self._get_easyocr_reader(language_code)
        if reader is None:
            return "", None
        results = reader.readtext(region_image, detail=1, paragraph=True)
        fragments: list[str] = []
        scores: list[float] = []
        for item in results:
            if not isinstance(item, (list, tuple)) or len(item) < 2:
                continue
            text = self.cleaner.clean_text(str(item[1] or ""))
            if not text:
                continue
            fragments.append(text)
            if len(item) > 2 and isinstance(item[2], (int, float)):
                scores.append(float(item[2]))
        combined = " ".join(fragments).strip()
        confidence = sum(scores) / len(scores) if scores else None
        return combined, confidence

    def _apple_vision_recognise(self, region_image: np.ndarray, language_code: str) -> tuple[str, float | None]:
        return self.apple_vision_ocr.recognize(region_image, language_code)

    def _manga_ocr_read(self, region_image: np.ndarray) -> str:
        with self._LOAD_LOCK:
            if self.__class__._MANGA_OCR is None:
                from manga_ocr import MangaOcr

                self.__class__._MANGA_OCR = MangaOcr()
        pil_image = Image.fromarray(region_image)
        return str(self.__class__._MANGA_OCR(pil_image))

    def _get_paddle_ocr(self, language_code: str):
        provider_language = self._paddle_language(language_code)
        cached = self._PADDLE_CACHE.get(provider_language)
        if cached is not None:
            return cached
        with self._LOAD_LOCK:
            cached = self._PADDLE_CACHE.get(provider_language)
            if cached is not None:
                return cached
            from paddleocr import PaddleOCR

            recognition_model = {
                "en": "en_PP-OCRv5_mobile_rec",
                "korean": "korean_PP-OCRv5_mobile_rec",
                "japan": "japan_PP-OCRv3_mobile_rec",
                "ch": "PP-OCRv5_mobile_rec",
                "es": "latin_PP-OCRv3_mobile_rec",
                "latin": "latin_PP-OCRv3_mobile_rec",
            }.get(provider_language)
            try:
                instance = PaddleOCR(
                    lang=provider_language,
                    text_detection_model_name="PP-OCRv5_mobile_det",
                    text_recognition_model_name=recognition_model,
                    use_doc_orientation_classify=False,
                    use_doc_unwarping=False,
                    use_textline_orientation=False,
                    textline_orientation_batch_size=1,
                    text_recognition_batch_size=4,
                    text_det_limit_side_len=960,
                    text_det_limit_type="max",
                    text_rec_score_thresh=0.45,
                )
            except Exception:
                instance = PaddleOCR(lang="en")
            self._PADDLE_CACHE[provider_language] = instance
            return instance

    def _get_easyocr_reader(self, language_code: str):
        reader_languages = self._easyocr_languages(language_code)
        if not reader_languages:
            return None
        cache_key = ",".join(reader_languages)
        cached = self._EASYOCR_CACHE.get(cache_key)
        if cached is not None:
            return cached
        with self._LOAD_LOCK:
            cached = self._EASYOCR_CACHE.get(cache_key)
            if cached is not None:
                return cached
            import easyocr

            reader = easyocr.Reader(reader_languages, gpu=False, verbose=False)
            self._EASYOCR_CACHE[cache_key] = reader
            return reader

    def _paddle_language(self, language_code: str) -> str:
        return {
            "en": "en",
            "ja": "japan",
            "ko": "korean",
            "zh": "ch",
            "es": "es",
            "pt": "latin",
            "de": "latin",
            "tr": "latin",
            "id": "latin",
            "fr": "latin",
            "it": "latin",
            "ro": "latin",
            "ca": "latin",
            "gl": "latin",
        }.get(self.language_detector.normalize_language_code(language_code), "en")

    def _easyocr_languages(self, language_code: str) -> list[str]:
        normalized = self.language_detector.normalize_language_code(language_code)
        mapping = {
            "en": ["en"],
            "ja": ["ja", "en"],
            "ko": ["ko", "en"],
            "zh": ["ch_sim", "en"],
            "es": ["es", "en"],
            "pt": ["pt", "en"],
            "de": ["de", "en"],
            "tr": ["tr", "en"],
            "id": ["id", "en"],
            "fr": ["fr", "en"],
            "it": ["it", "en"],
        }
        return mapping.get(normalized, ["en"])

    def _has_paddleocr(self) -> bool:
        try:
            import paddleocr  # noqa: F401

            return True
        except Exception:
            return False

    def _has_easyocr(self) -> bool:
        try:
            import easyocr  # noqa: F401

            return True
        except Exception:
            return False

    def _has_apple_vision(self) -> bool:
        if not self.settings.comic_ocr_apple_vision_enabled:
            return False
        return self.apple_vision_ocr.is_available()

    def _should_use_apple_vision(self, language_code: str) -> bool:
        if not self._has_apple_vision():
            return False
        normalized = self.language_detector.normalize_language_code(language_code)
        return normalized != "ja"

    def _has_manga_ocr(self) -> bool:
        try:
            import manga_ocr  # noqa: F401

            return True
        except Exception:
            return False
