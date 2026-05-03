from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import logging
import re
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageOps

from app.core.config import get_settings
from app.pipeline.image_loader import ImageLoader
from app.schemas.project import ChapterMetadata, PanelBox
from app.services.apple_vision_ocr import AppleVisionOCRService
from app.services.comic_ocr_service import ComicOCRService
from app.services.dialogue_cleaner import DialogueCleaner
from app.services.language_detector import LanguageDetector
from app.services.ocr_cleaner import clean_ocr_lines, clean_ocr_text, is_usable_ocr_text
from app.services.translate_text import TranslateTextService
from app.utils.files import ensure_dir, read_json, write_json

logger = logging.getLogger(__name__)

_EVIDENCE_VERSION = "panel_evidence_v1"


@dataclass(slots=True)
class PanelEvidenceRegion:
    bbox: list[int]
    text_original: str
    text_english: str
    language: str
    confidence: float | None
    detector: str
    ocr_engine: str
    region_type: str = "dialogue"


@dataclass(slots=True)
class PanelEvidenceRecord:
    panel_id: str
    panel_order: int
    page: int
    text_original: str
    text_english: str
    dialogue_text: str
    caption_text: str
    confidence: float
    regions: list[PanelEvidenceRegion]
    source_summary: dict[str, Any]
    needs_review: bool = False


def load_panel_evidence_records(project_dir: Path) -> list[dict[str, Any]]:
    payload = read_json(project_dir / "output" / "panel_evidence.json", default={})
    if isinstance(payload, dict):
        records = payload.get("panels") or []
    elif isinstance(payload, list):
        records = payload
    else:
        records = []
    return [item for item in records if isinstance(item, dict)]


def panel_evidence_by_id(project_dir: Path) -> dict[str, dict[str, Any]]:
    return {
        str(item.get("panel_id") or "").strip(): item
        for item in load_panel_evidence_records(project_dir)
        if str(item.get("panel_id") or "").strip()
    }


class PanelEvidenceExtractor:
    """Build clean text evidence from panel crops before narration.

    This is deliberately not the old OCR-first path. It extracts candidate
    speech/caption regions, translates them into English when needed, and writes
    a sidecar that downstream vision/script prompts may use as *clean evidence*
    when Gemini refuses or under-reads a panel.
    """

    MAX_REGIONS_PER_PANEL = 8

    def __init__(self) -> None:
        self.settings = get_settings()
        self.cleaner = DialogueCleaner()
        self.language_detector = LanguageDetector(self.cleaner)
        # Apple Vision/LiveText can hard-crash the Python process on some
        # manga crops in the local macOS runtime. Keep it opt-in and lazy so a
        # conservative evidence refresh cannot take the worker down.
        self.apple_vision: AppleVisionOCRService | None = None
        self.comic_ocr = ComicOCRService(self.cleaner, self.language_detector)
        self.translator = TranslateTextService(self.language_detector)
        self.cache_dir = ensure_dir(self.settings.data_dir / "_panel_evidence_cache")

    def run(
        self,
        *,
        project_dir: Path,
        page_paths: list[Path],
        panels: list[PanelBox],
        chapter_metadata: ChapterMetadata,
        force_refresh: bool = False,
        allow_crop_ocr: bool = False,
        allow_apple_vision: bool = False,
        allow_metadata_ocr: bool = False,
        progress_callback: Any | None = None,
        cancel_callback: Any | None = None,
    ) -> list[dict[str, Any]]:
        output_path = project_dir / "output" / "panel_evidence.json"
        if output_path.exists() and not force_refresh:
            return load_panel_evidence_records(project_dir)

        kept_panels = [panel for panel in sorted(panels, key=lambda item: item.order) if panel.keep]
        if not kept_panels:
            write_json(output_path, {"version": _EVIDENCE_VERSION, "panels": []})
            return []

        page_ocr_boxes = read_json(project_dir / "output" / "page_ocr_boxes.json", default={})
        if not isinstance(page_ocr_boxes, dict):
            page_ocr_boxes = {}

        loader = ImageLoader(project_dir=project_dir, page_paths=page_paths, max_edge=1800)
        language_hint = self.language_detector.normalize_language_code(chapter_metadata.language)
        context_hint = " ".join(
            part
            for part in (
                str(chapter_metadata.manga_title or "").strip(),
                str(chapter_metadata.chapter_title or "").strip(),
            )
            if part
        )
        records: list[dict[str, Any]] = []
        total = len(kept_panels)
        for index, panel in enumerate(kept_panels, start=1):
            if cancel_callback:
                cancel_callback()
            panel_path = loader.panel_image_path(panel)
            page_evidence = self._page_ocr_candidates(panel, page_ocr_boxes)
            cache_key = self._cache_key(
                panel,
                panel_path,
                language_hint,
                page_evidence,
                allow_crop_ocr=allow_crop_ocr,
                allow_apple_vision=allow_apple_vision,
                allow_metadata_ocr=allow_metadata_ocr,
            )
            cache_path = self.cache_dir / f"{cache_key}.json"
            if cache_path.exists() and not force_refresh:
                payload = read_json(cache_path, default={})
                if isinstance(payload, dict) and payload.get("panel_id"):
                    records.append(payload)
                    if progress_callback:
                        progress_callback(round(index / total * 100, 2), f"Panel evidence {index}/{total}")
                    continue

            record = self._extract_panel(
                panel=panel,
                panel_path=panel_path,
                language_hint=language_hint,
                context_hint=context_hint,
                page_evidence=page_evidence,
                allow_crop_ocr=allow_crop_ocr,
                allow_apple_vision=allow_apple_vision,
                allow_metadata_ocr=allow_metadata_ocr,
            )
            payload = self._record_to_dict(record)
            write_json(cache_path, payload)
            records.append(payload)
            if progress_callback:
                progress_callback(round(index / total * 100, 2), f"Panel evidence {index}/{total}")

        write_json(output_path, {"version": _EVIDENCE_VERSION, "panels": records})
        return records

    def _extract_panel(
        self,
        *,
        panel: PanelBox,
        panel_path: Path | None,
        language_hint: str,
        context_hint: str,
        page_evidence: list[dict[str, Any]],
        allow_crop_ocr: bool,
        allow_apple_vision: bool,
        allow_metadata_ocr: bool,
    ) -> PanelEvidenceRecord:
        image: np.ndarray | None = None
        candidates: list[dict[str, Any]] = []
        candidates.extend(page_evidence)
        existing_text = clean_ocr_text(str(panel.ocr_text or "").strip())
        if allow_metadata_ocr and self._usable_original(existing_text):
            candidates.append(
                {
                    "bbox": [0, 0, max(int(panel.width), 1), max(int(panel.height), 1)],
                    "text": existing_text,
                    "confidence": 0.45,
                    "detector": "existing-panel-ocr",
                    "ocr_engine": "panel-metadata",
                }
            )

        only_existing_panel_ocr = bool(candidates) and all(
            str(candidate.get("detector") or "") == "existing-panel-ocr"
            for candidate in candidates
        )
        should_try_image_ocr = only_existing_panel_ocr or not self._candidate_set_has_substantial_signal(candidates)
        if should_try_image_ocr and panel_path is not None and panel_path.exists():
            try:
                with Image.open(panel_path) as source:
                    image = np.array(ImageOps.exif_transpose(source).convert("RGB"))
            except Exception as exc:
                logger.warning("Could not load panel crop for evidence %s: %s", panel.id, exc)

        if image is not None and allow_apple_vision:
            candidates.extend(self._apple_vision_candidates(image, language_hint))
        if image is not None:
            if allow_crop_ocr and not self._candidate_set_has_substantial_signal(candidates):
                candidates.extend(self._comic_candidates(image, language_hint))
            if allow_crop_ocr and not self._candidate_set_has_substantial_signal(candidates):
                candidates.extend(self._opencv_region_candidates(image, language_hint))
            if allow_crop_ocr and not candidates:
                candidates.extend(self._full_panel_candidate(image, language_hint))

        candidates = self._dedupe_candidates(candidates, max(int(panel.width), 1), max(int(panel.height), 1))
        regions = self._regions_from_candidates(candidates, language_hint, context_hint, panel)
        return self._build_record(panel, regions, candidates)

    def _comic_candidates(self, image: np.ndarray, language_hint: str) -> list[dict[str, Any]]:
        candidates: list[dict[str, Any]] = []
        try:
            raw_candidates = self.comic_ocr.detect_candidates(image, language_hint)
        except Exception as exc:
            logger.debug("Comic OCR candidate extraction failed: %s", exc)
            raw_candidates = []
        for item in raw_candidates:
            bbox = self._normalise_bbox(item.get("bbox"), image.shape[1], image.shape[0])
            if bbox is None:
                continue
            text = clean_ocr_text(str(item.get("text") or "").strip())
            if not self._usable_original(text):
                continue
            candidates.append(
                {
                    "bbox": bbox,
                    "text": text,
                    "confidence": float(item["confidence"]) if isinstance(item.get("confidence"), (int, float)) else None,
                    "detector": "comic-ocr",
                    "ocr_engine": str(item.get("ocr_engine") or "unknown"),
                }
            )
        return candidates

    def _apple_vision_candidates(self, image: np.ndarray, language_hint: str) -> list[dict[str, Any]]:
        candidates: list[dict[str, Any]] = []
        try:
            if self.apple_vision is None:
                self.apple_vision = AppleVisionOCRService(self.cleaner)
            fragments = self.apple_vision.extract(image, language_hint)
        except Exception as exc:
            logger.debug("Apple Vision OCR failed: %s", exc)
            fragments = []
        for fragment in fragments:
            bbox = self._normalise_bbox(fragment.bbox, image.shape[1], image.shape[0])
            if bbox is None:
                continue
            text = clean_ocr_text(fragment.text)
            if not self._usable_original(text):
                continue
            candidates.append(
                {
                    "bbox": bbox,
                    "text": text,
                    "confidence": fragment.confidence,
                    "detector": "apple-vision",
                    "ocr_engine": "apple-vision",
                }
            )
        return candidates

    def _opencv_region_candidates(self, image: np.ndarray, language_hint: str) -> list[dict[str, Any]]:
        candidates: list[dict[str, Any]] = []
        for bbox in self._detect_text_regions_opencv(image)[: self.MAX_REGIONS_PER_PANEL]:
            crop = self._crop_box(image, bbox)
            try:
                result = self.comic_ocr.recognize_region(crop, language_hint)
            except Exception:
                continue
            text = clean_ocr_text(result.original_text)
            if not self._usable_original(text):
                continue
            candidates.append(
                {
                    "bbox": [int(value) for value in bbox],
                    "text": text,
                    "confidence": result.confidence,
                    "detector": "opencv-region",
                    "ocr_engine": result.ocr_engine,
                }
            )
        return candidates

    def _full_panel_candidate(self, image: np.ndarray, language_hint: str) -> list[dict[str, Any]]:
        try:
            text, confidence, engine = self.comic_ocr.recognize_panel_text(image, language_hint)
        except Exception:
            return []
        cleaned = clean_ocr_text(text)
        if not self._usable_original(cleaned):
            return []
        return [
            {
                "bbox": [0, 0, int(image.shape[1]), int(image.shape[0])],
                "text": cleaned,
                "confidence": confidence,
                "detector": "full-panel",
                "ocr_engine": engine,
            }
        ]

    def _regions_from_candidates(
        self,
        candidates: list[dict[str, Any]],
        language_hint: str,
        context_hint: str,
        panel: PanelBox,
    ) -> list[PanelEvidenceRegion]:
        payloads: list[dict[str, Any]] = []
        for candidate in candidates:
            original = clean_ocr_text(str(candidate.get("text") or "").strip())
            if not self._usable_original(original):
                continue
            language = self.language_detector.detect(original, language_hint)
            payloads.append({"candidate": candidate, "text_original": original, "language": language})

        translations = self._translate_payloads(payloads, context_hint)
        regions: list[PanelEvidenceRegion] = []
        for payload in payloads:
            candidate = payload["candidate"]
            original = payload["text_original"]
            language = payload["language"]
            english = translations.get((language, original), original)
            english = self._clean_translated_evidence(english, original, language)
            if not self._usable_english_evidence(english, original, language):
                continue
            bbox = [int(value) for value in candidate.get("bbox") or [0, 0, max(int(panel.width), 1), max(int(panel.height), 1)]]
            regions.append(
                PanelEvidenceRegion(
                    bbox=bbox,
                    text_original=original,
                    text_english=english,
                    language=language,
                    confidence=float(candidate["confidence"]) if isinstance(candidate.get("confidence"), (int, float)) else None,
                    detector=str(candidate.get("detector") or "unknown"),
                    ocr_engine=str(candidate.get("ocr_engine") or "unknown"),
                    region_type=self._region_type(bbox, max(int(panel.width), 1), max(int(panel.height), 1)),
                )
            )
        return regions

    def _translate_payloads(self, payloads: list[dict[str, Any]], context_hint: str) -> dict[tuple[str, str], str]:
        translations: dict[tuple[str, str], str] = {}
        grouped: dict[str, list[str]] = {}
        for payload in payloads:
            language = self.language_detector.normalize_language_code(payload["language"])
            original = str(payload["text_original"] or "").strip()
            key = (language, original)
            if not original:
                translations[key] = ""
            elif language in {"en", "a"}:
                translations[key] = original
            else:
                grouped.setdefault(language, [])
                if original not in grouped[language]:
                    grouped[language].append(original)

        for language, texts in grouped.items():
            try:
                translated = self.translator.translate_batch(texts, language, context_hint)
            except Exception as exc:
                logger.warning("Panel evidence translation failed for %s: %s", language, exc)
                translated = self.language_detector.translate_batch_to_english(texts, language)
            for original, english in zip(texts, translated, strict=False):
                translations[(language, original)] = english
        return translations

    def _build_record(
        self,
        panel: PanelBox,
        regions: list[PanelEvidenceRegion],
        raw_candidates: list[dict[str, Any]],
    ) -> PanelEvidenceRecord:
        regions = self._dedupe_regions(regions)
        dialogue_lines = clean_ocr_lines(region.text_english for region in regions if region.region_type != "caption")
        caption_lines = clean_ocr_lines(region.text_english for region in regions if region.region_type == "caption")
        original_lines = clean_ocr_lines(region.text_original for region in regions)
        english_lines = clean_ocr_lines([*caption_lines, *dialogue_lines] or [region.text_english for region in regions])
        confidences = [float(region.confidence) for region in regions if isinstance(region.confidence, (int, float))]
        confidence = sum(confidences) / len(confidences) if confidences else (0.45 if regions else 0.0)
        engines = sorted({region.ocr_engine for region in regions if region.ocr_engine})
        detectors = sorted({region.detector for region in regions if region.detector})
        return PanelEvidenceRecord(
            panel_id=panel.id,
            panel_order=int(panel.order),
            page=int(panel.page),
            text_original=" ".join(original_lines).strip(),
            text_english=" ".join(english_lines).strip(),
            dialogue_text=" ".join(dialogue_lines).strip(),
            caption_text=" ".join(caption_lines).strip(),
            confidence=round(float(confidence), 3),
            regions=regions,
            source_summary={
                "version": _EVIDENCE_VERSION,
                "region_count": len(regions),
                "raw_candidate_count": len(raw_candidates),
                "engines": engines,
                "detectors": detectors,
            },
            needs_review=bool(raw_candidates and not regions),
        )

    def _page_ocr_candidates(self, panel: PanelBox, page_text_boxes: dict[str, Any]) -> list[dict[str, Any]]:
        boxes = page_text_boxes.get(str(panel.page)) or page_text_boxes.get(int(panel.page)) or []
        if not isinstance(boxes, list):
            return []
        panel_box = (int(panel.x), int(panel.y), max(int(panel.width), 1), max(int(panel.height), 1))
        association_box = self._expand_panel_box(panel_box)
        candidates: list[dict[str, Any]] = []
        for box in boxes:
            if not isinstance(box, dict):
                continue
            text = clean_ocr_text(str(box.get("text") or "").strip())
            if not self._usable_original(text):
                continue
            candidate_box = self._page_box_from_payload(box)
            if candidate_box is None:
                continue
            candidate_area = max(candidate_box[2] * candidate_box[3], 1)
            direct_overlap = self._intersection_area(panel_box, candidate_box) / candidate_area
            expanded_overlap = self._intersection_area(association_box, candidate_box) / candidate_area
            if direct_overlap < 0.42 and expanded_overlap < 0.70:
                continue
            local_bbox = [
                max(int(candidate_box[0] - panel_box[0]), 0),
                max(int(candidate_box[1] - panel_box[1]), 0),
                max(int(candidate_box[2]), 1),
                max(int(candidate_box[3]), 1),
            ]
            candidates.append(
                {
                    "bbox": local_bbox,
                    "text": text,
                    "confidence": float(box["confidence"]) if isinstance(box.get("confidence"), (int, float)) else 0.5,
                    "detector": "page-ocr-backfill",
                    "ocr_engine": str(box.get("ocr_engine") or box.get("engine") or "page-ocr"),
                }
            )
        return candidates

    def _detect_text_regions_opencv(self, image: np.ndarray) -> list[list[int]]:
        try:
            import cv2
        except Exception:
            return []
        grayscale = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        blurred = cv2.GaussianBlur(grayscale, (3, 3), 0)
        binary = cv2.adaptiveThreshold(
            blurred,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            31,
            11,
        )
        dilation_width = max(9, image.shape[1] // 28)
        dilation_height = max(3, image.shape[0] // 120)
        merged = cv2.dilate(binary, np.ones((dilation_height, dilation_width), np.uint8), iterations=1)
        contours, _ = cv2.findContours(merged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes: list[list[int]] = []
        crop_area = image.shape[0] * image.shape[1]
        for contour in contours:
            x, y, width, height = cv2.boundingRect(contour)
            area = width * height
            if area < max(180, crop_area * 0.0035):
                continue
            if width < max(34, image.shape[1] * 0.08):
                continue
            if height < max(16, image.shape[0] * 0.022):
                continue
            if area > crop_area * 0.62:
                continue
            pad_x = max(8, width // 10)
            pad_y = max(6, height // 6)
            x1 = max(x - pad_x, 0)
            y1 = max(y - pad_y, 0)
            x2 = min(x + width + pad_x, image.shape[1])
            y2 = min(y + height + pad_y, image.shape[0])
            boxes.append([x1, y1, x2 - x1, y2 - y1])
        return self._dedupe_boxes(boxes, image.shape[1], image.shape[0])

    def _dedupe_candidates(self, candidates: list[dict[str, Any]], image_width: int, image_height: int) -> list[dict[str, Any]]:
        deduped: list[dict[str, Any]] = []
        for candidate in candidates:
            bbox = self._normalise_bbox(candidate.get("bbox"), image_width, image_height)
            if bbox is None and str(candidate.get("detector") or "") in {"existing-panel-ocr", "full-panel"}:
                bbox = [0, 0, max(int(image_width), 1), max(int(image_height), 1)]
            if bbox is None:
                continue
            text = clean_ocr_text(str(candidate.get("text") or "").strip())
            if not self._usable_original(text):
                continue
            candidate = {**candidate, "bbox": bbox, "text": text}
            duplicate_index: int | None = None
            for index, existing in enumerate(deduped):
                same_text = normalize_text_key(text) == normalize_text_key(str(existing.get("text") or ""))
                overlaps = self._iou(tuple(bbox), tuple(existing["bbox"])) >= 0.62
                contains = self._bbox_contains(tuple(bbox), tuple(existing["bbox"])) or self._bbox_contains(tuple(existing["bbox"]), tuple(bbox))
                if same_text or overlaps or contains:
                    duplicate_index = index
                    break
            if duplicate_index is None:
                deduped.append(candidate)
                continue
            if self._candidate_score(candidate) > self._candidate_score(deduped[duplicate_index]):
                deduped[duplicate_index] = candidate
        return self._sort_reading_order(deduped)

    def _dedupe_regions(self, regions: list[PanelEvidenceRegion]) -> list[PanelEvidenceRegion]:
        deduped: list[PanelEvidenceRegion] = []
        for region in regions:
            key = normalize_text_key(region.text_english)
            if not key:
                continue
            duplicate_index = next((index for index, existing in enumerate(deduped) if normalize_text_key(existing.text_english) == key or self._iou(tuple(existing.bbox), tuple(region.bbox)) >= 0.72), None)
            if duplicate_index is None:
                deduped.append(region)
                continue
            existing = deduped[duplicate_index]
            if self._region_score(region) > self._region_score(existing):
                deduped[duplicate_index] = region
        return sorted(deduped, key=lambda item: (item.bbox[1], item.bbox[0]))

    def _usable_original(self, text: str) -> bool:
        cleaned = clean_ocr_text(text)
        if not cleaned or not is_usable_ocr_text(cleaned):
            return False
        tokens = re.findall(r"[A-Za-zÀ-ÖØ-öø-ÿĀ-žƀ-ɏḀ-ỿ\u3040-\u30ff\u4e00-\u9fff\uac00-\ud7af']+", cleaned)
        if len(tokens) == 1 and len(tokens[0]) < 4:
            return False
        if re.search(r"\b(?:translated by|scanlation|discord|patreon|typesetter|proofreader)\b", cleaned, re.IGNORECASE):
            return False
        return True

    def _candidate_set_has_substantial_signal(self, candidates: list[dict[str, Any]]) -> bool:
        if not candidates:
            return False
        combined = " ".join(str(candidate.get("text") or "") for candidate in candidates)
        tokens = re.findall(r"[A-Za-zÀ-ÖØ-öø-ÿĀ-žƀ-ɏḀ-ỿ\u3040-\u30ff\u4e00-\u9fff\uac00-\ud7af']+", combined)
        if len(tokens) >= 5:
            return True
        return len(candidates) >= 2 and sum(len(token) for token in tokens) >= 8

    def _usable_english_evidence(self, english: str, original: str, language: str) -> bool:
        cleaned = clean_ocr_text(english)
        if not cleaned or not is_usable_ocr_text(cleaned):
            return False
        if self._looks_like_foreign_echo(cleaned, original, language):
            return False
        if re.search(r"\b(?:no new event is added|connective tissue|surrounding tension stays intact)\b", cleaned, re.IGNORECASE):
            return False
        return True

    def _clean_translated_evidence(self, english: str, original: str, language: str) -> str:
        cleaned = clean_ocr_text(english)
        if not cleaned:
            return ""
        cleaned = re.sub(r"[\u3040-\u30ff\u4e00-\u9fff\uac00-\ud7af]+", " ", cleaned)
        cleaned = re.sub(r"\b(?:chen\s+e\s+ku\s+hao|v[áa]\s+logo\s+a\s+neve|vai\s+sair\s+para\s+a)\b[.!?, ]*", " ", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"(?:\band\s+went\b[\s.,]*){3,}", " ", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\b([A-Za-zÀ-ÖØ-öø-ÿĀ-žƀ-ɏḀ-ỿ]{2,})(?:\s+\1\b){2,}", r"\1", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\b(?:[A-Za-z]-){3,}[A-Za-z]\b", "", cleaned)
        pieces = [
            piece.strip(" ,;:-")
            for piece in re.split(r"(?<=[.!?])\s+|,\s+", cleaned)
            if piece.strip(" ,;:-")
        ]
        survivors: list[str] = []
        for piece in pieces:
            if self._reject_evidence_piece(piece, language):
                continue
            survivors.append(piece)
        if not survivors and not self._reject_evidence_piece(cleaned, language):
            survivors.append(cleaned)
        return clean_ocr_text(" ".join(survivors))

    def _reject_evidence_piece(self, piece: str, language: str) -> bool:
        normalized = clean_ocr_text(piece)
        if not normalized:
            return True
        tokens = re.findall(r"[A-Za-zÀ-ÖØ-öø-ÿĀ-žƀ-ɏḀ-ỿ']+", normalized)
        if not tokens:
            return True
        if len(tokens) == 1 and len(tokens[0]) < 5:
            return True
        lowered = normalized.casefold()
        if re.search(r"\b([a-z]{2,})(?:\s+\1\b){2,}", lowered):
            return True
        if self._has_repeated_ngram(lowered):
            return True
        if re.search(r"\b(?:[a-z]-){3,}[a-z]\b", lowered):
            return True
        foreign_markers = {
            "agora", "causa", "destas", "destes", "estas", "estes", "hein", "logo", "morrer",
            "merece", "muito", "nao", "não", "neve", "pagar", "para", "pessoas", "por",
            "porque", "saia", "sair", "tudo", "voce", "você",
        }
        token_set = {token.strip("'").casefold() for token in tokens}
        if token_set & foreign_markers:
            english_markers = {
                "the", "a", "an", "and", "but", "if", "for", "from", "in", "it", "not",
                "of", "on", "our", "that", "then", "this", "to", "was", "will", "with",
                "you", "your", "he", "him", "his", "she", "her", "they", "them",
            }
            # Mixed OCR often has one good English clause next to a Portuguese
            # fragment. Keep the clause only when it has enough English signal.
            if len(token_set & english_markers) < 2:
                return True
        if len(tokens) <= 3 and not any(token.casefold() in {"go", "wait", "stop", "help", "snow"} for token in tokens):
            return True
        return False

    def _has_repeated_ngram(self, lowered: str) -> bool:
        tokens = re.findall(r"[a-z']+", lowered)
        if len(tokens) < 8:
            return False
        for size in (2, 3):
            counts: dict[tuple[str, ...], int] = {}
            for index in range(0, len(tokens) - size + 1):
                ngram = tuple(tokens[index : index + size])
                counts[ngram] = counts.get(ngram, 0) + 1
            if any(count >= 3 for count in counts.values()):
                return True
        return False

    def _looks_like_foreign_echo(self, english: str, original: str, language: str) -> bool:
        normalized_language = self.language_detector.normalize_language_code(language)
        if normalized_language in {"en", "a"}:
            return False
        if normalize_text_key(english) != normalize_text_key(original):
            return False
        markers = {
            "pt": {"voce", "você", "nao", "não", "morrer", "merece", "pessoas", "muito", "estas", "destas", "porque", "para"},
            "es": {"usted", "porque", "para", "estas", "muerte", "personas"},
            "fr": {"vous", "pourquoi", "personnes", "mort", "avec"},
        }.get(normalized_language, set())
        tokens = {token.casefold() for token in re.findall(r"[A-Za-zÀ-ÖØ-öø-ÿĀ-žƀ-ɏḀ-ỿ']+", english)}
        return len(tokens & markers) >= 1

    def _region_type(self, bbox: list[int], image_width: int, image_height: int) -> str:
        x, y, width, height = bbox
        wide = width / max(image_width, 1) >= 0.56
        near_edge = y <= image_height * 0.18 or y + height >= image_height * 0.84
        short = height / max(image_height, 1) <= 0.28
        return "caption" if wide and near_edge and short else "dialogue"

    def _cache_key(
        self,
        panel: PanelBox,
        panel_path: Path | None,
        language_hint: str,
        page_evidence: list[dict[str, Any]],
        *,
        allow_crop_ocr: bool,
        allow_apple_vision: bool,
        allow_metadata_ocr: bool,
    ) -> str:
        hasher = hashlib.sha256()
        hasher.update(_EVIDENCE_VERSION.encode("utf-8"))
        hasher.update(
            f"crop={int(allow_crop_ocr)}|apple={int(allow_apple_vision)}|metadata={int(allow_metadata_ocr)}".encode("utf-8")
        )
        hasher.update(language_hint.encode("utf-8"))
        hasher.update(str(panel.id).encode("utf-8"))
        hasher.update(str((panel.page, panel.x, panel.y, panel.width, panel.height, panel.order)).encode("utf-8"))
        hasher.update(clean_ocr_text(str(panel.ocr_text or "")).encode("utf-8"))
        hasher.update(clean_ocr_text(" ".join(str(item.get("text") or "") for item in page_evidence)).encode("utf-8"))
        if panel_path is not None and panel_path.exists():
            try:
                hasher.update(panel_path.read_bytes())
            except Exception:
                pass
        return hasher.hexdigest()

    def _record_to_dict(self, record: PanelEvidenceRecord) -> dict[str, Any]:
        payload = asdict(record)
        payload["regions"] = [asdict(region) for region in record.regions]
        return payload

    def _normalise_bbox(self, raw_bbox: Any, image_width: int, image_height: int) -> list[int] | None:
        if not isinstance(raw_bbox, (list, tuple)) or len(raw_bbox) < 4:
            return None
        try:
            x, y, width, height = [int(round(float(value))) for value in raw_bbox[:4]]
        except Exception:
            return None
        x = max(x, 0)
        y = max(y, 0)
        width = min(max(width, 1), max(image_width - x, 1))
        height = min(max(height, 1), max(image_height - y, 1))
        if width < 16 or height < 14:
            return None
        if width * height > image_width * image_height * 0.72:
            return None
        return [x, y, width, height]

    def _dedupe_boxes(self, boxes: list[list[int]], image_width: int, image_height: int) -> list[list[int]]:
        deduped: list[list[int]] = []
        for box in boxes:
            normalized = self._normalise_bbox(box, image_width, image_height)
            if normalized is None:
                continue
            if any(self._iou(tuple(normalized), tuple(existing)) >= 0.68 for existing in deduped):
                continue
            deduped.append(normalized)
        return sorted(deduped, key=lambda item: (item[1], item[0]))

    def _crop_box(self, image: np.ndarray, box: list[int]) -> np.ndarray:
        x, y, width, height = [int(value) for value in box]
        return image[y : y + height, x : x + width]

    def _sort_reading_order(self, candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return sorted(candidates, key=lambda item: (int(item.get("bbox", [0, 0, 0, 0])[1]), int(item.get("bbox", [0, 0, 0, 0])[0])))

    def _candidate_score(self, candidate: dict[str, Any]) -> float:
        confidence = float(candidate.get("confidence")) if isinstance(candidate.get("confidence"), (int, float)) else 0.45
        text = str(candidate.get("text") or "")
        return confidence + min(len(text), 120) / 240

    def _region_score(self, region: PanelEvidenceRegion) -> float:
        confidence = float(region.confidence) if isinstance(region.confidence, (int, float)) else 0.45
        return confidence + min(len(region.text_english), 120) / 240

    def _page_box_from_payload(self, payload: dict[str, Any]) -> tuple[int, int, int, int] | None:
        if isinstance(payload.get("bbox"), (list, tuple)) and len(payload["bbox"]) >= 4:
            try:
                return tuple(int(round(float(value))) for value in payload["bbox"][:4])  # type: ignore[return-value]
            except Exception:
                return None
        keys = ("x", "y", "width", "height")
        if all(key in payload for key in keys):
            try:
                return (
                    int(round(float(payload["x"]))),
                    int(round(float(payload["y"]))),
                    max(int(round(float(payload["width"]))), 1),
                    max(int(round(float(payload["height"]))), 1),
                )
            except Exception:
                return None
        return None

    def _expand_panel_box(self, panel_box: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
        x, y, width, height = panel_box
        pad_x = max(12, int(width * 0.08))
        pad_y = max(12, int(height * 0.08))
        return (x - pad_x, y - pad_y, width + pad_x * 2, height + pad_y * 2)

    def _intersection_area(self, box_a: tuple[int, int, int, int], box_b: tuple[int, int, int, int]) -> int:
        ax, ay, aw, ah = box_a
        bx, by, bw, bh = box_b
        x1 = max(ax, bx)
        y1 = max(ay, by)
        x2 = min(ax + aw, bx + bw)
        y2 = min(ay + ah, by + bh)
        if x2 <= x1 or y2 <= y1:
            return 0
        return int((x2 - x1) * (y2 - y1))

    def _iou(self, box_a: tuple[int, int, int, int], box_b: tuple[int, int, int, int]) -> float:
        intersection = self._intersection_area(box_a, box_b)
        aw, ah = box_a[2], box_a[3]
        bw, bh = box_b[2], box_b[3]
        union = aw * ah + bw * bh - intersection
        return intersection / max(union, 1)

    def _bbox_contains(self, outer: tuple[int, int, int, int], inner: tuple[int, int, int, int]) -> bool:
        ox, oy, ow, oh = outer
        ix, iy, iw, ih = inner
        return ox <= ix and oy <= iy and ox + ow >= ix + iw and oy + oh >= iy + ih


def normalize_text_key(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", clean_ocr_text(text).casefold()).strip()
