from __future__ import annotations

import hashlib
import io
import json
import logging
import math
import os
import re
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from app.core.config import get_settings
from app.schemas.project import ChapterMetadata, PanelBox
from app.services.character_clusterer import CharacterClusterer
from app.services.character_identity_tracker import CharacterIdentityTracker
from app.services.character_memory import CharacterMemory
from app.services.character_name_filters import looks_like_false_character_name
from app.services.character_name_service import CharacterNameService
from app.services.character_vision_recognizer import GeminiCharacterRecognizer
from app.services.comic_ocr_service import ComicOCRService
from app.services.dialogue_cleaner import DialogueCleaner
from app.services.language_detector import LanguageDetector
from app.services.llm_router import LLMRouter
from app.services.magi_service import MagiHFService
from app.services.panel_detection_service import MagiSpeakerAttributionService
from app.services.ocr_cleaner import clean_ocr_lines, clean_ocr_text, is_usable_ocr_text
from app.services.script_generator import ScriptGenerator
from app.services.translate_text import TranslateTextService
from app.utils.files import ensure_dir, read_json, write_json

os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class DialogueRegion:
    page: int
    panel: int
    panel_order: int
    bbox: list[int]
    language: str
    text_original: str
    text_english: str
    bubble_id: str | None = None
    bubble_bbox: list[int] | None = None
    confidence: float | None = None
    detector: str = "opencv"
    ocr_engine: str = "none"
    character_id: str | None = None
    stable_character_id: str | None = None
    speaker_name: str | None = None
    speaker_label: str | None = None
    character_display_name: str | None = None


@dataclass(slots=True)
class DialogueScene:
    scene: int
    panel_id: str
    page: int
    panel: int
    panel_order: int
    reading_mode: str
    dialogue: list[str]
    dialogue_original: list[str]
    languages: list[str]
    panel_bbox: list[int]
    panel_path: str
    has_dialogue: bool
    detected_text: str
    dialogue_entries: list[dict[str, str]]
    speaker_names: list[str]
    character_names: list[str]
    character_ids: list[str]
    character_labels: list[str]
    primary_speaker_name: str | None = None
    protagonist_name: str | None = None
    logical_panel_id: str | None = None
    multi_page_panel: bool = False
    spans_pages: list[int] = field(default_factory=list)


@dataclass(slots=True)
class SceneCluster:
    scene: int
    panels: list[int]
    panel_ids: list[str]
    pages: list[int]
    reading_mode: str
    dialogue: list[str]
    dialogue_original: list[str]
    languages: list[str]
    character_names: list[str]
    keywords: list[str]
    summary_hint: str
    logical_panel_ids: list[str] = field(default_factory=list)
    multi_page_panel: bool = False


@dataclass(slots=True)
class OCRCandidate:
    bbox: list[int]
    text: str
    confidence: float | None = None
    detector: str = "opencv"
    ocr_engine: str = "none"


class DialogueExtractionPipeline:
    _ARTIFACT_STRATEGY = "dialogue_pipeline_v3_applevision_recall"
    _GLOBAL_PADDLE_OCR: dict[str, Any] = {}
    _GLOBAL_TRANSLATOR_CACHE: dict[str, tuple[Any, Any]] = {}
    _GLOBAL_TRANSLATED_TEXT_CACHE: dict[tuple[str, str], str] = {}
    _GLOBAL_MANGA_OCR: Any | None = None

    def __init__(self) -> None:
        self.settings = get_settings()
        self._paddle_ocr = self._GLOBAL_PADDLE_OCR
        self._translator_cache = self._GLOBAL_TRANSLATOR_CACHE
        self._translated_text_cache = self._GLOBAL_TRANSLATED_TEXT_CACHE
        self._manga_ocr = self._GLOBAL_MANGA_OCR
        self._dialogue_cleaner = DialogueCleaner()
        self._language_detector = LanguageDetector(self._dialogue_cleaner)
        self._comic_ocr = ComicOCRService(self._dialogue_cleaner, self._language_detector)
        self._translator = TranslateTextService(self._language_detector)
        self._character_names = CharacterNameService()
        self._character_clusterer = CharacterClusterer()
        self._gemini_recognizer = GeminiCharacterRecognizer()
        self._character_identity_tracker = CharacterIdentityTracker()
        self._character_memory = CharacterMemory()
        self._script_generator = ScriptGenerator()
        self._llm_router = LLMRouter()
        self._magi_service = MagiHFService()
        self._panel_dialogue_cache_dir = ensure_dir(self.settings.data_dir / "_panel_dialogue_cache")

    def run(
        self,
        project_dir: Path,
        panels: list[PanelBox],
        metadata: ChapterMetadata,
        page_text_boxes: dict[str, list] | None = None,
        allow_expensive_ocr: bool = True,
        progress_callback: callable | None = None,
        cancel_callback: callable | None = None,
    ) -> dict[str, Any]:
        started_at = time.perf_counter()
        extraction_mode = "deep" if allow_expensive_ocr else "fast"
        metrics = {
            "panels_total": len(panels),
            "panels_kept": 0,
            "panels_with_dialogue": 0,
            "scene_clusters": 0,
            "regions_detected": 0,
            "translation_count": 0,
            "ocr_seconds": 0.0,
            "translation_seconds": 0.0,
            "panel_cache_hits": 0,
            "page_ocr_primary_panels": 0,
            "page_ocr_visual_only_skips": 0,
            "deep_ocr_skips": 0,
            "magi_pages": 0,
            "magi_text_boxes": 0,
            "triage_skipped": 0,
            "triage_light_ocr": 0,
            "fast_ocr_skips": 0,
            "expensive_ocr_enabled": bool(allow_expensive_ocr),
            "elapsed_seconds": 0.0,
        }
        page_paths = sorted((project_dir / "pages").glob("*"))
        kept_panels = [panel for panel in sorted(panels, key=lambda item: item.order) if panel.keep]
        metrics["panels_kept"] = len(kept_panels)
        reading_mode = self._reading_mode(metadata, page_paths)
        project_language = self._infer_project_language(page_paths, metadata.language)
        self._translation_context_hint = str(metadata.manga_title or "").strip()
        self._page_text_boxes = page_text_boxes or {}

        panels_dir = ensure_dir(project_dir / "panels")
        ocr_dir = ensure_dir(project_dir / "ocr")
        translations_dir = ensure_dir(project_dir / "translations")
        output_dir = ensure_dir(project_dir / "output")
        panel_signature = self._panel_signature(kept_panels)

        cached_artifacts = self._load_cached_artifacts(
            output_dir / "dialogue_pipeline_manifest.json",
            panel_signature,
            extraction_mode,
        )
        if cached_artifacts is not None:
            if progress_callback:
                progress_callback(100, "Reused cached dialogue extraction")
            return cached_artifacts

        magi_page_payloads: dict[int, dict[str, Any]] = {}
        requested_magi_pages = sorted({int(panel.page) for panel in kept_panels})
        use_magi_dialogue_ocr = bool(self.settings.magi_dialogue_ocr_enabled)
        if requested_magi_pages and self._page_text_boxes:
            available_page_ocr = sum(
                1
                for page_number in requested_magi_pages
                if self._page_ocr_has_substantial_signal(self._page_text_boxes.get(str(page_number)))
            )
            # Page-level OCR from panel detection is already available for many
            # projects. Re-running MAGI OCR on top of that is the main reason
            # script generation crawls through long chapters. In that case, use
            # MAGI only for layout / character associations and let the existing
            # page OCR plus per-panel fallback handle the text itself.
            if available_page_ocr >= max(1, int(len(requested_magi_pages) * 0.35)):
                use_magi_dialogue_ocr = False
        if requested_magi_pages:
            magi_page_payloads = self._load_cached_magi_page_payloads(
                output_dir / "magi_page_payloads.json",
                requested_magi_pages,
            )
        missing_magi_pages = [page_number for page_number in requested_magi_pages if page_number not in magi_page_payloads]
        if missing_magi_pages and self._magi_service.is_available() and allow_expensive_ocr:
            if progress_callback:
                progress_callback(
                    6,
                    "Scanning chapter structure with MAGI"
                    if use_magi_dialogue_ocr
                    else "Scanning character layout with MAGI",
                )
            fresh_payloads = self._magi_service.predict_page_payloads(
                page_paths,
                page_numbers=missing_magi_pages,
                do_ocr=use_magi_dialogue_ocr,
                batch_size=int(self.settings.magi_batch_size or 1),
                cancel_callback=cancel_callback,
                progress_callback=(
                    (lambda pct, message: progress_callback(6 + pct * 0.12, message))
                    if progress_callback
                    else None
                ),
                progress_label=(
                    "Scanning chapter structure with MAGI"
                    if use_magi_dialogue_ocr
                    else "Scanning character layout with MAGI"
                ),
            )
            magi_page_payloads.update(fresh_payloads)
        elif magi_page_payloads and progress_callback:
            progress_callback(18, "Reused cached MAGI chapter scan")

        if requested_magi_pages:
            if magi_page_payloads:
                metrics["magi_pages"] = len(magi_page_payloads)
                metrics["magi_text_boxes"] = sum(
                    len((payload or {}).get("texts", []) or [])
                    for payload in magi_page_payloads.values()
                )
                write_json(output_dir / "magi_page_payloads.json", magi_page_payloads)
            else:
                (output_dir / "magi_page_payloads.json").unlink(missing_ok=True)

        raw_regions: list[DialogueRegion] = []
        scenes: list[DialogueScene] = []

        if not kept_panels:
            artifacts = {
                "strategy": self._ARTIFACT_STRATEGY,
                "extraction_mode": extraction_mode,
                "reading_mode": reading_mode,
                "project_language": project_language,
                "dialogue_regions": [],
                "scenes": [],
                "scene_clusters": [],
                "character_clusters": [],
                "providers": self._provider_summary(),
                "panel_signature": panel_signature,
                "metrics": metrics,
            }
            self._write_artifacts(ocr_dir, translations_dir, output_dir, artifacts)
            return artifacts

        page_cache: dict[int, np.ndarray] = {}

        # Pre-warm the translation cache from already-computed MAGI and page OCR data
        # before the per-panel loop so most panels hit the cache instead of making
        # individual blocking Gemini translation calls.
        if project_language not in ("en", "a", ""):
            if progress_callback:
                progress_callback(19, "Pre-translating known dialogue text")
            self._prewarm_translation_cache(magi_page_payloads, project_language)

        if progress_callback:
            progress_callback(20, "Scanning panels for dialogue candidates")
        magi_speakers: dict[int, dict[str, Any]] = {}
        character_clusters: list[dict[str, Any]] = []
        panel_region_records: list[dict[str, Any]] = []

        for index, panel in enumerate(kept_panels, start=1):
            if cancel_callback:
                cancel_callback()
            if progress_callback:
                progress_callback(
                    22 + ((index - 1) / max(len(kept_panels), 1)) * 48,
                    f"Scanning dialogue context for panel {index}/{len(kept_panels)}",
                )

            image = self._load_page_image(panel.page, page_paths, page_cache)
            if image is None:
                continue

            crop, crop_bbox = self._panel_crop(image, panel)
            panel_path = panels_dir / f"panel_{panel.order:03d}.png"
            Image.fromarray(crop).save(panel_path, format="PNG", optimize=True)
            panel_hash = self._panel_image_hash(crop)
            triage = self._triage_panel_image(crop)
            if triage["mode"] == "skip":
                metrics["triage_skipped"] += 1
            elif triage["mode"] == "light":
                metrics["triage_light_ocr"] += 1

            cached_regions = self._load_cached_panel_dialogue(panel_hash, project_language, reading_mode, str(triage["mode"]))
            if cached_regions is not None:
                metrics["panel_cache_hits"] += 1
                scene_regions = self._hydrate_cached_dialogue_regions(panel, crop_bbox, cached_regions)
            else:
                ocr_started_at = time.perf_counter()
                scene_regions: list[DialogueRegion] = []
                candidates: list[OCRCandidate] = []
                magi_candidates: list[OCRCandidate] = []
                page_key = str(panel.page)
                page_ocr_available = page_key in self._page_text_boxes
                page_ocr_boxes = self._page_text_boxes.get(page_key, [])
                page_backfill_regions: list[DialogueRegion] = []
                existing_panel_region = self._region_from_existing_panel_text(panel, crop_bbox, project_language)
                if existing_panel_region is not None:
                    scene_regions.append(existing_panel_region)
                if page_ocr_available:
                    page_backfill_regions = self._backfill_from_page_ocr(
                        panel,
                        crop_bbox,
                        image.shape[1],
                        image.shape[0],
                        project_language,
                        metrics,
                    )
                    if self._scene_regions_have_substantial_signal(page_backfill_regions):
                        scene_regions.extend(page_backfill_regions)
                    if scene_regions:
                        metrics["page_ocr_primary_panels"] += 1

                if not scene_regions:
                    magi_candidates = self._magi_candidates_for_panel(
                        panel,
                        crop_bbox,
                        magi_page_payloads.get(int(panel.page)),
                        image.shape[1],
                        image.shape[0],
                        reading_mode,
                    )
                    if magi_candidates:
                        metrics["regions_detected"] += len(magi_candidates)
                        translation_started_at = time.perf_counter()
                        scene_regions = self._build_dialogue_regions(
                            panel,
                            magi_candidates,
                            project_language,
                            metrics,
                        )
                        metrics["translation_seconds"] += time.perf_counter() - translation_started_at

                skip_panel_ocr_after_page_scan = (
                    not scene_regions
                    and page_ocr_available
                    and self._page_ocr_has_substantial_signal(page_ocr_boxes)
                    and self._should_trust_empty_page_ocr_for_panel(crop, triage, page_ocr_boxes)
                )
                skip_expensive_panel_ocr = self._should_skip_expensive_panel_ocr(
                    panel,
                    crop,
                    triage,
                    bool(page_ocr_available or magi_page_payloads.get(int(panel.page))),
                    bool(page_backfill_regions),
                    bool(magi_candidates),
                )

                if skip_panel_ocr_after_page_scan:
                    # Page-level OCR already scanned this page and this panel looks
                    # visual-only, so avoid spending minutes on splash art.
                    metrics["page_ocr_visual_only_skips"] += 1
                    pass
                elif skip_expensive_panel_ocr:
                    metrics["deep_ocr_skips"] += 1
                    pass
                elif not scene_regions and triage["mode"] == "full" and allow_expensive_ocr:
                    candidates = self._extract_panel_candidates(
                        crop,
                        project_language,
                        reading_mode,
                        cancel_callback=cancel_callback,
                    )
                    candidates = self._associate_candidates_to_panel(
                        candidates,
                        crop_bbox,
                        panel,
                        image.shape[1],
                        image.shape[0],
                        reading_mode,
                    )
                    metrics["regions_detected"] += len(candidates)
                    translation_started_at = time.perf_counter()
                    scene_regions = self._build_dialogue_regions(
                        panel,
                        candidates,
                        project_language,
                        metrics,
                    )
                    metrics["translation_seconds"] += time.perf_counter() - translation_started_at

                if not scene_regions and page_backfill_regions:
                    scene_regions.extend(page_backfill_regions)
                    metrics["page_ocr_primary_panels"] += 1

                if (
                    not scene_regions
                    and not skip_panel_ocr_after_page_scan
                    and not skip_expensive_panel_ocr
                    and allow_expensive_ocr
                    and triage["mode"] != "skip"
                ):
                    fallback_text, confidence, ocr_engine = self._ocr_full_panel(crop, project_language)
                    fallback_text = self._clean_text(fallback_text)
                    if fallback_text:
                        detected_language = self._panel_text_language(fallback_text, project_language)
                        text_english = self._translate_to_english(fallback_text, detected_language)
                        if text_english and text_english.strip() != fallback_text.strip():
                            metrics["translation_count"] += 1
                        scene_regions.append(
                            DialogueRegion(
                                page=panel.page,
                                panel=panel.panel,
                                panel_order=panel.order,
                                bbox=[0, 0, int(crop_bbox[2]), int(crop_bbox[3])],
                                language=detected_language,
                                text_original=fallback_text,
                                text_english=text_english,
                                confidence=confidence,
                                detector="panel-fallback",
                                ocr_engine=ocr_engine,
                            )
                        )

                if not allow_expensive_ocr and not scene_regions:
                    metrics["fast_ocr_skips"] += 1

                metrics["ocr_seconds"] += time.perf_counter() - ocr_started_at
                if allow_expensive_ocr:
                    self._store_cached_panel_dialogue(panel_hash, project_language, reading_mode, str(triage["mode"]), scene_regions)

            raw_regions.extend(scene_regions)
            panel_region_records.append(
                {
                    "panel": panel,
                    "crop_bbox": crop_bbox,
                    "panel_path": panel_path,
                    "scene_regions": scene_regions,
                    "image_width": image.shape[1],
                    "image_height": image.shape[0],
                }
            )

            if progress_callback:
                progress_callback(
                    22 + (index / max(len(kept_panels), 1)) * 48,
                    f"Extracted dialogue context for panel {index}/{len(kept_panels)}",
                )

        panel_region_records = self._merge_continuation_panel_regions(panel_region_records)

        tracking_panels = [
            record["panel"]
            for record in panel_region_records
            if self._should_track_panel_regions(record["scene_regions"])
        ]
        if tracking_panels:
            if progress_callback:
                progress_callback(74, "Analyzing speaker and character layout")
            tracking_pages = {int(panel.page) for panel in tracking_panels}
            magi_speakers = {
                page_number: payload
                for page_number, payload in magi_page_payloads.items()
                if int(page_number) in tracking_pages
            }
            if tracking_pages - set(magi_speakers):
                fallback_payloads = MagiSpeakerAttributionService().detect_page_associations(
                    page_paths,
                    tracking_panels,
                    cancel_callback=cancel_callback,
                )
                magi_speakers.update(
                    {
                        page_number: payload
                        for page_number, payload in fallback_payloads.items()
                        if int(page_number) in tracking_pages
                    }
                )
            if progress_callback:
                progress_callback(82, "Linking recurring characters across dialogue panels")
            cluster_payload = self._character_clusterer.cluster(
                page_paths,
                magi_speakers,
                tracking_panels,
                cancel_callback=cancel_callback,
            )
            magi_speakers = cluster_payload.get("page_payloads", magi_speakers)
            character_clusters = cluster_payload.get("clusters", [])

        # Gemini Vision character scan — runs whenever CLIP found no clusters.
        # This is the primary path for non-English manga where Magi/OCR struggles.
        if not character_clusters and self._gemini_recognizer.is_available():
            if progress_callback:
                progress_callback(86, "Identifying characters with Gemini Vision")
            try:
                vision_payload = self._gemini_recognizer.recognize(
                    page_paths=page_paths,
                    page_payloads=magi_speakers,
                    panels=kept_panels,
                    panel_image_dir=panels_dir,
                    cache_dir=output_dir,
                    cancel_callback=cancel_callback,
                )
                character_clusters = vision_payload.get("clusters", [])
                logger.info(
                    "Gemini Vision found %d character clusters for %s",
                    len(character_clusters),
                    project_dir.name,
                )
            except Exception as exc:
                logger.warning(
                    "Gemini Vision character recognition failed, continuing with empty clusters: %s",
                    exc,
                )

        for record in panel_region_records:
            record["scene_regions"] = self._attach_magi_speaker_candidates(
                record["scene_regions"],
                record["crop_bbox"],
                record["panel"],
                magi_speakers.get(record["panel"].page),
                record["image_width"],
                record["image_height"],
                reading_mode,
            )

        if progress_callback:
            progress_callback(90, "Resolving character names and speakers")
        character_dictionary, discovered_protagonist_name = self._character_names.discover(
            [region.text_english or region.text_original for region in raw_regions],
            metadata,
        )
        character_dictionary = {
            key: value
            for key, value in character_dictionary.items()
            if self._is_reliable_character_name(value)
        }
        if discovered_protagonist_name and not self._is_reliable_character_name(discovered_protagonist_name):
            discovered_protagonist_name = None
        character_clusters = self._character_clusterer.attach_dialogues(
            character_clusters,
            raw_regions,
            discovered_protagonist_name,
        )
        cluster_name_map = self._character_clusterer.resolve_names(
            character_clusters,
            metadata,
            character_dictionary,
            discovered_protagonist_name,
            router=self._llm_router,
        )
        cluster_name_map = {
            cluster_id: name
            for cluster_id, name in cluster_name_map.items()
            if self._is_reliable_character_name(name)
        }
        character_tracking = self._build_tracking_from_clusters(character_clusters)
        character_memory = self._character_memory.build(
            character_tracking,
            character_clusters,
            cluster_name_map,
            discovered_protagonist_name,
        )
        raw_regions = self._script_generator.apply_character_memory_to_regions(raw_regions, character_memory)
        character_memory, raw_regions, identity_report = self._character_identity_tracker.refine(
            page_paths,
            magi_speakers,
            character_clusters,
            character_memory,
            raw_regions,
            protagonist_name=discovered_protagonist_name,
        )
        raw_regions = self._script_generator.apply_character_memory_to_regions(raw_regions, character_memory)
        speaker_identity_map = self._resolve_speaker_identities(raw_regions, cluster_name_map=cluster_name_map)
        protagonist_name = discovered_protagonist_name or self._infer_protagonist_name(raw_regions, speaker_identity_map, metadata)
        if protagonist_name and not self._is_reliable_character_name(protagonist_name):
            protagonist_name = self._infer_protagonist_name(raw_regions, speaker_identity_map, metadata)
        if protagonist_name:
            protagonist_key = " ".join(re.findall(r"[a-z]+", protagonist_name.casefold())).strip()
            if protagonist_key:
                character_dictionary.setdefault(protagonist_key, protagonist_name)
        scenes = []
        metrics["panels_with_dialogue"] = 0
        if progress_callback:
            progress_callback(96, "Building panel-by-panel dialogue scenes")
        for record in panel_region_records:
            panel_regions = self._apply_speaker_identity_map(record["scene_regions"], speaker_identity_map, character_memory)
            prepared_scene = self._build_scene(
                record["panel"],
                record["crop_bbox"],
                reading_mode,
                record["panel_path"],
                panel_regions,
                character_dictionary=character_dictionary,
                protagonist_name=protagonist_name,
            )
            prepared_scene = self._script_generator.annotate_scene(prepared_scene, panel_regions, character_memory)
            if prepared_scene.has_dialogue:
                metrics["panels_with_dialogue"] += 1
            scenes.append(prepared_scene)

        scene_clusters = self._cluster_scenes(scenes, reading_mode)
        metrics["scene_clusters"] = len(scene_clusters)
        if progress_callback:
            progress_callback(100, "Dialogue and scene context ready")
        artifacts = {
            "extraction_mode": extraction_mode,
            "reading_mode": reading_mode,
            "project_language": project_language,
            "dialogue_regions": [asdict(item) for item in raw_regions],
            "scenes": [asdict(item) for item in scenes],
            "scene_clusters": [asdict(item) for item in scene_clusters],
            "character_clusters": character_clusters,
            "character_tracking": character_tracking,
            "characters": character_memory.get("characters", {}),
            "character_identity_report": identity_report,
            "character_dictionary": character_dictionary,
            "protagonist_name": protagonist_name,
            "providers": self._provider_summary(),
            "panel_signature": panel_signature,
            "metrics": {
                **metrics,
                "elapsed_seconds": round(time.perf_counter() - started_at, 2),
            },
        }
        logger.info(
            "Dialogue pipeline finished for %s with %s/%s dialogue scenes, %s regions, %s translations in %.2fs",
            project_dir.name,
            metrics["panels_with_dialogue"],
            metrics["panels_kept"],
            metrics["regions_detected"],
            metrics["translation_count"],
            artifacts["metrics"]["elapsed_seconds"],
        )
        self._write_artifacts(ocr_dir, translations_dir, output_dir, artifacts)
        return artifacts

    def _merge_continuation_panel_regions(
        self,
        records: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Merge dialogue regions from continuation panels into the primary panel.

        When a panel spans two pages the cross-page merger marks both halves with the
        same ``logical_panel_id``.  OCR runs independently on each crop so text is
        incomplete.  This method copies dialogue regions from the secondary panel(s)
        into the primary panel record so that downstream scene-building and
        narration generation sees the full text.
        """
        logical_groups: dict[str, list[int]] = {}
        for idx, record in enumerate(records):
            panel = record["panel"]
            logical_id = getattr(panel, "logical_panel_id", None) or ""
            if logical_id:
                logical_groups.setdefault(logical_id, []).append(idx)

        if not logical_groups:
            return records

        consumed: set[int] = set()
        for logical_id, indices in logical_groups.items():
            if len(indices) < 2:
                continue
            # The primary panel is the one with the lowest order (first appearance).
            indices.sort(key=lambda i: records[i]["panel"].order)
            primary_idx = indices[0]
            for secondary_idx in indices[1:]:
                secondary_regions = records[secondary_idx]["scene_regions"]
                if secondary_regions:
                    records[primary_idx]["scene_regions"].extend(secondary_regions)
                consumed.add(secondary_idx)

        if consumed:
            records = [r for i, r in enumerate(records) if i not in consumed]
            logger.info(
                "Merged OCR text from %d continuation panels into their primary panels.",
                len(consumed),
            )
        return records

    def _build_tracking_from_clusters(self, character_clusters: list[dict[str, Any]]) -> dict[str, Any]:
        ordered_clusters = sorted(
            (
                cluster
                for cluster in character_clusters
                if str(cluster.get("cluster_id") or "").strip()
            ),
            key=lambda item: (
                min((int(value) for value in item.get("pages", []) or [0]), default=0),
                min((int(value) for value in item.get("panels", []) or [0]), default=0),
                str(item.get("cluster_id") or ""),
            ),
        )
        characters: dict[str, dict[str, Any]] = {}
        source_to_character_id: dict[str, str] = {}
        panel_characters: dict[str, list[str]] = defaultdict(list)

        for index, cluster in enumerate(ordered_clusters, start=1):
            cluster_id = str(cluster.get("cluster_id") or "").strip()
            if not cluster_id:
                continue
            stable_id = f"Character_{index}"
            source_to_character_id[cluster_id] = stable_id
            panel_ids = [str(value) for value in cluster.get("panel_ids", []) or [] if str(value)]
            pages = [int(value) for value in cluster.get("pages", []) or [] if str(value).strip()]
            panels = [int(value) for value in cluster.get("panels", []) or [] if str(value).strip()]
            # Use rich per-appearance data if supplied by GeminiCharacterRecognizer
            # (contains proper page/panel per appearance, not just the first page).
            rich_appearances: list[dict[str, Any]] = cluster.get("_appearances") or []
            if rich_appearances:
                appearances = [
                    {
                        "page": int(app.get("page") or 0),
                        "panel": int(app.get("panel_order") or 0),
                        "panel_id": str(app.get("panel_id") or ""),
                        "bbox": list(app.get("bbox") or []),
                    }
                    for app in rich_appearances
                    if app.get("panel_id")
                ]
            else:
                appearances = [
                    {
                        "page": pages[0] if pages else 0,
                        "panel": panels[0] if panels else 0,
                        "panel_id": panel_id,
                        "bbox": [],
                    }
                    for panel_id in panel_ids
                ]
            characters[stable_id] = {
                "id": stable_id,
                "name": None,
                "description": "",
                "first_panel": min(panels) if panels else 0,
                "first_page": min(pages) if pages else 0,
                "appearances": appearances,
                "appearance_count": max(len(panel_ids), int(cluster.get("appearance_count") or 0)),
                "source_character_ids": [cluster_id],
            }
            for panel_id in panel_ids:
                if stable_id not in panel_characters[panel_id]:
                    panel_characters[panel_id].append(stable_id)

        return {
            "characters": characters,
            "source_to_character_id": source_to_character_id,
            "panel_characters": panel_characters,
            "provider": "cluster-tracker-v1",
        }

    def _load_page_image(self, page_number: int, page_paths: list[Path], page_cache: dict[int, np.ndarray]) -> np.ndarray | None:
        if page_number in page_cache:
            return page_cache[page_number]
        if page_number <= 0 or page_number > len(page_paths):
            return None

        image = np.array(Image.open(page_paths[page_number - 1]).convert("RGB"))
        page_cache[page_number] = image
        while len(page_cache) > 2:
            oldest_page = next(iter(page_cache))
            if oldest_page == page_number:
                break
            page_cache.pop(oldest_page, None)
        return image

    def _load_cached_magi_page_payloads(
        self,
        cache_path: Path,
        page_numbers: list[int],
    ) -> dict[int, dict[str, Any]]:
        cached_payload = read_json(cache_path, default=None)
        if not isinstance(cached_payload, dict):
            return {}
        provider_tag = self._magi_service.provider_tag()
        reusable: dict[int, dict[str, Any]] = {}
        for page_number in page_numbers:
            candidate = cached_payload.get(page_number)
            if candidate is None:
                candidate = cached_payload.get(str(page_number))
            if not isinstance(candidate, dict):
                continue
            if str(candidate.get("provider") or "") != provider_tag:
                continue
            reusable[int(page_number)] = candidate
        return reusable

    def _load_cached_artifacts(
        self,
        manifest_path: Path,
        panel_signature: list[dict[str, Any]],
        extraction_mode: str,
    ) -> dict[str, Any] | None:
        cached = read_json(manifest_path, default=None)
        if not cached:
            return None
        if cached.get("strategy") != self._ARTIFACT_STRATEGY:
            return None
        if str(cached.get("extraction_mode") or "deep") != extraction_mode:
            return None
        if cached.get("panel_signature") != panel_signature:
            return None
        if cached.get("providers") != self._provider_summary():
            return None
        if "scene_clusters" not in cached:
            return None
        if "character_clusters" not in cached:
            return None
        if "characters" not in cached:
            return None
        return cached

    def _panel_cache_path(
        self,
        panel_hash: str,
        language_hint: str,
        reading_mode: str,
        triage_mode: str,
    ) -> Path:
        digest = hashlib.sha1(
            json.dumps(
                {
                    "strategy": "panel_dialogue_cache_v5_applevision_recall",
                    "panel_hash": panel_hash,
                    "language_hint": language_hint,
                    "reading_mode": reading_mode,
                    "triage_mode": triage_mode,
                    "providers": self._provider_summary(),
                },
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()
        return self._panel_dialogue_cache_dir / f"{digest}.json"

    def _panel_image_hash(self, panel_image: np.ndarray) -> str:
        digest = hashlib.sha1()
        digest.update(str(panel_image.shape).encode("utf-8"))
        digest.update(panel_image.tobytes())
        return digest.hexdigest()

    def _triage_panel_image(self, panel_image: np.ndarray) -> dict[str, float | str]:
        import cv2

        if panel_image.size == 0:
            return {"mode": "skip", "edge_density": 0.0, "white_ratio": 1.0, "contrast": 0.0}

        height, width = panel_image.shape[:2]
        grayscale = cv2.cvtColor(panel_image, cv2.COLOR_RGB2GRAY)
        if max(height, width) > 900:
            scale = 900 / max(height, width)
            grayscale = cv2.resize(
                grayscale,
                (max(1, int(round(width * scale))), max(1, int(round(height * scale)))),
                interpolation=cv2.INTER_AREA,
            )
        blurred = cv2.GaussianBlur(grayscale, (3, 3), 0)
        edges = cv2.Canny(blurred, 60, 160)
        edge_density = float(np.mean(edges > 0))
        contrast = float(np.std(blurred))
        white_ratio = float(np.mean(blurred > 244))
        dark_ratio = float(np.mean(blurred < 208))
        aspect_ratio = width / max(height, 1)

        if (
            white_ratio >= 0.955
            and edge_density <= 0.012
            and contrast <= 16
        ) or (
            min(width, height) <= 72
            and edge_density <= 0.02
            and contrast <= 18
        ):
            mode = "skip"
        elif (
            white_ratio >= 0.84
            and edge_density <= 0.038
            and contrast <= 30
        ) or (
            aspect_ratio > 2.6
            and white_ratio >= 0.6
            and edge_density <= 0.048
        ) or (
            dark_ratio <= 0.12
            and contrast <= 26
            and edge_density <= 0.04
        ):
            mode = "light"
        else:
            mode = "full"

        return {
            "mode": mode,
            "edge_density": edge_density,
            "white_ratio": white_ratio,
            "contrast": contrast,
        }

    def _region_from_existing_panel_text(
        self,
        panel: PanelBox,
        crop_bbox: tuple[int, int, int, int],
        language_hint: str,
    ) -> DialogueRegion | None:
        """Reuse high-confidence existing panel OCR before running expensive OCR again."""
        raw_text = str(getattr(panel, "ocr_text", "") or "").strip()
        if not raw_text:
            return None
        cleaned = clean_ocr_text(raw_text)
        manual = bool(getattr(panel, "manual_ocr_text", False))
        if not self._existing_panel_text_is_strong(cleaned, manual=manual):
            return None
        detected_language = self._panel_text_language(cleaned, language_hint)
        text_english = self._translate_to_english(cleaned, detected_language)
        return DialogueRegion(
            page=panel.page,
            panel=panel.panel,
            panel_order=panel.order,
            bbox=[0, 0, int(crop_bbox[2]), int(crop_bbox[3])],
            language=detected_language,
            text_original=cleaned,
            text_english=text_english,
            confidence=0.82 if manual else 0.62,
            detector="existing-panel-ocr",
            ocr_engine="panel-cache",
        )

    def _existing_panel_text_is_strong(self, text: str, *, manual: bool = False) -> bool:
        cleaned = clean_ocr_text(text)
        if not cleaned or not is_usable_ocr_text(cleaned) or self._looks_like_noise(cleaned):
            return False
        if manual:
            return True
        if re.search(r"[\u3040-\u30ff\u4e00-\u9fff\uac00-\ud7af]", cleaned):
            return True

        tokens = [
            token.casefold()
            for token in re.findall(r"[A-Za-zÀ-ÖØ-öø-ÿĀ-žƀ-ɏḀ-ỿ']+", cleaned)
            if token.strip("'")
        ]
        if not tokens:
            return False
        if sum(1 for token in tokens if len(token) <= 2) > max(1, len(tokens) // 2):
            return False

        common_dialogue_tokens = {
            "again",
            "because",
            "come",
            "danger",
            "enemy",
            "fine",
            "go",
            "help",
            "here",
            "how",
            "means",
            "need",
            "never",
            "now",
            "okay",
            "please",
            "ready",
            "run",
            "stop",
            "there",
            "this",
            "wait",
            "what",
            "when",
            "where",
            "who",
            "why",
            "yes",
        }
        long_tokens = [token for token in tokens if len(token) >= 3]
        if len(tokens) == 1:
            return tokens[0] in common_dialogue_tokens
        if any(token in common_dialogue_tokens for token in tokens):
            return True
        if re.search(r"[?!]", cleaned) and len(long_tokens) >= 2:
            return True
        return False

    def _should_skip_expensive_panel_ocr(
        self,
        panel: PanelBox,
        panel_image: np.ndarray,
        triage: dict[str, float | str],
        layout_scan_available: bool,
        has_page_text_in_panel: bool,
        has_magi_text_in_panel: bool,
    ) -> bool:
        """Avoid pathological deep OCR on large visual panels with no text evidence.

        MAGI/page OCR already gives us a cheap text-location pass. If both say a
        large panel has no text and any existing OCR is weak, spending minutes on
        full-panel Paddle/OpenCV OCR usually produces sound effects or garbage
        while blocking the entire script job.
        """
        if not layout_scan_available:
            return False
        if bool(getattr(panel, "manual_ocr_text", False)):
            return False
        if has_page_text_in_panel or has_magi_text_in_panel:
            return False
        if panel_image.size == 0:
            return True
        existing_text = clean_ocr_text(str(getattr(panel, "ocr_text", "") or ""))
        if self._existing_panel_text_is_strong(existing_text):
            return False
        if self._panel_likely_contains_text_signal(triage):
            return False

        height, width = panel_image.shape[:2]
        area = int(height) * int(width)
        mode = str(triage.get("mode") or "")
        # This threshold intentionally targets page-sized / action-sized manga
        # panels. Small speech bubbles still get the deeper OCR fallback.
        return mode in {"full", "light"} and (area >= 650_000 or max(height, width) >= 1000)

    def _should_trust_empty_page_ocr_for_panel(
        self,
        panel_image: np.ndarray,
        triage: dict[str, float | str],
        page_ocr_boxes: list[Any] | None,
    ) -> bool:
        """Decide whether a page OCR miss is enough evidence to skip panel OCR.

        Page-level OCR is fast and usually reliable, but it can miss good manga
        dialogue on busy pages. Only trust the miss when the panel has weak text
        signals; otherwise let the bounded panel OCR fallback try again.
        """
        if panel_image.size == 0:
            return True

        height, width = panel_image.shape[:2]
        area = int(height) * int(width)
        white_ratio = float(triage.get("white_ratio") or 0.0)
        edge_density = float(triage.get("edge_density") or 0.0)
        contrast = float(triage.get("contrast") or 0.0)
        if self._panel_likely_contains_text_signal(triage):
            return False

        # Speech bubbles and captions usually produce a visible white/edge
        # signal. If we see that, do not let page OCR silence the panel.
        likely_text_panel = (
            white_ratio >= 0.09
            or (white_ratio >= 0.055 and edge_density >= 0.04)
            or (white_ratio >= 0.03 and edge_density >= 0.065 and contrast >= 34)
        )
        if likely_text_panel:
            return False

        full_page_like = area >= 1_300_000 or max(height, width) >= 1350
        has_any_page_text = self._page_ocr_has_substantial_signal(page_ocr_boxes)
        if has_any_page_text:
            return full_page_like and white_ratio < 0.045 and edge_density < 0.05 and contrast < 24
        return full_page_like and white_ratio < 0.06 and edge_density < 0.065 and contrast < 22

    def _panel_likely_contains_text_signal(self, triage: dict[str, float | str]) -> bool:
        white_ratio = float(triage.get("white_ratio") or 0.0)
        edge_density = float(triage.get("edge_density") or 0.0)
        contrast = float(triage.get("contrast") or 0.0)
        return (
            white_ratio >= 0.09
            or (white_ratio >= 0.05 and edge_density >= 0.035)
            or (edge_density >= 0.028 and contrast >= 28)
            or contrast >= 42
        )

    def _meaningful_page_ocr_boxes(self, page_ocr_boxes: list[Any] | None) -> list[dict[str, Any]]:
        meaningful: list[dict[str, Any]] = []
        for box in page_ocr_boxes or []:
            if not isinstance(box, dict):
                continue
            text = self._clean_text(str(box.get("text") or ""))
            if not text or self._looks_like_noise(text):
                continue
            meaningful.append(box)
        return meaningful

    def _page_ocr_has_substantial_signal(self, page_ocr_boxes: list[Any] | None) -> bool:
        meaningful = self._meaningful_page_ocr_boxes(page_ocr_boxes)
        if not meaningful:
            return False

        total_tokens = 0
        for box in meaningful:
            text = self._clean_text(str(box.get("text") or ""))
            token_count = len(
                re.findall(
                    r"[A-Za-z0-9\u00C0-\u024F\u1E00-\u1EFF\u3040-\u30ff\u4e00-\u9fff\uac00-\ud7af']+",
                    text,
                )
            )
            total_tokens += token_count
            if token_count >= 4:
                return True
            if token_count >= 3 and len(text) >= 8:
                return True
            if token_count >= 2 and len(text) >= 10:
                return True

        return len(meaningful) >= 2 and total_tokens >= 3

    def _text_has_substantial_signal(self, text: str) -> bool:
        cleaned = self._clean_text(text)
        if not cleaned or self._looks_like_noise(cleaned):
            return False

        token_count = len(
            re.findall(
                r"[A-Za-z0-9\u00C0-\u024F\u1E00-\u1EFF\u3040-\u30ff\u4e00-\u9fff\uac00-\ud7af']+",
                cleaned,
            )
        )
        if token_count >= 4:
            return True
        if token_count >= 2 and len(cleaned) >= 10:
            return True
        return False

    def _candidate_set_has_substantial_signal(self, candidates: list[OCRCandidate]) -> bool:
        if not candidates:
            return False

        total_tokens = 0
        for candidate in candidates:
            cleaned = self._clean_text(candidate.text)
            token_count = len(
                re.findall(
                    r"[A-Za-z0-9\u00C0-\u024F\u1E00-\u1EFF\u3040-\u30ff\u4e00-\u9fff\uac00-\ud7af']+",
                    cleaned,
                )
            )
            total_tokens += token_count
            if self._text_has_substantial_signal(cleaned):
                return True

        return len(candidates) >= 2 and total_tokens >= 3

    def _scene_regions_have_substantial_signal(self, regions: list[DialogueRegion]) -> bool:
        if not regions:
            return False

        total_tokens = 0
        for region in regions:
            cleaned = self._clean_text(region.text_english or region.text_original or "")
            token_count = len(
                re.findall(
                    r"[A-Za-z0-9\u00C0-\u024F\u1E00-\u1EFF\u3040-\u30ff\u4e00-\u9fff\uac00-\ud7af']+",
                    cleaned,
                )
            )
            total_tokens += token_count
            if self._text_has_substantial_signal(cleaned):
                return True

        return len(regions) >= 2 and total_tokens >= 3

    def _load_cached_panel_dialogue(
        self,
        panel_hash: str,
        language_hint: str,
        reading_mode: str,
        triage_mode: str,
    ) -> list[dict[str, Any]] | None:
        path = self._panel_cache_path(panel_hash, language_hint, reading_mode, triage_mode)
        payload = read_json(path, default=None)
        if not isinstance(payload, dict):
            return None
        regions = payload.get("regions")
        return regions if isinstance(regions, list) else None

    def _store_cached_panel_dialogue(
        self,
        panel_hash: str,
        language_hint: str,
        reading_mode: str,
        triage_mode: str,
        scene_regions: list[DialogueRegion],
    ) -> None:
        path = self._panel_cache_path(panel_hash, language_hint, reading_mode, triage_mode)
        write_json(
            path,
            {
                "regions": [
                    {
                        "bbox": [int(value) for value in region.bbox],
                        "language": region.language,
                        "text_original": region.text_original,
                        "text_english": region.text_english,
                        "bubble_bbox": [int(value) for value in region.bubble_bbox] if region.bubble_bbox else None,
                        "confidence": region.confidence,
                        "detector": region.detector,
                        "ocr_engine": region.ocr_engine,
                    }
                    for region in scene_regions
                ]
            },
        )

    def _hydrate_cached_dialogue_regions(
        self,
        panel: PanelBox,
        crop_bbox: tuple[int, int, int, int],
        cached_regions: list[dict[str, Any]],
    ) -> list[DialogueRegion]:
        hydrated: list[DialogueRegion] = []
        default_bbox = [0, 0, int(crop_bbox[2]), int(crop_bbox[3])]
        for index, item in enumerate(cached_regions, start=1):
            if not isinstance(item, dict):
                continue
            bbox = item.get("bbox") if isinstance(item.get("bbox"), list) else default_bbox
            bubble_bbox = item.get("bubble_bbox") if isinstance(item.get("bubble_bbox"), list) else bbox
            hydrated.append(
                DialogueRegion(
                    page=panel.page,
                    panel=panel.panel,
                    panel_order=panel.order,
                    bbox=[int(value) for value in bbox],
                    language=str(item.get("language") or ""),
                    text_original=str(item.get("text_original") or ""),
                    text_english=str(item.get("text_english") or ""),
                    bubble_id=f"panel-{panel.order}-bubble-{index}",
                    bubble_bbox=[int(value) for value in bubble_bbox],
                    confidence=float(item["confidence"]) if isinstance(item.get("confidence"), (int, float)) else None,
                    detector=str(item.get("detector") or "cache"),
                    ocr_engine=str(item.get("ocr_engine") or "cache"),
                )
            )
        return hydrated

    def _panel_signature(self, panels: list[PanelBox]) -> list[dict[str, Any]]:
        return [
            {
                "id": panel.id,
                "order": panel.order,
                "page": panel.page,
                "x": panel.x,
                "y": panel.y,
                "width": panel.width,
                "height": panel.height,
                "keep": panel.keep,
                "manual_keep": panel.manual_keep,
                "logical_panel_id": panel.logical_panel_id,
                "multi_page_panel": panel.multi_page_panel,
            }
            for panel in panels
        ]

    def _write_artifacts(self, ocr_dir: Path, translations_dir: Path, output_dir: Path, artifacts: dict[str, Any]) -> None:
        artifacts["strategy"] = self._ARTIFACT_STRATEGY
        raw_payload = [
            {
                "page": item["page"],
                "panel": item["panel"],
                "panel_order": item["panel_order"],
                "bbox": item["bbox"],
                "bubble_id": item.get("bubble_id"),
                "bubble_bbox": item.get("bubble_bbox"),
                "language": item["language"],
                "text_original": item["text_original"],
                "confidence": item["confidence"],
                "detector": item["detector"],
                "ocr_engine": item["ocr_engine"],
                "character_id": item.get("character_id"),
                "stable_character_id": item.get("stable_character_id"),
                "speaker_name": item.get("speaker_name"),
                "speaker_label": item.get("speaker_label"),
                "character_display_name": item.get("character_display_name"),
            }
            for item in artifacts["dialogue_regions"]
        ]
        translated_payload = [
            {
                **item,
                "text_english": item["text_english"],
            }
            for item in artifacts["dialogue_regions"]
        ]
        write_json(ocr_dir / "dialogue_regions.json", raw_payload)
        write_json(translations_dir / "dialogue_regions_translated.json", translated_payload)
        write_json(output_dir / "character_identity_report.json", artifacts.get("character_identity_report", {}))
        write_json(
            output_dir / "speaker_attributions.json",
            [
                {
                    "panel_id": scene.get("panel_id"),
                    "panel_order": scene.get("panel_order"),
                    "primary_speaker_name": scene.get("primary_speaker_name"),
                    "protagonist_name": scene.get("protagonist_name"),
                    "speaker_names": scene.get("speaker_names", []),
                    "character_names": scene.get("character_names", []),
                    "character_ids": scene.get("character_ids", []),
                    "character_labels": scene.get("character_labels", []),
                    "dialogue_entries": scene.get("dialogue_entries", []),
                }
                for scene in artifacts["scenes"]
            ],
        )
        write_json(output_dir / "gemini_scenes.json", artifacts["scenes"])
        write_json(output_dir / "scene_clusters.json", artifacts.get("scene_clusters", []))
        write_json(output_dir / "character_clusters.json", artifacts.get("character_clusters", []))
        write_json(output_dir / "character_tracking.json", artifacts.get("character_tracking", {}))
        write_json(output_dir / "characters.json", artifacts.get("characters", {}))
        write_json(output_dir / "character_dictionary.json", artifacts.get("character_dictionary", {}))
        write_json(output_dir / "dialogue_pipeline_manifest.json", artifacts)

    def _extract_panel_candidates(
        self,
        panel_image: np.ndarray,
        language_hint: str,
        reading_mode: str,
        cancel_callback: callable | None = None,
    ) -> list[OCRCandidate]:
        candidates = self._extract_paddle_candidates(panel_image, language_hint)
        if candidates:
            candidates = self._repair_low_confidence_candidates(panel_image, candidates, language_hint, cancel_callback)
            if self._candidate_set_has_substantial_signal(candidates):
                return self._sort_candidates(candidates, reading_mode)
            fallback_candidates = self._extract_opencv_fallback_candidates(
                panel_image,
                language_hint,
                reading_mode,
                cancel_callback=cancel_callback,
                limit=16,
            )
            merged = self._dedupe_candidates(
                candidates + fallback_candidates,
                panel_image.shape[1],
                panel_image.shape[0],
            )
            return self._sort_candidates(merged, reading_mode)

        return self._extract_opencv_fallback_candidates(
            panel_image,
            language_hint,
            reading_mode,
            cancel_callback=cancel_callback,
            limit=16,
        )

    def _extract_opencv_fallback_candidates(
        self,
        panel_image: np.ndarray,
        language_hint: str,
        reading_mode: str,
        cancel_callback: callable | None = None,
        *,
        limit: int = 10,
    ) -> list[OCRCandidate]:
        opencv_boxes = self._sort_regions(self._detect_text_regions_opencv(panel_image), reading_mode)
        fallback_candidates: list[OCRCandidate] = []
        for region_box in opencv_boxes[:limit]:
            if cancel_callback:
                cancel_callback()
            region_crop = self._crop_box(panel_image, region_box)
            text_original, confidence, ocr_engine = self._ocr_region(region_crop, language_hint)
            cleaned = self._clean_text(text_original)
            if not cleaned or self._looks_like_noise(cleaned):
                continue
            fallback_candidates.append(
                OCRCandidate(
                    bbox=[int(region_box[0]), int(region_box[1]), int(region_box[2]), int(region_box[3])],
                    text=cleaned,
                    confidence=confidence,
                    detector="opencv",
                    ocr_engine=ocr_engine,
                )
            )
        return self._dedupe_candidates(fallback_candidates, panel_image.shape[1], panel_image.shape[0])

    def _extract_paddle_candidates(self, panel_image: np.ndarray, language_hint: str) -> list[OCRCandidate]:
        candidates: list[OCRCandidate] = []
        for item in self._comic_ocr.detect_candidates(panel_image, language_hint):
            bbox = self._normalise_bbox(item["bbox"], panel_image.shape[1], panel_image.shape[0])
            if bbox is None:
                continue
            if item.get("confidence") is not None and float(item["confidence"]) < 0.35:
                continue
            candidates.append(
                OCRCandidate(
                    bbox=bbox,
                    text=self._clean_text(str(item.get("text") or "")),
                    confidence=float(item["confidence"]) if isinstance(item.get("confidence"), (int, float)) else None,
                    detector="hybrid-comic-ocr",
                    ocr_engine=str(item.get("ocr_engine") or "paddleocr"),
                )
            )
        return self._dedupe_candidates(candidates, panel_image.shape[1], panel_image.shape[0])

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

    def _dedupe_candidates(self, candidates: list[OCRCandidate], image_width: int, image_height: int) -> list[OCRCandidate]:
        cleaned: list[OCRCandidate] = []
        for candidate in candidates:
            bbox = self._normalise_bbox(candidate.bbox, image_width, image_height)
            if bbox is None:
                continue
            candidate.bbox = bbox
            duplicate_index: int | None = None
            for index, other in enumerate(cleaned):
                same_text = candidate.text.casefold() == other.text.casefold()
                overlaps = self._iou(tuple(candidate.bbox), tuple(other.bbox)) >= 0.55
                contains = self._bbox_contains(tuple(candidate.bbox), tuple(other.bbox)) or self._bbox_contains(tuple(other.bbox), tuple(candidate.bbox))
                if same_text or overlaps or contains:
                    duplicate_index = index
                    break
            if duplicate_index is None:
                cleaned.append(candidate)
                continue

            other = cleaned[duplicate_index]
            other_score = other.confidence if other.confidence is not None else -1.0
            candidate_score = candidate.confidence if candidate.confidence is not None else -1.0
            if candidate_score > other_score or len(candidate.text) > len(other.text):
                cleaned[duplicate_index] = candidate
        return cleaned

    def _sort_candidates(self, candidates: list[OCRCandidate], reading_mode: str) -> list[OCRCandidate]:
        if reading_mode == "webtoon":
            return sorted(candidates, key=lambda item: (item.bbox[1], item.bbox[0]))
        return sorted(candidates, key=lambda item: (item.bbox[1], -item.bbox[0]))

    def _repair_low_confidence_candidates(
        self,
        panel_image: np.ndarray,
        candidates: list[OCRCandidate],
        language_hint: str,
        cancel_callback: callable | None = None,
    ) -> list[OCRCandidate]:
        repaired: list[OCRCandidate] = []
        for candidate in candidates:
            if cancel_callback:
                cancel_callback()
            needs_retry = candidate.confidence is None or candidate.confidence < 0.72 or len(candidate.text) < 6
            if not needs_retry:
                repaired.append(candidate)
                continue

            region_crop = self._crop_box(panel_image, tuple(candidate.bbox))
            retry_text, retry_confidence, retry_engine = self._ocr_region(region_crop, language_hint)
            retry_text = self._clean_text(retry_text)
            if self._is_better_candidate(retry_text, retry_confidence, candidate.text, candidate.confidence):
                candidate.text = retry_text
                candidate.confidence = retry_confidence
                candidate.ocr_engine = retry_engine
            repaired.append(candidate)

        return self._dedupe_candidates(repaired, panel_image.shape[1], panel_image.shape[0])

    def _build_dialogue_regions(
        self,
        panel: PanelBox,
        candidates: list[OCRCandidate],
        language_hint: str,
        metrics: dict[str, Any],
    ) -> list[DialogueRegion]:
        if not candidates:
            return []

        region_payloads: list[dict[str, Any]] = []
        for candidate in candidates:
            detected_language = self._panel_text_language(candidate.text, language_hint)
            region_payloads.append(
                {
                    "candidate": candidate,
                    "language": detected_language,
                    "text_original": self._dialogue_cleaner.merge_broken_lines(candidate.text),
                }
            )

        translations = self._translate_payloads(region_payloads)
        scene_regions: list[DialogueRegion] = []
        for bubble_index, payload in enumerate(region_payloads, start=1):
            candidate = payload["candidate"]
            text_original = payload["text_original"]
            text_english = translations[(payload["language"], text_original)]
            if text_english and text_english.strip() != text_original.strip():
                metrics["translation_count"] += 1
            scene_regions.append(
                DialogueRegion(
                    page=panel.page,
                    panel=panel.panel,
                    panel_order=panel.order,
                    bbox=[int(candidate.bbox[0]), int(candidate.bbox[1]), int(candidate.bbox[2]), int(candidate.bbox[3])],
                    language=payload["language"],
                    text_original=text_original,
                    text_english=text_english,
                    bubble_id=f"panel-{panel.order}-bubble-{bubble_index}",
                    bubble_bbox=[int(candidate.bbox[0]), int(candidate.bbox[1]), int(candidate.bbox[2]), int(candidate.bbox[3])],
                    confidence=candidate.confidence,
                    detector=candidate.detector,
                    ocr_engine=candidate.ocr_engine,
                )
            )
        return scene_regions

    def _prewarm_translation_cache(
        self,
        magi_page_payloads: dict[int, dict[str, Any]],
        project_language: str,
    ) -> None:
        """Before the per-panel loop, collect all text visible in MAGI and page OCR
        data and translate it in large batches.  This pre-populates the shared
        translation cache so most per-panel calls return instantly from cache
        instead of making individual blocking HTTP requests to Gemini.
        """
        lang_norm = self._normalise_language_code(project_language)
        if lang_norm in ("en", "a", ""):
            return

        # Collect candidate texts from both pre-computed sources.
        texts: list[str] = []
        for boxes in self._page_text_boxes.values():
            for box in (boxes or []):
                t = self._clean_text(str(box.get("text") or "")).strip()
                if t and len(t) > 2:
                    texts.append(t)
        for payload in magi_page_payloads.values():
            for box in (payload.get("text_boxes") or []):
                t = self._clean_text(str(box.get("text") or "")).strip()
                if t and len(t) > 2:
                    texts.append(t)

        if not texts:
            return

        # Deduplicate and skip already-cached entries.
        unique: list[str] = list(dict.fromkeys(
            t for t in texts
            if (lang_norm, t) not in self._translated_text_cache
        ))

        if not unique:
            return

        logger.info(
            "Pre-warming translation cache: %d unique texts in '%s'",
            len(unique),
            lang_norm,
        )

        # Translate in large batches — one Gemini call covers up to 80 strings.
        _PREWARM_BATCH = 80
        hint = self._translation_context_hint
        for i in range(0, len(unique), _PREWARM_BATCH):
            batch = unique[i : i + _PREWARM_BATCH]
            try:
                translated = self._translator.translate_batch(batch, lang_norm, hint)
                for orig, trans in zip(batch, translated, strict=False):
                    self._translated_text_cache[(lang_norm, orig)] = trans
            except Exception as exc:
                logger.warning(
                    "Pre-warm translation batch %d/%d failed: %s",
                    i // _PREWARM_BATCH + 1,
                    math.ceil(len(unique) / _PREWARM_BATCH),
                    exc,
                )

    def _translate_payloads(self, region_payloads: list[dict[str, Any]]) -> dict[tuple[str, str], str]:
        translations: dict[tuple[str, str], str] = {}
        grouped: dict[str, list[str]] = defaultdict(list)

        for payload in region_payloads:
            language = self._normalise_language_code(payload["language"])
            text_original = payload["text_original"].strip()
            key = (language, text_original)
            if not text_original:
                translations[key] = ""
                continue
            if language in {"en", "a"}:
                translations[key] = text_original
                continue
            cached = self._translated_text_cache.get(key)
            if cached is not None:
                translations[key] = cached
                continue
            if text_original not in grouped[language]:
                grouped[language].append(text_original)

        for language, texts in grouped.items():
            translated_batch = self._translator.translate_batch(texts, language, getattr(self, "_translation_context_hint", ""))
            for original, translated in zip(texts, translated_batch, strict=False):
                key = (language, original)
                translations[key] = translated
                self._translated_text_cache[key] = translated

        return translations

    def _translate_batch_to_english(self, texts: list[str], language_code: str) -> list[str]:
        return self._translator.translate_batch(texts, language_code)

    def _bbox_contains(self, outer: tuple[int, int, int, int], inner: tuple[int, int, int, int]) -> bool:
        ox, oy, ow, oh = outer
        ix, iy, iw, ih = inner
        return ox <= ix and oy <= iy and ox + ow >= ix + iw and oy + oh >= iy + ih

    def _build_scene(
        self,
        panel: PanelBox,
        panel_bbox: tuple[int, int, int, int],
        reading_mode: str,
        panel_path: Path,
        scene_regions: list[DialogueRegion],
        character_dictionary: dict[str, str] | None = None,
        protagonist_name: str | None = None,
    ) -> DialogueScene:
        dialogue_original = self._prepare_scene_lines([region.text_original for region in scene_regions if region.text_original])
        dialogue_english = self._prepare_scene_lines([region.text_english for region in scene_regions if region.text_english])
        detected_lines = dialogue_english or dialogue_original
        detected_text = " ".join(detected_lines).strip()
        dialogue_entries = self._dialogue_entries_from_regions(scene_regions)
        speaker_names = sorted(
            {
                str(entry.get("speaker_name") or "").strip()
                for entry in dialogue_entries
                if self._is_reliable_character_name(str(entry.get("speaker_name") or "").strip())
            }
        )
        character_ids = sorted(
            {
                str(region.stable_character_id or "").strip()
                for region in scene_regions
                if str(region.stable_character_id or "").strip()
            }
        )
        character_labels = sorted(
            {
                str(region.character_display_name or "").strip()
                for region in scene_regions
                if self._is_reliable_character_name(str(region.character_display_name or "").strip())
            }
        )
        detected_names: list[str] = []
        if character_dictionary:
            for line in dialogue_original + dialogue_english:
                for name in self._character_names.character_names_in_text(line, character_dictionary):
                    if name not in detected_names:
                        detected_names.append(name)
        for label in character_labels:
            if label not in detected_names:
                detected_names.append(label)
        for name in speaker_names:
            if name not in detected_names:
                detected_names.append(name)
        if protagonist_name and protagonist_name not in detected_names and any(re.search(r"\b(i|i'm|i’ve|i'll|my|me)\b", line.casefold()) for line in dialogue_original + dialogue_english):
            detected_names.append(protagonist_name)
        return DialogueScene(
            scene=panel.order,
            panel_id=panel.id,
            page=panel.page,
            panel=panel.panel,
            panel_order=panel.order,
            reading_mode=reading_mode,
            dialogue=dialogue_english,
            dialogue_original=dialogue_original,
            languages=sorted({region.language for region in scene_regions if region.language}),
            panel_bbox=[int(panel_bbox[0]), int(panel_bbox[1]), int(panel_bbox[2]), int(panel_bbox[3])],
            panel_path=str(panel_path),
            has_dialogue=bool(detected_text),
            detected_text=detected_text,
            dialogue_entries=dialogue_entries,
            speaker_names=speaker_names,
            character_names=detected_names,
            character_ids=character_ids,
            character_labels=character_labels,
            primary_speaker_name=self._primary_speaker_name(dialogue_entries),
            protagonist_name=protagonist_name,
            logical_panel_id=panel.logical_panel_id or panel.id,
            multi_page_panel=bool(panel.multi_page_panel),
            spans_pages=sorted({int(panel.page), *(int(value) for value in panel.spans_pages or [])}),
        )

    def _magi_candidates_for_panel(
        self,
        panel: PanelBox,
        crop_bbox: tuple[int, int, int, int],
        page_payload: dict[str, Any] | None,
        image_width: int,
        image_height: int,
        reading_mode: str,
    ) -> list[OCRCandidate]:
        if not page_payload:
            return []

        panel_box = (int(panel.x), int(panel.y), int(panel.width), int(panel.height))
        association_box = self._expand_panel_association_box(panel_box, image_width, image_height)
        crop_x, crop_y, crop_w, crop_h = [int(value) for value in crop_bbox]
        candidates: list[OCRCandidate] = []

        for item in page_payload.get("texts", []) or []:
            text = self._clean_text(str(item.get("text") or "").strip())
            if not text or not bool(item.get("is_dialogue", True)):
                continue
            bbox = self._coerce_xywh_bbox(item.get("bbox"))
            if bbox is None:
                continue
            if not self._candidate_belongs_to_panel(bbox, panel_box, association_box):
                continue

            local_x = max(int(bbox[0]) - crop_x, 0)
            local_y = max(int(bbox[1]) - crop_y, 0)
            local_w = max(1, min(int(bbox[2]), max(crop_w - local_x, 1)))
            local_h = max(1, min(int(bbox[3]), max(crop_h - local_y, 1)))
            candidates.append(
                OCRCandidate(
                    bbox=[local_x, local_y, local_w, local_h],
                    text=text,
                    detector="magi-hf",
                    ocr_engine="magi-hf",
                )
            )

        candidates.sort(
            key=lambda candidate: (
                int(candidate.bbox[1]),
                -int(candidate.bbox[0]) if reading_mode != "webtoon" else int(candidate.bbox[0]),
            )
        )
        return self._dedupe_candidates(candidates, crop_w, crop_h)

    def _coerce_xywh_bbox(self, value: Any) -> tuple[int, int, int, int] | None:
        candidate = value.tolist() if hasattr(value, "tolist") else value
        if not isinstance(candidate, (list, tuple)) or len(candidate) < 4:
            return None
        try:
            x, y, width, height = [int(round(float(component))) for component in candidate[:4]]
        except (TypeError, ValueError):
            return None
        width = max(width, 1)
        height = max(height, 1)
        return (x, y, width, height)

    def _attach_magi_speaker_candidates(
        self,
        scene_regions: list[DialogueRegion],
        crop_bbox: tuple[int, int, int, int],
        panel: PanelBox,
        page_payload: dict[str, Any] | None,
        image_width: int,
        image_height: int,
        reading_mode: str,
    ) -> list[DialogueRegion]:
        if not scene_regions or not page_payload:
            return scene_regions

        panel_box = (int(panel.x), int(panel.y), int(panel.width), int(panel.height))
        association_box = self._expand_panel_association_box(panel_box, image_width, image_height)
        page_texts = [
            item
            for item in page_payload.get("texts", [])
            if self._candidate_belongs_to_panel(
                tuple(int(value) for value in item.get("bbox", [0, 0, 0, 0])),
                panel_box,
                association_box,
            )
        ]
        if not page_texts:
            return scene_regions

        page_texts = sorted(
            page_texts,
            key=lambda item: (
                int(item["bbox"][1]),
                -int(item["bbox"][0]) if reading_mode != "webtoon" else int(item["bbox"][0]),
            ),
        )
        used_text_indexes: set[int] = set()
        crop_x, crop_y, _, _ = crop_bbox

        for region_index, region in enumerate(scene_regions):
            region_page_box = (
                int(crop_x + region.bbox[0]),
                int(crop_y + region.bbox[1]),
                int(region.bbox[2]),
                int(region.bbox[3]),
            )
            best_match = self._match_region_to_magi_text(region_page_box, page_texts, used_text_indexes)
            if best_match is None and region_index < len(page_texts):
                candidate = page_texts[region_index]
                if int(candidate.get("text_index", -1)) not in used_text_indexes:
                    best_match = candidate
            if best_match is None:
                continue

            text_index = int(best_match.get("text_index", -1))
            if text_index >= 0:
                used_text_indexes.add(text_index)
            region.character_id = str(best_match.get("character_id") or "").strip() or None
            region.speaker_label = "Protagonist" if region.character_id else "Other"

        return scene_regions

    def _match_region_to_magi_text(
        self,
        region_page_box: tuple[int, int, int, int],
        page_texts: list[dict[str, Any]],
        used_text_indexes: set[int],
    ) -> dict[str, Any] | None:
        best_match: dict[str, Any] | None = None
        best_score = -1.0

        for item in page_texts:
            text_index = int(item.get("text_index", -1))
            if text_index in used_text_indexes:
                continue
            candidate_box = tuple(int(value) for value in item.get("bbox", [0, 0, 0, 0]))
            iou = self._iou(region_page_box, candidate_box)
            overlap = self._intersection_area(region_page_box, candidate_box)
            center_score = self._center_proximity_score(region_page_box, candidate_box)
            contains_bonus = 0.35 if self._bbox_contains(candidate_box, region_page_box) or self._bbox_contains(region_page_box, candidate_box) else 0.0
            score = iou * 2.4 + center_score + contains_bonus
            if overlap > 0:
                score += 0.3
            if score > best_score:
                best_score = score
                best_match = item

        if best_match is not None and best_score >= 0.48:
            return best_match
        return None

    def _center_proximity_score(
        self,
        left: tuple[int, int, int, int],
        right: tuple[int, int, int, int],
    ) -> float:
        left_center_x = left[0] + left[2] / 2
        left_center_y = left[1] + left[3] / 2
        right_center_x = right[0] + right[2] / 2
        right_center_y = right[1] + right[3] / 2
        distance = math.dist((left_center_x, left_center_y), (right_center_x, right_center_y))
        normalizer = max(left[2], right[2], left[3], right[3], 1)
        return max(0.0, 1.0 - (distance / normalizer))

    def _apply_speaker_identity_map(
        self,
        scene_regions: list[DialogueRegion],
        speaker_identity_map: dict[str, dict[str, str]],
        character_memory_payload: dict[str, Any] | None = None,
    ) -> list[DialogueRegion]:
        characters = character_memory_payload.get("characters", {}) if isinstance(character_memory_payload, dict) else {}
        for region in scene_regions:
            if region.character_id and region.character_id in speaker_identity_map:
                identity = speaker_identity_map[region.character_id]
                region.character_id = identity.get("character_id") or region.character_id
                region.speaker_name = identity.get("speaker_name") or None
                region.speaker_label = identity.get("speaker_label") or region.speaker_label
            stable_id = str(region.stable_character_id or "").strip()
            if stable_id and stable_id in characters:
                display_name = str(characters[stable_id].get("display_name") or stable_id).strip()
                region.character_display_name = display_name or region.character_display_name
                if (not region.speaker_name or region.speaker_name in {"Other", "Protagonist"}) and display_name:
                    region.speaker_name = display_name
                if (not region.speaker_label or region.speaker_label in {"Other", "Protagonist"}) and display_name:
                    region.speaker_label = display_name
            elif not region.speaker_label:
                region.speaker_label = "Other"
        return scene_regions

    def _resolve_speaker_identities(
        self,
        raw_regions: list[DialogueRegion],
        cluster_name_map: dict[str, str] | None = None,
    ) -> dict[str, dict[str, str]]:
        grouped: dict[str, list[DialogueRegion]] = defaultdict(list)
        addressed_names: Counter[str] = Counter()
        first_person_scores: Counter[str] = Counter()

        for region in raw_regions:
            if not region.character_id:
                continue
            grouped[region.character_id].append(region)
            text = region.text_original or region.text_english
            first_person_scores[region.character_id] += self._first_person_score(text)
            addressed_names.update(self._extract_addressed_names(text))

        identity_map: dict[str, dict[str, str]] = {}
        used_names: set[str] = set()
        protagonist_key: str | None = None
        protagonist_score = float("-inf")

        for character_id, regions in grouped.items():
            if cluster_name_map and character_id in cluster_name_map:
                speaker_name = str(cluster_name_map[character_id] or "").strip()
                if speaker_name:
                    identity_map[character_id] = {
                        "character_id": self._speaker_slug(speaker_name),
                        "speaker_name": speaker_name,
                        "speaker_label": speaker_name,
                    }
                    used_names.add(speaker_name.casefold())
            score = first_person_scores[character_id] + len(regions) * 0.2
            if score > protagonist_score:
                protagonist_key = character_id
                protagonist_score = score

            if character_id in identity_map:
                continue
            self_named: Counter[str] = Counter()
            for region in regions:
                self_named.update(self._extract_self_identified_names(region.text_original or region.text_english))
            if self_named:
                speaker_name = self_named.most_common(1)[0][0]
                identity_map[character_id] = {
                    "character_id": self._speaker_slug(speaker_name),
                    "speaker_name": speaker_name,
                    "speaker_label": speaker_name,
                }
                used_names.add(speaker_name.casefold())

        if protagonist_key and protagonist_key not in identity_map:
            protagonist_name = ""
            for candidate_name, count in addressed_names.most_common():
                if count < 2:
                    continue
                if candidate_name.casefold() in used_names:
                    continue
                protagonist_name = candidate_name
                break
            if protagonist_name:
                identity_map[protagonist_key] = {
                    "character_id": self._speaker_slug(protagonist_name),
                    "speaker_name": protagonist_name,
                    "speaker_label": protagonist_name,
                }
            else:
                identity_map[protagonist_key] = {
                    "character_id": "protagonist",
                    "speaker_name": "Protagonist",
                    "speaker_label": "Protagonist",
                }

        protagonist_identity = identity_map.get(protagonist_key) if protagonist_key else None
        if protagonist_identity and protagonist_identity.get("speaker_name") not in {"", "Other"} and not cluster_name_map:
            for character_id, regions in grouped.items():
                if character_id in identity_map:
                    continue
                if first_person_scores[character_id] <= 0:
                    continue
                if len(regions) <= 1 and first_person_scores[character_id] < 2:
                    continue
                identity_map[character_id] = protagonist_identity

        for character_id in grouped:
            if character_id in identity_map:
                continue
            identity_map[character_id] = {
                "character_id": character_id,
                "speaker_name": "",
                "speaker_label": "Other",
            }

        alias_lookup: dict[str, dict[str, str]] = {}
        for character_id, identity in identity_map.items():
            speaker_name = str(identity.get("speaker_name") or "").strip()
            if not speaker_name:
                continue
            key = speaker_name.casefold()
            if key not in alias_lookup:
                alias_lookup[key] = identity
                continue
            identity_map[character_id] = alias_lookup[key]

        return identity_map

    def _infer_protagonist_name(
        self,
        raw_regions: list[DialogueRegion],
        speaker_identity_map: dict[str, dict[str, str]],
        metadata: ChapterMetadata,
    ) -> str | None:
        metadata_name = self._metadata_character_name(metadata)
        if metadata_name:
            return metadata_name

        explicit_panels: dict[str, set[int]] = defaultdict(set)
        for region in raw_regions:
            text = region.text_original or region.text_english
            for candidate_name in self._extract_self_identified_names(text):
                explicit_panels[candidate_name].add(int(region.panel_order))
            for candidate_name in self._extract_addressed_names(text):
                explicit_panels[candidate_name].add(int(region.panel_order))

        ranked_explicit = sorted(
            ((len(panel_orders), candidate_name) for candidate_name, panel_orders in explicit_panels.items()),
            reverse=True,
        )
        for panel_count, candidate_name in ranked_explicit:
            if panel_count >= 2:
                return candidate_name

        identity_counts: Counter[str] = Counter()
        for region in raw_regions:
            if not region.character_id:
                continue
            identity = speaker_identity_map.get(region.character_id, {})
            speaker_name = str(identity.get("speaker_name") or "").strip()
            if not speaker_name or speaker_name in {"Other", "Protagonist"}:
                continue
            identity_counts[speaker_name] += 1

        return identity_counts.most_common(1)[0][0] if identity_counts else None

    def _metadata_character_name(self, metadata: ChapterMetadata) -> str | None:
        raw = metadata.raw if isinstance(metadata.raw, dict) else {}
        texts: list[str] = []
        for relation in raw.get("relationships", []) if isinstance(raw.get("relationships"), list) else []:
            if not isinstance(relation, dict) or relation.get("type") != "manga":
                continue
            attributes = relation.get("attributes")
            if not isinstance(attributes, dict):
                continue
            description = attributes.get("description")
            excerpt = self._metadata_description_excerpt(description)
            if excerpt:
                texts.append(excerpt)

        patterns = (
            r"\bhero\s+([A-Z][a-z]{1,14}(?:\s+[A-Z][a-z]{1,14}){0,2})\b",
            r"\b([A-Z][a-z]{1,14}(?:\s+[A-Z][a-z]{1,14}){0,2})\s+(?:was|is|has|wakes|awakens|returns|reborn|uses|must|begins)\b",
        )
        stop_tokens = {
            "Apocalypse", "Freeze", "Frozen", "Global", "Tianjin", "World", "Winter", "Official", "Traditional",
            "Simplified", "Chinese", "Japanese", "Spanish", "German", "French", "Indonesian", "Thai", "English",
            "Trailer", "Website", "Translation", "Webtoon",
        }
        for text in texts:
            for pattern in patterns:
                match = re.search(pattern, text)
                if not match:
                    continue
                candidate = match.group(1).strip()
                tokens = candidate.split()
                if any(token in stop_tokens for token in tokens):
                    continue
                return candidate
        return None

    def _is_reliable_character_name(self, name: str | None) -> bool:
        if looks_like_false_character_name(name):
            return False
        tokens = [token for token in re.findall(r"[a-z]+", str(name or "").casefold()) if token]
        if not tokens:
            return False
        banned = {
            "sorry", "wait", "okay", "thanks", "please", "hello", "customer", "manager", "world",
            "freeze", "apocalypse", "yes", "yeah", "yep", "no", "nope",
            "be", "dead", "jle", "trle", "nati", "salur", "sauri",
        }
        if any(token in banned for token in tokens):
            return False
        if len(tokens) == 1 and len(tokens[0]) < 4:
            return False
        return True

    def _metadata_description_excerpt(self, description: Any) -> str:
        if isinstance(description, dict):
            ordered_values: list[str] = []
            for key in ("en", "en-us", "en-gb", "ja", "ko", "es", "fr", "de"):
                value = description.get(key)
                if isinstance(value, str) and value.strip():
                    ordered_values.append(value)
            if not ordered_values:
                ordered_values.extend(str(value) for value in description.values() if isinstance(value, str) and value.strip())
            text = ordered_values[0] if ordered_values else ""
        elif isinstance(description, str):
            text = description
        else:
            text = ""
        if not text:
            return ""
        text = re.split(r"\n\s*---+\s*\n", text, maxsplit=1)[0]
        text = re.sub(r"\[[^\]]+\]\([^)]+\)", " ", text)
        text = re.sub(r"https?://\S+", " ", text)
        lines = []
        for raw_line in text.splitlines():
            cleaned = raw_line.strip()
            if not cleaned or cleaned.startswith(("-", ">", "*")):
                continue
            lines.append(cleaned)
        return re.sub(r"\s+", " ", " ".join(lines)).strip()

    def _dialogue_entries_from_regions(self, scene_regions: list[DialogueRegion]) -> list[dict[str, str]]:
        entries: list[dict[str, str]] = []
        for region in scene_regions:
            text = str(region.text_english or region.text_original or "").strip()
            if not text:
                continue
            speaker_name = str(region.speaker_name or region.speaker_label or "Other").strip() or "Other"
            entries.append(
                {
                    "speaker_name": speaker_name,
                    "text": text,
                }
            )
        return entries

    def _primary_speaker_name(self, dialogue_entries: list[dict[str, str]]) -> str | None:
        counts: Counter[str] = Counter()
        for entry in dialogue_entries:
            speaker_name = str(entry.get("speaker_name") or "").strip()
            if not speaker_name:
                continue
            counts[speaker_name] += 1
        for speaker_name, _ in counts.most_common():
            if speaker_name != "Other":
                return speaker_name
        return counts.most_common(1)[0][0] if counts else None

    def _extract_self_identified_names(self, text: str) -> list[str]:
        cleaned = clean_ocr_lines([text])
        if not cleaned:
            return []
        line = cleaned[0]
        patterns = (
            r"(?i)\bmy name is\s+([a-z][a-z]+(?:\s+[a-z][a-z]+){1,2})\b",
            r"(?i)\bi(?:'m| am)\s+([a-z][a-z]+(?:\s+[a-z][a-z]+){1,2})\b",
            r"(?i)\bthis is\s+([a-z][a-z]+(?:\s+[a-z][a-z]+){1,2})\b",
        )
        names: list[str] = []
        for pattern in patterns:
            match = re.search(pattern, line)
            if match:
                names.append(self._normalize_name(match.group(1)))
        return [name for name in names if name]

    def _extract_addressed_names(self, text: str) -> list[str]:
        cleaned = clean_ocr_lines([text])
        if not cleaned:
            return []
        line = cleaned[0]
        names: list[str] = []
        for match in re.finditer(
            r"(?i)\b([a-z][a-z]+(?:\s+[a-z][a-z]+){1,2})(?=[!,:])",
            line,
        ):
            normalized = self._normalize_name(match.group(1))
            if normalized:
                names.append(normalized)
        for match in re.finditer(r"(?i)\bmr\.?\s+([a-z][a-z]+(?:\s+[a-z][a-z]+)?)\b", line):
            normalized = self._normalize_name(match.group(1))
            if normalized:
                names.append(normalized)
        return names

    def _normalize_name(self, raw_name: str) -> str:
        tokens = [token for token in re.findall(r"[a-z]+", str(raw_name or "").casefold()) if token]
        if not tokens:
            return ""
        filler_leads = {"now", "hey", "wait", "sorry", "okay", "well", "so"}
        while len(tokens) > 1 and tokens[0] in filler_leads:
            tokens = tokens[1:]
        stop_tokens = {
            "a", "an", "am", "and", "apocalypse", "are", "can", "customer", "did", "do", "doing", "for", "freeze",
            "going", "gonna", "hello", "help", "hey", "his", "how", "i", "im", "i'm", "is", "it", "just", "lot",
            "money", "my", "name", "need", "please", "she", "spending", "thank", "that", "the", "their", "they",
            "this", "tired", "too", "very", "was", "we", "what", "world", "you", "your", "sorry", "okay",
        }
        if any(token in {"hello", "thank", "please", "world", "freeze", "apocalypse", "customer"} for token in tokens):
            return ""
        if any(token in stop_tokens for token in tokens):
            return ""
        if len(tokens) < 2:
            return ""
        if len(tokens) > 3:
            tokens = tokens[:3]
        return " ".join(token.capitalize() for token in tokens)

    def _first_person_score(self, text: str) -> int:
        lowered = clean_ocr_text(text).casefold()
        return len(
            re.findall(
                r"\b(i|i'm|i've|i'll|my|me|mine)\b",
                lowered,
            )
        )

    def _speaker_slug(self, speaker_name: str) -> str:
        normalized = re.sub(r"[^a-z0-9]+", "-", speaker_name.casefold()).strip("-")
        return normalized or "speaker"

    def _prepare_scene_lines(self, lines: list[str]) -> list[str]:
        prepared = clean_ocr_lines(lines)
        collapsed: list[str] = []
        for cleaned in prepared:
            if collapsed and (cleaned in collapsed[-1] or collapsed[-1] in cleaned):
                if len(cleaned) > len(collapsed[-1]):
                    collapsed[-1] = cleaned
                continue
            collapsed.append(cleaned)
        return collapsed

    def _provider_summary(self) -> dict[str, str]:
        return {
            "scene_pipeline": "scene-v5-applevision-recall",
            "text_detector": "paddleocr-opencv-hybrid" if self._has_paddleocr() else "opencv-heuristic",
            "ocr_primary": "comic-ocr-hybrid+paddle+easy+applevision" if self._has_paddleocr() else "none",
            "ocr_japanese": "manga-ocr+paddleocr" if self._has_manga_ocr() else ("paddleocr" if self._has_paddleocr() else "none"),
            "magi": self._magi_service.provider_tag(),
            "translation": "gemini-translation-with-local-fallback",
            "character_discovery": "spacy-ner-v3-metadata-excerpt",
            "speaker_attribution": "magi-hf-page-payloads-v1-character-clustering",
            "character_clustering": "clip-greedy-v1",
            "character_tracking": "clip-memory-v1",
            "llm_name_resolution": ",".join(self._llm_router.available_providers()) or "none",
        }

    def _cluster_scenes(self, scenes: list[DialogueScene], reading_mode: str) -> list[SceneCluster]:
        usable_scenes = [scene for scene in scenes if scene.has_dialogue and (scene.dialogue or scene.dialogue_original)]
        if not usable_scenes:
            return []

        clusters: list[list[DialogueScene]] = []
        current_cluster: list[DialogueScene] = []
        current_keywords: set[str] = set()
        current_character_names: set[str] = set()
        current_word_count = 0

        for scene in usable_scenes:
            scene_lines = scene.dialogue or scene.dialogue_original
            scene_keywords = self._scene_keywords(scene_lines)
            scene_character_names = {str(name).strip() for name in scene.character_names if str(name).strip()}
            scene_word_count = sum(len(line.split()) for line in scene_lines)
            should_split = False

            if current_cluster:
                last_scene = current_cluster[-1]
                page_gap = max(0, scene.page - last_scene.page)
                shared_keywords = current_keywords & scene_keywords
                shared_characters = current_character_names & scene_character_names
                transition_signal = self._scene_has_transition(scene_lines)
                current_has_transition = self._scene_has_transition(
                    current_cluster[-1].dialogue or current_cluster[-1].dialogue_original
                )
                same_logical_panel = bool(scene.logical_panel_id and scene.logical_panel_id == last_scene.logical_panel_id)
                should_split = any(
                    (
                        page_gap > 1,
                        len(current_cluster) >= 5,
                        current_word_count + scene_word_count > 140,
                        len(current_cluster) >= 2 and not shared_keywords and transition_signal,
                        len(current_cluster) >= 3 and not shared_keywords and page_gap >= 1,
                        len(current_cluster) >= 2 and current_has_transition and transition_signal,
                        len(current_cluster) >= 2 and current_character_names and scene_character_names and not shared_characters,
                    )
                )
                if same_logical_panel:
                    should_split = False

            if should_split and current_cluster:
                clusters.append(current_cluster)
                current_cluster = []
                current_keywords = set()
                current_character_names = set()
                current_word_count = 0

            current_cluster.append(scene)
            current_keywords.update(scene_keywords)
            current_character_names.update(scene_character_names)
            current_word_count += scene_word_count

        if current_cluster:
            clusters.append(current_cluster)

        return [self._build_scene_cluster(index, cluster, reading_mode) for index, cluster in enumerate(clusters, start=1)]

    def _build_scene_cluster(self, scene_number: int, cluster: list[DialogueScene], reading_mode: str) -> SceneCluster:
        dialogue = self._prepare_scene_lines(
            [line for scene in cluster for line in (scene.dialogue or [])]
        )
        dialogue_original = self._prepare_scene_lines(
            [line for scene in cluster for line in (scene.dialogue_original or [])]
        )
        keywords = self._scene_keywords(dialogue or dialogue_original)
        summary_source = dialogue or dialogue_original
        summary_hint = " ".join(summary_source[:4]).strip()[:420]
        return SceneCluster(
            scene=scene_number,
            panels=[scene.panel_order for scene in cluster],
            panel_ids=[scene.panel_id for scene in cluster],
            pages=sorted({scene.page for scene in cluster}),
            reading_mode=reading_mode,
            dialogue=dialogue,
            dialogue_original=dialogue_original,
            languages=sorted({language for scene in cluster for language in scene.languages}),
            character_names=sorted(
                {
                    str(name).strip()
                    for scene in cluster
                    for name in scene.character_names
                    if str(name).strip()
                }
            ),
            keywords=sorted(keywords),
            summary_hint=summary_hint,
            logical_panel_ids=sorted({str(scene.logical_panel_id or scene.panel_id) for scene in cluster if str(scene.logical_panel_id or scene.panel_id).strip()}),
            multi_page_panel=any(bool(scene.multi_page_panel) for scene in cluster),
        )

    def _scene_keywords(self, lines: list[str]) -> set[str]:
        stop_words = {
            "about",
            "after",
            "again",
            "before",
            "being",
            "from",
            "have",
            "into",
            "just",
            "like",
            "only",
            "over",
            "that",
            "their",
            "there",
            "they",
            "this",
            "what",
            "when",
            "where",
            "with",
            "would",
            "your",
        }
        keywords: set[str] = set()
        for line in lines:
            for token in re.findall(r"[A-Za-z]{4,}", line.lower()):
                if token in stop_words:
                    continue
                keywords.add(token)
        return keywords

    def _scene_has_transition(self, lines: list[str]) -> bool:
        transition_markers = (
            "later",
            "meanwhile",
            "suddenly",
            "before",
            "after",
            "wake up",
            "wakes up",
            "regress",
            "second chance",
            "back in time",
            "at that moment",
            "the next day",
        )
        lowered = " ".join(lines).casefold()
        return any(marker in lowered for marker in transition_markers)

    def _reading_mode(self, metadata: ChapterMetadata, page_paths: list[Path]) -> str:
        tag_payload = metadata.raw.get("relationships", []) if isinstance(metadata.raw, dict) else []
        tag_names: set[str] = set()
        for relationship in tag_payload:
            attributes = relationship.get("attributes", {})
            name = attributes.get("name", {})
            if isinstance(name, dict):
                tag_names.update(value.lower() for value in name.values() if isinstance(value, str))

        if {"long strip", "web comic", "webtoon"} & tag_names:
            return "webtoon"

        if page_paths:
            width, height = Image.open(page_paths[0]).size
            if height / max(width, 1) > 1.35:
                return "webtoon"

        return "manga"

    def _infer_project_language(self, page_paths: list[Path], metadata_language: str | None) -> str:
        hint = self._normalise_language_code(metadata_language)
        if metadata_language and hint not in {"", "a"}:
            return hint
        if not page_paths:
            return hint

        detected: Counter[str] = Counter()
        for page_path in page_paths[:3]:
            try:
                page_image = np.array(Image.open(page_path).convert("RGB"))
            except Exception:
                continue
            for sample in self._language_sample_crops(page_image):
                text, _, _ = self._comic_ocr.recognize_panel_text(sample, hint)
                cleaned = self._clean_text(text)
                if not cleaned or self._looks_like_noise(cleaned):
                    continue
                detected[self._detect_language(cleaned, hint)] += 1
                if sum(detected.values()) >= 6:
                    break
            if sum(detected.values()) >= 6:
                break

        if not detected:
            return hint
        candidate, _ = detected.most_common(1)[0]
        if candidate in {"en", "a"} and hint not in {"en", "a"}:
            return hint
        return candidate or hint

    def _language_sample_crops(self, page_image: np.ndarray) -> list[np.ndarray]:
        height, width = page_image.shape[:2]
        if height <= 0 or width <= 0:
            return []

        samples: list[np.ndarray] = []
        sample_height = min(max(int(width * 1.15), 420), height)
        start_positions = [0]
        if height > sample_height:
            start_positions.append(max((height - sample_height) // 2, 0))
            start_positions.append(max(height - sample_height, 0))

        for start_y in dict.fromkeys(start_positions):
            end_y = min(start_y + sample_height, height)
            crop = page_image[start_y:end_y, :]
            if crop.size:
                samples.append(self._resize_language_sample(crop))
        return samples

    def _resize_language_sample(self, crop: np.ndarray) -> np.ndarray:
        """Keep language-probe OCR bounded on large comic pages."""
        height, width = crop.shape[:2]
        longest_side = max(height, width)
        if longest_side <= 1400:
            return crop
        scale = 1400 / max(longest_side, 1)
        resized = Image.fromarray(crop).resize(
            (max(1, int(width * scale)), max(1, int(height * scale))),
            Image.Resampling.LANCZOS,
        )
        return np.array(resized)

    def _panel_crop(self, image: np.ndarray, panel: PanelBox) -> tuple[np.ndarray, tuple[int, int, int, int]]:
        height, width = image.shape[:2]
        pad_x = max(18, int(panel.width * self.settings.panel_crop_margin_x_ratio))
        pad_top = max(24, int(panel.height * self.settings.panel_crop_margin_top_ratio))
        pad_bottom = max(18, int(panel.height * self.settings.panel_crop_margin_bottom_ratio))
        x1 = max(int(panel.x - pad_x), 0)
        y1 = max(int(panel.y - pad_top), 0)
        x2 = min(int(panel.x + panel.width + pad_x), width)
        y2 = min(int(panel.y + panel.height + pad_bottom), height)
        if x2 <= x1 or y2 <= y1:
            return image, (0, 0, width, height)
        return image[y1:y2, x1:x2], (x1, y1, x2 - x1, y2 - y1)

    def _associate_candidates_to_panel(
        self,
        candidates: list[OCRCandidate],
        crop_bbox: tuple[int, int, int, int],
        panel: PanelBox,
        image_width: int,
        image_height: int,
        reading_mode: str,
    ) -> list[OCRCandidate]:
        if not candidates:
            return []

        crop_x, crop_y, _, _ = crop_bbox
        panel_box = (int(panel.x), int(panel.y), int(panel.width), int(panel.height))
        association_box = self._expand_panel_association_box(panel_box, image_width, image_height)
        associated: list[OCRCandidate] = []

        for candidate in candidates:
            local_bbox = tuple(int(value) for value in candidate.bbox)
            page_bbox = (
                int(crop_x + local_bbox[0]),
                int(crop_y + local_bbox[1]),
                int(local_bbox[2]),
                int(local_bbox[3]),
            )
            if not self._candidate_belongs_to_panel(page_bbox, panel_box, association_box):
                continue
            associated.append(candidate)

        if associated:
            return self._sort_candidates(associated, reading_mode)
        return self._sort_candidates(candidates, reading_mode)

    def _expand_panel_association_box(
        self,
        panel_box: tuple[int, int, int, int],
        image_width: int,
        image_height: int,
    ) -> tuple[int, int, int, int]:
        x, y, width, height = panel_box
        pad_x = max(24, int(width * self.settings.panel_assoc_margin_x_ratio))
        pad_top = max(28, int(height * self.settings.panel_assoc_margin_top_ratio))
        pad_bottom = max(20, int(height * self.settings.panel_assoc_margin_bottom_ratio))
        x1 = max(x - pad_x, 0)
        y1 = max(y - pad_top, 0)
        x2 = min(x + width + pad_x, image_width)
        y2 = min(y + height + pad_bottom, image_height)
        return (x1, y1, x2 - x1, y2 - y1)

    def _candidate_belongs_to_panel(
        self,
        candidate_box: tuple[int, int, int, int],
        panel_box: tuple[int, int, int, int],
        association_box: tuple[int, int, int, int],
    ) -> bool:
        if self._intersection_area(candidate_box, panel_box) > 0:
            return True
        if not self._bbox_contains(association_box, candidate_box) and self._intersection_area(candidate_box, association_box) <= 0:
            return False

        panel_center_x = panel_box[0] + panel_box[2] / 2
        panel_center_y = panel_box[1] + panel_box[3] / 2
        candidate_center_x = candidate_box[0] + candidate_box[2] / 2
        candidate_center_y = candidate_box[1] + candidate_box[3] / 2
        horizontal_distance = abs(candidate_center_x - panel_center_x)
        vertical_distance = abs(candidate_center_y - panel_center_y)

        max_horizontal = panel_box[2] * 0.75 + candidate_box[2] * 0.5
        max_vertical = panel_box[3] * 0.7 + candidate_box[3] * 0.9
        if horizontal_distance <= max_horizontal and vertical_distance <= max_vertical:
            return True

        return self._intersection_area(candidate_box, association_box) >= max(80, candidate_box[2] * candidate_box[3] * 0.45)

    def _detect_text_regions(self, panel_image: np.ndarray, language_hint: str) -> tuple[list[tuple[int, int, int, int]], str]:
        boxes: list[tuple[int, int, int, int]] = []
        if self._has_paddleocr():
            try:
                boxes = self._paddle_detect(panel_image, language_hint)
            except Exception:
                boxes = []

        if boxes:
            return self._dedupe_boxes(boxes, panel_image.shape[1], panel_image.shape[0]), "paddleocr"

        return self._detect_text_regions_opencv(panel_image), "opencv"

    def _detect_text_regions_opencv(self, panel_image: np.ndarray) -> list[tuple[int, int, int, int]]:
        import cv2

        grayscale = cv2.cvtColor(panel_image, cv2.COLOR_RGB2GRAY)
        blurred = cv2.GaussianBlur(grayscale, (3, 3), 0)
        binary = cv2.adaptiveThreshold(
            blurred,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            31,
            11,
        )
        dilation_width = max(9, panel_image.shape[1] // 28)
        dilation_height = max(3, panel_image.shape[0] // 120)
        kernel = np.ones((dilation_height, dilation_width), np.uint8)
        merged = cv2.dilate(binary, kernel, iterations=1)
        contours, _ = cv2.findContours(merged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        boxes: list[tuple[int, int, int, int]] = []
        crop_area = panel_image.shape[0] * panel_image.shape[1]
        for contour in contours:
            x, y, width, height = cv2.boundingRect(contour)
            area = width * height
            if area < max(180, crop_area * 0.0035):
                continue
            if width < max(40, panel_image.shape[1] * 0.09):
                continue
            if height < max(18, panel_image.shape[0] * 0.025):
                continue
            if height > panel_image.shape[0] * 0.55:
                continue
            pad_x = max(8, width // 10)
            pad_y = max(6, height // 6)
            x1 = max(x - pad_x, 0)
            y1 = max(y - pad_y, 0)
            x2 = min(x + width + pad_x, panel_image.shape[1])
            y2 = min(y + height + pad_y, panel_image.shape[0])
            boxes.append((x1, y1, x2 - x1, y2 - y1))

        return self._dedupe_boxes(boxes, panel_image.shape[1], panel_image.shape[0])

    def _ocr_region(self, region_image: np.ndarray, language_hint: str) -> tuple[str, float | None, str]:
        result = self._comic_ocr.recognize_region(region_image, language_hint)
        return self._clean_text(result.original_text), result.confidence, result.ocr_engine

    def _backfill_from_page_ocr(
        self,
        panel: Any,
        crop_bbox: tuple[int, int, int, int],
        image_width: int,
        image_height: int,
        project_language: str,
        metrics: dict[str, Any],
    ) -> list:
        """Fill panels that have no OCR text using page-level OCR boxes from reconstruction."""
        page_key = str(panel.page)
        boxes = self._page_text_boxes.get(page_key, [])
        if not boxes:
            return []

        panel_box = (int(panel.x), int(panel.y), int(panel.width), int(panel.height))
        association_box = self._expand_panel_association_box(panel_box, image_width, image_height)

        matched_texts: list[tuple[int, int, str, float]] = []
        for box in boxes:
            text = str(box.get("text") or "").strip()
            if not text or len(text) < 2:
                continue
            bx = int(box.get("x", 0))
            by = int(box.get("y", 0))
            bw = int(box.get("width", 0))
            bh = int(box.get("height", 0))
            confidence = float(box.get("confidence", 0.0))
            candidate_box = (bx, by, bw, bh)
            if self._candidate_belongs_to_panel(candidate_box, panel_box, association_box):
                cleaned = self._clean_text(text)
                if cleaned and not self._looks_like_noise(cleaned):
                    matched_texts.append((by, bx, cleaned, confidence))

        if not matched_texts:
            return []

        matched_texts.sort(key=lambda item: (item[0], item[1]))
        combined_text = " ".join(text for _, _, text, _ in matched_texts)
        avg_confidence = (
            sum(confidence for _, _, _, confidence in matched_texts) / len(matched_texts)
            if matched_texts
            else 0.5
        )
        detected_language = self._panel_text_language(combined_text, project_language)
        text_english = self._translate_to_english(combined_text, detected_language)
        if text_english and text_english.strip() != combined_text.strip():
            metrics["translation_count"] += 1

        return [
            DialogueRegion(
                page=panel.page,
                panel=panel.panel,
                panel_order=panel.order,
                bbox=[int(panel_box[0]), int(panel_box[1]), int(panel_box[2]), int(panel_box[3])],
                language=detected_language,
                text_original=combined_text,
                text_english=text_english or combined_text,
                confidence=avg_confidence,
                detector="page-ocr-backfill",
                ocr_engine="paddleocr-page",
            )
        ]

    def _ocr_full_panel(self, panel_image: np.ndarray, language_hint: str) -> tuple[str, float | None, str]:
        fragments: list[str] = []
        confidences: list[float] = []
        engines: list[str] = []
        for slice_image in self._panel_slices(panel_image):
            text, confidence, engine = self._comic_ocr.recognize_panel_text(slice_image, language_hint)
            cleaned = self._clean_text(text)
            if not cleaned or self._looks_like_noise(cleaned):
                continue
            if fragments and cleaned.casefold() == fragments[-1].casefold():
                continue
            fragments.append(cleaned)
            if isinstance(confidence, (int, float)):
                confidences.append(float(confidence))
            if engine:
                engines.append(engine)
        text = self._clean_text(" ".join(fragments))
        confidence = sum(confidences) / len(confidences) if confidences else None
        return text, confidence, (engines[0] if engines else "none")

    def _panel_slices(self, panel_image: np.ndarray) -> list[np.ndarray]:
        height = panel_image.shape[0]
        width = panel_image.shape[1]
        if height <= width * 1.4:
            return [panel_image]

        slice_height = min(max(width * 2, 960), height)
        overlap = int(slice_height * 0.18)
        slices: list[np.ndarray] = []
        start = 0
        while start < height:
            end = min(start + slice_height, height)
            slices.append(panel_image[start:end, :])
            if end >= height:
                break
            start = max(end - overlap, start + 1)
        return slices

    def _detect_language(self, text: str, language_hint: str) -> str:
        return self._language_detector.detect(text, language_hint)

    def _panel_text_language(self, text: str, language_hint: str) -> str:
        hint = self._normalise_language_code(language_hint)
        if hint in {"en", "a"} and re.search(r"[A-Za-z]", text or "") and not re.search(
            r"[\u3040-\u30ff\u31f0-\u31ff\u3400-\u9fff\uac00-\ud7af]",
            text or "",
        ):
            return "en"
        return self._detect_language(text, language_hint)

    def _heuristic_language(self, text: str, language_hint: str) -> str:
        normalised_hint = self._normalise_language_code(language_hint)
        ranges = {
            "ja": r"[\u3040-\u30ff\u31f0-\u31ff]",
            "ko": r"[\uac00-\ud7af]",
            "zh": r"[\u4e00-\u9fff]",
        }
        for code, pattern in ranges.items():
            if re.search(pattern, text):
                if code == "zh" and re.search(r"[\u3040-\u30ff]", text):
                    return "ja"
                return code
        if re.search(r"[A-Za-z]", text):
            if normalised_hint in {"pt", "es", "fr", "it", "ro", "ca", "gl"}:
                return normalised_hint
            return "en"
        return normalised_hint

    def _translate_to_english(self, text: str, language_code: str) -> str:
        if self._normalise_language_code(language_code) in {"en", "a"}:
            return text
        return self._translator.translate(text, language_code)

    def _get_translator(self, model_name: str) -> tuple[Any, Any]:
        cached = self._translator_cache.get(model_name)
        if cached is not None:
            return cached

        from transformers import MarianMTModel, MarianTokenizer

        tokenizer = MarianTokenizer.from_pretrained(model_name)
        model = MarianMTModel.from_pretrained(model_name)
        try:
            model.eval()
        except Exception:
            pass
        self._translator_cache[model_name] = (tokenizer, model)
        return tokenizer, model

    def _has_paddleocr(self) -> bool:
        try:
            import paddleocr  # noqa: F401

            return True
        except Exception:
            return False

    def _has_manga_ocr(self) -> bool:
        try:
            import manga_ocr  # noqa: F401

            return True
        except Exception:
            return False

    def _get_paddle_ocr(self, language_code: str):
        provider_language = self._paddle_language(language_code)
        if provider_language in self._paddle_ocr:
            return self._paddle_ocr[provider_language]

        from paddleocr import PaddleOCR
        recognition_model = {
            "en": "en_PP-OCRv5_mobile_rec",
            "korean": "korean_PP-OCRv5_mobile_rec",
            "japan": "japan_PP-OCRv3_mobile_rec",
            "ch": "PP-OCRv5_mobile_rec",
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
                text_recognition_batch_size=1,
                text_det_limit_side_len=960,
                text_det_limit_type="max",
                text_rec_score_thresh=0.45,
            )
        except (TypeError, ValueError):
            instance = PaddleOCR(lang=provider_language)
        self._paddle_ocr[provider_language] = instance
        return instance

    def _paddle_detect(self, panel_image: np.ndarray, language_code: str) -> list[tuple[int, int, int, int]]:
        ocr = self._get_paddle_ocr(language_code)
        results = ocr.ocr(panel_image)
        boxes, _, _ = self._extract_paddle_results(results)
        return boxes

    def _paddle_recognise(self, region_image: np.ndarray, language_code: str) -> tuple[str, float | None]:
        ocr = self._get_paddle_ocr(language_code)
        results = ocr.ocr(region_image)
        _, fragments, scores = self._extract_paddle_results(results)
        confidences = [float(score) for score in scores if isinstance(score, (int, float))]
        text = " ".join(fragment.strip() for fragment in fragments if fragment and fragment.strip())
        confidence = sum(confidences) / len(confidences) if confidences else None
        return text, confidence

    def _extract_paddle_results(self, results: Any) -> tuple[list[tuple[int, int, int, int]], list[str], list[float | None]]:
        boxes: list[tuple[int, int, int, int]] = []
        texts: list[str] = []
        scores: list[float | None] = []

        def add_poly(points: Any) -> None:
            if points is None:
                return
            xs = [int(point[0]) for point in points]
            ys = [int(point[1]) for point in points]
            x1, y1 = min(xs), min(ys)
            x2, y2 = max(xs), max(ys)
            boxes.append((x1, y1, x2 - x1, y2 - y1))

        def walk(node: Any) -> None:
            if isinstance(node, dict):
                for poly in node.get("dt_polys", []) or []:
                    add_poly(poly)
                rec_texts = node.get("rec_texts", []) or []
                rec_scores = node.get("rec_scores", []) or []
                for index, text in enumerate(rec_texts):
                    cleaned = self._clean_text(str(text))
                    if not cleaned or self._looks_like_noise(cleaned):
                        continue
                    texts.append(cleaned)
                    scores.append(float(rec_scores[index]) if index < len(rec_scores) and isinstance(rec_scores[index], (int, float)) else None)
                return

            if isinstance(node, (list, tuple)):
                if len(node) == 2 and isinstance(node[0], str):
                    cleaned = self._clean_text(node[0])
                    if cleaned and not self._looks_like_noise(cleaned):
                        texts.append(cleaned)
                        scores.append(float(node[1]) if isinstance(node[1], (int, float)) else None)
                    return
                if len(node) == 2 and isinstance(node[1], (list, tuple)) and len(node[1]) == 2 and isinstance(node[1][0], str):
                    add_poly(node[0])
                    cleaned = self._clean_text(node[1][0])
                    if cleaned and not self._looks_like_noise(cleaned):
                        texts.append(cleaned)
                        scores.append(float(node[1][1]) if isinstance(node[1][1], (int, float)) else None)
                    return
                for item in node:
                    walk(item)

        walk(results)
        return boxes, texts, scores

    def _manga_ocr_read(self, region_image: np.ndarray) -> str:
        if self._manga_ocr is None:
            from manga_ocr import MangaOcr

            self._manga_ocr = MangaOcr()
            self.__class__._GLOBAL_MANGA_OCR = self._manga_ocr

        pil_image = Image.fromarray(region_image)
        return str(self._manga_ocr(pil_image))

    def _paddle_language(self, language_code: str) -> str:
        return {
            "en": "en",
            "ja": "japan",
            "ko": "korean",
            "zh": "ch",
        }.get(self._normalise_language_code(language_code), "en")

    def _normalise_language_code(self, language_code: str | None) -> str:
        return self._language_detector.normalize_language_code(language_code)

    def _should_prefer_language_hint(self, text: str, detected_language: str, language_hint: str) -> bool:
        if not language_hint or language_hint in {"en", "a"} or detected_language == language_hint:
            return False
        if language_hint not in {"pt", "es", "fr", "it", "ro", "ca", "gl"}:
            return False
        if not re.search(r"[A-Za-z]", text):
            return False
        tokens = re.findall(r"[a-z]{2,}", clean_ocr_text(text).casefold())
        if not tokens:
            return False
        suspicious_detections = {"en", "cs", "so", "sk", "sl", "hr", "da", "nl", "sv", "tl", "fi", "id"}
        if detected_language in suspicious_detections:
            return True
        if detected_language not in {"pt", "es", "fr", "it", "ro", "ca", "gl", "ja", "ko", "zh"} and len(tokens) <= 14:
            return True
        return False

    def _sort_regions(self, boxes: list[tuple[int, int, int, int]], reading_mode: str) -> list[tuple[int, int, int, int]]:
        if reading_mode == "webtoon":
            return sorted(boxes, key=lambda item: (item[1], item[0]))
        return sorted(boxes, key=lambda item: (item[1], -item[0]))

    def _dedupe_boxes(
        self,
        boxes: list[tuple[int, int, int, int]],
        image_width: int,
        image_height: int,
    ) -> list[tuple[int, int, int, int]]:
        cleaned: list[tuple[int, int, int, int]] = []
        crop_area = image_width * image_height
        for box in sorted(boxes, key=lambda item: (item[1], item[0])):
            x, y, width, height = [int(value) for value in box]
            if width < 20 or height < 20:
                continue
            if width * height > crop_area * 0.65:
                continue
            duplicate = False
            for other in cleaned:
                if self._iou(box, other) >= 0.7:
                    duplicate = True
                    break
            if not duplicate:
                cleaned.append((max(x, 0), max(y, 0), width, height))
        return cleaned

    def _iou(self, box_a: tuple[int, int, int, int], box_b: tuple[int, int, int, int]) -> float:
        intersection = self._intersection_area(box_a, box_b)
        if intersection <= 0:
            return 0.0
        _, _, aw, ah = box_a
        _, _, bw, bh = box_b
        union = aw * ah + bw * bh - intersection
        return intersection / max(union, 1)

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

    def _crop_box(self, image: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
        x, y, width, height = box
        return image[y : y + height, x : x + width]

    def _ocr_variants(self, image: np.ndarray, language_hint: str) -> list[np.ndarray]:
        variants = [image]
        enhanced = self._preprocess_for_ocr(image)
        if enhanced is not None:
            variants.append(enhanced)

        height, width = image.shape[:2]
        if height > width * 1.2:
            scaled = self._scale_for_ocr(image)
            if scaled is not None:
                variants.append(scaled)

        if language_hint in {"ja", "zh"} and height > width * 1.15:
            rotated = np.rot90(image, k=1).copy()
            variants.append(rotated)
            rotated_enhanced = self._preprocess_for_ocr(rotated)
            if rotated_enhanced is not None:
                variants.append(rotated_enhanced)

        unique: list[np.ndarray] = []
        seen_shapes: set[tuple[int, int, int, int]] = set()
        for candidate in variants:
            signature = (
                candidate.shape[0],
                candidate.shape[1],
                candidate.shape[2] if candidate.ndim == 3 else 1,
                int(np.mean(candidate)),
            )
            if signature in seen_shapes:
                continue
            seen_shapes.add(signature)
            unique.append(candidate)
        return unique

    def _preprocess_for_ocr(self, image: np.ndarray) -> np.ndarray | None:
        try:
            import cv2

            grayscale = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            grayscale = cv2.fastNlMeansDenoising(grayscale, None, 10, 7, 21)
            upscaled = cv2.resize(grayscale, None, fx=1.6, fy=1.6, interpolation=cv2.INTER_CUBIC)
            thresholded = cv2.adaptiveThreshold(
                upscaled,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                31,
                9,
            )
            return cv2.cvtColor(thresholded, cv2.COLOR_GRAY2RGB)
        except Exception:
            return None

    def _scale_for_ocr(self, image: np.ndarray) -> np.ndarray | None:
        try:
            import cv2

            return cv2.resize(image, None, fx=1.4, fy=1.4, interpolation=cv2.INTER_CUBIC)
        except Exception:
            return None

    def _is_better_candidate(
        self,
        candidate_text: str,
        candidate_confidence: float | None,
        best_text: str,
        best_confidence: float | None,
    ) -> bool:
        if not candidate_text or self._looks_like_noise(candidate_text):
            return False
        if not best_text:
            return True
        if candidate_confidence is not None and best_confidence is not None:
            if candidate_confidence > best_confidence + 0.04:
                return True
            if best_confidence > candidate_confidence + 0.04:
                return False
        return len(candidate_text) > len(best_text)

    def _clean_text(self, text: str) -> str:
        return self._dialogue_cleaner.clean_text(text).lower()

    def _looks_like_noise(self, text: str) -> bool:
        return not self._dialogue_cleaner.is_usable(text)

    def _should_track_panel_regions(self, regions: list[DialogueRegion]) -> bool:
        usable_lines: list[str] = []
        for region in regions:
            cleaned = self._dialogue_cleaner.clean_text(region.text_english or region.text_original or "")
            if not self._dialogue_cleaner.is_usable(cleaned):
                continue
            usable_lines.append(cleaned)
        if not usable_lines:
            return False
        combined = " ".join(usable_lines)
        tokens = re.findall(
            r"[A-Za-z0-9\u00C0-\u024F\u1E00-\u1EFF\u3040-\u30ff\u4e00-\u9fff\uac00-\ud7af']+",
            combined,
        )
        if len(tokens) >= 5:
            return True
        if any("?" in line for line in usable_lines):
            return True
        if len(usable_lines) >= 2 and len(tokens) >= 3:
            return True
        return False
