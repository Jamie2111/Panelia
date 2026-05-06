from __future__ import annotations

import json
import logging
import re
import shutil
import time
from pathlib import Path
from typing import Any
from PIL import Image

from app.pipeline.context import PipelineContext
from app.pipeline.auto_run import continue_auto_run_pipeline
from app.pipeline.orchestration import queue_stage_once
from app.schemas.project import (
    CanonicalCharacterRecord,
    MusicConfig,
    NarrationMode,
    PanelVisionRecord,
    PipelineStage,
    StageStatus,
    VideoConfig,
    VoiceConfig,
)
from app.services.character_portrait_pass import CharacterPortraitPass, _PLACEHOLDER_NAME_PATTERN as _CANONICAL_PLACEHOLDER_PATTERN
from app.services.dialogue_pipeline import DialogueExtractionPipeline
from app.services.cross_page_panel_merger import CrossPagePanelMerger
from app.services.character_review_service import CharacterReviewService
from app.services.character_name_filters import looks_like_false_character_name
from app.services.character_name_service import CharacterNameService
from app.services.character_visual_profiler import CharacterVisualProfiler
from app.services.generate_narration import generate_narration
from app.services.ingestion import PageIngestionService
from app.services.panel_reconstruction_engine import PanelReconstructionEngine
from app.services.panel_detection_service import MagiPanelDetectionService
from app.services.panel_evidence_extractor import PanelEvidenceExtractor, load_panel_evidence_records
from app.services.panel_vision_extractor import PanelVisionExtractor, story_bible_canonical_fallback
from app.services.panel_vision_quality_service import PanelVisionQualityService
from app.services.ocr_cleaner import clean_ocr_lines, clean_ocr_text, combined_dialogue_entry_lines, is_usable_ocr_text
from app.services.llm_router import LLMRouter
from app.services.project_store import ProjectStore
from app.services.story_segment_repair_service import StorySegmentRepairService
from app.services.story_script_service import StoryScriptService
from app.services.style_vocabulary import StyleVocabulary, build_style_vocabulary
from app.services.video_service import VideoRenderService
from app.utils.files import read_json, write_json

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Non-name tokens that should never appear as character names in the dict.
# These are mostly Portuguese/Spanish filler words that OCR mis-tags as speakers.
# ---------------------------------------------------------------------------
_FALSE_CHARACTER_NAME_TOKENS: frozenset[str] = frozenset({
    "por favor", "favor", "claro", "de novo", "ola", "olá", "sim", "nao",
    "não", "gracias", "obrigado", "obrigada", "please", "okay", "sure",
    "yes", "no", "hello", "hey", "stop", "wait", "now", "agora",
    "be dead", "jle trle", "nati a", "salur", "sauri",
})


def _filter_character_dictionary(char_dict: dict) -> dict:
    """Remove OCR-artifact / non-name entries from the character dictionary."""
    filtered: dict = {}
    for key, value in char_dict.items():
        display_value = value.get("display_name") if isinstance(value, dict) else value
        if looks_like_false_character_name(key) or looks_like_false_character_name(display_value):
            continue
        key_norm = " ".join(re.findall(r"[a-záàâãéêíóôõúçñ]+", str(key or "").casefold())).strip()
        val_norm = " ".join(re.findall(r"[a-záàâãéêíóôõúçñ]+", str(display_value or "").casefold())).strip()
        if key_norm in _FALSE_CHARACTER_NAME_TOKENS or val_norm in _FALSE_CHARACTER_NAME_TOKENS:
            continue
        filtered[key] = value
    return filtered


def _coerce_positive_int(value: Any) -> int | None:
    try:
        coerced = int(value)
    except (TypeError, ValueError):
        return None
    return coerced if coerced > 0 else None


def _box_intersection_area(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> int:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x1 = max(ax, bx)
    y1 = max(ay, by)
    x2 = min(ax + aw, bx + bw)
    y2 = min(ay + ah, by + bh)
    if x2 <= x1 or y2 <= y1:
        return 0
    return (x2 - x1) * (y2 - y1)


def _expanded_panel_box(panel: Any) -> tuple[int, int, int, int]:
    width = max(int(getattr(panel, "width", 0) or 0), 1)
    height = max(int(getattr(panel, "height", 0) or 0), 1)
    pad_x = max(12, int(width * 0.08))
    pad_y = max(12, int(height * 0.08))
    return (
        int(getattr(panel, "x", 0) or 0) - pad_x,
        int(getattr(panel, "y", 0) or 0) - pad_y,
        width + pad_x * 2,
        height + pad_y * 2,
    )


def _page_ocr_text_for_panel(panel: Any, page_text_boxes: dict[str, list] | None) -> str:
    """Return page-level OCR text whose boxes land inside this panel.

    This lets script generation recover auto-skipped panels when the earlier
    panel-splitting stage was too aggressive, without running extra OCR.
    """
    if not page_text_boxes:
        return ""
    boxes = page_text_boxes.get(str(getattr(panel, "page", "")) or "")
    if not boxes:
        return ""

    panel_box = (
        int(getattr(panel, "x", 0) or 0),
        int(getattr(panel, "y", 0) or 0),
        max(int(getattr(panel, "width", 0) or 0), 1),
        max(int(getattr(panel, "height", 0) or 0), 1),
    )
    association_box = _expanded_panel_box(panel)
    matched: list[str] = []
    for box in boxes:
        if not isinstance(box, dict):
            continue
        text = clean_ocr_text(str(box.get("text") or "")).strip()
        if not text or len(text) < 2:
            continue
        candidate_box = (
            int(box.get("x", 0) or 0),
            int(box.get("y", 0) or 0),
            max(int(box.get("width", 0) or 0), 1),
            max(int(box.get("height", 0) or 0), 1),
        )
        candidate_area = max(candidate_box[2] * candidate_box[3], 1)
        direct_overlap = _box_intersection_area(panel_box, candidate_box) / candidate_area
        expanded_overlap = _box_intersection_area(association_box, candidate_box) / candidate_area
        if direct_overlap >= 0.42 or expanded_overlap >= 0.70:
            matched.append(text)

    return " ".join(clean_ocr_lines(matched)).strip()


_NON_RECOVERABLE_AUTO_SKIP_REASONS: tuple[str, ...] = (
    "speech-bubble strip below larger panel",
    "low-content white-gap panel",
    "low-content boundary fragment",
    "low-content top gap fragment",
    "top continuation fragment",
    "duplicate of earlier panel",
    "continuation of cross-page panel",
    "split page-boundary panel",
    "likely credit/watermark page",
)


def _panel_looks_like_text_strip(panel: Any, page_size: tuple[int, int] | None) -> bool:
    if not page_size:
        return False
    page_w, page_h = page_size
    width = max(int(getattr(panel, "width", 0) or 0), 1)
    height = max(int(getattr(panel, "height", 0) or 0), 1)
    width_ratio = width / max(page_w, 1)
    height_ratio = height / max(page_h, 1)
    area_ratio = (width * height) / max(page_w * page_h, 1)
    return bool(
        (width_ratio >= 0.72 and height_ratio <= 0.16)
        or (width_ratio >= 0.84 and height_ratio <= 0.22)
        or (width_ratio >= 0.88 and height_ratio <= 0.28 and area_ratio <= 0.18)
    )


def _should_recover_auto_skipped_panel_with_text(
    panel: Any,
    page_size: tuple[int, int] | None,
) -> bool:
    reason = str(getattr(panel, "skip_reason", "") or "").casefold().strip()
    if not reason:
        return True
    if any(blocked_reason in reason for blocked_reason in _NON_RECOVERABLE_AUTO_SKIP_REASONS):
        return False
    if (
        "overlapping strip from same-page panel split" in reason
        and _panel_looks_like_text_strip(panel, page_size)
    ):
        return False
    return True


_LOCATION_SUFFIX_PATTERN = re.compile(
    r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+"
    r"(City|Town|Village|Mountain|Mount|River|Lake|Island|Temple|Palace|Province|District|Station|Tower|Hall|Gate|Road|Street|Bridge|Park|Forest|Bay|Port|Harbor|Harbour|Castle|Fortress|Kingdom|Empire|Academy|School|University|Sect|Clan|Guild)\b"
)
_TITLE_CASE_PATTERN = re.compile(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,2})\b")
_PRONUNCIATION_NAME_PREFIX_BLACKLIST = {
    "A", "An", "And", "As", "At", "But", "By", "For", "From", "In", "Into", "Now", "Of", "On", "Or", "Por", "Right", "The", "Then", "Without",
}
_PRONUNCIATION_NAME_EXACT_BLACKLIST = {
    "Por Favor",
    "Global Freeze",
    "One Man",
    "The World",
    "The Apocalypse",
    "The Global Freeze",
    "A Voice",
}


def _clean_pronunciation_candidate(raw_name: str) -> str | None:
    tokens = re.findall(r"[A-Za-z]+", str(raw_name or ""))
    if not tokens:
        return None
    while len(tokens) > 1 and tokens[0] in _PRONUNCIATION_NAME_PREFIX_BLACKLIST:
        tokens = tokens[1:]
    if not tokens:
        return None
    candidate = " ".join(tokens[:3]).strip()
    if not candidate or candidate in _PRONUNCIATION_NAME_EXACT_BLACKLIST:
        return None
    if len(tokens) == 1 and len(tokens[0]) < 3:
        return None
    return candidate


def _collect_pronunciation_names(project_dir: Path, narrated_texts: list[str]) -> list[str]:
    name_service = CharacterNameService()
    names: list[str] = []

    char_dict_path = project_dir / "output" / "character_dictionary.json"
    if char_dict_path.exists():
        try:
            char_dict = json.loads(char_dict_path.read_text())
            for value in char_dict.values():
                cleaned = _clean_pronunciation_candidate(str(value))
                if cleaned:
                    names.append(cleaned)
        except Exception:
            logger.exception("Failed to read character_dictionary.json for pronunciation names")

    gemini_cache_path = project_dir / "output" / "gemini_summary_cache.json"
    if gemini_cache_path.exists():
        try:
            gemini_cache = json.loads(gemini_cache_path.read_text())
            for entry in gemini_cache.get("panel_script_map", []):
                for candidate in entry.get("character_names") or []:
                    cleaned = _clean_pronunciation_candidate(str(candidate))
                    if cleaned:
                        names.append(cleaned)
                narration = str(entry.get("narration") or "").strip()
                if narration:
                    for candidate in name_service.extract_names(narration):
                        cleaned = _clean_pronunciation_candidate(candidate)
                        if cleaned:
                            names.append(cleaned)
                    for match in _TITLE_CASE_PATTERN.findall(narration):
                        cleaned = _clean_pronunciation_candidate(match)
                        if cleaned:
                            names.append(cleaned)
                    for prefix, _suffix in _LOCATION_SUFFIX_PATTERN.findall(narration):
                        cleaned = _clean_pronunciation_candidate(prefix)
                        if cleaned:
                            names.append(cleaned)
        except Exception:
            logger.exception("Failed to read gemini_summary_cache.json for pronunciation names")

    for text in narrated_texts:
        for candidate in name_service.extract_names(text):
            cleaned = _clean_pronunciation_candidate(candidate)
            if cleaned:
                names.append(cleaned)
        for match in _TITLE_CASE_PATTERN.findall(text):
            cleaned = _clean_pronunciation_candidate(match)
            if cleaned:
                names.append(cleaned)
        for prefix, _suffix in _LOCATION_SUFFIX_PATTERN.findall(text):
            cleaned = _clean_pronunciation_candidate(prefix)
            if cleaned:
                names.append(cleaned)

    unique_names: list[str] = []
    seen: set[str] = set()
    for name in names:
        key = name.casefold()
        if key in seen:
            continue
        seen.add(key)
        unique_names.append(name)
    return unique_names


def run_ingestion(context: PipelineContext) -> None:
    store = context.store
    project = store.get_project(context.project_id)
    job = store.get_job(context.project_id, context.job_id)
    direct_runner = bool(job.payload.get("direct_runner"))
    ingestion = PageIngestionService(store)
    context.start("Preparing pages")
    metadata = ingestion.ingest(
        context.project_id,
        project.source_type,
        project.source_reference,
        progress_callback=context.progress,
        cancel_callback=context.ensure_not_cancelled,
    )
    store.update_project_metadata(context.project_id, chapter_metadata=metadata.model_dump(mode="json"))
    store.update_stage_state(context.project_id, PipelineStage.PANEL_DETECTION, StageStatus.READY, progress=0, message="Starting panel detection automatically")
    context.complete("Pages imported successfully")
    if direct_runner:
        return
    queue_stage_once(
        store,
        context.queue,
        context.project_id,
        PipelineStage.PANEL_DETECTION,
        "Queued automatically after page ingestion",
    )


def run_panel_detection(context: PipelineContext) -> None:
    store = context.store
    project = store.get_project(context.project_id)
    page_paths = store.list_page_paths(context.project_id)
    if not page_paths:
        raise ValueError("No imported pages were found. Re-import the source files or MangaDex chapter and try again.")
    detector = MagiPanelDetectionService()
    reconstruction = PanelReconstructionEngine()
    cross_page_merger = CrossPagePanelMerger()
    context.start("Running hybrid panel detection")
    context.progress(4, "Preparing panel detector")
    detected_panels = detector.detect_panels(
        page_paths,
        progress_callback=lambda progress, message: context.progress(8 + progress * 0.6, message),
        cancel_callback=context.ensure_not_cancelled,
    )
    detector_text_boxes_by_page: dict[int, list[tuple[int, int, int, int]]] = {}
    try:
        page_payloads = detector.export_character_review_page_payloads()
        if page_payloads:
            for page_number, payload in page_payloads.items():
                boxes: list[tuple[int, int, int, int]] = []
                for item in payload.get("texts", []) or []:
                    try:
                        x, y, width, height = [int(value) for value in (item.get("bbox") or [])[:4]]
                    except Exception:
                        continue
                    if width > 0 and height > 0:
                        boxes.append((x, y, width, height))
                if boxes:
                    detector_text_boxes_by_page[int(page_number)] = boxes
            write_json(
                store._project_dir(context.project_id) / "output" / "character_review_page_payloads.json",
                page_payloads,
            )
    except Exception:
        logger.warning("Could not persist character-review page payload cache for %s", context.project_id)
    context.progress(70, "Reconstructing panels from full-page OCR")
    reconstructed_panels, reconstruction_report = reconstruction.reconstruct(
        page_paths,
        detected_panels,
        metadata=project.chapter_metadata,
        detector_text_boxes_by_page=detector_text_boxes_by_page,
        progress_callback=lambda progress, message: context.progress(70 + progress * 0.18, message),
        cancel_callback=context.ensure_not_cancelled,
    )
    context.progress(90, "Linking cross-page continuation panels")
    merged_panels, merge_report = cross_page_merger.merge(
        page_paths,
        reconstructed_panels,
        progress_callback=lambda progress, message: context.progress(90 + progress * 0.05, message),
        cancel_callback=context.ensure_not_cancelled,
    )
    write_json(store._project_dir(context.project_id) / "output" / "panel_reconstruction.json", reconstruction_report)
    write_json(store._project_dir(context.project_id) / "output" / "cross_page_panel_merges.json", merge_report)
    page_text_boxes = reconstruction_report.get("page_text_boxes")
    if page_text_boxes:
        write_json(store._project_dir(context.project_id) / "output" / "page_ocr_boxes.json", page_text_boxes)
    context.progress(96, "Saving detected panels")

    # Preserve pages the user has manually reviewed — their corrections override fresh detections
    existing_panels = store.load_panels(context.project_id) or []
    locked_pages = {int(p.page) for p in existing_panels if p.detection_locked}
    if locked_pages:
        locked_by_page: dict[int, list] = {}
        for p in existing_panels:
            if int(p.page) in locked_pages:
                locked_by_page.setdefault(int(p.page), []).append(p)
        unlocked_fresh = [p for p in merged_panels if int(p.page) not in locked_pages]
        locked_preserved = [p for page_panels in locked_by_page.values() for p in page_panels]
        merged_panels = sorted(
            unlocked_fresh + locked_preserved,
            key=lambda p: (int(p.page), int(p.order)),
        )

    store.save_panels(context.project_id, merged_panels)
    panel_quality = store.load_panel_quality_report(context.project_id)
    store.update_stage_state(
        context.project_id,
        PipelineStage.PANEL_REVIEW,
        StageStatus.NEEDS_REVIEW,
        progress=100,
        message=(
            str(panel_quality.get("summary") or "").strip()
            if bool(panel_quality.get("should_block_script"))
            else "Panels detected. Review and save them to continue automatically."
        ),
    )
    store.update_stage_state(
        context.project_id,
        PipelineStage.CHARACTER_REVIEW,
        StageStatus.PENDING,
        progress=0,
        message="Character review will unlock after panel review is saved.",
    )
    store.update_stage_state(
        context.project_id,
        PipelineStage.SCRIPT_GENERATION,
        StageStatus.PENDING,
        progress=0,
        message="Waiting for panel review changes.",
    )
    store.update_stage_state(
        context.project_id,
        PipelineStage.NARRATION_GENERATION,
        StageStatus.PENDING,
        progress=0,
        message="Generate a script before creating audio.",
    )
    store.update_stage_state(
        context.project_id,
        PipelineStage.VIDEO_RENDERING,
        StageStatus.PENDING,
        progress=0,
        message="Generate audio before rendering video.",
    )
    context.complete("Detected panels with hybrid MAGI + OCR reconstruction")
    continue_auto_run_pipeline(store, context.queue, context.project_id, source="panel detection")



def _clear_script_generation_caches(project_dir: Path) -> None:
    output_dir = project_dir / "output"
    for path in (
        output_dir / "gemini_summary_cache.json",
        output_dir / "page_vision_cache.json",
        output_dir / "panel_captions_cache.json",
        output_dir / "panel_script_blocks.json",
        output_dir / "scene_summaries.json",
        output_dir / "story_bible.json",
        output_dir / "story_grounding.json",
        output_dir / "story_segments.json",
        output_dir / "narration_story.txt",
    ):
        path.unlink(missing_ok=True)


def _persist_dialogue_artifacts(project_dir: Path, artifacts: dict[str, Any]) -> None:
    write_json(project_dir / "output" / "character_identity_report.json", artifacts.get("character_identity_report", {}))
    write_json(project_dir / "output" / "scene_clusters.json", artifacts.get("scene_clusters", []))
    write_json(project_dir / "output" / "character_clusters.json", artifacts.get("character_clusters", []))
    write_json(project_dir / "output" / "character_tracking.json", artifacts.get("character_tracking", {}))
    write_json(project_dir / "output" / "characters.json", artifacts.get("characters", {}))
    write_json(project_dir / "output" / "character_dictionary.json", artifacts.get("character_dictionary", {}))
    write_json(
        project_dir / "output" / "speaker_attributions.json",
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
            for scene in artifacts.get("scenes", []) or []
            if isinstance(scene, dict)
        ],
    )
    write_json(project_dir / "output" / "dialogue_pipeline_manifest.json", artifacts)


def _dialogue_lines_for_scene(scene: dict[str, Any]) -> list[str]:
    lines = combined_dialogue_entry_lines(scene.get("dialogue_entries", []) or [])
    if not lines:
        lines = clean_ocr_lines(scene.get("dialogue", []) or [])
    if not lines:
        lines = clean_ocr_lines(scene.get("dialogue_original", []) or [])
    detected_text = str(scene.get("detected_text") or "").strip()
    if not lines and detected_text:
        lines = clean_ocr_lines([detected_text])
    return [line for line in lines if line]


def _build_script_slot_evidence(
    kept_panels: list[Any],
    ordered_payloads: list[dict[str, Any]],
    scenes: list[dict[str, Any]],
    scene_seeds: list[dict[str, Any]],
    scene_summaries: list[dict[str, Any]],
    protagonist_name: str | None,
) -> list[dict[str, Any]]:
    payload_by_order = {
        int(payload.get("panel") or 0): payload
        for payload in ordered_payloads
        if int(payload.get("panel") or 0)
    }
    scene_by_order = {
        int(scene.get("panel_order") or 0): scene
        for scene in scenes
        if int(scene.get("panel_order") or 0)
    }
    scene_lookup: dict[int, dict[str, Any]] = {}
    for seed in scene_seeds:
        scene_id = int(seed.get("scene_id") or 0)
        for panel_order in seed.get("panels", []) or []:
            try:
                scene_lookup[int(panel_order)] = seed
            except Exception:
                continue
    summary_lookup = {
        int(item.get("scene_id") or item.get("beat_id") or 0): item
        for item in scene_summaries
        if int(item.get("scene_id") or item.get("beat_id") or 0)
    }

    evidence: list[dict[str, Any]] = []
    for panel in kept_panels:
        payload = payload_by_order.get(int(panel.order), {})
        scene = scene_by_order.get(int(panel.order), {})
        seed = scene_lookup.get(int(panel.order), {})
        summary_item = summary_lookup.get(int(seed.get("scene_id") or 0), {})
        character_names = [
            str(name).strip()
            for name in (
                payload.get("character_names", [])
                or scene.get("character_names", [])
                or seed.get("character_names", [])
                or []
            )
            if str(name).strip()
        ]
        preferred_subject = character_names[0] if character_names else str(protagonist_name or "").strip()
        evidence.append(
            {
                "panel_id": panel.id,
                "panel_order": int(panel.order),
                "page": int(panel.page),
                "ocr_text": str(payload.get("text") or "").strip(),
                "dialogue": _dialogue_lines_for_scene(scene)[:3],
                "character_names": list(dict.fromkeys(character_names)),
                "preferred_subject": preferred_subject,
                "visual_caption": str(payload.get("visual_caption") or panel.visual_caption or "").strip(),
                "scene_id": int(seed.get("scene_id") or 0),
                "scene_summary": str(
                    summary_item.get("description")
                    or summary_item.get("summary")
                    or seed.get("combined_text")
                    or ""
                ).strip(),
            }
        )
    return evidence


def _reset_unlocked_manual_narration(project: object, store: ProjectStore, project_id: str) -> list:
    refreshed_panels = []
    changed = False
    for panel in getattr(project, "panels", []):
        next_manual = bool(panel.narration_locked and (panel.narration or "").strip())
        if bool(panel.manual_narration) != next_manual:
            changed = True
        refreshed_panels.append(panel.model_copy(update={"manual_narration": next_manual}))
    if changed:
        store.save_panels(project_id, refreshed_panels)
    return refreshed_panels


def _effective_narration_mode(project: object, payload: dict[str, Any] | None = None) -> str:
    return NarrationMode.PANEL.value



def _vision_stage_chain() -> tuple[PipelineStage, ...]:
    return (
        PipelineStage.CHARACTER_PORTRAIT,
        PipelineStage.PANEL_VISION_EXTRACTION,
        PipelineStage.PANEL_VISION_QUALITY,
    )


def _queue_requested_script_continuation(context: PipelineContext, next_stage: PipelineStage) -> bool:
    job = context.store.get_job(context.project_id, context.job_id)
    if not bool(job.payload.get("continue_to_script_generation")):
        return False
    payload = dict(job.payload or {})
    next_job = queue_stage_once(
        context.store,
        context.queue,
        context.project_id,
        next_stage,
        "Continuing requested script regeneration",
        payload=payload,
    )
    context.store.update_job(context.project_id, next_job.id, payload={**dict(next_job.payload or {}), **payload})
    return True


def run_character_review(context: PipelineContext) -> None:
    store = context.store
    project = store.get_project(context.project_id)
    job = store.get_job(context.project_id, context.job_id)
    page_paths = store.list_page_paths(context.project_id)
    kept_panels = [panel for panel in project.panels if panel.keep]
    if not page_paths:
        raise ValueError("No imported pages were found. Re-import the source files or MangaDex chapter and try again.")
    if not kept_panels:
        raise ValueError("No kept panels are available yet. Save panel review changes before preparing characters.")

    project_dir = store._project_dir(context.project_id)
    if bool(job.payload.get("force_refresh")):
        output_dir = project_dir / "output"
        for path in (
            output_dir / "character_review_state.json",
            output_dir / "character_identity_report.json",
            output_dir / "character_clusters.json",
            output_dir / "character_tracking.json",
            output_dir / "characters.json",
            output_dir / "character_dictionary.json",
            output_dir / "character_review_page_payloads.json",
            output_dir / "anime_face_page_payloads.json",
        ):
            path.unlink(missing_ok=True)
        shutil.rmtree(project_dir / "characters" / "review", ignore_errors=True)

    review_service = CharacterReviewService()
    context.start("Preparing character review suggestions")
    context.progress(4, "Preparing cached character data")
    persist_full_manifest = False
    try:
        artifacts, persist_full_manifest = review_service.prepare_review_artifacts(
            project_dir,
            project.chapter_metadata,
            project.panels,
            page_paths,
            progress_callback=lambda progress, message: context.progress(6 + progress * 0.86, message),
            cancel_callback=context.ensure_not_cancelled,
        )
    except Exception as exc:
        logger.warning("Fast character review path failed; falling back to full dialogue extraction: %s", exc)
        dialogue_pipeline = DialogueExtractionPipeline()
        page_ocr_boxes_path = project_dir / "output" / "page_ocr_boxes.json"
        page_text_boxes: dict[str, list] | None = None
        if page_ocr_boxes_path.exists():
            try:
                with open(page_ocr_boxes_path) as fh:
                    page_text_boxes = json.load(fh)
            except Exception:
                logger.warning("Could not load page_ocr_boxes.json while preparing character review")

        artifacts = dialogue_pipeline.run(
            project_dir,
            project.panels,
            project.chapter_metadata,
            page_text_boxes=page_text_boxes,
            progress_callback=lambda progress, message: context.progress(6 + progress * 0.86, message),
            cancel_callback=context.ensure_not_cancelled,
        )
        persist_full_manifest = True
    context.ensure_not_cancelled()
    context.progress(94, "Building character review cards")
    review_state = review_service.build_review_state(
        context.project_id,
        project_dir,
        project.name,
        project.chapter_metadata,
        project.panels,
        artifacts,
    )
    review_state = review_service.save_review_state(project_dir, project.name, project.chapter_metadata, review_state)
    if persist_full_manifest:
        _persist_dialogue_artifacts(project_dir, artifacts)
    else:
        write_json(project_dir / "output" / "character_identity_report.json", artifacts.get("character_identity_report", {}))
        write_json(project_dir / "output" / "character_clusters.json", artifacts.get("character_clusters", []))
        write_json(project_dir / "output" / "character_tracking.json", artifacts.get("character_tracking", {}))
        write_json(project_dir / "output" / "characters.json", artifacts.get("characters", {}))
        write_json(project_dir / "output" / "character_dictionary.json", artifacts.get("character_dictionary", {}))

    if review_state.identities:
        context.complete("Character review suggestions ready")
        store.update_stage_state(
            context.project_id,
            PipelineStage.CHARACTER_REVIEW,
            StageStatus.NEEDS_REVIEW,
            progress=100,
            message=f"Prepared {len(review_state.identities)} character groups. Review and save them, or continue to the script when ready.",
        )
    else:
        context.complete("No recurring characters needed review")
        store.update_stage_state(
            context.project_id,
            PipelineStage.CHARACTER_REVIEW,
            StageStatus.COMPLETED,
            progress=100,
            message="No recurring characters were found. Script generation is ready.",
        )
    next_stage = PipelineStage.SCRIPT_GENERATION
    next_message = "Character suggestions are ready. Review them or generate the script when you're ready."
    store.update_stage_state(context.project_id, next_stage, StageStatus.READY, progress=0, message=next_message)
    store.update_stage_state(
        context.project_id,
        PipelineStage.NARRATION_GENERATION,
        StageStatus.PENDING,
        progress=0,
        message="Generate a script before creating audio.",
    )
    store.update_stage_state(
        context.project_id,
        PipelineStage.VIDEO_RENDERING,
        StageStatus.PENDING,
        progress=0,
        message="Generate audio before rendering video.",
    )
    continue_auto_run_pipeline(store, context.queue, context.project_id, source="character review")


def run_character_portrait(context: PipelineContext) -> None:
    store = context.store
    project = store.get_project(context.project_id)
    job = store.get_job(context.project_id, context.job_id)
    force_refresh = bool(job.payload.get("force_refresh"))
    page_paths = store.list_page_paths(context.project_id)
    project_dir = store._project_dir(context.project_id)
    context.start("Enumerating canonical characters from page images")
    service = CharacterPortraitPass()
    records = service.run(
        project_dir=project_dir,
        page_paths=page_paths,
        panels=project.panels,
        project_title=project.name,
        chapter_metadata=project.chapter_metadata,
        force_refresh=force_refresh,
        progress_callback=context.progress,
        cancel_callback=context.ensure_not_cancelled,
    )
    store.update_stage_state(
        context.project_id,
        PipelineStage.PANEL_VISION_EXTRACTION,
        StageStatus.READY,
        progress=0,
        message="Canonical character roster ready. Panel vision extraction can start.",
    )
    context.complete(f"Canonical character roster prepared ({len(records)} characters)")
    if _queue_requested_script_continuation(context, PipelineStage.PANEL_VISION_EXTRACTION):
        return
    continue_auto_run_pipeline(store, context.queue, context.project_id, source="character portrait")


def run_panel_vision_extraction(context: PipelineContext) -> None:
    store = context.store
    project = store.get_project(context.project_id)
    job = store.get_job(context.project_id, context.job_id)
    force_refresh = bool(job.payload.get("force_refresh"))
    page_paths = store.list_page_paths(context.project_id)
    project_dir = store._project_dir(context.project_id)
    canonical_payload = read_json(project_dir / "output" / "canonical_characters.json", default=[])
    canonical_characters = canonical_payload if isinstance(canonical_payload, list) else []
    context.start("Extracting clean panel text evidence")
    evidence_extractor = PanelEvidenceExtractor()
    evidence_records = evidence_extractor.run(
        project_dir=project_dir,
        page_paths=page_paths,
        panels=project.panels,
        chapter_metadata=project.chapter_metadata,
        force_refresh=force_refresh or bool(job.payload.get("refresh_panel_evidence")),
        allow_crop_ocr=bool(job.payload.get("deep_panel_evidence_scan")),
        allow_apple_vision=bool(job.payload.get("apple_vision_panel_evidence")),
        allow_metadata_ocr=bool(job.payload.get("metadata_panel_evidence")),
        progress_callback=lambda progress, message: context.progress(progress * 0.35, message),
        cancel_callback=context.ensure_not_cancelled,
    )
    context.start("Reading kept panels with Gemini Vision")
    extractor = PanelVisionExtractor()
    records = extractor.run(
        project_dir=project_dir,
        page_paths=page_paths,
        panels=project.panels,
        canonical_characters=canonical_characters,
        project_title=project.name,
        chapter_metadata=project.chapter_metadata,
        force_refresh=force_refresh,
        progress_callback=lambda progress, message: context.progress(35 + progress * 0.65, message),
        cancel_callback=context.ensure_not_cancelled,
    )
    store.update_stage_state(
        context.project_id,
        PipelineStage.PANEL_VISION_QUALITY,
        StageStatus.READY,
        progress=0,
        message="Panel vision draft ready. Quality rescue can start.",
    )
    context.complete(f"Panel vision extracted for {len(records)} kept panels ({len(evidence_records)} clean evidence records)")
    if _queue_requested_script_continuation(context, PipelineStage.PANEL_VISION_QUALITY):
        return
    continue_auto_run_pipeline(store, context.queue, context.project_id, source="panel vision extraction")


def run_panel_vision_quality(context: PipelineContext) -> None:
    store = context.store
    project = store.get_project(context.project_id)
    job = store.get_job(context.project_id, context.job_id)
    force_refresh = bool(job.payload.get("force_refresh"))
    page_paths = store.list_page_paths(context.project_id)
    project_dir = store._project_dir(context.project_id)
    canonical_payload = read_json(project_dir / "output" / "canonical_characters.json", default=[])
    canonical_characters = canonical_payload if isinstance(canonical_payload, list) else []
    context.start("Rescuing low-confidence panel vision reads")
    service = PanelVisionQualityService()
    records = service.run(
        project_dir=project_dir,
        page_paths=page_paths,
        panels=project.panels,
        canonical_characters=canonical_characters,
        project_title=project.name,
        chapter_metadata=project.chapter_metadata,
        force_refresh=force_refresh,
        progress_callback=context.progress,
        cancel_callback=context.ensure_not_cancelled,
    )
    store.update_stage_state(
        context.project_id,
        PipelineStage.SCRIPT_GENERATION,
        StageStatus.READY,
        progress=0,
        message="Vision artefacts are ready. Script generation can start.",
    )
    context.complete(f"Panel vision quality pass finished ({len(records)} panels)")
    if _queue_requested_script_continuation(context, PipelineStage.SCRIPT_GENERATION):
        return
    continue_auto_run_pipeline(store, context.queue, context.project_id, source="panel vision quality")


def run_script_generation(context: PipelineContext) -> None:
    store = context.store
    project = store.get_project(context.project_id)
    project_dir = store._project_dir(context.project_id)
    job = store.get_job(context.project_id, context.job_id)
    repair_weak_segments = bool(job.payload.get("repair_weak_segments"))
    stop_after_stage = bool(job.payload.get("stop_after_stage"))
    force_refresh = bool(job.payload.get("force_refresh"))
    stage_started_at = time.perf_counter()

    if repair_weak_segments:
        context.start("Repairing weak story segments")
        repair_service = StorySegmentRepairService(store=store)
        result = repair_service.repair_project(
            context.project_id,
            batch_size=_coerce_positive_int(job.payload.get("repair_batch_size")),
            max_segments=_coerce_positive_int(job.payload.get("max_repair_segments")),
            use_local_ocr_rescue=bool(job.payload.get("repair_with_local_ocr")),
            progress_callback=context.progress,
            cancel_callback=context.ensure_not_cancelled,
        )
        store.update_stage_state(
            context.project_id,
            PipelineStage.NARRATION_GENERATION,
            StageStatus.READY,
            progress=0,
            message="Story repair complete. Generate fresh audio when you are ready.",
        )
        store.update_stage_state(
            context.project_id,
            PipelineStage.VIDEO_RENDERING,
            StageStatus.PENDING,
            progress=0,
            message="Video will be available after fresh audio generation.",
        )
        logger.info(
            "Incremental story repair for %s completed in %.2fs (%d/%d targets repaired)",
            context.project_id,
            time.perf_counter() - stage_started_at,
            result.repaired_segments,
            result.target_segments,
        )
        context.complete(
            f"Story repair complete: {result.repaired_segments}/{result.target_segments} weak segment"
            f"{'s' if result.target_segments != 1 else ''} improved"
        )
        return

    context.start("Generating recap script")
    if force_refresh:
        _clear_script_generation_caches(project_dir)
        project = project.model_copy(
            update={"panels": _reset_unlocked_manual_narration(project, store, context.project_id)}
        )
        logger.info("Force-refreshing script generation caches for %s", context.project_id)
    page_paths = store.list_page_paths(context.project_id)
    page_path_lookup = {index: path for index, path in enumerate(page_paths, start=1)}
    page_size_cache: dict[int, tuple[int, int] | None] = {}

    def page_size_for(page_number: int) -> tuple[int, int] | None:
        if page_number not in page_size_cache:
            page_path = page_path_lookup.get(page_number)
            if page_path is None:
                page_size_cache[page_number] = None
            else:
                try:
                    with Image.open(page_path) as image:
                        page_size_cache[page_number] = tuple(int(value) for value in image.size)
                except Exception:
                    logger.warning("Could not read page size for %s page %s", context.project_id, page_number)
                    page_size_cache[page_number] = None
        return page_size_cache[page_number]

    narration_mode = NarrationMode.PANEL.value
    panel_vision_records = None
    canonical_characters = None

    # ── Step 1: Dialogue extraction (unchanged) ─────────────────────
    context.progress(10, "Extracting dialogue and scene context")
    dialogue_pipeline = DialogueExtractionPipeline()
    page_ocr_boxes_path = project_dir / "output" / "page_ocr_boxes.json"
    page_text_boxes: dict[str, list] | None = None
    if page_ocr_boxes_path.exists():
        try:
            with open(page_ocr_boxes_path) as fh:
                page_text_boxes = json.load(fh)
        except Exception:
            logger.warning("Could not load page_ocr_boxes.json, skipping page-level backfill")
    has_existing_panel_text = any(
        is_usable_ocr_text(clean_ocr_text(str(panel.ocr_text or "")))
        for panel in project.panels
    )
    allow_expensive_dialogue_ocr = bool(
        job.payload.get("refresh_dialogue_context")
        or job.payload.get("deep_dialogue_scan")
        or (not page_text_boxes and not has_existing_panel_text)
    )
    if not allow_expensive_dialogue_ocr:
        logger.info(
            "Using fast dialogue-context refresh for %s; pass refresh_dialogue_context=true for a deep OCR rebuild",
            context.project_id,
        )
    ocr_started_at = time.perf_counter()
    artifacts = dialogue_pipeline.run(
        project_dir,
        project.panels,
        project.chapter_metadata,
        page_text_boxes=page_text_boxes,
        allow_expensive_ocr=allow_expensive_dialogue_ocr,
        progress_callback=lambda progress, message: context.progress(10 + progress * 0.25, message),
        cancel_callback=context.ensure_not_cancelled,
    )
    review_service = CharacterReviewService()
    artifacts = review_service.apply_review_to_artifacts(project_dir, artifacts)
    _persist_dialogue_artifacts(project_dir, artifacts)
    logger.info(
        "Script generation OCR stage for %s finished in %.2fs with metrics=%s",
        context.project_id,
        time.perf_counter() - ocr_started_at,
        artifacts.get("metrics", {}),
    )
    context.ensure_not_cancelled()
    project = store.get_project(context.project_id)
    scenes = artifacts["scenes"]
    scene_clusters = artifacts.get("scene_clusters", [])
    character_dictionary = artifacts.get("character_dictionary") or {}
    protagonist_name = artifacts.get("protagonist_name")
    # Strip out OCR artifacts and non-name phrases (e.g. Portuguese filler words
    # like "por favor" that the dialogue extractor may mis-identify as speakers).
    character_dictionary = _filter_character_dictionary(character_dictionary)
    write_json(project_dir / "output" / "character_dictionary.json", character_dictionary)
    write_json(
        project_dir / "output" / "character_identity_report.json",
        {
            "character_dictionary": character_dictionary,
            "protagonist_name": protagonist_name,
        },
    )

    # ── Load cached vision evidence ─────────────────────────────────
    # If a panel_vision_final.json exists from a prior portrait + vision
    # extraction run, load it now so panel-mode narration gets rich
    # per-panel evidence (action_beat, dialogue, caption, visual_cues).
    # This runs non-fatally — if the file is missing or corrupt, panel
    # mode falls back to OCR-only evidence without crashing.
    vision_final_path = project_dir / "output" / "panel_vision_final.json"
    canonical_path = project_dir / "output" / "canonical_characters.json"
    if vision_final_path.exists():
        try:
            vision_payload = read_json(vision_final_path, default=[])
            vision_list = vision_payload.get("records") if isinstance(vision_payload, dict) else vision_payload
            if isinstance(vision_list, list) and vision_list:
                panel_vision_records = [
                    PanelVisionRecord(**r) for r in vision_list if isinstance(r, dict)
                ]
                logger.info(
                    "Loaded %d panel vision records for %s (panel-mode evidence enrichment)",
                    len(panel_vision_records),
                    context.project_id,
                )
        except Exception as exc:
            logger.warning("Could not load panel_vision_final.json (non-fatal): %s", exc)
    if canonical_path.exists():
        try:
            canonical_list = read_json(canonical_path, default=[])
            if isinstance(canonical_list, list) and canonical_list:
                canonical_characters = [
                    CanonicalCharacterRecord(**r) for r in canonical_list if isinstance(r, dict)
                ]
                logger.info(
                    "Loaded %d canonical characters for %s",
                    len(canonical_characters),
                    context.project_id,
                )
        except Exception as exc:
            logger.warning("Could not load canonical_characters.json (non-fatal): %s", exc)

    # Update panel OCR text from dialogue extraction
    scene_lookup = {scene["panel_id"]: scene for scene in scenes if scene.get("panel_id")}
    updated_panels = []
    kept_after_skip = 0
    recovered_auto_skipped_panels = 0
    for panel in project.panels:
        scene = scene_lookup.get(panel.id)
        detected_text = ""
        has_dialogue = False

        if panel.manual_ocr_text:
            detected_text = clean_ocr_text(panel.ocr_text or "").strip()
            has_dialogue = is_usable_ocr_text(detected_text)
        elif scene:
            cleaned_lines = combined_dialogue_entry_lines(scene.get("dialogue_entries", []) or [])
            if not cleaned_lines:
                cleaned_lines = clean_ocr_lines(scene.get("dialogue", []))
            if not cleaned_lines:
                cleaned_lines = clean_ocr_lines(scene.get("dialogue_original", []))
            if not cleaned_lines and scene.get("detected_text"):
                cleaned_lines = clean_ocr_lines([str(scene.get("detected_text", ""))])
            detected_text = " ".join(cleaned_lines).strip()
            has_dialogue = is_usable_ocr_text(detected_text)
        else:
            detected_text = _page_ocr_text_for_panel(panel, page_text_boxes)
            has_dialogue = is_usable_ocr_text(detected_text)

        if panel.manual_ocr_text:
            has_dialogue = is_usable_ocr_text(detected_text)
        keep = panel.keep
        auto_skipped = False if keep else panel.auto_skipped
        skip_reason = None if keep else panel.skip_reason

        if has_dialogue:
            if (
                not keep
                and panel.auto_skipped
                and _should_recover_auto_skipped_panel_with_text(
                    panel,
                    page_size_for(int(panel.page)),
                )
            ):
                keep = True
                auto_skipped = False
                skip_reason = None
                recovered_auto_skipped_panels += 1
            if keep:
                kept_after_skip += 1
        elif panel.manual_keep:
            keep = True
            kept_after_skip += 1
            auto_skipped = False
            skip_reason = None
        elif panel.keep:
            kept_after_skip += 1
            auto_skipped = False
            skip_reason = None

        updated_panels.append(
            panel.model_copy(
                update={
                    "keep": keep,
                    "ocr_text": detected_text or None,
                    "text_detected": has_dialogue,
                    "auto_skipped": auto_skipped,
                    "skip_reason": skip_reason,
                    "manual_ocr_text": panel.manual_ocr_text,
                }
            )
        )

    if kept_after_skip == 0:
        raise ValueError("No kept panels remain after script preparation. Review the editor and add back any panels you still want narrated.")

    store.save_panels(context.project_id, updated_panels)
    if recovered_auto_skipped_panels:
        logger.info(
            "Recovered %d auto-skipped panels with page-level OCR text for %s",
            recovered_auto_skipped_panels,
            context.project_id,
        )
    context.ensure_not_cancelled()

    # ── Step 1b: Character visual profiling ────────────────────────
    # Enrich character_dictionary with appearance descriptions so the
    # LLM can identify characters visually in panels with no OCR text.
    context.progress(36, "Profiling character appearances")
    try:
        profiler = CharacterVisualProfiler()
        character_dictionary = profiler.enrich_character_dictionary(
            character_dictionary,
            updated_panels,
            scenes,
            panel_image_dir=project_dir / "panels",
            cache_dir=project_dir / "output",
        )
        write_json(project_dir / "output" / "character_dictionary.json", character_dictionary)
    except Exception as exc:
        logger.warning("Character visual profiling failed (non-fatal): %s", exc)

    # ── Step 2: Story-first script generation ───────────────────────
    context.progress(38, "Building scene-level story script")
    router = LLMRouter()
    story_service = StoryScriptService(router)
    style_vocab: StyleVocabulary | None = None
    existing_story_bible = read_json(project_dir / "output" / "story_bible.json", default={})
    if isinstance(existing_story_bible, dict) and existing_story_bible:
        existing_scene_summaries = read_json(project_dir / "output" / "scene_summaries.json", default={})
        style_vocab = build_style_vocabulary(
            canonical_characters=canonical_characters or [],
            character_dictionary=character_dictionary,
            story_bible=existing_story_bible,
            scene_summaries=existing_scene_summaries,
            chapter_summary=(
                str(existing_scene_summaries.get("chapter_summary") or existing_story_bible.get("chapter_premise") or "")
                if isinstance(existing_scene_summaries, dict)
                else str(existing_story_bible.get("chapter_premise") or "")
            ),
        )

    # Fetch external series context via Gemini grounded search (non-fatal).
    series_context: dict = {}
    series_title = (project.chapter_metadata.manga_title or project.name or "").strip()
    if series_title:
        context.progress(39, f"Looking up series context for '{series_title}'")
        try:
            import asyncio as _asyncio
            series_context = _asyncio.run(
                router.fetch_series_context(
                    series_title,
                    story_service._chapter_metadata_payload(project.chapter_metadata),
                )
            )
            if series_context.get("search_context"):
                logger.info(
                    "Fetched %d chars of grounded context for '%s'",
                    len(series_context["search_context"]),
                    series_title,
                )
        except Exception as exc:
            logger.warning("Series context fetch failed (non-fatal): %s", exc)
            series_context = {}

    story_started_at = time.perf_counter()
    story_bundle = story_service.generate(
        project_title=project.name or "",
        chapter_metadata=project.chapter_metadata,
        panels=updated_panels,
        scenes=scenes,
        scene_clusters=scene_clusters,
        character_dictionary=character_dictionary,
        protagonist_name=protagonist_name,
        cache_dir=project_dir / "output",
        narration_mode=narration_mode,
        series_context=series_context,
        progress_callback=lambda p, msg: context.progress(39 + p * 0.55, msg),
        panel_vision_records=panel_vision_records,
        panel_evidence_records=None,
        canonical_characters=canonical_characters,
        style_vocab=style_vocab,
        disable_multimodal_rescue=bool(job.payload.get("disable_multimodal_rescue")),
    )
    disable_image_repair = bool(
        job.payload.get("disable_image_repair")
        or job.payload.get("disable_multimodal_rescue")
    )
    logger.info(
        "Story-first script generation for %s completed in %.2fs (%d segments)",
        context.project_id,
        time.perf_counter() - story_started_at,
        len(story_bundle.story_segments),
    )
    write_json(
        project_dir / "output" / "scene_summaries.json",
        {
            "chapter_summary": story_bundle.chapter_summary,
            "scenes": story_bundle.scene_summaries,
            "scene_seeds": story_bundle.scene_seeds,
        },
    )
    write_json(project_dir / "output" / "story_bible.json", story_bundle.story_bible)
    write_json(project_dir / "output" / "story_grounding.json", story_bundle.grounding_state)
    if story_bundle.style_vocabulary:
        write_json(project_dir / "output" / "style_vocabulary.json", story_bundle.style_vocabulary.to_dict())
    context.ensure_not_cancelled()

    # ── Step 3: Save ────────────────────────────────────────────────
    context.progress(95, "Saving story script")
    store.save_story_segments(
        context.project_id,
        story_bundle.story_segments,
        story_block=story_bundle.story_text,
    )
    if story_bundle.story_text:
        story_path = project_dir / "output" / "narration_story.txt"
        story_path.write_text(story_bundle.story_text.strip() + "\n", encoding="utf-8")
    context.ensure_not_cancelled()

    # ── Step 4: Auto-repair weak segments ───────────────────────────
    # Runs the incremental repair pass automatically so any blank /
    # visual-only / generic beats are fixed before audio generation,
    # without requiring a separate manual "Repair segments" click.
    if disable_image_repair:
        logger.info("Skipping auto-repair for %s because image/script repair is disabled", context.project_id)
    else:
        context.progress(96, "Repairing weak story segments")
        try:
            repair_service = StorySegmentRepairService(store=store, style_vocab=story_bundle.style_vocabulary)
            repair_result = repair_service.repair_project(
                context.project_id,
                batch_size=3,
                use_local_ocr_rescue=True,
                use_multimodal_rescue=True,
                progress_callback=lambda p, msg: context.progress(96 + p * 0.03, msg),
                cancel_callback=context.ensure_not_cancelled,
            )
            logger.info(
                "Auto-repair for %s: %d/%d weak segments improved",
                context.project_id,
                repair_result.repaired_segments,
                repair_result.target_segments,
            )
        except Exception as exc:
            logger.warning("Auto-repair pass failed (non-fatal): %s", exc)

    if stop_after_stage:
        store.update_stage_state(
            context.project_id,
            PipelineStage.NARRATION_GENERATION,
            StageStatus.READY,
            progress=0,
            message="Script ready. Generate audio when you want to continue.",
        )
    else:
        store.update_stage_state(
            context.project_id,
            PipelineStage.NARRATION_GENERATION,
            StageStatus.READY,
            progress=0,
            message="Starting narration audio automatically",
        )
    logger.info(
        "Script generation stage for %s completed in %.2fs",
        context.project_id,
        time.perf_counter() - stage_started_at,
    )
    context.complete("Narration script ready")
    if not stop_after_stage:
        queue_stage_once(
            store,
            context.queue,
            context.project_id,
            PipelineStage.NARRATION_GENERATION,
            "Queued automatically after script generation",
        )


def run_narration_generation(context: PipelineContext) -> None:
    store = context.store
    project = store.get_project(context.project_id)
    story_segments = [
        segment
        for segment in list(project.story_segments or store.load_story_segments(context.project_id))
        if bool(getattr(segment, "keep", True))
    ]
    script_lines = [segment.text.strip() for segment in story_segments if segment.text.strip()] or list(project.script_lines)
    if not script_lines:
        raise ValueError("No script is available. Generate or save a script before creating audio.")
    quality_report = store.load_script_quality_report(context.project_id)
    _job = store.get_job(context.project_id, context.job_id)
    force_bypass = bool((_job.payload or {}).get("force_quality_bypass"))
    if bool(quality_report.get("should_block_tts")) and not force_bypass:
        summary = str(quality_report.get("summary") or "Script quality checks found too many problems for automatic TTS.")
        raise ValueError(
            f"{summary} Review the story segments, regenerate the script, or manually fix the flagged text before creating audio."
        )

    context.start("Generating voice narration")
    project_dir = store._project_dir(context.project_id)
    narration_texts = [segment.text.strip() for segment in story_segments if segment.text.strip()] or script_lines
    narration_ids = [segment.id for segment in story_segments if segment.text.strip()] or [f"segment_{index:03d}" for index, _ in enumerate(narration_texts, start=1)]
    character_names = _collect_pronunciation_names(project_dir, narration_texts + list(script_lines))
    generate_narration(
        narration_texts,
        Path(store._project_dir(context.project_id) / "audio"),
        project.voice_config,
        panel_ids=narration_ids,
        progress_callback=context.progress,
        cancel_callback=context.ensure_not_cancelled,
        language_hint=project.chapter_metadata.language,
        pronunciation_dictionary={},
        character_names=character_names,
    )
    store.update_stage_state(
        context.project_id,
        PipelineStage.VIDEO_RENDERING,
        StageStatus.READY,
        progress=0,
        message="Starting video rendering automatically",
    )
    context.complete("Narration audio generated")
    queue_stage_once(
        store,
        context.queue,
        context.project_id,
        PipelineStage.VIDEO_RENDERING,
        "Queued automatically after audio generation",
    )


def run_video_rendering(context: PipelineContext) -> None:
    store = context.store
    project = store.get_project(context.project_id)
    story_segments = [
        segment
        for segment in list(project.story_segments or store.load_story_segments(context.project_id))
        if bool(getattr(segment, "keep", True))
    ]
    script_lines = [segment.text.strip() for segment in story_segments if segment.text.strip()] or list(project.script_lines)
    if not script_lines:
        raise ValueError("No script is available. Generate or save a script before rendering the video.")
    if not project.audio_files:
        raise ValueError("No narration audio is available. Generate audio before rendering the video.")
    service = VideoRenderService()
    context.start("Rendering video with FFmpeg")
    render_panels = project.panels
    service.render_project_video(
        store._project_dir(context.project_id),
        render_panels,
        story_segments,
        project.video_config,
        project.music_config,
        progress_callback=context.progress,
        cancel_callback=context.ensure_not_cancelled,
    )
    context.complete("Video exported")


STAGE_HANDLERS = {
    PipelineStage.INGESTION: run_ingestion,
    PipelineStage.PANEL_DETECTION: run_panel_detection,
    PipelineStage.CHARACTER_REVIEW: run_character_review,
    PipelineStage.CHARACTER_PORTRAIT: run_character_portrait,
    PipelineStage.PANEL_VISION_EXTRACTION: run_panel_vision_extraction,
    PipelineStage.PANEL_VISION_QUALITY: run_panel_vision_quality,
    PipelineStage.SCRIPT_GENERATION: run_script_generation,
    PipelineStage.NARRATION_GENERATION: run_narration_generation,
    PipelineStage.VIDEO_RENDERING: run_video_rendering,
}
