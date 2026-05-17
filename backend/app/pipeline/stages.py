from __future__ import annotations

import json
import logging
import re
import shutil
import time
import asyncio
from pathlib import Path
from typing import Any
from PIL import Image

from app.pipeline.context import PipelineContext
from app.pipeline.auto_run import continue_auto_run_pipeline
from app.pipeline.orchestration import queue_stage_once
from app.schemas.project import (
    CanonicalCharacterRecord,
    JobStatus,
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
from app.services.script_quality_service import ScriptQualityService
from app.services.script_generation_vnext import ScriptGenerationVNextService, ScriptVNextRedraftConfig
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


def _collect_supported_narration_names(project_dir: Path, project: Any) -> list[str]:
    names: list[str] = []

    def add(value: object) -> None:
        cleaned = _clean_pronunciation_candidate(str(value or ""))
        if cleaned and not looks_like_false_character_name(cleaned):
            names.append(cleaned)

    style_payload = read_json(project_dir / "output" / "style_vocabulary.json", default={})
    if isinstance(style_payload, dict):
        for value in style_payload.get("named_characters") or []:
            add(value)
        protagonist = style_payload.get("protagonist")
        if protagonist:
            add(protagonist)

    appearances_payload = read_json(project_dir / "output" / "character_appearances.json", default={})
    if isinstance(appearances_payload, dict):
        for key in appearances_payload.keys():
            add(key)

    canonical_payload = read_json(project_dir / "output" / "canonical_characters.json", default=[])
    if isinstance(canonical_payload, list):
        for item in canonical_payload:
            if not isinstance(item, dict):
                continue
            add(item.get("name"))
            for alias in item.get("aliases") or []:
                add(alias)

    metadata_raw = getattr(project.chapter_metadata, "raw", {}) or {}
    manga = metadata_raw.get("manga") if isinstance(metadata_raw, dict) else {}
    if isinstance(manga, dict):
        for value in manga.get("cast") or manga.get("characters") or []:
            if isinstance(value, dict):
                add(value.get("name") or value.get("display_name"))
            else:
                add(value)

    unique: list[str] = []
    seen: set[str] = set()
    for name in names:
        key = name.casefold()
        if key in seen:
            continue
        seen.add(key)
        unique.append(name)
    return unique


def _collect_narration_world_terms(project_dir: Path, project: Any) -> list[str]:
    terms: list[str] = []
    style_payload = read_json(project_dir / "output" / "style_vocabulary.json", default={})
    if isinstance(style_payload, dict):
        terms.extend(str(value).strip() for value in style_payload.get("world_terms") or [] if str(value).strip())
        for key in ("antagonist_term", "team_term"):
            value = str(style_payload.get(key) or "").strip()
            if value:
                terms.append(value)
    metadata_raw = getattr(project.chapter_metadata, "raw", {}) or {}
    manga = metadata_raw.get("manga") if isinstance(metadata_raw, dict) else {}
    if isinstance(manga, dict):
        terms.extend(str(value).strip() for value in manga.get("world_terms") or [] if str(value).strip())
    for value in (
        getattr(project.chapter_metadata, "manga_title", None),
        getattr(project.chapter_metadata, "chapter_title", None),
    ):
        if str(value or "").strip():
            terms.append(str(value).strip())
    unique: list[str] = []
    seen: set[str] = set()
    for term in terms:
        key = term.casefold()
        if key in seen:
            continue
        seen.add(key)
        unique.append(term)
    return unique


def _build_cast_block_for_name_resolution(
    canonical_characters: list[Any] | None,
) -> str:
    """Render canonical characters as the cast block the name resolver
    expects. Mirrors CastBibleService.format_for_prompt format so the
    resolver prompt sees the same shape regardless of where the cast
    came from (CastBible or canonical_characters.json).

    Empty / None canonical list returns "" so the resolver skips the
    pass entirely instead of running against an empty bible.
    """
    if not canonical_characters:
        return ""
    lines: list[str] = ["KNOWN CAST (use these names when you can match them in the panel):"]
    seen: set[str] = set()
    for character in canonical_characters:
        name = str(getattr(character, "name", "") or "").strip()
        if not name or name.casefold() in seen:
            continue
        seen.add(name.casefold())
        description = str(getattr(character, "visual_description", "") or "").strip()
        role = str(getattr(character, "role", "") or "").strip()
        aliases = list(getattr(character, "aliases", []) or [])
        piece = f"  • {name}"
        if description:
            piece += f" - {description}"
        elif role:
            piece += f" - {role}"
        if aliases:
            piece += f" (also known as: {', '.join(aliases[:3])})"
        lines.append(piece)
    return "\n".join(lines) if len(lines) > 1 else ""


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

    # Preserve pages the user has manually reviewed - their corrections override fresh detections
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
        output_dir / "dialogue_pipeline_manifest.json",
        output_dir / "ocr_results.json",
        output_dir / "transcript.json",
        output_dir / "ocr_coverage.json",
        output_dir / "gemini_scenes.json",
        output_dir / "speaker_attributions.json",
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


def _script_dialogue_progress(progress: float, message: str) -> float:
    """Map dialogue extraction's internal percent onto the script stage without a jumpy start."""
    normalized = str(message or "").lower()
    count_match = re.search(r"panel\s+(\d+)\s*/\s*(\d+)", normalized)
    if count_match:
        current = max(int(count_match.group(1)), 0)
        total = max(int(count_match.group(2)), 1)
        return 10 + min(current / total, 1.0) * 24

    clamped = max(0.0, min(float(progress or 0.0), 100.0))
    if clamped < 22:
        return 10
    if clamped < 74:
        return 10 + ((clamped - 22) / 52) * 24
    return 34 + ((clamped - 74) / 26) * 4


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


def _transcript_lines_by_panel_order(project_dir: Path) -> dict[int, list[str]]:
    transcript = read_json(project_dir / "output" / "transcript.json", default={})
    fragments = transcript.get("fragments", []) if isinstance(transcript, dict) else []
    if not isinstance(fragments, list):
        return {}
    lines_by_order: dict[int, list[str]] = {}
    seen_by_order: dict[int, set[str]] = {}
    for item in fragments:
        if not isinstance(item, dict) or not bool(item.get("accepted", True)):
            continue
        try:
            order = int(item.get("panel_order") or 0)
        except Exception:
            order = 0
        if order <= 0:
            continue
        text = clean_ocr_text(
            str(item.get("repaired_text") or item.get("text") or item.get("cleaned_text") or "")
        ).strip()
        if not text or not is_usable_ocr_text(text):
            continue
        key = re.sub(r"\W+", " ", text.casefold()).strip()
        if not key:
            continue
        seen = seen_by_order.setdefault(order, set())
        if key in seen:
            continue
        seen.add(key)
        lines_by_order.setdefault(order, []).append(text)
    return lines_by_order


def _transcript_evidence_records(project_dir: Path) -> list[dict[str, Any]]:
    transcript = read_json(project_dir / "output" / "transcript.json", default={})
    fragments = transcript.get("fragments", []) if isinstance(transcript, dict) else []
    if not isinstance(fragments, list):
        return []
    merged_by_order: dict[int, dict[str, Any]] = {}
    seen_by_order: dict[int, set[str]] = {}
    for item in fragments:
        if not isinstance(item, dict) or not bool(item.get("accepted", True)):
            continue
        try:
            order = int(item.get("panel_order") or 0)
        except Exception:
            order = 0
        if order <= 0:
            continue
        text = clean_ocr_text(
            str(item.get("repaired_text") or item.get("text") or item.get("cleaned_text") or "")
        ).strip()
        if not text or not is_usable_ocr_text(text):
            continue
        key = re.sub(r"\W+", " ", text.casefold()).strip()
        if not key:
            continue
        seen = seen_by_order.setdefault(order, set())
        if key in seen:
            continue
        seen.add(key)
        record = merged_by_order.setdefault(
            order,
            {
                "panel_id": str(item.get("panel_id") or "").strip(),
                "panel_order": order,
                "dialogue_text": "",
                "text_english": "",
                "source": "transcript",
                "regions": [],
            },
        )
        if not record.get("panel_id") and str(item.get("panel_id") or "").strip():
            record["panel_id"] = str(item.get("panel_id") or "").strip()
        record["dialogue_text"] = " ".join(part for part in [record.get("dialogue_text", ""), text] if part).strip()
        record["text_english"] = record["dialogue_text"]
        record.setdefault("regions", []).append(
            {
                "bbox": item.get("bbox"),
                "text_english": text,
                "text_original": str(item.get("raw_text") or item.get("text") or text),
                "confidence": item.get("confidence"),
                "detector": item.get("detector") or item.get("source") or "transcript",
                "ocr_engine": item.get("backend") or item.get("ocr_engine") or "transcript",
            }
        )
    return list(merged_by_order.values())


def _merge_panel_evidence_records(primary: list[dict[str, Any]] | None, secondary: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    merged: dict[int | str, dict[str, Any]] = {}

    def key_for(item: dict[str, Any]) -> int | str:
        try:
            order = int(item.get("panel_order") or 0)
        except Exception:
            order = 0
        if order:
            return order
        return str(item.get("panel_id") or "").strip()

    for records in (primary or [], secondary or []):
        for source in records:
            if not isinstance(source, dict):
                continue
            key = key_for(source)
            if not key:
                continue
            if key not in merged:
                merged[key] = dict(source)
                continue
            current = merged[key]
            for field in ("dialogue_text", "text_english", "caption_text", "text_original"):
                incoming = str(source.get(field) or "").strip()
                existing = str(current.get(field) or "").strip()
                if incoming and incoming.casefold() not in existing.casefold():
                    current[field] = " ".join(part for part in [existing, incoming] if part).strip()
            if not current.get("panel_id") and source.get("panel_id"):
                current["panel_id"] = source.get("panel_id")
            regions = list(current.get("regions") or [])
            regions.extend(source.get("regions") or [])
            current["regions"] = regions
    return list(merged.values())


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


def _panel_vision_record_ids(records: list[PanelVisionRecord] | None) -> set[str]:
    return {str(record.panel_id) for record in records or [] if str(record.panel_id or "").strip()}


def _script_evidence_snapshot(
    kept_panels: list,
    panel_vision_records: list[PanelVisionRecord] | None,
) -> dict[str, float | int | bool]:
    kept_count = len(kept_panels)
    usable_text_count = sum(
        1
        for panel in kept_panels
        if is_usable_ocr_text(clean_ocr_text(str(getattr(panel, "ocr_text", "") or "")))
    )
    vision_ids = _panel_vision_record_ids(panel_vision_records)
    vision_count = sum(1 for panel in kept_panels if str(getattr(panel, "id", "")) in vision_ids)
    return {
        "kept_count": kept_count,
        "usable_text_count": usable_text_count,
        "usable_text_ratio": usable_text_count / max(kept_count, 1),
        "vision_count": vision_count,
        "vision_ratio": vision_count / max(kept_count, 1),
        "has_panel_vision": bool(vision_count),
    }


def _should_defer_script_for_vision(evidence: dict[str, float | int | bool]) -> bool:
    kept_count = int(evidence.get("kept_count") or 0)
    if kept_count <= 0:
        return False
    usable_text_ratio = float(evidence.get("usable_text_ratio") or 0.0)
    vision_ratio = float(evidence.get("vision_ratio") or 0.0)
    # OCR-light comics need visual evidence before a recap script can be
    # grounded. The thresholds are intentionally proportional so this applies
    # to short tests, long manga batches, and webtoon chapters without naming
    # any specific series.
    return usable_text_ratio < 0.25 and vision_ratio < 0.80


def _defer_script_until_vision(
    context: PipelineContext,
    *,
    project_dir: Path,
    panel_vision_records: list[PanelVisionRecord] | None,
    force_refresh: bool,
    reason: str,
) -> None:
    canonical_path = project_dir / "output" / "canonical_characters.json"
    vision_final_path = project_dir / "output" / "panel_vision_final.json"
    if not canonical_path.exists():
        next_stage = PipelineStage.CHARACTER_PORTRAIT
        message = f"{reason} Queued character portraits before script generation."
    elif not vision_final_path.exists() or not panel_vision_records:
        next_stage = PipelineStage.PANEL_VISION_EXTRACTION
        message = f"{reason} Queued panel vision before script generation."
    else:
        next_stage = PipelineStage.PANEL_VISION_QUALITY
        message = f"{reason} Queued panel vision quality before script generation."

    current_job = context.store.get_job(context.project_id, context.job_id)
    payload = {
        **dict(current_job.payload or {}),
        "continue_to_script_generation": True,
        "force_refresh": force_refresh,
    }
    queue_stage_once(
        context.store,
        context.queue,
        context.project_id,
        next_stage,
        message,
        payload=payload,
    )
    context.store.update_job(
        context.project_id,
        context.job_id,
        status=JobStatus.COMPLETED.value,
        progress=100,
        finished_at=context.store._now().isoformat(),
        message=message,
    )
    context.store.update_stage_state(
        context.project_id,
        PipelineStage.SCRIPT_GENERATION,
        StageStatus.READY,
        progress=0,
        message="Waiting for panel vision evidence before generating the script.",
    )
    context.queue.clear_cancel(context.job_id)


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


def _panel_vision_text_for_segment(segment: Any, panel_vision_records: list[PanelVisionRecord] | None) -> str:
    record_by_id = {str(record.panel_id): record for record in panel_vision_records or [] if str(record.panel_id or "").strip()}
    parts: list[str] = []
    for index, panel_id in enumerate(getattr(segment, "panel_ids", []) or [], start=1):
        record = record_by_id.get(str(panel_id))
        if record is None:
            continue
        evidence_bits = []
        speaker = str(getattr(record, "speaker", "") or "").strip()
        if speaker and speaker.casefold() != "unknown":
            evidence_bits.append(f"speaker: {speaker}")
        character_names = [
            str(name).strip()
            for name in getattr(record, "character_names", []) or []
            if str(name).strip()
        ][:4]
        if character_names:
            evidence_bits.append(f"characters: {', '.join(character_names)}")
        for label, value in (
            ("action", getattr(record, "action_beat", "")),
            ("dialogue", getattr(record, "dialogue", "")),
            ("caption", getattr(record, "caption", "")),
            ("emotion", getattr(record, "emotion", "")),
        ):
            text = str(value or "").strip()
            if text:
                evidence_bits.append(f"{label}: {text}")
        if evidence_bits:
            parts.append(f"panel {index}: " + "; ".join(evidence_bits))
    return "\n".join(parts)


def _story_quality_report_for_segments(
    segments: list[Any],
    panel_vision_records: list[PanelVisionRecord] | None,
) -> dict[str, Any]:
    records = [
        record.model_dump() if hasattr(record, "model_dump") else dict(record)
        for record in panel_vision_records or []
    ]
    return ScriptQualityService().analyze_story_segments(segments, panel_vision_records=records)


def _youtube_rewrite_min_count(total: int) -> int:
    if total <= 0:
        return 0
    # Final rewrite should be close to complete. Partial model responses were
    # the main mini-test failure mode, so leave only a tiny allowance for huge
    # projects where one or two conservative unchanged segments are acceptable.
    return max(1, min(total, round(total * 0.92)))


def _youtube_candidate_is_safe(
    *,
    current_report: dict[str, Any],
    candidate_report: dict[str, Any],
) -> bool:
    hard_failure_counts = (
        int(candidate_report.get("malformed_lines") or 0),
        int(candidate_report.get("visual_lines") or 0),
        int(candidate_report.get("ocr_contamination_lines") or 0),
        int(candidate_report.get("first_person_lines") or 0),
    )
    if any(count > 0 for count in hard_failure_counts):
        return False
    candidate_score = int(candidate_report.get("quality_score") or 0)
    current_score = int(current_report.get("quality_score") or 0)
    if candidate_score < 74:
        return False
    if (
        not bool(current_report.get("should_block_tts"))
        and bool(candidate_report.get("should_block_tts"))
    ):
        return False
    if current_score and candidate_score < current_score - 4:
        return False
    return True


def _rewrite_blocked_story_segments_once(
    *,
    store: ProjectStore,
    project_id: str,
    project_title: str,
    chapter_summary: str,
    panel_vision_records: list[PanelVisionRecord] | None,
    quality_report: dict[str, Any],
    router: LLMRouter,
) -> bool:
    risky_by_order = {
        int(item.get("order") or 0): item
        for item in quality_report.get("risky_segments", []) or []
        if isinstance(item, dict)
    }
    segments = store.load_story_segments(project_id)
    payload: list[dict[str, Any]] = []
    for index, segment in enumerate(segments):
        risky = risky_by_order.get(int(segment.order))
        text = str(segment.text or "").strip()
        if not risky and text:
            continue
        local_evidence = _panel_vision_text_for_segment(segment, panel_vision_records)
        payload.append(
            {
                "index": index,
                "segment_id": segment.id,
                "panel_start": segment.panel_start or 0,
                "panel_end": segment.panel_end or 0,
                "panel_count": len(segment.panel_ids or []),
                "current": text,
                "risk_reasons": list((risky or {}).get("reasons") or []),
                "local_evidence": local_evidence,
                "previous_line": str(segments[index - 1].text or "").strip() if index > 0 else "",
                "next_line": str(segments[index + 1].text or "").strip() if index + 1 < len(segments) else "",
            }
        )
    if not payload:
        return False
    result = asyncio.run(
        router.rewrite_blocked_story_segments(
            payload,
            {
                "project_title": project_title,
                "chapter_summary": chapter_summary,
            },
            provider="gemini",
        )
    )
    rewrites: dict[int, str] = {}
    for item in result.payload.get("rewrites", []) or []:
        if not isinstance(item, dict):
            continue
        try:
            rewrite_index = int(item.get("index"))
        except (TypeError, ValueError):
            continue
        rewrites[rewrite_index] = str(item.get("line") or item.get("text") or "").strip()
    changed = False
    updated = []
    for index, segment in enumerate(segments):
        if index not in rewrites:
            updated.append(segment)
            continue
        line = clean_ocr_text(rewrites[index])
        if line and line != str(segment.text or "").strip():
            updated.append(segment.model_copy(update={"text": line, "visual_only": False, "suppression_reason": None}))
            changed = True
        elif not line and str(segment.text or "").strip():
            updated.append(segment.model_copy(update={"text": "", "visual_only": True, "suppression_reason": "quality_rewrite_rejected"}))
            changed = True
        else:
            updated.append(segment)
    if changed:
        store.save_story_segments(project_id, updated, story_block="\n\n".join(segment.text.strip() for segment in updated if segment.text.strip()))
    return changed


def _rewrite_story_segments_for_youtube_once(
    *,
    store: ProjectStore,
    project_id: str,
    project_title: str,
    chapter_summary: str,
    panel_vision_records: list[PanelVisionRecord] | None,
    canonical_characters: list[CanonicalCharacterRecord] | None,
    router: LLMRouter,
) -> bool:
    segments = store.load_story_segments(project_id)
    if not segments:
        return False
    roster = [
        {
            "name": str(character.name or "").strip(),
            "role": str(character.role or "").strip(),
            "aliases": list(character.aliases or []),
        }
        for character in canonical_characters or []
        if str(character.name or "").strip()
    ]
    payload: list[dict[str, Any]] = []
    for index, segment in enumerate(segments):
        payload.append(
            {
                "index": index,
                "segment_id": segment.id,
                "scene_id": segment.scene_id or 0,
                "order": segment.order,
                "panel_start": segment.panel_start or 0,
                "panel_end": segment.panel_end or 0,
                "panel_count": len(segment.panel_ids or []),
                "current": str(segment.text or "").strip(),
                "local_evidence": _panel_vision_text_for_segment(segment, panel_vision_records),
                "previous_line": str(segments[index - 1].text or "").strip() if index > 0 else "",
                "next_line": str(segments[index + 1].text or "").strip() if index + 1 < len(segments) else "",
            }
        )
    rewrite_context = {
        "project_title": project_title,
        "chapter_summary": chapter_summary,
        "character_roster": roster,
    }

    id_to_index = {str(segment.id): index for index, segment in enumerate(segments)}
    rewrites: dict[int, str] = {}
    try:
        generated = asyncio.run(
            router.generate_youtube_recap_segments(payload, rewrite_context, provider="gemini")
        )
        for item in generated.payload.get("segments", []) or []:
            if not isinstance(item, dict):
                continue
            segment_id = str(item.get("segment_id") or "").strip()
            text = clean_ocr_text(str(item.get("text") or "").strip())
            rewrite_index = id_to_index.get(segment_id)
            if rewrite_index is not None and text:
                rewrites[rewrite_index] = text
    except Exception as exc:
        logger.warning("YouTube recap segment generation failed for %s: %s", project_id, exc)

    min_rewrite_count = _youtube_rewrite_min_count(len(segments))
    if rewrites and len(rewrites) < min_rewrite_count:
        logger.warning(
            "YouTube recap segment generation for %s returned only %d/%d usable segments; filling gaps with rewrites",
            project_id,
            len(rewrites),
            len(segments),
        )

    def _call_rewriter(items: list[dict[str, Any]]) -> dict[int, str]:
        result = asyncio.run(router.rewrite_story_segments_for_youtube(items, rewrite_context, provider="gemini"))
        chunk_rewrites: dict[int, str] = {}
        for item in result.payload.get("rewrites", []) or []:
            if not isinstance(item, dict):
                continue
            try:
                rewrite_index = int(item.get("index"))
            except (TypeError, ValueError):
                continue
            chunk_rewrites[rewrite_index] = clean_ocr_text(str(item.get("line") or item.get("text") or "").strip())
        return chunk_rewrites

    if len(rewrites) < min_rewrite_count:
        missing_payload = [
            item
            for item in payload
            if int(item.get("index") or 0) not in rewrites
            or not str(rewrites.get(int(item.get("index") or 0)) or "").strip()
        ]
        if len(missing_payload) != len(payload):
            for start in range(0, len(missing_payload), 4):
                chunk = missing_payload[start:start + 4]
                try:
                    rewrites.update(_call_rewriter(chunk))
                except Exception as exc:
                    logger.warning("YouTube missing-rewrite chunk %d failed for %s: %s", start // 4 + 1, project_id, exc)
    if len(rewrites) < min_rewrite_count:
        try:
            rewrites.update(_call_rewriter(payload))
        except Exception as exc:
            logger.warning("Full YouTube rewrite failed for %s: %s", project_id, exc)
    if len(rewrites) < min_rewrite_count:
        logger.warning(
            "YouTube rewrite for %s returned only %d/%d entries; retrying in chunks",
            project_id,
            len(rewrites),
            len(segments),
        )
        rewrites = {}
        for start in range(0, len(payload), 4):
            chunk = payload[start:start + 4]
            try:
                rewrites.update(_call_rewriter(chunk))
            except Exception as exc:
                logger.warning("YouTube rewrite chunk %d failed for %s: %s", start // 4 + 1, project_id, exc)
    missing_after_chunks = [
        item
        for item in payload
        if int(item.get("index") or 0) not in rewrites
        or not str(rewrites.get(int(item.get("index") or 0)) or "").strip()
    ]
    if len(rewrites) < min_rewrite_count and (len(payload) <= 60 or len(missing_after_chunks) <= 12):
        logger.warning(
            "YouTube rewrite for %s still returned only %d/%d entries; retrying segment-by-segment",
            project_id,
            len(rewrites),
            len(segments),
        )
        for item in missing_after_chunks:
            rewrite_index = int(item.get("index") or 0)
            if rewrite_index in rewrites and rewrites[rewrite_index].strip():
                continue
            try:
                rewrites.update(_call_rewriter([item]))
            except Exception as exc:
                logger.warning("YouTube rewrite single index %d failed for %s: %s", rewrite_index, project_id, exc)
    if len(rewrites) < min_rewrite_count:
        logger.warning(
            "Rejected YouTube rewrite for %s: only %d/%d segment rewrites returned",
            project_id,
            len(rewrites),
            len(segments),
        )
        return False
    changed = False
    updated = []
    for index, segment in enumerate(segments):
        line = rewrites.get(index, str(segment.text or "").strip())
        if line and line != str(segment.text or "").strip():
            updated.append(segment.model_copy(update={"text": line, "visual_only": False, "suppression_reason": None}))
            changed = True
        elif not line and str(segment.text or "").strip():
            updated.append(segment.model_copy(update={"text": "", "visual_only": True, "suppression_reason": "youtube_rewrite_rejected"}))
            changed = True
        else:
            updated.append(segment)
    if changed:
        current_report = _story_quality_report_for_segments(segments, panel_vision_records)
        candidate_report = _story_quality_report_for_segments(updated, panel_vision_records)
        if not _youtube_candidate_is_safe(
            current_report=current_report,
            candidate_report=candidate_report,
        ):
            logger.warning(
                "Rejected YouTube rewrite for %s after quality check: current=%s candidate=%s",
                project_id,
                current_report.get("summary"),
                candidate_report.get("summary"),
            )
            return False
    if changed:
        store.save_story_segments(
            project_id,
            updated,
            story_block="\n\n".join(segment.text.strip() for segment in updated if segment.text.strip()),
        )
    return changed


def _run_script_generation_vision(
    context: PipelineContext,
    project: Any,
    project_dir: Path,
    job: Any,
) -> None:
    """Vision-grounded script generation. The new pipeline path.

    Sends every kept panel image to Gemini Vision in visual reading order,
    with rolling continuity context, and writes a single canonical
    script_manifest.json + mirrors to panels.json + script.json + script.txt.

    No polish/repair cascade. Panels that fail are flagged for in-place
    regeneration in the UI - they are never silently filled with garbage.
    """
    import asyncio
    import json as _json

    from app.services.cast_bible_service import CastBibleService
    from app.services.panel_vision_narrator import (
        PanelVisionNarrator,
        panels_from_store,
        write_narration_outputs,
    )

    store = context.store
    panels_path = project_dir / "panels.json"
    if not panels_path.exists():
        raise RuntimeError(f"panels.json missing for project {context.project_id}")

    panels_json = _json.loads(panels_path.read_text(encoding="utf-8"))
    # Two character-hint signals feed character_hints, BEFORE the vision
    # narrator runs - so every per-panel Gemini Vision call already knows
    # who is on-screen, dramatically improving naming.
    #
    #   Signal A: cast bible name list -> word-boundary scan over OCR text
    #             per panel ("hiro!" mention -> hint=Hiro). Cheap, no
    #             extra API calls.
    #   Signal B: anime face detection -> CLIP clustering of all detected
    #             faces -> ONE Gemini call to label each cluster with a
    #             cast name -> per-panel hint backfill from cluster
    #             membership. Single CLIP load, one Gemini call total,
    #             yields hints for the ~50-80% of panels containing a
    #             detectable face. Service is idempotent + cached.
    cast_names_for_hints: list[str] = []
    cached_bible_for_id = None
    try:
        from app.services.cast_bible_service import CastBibleService as _CastSvc
        cached_bible_for_id = _CastSvc().load_cached(project_dir)
        if cached_bible_for_id and cached_bible_for_id.members:
            cast_names_for_hints = [m.name for m in cached_bible_for_id.members if m.name]
    except Exception:
        pass

    panel_hint_index: dict[str, list[str]] = {}
    portrait_lookup: dict[str, Path] = {}
    try:
        if cast_names_for_hints:
            context.progress(3, "Identifying characters across panels (face cluster + Gemini)")
            from app.services.character_identifier_service import (
                build_character_identity, load_panel_hint_index,
                load_character_portraits,
            )
            build_character_identity(
                project_dir,
                bible=cached_bible_for_id,
                cancel_callback=context.ensure_not_cancelled,
            )
            panel_hint_index = load_panel_hint_index(project_dir)
            portrait_lookup = load_character_portraits(project_dir)
            logger.info(
                "Character identity index: %d panels with face-based hints, %d portraits available",
                len(panel_hint_index), len(portrait_lookup),
            )
    except Exception as exc:  # noqa: BLE001
        logger.warning("Character identifier skipped (non-fatal): %s", exc)

    panel_inputs = panels_from_store(
        project_dir,
        panels_json,
        cast_member_names=cast_names_for_hints,
        panel_hint_index=panel_hint_index,
    )
    if not panel_inputs:
        context.fail("No kept panels to narrate.")
        return

    # ── Build / load the cast bible BEFORE narrating ─────────────────────
    # One Gemini call per project, cached at output/cast_bible.json. For
    # popular series the model already knows the cast, so we get reliable
    # character names ("Zero Two") instead of generic descriptions
    # ("a pink-haired girl"). For obscure series the bible is empty and
    # the narrator falls back to its existing generic-description prompt.
    context.progress(2, "Looking up cast for character names")
    chapter_meta = getattr(project, "chapter_metadata", None)
    manga_title = getattr(chapter_meta, "manga_title", None) or project.name
    chapter_title = getattr(chapter_meta, "chapter_title", None) or ""
    try:
        cast_service = CastBibleService()
        bible = cast_service.ensure_bible(
            project_dir,
            manga_title=manga_title or "(unknown)",
            chapter_title=chapter_title or "(unknown)",
        )
        cast_block = CastBibleService.format_for_prompt(bible)
        if bible and bible.members:
            logger.info(
                "Cast bible loaded for %s: %d characters (%s)",
                context.project_id, len(bible.members), bible.source,
            )
    except Exception as exc:  # noqa: BLE001
        logger.warning("Cast bible step failed (continuing without): %s", exc)
        cast_block = ""

    context.progress(5, f"Starting vision narration for {len(panel_inputs)} panels")
    narrator = PanelVisionNarrator()

    def _progress(pct: float, msg: str) -> None:
        # Reserve 5%-95% of the stage bar for the actual narration loop.
        scaled = 5.0 + (pct * 0.9)
        context.progress(scaled, msg)

    batch = asyncio.run(
        narrator.narrate_chapter(
            panel_inputs,
            cast_block=cast_block,
            portrait_lookup=portrait_lookup,
            progress_callback=_progress,
            cancel_callback=context.ensure_not_cancelled,
        )
    )

    context.progress(95, "Persisting narration outputs")
    summary = write_narration_outputs(
        project_dir, panel_inputs, batch.results, panels_json
    )

    # ── Auto-polish the opening 20 lines ─────────────────────────────────
    # The first 20 lines = first ~3 minutes of audio = the make-or-break
    # retention window on YouTube. The per-panel vision narrator does an
    # okay job but tends to write panel descriptions ("a uniformed
    # character with a reflective visor holds a small orb") for the early
    # set-up panels where the cast isn't visible yet. This step rewrites
    # those into cinematic recap narration ("In a dying world, humanity
    # hides inside moving fortresses called Plantations") - the same
    # transformation backend/scripts/polish_opening_lines.py did as a
    # one-off, now wired into the pipeline so every project gets it
    # automatically. Non-fatal: failures fall through and ship the
    # un-polished version.
    try:
        context.progress(96, "Polishing opening narration lines")
        from app.services.opening_polish_service import polish_opening_narration
        polish_opening_narration(
            project_dir,
            cast_block=cast_block,
            manga_title=manga_title,
            chapter_title=chapter_title,
            line_count=20,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("Opening-lines polish skipped (non-fatal): %s", exc)

    # ── Name resolution pass ─────────────────────────────────────────────
    # User rule: "if a character's name has never been said then and only
    # then can their description be used." The per-panel vision narrator
    # sometimes falls back to "the boy with dark hair" when the cast
    # bible clearly identifies that boy as Hiro. This pass scans the
    # entire script with the bible in context and rewrites every such
    # descriptor back to the cast name, in batches of 80 lines for
    # rolling continuity. Non-fatal: if Gemini hiccups we just keep
    # the original descriptors.
    try:
        if cast_block:
            context.progress(97, "Resolving character names across the script")
            from app.services.name_resolution_service import resolve_character_names
            report = resolve_character_names(
                project_dir, cast_block=cast_block
            )
            if report.get("updated", 0) > 0:
                logger.info(
                    "Name resolution: %d lines rewritten across %d batches",
                    report.get("updated"), report.get("batches", 0),
                )
    except Exception as exc:  # noqa: BLE001
        logger.warning("Name-resolution pass skipped (non-fatal): %s", exc)

    # Post-write consistency check - catches any panels↔script drift before
    # we hand off to TTS. Failing fast here beats discovering a desync after
    # audio has been generated.
    try:
        from app.services.script_consistency_check import (
            check_project, format_report,
        )
        report = check_project(project_dir, project_id=context.project_id)
        if report.issues:
            logger.warning(
                "Script consistency report for %s:\n%s",
                context.project_id,
                format_report(report),
            )
        if report.has_errors:
            error_messages = "; ".join(
                f"{i.code}: {i.message}" for i in report.issues if i.severity == "error"
            )[:400]
            raise RuntimeError(
                f"Script consistency check failed: {error_messages}"
            )
    except RuntimeError:
        raise
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "Consistency check could not run for %s: %s",
            context.project_id, exc,
        )

    store.update_stage_state(
        context.project_id,
        PipelineStage.NARRATION_GENERATION,
        StageStatus.READY,
        progress=0,
        message=(
            f"Vision script ready: {batch.successful}/{len(panel_inputs)} narrations. "
            f"{summary['panels_needing_review']} panels flagged for review. "
            f"Generate audio when ready."
        ),
    )

    logger.info(
        "Vision script generation for %s done in %.1fs: %d ok / %d need review",
        context.project_id,
        batch.elapsed_seconds,
        batch.successful,
        batch.failed,
    )
    context.complete(
        f"Vision script done: {batch.successful}/{len(panel_inputs)} panels narrated "
        f"({batch.elapsed_seconds:.0f}s, {summary['panels_needing_review']} need review)"
    )


def _run_script_generation_vnext(
    context: PipelineContext,
    project: Any,
    project_dir: Path,
    job: Any,
) -> None:
    stop_after_stage = bool(job.payload.get("stop_after_stage"))
    max_cost_usd = float(job.payload.get("max_cost_usd") or job.payload.get("vnext_redraft_max_cost_usd") or 0.0)
    redraft_config = ScriptVNextRedraftConfig(
        enabled=bool(job.payload.get("vnext_redraft_enabled")) or max_cost_usd > 0,
        dry_run=bool(job.payload.get("vnext_redraft_dry_run")),
        max_calls=int(job.payload.get("vnext_redraft_max_calls") or 4),
        max_scenes_per_batch=int(job.payload.get("vnext_redraft_max_scenes_per_batch") or 4),
        max_prompt_chars=int(job.payload.get("vnext_redraft_max_prompt_chars") or 12000),
        max_output_tokens=int(job.payload.get("vnext_redraft_max_output_tokens") or 1800),
        max_estimated_cost_usd=max_cost_usd,
        style_threshold=int(job.payload.get("vnext_redraft_style_threshold") or 68),
    )
    context.progress(8, "Planning chronological vNext scenes from existing artifacts")
    service = ScriptGenerationVNextService()
    result = service.run(
        project_id=context.project_id,
        project_name=project.name or "",
        project_dir=project_dir,
        chapter_metadata=project.chapter_metadata,
        panels=project.panels,
        job_id=job.id,
        max_cost_usd=max_cost_usd,
        redraft_config=redraft_config,
    )
    context.ensure_not_cancelled()
    context.progress(88, "Saving vNext scene-level script artifacts")
    if stop_after_stage:
        context.complete("vNext side-by-side script artifacts ready")
        return

    context.store.save_story_segments(
        context.project_id,
        result.story_segments,
        story_block=result.story_text,
        job_id=job.id,
    )
    output_dir = project_dir / "output"
    write_json(output_dir / "scene_plan.json", result.scene_plan)
    write_json(output_dir / "narration_chunks.json", result.narration_chunks)
    write_json(output_dir / "qc_report.json", result.qc_report)
    write_json(output_dir / "script_quality.json", result.qc_report)
    write_json(output_dir / "cost_report.json", result.cost_report)
    write_json(output_dir / "benchmark_report.json", {
        "script_pipeline_version": "vNext",
        "quality_score": result.qc_report.get("quality_score"),
        "should_block_tts": result.qc_report.get("should_block_tts"),
        "meaningful_panel_usage_rate": result.qc_report.get("meaningful_panel_usage_rate"),
        "long_no_tts_gap_count": result.qc_report.get("long_no_tts_gap_count"),
        "estimated_cost_usd": result.cost_report.get("estimated_cost_usd"),
        "gemini_calls_total": result.cost_report.get("gemini_calls_total"),
    })
    (output_dir / "final_script.md").write_text(result.story_text.strip() + ("\n" if result.story_text.strip() else ""), encoding="utf-8")

    if bool(result.qc_report.get("should_block_tts")):
        context.store.update_stage_state(
            context.project_id,
            PipelineStage.NARRATION_GENERATION,
            StageStatus.NEEDS_REVIEW,
            progress=0,
            message=str(result.qc_report.get("summary") or "vNext QC blocked audio generation."),
        )
        context.complete("vNext script needs review before audio")
        return

    context.store.update_stage_state(
        context.project_id,
        PipelineStage.NARRATION_GENERATION,
        StageStatus.READY,
        progress=0,
        message="vNext script ready. Generate audio when you want to continue." if stop_after_stage else "Starting narration audio automatically",
    )
    context.complete("vNext narration script ready")
    if not stop_after_stage:
        queue_stage_once(
            context.store,
            context.queue,
            context.project_id,
            PipelineStage.NARRATION_GENERATION,
            "Queued automatically after vNext script generation",
        )


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
    configured_script_version = str(
        job.payload.get("script_pipeline_version")
        or getattr(project.pipeline_config, "script_pipeline_version", "legacy")
        or "legacy"
    ).strip()
    version_lower = configured_script_version.casefold()
    if version_lower == "vision":
        _run_script_generation_vision(context, project, project_dir, job)
        return
    if version_lower == "vnext":
        _run_script_generation_vnext(context, project, project_dir, job)
        return

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
    panel_evidence_records = None
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
        force_refresh=force_refresh or bool(job.payload.get("refresh_dialogue_context")),
        progress_callback=lambda progress, message: context.progress(_script_dialogue_progress(progress, message), message),
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
    # extraction run, load it now so story script generation gets rich
    # per-panel evidence (action_beat, dialogue, caption, visual_cues).
    # This runs non-fatally - if the file is missing or corrupt, panel
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
                    "Loaded %d panel vision records for %s (story-script evidence enrichment)",
                    len(panel_vision_records),
                    context.project_id,
                )
        except Exception as exc:
            logger.warning("Could not load panel_vision_final.json (non-fatal): %s", exc)
    try:
        panel_evidence_records = load_panel_evidence_records(project_dir)
        if panel_evidence_records:
            logger.info(
                "Loaded %d panel evidence records for %s (story-script OCR enrichment)",
                len(panel_evidence_records),
                context.project_id,
            )
    except Exception as exc:
        logger.warning("Could not load panel_evidence.json (non-fatal): %s", exc)
    had_panel_evidence_records = bool(panel_evidence_records)
    transcript_evidence_records = _transcript_evidence_records(project_dir)
    if transcript_evidence_records:
        panel_evidence_records = _merge_panel_evidence_records(panel_evidence_records, transcript_evidence_records)
        logger.info(
            "Merged %d transcript evidence records into story-script evidence for %s",
            len(transcript_evidence_records),
            context.project_id,
        )
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
    transcript_by_order = _transcript_lines_by_panel_order(project_dir)
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
        elif transcript_by_order.get(int(panel.order)):
            detected_text = " ".join(transcript_by_order[int(panel.order)]).strip()
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

    if not had_panel_evidence_records:
        context.progress(34, "Extracting clean panel text evidence")
        evidence_extractor = PanelEvidenceExtractor()
        panel_evidence_records = evidence_extractor.run(
            project_dir=project_dir,
            page_paths=page_paths,
            panels=updated_panels,
            chapter_metadata=project.chapter_metadata,
            force_refresh=force_refresh or bool(job.payload.get("refresh_panel_evidence")),
            allow_crop_ocr=bool(job.payload.get("deep_panel_evidence_scan")),
            allow_apple_vision=bool(job.payload.get("apple_vision_panel_evidence")),
            allow_metadata_ocr=bool(job.payload.get("metadata_panel_evidence")),
            progress_callback=lambda progress, message: context.progress(34 + progress * 0.02, message),
            cancel_callback=context.ensure_not_cancelled,
        )
        context.ensure_not_cancelled()
        transcript_evidence_records = _transcript_evidence_records(project_dir)
        if transcript_evidence_records:
            panel_evidence_records = _merge_panel_evidence_records(panel_evidence_records, transcript_evidence_records)

    evidence_snapshot = _script_evidence_snapshot(
        [panel for panel in updated_panels if bool(getattr(panel, "keep", False))],
        panel_vision_records,
    )
    if (
        not bool(job.payload.get("allow_sparse_script"))
        and _should_defer_script_for_vision(evidence_snapshot)
    ):
        logger.info(
            "Deferring script generation for %s until vision evidence is available: %s",
            context.project_id,
            evidence_snapshot,
        )
        _defer_script_until_vision(
            context,
            project_dir=project_dir,
            panel_vision_records=panel_vision_records,
            force_refresh=force_refresh,
            reason=(
                "Script evidence is too sparse "
                f"({int(evidence_snapshot['usable_text_count'])}/{int(evidence_snapshot['kept_count'])} kept panels have usable OCR; "
                f"{int(evidence_snapshot['vision_count'])}/{int(evidence_snapshot['kept_count'])} have panel vision)."
            ),
        )
        return

    # ── Step 1b: Character visual profiling ────────────────────────
    # Enrich character_dictionary with appearance descriptions so the
    # LLM can identify characters visually in panels with no OCR text.
    context.progress(36, "Profiling character appearances")
    kept_panel_count = sum(1 for panel in updated_panels if bool(getattr(panel, "keep", False)))
    if kept_panel_count > 240 and (panel_vision_records or canonical_characters):
        logger.info(
            "Skipping character visual profiling for %s (%d kept panels; existing vision/canonical evidence is available)",
            context.project_id,
            kept_panel_count,
        )
    else:
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
        panel_evidence_records=panel_evidence_records,
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
        job_id=job.id,
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

    if not bool(job.payload.get("skip_youtube_rewrite")):
        context.progress(98, "Rewriting for YouTube recap flow")
        try:
            changed = _rewrite_story_segments_for_youtube_once(
                store=store,
                project_id=context.project_id,
                project_title=project.name or "",
                chapter_summary=story_bundle.chapter_summary,
                panel_vision_records=panel_vision_records,
                canonical_characters=canonical_characters,
                router=router,
            )
            if changed:
                logger.info("Applied YouTube recap rewrite pass for %s", context.project_id)
        except Exception as exc:
            logger.warning("YouTube recap rewrite pass failed (non-fatal): %s", exc)

    # Name resolution pass: rewrite generic descriptors ("the boy",
    # "a student", "the woman") to canonical cast names. The vision
    # narrator emits names when it visually matches a reference portrait
    # but otherwise drops to descriptors; this pass uses Gemini to fix
    # those after the fact via 3 strategies (feature match, context
    # inference, speaker inference). Non-fatal: a quota / safety
    # failure leaves the script as-is and the run continues. Model
    # cascade inside the service handles per-model 429 fallback.
    if not bool(job.payload.get("skip_name_resolution")):
        context.progress(99, "Resolving generic descriptors to cast names")
        try:
            from app.services.name_resolution_service import resolve_character_names
            project_dir = store._project_dir(context.project_id)
            cast_block = _build_cast_block_for_name_resolution(canonical_characters)
            if cast_block:
                report = resolve_character_names(project_dir, cast_block=cast_block, batch_size=80)
                if report.get("updated"):
                    logger.info(
                        "Name resolution rewrote %d/%d lines across %d batches for %s",
                        report.get("updated"), report.get("total_lines"),
                        report.get("batches"), context.project_id,
                    )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Name resolution pass failed (non-fatal): %s", exc)

    quality_report = store.load_script_quality_report(context.project_id)
    if bool(quality_report.get("should_block_tts")) and not bool(job.payload.get("skip_quality_rewrite")):
        context.progress(99, "Rewriting blocked script beats from panel evidence")
        try:
            changed = _rewrite_blocked_story_segments_once(
                store=store,
                project_id=context.project_id,
                project_title=project.name or "",
                chapter_summary=story_bundle.chapter_summary,
                panel_vision_records=panel_vision_records,
                quality_report=quality_report,
                router=router,
            )
            if changed:
                quality_report = store.load_script_quality_report(context.project_id)
        except Exception as exc:
            logger.warning("Blocked-script rewrite pass failed (non-fatal): %s", exc)

    if bool(quality_report.get("should_block_tts")):
        summary = str(
            quality_report.get("summary")
            or "Script quality checks found problems that need review before audio generation."
        )
        store.update_stage_state(
            context.project_id,
            PipelineStage.NARRATION_GENERATION,
            StageStatus.NEEDS_REVIEW,
            progress=0,
            message=summary,
        )
        logger.info("Script generation completed but TTS remains blocked for %s: %s", context.project_id, summary)
        context.complete("Narration script needs review before audio")
        return

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
        # Auto-cascade safety net (mirrors panel stages). Pairs with
        # _resume_orphaned_cascades in the worker so a queue/redis hiccup
        # between context.complete and queue_stage_once doesn't strand
        # the project at "narration_generation: ready" forever.
        continue_auto_run_pipeline(store, context.queue, context.project_id, source="script generation")


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
    # The legacy ScriptQualityService gate was tuned for the old multi-pass
    # cascade; it false-positives heavily on vision-narrator output (which
    # writes panel-specific descriptions the gate flags as "caption-like").
    # Bypass it entirely for vision-mode projects - content quality is
    # already enforced by PanelVisionNarrator's per-panel post-process.
    pipeline_version = (
        getattr(project.pipeline_config, "script_pipeline_version", "legacy") or "legacy"
    ).strip().lower()
    if pipeline_version == "vision":
        force_bypass = True
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
    supported_character_names = _collect_supported_narration_names(project_dir, project)
    world_terms = _collect_narration_world_terms(project_dir, project)
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
        supported_character_names=supported_character_names,
        world_terms=world_terms,
        # Same rationale as the quality-gate bypass above: vision-mode
        # projects shouldn't be blocked by the legacy contamination gate.
        skip_contamination_guard=(pipeline_version == "vision"),
    )

    # Coverage validator: every narration_id we passed in must map to
    # *some* on-disk WAV via the audio manifest.
    #
    # The contamination guard inside generate_narration intentionally
    # merges adjacent units (deduping near-identical lines, fusing
    # sentence runs that belong together). So len(narration_ids) > len(
    # output_wavs) is BY DESIGN - what matters is whether every input
    # segment_id appears as a panel_id in the manifest, meaning some
    # WAV will play during its panel time at render. A "real" coverage
    # gap looks like: input segment_id never appears in the manifest,
    # AND no wav has its panel_id.
    audio_dir = Path(store._project_dir(context.project_id) / "audio")
    manifest_path = audio_dir / "manifest.json"
    represented_ids: set[str] = set()
    if manifest_path.exists():
        try:
            mf = read_json(manifest_path, default={})
            if isinstance(mf, dict):
                for entry in mf.values():
                    if isinstance(entry, dict):
                        pid = str(entry.get("panel_id") or "").strip()
                        if pid:
                            represented_ids.add(pid)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Narration coverage validator: could not read manifest: %s", exc)
    expected_ids = {str(pid).strip() for pid in narration_ids if str(pid).strip()}
    missing_ids = sorted(expected_ids - represented_ids) if represented_ids else []
    # Also flag empty / zero-byte WAVs so a broken file still surfaces.
    empty_wavs: list[str] = []
    for wav in sorted(audio_dir.glob("panel_*.wav")):
        if wav.stat().st_size < 1024:
            empty_wavs.append(wav.name)
    if missing_ids or empty_wavs:
        problem_count = len(missing_ids) + len(empty_wavs)
        sample_missing = missing_ids[:5]
        sample_empty = empty_wavs[:5]
        detail = (
            f"Narration coverage gap: "
            f"{len(missing_ids)} of {len(expected_ids)} input segments have no audio "
            f"(first few missing segment_ids: {sample_missing}), "
            f"and {len(empty_wavs)} WAV files are empty "
            f"(first few: {sample_empty}). "
            f"Retry the narration stage; do not proceed to render until coverage is complete."
        )
        logger.error(detail)
        raise RuntimeError(detail)

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
    # Auto-cascade safety net (mirrors the panel stages). If queue_stage_once
    # silently no-ops (e.g. a stale active-job lookup), this still triggers
    # the next forward stage based on stage_states.
    continue_auto_run_pipeline(store, context.queue, context.project_id, source="narration generation")


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

    # Audio coverage guard: defense-in-depth match to the one in
    # run_narration_generation. Uses manifest-based coverage (segment_id
    # -> wav mapping) because the narration engine intentionally merges
    # adjacent units, so the wav count is < segment count by design.
    audio_dir = Path(store._project_dir(context.project_id) / "audio")
    manifest_path = audio_dir / "manifest.json"
    text_segments = [s for s in story_segments if str(getattr(s, "text", "") or "").strip()]
    expected_ids = {str(getattr(s, "id", "") or "").strip() for s in text_segments}
    expected_ids.discard("")
    represented_ids: set[str] = set()
    if manifest_path.exists():
        try:
            mf = read_json(manifest_path, default={})
            if isinstance(mf, dict):
                for entry in mf.values():
                    if isinstance(entry, dict):
                        pid = str(entry.get("panel_id") or "").strip()
                        if pid:
                            represented_ids.add(pid)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Pre-render coverage guard: could not read manifest: %s", exc)
    # Only enforce coverage if we actually have a manifest to compare
    # against (otherwise an old legacy project pre-manifest gets
    # blocked from rendering). When manifest exists, every text segment
    # must be represented as a panel_id in some wav entry.
    if represented_ids:
        missing_ids = sorted(expected_ids - represented_ids)
        if missing_ids:
            detail = (
                f"Cannot render: {len(missing_ids)} of {len(expected_ids)} narrated "
                f"segments have no audio in the manifest "
                f"(first few segment_ids: {missing_ids[:5]}). "
                f"Re-run narration_generation, then retry the render."
            )
            logger.error(detail)
            raise RuntimeError(detail)

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
    # Auto-cascade safety net: video_rendering used to leave the bundle
    # stage at PENDING/READY with no active job, requiring the cascade
    # sweeper to catch up. Trigger the next stage inline so auto-run
    # projects flow straight into bundle generation.
    continue_auto_run_pipeline(store, context.queue, context.project_id, source="video rendering")


def run_youtube_bundle(context: PipelineContext) -> None:
    """Build the YouTube publish bundle (title + description + thumbnail).

    Runs after the video has rendered. Output goes to
    `<project>/youtube_bundle/` with title.txt, description.md,
    thumbnail.png, thumbnail_source.png, and manifest.json.
    """
    import json as _json
    from app.services.youtube_bundle_service import YouTubeBundleService

    store = context.store
    project = store.get_project(context.project_id)
    project_dir = store._project_dir(context.project_id)

    panels_path = project_dir / "panels.json"
    if not panels_path.exists():
        raise RuntimeError("panels.json is missing - cannot pick a thumbnail.")
    panels_json = _json.loads(panels_path.read_text(encoding="utf-8"))
    if not any(p.get("keep") for p in panels_json):
        raise RuntimeError("No kept panels - nothing to thumbnail.")

    script_lines = list(project.script_lines or store.load_script(context.project_id))
    if not script_lines:
        raise RuntimeError("No script is available - generate a script first.")

    context.start("Generating your YouTube bundle")

    # Pull the audio manifest so the bundle service can plan chapter
    # markers from real per-panel durations.
    audio_manifest_path = project_dir / "audio" / "manifest.json"
    audio_manifest: dict | None = None
    if audio_manifest_path.exists():
        try:
            audio_manifest = _json.loads(audio_manifest_path.read_text(encoding="utf-8"))
        except Exception:
            audio_manifest = None

    # Locate the main rendered video so the finishing renderer can
    # prepend the cold open and append the outro.
    main_video_path = project_dir / "video" / "final.mp4"
    if not main_video_path.exists():
        # Fall back to whichever video the store last picked up. CRITICAL:
        # skip `final_publish.mp4` (the OUTPUT of the finishing renderer)
        # or each rerun would recursively wrap the previous publish file
        # with another cold-open + outro and re-encode it at a lower
        # bitrate. We want the un-finished main video.
        videos = store.list_videos(context.project_id) or []
        candidates = [
            v for v in videos
            if getattr(v, "name", "") not in {"final_publish.mp4", "short.mp4"}
        ]
        if candidates:
            main_video_path = Path(str(candidates[-1].path))
        elif videos:
            # No clean candidate - fall back to whatever exists.
            main_video_path = Path(str(videos[-1].path))

    service = YouTubeBundleService()
    result = service.build(
        project_dir=project_dir,
        project_name=project.name,
        chapter_title=getattr(project.chapter_metadata, "chapter_title", None),
        manga_title=getattr(project.chapter_metadata, "manga_title", None),
        panels_json=panels_json,
        script_lines=script_lines,
        audio_manifest=audio_manifest,
        voice_config=project.voice_config,
        video_config=project.video_config,
        main_video_path=main_video_path if main_video_path.exists() else None,
        progress_callback=context.progress,
    )

    logger.info(
        "YouTube bundle for %s: title=%r thumbnail=%s",
        context.project_id, result.title, result.thumbnail_path,
    )
    context.complete(
        f"Bundle ready. Title: {result.title[:60]}{'…' if len(result.title) > 60 else ''}"
    )


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
    PipelineStage.YOUTUBE_BUNDLE: run_youtube_bundle,
}
