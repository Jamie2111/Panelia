from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
import logging
from pathlib import Path
import re
from typing import Any

from app.pipeline.image_loader import ImageLoader
from app.schemas.project import CanonicalCharacterRecord, ChapterMetadata, PanelBox, PanelVisionRecord
from app.services.llm_router import LLMRouter
from app.services.panel_vision_extractor import story_bible_canonical_fallback
from app.services.script_polisher import ScriptPolisher
from app.utils.files import read_json, write_json

logger = logging.getLogger(__name__)


class PanelVisionQualityService:
    RESCUE_WORKERS = 2

    def __init__(self, router: LLMRouter | None = None) -> None:
        self.router = router or LLMRouter()
        self.polisher = ScriptPolisher(self.router)

    def run(
        self,
        *,
        project_dir: Path,
        page_paths: list[Path],
        panels: list[PanelBox],
        canonical_characters: list[CanonicalCharacterRecord],
        project_title: str,
        chapter_metadata: ChapterMetadata,
        force_refresh: bool = False,
        progress_callback: Any | None = None,
        cancel_callback: Any | None = None,
    ) -> list[PanelVisionRecord]:
        raw_path = project_dir / "output" / "panel_vision.json"
        final_path = project_dir / "output" / "panel_vision_final.json"
        if final_path.exists() and not force_refresh:
            payload = read_json(final_path, default=[])
            if isinstance(payload, list):
                return [PanelVisionRecord.model_validate(item) for item in payload]

        raw_records = [PanelVisionRecord.model_validate(item) for item in read_json(raw_path, default=[])]
        if not raw_records:
            write_json(final_path, [])
            return []
        normalized_characters = [
            character
            if isinstance(character, CanonicalCharacterRecord)
            else CanonicalCharacterRecord.model_validate(character)
            for character in canonical_characters
        ]
        if not normalized_characters:
            normalized_characters = story_bible_canonical_fallback(project_dir)

        panels_by_id = {panel.id: panel for panel in panels}
        ordered_records = sorted(raw_records, key=lambda item: item.panel_order)
        loader = ImageLoader(project_dir=project_dir, page_paths=page_paths, max_edge=1024)
        roster = [character.model_dump(mode="json") for character in normalized_characters]
        rescued: list[PanelVisionRecord] = []
        flagged_indexes = [
            index
            for index, record in enumerate(ordered_records)
            if self._record_needs_llm_rescue(record)
        ]
        rescue_updates: dict[int, PanelVisionRecord] = {}

        def _process_flagged(index: int) -> tuple[int, PanelVisionRecord | None]:
            record = ordered_records[index]
            return (
                index,
                self._rescue_record(
                    record=record.model_copy(),
                    all_records=ordered_records,
                    record_index=index,
                    loader=loader,
                    roster=roster,
                    project_title=project_title,
                    chapter_metadata=chapter_metadata,
                    panels_by_id=panels_by_id,
                ),
            )

        if flagged_indexes:
            with ThreadPoolExecutor(max_workers=max(1, min(self.RESCUE_WORKERS, len(flagged_indexes)))) as pool:
                for completed, (index, rescued_record) in enumerate(pool.map(_process_flagged, flagged_indexes), start=1):
                    if rescued_record is not None:
                        rescue_updates[index] = rescued_record
                    if cancel_callback:
                        cancel_callback()
                    if progress_callback:
                        panel_order = ordered_records[index].panel_order
                        progress_callback(
                            round(completed / max(len(flagged_indexes), 1) * 100, 2),
                            f"Rescuing low-confidence panel {panel_order}",
                        )

        for index, record in enumerate(ordered_records):
            current = rescue_updates.get(index, record.model_copy())
            if self._needs_rescue(current):
                soft_label = self._soft_speaker_label(current)
                if soft_label:
                    current.speaker = soft_label
                    current.visual_only = False
                    current.suppression_reason = None
                elif self._record_has_meaningful_evidence(current):
                    current.visual_only = False
                    current.suppression_reason = str(current.suppression_reason or "low_confidence_kept").strip()
                else:
                    current.visual_only = True
                    current.suppression_reason = "vision_unreadable"
            else:
                current.visual_only = False
                if str(current.suppression_reason or "").strip() == "vision_unreadable":
                    current.suppression_reason = None
            rescued.append(current)

        write_json(final_path, [record.model_dump(mode="json") for record in rescued])
        return rescued

    def _needs_rescue(self, record: PanelVisionRecord) -> bool:
        action_beat = str(record.action_beat or "").strip()
        has_text = bool(str(record.dialogue or "").strip() or str(record.caption or "").strip())
        speaker_unknown = str(record.speaker or "").strip().casefold() == "unknown"
        return bool(
            record.confidence < 0.55
            or (speaker_unknown and has_text)
            or not action_beat
        )

    def _record_is_empty_shell(self, record: PanelVisionRecord) -> bool:
        return not any(
            str(value or "").strip()
            for value in (record.action_beat, record.dialogue, record.caption)
        )

    def _text_has_alpha_content(self, text: str) -> bool:
        cleaned = str(text or "").strip()
        if not cleaned:
            return False
        if re.fullmatch(r"[\W_]+", cleaned):
            return False
        return bool(re.search(r"[A-Za-z]", cleaned))

    def _action_beat_has_meaningful_content(self, text: str) -> bool:
        cleaned = str(text or "").strip()
        if not cleaned:
            return False
        gibberish_detector = getattr(self.polisher, "_is_gibberish", None)
        if callable(gibberish_detector) and gibberish_detector(cleaned):
            return False
        if len(re.findall(r"[A-Za-z']+", cleaned)) < 4:
            return False
        if re.match(r"^(?:a|an|the)\s+character\b", cleaned, flags=re.IGNORECASE):
            return False
        return True

    def _record_has_meaningful_evidence(self, record: PanelVisionRecord) -> bool:
        if self._text_has_alpha_content(record.dialogue):
            return True
        if self._text_has_alpha_content(record.caption):
            return True
        if self._action_beat_has_meaningful_content(record.action_beat):
            return True
        return False

    def _record_needs_llm_rescue(self, record: PanelVisionRecord) -> bool:
        if not self._needs_rescue(record):
            return False
        if not self._record_has_meaningful_evidence(record):
            return False
        action_beat = str(record.action_beat or "").strip()
        has_text = self._text_has_alpha_content(record.dialogue) or self._text_has_alpha_content(record.caption)
        if has_text and not action_beat:
            return True
        if action_beat.endswith("-"):
            return True
        if action_beat and len(re.findall(r"[A-Za-z']+", action_beat)) <= 4:
            return True
        speaker_unknown = str(record.speaker or "").strip().casefold() == "unknown"
        if speaker_unknown and has_text and float(record.confidence or 0.0) < 0.35:
            return True
        return False

    def _soft_speaker_label(self, record: PanelVisionRecord) -> str | None:
        if str(record.speaker or "").strip().casefold() != "unknown":
            return None
        if not str(record.dialogue or "").strip():
            return None
        if float(record.confidence or 0.0) < 0.8:
            return None
        lowered = str(record.action_beat or "").casefold()
        if not lowered:
            return None
        if "neighbor" in lowered:
            return "neighbor"
        if "off-panel" in lowered or "unseen" in lowered or "foreground tells" in lowered:
            return "off-screen speaker"
        if "crowd" in lowered or "bystander" in lowered:
            return "bystander"
        return "unseen speaker"

    def _rescue_record(
        self,
        *,
        record: PanelVisionRecord,
        all_records: list[PanelVisionRecord],
        record_index: int,
        loader: ImageLoader,
        roster: list[dict[str, Any]],
        project_title: str,
        chapter_metadata: ChapterMetadata,
        panels_by_id: dict[str, PanelBox],
    ) -> PanelVisionRecord | None:
        panel = panels_by_id.get(record.panel_id)
        if panel is None:
            return None
        full_context_paths: list[tuple[str, Path]] = []
        previous = all_records[record_index - 1] if record_index > 0 else None
        next_record = all_records[record_index + 1] if record_index + 1 < len(all_records) else None

        current_path = loader.panel_image_path(panel)
        if current_path is not None:
            full_context_paths.append((f"Current panel {record.panel_order}", current_path))
        if previous is not None:
            previous_panel = panels_by_id.get(previous.panel_id)
            previous_path = loader.panel_image_path(previous_panel) if previous_panel is not None else None
            if previous_path is not None:
                full_context_paths.append((f"Previous panel {previous.panel_order}", previous_path))
        if next_record is not None:
            next_panel = panels_by_id.get(next_record.panel_id)
            next_path = loader.panel_image_path(next_panel) if next_panel is not None else None
            if next_path is not None:
                full_context_paths.append((f"Next panel {next_record.panel_order}", next_path))
        single_paths = [pair for pair in full_context_paths if pair[0].startswith("Current panel")]
        prefer_single_image = self._record_is_empty_shell(record)
        primary_paths = single_paths if prefer_single_image and single_paths else full_context_paths
        secondary_paths = full_context_paths if primary_paths is single_paths and len(full_context_paths) > len(single_paths) else single_paths if len(single_paths) < len(full_context_paths) else []
        rescue_args = {
            "panel": {
                "panel_id": record.panel_id,
                "panel_order": record.panel_order,
                "page": record.page,
                "existing_hint": record.action_beat,
            },
            "context": {
                "character_roster": roster,
                "chapter_context": {
                    "project_title": project_title,
                    "manga_title": chapter_metadata.manga_title,
                    "chapter_title": chapter_metadata.chapter_title,
                    "chapter_number": chapter_metadata.chapter_number,
                },
                "previous_panel_action": str(previous.action_beat if previous else ""),
                "next_panel_action": str(next_record.action_beat if next_record else ""),
            },
        }
        try:
            result = asyncio.run(
                self.router.rescue_panel_vision(
                    rescue_args["panel"],
                    rescue_args["context"],
                    provider="gemini",
                    labeled_image_paths=primary_paths,
                )
            )
        except Exception as exc:
            logger.warning(
                "Panel vision rescue failed for %s with %s images: %s. Retrying alternate context.",
                record.panel_id,
                len(primary_paths),
                exc,
            )
            if not secondary_paths:
                return None
            try:
                result = asyncio.run(
                    self.router.rescue_panel_vision(
                        rescue_args["panel"],
                        rescue_args["context"],
                        provider="gemini",
                        labeled_image_paths=secondary_paths,
                    )
                )
            except Exception as exc_retry:
                logger.warning(
                    "Panel vision rescue still failed for %s with alternate context: %s",
                    record.panel_id,
                    exc_retry,
                )
                return None

        payloads = result.payload.get("panels", []) or []
        if not payloads:
            return None
        item = payloads[0]
        character_names = [
            str(name).strip()
            for name in item.get("character_names", []) or record.character_names
            if str(name).strip()
        ][:8]
        rescued = record.model_copy(
            update={
                "speaker": str(item.get("speaker") or record.speaker).strip() or "unknown",
                "dialogue": str(item.get("dialogue") or record.dialogue).strip(),
                "caption": str(item.get("caption") or record.caption).strip(),
                "action_beat": str(item.get("action_beat") or record.action_beat).strip(),
                "emotion": str(item.get("emotion") or record.emotion).strip(),
                "scene_change": bool(item.get("scene_change", record.scene_change)),
                "confidence": max(float(item.get("confidence") or 0.0), float(record.confidence or 0.0)),
                "character_names": character_names,
                "visual_only": False,
                "suppression_reason": None,
            }
        )
        return rescued
