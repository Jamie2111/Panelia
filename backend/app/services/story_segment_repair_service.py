from __future__ import annotations

import asyncio
import logging
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger(__name__)

from app.schemas.project import PanelBox, PanelVisionRecord, StorySegment
from app.services.llm_router import LLMRouter
from app.services.ocr_cleaner import clean_ocr_text
from app.services.project_store import ProjectStore
from app.services.style_vocabulary import StyleVocabulary, build_style_vocabulary
from app.services.story_grounding import build_name_grounding, extract_proper_name_candidates, normalize_name_key
from app.services.story_script_service import StoryScriptService
from app.utils.files import read_json, write_json


ProgressCallback = Callable[[float, str], None]
CancelCallback = Callable[[], None]


@dataclass(slots=True)
class StorySegmentRepairResult:
    project_id: str
    total_segments: int
    target_segments: int
    repaired_segments: int
    spoken_segments: int
    visual_only_segments: int
    quality_report: dict[str, Any]


class StorySegmentRepairService:
    """Incrementally repair weak story-first narration without redrafting the whole project.

    The full story generator can be expensive and fragile on large manga chapters.
    This repair pass targets only blank/visual-only/generic beats, saves after each
    small batch, and keeps the original segment-to-panel mapping intact.
    """

    DEFAULT_BATCH_SIZE = 3

    def __init__(
        self,
        *,
        store: ProjectStore | None = None,
        story_service: StoryScriptService | None = None,
        style_vocab: StyleVocabulary | None = None,
    ) -> None:
        self.store = store or ProjectStore()
        self.story_service = story_service or StoryScriptService(router=LLMRouter())
        self.style_vocab = style_vocab
        self._style_vocab_project_id: str | None = None

    # Maximum number of full-pass attempts before we accept that the remaining
    # weak segments cannot be filled with real narration and fall back to
    # marking them visual_only.
    MAX_CONVERGENCE_PASSES = 3

    def repair_project(
        self,
        project_id: str,
        *,
        batch_size: int | None = None,
        max_segments: int | None = None,
        use_local_ocr_rescue: bool = False,
        use_multimodal_rescue: bool = True,
        progress_callback: ProgressCallback | None = None,
        cancel_callback: CancelCallback | None = None,
    ) -> StorySegmentRepairResult:
        project = self.store.get_project(project_id)
        project_dir = self.store._project_dir(project_id)
        segments = sorted(project.story_segments or self.store.load_story_segments(project_id), key=lambda item: item.order)
        if not segments:
            raise ValueError("No story segments are available to repair. Generate a script first.")

        batch_size = max(1, int(batch_size or self.DEFAULT_BATCH_SIZE))

        # Shared caches used across convergence passes.
        story_bible = self._load_dict(project_dir / "output" / "story_bible.json")
        grounding_state = self._load_dict(project_dir / "output" / "story_grounding.json")
        character_dictionary = self._load_dict(project_dir / "output" / "character_dictionary.json")
        scene_summary_payload = self._load_dict(project_dir / "output" / "scene_summaries.json")
        chapter_summary = str(scene_summary_payload.get("chapter_summary") or story_bible.get("chapter_premise") or "").strip()
        if self.style_vocab is None or (
            self._style_vocab_project_id is not None
            and self._style_vocab_project_id != project_id
        ):
            canonical_payload = read_json(project_dir / "output" / "canonical_characters.json", default=[])
            canonical_characters = canonical_payload if isinstance(canonical_payload, list) else []
            self.style_vocab = build_style_vocabulary(
                canonical_characters=canonical_characters,
                character_dictionary=character_dictionary,
                story_bible=story_bible,
                scene_summaries=scene_summary_payload,
                chapter_summary=chapter_summary,
            )
        self._style_vocab_project_id = project_id
        chapter_metadata = self.story_service._chapter_metadata_payload(project.chapter_metadata)
        protagonist_name = str(grounding_state.get("protagonist_name") or "").strip() or None
        if not grounding_state:
            grounding_state = build_name_grounding(chapter_metadata, character_dictionary, protagonist_name)

        payloads = [
            {
                "text": self.story_service._normalize_segment_text(segment.text, allow_empty=True),
                "visual_only": bool(segment.visual_only),
                "suppression_reason": segment.suppression_reason,
            }
            for segment in segments
        ]

        total_targets_initial: int | None = None
        total_repaired = 0
        max_segments_per_pass = max_segments
        remaining_budget = max(int(max_segments), 0) if max_segments is not None else None
        enhanced_rescue_cache: dict[str, str] = {}

        for pass_number in range(1, self.MAX_CONVERGENCE_PASSES + 1):
            if cancel_callback:
                cancel_callback()

            # Rebuild units against the current payloads so rewritten lines
            # don't keep qualifying as "weak" on the next pass.
            segments_current = self._segments_from_payloads(segments, payloads)
            units, panels_by_id = self._build_units(project_id, segments_current)
            local_filled = self._fill_from_local_evidence(payloads, units)
            if local_filled:
                total_repaired += local_filled
                segments_current = self._segments_from_payloads(segments, payloads)
                self._save(project_id, project_dir, segments_current)
                units, panels_by_id = self._build_units(project_id, segments_current)
            target_indices = self._target_indices(segments_current, units)
            if remaining_budget is not None:
                target_indices = target_indices[: remaining_budget]
            if total_targets_initial is None:
                total_targets_initial = len(target_indices)

            if not target_indices:
                break

            rescued_lines = [str(payload.get("text") or "").strip() for payload in payloads]

            if progress_callback:
                label = (
                    f"Pass {pass_number}/{self.MAX_CONVERGENCE_PASSES}: "
                    f"filling {len(target_indices)} weak segment"
                    f"{'s' if len(target_indices) != 1 else ''}"
                )
                progress_callback(3, label)

            repaired_this_pass = 0
            llm_disabled = not use_multimodal_rescue
            llm_empty_batches = 0
            for batch_number, start in enumerate(range(0, len(target_indices), batch_size), start=1):
                if cancel_callback:
                    cancel_callback()
                batch_indices = target_indices[start : start + batch_size]
                batch_units = [units[index] for index in batch_indices]
                scene_visual_paths = self.story_service._build_scene_visual_paths(
                    batch_units,
                    panels_by_id,
                    project_dir / "panels",
                    project_dir / "output" / "scene_visuals_incremental_repair",
                )

                if use_local_ocr_rescue:
                    language_hint = str(chapter_metadata.get("language") or "en").strip() or "en"
                    for index in batch_indices:
                        enriched = dict(units[index])
                        cache_key = "|".join(
                            (
                                str(enriched.get("segment_id") or index),
                                str(enriched.get("panel_start") or ""),
                                str(enriched.get("panel_end") or ""),
                                self.story_service._normalized_line_key(str(enriched.get("combined_text") or "")),
                            )
                        )
                        if cache_key not in enhanced_rescue_cache:
                            enhanced_rescue_cache[cache_key] = self.story_service._enhanced_rescue_text(
                                enriched,
                                scene_visual_paths=scene_visual_paths,
                                language_hint=language_hint,
                            )
                        enriched["combined_text"] = enhanced_rescue_cache[cache_key]
                        units[index] = enriched

                rewrite_by_index = (
                    {}
                    if llm_disabled
                    else self._repair_batch_with_gemini(
                        batch_indices,
                        rescued_lines,
                        units,
                        project_title=project.name or "",
                        chapter_metadata=chapter_metadata,
                        chapter_summary=chapter_summary,
                        character_dictionary=character_dictionary,
                        protagonist_name=protagonist_name,
                        story_bible=story_bible,
                        grounding_state=grounding_state,
                        scene_visual_paths=scene_visual_paths,
                        log_label=f"pass{pass_number}-batch{batch_number}",
                    )
                )
                if rewrite_by_index:
                    llm_empty_batches = 0
                elif not llm_disabled:
                    llm_empty_batches += 1
                    if llm_empty_batches >= 2:
                        llm_disabled = True

                for index in batch_indices:
                    previous = rescued_lines[index]
                    candidate = self._trim_repair_line(rewrite_by_index.get(index, ""))
                    if not candidate:
                        candidate = self._trim_repair_line(
                            self.story_service._fallback_scene_line(
                                units[index],
                                protagonist_name,
                                style_vocab=self.style_vocab,
                            )
                        )
                    candidate = self.story_service._normalize_segment_text(candidate, allow_empty=True)
                    candidate_key = self.story_service._normalized_line_key(candidate)
                    previous_key = self.story_service._normalized_line_key(rescued_lines[index - 1]) if index > 0 else ""
                    next_key = self.story_service._normalized_line_key(rescued_lines[index + 1]) if index + 1 < len(rescued_lines) else ""
                    if candidate_key and (candidate_key == previous_key or candidate_key == next_key):
                        fallback = self.story_service._fallback_scene_line(
                            units[index],
                            protagonist_name,
                            style_vocab=self.style_vocab,
                        )
                        fallback = self._trim_repair_line(fallback)
                        fallback_key = self.story_service._normalized_line_key(fallback)
                        if fallback_key and fallback_key not in {previous_key, next_key, candidate_key}:
                            candidate = fallback
                    if not self._usable_repair_line(candidate):
                        continue
                    rescued_lines[index] = candidate
                    payloads[index] = {
                        "text": candidate,
                        "visual_only": False,
                        "suppression_reason": None,
                    }
                    if candidate.strip() and candidate.strip() != previous.strip():
                        repaired_this_pass += 1

                updated_segments = self._segments_from_payloads(segments, payloads)
                self._save(project_id, project_dir, updated_segments)
                done = min(start + len(batch_indices), len(target_indices))
                if progress_callback:
                    base = 8 + ((pass_number - 1) / self.MAX_CONVERGENCE_PASSES) * 80
                    span = 80 / self.MAX_CONVERGENCE_PASSES
                    progress_callback(
                        base + (done / max(len(target_indices), 1)) * span,
                        f"Pass {pass_number}: repaired {done}/{len(target_indices)}",
                    )

            total_repaired += repaired_this_pass
            if remaining_budget is not None:
                remaining_budget = max(remaining_budget - len(target_indices), 0)
                if remaining_budget == 0:
                    break

            # If this pass didn't improve anything, further passes will not
            # either — break early to avoid wasting LLM budget.
            if repaired_this_pass == 0:
                break

        # Final safety net: any segment that is STILL empty or low-quality
        # after the convergence loop gets marked visual_only=True so it does
        # not leak OCR garbage into the narration audio. The frontend already
        # treats visual_only segments as silent gaps.
        final_units, _ = self._build_units(project_id, self._segments_from_payloads(segments, payloads))
        style_line_keys = [
            self.story_service._normalized_line_key(
                self.story_service._normalize_segment_text(str(payload.get("text") or ""), allow_empty=True)
            )
            for payload in payloads
        ]
        style_duplicate_counts = Counter(key for key in style_line_keys if key)
        style_rescue_indices = [
            index
            for index, payload in enumerate(payloads)
            if (
                not self._usable_repair_line(str(payload.get("text") or ""))
                or bool(style_line_keys[index] and style_duplicate_counts[style_line_keys[index]] > 1)
            )
        ]
        if use_multimodal_rescue:
            for start in range(0, len(style_rescue_indices), batch_size):
                batch_indices = style_rescue_indices[start : start + batch_size]
                rewrite_by_index = self._style_rescue_with_gemini(
                    batch_indices,
                    payloads,
                    final_units,
                    project_title=project.name or "",
                    chapter_metadata=chapter_metadata,
                    chapter_summary=chapter_summary,
                    character_dictionary=character_dictionary,
                    story_bible=story_bible,
                    grounding_state=grounding_state,
                )
                for index in batch_indices:
                    candidate = self._trim_repair_line(rewrite_by_index.get(index, ""))
                    if not candidate:
                        continue
                    payloads[index] = {
                        "text": self.story_service._normalize_segment_text(candidate, allow_empty=True),
                        "visual_only": False,
                        "suppression_reason": None,
                    }

        suppressed_in_fallback = 0
        fallback_filled = 0
        final_line_keys = [
            self.story_service._normalized_line_key(
                self.story_service._normalize_segment_text(str(payload.get("text") or ""), allow_empty=True)
            )
            for payload in payloads
        ]
        final_duplicate_counts = Counter(key for key in final_line_keys if key)
        for index, payload in enumerate(payloads):
            text = str(payload.get("text") or "").strip()
            duplicate_text = bool(final_line_keys[index] and final_duplicate_counts[final_line_keys[index]] > 1)
            if not text:
                fallback = self._trim_repair_line(self._emergency_repair_line(final_units[index], payloads, index))
                if fallback:
                    payload["text"] = fallback
                    payload["visual_only"] = False
                    payload["suppression_reason"] = None
                    fallback_filled += 1
                elif not payload.get("visual_only"):
                    payload["visual_only"] = True
                    payload.setdefault("suppression_reason", "fill_exhausted")
                    suppressed_in_fallback += 1
                continue
            if (
                self.story_service._line_is_low_quality(text)
                or self.story_service._line_is_overly_generic(text)
                or self.story_service._line_is_sentence_fragment(text)
                or self.story_service._line_is_dialogue_fragment(text)
                or duplicate_text
                or not self._line_supported_by_unit_vision(text, final_units[index])
            ):
                fallback = self._trim_repair_line(self._emergency_repair_line(final_units[index], payloads, index))
                if fallback:
                    payload["text"] = fallback
                    payload["visual_only"] = False
                    payload["suppression_reason"] = None
                    fallback_filled += 1
                else:
                    payload["text"] = ""
                    payload["visual_only"] = True
                    payload["suppression_reason"] = "fill_exhausted"
                    suppressed_in_fallback += 1

        # Final near-duplicate collapse on the post-repair payloads. Even with
        # the per-batch adjacent-neighbor guard above, the repair pass can
        # still produce text that is byte-identical to a non-adjacent
        # neighbour (for example when scene 6 is unchanged from the dedupe
        # output, scene 7 was blanked as a near_duplicate of scene 6, then
        # the dedupe-skip exclusion in ``_target_indices`` was added later
        # but earlier passes still wrote a candidate before the exclusion
        # took effect). Running the dedupe one last time guarantees that
        # ``narration_story.txt`` never contains the same line twice.
        units_for_dedupe, _ = self._build_units(project_id, self._segments_from_payloads(segments, payloads))
        before_blank_count = sum(1 for payload in payloads if not str(payload.get("text") or "").strip())
        deduped_payloads = self.story_service._collapse_near_duplicate_segments(
            [dict(payload) for payload in payloads],
            units_for_dedupe,
            blank_unresolved=False,
        )
        # Mirror dedupe results back into ``payloads`` so the save below sees them.
        for index, deduped in enumerate(deduped_payloads):
            payloads[index] = deduped
        # Dedupe may blank a repeated line. Immediately refill those slots
        # from local evidence before we attempt any chapter-level cohesion, so
        # coverage never depends on the optional Gemini pass.
        payloads = self.story_service._fill_blank_story_payloads(
            payloads,
            units_for_dedupe,
            protagonist_name=protagonist_name,
            grounding=grounding_state,
            story_bible=story_bible,
            style_vocab=self.style_vocab,
        )
        fallback_filled += self._fill_from_local_evidence(payloads, units_for_dedupe)
        after_blank_count = sum(1 for payload in payloads if not str(payload.get("text") or "").strip())
        post_repair_dedupes = max(after_blank_count - before_blank_count, 0)
        if post_repair_dedupes:
            logger.info(
                "Post-repair dedupe blanked %d segment(s) that the repair pass had filled with text matching a neighbour",
                post_repair_dedupes,
            )

        # Repair can leave the script technically safe but still choppy because
        # it works in tiny batches. Finish with the same chapter-level narrator
        # cohesion pass used by full generation, now enriched with local vision
        # evidence so alignment is preserved.
        cohesion_changed = False
        vision_backed_units = any(bool(unit.get("has_panel_vision")) for unit in final_units)
        if use_multimodal_rescue and not vision_backed_units and len(payloads) >= 3:
            before_payloads = [dict(payload) for payload in payloads]
            before_cohesion = [str(payload.get("text") or "").strip() for payload in before_payloads]
            before_blank_count = sum(1 for text in before_cohesion if not text)
            candidate_payloads = self.story_service._narrator_cohesion_pass(
                [dict(payload) for payload in before_payloads],
                units_for_dedupe,
                project_title=project.name or "",
                chapter_metadata=chapter_metadata,
                chapter_summary=chapter_summary,
                character_dictionary=character_dictionary,
                protagonist_name=protagonist_name,
                name_grounding=grounding_state,
                require_multi_sentence=True,
                style_vocab=self.style_vocab,
            )
            candidate_payloads = self.story_service._stabilize_reviewed_segments(
                candidate_payloads,
                units_for_dedupe,
                protagonist_name,
                grounding_state,
                story_bible,
            )
            candidate_payloads = self.story_service._collapse_internal_duplicate_sentences(candidate_payloads)
            candidate_payloads = self.story_service._collapse_near_duplicate_segments(
                candidate_payloads,
                units_for_dedupe,
                blank_unresolved=False,
            )
            candidate_payloads = self.story_service._fill_blank_story_payloads(
                candidate_payloads,
                units_for_dedupe,
                protagonist_name=protagonist_name,
                grounding=grounding_state,
                story_bible=story_bible,
                style_vocab=self.style_vocab,
            )
            after_cohesion = [str(payload.get("text") or "").strip() for payload in candidate_payloads]
            after_blank_count = sum(1 for text in after_cohesion if not text)
            if after_blank_count <= before_blank_count:
                payloads = candidate_payloads
                cohesion_changed = before_cohesion != after_cohesion
            else:
                merged_payloads = [dict(payload) for payload in before_payloads]
                accepted_richer = 0
                for index, candidate_payload in enumerate(candidate_payloads):
                    before_text = self.story_service._normalize_segment_text(
                        str(before_payloads[index].get("text") or ""),
                        allow_empty=True,
                    )
                    candidate_text = self.story_service._normalize_segment_text(
                        str(candidate_payload.get("text") or ""),
                        allow_empty=True,
                    )
                    if not before_text or not candidate_text:
                        continue
                    if self.story_service._sentence_count(candidate_text) < max(2, self.story_service._sentence_count(before_text)):
                        continue
                    if (
                        self.story_service._line_is_low_quality(candidate_text)
                        or self.story_service._line_is_overly_generic(candidate_text)
                        or self.story_service._line_is_dialogue_fragment(candidate_text)
                        or self.story_service._line_is_sentence_fragment(candidate_text)
                    ):
                        continue
                    merged_payloads[index]["text"] = candidate_text
                    merged_payloads[index]["visual_only"] = False
                    merged_payloads[index]["suppression_reason"] = None
                    accepted_richer += 1
                if accepted_richer:
                    payloads = merged_payloads
                    cohesion_changed = True
                    logger.info(
                        "Partially accepted %d richer narrator cohesion rewrites while preserving spoken coverage (%d -> %d blanks)",
                        accepted_richer,
                        before_blank_count,
                        after_blank_count,
                    )
                else:
                    logger.warning(
                        "Discarding narrator cohesion result because it reduced spoken coverage (%d -> %d blanks)",
                        before_blank_count,
                        after_blank_count,
                    )

        reinforced_payloads = self.story_service._reinforce_multi_sentence_scene_payloads(
            [dict(payload) for payload in payloads],
            units_for_dedupe,
            protagonist_name=protagonist_name,
            grounding=grounding_state,
            story_bible=story_bible,
            style_vocab=self.style_vocab,
        )
        if use_multimodal_rescue and not vision_backed_units:
            reinforced_payloads = self.story_service._expand_short_scene_payloads_with_llm(
                reinforced_payloads,
                units_for_dedupe,
                project_title=project.name or "",
                chapter_metadata=chapter_metadata,
                chapter_summary=chapter_summary,
                character_dictionary=character_dictionary,
                protagonist_name=protagonist_name,
                story_bible=story_bible,
                name_grounding=grounding_state,
                style_vocab=self.style_vocab,
            )
        reinforced_payloads = self.story_service._remove_overused_generic_sentences(reinforced_payloads)
        reinforced_payloads = self.story_service._fill_blank_story_payloads(
            reinforced_payloads,
            units_for_dedupe,
            protagonist_name=protagonist_name,
            grounding=grounding_state,
            story_bible=story_bible,
            style_vocab=self.style_vocab,
        )
        reinforced_payloads = self.story_service._collapse_internal_duplicate_sentences(reinforced_payloads)
        reinforced_payloads = self.story_service._collapse_near_duplicate_segments(
            reinforced_payloads,
            units_for_dedupe,
            blank_unresolved=False,
        )
        reinforced_payloads = self.story_service._fill_blank_story_payloads(
            reinforced_payloads,
            units_for_dedupe,
            protagonist_name=protagonist_name,
            grounding=grounding_state,
            story_bible=story_bible,
            style_vocab=self.style_vocab,
        )
        fallback_filled += self._fill_from_local_evidence(reinforced_payloads, units_for_dedupe)
        reinforced_payloads = self.story_service._remove_overused_generic_sentences(reinforced_payloads)
        reinforced_payloads = self.story_service._fill_blank_story_payloads(
            reinforced_payloads,
            units_for_dedupe,
            protagonist_name=protagonist_name,
            grounding=grounding_state,
            story_bible=story_bible,
            style_vocab=self.style_vocab,
        )
        fallback_filled += self._fill_from_local_evidence(reinforced_payloads, units_for_dedupe)
        reinforced_payloads = self.story_service._collapse_internal_duplicate_sentences(reinforced_payloads)
        reinforced_payloads = self.story_service._collapse_near_duplicate_segments(
            reinforced_payloads,
            units_for_dedupe,
            blank_unresolved=False,
        )
        reinforced_payloads = self.story_service._fill_blank_story_payloads(
            reinforced_payloads,
            units_for_dedupe,
            protagonist_name=protagonist_name,
            grounding=grounding_state,
            story_bible=story_bible,
            style_vocab=self.style_vocab,
        )
        fallback_filled += self._fill_from_local_evidence(reinforced_payloads, units_for_dedupe)
        reinforced_payloads = self.story_service._reinforce_multi_sentence_scene_payloads(
            reinforced_payloads,
            units_for_dedupe,
            protagonist_name=protagonist_name,
            grounding=grounding_state,
            story_bible=story_bible,
            style_vocab=self.style_vocab,
        )
        reinforced_payloads = self.story_service._remove_overused_generic_sentences(reinforced_payloads)
        reinforced_payloads = self.story_service._collapse_internal_duplicate_sentences(reinforced_payloads)
        reinforced_payloads = self.story_service._collapse_near_duplicate_segments(
            reinforced_payloads,
            units_for_dedupe,
            blank_unresolved=False,
        )
        reinforced_payloads = self.story_service._fill_blank_story_payloads(
            reinforced_payloads,
            units_for_dedupe,
            protagonist_name=protagonist_name,
            grounding=grounding_state,
            story_bible=story_bible,
            style_vocab=self.style_vocab,
        )
        fallback_filled += self._fill_from_local_evidence(reinforced_payloads, units_for_dedupe)
        reinforced_payloads = self.story_service._collapse_near_duplicate_segments(
            reinforced_payloads,
            units_for_dedupe,
            blank_unresolved=False,
        )
        reinforced_payloads = self.story_service._fill_blank_story_payloads(
            reinforced_payloads,
            units_for_dedupe,
            protagonist_name=protagonist_name,
            grounding=grounding_state,
            story_bible=story_bible,
            style_vocab=self.style_vocab,
        )
        fallback_filled += self._fill_from_local_evidence(reinforced_payloads, units_for_dedupe)
        if reinforced_payloads != payloads:
            payloads = reinforced_payloads
            cohesion_changed = True

        if suppressed_in_fallback or fallback_filled or post_repair_dedupes or cohesion_changed:
            updated_segments = self._segments_from_payloads(segments, payloads)
            self._save(project_id, project_dir, updated_segments)

        updated_segments = self.store.load_story_segments(project_id)
        report = self.store.load_script_quality_report(project_id)
        target_count = total_targets_initial or 0
        return self._result(
            project_id,
            updated_segments,
            target_count=target_count,
            repaired_count=total_repaired,
            quality_report=report,
        )

    def _fill_from_local_evidence(
        self,
        payloads: list[dict[str, Any]],
        units: list[dict[str, Any]],
    ) -> int:
        """Fill obvious weak slots before spending LLM calls.

        Gemini repair is useful for polish, but coverage should not depend on
        network availability. This pass uses trusted vision/dialogue evidence
        and conservative templates to keep every panel range represented.
        """
        filled = 0
        line_keys = [
            self.story_service._normalized_line_key(
                self.story_service._normalize_segment_text(str(payload.get("text") or ""), allow_empty=True)
            )
            for payload in payloads
        ]
        duplicate_counts = Counter(key for key in line_keys if key)
        for index, (payload, unit) in enumerate(zip(payloads, units, strict=False)):
            text = self.story_service._normalize_segment_text(str(payload.get("text") or ""), allow_empty=True)
            duplicate_text = bool(line_keys[index] and duplicate_counts[line_keys[index]] > 1)
            unsupported_vision_line = bool(text) and not self._line_supported_by_unit_vision(text, unit)
            needs_fill = (
                not text
                or bool(payload.get("visual_only"))
                or bool(payload.get("suppression_reason"))
                or duplicate_text
                or unsupported_vision_line
                or self.story_service._line_is_low_quality(text)
                or self.story_service._line_is_overly_generic(text)
                or self.story_service._line_is_dialogue_fragment(text)
                or self.story_service._line_is_sentence_fragment(text)
            )
            if not needs_fill:
                continue
            candidate = self._trim_repair_line(self._emergency_repair_line(unit, payloads, index))
            if not candidate:
                continue
            if candidate and candidate != text:
                filled += 1
            payload["text"] = candidate
            payload["visual_only"] = False
            payload["suppression_reason"] = None
        return filled

    def _repair_batch_with_gemini(
        self,
        batch_indices: list[int],
        rescued_lines: list[str],
        units: list[dict[str, Any]],
        *,
        project_title: str,
        chapter_metadata: dict[str, Any],
        chapter_summary: str,
        character_dictionary: dict[str, Any],
        protagonist_name: str | None,
        story_bible: dict[str, Any],
        grounding_state: dict[str, Any],
        scene_visual_paths: dict[str, list[Path]],
        log_label: str,
    ) -> dict[int, str]:
        try:
            if "gemini" not in self.story_service.router.available_providers():
                return {}
        except Exception:
            return {}

        return self.story_service._run_multimodal_rescue_batch(
            batch_indices,
            rescued_lines,
            units,
            project_title=project_title,
            chapter_metadata=chapter_metadata,
            chapter_summary=chapter_summary,
            character_dictionary=character_dictionary,
            protagonist_name=protagonist_name,
            prompt_story_bible=self.story_service._story_bible_prompt_payload(story_bible),
            allowed_character_names=list(grounding_state.get("allowed_character_names") or []),
            scene_visual_paths=scene_visual_paths,
            log_label=log_label,
        )

    def _target_indices(self, segments: list[StorySegment], units: list[dict[str, Any]]) -> list[int]:
        targets: list[int] = []
        line_keys = [
            self.story_service._normalized_line_key(
                self.story_service._normalize_segment_text(segment.text, allow_empty=True)
            )
            for segment in segments
        ]
        duplicate_counts = Counter(key for key in line_keys if key)
        # Suppression reasons that mark a segment as truly unrecoverable.
        # ``vision_unreadable`` means Gemini refused the underlying imagery and
        # we have no trustworthy signal to ground a rewrite in. ``near_duplicate``
        # is intentionally NOT terminal anymore: once a duplicate line has been
        # blanked, the repair pass gets a chance to refill that slot with a more
        # specific multimodal beat, and the final post-repair dedupe will still
        # blank it again if it collapses back into its neighbour.
        terminal_suppressions = {"vision_unreadable"}
        for index, (segment, unit) in enumerate(zip(segments, units, strict=False)):
            line = self.story_service._normalize_segment_text(segment.text, allow_empty=True)
            reason = self.story_service._multimodal_rescue_reason(line, unit)
            duplicate_neighbor = bool(
                line_keys[index]
                and (
                    (index > 0 and line_keys[index] == line_keys[index - 1])
                    or (index + 1 < len(line_keys) and line_keys[index] == line_keys[index + 1])
                )
            )
            duplicate_anywhere = bool(line_keys[index] and duplicate_counts[line_keys[index]] > 1)
            style_issue = bool(line) and self.story_service._line_needs_style_refinement(line)
            unsupported_vision_line = bool(line) and not self._line_supported_by_unit_vision(line, unit)
            if str(segment.suppression_reason or "").strip() in terminal_suppressions:
                continue
            if (
                not line
                or bool(segment.visual_only)
                or bool(segment.suppression_reason)
                or bool(reason)
                or duplicate_neighbor
                or duplicate_anywhere
                or style_issue
                or unsupported_vision_line
            ):
                targets.append(index)
        return targets

    def _line_supported_by_unit_vision(self, line: str, unit: dict[str, Any]) -> bool:
        if not bool(unit.get("has_panel_vision")):
            return True
        evidence = " ".join(
            str(unit.get(key) or "").strip()
            for key in ("vision_action_beat", "vision_caption", "vision_dialogue", "combined_text", "ocr_fallback_text")
            if str(unit.get(key) or "").strip()
        )
        if not evidence.strip():
            return True
        line_tokens = self.story_service._content_token_set(line)
        evidence_tokens = self.story_service._content_token_set(evidence)
        if not line_tokens or not evidence_tokens:
            return True
        overlap = len(line_tokens & evidence_tokens)
        containment = overlap / max(1, min(len(line_tokens), len(evidence_tokens)))
        if overlap >= 2 and containment >= 0.18:
            return True
        line_key = normalize_name_key(line)
        evidence_key = normalize_name_key(evidence)
        shared_names = [
            name
            for name in unit.get("character_names", []) or []
            if normalize_name_key(name) and normalize_name_key(name) in line_key and normalize_name_key(name) in evidence_key
        ]
        return bool(shared_names and overlap >= 1)

    def _usable_repair_line(self, line: str) -> bool:
        candidate = self.story_service._normalize_segment_text(line, allow_empty=True)
        if not candidate:
            return False
        if self.story_service._line_is_low_quality(candidate):
            return False
        if self.story_service._line_is_overly_generic(candidate):
            return False
        if self.story_service._line_is_dialogue_fragment(candidate):
            return False
        if self.story_service._line_is_sentence_fragment(candidate):
            return False
        if self.story_service._line_needs_style_refinement(candidate):
            return False
        if self.story_service._line_has_first_person_narration(candidate):
            return False
        if self.story_service.polisher._is_visual_description(candidate):
            return False
        return True

    def _trim_repair_line(self, line: str) -> str:
        candidate = self.story_service._normalize_segment_text(line, allow_empty=True)
        if not candidate:
            return ""
        if self._usable_repair_line(candidate):
            return candidate
        if len(self.story_service._split_sentences_for_cleanup(candidate)) < 2:
            return ""
        trimmed = self.story_service._remove_offending_sentences(candidate)
        if trimmed and self._usable_repair_line(trimmed):
            return trimmed
        return ""

    def _style_rescue_with_gemini(
        self,
        batch_indices: list[int],
        payloads: list[dict[str, Any]],
        units: list[dict[str, Any]],
        *,
        project_title: str,
        chapter_metadata: dict[str, Any],
        chapter_summary: str,
        character_dictionary: dict[str, Any],
        story_bible: dict[str, Any],
        grounding_state: dict[str, Any],
    ) -> dict[int, str]:
        if not batch_indices:
            return {}
        try:
            if "gemini" not in self.story_service.router.available_providers():
                return {}
        except Exception:
            return {}

        lines: list[dict[str, Any]] = []
        allowed_character_names = list(grounding_state.get("allowed_character_names") or [])
        prompt_story_bible = self.story_service._story_bible_prompt_payload(story_bible)
        for local_index, global_index in enumerate(batch_indices):
            unit = units[global_index]
            current_line = self._style_rescue_seed(unit)
            if not current_line:
                current_line = self._emergency_repair_line(unit, payloads, global_index)
            if not current_line:
                current_line = self._neighbor_bridge_seed(payloads, global_index)
            if not current_line:
                continue
            lines.append(
                {
                    "index": local_index,
                    "current_line": current_line,
                    "previous_line": str(payloads[global_index - 1].get("text") or "").strip() if global_index > 0 else "",
                    "next_line": str(payloads[global_index + 1].get("text") or "").strip() if global_index + 1 < len(payloads) else "",
                    "ocr_text": str(unit.get("combined_text") or "").strip(),
                    "scene_summary": str(unit.get("scene_summary") or "").strip(),
                    "visual_cues": self.story_service._style_evidence_text(unit),
                    "vision_dialogue": str(unit.get("vision_dialogue") or "").strip(),
                    "vision_caption": str(unit.get("vision_caption") or "").strip(),
                    "vision_action_beat": str(unit.get("vision_action_beat") or "").strip(),
                    "character_names": unit.get("character_names", []) or [],
                    "panel_count": int(unit.get("panel_count") or 0),
                }
            )

        if not lines:
            return {}

        try:
            result = asyncio.run(
                self.story_service.router.refine_story_segment_style(
                    lines,
                    {
                        "project_title": project_title,
                        "chapter_summary": chapter_summary,
                        "chapter_metadata": chapter_metadata,
                        "character_dictionary": character_dictionary,
                        "story_bible": prompt_story_bible,
                        "allowed_character_names": allowed_character_names,
                    },
                    provider="gemini",
                )
            )
        except Exception:
            return {}

        rewrites: dict[int, str] = {}
        for item in result.payload.get("rewrites", []):
            if not isinstance(item, dict):
                continue
            local_index = int(item.get("index") or 0)
            if local_index >= len(batch_indices):
                continue
            rewrites[batch_indices[local_index]] = self.story_service._normalize_segment_text(
                str(item.get("line") or "").strip(),
                allow_empty=True,
            )
        return rewrites

    def _style_rescue_seed(self, unit: dict[str, Any]) -> str:
        for candidate in (
            str(unit.get("vision_action_beat") or "").strip(),
            str(unit.get("visual_cues") or "").strip(),
            str(unit.get("combined_text") or "").strip(),
            str(unit.get("ocr_fallback_text") or "").strip(),
            str(unit.get("scene_summary") or "").strip(),
        ):
            normalized = self.story_service._normalize_segment_text(candidate, allow_empty=True)
            if normalized:
                return normalized
        return ""

    def _neighbor_bridge_seed(
        self,
        payloads: list[dict[str, Any]],
        index: int,
        unit: dict[str, Any] | None = None,
    ) -> str:
        previous = str(payloads[index - 1].get("text") or "").strip() if index > 0 else ""
        next_line = str(payloads[index + 1].get("text") or "").strip() if index + 1 < len(payloads) else ""
        raw_context = " ".join(part for part in (previous, next_line) if part)
        context = raw_context.casefold()
        if not context:
            return ""
        if self.style_vocab:
            templates = self._filled_style_templates(
                (
                    "{subject_a} and {subject_b} keep the nearby risk tied to {team}. Their connection gives the beat a consequence the group has to answer.",
                    "{team} gathers around {world_term} as the scene moves from reaction into decision.",
                    "{subject} keeps the nearby choice active while the surrounding group reacts.",
                ),
                unit or {},
                raw_context,
            )
            if templates:
                return templates[0]
        names = [
            name
            for name in extract_proper_name_candidates(raw_context)
            if not re.search(r"\b(?:unknown|speaker|narrator|protagonist|character|figure|someone)\b", name, flags=re.IGNORECASE)
        ]
        if names:
            return f"{names[0]} keeps the nearby choice active while the surrounding group reacts."
        return ""

    def _emergency_repair_line(
        self,
        unit: dict[str, Any],
        payloads: list[dict[str, Any]],
        index: int,
    ) -> str:
        """Last-resort coverage guard used only after Gemini repair fails.

        It prefers grounded unit evidence, then emits a conservative bridge so
        the editor/video never loses a story slot to silence.
        """
        candidates = [
            self.story_service._evidence_bridge_line(unit, None, style_vocab=self.style_vocab),
            self.story_service._fallback_scene_line(unit, None, style_vocab=self.style_vocab),
            str(unit.get("vision_action_beat") or "").strip(),
            str(unit.get("visual_cues") or "").strip(),
            str(unit.get("vision_caption") or "").strip(),
            str(unit.get("ocr_fallback_text") or "").strip(),
        ]
        for candidate in candidates:
            normalized = self.story_service._normalize_segment_text(candidate, allow_empty=True)
            trimmed = self._trim_repair_line(normalized)
            if trimmed and not self._line_already_used(trimmed, payloads, index):
                return trimmed

        if bool(unit.get("has_panel_vision")):
            return ""

        trusted_context = " ".join(
            str(value or "")
            for value in (
                unit.get("vision_action_beat"),
                unit.get("vision_dialogue"),
                unit.get("vision_caption"),
                unit.get("combined_text"),
                unit.get("ocr_fallback_text"),
            )
            if str(value or "").strip()
        )
        if bool(unit.get("has_panel_vision")) and not trusted_context.strip():
            return ""
        local_context_source = trusted_context
        if not bool(unit.get("has_panel_vision")):
            local_context_source = " ".join(
                str(value or "")
                for value in (
                    unit.get("title"),
                    trusted_context,
                    unit.get("scene_summary"),
                )
                if str(value or "").strip()
            )
        local_context = local_context_source.casefold()
        context = " ".join(
            str(value or "")
            for value in (
                local_context,
                payloads[index - 1].get("text") if index > 0 else "",
                payloads[index + 1].get("text") if index + 1 < len(payloads) else "",
            )
        ).casefold()
        names = [
            str(name).strip()
            for name in unit.get("character_names", []) or []
            if str(name).strip()
        ]
        neighbor_seed = self._neighbor_bridge_seed(payloads, index, unit)
        if (
            neighbor_seed
            and self._usable_repair_line(neighbor_seed)
            and not self._line_already_used(neighbor_seed, payloads, index)
        ):
            return neighbor_seed
        style_templates = self._repair_style_templates(local_context, unit)
        if style_templates:
            candidate = self._first_unused_template(style_templates, unit, payloads, index)
            if candidate:
                return candidate
        if names:
            joined = " and ".join(names[:2])
            candidate = f"{joined} has to respond before the situation moves out of reach."
            if not self._line_already_used(candidate, payloads, index):
                return candidate
        return self._first_unused_template(
            (
                "The beat shifts again, leaving the next choice harder to avoid.",
                "The scene moves into a strained moment where hesitation becomes dangerous.",
                "The nearby danger narrows the response available to the group.",
                "The moment carries enough consequence to keep the chapter moving.",
            ),
            unit,
            payloads,
            index,
        )

    def _repair_style_templates(self, local_context: str, unit: dict[str, Any]) -> tuple[str, ...]:
        if not self.style_vocab:
            return ()
        bucket = "neutral"
        if re.search(r"\b(?:fight|battle|attack|enemy|threat|clash|skirmish|combat)\b", local_context):
            bucket = "combat"
        elif re.search(r"\b(?:explain|order|brief|command|prepare|arrive|leave|enter)\w*\b", local_context):
            bucket = "transition"
        elif re.search(r"\b(?:missing|absence|loss|lost|dead|alone|without)\b", local_context):
            bucket = "absence_or_loss"
        elif re.search(r"\b(?:resource|supply|food|water|medicine|battery|power|shelter|stockpile)\w*\b", local_context):
            bucket = "resource_or_supply"
        buckets = {
            "combat": (
                "{team} is pushed deeper into {world_term} as {antagonist} refuses to give ground. The escalation leaves every response sharper than the last.",
                "{world_term} closes in around {team}, turning the local action into a survival problem.",
                "{antagonist}'s pressure turns {stakes} into another scramble that the group cannot ignore.",
            ),
            "transition": (
                "The room moves from explanation into orders, leaving less room for hesitation. What had been a pause becomes the next instruction everyone has to follow.",
                "{subject} faces {stakes} head-on as the scene turns toward action. The exchange gives the next beat a clearer direction.",
            ),
            "absence_or_loss": (
                "{subject}'s absence keeps weighing on {team} as {stakes} moves forward. The gap changes how everyone reacts to the next decision.",
                "Without {subject}, {team} is left trying to keep {stakes} from falling apart. The strain tests whether the group can still hold together.",
            ),
            "resource_or_supply": (
                "{subject} keeps {stakes} guarded as the world outside turns every request into a risk. The scene makes that caution feel less like suspicion and more like survival.",
                "{subject} connects {stakes} to {world_term}, making every request feel like a risk. That calculation pushes the group toward a harder choice.",
            ),
            "neutral": (
                "{subject} faces the moment head-on as the scene turns toward another response.",
                "The beat keeps moving because the last choice still has consequences.",
            ),
        }
        return self._filled_style_templates(buckets[bucket], unit, local_context)

    def _filled_style_templates(
        self,
        templates: tuple[str, ...],
        unit: dict[str, Any],
        context: str,
    ) -> tuple[str, ...]:
        vocab = self.style_vocab
        if not vocab:
            return ()
        context_key = normalize_name_key(context)

        placeholder_phrases = {
            "the immediate problem",
            "the next exchange",
            "the exchange",
            "the group tension",
            "the protected space",
            "the resource problem",
        }

        def _useful(value: str, *, allow_single: bool = False) -> str:
            cleaned = str(value or "").strip()
            if not cleaned or cleaned.casefold() in placeholder_phrases:
                return ""
            words = re.findall(r"[A-Za-z0-9]+", cleaned)
            if len(words) >= 2:
                return cleaned
            if allow_single and words and (cleaned.isupper() or re.search(r"[A-Z0-9]", cleaned)):
                return cleaned
            return ""

        def _phrase_supported(value: str, *, allow_single: bool = False) -> bool:
            useful = _useful(value, allow_single=allow_single)
            key = normalize_name_key(useful)
            return bool(useful and key and key in context_key)

        def _pick(values: tuple[str, ...] | list[str], fallback: str = "", *, allow_single: bool = False) -> str:
            choices = [str(value).strip() for value in values if str(value).strip()]
            for choice in choices:
                key = normalize_name_key(choice)
                if key and key in context_key and _useful(choice, allow_single=allow_single):
                    return choice
            if fallback and _phrase_supported(fallback, allow_single=allow_single):
                return _useful(fallback, allow_single=allow_single)
            return ""

        unit_names = [str(name).strip() for name in unit.get("character_names", []) or [] if str(name).strip()]
        subject = unit_names[0] if unit_names else next(
            (
                name
                for name in vocab.named_characters
                if normalize_name_key(name) and normalize_name_key(name) in context_key
            ),
            "",
        )
        subject_b = next(
            (
                name
                for name in (*unit_names[1:], *vocab.named_characters)
                if normalize_name_key(name) and normalize_name_key(name) != normalize_name_key(subject)
                and (name in unit_names or normalize_name_key(name) in context_key)
            ),
            "",
        )
        slots = {
            "subject": subject,
            "subject_a": subject,
            "subject_b": subject_b,
            "team": vocab.team_term if _phrase_supported(vocab.team_term or "") else "",
            "world_term": _pick(vocab.world_terms, "", allow_single=True),
            "stakes": _pick(vocab.stakes_phrases, "", allow_single=False),
            "action_verb": _pick(vocab.action_verbs, "pressing", allow_single=False) or "pressing",
            "antagonist": vocab.antagonist_term if _phrase_supported(vocab.antagonist_term or "") else "",
        }
        filled: list[str] = []
        for template in templates:
            required = set(re.findall(r"{([a-z_]+)}", template))
            if any(not slots.get(slot) for slot in required):
                continue
            candidate = template.format(**slots)
            normalized = self.story_service._normalize_segment_text(candidate, allow_empty=True)
            if normalized and self._usable_repair_line(normalized):
                filled.append(normalized)
        return tuple(filled)

    def _first_unused_template(
        self,
        templates: tuple[str, ...],
        unit: dict[str, Any],
        payloads: list[dict[str, Any]],
        index: int,
    ) -> str:
        if not templates:
            return ""
        start = self._template_index(unit, len(templates))
        for offset in range(len(templates)):
            candidate = templates[(start + offset) % len(templates)]
            if not self._line_already_used(candidate, payloads, index):
                return candidate
        return ""

    def _line_already_used(self, line: str, payloads: list[dict[str, Any]], index: int) -> bool:
        key = self.story_service._normalized_line_key(line)
        if not key:
            return False
        for other_index, payload in enumerate(payloads):
            if other_index == index:
                continue
            other_key = self.story_service._normalized_line_key(str(payload.get("text") or ""))
            if other_key and other_key == key:
                return True
        return False

    def _template_index(self, unit: dict[str, Any], length: int) -> int:
        if length <= 1:
            return 0
        return (
            int(unit.get("scene_id") or 0)
            + int(unit.get("sequence_in_scene") or 0)
            + int(unit.get("panel_start") or 0)
        ) % length

    def _build_units(
        self,
        project_id: str,
        segments: list[StorySegment],
    ) -> tuple[list[dict[str, Any]], dict[str, PanelBox]]:
        project = self.store.get_project(project_id)
        project_dir = self.store._project_dir(project_id)
        story_bible = self._load_dict(project_dir / "output" / "story_bible.json")
        panels_by_id = {panel.id: panel for panel in project.panels if panel.keep}
        scene_summary_lookup = self._scene_summary_lookup(self.store._project_dir(project_id))
        panel_vision_payload_by_id: dict[str, dict[str, Any]] = {}
        panel_vision_record_by_id: dict[str, PanelVisionRecord] = {}
        panel_vision_path = project_dir / "output" / "panel_vision_final.json"
        panel_vision_payload = read_json(panel_vision_path, default=[])
        if isinstance(panel_vision_payload, list) and panel_vision_payload:
            records: list[PanelVisionRecord] = []
            for item in panel_vision_payload:
                try:
                    records.append(
                        item
                        if isinstance(item, PanelVisionRecord)
                        else PanelVisionRecord.model_validate(item)
                    )
                except Exception:
                    continue
            prepared_payloads = self.story_service._prepare_vision_panel_payloads(
                sorted(panels_by_id.values(), key=lambda panel: panel.order),
                records,
                [],
            )
            panel_vision_payload_by_id = {
                str(item.get("panel_id") or "").strip(): dict(item)
                for item in prepared_payloads
                if str(item.get("panel_id") or "").strip()
            }
            panel_vision_record_by_id = {record.panel_id: record for record in records}
        has_panel_vision_payloads = bool(panel_vision_payload_by_id)
        scene_counts = Counter(int(segment.scene_id or 0) for segment in segments)
        scene_offsets: dict[int, int] = defaultdict(int)
        units: list[dict[str, Any]] = []

        for segment in segments:
            scene_id = int(segment.scene_id or 0) or segment.order
            scene_offsets[scene_id] += 1
            panel_ids = [panel_id for panel_id in segment.panel_ids if panel_id in panels_by_id]
            panels = [panels_by_id[panel_id] for panel_id in panel_ids]
            vision_payloads = [
                panel_vision_payload_by_id[panel_id]
                for panel_id in panel_ids
                if panel_id in panel_vision_payload_by_id
            ]
            vision_records = [
                panel_vision_record_by_id[panel_id]
                for panel_id in panel_ids
                if panel_id in panel_vision_record_by_id
            ]
            raw_ocr_text = clean_ocr_text(
                " ".join(str(panel.ocr_text or "").strip() for panel in panels if str(panel.ocr_text or "").strip())
            )
            ocr_text = raw_ocr_text
            if self.story_service._text_is_noisy_ocr(ocr_text):
                ocr_text = ""
            ocr_fallback_text = clean_ocr_text(
                " ".join(self.story_service._panel_ocr_fallback_text(panel) for panel in panels)
            )
            if self.story_service._text_is_noisy_ocr(ocr_fallback_text):
                ocr_fallback_text = ""
            vision_text = clean_ocr_text(
                " ".join(str(payload.get("text") or "").strip() for payload in vision_payloads if str(payload.get("text") or "").strip())
            )
            combined_text = vision_text if has_panel_vision_payloads else (vision_text or ocr_text)
            legacy_visual_cues = " ".join(
                str(panel.visual_caption or "").strip()
                for panel in panels
                if str(panel.visual_caption or "").strip()
            ).strip()
            vision_action_beats = clean_ocr_text(
                " ".join(
                    str(payload.get("vision_action_beat") or record.action_beat or "").strip()
                    for payload, record in zip(vision_payloads, vision_records)
                    if str(payload.get("vision_action_beat") or record.action_beat or "").strip()
                )
            )
            vision_dialogue = clean_ocr_text(
                " ".join(
                    str(payload.get("vision_dialogue") or record.dialogue or "").strip()
                    for payload, record in zip(vision_payloads, vision_records)
                    if str(payload.get("vision_dialogue") or record.dialogue or "").strip()
                )
            )
            vision_caption = clean_ocr_text(
                " ".join(
                    str(payload.get("vision_caption") or record.caption or "").strip()
                    for payload, record in zip(vision_payloads, vision_records)
                    if str(payload.get("vision_caption") or record.caption or "").strip()
                )
            )
            visual_cues = vision_action_beats or legacy_visual_cues
            character_names: list[str] = []
            seen_names: set[str] = set()
            vision_character_names = [
                str(name).strip()
                for payload in vision_payloads
                for name in payload.get("character_names", []) or []
                if str(name).strip()
            ]
            for raw_name in vision_character_names:
                name = str(raw_name or "").strip()
                key = name.casefold()
                if name and key not in seen_names:
                    seen_names.add(key)
                    character_names.append(name)
            for panel in panels:
                for raw_name in getattr(panel, "character_names", None) or []:
                    name = str(raw_name or "").strip()
                    key = name.casefold()
                    if name and key not in seen_names:
                        seen_names.add(key)
                        character_names.append(name)
            salvaged_evidence = ""
            if raw_ocr_text and not has_panel_vision_payloads:
                salvaged_evidence = self.story_service._salvage_noisy_ocr_evidence(
                    raw_ocr_text,
                    story_bible=story_bible,
                    character_names=character_names,
                )
            units.append(
                {
                    "segment_id": segment.id,
                    "title": str(segment.title or "").strip(),
                    "scene_id": scene_id,
                    "sequence_in_scene": scene_offsets[scene_id],
                    "scene_unit_count": scene_counts[scene_id] or 1,
                    "panel_start": int(segment.panel_start or 0),
                    "panel_end": int(segment.panel_end or 0),
                    "panel_count": len(panel_ids),
                    "panel_ids": panel_ids,
                    "has_panel_vision": has_panel_vision_payloads,
                    "character_names": character_names,
                    "combined_text": combined_text,
                    "ocr_fallback_text": ocr_fallback_text,
                    "salvaged_evidence": salvaged_evidence,
                    "visual_cues": visual_cues,
                    "vision_dialogue": vision_dialogue,
                    "vision_caption": vision_caption,
                    "vision_action_beat": vision_action_beats,
                    "scene_summary": scene_summary_lookup.get(scene_id, ""),
                }
            )
        return units, panels_by_id

    def _scene_summary_lookup(self, project_dir: Path) -> dict[int, str]:
        payload = self._load_dict(project_dir / "output" / "scene_summaries.json")
        lookup: dict[int, str] = {}
        for item in payload.get("scenes") or []:
            if not isinstance(item, dict):
                continue
            try:
                scene_id = int(item.get("scene_id") or item.get("beat_id") or 0)
            except Exception:
                continue
            summary = str(item.get("description") or item.get("summary") or "").strip()
            if scene_id and summary:
                lookup[scene_id] = summary
        return lookup

    def _segments_from_payloads(
        self,
        segments: list[StorySegment],
        payloads: list[dict[str, Any]],
    ) -> list[StorySegment]:
        repaired: list[StorySegment] = []
        for segment, payload in zip(segments, payloads, strict=False):
            text = self.story_service._fix_sentence_boundary_text(str(payload.get("text") or ""))
            visual_only = bool(payload.get("visual_only")) if not text else False
            suppression_reason = str(payload.get("suppression_reason") or "").strip() or None
            if text:
                suppression_reason = None
            elif not suppression_reason:
                suppression_reason = segment.suppression_reason or "weak_evidence"
            repaired.append(
                segment.model_copy(
                    update={
                        "text": text,
                        "visual_only": visual_only,
                        "suppression_reason": suppression_reason,
                    }
                )
            )
        return repaired

    def _save(self, project_id: str, project_dir: Path, segments: list[StorySegment]) -> None:
        story_text = self.story_service._compose_story_text(segments)
        self.store.save_story_segments(project_id, segments, story_block=story_text)
        write_json(project_dir / "output" / "story_segments.json", [segment.model_dump(mode="json") for segment in segments])
        (project_dir / "output" / "narration_story.txt").write_text(
            (story_text.strip() + "\n") if story_text.strip() else "",
            encoding="utf-8",
        )

    def _result(
        self,
        project_id: str,
        segments: list[StorySegment],
        *,
        target_count: int,
        repaired_count: int,
        quality_report: dict[str, Any],
    ) -> StorySegmentRepairResult:
        return StorySegmentRepairResult(
            project_id=project_id,
            total_segments=len(segments),
            target_segments=target_count,
            repaired_segments=repaired_count,
            spoken_segments=sum(1 for segment in segments if segment.text.strip()),
            visual_only_segments=sum(1 for segment in segments if segment.visual_only),
            quality_report=quality_report,
        )

    def _load_dict(self, path: Path) -> dict[str, Any]:
        payload = read_json(path, default={})
        return payload if isinstance(payload, dict) else {}
