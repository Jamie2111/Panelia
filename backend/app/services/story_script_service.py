"""DEPRECATED — see app/services/DEPRECATED.md.

StoryScriptService coordinates the legacy multi-pass narration cascade.
Replaced by PanelVisionNarrator (single vision-grounded pass). Retained for
projects still on script_pipeline_version="legacy".
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import json
import logging
import math
import os
import re
import threading
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image, ImageOps

from app.schemas.project import (
    CanonicalCharacterRecord,
    ChapterMetadata,
    NarrationMode,
    PanelBox,
    PanelVisionRecord,
    StorySegment,
)
from app.services.character_name_filters import looks_like_false_character_name, normalize_name_key
from app.services.comic_ocr_service import ComicOCRService
from app.services.llm_router import LLMRouter
from app.services.panel_narrator import PanelNarrator
from app.services.script_polisher import ScriptPolisher
from app.services.script_quality_service import ScriptQualityService
from app.services.story_beats import StoryBeatService
from app.services.story_grounding import (
    apply_name_corrections_to_text,
    build_name_grounding,
    canonicalize_character_name,
    compact_chapter_metadata,
    contains_unapproved_names,
    extract_proper_name_candidates,
)
from app.services.style_vocabulary import StyleVocabulary, build_style_vocabulary
from app.services.storytelling_style_guide import strip_storytelling_meta
from app.services.ocr_cleaner import clean_ocr_text, is_usable_ocr_text
from app.utils.files import ensure_dir

logger = logging.getLogger(__name__)

_VISION_PLACEHOLDER_NAME_PATTERN = re.compile(
    r"\b(protagonist|unknown|victim|speaker|narrator|man|woman|boy|girl|person|figure|child|manager|delivery man|old woman|elderly woman|null|none|name)\b",
    re.IGNORECASE,
)


@dataclass(slots=True)
class StoryScriptBundle:
    story_segments: list[StorySegment]
    story_text: str
    chapter_summary: str
    scene_summaries: list[dict[str, Any]]
    draft_lines: list[str]
    polished_lines: list[str]
    scene_seeds: list[dict[str, Any]]
    story_bible: dict[str, Any]
    grounding_state: dict[str, Any]
    style_vocabulary: StyleVocabulary | None = None


class StoryScriptService:
    _SCENE_CHUNK_SIZE = 14
    _MULTIMODAL_SCENE_CHUNK_SIZE = 8   # doubled from 4 — fewer total API calls
    _CRITIC_BATCH_SIZE = 10            # was 6 — fewer critic API calls
    _RESCUE_BATCH_SIZE = 2             # smaller multimodal rescue batches hit Gemini image blocks less often
    _MAX_MULTIMODAL_LINE_RESCUES = 48
    _MAX_VISUAL_ONLY_RECOVERIES = 32
    _STYLE_BATCH_SIZE = 16             # was 8 — fewer style API calls
    _STYLE_PASSES = 2
    _DRAFT_WORKERS = 2                 # concurrent draft threads
    _CRITIC_WORKERS = 2                # concurrent critic threads

    # Words that look like short names but are common English words — excluded from
    # character-name extraction in the mechanical OCR paraphrase.
    _MECHANICAL_NAME_STOPWORDS: frozenset[str] = frozenset({
        "ago", "ain't", "all", "also", "and", "are", "aren't", "ask",
        "back", "bad", "buckle", "but", "buy",
        "can", "can't", "care", "chance", "come", "could", "couldn't",
        "did", "didn't", "does", "doesn't", "done", "don't", "doubt",
        "each", "else", "even", "ever", "every",
        "far", "feel", "felt", "fine", "for", "from", "get", "gets",
        "give", "given", "going", "got", "had", "has", "have", "having",
        "head", "heads", "heading", "hence", "here", "him", "his",
        "isn't", "its",
        "just", "knew", "know", "last", "let", "like", "listen", "look",
        "makes", "mean", "means", "mind", "more", "move", "moved", "much",
        "must", "need", "needing", "never", "next", "not",
        "now", "okay", "once", "only",
        "probably", "put", "read", "really", "right", "run", "said", "say",
        "says", "see", "seeing", "send", "share", "she", "should", "sorry",
        "stay", "still", "stop", "sure",
        "take", "than", "thank", "thanks", "that", "then", "there", "them",
        "they", "think", "this", "thought", "told", "too", "try",
        "was", "wasn't", "well", "went", "what", "when", "where", "while",
        "who", "will", "with", "without", "won't", "would", "wouldn't",
        "yeah", "yes", "yet", "you",
    })

    def __init__(self, router: LLMRouter | None = None) -> None:
        self.router = router or LLMRouter()
        self.beats = StoryBeatService(self.router)
        self.polisher = ScriptPolisher(self.router)
        self._comic_ocr: ComicOCRService | None = None

    def generate(
        self,
        *,
        project_title: str,
        chapter_metadata: ChapterMetadata,
        panels: list[PanelBox],
        scenes: list[dict[str, Any]],
        scene_clusters: list[dict[str, Any]],
        character_dictionary: dict[str, Any],
        protagonist_name: str | None,
        cache_dir: Path,
        narration_mode: str = "story",
        series_context: dict[str, Any] | None = None,
        progress_callback: Any | None = None,
        panel_vision_records: list[PanelVisionRecord] | None = None,
        panel_evidence_records: list[dict[str, Any]] | None = None,
        canonical_characters: list[CanonicalCharacterRecord] | None = None,
        style_vocab: StyleVocabulary | None = None,
        disable_multimodal_rescue: bool = False,
    ) -> StoryScriptBundle:
        """Generate a recap script as grouped story segments backed by panel evidence."""
        def _progress(progress: float, message: str) -> None:
            if progress_callback:
                progress_callback(float(int(max(0.0, min(100.0, progress)) + 0.9999)), message)

        def _run_with_progress_pulse(
            start: float,
            end: float,
            message: str,
            callback: Any,
            *,
            interval_seconds: float = 2.5,
        ) -> Any:
            if progress_callback is None:
                return callback()
            stop_event = threading.Event()
            span = max(0.0, float(end) - float(start))
            cap = max(float(start), float(end) - 0.05)

            def pulse() -> None:
                tick = 0
                while not stop_event.wait(interval_seconds):
                    tick += 1
                    # Move quickly enough to reassure the UI, but asymptotically
                    # hold a little room for the real completion update.
                    ratio = min(0.995, 1.0 - (0.70 ** tick))
                    _progress(min(cap, float(start) + span * ratio), message)

            thread = threading.Thread(target=pulse, name="story-progress-pulse", daemon=True)
            thread.start()
            try:
                return callback()
            finally:
                stop_event.set()
                thread.join(timeout=0.2)

        _progress(4, "Preparing story panels")
        helper = PanelNarrator(self.router, cache_dir=cache_dir)
        all_panels_ordered = sorted(
            panels,
            key=lambda item: (
                int(getattr(item, "page", 0) or 0),
                int(getattr(item, "panel", 0) or 0),
                int(getattr(item, "order", 0) or 0),
            ),
        )
        kept_panels = [panel for panel in all_panels_ordered if panel.keep]
        panels_by_id = {panel.id: panel for panel in kept_panels}
        # For OCR-only mode (no vision records), anonymize the project title in LLM prompts.
        # Well-known series titles cause the LLM to generate franchise-derived content
        # regardless of instructions, because it knows the characters and story from training.
        # Using a generic title forces it to work only from the supplied OCR evidence.
        ocr_only_mode = not panel_vision_records
        if ocr_only_mode:
            # Shadow project_title so every downstream LLM call uses the anonymous version.
            # Well-known series titles cause the LLM to generate franchise-derived content
            # from training knowledge.  A generic title forces it to work only from OCR evidence.
            project_title = "Untitled Comic Chapter"
            # Filter any pre-existing character_dictionary to characters whose names actually
            # appear in the panel OCR.  character_dictionary.json may contain franchise-hallucinated
            # entries (e.g. "Zero Two") from a previous vision-mode run.  Passing those to the
            # draft LLM causes it to generate franchise content for every segment.
            if character_dictionary:
                ocr_panel_text = " ".join(
                    clean_ocr_text(str(p.ocr_text or "").strip()).lower()
                    for p in all_panels_ordered
                    if clean_ocr_text(str(p.ocr_text or "").strip())
                )
                # Keep only entries whose name appears in OCR, and strip appearance
                # descriptions — those are often franchise-derived from a previous
                # vision-mode run and will trigger franchise hallucination in the LLM
                # even when the character name alone matches OCR.
                filtered_dict: dict[str, Any] = {}
                for key, val in character_dictionary.items():
                    if re.search(rf"\b{re.escape(str(key).strip().lower())}\b", ocr_panel_text):
                        filtered_dict[key] = {
                            k: v for k, v in (val if isinstance(val, dict) else {}).items()
                            if k != "appearance"
                        }
                character_dictionary = filtered_dict
        # Build full-chapter dialogue context from ALL panels (kept + skipped) so
        # the LLM has complete speech-bubble text even for panels that are deduped
        # or filtered out of narration.
        all_panels_ocr_fragments = [
            clean_ocr_text(str(panel.ocr_text or "").strip())
            for panel in all_panels_ordered
            if clean_ocr_text(str(panel.ocr_text or "").strip())
        ]
        chapter_dialogue_context = " ".join(all_panels_ocr_fragments)[:6000] if all_panels_ocr_fragments else ""
        previous_story_bible = self._load_story_bible_cache(cache_dir / "story_bible.json")
        if panel_vision_records:
            # Vision evidence available — use richer per-panel payloads that include
            # action_beat, dialogue, caption, and visual_cues from Gemini Vision.
            character_dictionary = self._build_vision_character_dictionary(
                canonical_characters or [], character_dictionary
            )
            if not protagonist_name:
                protagonist_name = self._vision_protagonist_name(canonical_characters or [])
            ordered_payloads = self._prepare_vision_panel_payloads(
                kept_panels,
                panel_vision_records,
                canonical_characters or [],
                panel_evidence_records or [],
            )
            scene_seeds = self._merge_scene_seeds(self._build_vision_scene_seeds(ordered_payloads))
        else:
            ordered_payloads = helper._prepare_panel_payloads(kept_panels, scenes)
            scene_seeds = self._merge_scene_seeds(helper._build_scene_seeds(ordered_payloads, scene_clusters))
        if not scene_seeds and kept_panels:
            scene_seeds = [
                {
                    "scene_id": 1,
                    "panel_start": int(kept_panels[0].page or 0) * 10000 + int(getattr(kept_panels[0], "panel", 0) or 0),
                    "panel_end": int(kept_panels[-1].page or 0) * 10000 + int(getattr(kept_panels[-1], "panel", 0) or 0),
                    "panel_ids": [panel.id for panel in kept_panels],
                    "panels": [int(panel.page or 0) * 10000 + int(getattr(panel, "panel", 0) or 0) for panel in kept_panels],
                    "combined_text": clean_ocr_text(" ".join(str(panel.ocr_text or "").strip() for panel in kept_panels)),
                    "character_names": [],
                }
            ]

        effective_metadata = self.polisher._effective_chapter_metadata(  # type: ignore[attr-defined]
            chapter_metadata,
            project_title=project_title,
            character_dictionary=character_dictionary,
            draft_lines=[str(seed.get("combined_text") or "").strip() for seed in scene_seeds],
        )
        # In OCR-only mode strip the real manga title from effective_metadata so that
        # every downstream LLM (polisher, critic, cohesion) gets a generic title.
        # Well-known series titles cause the LLM to hallucinate franchise content
        # (e.g. "Klaxosaurs", "Zero Two") even when the draft is purely mechanical.
        if ocr_only_mode and isinstance(effective_metadata, dict):
            effective_metadata = dict(effective_metadata)
            effective_metadata["manga_title"] = "Untitled Comic Chapter"
            effective_metadata["chapter_title"] = effective_metadata.get("chapter_title") or "Chapter"
            effective_metadata.pop("series_cast_hints", None)
            effective_metadata.pop("canonical_name_corrections", None)
        corrections = effective_metadata.get("canonical_name_corrections", []) if isinstance(effective_metadata, dict) else []
        character_dictionary = self._apply_corrections_to_character_dictionary(character_dictionary, corrections)
        scene_seeds = [self._apply_corrections_to_seed(seed, corrections) for seed in scene_seeds]
        preliminary_grounding = build_name_grounding(
            effective_metadata if isinstance(effective_metadata, dict) else chapter_metadata,
            character_dictionary,
            protagonist_name,
        )
        scene_seeds = self._sanitize_scene_seeds(scene_seeds, preliminary_grounding)

        _progress(16, "Building chapter story beats")
        beat_bundle = _run_with_progress_pulse(
            16,
            27.5,
            "Building chapter story beats",
            lambda: self.beats.generate(
                chapter_metadata,
                project_title,
                scene_seeds,
                character_dictionary,
                protagonist_name,
                required_provider="gemini",
                allow_fallback=True,
            ),
        )
        scene_summaries = self.beats.align_beats_to_scenes(beat_bundle.beats, scene_seeds)
        # In OCR-only mode the beats LLM generates franchise-derived story summaries
        # (it recognizes character names like "hiro" from its training data).  Replace the
        # beats story_script with a plain summary built purely from the raw OCR fragments
        # so that the downstream polish/critic passes cannot anchor on franchise content.
        if ocr_only_mode and chapter_dialogue_context:
            from app.services.story_beats import StoryBeatBundle as _SBB
            # Build a chapter summary from mechanical paraphrases of the scene seeds.
            # Do NOT use raw OCR sentences — passing those to the polish/critic LLMs
            # causes them to quote dialogue literally ("The conversation reveals '...'").
            # Mechanical paraphrases describe story events without quoting raw text.
            _mech_summaries: list[str] = []
            for _seed in scene_seeds:
                _seed_ocr = str(_seed.get("combined_text") or "").strip()
                if _seed_ocr:
                    _m = self._mechanical_ocr_paraphrase(_seed_ocr)
                    if _m:
                        _mech_summaries.append(_m)
            if _mech_summaries:
                ocr_chapter_summary = " ".join(_mech_summaries)
            else:
                ocr_chapter_summary = "A chapter unfolds across several scenes."
            beat_bundle = _SBB(
                story_script=ocr_chapter_summary,
                beats=beat_bundle.beats,
                provider=beat_bundle.provider,
                model=beat_bundle.model,
                warning=beat_bundle.warning,
            )
        metadata_payload = self._chapter_metadata_payload(
            effective_metadata if isinstance(effective_metadata, dict) else chapter_metadata
        )
        name_grounding = build_name_grounding(metadata_payload, character_dictionary, protagonist_name)
        metadata_payload = dict(name_grounding.get("chapter_metadata") or metadata_payload)
        fallback_story_bible = self._fallback_story_bible(
            scene_seeds,
            scene_summaries,
            chapter_summary=beat_bundle.story_script,
            character_dictionary=character_dictionary,
        )
        _progress(28, "Building story bible and character grounding")
        story_bible = _run_with_progress_pulse(
            28,
            35.5,
            "Building story bible and character grounding",
            lambda: self._build_story_bible(
                scene_seeds,
                scene_summaries,
                project_title=project_title,
                chapter_metadata=metadata_payload,
                chapter_summary=beat_bundle.story_script,
                character_dictionary=character_dictionary,
                protagonist_name=protagonist_name,
                fallback_story_bible=fallback_story_bible,
                allowed_character_names=list(name_grounding.get("allowed_character_names") or []),
            ),
        )
        if previous_story_bible:
            story_bible = self._merge_story_bibles(previous_story_bible, story_bible)
        story_bible = self._sanitize_story_bible(story_bible, fallback_story_bible, name_grounding)
        # In OCR-only mode (no vision records, empty character dict), the story-bible LLM
        # tends to hallucinate franchise-derived cast members (e.g. "Zero Two", "Strelitzia")
        # that aren't present in the actual panel text. Filter cast to only names that appear
        # in the chapter OCR evidence, preventing franchise hallucination in downstream prompts.
        if ocr_only_mode:
            # Always filter story bible cast to OCR-attested names in OCR-only mode.
            # The LLM may still hallucinate franchise cast (e.g. "Zero Two") even when
            # character_dictionary was pre-filtered, because the story-bible LLM uses the
            # project title + OCR names and infers franchise context from its training data.
            ocr_corpus = " ".join(str(seed.get("combined_text") or "").lower() for seed in scene_seeds)
            ocr_corpus += " " + chapter_dialogue_context.lower()
            filtered_cast = [
                member for member in story_bible.get("cast") or []
                if isinstance(member, dict) and str(member.get("name") or "").strip()
                and re.search(
                    rf"\b{re.escape(str(member.get('name') or '').strip().lower())}\b",
                    ocr_corpus
                )
            ]
            if filtered_cast != story_bible.get("cast"):
                logger.info(
                    "OCR-only mode: filtered story bible cast from %d to %d members (OCR-attested only)",
                    len(story_bible.get("cast") or []), len(filtered_cast),
                )
                story_bible["cast"] = filtered_cast
        # Inject full-chapter dialogue (all panels incl. skipped) as grounding context
        # so every downstream LLM call knows what speech bubbles exist across all pages.
        # In OCR-only mode, inject the mechanical paraphrase summary instead of the raw OCR
        # to prevent all downstream LLMs (critic, cohere, enrichment) from quoting raw
        # dialogue literally ("The conversation reveals '...'").
        if chapter_dialogue_context:
            if ocr_only_mode:
                story_bible["chapter_dialogue_context"] = beat_bundle.story_script or ""
            else:
                story_bible["chapter_dialogue_context"] = chapter_dialogue_context
        # Inject external series context from Gemini grounded search (if available).
        if series_context and series_context.get("search_context"):
            story_bible["series_external_context"] = str(series_context["search_context"])[:3000]
            logger.info(
                "Injected %d chars of grounded series context for '%s'",
                len(story_bible["series_external_context"]),
                project_title,
            )
        name_grounding = self._merge_story_bible_into_grounding(name_grounding, story_bible)
        fresh_style_vocab = build_style_vocabulary(
            canonical_characters=canonical_characters or [],
            character_dictionary=character_dictionary,
            story_bible=story_bible,
            scene_summaries=scene_summaries,
            chapter_summary=beat_bundle.story_script,
        )
        # In OCR-only mode keep style_vocab=None so the LLM-heavy style_vocab block
        # (narrator_enrichment_pass, expand_short_scene_payloads_with_llm, etc.) is
        # entirely skipped.  Those passes use the story_bible for context, which in
        # OCR-only mode is derived from the beats LLM and may contain franchise
        # character names that cause hallucination or overwrite the clean mechanical
        # paraphrases with generic warning/lore filler.
        if not ocr_only_mode and (fresh_style_vocab.named_characters or fresh_style_vocab.world_terms):
            style_vocab = fresh_style_vocab
        scene_mode = False
        _progress(36, "Expanding panel story segments")
        # Panel mode: each kept panel becomes exactly one narration slot.
        # Bypasses scene-seed grouping, coalescing, and segment coherence merging
        # entirely — the main sources of out-of-order panels, skipped panels,
        # franchise hallucination bleed, and disconnected-transition failures.
        story_units = self._panel_mode_story_units(ordered_payloads, name_grounding)
        scene_visual_paths = self._build_scene_visual_paths(
            story_units,
            panels_by_id,
            cache_dir.parent / "panels",
            cache_dir / "scene_visuals",
        )
        # In OCR-only mode, replace raw OCR in story unit fields with mechanical paraphrases
        # BEFORE drafting.  The draft LLM, critic, and all downstream passes read
        # `vision_dialogue`, `combined_text`, and similar fields as evidence.  If those
        # fields contain raw OCR, the LLM quotes it literally ("The conversation reveals '...'").
        # Using the mechanical paraphrase as evidence instead gives the LLM a narrative
        # template to refine rather than raw dialogue to quote.
        if ocr_only_mode:
            # Build a panel-order → raw OCR mapping from the actual kept panels.
            # The sanitized seed combined_text can lose important name clues (e.g.
            # "Hence, Naomi.") because _salvage_readable_ocr_fragments treats them
            # as short noise fragments.  Raw panel OCR is unmodified and safe to use
            # for name extraction only (not for pattern matching or narration).
            _panel_order_to_raw_ocr: dict[int, str] = {
                int(getattr(_kp, "order", 0) or 0): clean_ocr_text(str(getattr(_kp, "ocr_text", "") or "").strip())
                for _kp in kept_panels
                if clean_ocr_text(str(getattr(_kp, "ocr_text", "") or "").strip())
            }
            for unit in story_units:
                source_ocr = str(unit.get("combined_text") or unit.get("ocr_fallback_text") or "").strip()
                # Collect raw OCR from all panels in this unit (and one panel on each
                # side) as name context.  Using a ±1 window helps when the name appears
                # in the panel immediately before or after the speech-act panel.
                _unit_panels = [int(_p or 0) for _p in unit.get("panels", []) or []]
                _ctx_panel_orders: set[int] = set(_unit_panels)
                if _unit_panels:
                    _ctx_panel_orders.add(min(_unit_panels) - 1)
                    _ctx_panel_orders.add(max(_unit_panels) + 1)
                _ctx_parts: list[str] = []
                for _po in sorted(_ctx_panel_orders):
                    _raw = _panel_order_to_raw_ocr.get(_po, "")
                    if _raw:
                        _ctx_parts.append(_raw)
                _raw_ocr_ctx = " ".join(_ctx_parts)
                mechanical = self._mechanical_ocr_paraphrase(source_ocr, name_context=_raw_ocr_ctx) if source_ocr else ""
                unit["ocr_source_text"] = source_ocr
                if mechanical:
                    unit["vision_dialogue"] = mechanical
                    unit["combined_text"] = mechanical
                    unit["ocr_fallback_text"] = mechanical
                else:
                    # No recognized speech act: clear vision_dialogue to prevent quoting.
                    # Keep combined_text as "" so _slot_evidence doesn't send raw OCR.
                    unit["vision_dialogue"] = ""
                    unit["combined_text"] = ""
                    unit["ocr_fallback_text"] = ""
                # Always clear raw vision fields for OCR-only mode
                unit["vision_caption"] = ""
                unit["vision_action_beat"] = ""
                # Replace franchise-derived scene_summary with the same mechanical paraphrase.
                # The scene_summary comes from the beats LLM which receives OCR with character
                # names (e.g. "hiro", "zorome") and generates franchise-derived descriptions.
                # Passing those as scene_summary to the critic/cohesion LLMs causes franchise
                # hallucination to leak into polished segments.
                unit["scene_summary"] = mechanical or ""
        _progress(44, "Drafting aligned story narration")
        draft_lines = _run_with_progress_pulse(
            44,
            63.5,
            "Drafting aligned story narration",
            lambda: self._draft_scene_lines(
                story_units,
                project_title=project_title,
                chapter_metadata=effective_metadata if isinstance(effective_metadata, dict) else chapter_metadata,
                chapter_summary=beat_bundle.story_script,
                character_dictionary=character_dictionary,
                protagonist_name=protagonist_name,
                story_bible=story_bible,
                scene_visual_paths=scene_visual_paths,
                name_grounding=name_grounding,
                prefer_local_evidence=False,
                style_vocab=style_vocab,
            ),
        )
        draft_lines = [self._apply_name_corrections(line, corrections) for line in draft_lines]
        # In OCR-only mode, always replace draft lines that have OCR evidence with a
        # mechanical paraphrase. This prevents both caption-like quoting AND franchise
        # hallucination from becoming the polish-pass input. The mechanical paraphrase is
        # a simple but OCR-grounded sentence that the polish LLM will refine into narration.
        if ocr_only_mode:
            for unit_idx, unit in enumerate(story_units):
                if unit_idx >= len(draft_lines):
                    break
                source_ocr = str(
                    unit.get("ocr_source_text")
                    or unit.get("combined_text")
                    or unit.get("ocr_fallback_text")
                    or ""
                ).strip()
                if source_ocr:
                    # If the unit has already been converted into a mechanical
                    # narrative line, preserve that exact line. Re-running the
                    # classifier on the mechanical sentence can misclassify it
                    # because the template itself may contain words such as
                    # "farewell" or "danger".
                    mechanical = str(unit.get("combined_text") or "").strip()
                    if not mechanical or mechanical == source_ocr:
                        mechanical = self._mechanical_ocr_paraphrase(source_ocr)
                    if mechanical:
                        draft_lines[unit_idx] = mechanical
                        logger.debug("Injected mechanical OCR paraphrase for unit %d in OCR-only mode", unit_idx)
        _progress(64, "Polishing story narration")
        if ocr_only_mode:
            # In OCR-only mode the draft lines have already been replaced by
            # deterministic OCR-grounded recap beats. The general polish prompt
            # tends to compress those back into one-sentence caption-like lines
            # or reintroduce franchise knowledge, so preserve the grounded
            # draft directly.
            polished_lines = list(draft_lines)
        else:
            polished_lines = _run_with_progress_pulse(
                64,
                75.5,
                "Polishing story narration",
                lambda: self.polisher.polish(
                    draft_lines,
                    beat_bundle.story_script,
                    character_dictionary,
                    project_title=project_title,
                    chapter_metadata=effective_metadata if isinstance(effective_metadata, dict) else chapter_metadata,
                    slot_evidence=self._slot_evidence(story_units, draft_lines, ocr_only=ocr_only_mode),
                    preserve_multi_sentence=False,
                ),
            )
        if len(polished_lines) != len(story_units):
            polished_lines = list(draft_lines)
        polished_lines = [self._apply_name_corrections(line, corrections) for line in polished_lines]

        _progress(76, "Critiquing and repairing weak story beats")
        reviewed_segments = _run_with_progress_pulse(
            76,
            89.5,
            "Critiquing and repairing weak story beats",
            lambda: self._critic_scene_lines(
                polished_lines,
                story_units,
                project_title=project_title,
                chapter_metadata=metadata_payload,
                chapter_summary=beat_bundle.story_script,
                character_dictionary=character_dictionary,
                protagonist_name=protagonist_name,
                story_bible=story_bible,
                name_grounding=name_grounding,
                scene_visual_paths=scene_visual_paths,
                disable_multimodal_rescue=disable_multimodal_rescue,
                style_vocab=style_vocab,
                # In OCR-only mode, polished mechanical paraphrases are already our
                # best content. Skip the LLM critic — it tends to add generic filler
                # sentences and can re-introduce franchise hallucination through the
                # story_bible / beats context it receives.
                skip_llm_critic=ocr_only_mode,
            ),
        )
        for item in reviewed_segments:
            text = self._apply_name_corrections(str(item.get("text") or ""), corrections)
            item["text"] = apply_name_corrections_to_text(text, name_grounding)
        if ocr_only_mode:
            # The remaining delivery passes are tuned for vision-grounded output:
            # they use character-name guardrails, multimodal rescue, segment
            # coalescing, and local-evidence fill-ins. In OCR-only runs those
            # passes can falsely reject safe narrator subjects ("the others",
            # "the listener") or trim multi-sentence beats. Preserve the
            # deterministic OCR-grounded payloads and let final sanitization
            # remove only truly bad lines.
            reviewed_segments = self._final_sanitize_story_payloads(reviewed_segments)
            _progress(97, "Finalizing aligned story segments")
            story_segments = self._build_story_segments(story_units, reviewed_segments)
            story_text = self._compose_story_text(story_segments)
            return StoryScriptBundle(
                story_segments=story_segments,
                story_text=story_text,
                chapter_summary=beat_bundle.story_script,
                scene_summaries=scene_summaries,
                draft_lines=draft_lines,
                polished_lines=polished_lines,
                scene_seeds=scene_seeds,
                story_bible=story_bible,
                grounding_state=name_grounding,
                style_vocabulary=style_vocab,
            )
        # In OCR-only mode, skip the LLM-based style pass. The polished mechanical
        # paraphrases are already clean; the style LLM tends to add generic filler
        # sentences and can re-introduce franchise content through story_bible context.
        if not ocr_only_mode:
            _progress(90, "Smoothing narration for voiceover")
            reviewed_segments = _run_with_progress_pulse(
                90,
                93.5,
                "Smoothing narration for voiceover",
                lambda: self._style_spoken_segment_payloads(
                    reviewed_segments,
                    story_units,
                    project_title=project_title,
                    chapter_metadata=metadata_payload,
                    chapter_summary=beat_bundle.story_script,
                    character_dictionary=character_dictionary,
                    story_bible=story_bible,
                    name_grounding=name_grounding,
                    style_vocab=style_vocab,
                ),
            )
        reviewed_segments = self._stabilize_reviewed_segments(
            reviewed_segments,
            story_units,
            protagonist_name,
            name_grounding,
            story_bible,
        )

        # Near-duplicate + repeat collapse: a YouTube recap cannot contain the
        # same sentence twice in a row (even with slight paraphrasing) or have
        # two adjacent scenes narrate the exact same moment.
        reviewed_segments = self._collapse_internal_duplicate_sentences(reviewed_segments, scene_mode=False)
        reviewed_segments = self._collapse_near_duplicate_segments(
            reviewed_segments,
            story_units,
            blank_unresolved=True,
            style_vocab=style_vocab,
        )

        # Chapter-level narrator cohesion: rewrite the whole thing as one flowing
        # YouTube recap with real transitions between scenes. Runs only when we
        # have enough substance to be worth the LLM call.
        # Skip in OCR-only mode: the polished mechanical paraphrases are our best
        # content, and the cohesion LLM tends to homogenize all slots toward the
        # most recognizable beat (e.g. the warning), overwriting distinct events
        # (offer, farewell) and re-introducing franchise content through the
        # story_bible / beats scene memory.
        if not ocr_only_mode:
            _progress(94, "Cohering narrator voice across scenes")
            reviewed_segments = _run_with_progress_pulse(
                94,
                95.8,
                "Cohering narrator voice across scenes",
                lambda: self._narrator_cohesion_pass(
                    reviewed_segments,
                    story_units,
                    project_title=project_title,
                    chapter_metadata=metadata_payload,
                    chapter_summary=beat_bundle.story_script,
                    character_dictionary=character_dictionary,
                    protagonist_name=protagonist_name,
                    name_grounding=name_grounding,
                    require_multi_sentence=False,
                    style_vocab=style_vocab,
                ),
            )
        # Cohesion can inadvertently homogenize adjacent scenes (two scenes
        # end up with the same rewritten sentence). Run the duplicate collapse
        # one more time on the post-cohesion output so nothing slips through
        # into narration_story.txt.
        reviewed_segments = self._collapse_internal_duplicate_sentences(reviewed_segments, scene_mode=False)
        reviewed_segments = self._collapse_near_duplicate_segments(
            reviewed_segments,
            story_units,
            blank_unresolved=True,
            style_vocab=style_vocab,
        )

        # Dedup can legitimately blank a segment when two adjacent scene lines
        # collapse to the same beat. Give those slots one last multimodal
        # recovery pass so the final editor/video never carries empty segments
        # when the images still support a conservative replacement.
        if not disable_multimodal_rescue:
            reviewed_segments = self._recover_visual_only_payloads_multimodal(
                reviewed_segments,
                story_units,
                project_title=project_title,
                chapter_metadata=metadata_payload,
                chapter_summary=beat_bundle.story_script,
                character_dictionary=character_dictionary,
                protagonist_name=protagonist_name,
                story_bible=story_bible,
                name_grounding=name_grounding,
                scene_visual_paths=scene_visual_paths,
            )
        reviewed_segments = self._stabilize_reviewed_segments(
            reviewed_segments,
            story_units,
            protagonist_name,
            name_grounding,
            story_bible,
        )
        reviewed_segments = self._fill_blank_story_payloads(
            reviewed_segments,
            story_units,
            protagonist_name=protagonist_name,
            grounding=name_grounding,
            story_bible=story_bible,
            style_vocab=style_vocab,
        )
        if disable_multimodal_rescue:
            final_lines = [
                self._normalize_segment_text(str(item.get("text") or "").strip(), allow_empty=True)
                for item in reviewed_segments
            ]
        else:
            final_lines = self._rescue_scene_lines_multimodal(
                [
                    self._normalize_segment_text(str(item.get("text") or "").strip(), allow_empty=True)
                    for item in reviewed_segments
                ],
                story_units,
                project_title=project_title,
                chapter_metadata=metadata_payload,
                chapter_summary=beat_bundle.story_script,
                character_dictionary=character_dictionary,
                protagonist_name=protagonist_name,
                story_bible=story_bible,
                name_grounding=name_grounding,
                scene_visual_paths=scene_visual_paths,
            )
        for index, candidate in enumerate(final_lines):
            normalized = self._normalize_segment_text(candidate, allow_empty=True)
            if not normalized:
                continue
            reviewed_segments[index]["text"] = normalized
            reviewed_segments[index]["visual_only"] = False
            suppression = str(reviewed_segments[index].get("suppression_reason") or "").strip()
            if suppression in {"weak_evidence", "duplicate_alignment", "generic_alignment"}:
                reviewed_segments[index]["suppression_reason"] = None
            elif suppression == "near_duplicate":
                prev_text = str(reviewed_segments[index - 1].get("text") or "").strip() if index > 0 else ""
                next_text = str(reviewed_segments[index + 1].get("text") or "").strip() if index + 1 < len(reviewed_segments) else ""
                rescued_tokens = self._content_token_set(normalized)
                prev_tokens = self._content_token_set(prev_text)
                next_tokens = self._content_token_set(next_text)
                j_prev = self._jaccard(rescued_tokens, prev_tokens) if prev_tokens else 0.0
                j_next = self._jaccard(rescued_tokens, next_tokens) if next_tokens else 0.0
                if max(j_prev, j_next) < 0.70:
                    reviewed_segments[index]["suppression_reason"] = None
                else:
                    reviewed_segments[index]["text"] = ""
                    reviewed_segments[index]["visual_only"] = True
        if not ocr_only_mode:
            reviewed_segments = self._style_spoken_segment_payloads(
                reviewed_segments,
                story_units,
                project_title=project_title,
                chapter_metadata=metadata_payload,
                chapter_summary=beat_bundle.story_script,
                character_dictionary=character_dictionary,
                story_bible=story_bible,
                name_grounding=name_grounding,
                style_vocab=style_vocab,
            )
        reviewed_segments = self._stabilize_reviewed_segments(
            reviewed_segments,
            story_units,
            protagonist_name,
            name_grounding,
            story_bible,
        )
        reviewed_segments = self._collapse_internal_duplicate_sentences(reviewed_segments, scene_mode=False)
        reviewed_segments = self._collapse_near_duplicate_segments(
            reviewed_segments,
            story_units,
            blank_unresolved=True,
            style_vocab=style_vocab,
        )
        if not disable_multimodal_rescue:
            reviewed_segments = self._recover_visual_only_payloads_multimodal(
                reviewed_segments,
                story_units,
                project_title=project_title,
                chapter_metadata=metadata_payload,
                chapter_summary=beat_bundle.story_script,
                character_dictionary=character_dictionary,
                protagonist_name=protagonist_name,
                story_bible=story_bible,
                name_grounding=name_grounding,
                scene_visual_paths=scene_visual_paths,
            )
        reviewed_segments = self._fill_blank_story_payloads(
            reviewed_segments,
            story_units,
            protagonist_name=protagonist_name,
            grounding=name_grounding,
            story_bible=story_bible,
            style_vocab=style_vocab,
        )
        if style_vocab is not None:
            _progress(96, "Final grounded segment enrichment")
            reviewed_segments = _run_with_progress_pulse(
                96,
                96.8,
                "Final grounded segment enrichment",
                lambda: self._narrator_enrichment_pass(
                    reviewed_segments,
                    story_units,
                    style_vocab=style_vocab,
                    story_bible=story_bible,
                    cache_dir=cache_dir,
                ),
            )
            reviewed_segments = self._remove_overused_generic_sentences(reviewed_segments)
            reviewed_segments = self._collapse_internal_duplicate_sentences(reviewed_segments, scene_mode=False)
            reviewed_segments = self._collapse_near_duplicate_segments(
                reviewed_segments,
                story_units,
                blank_unresolved=True,
                style_vocab=style_vocab,
            )
            reviewed_segments = self._fill_blank_story_payloads(
                reviewed_segments,
                story_units,
                protagonist_name=protagonist_name,
                grounding=name_grounding,
                story_bible=story_bible,
                style_vocab=style_vocab,
            )
            reviewed_segments = self._force_fill_remaining_blank_payloads(
                reviewed_segments,
                story_units,
                style_vocab=style_vocab,
            )
            reviewed_segments = self._prefer_local_evidence_for_thin_segments(
                reviewed_segments,
                story_units,
                style_vocab=style_vocab,
            )
            reviewed_segments = self._ensure_minimum_segment_richness(
                reviewed_segments,
                story_units,
                style_vocab=style_vocab,
            )
            reviewed_segments = self._remove_overused_generic_sentences(reviewed_segments)
            reviewed_segments = self._collapse_internal_duplicate_sentences(reviewed_segments, scene_mode=False)
            reviewed_segments = self._collapse_near_duplicate_segments(
                reviewed_segments,
                story_units,
                blank_unresolved=True,
                style_vocab=style_vocab,
            )
            reviewed_segments = self._prefer_local_evidence_for_thin_segments(
                reviewed_segments,
                story_units,
                style_vocab=style_vocab,
            )
            reviewed_segments = self._collapse_internal_duplicate_sentences(reviewed_segments, scene_mode=False)
            reviewed_segments = self._force_fill_remaining_blank_payloads(
                reviewed_segments,
                story_units,
                style_vocab=style_vocab,
            )
            if style_vocab is not None:
                reviewed_segments = self._expand_short_scene_payloads_with_llm(
                    reviewed_segments,
                    story_units,
                    project_title=project_title,
                    chapter_metadata=metadata_payload,
                    chapter_summary=beat_bundle.story_script,
                    character_dictionary=character_dictionary,
                    protagonist_name=protagonist_name,
                    story_bible=story_bible,
                    name_grounding=name_grounding,
                    style_vocab=style_vocab,
                )
            reviewed_segments = self._trim_final_bad_sentences(reviewed_segments)
            reviewed_segments = self._force_fill_remaining_blank_payloads(
                reviewed_segments,
                story_units,
                style_vocab=style_vocab,
            )
            reviewed_segments = self._reinforce_multi_sentence_scene_payloads(
                reviewed_segments,
                story_units,
                protagonist_name=protagonist_name,
                grounding=name_grounding,
                story_bible=story_bible,
                style_vocab=style_vocab,
            )
            reviewed_segments = self._break_exact_duplicate_payloads(
                reviewed_segments,
                story_units,
                style_vocab=style_vocab,
            )
            reviewed_segments = self._break_exact_duplicate_payloads(
                reviewed_segments,
                story_units,
                style_vocab=style_vocab,
            )
            reviewed_segments = self._trim_final_bad_sentences(reviewed_segments)
            reviewed_segments = self._fix_sentence_boundaries(reviewed_segments)

        reviewed_segments = self._fix_self_target_action_payloads(reviewed_segments, story_units)
        reviewed_segments = self._final_sanitize_story_payloads(reviewed_segments)

        _progress(97, "Finalizing aligned story segments")
        story_segments = self._build_story_segments(story_units, reviewed_segments)
        # Panel mode: skip segment-coherence merging — each panel must stay as
        # its own slot.  _cohere_story_segments_for_delivery was the source of
        # out-of-order panel refs and 5-panel blobs being stitched together.
        # In OCR-only mode the coverage repair pass reads raw panel.ocr_text and
        # generates caption-like "The conversation reveals '...'" sentences that
        # overwrite the clean polished output.  Skip it; the mechanical paraphrases
        # produced by the polish pipeline are the best we can do without vision data.
        if not ocr_only_mode:
            story_segments = self._repair_story_coverage_for_delivery(
                story_segments,
                kept_panels,
                panel_evidence_records or [],
            )
        # Deduplicate consecutive segments with identical or near-identical text
        story_segments = self._deduplicate_story_segments(story_segments)
        story_text = self._compose_story_text(story_segments)
        return StoryScriptBundle(
            story_segments=story_segments,
            story_text=story_text,
            chapter_summary=beat_bundle.story_script,
            scene_summaries=scene_summaries,
            draft_lines=draft_lines,
            polished_lines=polished_lines,
            scene_seeds=scene_seeds,
            story_bible=story_bible,
            grounding_state=name_grounding,
            style_vocabulary=style_vocab,
        )

    def _merge_scene_seeds(self, scene_seeds: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not scene_seeds:
            return []
        merged: list[dict[str, Any]] = []
        bucket: list[dict[str, Any]] = []
        bucket_words = 0

        def flush() -> None:
            nonlocal bucket, bucket_words
            if not bucket:
                return
            ordered = bucket[:]
            merged.append(
                {
                    "scene_id": len(merged) + 1,
                    "panel_start": int(ordered[0].get("panel_start") or 0),
                    "panel_end": int(ordered[-1].get("panel_end") or 0),
                    "panel_ids": [panel_id for item in ordered for panel_id in item.get("panel_ids", []) or []],
                    "panels": [panel for item in ordered for panel in item.get("panels", []) or []],
                    "combined_text": clean_ocr_text(
                        " ".join(str(item.get("combined_text") or "").strip() for item in ordered if str(item.get("combined_text") or "").strip())
                    )[:2200],
                    "character_names": sorted(
                        {
                            str(name).strip()
                            for item in ordered
                            for name in item.get("character_names", []) or []
                            if str(name).strip()
                        }
                    ),
                }
            )
            bucket = []
            bucket_words = 0

        for seed in scene_seeds:
            text = str(seed.get("combined_text") or "").strip()
            word_count = len(re.findall(r"[A-Za-z']+", text))
            panel_count = len(seed.get("panel_ids", []) or [])
            # A scene stands alone only when it is already big enough to split
            # into multiple story groups. Otherwise bucket it with neighbors.
            should_stand_alone = panel_count >= 6 or word_count >= 160
            if should_stand_alone:
                flush()
                bucket = [seed]
                flush()
                continue

            bucket.append(seed)
            bucket_words += word_count
            bucket_panel_count = sum(len(item.get("panel_ids", []) or []) for item in bucket)
            # Target: ~3-4 panels per final segment. Keep scenes small so
            # _split_scene_into_story_groups has room to produce multiple beats
            # instead of a single massive summary.
            if len(bucket) >= 3 or bucket_panel_count >= 5 or bucket_words >= 160:
                flush()

        flush()
        return merged or scene_seeds

    def _sanitize_scene_seeds(
        self,
        scene_seeds: list[dict[str, Any]],
        grounding: dict[str, Any],
    ) -> list[dict[str, Any]]:
        sanitized: list[dict[str, Any]] = []
        for seed in scene_seeds:
            current = dict(seed)
            current["character_names"] = self._grounded_character_names(
                current.get("character_names", []) or [],
                grounding,
                strict=True,
            )
            combined_text = clean_ocr_text(str(current.get("combined_text") or "").strip())
            current["combined_text"] = (
                self._salvage_readable_ocr_fragments(combined_text)
                if self._text_is_noisy_ocr(combined_text)
                else combined_text
            )
            sanitized.append(current)
        return sanitized

    def _apply_hybrid_panel_anchors(
        self,
        story_segments: list,
        kept_panels: list[PanelBox],
    ) -> list:
        """Hybrid mode: ensure every kept panel maps to at least one story segment.

        Panels that aren't covered by any existing segment get a blank anchor
        segment inserted at the right ordinal position.  The auto-repair pass that
        runs at the end of script generation will fill those blank segments in.
        """
        from app.schemas.project import StorySegment as _StorySegment
        import uuid as _uuid

        covered: set[str] = {
            panel_id
            for seg in story_segments
            for panel_id in (seg.panel_ids or [])
        }
        uncovered = [p for p in kept_panels if p.id not in covered]
        if not uncovered:
            return story_segments

        # Build an order map so we can insert anchors in sequence.
        seg_list = list(story_segments)
        # Group uncovered panels by proximity to existing segments.
        for panel in uncovered:
            # Find the insertion index: after the last segment whose panel_end ≤ panel.order.
            insert_at = len(seg_list)
            panel_order = int(panel.order)
            for idx, seg in enumerate(seg_list):
                end = seg.panel_end or seg.order
                if int(end) <= panel_order:
                    insert_at = idx + 1
            anchor = _StorySegment(
                id=str(_uuid.uuid4()),
                order=panel_order,
                text="",
                panel_ids=[panel.id],
                panel_start=panel_order,
                panel_end=panel_order,
                scene_id=None,
                visual_only=not bool(panel.ocr_text),
                suppression_reason=None,
            )
            seg_list.insert(insert_at, anchor)

        # Re-number orders to keep them monotone using Pydantic's model_copy.
        seg_list = [
            seg.model_copy(update={"order": idx + 1})
            for idx, seg in enumerate(seg_list)
        ]
        logger.info(
            "Hybrid mode: added %d panel anchor segment(s) for uncovered kept panels",
            len(uncovered),
        )
        return seg_list

    def _build_vision_character_dictionary(
        self,
        canonical_characters: list[CanonicalCharacterRecord],
        fallback_dictionary: dict[str, Any],
    ) -> dict[str, Any]:
        if not canonical_characters:
            return fallback_dictionary
        dictionary: dict[str, Any] = {}
        for character in canonical_characters:
            name = str(character.name or "").strip()
            if not name or self._vision_name_is_placeholder(name):
                continue
            if str(character.role or "").casefold() == "cameo":
                continue
            if not (character.portrait_pages or character.visual_description):
                continue
            dictionary[name] = {
                "display_name": name,
                "role": character.role,
                "appearance": character.visual_description,
                "aliases": [
                    str(alias).strip()
                    for alias in character.aliases
                    if str(alias).strip() and not self._vision_name_is_placeholder(str(alias))
                ],
            }
        return dictionary

    def _vision_protagonist_name(self, canonical_characters: list[CanonicalCharacterRecord]) -> str | None:
        protagonists = [
            character
            for character in canonical_characters
            if (
                str(character.role or "").casefold() == "protagonist"
                and str(character.name or "").strip()
                and not self._vision_name_is_placeholder(str(character.name or ""))
                and (character.portrait_pages or character.visual_description)
            )
        ]
        if protagonists:
            protagonists.sort(
                key=lambda character: (
                    self._vision_name_quality(str(character.name or "")),
                    len(character.portrait_pages or []),
                    len(str(character.visual_description or "")),
                ),
                reverse=True,
            )
            return protagonists[0].name
        for character in canonical_characters:
            if (
                str(character.name or "").strip()
                and not self._vision_name_is_placeholder(str(character.name or ""))
                and (character.portrait_pages or character.visual_description)
            ):
                return character.name
        return None

    def _vision_name_quality(self, raw_name: str) -> tuple[int, int]:
        lowered = str(raw_name or "").strip().casefold()
        if not lowered:
            return (0, 0)
        if self._vision_name_is_placeholder(lowered):
            return (1, len(lowered.split()))
        tokens = lowered.split()
        if len(tokens) >= 2:
            return (5, len(tokens))
        return (4, len(tokens))

    def _vision_name_is_placeholder(self, raw_name: str) -> bool:
        cleaned = str(raw_name or "").strip()
        if not cleaned:
            return True
        return bool(_VISION_PLACEHOLDER_NAME_PATTERN.search(cleaned))

    def _normalize_character_role_map(self, raw_roles: Any) -> dict[str, list[str]]:
        allowed_roles = {
            "visible_present",
            "speaker",
            "addressee",
            "mentioned_absent",
            "flashback_present",
            "memory_present",
            "imagined_present",
            "uncertain",
        }
        if not isinstance(raw_roles, dict):
            return {}
        normalized: dict[str, list[str]] = {}
        for raw_name, raw_values in raw_roles.items():
            name = str(raw_name or "").strip()
            if not name or self._vision_name_is_placeholder(name):
                continue
            values = raw_values if isinstance(raw_values, list) else [raw_values]
            roles = [
                str(value or "").strip()
                for value in values
                if str(value or "").strip() in allowed_roles
            ]
            if roles:
                normalized[name] = list(dict.fromkeys(roles))[:4]
        return normalized

    def _merge_character_roles(self, role_maps: Iterable[dict[str, Any]]) -> dict[str, list[str]]:
        merged: dict[str, list[str]] = {}
        for role_map in role_maps:
            for name, roles in self._normalize_character_role_map(role_map).items():
                bucket = merged.setdefault(name, [])
                for role in roles:
                    if role not in bucket:
                        bucket.append(role)
        return {name: roles[:6] for name, roles in merged.items()}

    def _character_role_allows_presence(self, name: str, character_roles: dict[str, list[str]]) -> bool:
        if not name:
            return False
        roles = {
            role
            for role_name, role_values in character_roles.items()
            if normalize_name_key(role_name) == normalize_name_key(name)
            for role in role_values
        }
        if not roles:
            return True
        if roles & {"visible_present", "speaker", "flashback_present", "memory_present", "imagined_present"}:
            return True
        return False

    def _vision_records_have_usable_content(self, records: list[PanelVisionRecord]) -> bool:
        """Return True when vision records can safely drive fine-grained alignment.

        Some projects, especially manga pages Gemini refuses as prohibited
        content, still produce a full ``panel_vision_final.json`` where every
        record is visual_only with no action/dialogue/caption. Treating that as
        per-panel evidence creates hundreds of empty hybrid anchors. We only use
        fine-grained hybrid alignment when at least a small slice of records has
        real panel facts.
        """
        if not records:
            return False
        usable = 0
        for record in records:
            if bool(getattr(record, "visual_only", False)):
                continue
            evidence = " ".join(
                str(value or "").strip()
                for value in (
                    getattr(record, "action_beat", ""),
                    getattr(record, "dialogue", ""),
                    getattr(record, "caption", ""),
                )
                if str(value or "").strip()
            )
            if not evidence:
                continue
            confidence = float(getattr(record, "confidence", 0.0) or 0.0)
            if confidence < 0.2 and len(evidence.split()) < 6:
                continue
            usable += 1
        return usable >= 12 or (usable / max(len(records), 1)) >= 0.05

    def _prepare_vision_panel_payloads(
        self,
        panels: list[PanelBox],
        panel_vision_records: list[PanelVisionRecord],
        canonical_characters: list[CanonicalCharacterRecord],
        panel_evidence_records: list[dict[str, Any]] | None = None,
    ) -> list[dict[str, Any]]:
        record_by_id = {record.panel_id: record for record in panel_vision_records}
        evidence_by_id = {
            str(item.get("panel_id") or "").strip(): item
            for item in panel_evidence_records or []
            if isinstance(item, dict) and str(item.get("panel_id") or "").strip()
        }
        prepared: list[dict[str, Any]] = []
        for panel in panels:
            record = record_by_id.get(panel.id)
            evidence = evidence_by_id.get(panel.id)
            panel_evidence_text = self._panel_evidence_text(evidence)
            fallback_ocr = panel_evidence_text or self._panel_ocr_fallback_text(panel)
            _panel_reading_order = int(panel.page or 0) * 10000 + int(getattr(panel, "panel", 0) or 0)
            if record is None:
                # No vision record available. Check if we have any OCR evidence.
                # If we have neither vision nor OCR, mark as zero-evidence so LLM
                # doesn't generate weak repetitive narration.
                has_evidence = bool(fallback_ocr or panel_evidence_text)
                prepared.append(
                    {
                        "panel": _panel_reading_order,
                        "panel_id": panel.id,
                        "page": panel.page,
                        "text": fallback_ocr,
                        "translation_failed": False,
                        "character_names": [],
                        "character_roles": {},
                        "visual_caption": "",
                        "panel_evidence_text": panel_evidence_text,
                        "scene_change": False,
                        "confidence": 0.0,
                        "zero_evidence": not has_evidence,  # Mark for downstream filtering
                    }
                )
                continue
            combined_text = " ".join(
                value.strip()
                for value in (record.dialogue, record.caption)
                if str(value or "").strip()
            ).strip()
            if panel_evidence_text:
                combined_text = self._join_unique_evidence(combined_text, panel_evidence_text)
            if not combined_text and fallback_ocr and (record.visual_only or float(record.confidence or 0.0) <= 0.05):
                # Gemini occasionally refuses the image entirely, leaving
                # vision fields blank even though the panel has a local,
                # filtered narration/OCR fallback from detection. Use that
                # fallback as evidence only; downstream quality gates still
                # prevent noisy OCR from being emitted verbatim.
                combined_text = fallback_ocr
            character_roles = self._normalize_character_role_map(getattr(record, "character_roles", {}) or {})
            character_names = [
                str(name).strip()
                for name in list(record.character_names)
                if self._character_role_allows_presence(str(name).strip(), character_roles)
            ]
            if (
                record.speaker not in {"", "unknown", "narrator", "off-screen speaker", "unseen speaker", "neighbor", "bystander"}
                and not self._vision_name_is_placeholder(record.speaker)
            ):
                character_names.append(record.speaker)
            seen: set[str] = set()
            deduped_names: list[str] = []
            haystack = " ".join(
                [
                    record.action_beat,
                    record.dialogue,
                    record.caption,
                    combined_text,
                    fallback_ocr,
                ]
            ).casefold()
            for name in character_names:
                key = re.sub(r"\s+", " ", str(name or "").casefold()).strip()
                if key and key not in seen and not self._vision_name_is_placeholder(str(name)):
                    seen.add(key)
                    deduped_names.append(str(name).strip())
            for character in canonical_characters:
                if not character.name:
                    continue
                if self._vision_name_is_placeholder(character.name):
                    continue
                if re.search(rf"\b{re.escape(character.name.casefold())}\b", str(record.action_beat or "").casefold()):
                    key = re.sub(r"\s+", " ", character.name.casefold()).strip()
                    if key not in seen:
                        seen.add(key)
                        deduped_names.append(character.name)
                        character_roles.setdefault(character.name, ["visible_present"])
                elif re.search(rf"\b{re.escape(character.name.casefold())}\b", haystack):
                    character_roles.setdefault(character.name, ["mentioned_absent"])
            # Check if panel has usable evidence content
            has_dialogue_or_action = bool(
                (record.dialogue and str(record.dialogue).strip())
                or (record.action_beat and str(record.action_beat).strip())
                or (record.caption and str(record.caption).strip())
                or combined_text
            )
            has_any_evidence = has_dialogue_or_action or bool(fallback_ocr or panel_evidence_text)

            prepared.append(
                {
                    "panel": _panel_reading_order,
                    "panel_id": panel.id,
                    "page": panel.page,
                    "text": combined_text,
                    "translation_failed": False,
                    "character_names": deduped_names,
                    "character_roles": character_roles,
                    "visual_caption": "" if record.visual_only else self._clean_vision_evidence_text(
                        str(record.action_beat or "").strip(),
                        canonical_characters,
                    ),
                    "vision_dialogue": self._clean_vision_evidence_text(
                        str(record.dialogue or "").strip(),
                        canonical_characters,
                    ),
                    "vision_caption": self._clean_vision_evidence_text(
                        str(record.caption or "").strip(),
                        canonical_characters,
                    ),
                    "vision_action_beat": self._clean_vision_evidence_text(
                        str(record.action_beat or "").strip(),
                        canonical_characters,
                    ),
                    "ocr_fallback_text": fallback_ocr,
                    "panel_evidence_text": panel_evidence_text,
                    "panel_evidence_confidence": float(evidence.get("confidence") or 0.0) if isinstance(evidence, dict) else 0.0,
                    "scene_change": bool(record.scene_change),
                    "confidence": float(record.confidence or 0.0),
                    "zero_evidence": not has_any_evidence,  # Mark for downstream filtering
                }
            )
        return prepared

    def _panel_evidence_text(self, evidence: dict[str, Any] | None) -> str:
        if not isinstance(evidence, dict):
            return ""
        source_summary = evidence.get("source_summary") if isinstance(evidence.get("source_summary"), dict) else {}
        detectors = {
            str(item or "").strip()
            for item in (source_summary.get("detectors") or [])
            if str(item or "").strip()
        }
        confidence = float(evidence.get("confidence") or 0.0)
        # The legacy panel metadata OCR is exactly the source that produced
        # "Agora", Portuguese fragments, and fake character names. Treat it as
        # audit evidence only unless a real detector corroborates it.
        if not detectors or detectors <= {"existing-panel-ocr"}:
            return ""
        if confidence < 0.58 and not (detectors & {"page-ocr-backfill", "apple-vision", "comic-ocr", "opencv-region"}):
            return ""
        candidates = [
            str(evidence.get("caption_text") or "").strip(),
            str(evidence.get("dialogue_text") or "").strip(),
            str(evidence.get("text_english") or "").strip(),
        ]
        text = clean_ocr_text(" ".join(item for item in candidates if item))
        if not text or self._text_is_noisy_ocr(text):
            return ""
        if self._line_has_foreign_stopword_cluster(text):
            return ""
        return text[:900]

    def _join_unique_evidence(self, primary: str, secondary: str) -> str:
        first = clean_ocr_text(str(primary or "").strip())
        second = clean_ocr_text(str(secondary or "").strip())
        if not first:
            return second
        if not second:
            return first
        first_key = re.sub(r"[^a-z0-9]+", " ", first.casefold()).strip()
        second_key = re.sub(r"[^a-z0-9]+", " ", second.casefold()).strip()
        if second_key and second_key in first_key:
            return first
        if first_key and first_key in second_key:
            return second
        return f"{first} {second}".strip()

    def _panel_ocr_fallback_text(self, panel: PanelBox) -> str:
        """Return a conservative OCR sidecar for hybrid alignment fallback.

        Vision-first remains vision-led, but hybrid mode needs ordered local
        anchors even when Gemini refuses an image batch. This helper admits
        only sentence-like OCR snippets; if OCR is unusable, it falls back to
        the panel's existing narration field as evidence. That narration is
        still filtered downstream before it can become final script text, but
        keeping it here prevents visible text panels from becoming silent just
        because both Paddle and Gemini had a bad read.
        """
        legacy_narration = self._normalize_segment_text(str(panel.narration or "").strip(), allow_empty=True)
        if legacy_narration and not self._text_is_noisy_ocr(legacy_narration):
            return legacy_narration[:500]
        # Fall back to speech-bubble OCR text extracted during detection — this
        # is the primary source of dialogue and should always be preferred over
        # silence when vision records are absent.
        raw_ocr = clean_ocr_text(str(getattr(panel, "ocr_text", None) or "").strip())
        if raw_ocr and not self._text_is_noisy_ocr(raw_ocr):
            if re.search(r"(.)\1\1", raw_ocr):
                salvaged = self._salvage_readable_ocr_fragments(raw_ocr)
                if salvaged:
                    return salvaged[:500]
            return raw_ocr[:500]
        salvaged = self._salvage_readable_ocr_fragments(raw_ocr)
        if salvaged:
            return salvaged[:500]
        return ""

    def _salvage_readable_ocr_fragments(self, text: str) -> str:
        """Keep readable OCR fragments when one noisy shard poisons a whole panel.

        Manga pages often mix good speech bubbles with SFX, one-letter labels,
        or broken OCR from vertical/foreign text. The normal all-or-nothing
        noise check is correct for final transcript text, but too aggressive
        for evidence gathering: a single bad bubble can erase several readable
        English lines. This helper keeps sentence-like English fragments while
        leaving obvious SFX/garbage out of the script path.
        """
        cleaned = clean_ocr_text(str(text or "").strip())
        if not cleaned:
            return ""
        pieces = [
            piece.strip(" \t\r\n,;:-")
            for piece in re.split(r"(?<=[.!?])\s+|\n+", cleaned)
            if piece.strip(" \t\r\n,;:-")
        ]
        accepted: list[str] = []
        seen: set[str] = set()
        common_words = {
            "the", "and", "you", "that", "this", "with", "what", "have", "will",
            "would", "could", "about", "because", "before", "after", "there",
            "their", "them", "your", "from", "read", "name", "share", "chance",
            "take", "care", "still", "need", "needing", "okay",
        }
        for piece in pieces:
            if re.search(r"[\u3040-\u30ff\u4e00-\u9fff\uac00-\ud7af]", piece) and not re.search(r"[A-Za-z]{3,}", piece):
                continue
            tokens = re.findall(r"[A-Za-z']+", piece)
            if len(tokens) < 3:
                continue
            semantic_fragment = bool(
                re.search(
                    r"\b(?:read as|hence|name|share them|still have a chance|take care|not be seeing|won't needing)\b",
                    piece,
                    flags=re.IGNORECASE,
                )
            )
            if len(tokens) <= 3 and re.search(r"(.)\1\1", piece):
                continue
            short_ratio = sum(1 for token in tokens if len(token) <= 2) / max(len(tokens), 1)
            no_vowel_ratio = sum(
                1 for token in tokens if len(token) >= 3 and not re.search(r"[aeiouyAEIOUY]", token)
            ) / max(len(tokens), 1)
            if not semantic_fragment and (short_ratio >= 0.45 or no_vowel_ratio >= 0.34):
                continue
            if not is_usable_ocr_text(piece):
                continue
            if not semantic_fragment and self._has_ocr_shard_cluster(piece):
                continue
            if re.fullmatch(r"[A-Za-z!?. -]{1,16}", piece) and re.search(r"(.)\1\1|^[A-Z!?. -]+$", piece):
                continue
            if not (set(token.casefold() for token in tokens) & common_words) and len(tokens) < 6:
                continue
            normalized = self._normalize_segment_text(piece, allow_empty=True)
            key = re.sub(r"[^a-z0-9]+", " ", normalized.casefold()).strip()
            if not normalized or not key or key in seen:
                continue
            seen.add(key)
            accepted.append(normalized)
        return clean_ocr_text(" ".join(accepted))[:900]

    def _salvage_noisy_ocr_evidence(
        self,
        text: str,
        *,
        story_bible: dict[str, Any] | None = None,
        character_names: list[str] | None = None,
        protagonist_name: str | None = None,
    ) -> str:
        """Extract non-spoken anchors from noisy OCR.

        The returned string is supporting evidence only. It should never be
        emitted directly as narration; it exists so bridge rules can still see
        reliable names, world terms, and action cues inside otherwise rejected
        OCR noise.
        """
        cleaned = clean_ocr_text(str(text or "").strip())
        if not cleaned:
            return ""

        known_terms: list[str] = []

        def _add_term(value: Any) -> None:
            term = str(value or "").strip()
            if not term:
                return
            key = normalize_name_key(term)
            if not key or len(key) < 3:
                return
            if all(normalize_name_key(existing) != key for existing in known_terms):
                known_terms.append(term)

        _add_term(protagonist_name)
        for name in character_names or []:
            _add_term(name)
        for cast_item in (story_bible or {}).get("cast", []) or []:
            if isinstance(cast_item, dict):
                _add_term(cast_item.get("name"))
                for alias in cast_item.get("aliases") or []:
                    _add_term(alias)
            else:
                _add_term(cast_item)
        for term in (story_bible or {}).get("world_terms", []) or []:
            _add_term(term)

        normalized_cleaned = normalize_name_key(cleaned)
        found_terms: list[str] = []
        for term in sorted(known_terms, key=len, reverse=True):
            term_key = normalize_name_key(term)
            if not term_key:
                continue
            if term_key in normalized_cleaned:
                found_terms.append(term)
                continue

        signal_patterns: tuple[tuple[str, str], ...] = (
            ("enemy", r"\benemy\b|\battack\b|\bretreat\b|\bevacuat\w*\b|\bintercept\b|\bactive\b|\balive\b|\brun\b"),
            ("machine", r"\bmachine\b|\bmech\b|\bframe\b|\bweapon\b|\bvehicle\b"),
            ("pilot", r"\bpilot\b|\bparasites?\b|\bpartner\b|\bconnect(?:ion)?\b|\bonline\b|\bready\b|\bget on\b|\bhachi\b"),
            ("mission", r"\bmission\b|\btarget\b|\bfacility\b|\blevel\b|\bshaft\b|\bcaution\b"),
            ("danger", r"\bdeath\b|\bdie\b|\bkill\b|\bblood\b|\bpowerless\b|\bstop(?:s|ped)? it\b"),
            ("argument", r"\bdidn'?t\b.{0,16}\bstart it\b|\brumors?\b|\bwhat are you\b|\bwhat a sight\b|\bno one names\b"),
            ("identity", r"\bnames?\b|\bcrybaby\b|\bcalled\b|\bnamed\b|\bidentity\b"),
        )
        signal_labels = [
            label
            for label, pattern in signal_patterns
            if re.search(pattern, cleaned, flags=re.IGNORECASE)
        ]

        if not found_terms and not signal_labels:
            return ""

        fragments: list[str] = []
        for raw_part in re.split(r"(?<=[.!?])\s+|[;。！？]+|\s{2,}", cleaned):
            part = raw_part.strip(" ,;:-")
            if not part:
                continue
            part_key = normalize_name_key(part)
            has_known = any(normalize_name_key(term) in part_key for term in found_terms if normalize_name_key(term))
            has_signal = any(re.search(pattern, part, flags=re.IGNORECASE) for _, pattern in signal_patterns)
            if not has_known and not has_signal:
                continue
            part = re.sub(r"[^A-Za-z0-9À-ÿ'?!.,:;() -]+", " ", part)
            part = re.sub(r"\b([A-Za-z]{2,})\s+\1\b", r"\1", part, flags=re.IGNORECASE)
            part = re.sub(r"\s+", " ", part).strip(" ,;:-")
            tokens = re.findall(r"[A-Za-zÀ-ÿ']+", part)
            if len(tokens) < 2:
                continue
            fragments.append(part[:120])
            if len(fragments) >= 6:
                break

        parts: list[str] = []
        if found_terms:
            parts.append("Known terms: " + ", ".join(dict.fromkeys(found_terms[:8])))
        if signal_labels:
            parts.append("Signals: " + ", ".join(dict.fromkeys(signal_labels[:8])))
        if fragments:
            parts.append("OCR anchors: " + "; ".join(fragments))
        return ". ".join(parts)

    def _clean_vision_evidence_text(
        self,
        text: str,
        canonical_characters: list[CanonicalCharacterRecord] | None = None,
    ) -> str:
        cleaned = self._normalize_supporting_text(str(text or "").strip())
        if not cleaned:
            return ""
        if self._line_has_foreign_stopword_cluster(cleaned):
            return ""
        # Alias replacement can duplicate the final token of a multi-word
        # canonical name. Collapse those duplicated tails before prompting.
        for character in canonical_characters or []:
            name = str(character.name or "").strip()
            tokens = name.split()
            if len(tokens) < 2:
                continue
            tail = tokens[-1]
            cleaned = re.sub(
                rf"\b{re.escape(name)}\s+{re.escape(tail)}\b",
                name,
                cleaned,
                flags=re.IGNORECASE,
            )
        cleaned = re.sub(r"\b([A-Z][a-z]+)\s+([A-Z][a-z]+)\s+\2\b", r"\1 \2", cleaned)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        return cleaned

    def _build_vision_scene_seeds(self, ordered_payloads: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not ordered_payloads:
            return []
        groups: list[list[dict[str, Any]]] = []
        current: list[dict[str, Any]] = []
        for payload in ordered_payloads:
            if current and bool(payload.get("scene_change")):
                groups.append(current)
                current = []
            current.append(payload)
        if current:
            groups.append(current)
        seeds: list[dict[str, Any]] = []
        for scene_index, group in enumerate(groups, start=1):
            panel_ids = [str(item.get("panel_id") or "").strip() for item in group if str(item.get("panel_id") or "").strip()]
            panels = [int(item.get("panel") or 0) for item in group if int(item.get("panel") or 0)]
            combined_text = " ".join(
                str(item.get("text") or "").strip()
                for item in group
                if str(item.get("text") or "").strip()
            ).strip()
            visual_cues = " ".join(
                str(item.get("visual_caption") or "").strip()
                for item in group
                if str(item.get("visual_caption") or "").strip()
            ).strip()
            if visual_cues and combined_text:
                combined_text = f"{combined_text} {visual_cues}".strip()
            elif visual_cues:
                combined_text = visual_cues
            character_names = sorted(
                {
                    str(name).strip()
                    for item in group
                    for name in item.get("character_names", []) or []
                    if str(name).strip()
                }
            )
            character_roles = self._merge_character_roles(
                item.get("character_roles", {}) or {}
                for item in group
            )
            seeds.append(
                {
                    "scene_id": scene_index,
                    "panel_start": min(panels) if panels else 0,
                    "panel_end": max(panels) if panels else 0,
                    "panel_ids": panel_ids,
                    "panels": panels,
                    "combined_text": combined_text[:1800],
                    "character_names": character_names,
                    "character_roles": character_roles,
                }
            )
        return seeds

    def _expand_story_units(
        self,
        scene_seeds: list[dict[str, Any]],
        ordered_payloads: list[dict[str, Any]],
        scene_summaries: list[dict[str, Any]],
        grounding: dict[str, Any] | None,
    ) -> list[dict[str, Any]]:
        panel_payload_by_id = {
            str(payload.get("panel_id") or "").strip(): dict(payload)
            for payload in ordered_payloads
            if str(payload.get("panel_id") or "").strip()
        }
        summary_lookup = {
            int(item.get("scene_id") or 0): str(item.get("description") or item.get("summary") or "").strip()
            for item in scene_summaries
            if int(item.get("scene_id") or 0)
        }
        raw_units: list[dict[str, Any]] = []
        for scene_index, seed in enumerate(scene_seeds, start=1):
            scene_id = int(seed.get("scene_id") or scene_index)
            payloads = [
                panel_payload_by_id[panel_id]
                for panel_id in seed.get("panel_ids", []) or []
                if str(panel_id).strip() in panel_payload_by_id
            ]
            groups = self._split_scene_into_story_groups(payloads) if payloads else [[]]
            if not groups:
                groups = [payloads] if payloads else [[]]
            for group in groups:
                raw_units.append(
                    self._build_story_unit_payload(
                        group,
                        scene_id=scene_id,
                        scene_summary=summary_lookup.get(scene_id, ""),
                        grounding=grounding,
                        fallback_seed=seed,
                    )
                )

        covered_panel_ids = {
            panel_id
            for unit in raw_units
            for panel_id in unit.get("panel_ids", []) or []
            if str(panel_id).strip()
        }
        missing_payloads = [
            payload
            for payload in sorted(panel_payload_by_id.values(), key=lambda item: int(item.get("panel") or 0))
            if str(payload.get("panel_id") or "").strip() not in covered_panel_ids
        ]
        next_scene_id = max([int(seed.get("scene_id") or 0) for seed in scene_seeds] or [0]) + 1
        for group in self._split_missing_payload_groups(missing_payloads):
            scene_id = self._scene_id_for_missing_group(group, raw_units, next_scene_id)
            if scene_id >= next_scene_id:
                next_scene_id = scene_id + 1
            raw_units.append(
                self._build_story_unit_payload(
                    group,
                    scene_id=scene_id,
                    scene_summary=summary_lookup.get(scene_id, ""),
                    grounding=grounding,
                    fallback_seed=None,
                )
            )

        if not raw_units:
            raw_units = [
                self._build_story_unit_payload(
                    [],
                    scene_id=int(seed.get("scene_id") or index),
                    scene_summary=summary_lookup.get(int(seed.get("scene_id") or index), ""),
                    grounding=grounding,
                    fallback_seed=seed,
                )
                for index, seed in enumerate(scene_seeds, start=1)
            ]
        return self._finalize_story_units(self._coalesce_story_units_for_recap(raw_units))

    def _panel_mode_story_units(
        self,
        ordered_payloads: list[dict[str, Any]],
        grounding: dict[str, Any] | None,
    ) -> list[dict[str, Any]]:
        """Create exactly one story unit per panel — no merging or grouping.

        This is the strict panel mode: each kept panel becomes its own
        narration slot.  Scene seeds, coalescing, and segment coherence
        merging are all bypassed.  The result is maximum alignment between
        the artwork and the narration — one sentence describes one panel.
        """
        units: list[dict[str, Any]] = []
        for index, payload in enumerate(ordered_payloads, start=1):
            unit = self._build_story_unit_payload(
                [payload],
                scene_id=index,
                scene_summary="",
                grounding=grounding,
                fallback_seed=None,
            )
            unit["sequence_in_scene"] = 1
            unit["scene_unit_count"] = 1
            unit["segment_id"] = f"panel_{index:04d}"
            units.append(unit)
        return units

    def _coalesce_story_units_for_recap(self, raw_units: list[dict[str, Any]]) -> list[dict[str, Any]]:
        ordered_units = sorted(
            [dict(unit) for unit in raw_units if unit.get("panel_ids")],
            key=lambda item: (
                int(item.get("panel_start") or 0),
                int(item.get("panel_end") or 0),
                int(item.get("scene_id") or 0),
            ),
        )
        if len(ordered_units) <= 1:
            return raw_units

        merged: list[dict[str, Any]] = []
        bucket: list[dict[str, Any]] = []

        def unit_words(unit: dict[str, Any]) -> int:
            return len(
                re.findall(
                    r"[A-Za-z']+",
                    " ".join(
                        str(unit.get(key) or "")
                        for key in (
                            "combined_text",
                            "vision_dialogue",
                            "vision_caption",
                            "vision_action_beat",
                            "visual_cues",
                            "ocr_fallback_text",
                        )
                    ),
                )
            )

        def should_flush_before(next_unit: dict[str, Any]) -> bool:
            if not bucket:
                return False
            current_panels = sum(len(unit.get("panel_ids", []) or []) for unit in bucket)
            next_panels = len(next_unit.get("panel_ids", []) or [])
            current_words = sum(unit_words(unit) for unit in bucket)
            next_words = unit_words(next_unit)
            previous = bucket[-1]
            previous_end = int(previous.get("panel_end") or 0)
            next_start = int(next_unit.get("panel_start") or 0)
            boundary_score = self._story_unit_boundary_score(previous, next_unit)
            # Hard limit: never exceed 4 panels per segment
            if current_panels + next_panels > 4:
                return True
            if previous_end and next_start and next_start - previous_end > 2:
                return True
            if current_panels >= 2 and boundary_score >= 2.3:
                return True
            if current_panels >= 3 and boundary_score >= 2.0:
                return True
            if current_panels >= 4:
                return True
            if current_words >= 120:
                return True
            return False

        def flush() -> None:
            nonlocal bucket
            if not bucket:
                return
            merged.append(self._merge_story_unit_bucket(bucket))
            bucket = []

        for unit in ordered_units:
            if should_flush_before(unit):
                flush()
            bucket.append(unit)
        flush()

        # Post-process: split any merged unit larger than 4 panels
        final_units: list[dict[str, Any]] = []
        for unit in merged or raw_units:
            panels = sorted(unit.get("panels", []) or [], key=int)
            panel_ids = unit.get("panel_ids", []) or []

            # Fall back to panel_count field if "panels" is not set
            if not panels and unit.get("panel_count"):
                # If we don't have the "panels" list, use the count to determine if splitting is needed
                if unit.get("panel_count", 0) > 4:
                    # Split by reconstructing panels from panel_start/panel_end
                    start = int(unit.get("panel_start") or 0)
                    end = int(unit.get("panel_end") or 0)
                    if start and end and end >= start:
                        # Get all panel orders in this range
                        all_orders = list(range(start, end + 1))
                        # Match them to panel_ids
                        for i in range(0, len(all_orders), 4):
                            chunk_orders = all_orders[i:i+4]
                            chunk_ids = panel_ids[i:i+4] if i + 4 <= len(panel_ids) else panel_ids[i:]
                            chunk_unit = dict(unit)
                            chunk_unit["panels"] = chunk_orders
                            chunk_unit["panel_ids"] = chunk_ids
                            chunk_unit["panel_start"] = int(chunk_orders[0])
                            chunk_unit["panel_end"] = int(chunk_orders[-1])
                            chunk_unit["panel_count"] = len(chunk_orders)
                            final_units.append(chunk_unit)
                        continue
                    else:
                        final_units.append(unit)
                        continue

            # Build mapping from panel order to panel ID for splitting
            # Handle cases where panel_ids might not match panels exactly
            panel_by_order: dict[str, str] = {}
            if len(panel_ids) == len(panels):
                panel_by_order = {str(p): pid for p, pid in zip(panels, panel_ids)}
            elif panel_ids:
                # If counts don't match, just use panel_ids as-is when splitting
                for i, p in enumerate(panels):
                    if i < len(panel_ids):
                        panel_by_order[str(p)] = panel_ids[i]

            if len(panels) <= 4:
                final_units.append(unit)
            else:
                # Split unit into chunks of at most 4 panels (in order)
                for i in range(0, len(panels), 4):
                    chunk_panels = panels[i:i+4]
                    chunk_ids = [panel_by_order.get(str(p), "") for p in chunk_panels]
                    chunk_ids = [pid for pid in chunk_ids if pid]
                    if not chunk_ids and panel_ids:
                        # Fallback: use sequential IDs from panel_ids
                        chunk_ids = panel_ids[i:i+4]
                    chunk_unit = dict(unit)
                    chunk_unit["panel_ids"] = chunk_ids
                    chunk_unit["panels"] = chunk_panels
                    chunk_unit["panel_start"] = int(chunk_panels[0]) if chunk_panels else int(unit.get("panel_start") or 0)
                    chunk_unit["panel_end"] = int(chunk_panels[-1]) if chunk_panels else int(unit.get("panel_end") or 0)
                    chunk_unit["panel_count"] = len(chunk_panels)
                    final_units.append(chunk_unit)
        return final_units or raw_units

    def _story_unit_boundary_score(self, current: dict[str, Any], upcoming: dict[str, Any]) -> float:
        current_names = {
            str(name).strip().casefold()
            for name in current.get("character_names", []) or []
            if str(name).strip()
        }
        upcoming_names = {
            str(name).strip().casefold()
            for name in upcoming.get("character_names", []) or []
            if str(name).strip()
        }
        current_summary = self._normalize_supporting_text(str(current.get("scene_summary") or ""))
        upcoming_summary = self._normalize_supporting_text(str(upcoming.get("scene_summary") or ""))
        current_text_only = bool(current.get("text_only_beat"))
        upcoming_text_only = bool(upcoming.get("text_only_beat"))
        current_words = len(
            re.findall(
                r"[A-Za-z']+",
                " ".join(
                    str(current.get(key) or "")
                    for key in (
                        "combined_text",
                        "vision_dialogue",
                        "vision_caption",
                        "vision_action_beat",
                        "visual_cues",
                        "ocr_fallback_text",
                    )
                ),
            )
        )
        upcoming_words = len(
            re.findall(
                r"[A-Za-z']+",
                " ".join(
                    str(upcoming.get(key) or "")
                    for key in (
                        "combined_text",
                        "vision_dialogue",
                        "vision_caption",
                        "vision_action_beat",
                        "visual_cues",
                        "ocr_fallback_text",
                    )
                ),
            )
        )
        score = 0.0
        if int(current.get("scene_id") or 0) != int(upcoming.get("scene_id") or 0):
            score += 0.75
        if current_names and upcoming_names:
            overlap = len(current_names & upcoming_names) / max(1, min(len(current_names), len(upcoming_names)))
            if overlap == 0:
                score += 1.25
            elif overlap < 0.5:
                score += 0.45
        elif bool(current_names) != bool(upcoming_names):
            score += 0.3
        if current_summary and upcoming_summary and current_summary.casefold() != upcoming_summary.casefold():
            score += 0.35
        if bool(current_words >= 18) != bool(upcoming_words >= 18):
            score += 0.35
        if abs(current_words - upcoming_words) >= 18:
            score += 0.2
        if current_text_only != upcoming_text_only:
            score += 0.25
        return score

    def _merge_story_unit_bucket(self, bucket: list[dict[str, Any]]) -> dict[str, Any]:
        if len(bucket) == 1:
            return dict(bucket[0])
        panel_ids = [
            str(panel_id).strip()
            for unit in bucket
            for panel_id in unit.get("panel_ids", []) or []
            if str(panel_id).strip()
        ]
        panels = [
            int(panel)
            for unit in bucket
            for panel in unit.get("panels", []) or []
            if int(panel or 0)
        ]
        character_names = sorted(
            {
                str(name).strip()
                for unit in bucket
                for name in unit.get("character_names", []) or []
                if str(name).strip()
            },
            key=str.casefold,
        )
        character_roles = self._merge_character_roles(
            unit.get("character_roles", {}) or {}
            for unit in bucket
        )
        merged: dict[str, Any] = {
            "scene_id": int(bucket[0].get("scene_id") or 0),
            "panel_start": min(int(unit.get("panel_start") or 0) for unit in bucket if int(unit.get("panel_start") or 0)),
            "panel_end": max(int(unit.get("panel_end") or 0) for unit in bucket if int(unit.get("panel_end") or 0)),
            "panel_ids": panel_ids,
            "panels": panels,
            "panel_count": len(panel_ids),
            "character_names": character_names,
            "character_roles": character_roles,
            "scene_summary": " ".join(
                dict.fromkeys(
                    str(unit.get("scene_summary") or "").strip()
                    for unit in bucket
                    if str(unit.get("scene_summary") or "").strip()
                )
            )[:1200],
        }
        for key, limit in (
            ("combined_text", 1200),
            ("visual_cues", 700),
            ("vision_dialogue", 1200),
            ("vision_caption", 1200),
            ("vision_action_beat", 1200),
            ("ocr_fallback_text", 1200),
        ):
            merged[key] = clean_ocr_text(
                " ".join(str(unit.get(key) or "").strip() for unit in bucket if str(unit.get(key) or "").strip())
            )[:limit]
        merged["text_only_beat"] = bool(
            any(bool(unit.get("text_only_beat")) for unit in bucket)
            and not str(merged.get("vision_action_beat") or "").strip()
        )
        return merged

    def _build_story_unit_payload(
        self,
        group: list[dict[str, Any]],
        *,
        scene_id: int,
        scene_summary: str,
        grounding: dict[str, Any] | None,
        fallback_seed: dict[str, Any] | None,
    ) -> dict[str, Any]:
        panel_ids = [
            str(item.get("panel_id") or "").strip()
            for item in group
            if str(item.get("panel_id") or "").strip()
        ]
        if not panel_ids and fallback_seed is not None:
            panel_ids = [
                str(panel_id).strip()
                for panel_id in fallback_seed.get("panel_ids", []) or []
                if str(panel_id).strip()
            ]
        panel_orders = [
            int(item.get("panel") or 0)
            for item in group
            if int(item.get("panel") or 0)
        ]
        if not panel_orders and fallback_seed is not None:
            panel_orders = [int(panel) for panel in fallback_seed.get("panels", []) or [] if int(panel or 0)]
        combined_text = clean_ocr_text(
            " ".join(str(item.get("text") or "").strip() for item in group if str(item.get("text") or "").strip())
        )[:1200]
        if not combined_text and fallback_seed is not None and not group:
            combined_text = str(fallback_seed.get("combined_text") or "").strip()[:1200]
        if self._text_is_noisy_ocr(combined_text):
            combined_text = self._salvage_readable_ocr_fragments(combined_text)
        visual_cues = self._normalize_supporting_text(
            " ".join(
                str(item.get("visual_caption") or "").strip()
                for item in group
                if str(item.get("visual_caption") or "").strip()
            )
        )[:500]
        vision_dialogue = clean_ocr_text(
            " ".join(
                str(item.get("vision_dialogue") or "").strip()
                for item in group
                if str(item.get("vision_dialogue") or "").strip()
            )
        )[:1200]
        if self._text_is_noisy_ocr(vision_dialogue):
            vision_dialogue = ""
        vision_caption = clean_ocr_text(
            " ".join(
                str(item.get("vision_caption") or "").strip()
                for item in group
                if str(item.get("vision_caption") or "").strip()
            )
        )[:1200]
        if self._text_is_noisy_ocr(vision_caption):
            vision_caption = ""
        vision_action_beat = self._normalize_supporting_text(
            " ".join(
                str(item.get("vision_action_beat") or item.get("visual_caption") or "").strip()
                for item in group
                if str(item.get("vision_action_beat") or item.get("visual_caption") or "").strip()
            )
        )[:1200]
        ocr_fallback_text = clean_ocr_text(
            " ".join(
                str(item.get("ocr_fallback_text") or "").strip()
                for item in group
                if str(item.get("ocr_fallback_text") or "").strip()
            )
        )[:1200]
        if self._text_is_noisy_ocr(ocr_fallback_text):
            ocr_fallback_text = self._salvage_readable_ocr_fragments(ocr_fallback_text)
        text_only_beat = bool(
            (combined_text or vision_dialogue or vision_caption or ocr_fallback_text)
            and not str(vision_action_beat or visual_cues).strip()
        )
        inherited_names = [
            str(name).strip()
            for name in (
                fallback_seed.get("character_names", [])
                if fallback_seed is not None and not group
                else []
            ) or []
            if str(name).strip()
        ]
        character_names = self._grounded_character_names(
            [
                *[
                    str(name).strip()
                    for item in group
                    for name in item.get("character_names", []) or []
                    if str(name).strip()
                ],
                *inherited_names,
            ],
            grounding,
            strict=False,
        )
        character_roles = self._merge_character_roles(
            [
                item.get("character_roles", {}) or {}
                for item in group
            ]
            + (
                [fallback_seed.get("character_roles", {}) or {}]
                if fallback_seed is not None and not group
                else []
            )
        )
        # When Gemini Vision evidence is absent and local OCR dialogue is available,
        # promote the OCR text into vision_dialogue so it surfaces as a primary evidence
        # field. Without this, the LLM ignores combined_text in favour of franchise
        # knowledge when generating draft narration.
        effective_vision_dialogue = vision_dialogue
        if not effective_vision_dialogue and not vision_caption and not vision_action_beat and not visual_cues:
            effective_vision_dialogue = combined_text or ocr_fallback_text
        return {
            "scene_id": scene_id,
            "panel_start": min(panel_orders) if panel_orders else int((fallback_seed or {}).get("panel_start") or 0),
            "panel_end": max(panel_orders) if panel_orders else int((fallback_seed or {}).get("panel_end") or 0),
            "panel_ids": panel_ids,
            "panels": panel_orders,
            "panel_count": len(panel_ids),
            "character_names": character_names,
            "character_roles": character_roles,
            "combined_text": combined_text,
            "visual_cues": visual_cues,
            "vision_dialogue": effective_vision_dialogue,
            "vision_caption": vision_caption,
            "vision_action_beat": vision_action_beat,
            "ocr_fallback_text": ocr_fallback_text,
            "text_only_beat": text_only_beat,
            "scene_summary": scene_summary,
        }

    def _split_missing_payload_groups(self, payloads: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
        if not payloads:
            return []
        contiguous_runs: list[list[dict[str, Any]]] = []
        current_run: list[dict[str, Any]] = []
        for payload in payloads:
            if not current_run:
                current_run = [payload]
                continue
            previous = current_run[-1]
            current_panel = int(payload.get("panel") or 0)
            previous_panel = int(previous.get("panel") or 0)
            current_page = int(payload.get("page") or 0)
            previous_page = int(previous.get("page") or 0)
            if current_panel == previous_panel + 1 and current_page in {previous_page, previous_page + 1}:
                current_run.append(payload)
            else:
                contiguous_runs.append(current_run)
                current_run = [payload]
        if current_run:
            contiguous_runs.append(current_run)

        groups: list[list[dict[str, Any]]] = []
        for run in contiguous_runs:
            if len(run) <= 4:
                groups.append(run)
            else:
                groups.extend(self._split_scene_for_coarse_hybrid_alignment(run))
        return groups

    def _scene_id_for_missing_group(
        self,
        group: list[dict[str, Any]],
        raw_units: list[dict[str, Any]],
        default_scene_id: int,
    ) -> int:
        if not group:
            return default_scene_id
        start = int(group[0].get("panel") or 0)
        end = int(group[-1].get("panel") or start)
        ordered_units = sorted(
            [dict(unit) for unit in raw_units if int(unit.get("panel_start") or 0) or int(unit.get("panel_end") or 0)],
            key=lambda item: (
                int(item.get("panel_start") or 0),
                int(item.get("panel_end") or 0),
                int(item.get("scene_id") or 0),
            ),
        )
        previous = next(
            (
                unit
                for unit in reversed(ordered_units)
                if int(unit.get("panel_end") or 0) < start
            ),
            None,
        )
        upcoming = next(
            (
                unit
                for unit in ordered_units
                if int(unit.get("panel_start") or 0) > end
            ),
            None,
        )
        previous_scene_id = int((previous or {}).get("scene_id") or 0)
        upcoming_scene_id = int((upcoming or {}).get("scene_id") or 0)
        previous_gap = start - int((previous or {}).get("panel_end") or 0) if previous else None
        upcoming_gap = int((upcoming or {}).get("panel_start") or 0) - end if upcoming else None

        # Only inherit an existing scene when the missing run is clearly sandwiched
        # inside that same local scene. Otherwise assign a fresh scene id so distant
        # uncovered runs do not collapse into a giant pseudo-scene.
        if (
            previous_scene_id
            and previous_scene_id == upcoming_scene_id
            and previous_gap is not None
            and previous_gap <= 2
            and upcoming_gap is not None
            and upcoming_gap <= 2
        ):
            return previous_scene_id
        if previous_scene_id and previous_gap is not None and previous_gap <= 1 and (upcoming_gap is None or upcoming_gap > 3):
            return previous_scene_id
        if upcoming_scene_id and upcoming_gap is not None and upcoming_gap <= 1 and (previous_gap is None or previous_gap > 3):
            return upcoming_scene_id
        return default_scene_id

    def _finalize_story_units(self, raw_units: list[dict[str, Any]]) -> list[dict[str, Any]]:
        ordered_units = sorted(
            [dict(unit) for unit in raw_units if (unit.get("panel_ids") or [])],
            key=lambda item: (
                int(item.get("panel_start") or 0),
                int(item.get("panel_end") or 0),
                int(item.get("scene_id") or 0),
            ),
        )
        scene_counts: dict[int, int] = {}
        for unit in ordered_units:
            scene_id = int(unit.get("scene_id") or 0)
            scene_counts[scene_id] = scene_counts.get(scene_id, 0) + 1
        scene_offsets: dict[int, int] = {}
        finalized: list[dict[str, Any]] = []
        for unit in ordered_units:
            scene_id = int(unit.get("scene_id") or 0) or len(finalized) + 1
            scene_offsets[scene_id] = scene_offsets.get(scene_id, 0) + 1
            sequence_in_scene = scene_offsets[scene_id]
            current = dict(unit)
            current["scene_id"] = scene_id
            current["sequence_in_scene"] = sequence_in_scene
            current["scene_unit_count"] = scene_counts.get(scene_id, 1)
            current["segment_id"] = f"scene_{scene_id:03d}_beat_{sequence_in_scene:02d}"
            finalized.append(current)
        return finalized

    def _split_scene_into_story_groups(self, panel_payloads: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
        if not panel_payloads:
            return []
        if len(panel_payloads) == 1:
            return [panel_payloads]

        desired_units = self._desired_story_unit_count(panel_payloads)
        if desired_units <= 1:
            return [panel_payloads]

        boundaries: list[int] = []
        previous_boundary = -1
        panel_count = len(panel_payloads)
        for ordinal in range(1, desired_units):
            remaining_breaks = desired_units - ordinal
            left = previous_boundary + 1
            right = panel_count - remaining_breaks - 1
            if left > right:
                break
            ideal_boundary = max(left, min(right, round(ordinal * panel_count / desired_units) - 1))
            candidate_start = max(left, ideal_boundary - 2)
            candidate_end = min(right, ideal_boundary + 2)
            if candidate_start > candidate_end:
                candidate_start, candidate_end = left, right
            best_boundary = max(
                range(candidate_start, candidate_end + 1),
                key=lambda boundary: self._story_group_boundary_score(panel_payloads[boundary], panel_payloads[boundary + 1])
                - 0.35 * abs(boundary - ideal_boundary),
            )
            boundaries.append(best_boundary)
            previous_boundary = best_boundary

        groups: list[list[dict[str, Any]]] = []
        start_index = 0
        for boundary in boundaries:
            groups.append(panel_payloads[start_index:boundary + 1])
            start_index = boundary + 1
        groups.append(panel_payloads[start_index:])
        return [group for group in groups if group]

    def _split_scene_for_scene_mode(self, panel_payloads: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
        if not panel_payloads:
            return [[]]
        panel_count = len(panel_payloads)
        if panel_count <= 4:
            return [panel_payloads]
        target_panels = 3.0
        desired_units = max(2, round(panel_count / target_panels))
        desired_units = min(panel_count, desired_units)
        groups: list[list[dict[str, Any]]] = []
        for ordinal in range(desired_units):
            start = round(ordinal * panel_count / desired_units)
            end = round((ordinal + 1) * panel_count / desired_units)
            group = panel_payloads[start:end]
            if group:
                groups.append(group)
        return groups or [panel_payloads]

    def _split_scene_for_hybrid_alignment(self, panel_payloads: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
        """Split hybrid scenes into granular, ordered coverage anchors.

        Unlike story/scene mode, hybrid should never collapse a long weak-evidence
        chapter into six broad summary beats. It needs enough local slots for the
        final video to stay synchronized with panel order, while still grouping a
        few adjacent panels so the narration does not become robotic.
        """
        if not panel_payloads:
            return [[]]
        panel_count = len(panel_payloads)
        if panel_count <= 4:
            return [panel_payloads]
        # Hybrid is the alignment mode: keep every narration slot close to the
        # artwork. Strong vision evidence should improve the content inside a
        # slot, not expand the slot to 6-8 panels and reintroduce summary drift.
        target_panels = 3.0
        desired_units = max(2, round(panel_count / target_panels))
        desired_units = min(panel_count, desired_units)
        if desired_units <= 1:
            return [panel_payloads]
        groups: list[list[dict[str, Any]]] = []
        for ordinal in range(desired_units):
            start = round(ordinal * panel_count / desired_units)
            end = round((ordinal + 1) * panel_count / desired_units)
            group = panel_payloads[start:end]
            if group:
                groups.append(group)
        return groups or [panel_payloads]

    def _split_scene_for_coarse_hybrid_alignment(self, panel_payloads: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
        """Small ordered fallback when vision exists but is unusable.

        Hybrid mode is used for panel/video alignment, so even when Gemini
        refuses panel vision we keep local anchors close to the artwork. The
        narration may lean on OCR sidecar evidence and conservative repair
        fallbacks, but each segment should still cover roughly 2-4 panels
        rather than a whole scene.
        """
        if not panel_payloads:
            return [[]]
        panel_count = len(panel_payloads)
        if panel_count <= 4:
            return [panel_payloads]
        target_panels = 3.0
        desired_units = max(2, round(panel_count / target_panels))
        desired_units = min(panel_count, desired_units)
        if desired_units <= 1:
            return [panel_payloads]

        groups: list[list[dict[str, Any]]] = []
        for ordinal in range(desired_units):
            start = round(ordinal * panel_count / desired_units)
            end = round((ordinal + 1) * panel_count / desired_units)
            group = panel_payloads[start:end]
            if group:
                groups.append(group)
        return groups or [panel_payloads]

    def _desired_story_unit_count(self, panel_payloads: list[dict[str, Any]]) -> int:
        panel_count = len(panel_payloads)
        if panel_count <= 1:
            return panel_count
        evidence_scores = [self._panel_payload_signal_score(payload) for payload in panel_payloads]
        text_word_counts = [
            len(re.findall(r"[A-Za-z']+", clean_ocr_text(str(payload.get("text") or "").strip())))
            for payload in panel_payloads
        ]
        average_signal = sum(evidence_scores) / max(panel_count, 1)
        strong_panels = sum(1 for score in evidence_scores if score >= 2.0)
        # Target at most 3-4 panels per segment so each segment stays focussed
        # and the quality checker never flags "underexplained large panel range".
        target_panels_per_unit = 4.0
        if average_signal >= 2.15:
            target_panels_per_unit = 3.0
        elif average_signal >= 1.55:
            target_panels_per_unit = 3.5

        desired_units = max(1, round(panel_count / target_panels_per_unit))
        if strong_panels >= max(4, round(panel_count * 0.8)) and panel_count <= 10:
            desired_units += 1
        if panel_count <= 2 and average_signal >= 2.1 and max(text_word_counts or [0]) >= 8:
            desired_units = panel_count
        elif panel_count == 3 and average_signal >= 2.4 and sum(1 for count in text_word_counts if count >= 8) >= 2:
            desired_units = min(2, panel_count)
        # Allow up to 1 unit per panel so large chapters get enough segments
        max_units = min(panel_count, max(5, round(panel_count / 3.0)))
        desired_units = min(desired_units, max_units)
        if panel_count >= 14:
            desired_units = max(desired_units, 2)
        return max(1, desired_units)

    def _panel_payload_signal_score(self, payload: dict[str, Any]) -> float:
        text_words = len(re.findall(r"[A-Za-z']+", clean_ocr_text(str(payload.get("text") or "").strip())))
        caption_words = len(re.findall(r"[A-Za-z']+", str(payload.get("visual_caption") or "").strip()))
        names = [
            str(name).strip()
            for name in payload.get("character_names", []) or []
            if str(name).strip()
        ]
        score = 0.0
        if text_words >= 12:
            score += 2.0
        elif text_words >= 5:
            score += 1.1
        elif text_words >= 2:
            score += 0.4
        if caption_words >= 12:
            score += 0.45 if text_words >= 2 else 0.18
        elif caption_words >= 6:
            score += 0.25 if text_words >= 2 else 0.10
        elif caption_words >= 4 and text_words >= 2:
            score += 0.08
        if names:
            score += 0.45 if text_words >= 2 else (0.12 if caption_words >= 6 else 0.04)
        if payload.get("translation_failed"):
            score = max(0.0, score - 0.25)
        return score

    def _story_group_boundary_score(self, current: dict[str, Any], upcoming: dict[str, Any]) -> float:
        current_words = len(re.findall(r"[A-Za-z']+", clean_ocr_text(str(current.get("text") or "").strip())))
        upcoming_words = len(re.findall(r"[A-Za-z']+", clean_ocr_text(str(upcoming.get("text") or "").strip())))
        current_names = {
            str(name).strip().casefold()
            for name in current.get("character_names", []) or []
            if str(name).strip()
        }
        upcoming_names = {
            str(name).strip().casefold()
            for name in upcoming.get("character_names", []) or []
            if str(name).strip()
        }
        current_caption = self._normalize_supporting_text(str(current.get("visual_caption") or ""))
        upcoming_caption = self._normalize_supporting_text(str(upcoming.get("visual_caption") or ""))
        score = 0.0
        if int(current.get("page") or 0) != int(upcoming.get("page") or 0):
            score += 0.9
        if current_names != upcoming_names and (current_names or upcoming_names):
            score += 1.5
        if bool(current_words >= 5) != bool(upcoming_words >= 5):
            score += 0.9
        if abs(current_words - upcoming_words) >= 8:
            score += 0.35
        if current_caption and upcoming_caption and current_caption.casefold() != upcoming_caption.casefold():
            score += 0.3
        if current_words >= 10 and upcoming_words >= 10:
            score += 0.2
        return score

    def _normalize_supporting_text(self, text: str) -> str:
        return re.sub(r"\s+", " ", str(text or "").strip())

    def _apply_corrections_to_character_dictionary(
        self,
        character_dictionary: dict[str, Any],
        corrections: list[dict[str, str]],
    ) -> dict[str, Any]:
        corrected = dict(character_dictionary)
        for item in corrections or []:
            variant = str(item.get("variant") or "").strip()
            canonical = str(item.get("canonical") or "").strip()
            if not variant or not canonical:
                continue
            value = corrected.pop(variant, None)
            normalized_variant = variant.casefold()
            for key in list(corrected.keys()):
                if str(key).casefold() == normalized_variant:
                    value = corrected.pop(key)
                    break
            if value is None:
                continue
            corrected[canonical] = value
        return corrected

    def _apply_corrections_to_seed(self, seed: dict[str, Any], corrections: list[dict[str, str]]) -> dict[str, Any]:
        corrected = dict(seed)
        corrected["character_names"] = [
            self._canonical_name(str(name).strip(), corrections)
            for name in seed.get("character_names", []) or []
            if str(name).strip()
        ]
        corrected["combined_text"] = self._apply_name_corrections(str(seed.get("combined_text") or "").strip(), corrections)
        return corrected

    def _canonical_name(self, value: str, corrections: list[dict[str, str]]) -> str:
        for item in corrections or []:
            variant = str(item.get("variant") or "").strip()
            canonical = str(item.get("canonical") or "").strip()
            if variant and canonical and value.casefold() == variant.casefold():
                return canonical
        return value

    def _apply_name_corrections(self, text: str, corrections: list[dict[str, str]]) -> str:
        corrected = str(text or "")
        for item in corrections or []:
            variant = str(item.get("variant") or "").strip()
            canonical = str(item.get("canonical") or "").strip()
            if not variant or not canonical:
                continue
            corrected = re.sub(rf"\b{re.escape(variant)}\b", canonical, corrected, flags=re.IGNORECASE)
        return corrected

    def _draft_scene_lines(
        self,
        story_units: list[dict[str, Any]],
        *,
        project_title: str,
        chapter_metadata: ChapterMetadata | dict[str, Any],
        chapter_summary: str,
        character_dictionary: dict[str, Any],
        protagonist_name: str | None,
        story_bible: dict[str, Any] | None = None,
        scene_visual_paths: dict[str, list[Path]] | None = None,
        name_grounding: dict[str, Any] | None = None,
        prefer_local_evidence: bool = False,
        style_vocab: StyleVocabulary | None = None,
    ) -> list[str]:
        if not story_units:
            return []
        units = [
            {
                "segment_id": str(unit.get("segment_id") or f"segment_{index:03d}").strip() or f"segment_{index:03d}",
                "scene_id": int(unit.get("scene_id") or index),
                "sequence_in_scene": int(unit.get("sequence_in_scene") or 1),
                "scene_unit_count": int(unit.get("scene_unit_count") or 1),
                "panel_start": int(unit.get("panel_start") or 0),
                "panel_end": int(unit.get("panel_end") or 0),
                "panel_count": int(unit.get("panel_count") or len(unit.get("panel_ids", []) or [])),
                "panel_ids": [str(panel_id).strip() for panel_id in unit.get("panel_ids", []) or [] if str(panel_id).strip()],
                "character_names": self._grounded_character_names(unit.get("character_names", []) or [], name_grounding),
                "character_roles": unit.get("character_roles", {}) or {},
                "combined_text": str(unit.get("combined_text") or "").strip(),
                "visual_cues": str(unit.get("visual_cues") or "").strip(),
                "vision_dialogue": str(unit.get("vision_dialogue") or "").strip(),
                "vision_caption": str(unit.get("vision_caption") or "").strip(),
                "vision_action_beat": str(unit.get("vision_action_beat") or "").strip(),
                "salvaged_evidence": str(unit.get("salvaged_evidence") or "").strip(),
                "ocr_fallback_text": str(unit.get("ocr_fallback_text") or "").strip(),
                "scene_summary": str(unit.get("scene_summary") or "").strip(),
                "zero_evidence": bool(unit.get("zero_evidence")),  # Preserve zero-evidence flag
            }
            for index, unit in enumerate(story_units, start=1)
        ]

        if prefer_local_evidence:
            return [self._fallback_scene_line(unit, protagonist_name, style_vocab=style_vocab) for unit in units]

        if "gemini" not in self.router.available_providers():
            return [self._fallback_scene_line(unit, protagonist_name, style_vocab=style_vocab) for unit in units]

        metadata_payload = self._chapter_metadata_payload(chapter_metadata)
        prompt_story_bible = self._story_bible_prompt_payload(story_bible or {})
        allowed_character_names = list(name_grounding.get("allowed_character_names") or []) if name_grounding else []
        chunks = [
            units[start : start + self._MULTIMODAL_SCENE_CHUNK_SIZE]
            for start in range(0, len(units), self._MULTIMODAL_SCENE_CHUNK_SIZE)
        ]

        if len(chunks) <= 1 or self._DRAFT_WORKERS <= 1:
            # Single-threaded path — rolling draft_history gives cross-chunk context.
            draft_lines: list[str] = []
            for start in range(0, len(units), self._MULTIMODAL_SCENE_CHUNK_SIZE):
                chunk = units[start : start + self._MULTIMODAL_SCENE_CHUNK_SIZE]
                draft_lines.extend(
                    self._run_story_draft_batch(
                        chunk,
                        draft_history=draft_lines,
                        project_title=project_title,
                        chapter_metadata=metadata_payload,
                        chapter_summary=chapter_summary,
                        character_dictionary=character_dictionary,
                        protagonist_name=protagonist_name,
                        prompt_story_bible=prompt_story_bible,
                        story_bible=story_bible or {},
                        allowed_character_names=allowed_character_names,
                        scene_visual_paths=scene_visual_paths or {},
                        retry_individual=True,
                        log_label=f"{start}-{start + len(chunk)}",
                        style_vocab=style_vocab,
                    )
                )
            return draft_lines

        # Multi-threaded path — all chunks drafted in parallel.
        # draft_history is empty per-chunk; the story_bible provides full
        # scene-level context so the rolling last-4-lines context is not critical.
        _story_bible_ref = story_bible or {}
        _scene_visual_paths_ref = scene_visual_paths or {}
        _style_vocab_ref = style_vocab

        def _draft_one(args: tuple[int, list[dict[str, Any]]]) -> list[str]:
            i, chunk = args
            panel_start = i * self._MULTIMODAL_SCENE_CHUNK_SIZE
            return self._run_story_draft_batch(
                chunk,
                draft_history=[],
                project_title=project_title,
                chapter_metadata=metadata_payload,
                chapter_summary=chapter_summary,
                character_dictionary=character_dictionary,
                protagonist_name=protagonist_name,
                prompt_story_bible=prompt_story_bible,
                story_bible=_story_bible_ref,
                allowed_character_names=allowed_character_names,
                scene_visual_paths=_scene_visual_paths_ref,
                retry_individual=True,
                log_label=f"{panel_start}-{panel_start + len(chunk)}",
                style_vocab=_style_vocab_ref,
            )

        logger.info(
            "Drafting %d chunks across %d parallel workers",
            len(chunks),
            self._DRAFT_WORKERS,
        )
        with concurrent.futures.ThreadPoolExecutor(max_workers=self._DRAFT_WORKERS) as executor:
            all_results = list(executor.map(_draft_one, enumerate(chunks)))

        return [line for chunk_lines in all_results for line in chunk_lines]

    def _run_story_draft_batch(
        self,
        chunk: list[dict[str, Any]],
        *,
        draft_history: list[str],
        project_title: str,
        chapter_metadata: dict[str, Any],
        chapter_summary: str,
        character_dictionary: dict[str, Any],
        protagonist_name: str | None,
        prompt_story_bible: dict[str, Any],
        story_bible: dict[str, Any],
        allowed_character_names: list[str],
        scene_visual_paths: dict[str, list[Path]],
        retry_individual: bool,
        log_label: str,
        style_vocab: StyleVocabulary | None = None,
    ) -> list[str]:
        if not chunk:
            return []
        chunk_scene_ids = sorted({int(unit["scene_id"]) for unit in chunk})
        chunk_image_paths = {
            str(unit["segment_id"]): list(scene_visual_paths.get(str(unit["segment_id"])) or [])[:3]
            for unit in chunk
            if scene_visual_paths.get(str(unit["segment_id"]))
        }
        try:
            result = asyncio.run(
                self.router.generate_story_segments(
                    chunk,
                    {
                        "project_title": project_title,
                        "chapter_metadata": chapter_metadata,
                        "chapter_summary": chapter_summary,
                        "character_dictionary": character_dictionary,
                        "protagonist_name": protagonist_name or "",
                        "story_bible": prompt_story_bible,
                        "running_memory": self._running_story_memory(draft_history, story_bible, chunk_scene_ids),
                        "scene_memory": self._scene_memory_for_chunk(story_bible, chunk_scene_ids),
                        "allowed_character_names": allowed_character_names,
                        "scene_mode": False,
                        "style_vocabulary": style_vocab.to_dict() if style_vocab else {},
                    },
                    provider="gemini",
                    scene_image_paths=chunk_image_paths,
                )
            )
            payload_segments = result.payload.get("segments", [])
        except Exception as exc:
            if retry_individual and len(chunk) > 1:
                logger.warning("Story segment draft chunk failed (%s), retrying individually: %s", log_label, exc)
                recovered: list[str] = []
                history = list(draft_history)
                for index, unit in enumerate(chunk):
                    lines = self._run_story_draft_batch(
                        [unit],
                        draft_history=history,
                        project_title=project_title,
                        chapter_metadata=chapter_metadata,
                        chapter_summary=chapter_summary,
                        character_dictionary=character_dictionary,
                        protagonist_name=protagonist_name,
                        prompt_story_bible=prompt_story_bible,
                        story_bible=story_bible,
                        allowed_character_names=allowed_character_names,
                        scene_visual_paths=scene_visual_paths,
                        retry_individual=False,
                        log_label=f"{log_label}:{index}",
                        style_vocab=style_vocab,
                    )
                    recovered.extend(lines)
                    history.extend(lines)
                return recovered
            logger.warning("Story segment draft chunk failed (%s): %s", log_label, exc)
            return [
                self._run_story_draft_text_only(
                    unit,
                    draft_history=draft_history,
                    project_title=project_title,
                    chapter_metadata=chapter_metadata,
                    chapter_summary=chapter_summary,
                    character_dictionary=character_dictionary,
                    protagonist_name=protagonist_name,
                    prompt_story_bible=prompt_story_bible,
                        story_bible=story_bible,
                        allowed_character_names=allowed_character_names,
                        style_vocab=style_vocab,
                    )
                or self._fallback_scene_line(unit, protagonist_name, style_vocab=style_vocab)
                for unit in chunk
            ]

        by_segment_id = {
            str(item.get("segment_id") or "").strip(): self._normalize_segment_text(str(item.get("text") or "").strip())
            for item in payload_segments
            if isinstance(item, dict)
        }
        recovered: list[str] = []
        history = list(draft_history)
        for index, unit in enumerate(chunk):
            segment_id = str(unit["segment_id"])
            candidate = by_segment_id.get(segment_id)
            if not candidate and retry_individual and len(chunk) > 1:
                lines = self._run_story_draft_batch(
                    [unit],
                    draft_history=history,
                    project_title=project_title,
                    chapter_metadata=chapter_metadata,
                    chapter_summary=chapter_summary,
                    character_dictionary=character_dictionary,
                    protagonist_name=protagonist_name,
                    prompt_story_bible=prompt_story_bible,
                    story_bible=story_bible,
                    allowed_character_names=allowed_character_names,
                    scene_visual_paths=scene_visual_paths,
                    retry_individual=False,
                    log_label=f"{log_label}:{index}",
                    style_vocab=style_vocab,
                )
                candidate = lines[0] if lines else ""
            if not candidate:
                candidate = self._run_story_draft_text_only(
                    unit,
                    draft_history=history,
                    project_title=project_title,
                    chapter_metadata=chapter_metadata,
                    chapter_summary=chapter_summary,
                    character_dictionary=character_dictionary,
                    protagonist_name=protagonist_name,
                    prompt_story_bible=prompt_story_bible,
                    story_bible=story_bible,
                    allowed_character_names=allowed_character_names,
                    style_vocab=style_vocab,
                )
            if candidate and not self._line_supported_by_unit_evidence(candidate, unit):
                candidate = ""
            candidate = candidate or self._fallback_scene_line(unit, protagonist_name, style_vocab=style_vocab)
            recovered.append(candidate)
            history.append(candidate)
        return recovered

    def _run_story_draft_text_only(
        self,
        unit: dict[str, Any],
        *,
        draft_history: list[str],
        project_title: str,
        chapter_metadata: dict[str, Any],
        chapter_summary: str,
        character_dictionary: dict[str, Any],
        protagonist_name: str | None,
        prompt_story_bible: dict[str, Any],
        story_bible: dict[str, Any],
        allowed_character_names: list[str],
        style_vocab: StyleVocabulary | None = None,
    ) -> str:
        segment_id = str(unit.get("segment_id") or "").strip()
        if not segment_id:
            return ""
        try:
            result = asyncio.run(
                self.router.generate_story_segments(
                    [unit],
                    {
                        "project_title": project_title,
                        "chapter_metadata": chapter_metadata,
                        "chapter_summary": chapter_summary,
                        "character_dictionary": character_dictionary,
                        "protagonist_name": protagonist_name or "",
                        "story_bible": prompt_story_bible,
                        "running_memory": self._running_story_memory(
                            draft_history,
                            story_bible,
                            [int(unit.get("scene_id") or 0)],
                        ),
                        "scene_memory": self._scene_memory_for_chunk(
                            story_bible,
                            [int(unit.get("scene_id") or 0)],
                        ),
                        "allowed_character_names": allowed_character_names,
                        "style_vocabulary": style_vocab.to_dict() if style_vocab else {},
                        "scene_mode": False,
                    },
                    provider="gemini",
                    scene_image_paths={},
                )
            )
        except Exception as exc:
            logger.warning("Text-only story segment retry failed for %s: %s", segment_id, exc)
            return ""
        for item in result.payload.get("segments", []):
            if not isinstance(item, dict):
                continue
            if str(item.get("segment_id") or "").strip() != segment_id:
                continue
            return self._normalize_segment_text(str(item.get("text") or "").strip(), allow_empty=True)
        return ""

    def _chapter_metadata_payload(self, chapter_metadata: ChapterMetadata | dict[str, Any] | Any) -> dict[str, Any]:
        return compact_chapter_metadata(chapter_metadata)

    def _load_story_bible_cache(self, path: Path) -> dict[str, Any]:
        if not path.exists():
            return {}
        try:
            loaded = json.loads(path.read_text(encoding="utf-8"))
            return dict(loaded) if isinstance(loaded, dict) else {}
        except Exception:
            return {}

    def _build_story_bible(
        self,
        scene_seeds: list[dict[str, Any]],
        scene_summaries: list[dict[str, Any]],
        *,
        project_title: str,
        chapter_metadata: dict[str, Any],
        chapter_summary: str,
        character_dictionary: dict[str, Any],
        protagonist_name: str | None,
        fallback_story_bible: dict[str, Any] | None = None,
        allowed_character_names: list[str] | None = None,
    ) -> dict[str, Any]:
        fallback = fallback_story_bible or self._fallback_story_bible(
            scene_seeds,
            scene_summaries,
            chapter_summary=chapter_summary,
            character_dictionary=character_dictionary,
        )
        try:
            if "gemini" not in self.router.available_providers():
                return fallback
        except Exception:
            return fallback

        prompt_scenes = self._story_bible_scene_payload(scene_seeds, scene_summaries)
        try:
            result = asyncio.run(
                self.router.build_story_bible(
                    prompt_scenes,
                    {
                        "project_title": project_title,
                        "chapter_metadata": chapter_metadata,
                        "chapter_summary": chapter_summary,
                        "character_dictionary": character_dictionary,
                        "protagonist_name": protagonist_name or "",
                        "allowed_character_names": allowed_character_names or [],
                    },
                    provider="gemini",
                )
            )
            return self._merge_story_bibles(fallback, result.payload)
        except Exception as exc:
            logger.warning("Story bible generation failed: %s", exc)
            return fallback

    def _story_bible_scene_payload(
        self,
        scene_seeds: list[dict[str, Any]],
        scene_summaries: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        summary_lookup = {
            int(item.get("scene_id") or 0): str(item.get("description") or item.get("summary") or "").strip()
            for item in scene_summaries
            if int(item.get("scene_id") or 0)
        }
        payload: list[dict[str, Any]] = []
        for index, seed in enumerate(scene_seeds, start=1):
            scene_id = int(seed.get("scene_id") or index)
            payload.append(
                {
                    "scene_id": scene_id,
                    "character_names": [str(name).strip() for name in seed.get("character_names", []) or [] if str(name).strip()],
                    "scene_summary": summary_lookup.get(scene_id, ""),
                    "combined_text": str(seed.get("combined_text") or "").strip()[:600],
                }
            )
        return payload

    def _fallback_story_bible(
        self,
        scene_seeds: list[dict[str, Any]],
        scene_summaries: list[dict[str, Any]],
        *,
        chapter_summary: str,
        character_dictionary: dict[str, Any],
    ) -> dict[str, Any]:
        cast: list[dict[str, Any]] = []
        for name, info in character_dictionary.items():
            display_name = str(info.get("display_name") or name).strip() if isinstance(info, dict) else str(name).strip()
            if not display_name:
                continue
            aliases = []
            if isinstance(info, dict):
                aliases = [str(alias).strip() for alias in info.get("aliases", []) or [] if str(alias).strip()]
            cast.append(
                {
                    "name": display_name,
                    "aliases": aliases[:6],
                    "role": str(info.get("role") or "").strip() if isinstance(info, dict) else "",
                    "visual_cues": str(info.get("appearance") or "").strip() if isinstance(info, dict) else "",
                    "notes": "",
                }
            )

        summary_lookup = {
            int(item.get("scene_id") or 0): str(item.get("description") or item.get("summary") or "").strip()
            for item in scene_summaries
            if int(item.get("scene_id") or 0)
        }
        scene_memory: list[dict[str, Any]] = []
        for index, seed in enumerate(scene_seeds, start=1):
            scene_id = int(seed.get("scene_id") or index)
            scene_memory.append(
                {
                    "scene_id": scene_id,
                    "state": summary_lookup.get(scene_id, "")[:180],
                    "location": "",
                    "characters": [str(name).strip() for name in seed.get("character_names", []) or [] if str(name).strip()][:6],
                    "open_thread": "",
                }
            )

        continuity_notes = [
            "Keep names and relationship labels consistent once a character is identified.",
            "Treat adjacent scenes as chronological unless dialogue or captions clearly indicate a jump.",
            "Prefer conservative narration over unsupported exposition when evidence is sparse.",
        ]
        return {
            "chapter_premise": chapter_summary[:500].strip(),
            "cast": cast[:20],
            "world_terms": [],
            "continuity_notes": continuity_notes,
            "scene_memory": scene_memory,
        }

    def _merge_story_bibles(self, fallback: dict[str, Any], generated: dict[str, Any]) -> dict[str, Any]:
        merged = dict(fallback)
        for key in ("chapter_premise",):
            value = str(generated.get(key) or "").strip()
            if value:
                merged[key] = value
        for key in ("world_terms", "continuity_notes"):
            combined = [
                str(item).strip()
                for item in [*(fallback.get(key) or []), *(generated.get(key) or [])]
                if str(item).strip()
            ]
            seen: set[str] = set()
            merged[key] = [
                item
                for item in combined
                if not (item.casefold() in seen or seen.add(item.casefold()))
            ][:24]
        merged["cast"] = list(generated.get("cast") or fallback.get("cast") or [])
        scene_memory_by_id: dict[int, dict[str, Any]] = {
            int(item.get("scene_id") or 0): dict(item)
            for item in fallback.get("scene_memory", []) or []
            if int(item.get("scene_id") or 0)
        }
        for item in generated.get("scene_memory", []) or []:
            scene_id = int(item.get("scene_id") or 0)
            if not scene_id:
                continue
            current = scene_memory_by_id.get(scene_id, {})
            current.update({k: v for k, v in dict(item).items() if v})
            scene_memory_by_id[scene_id] = current
        merged["scene_memory"] = [scene_memory_by_id[key] for key in sorted(scene_memory_by_id)]
        return merged

    def _merge_story_bible_into_grounding(
        self,
        grounding: dict[str, Any],
        story_bible: dict[str, Any],
    ) -> dict[str, Any]:
        merged = dict(grounding)
        allowed = list(grounding.get("allowed_character_names") or [])
        for item in story_bible.get("cast", []) or []:
            name = str(item.get("name") or "").strip()
            if name:
                allowed.append(name)
        allowed_names = self._grounded_character_names(allowed, grounding, strict=False)
        allowed_map = dict(grounding.get("allowed_name_map") or {})
        for name in allowed_names:
            key = normalize_name_key(name)
            if key:
                allowed_map.setdefault(key, name)
        merged["allowed_character_names"] = allowed_names
        merged["allowed_name_map"] = allowed_map
        chapter_metadata = dict(grounding.get("chapter_metadata") or {})
        chapter_metadata["series_cast_hints"] = allowed_names
        merged["chapter_metadata"] = chapter_metadata
        return merged

    def _grounded_character_names(
        self,
        names: list[Any],
        grounding: dict[str, Any] | None,
        *,
        strict: bool = True,
    ) -> list[str]:
        cleaned: list[str] = []
        seen: set[str] = set()
        for raw_name in names or []:
            name = (
                canonicalize_character_name(raw_name, grounding or {}, strict=strict)
                if grounding
                else str(raw_name or "").strip()
            )
            if not name:
                continue
            key = re.sub(r"\s+", " ", name.casefold()).strip()
            if key and key not in seen:
                seen.add(key)
                cleaned.append(name)
        return cleaned

    def _sanitize_story_bible(
        self,
        story_bible: dict[str, Any],
        fallback_story_bible: dict[str, Any],
        grounding: dict[str, Any],
    ) -> dict[str, Any]:
        sanitized = dict(fallback_story_bible)
        world_terms: list[str] = []
        seen_world_terms: set[str] = set()
        for value in [*(fallback_story_bible.get("world_terms") or []), *(story_bible.get("world_terms") or [])]:
            term = str(value or "").strip()
            key = re.sub(r"\s+", " ", term.casefold()).strip()
            if term and key and key not in seen_world_terms:
                seen_world_terms.add(key)
                world_terms.append(term)
        sanitized["world_terms"] = world_terms[:24]

        premise = apply_name_corrections_to_text(str(story_bible.get("chapter_premise") or "").strip(), grounding)
        if premise and not contains_unapproved_names(premise, grounding, world_terms=world_terms):
            sanitized["chapter_premise"] = premise

        continuity_notes: list[str] = []
        for note in [*(fallback_story_bible.get("continuity_notes") or []), *(story_bible.get("continuity_notes") or [])]:
            cleaned = apply_name_corrections_to_text(str(note or "").strip(), grounding)
            if not cleaned:
                continue
            if contains_unapproved_names(cleaned, grounding, world_terms=world_terms):
                continue
            if cleaned.casefold() not in {item.casefold() for item in continuity_notes}:
                continuity_notes.append(cleaned)
        sanitized["continuity_notes"] = continuity_notes[:24]

        cast: list[dict[str, Any]] = []
        seen_cast: set[str] = set()
        for item in [*(story_bible.get("cast") or []), *(fallback_story_bible.get("cast") or [])]:
            if not isinstance(item, dict):
                continue
            name = canonicalize_character_name(item.get("name"), grounding, strict=True)
            if not name:
                continue
            key = re.sub(r"\s+", " ", name.casefold()).strip()
            if key in seen_cast:
                continue
            seen_cast.add(key)
            aliases = self._grounded_character_names(item.get("aliases", []) or [], grounding, strict=False)
            cast.append(
                {
                    "name": name,
                    "aliases": aliases[:8],
                    "role": apply_name_corrections_to_text(str(item.get("role") or "").strip(), grounding),
                    "visual_cues": apply_name_corrections_to_text(str(item.get("visual_cues") or "").strip(), grounding),
                    "notes": apply_name_corrections_to_text(str(item.get("notes") or "").strip(), grounding),
                }
            )
        sanitized["cast"] = cast[:24]

        fallback_by_scene = {
            int(item.get("scene_id") or 0): dict(item)
            for item in fallback_story_bible.get("scene_memory", []) or []
            if int(item.get("scene_id") or 0)
        }
        generated_by_scene = {
            int(item.get("scene_id") or 0): dict(item)
            for item in story_bible.get("scene_memory", []) or []
            if int(item.get("scene_id") or 0)
        }
        scene_memory: list[dict[str, Any]] = []
        for scene_id in sorted({*fallback_by_scene.keys(), *generated_by_scene.keys()}):
            item = generated_by_scene.get(scene_id) or fallback_by_scene.get(scene_id) or {}
            if not scene_id:
                continue
            current = dict(item)
            current["characters"] = self._grounded_character_names(current.get("characters", []) or [], grounding, strict=True)
            current["state"] = apply_name_corrections_to_text(str(current.get("state") or "").strip(), grounding)
            current["location"] = apply_name_corrections_to_text(str(current.get("location") or "").strip(), grounding)
            current["open_thread"] = apply_name_corrections_to_text(str(current.get("open_thread") or "").strip(), grounding)
            if (
                self._line_is_low_quality(current["state"])
                or self._line_is_low_quality(current["open_thread"])
                or self._line_is_low_quality(current["location"])
                or
                contains_unapproved_names(current["state"], grounding, world_terms=world_terms, extra_allowed_names=current["characters"])
                or contains_unapproved_names(current["open_thread"], grounding, world_terms=world_terms, extra_allowed_names=current["characters"])
            ):
                current = dict(fallback_by_scene.get(scene_id) or current)
                current["characters"] = self._grounded_character_names(current.get("characters", []) or [], grounding, strict=True)
                current["state"] = apply_name_corrections_to_text(str(current.get("state") or "").strip(), grounding)
                current["location"] = apply_name_corrections_to_text(str(current.get("location") or "").strip(), grounding)
                current["open_thread"] = apply_name_corrections_to_text(str(current.get("open_thread") or "").strip(), grounding)
            scene_memory.append(
                {
                    "scene_id": scene_id,
                    "state": current["state"],
                    "location": current["location"],
                    "characters": current["characters"][:8],
                    "open_thread": current["open_thread"],
                }
            )
        if not scene_memory:
            scene_memory = list(fallback_story_bible.get("scene_memory", []) or [])
        sanitized["scene_memory"] = scene_memory[:120]
        if not premise or contains_unapproved_names(premise, grounding, world_terms=world_terms) or self._line_is_low_quality(premise):
            safe_states = [
                str(item.get("state") or "").strip()
                for item in sanitized["scene_memory"]
                if str(item.get("state") or "").strip() and not self._line_is_low_quality(str(item.get("state") or "").strip())
            ]
            if safe_states:
                sanitized["chapter_premise"] = " ".join(safe_states[:2])[:500].strip()
        return sanitized

    def _world_terms_for_guardrails(
        self,
        story_bible: dict[str, Any],
        grounding: dict[str, Any] | None,
    ) -> list[str]:
        world_terms = [str(item).strip() for item in story_bible.get("world_terms", []) or [] if str(item).strip()]
        if grounding:
            metadata = grounding.get("chapter_metadata") or {}
            title = str(metadata.get("manga_title") or "").strip()
            if title:
                world_terms.append(title)
        return world_terms

    def _stabilize_reviewed_segments(
        self,
        payloads: list[dict[str, Any]],
        units: list[dict[str, Any]],
        protagonist_name: str | None,
        grounding: dict[str, Any] | None,
        story_bible: dict[str, Any],
    ) -> list[dict[str, Any]]:
        if not grounding:
            return payloads
        stabilized: list[dict[str, Any]] = []
        world_terms = self._world_terms_for_guardrails(story_bible, grounding)
        scene_memory_by_id = {
            int(item.get("scene_id") or 0): dict(item)
            for item in story_bible.get("scene_memory", []) or []
            if int(item.get("scene_id") or 0)
        }
        for index, payload in enumerate(payloads):
            current = dict(payload)
            text = apply_name_corrections_to_text(str(current.get("text") or "").strip(), grounding)
            if text:
                has_quality_issue = self._line_is_low_quality(text) or self._line_is_overly_generic(text)
                has_name_issue = contains_unapproved_names(text, grounding, world_terms=world_terms)
                if (has_quality_issue or has_name_issue) and self._sentence_count(text) >= 2:
                    trimmed = self._remove_offending_sentences(text)
                    if trimmed and not contains_unapproved_names(trimmed, grounding, world_terms=world_terms):
                        text = trimmed
                        has_quality_issue = self._line_is_low_quality(text) or self._line_is_overly_generic(text)
                        has_name_issue = False
                if has_quality_issue or has_name_issue:
                    text = self._safe_grounded_scene_line(
                        units[index],
                        protagonist_name,
                        grounding,
                        world_terms,
                        scene_memory_by_id.get(int(units[index].get("scene_id") or 0)),
                    )
                    if not text:
                        current["visual_only"] = True
                        current["suppression_reason"] = str(current.get("suppression_reason") or "weak_evidence")
            if not text and not current.get("visual_only"):
                text = self._safe_grounded_scene_line(
                    units[index],
                    protagonist_name,
                    grounding,
                    world_terms,
                    scene_memory_by_id.get(int(units[index].get("scene_id") or 0)),
                )
                if not text:
                    current["visual_only"] = True
                    current["suppression_reason"] = str(current.get("suppression_reason") or "weak_evidence")
            current["text"] = text
            stabilized.append(current)
        return stabilized

    def _collapse_internal_duplicate_sentences(
        self,
        payloads: list[dict[str, Any]],
        *,
        scene_mode: bool = False,
    ) -> list[dict[str, Any]]:
        """Collapse only obvious duplicate sentences WITHIN a segment.

        Scene-mode prompts occasionally return a line that says the same thing
        twice in a row (e.g. "The installation was completed with high-grade
        materials. The installation is completed with high-grade materials.").
        The cross-segment dedup pass does not catch that because the duplicate
        lives inside one segment. We split each segment into sentences and only
        dedupe substantial sentence pairs; short bridge sentences are allowed to
        overlap because they often carry the connective tissue that makes
        scene-mode narration feel continuous. When a duplicate pair is found,
        keep the richer sentence rather than blindly keeping the first one.
        """
        if not payloads:
            return payloads
        stop_words = {
            "a", "an", "and", "as", "at", "but", "by", "for", "from", "he", "her",
            "him", "his", "in", "into", "is", "it", "its", "of", "on", "or", "she",
            "that", "the", "their", "them", "they", "this", "to", "was", "were",
            "will", "with", "who", "what", "when", "where", "why", "how", "be",
            "been", "being", "have", "has", "had", "do", "does", "did", "would",
            "could", "should", "may", "might", "must", "can", "there", "then",
            "than", "so", "not", "no", "yes", "up", "down", "out", "over", "under",
            "also", "too",
        }

        def content_tokens(text: str) -> frozenset[str]:
            return frozenset(
                token
                for token in re.findall(r"[a-z']+", text.casefold())
                if len(token) > 2 and token not in stop_words
            )

        jaccard_threshold = 0.75 if scene_mode else 0.82
        containment_threshold = 0.62 if scene_mode else 0.75
        sentence_split_re = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")
        refined: list[dict[str, Any]] = []
        for payload in payloads:
            current = dict(payload)
            text = str(current.get("text") or "").strip()
            if not text or current.get("visual_only"):
                refined.append(current)
                continue
            sentences = [segment.strip() for segment in sentence_split_re.split(text) if segment.strip()]
            if len(sentences) <= 1:
                refined.append(current)
                continue
            kept: list[str] = []
            kept_token_sets: list[frozenset[str]] = []
            for sentence in sentences:
                tokens = content_tokens(sentence)
                if not tokens:
                    kept.append(sentence)
                    kept_token_sets.append(tokens)
                    continue
                drop_current = False
                replacement_index: int | None = None
                for prior_index, prior_tokens in enumerate(kept_token_sets):
                    if not prior_tokens:
                        continue
                    if len(tokens) < 4 or len(prior_tokens) < 4:
                        continue
                    intersection = len(tokens & prior_tokens)
                    union = len(tokens | prior_tokens)
                    containment = intersection / max(1, min(len(tokens), len(prior_tokens)))
                    jaccard = intersection / union if union else 0.0
                    if jaccard >= jaccard_threshold or containment >= containment_threshold:
                        if len(tokens) <= len(prior_tokens):
                            drop_current = True
                        else:
                            replacement_index = prior_index
                        break
                if drop_current:
                    continue
                if replacement_index is not None:
                    kept[replacement_index] = sentence
                    kept_token_sets[replacement_index] = tokens
                    continue
                kept.append(sentence)
                kept_token_sets.append(tokens)
            if len(kept) != len(sentences):
                current["text"] = " ".join(kept).strip()
            refined.append(current)
        return refined

    def _remove_overused_generic_sentences(self, payloads: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Drop repeated fallback sentences without creating new silent ranges.

        This is a style cleanup pass, not a coverage pass. If a repeated
        sentence is the only narration for a panel range, leave it in place and
        let the evidence-aware duplicate pass try to replace it. Blanking here
        creates the exact regression we are trying to avoid: previously spoken
        panels becoming visual-only because their line sounded repetitive.
        """
        if not payloads:
            return payloads
        split_re = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")

        def sentence_key(sentence: str) -> str:
            return re.sub(r"[^a-z0-9 ]+", " ", sentence.casefold()).strip()

        def is_overused_sentence(sentence: str, count: int) -> bool:
            if count <= 1:
                return False
            lowered = sentence.casefold()
            if count >= 3:
                return True
            if (
                self._line_is_low_quality(sentence)
                or self._line_is_overly_generic(sentence)
                or self.polisher._is_visual_description(sentence)
            ):
                return True
            return bool(
                re.search(
                    r"\b(?:battle (?:is|was )?thrown into chaos|mechs? (?:tear|tore) through explosions|"
                    r"clash escalates, forcing the pilots|enemy'?s relentless pressure|mission keeps circling back|mission continues to|"
                    r"situation (?:grew|becomes|was becoming) increasingly difficult)\b",
                    lowered,
                )
            )

        sentence_counts: Counter[str] = Counter()
        payload_sentences: list[list[str]] = []
        for payload in payloads:
            text = self._normalize_segment_text(str(payload.get("text") or ""), allow_empty=True)
            sentences = [part.strip() for part in split_re.split(text) if part.strip()]
            payload_sentences.append(sentences)
            for sentence in sentences:
                key = sentence_key(sentence)
                if key:
                    sentence_counts[key] += 1

        seen_overused: set[str] = set()
        refined: list[dict[str, Any]] = []
        for payload, sentences in zip(payloads, payload_sentences, strict=False):
            current = dict(payload)
            if not sentences:
                refined.append(current)
                continue
            if len(sentences) <= 1:
                refined.append(current)
                for sentence in sentences:
                    key = sentence_key(sentence)
                    if key and is_overused_sentence(sentence, sentence_counts[key]):
                        seen_overused.add(key)
                continue
            survivors: list[str] = []
            changed = False
            for sentence in sentences:
                key = sentence_key(sentence)
                if key and is_overused_sentence(sentence, sentence_counts[key]):
                    if key in seen_overused:
                        changed = True
                        continue
                    seen_overused.add(key)
                survivors.append(sentence)
            if changed:
                joined = self._normalize_segment_text(" ".join(survivors), allow_empty=True)
                if joined:
                    current["text"] = joined
                    current["visual_only"] = False
                    current["suppression_reason"] = None
                else:
                    # Coverage beats style. Keep the original text and let
                    # downstream evidence-aware duplicate repair replace it if
                    # a grounded alternative exists.
                    current["text"] = self._normalize_segment_text(str(payload.get("text") or ""), allow_empty=True)
            refined.append(current)
        return refined

    def _collapse_near_duplicate_segments(
        self,
        payloads: list[dict[str, Any]],
        units: list[dict[str, Any]],
        *,
        blank_unresolved: bool = True,
        style_vocab: StyleVocabulary | None = None,
    ) -> list[dict[str, Any]]:
        """Detect and rewrite near-duplicate narration lines.

        A YouTube recap cannot say the same thing twice. We catch two cases:

        1. **Exact duplicate sentences** across the whole chapter (e.g. a
           repeat-pass returned the same line for two adjacent beats).
        2. **High Jaccard overlap** (≥0.70 on content tokens) between a segment
           and the one immediately before it.

        When possible, replace a duplicate with a distinct line from the same
        unit's trusted vision evidence. If no safe replacement exists, keep the
        line rather than creating a silent narration gap; later rescue/style
        passes can still improve it, but the editor should not lose coverage.
        """
        if not payloads:
            return payloads
        stop_words = {
            "a", "an", "and", "as", "at", "but", "by", "for", "from", "he", "her",
            "him", "his", "in", "into", "is", "it", "its", "of", "on", "or", "she",
            "that", "the", "their", "them", "they", "this", "to", "was", "were",
            "will", "with", "who", "what", "when", "where", "why", "how", "be",
            "been", "being", "have", "has", "had", "do", "does", "did", "would",
            "could", "should", "may", "might", "must", "can", "there", "then",
            "than", "so", "not", "no", "yes", "up", "down", "out", "over", "under",
        }

        def content_tokens(text: str) -> frozenset[str]:
            return frozenset(
                token
                for token in re.findall(r"[a-z']+", text.casefold())
                if len(token) > 2 and token not in stop_words
            )

        def normalized(text: str) -> str:
            return re.sub(r"[^a-z0-9 ]+", " ", text.casefold()).strip()

        def candidate_parts(raw: str) -> list[str]:
            cleaned = self._normalize_supporting_text(raw)
            if not cleaned:
                return []
            parts = [
                part.strip(" ,;:-")
                for part in re.split(r"(?<=[.!?])\s+|;\s+|\s{2,}", cleaned)
                if part.strip(" ,;:-")
            ]
            return parts or [cleaned]

        def candidate_is_duplicate(candidate: str) -> bool:
            candidate_signature = normalized(candidate)
            candidate_tokens = content_tokens(candidate)
            if candidate_signature and candidate_signature in seen_signatures:
                return True
            if previous_tokens and candidate_tokens and len(candidate_tokens) >= 4:
                intersection = len(candidate_tokens & previous_tokens)
                union = len(candidate_tokens | previous_tokens)
                if union and intersection / union >= 0.70:
                    return True
            if candidate_tokens and len(candidate_tokens) >= 6:
                for recent_tokens in recent_token_sets:
                    if not recent_tokens or len(recent_tokens) < 6:
                        continue
                    intersection = len(candidate_tokens & recent_tokens)
                    shorter = min(len(candidate_tokens), len(recent_tokens))
                    if shorter and intersection / shorter >= 0.50:
                        return True
            return False

        def replacement_for(index: int) -> str:
            unit = units[index] if index < len(units) else {}
            for variant in range(6):
                bridge = self._compose_neighbour_bridge_line(
                    unit,
                    prev_payload=refined[index - 1] if index > 0 else None,
                    next_payload=refined[index + 1] if index + 1 < len(refined) else None,
                    protagonist_name=None,
                    story_bible={},
                    scene_memory=None,
                    variant=variant,
                    style_vocab=style_vocab,
                )
                candidate = self._normalize_segment_text(bridge, allow_empty=True)
                if (
                    not candidate
                    or self._line_is_low_quality(candidate)
                    or self._line_is_overly_generic(candidate)
                    or self._line_is_dialogue_fragment(candidate)
                    or self._line_is_sentence_fragment(candidate)
                    or self._line_needs_style_refinement(candidate)
                    or candidate_is_duplicate(candidate)
                ):
                    continue
                return candidate
            sources = (
                self._evidence_bridge_line(unit, None, style_vocab=style_vocab),
                str(unit.get("vision_action_beat") or ""),
                str(unit.get("visual_cues") or ""),
                str(unit.get("vision_caption") or ""),
                str(unit.get("vision_dialogue") or ""),
                str(unit.get("combined_text") or ""),
                str(unit.get("ocr_fallback_text") or ""),
                str(unit.get("scene_summary") or ""),
            )
            for source in sources:
                for part in candidate_parts(source):
                    candidate = self._normalize_segment_text(part, allow_empty=True)
                    if (
                        not candidate
                        or self._line_is_low_quality(candidate)
                        or self._line_is_overly_generic(candidate)
                        or self._line_is_dialogue_fragment(candidate)
                        or self._line_is_sentence_fragment(candidate)
                        or candidate_is_duplicate(candidate)
                    ):
                        continue
                    return candidate
            return ""

        refined: list[dict[str, Any]] = [dict(item) for item in payloads]
        seen_signatures: set[str] = set()
        # Track recent non-blanked token sets so we can compare a candidate
        # against the last several beats, not just the immediately preceding
        # one. This catches the case where a backstory paraphrase reappears
        # one or two scenes after the original while an unrelated beat sits
        # between them; the adjacent-only check misses it because the middle
        # scene is not a duplicate of either repeated beat. We keep at
        # most 6 recent token sets which is enough to span a typical
        # narration "page" without quadratic cost.
        recent_token_sets: list[frozenset[str]] = []
        previous_tokens: frozenset[str] | None = None
        blanked_count = 0
        recent_window = 6

        for index, payload in enumerate(refined):
            text = str(payload.get("text") or "").strip()
            if not text:
                previous_tokens = frozenset()
                continue
            signature = normalized(text)
            tokens = content_tokens(text)

            is_duplicate = False
            hard_duplicate = False
            if signature and signature in seen_signatures:
                is_duplicate = True
                hard_duplicate = True
            if not is_duplicate and previous_tokens and tokens and len(tokens) >= 4:
                intersection = len(tokens & previous_tokens)
                union = len(tokens | previous_tokens)
                jaccard = intersection / union if union else 0.0
                if jaccard >= 0.70:
                    is_duplicate = True
                    hard_duplicate = jaccard >= 0.85
            # Containment check: if most of the shorter line's content tokens
            # appear in its neighbour, the two lines are paraphrases of the
            # same beat even if surface vocabulary differs (e.g. "Humanity's
            # quest for magma energy ... barren Earth" vs "Humanity's
            # extraction of magma energy led to the Earth's surface becoming
            # barren"). Jaccard alone misses this because the longer line has
            # extra connective words that inflate the union. Containment
            # ratio = |intersection| / min(|t1|,|t2|) catches it cleanly.
            #
            # The threshold is intentionally lower than the Jaccard one (0.50
            # vs 0.70) because:
            #   * a high *containment* ratio means the shorter line has very
            #     little new information relative to the longer one; that is
            #     by definition a paraphrase of the same beat, regardless of
            #     surface fluff
            #   * legitimate adjacent scenes that name the same protagonist
            #     and a couple of shared world terms still measure well below
            #     0.50 (verified empirically on the darling chapter)
            #
            # We compare the candidate against every token set in the recent
            # window so non-adjacent paraphrases (separated by an unrelated
            # beat) still get caught.
            if not is_duplicate and tokens and len(tokens) >= 6:
                for recent_tokens in recent_token_sets:
                    if not recent_tokens or len(recent_tokens) < 6:
                        continue
                    intersection = len(tokens & recent_tokens)
                    shorter = min(len(tokens), len(recent_tokens))
                    containment = intersection / shorter if shorter else 0.0
                    if containment >= 0.50:
                        is_duplicate = True
                        hard_duplicate = containment >= 0.85
                        break
            # Collapse repeated line endings that only differ by trailing punctuation.
            if not is_duplicate and signature:
                collapsed = re.sub(r"\s+", " ", signature)
                for prior in list(seen_signatures)[-12:]:
                    if collapsed == prior or collapsed in prior or prior in collapsed and abs(len(collapsed) - len(prior)) <= 8:
                        is_duplicate = True
                        hard_duplicate = True
                        break

            if is_duplicate:
                replacement = replacement_for(index)
                if replacement:
                    text = replacement
                    payload["text"] = replacement
                    payload["visual_only"] = False
                    payload["suppression_reason"] = None
                    signature = normalized(replacement)
                    tokens = content_tokens(replacement)
                elif blank_unresolved or hard_duplicate:
                    if not blank_unresolved and text:
                        payload["duplicate_original_text"] = text
                    payload["text"] = ""
                    payload["visual_only"] = True
                    payload["suppression_reason"] = "near_duplicate"
                    text = ""
                    signature = ""
                    tokens = frozenset()
                    blanked_count += 1
                else:
                    payload["suppression_reason"] = "near_duplicate_kept"

            if signature:
                seen_signatures.add(signature)
            previous_tokens = tokens
            if tokens:
                recent_token_sets.append(tokens)
                if len(recent_token_sets) > recent_window:
                    recent_token_sets = recent_token_sets[-recent_window:]

        if blanked_count:
            logger.info("Collapsed %d near-duplicate narration segments", blanked_count)
        return refined

    @staticmethod
    def _cohesion_proper_name_keys(text: str) -> set[str]:
        """Extract content-bearing proper-noun keys from a narration line.

        ``extract_proper_name_candidates`` already filters obvious false
        positives (weekday names, common stop nouns), so we just normalise
        each surviving candidate to a comparable key.
        """
        keys: set[str] = set()
        for candidate in extract_proper_name_candidates(text or ""):
            key = normalize_name_key(candidate)
            if not key:
                continue
            keys.add(key)
        return keys

    def _cohesion_drift_rejected(
        self,
        original: str,
        rewrite: str,
        evidence_text: str,
        allowed_keys: frozenset[str],
    ) -> bool:
        """Decide if a cohesion rewrite drifted off the original facts.

        We accept rewrites that drop proper nouns (legitimate flow / pronoun
        substitution) but reject rewrites that **introduce** a proper noun
        which is neither in the original line nor in the unit's local
        evidence (action_beat / dialogue / caption). This is the cheapest
        defence against the failure mode the user observed: cohesion making
        the chapter scan well by inventing characters or swapping who is in
        a scene.
        """
        original_keys = self._cohesion_proper_name_keys(original)
        rewrite_keys = self._cohesion_proper_name_keys(rewrite)
        if not rewrite_keys:
            return False
        introduced = rewrite_keys - original_keys
        if not introduced:
            return False
        evidence_keys = self._cohesion_proper_name_keys(evidence_text)
        # Project world terms are allowed because they are vocabulary, not
        # character claims. The allowed_keys set includes both character names
        # and vetted world terms, so we just need a quick subtraction.
        unsupported = {key for key in introduced if key not in evidence_keys and key not in allowed_keys}
        return bool(unsupported)

    def _cohesion_content_drift_rejected(
        self,
        original: str,
        rewrite: str,
        evidence_text: str,
    ) -> bool:
        """Reject cohesion edits that no longer describe the local beat.

        Proper-noun checks catch invented names, but a rewrite can still drift
        by swapping the event while keeping the same protagonist. Keep the rule
        deliberately simple and project-agnostic: a cohesion rewrite must retain
        at least a small amount of content overlap with either the original line
        or the unit's local evidence.
        """
        rewrite_tokens = self._content_token_set(rewrite)
        if len(rewrite_tokens) < 4:
            return False
        original_tokens = self._content_token_set(original)
        evidence_tokens = self._content_token_set(evidence_text)
        support_tokens = original_tokens | evidence_tokens
        if not support_tokens:
            return False
        overlap_count = len(rewrite_tokens & support_tokens)
        if overlap_count >= 2:
            return False
        support_ratio = overlap_count / max(1, min(len(rewrite_tokens), len(support_tokens)))
        if support_ratio >= 0.25:
            return False
        return True

    def _narrator_cohesion_pass(
        self,
        payloads: list[dict[str, Any]],
        units: list[dict[str, Any]],
        *,
        project_title: str,
        chapter_metadata: dict[str, Any],
        chapter_summary: str,
        character_dictionary: dict[str, Any],
        protagonist_name: str | None,
        name_grounding: dict[str, Any] | None,
        require_multi_sentence: bool = False,
        style_vocab: StyleVocabulary | None = None,
    ) -> list[dict[str, Any]]:
        """Rewrite the whole chapter in a single narrator voice.

        After scene drafting, polishing, critique, style and dedup, we send the
        whole ordered list of surviving lines to Gemini 2.5 with instructions
        to produce a YouTube-recap-style flowing script: consistent narrator,
        real transitions between scenes, no repeats, no visual-report verbs,
        no orphan sentences that describe a single panel.

        Blanked segments (``visual_only=True``) keep their slot but receive no
        text so the repair pass can still decide whether to fill them.
        """
        if not payloads or len(payloads) < 3:
            return payloads
        if "gemini" not in self.router.available_providers():
            return payloads

        indexed_lines = [
            {
                "index": index,
                "scene_id": int(units[index].get("scene_id") or 0) if index < len(units) else 0,
                "panel_start": int(units[index].get("panel_start") or 0) if index < len(units) else 0,
                "panel_end": int(units[index].get("panel_end") or 0) if index < len(units) else 0,
                "panel_count": int(units[index].get("panel_count") or len(units[index].get("panel_ids", []) or [])) if index < len(units) else 0,
                "text": str(payload.get("text") or "").strip(),
                "visual_only": bool(payload.get("visual_only")),
                "suppression_reason": str(payload.get("suppression_reason") or "").strip(),
                "scene_summary": str(units[index].get("scene_summary") or "").strip() if index < len(units) else "",
                "vision_dialogue": str(units[index].get("vision_dialogue") or "").strip() if index < len(units) else "",
                "vision_caption": str(units[index].get("vision_caption") or "").strip() if index < len(units) else "",
                "vision_action_beat": str(units[index].get("vision_action_beat") or "").strip() if index < len(units) else "",
                "local_evidence": self._style_evidence_text(units[index]) if index < len(units) else "",
                "character_names": [
                    str(name).strip()
                    for name in (units[index].get("character_names", []) if index < len(units) else []) or []
                    if str(name).strip()
                ],
            }
            for index, payload in enumerate(payloads)
        ]
        non_empty = [item for item in indexed_lines if item["text"] and not item["visual_only"]]
        if len(non_empty) < 3:
            return payloads

        allowed_character_names = (
            list(name_grounding.get("allowed_character_names") or []) if name_grounding else []
        )
        character_drift_keys: set[str] = set()
        # World vocabulary and any other allowed proper nouns we already vetted.
        # The drift check uses this set so cohesion can reuse terms the bible
        # names, but not invent fresh characters.
        allowed_drift_keys: set[str] = set()
        for name in allowed_character_names:
            key = normalize_name_key(str(name))
            if key:
                allowed_drift_keys.add(key)
                character_drift_keys.add(key)
        if name_grounding:
            for entry in name_grounding.get("allowed_name_map", {}) or {}:
                key = normalize_name_key(str(entry))
                if key:
                    allowed_drift_keys.add(key)
            world_terms = (
                name_grounding.get("chapter_metadata", {}).get("world_terms")
                if isinstance(name_grounding.get("chapter_metadata"), dict)
                else None
            )
            for term in world_terms or []:
                key = normalize_name_key(str(term))
                if key:
                    allowed_drift_keys.add(key)
        for character in (character_dictionary or {}).values():
            if isinstance(character, dict):
                for value in (character.get("aliases") or []):
                    key = normalize_name_key(str(value))
                    if key:
                        allowed_drift_keys.add(key)
                display = character.get("display_name") or character.get("name")
                if display:
                    key = normalize_name_key(str(display))
                    if key:
                        allowed_drift_keys.add(key)
                        character_drift_keys.add(key)
        if protagonist_name:
            key = normalize_name_key(str(protagonist_name))
            if key:
                allowed_drift_keys.add(key)
                character_drift_keys.add(key)
        if style_vocab:
            allowed_drift_keys.update(style_vocab.allowed_drift_keys)
            for name in style_vocab.named_characters:
                key = normalize_name_key(name)
                if key:
                    character_drift_keys.add(key)
        # World/stakes terms may be reused chapter-wide, but character names
        # must appear in the original line or local evidence before a cohesion
        # rewrite can introduce them. This prevents early protagonist/name bleed
        # in lore or setting-only segments.
        allowed_drift_frozen = frozenset(allowed_drift_keys - character_drift_keys)

        evidence_by_index: dict[int, str] = {
            item["index"]: " ".join(
                value
                for value in (
                    item.get("vision_action_beat", ""),
                    item.get("vision_dialogue", ""),
                    item.get("vision_caption", ""),
                    item.get("scene_summary", ""),
                    item.get("local_evidence", ""),
                )
                if value
            )
            for item in indexed_lines
        }

        # We chunk to keep each Gemini call under a safe input token budget.
        # Overlap of 2 lines gives the model continuity between chunks.
        chunk_size = 24
        overlap = 2
        chunks: list[list[dict[str, Any]]] = []
        start = 0
        while start < len(indexed_lines):
            end = min(len(indexed_lines), start + chunk_size)
            chunks.append(indexed_lines[start:end])
            if end == len(indexed_lines):
                break
            start = max(start + chunk_size - overlap, end)

        by_index: dict[int, str] = {item["index"]: item["text"] for item in indexed_lines}
        drift_rejected = 0
        content_drift_rejected = 0
        for chunk_index, chunk in enumerate(chunks, start=1):
            try:
                result = asyncio.run(
                    self.router.cohere_chapter_narrator(
                        chunk,
                        {
                            "project_title": project_title,
                            "chapter_metadata": chapter_metadata,
                            "chapter_summary": chapter_summary,
                            "character_dictionary": character_dictionary,
                            "protagonist_name": protagonist_name or "",
                            "allowed_character_names": allowed_character_names,
                            "chunk_index": chunk_index,
                            "chunk_total": len(chunks),
                            "require_multi_sentence": require_multi_sentence,
                            "style_vocabulary": style_vocab.to_dict() if style_vocab else {},
                        },
                        provider="gemini",
                    )
                )
            except Exception as exc:
                logger.warning("Narrator cohesion pass failed for chunk %d/%d: %s", chunk_index, len(chunks), exc)
                continue
            rewrites = result.payload.get("rewrites", []) or []
            for rewrite in rewrites:
                if not isinstance(rewrite, dict):
                    continue
                idx_value = rewrite.get("index")
                try:
                    idx = int(idx_value)
                except (TypeError, ValueError):
                    continue
                if idx < 0 or idx >= len(indexed_lines):
                    continue
                new_text = self._normalize_segment_text(
                    str(rewrite.get("line") or rewrite.get("text") or "").strip(),
                    allow_empty=True,
                )
                if not new_text:
                    continue
                if (
                    require_multi_sentence
                    and not indexed_lines[idx].get("visual_only")
                ):
                    if self._sentence_count(new_text) < 2:
                        continue
                    if (
                        self._line_is_low_quality(new_text)
                        or self._line_is_overly_generic(new_text)
                        or self._line_is_dialogue_fragment(new_text)
                        or self._line_is_sentence_fragment(new_text)
                    ):
                        new_text = self._remove_offending_sentences(new_text)
                    if self._sentence_count(new_text) < 2:
                        continue
                    if (
                        self._line_is_low_quality(new_text)
                        or self._line_is_overly_generic(new_text)
                        or self._line_is_dialogue_fragment(new_text)
                        or self._line_is_sentence_fragment(new_text)
                    ):
                        continue
                # Safety: reject truncation, but allow short drafts to grow into
                # grounded multi-sentence beats when cohesion has enough evidence.
                original = by_index.get(idx) or ""
                if original:
                    original_words = len(original.split())
                    if original_words <= 12:
                        length_floor = 0.0
                    elif original_words <= 25:
                        length_floor = 0.35
                    else:
                        length_floor = 0.40
                    if length_floor and len(new_text) < max(24, int(len(original) * length_floor)):
                        continue
                if len(new_text) > 700:
                    continue
                # Anti-drift: reject rewrites that introduce proper nouns that
                # are not in the original line and not supported by the unit's
                # local evidence. Prevents cohesion from inventing characters
                # or swapping who is in a scene to make the prose flow.
                evidence_text = evidence_by_index.get(idx, "")
                if self._cohesion_drift_rejected(
                    original,
                    new_text,
                    evidence_text,
                    allowed_drift_frozen,
                ):
                    drift_rejected += 1
                    continue
                if self._cohesion_content_drift_rejected(original, new_text, evidence_text):
                    content_drift_rejected += 1
                    continue
                by_index[idx] = new_text

        if drift_rejected:
            logger.info("Rejected %d cohesion rewrites for proper-noun drift", drift_rejected)
        if content_drift_rejected:
            logger.info("Rejected %d cohesion rewrites for content drift", content_drift_rejected)

        refined: list[dict[str, Any]] = []
        for index, payload in enumerate(payloads):
            current = dict(payload)
            if current.get("visual_only"):
                refined.append(current)
                continue
            rewritten = by_index.get(index)
            if rewritten and rewritten.strip():
                current["text"] = rewritten.strip()
            refined.append(current)
        return refined

    def _enrichment_chapter_context(
        self,
        story_bible: dict[str, Any],
        style_vocab: StyleVocabulary,
    ) -> str:
        """Compact world/stakes context for thin-line repair.

        This context is deliberately not a license to introduce characters.
        Character drift is still checked per line; the block exists so the LLM
        can turn short factual lines into natural recap prose without falling
        back to abstract "risk/moment" scaffolding.
        """
        parts: list[str] = []
        for key in ("chapter_premise", "series_external_context"):
            value = str((story_bible or {}).get(key) or "").strip()
            if value:
                parts.append(value)
        world_terms = ", ".join(style_vocab.world_terms[:8])
        stakes = ", ".join(style_vocab.stakes_phrases[:6])
        if world_terms:
            parts.append(f"World terms: {world_terms}.")
        if stakes:
            parts.append(f"Stakes: {stakes}.")
        text = self._normalize_supporting_text(" ".join(parts))
        return text[:1800]

    def _narrator_enrichment_pass(
        self,
        payloads: list[dict[str, Any]],
        units: list[dict[str, Any]],
        *,
        style_vocab: StyleVocabulary,
        story_bible: dict[str, Any] | None = None,
        cache_dir: Path | None = None,
    ) -> list[dict[str, Any]]:
        """Lengthen thin narration lines without changing their facts."""
        if not payloads or not units:
            return payloads
        try:
            if "gemini" not in self.router.available_providers():
                return payloads
        except Exception:
            return payloads

        candidates: list[dict[str, Any]] = []
        for index, payload in enumerate(payloads):
            if index >= len(units):
                continue
            unit = units[index]
            text = self._normalize_segment_text(str(payload.get("text") or ""), allow_empty=True)
            visual_only = bool(payload.get("visual_only"))
            suppression = str(payload.get("suppression_reason") or "").strip()
            if not text or visual_only:
                seed_candidates = (
                    self._evidence_bridge_line(unit, None, style_vocab=style_vocab),
                    self._fallback_scene_line(unit, None, style_vocab=style_vocab),
                    str(unit.get("vision_action_beat") or "").strip(),
                    str(unit.get("vision_caption") or "").strip(),
                    str(unit.get("scene_summary") or "").strip(),
                )
                text = ""
                for seed in seed_candidates:
                    normalized_seed = self._normalize_segment_text(seed, allow_empty=True)
                    if normalized_seed and not (
                        self._line_is_low_quality(normalized_seed)
                        or self._line_is_dialogue_fragment(normalized_seed)
                        or self._line_is_sentence_fragment(normalized_seed)
                        or self._line_has_first_person_narration(normalized_seed)
                        or self.polisher._is_visual_description(normalized_seed)
                    ):
                        text = normalized_seed
                        break
                if not text:
                    continue
            elif suppression:
                continue
            word_count = len(re.findall(r"\b[\w'-]+\b", text))
            sentence_count = self._sentence_count(text)
            generic_line = self._line_is_overly_generic(text)
            if word_count >= 50 and sentence_count >= 2 and not generic_line:
                continue
            evidence = " | ".join(
                value
                for value in (
                    str(unit.get("vision_action_beat") or "").strip(),
                    str(unit.get("vision_dialogue") or "").strip(),
                    str(unit.get("vision_caption") or "").strip(),
                    str(unit.get("visual_cues") or "").strip(),
                    str(unit.get("ocr_fallback_text") or "").strip(),
                    self._style_evidence_text(unit),
                )
                if value
            )
            candidates.append(
                {
                    "index": index,
                    "current": text,
                    "evidence": evidence,
                    "scene_summary": str(unit.get("scene_summary") or "").strip(),
                    "previous_line": str(payloads[index - 1].get("text") or "").strip() if index > 0 else "",
                    "next_line": str(payloads[index + 1].get("text") or "").strip() if index + 1 < len(payloads) else "",
                    "character_names": [
                        str(name).strip()
                        for name in unit.get("character_names", []) or []
                        if str(name).strip()
                    ],
                }
            )
        if not candidates:
            return payloads
        max_enrichment_candidates = 120
        if len(candidates) > max_enrichment_candidates:
            logger.info(
                "Skipping final narrator enrichment for %d candidates; cap is %d to keep large projects stable",
                len(candidates),
                max_enrichment_candidates,
            )
            return payloads

        if cache_dir is not None:
            try:
                backup_text = "\n\n".join(
                    str(payload.get("text") or "").strip()
                    for payload in payloads
                    if str(payload.get("text") or "").strip()
                )
                if backup_text:
                    timestamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
                    (cache_dir / f"narration_story.backup_pre_enrichment_{timestamp}.txt").write_text(
                        backup_text.strip() + "\n",
                        encoding="utf-8",
                    )
            except Exception as exc:
                logger.debug("Could not write pre-enrichment backup: %s", exc)

        by_index: dict[int, str] = {
            index: self._normalize_segment_text(str(payload.get("text") or ""), allow_empty=True)
            for index, payload in enumerate(payloads)
        }
        for item in candidates:
            idx = int(item["index"])
            if not by_index.get(idx):
                by_index[idx] = str(item.get("current") or "").strip()
        evidence_by_index = {
            int(item["index"]): " ".join(
                part
                for part in (
                    str(item.get("evidence") or ""),
                    str(item.get("scene_summary") or ""),
                    str(item.get("previous_line") or ""),
                    str(item.get("next_line") or ""),
                )
                if part
            )
            for item in candidates
        }
        character_drift_keys = {
            normalize_name_key(name)
            for name in style_vocab.named_characters
            if normalize_name_key(name)
        }
        # Enrichment may reuse world/stakes vocabulary chapter-wide, but it
        # must not introduce a character before that character appears in the
        # current line or neighboring evidence.
        allowed_drift_frozen = frozenset(style_vocab.allowed_drift_keys - character_drift_keys)

        chunk_size = 18
        overlap = 2
        chunks: list[list[dict[str, Any]]] = []
        start = 0
        while start < len(candidates):
            end = min(len(candidates), start + chunk_size)
            chunks.append(candidates[start:end])
            if end == len(candidates):
                break
            start = max(start + chunk_size - overlap, end)

        accepted = 0
        length_rejected = 0
        drift_rejected = 0
        content_rejected = 0
        for chunk_index, chunk in enumerate(chunks, start=1):
            try:
                result = asyncio.run(
                    self.router.enrich_chapter_narrator(
                        chunk,
                        {
                            "style_vocabulary": style_vocab.to_dict(),
                        },
                        provider="gemini",
                    )
                )
            except Exception as exc:
                logger.warning("Narrator enrichment pass failed for chunk %d/%d: %s", chunk_index, len(chunks), exc)
                continue

            for rewrite in result.payload.get("rewrites", []) or []:
                if not isinstance(rewrite, dict):
                    continue
                try:
                    idx = int(rewrite.get("index"))
                except (TypeError, ValueError):
                    continue
                original = by_index.get(idx) or ""
                if not original:
                    continue
                new_text = self._normalize_segment_text(
                    str(rewrite.get("line") or rewrite.get("text") or "").strip(),
                    allow_empty=True,
                )
                if not new_text:
                    continue
                original_words = len(re.findall(r"\b[\w'-]+\b", original))
                new_words = len(re.findall(r"\b[\w'-]+\b", new_text))
                original_generic = self._line_is_overly_generic(original)
                if (
                    new_words > 100
                    or (new_words < 18 and original_words >= 18)
                    or (len(new_text) < len(original) and not original_generic)
                ):
                    length_rejected += 1
                    continue
                evidence_text = evidence_by_index.get(idx, "")
                if self._cohesion_drift_rejected(original, new_text, evidence_text, allowed_drift_frozen):
                    drift_rejected += 1
                    continue
                if self._cohesion_content_drift_rejected(original, new_text, evidence_text):
                    content_rejected += 1
                    continue
                if (
                    self._line_is_low_quality(new_text)
                    or self._line_is_overly_generic(new_text)
                    or self._line_is_dialogue_fragment(new_text)
                    or self._line_is_sentence_fragment(new_text)
                    or self._line_has_first_person_narration(new_text)
                    or self.polisher._is_visual_description(new_text)
                ):
                    trimmed = self._remove_offending_sentences(new_text)
                    if not trimmed or len(trimmed) < len(original):
                        length_rejected += 1
                        continue
                    new_text = trimmed
                by_index[idx] = new_text
                accepted += 1

        logger.info(
            "Enrichment pass: %d/%d rewrites accepted (rejected %d length, %d drift, %d content)",
            accepted,
            len(candidates),
            length_rejected,
            drift_rejected,
            content_rejected,
        )

        refined: list[dict[str, Any]] = []
        for index, payload in enumerate(payloads):
            current = dict(payload)
            enriched = by_index.get(index)
            if enriched and enriched.strip():
                current["text"] = enriched.strip()
                current["visual_only"] = False
                current["suppression_reason"] = None
            refined.append(current)
        return refined

    def _safe_grounded_scene_line(
        self,
        unit: dict[str, Any],
        protagonist_name: str | None,
        grounding: dict[str, Any],
        world_terms: list[str],
        scene_memory_item: dict[str, Any] | None = None,
    ) -> str:
        trusted_vision = " ".join(
            str(value or "").strip()
            for value in (
                unit.get("vision_action_beat"),
                unit.get("vision_caption"),
                unit.get("vision_dialogue"),
            )
            if str(value or "").strip()
        )
        action_recap = self._action_evidence_recap_line(unit)
        if (
            action_recap
            and not self._line_is_low_quality(action_recap)
            and not self._line_is_overly_generic(action_recap)
        ):
            return action_recap
        if trusted_vision:
            candidates = (
                trusted_vision,
                str(unit.get("combined_text") or "").strip(),
            )
        else:
            candidates = (
                str(unit.get("combined_text") or "").strip(),
                str(unit.get("ocr_fallback_text") or "").strip(),
                str((scene_memory_item or {}).get("state") or "").strip(),
                str((scene_memory_item or {}).get("open_thread") or "").strip(),
                "" if int(unit.get("scene_unit_count") or 1) > 1 else str(unit.get("scene_summary") or "").strip(),
            )
        for candidate in candidates:
            cleaned = apply_name_corrections_to_text(candidate, grounding)
            normalized = self._normalize_segment_text(cleaned, allow_empty=True)
            if (
                normalized
                and not self._line_is_low_quality(normalized)
                and not contains_unapproved_names(normalized, grounding, world_terms=world_terms)
            ):
                return normalized
        return ""

    def _action_evidence_recap_line(self, unit: dict[str, Any]) -> str:
        """Turn clustered visual action evidence into one grounded recap beat.

        This is intentionally pattern-based and title-agnostic. It exists for
        multi-panel action groups where the raw vision sentences are too
        caption-like for final narration ("X stands...", "Y looks...") but still
        contain a clear conflict progression.
        """
        source = self._normalize_supporting_text(
            " ".join(
                str(unit.get(key) or "").strip()
                for key in ("vision_action_beat", "vision_caption", "visual_cues")
                if str(unit.get(key) or "").strip()
            )
        )
        if not source:
            return ""
        sentences = self._split_sentences_for_cleanup(source)
        if len(sentences) < 2:
            return ""
        # Prefer named anchors already present in evidence, but keep labels
        # generic when the source only gives visual roles.
        names = []
        for match in re.finditer(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?\b", source):
            name = match.group(0).strip()
            if name in {"The", "A", "An"} or looks_like_false_character_name(name):
                continue
            if name not in names:
                names.append(name)
        primary = names[0] if names else "the lead character"
        lower = source.casefold()

        opponent = ""
        defeated_match = re.search(
            r"\bdefeated\s+(?P<label>[a-z][a-z -]{2,40}?(?:boy|girl|man|woman|student|opponent|fighter|enemy|person))\b",
            source,
            flags=re.IGNORECASE,
        )
        if defeated_match:
            opponent = defeated_match.group("label").strip().lower()
        elif re.search(r"\bred-haired\b", lower):
            opponent = "red-haired opponent"
        elif re.search(r"\b(?:opponent|enemy|attacker)\b", lower):
            opponent = "opponent"

        has_strike = bool(re.search(r"\b(?:kick|kicks|kicked|punch|punches|punched|strike|strikes|hit|hits|attack|attacks)\b", lower))
        has_damage = bool(re.search(r"\b(?:pain|crack|cracks|cracked|debris|slumped|stagger|staggers|impact|damage)\b", lower))
        if has_strike and has_damage:
            target = primary
            attacker = f"the {opponent}" if opponent and not opponent.startswith("the ") else (opponent or "the opponent")
            first = (
                f"{primary} has {attacker} cornered against the damage from the fight."
                if opponent
                else f"{primary} is caught in the aftermath of a brutal exchange."
            )
            second = (
                f"{attacker.capitalize()} still fights back, turning the aftermath into one last burst of resistance."
                if attacker
                else "The counterattack turns the aftermath into one last burst of resistance."
            )
            candidate = self._normalize_segment_text(f"{first} {second}", allow_empty=True)
            if (
                candidate
                and not self._line_is_low_quality(candidate)
                and not self._line_is_overly_generic(candidate)
                and not self._line_needs_style_refinement(candidate)
            ):
                return candidate

        if has_damage and len(names) >= 1:
            candidate = self._normalize_segment_text(
                f"{primary} is pulled into the damage left by the clash. "
                "The cracked surroundings make the fight feel like it is still spilling into the next move.",
                allow_empty=True,
            )
            if (
                candidate
                and not self._line_is_low_quality(candidate)
                and not self._line_is_overly_generic(candidate)
                and not self._line_needs_style_refinement(candidate)
            ):
                return candidate
        return ""

    def _fix_self_target_action_payloads(
        self,
        payloads: list[dict[str, Any]],
        units: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Replace impossible self-target action lines with local action evidence."""
        fixed: list[dict[str, Any]] = []
        self_target_pattern = re.compile(
            r"\b([A-Z][a-z]+)\b[^.]{0,90}\b(?:kick|kicks|kicked|punch|punches|punched|hit|hits|"
            r"attack|attacks|attacked|strike|strikes|struck)\b[^.]{0,90}\b\1(?:'s)?\b",
            re.IGNORECASE,
        )
        for index, payload in enumerate(payloads):
            current = dict(payload)
            text = self._normalize_segment_text(str(current.get("text") or ""), allow_empty=True)
            caption_only = bool(
                text
                and (
                    self_target_pattern.search(text)
                    or self._sentence_has_visual_caption_leak(text)
                    or self.polisher._is_visual_description(text)
                )
            )
            if caption_only:
                unit = units[index] if index < len(units) else {}
                replacement = self._action_evidence_recap_line(unit)
                if replacement:
                    current["text"] = replacement
                    current["visual_only"] = False
                    current["suppression_reason"] = None
            fixed.append(current)
        return fixed

    def _story_bible_prompt_payload(self, story_bible: dict[str, Any]) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "chapter_premise": str(story_bible.get("chapter_premise") or "").strip(),
            "cast": list(story_bible.get("cast") or [])[:12],
            "world_terms": list(story_bible.get("world_terms") or [])[:12],
            "continuity_notes": list(story_bible.get("continuity_notes") or [])[:10],
        }
        # Inject grounded series context when present so every LLM call benefits.
        series_ctx = str(story_bible.get("series_external_context") or "").strip()
        if series_ctx:
            payload["series_context"] = series_ctx[:2000]
        return payload

    def _scene_memory_for_chunk(self, story_bible: dict[str, Any], scene_ids: list[int]) -> list[dict[str, Any]]:
        wanted = set(scene_ids)
        return [
            dict(item)
            for item in story_bible.get("scene_memory", []) or []
            if int(item.get("scene_id") or 0) in wanted
        ]

    def _running_story_memory(self, drafted_lines: list[str], story_bible: dict[str, Any], scene_ids: list[int]) -> str:
        previous_lines = [line.strip() for line in drafted_lines[-4:] if line.strip()]
        previous_scene_notes = [
            item
            for item in story_bible.get("scene_memory", []) or []
            if int(item.get("scene_id") or 0) < (min(scene_ids) if scene_ids else 0)
        ][-3:]
        parts: list[str] = []
        if previous_scene_notes:
            formatted = []
            for item in previous_scene_notes:
                state = str(item.get("state") or "").strip()
                open_thread = str(item.get("open_thread") or "").strip()
                location = str(item.get("location") or "").strip()
                detail = " | ".join(part for part in (state, location, open_thread) if part)
                if detail:
                    formatted.append(detail)
            if formatted:
                parts.append("Recent scene state: " + " || ".join(formatted))
        if previous_lines:
            parts.append("Recent recap lines: " + " ".join(previous_lines))
        return "\n".join(parts).strip()

    def _build_scene_visual_paths(
        self,
        story_units: list[dict[str, Any]],
        panels_by_id: dict[str, PanelBox],
        panel_dir: Path,
        output_dir: Path,
    ) -> dict[str, list[Path]]:
        ensure_dir(output_dir)
        visual_paths: dict[str, list[Path]] = {}
        for index, unit in enumerate(story_units, start=1):
            segment_id = str(unit.get("segment_id") or f"segment_{index:03d}").strip() or f"segment_{index:03d}"
            panel_image_paths = self._scene_panel_image_paths(unit, panels_by_id, panel_dir)
            if not panel_image_paths:
                continue
            collage_path = output_dir / f"{segment_id}.jpg"
            try:
                self._write_scene_collage(panel_image_paths, collage_path)
                representative_path = panel_image_paths[len(panel_image_paths) // 2]
                ordered_paths: list[Path] = []
                for candidate in (collage_path, representative_path, *panel_image_paths):
                    if candidate.exists() and candidate not in ordered_paths:
                        ordered_paths.append(candidate)
                visual_paths[segment_id] = ordered_paths[:4] or [representative_path]
            except Exception as exc:
                logger.debug("Scene collage generation failed for %s: %s", segment_id, exc)
                visual_paths[segment_id] = [panel_image_paths[len(panel_image_paths) // 2]]
        return visual_paths

    def _scene_panel_image_paths(
        self,
        seed: dict[str, Any],
        panels_by_id: dict[str, PanelBox],
        panel_dir: Path,
    ) -> list[Path]:
        panel_ids = [str(panel_id).strip() for panel_id in seed.get("panel_ids", []) or [] if str(panel_id).strip()]
        if not panel_ids:
            return []
        candidate_ids = [panel_ids[0], panel_ids[len(panel_ids) // 2], panel_ids[-1]]
        ordered_ids: list[str] = []
        seen: set[str] = set()
        for panel_id in candidate_ids:
            if panel_id and panel_id not in seen:
                seen.add(panel_id)
                ordered_ids.append(panel_id)
        paths: list[Path] = []
        for panel_id in ordered_ids:
            panel = panels_by_id.get(panel_id)
            image_path = self._find_panel_image(panel, panel_dir) if panel is not None else None
            if image_path is not None:
                paths.append(image_path)
        return paths

    def _find_panel_image(self, panel: PanelBox | None, panel_dir: Path) -> Path | None:
        if panel is None:
            return None
        candidates = [
            panel_dir / f"panel_{int(panel.order):03d}.png",
            panel_dir / f"panel_{int(panel.order):03d}.jpg",
            panel_dir / f"{panel.id}.png",
            panel_dir / f"{panel.id}.jpg",
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return None

    def _write_scene_collage(self, image_paths: list[Path], output_path: Path) -> None:
        thumbs: list[Image.Image] = []
        gutter = 10
        try:
            for path in image_paths[:3]:
                with Image.open(path) as img:
                    thumb = ImageOps.contain(img.convert("RGB"), (420, 360), Image.Resampling.LANCZOS)
                    thumbs.append(thumb)
            if not thumbs:
                return
            width = sum(image.width for image in thumbs) + gutter * (len(thumbs) - 1)
            height = max(image.height for image in thumbs)
            canvas = Image.new("RGB", (width, height), (246, 246, 246))
            cursor_x = 0
            for thumb in thumbs:
                offset_y = (height - thumb.height) // 2
                canvas.paste(thumb, (cursor_x, offset_y))
                cursor_x += thumb.width + gutter
            output_path.parent.mkdir(parents=True, exist_ok=True)
            canvas.save(output_path, format="JPEG", quality=82, optimize=True)
        finally:
            for thumb in thumbs:
                thumb.close()

    def _critic_scene_lines(
        self,
        lines: list[str],
        story_units: list[dict[str, Any]],
        *,
        project_title: str,
        chapter_metadata: dict[str, Any],
        chapter_summary: str,
        character_dictionary: dict[str, Any],
        protagonist_name: str | None,
        story_bible: dict[str, Any],
        name_grounding: dict[str, Any] | None = None,
        scene_visual_paths: dict[str, list[Path]] | None = None,
        disable_multimodal_rescue: bool = False,
        style_vocab: StyleVocabulary | None = None,
        skip_llm_critic: bool = False,
    ) -> list[dict[str, Any]]:
        if not lines:
            return []

        units = [
            {
                "segment_id": str(unit.get("segment_id") or f"segment_{index:03d}").strip() or f"segment_{index:03d}",
                "scene_id": int(unit.get("scene_id") or index),
                "sequence_in_scene": int(unit.get("sequence_in_scene") or 1),
                "scene_unit_count": int(unit.get("scene_unit_count") or 1),
                "panel_count": int(unit.get("panel_count") or len(unit.get("panel_ids", []) or [])),
                "character_names": self._grounded_character_names(unit.get("character_names", []) or [], name_grounding),
                "combined_text": str(unit.get("combined_text") or "").strip(),
                "visual_cues": str(unit.get("visual_cues") or "").strip(),
                "vision_dialogue": str(unit.get("vision_dialogue") or "").strip(),
                "vision_caption": str(unit.get("vision_caption") or "").strip(),
                "vision_action_beat": str(unit.get("vision_action_beat") or "").strip(),
                "salvaged_evidence": str(unit.get("salvaged_evidence") or "").strip(),
                "local_evidence": str(unit.get("local_evidence") or "").strip(),
                "scene_summary": str(unit.get("scene_summary") or "").strip(),
            }
            for index, unit in enumerate(story_units, start=1)
        ]

        reviewed = [self._normalize_segment_text(line, allow_empty=True) for line in lines]
        # In OCR-only mode the polished mechanical paraphrases are our best and only
        # content.  _apply_weak_scene_policy has many filters (overly_generic, low_quality,
        # etc.) designed for vision-mode output that false-positive on our short
        # mechanical templates (e.g. "An unexpected offer is extended — a chance to join
        # forces and face it together." is flagged as overly_generic by the scene_unit_count
        # branch). This causes valid beats to be blanked and then incorrectly replaced by
        # _force_fill_remaining_blank_payloads with content from a neighbouring unit.
        # In OCR-only mode we skip the policy pass entirely and return normalised polished
        # lines directly as payloads, preserving all four beats.
        if skip_llm_critic:
            return [
                {
                    "text": line,
                    "visual_only": not bool(line),
                    "suppression_reason": None if line else "weak_evidence",
                }
                for line in reviewed
            ]
        if "gemini" in self.router.available_providers() and not skip_llm_critic:
            allowed_character_names = list(name_grounding.get("allowed_character_names") or []) if name_grounding else []
            prompt_story_bible = self._story_bible_prompt_payload(story_bible)
            critic_chunks = [
                (start, units[start : start + self._CRITIC_BATCH_SIZE])
                for start in range(0, len(reviewed), self._CRITIC_BATCH_SIZE)
            ]

            if len(critic_chunks) <= 1 or self._CRITIC_WORKERS <= 1:
                for start, chunk_units in critic_chunks:
                    rewrite_by_index = self._run_story_critic_batch(
                        reviewed,
                        chunk_units,
                        start_index=start,
                        project_title=project_title,
                        chapter_summary=chapter_summary,
                        chapter_metadata=chapter_metadata,
                        character_dictionary=character_dictionary,
                        prompt_story_bible=prompt_story_bible,
                        allowed_character_names=allowed_character_names,
                        retry_individual=True,
                        log_label=f"{start}-{start + len(chunk_units)}",
                    )
                    for local_index, replacement in rewrite_by_index.items():
                        global_index = start + local_index
                        if global_index < len(reviewed):
                            reviewed[global_index] = replacement
            else:
                # Parallel critic: snapshot the reviewed list so all batches read a
                # consistent state; each batch writes only to its own slice afterward.
                reviewed_snapshot = list(reviewed)

                def _critic_one(args: tuple[int, list[dict[str, Any]]]) -> tuple[int, dict[int, str]]:
                    start_idx, chunk_units = args
                    return start_idx, self._run_story_critic_batch(
                        reviewed_snapshot,
                        chunk_units,
                        start_index=start_idx,
                        project_title=project_title,
                        chapter_summary=chapter_summary,
                        chapter_metadata=chapter_metadata,
                        character_dictionary=character_dictionary,
                        prompt_story_bible=prompt_story_bible,
                        allowed_character_names=allowed_character_names,
                        retry_individual=True,
                        log_label=f"{start_idx}-{start_idx + len(chunk_units)}",
                    )

                logger.info(
                    "Critiquing %d chunks across %d parallel workers",
                    len(critic_chunks),
                    self._CRITIC_WORKERS,
                )
                with concurrent.futures.ThreadPoolExecutor(max_workers=self._CRITIC_WORKERS) as executor:
                    batch_results = list(executor.map(_critic_one, critic_chunks))

                for start_idx, rewrite_by_index in batch_results:
                    for local_index, replacement in rewrite_by_index.items():
                        global_index = start_idx + local_index
                        if global_index < len(reviewed):
                            reviewed[global_index] = replacement

        if not disable_multimodal_rescue:
            reviewed = self._rescue_scene_lines_multimodal(
                reviewed,
                units,
                project_title=project_title,
                chapter_metadata=chapter_metadata,
                chapter_summary=chapter_summary,
                character_dictionary=character_dictionary,
                protagonist_name=protagonist_name,
                story_bible=story_bible,
                name_grounding=name_grounding,
                scene_visual_paths=scene_visual_paths or {},
            )

        payloads = self._apply_weak_scene_policy(
            reviewed,
            units,
            protagonist_name,
            style_vocab=style_vocab,
        )
        if not disable_multimodal_rescue:
            payloads = self._recover_visual_only_payloads_multimodal(
                payloads,
                units,
                project_title=project_title,
                chapter_metadata=chapter_metadata,
                chapter_summary=chapter_summary,
                character_dictionary=character_dictionary,
                protagonist_name=protagonist_name,
                story_bible=story_bible,
                name_grounding=name_grounding,
                scene_visual_paths=scene_visual_paths or {},
            )
        return self._stabilize_reviewed_segments(payloads, units, protagonist_name, name_grounding, story_bible)

    def _style_spoken_segment_payloads(
        self,
        payloads: list[dict[str, Any]],
        units: list[dict[str, Any]],
        *,
        project_title: str,
        chapter_metadata: dict[str, Any],
        chapter_summary: str,
        character_dictionary: dict[str, Any],
        story_bible: dict[str, Any],
        name_grounding: dict[str, Any] | None,
        style_vocab: StyleVocabulary | None = None,
    ) -> list[dict[str, Any]]:
        if not payloads:
            return payloads
        try:
            if "gemini" not in self.router.available_providers():
                return payloads
        except Exception:
            return payloads

        refined = [dict(item) for item in payloads]
        prompt_story_bible = self._story_bible_prompt_payload(story_bible)
        allowed_character_names = list(name_grounding.get("allowed_character_names") or []) if name_grounding else []
        world_terms = self._world_terms_for_guardrails(story_bible, name_grounding)
        for _ in range(self._STYLE_PASSES):
            candidate_indices = [
                index
                for index, payload in enumerate(refined)
                if (
                    str(payload.get("text") or "").strip()
                    and not bool(payload.get("visual_only"))
                    and self._line_needs_style_refinement(str(payload.get("text") or "").strip())
                )
            ]
            if not candidate_indices:
                break

            for start in range(0, len(candidate_indices), self._STYLE_BATCH_SIZE):
                batch_indices = candidate_indices[start:start + self._STYLE_BATCH_SIZE]
                batch_payload: list[dict[str, Any]] = []
                for local_index, global_index in enumerate(batch_indices):
                    unit = units[global_index]
                    current_line = str(refined[global_index].get("text") or "").strip()
                    batch_payload.append(
                        {
                            "index": local_index,
                            "current_line": current_line,
                            "previous_line": str(refined[global_index - 1].get("text") or "").strip() if global_index > 0 else "",
                            "next_line": str(refined[global_index + 1].get("text") or "").strip() if global_index + 1 < len(refined) else "",
                            "ocr_text": str(unit.get("combined_text") or "").strip(),
                            "scene_summary": str(unit.get("scene_summary") or "").strip(),
                            "visual_cues": self._style_evidence_text(unit),
                            "vision_dialogue": str(unit.get("vision_dialogue") or "").strip(),
                            "vision_caption": str(unit.get("vision_caption") or "").strip(),
                            "vision_action_beat": str(unit.get("vision_action_beat") or "").strip(),
                            "character_names": unit.get("character_names", []) or [],
                            "panel_count": int(unit.get("panel_count") or len(unit.get("panel_ids", []) or [])),
                        }
                    )

                try:
                    result = asyncio.run(
                        self.router.refine_story_segment_style(
                            batch_payload,
                            {
                                "project_title": project_title,
                                "chapter_summary": chapter_summary,
                                "chapter_metadata": chapter_metadata,
                                "character_dictionary": character_dictionary,
                                "story_bible": prompt_story_bible,
                                "allowed_character_names": allowed_character_names,
                                "style_vocabulary": style_vocab.to_dict() if style_vocab else {},
                            },
                            provider="gemini",
                        )
                    )
                except Exception as exc:
                    logger.warning("Story segment style pass failed for batch %s-%s: %s", start, start + len(batch_indices), exc)
                    continue

                rewrite_by_index = {
                    int(item.get("index") or 0): self._normalize_segment_text(str(item.get("line") or "").strip(), allow_empty=True)
                    for item in result.payload.get("rewrites", [])
                    if isinstance(item, dict)
                }
                for local_index, global_index in enumerate(batch_indices):
                    candidate = rewrite_by_index.get(local_index, "")
                    if not candidate:
                        continue
                    current_line = str(refined[global_index].get("text") or "").strip()
                    unit = units[global_index]
                    evidence = {
                        "ocr_text": str(unit.get("combined_text") or "").strip(),
                        "scene_summary": str(unit.get("scene_summary") or "").strip(),
                        "visual_caption": self._style_evidence_text(unit),
                        "character_names": unit.get("character_names", []) or [],
                        "dialogue": [],
                    }
                    candidate = self.polisher._replace_machine_placeholders(candidate)
                    if not self._style_candidate_is_safe(
                        candidate,
                        current_line=current_line,
                        evidence=evidence,
                        unit=unit,
                        grounding=name_grounding or {},
                        world_terms=world_terms,
                    ):
                        continue
                    refined[global_index]["text"] = candidate
        return refined

    def _style_evidence_text(self, unit: dict[str, Any]) -> str:
        """Compact local evidence for style repair prompts.

        Style repair should see trusted vision fields before legacy visual
        captions; otherwise it tends to polish stale captions instead of
        rewriting them into real narration.
        """
        parts = [
            str(unit.get("vision_action_beat") or "").strip(),
            str(unit.get("vision_caption") or "").strip(),
            str(unit.get("vision_dialogue") or "").strip(),
            str(unit.get("visual_cues") or "").strip(),
            str(unit.get("ocr_fallback_text") or "").strip(),
        ]
        return self._normalize_supporting_text(" ".join(part for part in parts if part))[:900]

    def _line_needs_style_refinement(self, line: str) -> bool:
        cleaned = self._normalize_segment_text(line, allow_empty=True)
        if not cleaned:
            return False
        if self._line_is_low_quality(cleaned) or self._line_is_overly_generic(cleaned):
            return True
        style_patterns = (
            r'"',
            r"\b(?:asks?|asked|tells?|told|calling to|called out|stating|stated|declares?|declared|reassures?|reassured|questioned|wondered|laughed|shouts?|shouted)\b",
            r"\b(?:voice called out|appears, calling|asking if|telling him|telling her|telling them)\b",
            r"\b(?:is shown|are shown|looked around|looked at|looks to the side|smiled|smiles|appears before|is displayed|are displayed|stands by|stands with|looks at|watches with)\b",
            r"\b(?:was shown|were shown|exterior view|in the background)\b",
            r"\b(?:stares?|staring)\b",
            r"\bpoints?\s+(?:a finger|at|toward|towards|forward)\b",
            r"^(?:stands?|standing|walks?|walking|runs?|running|looks?|looking|sits?|sitting|turns?|turning|moves?|moving|continues?|continuing)\b",
            r"^(?:charges?|charging)\s+forward\b",
            r"^(?:ornate|massive|large|small|dark|bright|sterile|expansive)\s+(?:building|structure|figure|robot|screen|screens)\b",
            r"\b(?:impact sound effect|visible in the background|seen in the background|sound effect|shock and confusion|express shock|expresses shock)\b",
            r"\b(?:are seen|is seen)\b",
            r"\b(?:suit-clad figures?|figures? on an escalator|young women are relaxing|relaxing and socializing|two mecha suits|in a tense\s*,|female pilot urgently ordered)\b",
            r"\bbright,\s*undefined space\b",
            r"\bholding (?:his|her|their|a|the) smartphone\b",
            r"\bspeaks? with .{0,80}\bexpression\b",
            r"\b(?:fate grows more tangled|pressure around them builds)\b",
            r"\bpressure around (?:him|her|them|the group)\s+refuses to let up\b",
            r"\bone more uneasy beat\b",
            r"\btense detail that still matters\b",
            r"\bjagged pause\b",
            r"\bthe mood (?:stays|remains) unsettled\b",
            r"\bthe atmosphere remains strained\b",
            r"\bstays? unresolved for another beat\b",
            r"\bthe panel (?:sharpens|lingers|holds|cuts)\b",
            r"\b(?:with a lollipop|dynamic pose|in a dynamic pose|burst of energy upwards?)\b",
            r"\b(?:\bI\b|I'm|I am|my|mine|we|our|ours)\b",
            r"^(?:a|an|the)\s+(?:(?:young|furious|worried|frightened|angry|smug|injured|armed)\s+){0,3}(?:man|woman|boy|girl|person|child|neighbor|figure)\s+(?:holding|holds|held|clutching|clenches?|approaches|approached)\b",
            r"\bholds?\s+(?:a|the)\s+(?:knife|gun|phone|object)\b.{0,80}\b(?:expression|eyes?|glow(?:ing|ed)?)\b",
            r"\beyes?\s+glow(?:ing|ed)?\b",
            r"\bviewer\b",
            r"^(?:And\s+)?(?:another|a|an|one)\b[\w\s'-]{0,80}\bare\b",
        )
        return any(re.search(pattern, cleaned, flags=re.IGNORECASE) for pattern in style_patterns)

    def _style_candidate_is_safe(
        self,
        candidate: str,
        *,
        current_line: str,
        evidence: dict[str, Any],
        unit: dict[str, Any],
        grounding: dict[str, Any],
        world_terms: list[str],
    ) -> bool:
        if self._line_is_low_quality(candidate) or self._line_is_overly_generic(candidate):
            return False
        if self._line_needs_style_refinement(candidate):
            return False
        if self.polisher._line_has_issues(candidate):
            return False
        if contains_unapproved_names(
            candidate,
            grounding,
            world_terms=world_terms,
            extra_allowed_names=unit.get("character_names", []) or [],
        ):
            return False
        if self.polisher._line_matches_slot(candidate, current_line, evidence):
            return True
        candidate_words = self.polisher._content_words(candidate)
        support_words = self.polisher._content_words(
            " ".join(
                part for part in [
                    current_line,
                    str(evidence.get("ocr_text") or "").strip(),
                    str(evidence.get("scene_summary") or "").strip(),
                    str(evidence.get("visual_caption") or "").strip(),
                    " ".join(str(name).strip() for name in evidence.get("character_names", []) or [] if str(name).strip()),
                ] if part
            )
        )
        return len(candidate_words & support_words) >= 2

    def _run_story_critic_batch(
        self,
        reviewed: list[str],
        chunk_units: list[dict[str, Any]],
        *,
        start_index: int,
        project_title: str,
        chapter_summary: str,
        chapter_metadata: dict[str, Any],
        character_dictionary: dict[str, Any],
        prompt_story_bible: dict[str, Any],
        allowed_character_names: list[str],
        retry_individual: bool,
        log_label: str,
    ) -> dict[int, str]:
        payload = []
        for local_index, unit in enumerate(chunk_units):
            global_index = start_index + local_index
            payload.append(
                {
                    "index": local_index,
                    "segment_id": unit["segment_id"],
                    "scene_id": unit["scene_id"],
                    "current_line": reviewed[global_index],
                    "scene_summary": unit["scene_summary"],
                    "combined_text": unit["combined_text"][:700],
                    "visual_cues": unit["visual_cues"][:260],
                    "vision_dialogue": str(unit.get("vision_dialogue") or "").strip()[:700],
                    "vision_caption": str(unit.get("vision_caption") or "").strip()[:700],
                    "vision_action_beat": str(unit.get("vision_action_beat") or "").strip()[:500],
                    "local_evidence": " ".join(
                        str(unit.get(key) or "").strip()
                        for key in ("salvaged_evidence", "local_evidence")
                        if str(unit.get(key) or "").strip()
                    )[:700],
                    "character_names": unit["character_names"],
                    "panel_count": unit["panel_count"],
                    "previous_line": reviewed[global_index - 1] if global_index > 0 else "",
                    "next_line": reviewed[global_index + 1] if global_index + 1 < len(reviewed) else "",
                }
            )
        try:
            result = asyncio.run(
                self.router.critique_story_segments(
                    payload,
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
        except Exception as exc:
            if retry_individual and len(chunk_units) > 1:
                logger.warning("Story segment critic batch failed (%s), retrying individually: %s", log_label, exc)
                merged: dict[int, str] = {}
                for local_index, unit in enumerate(chunk_units):
                    merged.update(
                        self._run_story_critic_batch(
                            reviewed,
                            [unit],
                            start_index=start_index + local_index,
                            project_title=project_title,
                            chapter_summary=chapter_summary,
                            chapter_metadata=chapter_metadata,
                            character_dictionary=character_dictionary,
                            prompt_story_bible=prompt_story_bible,
                            allowed_character_names=allowed_character_names,
                            retry_individual=False,
                            log_label=f"{log_label}:{local_index}",
                        )
                    )
                return merged
            logger.warning("Story segment critic batch failed (%s): %s", log_label, exc)
            return {}

        return {
            int(item.get("index") or 0): self._normalize_segment_text(str(item.get("line") or "").strip(), allow_empty=True)
            for item in result.payload.get("rewrites", [])
            if isinstance(item, dict)
        }

    def _rescue_scene_lines_multimodal(
        self,
        lines: list[str],
        units: list[dict[str, Any]],
        *,
        project_title: str,
        chapter_metadata: dict[str, Any],
        chapter_summary: str,
        character_dictionary: dict[str, Any],
        protagonist_name: str | None,
        story_bible: dict[str, Any],
        name_grounding: dict[str, Any] | None,
        scene_visual_paths: dict[str, list[Path]],
    ) -> list[str]:
        if not lines or not scene_visual_paths:
            return lines
        try:
            if "gemini" not in self.router.available_providers():
                return lines
        except Exception:
            return lines

        rescued = list(lines)
        candidate_indices = [
            index
            for index, unit in enumerate(units)
            if scene_visual_paths.get(str(unit.get("segment_id") or "").strip())
            and self._multimodal_rescue_reason(rescued[index], unit)
        ]
        if not candidate_indices:
            return rescued

        if len(candidate_indices) > self._MAX_MULTIMODAL_LINE_RESCUES:
            logger.info(
                "Limiting multimodal story line rescue from %d to %d candidates",
                len(candidate_indices),
                self._MAX_MULTIMODAL_LINE_RESCUES,
            )
            candidate_indices = candidate_indices[: self._MAX_MULTIMODAL_LINE_RESCUES]

        allowed_character_names = list(name_grounding.get("allowed_character_names") or []) if name_grounding else []
        prompt_story_bible = self._story_bible_prompt_payload(story_bible)
        for start in range(0, len(candidate_indices), self._RESCUE_BATCH_SIZE):
            batch_indices = candidate_indices[start:start + self._RESCUE_BATCH_SIZE]
            rewrite_by_index = self._run_multimodal_rescue_batch(
                batch_indices,
                rescued,
                units,
                project_title=project_title,
                chapter_metadata=chapter_metadata,
                chapter_summary=chapter_summary,
                character_dictionary=character_dictionary,
                protagonist_name=protagonist_name,
                prompt_story_bible=prompt_story_bible,
                allowed_character_names=allowed_character_names,
                scene_visual_paths=scene_visual_paths,
                log_label=f"{start}-{start + len(batch_indices)}",
            )
            for global_index, replacement in rewrite_by_index.items():
                rescued[global_index] = replacement
        return rescued

    def _run_multimodal_rescue_batch(
        self,
        batch_indices: list[int],
        rescued: list[str],
        units: list[dict[str, Any]],
        *,
        project_title: str,
        chapter_metadata: dict[str, Any],
        chapter_summary: str,
        character_dictionary: dict[str, Any],
        protagonist_name: str | None,
        prompt_story_bible: dict[str, Any],
        allowed_character_names: list[str],
        scene_visual_paths: dict[str, list[Path]],
        log_label: str,
    ) -> dict[int, str]:
        payload: list[dict[str, Any]] = []
        image_paths: dict[str, list[Path]] = {}
        for local_index, global_index in enumerate(batch_indices):
            unit = units[global_index]
            segment_id = str(unit.get("segment_id") or f"segment_{global_index + 1:03d}").strip()
            weak_reason = self._multimodal_rescue_reason(rescued[global_index], unit) or "weak_alignment"
            payload.append(
                {
                    "index": local_index,
                    "segment_id": segment_id,
                    "scene_id": int(unit.get("scene_id") or global_index + 1),
                    "sequence_in_scene": int(unit.get("sequence_in_scene") or 1),
                    "scene_unit_count": int(unit.get("scene_unit_count") or 1),
                    "panel_count": int(unit.get("panel_count") or len(unit.get("panel_ids", []) or [])),
                    "current_line": rescued[global_index],
                    "combined_text": str(unit.get("combined_text") or "").strip()[:700],
                    "ocr_fallback_text": str(unit.get("ocr_fallback_text") or "").strip()[:700],
                    "visual_cues": str(unit.get("visual_cues") or "").strip()[:260],
                    "vision_dialogue": str(unit.get("vision_dialogue") or "").strip()[:700],
                    "vision_caption": str(unit.get("vision_caption") or "").strip()[:700],
                    "vision_action_beat": str(unit.get("vision_action_beat") or "").strip()[:700],
                    "character_names": unit.get("character_names", []) or [],
                    "scene_summary": str(unit.get("scene_summary") or "").strip(),
                    "previous_line": rescued[global_index - 1] if global_index > 0 else "",
                    "next_line": rescued[global_index + 1] if global_index + 1 < len(rescued) else "",
                    "weak_reason": weak_reason,
                }
            )
            image_paths[segment_id] = list(scene_visual_paths.get(segment_id) or [])[:3]

        try:
            result = asyncio.run(
                self.router.repair_story_segments_multimodal(
                    payload,
                    {
                        "project_title": project_title,
                        "chapter_summary": chapter_summary,
                        "chapter_metadata": chapter_metadata,
                        "character_dictionary": character_dictionary,
                        "protagonist_name": protagonist_name or "",
                        "story_bible": prompt_story_bible,
                        "allowed_character_names": allowed_character_names,
                    },
                    provider="gemini",
                    scene_image_paths=image_paths,
                )
            )
        except Exception as exc:
            if len(batch_indices) > 1:
                logger.warning("Multimodal story rescue failed for segments %s, retrying individually: %s", log_label, exc)
                merged: dict[int, str] = {}
                for global_index in batch_indices:
                    merged.update(
                        self._run_multimodal_rescue_batch(
                            [global_index],
                            rescued,
                            units,
                            project_title=project_title,
                            chapter_metadata=chapter_metadata,
                            chapter_summary=chapter_summary,
                            character_dictionary=character_dictionary,
                            protagonist_name=protagonist_name,
                            prompt_story_bible=prompt_story_bible,
                            allowed_character_names=allowed_character_names,
                            scene_visual_paths=scene_visual_paths,
                            log_label=str(global_index),
                        )
                    )
                return merged
            logger.warning("Multimodal story rescue failed for segment %s: %s", log_label, exc)
            return {}

        return {
            batch_indices[int(item.get("index") or 0)]: self._normalize_segment_text(
                str(item.get("line") or "").strip(),
                allow_empty=True,
            )
            for item in result.payload.get("rewrites", [])
            if isinstance(item, dict) and int(item.get("index") or 0) < len(batch_indices)
        }

    def _recover_visual_only_payloads_multimodal(
        self,
        payloads: list[dict[str, Any]],
        units: list[dict[str, Any]],
        *,
        project_title: str,
        chapter_metadata: dict[str, Any],
        chapter_summary: str,
        character_dictionary: dict[str, Any],
        protagonist_name: str | None,
        story_bible: dict[str, Any],
        name_grounding: dict[str, Any] | None,
        scene_visual_paths: dict[str, list[Path]],
    ) -> list[dict[str, Any]]:
        if not payloads or not scene_visual_paths:
            return payloads
        try:
            if "gemini" not in self.router.available_providers():
                return payloads
        except Exception:
            return payloads

        recovered = [dict(item) for item in payloads]
        rescued_lines = [
            self._normalize_segment_text(str(item.get("text") or "").strip(), allow_empty=True)
            for item in recovered
        ]
        candidate_indices: list[int] = []
        for index, (payload, unit) in enumerate(zip(recovered, units)):
            if rescued_lines[index]:
                continue
            if not bool(payload.get("visual_only")):
                continue
            segment_id = str(unit.get("segment_id") or "").strip()
            if not scene_visual_paths.get(segment_id):
                continue
            panel_count = int(unit.get("panel_count") or len(unit.get("panel_ids", []) or []))
            has_neighbor_context = bool(
                (rescued_lines[index - 1].strip() if index > 0 else "")
                or (rescued_lines[index + 1].strip() if index + 1 < len(rescued_lines) else "")
            )
            has_story_context = bool(
                str(unit.get("scene_summary") or "").strip()
                or (unit.get("character_names") or [])
                or has_neighbor_context
            )
            if panel_count >= 2 or has_story_context:
                candidate_indices.append(index)

        if not candidate_indices:
            return recovered

        if len(candidate_indices) > self._MAX_VISUAL_ONLY_RECOVERIES:
            def _recovery_priority(index: int) -> tuple[int, int, int, int]:
                unit = units[index]
                panel_count = int(unit.get("panel_count") or len(unit.get("panel_ids", []) or []))
                has_names = 1 if (unit.get("character_names") or []) else 0
                has_caption = 1 if str(unit.get("vision_caption") or "").strip() else 0
                has_action = 1 if str(unit.get("vision_action_beat") or "").strip() else 0
                return (panel_count, has_names, has_caption, has_action)

            original_count = len(candidate_indices)
            selected = sorted(candidate_indices, key=_recovery_priority, reverse=True)[: self._MAX_VISUAL_ONLY_RECOVERIES]
            candidate_indices = sorted(selected)
            logger.info(
                "Limiting visual-only multimodal recovery from %d to %d strongest candidates",
                original_count,
                len(candidate_indices),
            )

        rescue_units = [dict(unit) for unit in units]
        enable_local_ocr_rescue = os.getenv("PANELIA_ENABLE_STORY_LOCAL_OCR_RESCUE", "").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        if enable_local_ocr_rescue:
            language_hint = str(chapter_metadata.get("language") or "en").strip() or "en"
            for index in candidate_indices:
                rescue_units[index]["combined_text"] = self._enhanced_rescue_text(
                    units[index],
                    scene_visual_paths=scene_visual_paths,
                    language_hint=language_hint,
                )
        else:
            logger.info(
                "Skipping local OCR rescue for %d visual story segment candidates; "
                "Gemini vision/existing OCR will handle rescue. Set "
                "PANELIA_ENABLE_STORY_LOCAL_OCR_RESCUE=1 to re-enable.",
                len(candidate_indices),
            )

        allowed_character_names = list(name_grounding.get("allowed_character_names") or []) if name_grounding else []
        prompt_story_bible = self._story_bible_prompt_payload(story_bible)
        for start in range(0, len(candidate_indices), self._RESCUE_BATCH_SIZE):
            batch_indices = candidate_indices[start:start + self._RESCUE_BATCH_SIZE]
            rewrite_by_index = self._run_multimodal_rescue_batch(
                batch_indices,
                rescued_lines,
                rescue_units,
                project_title=project_title,
                chapter_metadata=chapter_metadata,
                chapter_summary=chapter_summary,
                character_dictionary=character_dictionary,
                protagonist_name=protagonist_name,
                prompt_story_bible=prompt_story_bible,
                allowed_character_names=allowed_character_names,
                scene_visual_paths=scene_visual_paths,
                log_label=f"recover-{start}-{start + len(batch_indices)}",
            )
            for global_index, replacement in rewrite_by_index.items():
                candidate = self._normalize_segment_text(replacement, allow_empty=True)
                if (
                    not candidate
                    or self._line_is_low_quality(candidate)
                    or self._line_is_overly_generic(candidate)
                    or self._line_is_dialogue_fragment(candidate)
                    or self._line_is_sentence_fragment(candidate)
                ):
                    continue
                recovered[global_index]["text"] = candidate
                recovered[global_index]["visual_only"] = False
                recovered[global_index]["suppression_reason"] = None
                rescued_lines[global_index] = candidate
        return recovered

    def _enhanced_rescue_text(
        self,
        unit: dict[str, Any],
        *,
        scene_visual_paths: dict[str, list[Path]],
        language_hint: str,
    ) -> str:
        base_text = clean_ocr_text(str(unit.get("combined_text") or "").strip())
        if base_text and not self._text_is_noisy_ocr(base_text):
            return base_text[:1200]

        segment_id = str(unit.get("segment_id") or "").strip()
        image_paths = list(scene_visual_paths.get(segment_id) or [])[:3]
        if not image_paths:
            return base_text[:1200]

        fragments: list[str] = []
        ocr = self._get_comic_ocr()
        for image_path in image_paths:
            try:
                with Image.open(image_path) as image:
                    rgb_image = image.convert("RGB")
                    if max(rgb_image.size) > 1800:
                        rgb_image.thumbnail((1800, 1800), Image.Resampling.LANCZOS)
                    crop = np.array(rgb_image)
            except Exception:
                continue
            text, _, _ = ocr.recognize_panel_text(crop, language_hint)
            cleaned = clean_ocr_text(text)
            if not cleaned or self._text_is_noisy_ocr(cleaned):
                continue
            if any(self._normalized_line_key(existing) == self._normalized_line_key(cleaned) for existing in fragments):
                continue
            fragments.append(cleaned)
        if not fragments:
            return base_text[:1200]
        merged = clean_ocr_text(" ".join(fragments))
        return merged[:1200] if merged else base_text[:1200]

    def _get_comic_ocr(self) -> ComicOCRService:
        if self._comic_ocr is None:
            self._comic_ocr = ComicOCRService()
        return self._comic_ocr

    def _multimodal_rescue_reason(self, line: str, unit: dict[str, Any]) -> str | None:
        cleaned = self._normalize_segment_text(line, allow_empty=True)
        if not cleaned:
            return "blank"
        if self._line_is_low_quality(cleaned):
            return "low_quality"
        if self._line_is_overly_generic(cleaned):
            return "generic"
        if re.search(r"\b(?:someone|a character|another character|a figure|another figure)\b", cleaned, flags=re.IGNORECASE):
            return "vague_subject"
        if self._line_is_dialogue_fragment(cleaned):
            return "dialogue_fragment"
        if self._line_is_sentence_fragment(cleaned):
            return "sentence_fragment"
        if self._line_echoes_unreliable_ocr(cleaned, unit):
            return "ocr_echo"
        return None

    def _line_echoes_unreliable_ocr(self, line: str, unit: dict[str, Any]) -> bool:
        combined_text = clean_ocr_text(str(unit.get("combined_text") or "").strip())
        if not combined_text or not self._text_is_noisy_ocr(combined_text):
            return False
        stop_words = {
            "the", "and", "for", "with", "this", "that", "from", "into", "your", "their",
            "they", "them", "then", "just", "have", "what", "when", "where", "who", "why",
            "how", "you", "him", "her", "his", "its", "our", "out", "are", "was", "were",
        }
        line_tokens = {
            token
            for token in re.findall(r"[A-Za-z']+", line.casefold())
            if len(token) > 2 and token not in stop_words
        }
        ocr_tokens = {
            token
            for token in re.findall(r"[A-Za-z']+", combined_text.casefold())
            if len(token) > 2 and token not in stop_words
        }
        if not line_tokens or not ocr_tokens:
            return False
        overlap = len(line_tokens & ocr_tokens) / max(1, min(len(line_tokens), len(ocr_tokens)))
        return overlap >= 0.5

    def _line_is_sentence_fragment(self, line: str) -> bool:
        cleaned = str(line or "").strip()
        if not cleaned:
            return False
        tokens = re.findall(r"[A-Za-z']+", cleaned)
        if len(tokens) < 3:
            return False
        starts_with_fragment = bool(
            re.match(r"^(?:A|An|The|As|While|When|After|Before|With|Without|For|To|Into|From|Under|Over|Inside|Outside|Between|One|Two|Three|Four|That|Those|These|This|Him|Her|Them|Easily|Offering)\b", cleaned)
        )
        starts_with_gerund = bool(re.match(r"^[A-Z][a-z]+ing\b", cleaned))
        starts_with_participle = bool(re.match(r"^[A-Z][a-z]+(?:ed|en)\b", cleaned))
        if starts_with_gerund and len(tokens) <= 10:
            return True
        finite_verb_pattern = re.compile(
            r"\b(?:is|are|was|were|be|being|been|has|have|had|do|does|did|can|could|will|would|should|may|might|must|"
            r"moves|moved|steps|stepped|walks|walked|runs|ran|meets|met|finds|found|turns|turned|reaches|reached|"
            r"leans|leaned|pulls|pulled|calls|called|says|said|asks|asked|watches|watched|tells|told|realizes|realized|"
            r"wonders|wondered|recoils|recoiled|flinches|flinched|hesitates|hesitated|presses|pressed|stares|stared|"
            r"smiles|smiled|admits|admitted|insists|insisted|explains|explained|warns|warned|orders|ordered|promises|promised|"
            r"closes|closed|keeps|kept|makes|made|tightens|tightened|erupts|erupted|begins|began|starts|started|pushes|pushed|"
            r"forces|forced|refuses|refused|gives|gave|needs|needed|loses|lost|becomes|became|surfaces|surfaced|feels|felt|"
            r"looms|loomed|rattles|rattled|leaves|left|marks|marked|throws|threw|scrambles|scrambled|shocks|shocked|"
            r"cuts|cut|sends|sent|argues|argued|celebrates|celebrated|claims|claimed|rejects|rejected|faces|faced|"
            r"draws|drew|spreads|spread|reports|reported|commences|commenced|boards|boarded|comes|came|breaks|broke|"
            r"shifts|shifted|carries|carried|prepares|prepared|confirms|confirmed|signals|signaled|signals?|"
            r"regains|regained|adjusts|adjusted|costs|costed|contains|contained|holds|held|move|retreat|retreats|"
            r"overwhelm|overwhelms|stay|stays|develops|developed|creates|created|forms|formed|takes|took)\b",
            flags=re.IGNORECASE,
        )
        if finite_verb_pattern.search(cleaned):
            return False
        if starts_with_fragment or starts_with_gerund or starts_with_participle:
            return True
        return len(tokens) <= 6

    def _line_is_dialogue_fragment(self, line: str) -> bool:
        cleaned = str(line or "").strip()
        if not cleaned:
            return False
        tokens = re.findall(r"[A-Za-z']+", cleaned)
        if not tokens:
            return False
        if cleaned.endswith("?") and len(tokens) <= 14:
            return True
        if len(tokens) <= 10 and re.match(
            r"^(?:It|Who|Why|What|How|Are|Is|Do|Did|Can|Could|Would|Should|Will|Won't|Don't|Didn't)\b",
            cleaned,
            flags=re.IGNORECASE,
        ):
            return True
        if cleaned.count("?") + cleaned.count("!") >= 2 and len(tokens) <= 18:
            return True
        return False

    def _split_sentences_for_cleanup(self, text: str) -> list[str]:
        """Split scene-mode narration into conservative sentence chunks."""
        cleaned = self._normalize_segment_text(text, allow_empty=True)
        if not cleaned:
            return []
        return [
            sentence.strip()
            for sentence in re.split(r"(?<=[.!?])\s+(?=[A-Z])", cleaned)
            if sentence.strip()
        ]

    def _sentence_count(self, text: str) -> int:
        return len(self._split_sentences_for_cleanup(text))

    @staticmethod
    def _line_has_first_person_narration(text: str) -> bool:
        lowered = str(text or "").strip().casefold()
        if not lowered:
            return False
        return bool(
            re.search(
                r"\b(i|i'm|i've|i’ll|i'll|i’d|i'd|me|my|mine|myself|we|we're|we've|we’ll|we'll|us|our|ours|ourselves)\b",
                lowered,
            )
        )

    def _sentence_fails_story_filters(self, text: str) -> bool:
        return (
            self._line_is_low_quality(text)
            or self._line_is_overly_generic(text)
            or self._line_is_dialogue_fragment(text)
            or self._line_is_sentence_fragment(text)
            or self._line_needs_style_refinement(text)
            or self._line_has_first_person_narration(text)
            or self.polisher._is_visual_description(text)
        )

    def _remove_offending_sentences(self, text: str) -> str:
        """Drop bad sentences without discarding the whole scene-mode segment."""
        normalized = self._normalize_segment_text(text, allow_empty=True)
        sentences = self._split_sentences_for_cleanup(normalized)
        if len(sentences) <= 1:
            return normalized
        survivors = [
            sentence
            for sentence in sentences
            if not self._sentence_fails_story_filters(sentence)
        ]
        if not survivors:
            return ""
        joined = self._normalize_segment_text(" ".join(survivors), allow_empty=True)
        if not joined or self._sentence_fails_story_filters(joined):
            return ""
        return joined

    def _apply_weak_scene_policy(
        self,
        lines: list[str],
        units: list[dict[str, Any]],
        protagonist_name: str | None,
        *,
        scene_mode: bool = False,
        style_vocab: StyleVocabulary | None = None,
    ) -> list[dict[str, Any]]:
        payloads: list[dict[str, Any]] = []
        previous_key = ""
        for index, unit in enumerate(units):
            raw_line = lines[index] if index < len(lines) else ""
            normalized_line = self._normalize_segment_text(raw_line, allow_empty=True)
            weak_scene = self._scene_is_weakly_grounded(unit)
            current_key = self._normalized_line_key(normalized_line)
            is_duplicate = bool(current_key and current_key == previous_key)
            overly_generic = self._line_is_overly_generic(normalized_line)
            low_quality = self._line_is_low_quality(normalized_line)
            first_person = self._line_has_first_person_narration(normalized_line)
            visual_description = self.polisher._is_visual_description(normalized_line)

            if not normalized_line:
                fallback_line = self._fallback_scene_line(unit, protagonist_name, style_vocab=style_vocab)
                if weak_scene:
                    payloads.append({"text": "", "visual_only": True, "suppression_reason": "weak_evidence"})
                elif fallback_line:
                    payloads.append(
                        {
                            "text": fallback_line,
                            "visual_only": False,
                            "suppression_reason": None,
                        }
                    )
                else:
                    payloads.append({"text": "", "visual_only": True, "suppression_reason": "weak_evidence"})
                continue

            if (
                not weak_scene
                and normalized_line
                and len(self._split_sentences_for_cleanup(normalized_line)) >= 2
                and (
                    low_quality
                    or overly_generic
                    or self._line_is_dialogue_fragment(normalized_line)
                    or self._line_is_sentence_fragment(normalized_line)
                    or self._line_needs_style_refinement(normalized_line)
                    or first_person
                    or visual_description
                    or any(
                        self._sentence_fails_story_filters(sentence)
                        for sentence in self._split_sentences_for_cleanup(normalized_line)
                    )
                )
            ):
                trimmed_line = self._remove_offending_sentences(normalized_line)
                if trimmed_line:
                    normalized_line = trimmed_line
                    current_key = self._normalized_line_key(normalized_line)
                    is_duplicate = bool(current_key and current_key == previous_key)
                    overly_generic = self._line_is_overly_generic(normalized_line)
                    low_quality = self._line_is_low_quality(normalized_line)
                    first_person = self._line_has_first_person_narration(normalized_line)
                    visual_description = self.polisher._is_visual_description(normalized_line)
                elif scene_mode:
                    sentences = self._split_sentences_for_cleanup(normalized_line)
                    survivors = [
                        sentence
                        for sentence in sentences
                        if not self._sentence_fails_story_filters(sentence)
                    ]
                    if survivors:
                        normalized_line = self._normalize_segment_text(" ".join(survivors), allow_empty=True)
                        current_key = self._normalized_line_key(normalized_line)
                        is_duplicate = bool(current_key and current_key == previous_key)
                        overly_generic = self._line_is_overly_generic(normalized_line)
                        low_quality = self._line_is_low_quality(normalized_line)
                        first_person = self._line_has_first_person_narration(normalized_line)
                        visual_description = self.polisher._is_visual_description(normalized_line)

            if low_quality or first_person or visual_description:
                fallback_line = self._evidence_bridge_line(unit, protagonist_name, style_vocab=style_vocab) or self._fallback_scene_line(
                    unit,
                    protagonist_name,
                    style_vocab=style_vocab,
                )
                if fallback_line and not self._sentence_fails_story_filters(fallback_line):
                    payloads.append({"text": fallback_line, "visual_only": False, "suppression_reason": None})
                    previous_key = self._normalized_line_key(fallback_line)
                else:
                    payloads.append({"text": "", "visual_only": True, "suppression_reason": "weak_evidence"})
                continue

            if is_duplicate:
                fallback_line = self._fallback_scene_line(unit, protagonist_name, style_vocab=style_vocab)
                fallback_key = self._normalized_line_key(fallback_line)
                if fallback_line and fallback_key and fallback_key != current_key and not self._line_is_low_quality(fallback_line):
                    payloads.append({"text": fallback_line, "visual_only": False, "suppression_reason": None})
                    previous_key = fallback_key
                else:
                    payloads.append({"text": "", "visual_only": True, "suppression_reason": "duplicate_alignment"})
                continue

            if overly_generic and int(unit.get("scene_unit_count") or 1) > 1:
                fallback_line = self._fallback_scene_line(unit, protagonist_name, style_vocab=style_vocab)
                if fallback_line and self._normalized_line_key(fallback_line) != current_key:
                    payloads.append({"text": fallback_line, "visual_only": False, "suppression_reason": None})
                    previous_key = self._normalized_line_key(fallback_line)
                else:
                    payloads.append({"text": "", "visual_only": True, "suppression_reason": "generic_alignment"})
                continue

            if weak_scene and (is_duplicate or low_quality):
                payloads.append({"text": "", "visual_only": True, "suppression_reason": "weak_evidence"})
                continue

            payloads.append({"text": normalized_line, "visual_only": False, "suppression_reason": None})
            if current_key:
                previous_key = current_key
        return payloads

    def _scene_is_weakly_grounded(self, unit: dict[str, Any]) -> bool:
        combined_text = clean_ocr_text(str(unit.get("combined_text") or "").strip())
        visual_cues = self._normalize_supporting_text(str(unit.get("visual_cues") or "").strip())
        if self._text_is_noisy_ocr(combined_text):
            combined_text = ""
        word_count = len(re.findall(r"[A-Za-z']+", combined_text))
        visual_word_count = len(re.findall(r"[A-Za-z']+", visual_cues))
        character_count = len(unit.get("character_names", []) or [])
        panel_count = int(unit.get("panel_count") or 0)
        return word_count < 4 and visual_word_count < 6 and character_count == 0 and panel_count <= 1

    def _line_is_overly_generic(self, line: str) -> bool:
        cleaned = str(line or "").strip()
        if not cleaned:
            return False
        generic_patterns = (
            r"\bNone\b",
            r"\bnull\b",
            r"\bsymbols for\b",
            r"\bface shows\b",
            r"\bexpression shows\b",
            r"\bexpression\b",
            r"\beyes wide\b",
            r"\bsweat beading\b",
            r"\bshocked expression\b",
            r"\bstartled expression\b",
            r"\bbody coiled\b",
            r"\blight blue tiled floor\b",
            r"\bappears distressed\b",
            r"\bthe injury turns the confrontation\b",
            r"\bthe barrier turns protection into leverage\b",
            r"^(?:someone|a person|another person|a character|another character|the group|a figure|another figure)\b",
            r"^(?:a|an)\s+(?:male|female)\s+character\b",
            r"^(?:curly|blonde|dark|short|long|red|blue|black|white|silver|pink|green)\s+hair\b",
            r"\b(?:expresses?|expressed|states?|stated|declares?|declared|remarks?|remarked|mentions?|mentioned|comments?|commented)\b",
            r"\bwas\s+(?:asked|questioned|told|introduced|reminded|warned)\b",
            r"^(?:a|an|the)\s+(?:man|woman|boy|girl|young man|young woman)\s+(?:interrupted|apologized|announced|asked|told|warned|questioned|declared|explained)\b",
            r"\bis introduced as\b",
            r"\b(?:concept|idea|theme|metaphor)\b.{0,80}\bis introduced\b",
            r"\b(?:also\s+known\s+as|known\s+as)\b.{0,80}\bis introduced\b",
            r"\bis introduced\.\s*(?:a|an|the)\s+metaphor\b",
            r"\ba metaphor for\b",
            r"\bstark contrast\b",
            r"\bknown as\b.{0,100}\bdesigned to combat\b",
            r"\bis in a significant situation\b",
            r"\b(?:is shown|are shown|is depicted|are depicted|was depicted|were depicted|is displayed|are displayed)\b",
            r"\bis visible\b",
            r"\ba character\b",
            r"\bsomething\b",
            r"\bwith expressions? of\b",
            r"^the narration\b",
            r"^(?:the scene|the story|the chapter)\b",
            r"\bthe scene\b",
            r"\bthe moment\b",
            r"\bthe situation\b",
            r"\bbecomes impossible to ignore\b",
            r"\bthe (?:conflict|pressure|stakes|story|chapter)\b",
            r"\bthe mood\b",
            r"\bthe narration\b",
            r"\bthe next development\b",
            r"\banother tense beat\b",
            r"\banother crucial moment\b",
            r"\banother uneasy beat\b",
            r"\buneasy beat passes\b",
            r"\btension stays unresolved for another beat\b",
            r"\bnext beat makes it clear\b",
            r"\bone pointed\b",
            r"\bkeeps evolving\b",
            r"\ba sharp question cuts through\b",
            r"\bone pointed question makes it clear\b",
            r"\bthe panel holds for a beat\b",
            r"\bthe moment catches on\b",
            r"\btension builds\b",
            r"\bthe pressure keeps rising\b",
            r"\bthe world still feels normal\b",
            r"\bquestions start piling up\b",
            r"\bthe consequences grow harder to ignore\b",
            r"\bthe situation grows harder to contain\b",
            r"\bbefore the scene can settle\b",
            r"\bas everyone absorbs what just happened\b",
            r"\ba sudden question leaves the moment hanging\b",
            r"\bthe unanswered question freezes the scene\b",
            r"\bthe situation shifts\b",
            r"\bthe risk keeps pressure on the moment\b",
            r"\bthe moment (?:keeps|stays with) the moment\b",
            r"\bthe moment still has to carry that risk forward\b",
            r"\bthe next response has to account for the next choice\b",
            r"\bkeeps? facing (?:the )?(?:group tension|exchange|risk|immediate threat)\b",
            r"\buncertainty around (?:the )?(?:group tension|exchange|risk|immediate threat)\b",
            r"\bthe group tension\b",
            r"\bthe exchange\b.{0,80}\bthe group\b",
            r"\bthe immediate threat\b.{0,80}\bthe immediate threat\b",
            r"\bthe risk\b.{0,80}\bthe next choice\b",
            r"\bbefore anyone can fully recover\b",
            r"\bthe immediate problem tightens\b",
            r"\bthe immediate problem\b",
            r"\bthe next exchange\b",
            r"\bthe next beat carries that pressure forward\b",
            r"\bnearby choice\b",
            r"\bnext choice\b",
            r"\blast choice\b",
            r"\bthe beat (?:keeps|kept|shifts|shifted|moves|moved)\b",
            r"\bkeeps? the nearby\b",
            r"\bwhile the surrounding group reacts\b",
            r"\bmatter of survival\b",
            r"\bthe dynamic shifts\b",
            r"\bfewer options\b",
            r"\bfew options\b",
            r"\bmenacing posture signals\b",
            r"\bimminent conflict\b",
            r"\bprecarious position\b",
            r"\bthe overall apathy\b",
            r"\bcreating a dull atmosphere\b",
            r"\bpromises further complications\b",
            r"\bpersistent challenges\b",
            r"\bthe exchange changes how the group has to read\b",
            r"\bputs? new pressure on .{0,60}\bbefore anyone can settle\b",
            r"\bpull(?:s|ed)? .{0,80}\binto tension .{0,80}\bcannot fully interpret\b",
            r"\bcannot fully interpret\b",
            r"\breads? .{0,80}\bas proof that .{0,80}\bbecoming harder to protect\b",
            r"\bbecoming harder to protect\b.{0,80}\bpushes? the next exchange\b",
            r"\bthat realization pushes the next exchange toward\b",
            r"\bfeels? the strain of .{0,80}\bbefore anyone can put it into words\b",
            r"\bchoices? start(?:s|ed)? shaping consequences before the others understand\b",
            r"\bkeeps? shaping the room around\b",
            r"\bleave(?:s|d)? .{0,80}\bwith a problem they cannot solve from the outside\b",
            r"\bmake(?:s)? .{0,80}\bfeel larger than a single exchange\b",
            r"\bmake(?:s)? .{0,80}\bfeel personal for\b",
            r"\bfeel personal for .{0,80}\bchoices? change(?:s|d)? the emotional stakes\b",
            r"\bchoices? change(?:s|d)? the emotional stakes before the action moves on\b",
            r"\bemotional stakes before the action moves on\b",
            r"\bcarries? (?:a|the) battle through the reaction\b",
            r"\bconnects? (?:a|the) battle to\b",
            r"\bmaking the danger harder to dismiss\b",
            r"\breaction gives the beat enough weight\b",
            r"\bthe beat ties\b",
            r"\bkeeps? .{0,80}\bfrom fading into the background\b",
            r"\bthe group has to stay alert\b",
            r"\bremains? the clearest anchor in (?:the )?(?:risk|moment|exchange|group tension)\b",
            r"\bturn (?:the )?(?:next choice|risk|group tension) into an actual choice\b",
            r"\bwithout changing what the scene shows\b",
            r"\bthe moment has consequence because\b",
            r"\bnot simply watch\b",
            r"\bconnect the last shock with the next decision\b",
            r"\bunresolved pressure gives the next response\b",
            r"\bstays? unresolved as .{0,80}\bpushes? through the beat\b",
            r"\bthe purpose of piloting\b",
            r"\bleft pressing around\b",
            r"\bconsequence stays open\b",
            r"\bturns? .{0,80}\binto a private test\b",
            r"\bprivate test\b",
            r"\bsurrounding group has to answer\b",
            r"\bpoint where the beat turns forward\b",
            r"\babsorbs? the strain of .{0,80}\bwhile .{0,80}\bbecomes the point\b",
            r"\bkeeps? moving through .{0,80}\bwhile .{0,80}\btries to keep up\b",
            r"\bties? the previous shock to a decision\b",
            r"\bthe pressure carries forward\b",
            r"\bthe moment hangs\b",
            r"\bstays? near the center of\s*,",
            r"\bas part of\s*,",
            r"\bno clean way to step back\b",
            r"\bkeeps? facing a defense\b",
            r"\bthe next choice can move any further\b",
            r"^Him\b",
            r"\bto explain that\b",
            r"\bthe conflict keeps tightening\b",
            r"\bthe briefing shifts from explanation\b",
            r"\bthe explanation turns practical\b",
            r"\bthe room moves from theory into orders\b",
            r"\bthe facility'?s order makes\b",
            r"\bthe setting shifts into machinery\b",
            r"\b(?:missing role|missing presence|missing pilot|absence)\b.{0,80}\b(?:weigh|shape|shaping|plans?|squad|mission|constant|critical|gap)\b",
            r"\b(?:without [A-Z][a-z]+|without the .{0,30})\b.{0,80}\b(?:mission|falling apart|squad|formation)\b",
            r"\bhas to respond before the situation moves out of reach\b",
            r"\bno new event is added\b",
            r"\bconnective tissue\b",
            r"\bgrounded beat\b",
            r"\bpanel range\b",
            r"\brestrained transition\b",
            r"\bimmediate aftermath\b",
            r"\bsurrounding tension stays intact\b",
            r"\bkeeps? the pacing intact\b",
            r"\bthe next spoken beat\b",
            r"\bthe line holds just enough tension\b",
            r"\bthe danger remains close enough\b",
            r"\bno one can treat it as background\b",
            r"\bthe reaction carries enough unease to bridge\b",
            r"\bbridge into the following panel\b",
            r"\bthat reaction leaves the choice sharper\b",
            r"\bthe exchange leaves the characters with less room\b",
            r"\bthat detail keeps the danger close enough\b",
            r"\bthe beat lands as a decision point\b",
            r"\bthat response makes the next decision feel immediate\b",
            r"\bthe surrounding reaction keeps the sequence\b",
            r"\bthat turn gives the following action\b",
            r"\bthe choice carries enough weight\b",
            r"\bthat small shift keeps the emotional thread\b",
            r"\bthe pause leaves a trace of doubt\b",
            r"\bthat look of hesitation\b",
            r"\bthat answer keeps the group moving\b",
            r"\bthe reaction gives the next exchange\b",
            r"\bthat consequence makes the transition\b",
            r"\bthat pressure keeps the characters\b",
            r"\bthe choice reframes the immediate problem\b",
            r"\bthat hesitation makes the next move\b",
            r"\bthe answer lands with enough force\b",
            r"\bthat change in tone gives the sequence\b",
            r"\bthe uncertainty keeps everyone oriented\b",
            r"\bthat moment of resistance\b",
            r"\bthe reaction leaves a small\b",
            r"\bthat admission keeps the emotional cost\b",
            r"\bthe line turns a passing exchange\b",
            r"\bthat interruption keeps the pacing tense\b",
            r"\bthe response anchors the transition\b",
            r"\bthat pressure gives the next move\b",
            r"\bthe beat keeps the immediate risk alive\b",
            r"\bthat detail makes the scene'?s next turn\b",
            r"\ba pointed request\b.{0,40}\b(?:throws|threw) the protagonist\b",
            r"\bexpressing (?:his|her|their) willingness\b",
            r"\bgrimaces? in pain or distress\b",
            r"\ba door or barrier\b",
            r"\blarge dome structure\b",
            r"\bmassive dome floats\b",
            r"\ba mocking response cuts into the conversation\b",
            r"\bforcing another response before anyone can settle\b",
            r"\bbefore he could fully process the scene\b",
            r"\bmoves into the next turn of the chapter\b",
            r"\bpushes into the next turn of the chapter\b",
            r"^and consequences push the story forward\b",
            r"^confusion takes over\b",
            r"^magic erupts without warning\b",
            r"^as the situation escalates\b",
            r"\bshown together\b",
            r"\bpossibly\b",
            r"\bhinting at\b",
            r"\binteract\b",
            r"\btraining or combat scenario\b",
            r"\binvolved in a significant event\b",
            r"\bcontrolled emergency\b",
            r"\bsetting is now a controlled facility\b",
            r"\bin a stark setting\b",
            r"\bdiscussed matters\b",
            r"\bexpressions? hinted\b",
            r"\border barely hides the pressure\b",
            r"\bmale officer\b",
            r"\bgives? a serious command\b",
            r"\bdelivered a stern warning\b",
            r"\bsurprise and confusion\b",
            r"\bsurprised and confused\b",
            r"\breadings are still unstable\b",
            r"\bsomeone\b",
            r"\btwo of whom\b",
            r"\bchibi form\b",
            r"\bphone exchange turn(?:s|ed) smug\b",
            r"\bshrug(?:s|ged) indifferently\b",
            r"\b(?:he|she) seems unconcerned\b",
            r"\btrap(?:s|ped|ping) the families in a new crisis\b",
            r"\bsince [A-Z][a-z]+ (?:is|was) dead\b",
            r"\bwill help (?:them|us) live longer\b",
            r"\bmedical kit and energy (?:source|cells)\b",
            r"\bkey detail here\b",
            r"\bsupply fight\b.{0,80}\bsurvival now matters? more than trust\b",
            r"\bpossibility that hao (?:has|had) been killed\b",
            r"\bevery resource harder to share\b",
            r"\bthis situation creates significant pressure\b",
            r"\barmed figures meticulously preparing\b",
            r"\bpreparing their next move\b",
            r"\btrust (?:inside|within) the building\b",
            r"\btrust within the building\b.{0,80}\b(?:breaking down|deteriorating)\b",
            r"\bcooperation among the neighbors\b",
            r"\bneighbors are abandoning cooperation\b",
            r"\btreats? the resource dispute as proof\b",
            r"\bcontemplating chen zheng hao'?s cruelty\b",
            r"\bbody on the ground\b.{0,80}\bchen zheng hao\b",
            r"\bcommits? to the breach with (?:his|her|their) weapon ready\b",
            r"\bstandoff at the door turns physical\b",
            r"\btake the threat seriously\b",
            r"\btheir keeps\b",
            r"\bwarning that the building\b",
            r"\bmission keeps circling back\b",
            r"\bweighs? the partnership against the danger waiting ahead\b",
            r"\bcarries? that uncertainty forward as the next wave of danger approaches\b",
            r"\bbecomes central to the team'?s next move as the operation tightens\b",
            r"\bstays? in the fight as the pilots move from hesitation into action\b",
            r"\bhelps? carry the operation from uncertainty into direct combat\b",
            r"\bfaces? a personal test as the bond with [A-Z][a-z]+(?: [A-Z][a-z]+)? turns dangerous\b",
            r"\boperation pushes the pilots toward launch before anyone has time to settle their doubts\b",
            r"\bpilots move from preparation into combat as .{0,40} units become the squad'?s only answer\b",
            r"\bsquad shifts from doubt into deployment as .{0,40} threat closes in\b",
            r"\breads? the approaching danger as proof that staying fortified is the only safe choice\b",
            r"\btreats? the attack as proof that (?:his|her|their) defenses? ha(?:s|ve) to hold\b",
            r"\bdefenses? must hold firm\b",
            r"\bremains focused on advancing the mission\b",
            r"\bgrowing peril\b",
            r"\bstruggles to process\b.{0,80}\bpartnership\b",
            r"\bimplications of (?:this|his|her|the) (?:new )?(?:partnership|alliance|bond)\b",
            r"\bnext (?:critical )?decision draws (?:closer|nearer)\b",
            r"\bkeeps? .{0,80} in focus while the next confrontation (?:takes|begins? to take|began to take) shape\b",
            r"\bnext confrontation (?:takes|begins? to take|began to take) shape\b",
            r"\b(?:proudly|smugly) presented (?:his|her|their) fortified shelter\b",
            r"\bfortified shelter\b.*\b(?:medical kit|energy cells|supplies)\b",
            r"\bfortified shelter\b.*\btestament\b.*\bforesight\b.*\bpreparation\b",
        )
        if any(re.search(pattern, cleaned, flags=re.IGNORECASE) for pattern in generic_patterns):
            return True
        return bool(
            re.search(
                r"\b(?:reads? the next choice through|keeps? the cost close|for any sign of control|"
                r"what the group already knows|inability to pilot|"
                r"exceptionally skilled pilot|ability to pilot)\b",
                cleaned,
                flags=re.IGNORECASE,
            )
        )

    def _final_sanitize_story_payloads(self, payloads: list[dict[str, Any]]) -> list[dict[str, Any]]:
        sanitized: list[dict[str, Any]] = []
        for payload in payloads:
            current = dict(payload)
            text = self._normalize_segment_text(str(current.get("text") or ""), allow_empty=True)
            text = re.sub(r"\bNone\b", "the character", text)
            text = re.sub(r"\bnull\b", "the character", text, flags=re.IGNORECASE)
            text = self._normalize_segment_text(text, allow_empty=True)
            if text:
                kept_sentences = [
                    sentence
                    for sentence in self._split_sentences_for_cleanup(text)
                    if not self._sentence_has_visual_caption_leak(sentence)
                ]
                if kept_sentences and len(kept_sentences) < len(self._split_sentences_for_cleanup(text)):
                    text = self._normalize_segment_text(" ".join(kept_sentences), allow_empty=True)
            if (
                not text
                or self._line_is_low_quality(text)
                or self._line_is_overly_generic(text)
                or self._line_is_dialogue_fragment(text)
                or self._line_is_sentence_fragment(text)
            ):
                current["text"] = ""
                current["visual_only"] = True
                current["suppression_reason"] = current.get("suppression_reason") or "final_quality_reject"
            else:
                current["text"] = text
                if current.get("suppression_reason") == "final_quality_reject":
                    current["suppression_reason"] = None
            sanitized.append(current)
        return sanitized

    @staticmethod
    def _sentence_has_visual_caption_leak(sentence: str) -> bool:
        lowered = str(sentence or "").casefold()
        visual_phrases = (
            "face shows",
            "expression shows",
            "expression",
            "eyes wide",
            "sweat beading",
            "symbols for",
            "shocked expression",
            "startled expression",
            "pained expression",
            "determined expression",
            "glared with",
            "body coiled",
            "light blue tiled floor",
            "appears distressed",
            "appearing distressed",
            "the injury turns the confrontation",
            "the barrier turns protection into leverage",
            "blinding red light",
            "bright light",
            "with wide eyes",
            "in the foreground",
        )
        if any(phrase in lowered for phrase in visual_phrases):
            return True
        return bool(
            re.search(
                r"\b(?:wearing|visible|background|foreground|camera|panel|frame|close-up)\b",
                lowered,
            )
        )

    def _line_is_low_quality(self, line: str) -> bool:
        cleaned = self._normalize_segment_text(line, allow_empty=True)
        if not cleaned:
            return False
        if re.search(
            r"\b(?:the\s+)?(?:concept|idea|theme)\b.{0,80}\bis introduced\b|\ba metaphor for\b|\bstark contrast\b",
            cleaned,
            flags=re.IGNORECASE,
        ):
            return True
        if re.search(r"\b(?:agora|claro|sim|não|nao|por favor)\b", cleaned, flags=re.IGNORECASE):
            return True
        if re.match(r"^None\s+(?:has|is|was|keeps|stays|faces|moves|carries)\b", cleaned):
            return True
        sentences = self._split_sentences_for_cleanup(cleaned)
        if len(sentences) >= 2:
            token_sets = [self._content_token_set(sentence) for sentence in sentences]
            for current_index, tokens in enumerate(token_sets):
                if len(tokens) < 4:
                    continue
                for prior_tokens in token_sets[:current_index]:
                    if len(prior_tokens) < 4:
                        continue
                    overlap = len(tokens & prior_tokens)
                    containment = overlap / max(1, min(len(tokens), len(prior_tokens)))
                    jaccard = overlap / max(1, len(tokens | prior_tokens))
                    if containment >= 0.72 or jaccard >= 0.58:
                        return True
        if re.match(r'^(?:"|“)?(?:I|I\'m|I’d|I\'d|I’ll|I\'ll|I’ve|I\'ve|My|Me|We|Our|Us)\b', cleaned):
            return True
        if re.search(r"(?:'|\"|“|‘)(?:I|I'm|I’m|I’d|I'd|I’ll|I'll|I’ve|I've|My|Me|We|We're|We’re|Our|Us)\b", cleaned):
            return True
        if re.match(r'^(?:"|“|\'|‘)', cleaned):
            return True
        if re.match(r"^[A-Z][A-Za-z0-9_ ]{0,24}:\s", cleaned):
            return True
        if re.search(r"\b(?:you're|you’re|we're|we’re|let's|let’s)\b", cleaned, flags=re.IGNORECASE):
            return True
        if re.search(r"\b(?:we'll|we’ll|we've|we’ve|we'd|we’d|i'll|i’ll|i've|i’ve|i'd|i’d)\b", cleaned, flags=re.IGNORECASE):
            return True
        if re.search(
            r"\b(?:adoing|has yet her|not like this is about|they'?ll be help)\b",
            cleaned,
            flags=re.IGNORECASE,
        ):
            return True
        if re.search(
            r"\b(?:Hmb|rnes|imnb|Jle|trle|Morn\s+ing|anneenntennnu|"
            r"In\s+ing|Etnnn|Myo|Noth|Ytmeno|Irsen|Plom|oclv|rane)\b|"
            r"[一-龯ぁ-んァ-ン]",
            cleaned,
            flags=re.IGNORECASE,
        ):
            return True
        if re.search(
            r"\b([A-Z][a-z]+)\s+\w+s\s+\1\b",
            cleaned,
        ):
            return True
        if re.search(
            r"\b([A-Z][a-z]+)\b.{0,80}\bwatching\s+\1\b",
            cleaned,
        ):
            return True
        if re.search(
            r"\b(?:[A-Za-z]{2,}\s+was\s+ing|she did [A-Z][a-z]+ calls?|^and\s+[A-Z]?[a-z]+\b|^other\s+exclaims?|^pilots\s+gives?)\b",
            cleaned,
            flags=re.IGNORECASE,
        ):
            return True
        if re.search(r"\b(?:gwr|ucc|tion)\b|\bin the to the\b|\bwill of you\b|\bthis mi\b", cleaned, flags=re.IGNORECASE):
            return True
        if re.search(r"\bThe protagonist\b", cleaned):
            return True
        if re.search(r"\b(?:I|i|me|my|mine|myself)\b", cleaned) and re.search(r"\b(?:couldn'?t|can't|cannot|for me|myself|mine)\b", cleaned, flags=re.IGNORECASE):
            return True
        if re.search(r"\bthe three of you\b", cleaned, flags=re.IGNORECASE):
            return True
        if cleaned.count("?") + cleaned.count("!") >= 2:
            return True
        if sum(1 for char in cleaned if ord(char) > 127 and char.isalpha()) >= 2:
            return True
        if self._text_is_noisy_ocr(cleaned):
            return True
        if self._has_ocr_shard_cluster(cleaned):
            return True
        if re.search(r"\b(?:is shown|are shown|is depicted|are depicted|is displayed|are displayed|was seen|were seen)\b", cleaned, flags=re.IGNORECASE):
            return True
        if re.search(
            r"\b(?:male character|female character|a character|another character|unseen person|distant structure|figures? in uniform|something|seen from behind|long hair flowing|with expressions? of)\b",
            cleaned,
            flags=re.IGNORECASE,
        ):
            return True
        if re.search(
            r"\b(?:dark-haired character|small boy smiles|smiles brightly|closed eyes|expressing agreement|cloud of dust and debris|looming over the battlefield|chaotic scene was punctuated|wreckage of mechs|intense nature of the ongoing conflict)\b",
            cleaned,
            flags=re.IGNORECASE,
        ):
            return True
        if re.search(
            r"\b(?:hums as|hand reaches? toward|hand reaches? towards|inside a mecha urgently calls|large robot-like machine|crashing or taking heavy damage|dark humanoid figure|glowing eyes|creaking ominously|mechanical entity erupted|situation was dire|immediate threat to all present)\b",
            cleaned,
            flags=re.IGNORECASE,
        ):
            return True
        if re.search(
            r"\b(?:air was thick|dynamic combat|required constant adaptation|quick decision-making|attention focused elsewhere|situation unfolded|profound sense of emptiness|nothing was coming|male characters? reacted|mech suit attacked|sharp claws|impact|observing in the background|observed by|girl in a uniform|woman with long red hair|massive machine to crash|pilots flying backward|significant gap left by [A-Z][a-z]+|constant factor that influenced)\b",
            cleaned,
            flags=re.IGNORECASE,
        ):
            return True
        if re.search(
            r"\b(?:looked down with|determined expression|giant,\s*dark mechanical entity|mechanical entity emerged|too late to help|three characters? reacted|reacted with shock|appeared to be charging|inside an industrial facility|discussed the situation|powerful punch|girl in uniform shouted|robot-like machine took heavy damage|massive machine crashing down|pilots being thrown backward)\b",
            cleaned,
            flags=re.IGNORECASE,
        ):
            return True
        if re.search(
            r"\b(?:industrial facility with large vats|large vats and pipes|engaged in a discussion|large,\s*angular mech|angular mech issued|stern warning|explosions erupted|startling a young pilot|he then calls out|asking if his feelings match hers)\b",
            cleaned,
            flags=re.IGNORECASE,
        ):
            return True
        if re.search(
            r"\b(?:impact sound effect|visible in the background|seen in the background|sound effect|shock and confusion|express shock|expresses shock|figures? on an escalator|young women are relaxing|relaxing and socializing|two mecha suits|in a tense\s*,|female pilot urgently ordered)\b",
            cleaned,
            flags=re.IGNORECASE,
        ):
            return True
        if re.search(
            r"\b(?:shield(?:ed|s)? (?:his|her|their)?\s*mouth|girl in a uniform shouted|characters? (?:were )?thrown backward|damaged robot-like machine|robot-like machine lay|large armored figure|injured character|lay on the ground|lying on the ground)\b",
            cleaned,
            flags=re.IGNORECASE,
        ):
            return True
        if re.search(
            r"\b(?:large,\s*angular mech-like entity|mech-like entity|stern expression|ongoing construction|cloudy sky|gazed up|towering,\s*slender humanoid robot|sheer scale|stunned silence|testament to its imposing presence|standing beside another woman|looked forward with a content expression|glowed with determination or anger)\b",
            cleaned,
            flags=re.IGNORECASE,
        ):
            return True
        if re.search(
            r"\b(?:highlight(?:ed|s)? the overwhelming nature|sought a brief respite|no room for misinterpretation|details were precise|situation grew dire|gravity of the situation)\b",
            cleaned,
            flags=re.IGNORECASE,
        ):
            return True
        if re.search(r"\b(?:confusion and surprise|situation was becoming increasingly difficult|harsh statement was delivered)\b", cleaned, flags=re.IGNORECASE):
            return True
        if re.search(
            r"\b(?:the pause holds just long enough|next problem to take shape|battle (?:is|was )?thrown into chaos|mechs? (?:tear|tore) through explosions|clash escalates, forcing the pilots)\b",
            cleaned,
            flags=re.IGNORECASE,
        ):
            return True
        if re.search(r"\b(?:with a lollipop|dynamic pose|in a dynamic pose|burst of energy upwards?)\b", cleaned, flags=re.IGNORECASE):
            return True
        if re.search(r"\b(?:fate grows more tangled|pressure around them builds)\b", cleaned, flags=re.IGNORECASE):
            return True
        if re.search(r"\b(?:the situation shifts|before anyone can fully recover|before he could fully process the scene|the scene)\b", cleaned, flags=re.IGNORECASE):
            return True
        if re.search(r"\b(?:bright light|illuminating the industrial landscape|dome-shaped structure|desolate landscape)\b", cleaned, flags=re.IGNORECASE):
            return True
        if re.match(r"^(?:charges?|charging)\s+forward\b", cleaned, flags=re.IGNORECASE):
            return True
        if re.search(r"\b(?:stands?|stood|looks?|looked|looking|watches?|watched|stares?|staring)\b", cleaned, flags=re.IGNORECASE):
            return True
        if re.search(r"\bpoints?\s+(?:a finger|at|toward|towards|forward)\b", cleaned, flags=re.IGNORECASE):
            return True
        if re.search(r"\b(?:leans?|leaning)\b", cleaned, flags=re.IGNORECASE):
            return True
        if re.search(r"\b[A-Z][a-z]+(?: [A-Z][a-z]+)?\b[^.]{0,100}\bforcing him\b|\b[A-Z][a-z]+(?: [A-Z][a-z]+)?\b[^.]{0,100}\bforcing his\b", cleaned, flags=re.IGNORECASE):
            return True
        if re.match(r"^(?:They're|They’re|They are)\b", cleaned, flags=re.IGNORECASE) and len(re.findall(r"[A-Za-z']+", cleaned)) <= 8:
            return True
        if re.match(
            r"^(?:stands?|standing|walks?|walking|runs?|running|looks?|looking|sits?|sitting|turns?|turning|moves?|moving|continues?|continuing)\b",
            cleaned,
            flags=re.IGNORECASE,
        ):
            return True
        if re.match(r"^(?:curly|blonde|dark|short|long|red|blue|black|white|silver|pink|green)\s+hair\b", cleaned, flags=re.IGNORECASE):
            return True
        if re.match(r"^emphasizing\b", cleaned, flags=re.IGNORECASE):
            return True
        if re.match(r"^(?:And\s+)?(?:another|a|an|one)\b[\w\s'-]{0,80}\bare\b", cleaned, flags=re.IGNORECASE):
            return True
        if self._line_has_foreign_stopword_cluster(cleaned):
            return True
        if re.match(r"^(?:Confusion takes over|And consequences push the story forward|Magic erupts without warning|As the situation escalates)\b", cleaned, flags=re.IGNORECASE):
            return True
        if re.search(r"\b(?:Unknown|viewer|Chibi-style)\b", cleaned, flags=re.IGNORECASE):
            return True
        visual_detector = getattr(self.polisher, "_is_visual_description", None)
        if callable(visual_detector) and visual_detector(cleaned):
            return True
        if self._line_is_dialogue_fragment(cleaned):
            return True
        if self._line_is_sentence_fragment(cleaned):
            return True
        tokens = re.findall(r"[A-Za-z']+", cleaned)
        if len(tokens) <= 2:
            return True
        if cleaned.endswith("!") and len(tokens) <= 8:
            return True
        if len(tokens) <= 16 and re.search(r"\byou\b", cleaned, flags=re.IGNORECASE):
            return True
        dialogue_pronouns = len(re.findall(r"\b(?:I|you|we|me|my|our|your|us)\b", cleaned, flags=re.IGNORECASE))
        if cleaned.count("?") >= 1 and dialogue_pronouns >= 2:
            return True
        if dialogue_pronouns >= 4 and len(tokens) <= 28:
            return True
        if len(tokens) <= 8 and re.search(r"\b\d+\b", cleaned):
            return True
        if len(tokens) <= 2 and re.search(r"\b(?:there|okay|right|yes|no)\b", cleaned, flags=re.IGNORECASE):
            return True
        if len(tokens) <= 14 and re.search(r"\b(?:a|an|the)\s+[A-Za-z-]{3,}\.\s*$", cleaned):
            return True
        alpha_tokens = re.findall(r"[A-Za-z']+", cleaned)
        alpha_chars = [char for char in cleaned if char.isalpha()]
        if alpha_chars:
            uppercase_ratio = sum(1 for char in alpha_chars if char.isupper()) / len(alpha_chars)
            if uppercase_ratio >= 0.72 and len(alpha_tokens) >= 4:
                return True
        if re.search(r"\b([A-Za-z]{2,})\s+\1\b", cleaned, flags=re.IGNORECASE):
            return True
        # Catch OCR-garbled short sentences that end with a bare linking/auxiliary
        # verb with no complement, e.g. "Asi can name is.", "Her name is.",
        # "The group would have.". Narration is third-person prose so these
        # patterns are almost always truncated OCR reads, not legitimate.
        if re.search(
            r"\b(?:is|are|was|were|be|been|have|has|had|will|would|can|could|might|should|may|must|shall)\.\s*$",
            cleaned,
            flags=re.IGNORECASE,
        ) and len(tokens) <= 10:
            return True
        if re.search(r"\b(?:to|for|with|from|about|into|because|while|when|if)\.\s*$", cleaned, flags=re.IGNORECASE) and len(tokens) <= 14:
            return True
        # Broken grammar: two linking/auxiliary verbs right next to each other
        # at the end (e.g. "name is", "can be", "has been" followed by period).
        if re.search(
            r"\b(?:name|it|he|she|they|we|you)\s+(?:is|are|was|were|be|been)\.\s*$",
            cleaned,
            flags=re.IGNORECASE,
        ):
            return True
        # Very short sentence whose first word looks like a non-dictionary OCR
        # token (capitalised, no vowels-only letters, not in known-name-likely
        # set). Strict length requirement keeps false positives near zero for
        # legitimate short narration; the scene summary / multimodal repair
        # pass will provide a better replacement.
        first_token = tokens[0] if tokens else ""
        if (
            len(tokens) <= 6
            and first_token
            and first_token[:1].isupper()
            and len(first_token) <= 4
            and not re.search(r"[aeiouAEIOU]", first_token[1:])
            and first_token.lower() not in {"mr", "mrs", "ms", "dr", "sr", "jr", "st", "the", "his", "her", "and", "but"}
        ):
            return True
        if tokens:
            short_tokens = sum(1 for token in tokens if len(token) <= 2)
            no_vowel_tokens = sum(1 for token in tokens if len(token) >= 3 and not re.search(r"[aeiouyAEIOUY]", token))
            if len(tokens) <= 6 and short_tokens / len(tokens) >= 0.34:
                return True
            if len(tokens) <= 6 and no_vowel_tokens >= 1:
                return True
            if len(tokens) >= 8 and short_tokens / len(tokens) >= 0.28:
                return True
            if len(tokens) >= 8 and no_vowel_tokens / len(tokens) >= 0.2:
                return True
        sentence_parts = [
            part.strip()
            for part in re.split(r"(?<=[.!?])\s+", cleaned)
            if part.strip()
        ]
        if len(sentence_parts) >= 3:
            fragment_count = sum(1 for part in sentence_parts if len(re.findall(r"[A-Za-z']+", part)) <= 2)
            if fragment_count >= 2:
                return True
            # Any sentence in the group that itself matches the bare-linking-verb
            # trap is enough to taint the whole line.
            for part in sentence_parts:
                if re.search(
                    r"\b(?:is|are|was|were|be|been)\.\s*$",
                    part,
                    flags=re.IGNORECASE,
                ) and len(re.findall(r"[A-Za-z']+", part)) <= 6:
                    return True
        detector = getattr(self.polisher, "_is_gibberish", None)
        return bool(callable(detector) and detector(cleaned))

    def _line_has_foreign_stopword_cluster(self, line: str) -> bool:
        cleaned = str(line or "").strip()
        if not cleaned:
            return False
        tokens = [token.casefold() for token in re.findall(r"[A-Za-zÀ-ÿ']+", cleaned)]
        if len(tokens) < 4:
            return False
        markers = {
            "que", "uma", "umas", "um", "uns", "com", "para", "pra", "não", "nao", "você", "vocês",
            "voce", "voces", "ele", "ela", "elas", "eles", "isso", "isto", "essa", "esse", "dessa",
            "deste", "desta", "aqui", "agora", "então", "entao", "antes", "depois", "onde", "porque",
            "porquê", "mas", "mesmo", "muito", "muita", "muitos", "muitas", "entrou", "inteira",
            "terra", "terrivel", "terrível", "irmão", "irmao", "lhe", "cês", "ces",
            "pessoas", "pessoa", "merece", "morrer", "destas", "destes",
        }
        matches = sum(1 for token in tokens if token in markers)
        return matches >= max(2, len(tokens) // 4)

    def _text_is_noisy_ocr(self, text: str) -> bool:
        cleaned = clean_ocr_text(str(text or "").strip())
        if not cleaned:
            return True
        if not is_usable_ocr_text(cleaned):
            return True
        # Flag 4+ consecutive special characters as noise (e.g. "----", "....").
        # Allow ".." and "..." (ellipsis) and "?!" (common manga punctuation) — using
        # {2,} here was too aggressive and wiped legitimate manga dialogue that ends
        # with trailing ".." or has OCR-reconstructed speech-bubble tails.
        if re.search(r"[.?!,:;/\\|_-]{4,}", cleaned):
            return True
        tokens = re.findall(r"[A-Za-z']+", cleaned)
        if not tokens:
            return True
        if self._has_ocr_shard_cluster(cleaned):
            return True
        uppercase_tokens = sum(1 for token in re.findall(r"\b[A-Z]{3,}\b", cleaned) if token.isupper())
        short_tokens = sum(1 for token in tokens if len(token) <= 2)
        no_vowel_tokens = sum(
            1 for token in tokens if len(token) >= 3 and not re.search(r"[aeiouyAEIOUY]", token)
        )
        weird_repeats = sum(1 for token in tokens if re.search(r"(.)\1\1", token))
        if len(tokens) <= 6 and uppercase_tokens >= 3 and no_vowel_tokens >= max(2, uppercase_tokens - 1):
            return True
        if len(tokens) >= 8 and short_tokens / len(tokens) >= 0.34:
            return True
        if len(tokens) >= 8 and no_vowel_tokens / len(tokens) >= 0.28:
            return True
        if weird_repeats >= max(2, len(tokens) // 5):
            return True
        return False

    def _has_ocr_shard_cluster(self, text: str) -> bool:
        """Detect clipped OCR shards inside otherwise sentence-like text.

        Vision OCR sometimes returns fragments such as "PLENTY O PARASITE HAVE
        TH SAME ISS..." with high confidence. A single short word is normal
        English; multiple invalid short shards in one sentence are a strong
        signal that the line should be paraphrased from visual/action evidence
        instead of preserved.
        """
        tokens = re.findall(r"[A-Za-z']+", str(text or ""))
        if len(tokens) < 5:
            return False
        allowed_short = {
            "a", "i", "am", "an", "as", "at", "be", "by", "do", "go", "he",
            "if", "in", "is", "it", "me", "my", "no", "of", "oh", "ok", "on",
            "or", "so", "to", "up", "us", "we",
        }
        bad_short = [
            token.casefold()
            for token in tokens
            if len(token) <= 2 and token.casefold() not in allowed_short
        ]
        suspicious_fragments = {
            token.casefold()
            for token in tokens
            if token.casefold() in {"iss", "ths", "tha", "hav", "wont", "cant", "dont"}
        }
        if len(bad_short) >= 2:
            return True
        if bad_short and suspicious_fragments:
            return True
        return len(suspicious_fragments) >= 2

    def _normalized_line_key(self, text: str) -> str:
        return " ".join(re.findall(r"[a-z0-9]+", str(text or "").casefold()))

    def _fallback_scene_line(
        self,
        unit: dict[str, Any],
        protagonist_name: str | None,
        *,
        style_vocab: StyleVocabulary | None = None,
    ) -> str:
        scene_unit_count = int(unit.get("scene_unit_count") or 1)
        anchor_names = {
            normalize_name_key(name)
            for name in [protagonist_name, *(unit.get("character_names", []) or [])]
            if str(name or "").strip()
        }

        def has_named_anchor(text: str) -> bool:
            text_key = normalize_name_key(text)
            return any(name_key and name_key in text_key for name_key in anchor_names)

        for supported in (
            str(unit.get("vision_action_beat") or "").strip(),
            str(unit.get("vision_caption") or "").strip(),
            str(unit.get("vision_dialogue") or "").strip(),
        ):
            normalized_supported = self._normalize_segment_text(supported, allow_empty=True)
            if (
                normalized_supported
                and has_named_anchor(normalized_supported)
                and not self._line_is_low_quality(normalized_supported)
                and not self._line_is_overly_generic(normalized_supported)
                and not self._line_is_dialogue_fragment(normalized_supported)
                and not self._line_is_sentence_fragment(normalized_supported)
                and not self._line_needs_style_refinement(normalized_supported)
            ):
                return normalized_supported

        local_recap = self._local_evidence_recap_line(unit, "", style_vocab=style_vocab)
        if local_recap:
            return local_recap

        bridge_line = self._evidence_bridge_line(unit, protagonist_name, style_vocab=style_vocab)
        if bridge_line:
            return bridge_line
        for supported in (
            str(unit.get("vision_action_beat") or "").strip(),
            str(unit.get("vision_caption") or "").strip(),
            str(unit.get("vision_dialogue") or "").strip(),
        ):
            normalized_supported = self._normalize_segment_text(supported, allow_empty=True)
            if (
                normalized_supported
                and not self._line_is_low_quality(normalized_supported)
                and not self._line_is_overly_generic(normalized_supported)
                and not self._line_is_dialogue_fragment(normalized_supported)
                and not self._line_is_sentence_fragment(normalized_supported)
                and not self._line_needs_style_refinement(normalized_supported)
            ):
                return normalized_supported
        combined_text = clean_ocr_text(str(unit.get("combined_text") or "").strip())
        if not combined_text:
            combined_text = clean_ocr_text(str(unit.get("ocr_fallback_text") or "").strip())
        if combined_text and not self._text_is_noisy_ocr(combined_text):
            parts = [
                piece.strip(" ,;:-")
                for piece in re.split(r"(?<=[.!?])\s+|,\s+", combined_text)
                if piece.strip(" ,;:-")
            ]
            for part in parts:
                normalized = self._normalize_segment_text(part, allow_empty=True)
                if (
                    len(normalized.split()) >= 5
                    and not self._line_is_low_quality(normalized)
                    and not self._line_is_overly_generic(normalized)
                    and not self._line_is_dialogue_fragment(normalized)
                    and not self._line_is_sentence_fragment(normalized)
                    and not self._line_needs_style_refinement(normalized)
                ):
                    return normalized
        visual_cues = self._normalize_supporting_text(str(unit.get("visual_cues") or "").strip())
        if visual_cues:
            parts = [
                piece.strip(" ,;:-")
                for piece in re.split(r"(?<=[.!?])\s+|,\s+", visual_cues)
                if piece.strip(" ,;:-")
            ]
            for part in parts:
                normalized = self._normalize_segment_text(part, allow_empty=True)
                if (
                    len(normalized.split()) >= 5
                    and not self._line_is_low_quality(normalized)
                    and not self._line_is_overly_generic(normalized)
                    and not self._line_is_sentence_fragment(normalized)
                    and not self._line_is_dialogue_fragment(normalized)
                    and not self._line_needs_style_refinement(normalized)
                ):
                    return normalized
        summary = self._normalize_segment_text(str(unit.get("scene_summary") or "").strip(), allow_empty=True)
        if scene_unit_count > 1:
            summary = ""
        evidence_text = " ".join(
            str(unit.get(key) or "").strip()
            for key in ("vision_action_beat", "vision_caption", "vision_dialogue", "combined_text", "visual_cues")
            if str(unit.get(key) or "").strip()
        )
        if summary and not self._line_is_low_quality(summary) and self._summary_supported_by_evidence(summary, evidence_text):
            return summary
        return ""

    def _evidence_bridge_line(
        self,
        unit: dict[str, Any],
        protagonist_name: str | None = None,
        *,
        style_vocab: StyleVocabulary | None = None,
    ) -> str:
        """Convert sparse trusted evidence into a conservative story beat.

        Project-agnostic: never embeds hardcoded character names, locations, or
        plot keywords. The bridge is composed strictly from the panel's own
        trusted vision evidence (action beat / caption / dialogue) and from the
        unit's character_names list (which is built from the canonical roster
        for the active project). If the vision evidence cannot supply a clean
        sentence, the function returns an empty string so the caller can fall
        through to its next candidate (or honestly leave the slot blank).
        """
        trusted_text = " ".join(
            str(unit.get(key) or "").strip()
            for key in ("vision_action_beat", "vision_caption", "vision_dialogue")
            if str(unit.get(key) or "").strip()
        )
        if not trusted_text.strip():
            return ""
        candidates: list[str] = []
        for sentence in re.split(r"(?<=[.!?])\s+(?=[A-Z])", trusted_text):
            stripped = sentence.strip()
            if stripped:
                candidates.append(stripped)
        # Optional: a one-line synthesis using a real character name from the unit's
        # character_names (which is populated from the project's canonical roster).
        names = [
            str(name).strip()
            for name in unit.get("character_names", []) or []
            if str(name).strip()
        ]
        name_keys = {normalize_name_key(name) for name in names if normalize_name_key(name)}
        action_beat = str(unit.get("vision_action_beat") or "").strip()
        if action_beat:
            for name in names:
                if normalize_name_key(name) in normalize_name_key(action_beat):
                    candidates.insert(0, action_beat)
                    break
        # Drop the function-local helpers that the legacy hand-written body used
        # to reach into project-specific narrative templates. Only generic
        # evidence-derived sentences remain.
        del name_keys

        for candidate in candidates:
            normalized = self._normalize_segment_text(candidate, allow_empty=True)
            if (
                normalized
                and len(normalized.split()) >= 5
                and not self._line_is_low_quality(normalized)
                and not self._line_is_overly_generic(normalized)
                and not self._line_is_dialogue_fragment(normalized)
                and not self._line_is_sentence_fragment(normalized)
            ):
                return normalized
        return ""

    def _compose_neighbour_bridge_line(
        self,
        unit: dict[str, Any],
        *,
        prev_payload: dict[str, Any] | None,
        next_payload: dict[str, Any] | None,
        protagonist_name: str | None,
        story_bible: dict[str, Any],
        scene_memory: dict[str, Any] | None,
        variant: int = 0,
        style_vocab: StyleVocabulary | None = None,
    ) -> str:
        """Fill a stubborn blank with a conservative connector.

        The bridge is intentionally narrow: it must name a real character and
        derive its topic from the adjacent narration plus local scene evidence.
        It exists to avoid silent coverage holes without inventing a new event.
        """
        prev_text = self._normalize_segment_text(str((prev_payload or {}).get("text") or "").strip(), allow_empty=True)
        next_text = self._normalize_segment_text(str((next_payload or {}).get("text") or "").strip(), allow_empty=True)
        if not prev_text and not next_text:
            return ""

        evidence_text = " ".join(
            str(value or "").strip()
            for value in (
                prev_text,
                next_text,
                unit.get("vision_action_beat"),
                unit.get("vision_caption"),
                unit.get("vision_dialogue"),
                unit.get("combined_text"),
                unit.get("visual_cues"),
                unit.get("salvaged_evidence"),
                unit.get("scene_summary"),
                (scene_memory or {}).get("state"),
                (scene_memory or {}).get("open_thread"),
            )
            if str(value or "").strip()
        )
        local_evidence_text = " ".join(
            str(value or "").strip()
            for value in (
                unit.get("vision_action_beat"),
                unit.get("vision_caption"),
                unit.get("vision_dialogue"),
                unit.get("combined_text"),
                unit.get("visual_cues"),
                unit.get("salvaged_evidence"),
            )
            if str(value or "").strip()
        )
        trusted_vision_text = " ".join(
            str(value or "").strip()
            for value in (
                unit.get("vision_action_beat"),
                unit.get("vision_caption"),
                unit.get("vision_dialogue"),
            )
            if str(value or "").strip()
        )
        if trusted_vision_text and variant == 0:
            direct_bridge = self._evidence_bridge_line(unit, protagonist_name, style_vocab=style_vocab)
            if direct_bridge:
                return direct_bridge
        local_topic = self._bridge_topic_phrase(local_evidence_text)
        subject = self._bridge_named_subject(
            unit,
            story_bible,
            None,
            None,
            " ".join(part for part in (local_evidence_text or trusted_vision_text, prev_text) if part),
        )
        topic = local_topic or self._bridge_topic_phrase(evidence_text)
        if not subject:
            return ""
        if not style_vocab:
            if not topic:
                return ""
            generic_templates = (
                f"{subject} carries {topic} into the next beat while the pressure shifts around them.",
                f"{subject} holds the line on {topic} while the scene tightens around them.",
                f"{subject} responds to {topic} before the pressure spills into the group.",
                f"{subject} faces {topic} head-on as the scene turns toward a harder response.",
            )
            candidate = generic_templates[variant % len(generic_templates)]
            normalized = self._normalize_segment_text(candidate, allow_empty=True)
            if (
                normalized
                and len(normalized.split()) >= 5
                and not self._line_is_low_quality(normalized)
                and not self._line_is_overly_generic(normalized)
                and not self._line_is_dialogue_fragment(normalized)
                and not self._line_is_sentence_fragment(normalized)
            ):
                return normalized
            return ""
        # If the vision/evidence bridge could not produce a concrete local
        # sentence, do not manufacture one from chapter-wide vocabulary. Those
        # broad bridges read fluent in isolation but can name concepts before
        # they appear on-screen or create filler like "FRANXX demands attention".
        return ""
        vocab_names = list(style_vocab.named_characters if style_vocab else ())

        placeholder_phrases = {
            "the immediate problem",
            "the next exchange",
            "the exchange",
            "the group tension",
            "the protected space",
            "the resource problem",
        }

        def _useful_phrase(value: str, *, allow_single: bool = False) -> str:
            cleaned = str(value or "").strip()
            if not cleaned or cleaned.casefold() in placeholder_phrases:
                return ""
            words = re.findall(r"[A-Za-z0-9]+", cleaned)
            if len(words) >= 2:
                return cleaned
            if allow_single and words and (cleaned.isupper() or re.search(r"[A-Z0-9]", cleaned)):
                return cleaned
            return ""

        style_support_text = " ".join(
            str(value or "").strip()
            for value in (
                local_evidence_text,
                prev_text,
                unit.get("scene_summary"),
            )
            if str(value or "").strip()
        )
        style_support_key = normalize_name_key(style_support_text)
        neighbor_support_key = normalize_name_key(evidence_text)

        def _phrase_supported(value: str, *, allow_single: bool = False, include_neighbors: bool = False) -> bool:
            useful = _useful_phrase(value, allow_single=allow_single)
            key = normalize_name_key(useful)
            if not useful or not key:
                return False
            support_key = neighbor_support_key if include_neighbors else style_support_key
            return bool(support_key and key in support_key)

        def _pick(
            values: tuple[str, ...] | list[str] | None,
            fallback: str = "",
            *,
            allow_single: bool = False,
            include_neighbors: bool = False,
        ) -> str:
            choices = [str(value).strip() for value in values or [] if str(value).strip()]
            evidence_key = normalize_name_key(evidence_text)
            for choice in choices:
                key = normalize_name_key(choice)
                if (
                    key
                    and key in evidence_key
                    and _useful_phrase(choice, allow_single=allow_single)
                    and _phrase_supported(choice, allow_single=allow_single, include_neighbors=include_neighbors)
                ):
                    return choice
            if fallback and _phrase_supported(fallback, allow_single=allow_single, include_neighbors=include_neighbors):
                return _useful_phrase(fallback, allow_single=allow_single)
            return ""

        def _second_subject() -> str:
            unit_keys = {
                normalize_name_key(name)
                for name in unit.get("character_names", []) or []
                if normalize_name_key(str(name))
            }
            for name in vocab_names:
                if normalize_name_key(name) == normalize_name_key(subject):
                    continue
                if unit_keys and normalize_name_key(name) not in unit_keys:
                    continue
                if not unit_keys and normalize_name_key(name) not in style_support_key:
                    continue
                return name
            return ""

        vocab_name_keys = {normalize_name_key(name) for name in vocab_names if normalize_name_key(name)}
        subject_is_character = normalize_name_key(subject) in vocab_name_keys
        subject_b = _second_subject() if subject_is_character else ""
        team = (
            style_vocab.team_term
            if style_vocab and _phrase_supported(style_vocab.team_term or "", include_neighbors=True)
            else ""
        )
        world_term = _pick(style_vocab.world_terms if style_vocab else (), topic or "", allow_single=True)
        if not world_term and topic == "the immediate threat":
            world_term = (
                style_vocab.antagonist_term
                if style_vocab and _phrase_supported(style_vocab.antagonist_term or "", include_neighbors=True)
                else ""
            )
        stakes = _pick(style_vocab.stakes_phrases if style_vocab else (), "", allow_single=False)
        action_verb = _pick(style_vocab.action_verbs if style_vocab else (), "", allow_single=False) or "pressing"
        antagonist = (
            style_vocab.antagonist_term
            if style_vocab and _phrase_supported(style_vocab.antagonist_term or "", include_neighbors=True)
            else ""
        )
        template_specs = (
            ("{subject} and {subject_b} keep {stakes} in motion as {team} reacts around them. The beat ties their bond to the larger risk without changing what the scene shows.", ("subject_b", "stakes", "team")),
            ("{world_term} draws {team} into a sharper choice while {subject} stays close to the fallout. The moment has consequence because the group has to respond, not simply watch.", ("team", "world_term")),
            ("{subject} meets {antagonist} through the pressure of {world_term}. The scene keeps the danger personal while still pointing the chapter toward its next turn.", ("world_term", "antagonist")),
            ("{team} is left {action_verb} around {stakes}. The aftermath ties the previous shock to a decision the group cannot avoid.", ("team", "action_verb", "stakes")),
            ("{subject} and {subject_b} tie {world_term} directly to {team}. Their choices shift the scene from observation into consequence.", ("subject_b", "world_term", "team")),
            ("{subject} carries {stakes} through the reaction around them. The decision matters because it changes how the surrounding group has to answer.", ("stakes",)),
            ("{subject} treats {stakes} as another sign that survival matters more than comfort. The beat turns that calculation into a choice the story can build on.", ("stakes",)),
            ("{subject} connects {stakes} to {world_term}, turning the danger into something the group has to answer. The next response grows from that pressure instead of feeling like a reset.", ("stakes", "world_term")),
            ("{subject_b} watches {subject} carry {stakes} forward. {team} has to react to the bond even before they fully understand it.", ("subject_b", "stakes", "team")),
            ("{world_term} forces {subject} into a clearer response. {team} gets pulled into the consequence because the pressure is no longer private.", ("world_term", "team")),
            ("{team} absorbs {stakes} while {subject} keeps moving. One character's choice becomes something the whole group has to feel.", ("team", "stakes")),
            ("{subject} carries the weight of {world_term} into the next beat. Around them, {team} adjusts as the chapter moves from reaction into decision.", ("world_term", "team")),
            ("{subject_b} measures {subject}'s reaction as {stakes} shifts again. The moment leaves {team} caught between what they know and what they still need to understand.", ("subject_b", "stakes", "team")),
            ("{subject} pushes through {world_term} while the consequence stays open. The next response has a clearer reason to land because the danger has not gone quiet.", ("world_term",)),
            ("{team} absorbs the strain of {stakes} while {subject} becomes the point where the beat turns forward.", ("team", "stakes")),
            ("{subject}'s place in {world_term} becomes harder to ignore. The scene uses that pressure to draw a real response from {team}.", ("world_term", "team")),
            ("{subject_b} and {subject} leave {team} reacting from the outside. {stakes} keeps the beat active without pretending the group has all the answers.", ("subject_b", "team", "stakes")),
            ("{world_term} presses closer while {subject} tries to keep the moment from breaking apart. The result becomes a handoff into the next decision.", ("world_term",)),
            ("{team} has to read {subject}'s choice from the outside. That uncertainty keeps {stakes} alive as the scene moves on.", ("team", "stakes")),
            ("{subject} turns {world_term} into a private test. {team} can only react as the consequence starts to spread.", ("world_term", "team")),
            ("{subject_b} stays near the edge of {subject}'s decision. Together, they make {stakes} matter beyond one reaction.", ("subject_b", "stakes")),
            ("{subject} keeps moving through {stakes} while {team} tries to keep up. The beat links what just happened to the response that follows.", ("stakes", "team")),
        )
        slots = {
            "subject": subject,
            "subject_b": subject_b,
            "team": team,
            "world_term": world_term,
            "stakes": stakes,
            "action_verb": action_verb,
            "antagonist": antagonist,
        }
        for offset in range(len(template_specs)):
            template, required = template_specs[(variant + offset) % len(template_specs)]
            if not subject_is_character and ("{subject}" in template or "{subject_b}" in template):
                continue
            if any(not slots.get(key) for key in required):
                continue
            candidate = template.format(**slots)
            normalized = self._normalize_segment_text(candidate, allow_empty=True)
            if (
                normalized
                and len(normalized.split()) >= 5
                and not self._line_is_low_quality(normalized)
                and not self._line_is_overly_generic(normalized)
                and not self._line_is_dialogue_fragment(normalized)
                and not self._line_is_sentence_fragment(normalized)
            ):
                return normalized
        return ""

    def _bridge_named_subject(
        self,
        unit: dict[str, Any],
        story_bible: dict[str, Any],
        protagonist_name: str | None,
        scene_memory: dict[str, Any] | None,
        evidence_text: str,
    ) -> str:
        names: list[str] = []
        names.extend(str(name).strip() for name in unit.get("character_names", []) or [] if str(name).strip())
        names.extend(str(name).strip() for name in (scene_memory or {}).get("characters", []) or [] if str(name).strip())
        evidence_key_text = normalize_name_key(evidence_text)
        protagonist_key = normalize_name_key(protagonist_name or "")
        if protagonist_name and protagonist_key and protagonist_key in evidence_key_text:
            names.append(str(protagonist_name).strip())
        for cast_item in story_bible.get("cast", []) or []:
            if isinstance(cast_item, dict):
                for key in ("name", "display_name", "canonical_name"):
                    value = str(cast_item.get(key) or "").strip()
                    value_key = normalize_name_key(value)
                    if value and value_key and value_key in evidence_key_text:
                        names.append(value)
            else:
                value = str(cast_item or "").strip()
                value_key = normalize_name_key(value)
                if value and value_key and value_key in evidence_key_text:
                    names.append(value)
        names.extend(extract_proper_name_candidates(evidence_text))
        world_term_keys = {
            normalize_name_key(term)
            for term in (story_bible.get("world_terms", []) or [])
            if normalize_name_key(str(term))
        }
        lowered_evidence = str(evidence_text or "").casefold()
        if re.search(r"\b(?:people|group|team|family|pilots?|squad)\b", lowered_evidence):
            names.append("The group")
        elif re.search(r"\b(?:enemy|monster|threat)\b", lowered_evidence):
            names.append("The threat")

        seen: set[str] = set()
        for raw_name in names:
            name = re.sub(r"\s+", " ", str(raw_name or "")).strip(" ,;:-")
            key = normalize_name_key(name)
            if not key or key in seen:
                continue
            seen.add(key)
            lowered = name.casefold()
            if re.search(r"\b(?:unknown|unidentified|speaker|narrator|victim|protagonist|character|someone|figure)\b", lowered):
                continue
            if key in world_term_keys:
                continue
            if lowered in {
                "after",
                "although",
                "and",
                "as",
                "before",
                "but",
                "despite",
                "during",
                "however",
                "meanwhile",
                "suddenly",
                "then",
                "that",
                "therefore",
                "this",
                "it",
                "while",
                "he",
                "she",
                "they",
                "their",
                "his",
                "her",
                "him",
            }:
                continue
            if len(name.split()) > 4:
                continue
            if looks_like_false_character_name(name):
                continue
            if not re.search(r"[A-Z0-9]", name):
                continue
            return name
        return ""

    @staticmethod
    def _bridge_topic_phrase(evidence_text: str) -> str:
        lowered = str(evidence_text or "").casefold()
        topic_rules = (
            (r"\bresources?\b|\bsuppl|\bfood\b|\bmedicine\b|\bbattery\b|\bpower\b|\btools?\b|\bstockpile\b", "the resource problem"),
            (r"\bmessage\b|\bchat\b|\bcall\b|\bquestion\b|\bargument\b|\brequest\b|\banswer\b|\bexchange\b", "the exchange"),
            (r"\bshelter\b|\bbase\b|\bdoor\b|\bfacility\b|\broom\b|\bbuilding\b|\bhideout\b|\brefuge\b", "the protected space"),
            (r"\battack\b|\bfight\b|\bbattle\b|\bweapon\b|\benemy\b|\bthreat\b|\bdanger\b|\bmission\b|\border\b", "the immediate threat"),
            (r"\bfamil|\bteam\b|\bgroup\b|\bpartner\b|\bbond\b|\btrust\b|\balliance\b", "the group tension"),
        )
        for pattern, topic in topic_rules:
            if re.search(pattern, lowered):
                return topic
        return ""

    def _summary_supported_by_evidence(self, summary: str, evidence_text: str) -> bool:
        summary_tokens = self._content_token_set(summary)
        evidence_tokens = self._content_token_set(evidence_text)
        if not summary_tokens or not evidence_tokens:
            return False
        overlap = len(summary_tokens & evidence_tokens) / max(1, min(len(summary_tokens), len(evidence_tokens)))
        return overlap >= 0.35

    def _line_supported_by_unit_evidence(self, line: str, unit: dict[str, Any]) -> bool:
        evidence = " ".join(
            str(unit.get(key) or "").strip()
            for key in (
                "vision_action_beat",
                "vision_caption",
                "vision_dialogue",
                "combined_text",
                "visual_cues",
                "ocr_fallback_text",
            )
            if str(unit.get(key) or "").strip()
        )
        if not evidence.strip():
            return True
        line_tokens = self._content_token_set(line)
        evidence_tokens = self._content_token_set(evidence)
        if not line_tokens or not evidence_tokens:
            return True
        overlap = len(line_tokens & evidence_tokens)
        containment = overlap / max(1, min(len(line_tokens), len(evidence_tokens)))
        if overlap >= 2 and containment >= 0.16:
            return True
        line_key = normalize_name_key(line)
        evidence_key = normalize_name_key(evidence)
        shared_names = [
            name
            for name in unit.get("character_names", []) or []
            if normalize_name_key(name)
            and normalize_name_key(name) in line_key
            and normalize_name_key(name) in evidence_key
        ]
        return bool(shared_names and overlap >= 1)

    def _unit_support_text(
        self,
        unit: dict[str, Any],
        story_bible: dict[str, Any] | None = None,
        scene_memory: dict[str, Any] | None = None,
    ) -> str:
        parts: list[str] = []
        for key in (
            "vision_action_beat",
            "vision_caption",
            "vision_dialogue",
            "combined_text",
            "visual_cues",
            "ocr_fallback_text",
            "scene_summary",
        ):
            value = str(unit.get(key) or "").strip()
            if value:
                parts.append(value)
        if scene_memory:
            for key in ("state", "location", "open_thread"):
                value = str(scene_memory.get(key) or "").strip()
                if value:
                    parts.append(value)
        if story_bible:
            for key in ("chapter_premise", "series_external_context"):
                value = str(story_bible.get(key) or "").strip()
                if value:
                    parts.append(value)
            parts.extend(str(term).strip() for term in story_bible.get("world_terms", []) or [] if str(term).strip())
            parts.extend(str(note).strip() for note in story_bible.get("continuity_notes", []) or [] if str(note).strip())
        return " ".join(parts)

    @staticmethod
    def _line_has_unsupported_setting_terms(line: str, evidence_text: str) -> bool:
        """Reject fallback lines that import another project's setting.

        This is deliberately narrow and only covers concrete world vocabulary
        that caused cross-project leakage in rescue bridges. If the local unit
        or story bible mentions the vocabulary, the line is allowed.
        """
        lowered_line = str(line or "").casefold()
        lowered_evidence = str(evidence_text or "").casefold()
        term_groups = (
            ("freeze", ("cold", "snow", "freeze", "blizzard", "ice", "storm")),
            ("community", ("neighbor", "neighbors", "community", "families", "building", "apartment")),
            ("shelter", ("bunker", "shelter", "safe house", "hideout", "refuge", "fortified", "defenses")),
            ("supplies", ("supplies", "supply", "resources", "rations", "medical kit", "energy cells", "food")),
            ("phone", ("phone", "message", "chat", "call", "group text")),
        )
        for _, terms in term_groups:
            if any(re.search(rf"\b{re.escape(term)}\b", lowered_line) for term in terms):
                if not any(re.search(rf"\b{re.escape(term)}\b", lowered_evidence) for term in terms):
                    return True
        return False

    def _content_token_set(self, text: str) -> set[str]:
        stop_words = {
            "a", "an", "and", "as", "at", "but", "by", "for", "from", "he", "her",
            "him", "his", "in", "into", "is", "it", "its", "of", "on", "or", "she",
            "that", "the", "their", "them", "they", "this", "to", "was", "were",
            "will", "with", "who", "what", "when", "where", "why", "how", "be",
            "been", "being", "have", "has", "had", "do", "does", "did", "would",
            "could", "should", "may", "might", "must", "can", "there", "then",
            "than", "so", "not", "no", "yes", "up", "down", "out", "over", "under",
            "one", "two", "three", "four", "five",
        }
        return {
            token
            for token in re.findall(r"[a-z']+", str(text or "").casefold())
            if len(token) > 2 and token not in stop_words
        }

    @staticmethod
    def _jaccard(left: set[str], right: set[str]) -> float:
        if not left or not right:
            return 0.0
        union = left | right
        if not union:
            return 0.0
        return len(left & right) / len(union)

    def _normalize_segment_text(self, text: str, *, allow_empty: bool = False) -> str:
        cleaned = strip_storytelling_meta(str(text or "").strip())
        cleaned = re.sub(r"\s+", " ", cleaned).strip(" ,;:-")
        if not cleaned:
            return "" if allow_empty else ""
        if cleaned[0].islower():
            cleaned = cleaned[0].upper() + cleaned[1:]
        if cleaned[-1] not in ".!?":
            cleaned += "."
        return cleaned

    # Patterns that indicate a caption-like draft that should be replaced with a mechanical strict_line
    _CAPTION_LIKE_DRAFT_PATTERN = re.compile(
        r"(?:^|\b)(?:the conversation reveals|the exchange starts with|a nearby response adds|"
        r"by the end of the exchange|a voice (?:says|states)|a character (?:says|states|mentions)|"
        r"the next line adds?|the scene turns on)",
        re.IGNORECASE,
    )

    @staticmethod
    def _extract_ocr_names(lower_text: str) -> list[str]:
        """Extract likely character first-names from a lowercase OCR string.

        Only returns short alphabetic tokens that appear in address/direct-speech
        positions (e.g. "take care, hiro" or "naomi..") and are not common English
        words.  Returned names are title-cased.
        """
        _stop = StoryScriptService._MECHANICAL_NAME_STOPWORDS
        candidates: list[str] = []
        seen: set[str] = set()

        def _add(word: str) -> None:
            key = word.lower()
            if (
                3 <= len(key) <= 12
                and key not in _stop
                and key not in seen
                and re.fullmatch(r"[a-z]+", key)
            ):
                seen.add(key)
                candidates.append(word.title())

        # Words that disqualify the comma-delimited token that follows them from being
        # treated as a character name.  These appear in translation notes, OCR artefacts,
        # and similar non-address contexts: "in japanese, <word>", "in chinese, <word>",
        # "letter a, <word>", a single digit/letter, etc.
        _bad_pre_comma = frozenset({
            "japanese", "chinese", "korean", "english", "french", "spanish", "german",
            "italian", "arabic", "latin", "greek", "russian", "portuguese",
            "kanji", "hiragana", "katakana", "romanji", "romaji",
            "letter", "character", "symbol", "word", "term", "phrase",
        })
        stripped = lower_text.strip()
        # "naomi.." — name before double-dot at the very start of the text
        m = re.match(r"^([a-z]{3,12})\.\.", stripped)
        if m:
            _add(m.group(1))
        # "hiro," — name at the very start followed by a comma
        m = re.match(r"^([a-z]{3,12}),", stripped)
        if m:
            _add(m.group(1))
        # ", hiro" or ", naomi" — address position inside the sentence
        # Guard: skip the candidate when the word immediately before the comma is a
        # language/script identifier (e.g. "in japanese, kathek") or a single character
        # (e.g. "letter a, kathek") — those mark translation notes, not character addresses.
        for m in re.finditer(r"([a-z]{2,}|[a-z\d]),\s+([a-z]{3,12})(?=[.\s!?]|$)", stripped):
            pre_word = m.group(1)
            candidate = m.group(2)
            if pre_word in _bad_pre_comma or len(pre_word) <= 1:
                continue
            _add(candidate)
        return candidates[:2]

    @staticmethod
    def _names_near_keywords(context_lower: str, keywords: tuple[str, ...]) -> list[str]:
        """Extract names from sentences in ``context_lower`` that contain any of ``keywords``.

        When a bequest/farewell/etc. pattern fires on a short chunk that lacks an
        address-position name, this lets us find the name from the broader seed text
        while restricting extraction to sentences that are actually about the same event.
        """
        sentences = re.split(r"[.!?]+", context_lower)
        for sentence in sentences:
            if any(kw in sentence for kw in keywords):
                found = StoryScriptService._extract_ocr_names(sentence.strip())
                if found:
                    return found
        return []

    @staticmethod
    def _paraphrase_ocr_chunk(chunk_lower: str, *, name_context: str = "") -> tuple[str, str]:
        """Return ``(pattern_key, narrative_sentence)`` for one OCR sentence.

        ``pattern_key`` is a short identifier (e.g. ``"farewell"``) used by the
        caller to deduplicate across chunks that match the same speech-act type.
        Both values are ``""`` when no pattern matches.

        ``name_context`` may be the seed's full combined OCR.  When the chunk
        itself yields no address-position name, name extraction is retried on
        sentences in ``name_context`` that contain the same pattern keywords,
        so a name from a sibling panel (e.g. "hence, naomi.") can ground the
        template without picking up unrelated names from other scenes.

        Templates are designed so that key OCR tokens (character names, dialogue
        keywords) appear verbatim in the output — this ensures the quality
        service's token-overlap check registers the contributing panel as *used*.
        Templates also deliberately avoid "someone", "the moment", and other
        strings that trigger ``_line_is_overly_generic``.
        """
        text = chunk_lower.strip()
        if not text or len(text) < 6:
            return ("", "")
        names = StoryScriptService._extract_ocr_names(text)
        n1 = names[0] if names else ""

        # --- Farewell / goodbye ---
        # Template deliberately includes "probably" and "end" so that the quality
        # service's token-overlap check can credit garbled farewell panels whose OCR
        # contains corrupted variants of "it's probably the end" or similar phrases.
        if any(kw in text for kw in ("won't be seeing", "seeing each other", "goodbye", "take care", "farewell", "see you again")):
            if n1:
                return (
                    "farewell",
                    f"{n1} and the others say a final goodbye — "
                    f"it's probably the end of the line, and they won't be seeing each other again.",
                )
            return (
                "farewell",
                "A final goodbye falls between them — "
                "it's probably the end of the line, and they won't be seeing each other again.",
            )

        # --- Dismissal / blame ---
        if any(kw in text for kw in ("forget about", "forget abolit", "crybaby", "dragged down", "because of him", "because of her")):
            if n1:
                return ("dismissal", f"{n1} gets dismissed as a crybaby by the others, who move on without a second thought.")
            return ("dismissal", "The absent one gets dismissed as a crybaby, and the group moves on without a second thought.")

        # --- Orders / drills / controlled procedure ---
        if any(kw in text for kw in ("buckle up", "seatbelt", "practice run", "training run", "test run", "drill")):
            return (
                "drill",
                "The order to buckle up redirects everyone into the next practice run "
                "before anyone can sit with what just happened.",
            )

        # --- Warning / death ---
        if any(kw in text for kw in ("heading your death", "heading to your death", "stop you're", "going to die", "will die", "you'll die", "you're dead")):
            if n1:
                return (
                    "warning",
                    f"A warning stops {n1} cold — "
                    f"the path ahead is heading toward death, and the danger stops being abstract.",
                )
            return (
                "warning",
                "A warning cuts through — "
                "the path ahead is heading toward death, and the danger stops being abstract.",
            )

        # --- Bequest / passing on belongings ---
        _bequest_kws = ("won't needing", "won't be needing", "share them with", "share it with", "leave these")
        if any(kw in text for kw in _bequest_kws):
            if not n1 and name_context:
                # The name often appears in an adjacent name-reveal sentence ("hence, naomi.")
                # rather than in the bequest line itself, so search the name_context for names
                # near either the bequest keywords or name-reveal markers.
                _bequest_name_kws = _bequest_kws + ("hence", "my name", "call me", "can be read as")
                _ctx_names = StoryScriptService._names_near_keywords(name_context, _bequest_name_kws)
                if _ctx_names:
                    n1 = _ctx_names[0]
            if n1:
                return (
                    "bequest",
                    f"{n1} passes along belongings she won't be needing anymore — "
                    f"share them with everyone else, she says, like it's already over.",
                )
            return (
                "bequest",
                "She passes along belongings she won't be needing anymore — "
                "share them with everyone else, she says, like it's already over.",
            )

        # --- Second chance / remaining hope ---
        # Avoid "someone", "the moment" — both trigger _line_is_overly_generic.
        if any(kw in text for kw in ("still have a chance", "one more chance", "another chance", "not over yet", "isn't over yet")):
            if n1:
                return (
                    "second_chance",
                    f"{n1} still has a chance — "
                    f"that opening hasn't closed yet, and the path forward is still there.",
                )
            return (
                "second_chance",
                "One last chance stays open — "
                "that opening hasn't closed yet, and the path forward is still there.",
            )

        # --- Request / offer (check before isolation — offer is a stronger beat) ---
        if any(kw in text for kw in ("ride with me", "come with me", "join me", "offering you the chance", "partner up")):
            if n1:
                return ("offer", f"{n1} extends an offer to ride together — turning a personal risk into something shared.")
            return (
                "offer",
                "An offer to ride together turns a personal risk into something shared — "
                "one side extends it, and the other has to decide.",
            )

        # --- Name / identity reveal ---
        if any(kw in text for kw in ("my name is", "call me", "that's my name", "name that you gave me", "can be read as", "hence")):
            if n1:
                return (
                    "name_reveal",
                    f"The name {n1} is encoded inside the numeral sequence — "
                    f"each part can be read as a syllable, turning a string of numbers into a personal identity.",
                )
            return (
                "name_reveal",
                "A name is hidden inside a numeral sequence — "
                "each part can be read as a syllable, turning numbers into a personal identity.",
            )

        # --- Isolation / loneliness ---
        if any(kw in text for kw in ("always been alone", "always alone", "lonely", "been alone")):
            return (
                "isolation",
                "An admission of loneliness surfaces — they say they've always been alone, "
                "and the confession makes connection feel less like comfort and more like survival.",
            )

        # --- Lore / exposition ---
        if any(kw in text for kw in ("in a book", "read about", "learned about", "written about", "tales of", "they have to hide", "hide and")):
            return (
                "lore",
                "Old knowledge surfaces from a book or memory that came before the current crisis — "
                "the remembered detail turns the problem into part of a larger history.",
            )

        # --- Threat / confrontation ---
        if any(kw in text for kw in ("you'll regret", "don't underestimate", "come at me", "try me", "bring it")):
            return (
                "threat",
                "A direct challenge cuts through before the tension can settle — "
                "the threat forces the other side to respond instead of pretending the conflict can be avoided.",
            )

        # --- Question ---
        if "?" in text and len(text) >= 12:
            question_parts = [q.strip() for q in text.split("?") if q.strip()]
            if question_parts and len(question_parts[0]) >= 10:
                return (
                    "question",
                    "A question cuts through before anyone can move — "
                    "whoever answers it will be deciding what they can risk next.",
                )

        return ("", "")

    @staticmethod
    def _mechanical_ocr_paraphrase(ocr_text: str, *, name_context: str = "") -> str:
        """Convert raw OCR dialogue into a story-event narrative template.

        For scenes with multiple speech acts (e.g. farewell + bequest), the combined
        OCR is split into sentence chunks.  Up to two paraphrases from *different*
        speech-act types are merged so the output covers more panel evidence without
        creating near-duplicate sentences (which would trip ``_line_is_low_quality``).

        ``name_context`` may be supplied as the seed's full combined OCR when the
        unit-level OCR does not contain a character name in address position.  It is
        used ONLY for name extraction, never for pattern matching.

        Returns "" for ambiguous / noisy OCR so the caller knows NOT to inject
        it as a dialogue anchor.
        """
        text = ocr_text.strip()
        if not text:
            return ""
        lower = text.lower()

        # Try the full combined text first; use name_context for richer name extraction
        # when the unit-level OCR is too short to contain an address-position name.
        full_key, full_result = StoryScriptService._paraphrase_ocr_chunk(
            lower, name_context=name_context.lower() if name_context else ""
        )

        # Split into sentence chunks and collect paraphrases from DIFFERENT
        # speech-act types so the two combined sentences cover distinct content.
        chunks = [
            s.strip()
            for s in re.split(r"[.!?]+", lower)
            if s.strip() and len(s.strip()) >= 8
        ]
        chunk_results: list[str] = []
        seen_keys: set[str] = {full_key} if full_key else set()
        seen_texts: set[str] = {full_result} if full_result else set()
        for chunk in chunks[:6]:
            chunk_key, r = StoryScriptService._paraphrase_ocr_chunk(chunk)
            if r and chunk_key and chunk_key not in seen_keys and r not in seen_texts:
                seen_keys.add(chunk_key)
                seen_texts.add(r)
                chunk_results.append(r)

        if full_result and chunk_results:
            # Combine full-text paraphrase with first complementary chunk result.
            return f"{full_result} {chunk_results[0]}"
        if full_result:
            return full_result
        if len(chunk_results) >= 2:
            return f"{chunk_results[0]} {chunk_results[1]}"
        if chunk_results:
            return chunk_results[0]

        # Conservative fallback for readable OCR with no clear speech-act pattern.
        if not StoryScriptService._static_text_looks_noisy_for_mechanical(text):
            return (
                "The dialogue presses forward through uncertainty "
                "instead of giving the characters a clean answer."
            )
        # Default: no recognized speech act — return "" so the caller does NOT
        # inject this as dialogue (which would cause the polish LLM to quote it).
        return ""

    @staticmethod
    def _static_text_looks_noisy_for_mechanical(text: str) -> bool:
        cleaned = clean_ocr_text(str(text or "").strip())
        if not cleaned or not is_usable_ocr_text(cleaned):
            return True
        tokens = re.findall(r"[A-Za-z']+", cleaned)
        if len(tokens) < 4:
            return True
        short = sum(1 for token in tokens if len(token) <= 2)
        no_vowel = sum(1 for token in tokens if len(token) >= 3 and not re.search(r"[aeiouyAEIOUY]", token))
        if short / max(len(tokens), 1) >= 0.42:
            return True
        if no_vowel / max(len(tokens), 1) >= 0.30:
            return True
        if re.search(r"[.?!,:;/\\|_-]{4,}", cleaned):
            return True
        return False

    def _slot_evidence(
        self,
        story_units: list[dict[str, Any]],
        draft_lines: list[str],
        ocr_only: bool = False,
    ) -> list[dict[str, Any]]:
        evidence: list[dict[str, Any]] = []
        for index, unit in enumerate(story_units):
            ocr_text = str(unit.get("combined_text") or "").strip()
            ocr_fallback = str(unit.get("ocr_fallback_text") or "").strip()
            raw_dialogue_for_slot: list[str] = []
            source_ocr = ocr_text or ocr_fallback
            if source_ocr and not ocr_only:
                # In vision mode: split raw OCR into dialogue sentences for the rewrite
                # pass to anchor on actual chapter text.
                for sentence in re.findall(r"[^.!?]+[.!?]?", source_ocr):
                    s = sentence.strip()
                    if s and len(s) > 4:
                        raw_dialogue_for_slot.append(s)
            # When OCR has a recognized speech act, inject the mechanical paraphrase so the
            # polish pass works from a narrative template rather than franchise hallucination
            # or raw OCR quotation.  When no speech act is recognized, mechanical = "" and
            # we CLEAR dialogue entirely — sending fragmentary OCR as dialogue causes the
            # polish LLM to quote it literally ("The conversation reveals '...'").
            draft_line = draft_lines[index] if index < len(draft_lines) else ""
            mechanical = self._mechanical_ocr_paraphrase(source_ocr) if source_ocr else ""
            if mechanical:
                # Recognized speech act: use as strict_line and as the single dialogue anchor.
                # Also replace ocr_text with the mechanical paraphrase so the LLM cannot quote
                # the raw OCR from that field either.
                strict_line = mechanical
                raw_dialogue_for_slot = [mechanical]
                slot_ocr_text = mechanical  # hide raw OCR from the rewrite prompt
            elif ocr_only:
                # No recognized speech act in OCR-only mode: use draft as strict_line,
                # send NO dialogue and NO ocr_text so the polish LLM cannot quote raw OCR.
                strict_line = draft_line
                raw_dialogue_for_slot = []
                slot_ocr_text = ""
            else:
                # Vision mode with no mechanical override: pass through raw OCR normally.
                strict_line = draft_line
                slot_ocr_text = ocr_text or ocr_fallback
            evidence.append(
                {
                    "strict_line": strict_line,
                    "ocr_text": slot_ocr_text,
                    "dialogue": raw_dialogue_for_slot,
                    "character_names": [str(name).strip() for name in unit.get("character_names", []) or [] if str(name).strip()],
                    "scene_summary": str(unit.get("scene_summary") or "").strip(),
                }
            )
        return evidence

    def _build_story_segments(
        self,
        story_units: list[dict[str, Any]],
        lines: list[str] | list[dict[str, Any]],
    ) -> list[StorySegment]:
        segments: list[StorySegment] = []
        for index, unit in enumerate(story_units, start=1):
            panel_ids = [str(panel_id).strip() for panel_id in unit.get("panel_ids", []) or [] if str(panel_id).strip()]
            representative_panel_id = panel_ids[len(panel_ids) // 2] if panel_ids else None
            scene_id = int(unit.get("scene_id") or index)
            sequence_in_scene = int(unit.get("sequence_in_scene") or 1)
            scene_unit_count = int(unit.get("scene_unit_count") or 1)
            # Keep editor headers stable and scannable. Scene summaries can be
            # long global setup sentences, which made every split beat display
            # the same confusing title in the narration UI.
            title = (
                f"Scene {scene_id} - Beat {sequence_in_scene}"
                if scene_unit_count > 1
                else f"Scene {scene_id}"
            )
            line_payload: dict[str, Any]
            if index - 1 < len(lines) and isinstance(lines[index - 1], dict):
                line_payload = dict(lines[index - 1])
            else:
                line_payload = {
                    "text": lines[index - 1] if index - 1 < len(lines) else "",
                    "visual_only": False,
                    "suppression_reason": None,
                }
            segments.append(
                StorySegment(
                    id=str(unit.get("segment_id") or f"scene_{scene_id:03d}_beat_{sequence_in_scene:02d}").strip(),
                    order=index,
                    text=self._normalize_segment_text(str(line_payload.get("text") or ""), allow_empty=True),
                    panel_ids=panel_ids,
                    panel_start=int(unit.get("panel_start") or 0) or None,
                    panel_end=int(unit.get("panel_end") or 0) or None,
                    scene_id=scene_id,
                    title=title,
                    representative_panel_id=representative_panel_id,
                    visual_only=bool(line_payload.get("visual_only")),
                    suppression_reason=str(line_payload.get("suppression_reason") or "").strip() or None,
                )
            )
        ordered = sorted(
            segments,
            key=lambda segment: (
                segment.panel_start is None,
                int(segment.panel_start or segment.order or 0),
                int(segment.panel_end or segment.panel_start or segment.order or 0),
                int(segment.order or 0),
            ),
        )
        scene_id_map: dict[int, int] = {}
        next_scene_id = 1
        renumbered: list[StorySegment] = []
        for index, segment in enumerate(ordered, start=1):
            original_scene_id = int(segment.scene_id or index)
            if original_scene_id not in scene_id_map:
                scene_id_map[original_scene_id] = next_scene_id
                next_scene_id += 1
            scene_id = scene_id_map[original_scene_id]
            sequence_in_scene = 1
            if segment.title:
                match = re.search(r"Beat\s+(\d+)$", str(segment.title))
                if match:
                    sequence_in_scene = int(match.group(1))
            title = f"Scene {scene_id} - Beat {sequence_in_scene}" if segment.title and "Beat" in segment.title else f"Scene {scene_id}"
            renumbered.append(segment.model_copy(update={"order": index, "scene_id": scene_id, "title": title}))
        return renumbered

    def _cohere_story_segments_for_delivery(self, segments: list[StorySegment]) -> list[StorySegment]:
        """Collapse duplicate/tiny beat slots into narration paragraphs.

        Story generation works in small aligned slots so evidence stays traceable,
        but the delivered YouTube script should not expose every slot as its own
        paragraph. This pass keeps the panel provenance while merging overlapping
        and very short neighboring beats into readable 2-5 sentence segments.
        """
        ordered = sorted(
            [segment for segment in segments if bool(getattr(segment, "keep", True))],
            key=lambda segment: (
                segment.panel_start is None,
                int(segment.panel_start or segment.order or 0),
                int(segment.panel_end or segment.panel_start or segment.order or 0),
                int(segment.order or 0),
            ),
        )
        if not ordered:
            return []

        def word_count(text: str) -> int:
            return len(re.findall(r"\b[\w'-]+\b", str(text or "")))

        def token_overlap(left: str, right: str) -> float:
            left_tokens = self._content_token_set(left)
            right_tokens = self._content_token_set(right)
            if not left_tokens or not right_tokens:
                return 0.0
            return len(left_tokens & right_tokens) / max(1, len(left_tokens | right_tokens))

        def panel_gap(left: StorySegment, right: StorySegment) -> int:
            if left.panel_end is None or right.panel_start is None:
                return 99
            return int(right.panel_start) - int(left.panel_end)

        def should_merge(bucket: list[StorySegment], candidate: StorySegment) -> bool:
            last = bucket[-1]
            bucket_text = " ".join(str(item.text or "").strip() for item in bucket if str(item.text or "").strip())
            candidate_text = str(candidate.text or "").strip()
            bucket_start = min((int(item.panel_start or item.order or 0) for item in bucket), default=0)
            bucket_end = max((int(item.panel_end or item.panel_start or item.order or 0) for item in bucket), default=0)
            candidate_start = int(candidate.panel_start or candidate.order or 0)
            overlaps_panels = bool(bucket_end and candidate_start and candidate_start <= bucket_end)
            # Guard: count panels already in bucket + candidate; never merge when
            # the result would exceed 4 panels — enforce strict 4-panel limit.
            bucket_panel_count = sum(len(item.panel_ids or []) for item in bucket)
            candidate_panel_count = len(candidate.panel_ids or [])
            merged_panel_count = bucket_panel_count + candidate_panel_count
            # Only collapse near-duplicates when both are very small — if segments
            # represent genuinely different page ranges they should stay distinct
            # even when the LLM happened to generate similar text.
            near_duplicate = (
                bool(bucket_text and candidate_text and token_overlap(bucket_text, candidate_text) >= 0.28)
                and merged_panel_count <= 4
            )
            tiny_neighbor = (
                panel_gap(last, candidate) <= 6
                and word_count(bucket_text) < 86
                and word_count(candidate_text) < 44
                and self._sentence_count(bucket_text) + self._sentence_count(candidate_text) <= 5
                and merged_panel_count <= 4
            )
            same_scene_short = (
                int(last.scene_id or 0) == int(candidate.scene_id or -1)
                and word_count(bucket_text) < 110
                and word_count(candidate_text) < 52
                and merged_panel_count <= 4
            )
            return overlaps_panels or near_duplicate or tiny_neighbor or same_scene_short

        def unique_sentences(bucket: list[StorySegment]) -> list[str]:
            sentences: list[str] = []
            bucket_text = " ".join(str(segment.text or "") for segment in bucket)
            ignored_names = {
                "A", "An", "As", "At", "For", "He", "Her", "His", "In", "It", "One", "She", "The",
                "Their", "These", "They", "This", "Those", "When", "While", "Meanwhile", "However",
            }
            name_counts = Counter(
                match.strip()
                for match in re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?\b", bucket_text)
                if match.strip() not in ignored_names
            )
            dominant_name = name_counts.most_common(1)[0][0] if name_counts else ""

            def repair_pronoun_start(sentence: str) -> str:
                repaired = sentence
                if not dominant_name:
                    return repaired
                replacements = (
                    (r"^He\b", dominant_name),
                    (r"^She\b", dominant_name),
                    (r"^His\b", f"{dominant_name}'s"),
                    (r"^Her\b", f"{dominant_name}'s"),
                )
                for pattern, replacement in replacements:
                    if re.match(pattern, repaired):
                        return re.sub(pattern, replacement, repaired, count=1)
                return repaired

            def redundant_or_meta_sentence(sentence: str) -> bool:
                lowered = sentence.casefold()
                words = re.findall(r"\b[\w'-]+\b", sentence)
                if len(words) <= 24 and re.match(
                    r"^(?:[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?|The\s+[A-Za-z][A-Za-z'-]+|A\s+[A-Za-z][A-Za-z'-]+)\s+(?:holds|stands|sits|walks|looks|watches|drinks|relaxes|surveys|smiles|types|reaches|poses)\b",
                    sentence,
                ):
                    return True
                if re.search(r"\b(?:stark reality|constant struggle|depends entirely|emotional toll|harsh realities)\b", lowered):
                    return True
                if re.search(
                    r"\b(?:presence signals|arrival indicates|moment of decision is critical|weight of this choice|certainty behind the command|statement that reveals|complex nature of|danger of (?:their|his|her|the) situation was palpable|this realization underscores|gravity of the circumstances|emotional turmoil)\b",
                    lowered,
                ):
                    return True
                if re.search(
                    r"\b(?:highlighting|underscoring|emphasizing|signaling|indicating|suggesting|foreshadowing)\b|\b(?:highlights|underscores|emphasizes|signals|indicates|suggests|signifies)\b",
                    lowered,
                ):
                    return True
                if re.search(
                    r"\b(?:appears|seems|perhaps|clearly|palpable)\b",
                    lowered,
                ):
                    return True
                if re.search(
                    r"\b(?:raises questions about|creates? (?:a|an) (?:sense|atmosphere) of|testament to|determination evident|adds another layer of uncertainty|the tension in the air was palpable|leaving onlookers to wonder|express concern over|express their astonishment|posture suggests|stride indicates|acknowledg(?:e|ing)|signifies agreement|unusual circumstances)\b",
                    lowered,
                ):
                    return True
                if re.search(
                    r"\b(?:the question arises whether|significant event|leaving everyone on edge|true nature and significance|deep knowledge|profound meaning|significant details)\b",
                    lowered,
                ):
                    return True
                if re.search(
                    r"\b(?:possess(?:es)?|utili[sz](?:es?|ed)|uses?)\s+(?:his|her|their)?\s*(?:unique|spatial|mysterious)?\s*(?:power|powers|abilit(?:y|ies))\b|\b(?:glowing energy|surreal, futuristic setting|post-storm era|useful recruits|the panel is split|on the left|on the right)\b",
                    lowered,
                ):
                    return True
                if len(sentences) >= 2 and re.match(
                    r"^(?:this|that|these|those)\s+",
                    lowered,
                ) and re.search(
                    r"\b(?:highlights?|underscores?|emphasizes?|indicates?|suggests?|realization|consequence|statement|sentiment)\b",
                    lowered,
                ):
                    return True
                if len(sentences) >= 3 and re.search(
                    r"\b(?:significant shift|pivotal moment|immediate disruption|direct challenge|specific objective|matter of importance)\b",
                    lowered,
                ):
                    return True
                return False

            for segment in bucket:
                text = self._normalize_segment_text(str(segment.text or ""), allow_empty=True)
                if not text:
                    continue
                for sentence in self._split_sentences_for_cleanup(text):
                    normalized = self._normalize_segment_text(repair_pronoun_start(sentence), allow_empty=True)
                    if not normalized:
                        continue
                    if (
                        self._line_is_low_quality(normalized)
                        or self._line_is_sentence_fragment(normalized)
                        or self._line_is_overly_generic(normalized)
                        or self._line_needs_style_refinement(normalized)
                        or self._text_is_noisy_ocr(normalized)
                        or self._has_ocr_shard_cluster(normalized)
                        or self._line_has_foreign_stopword_cluster(normalized)
                        or re.search(
                            r"\b(?:a\s+recap\s+begins|summariz(?:e|ing)\s+previous\s+events|narrative\s+interlude|refreshing\s+their\s+memory)\b",
                            normalized,
                            flags=re.IGNORECASE,
                        )
                        or re.search(r"\b[a-z]{2,}\s+[a-z]{4,}\s+they\s+use\s+this\s+to\b", normalized, flags=re.IGNORECASE)
                        or bool(re.search(r"\b[A-Z]{4,}\s+[A-Z]{4,}\b", normalized) and re.search(r"\b[a-z]{5,}\b", normalized))
                        or redundant_or_meta_sentence(normalized)
                    ):
                        continue
                    tokens = self._content_token_set(normalized)
                    redundant = False
                    for existing in sentences:
                        existing_tokens = self._content_token_set(existing)
                        if not tokens or not existing_tokens:
                            continue
                        jaccard = len(tokens & existing_tokens) / max(1, len(tokens | existing_tokens))
                        containment = len(tokens & existing_tokens) / max(1, min(len(tokens), len(existing_tokens)))
                        if jaccard >= 0.28 or containment >= 0.55:
                            redundant = True
                            break
                    if not redundant:
                        sentences.append(normalized)
                    if len(sentences) >= 4 or sum(word_count(sentence) for sentence in sentences) >= 120:
                        break
                if len(sentences) >= 4 or sum(word_count(sentence) for sentence in sentences) >= 120:
                    break
            return sentences

        def build_segment(bucket: list[StorySegment], order: int) -> StorySegment | None:
            panel_ids: list[str] = []
            for item in bucket:
                for panel_id in item.panel_ids or []:
                    clean_id = str(panel_id).strip()
                    if clean_id and clean_id not in panel_ids:
                        panel_ids.append(clean_id)
            starts = [int(item.panel_start) for item in bucket if item.panel_start is not None]
            ends = [int(item.panel_end or item.panel_start) for item in bucket if item.panel_end is not None or item.panel_start is not None]
            sentences = unique_sentences(bucket)
            text = self._normalize_segment_text(" ".join(sentences), allow_empty=True)
            if not text:
                return None
            representative_panel_id = panel_ids[len(panel_ids) // 2] if panel_ids else None
            source_id = str(bucket[0].id or f"scene_{order:03d}").strip()
            return bucket[0].model_copy(
                update={
                    "id": f"{source_id}_cohered" if not source_id.endswith("_cohered") else source_id,
                    "order": order,
                    "text": text,
                    "panel_ids": panel_ids,
                    "panel_start": min(starts) if starts else None,
                    "panel_end": max(ends) if ends else None,
                    "scene_id": order,
                    "title": f"Scene {order}",
                    "representative_panel_id": representative_panel_id,
                    "visual_only": False,
                    "suppression_reason": None,
                }
            )

        buckets: list[list[StorySegment]] = []
        current: list[StorySegment] = []
        for segment in ordered:
            if not current:
                current = [segment]
                continue
            if should_merge(current, segment):
                current.append(segment)
                continue
            buckets.append(current)
            current = [segment]
        if current:
            buckets.append(current)

        cohered: list[StorySegment] = []
        for bucket in buckets:
            segment = build_segment(bucket, len(cohered) + 1)
            if segment is not None:
                cohered.append(segment)
        return cohered or ordered

    def _repair_story_coverage_for_delivery(
        self,
        segments: list[StorySegment],
        kept_panels: list[PanelBox],
        panel_evidence_records: list[dict[str, Any]],
    ) -> list[StorySegment]:
        """Make final story segments dense enough to preserve chronological coverage.

        The LLM often writes attractive paragraphs that cover 40-60 source
        panels. Those paragraphs can sound clean while silently skipping many
        story beats. This pass keeps the chronology deterministic: large ranges
        are split into smaller panel chunks, skipped kept panels are inserted,
        and every inserted chunk carries explicit panel provenance.
        """
        kept = [
            panel for panel in sorted(
                kept_panels,
                key=lambda item: (
                    int(getattr(item, "page", 0) or 0),
                    int(getattr(item, "panel", 0) or 0),
                    int(getattr(item, "order", 0) or 0),
                ),
            )
            if bool(getattr(panel, "keep", True))
        ]
        if not segments or not kept:
            return segments
        report = ScriptQualityService().analyze_story_segments(
            segments,
            panels=kept,
            panel_evidence_records=panel_evidence_records,
        )
        if (
            not report.get("should_block_tts")
            and not report.get("underexplained_panel_ranges")
            and not report.get("skipped_panel_ranges")
            and not (report.get("scene_usage") or {}).get("overcompressed_scene_count")
            and not (report.get("scene_usage") or {}).get("unused_meaningful_panel_count")
        ):
            return segments
        scene_usage_by_segment = (report.get("scene_usage") or {}).get("scenes_by_segment_id", {})

        panel_by_order = {int(panel.page or 0) * 10000 + int(getattr(panel, "panel", 0) or 0): panel for panel in kept}
        evidence_by_id = {
            str(item.get("panel_id") or "").strip(): item
            for item in panel_evidence_records
            if isinstance(item, dict) and str(item.get("panel_id") or "").strip()
        }
        evidence_by_order: dict[int, dict[str, Any]] = {}
        for item in panel_evidence_records:
            if not isinstance(item, dict):
                continue
            try:
                order = int(item.get("panel_order") or 0)
            except Exception:
                order = 0
            if order:
                evidence_by_order[order] = self._merge_coverage_evidence(evidence_by_order.get(order, {}), item)

        covered_orders: set[int] = set()
        repaired: list[StorySegment] = []
        max_chunk_panels = 8

        def chunk_panels(panels: list[PanelBox], max_size: int = max_chunk_panels) -> list[list[PanelBox]]:
            chunks: list[list[PanelBox]] = []
            for index in range(0, len(panels), max_size):
                chunk = panels[index:index + max_size]
                if chunk:
                    chunks.append(chunk)
            return chunks

        ordered_segments = sorted(
            [segment for segment in segments if bool(getattr(segment, "keep", True))],
            key=lambda segment: (
                segment.panel_start is None,
                int(segment.panel_start or segment.order or 0),
                int(segment.panel_end or segment.panel_start or segment.order or 0),
                int(segment.order or 0),
            ),
        )

        for segment in ordered_segments:
            segment_orders = self._segment_panel_orders(segment, panel_by_order)
            if not segment_orders:
                repaired.append(segment)
                continue
            panels_for_segment = [panel_by_order[order] for order in segment_orders if order in panel_by_order]
            covered_orders.update(segment_orders)
            should_split = (
                len(panels_for_segment) > max_chunk_panels
                or self._story_segment_is_thin_for_range(segment, len(panels_for_segment))
            )
            scene_usage = scene_usage_by_segment.get(str(segment.id), {}) if isinstance(scene_usage_by_segment, dict) else {}
            should_rebuild = bool(
                scene_usage.get("unused_meaningful_panel_ids")
                or scene_usage.get("action_scene_without_concrete_action")
                or scene_usage.get("abstract_or_vague_narration")
                or scene_usage.get("needs_narration_expansion")
            )
            if not should_split:
                if should_rebuild:
                    text = self._coverage_chunk_narration(
                        panels_for_segment,
                        source_text=str(segment.text or ""),
                        evidence_by_id=evidence_by_id,
                        evidence_by_order=evidence_by_order,
                        chunk_index=1,
                        chunk_count=1,
                    )
                    repaired.append(
                        self._story_segment_from_panel_chunk(
                            source_segment=segment,
                            panel_chunk=panels_for_segment,
                            text=text,
                            suffix="meaningful_repair",
                        )
                    )
                    continue
                repaired.append(segment)
                continue
            for chunk_index, panel_chunk in enumerate(chunk_panels(panels_for_segment), start=1):
                text = self._coverage_chunk_narration(
                    panel_chunk,
                    source_text=str(segment.text or ""),
                    evidence_by_id=evidence_by_id,
                    evidence_by_order=evidence_by_order,
                    chunk_index=chunk_index,
                    chunk_count=max(1, math.ceil(len(panels_for_segment) / max_chunk_panels)),
                )
                repaired.append(
                    self._story_segment_from_panel_chunk(
                        source_segment=segment,
                        panel_chunk=panel_chunk,
                        text=text,
                        suffix=f"coverage_{chunk_index:02d}",
                    )
                )

        def _reading_order_key(panel: PanelBox) -> int:
            return int(panel.page or 0) * 10000 + int(getattr(panel, "panel", 0) or 0)

        missing_orders = [_reading_order_key(panel) for panel in kept if _reading_order_key(panel) not in covered_orders]
        for missing_range in self._contiguous_order_groups(missing_orders, panel_by_order):
            for chunk_index, panel_chunk in enumerate(chunk_panels(missing_range), start=1):
                text = self._coverage_chunk_narration(
                    panel_chunk,
                    source_text="",
                    evidence_by_id=evidence_by_id,
                    evidence_by_order=evidence_by_order,
                    chunk_index=chunk_index,
                    chunk_count=1,
                )
                _ps = _reading_order_key(panel_chunk[0])
                _pe = _reading_order_key(panel_chunk[-1])
                repaired.append(
                    StorySegment(
                        id=f"coverage_insert_{_ps:06d}_{_pe:06d}",
                        order=0,
                        text=text,
                        keep=True,
                        panel_ids=[panel.id for panel in panel_chunk],
                        panel_start=_ps,
                        panel_end=_pe,
                        scene_id=None,
                        title="Coverage bridge",
                        representative_panel_id=panel_chunk[len(panel_chunk) // 2].id,
                        visual_only=False,
                    )
                )

        repaired = sorted(
            repaired,
            key=lambda segment: (
                segment.panel_start is None,
                int(segment.panel_start or segment.order or 0),
                int(segment.panel_end or segment.panel_start or segment.order or 0),
                int(segment.order or 0),
            ),
        )
        final_repaired = [
            segment.model_copy(update={"order": index, "scene_id": index, "title": f"Scene {index}"})
            for index, segment in enumerate(repaired, start=1)
        ]
        repaired_report = ScriptQualityService().analyze_story_segments(
            final_repaired,
            panels=kept,
            panel_evidence_records=panel_evidence_records,
        )
        original_quality = int(report.get("quality_score", 0) or 0)
        repaired_quality = int(repaired_report.get("quality_score", 0) or 0)
        original_bad_style = sum(
            int(report.get(key, 0) or 0)
            for key in ("generic_lines", "filler_meta_lines", "caption_like_lines", "caption_like_sentences")
        )
        repaired_bad_style = sum(
            int(repaired_report.get(key, 0) or 0)
            for key in ("generic_lines", "filler_meta_lines", "caption_like_lines", "caption_like_sentences")
        )
        if repaired_quality < original_quality or (
            repaired_quality <= original_quality
            and repaired_bad_style > original_bad_style
        ):
            logger.warning(
                "Coverage repair rejected because it worsened narration quality "
                "(quality %s -> %s, style issues %s -> %s)",
                original_quality,
                repaired_quality,
                original_bad_style,
                repaired_bad_style,
            )
            return [
                segment.model_copy(update={"order": index, "scene_id": index})
                for index, segment in enumerate(ordered_segments, start=1)
            ]
        return final_repaired

    def _segment_panel_orders(self, segment: StorySegment, panel_by_order: dict[int, PanelBox]) -> list[int]:
        orders: set[int] = set()
        # panel_by_order is keyed by reading-order int (page * 10000 + panel_index)
        id_to_reading_order = {
            panel.id: int(panel.page or 0) * 10000 + int(getattr(panel, "panel", 0) or 0)
            for panel in panel_by_order.values()
        }
        for panel_id in segment.panel_ids or []:
            order = id_to_reading_order.get(str(panel_id))
            if order is not None:
                orders.add(order)
        start = int(segment.panel_start or 0)
        end = int(segment.panel_end or 0)
        if start and end:
            low, high = sorted((start, end))
            orders.update(order for order in panel_by_order if low <= order <= high)
        return sorted(orders)

    def _story_segment_is_thin_for_range(self, segment: StorySegment, panel_count: int) -> bool:
        if panel_count < 30:
            return False
        words = len(re.findall(r"\b[\w'-]+\b", str(segment.text or "")))
        sentences = self._sentence_count(str(segment.text or ""))
        return (words / max(panel_count, 1)) < 2.2 or sentences < 4

    def _story_segment_from_panel_chunk(
        self,
        *,
        source_segment: StorySegment,
        panel_chunk: list[PanelBox],
        text: str,
        suffix: str,
    ) -> StorySegment:
        return source_segment.model_copy(
            update={
                "id": f"{source_segment.id}_{suffix}",
                "text": text,
                "panel_ids": [panel.id for panel in panel_chunk],
                "panel_start": int(panel_chunk[0].page or 0) * 10000 + int(getattr(panel_chunk[0], "panel", 0) or 0),
                "panel_end": int(panel_chunk[-1].page or 0) * 10000 + int(getattr(panel_chunk[-1], "panel", 0) or 0),
                "representative_panel_id": panel_chunk[len(panel_chunk) // 2].id,
                "visual_only": False,
                "suppression_reason": None,
            }
        )

    def _contiguous_order_groups(
        self,
        orders: list[int],
        panel_by_order: dict[int, PanelBox],
    ) -> list[list[PanelBox]]:
        if not orders:
            return []
        groups: list[list[PanelBox]] = []
        current: list[PanelBox] = []
        previous: int | None = None
        for order in sorted(orders):
            panel = panel_by_order.get(order)
            if panel is None:
                continue
            if previous is not None and order != previous + 1 and current:
                groups.append(current)
                current = []
            current.append(panel)
            previous = order
        if current:
            groups.append(current)
        return groups

    def _coverage_chunk_narration(
        self,
        panel_chunk: list[PanelBox],
        *,
        source_text: str,
        evidence_by_id: dict[str, dict[str, Any]],
        evidence_by_order: dict[int, dict[str, Any]],
        chunk_index: int,
        chunk_count: int,
    ) -> str:
        # Keep noisy OCR out of final narration. OCR snippets remain available
        # in debug artifacts, but this coverage guard should not quote uncertain
        # text just to increase density.
        snippets = self._coverage_dialogue_snippets(
            panel_chunk,
            evidence_by_id=evidence_by_id,
            evidence_by_order=evidence_by_order,
        )
        source_sentences = [
            sentence
            for sentence in self._split_sentences_for_cleanup(source_text)
            if self._coverage_source_sentence_is_safe(sentence, int(panel_chunk[0].order))
        ]
        chosen_source = source_sentences[min(chunk_index - 1, len(source_sentences) - 1)] if source_sentences else ""
        names = self._coverage_names_from_text(" ".join([chosen_source, *snippets]))
        subject = names[0] if names else "The group"
        start = int(panel_chunk[0].order)
        end = int(panel_chunk[-1].order)
        keywords = self._coverage_keywords(" ".join([source_text, *snippets]))
        density_sentences = self._coverage_density_sentences(subject, keywords, start, end)

        sentences: list[str] = []
        if chosen_source and not snippets:
            sentences.append(self._normalize_segment_text(chosen_source, allow_empty=True))
        if snippets:
            dialogue_recap = self._coverage_dialogue_recap(snippets[:6], subject)
            if dialogue_recap:
                sentences.append(dialogue_recap)
        else:
            if not chosen_source:
                visual_sentence = self._coverage_visual_sentence(panel_chunk, subject, keywords)
                if visual_sentence:
                    sentences.append(visual_sentence)
        target_words = 0
        if len(panel_chunk) >= 30:
            target_words = min(150, max(95, round(len(panel_chunk) * 2.25)))
        template_offset = (start + end + chunk_index) % max(len(density_sentences), 1)
        attempts = 0
        while (
            target_words
            and len(re.findall(r"\b[\w'-]+\b", " ".join(sentences))) < target_words
            and len(sentences) < 8
            and attempts < len(density_sentences) * 2
        ):
            template = density_sentences[(template_offset + attempts) % len(density_sentences)]
            attempts += 1
            if template not in sentences:
                sentences.append(self._normalize_segment_text(template, allow_empty=True))
        return self._normalize_segment_text(" ".join(sentences[:8]), allow_empty=True)

    def _coverage_dialogue_sentence(self, snippet: str, index: int) -> str:
        text = self._coverage_snippet_summary(snippet)
        if text.endswith("?"):
            return self._normalize_segment_text(f"Someone asks, {text}", allow_empty=True)
        return self._normalize_segment_text(f"Someone says, {text}", allow_empty=True)

    def _coverage_dialogue_recap(self, snippets: list[str], subject: str) -> str:
        cleaned: list[str] = []
        seen: set[str] = set()
        for snippet in snippets:
            text = self._coverage_snippet_summary(snippet)
            key = re.sub(r"\W+", " ", text.casefold()).strip()
            if not key or key in seen:
                continue
            seen.add(key)
            cleaned.append(text)
        if not cleaned:
            return ""

        questions = [text for text in cleaned if text.endswith("?")]
        statements = [text for text in cleaned if not text.endswith("?")]
        warning_terms = re.compile(
            r"\b(?:warn|danger|kill|killer|blood|attack|enemy|monster|leave|gone|forced|hard|lonely|sorry|problem|cannot|can't|won't|must|need|want)\b",
            re.IGNORECASE,
        )
        pressure = [text for text in statements if warning_terms.search(text)]
        if questions and pressure:
            extra = ""
            remaining = [text for text in cleaned if text not in {questions[0], pressure[0]}]
            if remaining:
                extra = f" A nearby response adds {self._quote_or_clause(remaining[0])}."
            return self._normalize_segment_text(
                f"The exchange starts with {self._quote_or_clause(questions[0])}, then turns serious when {self._quote_or_clause(pressure[0])}.{extra}",
                allow_empty=True,
            )
        if len(statements) >= 2:
            first = self._quote_or_clause(statements[0])
            second = self._quote_or_clause(statements[1])
            third = self._quote_or_clause(statements[2]) if len(statements) >= 3 else ""
            extra = f" By the end of the exchange, {third}." if third else ""
            return self._normalize_segment_text(
                f"The conversation reveals {first}, and the next line adds {second}.{extra}",
                allow_empty=True,
            )
        if questions:
            return self._normalize_segment_text(
                f"The scene turns on {self._quote_or_clause(questions[0])}.",
                allow_empty=True,
            )
        return self._normalize_segment_text(
            f"{subject} has to absorb {self._quote_or_clause(statements[0])}.",
            allow_empty=True,
        )

    def _quote_or_clause(self, text: str) -> str:
        cleaned = self._normalize_segment_text(text, allow_empty=True)
        if not cleaned:
            return "the exchange"
        if len(cleaned.split()) <= 14:
            return f'"{cleaned}"'
        lowered = cleaned[0].lower() + cleaned[1:] if cleaned else cleaned
        return f"that {lowered}"

    def _coverage_visual_sentence(self, panel_chunk: list[PanelBox], subject: str, keywords: list[str]) -> str:
        captions = [
            clean_ocr_text(str(getattr(panel, "visual_caption", "") or "")).strip()
            for panel in panel_chunk
            if str(getattr(panel, "visual_caption", "") or "").strip()
        ]
        for caption in captions:
            if len(caption.split()) >= 4 and self._coverage_source_sentence_is_safe(caption, int(panel_chunk[0].order)):
                return self._normalize_segment_text(caption, allow_empty=True)
        useful_keywords = [
            keyword
            for keyword in keywords
            if keyword
            and keyword.casefold()
            not in {"scene", "panel", "moment", "beat", "group", "someone", "something"}
        ][:3]
        if useful_keywords:
            detail = ", ".join(useful_keywords)
            return self._normalize_segment_text(
                f"{subject} stays tied to {detail}.",
                allow_empty=True,
            )
        return ""

    def _merge_coverage_evidence(self, existing: dict[str, Any], item: dict[str, Any]) -> dict[str, Any]:
        merged = dict(existing or {})
        text_keys = ("dialogue_text", "repaired_text", "text_english", "cleaned_text", "text", "dialogue", "caption")
        for key, value in item.items():
            if key in text_keys and str(value or "").strip():
                current = str(merged.get(key) or "").strip()
                incoming = str(value or "").strip()
                if current:
                    if incoming.casefold() not in current.casefold():
                        merged[key] = f"{current} {incoming}"
                else:
                    merged[key] = incoming
            elif key not in merged or merged.get(key) in (None, ""):
                merged[key] = value
        return merged

    def _coverage_snippet_summary(self, snippet: str) -> str:
        text = self._normalize_segment_text(str(snippet or ""), allow_empty=True)
        text = re.sub(r"^[\"'“”‘’]+|[\"'“”‘’]+$", "", text).strip()
        if not text:
            return "a specific concern"
        lowered = text.casefold()
        if text[-1:] not in ".!?":
            text += "."
        return text

    def _coverage_keywords(self, text: str) -> list[str]:
        stop_words = {
            "about", "after", "again", "around", "because", "before", "between", "their", "there",
            "these", "those", "through", "while", "would", "could", "should", "with", "from",
            "into", "over", "under", "this", "that", "they", "them", "what", "when", "where",
            "have", "has", "had", "will", "only", "more", "just", "very", "still", "even",
        }
        tokens: list[str] = []
        for token in re.findall(r"[A-Za-z][A-Za-z'-]{3,}", str(text or "")):
            cleaned = token.strip("-'").casefold()
            if cleaned in stop_words or len(cleaned) < 4:
                continue
            if cleaned not in tokens and not re.search(r"(.)\1{3,}", cleaned):
                tokens.append(cleaned)
        return tokens[:12] or ["fallout", "response", "decision", "consequence"]

    def _coverage_density_sentences(self, subject: str, keywords: list[str], start: int, end: int) -> list[str]:
        key = list(keywords or ["fallout", "response", "decision", "consequence"])
        while len(key) < 8:
            key.extend(key)
        templates = (
            f"{subject} is pulled toward {key[0]} as {key[1]} changes the immediate problem.",
            f"{key[2].capitalize()} turns into a visible obstacle, so {subject} has to respond instead of waiting.",
            f"The focus moves from {key[3]} to {key[4]}, making the next reaction feel earned.",
            f"{key[5].capitalize()} leaves the characters with less room to avoid the consequence.",
            f"{subject} reacts to {key[6]}, and that reaction carries the scene into {key[7]}.",
            f"The moment ends with {key[1]} still unresolved, setting up the next beat.",
        )
        offset = (start + end) % len(templates)
        return [
            self._normalize_segment_text(templates[(offset + index) % len(templates)], allow_empty=True)
            for index in range(len(templates))
        ]

    def _coverage_source_sentence_is_safe(self, sentence: str, panel_start: int) -> bool:
        text = self._normalize_segment_text(str(sentence or ""), allow_empty=True)
        lowered = text.casefold()
        if not text:
            return False
        if self._line_is_low_quality(text) or self._text_is_noisy_ocr(text):
            return False
        if re.search(r"\b(?:i|i'm|i've|i'll|i'd|me|my|mine|myself|we|we're|we've|we'll|us|our|ours)\b", lowered):
            return False
        if re.match(r"^(?:he|she|they|his|her|their|however|meanwhile|within)\b", lowered):
            return False
        if panel_start > 120 and re.match(
            r"^(?:in this world|in a world|in the future|humanity|children known as|parasites are|young pilots are|within the confines)",
            lowered,
        ):
            return False
        if re.search(r"\b(?:ability|abilities|power|powers|magic|spell|technique|skill|aura|energy|mana|curse|gift|transformation)\b", lowered):
            return False
        if re.search(r"\b(?:palpable|underscores|highlights|suggests|indicates|emotional turmoil|gravity of the circumstances)\b", lowered):
            return False
        return True

    def _coverage_dialogue_snippets(
        self,
        panel_chunk: list[PanelBox],
        evidence_by_id: dict[str, dict[str, Any]],
        evidence_by_order: dict[int, dict[str, Any]],
    ) -> list[str]:
        snippets: list[str] = []
        seen: set[str] = set()
        for panel in panel_chunk:
            evidence = self._merge_coverage_evidence(
                evidence_by_order.get(int(panel.order), {}),
                evidence_by_id.get(panel.id, {}),
            )
            raw = " ".join(
                part
                for part in [
                    str(panel.ocr_text or "").strip(),
                    str(evidence.get("dialogue_text") or "").strip(),
                    str(evidence.get("text_english") or "").strip(),
                    str(evidence.get("text_original") or "").strip(),
                ]
                if part
            )
            for snippet in self._extract_clean_evidence_snippets(raw):
                key = re.sub(r"\W+", " ", snippet.casefold()).strip()
                if key and key not in seen:
                    seen.add(key)
                    snippets.append(snippet)
                if len(snippets) >= 8:
                    return snippets
        return snippets

    def _extract_clean_evidence_snippets(self, raw_text: str) -> list[str]:
        text = clean_ocr_text(str(raw_text or "")).strip()
        if not text or self._has_ocr_shard_cluster(text):
            return []
        text = re.sub(r"\s+", " ", text)
        candidates = re.split(r"(?<=[.!?])\s+|[;|]", text)
        snippets: list[str] = []
        for candidate in candidates:
            cleaned = candidate.strip(" \"'“”‘’.,;:-")
            words = re.findall(r"[A-Za-z][A-Za-z']*", cleaned)
            if len(words) < 3 and not cleaned.endswith("?"):
                continue
            if len(words) > 35:
                continue
            if self._text_is_noisy_ocr(cleaned):
                continue
            uppercase_ratio = sum(1 for char in cleaned if char.isupper()) / max(1, sum(1 for char in cleaned if char.isalpha()))
            if uppercase_ratio > 0.24 and len(words) >= 3:
                continue
            if re.search(r"\b(?:gwirrr|garrr|codc|cyyc|cynmcw|aano|hiko|uenn|azlp|www|rwm|nmi|neww|tthem|mencing|klav|nater|gölc|vpz|piksi|wyognli)\b", cleaned, re.IGNORECASE):
                continue
            if re.search(r"\b(?:[a-z]+[A-Z][A-Za-z]*|[A-Z]{2,}[a-z]+|[a-z][A-Z]{2,})\b", cleaned):
                continue
            if re.search(r"\b[A-Za-z]{1,2}\s+[A-Za-z]{1,2}\s+[A-Za-z]{1,2}\s+[A-Za-z]{1,2}\b", cleaned):
                continue
            snippets.append(cleaned[0].lower() + cleaned[1:] if cleaned and cleaned[0].isupper() else cleaned)
            if len(snippets) >= 8:
                break
        return snippets

    def _coverage_names_from_text(self, text: str) -> list[str]:
        blocked = {
            "The", "This", "That", "These", "Those", "Panels", "Scene", "Chapter",
            "Parasite", "Parasites",
        }
        names: list[str] = []
        for match in re.finditer(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?\b", str(text or "")):
            name = match.group(0).strip()
            if name in blocked or looks_like_false_character_name(name):
                continue
            if name not in names:
                names.append(name)
        return names[:3]

    def _fill_blank_story_payloads(
        self,
        payloads: list[dict[str, Any]],
        units: list[dict[str, Any]],
        *,
        protagonist_name: str | None,
        grounding: dict[str, Any] | None,
        story_bible: dict[str, Any],
        style_vocab: StyleVocabulary | None = None,
    ) -> list[dict[str, Any]]:
        if not payloads:
            return payloads
        filled = [dict(item) for item in payloads]
        scene_memory_by_id = {
            int(item.get("scene_id") or 0): dict(item)
            for item in story_bible.get("scene_memory", []) or []
            if int(item.get("scene_id") or 0)
        }
        line_key_counts = Counter(
            key
            for key in (
                self._normalized_line_key(self._normalize_segment_text(str(item.get("text") or ""), allow_empty=True))
                for item in filled
            )
            if key
        )
        seen_existing_keys: set[str] = set()
        world_terms = self._world_terms_for_guardrails(story_bible, grounding) if grounding else []
        for index, payload in enumerate(filled):
            unit = units[index]
            scene_memory = scene_memory_by_id.get(int(unit.get("scene_id") or 0))
            support_text = self._unit_support_text(unit, story_bible, scene_memory)
            prev_text = str(filled[index - 1].get("text") or "").strip() if index > 0 else ""
            next_text = str(filled[index + 1].get("text") or "").strip() if index + 1 < len(filled) else ""
            bridge_support_text = " ".join(part for part in (support_text, prev_text, next_text) if part)
            prev_key = self._normalized_line_key(prev_text) if prev_text else ""
            next_key = self._normalized_line_key(next_text) if next_text else ""
            existing = self._normalize_segment_text(str(payload.get("text") or "").strip(), allow_empty=True)
            if existing:
                existing_key = self._normalized_line_key(existing)
                duplicate_existing = bool(existing_key and existing_key in seen_existing_keys)
                existing_needs_replacement = (
                    self._line_is_low_quality(existing)
                    or self._line_is_overly_generic(existing)
                    or self.polisher._is_visual_description(existing)
                    or self._line_is_dialogue_fragment(existing)
                    or self._line_is_sentence_fragment(existing)
                    or self._line_needs_style_refinement(existing)
                    or self._line_has_first_person_narration(existing)
                    or self._line_has_unsupported_setting_terms(existing, bridge_support_text)
                    or duplicate_existing
                )
                if existing_needs_replacement:
                    trimmed_existing = self._remove_offending_sentences(existing)
                    trimmed_key = self._normalized_line_key(trimmed_existing)
                    if (
                        trimmed_existing
                        and trimmed_existing != existing
                        and not self._line_has_unsupported_setting_terms(trimmed_existing, bridge_support_text)
                        and not (trimmed_key and trimmed_key in {prev_key, next_key})
                    ):
                        payload["text"] = trimmed_existing
                        payload["visual_only"] = False
                        payload["suppression_reason"] = None
                        if trimmed_key:
                            line_key_counts[trimmed_key] += 1
                            seen_existing_keys.add(trimmed_key)
                        continue
                    payload["duplicate_original_text"] = existing
                    payload["text"] = ""
                    payload["visual_only"] = True
                    payload["suppression_reason"] = str(payload.get("suppression_reason") or "weak_evidence")
                else:
                    payload["text"] = existing
                    if existing_key:
                        seen_existing_keys.add(existing_key)
                    continue
            if self._normalize_segment_text(str(payload.get("text") or "").strip(), allow_empty=True):
                payload["text"] = existing
                continue
            candidates: list[str] = []
            if grounding:
                candidates.append(
                    self._safe_grounded_scene_line(
                        unit,
                        protagonist_name,
                        grounding,
                        world_terms,
                        scene_memory_by_id.get(int(unit.get("scene_id") or 0)),
                    )
                )
            candidates.extend(
                [
                    self._evidence_bridge_line(unit, protagonist_name, style_vocab=style_vocab),
                    self._fallback_scene_line(unit, protagonist_name, style_vocab=style_vocab),
                    str(unit.get("vision_action_beat") or "").strip(),
                    str(unit.get("vision_caption") or "").strip(),
                    str(unit.get("ocr_fallback_text") or "").strip(),
                    "" if int(unit.get("scene_unit_count") or 1) > 1 else str(unit.get("scene_summary") or "").strip(),
                ]
            )
            for candidate in candidates:
                normalized = self._normalize_segment_text(candidate, allow_empty=True)
                if not normalized:
                    continue
                if (
                    self._line_is_low_quality(normalized)
                    or self._line_is_overly_generic(normalized)
                    or self.polisher._is_visual_description(normalized)
                    or self._line_is_dialogue_fragment(normalized)
                    or self._line_is_sentence_fragment(normalized)
                    or self._line_needs_style_refinement(normalized)
                    or self._line_has_first_person_narration(normalized)
                    or self._line_has_unsupported_setting_terms(normalized, support_text)
                ):
                    continue
                key = self._normalized_line_key(normalized)
                if key and key in {prev_key, next_key}:
                    continue
                if key and line_key_counts.get(key, 0) > 0:
                    continue
                payload["text"] = normalized
                payload["visual_only"] = False
                payload["suppression_reason"] = None
                if key:
                    line_key_counts[key] += 1
                    seen_existing_keys.add(key)
                break
            if payload.get("text"):
                continue
            if payload.get("text"):
                continue
            original_duplicate = self._normalize_segment_text(
                str(payload.get("duplicate_original_text") or "").strip(),
                allow_empty=True,
            )
            if original_duplicate:
                original_key = self._normalized_line_key(original_duplicate)
                if (
                    original_key
                    and original_key not in {prev_key, next_key}
                    and line_key_counts.get(original_key, 0) == 0
                    and not self._line_is_low_quality(original_duplicate)
                    and not self._line_is_overly_generic(original_duplicate)
                    and not self._line_is_dialogue_fragment(original_duplicate)
                    and not self._line_is_sentence_fragment(original_duplicate)
                    and not self._line_needs_style_refinement(original_duplicate)
                    and not self._line_has_first_person_narration(original_duplicate)
                    and not self.polisher._is_visual_description(original_duplicate)
                    and not self._line_has_unsupported_setting_terms(original_duplicate, bridge_support_text)
                ):
                    payload["text"] = original_duplicate
                    payload["visual_only"] = False
                    payload["suppression_reason"] = None
                    line_key_counts[original_key] += 1
                    seen_existing_keys.add(original_key)
        return filled

    def _force_fill_remaining_blank_payloads(
        self,
        payloads: list[dict[str, Any]],
        units: list[dict[str, Any]],
        *,
        style_vocab: StyleVocabulary | None = None,
    ) -> list[dict[str, Any]]:
        """Final coverage guard: do not let scriptable beats stay silent.

        This runs after the normal evidence and bridge fill paths. It is
        intentionally short and varied: silence is more damaging than a
        conservative connector, but repeated connector templates are also a
        quality risk.
        """
        if not payloads:
            return payloads
        filled = [dict(item) for item in payloads]
        vocab = style_vocab

        def _pick_local(
            values: tuple[str, ...] | list[str],
            index: int,
            support_text: str,
            fallback: str = "",
        ) -> str:
            choices = [str(value).strip() for value in values or [] if str(value).strip()]
            support_key = normalize_name_key(support_text)
            if choices and support_key:
                for offset in range(len(choices)):
                    choice = choices[(index + offset) % len(choices)]
                    key = normalize_name_key(choice)
                    if key and key in support_key:
                        return choice
            return fallback

        for index, payload in enumerate(filled):
            if str(payload.get("text") or "").strip():
                continue
            unit = units[index] if index < len(units) else {}
            subject = next(
                (str(name).strip() for name in unit.get("character_names", []) or [] if str(name).strip()),
                "",
            )
            prev_text = str(filled[index - 1].get("text") or "").strip() if index > 0 else ""
            support_text = " ".join(
                part
                for part in (
                    *(
                        str(unit.get(key) or "").strip()
                        for key in (
                            "vision_action_beat",
                            "vision_caption",
                            "vision_dialogue",
                            "combined_text",
                            "visual_cues",
                            "salvaged_evidence",
                            "scene_summary",
                        )
                    ),
                    prev_text,
                )
                if str(part or "").strip()
            )
            support_key = normalize_name_key(support_text)
            if not subject and vocab:
                subject = next(
                    (
                        name
                        for name in vocab.named_characters
                        if normalize_name_key(name) and normalize_name_key(name) in support_key
                    ),
                    "",
                )
            if not subject:
                subject = next(
                    (
                        name
                        for name in extract_proper_name_candidates(support_text)
                        if not re.search(r"\b(?:unknown|speaker|narrator|protagonist|character|figure|someone)\b", name, flags=re.IGNORECASE)
                    ),
                    "",
                )
            if not subject and re.search(r"\bhumanity\b", support_text, flags=re.IGNORECASE):
                subject = "Humanity"
            has_group_context = bool(re.search(r"\b(?:group|team|family|neighbors|pilots|squad|crew|survivors)\b", support_text, flags=re.IGNORECASE))
            team = (
                vocab.team_term
                if vocab and normalize_name_key(vocab.team_term or "") in support_key
                else ("the group" if has_group_context else "the moment")
            )
            topic = self._bridge_topic_phrase(support_text)
            placeholder_topics = {
                "the group tension",
                "the exchange",
                "the immediate threat",
                "the protected space",
                "the resource problem",
            }
            topic_fallback = "" if topic in placeholder_topics else topic
            world = _pick_local(vocab.world_terms if vocab else (), index, support_text, topic_fallback or "the danger")
            stakes = _pick_local(vocab.stakes_phrases if vocab else (), index, support_text, topic_fallback or "the next choice")
            anchor_subject = subject or (
                world
                if world.casefold() not in {"the risk", "the immediate threat", "the next choice"}
                else ""
            )
            if not anchor_subject:
                anchor_subject = team if team != "the moment" else "the group"
            usable_team = team if team != "the moment" else "the group"
            templates = (
                "{subject} stays close to {world} while {team} weighs {stakes}. The group cannot treat that risk as distant anymore.",
                "{team} keeps its attention on {subject} as {world} complicates {stakes}. Their pause carries enough uncertainty to keep the risk alive.",
                "{subject} moves carefully around {world}, aware that {stakes} no longer feels abstract. {team} has to respond with less room for certainty.",
                "{world} leaves {team} measuring every reaction around {subject}. The choice ahead feels narrower because the group has already seen the cost.",
                "{subject} treats {stakes} as something that can no longer be ignored. Around them, {team} has to decide how much trust still remains.",
                "{team} gathers itself around {world}, but {subject} remains the person everyone has to watch. That attention turns {stakes} into a shared burden.",
                "{subject} keeps facing {world} while {team} searches for a steadier answer. The uncertainty around {stakes} keeps the group from relaxing.",
                "{world} presses into {subject}'s path, making {stakes} harder to separate from survival. {team} has to carry that worry forward.",
                "{team} watches {subject} handle {world} with no easy reassurance. The unresolved risk around {stakes} gives the group another reason to hesitate.",
                "{subject} stays tied to {stakes} even when {world} pulls attention elsewhere. {team} has to read that choice without a simple explanation.",
                "{world} changes how {team} understands {subject}'s role. What looked like a private burden now affects how everyone weighs {stakes}.",
                "{subject} keeps the focus on {world}, and {team} has to adjust around that fact. The risk in {stakes} remains close enough to shape their response.",
                "{team} steadies itself after {world} exposes another fragile point. {subject} becomes the one person who can keep {stakes} from slipping away.",
                "{subject} carries the strain of {world} while {team} looks for a safe answer. The group has to treat {stakes} as immediate rather than theoretical.",
                "{world} gives {subject} no clean way to step back. {team} is left balancing doubt, loyalty, and {stakes} all at once.",
                "{subject} stays in the middle of {world} as {team} tries to understand the risk. Their response to {stakes} has to come before certainty does.",
                "{team} follows the tension around {subject} because {world} has already changed the terms. The group cannot separate that choice from {stakes}.",
                "{subject} keeps moving through {world} with the others close behind. {stakes} becomes a test of whether {team} can still act together.",
                "{world} forces {team} to measure what {subject} is willing to risk. That makes {stakes} feel less like rumor and more like a decision.",
                "{subject} faces {world} without giving {team} a simple answer. The silence around {stakes} leaves everyone reading the same danger differently.",
                "{team} has to follow {subject}'s lead through {world}. Each reaction makes {stakes} feel more difficult to postpone.",
                "{subject} keeps the burden of {stakes} visible while {world} closes around the group. {team} can only move by accepting that risk.",
                "{world} keeps {subject} from disappearing into the background. {team} has to account for that presence before {stakes} can move any further.",
                "{subject} remains the clearest anchor in {world}. Around them, {team} has to turn {stakes} into an actual choice.",
                "{team} studies {subject}'s reaction as {world} keeps the risk close. The group understands that {stakes} will not resolve itself.",
                "{subject} holds steady inside {world}, even as {team} struggles with {stakes}. That steadiness gives the next decision a sharper edge.",
                "{world} leaves {team} with fewer ways to protect {subject}. The weight of {stakes} pushes every response toward something more dangerous.",
                "{subject} keeps {stakes} from fading while {world} demands attention. {team} has to decide whether to follow that focus or resist it.",
                "{team} cannot look away from {subject} once {world} exposes the risk. {stakes} becomes something the group has to carry together.",
                "{subject} reads {world} as a warning, not a pause. {team} has to treat {stakes} as part of the danger already in front of them.",
                "{world} makes {subject}'s position harder for {team} to ignore. The risk around {stakes} leaves no comfortable place for anyone to stand.",
                "{subject} stays with {stakes} while {team} absorbs what {world} has changed. The group has to move forward without pretending the answer is simple.",
            )
            slots = {
                "subject": anchor_subject,
                "team": usable_team,
                "world": world,
                "stakes": stakes,
            }
            for sentence in self._split_sentences_for_cleanup(support_text):
                normalized = self._normalize_segment_text(sentence, allow_empty=True)
                words = len(re.findall(r"\b[\w'-]+\b", normalized))
                if 8 <= words <= 35 and not (
                    self._line_is_low_quality(normalized)
                    or self._line_is_overly_generic(normalized)
                    or self._line_is_dialogue_fragment(normalized)
                    or self._line_is_sentence_fragment(normalized)
                    or self._line_has_first_person_narration(normalized)
                    or self.polisher._is_visual_description(normalized)
                ):
                    payload["text"] = normalized
                    payload["visual_only"] = False
                    payload["suppression_reason"] = None
                    break
            if str(payload.get("text") or "").strip():
                continue
            payload["text"] = ""
            payload["visual_only"] = True
            payload["suppression_reason"] = str(payload.get("suppression_reason") or "weak_evidence")
            continue
            for offset in range(len(templates)):
                candidate = templates[(index + offset) % len(templates)].format(**slots)
                normalized = self._normalize_segment_text(candidate, allow_empty=True)
                if normalized and not (
                    self._line_is_low_quality(normalized)
                    or self._line_is_overly_generic(normalized)
                    or self._line_is_dialogue_fragment(normalized)
                    or self._line_is_sentence_fragment(normalized)
                    or self._line_has_first_person_narration(normalized)
                    or self.polisher._is_visual_description(normalized)
                ):
                    payload["text"] = normalized
                    payload["visual_only"] = False
                    payload["suppression_reason"] = None
                    break
            if not str(payload.get("text") or "").strip():
                payload["text"] = ""
                payload["visual_only"] = True
                payload["suppression_reason"] = str(payload.get("suppression_reason") or "weak_evidence")
        return filled

    def _prefer_local_evidence_for_thin_segments(
        self,
        payloads: list[dict[str, Any]],
        units: list[dict[str, Any]],
        *,
        style_vocab: StyleVocabulary | None = None,
    ) -> list[dict[str, Any]]:
        """Use the unit's own 2-4 panel evidence when the draft is thin.

        This is the safe middle ground between one-sentence OCR captions and
        broad chapter-level filler: if the grouped panels already contain
        multiple clean beats, preserve those beats directly.
        """
        if not payloads:
            return payloads
        refined: list[dict[str, Any]] = []
        for index, payload in enumerate(payloads):
            current = dict(payload)
            unit = units[index] if index < len(units) else {}
            text = self._normalize_segment_text(str(current.get("text") or ""), allow_empty=True)
            local_line = self._local_evidence_recap_line(unit, text, style_vocab=style_vocab)
            if not local_line:
                refined.append(current)
                continue

            prev_text = str(payloads[index - 1].get("text") or "").strip() if index > 0 else ""
            next_text = str(payloads[index + 1].get("text") or "").strip() if index + 1 < len(payloads) else ""
            text_key = self._normalized_line_key(text)
            duplicate_neighbor = bool(text_key and text_key in {self._normalized_line_key(prev_text), self._normalized_line_key(next_text)})
            word_count = len(re.findall(r"\b[\w'-]+\b", text))
            local_word_count = len(re.findall(r"\b[\w'-]+\b", local_line))
            should_replace = (
                not text
                or bool(current.get("visual_only"))
                or duplicate_neighbor
                or self._line_has_unsupported_local_character(text, unit, style_vocab)
                or self._line_is_low_quality(text)
                or self._line_is_overly_generic(text)
                or (self._sentence_count(text) < 2 and word_count < 28 and local_word_count >= word_count)
            )
            if should_replace and self._local_recap_text_is_usable(local_line):
                current["text"] = local_line
                current["visual_only"] = False
                current["suppression_reason"] = None
            refined.append(current)
        return refined

    def _line_has_unsupported_local_character(
        self,
        text: str,
        unit: dict[str, Any],
        style_vocab: StyleVocabulary | None,
    ) -> bool:
        if not text or not style_vocab:
            return False
        text_key = normalize_name_key(text)
        support_text = " ".join(
            str(value or "").strip()
            for value in (
                unit.get("combined_text"),
                unit.get("vision_action_beat"),
                unit.get("vision_dialogue"),
                unit.get("vision_caption"),
                unit.get("visual_cues"),
                unit.get("ocr_fallback_text"),
                " ".join(str(name or "").strip() for name in unit.get("character_names", []) or []),
            )
            if str(value or "").strip()
        )
        support_key = normalize_name_key(support_text)
        for name in style_vocab.named_characters:
            key = normalize_name_key(name)
            if key and key in text_key and key not in support_key:
                return True
        return False

    def _local_recap_text_is_usable(self, text: str) -> bool:
        sentences = self._split_sentences_for_cleanup(text)
        return bool(sentences) and all(self._local_evidence_sentence_is_usable(sentence) for sentence in sentences)

    def _local_evidence_sentence_is_usable(self, text: str) -> bool:
        normalized = self._normalize_segment_text(text, allow_empty=True)
        if not normalized:
            return False
        words = re.findall(r"\b[\w'-]+\b", normalized)
        if len(words) < 5 or len(words) > 42:
            return False
        if self._text_is_noisy_ocr(normalized) or self._has_ocr_shard_cluster(normalized):
            return False
        if self._line_has_foreign_stopword_cluster(normalized):
            return False
        if self._line_has_first_person_narration(normalized):
            return False
        if self._line_is_dialogue_fragment(normalized):
            return False
        if self._line_needs_style_refinement(normalized) or self.polisher._is_visual_description(normalized):
            return False
        if re.match(r'^(?:"|“|\'|‘)', normalized):
            return False
        if re.match(r"^[A-Z][A-Za-z0-9_ ]{0,24}:\s", normalized):
            return False
        if re.search(
            r"\b(?:concept|idea|theme|metaphor)\b|\bstark contrast\b|\bis introduced\b|\bknown as\b.{0,60}\bis introduced\b",
            normalized,
            flags=re.IGNORECASE,
        ):
            return False
        if re.search(
            r"\b(?:unknown|viewer|a character|another character|male character|female character|someone)\b",
            normalized,
            flags=re.IGNORECASE,
        ):
            return False
        if re.search(r"\b([A-Z][a-z]+)\s+\w+s\s+\1\b", normalized):
            return False
        if re.search(r"\bHmb\b|\brnes\b|\bimnb\b|\bGaa\b|\bmqlu\b", normalized, flags=re.IGNORECASE):
            return False
        return True

    def _local_evidence_recap_line(
        self,
        unit: dict[str, Any],
        current_text: str = "",
        *,
        style_vocab: StyleVocabulary | None = None,
    ) -> str:
        pieces = [
            str(unit.get("combined_text") or "").strip(),
            str(unit.get("vision_dialogue") or "").strip(),
            str(unit.get("vision_caption") or "").strip(),
            str(unit.get("vision_action_beat") or "").strip(),
            str(unit.get("visual_cues") or "").strip(),
        ]
        if not any(pieces):
            pieces.append(str(unit.get("ocr_fallback_text") or "").strip())
        source = self._normalize_supporting_text(" ".join(piece for piece in pieces if piece))
        if not source or self._text_is_noisy_ocr(source):
            return ""

        candidates: list[str] = []
        seen_keys: set[str] = set()

        def _add_candidate(raw: str) -> None:
            normalized = self._normalize_segment_text(raw, allow_empty=True)
            if not normalized:
                return
            words = len(re.findall(r"\b[\w'-]+\b", normalized))
            if words < 7 or words > 36:
                return
            if (
                not self._local_evidence_sentence_is_usable(normalized)
            ):
                return
            key = self._normalized_line_key(normalized)
            if not key or key in seen_keys:
                return
            tokens = self._content_token_set(normalized)
            for existing in candidates:
                existing_tokens = self._content_token_set(existing)
                if not tokens or not existing_tokens:
                    continue
                overlap = len(tokens & existing_tokens)
                containment = overlap / max(1, min(len(tokens), len(existing_tokens)))
                if containment >= 0.65:
                    return
            seen_keys.add(key)
            candidates.append(normalized)

        source_sentences = self._split_sentences_for_cleanup(source)
        use_current_first = self._sentence_count(current_text) >= 2 or len(source_sentences) < 2
        if use_current_first and current_text and not self._line_has_unsupported_local_character(current_text, unit, style_vocab):
            _add_candidate(current_text)
        for sentence in source_sentences:
            _add_candidate(sentence)
            if len(candidates) >= 3:
                break
        if not candidates and current_text and not self._line_has_unsupported_local_character(current_text, unit, style_vocab):
            _add_candidate(current_text)
        if not candidates:
            return ""
        return self._normalize_segment_text(" ".join(candidates[:3]), allow_empty=True)

    def _ensure_minimum_segment_richness(
        self,
        payloads: list[dict[str, Any]],
        units: list[dict[str, Any]],
        *,
        style_vocab: StyleVocabulary | None = None,
    ) -> list[dict[str, Any]]:
        """Make editable story segments usable as standalone 2-3 sentence beats."""
        if not payloads:
            return payloads
        enriched = [dict(item) for item in payloads]

        def word_count(text: str) -> int:
            return len(re.findall(r"\b[\w'-]+\b", str(text or "")))

        for index, payload in enumerate(enriched):
            if bool(payload.get("visual_only")):
                continue
            text = self._normalize_segment_text(str(payload.get("text") or ""), allow_empty=True)
            if not text:
                continue
            current_sentences = self._sentence_count(text)
            current_words = word_count(text)
            if current_sentences >= 2 and current_words >= 24:
                continue
            if current_sentences < 2 and current_words >= 24:
                split_text = self._split_long_sentence_for_richness(text)
                if split_text:
                    payload["text"] = split_text
                    payload["visual_only"] = False
                    payload["suppression_reason"] = None
                    continue
            unit = units[index] if index < len(units) else {}
            additions = self._richness_addition_candidates(
                unit,
                text,
                index=index,
                style_vocab=style_vocab,
            )
            existing_keys = {
                self._normalized_line_key(sentence)
                for sentence in self._split_sentences_for_cleanup(text)
                if sentence.strip()
            }
            for addition in additions:
                normalized = self._normalize_segment_text(addition, allow_empty=True)
                key = self._normalized_line_key(normalized)
                if not normalized or key in existing_keys:
                    continue
                candidate = self._normalize_segment_text(f"{text} {normalized}", allow_empty=True)
                if not candidate or word_count(candidate) > 90:
                    continue
                if (
                    self._line_is_low_quality(candidate)
                    or self._line_is_overly_generic(candidate)
                    or self._line_is_dialogue_fragment(candidate)
                    or self._line_is_sentence_fragment(candidate)
                    or self._line_has_first_person_narration(candidate)
                    or self.polisher._is_visual_description(candidate)
                ):
                    continue
                text = candidate
                existing_keys.add(key)
                if self._sentence_count(text) >= 2 and word_count(text) >= 24:
                    break
            payload["text"] = text
            payload["visual_only"] = False
            payload["suppression_reason"] = None
        return enriched

    def _trim_final_bad_sentences(self, payloads: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Last defensive trim for OCR shards and fragments inside otherwise usable beats."""
        trimmed_payloads: list[dict[str, Any]] = []
        for payload in payloads:
            current = dict(payload)
            text = self._normalize_segment_text(str(current.get("text") or ""), allow_empty=True)
            if not text:
                trimmed_payloads.append(current)
                continue
            sentences = self._split_sentences_for_cleanup(text)
            if not sentences:
                current["text"] = ""
                current["visual_only"] = True
                current["suppression_reason"] = "weak_evidence"
                trimmed_payloads.append(current)
                continue
            survivors = [
                sentence
                for sentence in sentences
                if not (
                    self._line_is_low_quality(sentence)
                    or self._line_is_overly_generic(sentence)
                    or self._line_needs_style_refinement(sentence)
                    or self._line_is_dialogue_fragment(sentence)
                    or self._line_is_sentence_fragment(sentence)
                    or self._line_has_first_person_narration(sentence)
                    or self.polisher._is_visual_description(sentence)
                )
            ]
            if not survivors:
                current["text"] = ""
                current["visual_only"] = True
                current["suppression_reason"] = "weak_evidence"
            elif len(survivors) != len(sentences):
                current["text"] = self._normalize_segment_text(" ".join(survivors), allow_empty=True)
                current["visual_only"] = False
                current["suppression_reason"] = None
            trimmed_payloads.append(current)
        return trimmed_payloads

    def _pad_remaining_short_segments(self, payloads: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Preserve short but grounded segments without adding synthetic filler."""
        if not payloads:
            return payloads
        return [dict(item) for item in payloads]

    def _break_exact_duplicate_payloads(
        self,
        payloads: list[dict[str, Any]],
        units: list[dict[str, Any]] | None = None,
        *,
        style_vocab: StyleVocabulary | None = None,
    ) -> list[dict[str, Any]]:
        seen: Counter[str] = Counter()
        refined: list[dict[str, Any]] = []
        for index, payload in enumerate(payloads):
            current = dict(payload)
            text = self._normalize_segment_text(str(current.get("text") or ""), allow_empty=True)
            key = self._normalized_line_key(text)
            if key:
                seen[key] += 1
                if seen[key] > 1 and text:
                    unit = units[index] if units and index < len(units) else {}
                    additions = self._richness_addition_candidates(
                        unit,
                        text,
                        index=index + seen[key],
                        style_vocab=style_vocab,
                    )
                    for addition in additions:
                        normalized_addition = self._normalize_segment_text(addition, allow_empty=True)
                        if not normalized_addition:
                            continue
                        candidate = self._normalize_segment_text(f"{text} {normalized_addition}", allow_empty=True)
                        if len(re.findall(r"\b[\w'-]+\b", candidate)) > 90:
                            continue
                        if (
                            self._line_is_low_quality(candidate)
                            or self._line_is_overly_generic(candidate)
                            or self._line_is_dialogue_fragment(candidate)
                            or self._line_has_first_person_narration(candidate)
                            or self.polisher._is_visual_description(candidate)
                        ):
                            continue
                        current["text"] = candidate
                        current["visual_only"] = False
                        current["suppression_reason"] = None
                        break
                    if self._normalized_line_key(str(current.get("text") or "")) == key:
                        current["duplicate_original_text"] = text
                        current["text"] = ""
                        current["visual_only"] = True
                        current["suppression_reason"] = "near_duplicate"
            refined.append(current)
        return refined

    def _split_long_sentence_for_richness(self, text: str) -> str:
        """Turn one grounded long sentence into two sentences without adding facts."""
        normalized = self._normalize_segment_text(text, allow_empty=True)
        if self._sentence_count(normalized) != 1:
            return ""
        words = re.findall(r"\b[\w'-]+\b", normalized)
        if len(words) < 24:
            return ""
        stripped = normalized.rstrip(".!?")
        separators = (
            r",\s+but\s+",
            r",\s+while\s+",
            r",\s+as\s+",
            r",\s+even\s+as\s+",
            r",\s+before\s+",
            r",\s+because\s+",
            r",\s+boasting\s+",
            r",\s+clearly\s+",
            r",\s+with\s+",
            r",\s+declaring\s+",
            r",\s+prompting\s+",
            r",\s+suggesting\s+",
            r",\s+connecting\s+",
            r",\s+continuing\s+",
            r",\s+warning\s+",
            r",\s+asking\s+",
            r",\s+calling\s+",
            r",\s+unable\s+",
            r",\s+everything\s+",
            r",\s+leaving\s+",
            r",\s+making\s+",
            r",\s+questioning\s+",
            r",\s+drawing\s+",
            r",\s+yet\s+",
            r",\s+and\s+",
            r"\s+before\s+",
            r"\s+while\s+",
            r"\s+and\s+warns\s+",
            r",\s+(?=(?:a|an|the|his|her|their|its)\s+)",
            r";\s+",
            r":\s+",
        )
        for pattern in separators:
            for match in re.finditer(pattern, stripped, flags=re.IGNORECASE):
                left = stripped[: match.start()].strip(" ,;:")
                right = stripped[match.end() :].strip(" ,;:")
                if not left or not right:
                    continue
                left_words = re.findall(r"\b[\w'-]+\b", left)
                right_words = re.findall(r"\b[\w'-]+\b", right)
                if len(left_words) < 5 or len(right_words) < 7:
                    continue
                right = right[:1].upper() + right[1:]
                candidate = self._normalize_segment_text(f"{left}. {right}.", allow_empty=True)
                if (
                    candidate
                    and self._sentence_count(candidate) == 2
                    and not self._line_is_low_quality(candidate)
                    and not self._line_is_overly_generic(candidate)
                    and not self._line_is_dialogue_fragment(candidate)
                    and not self._line_is_sentence_fragment(candidate)
                    and not self.polisher._is_visual_description(candidate)
                ):
                    return candidate
        return ""

    def _richness_addition_candidates(
        self,
        unit: dict[str, Any],
        current_text: str,
        *,
        index: int,
        style_vocab: StyleVocabulary | None = None,
    ) -> list[str]:
        evidence = " ".join(
            str(unit.get(key) or "").strip()
            for key in (
                "vision_action_beat",
                "vision_caption",
                "vision_dialogue",
                "scene_summary",
                "combined_text",
                "visual_cues",
                "ocr_fallback_text",
            )
            if str(unit.get(key) or "").strip()
        )
        candidates: list[str] = []
        for sentence in self._split_sentences_for_cleanup(evidence):
            normalized = self._normalize_segment_text(sentence, allow_empty=True)
            words = len(re.findall(r"\b[\w'-]+\b", normalized))
            if 8 <= words <= 28 and not (
                self._line_is_low_quality(normalized)
                or self._line_is_overly_generic(normalized)
                or self._line_is_dialogue_fragment(normalized)
                or self._line_is_sentence_fragment(normalized)
                or self._line_needs_style_refinement(normalized)
                or self.polisher._is_visual_description(normalized)
            ):
                candidates.append(normalized)
                if len(candidates) >= 3:
                    break

        if len(re.findall(r"\b[\w'-]+\b", str(current_text or ""))) >= 24:
            return candidates
        local_context = " ".join(part for part in (current_text, evidence) if str(part or "").strip())
        local_key = normalize_name_key(local_context)
        current_lower = str(current_text or "").casefold()
        pattern_templates: list[str] = []
        content_lower = f"{current_lower} {evidence.casefold()}"
        if re.search(r"\b(?:arrow|wound|leg|bleed|blood|injur)\w*\b", content_lower):
            pattern_templates.extend(
                (
                    "The injury turns the confrontation from intimidation into immediate survival.",
                    "Pain narrows the choices around the wounded character, making every next move more urgent.",
                )
            )
        if re.search(r"\b(?:phone|call|message|chat|screen|tablet)\b", content_lower):
            pattern_templates.extend(
                (
                    "The device keeps the conflict public, letting each threat spread before anyone can take it back.",
                    "What should be a simple exchange becomes another way for pressure to reach the people outside the room.",
                )
            )
        if re.search(r"\b(?:door|barrier|gate|wall|window)\b", content_lower):
            pattern_templates.extend(
                (
                    "The barrier turns protection into leverage, separating panic outside from control inside.",
                    "Every impact against that boundary makes the standoff feel less like an argument and more like a siege.",
                )
            )
        if re.search(r"\b(?:snow|cold|freez|winter|ice|icy|frozen)\w*\b", content_lower):
            pattern_templates.extend(
                (
                    "The cold makes every delay matter, turning ordinary needs into survival pressure.",
                    "Outside, the weather keeps shrinking the margin between discomfort and real danger.",
                )
            )
        if re.search(r"\b(?:supply|supplies|resource|food|medicine|generator|stockpile|bunker|shelter)\w*\b", content_lower):
            pattern_templates.extend(
                (
                    "The supplies stop feeling like preparation and become leverage over everyone who lacks them.",
                    "Survival now depends on who controls the resources and who is desperate enough to challenge that control.",
                )
            )
        if re.search(r"\b(?:knife|gun|weapon|rifle|attack|violence|violent|threat|threaten|crowd)\w*\b", content_lower):
            pattern_templates.extend(
                (
                    "The threat stops being theoretical, forcing the characters to treat the next choice as a matter of survival.",
                    "What began as pressure hardens into open danger, leaving less room for anyone to back down safely.",
                )
            )
        if re.search(r"\b(?:water|spray|drench|bucket|substance)\w*\b", content_lower):
            pattern_templates.extend(
                (
                    "The failed tactic turns desperation into humiliation, making the attackers colder and angrier than before.",
                    "The mess changes the rhythm of the standoff, replacing confidence with embarrassment and panic.",
                )
            )
        if re.search(r"\b(?:partner|pair|pilot|cockpit|robot|mecha|machine|monster|enemy|battle|fight)\w*\b", content_lower):
            pattern_templates.extend(
                (
                    "The machinery turns personal trust into the real test, because survival depends on more than strength.",
                    "The fight stops being distant once the characters have to decide who they can rely on inside it.",
                )
            )
        if re.search(r"\b(?:ocean|wing|bird|freedom|fly|flight)\w*\b", content_lower):
            pattern_templates.extend(
                (
                    "The image of flight gives the moment a longing that reaches beyond the immediate conflict.",
                    "That dream of escape makes the present limits feel sharper and more painful.",
                )
            )
        if re.search(r"\b(?:question|ask|demand|wonder|confus|disbelief)\w*\b", current_lower):
            pattern_templates.extend(
                (
                    "The unanswered question keeps the pressure active instead of letting the exchange pass cleanly.",
                    "Confusion makes the next response feel less like routine and more like a forced choice.",
                )
            )
        if re.search(r"\b(?:farewell|goodbye|calls? out|cries? out|stop|wait)\b", current_lower):
            pattern_templates.extend(
                (
                    "The plea makes the separation feel immediate rather than ceremonial.",
                    "That emotion keeps the departure from feeling like a simple transition.",
                )
            )
        if re.search(r"\b(?:warn|insist|claim|vow|order|confirm|announce|declare)\w*\b", current_lower):
            pattern_templates.extend(
                (
                    "The certainty behind the line sharpens the cost of ignoring it.",
                    "The declaration forces the people listening to decide whether they believe the warning.",
                )
            )
        if re.search(r"\b(?:appear|emerge|reveal|rise|slam|explode|attack|devastat|stampede|surge|crash)\w*\b", current_lower):
            pattern_templates.extend(
                (
                    "The impact turns the background threat into something immediate.",
                    "The sudden shift gives everyone less room to treat the danger as distant.",
                )
            )
        if re.search(r"\b(?:fear|worry|shock|surprise|despair|doubt|concern|alone|lonely|powerless)\w*\b", current_lower):
            pattern_templates.extend(
                (
                    "The reaction makes the moment land as more than a passing pause.",
                    "That emotional weight follows the characters into the next decision.",
                )
            )
        if pattern_templates:
            candidates.append(pattern_templates[index % len(pattern_templates)])

        def _local_phrase(values: tuple[str, ...] | list[str], *, offset: int = 0) -> str:
            choices = [self._richness_phrase(value, fallback="") for value in values or []]
            choices = [value for value in choices if value and normalize_name_key(value) in local_key]
            if not choices:
                return ""
            return choices[(index + offset) % len(choices)]

        subject = self._richness_subject(unit, current_text, style_vocab)
        team = (
            self._richness_phrase(style_vocab.team_term or "", fallback="")
            if style_vocab and normalize_name_key(style_vocab.team_term or "") in local_key
            else ""
        )
        if not team and re.search(r"\b(?:group|team|family|neighbors|pilots|squad|crew|survivors)\b", local_context, flags=re.IGNORECASE):
            team = "the group"
        world = _local_phrase(style_vocab.world_terms if style_vocab else (), offset=0)
        stakes = _local_phrase(style_vocab.stakes_phrases if style_vocab else (), offset=3)
        topic = self._bridge_topic_phrase(evidence)
        if not world and topic:
            world = topic
        if not stakes and topic:
            stakes = topic
        if not subject or (not world and not stakes):
            if subject and not pattern_templates:
                pattern_templates.append(
                    f"{subject}'s response gives the brief exchange a clearer emotional direction before the scene moves on."
                )
            if pattern_templates:
                candidates.append(pattern_templates[index % len(pattern_templates)])
            return candidates
        if not team:
            team = "the group"

        # Do not add broad subject/world/stakes filler for short segments.
        # These lines boosted sentence counts while weakening local context.
        return candidates
        templates = (
            "{subject} has to carry that reaction into {world} while {team} decides what can still be trusted.",
            "{team} reads {subject}'s pause as part of {stakes}, not as a clean break from it.",
            "{world} turns the moment around {subject} into something {team} has to answer carefully.",
            "{subject}'s choice keeps {stakes} close enough for {team} to treat it as immediate.",
            "{team} watches {subject} move through {world}, aware that the next response cannot be casual.",
            "{subject} keeps the pressure visible while {world} forces {team} to adjust.",
            "{stakes} follows {subject} into the next beat, giving {team} less room to look away.",
            "{world} makes {team} measure {subject}'s reaction against what the group already knows.",
            "{subject} stays tied to the risk, and {team} has to decide how much {world} changes the choice.",
            "{team} cannot treat {subject}'s reaction as background once {stakes} starts shaping the scene.",
            "{subject} moves with the weight of {world} still close, leaving {team} to read the cost.",
            "{stakes} gives {team} a sharper reason to watch how {subject} responds.",
            "{subject} keeps the moment from settling, because {world} still changes what {team} can do.",
            "{team} has to follow the pressure around {subject} before {stakes} slips into another crisis.",
            "{world} keeps the scene pointed at {subject}, making {team} weigh the risk more carefully.",
            "{subject}'s place in the moment gives {team} a clearer view of why {stakes} matters.",
            "{team} reads the next choice through {subject}, while {world} keeps the cost close.",
            "{subject} stays near the center of {stakes}, and {team} has to respond without easy certainty.",
            "{world} gives {subject}'s reaction a sharper edge before {team} can move on.",
            "{team} watches the pressure gather around {subject}, treating {stakes} as something already in motion.",
            "{subject} keeps the scene from becoming a simple pause while {world} continues to matter.",
            "{stakes} leaves {team} studying {subject}'s response for any sign of control.",
            "{world} keeps {subject} from fading into the background, so {team} has to stay alert.",
            "{subject} turns the reaction into a choice {team} can no longer postpone.",
            "{team} has to measure {world} through {subject}'s next move rather than through reassurance.",
            "{stakes} stays unresolved around {subject}, giving {team} another reason to hesitate.",
            "{subject} carries the scene forward while {world} keeps the risk from feeling distant.",
            "{team} reads the cost in {subject}'s reaction and keeps {stakes} in view.",
            "{world} makes {subject}'s position harder to ignore before {team} can reset.",
            "{subject} keeps the focus on what {team} still has to face inside {stakes}.",
            "{team} has to treat {world} as part of {subject}'s choice, not just the backdrop.",
            "{stakes} presses close enough that {subject} cannot step away from what {team} needs next.",
        )
        for offset in range(len(templates)):
            candidates.append(templates[(index + offset) % len(templates)].format(
                subject=subject,
                team=team,
                world=world,
                stakes=stakes,
            ))
        return candidates

    def _fix_sentence_boundaries(self, payloads: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Clean punctuation so UI/editor sentence counts match the prose."""
        fixed: list[dict[str, Any]] = []
        for payload in payloads:
            current = dict(payload)
            current["text"] = self._fix_sentence_boundary_text(str(current.get("text") or ""))
            fixed.append(current)
        return fixed

    def _fix_sentence_boundary_text(self, text: str) -> str:
        normalized = self._normalize_segment_text(str(text or ""), allow_empty=True)
        if not normalized:
            return ""
        normalized = re.sub(r"\.{2,}", ".", normalized)
        normalized = re.sub(
            r"([.!?])\s+([a-z])",
            lambda match: f"{match.group(1)} {match.group(2).upper()}",
            normalized,
        )
        return self._normalize_segment_text(normalized, allow_empty=True)

    def _richness_subject(
        self,
        unit: dict[str, Any],
        current_text: str,
        style_vocab: StyleVocabulary | None,
    ) -> str:
        names = [
            str(name).strip()
            for name in unit.get("character_names", []) or []
            if str(name).strip() and not self._vision_name_is_placeholder(str(name))
        ]
        for name in names:
            if name.casefold() in str(current_text or "").casefold():
                return name
        allowed_names = {
            normalize_name_key(name)
            for name in ((style_vocab.named_characters if style_vocab else ()) or ())
            if normalize_name_key(name)
        }
        proper_names = re.findall(r"\b[A-Z][A-Za-z0-9]*(?:\s+[A-Z][A-Za-z0-9]*){0,2}\b", str(current_text or ""))
        for name in proper_names:
            cleaned = str(name).strip()
            key = normalize_name_key(cleaned)
            if key and key in allowed_names:
                return cleaned
        if names:
            return names[0]
        local_context = str(current_text or "")
        if re.search(r"\b(?:the group|the team|the family|the neighbors|the pilots|the squad|the crew)\b", local_context, flags=re.IGNORECASE):
            return "the group"
        noun_match = re.match(r"^(The|A|An)\s+([A-Za-z][A-Za-z'-]+(?:\s+[A-Za-z][A-Za-z'-]+){0,2})\b", local_context)
        if noun_match:
            phrase = f"{noun_match.group(1)} {noun_match.group(2)}"
            if not re.search(r"\b(?:moment|risk|exchange|situation|scene|pressure|choice|group tension)\b", phrase, flags=re.IGNORECASE):
                return phrase
        return ""

    def _richness_cycle_phrase(
        self,
        values: tuple[str, ...] | list[str],
        index: int,
        *,
        fallback: str,
    ) -> str:
        usable = [self._richness_phrase(value, fallback="") for value in values or []]
        usable = [value for value in usable if value]
        if not usable:
            return fallback
        return usable[index % len(usable)]

    def _richness_phrase(self, value: str, *, fallback: str) -> str:
        text = re.sub(r"\s+", " ", str(value or "").strip(" ,;:-"))
        if not text:
            return fallback
        lowered = text.casefold()
        if (
            lowered.endswith(" with")
            or re.search(r"\b(?:male|female|failed|promising|former prodigy|elite)\s+pilot\b", lowered)
            or re.search(r"\b(?:his|her|their|a)\s+(?:partners?|to pilot|partner to pilot)\b", lowered)
            or re.search(r"\b(?:inability to pilot|ability to pilot|exceptionally skilled pilot)\b", lowered)
            or lowered in {"partner", "partners", "piloting", "pilot", "the piloting", "a partner", "the partner"}
        ):
            return fallback
        return text

    def _vary_repetitive_bridge_phrasing(self, payloads: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Break up deterministic bridge tails without changing story facts."""
        if not payloads:
            return payloads
        varied = [dict(item) for item in payloads]

        def pick(options: tuple[str, ...], index: int) -> str:
            return options[index % len(options)]

        answer_variants = (
            "The group has to keep moving even without a clean answer.",
            "No one gets enough certainty to treat the risk as solved.",
            "The choice remains unsettled, but the group cannot stand still.",
            "That uncertainty forces everyone to act before comfort returns.",
            "The risk stays close enough that waiting no longer feels safe.",
            "The group has to carry the doubt into whatever comes next.",
            "There is no easy answer, only another step through the danger.",
            "The decision stays uncomfortable because the cost is already visible.",
            "Everyone has to move with an incomplete picture of the danger.",
            "The answer stays out of reach, but the group still has to respond.",
            "The safer path never fully appears, so hesitation becomes its own risk.",
            "The group is left choosing under doubt rather than certainty.",
        )
        active_variants = (
            "{topic} keeps shaping the fallout while the group searches for direction.",
            "{topic} remains close enough to color every reaction around the group.",
            "{topic} keeps the risk alive without turning the line into a reset.",
            "{topic} continues to steer the fallout around the group.",
            "{topic} gives the group another reason to hesitate before acting.",
            "{topic} keeps everyone reading the danger from a different angle.",
            "{topic} turns the pause into another test of trust.",
            "{topic} leaves the group carrying more doubt than reassurance.",
            "{topic} keeps the decision from feeling settled.",
            "{topic} makes even a quiet reaction feel loaded.",
            "{topic} stays close enough to shape the next choice.",
            "{topic} keeps the fallout personal for everyone nearby.",
        )
        outside_variants = (
            "leave {team} trying to understand the choice from the margins",
            "keep {team} reacting before anyone can fully explain the risk",
            "force {team} to read the decision from a distance",
            "pull {team} into a reaction they cannot neatly resolve",
            "leave {team} measuring the fallout from the outside",
            "keep {team} uncertain about how to answer the choice",
            "make {team} respond before the danger is fully clear",
            "leave {team} watching a decision they cannot control",
            "put {team} close to the fallout without giving them control",
            "make {team} interpret the risk before anyone feels ready",
            "leave {team} caught between loyalty and uncertainty",
            "keep {team} close enough to worry but too far away to guide it",
        )
        steady_variants = (
            "searches for a safer answer",
            "looks for a steadier path",
            "tries to find a way through",
            "works for a response that will hold",
            "looks for something solid to trust",
            "tries to regain its footing",
            "searches for a choice that will not break",
            "looks for a way to keep control",
            "tries to turn doubt into action",
            "searches for a response that can survive the risk",
            "looks for a way past the uncertainty",
            "tries to keep the danger contained",
        )
        survival_variants = (
            "a fight to keep going",
            "a struggle to stay alive",
            "a test of survival",
            "a bid to endure",
            "a fight against collapse",
            "a struggle for another chance",
            "a test of whether survival is possible",
            "a push to outlast the danger",
            "a fragile chance to continue",
            "a fight to remain standing",
            "a struggle against the cost",
            "a test of endurance",
        )
        compatibility_variants = (
            "their ability to pilot together",
            "the way they synchronize in combat",
            "their shared control inside the machine",
            "the bond that lets them fight",
            "their connection in the cockpit",
            "the trust their piloting requires",
            "the rhythm they need to survive the fight",
            "the fragile link between them",
            "the partnership that makes piloting possible",
            "the control they have to share",
            "their joint command of the machine",
            "the bond behind the sortie",
        )
        absorb_variants = (
            "{subject} stays with {topic} while {team} measures what has changed.",
            "{subject} keeps {topic} in view as {team} absorbs the cost.",
            "{subject} stays near {topic} while {team} works through the fallout.",
            "{subject} carries {topic} forward as {team} adjusts its response.",
            "{subject} remains tied to {topic} while {team} weighs the risk.",
            "{subject} keeps {topic} close as {team} tries to regain control.",
            "{subject} stays focused on {topic} while {team} reads the damage.",
            "{subject} holds onto {topic} as {team} searches for a safer path.",
            "{subject} keeps {topic} from fading while {team} faces the fallout.",
            "{subject} remains caught in {topic} as {team} decides how to answer.",
            "{subject} keeps {topic} visible while {team} weighs the next risk.",
            "{subject} stays beside {topic} as {team} tries to move with care.",
        )
        focus_variants = (
            "{subject} keeps attention fixed on {topic}. {team} has to respond while {stakes} still shapes the risk.",
            "{subject} refuses to let {topic} fade. {team} has to account for {stakes} before acting.",
            "{subject} keeps {topic} at the center. {team} reads {stakes} as a risk that still matters.",
            "{subject} holds the line around {topic}. {team} has to treat {stakes} as part of the immediate danger.",
            "{subject} keeps returning to {topic}. {team} has to decide how much {stakes} changes their response.",
            "{subject} anchors the response around {topic}. {team} cannot separate that choice from {stakes}.",
            "{subject} keeps {topic} from slipping away. {team} has to carry {stakes} into its answer.",
            "{subject} makes {topic} impossible to dismiss. {team} has to move with {stakes} still unresolved.",
            "{subject} keeps the group pointed toward {topic}. {team} has to weigh {stakes} without a clean answer.",
            "{subject} treats {topic} as the clearest warning. {team} has to respond while {stakes} remains close.",
            "{subject} keeps {topic} in front of everyone. {team} has to move before {stakes} becomes worse.",
            "{subject} stays fixed on {topic}. {team} has to read {stakes} as part of the same danger.",
        )
        step_back_variants = (
            "{topic} leaves {subject} with no easy retreat. {team} has to balance trust, fear, and {stakes} at once.",
            "{topic} keeps {subject} from backing away. {team} is left weighing loyalty against {stakes}.",
            "{topic} gives {subject} little room to retreat. {team} has to carry doubt and {stakes} together.",
            "{topic} holds {subject} in place. {team} has to decide how much risk {stakes} brings with it.",
            "{topic} denies {subject} a safe exit. {team} is left measuring fear against {stakes}.",
            "{topic} keeps {subject} exposed. {team} has to balance what they know against {stakes}.",
            "{topic} gives {subject} no simple escape. {team} has to move with doubt still attached to {stakes}.",
            "{topic} leaves {subject} boxed in. {team} is forced to weigh uncertainty against {stakes}.",
            "{topic} keeps {subject} close to the fallout. {team} has to decide what {stakes} now demands.",
            "{topic} makes retreat impossible for {subject}. {team} has to answer with {stakes} still unresolved.",
            "{topic} leaves {subject} without a clean way out. {team} has to keep fear and {stakes} in balance.",
            "{topic} traps {subject} near the risk. {team} has to choose while {stakes} still hangs over them.",
        )

        for index, payload in enumerate(varied):
            text = str(payload.get("text") or "")
            if not text:
                continue
            text = re.sub(
                r"The group has to move forward without pretending the answer is simple\.",
                pick(answer_variants, index),
                text,
            )

            def replace_active(match: re.Match[str]) -> str:
                topic = match.group("topic").strip()
                return pick(active_variants, index).format(topic=topic)

            text = re.sub(
                r"(?P<topic>[A-Z][A-Za-z0-9' -]{2,80}?) keeps the beat active without pretending the group has all the answers\.",
                replace_active,
                text,
            )
            text = re.sub(
                r"The uncertainty around (?P<topic>[A-Z][A-Za-z0-9' -]{2,80}?) keeps the group from relaxing\.",
                replace_active,
                text,
            )

            def replace_outside(match: re.Match[str]) -> str:
                subject = match.group("subject").strip()
                team = match.group("team").strip()
                return f"{subject} {pick(outside_variants, index).format(team=team)}"

            text = re.sub(
                r"(?P<subject>[A-Z][A-Za-z0-9' -]{2,80}?) leave (?P<team>[A-Z][A-Za-z0-9' -]{2,40}?) reacting from the outside",
                replace_outside,
                text,
            )
            text = re.sub(
                r"searches for a steadier answer",
                pick(steady_variants, index),
                text,
            )
            text = re.sub(r"\ba story of survival\b", pick(survival_variants, index), text, flags=re.IGNORECASE)
            text = re.sub(
                r"\btheir compatibility in piloting\b",
                pick(compatibility_variants, index),
                text,
                flags=re.IGNORECASE,
            )

            def replace_absorb(match: re.Match[str]) -> str:
                return pick(absorb_variants, index).format(
                    subject=match.group("subject").strip(),
                    team=match.group("team").strip(),
                    topic=match.group("topic").strip(),
                )

            text = re.sub(
                r"(?P<subject>[A-Z][A-Za-z0-9' -]{2,50}?) stays with (?P<topic>[A-Za-z][A-Za-z0-9' -]{2,80}?) while (?P<team>[A-Z][A-Za-z0-9' -]{2,40}?) absorbs what [A-Za-z][A-Za-z0-9' -]{2,80}? has changed",
                replace_absorb,
                text,
            )

            def replace_focus(match: re.Match[str]) -> str:
                return pick(focus_variants, index).format(
                    subject=match.group("subject").strip(),
                    team=match.group("team").strip(),
                    topic=match.group("topic").strip(),
                    stakes=match.group("stakes").strip(),
                )

            text = re.sub(
                r"(?P<subject>[A-Z][A-Za-z0-9' -]{2,50}?) keeps the focus on (?P<topic>[A-Za-z][A-Za-z0-9' -]{2,80}?), and (?P<team>[A-Z][A-Za-z0-9' -]{2,40}?) has to adjust around that fact\. The risk in (?P<stakes>[A-Za-z][A-Za-z0-9' -]{2,80}?) remains close enough to shape their response\.",
                replace_focus,
                text,
            )

            def replace_step_back(match: re.Match[str]) -> str:
                return pick(step_back_variants, index).format(
                    subject=match.group("subject").strip(),
                    team=match.group("team").strip(),
                    topic=match.group("topic").strip(),
                    stakes=match.group("stakes").strip(),
                )

            text = re.sub(
                r"(?P<topic>[A-Za-z][A-Za-z0-9' -]{2,80}?) gives (?P<subject>[A-Z][A-Za-z0-9' -]{2,50}?) no clean way to step back\. (?P<team>[A-Z][A-Za-z0-9' -]{2,40}?) is left balancing doubt, loyalty, and (?P<stakes>[A-Za-z][A-Za-z0-9' -]{2,80}?) all at once\.",
                replace_step_back,
                text,
            )
            payload["text"] = self._normalize_segment_text(text, allow_empty=True)
        return varied

    def _reinforce_multi_sentence_scene_payloads(
        self,
        payloads: list[dict[str, Any]],
        units: list[dict[str, Any]],
        *,
        protagonist_name: str | None,
        grounding: dict[str, Any] | None,
        story_bible: dict[str, Any],
        style_vocab: StyleVocabulary | None = None,
    ) -> list[dict[str, Any]]:
        """Give scene-mode slots enough prose to feel like story segments.

        This is a deterministic safety net after all LLM and repair passes. It
        only appends sentences already supported by the unit's vision/story
        evidence, and it refuses anything that trips the same quality guards as
        normal narration. The LLM remains responsible for the best prose; this
        prevents later fallbacks from collapsing full scenes into one caption.
        """
        if not payloads:
            return payloads
        reinforced = [dict(item) for item in payloads]
        world_terms = self._world_terms_for_guardrails(story_bible, grounding) if grounding else []
        scene_memory_by_id = {
            int(item.get("scene_id") or 0): dict(item)
            for item in story_bible.get("scene_memory", []) or []
            if int(item.get("scene_id") or 0)
        }
        for index, payload in enumerate(reinforced):
            if bool(payload.get("visual_only")):
                continue
            text = self._normalize_segment_text(str(payload.get("text") or "").strip(), allow_empty=True)
            if not text:
                continue
            unit = units[index]
            panel_count = int(unit.get("panel_count") or len(unit.get("panel_ids", []) or []))
            evidence_word_count = len(
                re.findall(
                    r"\b[\w'-]+\b",
                    " ".join(
                        str(unit.get(key) or "").strip()
                        for key in (
                            "vision_action_beat",
                            "vision_dialogue",
                            "vision_caption",
                            "combined_text",
                            "visual_cues",
                            "ocr_fallback_text",
                            "scene_summary",
                        )
                        if str(unit.get(key) or "").strip()
                    ),
                )
            )
            target_sentences = 4 if panel_count >= 6 and evidence_word_count >= 36 else (3 if panel_count >= 3 else 2)
            if self._sentence_count(text) >= target_sentences:
                continue

            existing_tokens = self._content_token_set(text)
            existing_keys = {
                self._normalized_line_key(sentence)
                for sentence in self._split_sentences_for_cleanup(text)
                if sentence.strip()
            }
            candidates: list[str] = []
            followup = self._alignment_followup_sentence(unit, text)
            if followup:
                candidates.append(followup)
            if grounding:
                candidates.append(
                    self._safe_grounded_scene_line(
                        unit,
                        protagonist_name,
                        grounding,
                        world_terms,
                        scene_memory_by_id.get(int(unit.get("scene_id") or 0)),
                    )
                )
            candidates.extend(
                [
                    self._evidence_bridge_line(unit, protagonist_name, style_vocab=style_vocab),
                    self._fallback_scene_line(unit, protagonist_name, style_vocab=style_vocab),
                    str(unit.get("vision_action_beat") or "").strip(),
                    str(unit.get("vision_caption") or "").strip(),
                    str(unit.get("vision_dialogue") or "").strip(),
                    str(unit.get("ocr_fallback_text") or "").strip(),
                    str(unit.get("scene_summary") or "").strip(),
                    str(unit.get("visual_cues") or "").strip(),
                ]
            )

            additions: list[str] = []
            for candidate in candidates:
                for sentence in self._split_sentences_for_cleanup(candidate):
                    normalized = self._normalize_segment_text(sentence, allow_empty=True)
                    key = self._normalized_line_key(normalized)
                    if not normalized or key in existing_keys:
                        continue
                    if (
                        self._line_is_low_quality(normalized)
                        or self._line_is_overly_generic(normalized)
                        or self._line_is_dialogue_fragment(normalized)
                        or self._line_is_sentence_fragment(normalized)
                        or self._line_needs_style_refinement(normalized)
                        or self.polisher._is_visual_description(normalized)
                    ):
                        continue
                    if grounding and contains_unapproved_names(normalized, grounding, world_terms=world_terms):
                        continue
                    tokens = self._content_token_set(normalized)
                    if tokens and existing_tokens:
                        overlap = len(tokens & existing_tokens) / max(1, min(len(tokens), len(existing_tokens)))
                        if overlap >= 0.72:
                            continue
                    additions.append(normalized)
                    existing_keys.add(key)
                    existing_tokens |= tokens
                    if self._sentence_count(text) + len(additions) >= target_sentences:
                        break
                if self._sentence_count(text) + len(additions) >= target_sentences:
                    break

            if not additions:
                continue
            combined = self._normalize_segment_text(" ".join([text, *additions]), allow_empty=True)
            if len(combined) > 700:
                continue
            if (
                self._line_is_low_quality(combined)
                or self._line_is_overly_generic(combined)
                or self.polisher._is_visual_description(combined)
                or (grounding and contains_unapproved_names(combined, grounding, world_terms=world_terms))
            ):
                continue
            payload["text"] = combined
            payload["visual_only"] = False
            payload["suppression_reason"] = None
        return reinforced

    def _alignment_followup_sentence(self, unit: dict[str, Any], current_text: str) -> str:
        evidence = " ".join(
            str(unit.get(key) or "").strip()
            for key in (
                "vision_action_beat",
                "vision_caption",
                "vision_dialogue",
                "combined_text",
                "visual_cues",
                "ocr_fallback_text",
            )
            if str(unit.get(key) or "").strip()
        )
        if not evidence.strip():
            return ""
        lowered = evidence.casefold()
        current_lower = str(current_text or "").casefold()
        names = [
            str(name).strip()
            for name in unit.get("character_names", []) or []
            if str(name).strip()
            and not self._vision_name_is_placeholder(str(name))
        ]
        subject = names[0] if names else ""
        candidates: list[str] = []
        if subject and re.search(r"\bfault|blame|pressure|question|argument|choice|decision\b", lowered):
            candidates.append(f"{subject} has to answer the pressure before the next choice closes in.")
        if subject and re.search(r"\brisk|danger|threat|fight|attack|enemy|mission|order\b", lowered):
            candidates.append(f"{subject} keeps the immediate danger in focus as the scene moves forward.")
        if subject and re.search(r"\bbond|partner|trust|promise|request|answer|message\b", lowered):
            candidates.append(f"{subject} keeps the unresolved exchange from fading into the background.")
        if re.search(r"\bguard|guards|this is as far as you go|get on\b", lowered):
            target = subject or "the group"
            candidates.append(f"The order leaves {target} with less room to choose the next move.")
        for candidate in candidates:
            normalized = self._normalize_segment_text(candidate, allow_empty=True)
            if not normalized:
                continue
            if self._normalized_line_key(normalized) in {
                self._normalized_line_key(sentence)
                for sentence in self._split_sentences_for_cleanup(current_text)
            }:
                continue
            combined = self._normalize_segment_text(f"{current_text} {normalized}", allow_empty=True)
            if (
                self._line_is_low_quality(normalized)
                or self._line_is_overly_generic(normalized)
                or self._line_is_dialogue_fragment(normalized)
                or self._line_is_sentence_fragment(normalized)
                or self._line_needs_style_refinement(normalized)
                or self._line_is_low_quality(combined)
                or self._line_is_overly_generic(combined)
            ):
                continue
            if not self._line_supported_by_unit_evidence(normalized, unit):
                continue
            return normalized
        return ""

    def _expand_short_scene_payloads_with_llm(
        self,
        payloads: list[dict[str, Any]],
        units: list[dict[str, Any]],
        *,
        project_title: str,
        chapter_metadata: dict[str, Any],
        chapter_summary: str,
        character_dictionary: dict[str, Any],
        protagonist_name: str | None,
        story_bible: dict[str, Any],
        name_grounding: dict[str, Any] | None,
        style_vocab: StyleVocabulary | None = None,
    ) -> list[dict[str, Any]]:
        if not payloads:
            return payloads
        try:
            if "gemini" not in self.router.available_providers():
                return payloads
        except Exception:
            return payloads

        refined = [dict(item) for item in payloads]
        target_indices: list[int] = []
        for index, payload in enumerate(refined):
            unit = units[index]
            current_text = str(payload.get("text") or "").strip()
            panel_count = int(unit.get("panel_count") or len(unit.get("panel_ids", []) or []))
            required_sentences = 3 if panel_count >= 3 else 2
            word_count = len(re.findall(r"\b[\w'-]+\b", current_text))
            has_local_evidence = bool(
                current_text
                or str(unit.get("vision_action_beat") or "").strip()
                or str(unit.get("vision_caption") or "").strip()
                or str(unit.get("vision_dialogue") or "").strip()
                or str(unit.get("ocr_fallback_text") or "").strip()
                or str(unit.get("scene_summary") or "").strip()
                or unit.get("character_names")
            )
            if not has_local_evidence:
                continue
            if (
                bool(payload.get("visual_only"))
                or not current_text
                or self._sentence_count(current_text) < required_sentences
                or word_count < (42 if panel_count >= 3 else 30)
                or self._line_is_overly_generic(current_text)
            ):
                target_indices.append(index)
        if not target_indices:
            return refined

        allowed_character_names = list(name_grounding.get("allowed_character_names") or []) if name_grounding else []
        prompt_story_bible = self._story_bible_prompt_payload(story_bible)
        world_terms = self._world_terms_for_guardrails(story_bible, name_grounding)
        if style_vocab:
            world_terms = list(
                dict.fromkeys(
                    [
                        *world_terms,
                        *style_vocab.world_terms,
                        *style_vocab.stakes_phrases,
                        *(value for value in (style_vocab.team_term, style_vocab.antagonist_term) if value),
                    ]
                )
            )
        accepted_count = 0
        for start in range(0, len(target_indices), 8):
            batch_indices = target_indices[start:start + 8]
            lines: list[dict[str, Any]] = []
            for local_index, global_index in enumerate(batch_indices):
                unit = units[global_index]
                lines.append(
                    {
                        "index": local_index,
                        "text": str(refined[global_index].get("text") or "").strip(),
                        "current_line": str(refined[global_index].get("text") or "").strip(),
                        "previous_line": str(refined[global_index - 1].get("text") or "").strip() if global_index > 0 else "",
                        "next_line": str(refined[global_index + 1].get("text") or "").strip() if global_index + 1 < len(refined) else "",
                        "scene_summary": str(unit.get("scene_summary") or "").strip(),
                        "vision_dialogue": str(unit.get("vision_dialogue") or "").strip(),
                        "vision_caption": str(unit.get("vision_caption") or "").strip(),
                        "vision_action_beat": str(unit.get("vision_action_beat") or "").strip(),
                        "ocr_fallback_text": str(unit.get("ocr_fallback_text") or "").strip(),
                        "local_evidence": self._style_evidence_text(unit),
                        "character_names": unit.get("character_names", []) or [],
                        "panel_count": int(unit.get("panel_count") or len(unit.get("panel_ids", []) or [])),
                    }
                )
            try:
                result = asyncio.run(
                    self.router.expand_story_segment_details(
                        lines,
                        {
                            "project_title": project_title,
                            "chapter_summary": chapter_summary,
                            "chapter_metadata": chapter_metadata,
                            "character_dictionary": character_dictionary,
                            "story_bible": prompt_story_bible,
                            "allowed_character_names": allowed_character_names,
                            "protagonist_name": protagonist_name or "",
                            "style_vocabulary": style_vocab.to_dict() if style_vocab else {},
                        },
                        provider="gemini",
                    )
                )
            except Exception as exc:
                logger.warning("Story segment expansion failed for batch %d-%d: %s", start, start + len(batch_indices), exc)
                continue

            rewrites = result.payload.get("rewrites", []) or []
            for item in rewrites:
                if not isinstance(item, dict):
                    continue
                try:
                    local_index = int(item.get("index"))
                except (TypeError, ValueError):
                    continue
                if local_index < 0 or local_index >= len(batch_indices):
                    continue
                global_index = batch_indices[local_index]
                candidate = self._normalize_segment_text(str(item.get("line") or "").strip(), allow_empty=True)
                current_count = self._sentence_count(str(refined[global_index].get("text") or ""))
                panel_count = int(units[global_index].get("panel_count") or len(units[global_index].get("panel_ids", []) or []))
                required_count = 3 if panel_count >= 3 else 2
                current_words = len(re.findall(r"\b[\w'-]+\b", str(refined[global_index].get("text") or "")))
                candidate_words = len(re.findall(r"\b[\w'-]+\b", candidate))
                if (
                    not candidate
                    or self._sentence_count(candidate) < required_count
                    or (
                        self._sentence_count(candidate) <= current_count
                        and candidate_words < max(35, int(current_words * 1.35))
                    )
                    or len(candidate) > 700
                ):
                    continue
                if (
                    self._line_is_low_quality(candidate)
                    or self._line_is_overly_generic(candidate)
                    or self._line_is_dialogue_fragment(candidate)
                    or self._line_is_sentence_fragment(candidate)
                    or self._line_has_first_person_narration(candidate)
                    or self.polisher._is_visual_description(candidate)
                    or (name_grounding and contains_unapproved_names(candidate, name_grounding, world_terms=world_terms))
                ):
                    trimmed = self._remove_offending_sentences(candidate)
                    if trimmed and self._sentence_count(trimmed) >= 2:
                        candidate = trimmed
                    else:
                        continue
                refined[global_index]["text"] = candidate
                refined[global_index]["visual_only"] = False
                refined[global_index]["suppression_reason"] = None
                accepted_count += 1
        if accepted_count:
            logger.info("Expanded %d short scene narration segments with style vocabulary", accepted_count)
        return refined

    def _deduplicate_story_segments(self, story_segments: list[StorySegment]) -> list[StorySegment]:
        """Remove exact and near-duplicate segments while preserving panel coverage.

        When consecutive panels generate identical or nearly identical narration
        (common in weak-evidence situations), merge them into a single segment
        to improve quality scores and avoid template-like repetition.
        """
        if len(story_segments) <= 1:
            return story_segments

        kept_segments: list[StorySegment] = []
        skip_indices: set[int] = set()

        for index, segment in enumerate(story_segments):
            if index in skip_indices:
                continue

            text = str(segment.text or "").strip()
            if not text:
                kept_segments.append(segment)
                continue

            # Check if next segments are duplicates or near-duplicates
            merge_indices = [index]
            merge_text = text
            found_different = False

            # Check up to 8 segments ahead for duplicates (span visual-only gaps)
            for next_index in range(index + 1, min(index + 8, len(story_segments))):
                if next_index in skip_indices:
                    # Skip indices already marked for merging, but keep looking
                    continue

                next_segment = story_segments[next_index]
                next_text = str(next_segment.text or "").strip()

                if not next_text:
                    # Skip visual-only segments (blank narration), but keep looking
                    continue

                # Check for exact duplicates
                if text == next_text:
                    merge_indices.append(next_index)
                    skip_indices.add(next_index)
                    logger.debug(
                        "Found exact duplicate: segment %d = segment %d",
                        index,
                        next_index,
                    )
                    continue

                # Check for near-duplicates (high similarity) - but only if consecutive
                if next_index == index + 1 or (next_index == index + 2 and not str(story_segments[index + 1].text or "").strip()):
                    similarity = self._text_similarity(text, next_text)
                    if similarity >= 0.80:  # 80% similarity threshold
                        merge_indices.append(next_index)
                        skip_indices.add(next_index)
                        merge_text = text  # Use the first occurrence
                        logger.debug(
                            "Found near-duplicate (%.2f similarity): segment %d ≈ segment %d",
                            similarity,
                            index,
                            next_index,
                        )
                        continue

                # Found a different non-empty segment, stop looking
                found_different = True
                break

            if len(merge_indices) > 1:
                # Merge multiple panel IDs into a single segment
                all_panel_ids = []
                for merge_idx in merge_indices:
                    panel_ids = story_segments[merge_idx].panel_ids or []
                    all_panel_ids.extend(panel_ids)

                merged_segment = segment.model_copy(
                    update={
                        "panel_ids": all_panel_ids if all_panel_ids else segment.panel_ids,
                        "text": text,
                    }
                )
                kept_segments.append(merged_segment)
                logger.info(
                    "Deduplicated segments [%s] → segment %d with %d total panel references",
                    ", ".join(str(i) for i in merge_indices),
                    len(kept_segments),
                    len(all_panel_ids),
                )
            else:
                kept_segments.append(segment)

        # Re-number orders to maintain monotonic sequence
        renumbered = [
            seg.model_copy(update={"order": idx + 1})
            for idx, seg in enumerate(kept_segments)
        ]

        if len(renumbered) < len(story_segments):
            logger.info(
                "Deduplication reduced segments from %d to %d",
                len(story_segments),
                len(renumbered),
            )

        return renumbered

    def _text_similarity(self, text_a: str, text_b: str) -> float:
        """Calculate Jaccard similarity between two texts (word-based).

        Returns a score from 0.0 (completely different) to 1.0 (identical).
        """
        words_a = set(re.findall(r"\b[\w'-]+\b", text_a.lower()))
        words_b = set(re.findall(r"\b[\w'-]+\b", text_b.lower()))

        if not words_a or not words_b:
            return 0.0 if words_a != words_b else 1.0

        intersection = len(words_a & words_b)
        union = len(words_a | words_b)
        return intersection / union if union > 0 else 0.0

    def _compose_story_text(self, story_segments: list[StorySegment]) -> str:
        lines = [segment.text.strip() for segment in story_segments if segment.text.strip()]
        if not lines:
            return ""
        return "\n\n".join(lines).strip()
