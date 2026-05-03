"""Script polishing service.

Post-processes draft narration for cohesion and quality.
Replaces the cohesive rewrite in stages.py, all dedup/repair in
panel_script_builder.py, and script_cleaner_service.py with a single
LLM rewrite pass followed by one programmatic validation pass.
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
from pathlib import Path
from typing import Any

from app.services.character_name_filters import looks_like_false_character_name
from app.services.llm_router import LLMRouter
from app.services.story_grounding import compact_chapter_metadata

logger = logging.getLogger(__name__)

_PROMPT_DIR = Path(__file__).resolve().parents[3] / "services" / "prompts"

_STOPWORDS = frozenset({
    "the", "a", "an", "i", "you", "he", "she", "it", "we", "they",
    "is", "are", "was", "were", "be", "been", "being", "am",
    "has", "have", "had", "do", "does", "did", "will", "would",
    "could", "should", "may", "might", "can", "shall", "must",
    "to", "of", "in", "for", "on", "with", "at", "by", "from",
    "as", "into", "through", "during", "before", "after",
    "and", "but", "or", "not", "no", "if", "so", "that", "this",
    "what", "who", "how", "when", "where", "why", "which",
    "his", "her", "its", "my", "your", "our", "their",
    "him", "them", "me", "us", "up", "out", "about", "over",
    "all", "one", "two", "new", "just", "also", "than", "more",
    "very", "too", "only", "now", "then", "here", "there",
})

_BASIC_ENGLISH = _STOPWORDS | frozenset({
    "said", "says", "told", "asked", "went", "came", "got",
    "made", "took", "gave", "know", "think", "see", "look",
    "want", "need", "let", "kill", "die", "fight", "run",
    "come", "go", "get", "make", "take", "give", "keep",
    "back", "down", "still", "even", "first", "last", "long",
    "good", "bad", "old", "other", "each", "every", "some",
    "any", "many", "much", "way", "time", "life", "world",
    "man", "hand", "part", "place", "case", "day", "eye",
    "right", "away", "own", "off", "another", "between",
    "same", "both", "few", "while", "against", "already",
    "yet", "never", "always", "enough", "because", "since",
    "until", "though", "around", "under", "along", "without",
})

# Patterns for visual descriptions
_VISUAL_PATTERNS = [
    re.compile(r"^(a|an|the)\s+\w+\s+(stands?|sits?|walks?|stood|sat|walked|lying|kneeling)\b", re.IGNORECASE),
    re.compile(r"^(a|an|the)\s+\w+\s+(with|in|on|at)\s+.*\b(room|floor|background|foreground|scene|frame)\b", re.IGNORECASE),
    re.compile(r"\blying on the\b", re.IGNORECASE),
    re.compile(r"\blooking (at|towards|concerned|angry|serious|worried)\b", re.IGNORECASE),
    # "A woman with [physical appearance]" or "The man with [appearance]"
    re.compile(r"^(a|an|the)\s+(man|woman|boy|girl|person|figure|character)\s+with\b", re.IGNORECASE),
    re.compile(r"\bwearing\b|\bdressed in\b|\bclad in\b", re.IGNORECASE),
    re.compile(r"\b(illuminated|bathed|silhouetted)\b", re.IGNORECASE),
    re.compile(r"\bspeech bubble\b|\btext bubble\b", re.IGNORECASE),
    re.compile(r"^(a|an|the)\s+(young|old|tall|short)?\s*(man|woman|boy|girl|person|figure|character)\s+(stands?|sits?|is\s+standing|is\s+sitting)", re.IGNORECASE),
    re.compile(r"\b(close-up|wide shot|panel shows|scene depicts|frame captures)\b", re.IGNORECASE),
    # Generic "A [adjective] [person] with [physical description]" — visual, not a story event
    re.compile(r"^(a|an|the)\s+\w+\s+(man|woman|boy|girl|person|figure)\s+(with|in|wearing|holding|sitting|standing)\b", re.IGNORECASE),
]

# Patterns for sentence fragments
# Gerund/participle openers that indicate a missing main clause when no comma follows.
# Valid: "Having checked the map, Zhang Yi sets out." (comma separates gerund phrase from main clause)
# Fragment: "Having been granted a second chance." (no main clause at all)
_GERUND_FRAGMENT_PATTERN = re.compile(
    r"^(Mocking|Having|Believing|Realizing|Observing|Grappling|Reviewing|Following|"
    r"Being|Feeling|Knowing|Seeing|Hearing|Watching|Noticing|Sensing|Thinking|Deciding|"
    r"Questioning|Wondering|Understanding|Recognizing|Considering|Accepting|Rejecting)\b",
    re.IGNORECASE,
)
# Lowercase connectors that start a dependent clause without a main clause
_LOWERCASE_FRAGMENT_PATTERNS = [
    re.compile(r"^(others|who|where|that|which)\b"),
]

_ROBOTIC_SUBJECT_OPENERS = re.compile(
    r"^(?:Other|A character|Another character|One character|A figure|Another figure|The figure|"
    r"Someone(?: nearby)?|Somebody|The speaker|A voice|Another voice)\b",
    re.IGNORECASE,
)

_ROBOTIC_REPORTING_PATTERNS = [
    re.compile(
        r"\b(?:expresses?|questions?|states?|declares?|mentions?|notes?|remarks?|observes?|"
        r"admits?|explains?|confirms?|announces?|informs?|replies?|responds?)\b",
        re.IGNORECASE,
    ),
    re.compile(r"\b(?:is|looks)\s+(?:startled|shocked|surprised|confused|worried|concerned)\b", re.IGNORECASE),
    re.compile(r"\b(?:reacts?|responds?)\s+(?:with|in)\b", re.IGNORECASE),
    re.compile(r"\b(?:appears|seems)\s+to\b", re.IGNORECASE),
    re.compile(r"\b(?:perhaps|presumably|seemingly)\b", re.IGNORECASE),
    re.compile(r"\b(?:is heard|echoes)\b", re.IGNORECASE),
]

_ROBOTIC_META_PATTERNS = [
    re.compile(r"^(?:This chapter|The narration|The text)\b", re.IGNORECASE),
    re.compile(r"^(?:A|An)\s+(?:loud|soft|sudden)\s+\w+", re.IGNORECASE),
]


class ScriptPolisher:
    """Post-processes draft narration for cohesion and quality."""

    _COHESIVE_REWRITE_CHUNK_SIZE = 60
    _FINAL_PASS_CHUNK_SIZE = 60
    _FINAL_PASS_OVERLAP = 8
    _FINAL_PASS_ANCHOR_LINES = 4
    _ROBOTIC_REPAIR_BATCH_SIZE = 8

    def __init__(self, router: LLMRouter | None = None) -> None:
        self.router = router or LLMRouter()
        self._last_artifacts: dict[str, list[str]] = {
            "strict_lines": [],
            "slot_locked_lines": [],
            "final_lines": [],
        }

    @property
    def last_artifacts(self) -> dict[str, list[str]]:
        return {
            key: list(value)
            for key, value in self._last_artifacts.items()
        }

    def polish(
        self,
        draft_lines: list[str],
        chapter_summary: str,
        character_dictionary: dict[str, str],
        *,
        project_title: str = "",
        chapter_metadata: Any | None = None,
        narrator: Any | None = None,
        locked_examples: str = "",
        slot_evidence: list[dict[str, Any]] | None = None,
        preserve_multi_sentence: bool = False,
    ) -> list[str]:
        """Rewrite draft script for cohesion, then validate.

        Args:
            draft_lines: Raw narration lines from PanelNarrator.
            chapter_summary: Chapter recap text.
            character_dictionary: {name: info} map.
            narrator: Optional PanelNarrator for retrying failed lines.
            locked_examples: Human-reviewed narrations from this chapter,
                formatted as "Panel N: '<text>'". Used as ground truth for
                character names and style during the cohesive rewrite.

        Returns:
            Polished narration lines.
        """
        if not draft_lines:
            return []

        effective_metadata = self._effective_chapter_metadata(
            chapter_metadata,
            project_title=project_title,
            character_dictionary=character_dictionary,
            draft_lines=draft_lines,
        )
        strict_lines = self._prepare_strict_lines(draft_lines)
        draft_blanks = sum(1 for l in draft_lines if not l.strip())
        logger.info("Polish input: %d lines, %d blank", len(draft_lines), draft_blanks)
        logger.info("Strict draft prepared: %d lines, %d blank", len(strict_lines), sum(1 for l in strict_lines if not l.strip()))

        # Step 1: Cohesive LLM rewrite
        rewritten = self._cohesive_rewrite(
            strict_lines,
            chapter_summary,
            character_dictionary,
            project_title=project_title,
            chapter_metadata=effective_metadata,
            locked_examples=locked_examples,
            preserve_multi_sentence=preserve_multi_sentence,
        )
        rewrite_blanks = sum(1 for l in rewritten if not l.strip()) if rewritten else 0
        logger.info(
            "Cohesive rewrite: %d lines (expected %d), %d blank",
            len(rewritten) if rewritten else 0, len(strict_lines), rewrite_blanks,
        )
        rewrite_looks_failed = rewritten == strict_lines and len(strict_lines) > self._COHESIVE_REWRITE_CHUNK_SIZE
        if rewrite_looks_failed:
            logger.warning("Cohesive rewrite produced an unchanged draft; retrying chunked rewrite")
        if not rewritten or len(rewritten) != len(strict_lines) or rewrite_looks_failed:
            logger.warning(
                "Cohesive rewrite returned %d lines (expected %d), retrying chunked rewrite",
                len(rewritten) if rewritten else 0, len(strict_lines),
            )
            rewritten = self._cohesive_rewrite_chunked(
                strict_lines,
                chapter_summary,
                character_dictionary,
                project_title=project_title,
                chapter_metadata=effective_metadata,
                locked_examples=locked_examples,
                preserve_multi_sentence=preserve_multi_sentence,
            )
        if not rewritten or len(rewritten) != len(strict_lines):
            logger.warning(
                "Chunked cohesive rewrite returned %d lines (expected %d), using draft",
                len(rewritten) if rewritten else 0, len(strict_lines),
            )
            rewritten = strict_lines

        # Step 2: One-pass programmatic validation
        validated = self._validate_lines(
            rewritten,
            strict_lines,
            preserve_multi_sentence=preserve_multi_sentence,
        )
        val_blanks = sum(1 for l in validated if not l.strip())
        logger.info("Validation output: %d lines, %d blank", len(validated), val_blanks)
        slot_locked = self._verify_slot_alignment(
            validated,
            strict_lines,
            slot_evidence or [],
            character_dictionary,
            preserve_multi_sentence=preserve_multi_sentence,
        )
        logger.info(
            "Slot-locked draft: %d lines, %d blank",
            len(slot_locked),
            sum(1 for l in slot_locked if not l.strip()),
        )

        # Step 3: Final Gemini-only continuity pass
        final_pass = self._final_continuity_pass(
            slot_locked,
            chapter_summary,
            character_dictionary,
            project_title=project_title,
            chapter_metadata=effective_metadata,
            locked_examples=locked_examples,
            slot_evidence=slot_evidence or [],
            preserve_multi_sentence=preserve_multi_sentence,
        )
        final_lines = slot_locked
        if final_pass and len(final_pass) == len(slot_locked):
            final_blanks = sum(1 for l in final_pass if not l.strip())
            logger.info("Final continuity pass: %d lines, %d blank", len(final_pass), final_blanks)
            validated = self._validate_lines(
                final_pass,
                slot_locked,
                preserve_multi_sentence=preserve_multi_sentence,
            )
            logger.info(
                "Post-continuity validation output: %d lines, %d blank",
                len(validated),
                sum(1 for l in validated if not l.strip()),
            )
            final_lines = self._verify_slot_alignment(
                validated,
                slot_locked,
                slot_evidence or [],
                character_dictionary,
                preserve_multi_sentence=preserve_multi_sentence,
            )

        repaired_lines = self._repair_robotic_lines(
            final_lines,
            slot_locked,
            chapter_summary,
            character_dictionary,
            project_title=project_title,
            chapter_metadata=effective_metadata,
            locked_examples=locked_examples,
            slot_evidence=slot_evidence or [],
            preserve_multi_sentence=preserve_multi_sentence,
        )
        self._last_artifacts = {
            "strict_lines": list(strict_lines),
            "slot_locked_lines": list(slot_locked),
            "final_lines": list(repaired_lines),
        }
        return repaired_lines

    def _cohesive_rewrite_chunked(
        self,
        lines: list[str],
        summary: str,
        characters: dict[str, str],
        *,
        project_title: str = "",
        chapter_metadata: Any | None = None,
        locked_examples: str = "",
        preserve_multi_sentence: bool = False,
    ) -> list[str]:
        if not lines:
            return []
        rewritten_all: list[str] = []
        for start in range(0, len(lines), self._COHESIVE_REWRITE_CHUNK_SIZE):
            chunk = lines[start:start + self._COHESIVE_REWRITE_CHUNK_SIZE]
            rewritten = self._cohesive_rewrite(
                chunk,
                summary,
                characters,
                project_title=project_title,
                chapter_metadata=chapter_metadata,
                locked_examples=locked_examples,
                preserve_multi_sentence=preserve_multi_sentence,
            )
            if not rewritten or len(rewritten) != len(chunk):
                logger.warning(
                    "Chunked cohesive rewrite failed for lines %d-%d (got %d, expected %d); using strict chunk",
                    start,
                    start + len(chunk),
                    len(rewritten) if rewritten else 0,
                    len(chunk),
                )
                rewritten_all.extend(chunk)
            else:
                rewritten_all.extend(rewritten)
        return rewritten_all

    def _effective_chapter_metadata(
        self,
        chapter_metadata: Any | None,
        *,
        project_title: str,
        character_dictionary: dict[str, Any],
        draft_lines: list[str],
    ) -> dict[str, Any]:
        payload = compact_chapter_metadata(chapter_metadata)
        hints = self._suggest_series_cast_hints(
            project_title,
            payload,
            character_dictionary,
            draft_lines,
        )
        if not hints:
            return payload
        merged = dict(payload)
        cast_hints = hints.get("series_cast_hints") or []
        corrections = hints.get("canonical_name_corrections") or []
        if cast_hints:
            merged["series_cast_hints"] = cast_hints
        if corrections:
            merged["canonical_name_corrections"] = corrections
        return merged

    def _suggest_series_cast_hints(
        self,
        project_title: str,
        chapter_metadata: dict[str, Any],
        character_dictionary: dict[str, Any],
        draft_lines: list[str],
    ) -> dict[str, Any]:
        try:
            if "gemini" not in self.router.available_providers():
                return {}
        except Exception:
            return {}

        observed_names = self._observed_name_candidates(draft_lines, character_dictionary)
        if not project_title and not chapter_metadata.get("manga_title"):
            return {}

        try:
            result = asyncio.run(
                self.router.suggest_series_cast_hints(
                    {
                        "project_title": project_title,
                        "chapter_metadata": chapter_metadata,
                        "character_dictionary": character_dictionary,
                        "observed_names": observed_names,
                    },
                    provider="gemini",
                )
            )
            return dict(result.payload)
        except Exception as exc:
            logger.warning("Series cast hint generation failed: %s", exc)
            return {}

    def _observed_name_candidates(
        self,
        draft_lines: list[str],
        character_dictionary: dict[str, Any],
    ) -> list[str]:
        candidates: set[str] = set()

        def add_candidate(raw: str) -> None:
            value = " ".join(str(raw or "").split()).strip(" ,.;:-")
            if not value or looks_like_false_character_name(value):
                return
            if len(value) < 2 or len(value) > 40:
                return
            if re.fullmatch(r"(?:The|A|An|Other|Someone|Somebody)", value):
                return
            candidates.add(value)

        for key, value in character_dictionary.items():
            add_candidate(str(key))
            if isinstance(value, dict):
                add_candidate(str(value.get("display_name") or ""))
                for alias in value.get("aliases", []) or []:
                    add_candidate(str(alias))
            else:
                add_candidate(str(value))

        for line in draft_lines[:]:
            for phrase in re.findall(r"\b(?:[A-Z][a-z0-9]+|[A-Z]{2,})(?:\s+(?:[A-Z][a-z0-9]+|[A-Z]{2,})){0,2}\b", str(line or "")):
                add_candidate(phrase)

        return sorted(candidates)[:40]

    # ------------------------------------------------------------------
    # Cohesive rewrite
    # ------------------------------------------------------------------

    def _cohesive_rewrite(
        self,
        lines: list[str],
        summary: str,
        characters: dict[str, str],
        *,
        project_title: str = "",
        chapter_metadata: Any | None = None,
        locked_examples: str = "",
        preserve_multi_sentence: bool = False,
    ) -> list[str]:
        """Single LLM call to rewrite full script for narrative flow."""
        if len(lines) <= 80:
            return self._rewrite_chunk(
                lines,
                summary,
                characters,
                project_title=project_title,
                chapter_metadata=chapter_metadata,
                locked_examples=locked_examples,
                preserve_multi_sentence=preserve_multi_sentence,
            )

        # Chunk large scripts with overlap
        overlap = 10
        chunk_size = 70
        all_rewritten: list[str] = []
        start = 0
        while start < len(lines):
            end = min(start + chunk_size, len(lines))
            chunk = lines[start:end]
            rewritten = self._rewrite_chunk(
                chunk,
                summary,
                characters,
                project_title=project_title,
                chapter_metadata=chapter_metadata,
                locked_examples=locked_examples,
                preserve_multi_sentence=preserve_multi_sentence,
            )

            if start == 0:
                all_rewritten.extend(rewritten)
            else:
                # Skip overlap lines, take the rest
                skip = min(overlap, len(rewritten))
                all_rewritten.extend(rewritten[skip:])

            start = end - overlap if end < len(lines) else end

        # Trim to exact length
        return all_rewritten[:len(lines)]

    def _rewrite_chunk(
        self,
        lines: list[str],
        summary: str,
        characters: dict[str, str],
        *,
        project_title: str = "",
        chapter_metadata: Any | None = None,
        locked_examples: str = "",
        preserve_multi_sentence: bool = False,
    ) -> list[str]:
        """Rewrite a single chunk via LLM."""
        template = (_PROMPT_DIR / "narrator-story-polish.md").read_text(encoding="utf-8")
        if preserve_multi_sentence:
            template = template.replace(
                "The draft was assembled panel-by-panel.",
                "The draft was assembled as story segments, and some lines intentionally contain multiple sentences.",
            )
            template = template.replace(
                "- Each output line corresponds to the same panel index as the input line.",
                "- Each output line corresponds to the same story segment index as the input line.",
            )
            template = template.replace(
                "- Keep each line to one sentence, usually 8-20 words.",
                "- Preserve multi-sentence scene lines. If an input line has 2-4 sentences, return 2-4 sentences covering the same setup, action, and consequence beats; do not compress a full scene into one sentence.",
            )

        char_block = ""
        if characters:
            entries = []
            for name, info in characters.items():
                if isinstance(info, dict):
                    display = info.get("display_name", name)
                    aliases = info.get("aliases", [])
                    role = info.get("role", "")
                    appearance = info.get("appearance", "")
                    parts = [display]
                    if aliases:
                        parts.append(f"(also: {', '.join(aliases)})")
                    if role:
                        parts.append(f"— {role}")
                    if appearance:
                        parts.append(f"[appearance: {appearance}]")
                    entries.append(" ".join(parts))
                else:
                    entries.append(f"{name}: {info}")
            char_block = "\n".join(entries)

        numbered_draft = "\n".join(f"{i}: {line}" for i, line in enumerate(lines))
        metadata_payload = (
            chapter_metadata
            if isinstance(chapter_metadata, dict)
            else chapter_metadata.model_dump(mode="json")
            if hasattr(chapter_metadata, "model_dump")
            else {}
        )

        prompt = (
            template
            .replace("{line_count}", str(len(lines)))
            .replace("{project_title}", project_title or "(unknown)")
            .replace("{chapter_metadata}", json.dumps(metadata_payload or {}, ensure_ascii=False) or "{}")
            .replace("{character_dictionary}", char_block or "(none)")
            .replace("{chapter_summary}", summary or "(none)")
            .replace("{locked_examples}", locked_examples or "(none — this is a fresh run)")
            .replace("{draft_script}", numbered_draft)
        )

        try:
            dynamic_budget = min(8000, max(1200, 55 * max(len(lines), 1)))
            result = asyncio.run(
                self.router._route_json(
                    task_name="script cohesive rewrite",
                    prompt=prompt,
                    validator=self._validate_rewrite_response,
                    max_output_tokens=dynamic_budget,
                )
            )
            rewritten = result.payload.get("rewritten_lines", [])
            return [str(line or "").strip() for line in rewritten]
        except Exception as exc:
            logger.warning("Cohesive rewrite failed: %s", exc)
            return lines

    def _final_continuity_pass(
        self,
        lines: list[str],
        summary: str,
        characters: dict[str, str],
        *,
        project_title: str = "",
        chapter_metadata: Any | None = None,
        locked_examples: str = "",
        slot_evidence: list[dict[str, Any]] | None = None,
        preserve_multi_sentence: bool = False,
    ) -> list[str]:
        """Run one final Gemini-only pass to make the script sound continuous.

        This is deliberately last in the chain so it can see already-clean lines
        and focus on story flow rather than basic repair. Large scripts are
        chunked with overlap to keep token usage bounded while still giving the
        model neighboring context.
        """
        if not lines:
            return lines

        try:
            if "gemini" not in self.router.available_providers():
                return lines
        except Exception:
            return lines

        ranges = self._chunk_ranges(
            len(lines),
            self._FINAL_PASS_CHUNK_SIZE,
            self._FINAL_PASS_OVERLAP,
        )
        if not ranges:
            return lines

        rewritten_all: list[str] = []
        total_chunks = len(ranges)
        for chunk_number, (start, end) in enumerate(ranges, start=1):
            chunk = lines[start:end]
            previous_lines = lines[max(0, start - self._FINAL_PASS_ANCHOR_LINES):start]
            next_lines = lines[end:min(len(lines), end + self._FINAL_PASS_ANCHOR_LINES)]
            rewritten = self._rewrite_final_continuity_chunk(
                chunk,
                summary,
                characters,
                project_title=project_title,
                chapter_metadata=chapter_metadata,
                locked_examples=locked_examples,
                previous_lines=previous_lines,
                next_lines=next_lines,
                chunk_index=chunk_number,
                chunk_total=total_chunks,
                slot_evidence=self._chunk_slot_evidence(slot_evidence or [], lines, start, end),
                preserve_multi_sentence=preserve_multi_sentence,
            )
            if chunk_number == 1:
                rewritten_all.extend(rewritten)
                continue
            skip = min(self._FINAL_PASS_OVERLAP, len(rewritten))
            rewritten_all.extend(rewritten[skip:])

        return rewritten_all[: len(lines)] if rewritten_all else lines

    def _rewrite_final_continuity_chunk(
        self,
        lines: list[str],
        summary: str,
        characters: dict[str, str],
        *,
        project_title: str,
        chapter_metadata: Any | None,
        locked_examples: str,
        previous_lines: list[str],
        next_lines: list[str],
        chunk_index: int,
        chunk_total: int,
        slot_evidence: list[dict[str, Any]],
        preserve_multi_sentence: bool = False,
    ) -> list[str]:
        metadata_payload = (
            chapter_metadata
            if isinstance(chapter_metadata, dict)
            else chapter_metadata.model_dump(mode="json")
            if hasattr(chapter_metadata, "model_dump")
            else {}
        )
        try:
            result = asyncio.run(
                self.router.rewrite_full_story(
                    lines,
                    summary,
                    characters,
                    project_title=project_title,
                    chapter_metadata=metadata_payload,
                    locked_examples=locked_examples,
                    previous_lines=previous_lines,
                    next_lines=next_lines,
                    chunk_index=chunk_index,
                    chunk_total=chunk_total,
                    slot_evidence=slot_evidence,
                    preserve_multi_sentence=preserve_multi_sentence,
                    provider="gemini",
                )
            )
            rewritten = [str(line or "").strip() for line in result.payload.get("rewritten_lines", [])]
            if len(rewritten) != len(lines):
                logger.warning(
                    "Final continuity pass returned %d lines for chunk %d/%d (expected %d); using pre-pass lines",
                    len(rewritten),
                    chunk_index,
                    chunk_total,
                    len(lines),
                )
                return lines
            return rewritten
        except Exception as exc:
            logger.warning("Final continuity pass failed for chunk %d/%d: %s", chunk_index, chunk_total, exc)
            return lines

    def _chunk_ranges(self, total_lines: int, chunk_size: int, overlap: int) -> list[tuple[int, int]]:
        if total_lines <= 0:
            return []
        if total_lines <= chunk_size:
            return [(0, total_lines)]

        ranges: list[tuple[int, int]] = []
        start = 0
        while start < total_lines:
            end = min(start + chunk_size, total_lines)
            ranges.append((start, end))
            if end >= total_lines:
                break
            start = max(end - overlap, start + 1)
        return ranges

    def _validate_rewrite_response(self, payload: Any) -> dict[str, Any]:
        """Validate story rewrite response."""
        if not isinstance(payload, dict):
            raise ValueError("Rewrite response is not a JSON object")
        lines = payload.get("rewritten_lines")
        if not isinstance(lines, list) or not lines:
            raise ValueError("Response missing rewritten_lines array")
        cleaned = [str(line or "").strip() for line in lines]
        return {"rewritten_lines": cleaned}

    # ------------------------------------------------------------------
    # Validation pass
    # ------------------------------------------------------------------

    # Transition prefixes the LLM sometimes adds despite the prompt forbidding them.
    # Strip these so the sentence reads naturally.
    _TRANSITION_PREFIXES = re.compile(
        r"^(?:By now|Soon|Then|Next|After that|At this point|From there|In practice|"
        r"Meanwhile|At the same time|In turn|Once again|As a result|In response|"
        r"With that|From this point),\s+",
        re.IGNORECASE,
    )

    def _prepare_strict_lines(self, lines: list[str]) -> list[str]:
        """Create a conservative slot-locked draft without dedupe blanking.

        This preserves one line per slot even when adjacent panels are similar,
        while still stripping the most obvious corruption that would poison the
        later cinematic passes.
        """
        strict: list[str] = []
        for raw in lines:
            line = " ".join(str(raw or "").split()).strip()
            if not line:
                strict.append("")
                continue

            stripped = self._TRANSITION_PREFIXES.sub("", line)
            if stripped != line:
                line = stripped[:1].upper() + stripped[1:] if stripped else ""

            if line and re.search(r"\b(?:Mr|Mrs|Ms|Dr|Sr|Jr)\.\s*$", line):
                line = ""
            if line and self._is_gibberish(line):
                line = ""

            strict.append(line)
        return strict

    def _validate_lines(
        self,
        lines: list[str],
        draft_lines: list[str],
        *,
        preserve_multi_sentence: bool = False,
    ) -> list[str]:
        """One-pass validation. Returns cleaned lines with bad ones blanked."""
        cleaned: list[str] = []
        seen_keywords: list[set[str]] = []
        # Track normalized content for exact-duplicate detection across the whole script.
        seen_normalized: set[str] = set()

        for i, line in enumerate(lines):
            draft = draft_lines[i] if i < len(draft_lines) else ""

            # Skip empty lines
            if not line.strip():
                # Try draft as fallback
                if draft.strip() and not self._line_has_issues(draft):
                    cleaned.append(draft)
                    seen_keywords.append(self._content_words(draft))
                    seen_normalized.add(re.sub(r"\s+", " ", draft.casefold()).strip())
                else:
                    cleaned.append("")
                    seen_keywords.append(set())
                continue

            # Strip filler transition prefixes the LLM adds despite being asked not to.
            stripped = self._TRANSITION_PREFIXES.sub("", line)
            if stripped != line:
                # Re-capitalize first letter after stripping the prefix.
                line = stripped[:1].upper() + stripped[1:] if stripped else ""

            line = self._replace_machine_placeholders(line)

            # Blank sentences that end abruptly on an honorific ("Zhang Yi meets with Mr.")
            # — these are truncated scene-summary copies where the LLM stopped mid-sentence.
            if line and re.search(r"\b(?:Mr|Mrs|Ms|Dr|Sr|Jr)\.\s*$", line):
                line = ""

            # Blank exact duplicates — same normalized text already seen earlier.
            if line:
                norm_key = re.sub(r"\s+", " ", line.casefold()).strip()
                if norm_key in seen_normalized:
                    line = ""

            # Blank near-duplicates — ≥80% keyword overlap with any of the last 5 lines.
            # This catches "Zhang Yi begins his preparations" vs
            # "Zhang Yi starts preparing for the apocalypse" that exact-dup misses.
            if line:
                recent_keywords = [kw for kw in seen_keywords[-5:] if kw]
                if recent_keywords and self._is_duplicate(line, recent_keywords, threshold=0.80):
                    line = ""

            if preserve_multi_sentence and line and self._line_has_issues(line):
                trimmed = self._trim_offending_sentences(line)
                if trimmed:
                    line = trimmed

            if line and self._is_visual_description(line):
                # Try draft as fallback
                if draft.strip() and not self._is_visual_description(draft):
                    line = draft
                else:
                    line = ""

            if line and self._is_fragment(line):
                line = ""

            if line and self._is_gibberish(line):
                line = ""

            cleaned.append(line)
            if line:
                norm_key = re.sub(r"\s+", " ", line.casefold()).strip()
                seen_normalized.add(norm_key)
                seen_keywords.append(self._content_words(line))
            else:
                seen_keywords.append(set())

        return cleaned

    def _replace_machine_placeholders(self, line: str) -> str:
        cleaned = re.sub(r"^Other of nowhere\b", "Out of nowhere", line)
        cleaned = re.sub(r"^Character[_\s-]*\d+\b", "The speaker", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\bCharacter[_\s-]*\d+\b", "the speaker", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"^Other\b(?!\s+figures\b)", "Another figure", cleaned)
        return cleaned

    def _split_sentences_for_cleanup(self, line: str) -> list[str]:
        return [
            sentence.strip()
            for sentence in re.split(r"(?<=[.!?])\s+(?=[A-Z])", " ".join(str(line or "").split()).strip())
            if sentence.strip()
        ]

    def _trim_offending_sentences(self, line: str) -> str:
        sentences = self._split_sentences_for_cleanup(line)
        if len(sentences) <= 1:
            return " ".join(str(line or "").split()).strip()
        survivors = [sentence for sentence in sentences if not self._line_has_issues(sentence)]
        if not survivors:
            return ""
        joined = " ".join(survivors).strip()
        return "" if self._line_has_issues(joined) else joined

    def _repair_robotic_lines(
        self,
        lines: list[str],
        fallback_lines: list[str],
        summary: str,
        characters: dict[str, Any],
        *,
        project_title: str = "",
        chapter_metadata: Any | None = None,
        locked_examples: str = "",
        slot_evidence: list[dict[str, Any]],
        preserve_multi_sentence: bool = False,
    ) -> list[str]:
        if not lines:
            return lines

        try:
            if "gemini" not in self.router.available_providers():
                return lines
        except Exception:
            return lines

        flagged_indexes = [
            index
            for index, line in enumerate(lines)
            if not (
                preserve_multi_sentence
                and len(self._split_sentences_for_cleanup(line)) >= 2
            )
            if self._needs_robotic_repair(
                line,
                previous_line=lines[index - 1] if index > 0 else "",
                next_line=lines[index + 1] if index + 1 < len(lines) else "",
                evidence=slot_evidence[index] if index < len(slot_evidence) and isinstance(slot_evidence[index], dict) else {},
            )
        ]
        if not flagged_indexes:
            return lines

        repaired_lines = list(lines)
        known_names = self._known_character_names(characters)
        metadata_payload = (
            chapter_metadata
            if isinstance(chapter_metadata, dict)
            else chapter_metadata.model_dump(mode="json")
            if hasattr(chapter_metadata, "model_dump")
            else {}
        )

        for start in range(0, len(flagged_indexes), self._ROBOTIC_REPAIR_BATCH_SIZE):
            batch_indexes = flagged_indexes[start:start + self._ROBOTIC_REPAIR_BATCH_SIZE]
            batch_payload: list[dict[str, Any]] = []
            for index in batch_indexes:
                evidence = slot_evidence[index] if index < len(slot_evidence) and isinstance(slot_evidence[index], dict) else {}
                batch_payload.append(
                    {
                        "index": index,
                        "current_line": repaired_lines[index],
                        "strict_line": fallback_lines[index] if index < len(fallback_lines) else "",
                        "previous_line": repaired_lines[index - 1] if index > 0 else "",
                        "next_line": repaired_lines[index + 1] if index + 1 < len(repaired_lines) else "",
                        "ocr_text": str(evidence.get("ocr_text") or evidence.get("text") or "").strip(),
                        "dialogue": [
                            str(item).strip()
                            for item in evidence.get("dialogue", []) or []
                            if str(item).strip()
                        ][:3],
                        "character_names": [
                            str(name).strip()
                            for name in evidence.get("character_names", []) or []
                            if str(name).strip() and not looks_like_false_character_name(name)
                        ][:5],
                        "preferred_subject": str(evidence.get("preferred_subject") or "").strip(),
                        "scene_summary": str(evidence.get("scene_summary") or "").strip(),
                    }
                )

            try:
                result = asyncio.run(
                    self.router.repair_story_lines(
                        batch_payload,
                        {
                            "project_title": project_title,
                            "chapter_summary": summary,
                            "chapter_metadata": metadata_payload,
                            "character_dictionary": characters,
                            "locked_examples": locked_examples,
                        },
                        provider="gemini",
                    )
                )
                rewrites = result.payload.get("rewrites", [])
            except Exception as exc:
                logger.warning("Robotic-line repair failed for %d lines: %s", len(batch_payload), exc)
                continue

            for item in rewrites:
                try:
                    index = int(item.get("index"))
                except (TypeError, ValueError):
                    continue
                if index < 0 or index >= len(repaired_lines):
                    continue
                candidate = " ".join(str(item.get("line") or "").split()).strip()
                if not candidate:
                    continue

                candidate = self._replace_machine_placeholders(candidate)
                evidence = slot_evidence[index] if index < len(slot_evidence) and isinstance(slot_evidence[index], dict) else {}
                fallback_line = fallback_lines[index] if index < len(fallback_lines) else ""
                candidate = self._repair_subject_naming(candidate, evidence, known_names, fallback_line)

                if preserve_multi_sentence and self._line_has_issues(candidate):
                    trimmed = self._trim_offending_sentences(candidate)
                    if trimmed:
                        candidate = trimmed

                if self._line_has_issues(candidate):
                    continue
                if not self._line_matches_slot(candidate, fallback_line, evidence):
                    continue

                repaired_lines[index] = candidate

        return repaired_lines

    def _needs_robotic_repair(
        self,
        line: str,
        *,
        previous_line: str = "",
        next_line: str = "",
        evidence: dict[str, Any] | None = None,
    ) -> bool:
        normalized = " ".join(str(line or "").split()).strip()
        if not normalized:
            return False

        lowered = normalized.casefold()
        if _ROBOTIC_SUBJECT_OPENERS.match(normalized):
            return True
        if any(pattern.search(normalized) for pattern in _ROBOTIC_META_PATTERNS):
            return True
        if any(pattern.search(normalized) for pattern in _ROBOTIC_REPORTING_PATTERNS):
            return True
        if re.search(r"\b(?:asks?|tells?|calls out to|greets?)\b", lowered) and not re.search(r"\b(?:why|how|what|whether|if|that)\b", lowered):
            return True
        if re.search(r"\b(?:sound effect|whoosh|swoosh|boom|bang|thud|vwoom|bazz?t)\b", lowered):
            return True
        if normalized.startswith("Chapter ") or normalized.startswith("This moment"):
            return True

        evidence_text = " ".join(
            part
            for part in [
                str(evidence.get("ocr_text") or evidence.get("text") or "").strip() if evidence else "",
                " ".join(str(item).strip() for item in (evidence.get("dialogue", []) if evidence else []) or [] if str(item).strip()),
                str(evidence.get("scene_summary") or "").strip() if evidence else "",
            ]
            if part
        ).casefold()
        if "?" in evidence_text and re.search(r"\b(?:questions?|asks?)\b", lowered):
            return True

        previous_opener = re.match(r"[A-Za-z']+", previous_line or "")
        current_opener = re.match(r"[A-Za-z']+", normalized)
        next_opener = re.match(r"[A-Za-z']+", next_line or "")
        if (
            current_opener
            and previous_opener
            and next_opener
            and current_opener.group(0).casefold() == previous_opener.group(0).casefold() == next_opener.group(0).casefold()
        ):
            return True

        return False

    def _chunk_slot_evidence(
        self,
        slot_evidence: list[dict[str, Any]],
        lines: list[str],
        start: int,
        end: int,
    ) -> list[dict[str, Any]]:
        compact: list[dict[str, Any]] = []
        for local_index, global_index in enumerate(range(start, min(end, len(lines)))):
            raw = slot_evidence[global_index] if global_index < len(slot_evidence) and isinstance(slot_evidence[global_index], dict) else {}
            dialogue = [
                str(item).strip()
                for item in raw.get("dialogue", []) or []
                if str(item).strip()
            ][:3]
            names = [
                str(name).strip()
                for name in raw.get("character_names", []) or []
                if str(name).strip() and not looks_like_false_character_name(name)
            ][:5]
            preferred_subject = str(raw.get("preferred_subject") or "").strip()
            if looks_like_false_character_name(preferred_subject):
                preferred_subject = ""
            compact.append(
                {
                    "index": local_index,
                    "panel_order": int(raw.get("panel_order") or raw.get("panel") or 0),
                    "page": int(raw.get("page") or 0),
                    "strict_line": str(lines[global_index] or "").strip(),
                    "ocr_text": str(raw.get("ocr_text") or raw.get("text") or "").strip()[:220],
                    "dialogue": dialogue,
                    "character_names": names,
                    "preferred_subject": preferred_subject,
                    "scene_summary": str(raw.get("scene_summary") or "").strip()[:220],
                    "visual_caption": str(raw.get("visual_caption") or "").strip()[:120],
                }
            )
        return compact

    def _line_has_issues(self, line: str) -> bool:
        """Quick check if a line has any quality issues."""
        return (
            self._is_fragment(line)
            or self._is_gibberish(line)
            or self._is_visual_description(line)
        )

    def _verify_slot_alignment(
        self,
        candidate_lines: list[str],
        fallback_lines: list[str],
        slot_evidence: list[dict[str, Any]],
        character_dictionary: dict[str, Any],
        *,
        preserve_multi_sentence: bool = False,
    ) -> list[str]:
        verified: list[str] = []
        known_names = self._known_character_names(character_dictionary)
        for index, fallback in enumerate(fallback_lines):
            candidate = candidate_lines[index] if index < len(candidate_lines) else ""
            evidence = slot_evidence[index] if index < len(slot_evidence) and isinstance(slot_evidence[index], dict) else {}
            line = " ".join(str(candidate or "").split()).strip()
            fallback_line = " ".join(str(fallback or "").split()).strip()

            if not line:
                verified.append(fallback_line)
                continue

            repaired = self._repair_subject_naming(line, evidence, known_names, fallback_line)
            if repaired:
                line = repaired

            if self._line_has_issues(line):
                if preserve_multi_sentence:
                    trimmed = self._trim_offending_sentences(line)
                    if trimmed:
                        line = trimmed
                    else:
                        verified.append(fallback_line)
                        continue
                else:
                    verified.append(fallback_line)
                    continue

            if self._line_has_issues(line):
                verified.append(fallback_line)
                continue

            if not self._line_matches_slot(line, fallback_line, evidence):
                verified.append(fallback_line)
                continue

            verified.append(line)
        return verified

    def _line_matches_slot(
        self,
        candidate: str,
        fallback: str,
        evidence: dict[str, Any],
    ) -> bool:
        candidate_words = self._content_words(candidate)
        if not candidate_words:
            return False

        evidence_text = " ".join(
            part
            for part in [
                fallback,
                str(evidence.get("ocr_text") or evidence.get("text") or "").strip(),
                " ".join(str(item).strip() for item in evidence.get("dialogue", []) or [] if str(item).strip()),
                str(evidence.get("scene_summary") or "").strip(),
                str(evidence.get("visual_caption") or "").strip(),
                " ".join(str(name).strip() for name in evidence.get("character_names", []) or [] if str(name).strip()),
            ]
            if part
        )
        support_words = self._content_words(evidence_text)
        if not support_words:
            return True

        overlap = len(candidate_words & support_words)
        denominator = max(1, min(len(candidate_words), len(support_words)))
        ratio = overlap / denominator
        if ratio >= 0.14:
            return True

        # If support is rich and the candidate shares nothing meaningful with it,
        # keep the stricter fallback for this slot.
        return len(support_words) < 3

    def _repair_subject_naming(
        self,
        line: str,
        evidence: dict[str, Any],
        known_names: set[str],
        fallback_line: str,
    ) -> str:
        preferred_names = [
            str(name).strip()
            for name in evidence.get("character_names", []) or []
            if str(name).strip() and not looks_like_false_character_name(name)
        ]
        preferred_subject = str(evidence.get("preferred_subject") or "").strip()
        if looks_like_false_character_name(preferred_subject):
            preferred_subject = ""
        if preferred_subject and preferred_subject not in preferred_names:
            preferred_names.insert(0, preferred_subject)
        if not preferred_names:
            return line

        preferred = preferred_names[0]
        possessive_pattern = re.compile(r"^(?:His|Her)\b", re.IGNORECASE)
        if possessive_pattern.match(line):
            return possessive_pattern.sub(f"{preferred}'s", line, count=1)

        generic_pattern = re.compile(
            r"^(?:Someone(?: nearby)?|Somebody|Other|A character|One character|Another character|A figure|Another figure|The figure|"
            r"The speaker|A voice|Another voice|A person|A young man|A young woman|"
            r"The young man|The young woman|The man|The woman|A boy|A girl|He|She)\b",
            re.IGNORECASE,
        )
        if generic_pattern.match(line):
            return generic_pattern.sub(preferred, line, count=1)

        if any(name.casefold() in line.casefold() for name in preferred_names):
            return line

        # Replace only explicit placeholder/OCR-noise names with the preferred
        # local name. Do not rewrite every capitalized sentence opener: ordinary
        # lines like "Massive structures hover..." are capitalized because they
        # start a sentence, not because "Massive" is a bad character name.
        placeholder_pattern = re.compile(r"^Character[_\s-]*\d+\b", re.IGNORECASE)
        if placeholder_pattern.match(line):
            return placeholder_pattern.sub(preferred, line, count=1)

        leading_phrase = re.match(r"^([A-Z][A-Za-z0-9]+(?:\s+[A-Z][A-Za-z0-9]+){0,2})\b", line)
        if leading_phrase:
            subject = leading_phrase.group(1).strip()
            if subject.isupper():
                return line
            normalized_subject = self._normalize_name_phrase(subject)
            if (
                normalized_subject
                and normalized_subject not in known_names
                and looks_like_false_character_name(subject)
                and not self._looks_like_common_sentence_opener(subject)
            ):
                return preferred + line[leading_phrase.end():]

        return line

    def _known_character_names(self, character_dictionary: dict[str, Any]) -> set[str]:
        names: set[str] = set()
        for key, value in character_dictionary.items():
            for raw in (key, value.get("display_name") if isinstance(value, dict) else value):
                normalized = self._normalize_name_phrase(str(raw or ""))
                if normalized and not looks_like_false_character_name(raw):
                    names.add(normalized)
            if isinstance(value, dict):
                for alias in value.get("aliases", []) or []:
                    normalized = self._normalize_name_phrase(str(alias or ""))
                    if normalized and not looks_like_false_character_name(alias):
                        names.add(normalized)
        return names

    def _normalize_name_phrase(self, value: str) -> str:
        return " ".join(re.findall(r"[a-z0-9]+", value.casefold())).strip()

    def _looks_like_common_sentence_opener(self, value: str) -> bool:
        return value.casefold() in {
            "after", "before", "when", "while", "meanwhile", "then", "as", "if",
            "because", "although", "though", "once", "today", "tomorrow", "yesterday",
            "the", "a", "an", "this", "that", "these", "those",
        }

    # ------------------------------------------------------------------
    # Individual validators
    # ------------------------------------------------------------------

    def _is_fragment(self, line: str) -> bool:
        """Detect sentence fragments: starts with lowercase connector or is very short.

        After a proper LLM cohesive rewrite, fragments are rare. This check
        only catches the clearest cases to avoid false positives on valid
        narration that happens to start with a gerund or contain proper nouns.
        """
        stripped = line.strip()
        if not stripped:
            return False

        # Gerund/participle opener without a following main clause.
        # A gerund opener is valid if followed by a comma: "Having checked, Zhang Yi acts."
        # Without a comma, there is no main clause and the line is a fragment.
        if _GERUND_FRAGMENT_PATTERN.match(stripped) and "," not in stripped:
            return True

        # Starts with a lowercase connector word (not "i " / "i'")
        if stripped[0].islower() and not stripped.startswith(("i ", "i'")):
            for pattern in _LOWERCASE_FRAGMENT_PATTERNS:
                if pattern.match(stripped):
                    return True

        # Very short lines (≤3 words) ending with period — almost certainly a fragment
        words = stripped.split()
        if len(words) <= 3 and stripped.endswith("."):
            return True

        return False

    def _is_gibberish(self, line: str) -> bool:
        """Detect garbled/nonsensical text.

        Catches clear structural corruption (heavy punctuation/digits or
        heavy accented characters), broken-grammar OCR truncations, and
        stuck phrase loops. The former word-validity ratio check has
        been removed because character names and specific story vocabulary
        legitimately score low against a basic English word list, causing
        false positives on perfectly valid narration lines.
        """
        stripped = line.strip()
        if not stripped or len(stripped) < 10:
            return False

        # Character soup: > 30% digits/punctuation (structural corruption)
        non_alpha = sum(1 for c in stripped if not c.isalpha() and not c.isspace())
        if non_alpha / len(stripped) > 0.30:
            return True

        # Accented character clusters > 25% (untranslated foreign text)
        alpha_chars = [c for c in stripped if c.isalpha()]
        if alpha_chars:
            accented = sum(1 for c in alpha_chars if ord(c) > 127)
            if accented / len(alpha_chars) > 0.25:
                return True

        # Repeated word triplet: "it's and it's and it's" (LLM loop artifact)
        if re.search(r"\b(\w{3,})\b.{0,10}\b\1\b.{0,10}\b\1\b", stripped, re.IGNORECASE):
            return True

        # Broken OCR: any sentence in the line ends with a bare linking or
        # auxiliary verb without a complement. E.g. "Asi can name is.",
        # "Her name is.", "The group would have.". These almost always come
        # from truncated OCR being passed through as narration.
        sentences = re.split(r"(?<=[.!?])\s+", stripped)
        for sentence in sentences:
            part = sentence.strip()
            if not part:
                continue
            toks = re.findall(r"[A-Za-z']+", part)
            if len(toks) <= 8 and re.search(
                r"\b(?:is|are|was|were|be|been|have|has|had|do|does|did|can|could|will|would|should|may|must|might|shall)\.\s*$",
                part,
                flags=re.IGNORECASE,
            ):
                # Allow sentences where a real subject-complement is clearly
                # present, e.g. "Her eyes narrow and she nods." — checked by
                # looking for at least 5 tokens AND a content verb before the
                # linking verb. For narration we're strict: these are almost
                # always OCR fragments.
                if len(toks) <= 6:
                    return True

        # Stretch of short (≤ 3 letter) tokens dominating a short phrase,
        # a very common OCR-garbage signature. Threshold is set conservatively
        # (65 %) so that normal narration like
        # "The doctor observes that the fog is headed their way and it is not
        # looking good." (≈ 56 % short tokens) is not flagged.
        toks_all = re.findall(r"[A-Za-z']+", stripped)
        if 6 <= len(toks_all) <= 40:
            short = sum(1 for t in toks_all if len(t) <= 3)
            if short / len(toks_all) >= 0.65:
                return True

        # OCR-garbage token signatures: 3-letter tokens with no vowels
        # (e.g. "Gwr", "ucc", "tion" without the "a" ending, "Klx", "Grr"),
        # or 4-letter tokens that look like suffix-only fragments ("tion",
        # "ment" by itself).
        if len(toks_all) >= 5:
            garbage_tokens = 0
            SUFFIX_ONLY_FRAGMENTS = {"tion", "ment", "ness", "ship", "ing", "tions"}
            for token in toks_all:
                lower = token.lower()
                # No-vowel consonant cluster 3+ chars: "Gwr", "Klx", "ucc"
                if len(token) >= 3 and not re.search(r"[aeiouy]", lower):
                    garbage_tokens += 1
                    continue
                # Suffix-only fragments surviving as their own words
                if lower in SUFFIX_ONLY_FRAGMENTS:
                    garbage_tokens += 1
            if garbage_tokens >= 2 and garbage_tokens / len(toks_all) >= 0.08:
                return True

        return False

    def _is_visual_description(self, line: str) -> bool:
        """Detect lines describing what a panel image looks like."""
        lowered = line.casefold().strip()
        if not lowered:
            return False

        for pattern in _VISUAL_PATTERNS:
            if pattern.search(lowered):
                return True

        return False

    def _is_duplicate(
        self,
        line: str,
        previous: list[set[str]],
        threshold: float = 0.70,
    ) -> bool:
        """Keyword-overlap duplicate detection. Single pass."""
        words = self._content_words(line)
        if len(words) < 3:
            return False
        for prev_words in previous:
            if not prev_words:
                continue
            overlap = len(words & prev_words)
            denominator = max(len(words), len(prev_words))
            if denominator > 0 and overlap / denominator >= threshold:
                return True
        return False

    def _content_words(self, text: str) -> set[str]:
        """Extract content words (excluding stopwords)."""
        return set(re.findall(r"[a-z']+", text.casefold())) - _STOPWORDS
