"""
VideoFinishingService - the three "look like a real YouTuber" finishing
touches that bolt onto an already-rendered panel video:

  1. Cold open    - a 5-7 second hook from the climax panel + teaser
                    voiceover, placed BEFORE the chronological narration
  2. Title card   - channel-branded title slide for 2-3 seconds after
                    the cold open
  3. Outro card   - a subscribe-CTA card for ~5 seconds at the end of
                    the video, sized for YouTube's end-screen overlay
  4. Chapter      - emit a `chapter_markers.json` with timecodes the
     timestamps     YouTube bundle service folds into the description

The service doesn't actually render video frames - it builds two
artifact files the existing video pipeline + YouTube bundle consume:

  • <project>/output/cold_open_plan.json    - telling video_service.py
    what to prepend (panel id, teaser text, hold time)
  • <project>/output/chapter_markers.json   - list of (sec, label)
    that the bundle service writes into description.md

Rendering of cold open / outro frames happens in video_service.py
using the channel preset for branding.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from app.core.config import get_settings
from app.services.channel_preset_service import ChannelPreset
from app.utils.files import write_json

logger = logging.getLogger(__name__)

try:
    import google.generativeai as genai
    _GEMINI_AVAILABLE = True
except ImportError:
    _GEMINI_AVAILABLE = False


# Keywords that signal a moment worth opening with - same vocabulary the
# thumbnail picker uses, since "best thumbnail" and "best cold open" are
# the same problem (find the dramatic peak).
_CLIMAX_KEYWORDS = (
    "shock", "shouts", "screams", "explodes", "explosion", "destroyed",
    "reveal", "appears", "transforms", "kisses", "kiss", "punch", "fall",
    "dies", "monster", "giant", "huge", "massive", "tears", "blood",
    "weapon", "sword", "burning", "fire", "lightning",
)


@dataclass
class ColdOpenPlan:
    """What goes before the title card."""
    panel_id: str
    panel_order: int
    teaser_text: str
    hold_seconds: float = 6.0


@dataclass
class ChapterMarker:
    """One row in the YouTube chapters list."""
    timecode_seconds: float
    label: str

    def to_timecode(self) -> str:
        m, s = divmod(int(self.timecode_seconds), 60)
        h, m = divmod(m, 60)
        if h:
            return f"{h:d}:{m:02d}:{s:02d}"
        return f"{m:d}:{s:02d}"


@dataclass
class FinishingPlan:
    cold_open: ColdOpenPlan | None
    chapter_markers: list[ChapterMarker] = field(default_factory=list)


class VideoFinishingService:
    """Build the cold-open + chapter-marker plan for a project."""

    def __init__(self, settings: Any | None = None) -> None:
        self.settings = settings or get_settings()
        self._model = None

    def _gemini(self):
        if self._model is not None:
            return self._model
        if not _GEMINI_AVAILABLE or not self.settings.gemini_api_key:
            return None
        genai.configure(api_key=self.settings.gemini_api_key)
        preferred = (self.settings.gemini_model or "gemini-2.5-flash").strip()
        if preferred in {"gemini-2.0-flash", "gemini-2.0-flash-exp"}:
            preferred = "gemini-2.5-flash"
        self._model = genai.GenerativeModel(preferred)
        return self._model

    # ── Public entry point ────────────────────────────────────────────────

    def plan(
        self,
        *,
        panels_json: list[dict[str, Any]],
        script_lines: list[str],
        audio_manifest: dict[str, Any],
        preset: ChannelPreset,
        project_dir: Path,
    ) -> FinishingPlan:
        """Build cold-open + chapter-marker plan. Writes intermediate
        artifacts to disk so video_service and youtube_bundle_service
        can pick them up without re-doing the work."""

        kept = sorted(
            [p for p in panels_json if p.get("keep")],
            key=lambda p: (int(p.get("page", 0)), int(p.get("panel", 0))),
        )
        cold_open = None
        if preset.cold_open_enabled and kept:
            cold_open = self._select_cold_open(kept, script_lines, preset)

        markers = self._select_chapter_markers(kept, script_lines, audio_manifest)

        plan = FinishingPlan(cold_open=cold_open, chapter_markers=markers)

        # Persist intermediates so the renderer + bundle service can read
        # them on a fresh process (auto-run, retries, etc).
        output_dir = project_dir / "output"
        output_dir.mkdir(parents=True, exist_ok=True)
        if cold_open is not None:
            write_json(
                output_dir / "cold_open_plan.json",
                {
                    "panel_id": cold_open.panel_id,
                    "panel_order": cold_open.panel_order,
                    "teaser_text": cold_open.teaser_text,
                    "hold_seconds": cold_open.hold_seconds,
                },
            )
        write_json(
            output_dir / "chapter_markers.json",
            [{"timecode_seconds": m.timecode_seconds, "label": m.label} for m in markers],
        )
        return plan

    # ── Cold-open selection ──────────────────────────────────────────────

    def _select_cold_open(
        self,
        kept_sorted: list[dict[str, Any]],
        script_lines: list[str],
        preset: ChannelPreset,
    ) -> ColdOpenPlan:
        """Pick the climax panel and write a punchy teaser line for it."""
        # Same scoring shape as the thumbnail picker, but biased even
        # harder toward the back half of the chapter - viewers tolerate
        # spoilers in cold opens, in fact they hook on them.
        total = len(kept_sorted)
        scores: list[tuple[float, int, dict[str, Any], str]] = []
        for idx, panel in enumerate(kept_sorted):
            narr = (
                (panel.get("narration") or "").strip()
                or (script_lines[idx].strip() if idx < len(script_lines) else "")
            )
            lower = narr.lower()
            score = 0.0
            for keyword in _CLIMAX_KEYWORDS:
                if keyword in lower:
                    score += 5.0
                    break
            # Strong climax bias: peak around the 75% mark
            t = idx / max(1, total - 1)
            score += max(0.0, 1.0 - abs(t - 0.78) * 2.5) * 4.0
            if len(narr.split()) >= 12:
                score += 1.5
            scores.append((score, idx, panel, narr))

        scores.sort(key=lambda s: s[0], reverse=True)
        _, climax_idx, climax_panel, climax_narration = scores[0]

        teaser = self._generate_teaser(
            climax_narration,
            preset=preset,
            full_script=script_lines,
        )

        return ColdOpenPlan(
            panel_id=str(climax_panel.get("id")),
            panel_order=int(climax_panel.get("order", 0)),
            teaser_text=teaser,
            hold_seconds=float(preset.cold_open_duration_seconds),
        )

    def _generate_teaser(
        self,
        climax_narration: str,
        *,
        preset: ChannelPreset,
        full_script: list[str],
    ) -> str:
        """One punchy teaser line under 18 words. Try Gemini; fall back to
        a rule-based derivation if the call fails."""
        model = self._gemini()
        if model is None:
            return self._fallback_teaser(climax_narration)
        try:
            opener = " ".join(line for line in full_script[:30] if line.strip())
            prompt = (
                "Write ONE cold-open teaser line for the start of a YouTube "
                "manga recap video. Goal: hook the viewer in the first 6 "
                "seconds with a tease of where the chapter is heading.\n\n"
                "RULES:\n"
                "  • 8-18 words\n"
                "  • Hint at the climax WITHOUT naming the exact outcome\n"
                "  • End on suspense (no 'and they lived happily ever after')\n"
                "  • Do not use 'In this chapter', 'Today on', or any meta wrapper\n"
                "  • Return JUST the line - no quotes, no prefix\n\n"
                f"Opening of the recap: {opener[:1200]}\n\n"
                f"Climax moment: {climax_narration[:200]}"
            )
            gen_kwargs: dict[str, Any] = {
                "temperature": 0.75,
                "top_p": 0.9,
                "max_output_tokens": 256,
            }
            try:
                from google.generativeai.types import ThinkingConfig  # type: ignore
                gen_kwargs["thinking_config"] = ThinkingConfig(thinking_budget=0)
            except Exception:
                pass
            response = model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(**gen_kwargs),
            )
            text = (getattr(response, "text", "") or "").strip()
            text = re.sub(r"^[\"'`]+|[\"'`]+$", "", text)
            text = text.split("\n", 1)[0].strip()
            if 4 <= len(text.split()) <= 28:
                return text
        except Exception as exc:  # noqa: BLE001
            logger.warning("Cold-open teaser LLM failed: %s", exc)
        return self._fallback_teaser(climax_narration)

    @staticmethod
    def _fallback_teaser(climax_narration: str) -> str:
        """Heuristic teaser when LLM is unavailable: trim the climax
        narration to the most evocative clause and reframe as
        cliffhanger."""
        # Take the first ≤14 words of the climax narration.
        words = [w for w in climax_narration.split() if w.strip()]
        snippet = " ".join(words[:14])
        if not snippet:
            return "Wait until you see how this one ends."
        if not snippet.endswith("..."):
            snippet = snippet.rstrip(".!?") + "…"
        return snippet

    # ── Chapter markers ──────────────────────────────────────────────────

    def _select_chapter_markers(
        self,
        kept_sorted: list[dict[str, Any]],
        script_lines: list[str],
        audio_manifest: dict[str, Any],
    ) -> list[ChapterMarker]:
        """Convert per-panel audio durations into a small list of YouTube
        chapter markers based on natural scene breaks.

        Strategy:
          • Compute the cumulative timecode at the start of each kept panel
          • Scene break = page jump (page number changes by ≥2) OR
            narration emotion shift (heuristic via keyword change)
          • Target 5-10 markers for a typical 10-minute video. Sample
            evenly if the heuristic produces too few.

        The labels are short scene names auto-generated from the panel's
        narration (first 4-6 informative words).
        """
        if not kept_sorted:
            return []

        # Build cumulative time per panel from the audio manifest.
        cumulative: list[float] = []
        running = 0.0
        for idx, panel in enumerate(kept_sorted):
            # Audio manifest is keyed by `panel_{NNN}.wav` based on the
            # narration order, NOT the panel's `order` field. We use
            # 1-based index into the kept-sorted list to match.
            audio_key = f"panel_{idx + 1:03d}.wav"
            entry = audio_manifest.get(audio_key) or {}
            duration = float(entry.get("duration_seconds") or 0.0)
            if duration <= 0:
                # Fall back to the panel's display duration estimate.
                duration = float(panel.get("duration_seconds") or 4.0)
            cumulative.append(running)
            running += duration

        # Detect candidate scene boundaries.
        boundaries: list[int] = [0]  # Always mark the opening.
        last_page = int(kept_sorted[0].get("page", 0))
        for idx in range(1, len(kept_sorted)):
            page = int(kept_sorted[idx].get("page", 0))
            if page - last_page >= 2:
                boundaries.append(idx)
                last_page = page
            else:
                last_page = page

        # If we got too few natural boundaries, sprinkle additional
        # markers at evenly-spaced intervals to give viewers something
        # to scrub to. Target ~6 markers minimum for any video > 5 min.
        total_seconds = running
        if total_seconds > 0:
            target_count = max(5, min(10, int(total_seconds / 90)))
            if len(boundaries) < target_count:
                step = max(1, len(kept_sorted) // target_count)
                for idx in range(0, len(kept_sorted), step):
                    if idx not in boundaries:
                        boundaries.append(idx)
        boundaries = sorted(set(boundaries))[:12]

        # Convert each boundary into a ChapterMarker with a punchy label.
        markers: list[ChapterMarker] = []
        for boundary_idx in boundaries:
            panel = kept_sorted[boundary_idx]
            narr = (
                (panel.get("narration") or "").strip()
                or (script_lines[boundary_idx].strip() if boundary_idx < len(script_lines) else "")
            )
            label = self._make_marker_label(narr, boundary_idx)
            markers.append(
                ChapterMarker(
                    timecode_seconds=cumulative[boundary_idx],
                    label=label,
                )
            )
        # YouTube requires the first marker to be at 0:00 - guarantee it.
        if markers and markers[0].timecode_seconds > 0.01:
            markers.insert(0, ChapterMarker(timecode_seconds=0.0, label="Intro"))
        elif not markers:
            markers.append(ChapterMarker(timecode_seconds=0.0, label="Intro"))
        else:
            markers[0] = ChapterMarker(timecode_seconds=0.0, label="Intro")
        return markers

    @staticmethod
    def _make_marker_label(narration: str, idx: int) -> str:
        """Short, scrubbable chapter label from a narration line.

        Target: 2-3 words. Modeled on the manga-narration channels that
        get the highest CTR on chapter timestamps (Mr Recap, Manga Crash):
        the labels read like scene names ("Cold open", "The arrival",
        "Sortie") rather than beat-by-beat descriptions
        ("Nana questions Doctor about Hachi's progress decision").
        """
        cleaned = re.sub(r"[\"'`]", "", narration or "").strip()
        if not cleaned:
            return f"Scene {idx + 1}"

        # Take only the first sentence to avoid running into multi-action
        # descriptions (those produce 7-word labels that read like spam).
        first_sentence = re.split(r"(?<=[.!?])\s+", cleaned, maxsplit=1)[0]
        words = first_sentence.split()
        skip = {
            "the", "a", "an", "and", "but", "or", "of", "to", "in", "on",
            "is", "was", "are", "were", "be", "been", "being",
            "with", "by", "for", "from", "at",
            "his", "her", "their", "its", "our", "your", "my",
            "he", "she", "they", "it", "we", "you",
        }
        meaningful: list[str] = []
        for word in words:
            clean_word = re.sub(r"[^A-Za-z'\-]", "", word)
            if not clean_word:
                continue
            if clean_word.lower() in skip and meaningful:
                continue
            meaningful.append(clean_word)
            if len(meaningful) >= 3:
                break
        if not meaningful:
            return f"Scene {idx + 1}"
        label = " ".join(meaningful).rstrip(",.;:!?")
        # Title-case for the chapter UI.
        label = " ".join(w[0].upper() + w[1:] if w else w for w in label.split())
        return label[:40] or f"Scene {idx + 1}"


def format_chapter_markers_for_description(
    markers: list[ChapterMarker],
    *,
    cold_open_offset_seconds: float = 0.0,
    title_card_offset_seconds: float = 0.0,
) -> str:
    """Format a chapter-marker list as the timestamp block YouTube
    auto-parses in descriptions. Pads timecodes by the cold-open +
    title-card durations so the markers point to the right moment in
    the FINAL video (which has the intro prepended)."""
    offset = float(cold_open_offset_seconds) + float(title_card_offset_seconds)
    lines: list[str] = []
    for marker in markers:
        shifted = ChapterMarker(
            timecode_seconds=max(0.0, marker.timecode_seconds + offset),
            label=marker.label,
        )
        lines.append(f"{shifted.to_timecode()} {shifted.label}")
    return "\n".join(lines)
