from __future__ import annotations

import re
from dataclasses import dataclass, field

from app.services.video_finishing_service import strip_panel_sfx


@dataclass(slots=True)
class NarrationUnit:
    index: int
    panel_id: str
    raw_text: str
    story_text: str
    spoken_text: str
    language: str | None = None
    emotion: str = "neutral narration"
    pauses_ms: list[int] = field(default_factory=list)
    metadata: dict[str, object] = field(default_factory=dict)


class StoryPreprocessor:
    _OPENERS = (
        "Moments later,",
        "Then,",
        "A beat later,",
        "Without warning,",
        "At that moment,",
    )

    def process(
        self,
        script_lines: list[str],
        panel_ids: list[str] | None = None,
    ) -> list[NarrationUnit]:
        units: list[NarrationUnit] = []
        previous_subject = ""
        for index, raw_line in enumerate(script_lines, start=1):
            cleaned = self._normalize(raw_line)
            panel_id = panel_ids[index - 1] if panel_ids and index - 1 < len(panel_ids) else f"panel_{index:03d}"
            if not cleaned:
                units.append(NarrationUnit(index=index, panel_id=panel_id, raw_text="", story_text="", spoken_text=""))
                continue

            softened = self._soften_repeated_subject(cleaned, previous_subject, index)
            segments = self._segment_line(softened)
            story_text = "\n\n".join(segment for segment in segments if segment).strip()
            previous_subject = self._lead_subject(cleaned) or previous_subject
            pauses_ms = [self._pause_for_segment(segment) for segment in segments[:-1]]
            units.append(
                NarrationUnit(
                    index=index,
                    panel_id=panel_id,
                    raw_text=cleaned,
                    story_text=story_text,
                    spoken_text=story_text,
                    pauses_ms=pauses_ms,
                    metadata={"segments": segments},
                )
            )
        return units

    def _normalize(self, value: str) -> str:
        # Strip any visible-SFX/onomatopoeia descriptions BEFORE the
        # narration reaches the TTS engine. Without this, the model
        # speaks "G W O O" or "BOOM" letter-by-letter when the panel
        # vision narrator described the in-panel sound effect text.
        # `strip_panel_sfx` only removes SFX-pattern sentences; plain
        # story prose passes through untouched.
        cleaned_sfx = strip_panel_sfx(str(value or ""))
        # Tame the pauses Edge/Azure TTS gives to ellipses and dashes.
        # An "..." mid-sentence triggers a ~1.2 sec pause that breaks
        # narration flow; a free-standing " - " or " — " is similar.
        # Swap them for commas, which read at ~0.3 sec - still a beat,
        # but not a dead-stop. Hyphens INSIDE words (co-worker, well-known)
        # are left alone because the regex requires whitespace on at least
        # one side. Trailing "..." at the very end is preserved (gives a
        # nice fade-out beat at the end of a panel narration).
        text = cleaned_sfx.strip()
        text = re.sub(r"\s*\.{3,}\s*([A-Za-z])", r", \1", text)  # mid-sentence "..."
        text = re.sub(r"\s+[–—-]+\s+", ", ", text)               # spaced dashes
        text = re.sub(r"([A-Za-z])\s+[–—]\s+([A-Za-z])", r"\1, \2", text)  # em/en dash leftovers
        return re.sub(r"\s+", " ", text).strip()

    def _soften_repeated_subject(self, line: str, previous_subject: str, index: int) -> str:
        subject = self._lead_subject(line)
        if not subject or not previous_subject or subject.casefold() != previous_subject.casefold():
            return line
        first_name = subject.split()[0]
        shortened = re.sub(rf"^{re.escape(subject)}\b", first_name, line, count=1)
        if shortened != line:
            return shortened
        opener = self._OPENERS[(index - 1) % len(self._OPENERS)]
        return f"{opener} {line[0].lower() + line[1:]}" if len(line) > 1 else f"{opener} {line}"

    def _segment_line(self, line: str) -> list[str]:
        # Pass the full sentence to Kokoro as a single unit so its prosody model
        # handles pacing and intonation naturally. Splitting at conjunctions or
        # word-count midpoints produces hard silence gaps that sound like random
        # pauses mid-sentence.
        return [self._cinematic_finish(line)]

    def _cinematic_finish(self, value: str) -> str:
        trimmed = value.strip()
        if not trimmed:
            return ""
        if trimmed.endswith(("...", ".", "!", "?")):
            return trimmed
        if len(trimmed.split()) <= 3:
            return f"{trimmed}..."
        return f"{trimmed}."

    def _lead_subject(self, line: str) -> str:
        match = re.match(r"^([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})\b", line)
        return match.group(1).strip() if match else ""

    def _pause_for_segment(self, segment: str) -> int:
        word_count = max(len(segment.split()), 1)
        if word_count <= 3:
            return 450
        if word_count <= 8:
            return 320
        return 220
