"""
CastBibleService — generate a "cast bible" once per project so the vision
narrator can refer to characters by name in every panel.

Problem this solves:
  Without character context, the vision narrator says "a uniformed
  woman with pink hair holds a lollipop" — accurate but cold. With a
  cast bible the same panel becomes "Zero Two casually licks a
  lollipop while sneaking through Plantation 13" — actually useful
  for a YouTube recap.

How it works:
  1. Once at the start of script generation, we ask Gemini one
     question: "Given this manga/manhwa/comic title and chapter, list
     the likely cast as JSON: [{name, role, visual_description}]".
     For popular series (Darling, One Piece, Solo Leveling, Bleach,
     etc.) the model already knows the cast from its training data —
     for ~$0.0003 in tokens we get a reliable bible.
  2. The bible is cached per project at output/cast_bible.json so we
     don't pay for it on retries.
  3. PanelVisionNarrator's per-panel prompt gets a "KNOWN CAST" block
     listing each character + visual description, so the model can
     match panel content to names with confidence.

For obscure / unknown manga: the LLM will return an empty / generic
list. The vision narrator then falls back to its existing behaviour
(describing characters generically). No regression.

Public API:
  • build_cast_bible(project_metadata, manga_title, chapter_title)
    → returns CastBible
  • load_cached(project_dir) → CastBible | None
  • format_for_prompt(bible) → str (drop-in for the prompt block)
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from app.core.config import get_settings
from app.utils.files import read_json, write_json

logger = logging.getLogger(__name__)

try:
    import google.generativeai as genai
    _GEMINI_AVAILABLE = True
except ImportError:
    _GEMINI_AVAILABLE = False


@dataclass
class CastMember:
    name: str
    role: str = ""
    visual_description: str = ""

    def to_dict(self) -> dict[str, str]:
        return {
            "name": self.name,
            "role": self.role,
            "visual_description": self.visual_description,
        }


@dataclass
class CastBible:
    """Result of the cast lookup. May be empty for obscure titles."""
    manga_title: str
    chapter_title: str
    members: list[CastMember] = field(default_factory=list)
    source: str = "unknown"  # "llm" | "fallback" | "cached"

    def is_empty(self) -> bool:
        return not self.members

    def to_dict(self) -> dict[str, Any]:
        return {
            "manga_title": self.manga_title,
            "chapter_title": self.chapter_title,
            "members": [m.to_dict() for m in self.members],
            "source": self.source,
        }


_BIBLE_PROMPT = """You are building a character cast bible for a YouTube
manga / manhwa / webtoon / comic recap pipeline.

Series: {manga_title}
Chapter focus: {chapter_title}

Return a JSON object with a single key "cast" whose value is an array of
character entries. Include the main / recurring characters of the series
who are likely to appear in this chapter. For each entry:

  {{
    "name": "the canonical English name (e.g. 'Zero Two', 'Levi Ackerman')",
    "role": "one short phrase, e.g. 'pistil pilot of Strelizia'",
    "visual_description": "the distinctive visible features the model
      should use to identify them in a panel — hair color, eye color,
      clothing, horns, scars, etc. Keep under 25 words."
  }}

Rules:
  • Only include characters you can verify from the series; do NOT invent.
  • Cap the list at 12 most-likely-relevant characters for this chapter.
  • If the series is too obscure or you don't have confident knowledge of
    it, return {{"cast": []}} — empty is better than wrong.
  • Return ONLY the JSON object. No prose, no code fences."""


class CastBibleService:
    """Lazily-constructed Gemini wrapper that emits a per-project cast bible."""

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

    # ── Public API ────────────────────────────────────────────────────────

    def load_cached(self, project_dir: Path) -> CastBible | None:
        """Return a cached cast bible if one exists for this project."""
        path = project_dir / "output" / "cast_bible.json"
        if not path.exists():
            return None
        try:
            raw = read_json(path)
            if not isinstance(raw, dict):
                return None
            members = [
                CastMember(
                    name=str(item.get("name") or "").strip(),
                    role=str(item.get("role") or "").strip(),
                    visual_description=str(item.get("visual_description") or "").strip(),
                )
                for item in (raw.get("members") or [])
                if isinstance(item, dict) and (item.get("name") or "").strip()
            ]
            return CastBible(
                manga_title=str(raw.get("manga_title") or ""),
                chapter_title=str(raw.get("chapter_title") or ""),
                members=members,
                source="cached",
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Cast bible cache unreadable: %s", exc)
            return None

    def ensure_bible(
        self,
        project_dir: Path,
        *,
        manga_title: str,
        chapter_title: str,
        force_refresh: bool = False,
    ) -> CastBible:
        """Get a bible, building one if needed. Always returns; falls
        back to empty for obscure series so the caller can still narrate."""
        if not force_refresh:
            cached = self.load_cached(project_dir)
            if cached is not None:
                return cached
        bible = self._build_via_llm(manga_title=manga_title, chapter_title=chapter_title)
        # Persist even empty bibles so we don't keep retrying on every
        # script regen for obscure series.
        write_json(project_dir / "output" / "cast_bible.json", bible.to_dict())
        return bible

    # ── LLM call ──────────────────────────────────────────────────────────

    def _build_via_llm(self, *, manga_title: str, chapter_title: str) -> CastBible:
        model = self._gemini()
        if model is None:
            logger.info("Cast bible LLM unavailable — returning empty bible.")
            return CastBible(manga_title=manga_title, chapter_title=chapter_title, source="fallback")

        prompt = _BIBLE_PROMPT.format(
            manga_title=manga_title or "(unknown)",
            chapter_title=chapter_title or "(unknown)",
        )
        try:
            gen_kwargs: dict[str, Any] = {
                "temperature": 0.3,
                "top_p": 0.9,
                "max_output_tokens": 2048,
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
            raw = getattr(response, "text", "") or ""
            return self._parse(raw, manga_title=manga_title, chapter_title=chapter_title)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Cast bible LLM call failed: %s", exc)
            return CastBible(manga_title=manga_title, chapter_title=chapter_title, source="fallback")

    @staticmethod
    def _parse(raw: str, *, manga_title: str, chapter_title: str) -> CastBible:
        text = (raw or "").strip()
        if not text:
            return CastBible(manga_title=manga_title, chapter_title=chapter_title, source="fallback")
        if text.startswith("```"):
            text = re.sub(r"^```[a-zA-Z]*\s*", "", text)
            text = re.sub(r"```\s*$", "", text).strip()
        if not text.startswith("{"):
            m = re.search(r"\{[\s\S]*\}", text)
            if m:
                text = m.group(0)
        try:
            data = json.loads(text)
        except (json.JSONDecodeError, ValueError):
            return CastBible(manga_title=manga_title, chapter_title=chapter_title, source="fallback")
        if not isinstance(data, dict):
            return CastBible(manga_title=manga_title, chapter_title=chapter_title, source="fallback")
        cast = data.get("cast")
        if not isinstance(cast, list):
            return CastBible(manga_title=manga_title, chapter_title=chapter_title, source="fallback")
        members: list[CastMember] = []
        for item in cast[:12]:
            if not isinstance(item, dict):
                continue
            name = str(item.get("name") or "").strip()
            if not name:
                continue
            members.append(
                CastMember(
                    name=name,
                    role=str(item.get("role") or "").strip(),
                    visual_description=str(item.get("visual_description") or "").strip(),
                )
            )
        return CastBible(
            manga_title=manga_title,
            chapter_title=chapter_title,
            members=members,
            source="llm",
        )

    # ── Prompt formatting helper ──────────────────────────────────────────

    @staticmethod
    def format_for_prompt(bible: CastBible | None) -> str:
        """Render the cast as a compact list for injection into the
        per-panel narration prompt. Empty bible → empty string so the
        narrator falls back to its non-cast prompt naturally."""
        if not bible or not bible.members:
            return ""
        lines: list[str] = ["KNOWN CAST (use these names when you can match them in the panel):"]
        for m in bible.members:
            piece = f"  • {m.name}"
            if m.visual_description:
                piece += f" — {m.visual_description}"
            elif m.role:
                piece += f" — {m.role}"
            lines.append(piece)
        return "\n".join(lines)
