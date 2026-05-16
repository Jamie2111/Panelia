"""
ChannelIntelligenceService

Higher-level layer on top of LiquidMemoryClient that exposes the three
Panelia use cases we've identified as most valuable:

  1. CHARACTER VOICING memory   — remembers per-character speech
     patterns, voice id, prosody preference across every video for
     this channel. Surfaced to the panel-vision narrator + the TTS
     engine so a character sounds the same in every episode.

  2. HOOK HISTORY               — every cold-open teaser that has
     actually shipped, with the video it belonged to. Lets the cold-
     open generator avoid repeating itself and learn from past hooks
     that performed well (once the user feeds back CTR data).

  3. THUMBNAIL-CHOICE log       — which thumbnail variants the user
     picked out of the carousel. After ~10 projects we have enough
     signal to bias the variant scorer toward their taste.

All methods are no-op safe when Liquid Memory isn't configured.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from app.services.liquid_memory_client import LiquidMemoryClient, MemoryRecord

logger = logging.getLogger(__name__)


@dataclass
class CharacterMemory:
    name: str
    voice_id: str | None = None
    speech_patterns: str = ""
    last_seen_project: str | None = None


class ChannelIntelligenceService:
    """Top-level facade. One instance per process is fine."""

    def __init__(self, client: LiquidMemoryClient | None = None) -> None:
        self.client = client or LiquidMemoryClient()

    @property
    def enabled(self) -> bool:
        return self.client.enabled

    # ── 1. Character voicing memory ──────────────────────────────────────

    def remember_character(
        self,
        *,
        name: str,
        voice_id: str,
        speech_patterns: str,
        project_id: str,
    ) -> str | None:
        """Called at the end of script generation for each named character."""
        content = (
            f"Character {name} | voice={voice_id} | "
            f"speech: {(speech_patterns or '').strip()[:240]}"
        )
        return self.client.store(
            kind="character_voice",
            content=content,
            metadata={
                "name": name,
                "voice_id": voice_id,
                "project_id": project_id,
            },
        )

    def recall_character(self, name: str, *, limit: int = 3) -> list[CharacterMemory]:
        """Called from the vision narrator + TTS engine before a panel
        with this character runs, so the voice stays consistent."""
        if not self.enabled or not name:
            return []
        records = self.client.recall(
            query=f"Character {name}",
            kind="character_voice",
            limit=limit,
        )
        out: list[CharacterMemory] = []
        for r in records:
            md = r.metadata or {}
            if (md.get("name") or "").lower() != name.lower():
                continue
            out.append(CharacterMemory(
                name=name,
                voice_id=md.get("voice_id"),
                speech_patterns=r.content,
                last_seen_project=md.get("project_id"),
            ))
        return out

    # ── 2. Hook history ──────────────────────────────────────────────────

    def remember_hook(
        self,
        *,
        teaser_text: str,
        project_id: str,
        series_name: str,
        chapter_label: str,
    ) -> str | None:
        """Called when the cold-open teaser ships to a bundle."""
        return self.client.store(
            kind="hook_history",
            content=teaser_text,
            metadata={
                "project_id": project_id,
                "series": series_name,
                "chapter": chapter_label,
            },
        )

    def recall_recent_hooks(self, *, limit: int = 10) -> list[str]:
        """For the cold-open prompt: 'don't repeat any of these'."""
        if not self.enabled:
            return []
        records = self.client.recall(
            query="cold-open hook line",
            kind="hook_history",
            limit=limit,
        )
        return [r.content for r in records if r.content]

    # ── 3. Thumbnail-choice log ──────────────────────────────────────────

    def remember_thumbnail_pick(
        self,
        *,
        project_id: str,
        chosen_index: int,
        chosen_style_label: str,
        chosen_overlay_text: str,
        group: str = "main",
    ) -> str | None:
        """Called from PUT /youtube-bundle whenever the user picks a
        different thumbnail variant in the publish studio."""
        return self.client.store(
            kind=f"thumbnail_choice_{group}",
            content=f"Picked '{chosen_style_label}': {chosen_overlay_text}",
            metadata={
                "project_id": project_id,
                "chosen_index": chosen_index,
                "style_label": chosen_style_label,
                "overlay_text": chosen_overlay_text,
                "group": group,
            },
        )

    def thumbnail_style_preferences(self, *, group: str = "main", limit: int = 20) -> dict[str, int]:
        """Count how often each style was chosen, so the scorer can
        bias toward the user's taste over time."""
        if not self.enabled:
            return {}
        records = self.client.recall(
            query=f"thumbnail variant pick {group}",
            kind=f"thumbnail_choice_{group}",
            limit=limit,
        )
        counts: dict[str, int] = {}
        for r in records:
            label = (r.metadata or {}).get("style_label") or "unknown"
            counts[label] = counts.get(label, 0) + 1
        return counts
