"""
ChannelPresetService — the YouTuber's "brand kit" that applies across
every project.

After someone publishes 5 videos, viewers should be able to recognize
their channel by:
  • Title-card font + accent color
  • Watermark in the corner of every video and thumbnail
  • Outro card that always says the same thing
  • Consistent thumbnail styling

This service reads ONE JSON file (`backend/data/channel_preset.json`)
that the user edits via the settings page, and exposes a tiny API the
video renderer + youtube bundle service can call to apply the brand
everywhere.

Why a singleton JSON instead of per-project: branding should be
consistent across every video on a channel. If you ever want to run
multiple channels from one Panelia install we'd extend this to keyed
presets — for now, one.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from app.core.config import get_settings
from app.utils.files import read_json, write_json

logger = logging.getLogger(__name__)


@dataclass
class ChannelPreset:
    """One channel's branding configuration. All fields have defaults so
    new installs work without any setup."""

    # Identity
    channel_name: str = "Panelia"
    """Used in the outro card and watermark. The viewer's mental "this
    is what I'm subscribed to" text."""

    tagline: str = "Subscribe for chapter recaps as soon as they drop."
    """One short line displayed under the channel name in the outro."""

    # Color & type
    accent_color: str = "#7FFFD4"
    """Hex color for the title-card underline, outro accent, and
    thumbnail highlight word. Default = our mint accent."""

    title_font: str = "Impact"
    """Display font family for thumbnail text. Falls back to a sans
    family per OS if not installed."""

    # Watermark
    watermark_enabled: bool = True
    watermark_text: str = "@panelia"
    """Bottom-right text on every panel. Use "@yourhandle"."""

    # Outro
    outro_enabled: bool = True
    outro_duration_seconds: float = 5.0
    outro_message: str = "Subscribe so you don't miss the next chapter."
    """Big text on the outro card. Keep under 60 chars."""

    # Cold open
    cold_open_enabled: bool = True
    cold_open_duration_seconds: float = 6.0
    """How long the climax-panel teaser is held before the title card."""

    # Title-card line (rendered between cold-open and main script)
    title_card_enabled: bool = True
    title_card_duration_seconds: float = 2.5

    # Frame-zero thumbnail card. When enabled, a short still of the
    # chosen YouTube thumbnail is rendered at t=0 of the final video.
    # The user can then pause on their phone and screenshot it for
    # upload to YouTube Studio's thumbnail picker (so the saved
    # thumbnail.png never has to leave the device).
    thumbnail_card_enabled: bool = False
    thumbnail_card_duration_seconds: float = 1.5

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "ChannelPreset":
        if not data:
            return cls()
        # Filter to fields we know about so future JSON additions don't crash.
        known = {f for f in cls.__dataclass_fields__.keys()}
        return cls(**{k: v for k, v in data.items() if k in known})


class ChannelPresetService:
    """Read / write the global channel preset.

    The preset lives at `<data_dir>/channel_preset.json`. New installs
    auto-create it from defaults on first read so callers never have to
    handle a missing-file case.
    """

    FILENAME = "channel_preset.json"

    def __init__(self, settings: Any | None = None) -> None:
        self.settings = settings or get_settings()
        self._path = Path(self.settings.data_dir) / self.FILENAME

    def path(self) -> Path:
        return self._path

    def load(self) -> ChannelPreset:
        if not self._path.exists():
            preset = ChannelPreset()
            self.save(preset)
            return preset
        try:
            data = read_json(self._path)
            return ChannelPreset.from_dict(data if isinstance(data, dict) else None)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "channel_preset.json could not be read (%s); falling back to defaults.",
                exc,
            )
            return ChannelPreset()

    def save(self, preset: ChannelPreset) -> ChannelPreset:
        write_json(self._path, preset.to_dict())
        return preset

    def update(self, patch: dict[str, Any]) -> ChannelPreset:
        """Merge `patch` into the saved preset and return the new one."""
        current = self.load().to_dict()
        for key, value in (patch or {}).items():
            if key in current:
                current[key] = value
        next_preset = ChannelPreset.from_dict(current)
        return self.save(next_preset)
