"""Channel preset endpoints - global branding the user edits once."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.services.channel_preset_service import ChannelPreset, ChannelPresetService

router = APIRouter(prefix="/api/channel", tags=["channel"])


class ChannelPresetPayload(BaseModel):
    """Loose patch shape - every field is optional so the UI can PATCH
    one value at a time without re-sending the whole preset."""
    channel_name: str | None = None
    tagline: str | None = None
    accent_color: str | None = None
    title_font: str | None = None
    watermark_enabled: bool | None = None
    watermark_text: str | None = None
    outro_enabled: bool | None = None
    outro_duration_seconds: float | None = None
    outro_message: str | None = None
    cold_open_enabled: bool | None = None
    cold_open_duration_seconds: float | None = None
    title_card_enabled: bool | None = None
    title_card_duration_seconds: float | None = None


@router.get("/preset")
def get_channel_preset() -> dict[str, Any]:
    """Return the current channel preset. Auto-creates with defaults on
    first call so the UI can render a populated form."""
    return ChannelPresetService().load().to_dict()


@router.put("/preset")
def update_channel_preset(payload: ChannelPresetPayload) -> dict[str, Any]:
    """Merge any non-null fields from the payload into the saved preset."""
    try:
        patch = {k: v for k, v in payload.model_dump().items() if v is not None}
        return ChannelPresetService().update(patch).to_dict()
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc))
