from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


CharacterReviewStatus = Literal["suggested", "confirmed", "unknown"]


class CharacterReviewSample(BaseModel):
    sample_id: str
    image_url: str | None = None
    image_path: str | None = None
    panel_id: str | None = None
    page: int | None = None
    panel: int | None = None
    bbox: list[int] = Field(default_factory=list)


class CharacterReviewIdentity(BaseModel):
    review_id: str
    stable_character_ids: list[str] = Field(default_factory=list)
    source_character_ids: list[str] = Field(default_factory=list)
    suggested_name: str | None = None
    remembered_name: str | None = None
    memory_matches: list[str] = Field(default_factory=list)
    name: str | None = None
    status: CharacterReviewStatus = "suggested"
    role_hint: str | None = None
    appearance_count: int = 0
    pages: list[int] = Field(default_factory=list)
    panel_ids: list[str] = Field(default_factory=list)
    sample_images: list[CharacterReviewSample] = Field(default_factory=list)
    notes: str | None = None


class CharacterReviewState(BaseModel):
    analysis_version: int = 1
    project_id: str
    series_key: str
    protagonist_name: str | None = None
    memory_names: list[str] = Field(default_factory=list)
    identities: list[CharacterReviewIdentity] = Field(default_factory=list)
    generated_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class CharacterReviewUpdateRequest(BaseModel):
    protagonist_name: str | None = None
    identities: list[CharacterReviewIdentity] = Field(default_factory=list)
