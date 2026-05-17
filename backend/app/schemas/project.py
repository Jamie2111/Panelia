from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, ClassVar, Literal

from pydantic import BaseModel, Field, field_validator


class SourceType(str, Enum):
    MANGADEX_URL = "mangadex_url"
    COMIX_TO_URL = "comix_to_url"
    ZIP = "zip"
    PDF = "pdf"
    IMAGES = "images"
    FOLDER = "folder"


class DuplicateHandlingMode(str, Enum):
    AUTO_PICK_BEST = "auto_pick_best"
    PREFER_OFFICIAL = "prefer_official"
    PREFER_FAN = "prefer_fan"
    PREFER_CONSISTENT_GROUP = "prefer_consistent_group"


class JobStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class StageStatus(str, Enum):
    PENDING = "pending"
    READY = "ready"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    NEEDS_REVIEW = "needs_review"


class PipelineStage(str, Enum):
    INGESTION = "ingestion"
    PANEL_DETECTION = "panel_detection"
    PANEL_REVIEW = "panel_review"
    # ── Vision-pipeline path - these four are legacy ─────────────────────
    # They run only when script_pipeline_version="legacy"/"vNext". In
    # "vision" mode they're auto-completed by auto_run.py so the UI can
    # collapse them.
    CHARACTER_REVIEW = "character_review"
    CHARACTER_PORTRAIT = "character_portrait"
    PANEL_VISION_EXTRACTION = "panel_vision_extraction"
    PANEL_VISION_QUALITY = "panel_vision_quality"
    # ── Universal path ───────────────────────────────────────────────────
    SCRIPT_GENERATION = "script_generation"
    NARRATION_GENERATION = "narration_generation"
    VIDEO_RENDERING = "video_rendering"
    # YouTube publish bundle: title, description, viral thumbnail - runs
    # last so the user can drag-and-drop the result into YouTube Studio.
    YOUTUBE_BUNDLE = "youtube_bundle"


class OutputFormat(str, Enum):
    MP4 = "mp4"
    MOV = "mov"


class Orientation(str, Enum):
    LANDSCAPE = "landscape"
    VERTICAL = "vertical"


class PanelLayout(str, Enum):
    CARD = "card"
    FULLSCREEN = "fullscreen"


class StageState(BaseModel):
    stage: PipelineStage
    status: StageStatus = StageStatus.PENDING
    progress: float = 0.0
    message: str | None = None
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class PanelBox(BaseModel):
    id: str
    page: int
    panel: int
    x: int
    y: int
    width: int
    height: int
    order: int
    keep: bool = True
    duration_seconds: float | None = None
    narration: str | None = None
    zoom_hint: str | None = None
    merged_from: list[str] = Field(default_factory=list)
    ocr_text: str | None = None
    text_detected: bool | None = None
    auto_skipped: bool = False
    skip_reason: str | None = None
    manual_keep: bool = False
    manual_narration: bool = False
    narration_locked: bool = False
    manual_ocr_text: bool = False
    visual_caption: str | None = None
    narration_source: str | None = None
    review_flags: list[str] = Field(default_factory=list)
    # ── Content safety (YouTube monetization gating) ─────────────────────
    # Populated by the vision narrator when content_safety_enabled. Drives
    # the per-panel blur / skip logic in video_service.py.
    content_rating: Literal["safe", "borderline", "explicit"] | None = None
    content_rating_reason: str | None = None
    content_blur: bool = False
    logical_panel_id: str | None = None
    multi_page_panel: bool = False
    spans_pages: list[int] = Field(default_factory=list)
    continuation_panel_ids: list[str] = Field(default_factory=list)
    reconstruction_source: str | None = None
    reconstruction_confidence: float | None = None
    detection_locked: bool = False

    # Sanity bounds: coordinates and sizes beyond these indicate bad data
    _MAX_COORD: ClassVar[int] = 65535   # ~4K resolution * 16 zoom - no real panel exceeds this
    _MAX_DIMENSION: ClassVar[int] = 32767

    @field_validator("x", "y", mode="before")
    @classmethod
    def _coerce_position_to_int(cls, value: Any) -> int:
        if isinstance(value, (int, float)):
            v = int(round(value))
        else:
            v = int(value)
        # Clamp to valid coordinate range - reject injected extreme values
        return max(-cls._MAX_COORD, min(cls._MAX_COORD, v))

    @field_validator("width", "height", mode="before")
    @classmethod
    def _coerce_size_to_int(cls, value: Any) -> int:
        if isinstance(value, (int, float)):
            v = max(1, int(round(value)))
        else:
            v = max(1, int(value))
        # Clamp dimensions to prevent resource exhaustion via absurd values
        return min(cls._MAX_DIMENSION, v)


class VoiceConfig(BaseModel):
    # Default to Edge TTS (Microsoft Azure Neural). It's free, requires no
    # API key, and sounds dramatically more human than Kokoro. Per-sentence
    # retries inside edge_tts_service handle transient 503s; if Edge truly
    # fails after retries, the narration job FAILS LOUDLY (no silent
    # voice swap mid-video). Set allow_kokoro_fallback=True to revive
    # the old auto-fallback behavior at the cost of voice consistency.
    voice: str = "edge_ava"
    lang_code: str = "a"
    speed: float = 1.0
    allow_kokoro_fallback: bool = False


class MusicConfig(BaseModel):
    enabled: bool = True
    track_name: str | None = None
    volume: float = 0.14
    fade_in_seconds: float = 1.0
    fade_out_seconds: float = 2.0
    sfx_enabled: bool = True     # Procedural transition whooshes at panel boundaries
    sfx_volume: float = 0.055    # Mixed at ~-25 dB relative to full scale (subtle)


class WatermarkPosition(str, Enum):
    BOTTOM_RIGHT = "bottom_right"
    BOTTOM_LEFT = "bottom_left"
    TOP_RIGHT = "top_right"
    TOP_LEFT = "top_left"


class WatermarkConfig(BaseModel):
    enabled: bool = False
    image_path: str | None = None          # Relative path inside project dir, or absolute
    position: WatermarkPosition = WatermarkPosition.BOTTOM_RIGHT
    opacity: float = 0.45                  # 0.0 (invisible) → 1.0 (opaque)
    scale: float = 0.04                    # Fraction of video width (0.03 = 3%)
    margin_px: int = 24                    # Pixels from the nearest edges
    fade_in_seconds: float = 0.5           # Fade in at start of video
    fade_out_seconds: float = 0.5          # Fade out at end of video


class VideoConfig(BaseModel):
    width: int = 1920
    height: int = 1080
    orientation: Orientation = Orientation.LANDSCAPE
    panel_layout: PanelLayout = PanelLayout.CARD
    intro_thumbnail_enabled: bool = True
    intro_thumbnail_seconds: float = 1.5
    output_format: OutputFormat = OutputFormat.MP4
    fps: int = 24
    background_color: str = "#09090b"
    watermark: WatermarkConfig = Field(default_factory=WatermarkConfig)
    title_card_enabled: bool = False
    title_card_seconds: float = 3.0    # Duration of title card in seconds
    title_card_accent_color: str = "#e11d48"  # Rose-600 - matches Panelia UI accent
    # Audio-sync per-panel fix (commit add849d, Part D1-D3). When True
    # each panel's hold duration matches its specific audio fragment
    # duration, so narration starts exactly when the panel appears.
    # When False, audio is divided uniformly across panels in a segment
    # (legacy behavior — produces a ~1s lead-in on multi-panel segments
    # but preserves panel-clip cache compatibility with renders done
    # before the fix). Set False to reuse cached clips on a re-render.
    audio_sync_per_panel: bool = True


class NarrationMode(str, Enum):
    PANEL = "panel"   # One narration line per kept panel - maximum alignment / control


class PipelineConfig(BaseModel):
    # Default new projects to end-to-end auto-run so the user can paste a
    # URL and walk away with a ready-to-publish bundle. Existing projects
    # keep whatever value is already saved.
    auto_run_end_to_end: bool = True
    narration_mode: NarrationMode = NarrationMode.PANEL
    # Default for new projects: the vision-grounded narration pipeline.
    # Existing projects keep whatever value is already saved in metadata.
    script_pipeline_version: str = "vision"
    # When True (default), the vision narrator classifies every panel
    # and the renderer blurs / skips panels that would demonetize a
    # YouTube video. Turn off for adult-only channels or testing.
    content_safety_enabled: bool = True

    @field_validator("narration_mode", mode="before")
    @classmethod
    def _coerce_narration_mode(cls, v: object) -> object:
        # Legacy projects may have "story", "hybrid", or "vision_first" stored.
        # All non-panel modes are retired - coerce them to panel silently.
        if v not in (NarrationMode.PANEL, NarrationMode.PANEL.value):
            return NarrationMode.PANEL
        return v

    @field_validator("script_pipeline_version", mode="before")
    @classmethod
    def _coerce_script_pipeline_version(cls, v: object) -> str:
        value = str(v or "vision").strip().casefold()
        if value == "vision":
            return "vision"
        if value == "vnext":
            return "vNext"
        return "legacy"


class ChapterMetadata(BaseModel):
    chapter_id: str | None = None
    source_url: str | None = None
    manga_title: str | None = None
    chapter_title: str | None = None
    chapter_number: str | None = None
    volume_number: str | None = None
    language: str | None = None
    page_count: int | None = None
    raw: dict[str, Any] = Field(default_factory=dict)


class StorySegment(BaseModel):
    id: str
    order: int
    text: str = ""
    keep: bool = True
    panel_ids: list[str] = Field(default_factory=list)
    panel_start: int | None = None
    panel_end: int | None = None
    scene_id: int | None = None
    title: str | None = None
    representative_panel_id: str | None = None
    visual_only: bool = False
    suppression_reason: str | None = None


class CanonicalCharacterRecord(BaseModel):
    stable_id: str
    name: str
    role: str = "supporting"
    visual_description: str = ""
    portrait_panel_ids: list[str] = Field(default_factory=list)
    portrait_pages: list[int] = Field(default_factory=list)
    confidence: float | None = None
    aliases: list[str] = Field(default_factory=list)


class CharacterPortraitsUpdateRequest(BaseModel):
    characters: list[CanonicalCharacterRecord] = Field(default_factory=list)


class PanelVisionRecord(BaseModel):
    panel_id: str
    panel_order: int
    page: int
    speaker: str = "unknown"
    dialogue: str = ""
    caption: str = ""
    action_beat: str = ""
    emotion: str = ""
    scene_change: bool = False
    confidence: float = 0.0
    character_names: list[str] = Field(default_factory=list)
    character_roles: dict[str, list[str]] = Field(default_factory=dict)
    visual_only: bool = False
    suppression_reason: str | None = None


class AudioFile(BaseModel):
    panel_id: str
    path: str
    url: str
    duration_seconds: float


class VideoFile(BaseModel):
    name: str
    path: str
    url: str
    width: int
    height: int
    output_format: OutputFormat
    created_at: datetime
    duration_seconds: float | None = None


class JobRecord(BaseModel):
    id: str
    project_id: str
    stage: PipelineStage
    status: JobStatus
    progress: float = 0.0
    message: str | None = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    started_at: datetime | None = None
    finished_at: datetime | None = None
    error: str | None = None
    payload: dict[str, Any] = Field(default_factory=dict)


class ProjectSummary(BaseModel):
    id: str
    name: str
    source_type: SourceType
    source_reference: str | None = None
    created_at: datetime
    updated_at: datetime
    chapter_metadata: ChapterMetadata = Field(default_factory=ChapterMetadata)
    stage_states: dict[PipelineStage, StageState]
    page_count: int = 0
    panel_count: int = 0
    kept_panel_count: int = 0
    thumbnail_url: str | None = None
    video_thumbnail_url: str | None = None
    active_jobs: list[JobRecord] = Field(default_factory=list)
    latest_video: VideoFile | None = None
    voice_config: VoiceConfig = Field(default_factory=VoiceConfig)
    video_config: VideoConfig = Field(default_factory=VideoConfig)
    music_config: MusicConfig = Field(default_factory=MusicConfig)
    pipeline_config: PipelineConfig = Field(default_factory=PipelineConfig)


class ProjectDetail(ProjectSummary):
    panels: list[PanelBox] = Field(default_factory=list)
    script_lines: list[str] = Field(default_factory=list)
    script_story: str | None = None
    story_segments: list[StorySegment] = Field(default_factory=list)
    script_display_metadata: dict[str, Any] = Field(default_factory=dict)
    audio_files: list[AudioFile] = Field(default_factory=list)
    videos: list[VideoFile] = Field(default_factory=list)
    available_music_tracks: list[dict[str, Any]] = Field(default_factory=list)


class PanelUpdateRequest(BaseModel):
    panels: list[PanelBox]


class ScriptUpdateRequest(BaseModel):
    script_lines: list[str] = Field(default_factory=list)
    story_segments: list[StorySegment] = Field(default_factory=list)
    panel_keeps: dict[str, bool] = Field(default_factory=dict)
    panel_narrations: dict[str, str] = Field(default_factory=dict)
    panel_locks: dict[str, bool] = Field(default_factory=dict)


class PanelRewriteMode(str, Enum):
    BALANCED = "balanced"
    CLOSER_TO_OCR = "closer_to_ocr"
    SHORTEN = "shorten"


class PanelRewriteRequest(BaseModel):
    panel_id: str
    mode: PanelRewriteMode = PanelRewriteMode.BALANCED
    current_narration: str | None = None


class PanelRewriteResponse(BaseModel):
    panel_id: str
    narration: str
    mode: PanelRewriteMode


class QueueStageRequest(BaseModel):
    stage: PipelineStage
    payload: dict[str, Any] = Field(default_factory=dict)


class RewindStageRequest(BaseModel):
    stage: PipelineStage


class DuplicateProjectRequest(BaseModel):
    name: str | None = None
    video_name: str | None = None
    copy_all_videos: bool = False


class ProjectRenameRequest(BaseModel):
    name: str = Field(min_length=1, max_length=160)


class BatchProjectRequest(BaseModel):
    urls: list[str]
    base_name: str | None = None


class MergeVideosRequest(BaseModel):
    project_id: str
    video_paths: list[str]
    output_name: str
    video_config: VideoConfig = Field(default_factory=VideoConfig)


class LanguageOption(BaseModel):
    code: str
    label: str
    description: str
    sample_text: str


class VoiceOption(BaseModel):
    id: str
    lang_code: str
    label: str
    description: str
    quality_note: str | None = None
    style_tags: list[str] = Field(default_factory=list)


class CatalogOptionsResponse(BaseModel):
    languages: list[LanguageOption] = Field(default_factory=list)
    voices: list[VoiceOption] = Field(default_factory=list)
    music_tracks: list[dict[str, Any]] = Field(default_factory=list)


class ProjectSettingsUpdateRequest(BaseModel):
    voice_config: VoiceConfig | None = None
    music_config: MusicConfig | None = None
    video_config: VideoConfig | None = None
    pipeline_config: PipelineConfig | None = None
