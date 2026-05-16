import shutil
from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


REPO_ROOT = Path(__file__).resolve().parents[3]


def _resolve_binary(name: str) -> str:
    detected = shutil.which(name)
    if detected:
        return detected

    for candidate in (f"/opt/homebrew/bin/{name}", f"/usr/local/bin/{name}"):
        if Path(candidate).exists():
            return candidate
    return name


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=(REPO_ROOT / ".env", ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = "Panelia"
    app_env: str = "development"
    debug: bool = True
    frontend_origin: str = "http://localhost:3000"

    data_dir: Path = Field(default=Path(__file__).resolve().parents[2] / "data", validation_alias="PANELIA_DATA_DIR")
    training_data_dir: Path = Field(default=REPO_ROOT / "backend" / "training_data", validation_alias="PANELIA_TRAINING_DATA_DIR")
    ocr_training_data_dir: Path = Field(default=REPO_ROOT / "backend" / "training_data" / "ocr", validation_alias="PANELIA_OCR_TRAINING_DATA_DIR")
    panel_detector_models_dir: Path = Field(default=REPO_ROOT / "backend" / "models", validation_alias="PANELIA_PANEL_MODELS_DIR")
    panel_detector_checkpoint_glob: str = "panel_detector*.pt"
    panel_detector_score_threshold: float = 0.35
    panel_detector_training_min_new_annotations: int = 10
    panel_detector_auto_train_enabled: bool = False
    panel_reconstruction_enabled: bool = True
    panel_reconstruction_full_page_ocr_enabled: bool = Field(
        default=False,
        validation_alias="PANELIA_PANEL_RECONSTRUCTION_FULL_PAGE_OCR_ENABLED",
    )
    panel_reconstruction_text_margin_ratio: float = 0.08
    panel_reconstruction_overlap_threshold: float = 0.18
    panel_reconstruction_cluster_distance_ratio: float = 0.085
    cross_page_panel_merging_enabled: bool = True
    cross_page_merge_edge_ratio: float = 0.1
    cross_page_merge_similarity_threshold: float = 0.65
    panel_crop_margin_x_ratio: float = 0.16
    panel_crop_margin_top_ratio: float = 0.22
    panel_crop_margin_bottom_ratio: float = 0.18
    panel_assoc_margin_x_ratio: float = 0.20
    panel_assoc_margin_top_ratio: float = 0.26
    panel_assoc_margin_bottom_ratio: float = 0.20
    # Panel quality thresholds (prevent script generation if exceeded)
    panel_quality_score_manga: int = Field(
        default=55,
        validation_alias="PANELIA_PANEL_QUALITY_SCORE_MANGA",
        description="Quality score threshold for manga format (0-100)",
    )
    panel_quality_score_webtoon: int = Field(
        default=72,
        validation_alias="PANELIA_PANEL_QUALITY_SCORE_WEBTOON",
        description="Quality score threshold for webtoon format (0-100)",
    )
    narration_enhancement_enabled: bool = True
    narration_sentence_cache_workers: int = 2
    # Azure Speech Service (paid) - when both key + region are set,
    # the Edge TTS path routes calls to Azure first and only falls
    # back to the free Edge endpoint on Azure failure. Same voice
    # catalog (en-US-AvaNeural etc.) so audio character is identical.
    # NOTE the PANELIA_-prefixed env names; that's the convention in
    # this codebase (see data_dir / training_data_dir above).
    azure_speech_key: str = Field(default="", validation_alias="PANELIA_AZURE_SPEECH_KEY")
    azure_speech_region: str = Field(default="", validation_alias="PANELIA_AZURE_SPEECH_REGION")
    # Max concurrent Azure synth calls. Azure handles 16+ comfortably;
    # cap is here to protect the rest of the machine.
    azure_speech_max_workers: int = Field(default=16, validation_alias="PANELIA_AZURE_SPEECH_MAX_WORKERS")
    # How many panel clips to render concurrently in the per-panel ffmpeg
    # loop. h264_videotoolbox on Apple Silicon comfortably handles 4
    # simultaneous encoder sessions; libx264 on Intel/Linux also fits a
    # typical 4-core box. Going higher tends to oversubscribe. Override
    # via env var PANELIA_VIDEO_CLIP_RENDER_WORKERS.
    video_clip_render_workers: int = 6
    narration_mastering_enabled: bool = True
    openvoice_enabled: bool = False
    redis_url: str = "redis://redis:6379/0"

    mangadex_api_base: str = "https://api.mangadex.org"
    mangadex_public_base: str = "https://mangadex.org"
    mangadex_timeout_seconds: int = 30
    mangadex_retry_count: int = 3
    comix_api_base: str = "https://comix.to/api/v2"
    comix_public_base: str = "https://comix.to"
    comix_timeout_seconds: int = 30
    comix_retry_count: int = 3

    magi_enabled: bool = True
    magi_model_id: str = "ragavsachdeva/magiv2"
    magi_local_files_only: bool = Field(default=True, validation_alias="PANELIA_MAGI_LOCAL_FILES_ONLY")
    magi_batch_size: int = 1
    magi_max_image_edge: int = 1400
    magi_tall_page_ratio: float = 1.45
    magi_detect_webtoon_panels: bool = False
    magi_dialogue_ocr_enabled: bool = True
    gemini_api_key: str | None = None
    gemini_model: str = "gemini-2.5-flash-lite"
    llm_provider_order: str = "gemini,grok,deepseek"
    llm_gemini_model: str = "gemini-2.5-flash-lite"
    gemini_panel_rewrite_batch_size: int = 16
    gemini_panel_caption_batch_size: int = 12
    grok_api_key: str | None = None
    grok_api_base: str = "https://api.x.ai/v1"
    grok_model: str = "grok-2-mini"
    deepseek_api_key: str | None = None
    deepseek_api_base: str = "https://api.deepseek.com"
    deepseek_model: str = "deepseek-chat"
    clip_model_id: str = "openai/clip-vit-base-patch32"
    character_clip_sample_limit: int = 2500
    character_name_resolution_limit: int = 40
    anime_face_detection_enabled: bool = True
    anime_face_cascade_path: Path = Field(
        default=REPO_ROOT / "backend" / "assets" / "lbpcascade_animeface.xml",
        validation_alias="PANELIA_ANIME_FACE_CASCADE_PATH",
    )
    anime_face_max_image_edge: int = 1400
    anime_face_min_size: int = 24
    anime_face_scale_factor: float = 1.08
    anime_face_min_neighbors: int = 4
    apple_vision_ocr_enabled: bool = True
    comic_ocr_apple_vision_enabled: bool = Field(
        default=True,
        validation_alias="PANELIA_COMIC_OCR_APPLE_VISION_ENABLED",
    )

    kokoro_default_voice: str = "af_bella"
    kokoro_default_lang_code: str = "a"
    kokoro_sample_rate: int = 24000

    ffmpeg_binary: str = Field(default_factory=lambda: _resolve_binary("ffmpeg"))
    ffprobe_binary: str = Field(default_factory=lambda: _resolve_binary("ffprobe"))

    upload_max_mb: int = 250
    default_resolution: str = "1920x1080"
    default_orientation: str = "landscape"
    default_output_format: str = "mp4"
    default_fps: int = 24

    def ensure_data_dir(self) -> Path:
        try:
            self.data_dir.mkdir(parents=True, exist_ok=True)
            self.training_data_dir.mkdir(parents=True, exist_ok=True)
            self.ocr_training_data_dir.mkdir(parents=True, exist_ok=True)
            self.panel_detector_models_dir.mkdir(parents=True, exist_ok=True)
            return self.data_dir
        except OSError:
            fallback = Path(__file__).resolve().parents[2] / "data"
            fallback.mkdir(parents=True, exist_ok=True)
            self.data_dir = fallback
            self.training_data_dir = REPO_ROOT / "backend" / "training_data"
            self.training_data_dir.mkdir(parents=True, exist_ok=True)
            self.ocr_training_data_dir = self.training_data_dir / "ocr"
            self.ocr_training_data_dir.mkdir(parents=True, exist_ok=True)
            self.panel_detector_models_dir = REPO_ROOT / "backend" / "models"
            self.panel_detector_models_dir.mkdir(parents=True, exist_ok=True)
            return self.data_dir


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    settings.ffmpeg_binary = _resolve_binary(settings.ffmpeg_binary)
    settings.ffprobe_binary = _resolve_binary(settings.ffprobe_binary)
    settings.ensure_data_dir()
    return settings
