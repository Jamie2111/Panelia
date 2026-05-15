from __future__ import annotations

from io import BytesIO
import logging
import re
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response
from PIL import Image

from app.core.config import get_settings
from app.pipeline.auto_run import continue_auto_run_pipeline
from app.pipeline.orchestration import queue_stage_once
from app.schemas.character_identity import CharacterReviewState, CharacterReviewUpdateRequest
from app.schemas.project import (
    BatchProjectRequest,
    CharacterPortraitsUpdateRequest,
    DuplicateProjectRequest,
    DuplicateHandlingMode,
    JobStatus,
    MergeVideosRequest,
    MusicConfig,
    OutputFormat,
    Orientation,
    PanelBox,
    PanelUpdateRequest,
    PipelineConfig,
    PipelineStage,
    PanelRewriteRequest,
    PanelRewriteResponse,
    ProjectRenameRequest,
    ProjectSettingsUpdateRequest,
    QueueStageRequest,
    RewindStageRequest,
    ScriptUpdateRequest,
    SourceType,
    StageStatus,
    VideoConfig,
    VoiceConfig,
)
from app.services.character_review_service import CharacterReviewService
from app.services.detector_training_service import DetectorTrainingService
from app.services.comix_to import ComixToService
from app.services.mangadex import MangaDexService
from app.services.panel_narrator import PanelNarrator
from app.services.panel_training_annotations import changed_annotation_pages_for_detector_training
from app.services.project_store import ProjectStore
from app.services.queue_service import QueueService
from app.services.video_service import VideoRenderService
from app.services.llm_router import LLMRouter
from app.utils.files import read_json, write_json
from training.save_ocr_annotation import save_ocr_annotation
from training.save_panel_annotation import save_panel_annotation

router = APIRouter(prefix="/api/projects", tags=["projects"])

store = ProjectStore()
queue = QueueService()
video_service = VideoRenderService()
character_review_service = CharacterReviewService()
mangadex_service = MangaDexService()
comix_service = ComixToService()
logger = logging.getLogger(__name__)
settings = get_settings()

REWINDABLE_STAGES = {
    PipelineStage.INGESTION,
    PipelineStage.PANEL_DETECTION,
    PipelineStage.PANEL_REVIEW,
    PipelineStage.CHARACTER_REVIEW,
    PipelineStage.CHARACTER_PORTRAIT,
    PipelineStage.PANEL_VISION_EXTRACTION,
    PipelineStage.PANEL_VISION_QUALITY,
    PipelineStage.SCRIPT_GENERATION,
    PipelineStage.NARRATION_GENERATION,
}



def _changed_annotation_pages(before_panels: list[PanelBox], after_panels: list[PanelBox]) -> dict[int, list[PanelBox]]:
    return changed_annotation_pages_for_detector_training(before_panels, after_panels)


def _changed_ocr_panels(before_panels: list[PanelBox], after_panels: list[PanelBox]) -> list[tuple[PanelBox, PanelBox | None]]:
    before_by_id = {str(panel.id): panel for panel in before_panels}
    changed: list[tuple[PanelBox, PanelBox | None]] = []
    for panel in after_panels:
        corrected_text = str(panel.ocr_text or "").strip()
        if not panel.manual_ocr_text or not corrected_text:
            continue
        previous = before_by_id.get(str(panel.id))
        previous_text = str((previous.ocr_text if previous is not None else "") or "").strip()
        previous_manual = bool(previous.manual_ocr_text) if previous is not None else False
        if previous is not None and previous_manual and previous_text == corrected_text:
            continue
        if previous is not None and not previous_manual and previous_text == corrected_text:
            continue
        changed.append((panel, previous))
    return changed


def _save_human_panel_training_examples(project_id: str, before_panels: list[PanelBox], after_panels: list[PanelBox]) -> None:
    changed_pages = _changed_annotation_pages(before_panels, after_panels)
    if not changed_pages:
        return

    project = store.get_project(project_id)
    page_paths = store.list_page_paths(project_id)
    for page_number, corrected_panels in changed_pages.items():
        page_index = int(page_number) - 1
        if page_index < 0 or page_index >= len(page_paths):
            continue
        page_path = page_paths[page_index]
        try:
            save_panel_annotation(
                page_path,
                corrected_panels,
                image_name=f"{project_id}_page_{page_number:04d}.png",
                metadata={
                    "project_id": project.id,
                    "project_name": project.name,
                    "page": int(page_number),
                    "source_type": project.source_type.value,
                    "language": project.chapter_metadata.language,
                    "manga_title": project.chapter_metadata.manga_title,
                    "chapter_number": project.chapter_metadata.chapter_number,
                    "chapter_title": project.chapter_metadata.chapter_title,
                    "annotation_source": "panel_editor",
                },
            )
        except Exception:
            logger.exception("Failed to save human panel annotation for %s page %s", project_id, page_number)


def _save_human_ocr_training_examples(project_id: str, before_panels: list[PanelBox], after_panels: list[PanelBox]) -> None:
    changed_panels = _changed_ocr_panels(before_panels, after_panels)
    if not changed_panels:
        return

    project = store.get_project(project_id)
    page_paths = store.list_page_paths(project_id)
    for panel, previous in changed_panels:
        page_index = int(panel.page) - 1
        if page_index < 0 or page_index >= len(page_paths):
            continue
        page_path = page_paths[page_index]
        try:
            save_ocr_annotation(
                page_path,
                panel,
                corrected_text=str(panel.ocr_text or "").strip(),
                original_text=str((previous.ocr_text if previous is not None else "") or "").strip(),
                image_name=f"{project_id}_{panel.id}_ocr.png",
                metadata={
                    "project_id": project.id,
                    "project_name": project.name,
                    "page": int(panel.page),
                    "panel_id": panel.id,
                    "panel_order": int(panel.order),
                    "panel_number": int(panel.panel),
                    "language": project.chapter_metadata.language,
                    "manga_title": project.chapter_metadata.manga_title,
                    "chapter_number": project.chapter_metadata.chapter_number,
                    "chapter_title": project.chapter_metadata.chapter_title,
                    "annotation_source": "manual_ocr_override",
                },
            )
        except Exception:
            logger.exception("Failed to save human OCR annotation for %s panel %s", project_id, panel.id)


def _cancel_active_jobs(
    project_id: str,
    stages: set[PipelineStage],
    *,
    job_message: str = "Cancelled while rewinding the pipeline",
    stage_message: str = "Cancelled while rewinding the pipeline",
) -> None:
    project = store.get_project(project_id)
    for job in project.active_jobs:
        if job.stage not in stages:
            continue
        queue.request_cancel(job.id)
        store.update_job(
            project_id,
            job.id,
            status="cancelled",
            finished_at=store._now().isoformat(),
            message=job_message,
        )
        store.update_stage_state(project_id, job.stage, StageStatus.CANCELLED, progress=job.progress, message=stage_message)


def _default_video_config(
    resolution: str | None,
    orientation: str | None,
    output_format: str | None,
) -> VideoConfig:
    width, height = (resolution or "1920x1080").split("x")
    base_width = int(width)
    base_height = int(height)
    resolved_orientation = Orientation(orientation or "landscape")
    if resolved_orientation == Orientation.VERTICAL:
        base_width, base_height = min(base_width, base_height), max(base_width, base_height)
    else:
        base_width, base_height = max(base_width, base_height), min(base_width, base_height)
    return VideoConfig(
        width=base_width,
        height=base_height,
        orientation=resolved_orientation,
        output_format=OutputFormat(output_format or "mp4"),
    )


@router.get("")
def list_projects():
    return store.list_projects()


@router.get("/{project_id}/summary")
def get_project_summary(project_id: str):
    try:
        return store.get_project_summary(project_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/{project_id}")
def get_project(project_id: str):
    try:
        return store.get_project(project_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


def _download_filename(project_name: str, extension: str) -> str:
    cleaned = re.sub(r'[\\/:*?"<>|]+', "", project_name).strip().rstrip(".")
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    base = cleaned or "Panelia Export"
    ext = extension.lstrip(".").lower() or "mp4"
    return f"{base}.{ext}"


@router.get("/{project_id}/panels/{panel_id}/preview")
def get_panel_preview(project_id: str, panel_id: str):
    if not store.project_exists(project_id):
        raise HTTPException(status_code=404, detail="Project not found")
    panels = store.load_panels(project_id)
    panel = next((item for item in panels if item.id == panel_id), None)
    if panel is None:
        raise HTTPException(status_code=404, detail="Panel not found")
    saved_preview = store._panel_previews_dir(project_id) / f"panel_{int(panel.order):03d}.png"
    if saved_preview.exists():
        return FileResponse(saved_preview, media_type="image/png")
    panel = store.sanitize_panel_box(project_id, panel)
    page_paths = store.list_page_paths(project_id)
    page_index = max(panel.page - 1, 0)
    if page_index >= len(page_paths):
        raise HTTPException(status_code=404, detail="Page image not found")

    with Image.open(page_paths[page_index]) as image:
        source = image.convert("RGB")
        left = max(0, int(panel.x))
        top = max(0, int(panel.y))
        right = min(source.width, int(panel.x + panel.width))
        bottom = min(source.height, int(panel.y + panel.height))
        if right <= left or bottom <= top:
            raise HTTPException(status_code=422, detail="Panel crop is invalid for this page")
        crop = source.crop((left, top, right, bottom))
        buffer = BytesIO()
        crop.save(buffer, format="PNG", compress_level=1)
    return Response(content=buffer.getvalue(), media_type="image/png")


@router.get("/{project_id}/video/latest-download")
def download_latest_video(project_id: str):
    if not store.project_exists(project_id):
        raise HTTPException(status_code=404, detail="Project not found")
    project = store.get_project(project_id)
    latest_video = project.latest_video or (project.videos[-1] if project.videos else None)
    if latest_video is None:
        raise HTTPException(status_code=404, detail="No finished video found")
    video_path = Path(latest_video.path)
    if not video_path.exists():
        raise HTTPException(status_code=404, detail="Video file not found")
    filename = _download_filename(project.name, video_path.suffix or latest_video.output_format.value)
    media_type = "video/mp4" if video_path.suffix.lower() == ".mp4" else "video/quicktime"
    return FileResponse(video_path, media_type=media_type, filename=filename)


@router.post("/{project_id}/video-thumbnail")
async def upload_video_thumbnail(project_id: str, file: UploadFile = File(...)):
    if not store.project_exists(project_id):
        raise HTTPException(status_code=404, detail="Project not found")
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Upload an image file for the video thumbnail.")

    _MAX_THUMBNAIL_BYTES = 20 * 1024 * 1024  # 20 MB
    payload = await file.read(_MAX_THUMBNAIL_BYTES + 1)
    if len(payload) > _MAX_THUMBNAIL_BYTES:
        raise HTTPException(status_code=413, detail="Thumbnail file exceeds 20 MB limit.")
    try:
        with Image.open(BytesIO(payload)) as source:
            image = source.convert("RGB")
            image.thumbnail((2048, 2048))
            buffer = BytesIO()
            image.save(buffer, format="JPEG", quality=92)
    except Exception as exc:
        raise HTTPException(status_code=400, detail="The uploaded file is not a valid image.") from exc

    store.write_video_thumbnail(project_id, buffer.getvalue())
    return store.get_project(project_id)


@router.delete("/{project_id}/video-thumbnail")
def delete_video_thumbnail(project_id: str):
    if not store.project_exists(project_id):
        raise HTTPException(status_code=404, detail="Project not found")
    store.delete_video_thumbnail(project_id)
    return store.get_project(project_id)


@router.post("/{project_id}/watermark")
async def upload_watermark(project_id: str, file: UploadFile = File(...)):
    """Upload a PNG logo to use as a video watermark overlay."""
    if not store.project_exists(project_id):
        raise HTTPException(status_code=404, detail="Project not found")
    if not file.content_type or file.content_type not in {"image/png", "image/webp", "image/jpeg"}:
        raise HTTPException(status_code=400, detail="Upload a PNG, WebP, or JPEG image for the watermark.")

    _MAX_WM_BYTES = 5 * 1024 * 1024  # 5 MB
    payload = await file.read(_MAX_WM_BYTES + 1)
    if len(payload) > _MAX_WM_BYTES:
        raise HTTPException(status_code=413, detail="Watermark file exceeds 5 MB limit.")
    try:
        with Image.open(BytesIO(payload)) as source:
            # Convert to RGBA to preserve transparency
            image = source.convert("RGBA")
            image.thumbnail((512, 512))
            buffer = BytesIO()
            image.save(buffer, format="PNG")
    except Exception as exc:
        raise HTTPException(status_code=400, detail="The uploaded file is not a valid image.") from exc

    project_dir = store._project_dir(project_id)
    wm_path = project_dir / "watermark.png"
    wm_path.write_bytes(buffer.getvalue())

    # Update video_config.watermark to reference the new file
    project = store.get_project(project_id)
    wm_cfg = project.video_config.watermark.model_copy(update={"image_path": "watermark.png", "enabled": True})
    new_video_cfg = project.video_config.model_copy(update={"watermark": wm_cfg})
    store.update_project_metadata(project_id, video_config=new_video_cfg.model_dump(mode="json"))
    return store.get_project(project_id)


@router.delete("/{project_id}/watermark")
def delete_watermark(project_id: str):
    """Remove the watermark image from this project."""
    if not store.project_exists(project_id):
        raise HTTPException(status_code=404, detail="Project not found")
    project_dir = store._project_dir(project_id)
    (project_dir / "watermark.png").unlink(missing_ok=True)
    project = store.get_project(project_id)
    from app.schemas.project import WatermarkConfig
    wm_cfg = WatermarkConfig(enabled=False, image_path=None)
    new_video_cfg = project.video_config.model_copy(update={"watermark": wm_cfg})
    store.update_project_metadata(project_id, video_config=new_video_cfg.model_dump(mode="json"))
    return store.get_project(project_id)


@router.post("")
async def create_project(
    name: str = Form(...),
    source_type: SourceType = Form(...),
    mangadex_url: str | None = Form(default=None),
    comix_url: str | None = Form(default=None),
    chapter_range: str | None = Form(default=None),
    source_language: str | None = Form(default=None),
    duplicate_mode: DuplicateHandlingMode = Form(default=DuplicateHandlingMode.AUTO_PICK_BEST),
    files: list[UploadFile] | None = File(default=None),
    voice: str = Form(default="af_bella"),
    lang_code: str = Form(default="a"),
    speed: float = Form(default=1.0),
    resolution: str = Form(default="1920x1080"),
    orientation: str = Form(default="landscape"),
    output_format: str = Form(default="mp4"),
    music_enabled: bool = Form(default=True),
    music_track: str | None = Form(default=None),
    music_volume: float = Form(default=0.14),
):
    if source_type == SourceType.MANGADEX_URL and not mangadex_url:
        raise HTTPException(status_code=400, detail="A MangaDex URL is required for this source type.")
    if source_type == SourceType.COMIX_TO_URL and not comix_url:
        raise HTTPException(status_code=400, detail="A comix.to URL is required for this source type.")
    files = files or []

    if source_type not in {SourceType.MANGADEX_URL, SourceType.COMIX_TO_URL} and not files:
        raise HTTPException(status_code=400, detail="Please upload at least one file.")

    source_reference = mangadex_url if source_type == SourceType.MANGADEX_URL else comix_url
    try:
        if source_type == SourceType.MANGADEX_URL and mangadex_url:
            urls = [entry.strip() for entry in mangadex_url.splitlines() if entry.strip()]
            try:
                source_reference = "\n".join(
                    mangadex_service.resolve_import_urls(
                        urls,
                        chapter_range=chapter_range,
                        preferred_language=source_language,
                        duplicate_mode=duplicate_mode.value,
                    )
                )
            except Exception as exc:
                logger.warning(f"Failed to resolve MangaDex URLs, using raw URLs: {exc}")
                source_reference = mangadex_url
        elif source_type == SourceType.COMIX_TO_URL and comix_url:
            urls = [entry.strip() for entry in comix_url.splitlines() if entry.strip()]
            try:
                source_reference = "\n".join(
                    comix_service.resolve_import_urls(
                        urls,
                        chapter_range=chapter_range,
                        preferred_language=source_language,
                        duplicate_mode=duplicate_mode.value,
                    )
                )
            except Exception as exc:
                logger.warning(f"Failed to resolve comix.to URLs, using raw URLs: {exc}")
                source_reference = comix_url
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    project = store.create_project(name=name, source_type=source_type, source_reference=source_reference)
    if files:
        from app.services.ingestion import PageIngestionService

        ingestion = PageIngestionService(store)
        ingestion.save_upload_sources(project.id, files)

    store.update_project_metadata(
        project.id,
        voice_config=VoiceConfig(voice=voice, lang_code=lang_code, speed=speed).model_dump(mode="json"),
        video_config=_default_video_config(resolution, orientation, output_format).model_dump(mode="json"),
        music_config=MusicConfig(enabled=music_enabled, track_name=music_track, volume=music_volume).model_dump(mode="json"),
        pipeline_config=PipelineConfig().model_dump(mode="json"),
    )
    store.update_stage_state(project.id, PipelineStage.INGESTION, StageStatus.READY, progress=0, message="Queued for page import")

    job = store.create_job(project.id, PipelineStage.INGESTION)
    queue.enqueue(project.id, job.id, PipelineStage.INGESTION.value)
    return store.get_project(project.id)


@router.post("/batch")
def create_projects_batch(payload: BatchProjectRequest):
    urls = [url.strip() for url in payload.urls if url.strip()]
    if not urls:
        raise HTTPException(status_code=400, detail="Provide at least one MangaDex URL.")

    project = store.create_project(
        name=payload.base_name or "Combined MangaDex import",
        source_type=SourceType.MANGADEX_URL,
        source_reference="\n".join(urls),
    )
    store.update_stage_state(project.id, PipelineStage.INGESTION, StageStatus.READY, progress=0, message="Queued for page import")
    job = store.create_job(project.id, PipelineStage.INGESTION)
    queue.enqueue(project.id, job.id, PipelineStage.INGESTION.value)
    return [store.get_project(project.id)]


@router.post("/{project_id}/duplicate")
def duplicate_project(project_id: str, payload: DuplicateProjectRequest):
    if not store.project_exists(project_id):
        raise HTTPException(status_code=404, detail="Project not found")
    try:
        return store.duplicate_project(
            project_id,
            name=payload.name,
            video_name=payload.video_name,
            copy_all_videos=payload.copy_all_videos,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.put("/{project_id}/panels")
def update_panels(project_id: str, payload: PanelUpdateRequest):
    if not store.project_exists(project_id):
        raise HTTPException(status_code=404, detail="Project not found")
    existing_panels = store.load_panels(project_id)
    _cancel_active_jobs(
        project_id,
        {
            PipelineStage.CHARACTER_REVIEW,
            PipelineStage.CHARACTER_PORTRAIT,
            PipelineStage.PANEL_VISION_EXTRACTION,
            PipelineStage.PANEL_VISION_QUALITY,
            PipelineStage.SCRIPT_GENERATION,
            PipelineStage.NARRATION_GENERATION,
            PipelineStage.VIDEO_RENDERING,
        },
    )
    # Mark pages the user edited as detection_locked so corrections survive re-detection
    changed_page_numbers = set(_changed_annotation_pages(existing_panels, payload.panels).keys())
    panels_to_save = [
        panel.model_copy(update={"detection_locked": True})
        if int(panel.page) in changed_page_numbers
        else panel
        for panel in payload.panels
    ]
    store.save_panels(project_id, panels_to_save)
    output_dir = store._project_dir(project_id) / "output"
    for filename in (
        "canonical_characters.json",
        "panel_evidence.json",
        "panel_vision.json",
        "panel_vision_final.json",
        "ocr_audit.json",
    ):
        (output_dir / filename).unlink(missing_ok=True)
    store.invalidate_script_outputs(project_id, clear_generated_panel_narration=True)

    # Save training examples in background to avoid blocking the response
    def save_training_examples():
        try:
            _save_human_panel_training_examples(project_id, existing_panels, panels_to_save)
            _save_human_ocr_training_examples(project_id, existing_panels, panels_to_save)
        except Exception:
            logger.exception("Failed to save human training examples")

    queue.enqueue(save_training_examples)

    # Detector training is intentionally manual by default because background
    # training can make the laptop sluggish right after a panel-review save.
    try:
        _trainer = DetectorTrainingService()
        _training_status = _trainer.get_status()
        if settings.panel_detector_auto_train_enabled and _training_status.ready_to_train:
            _trainer.start_training()
            logger.info(
                "Auto-triggered panel detector training: %d new annotations",
                _training_status.new_panel_annotations,
            )
    except Exception:
        logger.exception("Failed to auto-trigger panel detector training")

    store.update_stage_state(project_id, PipelineStage.PANEL_REVIEW, StageStatus.COMPLETED, progress=100, message="Panel review saved")
    store.update_stage_state(project_id, PipelineStage.CHARACTER_REVIEW, StageStatus.READY, progress=0, message="Character review is ready whenever you want it.")
    store.update_stage_state(project_id, PipelineStage.SCRIPT_GENERATION, StageStatus.READY, progress=0, message="Panel review saved. Generate the script when you're ready.")
    store.update_stage_state(project_id, PipelineStage.NARRATION_GENERATION, StageStatus.PENDING, progress=0, message="Generate a script before creating audio.")
    store.update_stage_state(project_id, PipelineStage.VIDEO_RENDERING, StageStatus.PENDING, progress=0, message="Generate audio before rendering video.")
    continue_auto_run_pipeline(store, queue, project_id, source="panel review")
    return {"status": "ok", "project_id": project_id}


@router.post("/{project_id}/panels/pages/{page}/unlock")
def unlock_page_detection(project_id: str, page: int):
    """Clear detection_locked on all panels for a page so the next detection run re-detects it."""
    if not store.project_exists(project_id):
        raise HTTPException(status_code=404, detail="Project not found")
    panels = store.load_panels(project_id)
    updated = [
        p.model_copy(update={"detection_locked": False}) if int(p.page) == page else p
        for p in panels
    ]
    store.save_panels(project_id, updated)
    return {
        "unlocked_page": page,
        "panel_count": sum(1 for p in updated if int(p.page) == page),
    }


@router.put("/{project_id}/script")
def update_script(project_id: str, payload: ScriptUpdateRequest):
    if not store.project_exists(project_id):
        raise HTTPException(status_code=404, detail="Project not found")
    _cancel_active_jobs(
        project_id,
        {
            PipelineStage.SCRIPT_GENERATION,
            PipelineStage.NARRATION_GENERATION,
            PipelineStage.VIDEO_RENDERING,
        },
    )
    if payload.story_segments:
        if payload.panel_keeps:
            updated_panels = [
                panel.model_copy(update={"keep": bool(payload.panel_keeps.get(panel.id, panel.keep))})
                for panel in store.load_panels(project_id)
            ]
            store.save_panels(project_id, updated_panels)
        store.save_story_segments(project_id, payload.story_segments)
    elif payload.panel_keeps or payload.panel_narrations or payload.panel_locks:
        updated_panels = []
        for panel in store.load_panels(project_id):
            keep = bool(payload.panel_keeps.get(panel.id, panel.keep))
            existing_narration = str(panel.narration or "").strip()
            next_narration = str(payload.panel_narrations.get(panel.id, panel.narration or "")).strip()
            next_locked = bool(payload.panel_locks.get(panel.id, panel.narration_locked))
            next_manual = panel.manual_narration
            if panel.id in payload.panel_narrations:
                next_manual = next_narration != existing_narration and bool(next_narration)
            if next_locked and next_narration:
                next_manual = True
            updated_panels.append(
                panel.model_copy(
                    update={
                        "keep": keep,
                        "manual_keep": panel.manual_keep if keep else False,
                        "narration": next_narration or None,
                        "manual_narration": next_manual,
                        "narration_locked": next_locked,
                    }
                )
            )
        store.save_panels(project_id, updated_panels)
        current_lines = [
            str(panel.narration or "")
            for panel in sorted(updated_panels, key=lambda item: item.order)
            if panel.keep
        ]
    else:
        current_lines = payload.script_lines
        store.save_script(project_id, current_lines)
    store.update_stage_state(project_id, PipelineStage.SCRIPT_GENERATION, StageStatus.COMPLETED, progress=100, message="Narration script saved")
    store.update_stage_state(
        project_id,
        PipelineStage.NARRATION_GENERATION,
        StageStatus.READY,
        progress=0,
        message="Script saved. Generate audio when you are ready.",
    )
    store.update_stage_state(
        project_id,
        PipelineStage.VIDEO_RENDERING,
        StageStatus.PENDING,
        progress=0,
        message="Video will be available after fresh audio generation.",
    )
    return store.get_project(project_id)


@router.post("/{project_id}/script/rewrite-panel", response_model=PanelRewriteResponse)
def rewrite_script_panel(project_id: str, payload: PanelRewriteRequest):
    if not store.project_exists(project_id):
        raise HTTPException(status_code=404, detail="Project not found")
    project = store.get_project(project_id)
    panel = next((item for item in project.panels if item.id == payload.panel_id), None)
    if panel is None:
        raise HTTPException(status_code=404, detail="Panel not found")

    project_dir = store._project_dir(project_id)
    router = LLMRouter()
    narrator = PanelNarrator(router, cache_dir=project_dir / "output")

    # Build context from cached artifacts
    context = narrator.load_context_from_cache(project_dir / "output")
    # Add scene lookup from dialogue manifest
    dialogue_manifest = read_json(project_dir / "output" / "dialogue_pipeline_manifest.json", default={})
    context["scene_lookup"] = {
        str(item.get("panel_id", "")): item
        for item in dialogue_manifest.get("scenes", [])
        if isinstance(item, dict) and item.get("panel_id")
    }

    previous_lines = [
        str(item.narration or "").strip()
        for item in sorted(project.panels, key=lambda current: current.order)
        if item.keep and item.order < panel.order and (item.narration or "").strip()
    ]
    next_panel = next(
        (
            item
            for item in sorted(project.panels, key=lambda current: current.order)
            if item.keep and item.order > panel.order and (item.narration or "").strip()
        ),
        None,
    )
    next_line = str(next_panel.narration or "").strip() if next_panel else ""

    current_narration = payload.current_narration or panel.narration or ""

    # Find panel image
    panel_image_path = None
    for pattern in [
        f"panel_{panel.order:03d}.png",
        f"panel_{panel.order:03d}.jpg",
        f"{panel.id}.png",
        f"{panel.id}.jpg",
    ]:
        candidate = project_dir / "panels" / pattern
        if candidate.exists():
            panel_image_path = candidate
            break

    narration = narrator.narrate_single(
        panel,
        context,
        mode=payload.mode.value,
        current_narration=current_narration,
        panel_image_path=panel_image_path,
        previous_lines=previous_lines[-5:],
        next_line=next_line,
    )

    return PanelRewriteResponse(panel_id=panel.id, narration=narration, mode=payload.mode)


@router.post("/{project_id}/script/regenerate-panel-vision")
def regenerate_panel_vision(project_id: str, payload: PanelRewriteRequest):
    """Regenerate a single panel's narration using vision-grounded narration.

    This is the per-panel "fix-in-place" endpoint for the vision pipeline.
    Loads the panel image, sends it to Gemini Vision with surrounding panel
    narrations as continuity, and writes the result back to panels.json
    plus the manifest. Lets the UI fix flagged panels without rerunning
    the entire chapter.
    """
    import asyncio as _asyncio
    import json as _json
    from app.services.panel_vision_narrator import PanelInput, PanelVisionNarrator

    if not store.project_exists(project_id):
        raise HTTPException(status_code=404, detail="Project not found")
    project = store.get_project(project_id)
    panel = next((item for item in project.panels if item.id == payload.panel_id), None)
    if panel is None:
        raise HTTPException(status_code=404, detail="Panel not found")
    if not panel.keep:
        raise HTTPException(status_code=400, detail="Panel is not kept; cannot regenerate")

    project_dir = store._project_dir(project_id)
    image_path = project_dir / "panels" / f"panel_{panel.order:03d}.png"
    if not image_path.exists():
        # Fall back to alternate naming
        for ext in ("jpg", "jpeg", "webp"):
            alt = project_dir / "panels" / f"panel_{panel.order:03d}.{ext}"
            if alt.exists():
                image_path = alt
                break
    if not image_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Panel image not found: panel_{panel.order:03d}.*",
        )

    # Build continuity context from the immediately preceding kept panels
    kept_ordered = sorted(
        [p for p in project.panels if p.keep],
        key=lambda p: (p.page, p.panel),
    )
    panel_index = next(
        (i for i, p in enumerate(kept_ordered) if p.id == panel.id), 0
    )
    prior = kept_ordered[max(0, panel_index - 4):panel_index]
    context_str = "\n".join(
        f"  • {(p.narration or '').strip()}"
        for p in prior
        if (p.narration or "").strip()
    ) or "  (this is the opening panel)"

    panel_input = PanelInput(
        panel_id=panel.id,
        order=panel.order,
        page=panel.page,
        panel=panel.panel,
        image_path=image_path,
        ocr_text=panel.ocr_text or "",
        character_hints=[],
    )

    narrator = PanelVisionNarrator()
    async def _run_one() -> Any:
        # Use a fresh semaphore of size 1 so the call serializes properly
        semaphore = _asyncio.Semaphore(1)
        return await narrator._narrate_one(panel_input, context_str, semaphore)
    result = _asyncio.run(_run_one())

    if result.status != "ok":
        raise HTTPException(
            status_code=502,
            detail=f"Vision narration failed: {result.reason}",
        )

    # Persist: update panels.json and the manifest in lockstep.
    panels_path = project_dir / "panels.json"
    panels_json = _json.loads(panels_path.read_text(encoding="utf-8"))
    for p in panels_json:
        if p["id"] == panel.id:
            p["narration"] = result.narration
            p["narration_source"] = "panel_vision_narrator"
            flags = [f for f in (p.get("review_flags") or []) if not f.startswith("vision_")]
            p["review_flags"] = flags
            break
    panels_path.write_text(_json.dumps(panels_json, indent=2), encoding="utf-8")

    # Update manifest in-place (keep ordering, just edit this segment's text)
    manifest_path = project_dir / "script_manifest.json"
    if manifest_path.exists():
        manifest = _json.loads(manifest_path.read_text(encoding="utf-8"))
        for seg in manifest.get("story_segments", []):
            if panel.id in (seg.get("panel_ids") or []):
                seg["text"] = result.narration
                seg["narration"] = result.narration
                seg["needs_regenerate"] = False
                seg["regenerate_reason"] = ""
                break
        # Rebuild script_lines from segments to stay in sync
        manifest["script_lines"] = [s.get("text", "") for s in manifest.get("story_segments", [])]
        manifest["script_story"] = "\n".join(line for line in manifest["script_lines"] if line)
        manifest_path.write_text(_json.dumps(manifest, indent=2), encoding="utf-8")

    return PanelRewriteResponse(
        panel_id=panel.id,
        narration=result.narration,
        mode=payload.mode,
    )


@router.get("/{project_id}/youtube-bundle")
def get_youtube_bundle(project_id: str):
    """Return the latest YouTube publish bundle metadata for a project.

    Used by the frontend's "Ready to publish" card to show the title,
    description, and a preview of the thumbnail without the user having
    to dig through the file system. The thumbnail and source PNG are
    served as static media under `/media/...`.
    """
    import json as _json
    from app.services.project_store import ProjectStore

    if not store.project_exists(project_id):
        raise HTTPException(status_code=404, detail="Project not found")

    project_dir = store._project_dir(project_id)
    manifest_path = project_dir / "youtube_bundle" / "manifest.json"
    if not manifest_path.exists():
        raise HTTPException(
            status_code=404,
            detail="YouTube bundle has not been generated yet.",
        )
    try:
        manifest = _json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Bundle manifest corrupt: {exc}")

    # Convert internal relative paths into static media URLs the frontend
    # can use directly.
    def _media(url_path: str | None) -> str | None:
        if not url_path:
            return None
        return f"/media/projects/{project_id}/{url_path}"

    return {
        "project_id": project_id,
        "title": manifest.get("title"),
        "title_variants": manifest.get("title_variants") or [],
        "description": manifest.get("description"),
        "thumbnail_url": _media(manifest.get("thumbnail_path")),
        "thumbnail_source_url": _media(manifest.get("thumbnail_source_path")),
        "thumbnail_source_panel_id": manifest.get("thumbnail_source_panel_id"),
        "bundle_dir": manifest.get("bundle_dir"),
    }


@router.patch("/{project_id}/settings")
def update_project_settings(project_id: str, payload: ProjectSettingsUpdateRequest):
    if not store.project_exists(project_id):
        raise HTTPException(status_code=404, detail="Project not found")
    existing_project = store.get_project(project_id)
    updates: dict[str, object] = {}
    if payload.voice_config is not None:
        updates["voice_config"] = payload.voice_config.model_dump(mode="json")
    if payload.music_config is not None:
        updates["music_config"] = payload.music_config.model_dump(mode="json")
    if payload.video_config is not None:
        updates["video_config"] = payload.video_config.model_dump(mode="json")
    if payload.pipeline_config is not None:
        updates["pipeline_config"] = payload.pipeline_config.model_dump(mode="json")

    if updates:
        store.update_project_metadata(project_id, **updates)

    project = store.get_project(project_id)
    if payload.voice_config is not None:
        if project.script_lines:
            queue_stage_once(store, queue, project_id, PipelineStage.NARRATION_GENERATION, "Queued automatically after narrator change")
            store.update_stage_state(project_id, PipelineStage.VIDEO_RENDERING, StageStatus.PENDING, progress=0, message="Video will continue automatically after regenerated audio")
        else:
            store.update_stage_state(project_id, PipelineStage.NARRATION_GENERATION, StageStatus.READY, progress=0, message="Narration settings updated. Save or generate a script to continue.")
            store.update_stage_state(project_id, PipelineStage.VIDEO_RENDERING, StageStatus.PENDING, progress=0, message="Audio needs to be generated before video can continue.")
    elif payload.music_config is not None or payload.video_config is not None:
        if project.audio_files:
            reason = "music change" if payload.music_config is not None else "render setting change"
            queue_stage_once(store, queue, project_id, PipelineStage.VIDEO_RENDERING, f"Queued automatically after {reason}")
        else:
            store.update_stage_state(project_id, PipelineStage.VIDEO_RENDERING, StageStatus.READY, progress=0, message="Render settings updated. Generate audio to continue.")

    if (
        payload.pipeline_config is not None
        and payload.pipeline_config.auto_run_end_to_end
        and not existing_project.pipeline_config.auto_run_end_to_end
    ):
        continue_auto_run_pipeline(store, queue, project_id, source="enabling auto-run")

    return store.get_project(project_id)


@router.patch("/{project_id}/name")
def rename_project(project_id: str, payload: ProjectRenameRequest):
    if not store.project_exists(project_id):
        raise HTTPException(status_code=404, detail="Project not found")
    name = payload.name.strip()
    if not name:
        raise HTTPException(status_code=422, detail="Project name cannot be blank")
    return store.update_project_metadata(project_id, name=name)


@router.post("/{project_id}/jobs")
def queue_stage(project_id: str, payload: QueueStageRequest):
    if not store.project_exists(project_id):
        raise HTTPException(status_code=404, detail="Project not found")

    if payload.stage == PipelineStage.PANEL_REVIEW:
        raise HTTPException(status_code=400, detail="Panel review is completed from the editor UI, not the queue.")

    job_payload = dict(payload.payload or {})
    if payload.stage == PipelineStage.SCRIPT_GENERATION:
        _cancel_active_jobs(
            project_id,
            {PipelineStage.CHARACTER_REVIEW},
            job_message="Cancelled because script generation was requested explicitly.",
            stage_message="Skipped character review so script generation can start immediately.",
        )
    if payload.stage == PipelineStage.NARRATION_GENERATION:
        # A user-clicked audio queue is an explicit intent to proceed, even if an
        # older frontend build failed to include the manual bypass flag.
        job_payload.setdefault("force_quality_bypass", True)

    job = queue_stage_once(store, queue, project_id, payload.stage, "Queued", payload=job_payload)
    return store.get_job(project_id, job.id)


@router.get("/{project_id}/characters", response_model=CharacterReviewState)
def get_character_review(project_id: str):
    if not store.project_exists(project_id):
        raise HTTPException(status_code=404, detail="Project not found")
    project = store.get_project(project_id)
    try:
        return character_review_service.ensure_review_state(
            project.id,
            store._project_dir(project.id),
            project.name,
            project.chapter_metadata,
            project.panels,
        )
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=404,
            detail="Character suggestions have not been prepared yet. Save the panel review or run character review first.",
        ) from exc


@router.get("/{project_id}/character-dictionary")
def get_character_dictionary(project_id: str):
    if not store.project_exists(project_id):
        raise HTTPException(status_code=404, detail="Project not found")
    dictionary = read_json(store._project_dir(project_id) / "output" / "character_dictionary.json", default={})
    if not isinstance(dictionary, dict):
        dictionary = {}
    return {
        "project_id": project_id,
        "entries": [
            {"key": str(key), "name": str(value)}
            for key, value in sorted(dictionary.items(), key=lambda item: str(item[1]).lower())
            if str(value).strip()
        ],
    }


@router.get("/{project_id}/character-portraits")
def get_character_portraits(project_id: str):
    if not store.project_exists(project_id):
        raise HTTPException(status_code=404, detail="Project not found")
    characters = read_json(store._project_dir(project_id) / "output" / "canonical_characters.json", default=[])
    if not isinstance(characters, list):
        characters = []
    return {"project_id": project_id, "characters": characters}


@router.put("/{project_id}/character-portraits")
def update_character_portraits(project_id: str, payload: CharacterPortraitsUpdateRequest):
    if not store.project_exists(project_id):
        raise HTTPException(status_code=404, detail="Project not found")

    _cancel_active_jobs(
        project_id,
        {
            PipelineStage.CHARACTER_PORTRAIT,
            PipelineStage.PANEL_VISION_EXTRACTION,
            PipelineStage.PANEL_VISION_QUALITY,
            PipelineStage.SCRIPT_GENERATION,
            PipelineStage.NARRATION_GENERATION,
            PipelineStage.VIDEO_RENDERING,
        },
    )

    output_dir = store._project_dir(project_id) / "output"
    characters = [character.model_dump(mode="json") for character in payload.characters]
    write_json(output_dir / "canonical_characters.json", characters)

    for filename in (
        "panel_vision.json",
        "panel_vision_final.json",
        "story_bible.json",
        "story_grounding.json",
        "style_vocabulary.json",
    ):
        (output_dir / filename).unlink(missing_ok=True)
    store.invalidate_script_outputs(project_id, clear_generated_panel_narration=True)
    store._reset_directory(store._project_dir(project_id) / "audio")
    store._reset_directory(store._project_dir(project_id) / "video")
    store._reset_directory(store._project_dir(project_id) / "exports")

    store.update_stage_state(
        project_id,
        PipelineStage.CHARACTER_PORTRAIT,
        StageStatus.COMPLETED,
        progress=100,
        message=f"Character portraits saved ({len(characters)} characters)",
    )
    store.update_stage_state(
        project_id,
        PipelineStage.PANEL_VISION_EXTRACTION,
        StageStatus.READY,
        progress=0,
        message="Portrait edits saved. Re-run panel vision before generating the script.",
    )
    store.update_stage_state(
        project_id,
        PipelineStage.PANEL_VISION_QUALITY,
        StageStatus.PENDING,
        progress=0,
        message="Run panel vision before quality review.",
    )
    store.update_stage_state(
        project_id,
        PipelineStage.SCRIPT_GENERATION,
        StageStatus.READY,
        progress=0,
        message="Portrait edits saved. Generate the script after panel vision refreshes.",
    )
    store.update_stage_state(
        project_id,
        PipelineStage.NARRATION_GENERATION,
        StageStatus.PENDING,
        progress=0,
        message="Generate a script before creating audio.",
    )
    store.update_stage_state(
        project_id,
        PipelineStage.VIDEO_RENDERING,
        StageStatus.PENDING,
        progress=0,
        message="Generate audio before rendering video.",
    )
    store.update_project_metadata(project_id)
    return {"project_id": project_id, "characters": characters}


@router.put("/{project_id}/characters", response_model=CharacterReviewState)
def update_character_review(project_id: str, payload: CharacterReviewUpdateRequest):
    if not store.project_exists(project_id):
        raise HTTPException(status_code=404, detail="Project not found")
    project = store.get_project(project_id)
    existing_state = character_review_service.load_review_state(store._project_dir(project_id))
    next_state = CharacterReviewState(
        project_id=project_id,
        series_key=character_review_service.series_key(project.chapter_metadata, project.name),
        protagonist_name=payload.protagonist_name or (existing_state.protagonist_name if existing_state else None),
        memory_names=existing_state.memory_names if existing_state else [],
        identities=payload.identities,
        generated_at=existing_state.generated_at if existing_state else store._now(),
        updated_at=store._now(),
    )
    saved_state = character_review_service.save_review_state(
        store._project_dir(project_id),
        project.name,
        project.chapter_metadata,
        next_state,
    )
    for filename in ("canonical_characters.json", "panel_vision.json", "panel_vision_final.json"):
        (store._project_dir(project_id) / "output" / filename).unlink(missing_ok=True)
    store.update_stage_state(project_id, PipelineStage.CHARACTER_REVIEW, StageStatus.COMPLETED, progress=100, message="Character review saved")
    store.update_stage_state(project_id, PipelineStage.SCRIPT_GENERATION, StageStatus.READY, progress=0, message="Character review saved. Generate the script when you're ready.")
    store.update_stage_state(project_id, PipelineStage.NARRATION_GENERATION, StageStatus.PENDING, progress=0, message="Generate a script before creating audio.")
    store.update_stage_state(project_id, PipelineStage.VIDEO_RENDERING, StageStatus.PENDING, progress=0, message="Generate audio before rendering video.")
    continue_auto_run_pipeline(store, queue, project_id, source="character review")
    return saved_state


@router.post("/{project_id}/rewind")
def rewind_project(project_id: str, payload: RewindStageRequest):
    if not store.project_exists(project_id):
        raise HTTPException(status_code=404, detail="Project not found")
    if payload.stage not in REWINDABLE_STAGES:
        raise HTTPException(status_code=400, detail="Projects can only be rewound to supported pipeline stages.")

    affected_stages = set(PipelineStage)
    _cancel_active_jobs(project_id, affected_stages)
    review_state_exists = character_review_service.load_review_state(store._project_dir(project_id)) is not None

    if payload.stage == PipelineStage.PANEL_REVIEW:
        store.reset_pipeline_from_stage(project_id, PipelineStage.PANEL_REVIEW)
        store.update_stage_state(project_id, PipelineStage.PANEL_REVIEW, StageStatus.NEEDS_REVIEW, progress=100, message="Panel review reopened. Save panels to continue automatically.")
        store.update_stage_state(project_id, PipelineStage.CHARACTER_REVIEW, StageStatus.PENDING, progress=0, message="Character review will unlock after panel review is saved.")
        store.update_stage_state(project_id, PipelineStage.SCRIPT_GENERATION, StageStatus.PENDING, progress=0, message="Waiting for character review suggestions.")
        store.update_stage_state(project_id, PipelineStage.NARRATION_GENERATION, StageStatus.PENDING, progress=0, message="Audio will regenerate after the updated script is ready.")
        store.update_stage_state(project_id, PipelineStage.VIDEO_RENDERING, StageStatus.PENDING, progress=0, message="Video will regenerate after updated audio is ready.")
    elif payload.stage == PipelineStage.INGESTION:
        store.reset_pipeline_from_stage(project_id, PipelineStage.INGESTION)
        store.update_stage_state(project_id, PipelineStage.INGESTION, StageStatus.READY, progress=0, message="Re-importing source pages")
        store.update_stage_state(project_id, PipelineStage.PANEL_DETECTION, StageStatus.PENDING, progress=0, message="Waiting for the refreshed page import.")
        store.update_stage_state(project_id, PipelineStage.PANEL_REVIEW, StageStatus.PENDING, progress=0, message="Panel review will reopen after panel detection finishes.")
        store.update_stage_state(project_id, PipelineStage.CHARACTER_REVIEW, StageStatus.PENDING, progress=0, message="Character review will unlock after panel review is saved.")
        store.update_stage_state(project_id, PipelineStage.SCRIPT_GENERATION, StageStatus.PENDING, progress=0, message="Waiting for refreshed character review.")
        store.update_stage_state(project_id, PipelineStage.NARRATION_GENERATION, StageStatus.PENDING, progress=0, message="Audio will regenerate after the next script draft.")
        store.update_stage_state(project_id, PipelineStage.VIDEO_RENDERING, StageStatus.PENDING, progress=0, message="Video will regenerate after the refreshed audio is ready.")
        queue_stage_once(store, queue, project_id, PipelineStage.INGESTION, "Queued automatically after rewinding to import")
    elif payload.stage == PipelineStage.PANEL_DETECTION:
        store.reset_pipeline_from_stage(project_id, PipelineStage.PANEL_DETECTION)
        store.update_stage_state(project_id, PipelineStage.INGESTION, StageStatus.COMPLETED, progress=100, message="Pages are ready")
        store.update_stage_state(project_id, PipelineStage.PANEL_DETECTION, StageStatus.READY, progress=0, message="Re-running panel detection")
        store.update_stage_state(project_id, PipelineStage.PANEL_REVIEW, StageStatus.PENDING, progress=0, message="Panel review will reopen after the new detection pass.")
        store.update_stage_state(project_id, PipelineStage.CHARACTER_REVIEW, StageStatus.PENDING, progress=0, message="Character review will unlock after panel review is saved.")
        store.update_stage_state(project_id, PipelineStage.SCRIPT_GENERATION, StageStatus.PENDING, progress=0, message="Waiting for refreshed character review.")
        store.update_stage_state(project_id, PipelineStage.NARRATION_GENERATION, StageStatus.PENDING, progress=0, message="Audio will regenerate after the updated script is ready.")
        store.update_stage_state(project_id, PipelineStage.VIDEO_RENDERING, StageStatus.PENDING, progress=0, message="Video will regenerate after updated audio is ready.")
        queue_stage_once(store, queue, project_id, PipelineStage.PANEL_DETECTION, "Queued automatically after rewinding to panel detection")
    elif payload.stage == PipelineStage.CHARACTER_REVIEW:
        store.reset_generated_outputs_after_stage(project_id, PipelineStage.CHARACTER_REVIEW)
        store.update_stage_state(project_id, PipelineStage.PANEL_REVIEW, StageStatus.COMPLETED, progress=100, message="Panel review saved")
        store.update_stage_state(
            project_id,
            PipelineStage.CHARACTER_REVIEW,
            StageStatus.NEEDS_REVIEW if review_state_exists else StageStatus.READY,
            progress=100 if review_state_exists else 0,
            message="Character review reopened. Save your changes to continue."
            if review_state_exists
            else "Prepare character suggestions from the saved panels.",
        )
        store.update_stage_state(project_id, PipelineStage.SCRIPT_GENERATION, StageStatus.PENDING, progress=0, message="Waiting for character review changes.")
        store.update_stage_state(project_id, PipelineStage.NARRATION_GENERATION, StageStatus.PENDING, progress=0, message="Audio will regenerate after the updated script is ready.")
        store.update_stage_state(project_id, PipelineStage.VIDEO_RENDERING, StageStatus.PENDING, progress=0, message="Video will regenerate after updated audio is ready.")
    elif payload.stage == PipelineStage.SCRIPT_GENERATION:
        store.reset_generated_outputs_after_stage(project_id, PipelineStage.SCRIPT_GENERATION)
        store.update_stage_state(project_id, PipelineStage.PANEL_REVIEW, StageStatus.COMPLETED, progress=100, message="Panel review saved")
        store.update_stage_state(
            project_id,
            PipelineStage.CHARACTER_REVIEW,
            StageStatus.COMPLETED if review_state_exists else StageStatus.READY,
            progress=100 if review_state_exists else 0,
            message="Character review saved" if review_state_exists else "Character review is ready to prepare.",
        )
        store.update_stage_state(project_id, PipelineStage.SCRIPT_GENERATION, StageStatus.READY, progress=0, message="Open narration to edit or regenerate the script.")
        store.update_stage_state(project_id, PipelineStage.NARRATION_GENERATION, StageStatus.PENDING, progress=0, message="Waiting for the next script draft.")
        store.update_stage_state(project_id, PipelineStage.VIDEO_RENDERING, StageStatus.PENDING, progress=0, message="Video will continue after narration audio is regenerated.")
    else:
        store.reset_generated_outputs_after_stage(project_id, PipelineStage.NARRATION_GENERATION)
        store.update_stage_state(project_id, PipelineStage.PANEL_REVIEW, StageStatus.COMPLETED, progress=100, message="Panel review saved")
        store.update_stage_state(
            project_id,
            PipelineStage.CHARACTER_REVIEW,
            StageStatus.COMPLETED if review_state_exists else StageStatus.READY,
            progress=100 if review_state_exists else 0,
            message="Character review saved" if review_state_exists else "Character review is ready to prepare.",
        )
        store.update_stage_state(
            project_id,
            PipelineStage.SCRIPT_GENERATION,
            StageStatus.COMPLETED if store.load_script(project_id) else StageStatus.READY,
            progress=100 if store.load_script(project_id) else 0,
            message="Narration draft is ready to revise." if store.load_script(project_id) else "Create or regenerate a script before audio.",
        )
        store.update_stage_state(project_id, PipelineStage.NARRATION_GENERATION, StageStatus.READY, progress=0, message="Adjust the narrator and generate audio when ready.")
        store.update_stage_state(project_id, PipelineStage.VIDEO_RENDERING, StageStatus.PENDING, progress=0, message="Waiting for regenerated audio.")

    return store.get_project(project_id)


@router.get("/{project_id}/jobs")
def list_jobs(project_id: str):
    return store.list_jobs(project_id)


@router.post("/{project_id}/jobs/{job_id}/cancel")
def cancel_job(project_id: str, job_id: str):
    if not store.project_exists(project_id):
        raise HTTPException(status_code=404, detail="Project not found")
    queue.request_cancel(job_id)
    try:
        job = store.get_job(project_id, job_id)
        store.update_job(
            project_id,
            job_id,
            status="cancelled",
            finished_at=store._now().isoformat(),
            message="Cancellation requested by user",
        )
        store.update_stage_state(project_id, job.stage, StageStatus.CANCELLED, progress=job.progress, message="Cancelled from dashboard")
    except FileNotFoundError:
        pass
    return {"status": "cancellation_requested", "job_id": job_id}


@router.post("/{project_id}/jobs/{job_id}/pause")
def pause_job(project_id: str, job_id: str):
    """
    Pause a running or queued job.  The worker stops at its next checkpoint
    and the stage is left in 'paused' state so it can be resumed later.
    """
    if not store.project_exists(project_id):
        raise HTTPException(status_code=404, detail="Project not found")
    try:
        job = store.get_job(project_id, job_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Job not found")
    if job.status in {JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED}:
        raise HTTPException(status_code=400, detail=f"Cannot pause a job with status '{job.status.value}'")
    if job.status == JobStatus.PAUSED:
        return {"status": "already_paused", "job_id": job_id}
    # Signal the running worker (if any) to stop at the next checkpoint
    queue.request_pause(job_id)
    # Immediately mark as paused so the queue doesn't pick it up if it's still QUEUED
    store.update_job(
        project_id,
        job_id,
        status=JobStatus.PAUSED.value,
        finished_at=store._now().isoformat(),
        message="Paused",
    )
    store.update_stage_state(project_id, job.stage, StageStatus.PAUSED, progress=job.progress, message="Paused — click Resume to continue")
    return {"status": "paused", "job_id": job_id}


@router.post("/{project_id}/jobs/{job_id}/resume")
def resume_job(project_id: str, job_id: str):
    """
    Resume a paused job by re-queuing its stage.  A new job record is created;
    stages that cache work (narration, script) will skip already-completed panels.
    """
    if not store.project_exists(project_id):
        raise HTTPException(status_code=404, detail="Project not found")
    try:
        job = store.get_job(project_id, job_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Job not found")
    if job.status != JobStatus.PAUSED:
        raise HTTPException(status_code=400, detail=f"Job is not paused (status: '{job.status.value}')")
    # Clear the pause flag in case the worker hasn't consumed it yet
    queue.clear_pause(job_id)
    # Re-queue the same stage; the helper creates a new job record
    new_job = queue_stage_once(store, queue, project_id, job.stage, "Resuming", payload=dict(job.payload or {}))
    return store.get_job(project_id, new_job.id)


@router.post("/{project_id}/cancel")
def cancel_project(project_id: str):
    if not store.project_exists(project_id):
        raise HTTPException(status_code=404, detail="Project not found")

    project = store.get_project(project_id)
    cancelled_job_ids: list[str] = []
    for job in project.active_jobs:
        queue.request_cancel(job.id)
        store.update_job(
            project_id,
            job.id,
            status="cancelled",
            finished_at=store._now().isoformat(),
            message="Project cancelled from dashboard",
        )
        store.update_stage_state(project_id, job.stage, StageStatus.CANCELLED, progress=job.progress, message="Cancelled from dashboard")
        cancelled_job_ids.append(job.id)

    return {"status": "project_cancellation_requested", "job_ids": cancelled_job_ids}


@router.post("/{project_id}/merge")
def merge_videos(project_id: str, payload: MergeVideosRequest):
    if project_id != payload.project_id:
        raise HTTPException(status_code=400, detail="Project id mismatch")
    video_paths = [Path(path) for path in payload.video_paths]
    project_dir = store._project_dir(project_id)
    output_path = video_service.merge_videos(project_dir / "video", video_paths, payload.output_name, payload.video_config)
    return {"path": str(output_path), "url": store._relative_media_url(output_path)}


@router.delete("/{project_id}/video/{video_name}")
def delete_video(project_id: str, video_name: str):
    project_dir = store._project_dir(project_id)
    video_path = project_dir / "video" / video_name
    if not video_path.exists():
        raise HTTPException(status_code=404, detail="Video not found")
    video_path.unlink()
    manifest_path = project_dir / "video" / "manifest.json"
    if manifest_path.exists():
        manifest = read_json(manifest_path, default={})
        manifest.pop(video_name, None)
        write_json(manifest_path, manifest)
    return {"status": "deleted", "video_name": video_name}


@router.delete("/{project_id}")
def delete_project(project_id: str):
    if not store.project_exists(project_id):
        raise HTTPException(status_code=404, detail="Project not found")

    project = store.get_project(project_id)
    if project.active_jobs:
        raise HTTPException(status_code=409, detail="Cancel the active project jobs before deleting this project.")

    store.delete_project(project_id)
    return {"status": "deleted", "project_id": project_id}
