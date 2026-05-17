from __future__ import annotations

from datetime import datetime
import logging
import os
from pathlib import Path
import re
import shutil
import subprocess
from threading import Lock
from typing import Any, Iterable
from uuid import uuid4

import numpy as np
from PIL import Image

from app.core.config import get_settings
from app.schemas.project import (
    AudioFile,
    ChapterMetadata,
    JobRecord,
    JobStatus,
    MusicConfig,
    PanelBox,
    PipelineConfig,
    PipelineStage,
    ProjectDetail,
    ProjectSummary,
    StorySegment,
    SourceType,
    StageState,
    StageStatus,
    VideoConfig,
    VideoFile,
    VoiceConfig,
)
from app.services.ocr_cleaner import is_usable_ocr_text
from app.services.panel_quality_service import PanelQualityService
from app.services.script_cleaner_service import ScriptCleanerService
from app.services.script_quality_service import ScriptQualityService
from app.utils.files import ensure_dir, read_json, slugify, write_json


logger = logging.getLogger(__name__)

_LOW_SIGNAL_SCRIPT_FLAGS = {"corner_wedge", "side_void", "top_blank_band", "whitespace"}
PANEL_CROP_VERSION = "tightcrop_v5"


def _whole_progress(progress: object) -> int:
    try:
        value = float(progress or 0.0)
    except Exception:
        value = 0.0
    value = max(0.0, min(100.0, value))
    if value <= 0:
        return 0
    if value >= 100:
        return 100
    return int(value + 0.9999)


class ProjectStore:
    # Process-wide mtime-invalidated caches for the two hot loaders
    # called on every API request. Without these, opening the unord
    # editor (3447 pages, 3500 panels) blocks the API thread for
    # 25+ seconds because `_page_size_lookup` opens every page image
    # (just to read PIL header dimensions) and `load_panels` re-parses
    # a 4 MB JSON file through pydantic on every fetch.
    #
    # Cache key = (project_id, source_file_mtime). When the worker
    # writes panels.json or anything in the pages dir, the mtime
    # changes and the next access misses + rebuilds. Threadsafe via
    # a single class-level lock; the cached values are immutable
    # after construction.
    _PANELS_CACHE: dict[str, tuple[float, list["PanelBox"]]] = {}
    _PAGE_SIZE_CACHE: dict[str, tuple[tuple[float, int], dict[int, tuple[int, int]]]] = {}
    _CACHE_LOCK = Lock()

    def __init__(self) -> None:
        self.settings = get_settings()
        self.projects_root = ensure_dir(self.settings.data_dir / "projects")
        self.builtin_music_root = ensure_dir(Path(__file__).resolve().parents[2] / "assets" / "music")
        self.uploaded_music_root = ensure_dir(self.settings.data_dir / "music")
        self._script_cleaner = ScriptCleanerService()
        self._script_quality = ScriptQualityService()
        self._panel_quality = PanelQualityService()

    def _project_dir(self, project_id: str) -> Path:
        return self.projects_root / project_id

    def _metadata_path(self, project_id: str) -> Path:
        return self._project_dir(project_id) / "metadata.json"

    def _panels_path(self, project_id: str) -> Path:
        return self._project_dir(project_id) / "panels.json"

    def _panel_previews_dir(self, project_id: str) -> Path:
        return ensure_dir(self._project_dir(project_id) / "panels")

    def _script_path(self, project_id: str) -> Path:
        return self._project_dir(project_id) / "script.txt"

    def _script_manifest_path(self, project_id: str) -> Path:
        return self._project_dir(project_id) / "script.json"

    def _script_quality_path(self, project_id: str) -> Path:
        return self._project_dir(project_id) / "output" / "script_quality.json"

    def _script_artifact_metadata_path(self, project_id: str) -> Path:
        return self._project_dir(project_id) / "output" / "script_artifact.json"

    def _panel_quality_path(self, project_id: str) -> Path:
        return self._project_dir(project_id) / "output" / "panel_quality.json"

    def _panel_crop_report_path(self, project_id: str) -> Path:
        return self._project_dir(project_id) / "output" / "panel_crop_report.json"

    def _job_path(self, project_id: str, job_id: str) -> Path:
        return self._project_dir(project_id) / "jobs" / f"{job_id}.json"

    def _relative_media_url(self, path: Path) -> str:
        relative_path = path.relative_to(self.settings.data_dir).as_posix()
        return f"/media/{relative_path}"

    def _now(self) -> datetime:
        return datetime.utcnow()

    def _default_stage_states(self) -> dict[PipelineStage, StageState]:
        return {
            stage: StageState(stage=stage)
            for stage in PipelineStage
        }

    def create_project(
        self,
        name: str,
        source_type: SourceType,
        source_reference: str | None = None,
        chapter_metadata: ChapterMetadata | None = None,
    ) -> ProjectSummary:
        project_id = f"{slugify(name)}-{uuid4().hex[:8]}"
        project_dir = self._project_dir(project_id)
        ensure_dir(project_dir)
        for directory in ("pages", "audio", "video", "exports", "jobs", "input", "source", "panels", "ocr", "translations", "output", "thumbnails", "temp", "characters"):
            ensure_dir(project_dir / directory)

        metadata = {
            "id": project_id,
            "name": name,
            "source_type": source_type.value,
            "source_reference": source_reference,
            "created_at": self._now().isoformat(),
            "updated_at": self._now().isoformat(),
            "chapter_metadata": (chapter_metadata or ChapterMetadata()).model_dump(mode="json"),
            "stage_states": {
                stage.value: state.model_dump(mode="json")
                for stage, state in self._default_stage_states().items()
            },
            "voice_config": VoiceConfig().model_dump(mode="json"),
            "video_config": VideoConfig().model_dump(mode="json"),
            "music_config": MusicConfig().model_dump(mode="json"),
            "pipeline_config": PipelineConfig().model_dump(mode="json"),
        }
        write_json(self._metadata_path(project_id), metadata)
        write_json(self._panels_path(project_id), [])
        self._script_path(project_id).write_text("", encoding="utf-8")
        write_json(
            self._script_manifest_path(project_id),
            {
                "script_lines": [],
                "script_lines_strict": [],
                "script_lines_cinematic": [],
                "script_story": "",
                "story_segments": [],
                "script_mode": "story_segments_v1",
            },
        )
        return self.get_project(project_id)

    def project_exists(self, project_id: str) -> bool:
        return self._metadata_path(project_id).exists()

    def delete_project(self, project_id: str) -> None:
        project_dir = self._project_dir(project_id)
        if not project_dir.exists():
            raise FileNotFoundError(f"Unknown project: {project_id}")
        job_ids = self._job_ids_for_project(project_id)
        self.purge_project_artifacts(project_id, job_ids=job_ids)
        shutil.rmtree(project_dir)

    def _job_ids_for_project(self, project_id: str) -> set[str]:
        jobs_dir = self._project_dir(project_id) / "jobs"
        if not jobs_dir.exists():
            return set()
        return {path.stem for path in jobs_dir.glob("*.json") if path.is_file()}

    def _delete_artifact_path(self, path: Path) -> bool:
        if not path.exists() and not path.is_symlink():
            return False
        if path.is_dir() and not path.is_symlink():
            shutil.rmtree(path)
        else:
            path.unlink(missing_ok=True)
        return True

    def _project_artifact_dirs(self) -> tuple[Path, ...]:
        return (
            self.settings.training_data_dir / "annotations",
            self.settings.training_data_dir / "images",
            self.settings.ocr_training_data_dir / "annotations",
            self.settings.ocr_training_data_dir / "images",
            self.settings.data_dir / "previews",
        )

    def purge_project_artifacts(self, project_id: str, job_ids: Iterable[str] | None = None) -> dict[str, int]:
        counts = {"project_artifacts": 0, "queue_artifacts": 0}
        prefix = f"{project_id}_"
        for directory in self._project_artifact_dirs():
            if not directory.exists():
                continue
            for path in directory.glob(f"{prefix}*"):
                if self._delete_artifact_path(path):
                    counts["project_artifacts"] += 1

        for job_id in job_ids or ():
            for queue_name in ("cancel", "pause", "messages"):
                queue_dir = self.settings.data_dir / "_queue" / queue_name
                if not queue_dir.exists():
                    continue
                for path in queue_dir.glob(f"{job_id}*"):
                    if self._delete_artifact_path(path):
                        counts["queue_artifacts"] += 1
        return counts

    def purge_orphaned_artifacts(self) -> dict[str, int]:
        active_project_ids = {
            path.parent.name
            for path in self.projects_root.glob("*/metadata.json")
            if path.is_file()
        }
        active_job_ids = {
            path.stem
            for path in self.projects_root.glob("*/jobs/*.json")
            if path.is_file()
        }
        counts = {"project_artifacts": 0, "queue_artifacts": 0}
        project_artifact_pattern = re.compile(r"^(.+?)_(?:page|panel)_")

        for directory in self._project_artifact_dirs():
            if not directory.exists():
                continue
            for path in directory.iterdir():
                match = project_artifact_pattern.match(path.name)
                if match and match.group(1) not in active_project_ids:
                    if self._delete_artifact_path(path):
                        counts["project_artifacts"] += 1

        for queue_name in ("cancel", "pause", "messages"):
            queue_dir = self.settings.data_dir / "_queue" / queue_name
            if not queue_dir.exists():
                continue
            for path in queue_dir.iterdir():
                if path.stem not in active_job_ids:
                    if self._delete_artifact_path(path):
                        counts["queue_artifacts"] += 1
        return counts

    def duplicate_project(
        self,
        project_id: str,
        name: str | None = None,
        video_name: str | None = None,
        copy_all_videos: bool = False,
    ) -> ProjectDetail:
        source_project = self.get_project(project_id)
        if video_name and not any(video.name == video_name for video in source_project.videos):
            raise FileNotFoundError(f"Video not found: {video_name}")

        duplicated = self.create_project(
            name=name or f"{source_project.name} Copy",
            source_type=source_project.source_type,
            source_reference=source_project.source_reference,
            chapter_metadata=source_project.chapter_metadata,
        )
        duplicated_id = duplicated.id
        source_dir = self._project_dir(project_id)
        target_dir = self._project_dir(duplicated_id)

        for directory_name in ("pages", "input", "source", "panels", "ocr", "translations", "output", "thumbnails", "exports", "characters"):
            self._copy_directory_contents(source_dir / directory_name, target_dir / directory_name)

        self._copy_directory_contents(source_dir / "audio", target_dir / "audio")
        self._copy_selected_videos(source_dir / "video", target_dir / "video", video_name, copy_all_videos)

        source_panels = self._panels_path(project_id)
        if source_panels.exists():
            shutil.copy2(source_panels, self._panels_path(duplicated_id))

        source_script = self._script_path(project_id)
        if source_script.exists():
            shutil.copy2(source_script, self._script_path(duplicated_id))
        source_script_manifest = self._script_manifest_path(project_id)
        if source_script_manifest.exists():
            shutil.copy2(source_script_manifest, self._script_manifest_path(duplicated_id))

        stage_states = self._default_stage_states()
        page_count = len(list((target_dir / "pages").glob("*")))
        panels = self.load_panels(duplicated_id)
        has_script = bool(self.load_script(duplicated_id))
        has_audio = bool(self.list_audio_files(duplicated_id))
        has_videos = bool(self.list_videos(duplicated_id))

        if page_count:
            stage_states[PipelineStage.INGESTION] = StageState(
                stage=PipelineStage.INGESTION,
                status=StageStatus.COMPLETED,
                progress=100,
                message="Pages copied from the source project",
            )
            if not panels:
                stage_states[PipelineStage.PANEL_DETECTION] = StageState(
                    stage=PipelineStage.PANEL_DETECTION,
                    status=StageStatus.READY,
                    progress=0,
                    message="Pages copied. Run panel detection when you are ready.",
                )
        if panels:
            stage_states[PipelineStage.PANEL_DETECTION] = StageState(
                stage=PipelineStage.PANEL_DETECTION,
                status=StageStatus.COMPLETED,
                progress=100,
                message="Panels copied from the source project",
            )
            stage_states[PipelineStage.PANEL_REVIEW] = StageState(
                stage=PipelineStage.PANEL_REVIEW,
                status=StageStatus.COMPLETED,
                progress=100,
                message="Panel review copied. You can re-open it at any time.",
            )
            character_review_path = target_dir / "output" / "character_review_state.json"
            stage_states[PipelineStage.CHARACTER_REVIEW] = StageState(
                stage=PipelineStage.CHARACTER_REVIEW,
                status=StageStatus.COMPLETED if character_review_path.exists() else StageStatus.READY,
                progress=100 if character_review_path.exists() else 0,
                message="Character review copied from the source project"
                if character_review_path.exists()
                else "Character review is ready to prepare from the copied chapter.",
            )
            canonical_characters_path = target_dir / "output" / "canonical_characters.json"
            panel_vision_path = target_dir / "output" / "panel_vision.json"
            panel_vision_final_path = target_dir / "output" / "panel_vision_final.json"
            if canonical_characters_path.exists():
                stage_states[PipelineStage.CHARACTER_PORTRAIT] = StageState(
                    stage=PipelineStage.CHARACTER_PORTRAIT,
                    status=StageStatus.COMPLETED,
                    progress=100,
                    message="Canonical character roster copied from the source project",
                )
            if panel_vision_path.exists():
                stage_states[PipelineStage.PANEL_VISION_EXTRACTION] = StageState(
                    stage=PipelineStage.PANEL_VISION_EXTRACTION,
                    status=StageStatus.COMPLETED,
                    progress=100,
                    message="Panel vision draft copied from the source project",
                )
            if panel_vision_final_path.exists():
                stage_states[PipelineStage.PANEL_VISION_QUALITY] = StageState(
                    stage=PipelineStage.PANEL_VISION_QUALITY,
                    status=StageStatus.COMPLETED,
                    progress=100,
                    message="Panel vision quality pass copied from the source project",
                )
        if has_script:
            stage_states[PipelineStage.SCRIPT_GENERATION] = StageState(
                stage=PipelineStage.SCRIPT_GENERATION,
                status=StageStatus.COMPLETED,
                progress=100,
                message="Scene-based recap script copied from the source project",
            )
        if has_audio:
            stage_states[PipelineStage.NARRATION_GENERATION] = StageState(
                stage=PipelineStage.NARRATION_GENERATION,
                status=StageStatus.COMPLETED,
                progress=100,
                message="Narration audio copied from the source project",
            )
        if has_audio:
            stage_states[PipelineStage.VIDEO_RENDERING] = StageState(
                stage=PipelineStage.VIDEO_RENDERING,
                status=StageStatus.COMPLETED if has_videos else StageStatus.READY,
                progress=100 if has_videos else 0,
                message="Copied finished video. Adjust settings and re-render this duplicate."
                if has_videos
                else "Duplicate ready. Adjust settings and render a fresh cut.",
            )

        self.update_project_metadata(
            duplicated_id,
            chapter_metadata=source_project.chapter_metadata.model_dump(mode="json"),
            voice_config=source_project.voice_config.model_dump(mode="json"),
            video_config=source_project.video_config.model_dump(mode="json"),
            music_config=source_project.music_config.model_dump(mode="json"),
            pipeline_config=source_project.pipeline_config.model_dump(mode="json"),
            stage_states={
                stage.value: state.model_dump(mode="json")
                for stage, state in stage_states.items()
            },
        )
        return self.get_project(duplicated_id)

    def list_projects(self) -> list[ProjectSummary]:
        projects: list[ProjectSummary] = []
        for metadata_path in sorted(self.projects_root.glob("*/metadata.json"), reverse=True):
            projects.append(self.get_project_summary(metadata_path.parent.name))
        return sorted(projects, key=lambda item: item.updated_at, reverse=True)

    def _build_project_summary(
        self,
        project_id: str,
        metadata: dict[str, Any],
        *,
        panels: list[PanelBox],
        jobs: list[JobRecord],
        videos: list[VideoFile],
    ) -> ProjectSummary:
        stage_states = self._default_stage_states()
        for stage, state in metadata.get("stage_states", {}).items():
            stage_states[PipelineStage(stage)] = StageState.model_validate(state)
        thumbnail_path = self._project_dir(project_id) / "thumbnails" / "cover.jpg"
        video_thumbnail_path = self._project_dir(project_id) / "thumbnails" / "video_intro.jpg"

        return ProjectSummary(
            id=metadata["id"],
            name=metadata["name"],
            source_type=SourceType(metadata["source_type"]),
            source_reference=metadata.get("source_reference"),
            created_at=datetime.fromisoformat(metadata["created_at"]),
            updated_at=datetime.fromisoformat(metadata["updated_at"]),
            chapter_metadata=ChapterMetadata.model_validate(metadata.get("chapter_metadata", {})),
            stage_states=stage_states,
            page_count=len(list((self._project_dir(project_id) / "pages").glob("*"))),
            panel_count=len(panels),
            kept_panel_count=sum(1 for panel in panels if panel.keep),
            thumbnail_url=self._relative_media_url(thumbnail_path) if thumbnail_path.exists() else None,
            video_thumbnail_url=self._relative_media_url(video_thumbnail_path) if video_thumbnail_path.exists() else None,
            # Filter out "zombie" jobs whose stage is already completed
            # but whose status is still QUEUED/RUNNING in the job log.
            # This happens after worker crashes that leave entries marked
            # "running" with no actual process, and after worker restarts
            # that re-queue old jobs as "Recovered after worker restart".
            # Surfacing those as active_jobs makes the Preview page show
            # a permanent "Rendering..." overlay on a video that's been
            # done for hours. Treat a job as zombie if its stage's
            # state is already `completed`.
            active_jobs=[
                job for job in jobs
                if job.status in {JobStatus.QUEUED, JobStatus.RUNNING, JobStatus.PAUSED}
                and not (
                    job.stage in stage_states
                    and getattr(stage_states[job.stage], "status", None) == StageStatus.COMPLETED
                )
            ],
            latest_video=videos[-1] if videos else None,
            voice_config=VoiceConfig.model_validate(metadata.get("voice_config", {})),
            video_config=VideoConfig.model_validate(metadata.get("video_config", {})),
            music_config=MusicConfig.model_validate(metadata.get("music_config", {})),
            pipeline_config=PipelineConfig.model_validate(metadata.get("pipeline_config", {})),
        )

    def get_project_summary(self, project_id: str) -> ProjectSummary:
        metadata = read_json(self._metadata_path(project_id))
        if not metadata:
            raise FileNotFoundError(f"Unknown project: {project_id}")
        panels = self.load_panels(project_id)
        videos = self.list_videos(project_id)
        jobs = self.list_jobs(project_id)
        return self._build_project_summary(
            project_id,
            metadata,
            panels=panels,
            jobs=jobs,
            videos=videos,
        )

    def get_project(self, project_id: str) -> ProjectDetail:
        metadata = read_json(self._metadata_path(project_id))
        if not metadata:
            raise FileNotFoundError(f"Unknown project: {project_id}")
        panels = self.sanitize_panel_boxes(project_id, self.load_panels(project_id))
        panels = self._enrich_panels_from_script_blocks(project_id, panels)
        panels = self._enrich_panels_from_quality_report(project_id, panels)
        audio_files = self.list_audio_files(project_id)
        videos = self.list_videos(project_id)
        jobs = self.list_jobs(project_id)
        summary = self._build_project_summary(
            project_id,
            metadata,
            panels=panels,
            jobs=jobs,
            videos=videos,
        )

        return ProjectDetail(
            id=summary.id,
            name=summary.name,
            source_type=summary.source_type,
            source_reference=summary.source_reference,
            created_at=summary.created_at,
            updated_at=summary.updated_at,
            chapter_metadata=summary.chapter_metadata,
            stage_states=summary.stage_states,
            page_count=summary.page_count,
            panel_count=summary.panel_count,
            kept_panel_count=summary.kept_panel_count,
            panels=panels,
            script_lines=self.load_script(project_id),
            script_story=self.load_script_story(project_id),
            story_segments=self.load_story_segments(project_id),
            script_display_metadata=self.load_script_display_metadata(project_id, jobs=jobs),
            audio_files=audio_files,
            videos=videos,
            thumbnail_url=summary.thumbnail_url,
            video_thumbnail_url=summary.video_thumbnail_url,
            active_jobs=summary.active_jobs,
            latest_video=summary.latest_video,
            voice_config=summary.voice_config,
            video_config=summary.video_config,
            music_config=summary.music_config,
            pipeline_config=summary.pipeline_config,
            available_music_tracks=self.list_music_tracks(),
        )

    def update_project_metadata(self, project_id: str, **updates: object) -> ProjectDetail:
        metadata = read_json(self._metadata_path(project_id))
        metadata.update(updates)
        metadata["updated_at"] = self._now().isoformat()
        write_json(self._metadata_path(project_id), metadata)
        return self.get_project(project_id)

    def update_stage_state(
        self,
        project_id: str,
        stage: PipelineStage,
        status: StageStatus,
        progress: float | None = None,
        message: str | None = None,
    ) -> None:
        metadata = read_json(self._metadata_path(project_id))
        state = metadata["stage_states"].get(stage.value, {})
        state["stage"] = stage.value
        state["status"] = status.value
        if progress is not None:
            state["progress"] = _whole_progress(progress)
        if message is not None:
            state["message"] = message
        state["updated_at"] = self._now().isoformat()
        metadata["stage_states"][stage.value] = state
        metadata["updated_at"] = self._now().isoformat()
        write_json(self._metadata_path(project_id), metadata)

    def save_panels(self, project_id: str, panels: Iterable[PanelBox]) -> None:
        ordered_panels = self._normalize_panels_reading_order(panels)
        sanitized_panels = self.sanitize_panel_boxes(project_id, ordered_panels)
        normalized_panels: list[PanelBox] = []
        serialised = []
        for panel in sanitized_panels:
            normalized_panels.append(panel)
            serialised.append(panel.model_dump(mode="json"))
        write_json(self._panels_path(project_id), serialised)
        self._write_panel_previews(project_id, normalized_panels)
        self._invalidate_panel_derived_media(project_id)
        panel_quality = self._panel_quality.analyze(self._project_dir(project_id), normalized_panels)
        write_json(self._panel_quality_path(project_id), panel_quality)
        self.update_project_metadata(project_id)

    def invalidate_script_outputs(
        self,
        project_id: str,
        *,
        clear_generated_panel_narration: bool = False,
    ) -> None:
        """Clear script artifacts that no longer match the saved panel list."""

        project_dir = self._project_dir(project_id)
        output_dir = project_dir / "output"
        self._script_path(project_id).write_text("", encoding="utf-8")
        write_json(
            self._script_manifest_path(project_id),
            {
                "script_lines": [],
                "script_lines_strict": [],
                "script_lines_cinematic": [],
                "script_story": "",
                "story_segments": [],
                "script_mode": "story_segments_v1",
            },
        )
        for path in (
            self._script_quality_path(project_id),
            output_dir / "gemini_summary_cache.json",
            output_dir / "page_vision_cache.json",
            output_dir / "panel_captions_cache.json",
            output_dir / "panel_script_blocks.json",
            output_dir / "scene_summaries.json",
            output_dir / "story_bible.json",
            output_dir / "story_grounding.json",
            output_dir / "story_segments.json",
            output_dir / "narration_story.txt",
        ):
            path.unlink(missing_ok=True)

        if clear_generated_panel_narration:
            panels = self.load_panels(project_id)
            updated_panels: list[PanelBox] = []
            changed = False
            for panel in panels:
                keep_narration = bool(panel.manual_narration or panel.narration_locked)
                if keep_narration:
                    updated_panels.append(panel)
                    continue
                if panel.narration or panel.narration_source:
                    changed = True
                updated_panels.append(
                    panel.model_copy(
                        update={
                            "narration": None,
                            "manual_narration": False,
                            "narration_source": None,
                        }
                    )
                )
            if changed:
                write_json(self._panels_path(project_id), [panel.model_dump(mode="json") for panel in updated_panels])

        self.update_project_metadata(project_id)

    def load_panels(self, project_id: str) -> list[PanelBox]:
        """Read + validate panels.json with mtime-keyed in-process cache.

        Unord's panels.json is ~4 MB and pydantic-validating each entry
        ~3500 times takes hundreds of ms. Every API request hit this
        path. Now we stat() the file (cheap), key the cache by mtime,
        and return the cached list as long as the file hasn't changed.
        Worker writes bump the mtime -> next call misses + rebuilds.
        Returns a SHALLOW COPY so downstream mutation is safe.
        """
        panels_path = self._panels_path(project_id)
        try:
            mtime = panels_path.stat().st_mtime
        except FileNotFoundError:
            return []
        with type(self)._CACHE_LOCK:
            cached = type(self)._PANELS_CACHE.get(project_id)
            if cached is not None and cached[0] == mtime:
                return list(cached[1])
        payload = read_json(panels_path, default=[])
        parsed = [PanelBox.model_validate(item) for item in payload]
        with type(self)._CACHE_LOCK:
            type(self)._PANELS_CACHE[project_id] = (mtime, parsed)
        return list(parsed)

    def sanitize_panel_boxes(self, project_id: str, panels: Iterable[PanelBox]) -> list[PanelBox]:
        page_sizes = self._page_size_lookup(project_id)
        return [self._sanitize_panel_box(panel, page_sizes) for panel in panels]

    def sanitize_panel_box(self, project_id: str, panel: PanelBox) -> PanelBox:
        return self._sanitize_panel_box(panel, self._page_size_lookup(project_id))

    def _normalize_panels_reading_order(self, panels: Iterable[PanelBox]) -> list[PanelBox]:
        ordered = sorted(
            list(panels),
            key=lambda panel: (
                int(panel.page),
                int(panel.y),
                int(panel.x),
                int(panel.height),
                int(panel.width),
                int(panel.order),
                int(panel.panel),
                str(panel.id),
            ),
        )
        page_counters: dict[int, int] = {}
        normalized: list[PanelBox] = []
        for index, panel in enumerate(ordered, start=1):
            page_number = int(panel.page)
            page_counters[page_number] = page_counters.get(page_number, 0) + 1
            normalized.append(
                panel.model_copy(
                    update={
                        "order": index,
                        "panel": page_counters[page_number],
                    }
                )
            )
        return normalized

    def _page_size_lookup(self, project_id: str) -> dict[int, tuple[int, int]]:
        # Cache page dimensions per project. The hot path is the panel
        # editor on a 3447-page project, which used to open every page
        # image (PIL header read) on every API request — about 25 s of
        # blocking I/O. Pages do not change after ingestion, so we key
        # the cache on the pages-directory mtime plus the entry count
        # (cheap stat). If either changes we rebuild.
        pages_dir = self._project_dir(project_id) / "pages"
        try:
            dir_stat = pages_dir.stat()
            cache_key = (dir_stat.st_mtime, dir_stat.st_size)
        except FileNotFoundError:
            cache_key = (0.0, 0)

        cls = type(self)
        with cls._CACHE_LOCK:
            cached = cls._PAGE_SIZE_CACHE.get(project_id)
            if cached is not None and cached[0] == cache_key:
                return dict(cached[1])

        page_sizes: dict[int, tuple[int, int]] = {}
        for index, page_path in enumerate(self.list_page_paths(project_id), start=1):
            try:
                with Image.open(page_path) as image:
                    page_sizes[index] = image.size
            except Exception:
                logger.exception("Unable to read page size for %s", page_path)

        with cls._CACHE_LOCK:
            cls._PAGE_SIZE_CACHE[project_id] = (cache_key, page_sizes)
        return dict(page_sizes)

    def _sanitize_panel_box(self, panel: PanelBox, page_sizes: dict[int, tuple[int, int]]) -> PanelBox:
        page_size = page_sizes.get(int(panel.page))
        if not page_size:
            return panel.model_copy(
                update={
                    "x": max(int(panel.x), 0),
                    "y": max(int(panel.y), 0),
                    "width": max(int(panel.width), 1),
                    "height": max(int(panel.height), 1),
                }
            )

        page_width, page_height = page_size
        width = min(max(int(panel.width), 1), max(page_width, 1))
        height = min(max(int(panel.height), 1), max(page_height, 1))
        max_x = max(page_width - width, 0)
        max_y = max(page_height - height, 0)
        x = min(max(int(panel.x), 0), max_x)
        y = min(max(int(panel.y), 0), max_y)
        return panel.model_copy(update={"x": x, "y": y, "width": width, "height": height})

    def save_script(
        self,
        project_id: str,
        script_lines: list[str],
        story_block: str | None = None,
        *,
        strict_lines: list[str] | None = None,
        slot_evidence: list[dict[str, Any]] | None = None,
        job_id: str | None = None,
    ) -> None:
        aligned_lines = self._align_script_lines(project_id, script_lines)
        aligned_strict_lines = self._align_script_lines(project_id, strict_lines) if strict_lines is not None else list(aligned_lines)
        mapped_panels = self._panels_with_script_mapping(project_id, aligned_lines)
        final_lines = aligned_lines
        final_strict_lines = aligned_strict_lines
        if mapped_panels is not None:
            final_lines = [
                str(panel.narration or "")
                for panel in sorted(mapped_panels, key=lambda item: item.order)
                if panel.keep
            ]
            final_strict_lines = []
            strict_index = 0
            for panel in sorted(mapped_panels, key=lambda item: item.order):
                if not panel.keep:
                    continue
                strict_line = aligned_strict_lines[strict_index] if strict_index < len(aligned_strict_lines) else ""
                if panel.narration_locked and (panel.narration or "").strip():
                    strict_line = str(panel.narration or "")
                final_strict_lines.append(strict_line)
                strict_index += 1
            write_json(self._panels_path(project_id), [panel.model_dump(mode="json") for panel in mapped_panels])
        story_text = self._script_cleaner.clean_story_block(str(story_block or "").strip())
        if not story_text or final_lines != aligned_lines or any(panel.narration_locked for panel in mapped_panels or []):
            story_text = self._compose_story_from_lines(final_lines).strip()
        self._script_path(project_id).write_text("\n".join(final_lines) + ("\n" if final_lines else ""), encoding="utf-8")
        write_json(
            self._script_manifest_path(project_id),
            {
                "script_lines": final_lines,
                "script_lines_strict": final_strict_lines,
                "script_lines_cinematic": final_lines,
                "script_story": story_text,
            },
        )
        self._write_script_artifact_metadata(project_id, job_id=job_id, script_mode="panel_lines")
        if mapped_panels is not None:
            write_json(
                self._project_dir(project_id) / "output" / "panel_script_blocks.json",
                self._build_panel_script_blocks(
                    mapped_panels,
                    final_lines,
                    final_strict_lines,
                    slot_evidence or [],
                ),
            )
        quality_report = self._script_quality.analyze(
            mapped_panels or self._enrich_panels_from_script_blocks(project_id, self.load_panels(project_id)),
            final_lines,
        )
        write_json(self._script_quality_path(project_id), quality_report)
        self.update_project_metadata(project_id)

    def save_story_segments(
        self,
        project_id: str,
        story_segments: list[StorySegment | dict[str, Any]],
        *,
        story_block: str | None = None,
        job_id: str | None = None,
    ) -> None:
        input_segment_ids = [
            str((segment.id if isinstance(segment, StorySegment) else segment.get("id", "")) or "").strip()
            for segment in story_segments
        ]
        normalized_segments = self._normalize_story_segments_payload(project_id, story_segments)
        normalized_segment_ids = [segment.id for segment in normalized_segments]
        script_lines = [segment.text.strip() for segment in normalized_segments if segment.keep]
        story_text = (
            self._script_cleaner.clean_story_block(str(story_block or "").strip())
            if input_segment_ids == normalized_segment_ids
            else ""
        )
        if not story_text:
            story_text = self._compose_story_from_lines(script_lines).strip()

        self._script_path(project_id).write_text(
            "\n".join(script_lines) + ("\n" if script_lines else ""),
            encoding="utf-8",
        )
        write_json(
            self._script_manifest_path(project_id),
            {
                "script_lines": script_lines,
                "script_lines_strict": list(script_lines),
                "script_lines_cinematic": list(script_lines),
                "script_story": story_text,
                "story_segments": [segment.model_dump(mode="json") for segment in normalized_segments],
                "script_mode": "story_segments_v1",
            },
        )

        output_dir = self._project_dir(project_id) / "output"
        write_json(output_dir / "story_segments.json", [segment.model_dump(mode="json") for segment in normalized_segments])
        write_json(output_dir / "panel_script_blocks.json", [])
        (output_dir / "narration_story.txt").write_text(
            story_text.strip() + ("\n" if story_text.strip() else ""),
            encoding="utf-8",
        )
        self._write_script_artifact_metadata(project_id, job_id=job_id, script_mode="story_segments")

        panel_vision_records = read_json(output_dir / "panel_vision_final.json", default=[])
        panel_evidence_records = self._script_evidence_records(project_id)
        quality_report = self._script_quality.analyze_story_segments(
            normalized_segments,
            panel_vision_records=panel_vision_records if isinstance(panel_vision_records, list) else None,
            panel_evidence_records=panel_evidence_records if isinstance(panel_evidence_records, list) else None,
            panels=self.load_panels(project_id),
        )
        write_json(self._script_quality_path(project_id), quality_report)
        self.update_project_metadata(project_id)

    def load_script_quality_report(self, project_id: str) -> dict[str, object]:
        report = read_json(self._script_quality_path(project_id), default=None)
        if isinstance(report, dict) and int(report.get("analysis_version", 0) or 0) >= 3:
            return report
        project = self.get_project(project_id)
        story_segments = self.load_story_segments(project_id)
        panel_vision_records = read_json(self._project_dir(project_id) / "output" / "panel_vision_final.json", default=[])
        panel_evidence_records = self._script_evidence_records(project_id)
        if story_segments:
            report = self._script_quality.analyze_story_segments(
                story_segments,
                panel_vision_records=panel_vision_records if isinstance(panel_vision_records, list) else None,
                panel_evidence_records=panel_evidence_records if isinstance(panel_evidence_records, list) else None,
                panels=self.load_panels(project_id),
            )
        else:
            report = self._script_quality.analyze(project.panels, project.script_lines)
        write_json(self._script_quality_path(project_id), report)
        return report

    def _script_evidence_records(self, project_id: str) -> list[dict[str, object]]:
        output_dir = self._project_dir(project_id) / "output"
        records: list[dict[str, object]] = []
        transcript = read_json(output_dir / "transcript.json", default={})
        fragments = transcript.get("fragments", []) if isinstance(transcript, dict) else []
        if isinstance(fragments, list):
            records.extend([item for item in fragments if isinstance(item, dict) and bool(item.get("accepted", True))])
        if records:
            return records
        panel_evidence_payload = read_json(output_dir / "panel_evidence.json", default={})
        panel_evidence_records = (
            panel_evidence_payload.get("panels", [])
            if isinstance(panel_evidence_payload, dict)
            else panel_evidence_payload
        )
        if isinstance(panel_evidence_records, list):
            records.extend([item for item in panel_evidence_records if isinstance(item, dict)])
        return records

    def load_panel_quality_report(self, project_id: str) -> dict[str, object]:
        report = read_json(self._panel_quality_path(project_id), default=None)
        if isinstance(report, dict) and int(report.get("analysis_version", 0) or 0) >= 2:
            return report
        panels = self.load_panels(project_id)
        report = self._panel_quality.analyze(self._project_dir(project_id), panels)
        write_json(self._panel_quality_path(project_id), report)
        return report

    def _enrich_panels_from_quality_report(self, project_id: str, panels: list[PanelBox]) -> list[PanelBox]:
        report = self.load_panel_quality_report(project_id)
        risky_lookup: dict[str, list[str]] = {}
        for item in report.get("risky_panels", []):
            if not isinstance(item, dict):
                continue
            panel_id = str(item.get("panel_id", "")).strip()
            if not panel_id:
                continue
            reasons = [
                str(reason).strip()
                for reason in item.get("reasons", [])
                if str(reason).strip()
            ]
            if "corner_wedge" in reasons and not bool(item.get("corner_wedge")):
                corner_score = float(item.get("corner_wedge_score", 0.0) or 0.0)
                if corner_score < 0.42:
                    reasons = [reason for reason in reasons if reason != "corner_wedge"]
            risky_lookup[panel_id] = reasons

        enriched: list[PanelBox] = []
        for panel in panels:
            review_flags = list(risky_lookup.get(panel.id, []))
            if panel.auto_skipped and "auto_skipped" not in review_flags:
                review_flags.insert(0, "auto_skipped")
            enriched.append(panel.model_copy(update={"review_flags": review_flags}))
        return enriched

    def load_script(self, project_id: str) -> list[str]:
        manifest = read_json(self._script_manifest_path(project_id), default=None)
        if isinstance(manifest, dict) and isinstance(manifest.get("script_lines"), list):
            return [str(line) for line in manifest["script_lines"]]
        segments = self.load_story_segments(project_id)
        if segments:
            return [segment.text for segment in segments if segment.keep]
        path = self._script_path(project_id)
        if not path.exists():
            return []
        content = path.read_text(encoding="utf-8")
        if content == "":
            return []
        return content.splitlines()

    def load_story_segments(self, project_id: str) -> list[StorySegment]:
        manifest = read_json(self._script_manifest_path(project_id), default=None)
        if isinstance(manifest, dict) and isinstance(manifest.get("story_segments"), list):
            segments: list[StorySegment] = []
            for index, item in enumerate(manifest.get("story_segments") or [], start=1):
                if not isinstance(item, dict):
                    continue
                payload = dict(item)
                payload.setdefault("id", f"segment_{index:03d}")
                payload.setdefault("order", index)
                payload.setdefault("text", "")
                payload.setdefault("keep", True)
                payload.setdefault("panel_ids", [])
                segments.append(StorySegment.model_validate(payload))
            return sorted(segments, key=lambda item: item.order)

        script_lines = []
        path = self._script_path(project_id)
        if path.exists():
            content = path.read_text(encoding="utf-8")
            script_lines = content.splitlines() if content else []
        if not script_lines:
            return []

        kept_panels = [panel for panel in sorted(self.load_panels(project_id), key=lambda item: item.order) if panel.keep]
        fallback_segments: list[StorySegment] = []
        for index, line in enumerate(script_lines, start=1):
            panel = kept_panels[index - 1] if index - 1 < len(kept_panels) else None
            panel_ids = [panel.id] if panel else []
            panel_order = int(panel.order) if panel is not None else None
            fallback_segments.append(
                StorySegment(
                    id=f"segment_{index:03d}",
                    order=index,
                    text=str(line or "").strip(),
                    keep=True,
                    panel_ids=panel_ids,
                    panel_start=panel_order,
                    panel_end=panel_order,
                    scene_id=index,
                    title=f"Segment {index}",
                    representative_panel_id=panel.id if panel is not None else None,
                )
            )
        return fallback_segments

    def _normalize_story_segments_payload(
        self,
        project_id: str,
        story_segments: list[StorySegment | dict[str, Any]],
    ) -> list[StorySegment]:
        panels_by_id = {panel.id: panel for panel in self.load_panels(project_id)}
        normalized: list[StorySegment] = []
        for index, raw in enumerate(story_segments, start=1):
            segment = raw if isinstance(raw, StorySegment) else StorySegment.model_validate(raw)
            panel_ids = [
                str(panel_id).strip()
                for panel_id in segment.panel_ids
                if str(panel_id).strip() in panels_by_id
            ]
            covered_orders = sorted(
                int(panels_by_id[panel_id].order)
                for panel_id in panel_ids
                if panel_id in panels_by_id
            )
            representative_panel_id = (
                str(segment.representative_panel_id).strip()
                if segment.representative_panel_id and str(segment.representative_panel_id).strip() in panels_by_id
                else panel_ids[0] if panel_ids else None
            )
            normalized.append(
                StorySegment(
                    id=str(segment.id or f"segment_{index:03d}").strip() or f"segment_{index:03d}",
                    order=index,
                    text=" ".join(str(segment.text or "").split()).strip(),
                    keep=bool(getattr(segment, "keep", True)),
                    panel_ids=panel_ids,
                    panel_start=covered_orders[0] if covered_orders else segment.panel_start,
                    panel_end=covered_orders[-1] if covered_orders else segment.panel_end,
                    scene_id=segment.scene_id if segment.scene_id is not None else index,
                    title=str(segment.title or f"Segment {index}").strip() or f"Segment {index}",
                    representative_panel_id=representative_panel_id,
                    visual_only=bool(getattr(segment, "visual_only", False)),
                    suppression_reason=str(getattr(segment, "suppression_reason", "") or "").strip() or None,
                )
            )
        chronological = sorted(
            normalized,
            key=lambda segment: (
                segment.panel_start is None,
                int(segment.panel_start or segment.order or 0),
                int(segment.panel_end or segment.panel_start or segment.order or 0),
                int(segment.order or 0),
            ),
        )
        return [segment.model_copy(update={"order": index}) for index, segment in enumerate(chronological, start=1)]

    def load_script_story(self, project_id: str) -> str | None:
        manifest = read_json(self._script_manifest_path(project_id), default=None)
        if isinstance(manifest, dict):
            story = manifest.get("script_story")
            if isinstance(story, str):
                return story
        segments = self.load_story_segments(project_id)
        if segments:
            story = self._compose_story_from_lines([segment.text for segment in segments if segment.keep and segment.text.strip()])
            return story or None
        lines = self.load_script(project_id)
        story = self._compose_story_from_lines(lines)
        return story or None

    def _write_script_artifact_metadata(
        self,
        project_id: str,
        *,
        job_id: str | None,
        script_mode: str,
    ) -> None:
        output_dir = ensure_dir(self._project_dir(project_id) / "output")
        now = self._now().isoformat()
        write_json(
            output_dir / "script_artifact.json",
            {
                "script_path": str(self._script_path(project_id).resolve()),
                "script_manifest_path": str(self._script_manifest_path(project_id).resolve()),
                "story_path": str((output_dir / "narration_story.txt").resolve()),
                "job_id": job_id,
                "script_mode": script_mode,
                "created_at": now,
            },
        )

    def load_script_display_metadata(self, project_id: str, *, jobs: list[JobRecord] | None = None) -> dict[str, Any]:
        project_dir = self._project_dir(project_id)
        output_dir = project_dir / "output"
        script_path = self._script_path(project_id)
        manifest_path = self._script_manifest_path(project_id)
        story_path = output_dir / "narration_story.txt"
        artifact = read_json(self._script_artifact_metadata_path(project_id), default={})
        if not isinstance(artifact, dict):
            artifact = {}

        script_jobs = [
            job
            for job in (jobs if jobs is not None else self.list_jobs(project_id))
            if job.stage == PipelineStage.SCRIPT_GENERATION
        ]
        latest_job = script_jobs[0] if script_jobs else None
        latest_completed = next((job for job in script_jobs if job.status == JobStatus.COMPLETED), None)
        latest_active = next((job for job in script_jobs if job.status in {JobStatus.QUEUED, JobStatus.RUNNING, JobStatus.PAUSED}), None)

        created_at = str(artifact.get("created_at") or "").strip()
        if not created_at:
            candidates = [path for path in (manifest_path, script_path, story_path) if path.exists()]
            if candidates:
                created_at = datetime.fromtimestamp(max(path.stat().st_mtime for path in candidates)).isoformat()

        displayed_script_path = str(manifest_path.resolve()) if manifest_path.exists() else str(script_path.resolve()) if script_path.exists() else None
        displayed_job_id = str(artifact.get("job_id") or "").strip() or None
        latest_completed_path = str(story_path.resolve()) if latest_completed and story_path.exists() else None

        stale_reason = ""
        is_stale = False
        if latest_active and displayed_script_path:
            active_created = latest_active.created_at.isoformat()
            if not created_at or active_created > created_at:
                is_stale = True
                stale_reason = "script_generation_in_progress"
        elif latest_completed and displayed_script_path:
            if displayed_job_id and displayed_job_id != latest_completed.id:
                is_stale = True
                stale_reason = "newer_completed_job_available"

        return {
            "displayed_script_path": displayed_script_path,
            "displayed_script_job_id": displayed_job_id,
            "displayed_script_created_at": created_at or None,
            "latest_job_id": latest_job.id if latest_job else None,
            "latest_job_status": latest_job.status.value if latest_job else None,
            "latest_completed_script_path": latest_completed_path,
            "latest_completed_script_job_id": latest_completed.id if latest_completed else None,
            "is_displaying_stale_script": is_stale,
            "stale_reason": stale_reason,
        }

    def reset_pipeline_from_stage(self, project_id: str, stage: PipelineStage) -> None:
        project_dir = self._project_dir(project_id)
        self._reset_directory(self._panel_previews_dir(project_id))

        if stage == PipelineStage.INGESTION:
            self._reset_directory(project_dir / "pages")
            self._reset_directory(project_dir / "thumbnails")

        if stage in {PipelineStage.INGESTION, PipelineStage.PANEL_DETECTION}:
            write_json(self._panels_path(project_id), [])
            self._reset_directory(project_dir / "ocr")
            self._reset_directory(project_dir / "translations")
            self._reset_directory(project_dir / "output")

        if stage in {
            PipelineStage.INGESTION,
            PipelineStage.PANEL_DETECTION,
            PipelineStage.PANEL_REVIEW,
            PipelineStage.CHARACTER_REVIEW,
            PipelineStage.CHARACTER_PORTRAIT,
            PipelineStage.PANEL_VISION_EXTRACTION,
            PipelineStage.PANEL_VISION_QUALITY,
        }:
            shutil.rmtree(project_dir / "characters" / "review", ignore_errors=True)
            (project_dir / "output" / "character_review_state.json").unlink(missing_ok=True)
            (project_dir / "output" / "canonical_characters.json").unlink(missing_ok=True)
            (project_dir / "output" / "panel_vision.json").unlink(missing_ok=True)
            (project_dir / "output" / "panel_vision_final.json").unlink(missing_ok=True)
            (project_dir / "output" / "ocr_audit.json").unlink(missing_ok=True)

        if stage in {
            PipelineStage.INGESTION,
            PipelineStage.PANEL_DETECTION,
            PipelineStage.PANEL_REVIEW,
            PipelineStage.CHARACTER_REVIEW,
            PipelineStage.CHARACTER_PORTRAIT,
            PipelineStage.PANEL_VISION_EXTRACTION,
            PipelineStage.PANEL_VISION_QUALITY,
            PipelineStage.SCRIPT_GENERATION,
        }:
            self._script_path(project_id).write_text("", encoding="utf-8")
            write_json(
                self._script_manifest_path(project_id),
                {
                    "script_lines": [],
                    "script_lines_strict": [],
                    "script_lines_cinematic": [],
                    "script_story": "",
                    "story_segments": [],
                    "script_mode": "story_segments_v1",
                },
            )
            (self._script_quality_path(project_id)).unlink(missing_ok=True)

        if stage in {
            PipelineStage.INGESTION,
            PipelineStage.PANEL_DETECTION,
            PipelineStage.PANEL_REVIEW,
            PipelineStage.CHARACTER_REVIEW,
            PipelineStage.CHARACTER_PORTRAIT,
            PipelineStage.PANEL_VISION_EXTRACTION,
            PipelineStage.PANEL_VISION_QUALITY,
            PipelineStage.SCRIPT_GENERATION,
            PipelineStage.NARRATION_GENERATION,
        }:
            self._reset_directory(project_dir / "audio")

        if stage in {
            PipelineStage.INGESTION,
            PipelineStage.PANEL_DETECTION,
            PipelineStage.PANEL_REVIEW,
            PipelineStage.CHARACTER_REVIEW,
            PipelineStage.CHARACTER_PORTRAIT,
            PipelineStage.PANEL_VISION_EXTRACTION,
            PipelineStage.PANEL_VISION_QUALITY,
            PipelineStage.SCRIPT_GENERATION,
            PipelineStage.NARRATION_GENERATION,
            PipelineStage.VIDEO_RENDERING,
        }:
            self._reset_directory(project_dir / "video")
            self._reset_directory(project_dir / "exports")

        self.update_project_metadata(project_id)

    def reset_generated_outputs_after_stage(self, project_id: str, stage: PipelineStage) -> None:
        project_dir = self._project_dir(project_id)

        if stage == PipelineStage.PANEL_REVIEW:
            shutil.rmtree(project_dir / "characters" / "review", ignore_errors=True)
            (project_dir / "output" / "character_review_state.json").unlink(missing_ok=True)

        if stage in {PipelineStage.PANEL_REVIEW, PipelineStage.CHARACTER_REVIEW}:
            for filename in (
                "canonical_characters.json",
                "panel_vision.json",
                "panel_vision_final.json",
                "ocr_audit.json",
            ):
                (project_dir / "output" / filename).unlink(missing_ok=True)

        if stage in {
            PipelineStage.PANEL_REVIEW,
            PipelineStage.CHARACTER_REVIEW,
            PipelineStage.CHARACTER_PORTRAIT,
            PipelineStage.PANEL_VISION_EXTRACTION,
            PipelineStage.PANEL_VISION_QUALITY,
        }:
            self._script_path(project_id).write_text("", encoding="utf-8")
            write_json(
                self._script_manifest_path(project_id),
                {
                    "script_lines": [],
                    "script_lines_strict": [],
                    "script_lines_cinematic": [],
                    "script_story": "",
                    "story_segments": [],
                    "script_mode": "story_segments_v1",
                },
            )
            self._script_quality_path(project_id).unlink(missing_ok=True)

        if stage in {
            PipelineStage.PANEL_REVIEW,
            PipelineStage.CHARACTER_REVIEW,
            PipelineStage.CHARACTER_PORTRAIT,
            PipelineStage.PANEL_VISION_EXTRACTION,
            PipelineStage.PANEL_VISION_QUALITY,
            PipelineStage.SCRIPT_GENERATION,
            PipelineStage.NARRATION_GENERATION,
        }:
            self._reset_directory(project_dir / "audio")

        if stage in {
            PipelineStage.PANEL_REVIEW,
            PipelineStage.CHARACTER_REVIEW,
            PipelineStage.CHARACTER_PORTRAIT,
            PipelineStage.PANEL_VISION_EXTRACTION,
            PipelineStage.PANEL_VISION_QUALITY,
            PipelineStage.SCRIPT_GENERATION,
            PipelineStage.NARRATION_GENERATION,
            PipelineStage.VIDEO_RENDERING,
        }:
            self._reset_directory(project_dir / "video")
            self._reset_directory(project_dir / "exports")

        self.update_project_metadata(project_id)

    def _align_script_lines(self, project_id: str, script_lines: list[str]) -> list[str]:
        panels = [panel for panel in sorted(self.load_panels(project_id), key=lambda item: item.order) if panel.keep]
        if not panels:
            return [str(line) for line in script_lines]

        cleaned_lines = [str(line) for line in script_lines]
        block_lookup = self._panel_script_block_lookup(project_id)
        line_target_indexes = [
            index
            for index, panel in enumerate(panels)
            if self._panel_should_hold_script_line(panel, block_lookup.get(panel.id, {}))
        ]
        aligned = ["" for _ in panels]
        if not cleaned_lines or not line_target_indexes:
            return aligned

        if len(cleaned_lines) == len(panels):
            if len(line_target_indexes) == len(panels):
                return cleaned_lines
            compressed_lines = [cleaned_lines[index] for index in line_target_indexes]
            for target_index, line in zip(line_target_indexes, compressed_lines, strict=False):
                aligned[target_index] = line
            return aligned

        if len(cleaned_lines) == len(line_target_indexes):
            for target_index, line in zip(line_target_indexes, cleaned_lines, strict=False):
                aligned[target_index] = line
            return aligned

        target_groups = self._build_script_target_groups(project_id, panels, line_target_indexes)
        grouped_lines = self._align_lines_to_slot_count(cleaned_lines, len(target_groups))

        for group, group_line in zip(target_groups, grouped_lines, strict=False):
            if not group:
                continue
            distributed_lines = self._distribute_group_line(group_line, len(group))
            for panel_index, distributed_line in zip(group, distributed_lines, strict=False):
                aligned[panel_index] = distributed_line

        return aligned

    def _build_script_target_groups(
        self,
        project_id: str,
        panels: list[PanelBox],
        line_target_indexes: list[int],
    ) -> list[list[int]]:
        if not line_target_indexes:
            return []

        output_dir = self._project_dir(project_id) / "output"
        scene_clusters = read_json(output_dir / "scene_clusters.json", default=[])
        target_set = set(line_target_indexes)
        order_to_index = {panel.order: index for index, panel in enumerate(panels)}
        groups: list[list[int]] = []
        seen: set[int] = set()

        for cluster in scene_clusters:
            panel_orders = cluster.get("panels") if isinstance(cluster, dict) else None
            if not isinstance(panel_orders, list):
                continue
            indexes = [
                order_to_index[panel_order]
                for panel_order in panel_orders
                if panel_order in order_to_index and order_to_index[panel_order] in target_set
            ]
            indexes = sorted(dict.fromkeys(indexes))
            if not indexes:
                continue
            groups.append(indexes)
            seen.update(indexes)

        for target_index in line_target_indexes:
            if target_index in seen:
                continue
            groups.append([target_index])

        return groups or [[target_index] for target_index in line_target_indexes]

    def _spread_line_positions(self, total_targets: int, total_lines: int) -> list[int]:
        if total_lines <= 0 or total_targets <= 0:
            return []
        if total_lines == 1:
            return [0]

        positions = [
            round(line_index * (total_targets - 1) / max(total_lines - 1, 1))
            for line_index in range(total_lines)
        ]

        for index in range(1, len(positions)):
            positions[index] = max(positions[index], positions[index - 1] + 1)

        for index in range(len(positions) - 2, -1, -1):
            positions[index] = min(positions[index], positions[index + 1] - 1)

        return [max(0, min(total_targets - 1, position)) for position in positions]

    def _align_lines_to_slot_count(self, script_lines: list[str], slot_count: int) -> list[str]:
        if slot_count <= 0:
            return []
        if not script_lines:
            return ["" for _ in range(slot_count)]
        if len(script_lines) == slot_count:
            return script_lines

        if len(script_lines) < slot_count:
            expanded_lines = self._expand_script_lines(script_lines, slot_count)
            if len(expanded_lines) == slot_count:
                return expanded_lines
            result = ["" for _ in range(slot_count)]
            positions = self._spread_line_positions(slot_count, len(expanded_lines))
            for line_index, slot in enumerate(positions):
                result[slot] = expanded_lines[line_index]
            return result

        return self._merge_script_lines_into_targets(script_lines, slot_count)

    def _distribute_group_line(self, line: str, target_count: int) -> list[str]:
        if target_count <= 0:
            return []
        normalized_line = " ".join(str(line).split()).strip()
        if not normalized_line:
            return ["" for _ in range(target_count)]

        fragments = self._expand_group_line_fragments(normalized_line, target_count)
        if not fragments:
            fragments = [self._normalize_script_fragment(normalized_line)]

        if len(fragments) >= target_count:
            return fragments[:target_count]

        distributed = ["" for _ in range(target_count)]
        positions = self._spread_line_positions(target_count, len(fragments))
        for fragment_index, slot in enumerate(positions):
            distributed[slot] = fragments[fragment_index]
        return distributed

    def _expand_group_line_fragments(self, line: str, target_count: int) -> list[str]:
        fragments = self._split_story_line_fragments(line)
        if len(fragments) >= target_count:
            return fragments

        expanded: list[str] = []
        for fragment in fragments or [line]:
            aggressive_parts = self._aggressive_story_subfragments(fragment)
            if aggressive_parts:
                expanded.extend(aggressive_parts)
            else:
                expanded.append(fragment)
            if len(expanded) >= target_count:
                break

        if len(expanded) >= target_count:
            return expanded

        final_expanded: list[str] = []
        for fragment in expanded or [line]:
            split_halves = self._split_fragment_by_words(fragment)
            final_expanded.extend(split_halves or [fragment])
            if len(final_expanded) >= target_count:
                break

        return final_expanded or fragments

    def _expand_script_lines(self, script_lines: list[str], target_count: int) -> list[str]:
        expanded: list[str] = []
        for line in script_lines:
            fragments = self._split_story_line_fragments(line)
            if not fragments:
                continue
            expanded.extend(fragments)
            if len(expanded) >= target_count:
                return expanded[:target_count]
        return expanded

    def _split_story_line_fragments(self, line: str) -> list[str]:
        text = " ".join(str(line).split()).strip()
        if not text:
            return []

        fragments = [
            fragment.strip()
            for fragment in re.split(r"(?<=[.!?])\s+|(?<=;)\s+", text)
            if fragment.strip()
        ]
        refined: list[str] = []
        for fragment in fragments:
            if len(fragment.split()) > 16 and "," in fragment:
                comma_parts = [part.strip() for part in re.split(r",\s+", fragment) if part.strip()]
                refined.extend(comma_parts or [fragment])
            else:
                refined.append(fragment)

        final_fragments: list[str] = []
        for fragment in refined:
            if len(fragment.split()) > 16:
                conjunction_parts = [
                    part.strip()
                    for part in re.split(r"\b(?:and|but|while|as|then)\b", fragment, flags=re.IGNORECASE)
                    if part.strip()
                ]
                final_fragments.extend(conjunction_parts or [fragment])
            else:
                final_fragments.append(fragment)

        normalized = [self._normalize_script_fragment(fragment) for fragment in final_fragments]
        return [fragment for fragment in normalized if fragment]

    def _aggressive_story_subfragments(self, fragment: str) -> list[str]:
        text = " ".join(str(fragment).split()).strip()
        if not text:
            return []

        parts = [part.strip() for part in re.split(r",\s+|;\s+|\b(?:and|but|while|then)\b", text, flags=re.IGNORECASE) if part.strip()]
        normalized = [self._normalize_script_fragment(part) for part in parts]
        return [part for part in normalized if part and part.casefold() != self._normalize_script_fragment(text).casefold()]

    def _split_fragment_by_words(self, fragment: str) -> list[str]:
        words = fragment.split()
        if len(words) < 8:
            return [self._normalize_script_fragment(fragment)]

        midpoint = len(words) // 2
        first = self._normalize_script_fragment(" ".join(words[:midpoint]))
        second = self._normalize_script_fragment(" ".join(words[midpoint:]))
        return [part for part in (first, second) if part]

    def _merge_script_lines_into_targets(self, script_lines: list[str], total_targets: int) -> list[str]:
        if total_targets <= 0:
            return []

        merged: list[str] = []
        total_lines = len(script_lines)
        for target_index in range(total_targets):
            start = round(target_index * total_lines / total_targets)
            end = round((target_index + 1) * total_lines / total_targets)
            chunk = [line.strip() for line in script_lines[start:end] if line.strip()]
            merged.append(" ".join(chunk))
        return merged

    def _normalize_script_fragment(self, fragment: str) -> str:
        text = " ".join(fragment.split()).strip(" ,;")
        if not text:
            return ""
        text = text[0].upper() + text[1:] if len(text) > 1 else text.upper()
        if text[-1] not in ".!?":
            text += "."
        return text

    def _panels_with_script_mapping(self, project_id: str, script_lines: list[str]) -> list[PanelBox] | None:
        panels = self.load_panels(project_id)
        if not panels:
            return None

        kept_panels = [panel for panel in sorted(panels, key=lambda item: item.order) if panel.keep]
        block_lookup = self._panel_script_block_lookup(project_id)
        script_lookup: dict[str, str | None] = {}
        for index, panel in enumerate(kept_panels):
            script_lookup[panel.id] = script_lines[index] if index < len(script_lines) else ""
        mapped_panels: list[PanelBox] = []
        for panel in panels:
            narration = script_lookup.get(panel.id) if panel.keep else panel.narration
            if panel.keep and panel.narration_locked and (panel.narration or "").strip():
                narration = panel.narration
            manual_narration = bool(panel.manual_narration)
            if panel.narration_locked and (narration or "").strip():
                manual_narration = True
            block = block_lookup.get(panel.id, {})
            visual_caption = self._panel_visual_caption(block, panel)
            mapped_panels.append(
                panel.model_copy(
                    update={
                        "narration": narration,
                        # Only preserve true human edits / locks as manual narration.
                        # Auto-generated fallback lines on textless panels should not
                        # become sticky evidence that forces the same bad slot mapping
                        # on every later save or regeneration.
                        "manual_narration": manual_narration,
                        "visual_caption": visual_caption,
                        "narration_source": self._infer_narration_source(panel, block, narration, visual_caption),
                    }
                )
            )
        return mapped_panels

    def _build_panel_script_blocks(
        self,
        panels: list[PanelBox],
        final_lines: list[str],
        strict_lines: list[str],
        slot_evidence: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        kept_panels = [panel for panel in sorted(panels, key=lambda item: item.order) if panel.keep]
        blocks: list[dict[str, Any]] = []
        for index, panel in enumerate(kept_panels):
            evidence = slot_evidence[index] if index < len(slot_evidence) and isinstance(slot_evidence[index], dict) else {}
            character_names = [
                str(name).strip()
                for name in evidence.get("character_names", []) or []
                if str(name).strip()
            ]
            dialogue = [
                str(item).strip()
                for item in evidence.get("dialogue", []) or []
                if str(item).strip()
            ]
            blocks.append(
                {
                    "panel_id": panel.id,
                    "panel_order": int(panel.order),
                    "page": int(panel.page),
                    "strict_narration": strict_lines[index] if index < len(strict_lines) else "",
                    "final_narration": final_lines[index] if index < len(final_lines) else "",
                    "text": str(evidence.get("ocr_text") or evidence.get("text") or panel.ocr_text or "").strip(),
                    "dialogue": dialogue,
                    "character_names": character_names,
                    "preferred_subject": str(evidence.get("preferred_subject") or "").strip(),
                    "visual_caption": str(
                        evidence.get("visual_caption")
                        or panel.visual_caption
                        or ""
                    ).strip(),
                    "scene_id": int(evidence.get("scene_id") or 0),
                    "scene_summary": str(evidence.get("scene_summary") or "").strip(),
                }
            )
        return blocks

    def _panel_script_block_lookup(self, project_id: str) -> dict[str, dict[str, Any]]:
        blocks = read_json(self._project_dir(project_id) / "output" / "panel_script_blocks.json", default=[])
        return {
            str(block.get("panel_id") or ""): block
            for block in blocks
            if isinstance(block, dict) and str(block.get("panel_id") or "").strip()
        }

    def _panel_visual_caption(self, block: dict[str, Any], panel: PanelBox) -> str | None:
        caption = str(block.get("visual_caption") or block.get("caption") or panel.visual_caption or "").strip()
        if caption:
            return caption
        visual = str(block.get("visual") or "").strip()
        if visual:
            return visual
        return str(panel.visual_caption or "").strip() or None

    def _infer_narration_source(
        self,
        panel: PanelBox,
        block: dict[str, Any],
        narration: str | None,
        visual_caption: str | None,
    ) -> str | None:
        block_source = str(block.get("narration_source") or "").strip().lower()
        if block_source in {"gemini", "ocr", "vision", "vision_caption", "fallback"}:
            return block_source

        narration_text = str(narration or "").strip()
        if not narration_text:
            return None

        extracted_text = str(block.get("extracted_text") or panel.ocr_text or "").strip().lower()
        if is_usable_ocr_text(extracted_text):
            return "ocr"
        if str(visual_caption or "").strip():
            return "vision"
        return "fallback"

    def _enrich_panels_from_script_blocks(self, project_id: str, panels: list[PanelBox]) -> list[PanelBox]:
        block_lookup = self._panel_script_block_lookup(project_id)
        if not block_lookup:
            return panels

        enriched: list[PanelBox] = []
        for panel in panels:
            block = block_lookup.get(panel.id, {})
            visual_caption = self._panel_visual_caption(block, panel)
            narration = panel.narration
            if not narration and panel.keep:
                narration = str(block.get("narration") or "").strip() or None
            enriched.append(
                panel.model_copy(
                    update={
                        "narration": narration,
                        "visual_caption": visual_caption,
                        "narration_source": self._infer_narration_source(panel, block, narration, visual_caption),
                    }
                )
            )
        return enriched

    def _panel_has_extracted_text(self, panel: PanelBox) -> bool:
        return is_usable_ocr_text(panel.ocr_text or "")

    def _panel_is_low_signal_script_fragment(self, panel: PanelBox, block: dict[str, Any] | None = None) -> bool:
        block = block or {}
        if panel.manual_narration or (panel.narration_locked and str(panel.narration or "").strip()):
            return False
        if self._panel_has_extracted_text(panel):
            return False
        if str(panel.visual_caption or "").strip():
            return False
        if str(block.get("visual_caption") or block.get("visual") or block.get("caption") or "").strip():
            return False
        if block.get("vision_rescue_eligible") or block.get("visual_signal"):
            return False

        flags = {
            str(flag).strip()
            for flag in panel.review_flags or []
            if str(flag).strip()
        }
        if not flags:
            return False
        if flags == {"whitespace"}:
            return False
        return flags.issubset(_LOW_SIGNAL_SCRIPT_FLAGS)

    def _panel_should_hold_script_line(self, panel: PanelBox, block: dict[str, Any] | None = None) -> bool:
        block = block or {}
        carries_existing_narration = bool(str(panel.narration or "").strip()) and not self._panel_is_low_signal_script_fragment(
            panel,
            block,
        )
        return bool(
            panel.manual_narration
            or (panel.narration_locked and str(panel.narration or "").strip())
            or self._panel_has_extracted_text(panel)
            or str(panel.visual_caption or "").strip()
            or carries_existing_narration
            or str(block.get("visual_caption") or block.get("visual") or block.get("caption") or "").strip()
            or block.get("vision_rescue_eligible")
            or block.get("visual_signal")
        )

    def _compose_story_from_lines(self, script_lines: list[str]) -> str:
        non_empty_lines = [line.strip() for line in script_lines if line.strip()]
        if not non_empty_lines:
            return ""
        paragraphs: list[str] = []
        chunk: list[str] = []
        for line in non_empty_lines:
            chunk.append(line)
            if len(chunk) >= 3:
                paragraphs.append(" ".join(chunk))
                chunk = []
        if chunk:
            paragraphs.append(" ".join(chunk))
        return "\n\n".join(paragraphs)

    def _ensure_panel_previews(self, project_id: str, panels: list[PanelBox]) -> None:
        if not panels:
            return
        preview_dir = self._panel_previews_dir(project_id)
        expected = {
            f"panel_{panel.order:03d}.png"
            for panel in sorted(panels, key=lambda item: item.order)
        }
        existing = {path.name for path in preview_dir.glob("panel_*.png")}
        crop_report = read_json(self._panel_crop_report_path(project_id), default={})
        if expected == existing and crop_report.get("crop_version") == PANEL_CROP_VERSION:
            return
        self._write_panel_previews(project_id, panels)

    def _write_panel_previews(self, project_id: str, panels: list[PanelBox]) -> None:
        preview_dir = self._panel_previews_dir(project_id)
        for existing in preview_dir.glob("panel_*.png"):
            existing.unlink(missing_ok=True)

        page_paths = self.list_page_paths(project_id)
        if not page_paths:
            return

        sanitized_panels = self.sanitize_panel_boxes(project_id, panels)
        crop_report: dict[str, Any] = {
            "crop_version": PANEL_CROP_VERSION,
            "project_id": project_id,
            "panel_count": len(sanitized_panels),
            "tightened_count": 0,
            "panels": [],
        }

        # Build id→panel lookup for cross-page stitching
        panel_by_id: dict[str, PanelBox] = {p.id: p for p in sanitized_panels}

        page_cache: dict[int, Image.Image] = {}
        try:
            for panel in sorted(sanitized_panels, key=lambda item: item.order):
                if panel.page <= 0 or panel.page > len(page_paths):
                    continue
                source_image = page_cache.get(panel.page)
                if source_image is None:
                    source_image = Image.open(page_paths[panel.page - 1]).convert("RGB")
                    page_cache[panel.page] = source_image

                left = max(0, int(panel.x))
                top = max(0, int(panel.y))
                right = min(source_image.width, int(panel.x + panel.width))
                bottom = min(source_image.height, int(panel.y + panel.height))
                if right <= left or bottom <= top:
                    crop = source_image.crop((0, 0, source_image.width, source_image.height))
                else:
                    crop = source_image.crop((left, top, right, bottom))

                # Stitch continuation pages for primary cross-page panels.
                # Follows the chain for panels spanning 3+ pages.
                is_primary = (
                    panel.multi_page_panel
                    and panel.spans_pages
                    and int(panel.page) == min(panel.spans_pages)
                    and panel.continuation_panel_ids
                )
                if is_primary:
                    # Walk the continuation chain in page order, collecting crops
                    crops: list[Image.Image] = [crop]
                    visited: set[str] = {panel.id}
                    cont_ids = sorted(
                        panel.continuation_panel_ids,
                        key=lambda cid: panel_by_id[cid].page if cid in panel_by_id else 0,
                    )
                    for cont_id in cont_ids:
                        if cont_id in visited:
                            continue
                        visited.add(cont_id)
                        continuation = panel_by_id.get(cont_id)
                        if continuation is None or not (0 < continuation.page <= len(page_paths)):
                            continue
                        cont_image = page_cache.get(continuation.page)
                        if cont_image is None:
                            cont_image = Image.open(page_paths[continuation.page - 1]).convert("RGB")
                            page_cache[continuation.page] = cont_image
                        c_left = max(0, int(continuation.x))
                        c_top = max(0, int(continuation.y))
                        c_right = min(cont_image.width, int(continuation.x + continuation.width))
                        c_bottom = min(cont_image.height, int(continuation.y + continuation.height))
                        if c_right > c_left and c_bottom > c_top:
                            crops.append(cont_image.crop((c_left, c_top, c_right, c_bottom)))
                    if len(crops) > 1:
                        stitch_width = max(c.width for c in crops)
                        parts = [
                            c.resize((stitch_width, c.height)) if c.width != stitch_width else c
                            for c in crops
                        ]
                        total_height = sum(p.height for p in parts)
                        stitched = Image.new("RGB", (stitch_width, total_height))
                        y_offset = 0
                        for part in parts:
                            stitched.paste(part, (0, y_offset))
                            y_offset += part.height
                        crop = stitched

                original_size = crop.size
                crop, crop_meta = self._tighten_panel_preview_crop(crop)
                if crop_meta["was_tightened"]:
                    crop_report["tightened_count"] = int(crop_report["tightened_count"]) + 1
                output_path = preview_dir / f"panel_{panel.order:03d}.png"
                crop.save(
                    output_path,
                    format="PNG",
                    compress_level=1,
                )
                crop_report["panels"].append({
                    "panel_id": panel.id,
                    "order": int(panel.order),
                    "page": int(panel.page),
                    "crop_version": PANEL_CROP_VERSION,
                    "crop_image_path": str(output_path),
                    "original_size": {"width": int(original_size[0]), "height": int(original_size[1])},
                    "saved_size": {"width": int(crop.size[0]), "height": int(crop.size[1])},
                    **crop_meta,
                })
        finally:
            for image in page_cache.values():
                image.close()
        write_json(self._panel_crop_report_path(project_id), crop_report)

    def _tighten_panel_preview_crop(self, crop: Image.Image) -> tuple[Image.Image, dict[str, Any]]:
        rgb = crop.convert("RGB")
        arr = np.array(rgb)
        height, width = arr.shape[:2]
        empty_meta = {
            "was_tightened": False,
            "content_bbox_in_crop": {"x": 0, "y": 0, "width": int(width), "height": int(height)},
            "tightened_bbox": {"x": 0, "y": 0, "width": int(width), "height": int(height)},
            "margin_percent_before": {"left": 0.0, "right": 0.0, "top": 0.0, "bottom": 0.0},
            "margin_percent_after": {"left": 0.0, "right": 0.0, "top": 0.0, "bottom": 0.0},
            "edge_trim_percent": {"left": 0.0, "right": 0.0, "top": 0.0, "bottom": 0.0},
        }
        if arr.size == 0 or min(width, height) < 48:
            return rgb, empty_meta

        border = max(3, min(16, min(width, height) // 20))
        border_pixels = np.concatenate([
            arr[:border, :, :].reshape(-1, 3),
            arr[-border:, :, :].reshape(-1, 3),
            arr[:, :border, :].reshape(-1, 3),
            arr[:, -border:, :].reshape(-1, 3),
        ])
        background = np.median(border_pixels.astype(np.float32), axis=0)
        distance = np.linalg.norm(arr.astype(np.float32) - background, axis=2)
        gray = np.dot(arr[..., :3], [0.299, 0.587, 0.114]).astype(np.uint8)

        try:
            import cv2  # Local import keeps ProjectStore lightweight for non-image tests.
            edges = cv2.Canny(gray, 40, 120) > 0
            edges = cv2.dilate(edges.astype(np.uint8), np.ones((2, 2), np.uint8), iterations=1) > 0
        except Exception:
            edges = np.zeros_like(gray, dtype=bool)

        bg_luma = float(np.dot(background, [0.299, 0.587, 0.114]))
        foreground = (distance > 14.0) | edges | (gray < max(210, bg_luma - 18))
        rows = np.where(foreground.mean(axis=1) > 0.006)[0]
        cols = np.where(foreground.mean(axis=0) > 0.006)[0]
        if rows.size == 0 or cols.size == 0:
            return rgb, empty_meta

        y1 = int(rows[0])
        y2 = int(rows[-1]) + 1
        x1 = int(cols[0])
        x2 = int(cols[-1]) + 1
        content_width = x2 - x1
        content_height = y2 - y1
        foreground_ratio = float(np.mean(foreground))
        if foreground_ratio < 0.08:
            pad_x = max(4, int(content_width * 0.05))
            pad_y = max(4, int(content_height * 0.10))
        else:
            pad_x = pad_y = max(0, int(min(width, height) * 0.002))
        left = max(x1 - pad_x, 0)
        top = max(y1 - pad_y, 0)
        right = min(x2 + pad_x, width)
        bottom = min(y2 + pad_y, height)

        trim_left = left
        trim_top = top
        trim_right = width - right
        trim_bottom = height - bottom
        before = {
            "left": round(x1 / max(width, 1), 4),
            "right": round((width - x2) / max(width, 1), 4),
            "top": round(y1 / max(height, 1), 4),
            "bottom": round((height - y2) / max(height, 1), 4),
        }
        meta = {
            "was_tightened": False,
            "content_bbox_in_crop": {"x": x1, "y": y1, "width": x2 - x1, "height": y2 - y1},
            "tightened_bbox": {"x": 0, "y": 0, "width": int(width), "height": int(height)},
            "margin_percent_before": before,
            "margin_percent_after": before,
            "edge_trim_percent": {"left": 0.0, "right": 0.0, "top": 0.0, "bottom": 0.0},
        }
        largest_trim = max(trim_left, trim_top, trim_right, trim_bottom)
        if largest_trim < max(2, int(min(width, height) * 0.004)):
            trimmed, edge_meta = self._trim_bright_preview_edges(rgb)
            if edge_meta["was_tightened"]:
                meta.update(edge_meta)
                return trimmed, meta
            return rgb, meta

        new_width = right - left
        new_height = bottom - top
        min_saved_dim = 24 if foreground_ratio < 0.08 else 40
        if new_width < min_saved_dim or new_height < min_saved_dim:
            return rgb, meta
        min_area_ratio = 0.015 if foreground_ratio < 0.08 else 0.48
        if (new_width * new_height) < (width * height * min_area_ratio):
            return rgb, meta

        tightened = rgb.crop((left, top, right, bottom))
        after = {
            "left": round(max(x1 - left, 0) / max(new_width, 1), 4),
            "right": round(max(right - x2, 0) / max(new_width, 1), 4),
            "top": round(max(y1 - top, 0) / max(new_height, 1), 4),
            "bottom": round(max(bottom - y2, 0) / max(new_height, 1), 4),
        }
        meta.update({
            "was_tightened": True,
            "tightened_bbox": {"x": left, "y": top, "width": new_width, "height": new_height},
            "margin_percent_after": after,
        })
        edge_trimmed, edge_meta = self._trim_bright_preview_edges(tightened)
        if edge_meta["was_tightened"]:
            edge_box = edge_meta["tightened_bbox"]
            combined_left = left + int(edge_box["x"])
            combined_top = top + int(edge_box["y"])
            combined_width = int(edge_box["width"])
            combined_height = int(edge_box["height"])
            meta.update({
                "tightened_bbox": {
                    "x": combined_left,
                    "y": combined_top,
                    "width": combined_width,
                    "height": combined_height,
                },
                "edge_trim_percent": edge_meta["edge_trim_percent"],
            })
            return edge_trimmed, meta
        return tightened, meta

    def _trim_bright_preview_edges(self, image: Image.Image) -> tuple[Image.Image, dict[str, Any]]:
        """Trim bright, low-information halos from saved panel previews.

        The detector coordinates remain unchanged for review/editing. This only
        affects the preview crop used by narration/video, where a slight inward
        crop is preferable to thin white gutters around manhwa panels.
        """
        rgb = image.convert("RGB")
        arr = np.array(rgb)
        height, width = arr.shape[:2]
        empty_meta = {
            "was_tightened": False,
            "tightened_bbox": {"x": 0, "y": 0, "width": int(width), "height": int(height)},
            "edge_trim_percent": {"left": 0.0, "right": 0.0, "top": 0.0, "bottom": 0.0},
        }
        if arr.size == 0 or min(width, height) < 80:
            return rgb, empty_meta

        gray = np.dot(arr[..., :3], [0.299, 0.587, 0.114]).astype(np.uint8)
        try:
            import cv2
            edges = cv2.Canny(gray, 45, 130) > 0
        except Exception:
            edges = np.zeros_like(gray, dtype=bool)

        step_x = max(2, min(10, width // 80))
        step_y = max(2, min(10, height // 80))
        max_x_trim = max(step_x, int(width * 0.20))
        max_y_trim = max(step_y, int(height * 0.20))

        def low_content_vertical(start: int, end: int) -> bool:
            strip = gray[:, start:end]
            strip_edges = edges[:, start:end]
            if strip.size == 0:
                return False
            white_ratio = float(np.mean(strip >= 238))
            dark_ratio = float(np.mean(strip <= 80))
            edge_ratio = float(np.mean(strip_edges))
            return white_ratio >= 0.86 and dark_ratio <= 0.08 and edge_ratio <= 0.020

        def low_content_horizontal(start: int, end: int) -> bool:
            strip = gray[start:end, :]
            strip_edges = edges[start:end, :]
            if strip.size == 0:
                return False
            white_ratio = float(np.mean(strip >= 238))
            dark_ratio = float(np.mean(strip <= 80))
            edge_ratio = float(np.mean(strip_edges))
            return white_ratio >= 0.88 and dark_ratio <= 0.08 and edge_ratio <= 0.026

        left = 0
        while left + step_x <= max_x_trim and low_content_vertical(left, left + step_x):
            left += step_x
        right = 0
        while right + step_x <= max_x_trim and low_content_vertical(width - right - step_x, width - right):
            right += step_x
        top = 0
        while top + step_y <= max_y_trim and low_content_horizontal(top, top + step_y):
            top += step_y
        bottom = 0
        while bottom + step_y <= max_y_trim and low_content_horizontal(height - bottom - step_y, height - bottom):
            bottom += step_y

        force_zoom_x = int(width * 0.035)
        force_zoom_y = int(height * 0.035)
        edge_is_bright = {
            "left": low_content_vertical(0, min(max(step_x, force_zoom_x), width)),
            "right": low_content_vertical(max(width - max(step_x, force_zoom_x), 0), width),
            "top": low_content_horizontal(0, min(max(step_y, force_zoom_y), height)),
            "bottom": low_content_horizontal(max(height - max(step_y, force_zoom_y), 0), height),
        }
        if edge_is_bright["left"]:
            left = max(left, force_zoom_x)
        if edge_is_bright["right"]:
            right = max(right, force_zoom_x)
        if edge_is_bright["top"]:
            top = max(top, force_zoom_y)
        if edge_is_bright["bottom"]:
            bottom = max(bottom, force_zoom_y)

        if max(left, right, top, bottom) < max(2, int(min(width, height) * 0.006)):
            return rgb, empty_meta
        new_left = min(left, max(width - 24, 0))
        new_top = min(top, max(height - 24, 0))
        new_right = max(width - right, new_left + 24)
        new_bottom = max(height - bottom, new_top + 24)
        if new_right > width or new_bottom > height or new_right <= new_left or new_bottom <= new_top:
            return rgb, empty_meta

        trimmed = rgb.crop((new_left, new_top, new_right, new_bottom))
        meta = {
            "was_tightened": True,
            "tightened_bbox": {
                "x": int(new_left),
                "y": int(new_top),
                "width": int(new_right - new_left),
                "height": int(new_bottom - new_top),
            },
            "edge_trim_percent": {
                "left": round(new_left / max(width, 1), 4),
                "right": round((width - new_right) / max(width, 1), 4),
                "top": round(new_top / max(height, 1), 4),
                "bottom": round((height - new_bottom) / max(height, 1), 4),
            },
        }
        return trimmed, meta

    def _reset_directory(self, directory: Path) -> None:
        shutil.rmtree(directory, ignore_errors=True)
        ensure_dir(directory)

    def _invalidate_panel_derived_media(self, project_id: str) -> None:
        project_dir = self._project_dir(project_id)
        shutil.rmtree(project_dir / "video" / "cache", ignore_errors=True)
        shutil.rmtree(project_dir / "temp" / "render", ignore_errors=True)
        for path in (
            project_dir / "output" / "dialogue_pipeline_manifest.json",
            project_dir / "output" / "scene_summaries.json",
            project_dir / "output" / "gemini_scenes.json",
            project_dir / "output" / "panel_script_blocks.json",
            project_dir / "output" / "scene_clusters.json",
            project_dir / "output" / "speaker_attributions.json",
            project_dir / "output" / "character_clusters.json",
            project_dir / "output" / "character_tracking.json",
            project_dir / "output" / "characters.json",
            project_dir / "output" / "character_dictionary.json",
            project_dir / "output" / "character_identity_report.json",
            project_dir / "output" / "character_review_state.json",
            project_dir / "ocr" / "dialogue_regions.json",
            project_dir / "translations" / "dialogue_regions_translated.json",
        ):
            try:
                path.unlink()
            except FileNotFoundError:
                pass
        shutil.rmtree(project_dir / "characters" / "review", ignore_errors=True)

    def list_page_paths(self, project_id: str) -> list[Path]:
        return sorted((self._project_dir(project_id) / "pages").glob("*"))

    def create_job(self, project_id: str, stage: PipelineStage, payload: dict[str, object] | None = None) -> JobRecord:
        job_payload = dict(payload or {})
        if job_payload.get("direct_runner"):
            job_payload.setdefault("runner_pid", os.getpid())
        job = JobRecord(
            id=uuid4().hex,
            project_id=project_id,
            stage=stage,
            status=JobStatus.QUEUED,
            payload=job_payload,
        )
        write_json(self._job_path(project_id, job.id), job.model_dump(mode="json"))
        return job

    def get_job(self, project_id: str, job_id: str) -> JobRecord:
        payload = read_json(self._job_path(project_id, job_id))
        if not payload:
            raise FileNotFoundError(f"Unknown job: {job_id}")
        return JobRecord.model_validate(payload)

    def list_jobs(self, project_id: str) -> list[JobRecord]:
        jobs: list[JobRecord] = []
        for path in sorted((self._project_dir(project_id) / "jobs").glob("*.json")):
            try:
                payload = read_json(path)
                if not payload:
                    continue
                jobs.append(JobRecord.model_validate(payload))
            except Exception:
                logger.exception("Skipping unreadable job record %s", path)
        jobs = self._reconcile_stale_direct_runner_jobs(project_id, jobs)
        return sorted(jobs, key=lambda job: job.created_at, reverse=True)

    def update_job(self, project_id: str, job_id: str, **updates: object) -> JobRecord:
        job = self.get_job(project_id, job_id)
        payload = job.model_dump(mode="json")
        if "progress" in updates:
            updates = {**updates, "progress": _whole_progress(updates.get("progress"))}
        payload.update(updates)
        write_json(self._job_path(project_id, job_id), payload)
        return self.get_job(project_id, job_id)

    def _reconcile_stale_direct_runner_jobs(self, project_id: str, jobs: list[JobRecord]) -> list[JobRecord]:
        updated = False
        current_stage_states = read_json(self._metadata_path(project_id)).get("stage_states", {})
        latest_live_job_by_stage: dict[PipelineStage, JobRecord] = {}

        for job in jobs:
            if job.status not in {JobStatus.QUEUED, JobStatus.RUNNING}:
                continue
            if not self._job_uses_direct_runner(job):
                continue
            if not self._job_process_is_alive(job):
                continue
            previous = latest_live_job_by_stage.get(job.stage)
            if previous is None or job.created_at > previous.created_at:
                latest_live_job_by_stage[job.stage] = job

        reconciled: list[JobRecord] = []
        for job in jobs:
            if job.status not in {JobStatus.QUEUED, JobStatus.RUNNING}:
                reconciled.append(job)
                continue
            if not self._job_uses_direct_runner(job):
                reconciled.append(job)
                continue
            if self._job_process_is_alive(job):
                reconciled.append(job)
                continue

            updated = True
            stale_message = "Recovered stale direct-run job after runner exit"
            repaired = job.model_copy(
                update={
                    "status": JobStatus.CANCELLED,
                    "finished_at": self._now(),
                    "message": stale_message,
                    "error": "Runner process is no longer alive",
                }
            )
            write_json(self._job_path(project_id, job.id), repaired.model_dump(mode="json"))
            reconciled.append(repaired)

            live_job = latest_live_job_by_stage.get(job.stage)
            stage_state = current_stage_states.get(job.stage.value, {})
            current_status = str(stage_state.get("status") or "")
            if live_job is None and current_status == StageStatus.RUNNING.value:
                self.update_stage_state(
                    project_id,
                    job.stage,
                    StageStatus.READY,
                    progress=float(stage_state.get("progress") or 0.0),
                    message="Recovered from stale runner. Ready to resume.",
                )

        if updated:
            logger.warning("Recovered stale direct-run jobs for %s", project_id)
        return reconciled

    def _job_uses_direct_runner(self, job: JobRecord) -> bool:
        payload = job.payload or {}
        return bool(payload.get("direct_runner"))

    def _job_process_is_alive(self, job: JobRecord) -> bool:
        payload = job.payload or {}
        runner_pid = payload.get("runner_pid")
        if runner_pid is None:
            return True
        try:
            pid = int(runner_pid)
        except (TypeError, ValueError):
            return True
        if pid <= 0:
            return True
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        except OSError:
            return False
        return True

    def list_audio_files(self, project_id: str) -> list[AudioFile]:
        audio_dir = self._project_dir(project_id) / "audio"
        files: list[AudioFile] = []
        for audio_path in sorted(audio_dir.glob("*.wav")):
            manifest_path = audio_dir / "manifest.json"
            manifest = read_json(manifest_path, default={})
            panel_id = manifest.get(audio_path.name, {}).get("panel_id", audio_path.stem)
            duration = float(manifest.get(audio_path.name, {}).get("duration_seconds", 0))
            files.append(
                AudioFile(
                    panel_id=panel_id,
                    path=str(audio_path),
                    url=self._relative_media_url(audio_path),
                    duration_seconds=duration,
                )
            )
        return files

    def list_videos(self, project_id: str) -> list[VideoFile]:
        video_dir = self._project_dir(project_id) / "video"
        videos: list[VideoFile] = []
        manifest = read_json(video_dir / "manifest.json", default={})
        if isinstance(manifest, dict) and manifest:
            for name, info in manifest.items():
                video_path = video_dir / name
                if not video_path.exists():
                    continue
                videos.append(
                    VideoFile(
                        name=video_path.name,
                        path=str(video_path),
                        url=self._relative_media_url(video_path),
                        width=int(info.get("width", 0)),
                        height=int(info.get("height", 0)),
                        output_format=info.get("output_format", video_path.suffix.lstrip(".")),
                        created_at=datetime.fromisoformat(info.get("created_at", self._now().isoformat())),
                        duration_seconds=info.get("duration_seconds"),
                    )
                )
            return sorted(videos, key=lambda item: item.created_at)

        for video_path in sorted(video_dir.glob("*.mp4")) + sorted(video_dir.glob("*.mov")):
            created_at = datetime.fromtimestamp(video_path.stat().st_mtime)
            videos.append(
                VideoFile(
                    name=video_path.name,
                    path=str(video_path),
                    url=self._relative_media_url(video_path),
                    width=0,
                    height=0,
                    output_format=video_path.suffix.lstrip("."),
                    created_at=created_at,
                    duration_seconds=None,
                )
            )
        return sorted(videos, key=lambda item: item.created_at)

    def write_thumbnail(self, project_id: str, image_bytes: bytes) -> Path:
        path = self._project_dir(project_id) / "thumbnails" / "cover.jpg"
        path.write_bytes(image_bytes)
        return path

    def write_video_thumbnail(self, project_id: str, image_bytes: bytes) -> Path:
        """User upload path. Writes the image and drops a marker that
        tells the bundle stage to leave this file alone (so an auto-pull
        from the chosen Shorts cover doesn't overwrite the user's pick).
        """
        thumbs_dir = self._project_dir(project_id) / "thumbnails"
        thumbs_dir.mkdir(parents=True, exist_ok=True)
        path = thumbs_dir / "video_intro.jpg"
        path.write_bytes(image_bytes)
        (thumbs_dir / ".video_intro_user_upload").write_text("user")
        self.update_project_metadata(project_id)
        return path

    def delete_video_thumbnail(self, project_id: str) -> None:
        """Removes both the user upload AND the marker, so the next
        bundle stage will auto-pull from the chosen Shorts cover again.
        """
        thumbs_dir = self._project_dir(project_id) / "thumbnails"
        (thumbs_dir / "video_intro.jpg").unlink(missing_ok=True)
        (thumbs_dir / ".video_intro_user_upload").unlink(missing_ok=True)
        self.update_project_metadata(project_id)

    def sync_video_thumbnail_from_short_cover(
        self, project_id: str, short_thumbnail_abs_path: Path,
    ) -> Path | None:
        """Copy the chosen Shorts cover into `video_intro.jpg` so it
        becomes the video lead-in.

        If the user has uploaded a custom video thumbnail (marker file
        exists), this is a no-op - we never clobber the user's choice.
        Returns the destination path on success, None if skipped.
        """
        import shutil
        from PIL import Image as _Image

        thumbs_dir = self._project_dir(project_id) / "thumbnails"
        thumbs_dir.mkdir(parents=True, exist_ok=True)
        marker = thumbs_dir / ".video_intro_user_upload"
        if marker.exists():
            return None
        if not short_thumbnail_abs_path.exists():
            return None
        dest = thumbs_dir / "video_intro.jpg"
        try:
            # Source is a PNG; re-encode as JPEG to match the on-disk
            # convention (and shrink the file).
            with _Image.open(short_thumbnail_abs_path) as src:
                src.convert("RGB").save(dest, format="JPEG", quality=92)
        except Exception:
            # Fall back to a straight copy if PIL chokes.
            shutil.copy2(short_thumbnail_abs_path, dest)
        return dest

    def list_music_tracks(self) -> list[dict[str, object]]:
        tracks: list[dict[str, object]] = []
        builtin_manifest = read_json(self.builtin_music_root / "manifest.json", default=[])
        uploaded_manifest = read_json(self.uploaded_music_root / "manifest.json", default=[])

        for item in builtin_manifest:
            asset_path = self.builtin_music_root / item["file"]
            tracks.append(self._music_track_payload(item, asset_path, source="builtin"))

        for item in uploaded_manifest:
            asset_path = self.uploaded_music_root / item["file"]
            tracks.append(self._music_track_payload(item, asset_path, source="uploaded"))

        return tracks

    def resolve_music_track(self, track_name: str) -> dict[str, object] | None:
        return next(
            (track for track in self.list_music_tracks() if track["name"] == track_name and track["available"]),
            None,
        )

    def add_uploaded_music_track(
        self,
        filename: str,
        file_bytes: bytes,
        track_name: str | None = None,
        mood: str | None = None,
    ) -> dict[str, object]:
        if Path(filename).suffix.lower() != ".mp3":
            raise ValueError("Only MP3 files are supported for uploaded music.")

        base_name = (track_name or Path(filename).stem).strip() or "Uploaded Track"
        unique_name = self._unique_music_name(base_name)
        stem = slugify(Path(filename).stem or unique_name)
        target = self.uploaded_music_root / f"{stem}.mp3"
        counter = 2
        while target.exists():
            target = self.uploaded_music_root / f"{stem}-{counter}.mp3"
            counter += 1

        target.write_bytes(file_bytes)
        duration_seconds = self._probe_media_duration(target)

        manifest_path = self.uploaded_music_root / "manifest.json"
        manifest = read_json(manifest_path, default=[])
        entry = {
            "name": unique_name,
            "file": target.name,
            "mood": (mood or "custom").strip() or "custom",
            "duration_seconds": duration_seconds,
        }
        manifest.append(entry)
        write_json(manifest_path, manifest)
        return self._music_track_payload(entry, target, source="uploaded")

    def _music_track_payload(self, item: dict[str, object], asset_path: Path, source: str) -> dict[str, object]:
        if source == "uploaded":
            url = self._relative_media_url(asset_path) if asset_path.exists() else None
        else:
            url = f"/assets/music/{item['file']}" if asset_path.exists() else None

        return {
            **item,
            "available": asset_path.exists(),
            "url": url,
            "path": str(asset_path),
            "source": source,
        }

    def _unique_music_name(self, base_name: str) -> str:
        existing_names = {str(track["name"]) for track in self.list_music_tracks()}
        candidate = base_name
        if candidate not in existing_names:
            return candidate

        counter = 1
        while True:
            suffix = "Uploaded" if counter == 1 else f"Uploaded {counter}"
            candidate = f"{base_name} ({suffix})"
            if candidate not in existing_names:
                return candidate
            counter += 1

    def _probe_media_duration(self, path: Path) -> float | None:
        command = [
            self.settings.ffprobe_binary,
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ]
        result = subprocess.run(command, capture_output=True, check=False, text=True)
        if result.returncode != 0:
            return None
        try:
            return round(float(result.stdout.strip()), 2)
        except ValueError:
            return None

    def _copy_directory_contents(self, source_dir: Path, target_dir: Path) -> None:
        if not source_dir.exists():
            return
        ensure_dir(target_dir)
        for item in source_dir.iterdir():
            destination = target_dir / item.name
            if item.is_dir():
                shutil.copytree(item, destination, dirs_exist_ok=True)
            else:
                shutil.copy2(item, destination)

    def _copy_selected_videos(self, source_dir: Path, target_dir: Path, video_name: str | None, copy_all_videos: bool) -> None:
        ensure_dir(target_dir)
        manifest = read_json(source_dir / "manifest.json", default={})
        selected_names: set[str]
        if copy_all_videos or not video_name:
            selected_names = {path.name for path in source_dir.glob("*.mp4")} | {path.name for path in source_dir.glob("*.mov")}
        else:
            selected_names = {video_name}

        copied_manifest = {}
        for name in selected_names:
            source_path = source_dir / name
            if not source_path.exists():
                continue
            shutil.copy2(source_path, target_dir / name)
            if name in manifest:
                copied_manifest[name] = manifest[name]

        if copied_manifest:
            write_json(target_dir / "manifest.json", copied_manifest)
