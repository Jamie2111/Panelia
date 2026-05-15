"""
Auto-run continuation logic.

Picks the next stage to queue after each completed stage, based on the
project's configured pipeline version.

For vision-mode projects (the default for new projects), the path is:
    ingestion → panel_detection → panel_review → script_generation
      → narration_generation → video_rendering → youtube_bundle

The legacy stages (`character_review`, `character_portrait`,
`panel_vision_extraction`, `panel_vision_quality`) are auto-completed in
vision mode so the UI can collapse them and the pipeline never blocks on
them. They still run normally for legacy projects.
"""

from __future__ import annotations

from app.pipeline.orchestration import queue_stage_once
from app.schemas.project import PipelineStage, StageStatus
from app.services.character_review_service import CharacterReviewService
from app.services.project_store import ProjectStore
from app.services.queue_service import QueueService


# Legacy stages that don't need to run in vision mode.
VISION_SKIPPED_STAGES: frozenset[PipelineStage] = frozenset({
    PipelineStage.CHARACTER_REVIEW,
    PipelineStage.CHARACTER_PORTRAIT,
    PipelineStage.PANEL_VISION_EXTRACTION,
    PipelineStage.PANEL_VISION_QUALITY,
})


def project_auto_run_enabled(project: object) -> bool:
    pipeline_config = getattr(project, "pipeline_config", None)
    return bool(getattr(pipeline_config, "auto_run_end_to_end", False))


def project_uses_vision_pipeline(project: object) -> bool:
    pipeline_config = getattr(project, "pipeline_config", None)
    version = str(getattr(pipeline_config, "script_pipeline_version", "") or "").casefold()
    return version == "vision"


def _mark_skipped_legacy_stages(store: ProjectStore, project_id: str) -> None:
    """Mark the four legacy stages as completed for vision-mode projects so
    the UI shows the pipeline accurately and auto-run never stalls on them."""
    for stage in VISION_SKIPPED_STAGES:
        store.update_stage_state(
            project_id,
            stage,
            StageStatus.COMPLETED,
            progress=100,
            message="Skipped - not used by the vision pipeline.",
        )


def continue_auto_run_pipeline(
    store: ProjectStore,
    queue: QueueService,
    project_id: str,
    *,
    source: str,
) -> bool:
    project = store.get_project(project_id)
    if not project_auto_run_enabled(project):
        return False
    if project.active_jobs:
        return False

    stage_states = project.stage_states
    using_vision = project_uses_vision_pipeline(project)

    # In vision mode, eagerly mark the four legacy stages as completed so
    # the UI collapses them and we don't even consider running them.
    if using_vision:
        _mark_skipped_legacy_stages(store, project_id)

    review_service = CharacterReviewService()
    review_state = review_service.load_review_state(store._project_dir(project_id))

    # ── Panel review handling ────────────────────────────────────────────
    panel_review_state = stage_states.get(PipelineStage.PANEL_REVIEW)
    if (
        panel_review_state
        and panel_review_state.status in {StageStatus.READY, StageStatus.NEEDS_REVIEW}
        and project.panels
    ):
        panel_quality = store.load_panel_quality_report(project_id)
        if bool(panel_quality.get("should_block_script")):
            store.update_stage_state(
                project_id,
                PipelineStage.PANEL_REVIEW,
                StageStatus.NEEDS_REVIEW,
                progress=100,
                message="Auto-run paused - detected panels need a quick human review before scripting.",
            )
            return False

        store.update_stage_state(
            project_id,
            PipelineStage.PANEL_REVIEW,
            StageStatus.COMPLETED,
            progress=100,
            message="Auto-run accepted the current panel selection. You can still edit it anytime.",
        )

        # In vision mode, skip character review entirely and go straight
        # to script generation.
        if using_vision:
            store.update_stage_state(
                project_id,
                PipelineStage.SCRIPT_GENERATION,
                StageStatus.READY,
                progress=0,
                message="Vision narrator starting automatically.",
            )
            queue_stage_once(
                store,
                queue,
                project_id,
                PipelineStage.SCRIPT_GENERATION,
                f"Queued automatically after {source}",
            )
            return True

        # Legacy path - keep prior behaviour.
        if review_state is not None:
            store.update_stage_state(
                project_id,
                PipelineStage.CHARACTER_REVIEW,
                StageStatus.COMPLETED,
                progress=100,
                message="Character suggestions are already prepared. Auto-run is continuing to the script.",
            )
            store.update_stage_state(
                project_id,
                PipelineStage.SCRIPT_GENERATION,
                StageStatus.READY,
                progress=0,
                message="Starting script generation automatically.",
            )
            queue_stage_once(
                store,
                queue,
                project_id,
                PipelineStage.SCRIPT_GENERATION,
                f"Queued automatically after {source}",
            )
            return True

        store.update_stage_state(
            project_id,
            PipelineStage.CHARACTER_REVIEW,
            StageStatus.READY,
            progress=0,
            message="Starting character review automatically.",
        )
        queue_stage_once(
            store,
            queue,
            project_id,
            PipelineStage.CHARACTER_REVIEW,
            f"Queued automatically after {source}",
        )
        return True

    # ── Character review (legacy only) ──────────────────────────────────
    if not using_vision:
        character_review_state = stage_states.get(PipelineStage.CHARACTER_REVIEW)
        if character_review_state and character_review_state.status == StageStatus.READY:
            queue_stage_once(
                store,
                queue,
                project_id,
                PipelineStage.CHARACTER_REVIEW,
                f"Queued automatically after {source}",
            )
            return True
        if (
            character_review_state
            and character_review_state.status == StageStatus.NEEDS_REVIEW
            and review_state is not None
        ):
            store.update_stage_state(
                project_id,
                PipelineStage.CHARACTER_REVIEW,
                StageStatus.COMPLETED,
                progress=100,
                message="Character suggestions were prepared. Auto-run is continuing to the script.",
            )
            store.update_stage_state(
                project_id,
                PipelineStage.SCRIPT_GENERATION,
                StageStatus.READY,
                progress=0,
                message="Starting script generation automatically.",
            )
            queue_stage_once(
                store,
                queue,
                project_id,
                PipelineStage.SCRIPT_GENERATION,
                f"Queued automatically after {source}",
            )
            return True

    # ── Linear progression for the remaining stages ──────────────────────
    # Order matters: any stage in READY state earlier in the list is queued
    # first. We include YOUTUBE_BUNDLE so a successful video render flows
    # straight into bundle generation.
    forward_stages = [
        PipelineStage.INGESTION,
        PipelineStage.PANEL_DETECTION,
        PipelineStage.SCRIPT_GENERATION,
        PipelineStage.NARRATION_GENERATION,
        PipelineStage.VIDEO_RENDERING,
        PipelineStage.YOUTUBE_BUNDLE,
    ]
    for stage in forward_stages:
        state = stage_states.get(stage)
        if not state or state.status != StageStatus.READY:
            continue
        queue_stage_once(
            store,
            queue,
            project_id,
            stage,
            f"Queued automatically after {source}",
        )
        return True

    # If video_rendering just completed, make sure YOUTUBE_BUNDLE moves to
    # READY (it may have been left at PENDING for legacy projects).
    video_state = stage_states.get(PipelineStage.VIDEO_RENDERING)
    bundle_state = stage_states.get(PipelineStage.YOUTUBE_BUNDLE)
    if (
        video_state
        and video_state.status == StageStatus.COMPLETED
        and (bundle_state is None or bundle_state.status == StageStatus.PENDING)
    ):
        store.update_stage_state(
            project_id,
            PipelineStage.YOUTUBE_BUNDLE,
            StageStatus.READY,
            progress=0,
            message="Preparing your publish bundle.",
        )
        queue_stage_once(
            store,
            queue,
            project_id,
            PipelineStage.YOUTUBE_BUNDLE,
            f"Queued automatically after {source}",
        )
        return True

    return False
