from __future__ import annotations

from app.pipeline.orchestration import queue_stage_once
from app.schemas.project import PipelineStage, StageStatus
from app.services.character_review_service import CharacterReviewService
from app.services.project_store import ProjectStore
from app.services.queue_service import QueueService

def project_auto_run_enabled(project: object) -> bool:
    pipeline_config = getattr(project, "pipeline_config", None)
    return bool(getattr(pipeline_config, "auto_run_end_to_end", False))


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
    review_service = CharacterReviewService()
    review_state = review_service.load_review_state(store._project_dir(project_id))

    panel_review_state = stage_states.get(PipelineStage.PANEL_REVIEW)
    if panel_review_state and panel_review_state.status in {StageStatus.READY, StageStatus.NEEDS_REVIEW} and project.panels:
        panel_quality = store.load_panel_quality_report(project_id)
        if bool(panel_quality.get("should_block_script")):
            store.update_stage_state(
                project_id,
                PipelineStage.PANEL_REVIEW,
                StageStatus.NEEDS_REVIEW,
                progress=100,
                message="Auto-run paused because the detected panels still need review before scripting.",
            )
            return False

        store.update_stage_state(
            project_id,
            PipelineStage.PANEL_REVIEW,
            StageStatus.COMPLETED,
            progress=100,
            message="Auto-run accepted the current panel selection. You can still edit it anytime.",
        )
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
                message="Starting script generation automatically",
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
            message="Starting character review automatically",
        )
        queue_stage_once(
            store,
            queue,
            project_id,
            PipelineStage.CHARACTER_REVIEW,
            f"Queued automatically after {source}",
        )
        return True

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

    if character_review_state and character_review_state.status == StageStatus.NEEDS_REVIEW and review_state is not None:
        store.update_stage_state(
            project_id,
            PipelineStage.CHARACTER_REVIEW,
            StageStatus.COMPLETED,
            progress=100,
            message="Character suggestions were prepared. Auto-run is continuing to the script.",
        )
        store.update_stage_state(project_id, PipelineStage.SCRIPT_GENERATION, StageStatus.READY, progress=0, message="Starting script generation automatically")
        queue_stage_once(
            store,
            queue,
            project_id,
            PipelineStage.SCRIPT_GENERATION,
            f"Queued automatically after {source}",
        )
        return True

    for stage in (
        PipelineStage.INGESTION,
        PipelineStage.PANEL_DETECTION,
        PipelineStage.SCRIPT_GENERATION,
        PipelineStage.NARRATION_GENERATION,
        PipelineStage.VIDEO_RENDERING,
    ):
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

    return False
