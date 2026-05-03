from __future__ import annotations

from app.schemas.project import JobRecord, JobStatus, PipelineStage, StageStatus
from app.services.project_store import ProjectStore
from app.services.queue_service import QueueService


ACTIVE_JOB_STATUSES = {JobStatus.QUEUED, JobStatus.RUNNING}


def find_active_stage_job(store: ProjectStore, project_id: str, stage: PipelineStage) -> JobRecord | None:
    return next(
        (job for job in store.list_jobs(project_id) if job.stage == stage and job.status in ACTIVE_JOB_STATUSES),
        None,
    )


def queue_stage_once(
    store: ProjectStore,
    queue: QueueService,
    project_id: str,
    stage: PipelineStage,
    message: str,
    payload: dict[str, object] | None = None,
) -> JobRecord:
    existing_job = find_active_stage_job(store, project_id, stage)
    if existing_job is not None:
        store.update_stage_state(project_id, stage, StageStatus.READY, progress=existing_job.progress, message=message)
        return existing_job

    job = store.create_job(project_id, stage, payload=payload)
    store.update_stage_state(project_id, stage, StageStatus.READY, progress=0, message=message)
    queue.enqueue(project_id, job.id, stage.value)
    return job
