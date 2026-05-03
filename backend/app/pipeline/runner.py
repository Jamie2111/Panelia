from __future__ import annotations

from app.pipeline.context import JobCancelledError, JobPausedError, PipelineContext
from app.pipeline.stages import STAGE_HANDLERS
from app.schemas.project import JobStatus, PipelineStage
from app.services.project_store import ProjectStore
from app.services.queue_service import QueueMessage, QueueService


def run_job(message: QueueMessage, store: ProjectStore | None = None, queue: QueueService | None = None) -> None:
    store = store or ProjectStore()
    queue = queue or QueueService()
    if not store.project_exists(message.project_id):
        return
    job = store.get_job(message.project_id, message.job_id)
    if job.status in {JobStatus.CANCELLED, JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.RUNNING, JobStatus.PAUSED}:
        return
    stage = PipelineStage(message.stage)
    context = PipelineContext(store=store, queue=queue, project_id=message.project_id, job_id=message.job_id, stage=stage)

    try:
        handler = STAGE_HANDLERS[stage]
        handler(context)
    except JobPausedError:
        context.pause("Paused by user")
    except JobCancelledError:
        context.cancel("Cancelled by user")
    except Exception as exc:  # pragma: no cover - service failures depend on runtime integrations
        context.fail(str(exc))
        raise
