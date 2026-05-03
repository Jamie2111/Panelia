from __future__ import annotations

from dataclasses import dataclass

from app.schemas.project import JobStatus, PipelineStage, StageStatus
from app.services.project_store import ProjectStore
from app.services.queue_service import QueueService


class JobCancelledError(RuntimeError):
    pass


class JobPausedError(RuntimeError):
    pass


@dataclass
class PipelineContext:
    store: ProjectStore
    queue: QueueService
    project_id: str
    job_id: str
    stage: PipelineStage

    def start(self, message: str) -> None:
        self.ensure_not_cancelled()
        self.store.update_job(
            self.project_id,
            self.job_id,
            status=JobStatus.RUNNING.value,
            started_at=self.store._now().isoformat(),
            message=message,
        )
        self.store.update_stage_state(self.project_id, self.stage, StageStatus.RUNNING, progress=0, message=message)

    def progress(self, progress: float, message: str) -> None:
        self.ensure_not_cancelled()
        self.store.update_job(
            self.project_id,
            self.job_id,
            status=JobStatus.RUNNING.value,
            progress=progress,
            message=message,
        )
        self.store.update_stage_state(self.project_id, self.stage, StageStatus.RUNNING, progress=progress, message=message)

    def complete(self, message: str) -> None:
        self.store.update_job(
            self.project_id,
            self.job_id,
            status=JobStatus.COMPLETED.value,
            progress=100,
            finished_at=self.store._now().isoformat(),
            message=message,
        )
        self.store.update_stage_state(self.project_id, self.stage, StageStatus.COMPLETED, progress=100, message=message)
        self.queue.clear_cancel(self.job_id)

    def fail(self, message: str) -> None:
        self.store.update_job(
            self.project_id,
            self.job_id,
            status=JobStatus.FAILED.value,
            finished_at=self.store._now().isoformat(),
            error=message,
            message=message,
        )
        self.store.update_stage_state(self.project_id, self.stage, StageStatus.FAILED, message=message)
        self.queue.clear_cancel(self.job_id)

    def cancel(self, message: str) -> None:
        self.store.update_job(
            self.project_id,
            self.job_id,
            status=JobStatus.CANCELLED.value,
            finished_at=self.store._now().isoformat(),
            message=message,
        )
        self.store.update_stage_state(self.project_id, self.stage, StageStatus.CANCELLED, message=message)
        self.queue.clear_cancel(self.job_id)

    def pause(self, message: str) -> None:
        """Suspend the job so it can be resumed later. Progress is preserved."""
        self.store.update_job(
            self.project_id,
            self.job_id,
            status=JobStatus.PAUSED.value,
            finished_at=self.store._now().isoformat(),
            message=message,
        )
        self.store.update_stage_state(self.project_id, self.stage, StageStatus.PAUSED, message=message)
        self.queue.clear_pause(self.job_id)

    def ensure_not_cancelled(self) -> None:
        if self.queue.is_cancel_requested(self.job_id):
            raise JobCancelledError(f"Job {self.job_id} cancelled")
        if self.queue.is_pause_requested(self.job_id):
            raise JobPausedError(f"Job {self.job_id} paused")
