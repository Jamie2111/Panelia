from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field

from app.schemas.project import JobStatus, PipelineStage, StageStatus
from app.services.project_store import ProjectStore
from app.services.queue_service import QueueService


class JobCancelledError(RuntimeError):
    pass


class JobPausedError(RuntimeError):
    pass


# Throttle config for the high-frequency `progress()` method. Pipelines
# like Edge/Azure TTS narration call progress thousands of times per
# minute; each unthrottled call did 2 reads + 2 writes of a 3 MB
# metadata.json + a job.json read+write. With 16 concurrent narration
# workers calling at once, the file locks on metadata.json starved the
# worker threads to a halt (the "metadata busy-loop"). Now we only
# flush to disk when:
#   - the integer progress changes, OR
#   - the message changes meaningfully, OR
#   - at least PROGRESS_MIN_INTERVAL_SECONDS have passed since the
#     last flush.
# Cancel-check still happens on every call (it just hits an in-memory
# queue lookup, no disk).
PROGRESS_MIN_INTERVAL_SECONDS = 0.5


@dataclass
class PipelineContext:
    store: ProjectStore
    queue: QueueService
    project_id: str
    job_id: str
    stage: PipelineStage

    # In-memory throttling state (one set per PipelineContext instance).
    # The lock keeps multi-threaded workers (TTS sentence cache, video
    # render parallelism) from racing on the flush check.
    _progress_lock: threading.Lock = field(default_factory=threading.Lock, repr=False, compare=False)
    _last_flushed_progress: int = field(default=-1, repr=False, compare=False)
    _last_flushed_message: str = field(default="", repr=False, compare=False)
    _last_flush_time: float = field(default=0.0, repr=False, compare=False)

    def start(self, message: str) -> None:
        self.ensure_not_cancelled()
        self.store.update_job(
            self.project_id,
            self.job_id,
            status=JobStatus.RUNNING.value,
            progress=0,
            started_at=self.store._now().isoformat(),
            message=message,
        )
        self.store.update_stage_state(self.project_id, self.stage, StageStatus.RUNNING, progress=0, message=message)

    def progress(self, progress: float, message: str) -> None:
        """Throttled progress update.

        Flushes to disk only when progress moves at least 1% OR the
        message materially changes OR `PROGRESS_MIN_INTERVAL_SECONDS`
        seconds have passed since the last flush. Always honors
        cancel/pause requests (in-memory check, no disk).
        """
        self.ensure_not_cancelled()
        new_progress_int = max(0, int(round(max(float(progress), 0.0))))
        now = time.monotonic()
        with self._progress_lock:
            elapsed_ok = (now - self._last_flush_time) >= PROGRESS_MIN_INTERVAL_SECONDS
            progress_changed = new_progress_int != self._last_flushed_progress
            message_changed = (message or "") != self._last_flushed_message
            if not (progress_changed or message_changed or elapsed_ok):
                return
            self._last_flushed_progress = new_progress_int
            self._last_flushed_message = message or ""
            self._last_flush_time = now

        # Use the throttled progress value; never go backwards relative
        # to what's on disk (a stale read could undercut a flush from
        # another thread).
        current_job = self.store.get_job(self.project_id, self.job_id)
        next_progress = max(float(new_progress_int), float(current_job.progress or 0.0))
        self.store.update_job(
            self.project_id,
            self.job_id,
            status=JobStatus.RUNNING.value,
            progress=next_progress,
            message=message,
        )
        self.store.update_stage_state(self.project_id, self.stage, StageStatus.RUNNING, progress=next_progress, message=message)

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
