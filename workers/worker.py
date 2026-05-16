from __future__ import annotations

import atexit
import fcntl
import logging
import os
import shutil
import time
from datetime import datetime, timezone
from multiprocessing import Process
from pathlib import Path
from threading import Thread
import warnings

from requests import RequestsDependencyWarning

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("PADDLE_NUM_THREADS", "1")
os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/panelia-mpl")

warnings.filterwarnings("ignore", category=RequestsDependencyWarning)
warnings.filterwarnings("ignore", message=".*No ccache found.*")
warnings.filterwarnings("ignore", message="`lang` and `ocr_version` will be ignored when model names or model directories are not `None`\\.")
warnings.filterwarnings("ignore", message="`torch.utils._pytree._register_pytree_node` is deprecated.*", category=FutureWarning)

from app.pipeline.runner import run_job
from app.services.dialogue_pipeline import DialogueExtractionPipeline
from app.services.panel_detection_service import MagiPanelDetectionService
from app.services.project_store import ProjectStore
from app.services.queue_service import QueueService
from app.schemas.project import JobStatus, StageStatus

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("panelia-worker")
_WORKER_LOCK_HANDLE = None
_WORKER_LOCK_PATH = Path("/tmp/panelia-worker.lock")
_STALE_JOB_RECOVERY_INTERVAL_SECONDS = 60

# Minimum free disk space required before we start any job. A panel-render
# or video-render can write 5-10 GB of intermediates (final_silent.mp4,
# final_silent_with_intro.mp4, final.mp4, final_music.mp4). Starting one
# with too little headroom means ffmpeg crashes mid-encode on ENOSPC and -
# worse - the worker can't always write the failure status back to disk
# (because the disk that just rejected the encode rejects the status
# update too), so the job sits at "running" forever and the recovery
# sweeper keeps re-spawning it. Cleaner to refuse to start and tell the
# user. 5 GB matches the worst-case ~4 GB silent video + a margin.
_MIN_FREE_DISK_GB_FOR_JOB = 5.0
# In-process memo of job IDs we've already refused this run, so the
# recovery sweeper doesn't keep handing them back to us in a tight loop
# if the user hasn't freed disk yet.
_REFUSED_JOB_IDS: set[str] = set()


def _check_disk_space_or_fail(
    message,
    store: ProjectStore,
    data_dir: Path,
    min_free_gb: float = _MIN_FREE_DISK_GB_FOR_JOB,
) -> bool:
    """Return True if there's enough headroom to start; otherwise mark the
    job FAILED with an actionable message and return False.

    This guard prevents the crash-loop pattern where a render starts on a
    near-full disk, ffmpeg hits ENOSPC at the merge/prepend step (which
    needs 5-8 GB scratch), and the worker can't even flush the failure
    status because the disk that broke ffmpeg also breaks the JSON write.
    """
    try:
        usage = shutil.disk_usage(data_dir)
        free_gb = usage.free / (1024 ** 3)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Disk-space check failed for %s: %s", data_dir, exc)
        return True  # Don't block on a bad probe

    if free_gb >= min_free_gb:
        return True

    error_msg = (
        f"Not enough disk space to start this job. "
        f"Only {free_gb:.1f} GB free at {data_dir} (need >= {min_free_gb:.0f} GB). "
        f"Free up space (delete old project outputs or clear data/_video_cache) and retry."
    )
    logger.error("[disk-guard] %s job_id=%s", error_msg, message.job_id)
    _REFUSED_JOB_IDS.add(message.job_id)

    try:
        job = store.get_job(message.project_id, message.job_id)
    except Exception:  # noqa: BLE001
        logger.exception("[disk-guard] Could not load job to mark failed")
        return False

    try:
        store.update_job(
            message.project_id,
            message.job_id,
            status=JobStatus.FAILED.value,
            finished_at=store._now().isoformat(),
            error=error_msg,
            message=error_msg,
        )
        store.update_stage_state(
            message.project_id,
            job.stage,
            StageStatus.FAILED,
            progress=job.progress,
            message=error_msg,
        )
    except Exception:  # noqa: BLE001
        # If the status write itself fails (disk is THAT full), we at
        # least have the in-memory _REFUSED_JOB_IDS set to stop the loop
        # in this worker process. The job will appear "running" in the
        # store until the user frees disk and restarts the worker, but
        # we won't burn CPU on retries.
        logger.exception(
            "[disk-guard] Could not persist failure status for %s; "
            "in-memory refusal set will stop the loop until worker restarts",
            message.job_id,
        )
    return False


def _run_job_child(project_id: str, job_id: str, stage: str) -> None:
    child_store = ProjectStore()
    child_queue = QueueService()
    from app.services.queue_service import QueueMessage

    run_job(
        QueueMessage(project_id=project_id, job_id=job_id, stage=stage),
        store=child_store,
        queue=child_queue,
    )


def _run_job_isolated(message, store: ProjectStore) -> int | None:
    process = Process(
        target=_run_job_child,
        args=(message.project_id, message.job_id, message.stage),
        name=f"panelia-job-{message.job_id[:8]}",
    )
    process.start()
    process.join()
    exit_code = process.exitcode
    if exit_code in (0, None):
        return exit_code

    try:
        job = store.get_job(message.project_id, message.job_id)
        if job.status in {JobStatus.QUEUED, JobStatus.RUNNING}:
            error_message = f"Worker child exited unexpectedly with code {exit_code}"
            store.update_job(
                message.project_id,
                message.job_id,
                status=JobStatus.FAILED.value,
                finished_at=store._now().isoformat(),
                error=error_message,
                message=error_message,
            )
            store.update_stage_state(
                message.project_id,
                job.stage,
                StageStatus.FAILED,
                progress=job.progress,
                message=error_message,
            )
    except Exception:
        logger.exception("Failed to mark crashed job %s as failed", message.job_id)
    return exit_code


def _prewarm_ocr_models() -> None:
    try:
        pipeline = DialogueExtractionPipeline()
        if pipeline._has_paddleocr():
            pipeline._get_paddle_ocr("en")
            logger.info("PaddleOCR cache warmed in background")
    except Exception:
        logger.exception("Failed to warm OCR models")


def _prewarm_magi_model() -> None:
    try:
        detector = MagiPanelDetectionService()
        detector._load_model()
        logger.info("MAGI cache warmed in background")
    except Exception:
        logger.exception("Failed to warm MAGI model")


def _prewarm_models_gently() -> None:
    try:
        time.sleep(2)
        _prewarm_ocr_models()
        time.sleep(6)
        _prewarm_magi_model()
    except Exception:
        logger.exception("Failed during staged model warmup")


def _recover_interrupted_jobs(
    store: ProjectStore,
    queue: QueueService,
    *,
    force: bool = False,
) -> None:
    recovered = 0
    now = datetime.now(timezone.utc)
    for project in store.list_projects():
        for job in project.active_jobs:
            if job.status != JobStatus.RUNNING:
                continue
            if bool((job.payload or {}).get("direct_runner")):
                continue
            stage_state = project.stage_states.get(job.stage)
            heartbeat_at = None
            if stage_state and stage_state.updated_at:
                heartbeat_at = stage_state.updated_at
            elif job.started_at:
                heartbeat_at = job.started_at
            if not force and heartbeat_at is not None:
                if heartbeat_at.tzinfo is None:
                    heartbeat_at = heartbeat_at.replace(tzinfo=timezone.utc)
                if (now - heartbeat_at).total_seconds() < 90:
                    continue
            store.update_job(
                project.id,
                job.id,
                status=JobStatus.QUEUED.value,
                started_at=None,
                message="Recovered after worker restart",
            )
            store.update_stage_state(
                project.id,
                job.stage,
                StageStatus.READY,
                progress=job.progress,
                message="Recovered after worker restart",
            )
            queue.enqueue(project.id, job.id, job.stage.value)
            recovered += 1
    if recovered:
        if force:
            logger.info("Recovered %s interrupted job(s) during worker startup", recovered)
        else:
            logger.info("Recovered %s stale interrupted job(s)", recovered)


def _acquire_worker_lock() -> None:
    global _WORKER_LOCK_HANDLE
    lock_path = _WORKER_LOCK_PATH
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    handle = lock_path.open("a+")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        handle.close()
        raise RuntimeError("Another Panelia worker is already running. Stop the older worker before starting a new one.")
    handle.seek(0)
    handle.truncate()
    handle.write(str(os.getpid()))
    handle.flush()
    _WORKER_LOCK_HANDLE = handle

    def _release() -> None:
        global _WORKER_LOCK_HANDLE
        if _WORKER_LOCK_HANDLE is None:
            return
        try:
            fcntl.flock(_WORKER_LOCK_HANDLE.fileno(), fcntl.LOCK_UN)
        except Exception:
            pass
        try:
            _WORKER_LOCK_HANDLE.close()
        except Exception:
            pass
        try:
            _WORKER_LOCK_PATH.unlink(missing_ok=True)
        except Exception:
            pass
        _WORKER_LOCK_HANDLE = None

    atexit.register(_release)


def main() -> None:
    try:
        _acquire_worker_lock()
    except RuntimeError as exc:
        logger.info("%s", exc)
        return
    store = ProjectStore()
    queue = QueueService()
    try:
        import cv2

        cv2.setNumThreads(1)
    except Exception:
        pass

    try:
        import torch

        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
    except Exception:
        pass

    logger.info("Worker started and listening for jobs")
    _recover_interrupted_jobs(store, queue, force=True)
    last_recovery_check = time.monotonic()
    if str(os.environ.get("PANELIA_WORKER_PREWARM") or "").strip().casefold() in {"1", "true", "yes"}:
        Thread(target=_prewarm_models_gently, daemon=True).start()

    while True:
        message = queue.reserve(timeout_seconds=5)
        if not message:
            now = time.monotonic()
            if now - last_recovery_check >= _STALE_JOB_RECOVERY_INTERVAL_SECONDS:
                _recover_interrupted_jobs(store, queue)
                last_recovery_check = now
            time.sleep(1)
            continue

        # Skip jobs we already refused this worker lifetime (disk-full
        # guard). Without this, the stale-job recovery sweeper keeps
        # handing them back and we'd burn cycles re-checking and re-failing.
        if message.job_id in _REFUSED_JOB_IDS:
            logger.warning(
                "Skipping previously-refused job %s (free up disk and restart worker to retry)",
                message.job_id,
            )
            continue

        # Disk-space gate. Cheaper to fail fast with a clear error than
        # to crash mid-encode and lose all the render-cache work we built.
        from app.core.config import get_settings  # local import to avoid startup cost
        settings_for_guard = get_settings()
        if not _check_disk_space_or_fail(message, store, settings_for_guard.data_dir):
            continue

        logger.info("Processing job %s for project %s stage %s", message.job_id, message.project_id, message.stage)
        started_at = time.perf_counter()
        exit_code = _run_job_isolated(message, store)
        if exit_code == 0:
            logger.info(
                "Completed job %s for project %s stage %s in %.2fs",
                message.job_id,
                message.project_id,
                message.stage,
                time.perf_counter() - started_at,
            )
        elif exit_code is None:
            logger.warning(
                "Job %s child exit was unavailable after %.2fs",
                message.job_id,
                time.perf_counter() - started_at,
            )
        else:
            logger.error(
                "Job %s crashed with child exit code %s after %.2fs",
                message.job_id,
                exit_code,
                time.perf_counter() - started_at,
            )


if __name__ == "__main__":
    main()
