from __future__ import annotations

import argparse
import atexit
import fcntl
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.pipeline.context import PipelineContext
from app.pipeline.stages import (
    run_ingestion,
    run_narration_generation,
    run_panel_detection,
    run_script_generation,
    run_video_rendering,
)
from app.schemas.project import PipelineStage, StageStatus
from app.services.project_store import ProjectStore
from app.services.queue_service import QueueService
from app.services.video_verifier import VideoVerifier

_PROJECT_LOCK_HANDLES: dict[str, object] = {}


def _acquire_project_lock(project_id: str) -> None:
    if project_id in _PROJECT_LOCK_HANDLES:
        return
    lock_path = Path("/tmp") / f"panelia-chunk-{project_id}.lock"
    handle = lock_path.open("w")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        handle.close()
        raise RuntimeError(f"Another direct chunk runner is already active for {project_id}.")
    handle.write(str(os.getpid()))
    handle.flush()
    _PROJECT_LOCK_HANDLES[project_id] = handle


def _release_project_locks() -> None:
    while _PROJECT_LOCK_HANDLES:
        _, handle = _PROJECT_LOCK_HANDLES.popitem()
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        except Exception:
            pass
        try:
            handle.close()
        except Exception:
            pass


atexit.register(_release_project_locks)


def _cancel_stale_jobs(store: ProjectStore, project_id: str) -> None:
    queue = QueueService()
    for job in store.list_jobs(project_id):
        status_value = job.status.value if hasattr(job.status, "value") else str(job.status)
        if status_value not in {"queued", "running"}:
            continue
        queue.request_cancel(job.id)
        store.update_job(project_id, job.id, status="cancelled", message="Superseded by direct chunk rerun")


def _run_stage(
    store: ProjectStore,
    queue: QueueService,
    project_id: str,
    stage: PipelineStage,
) -> str:
    payload: dict[str, object] = {"direct_runner": True}
    if stage == PipelineStage.SCRIPT_GENERATION:
        payload["stop_after_stage"] = True
    job = store.create_job(project_id, stage, payload=payload)
    context = PipelineContext(
        store=store,
        queue=queue,
        project_id=project_id,
        job_id=job.id,
        stage=stage,
    )
    if stage == PipelineStage.NARRATION_GENERATION:
        run_narration_generation(context)
    elif stage == PipelineStage.INGESTION:
        run_ingestion(context)
    elif stage == PipelineStage.SCRIPT_GENERATION:
        run_script_generation(context)
    elif stage == PipelineStage.VIDEO_RENDERING:
        run_video_rendering(context)
    else:
        raise ValueError(f"Unsupported stage: {stage}")
    return job.id


def _auto_accept_detected_panels(store: ProjectStore, project_id: str) -> None:
    project = store.get_project(project_id)
    if not project.panels:
        raise RuntimeError(f"{project_id} finished panel detection without any saved panels.")
    panel_quality = store.load_panel_quality_report(project_id)
    if bool(panel_quality.get("should_block_script")):
        summary = str(panel_quality.get("summary") or "Panel quality checks blocked automatic continuation.")
        store.update_stage_state(
            project_id,
            PipelineStage.PANEL_REVIEW,
            StageStatus.NEEDS_REVIEW,
            progress=100,
            message=summary,
        )
        raise RuntimeError(f"{project_id} blocked before script generation: {summary}")
    store.save_panels(project_id, project.panels)
    store.update_stage_state(
        project_id,
        PipelineStage.PANEL_REVIEW,
        StageStatus.COMPLETED,
        progress=100,
        message="Detected panels auto-accepted for chunk batch processing.",
    )
    store.update_stage_state(
        project_id,
        PipelineStage.SCRIPT_GENERATION,
        StageStatus.READY,
        progress=0,
        message="Panel review auto-accepted. Starting script generation.",
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


def _run_panel_detection_stage(
    store: ProjectStore,
    queue: QueueService,
    project_id: str,
) -> str:
    store.update_stage_state(
        project_id,
        PipelineStage.PANEL_REVIEW,
        StageStatus.PENDING,
        progress=0,
        message="Panel review will reopen after the new detection pass.",
    )
    store.update_stage_state(
        project_id,
        PipelineStage.SCRIPT_GENERATION,
        StageStatus.PENDING,
        progress=0,
        message="Waiting for panel detection to finish.",
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
    payload: dict[str, object] = {"direct_runner": True}
    job = store.create_job(project_id, PipelineStage.PANEL_DETECTION, payload=payload)
    context = PipelineContext(
        store=store,
        queue=queue,
        project_id=project_id,
        job_id=job.id,
        stage=PipelineStage.PANEL_DETECTION,
    )
    run_panel_detection(context)
    return job.id


def _resolve_video_path(store: ProjectStore, project_id: str) -> Path:
    project = store.get_project(project_id)
    if not project.latest_video:
        raise RuntimeError(f"{project_id} has no latest video after rendering.")
    relative = project.latest_video.path.replace(f"/media/projects/{project_id}/", "")
    return Path(store._project_dir(project_id)) / relative


def main() -> int:
    parser = argparse.ArgumentParser(description="Run narration+video stages directly for chunk projects.")
    parser.add_argument("project_ids", nargs="*", help="Chunk project ids to process.")
    parser.add_argument(
        "--manifest-parent",
        action="append",
        default=[],
        help="Parent project id whose output/chunk_projects.json should be expanded into chunk project ids.",
    )
    parser.add_argument(
        "--skip-existing-video",
        action="store_true",
        help="Skip projects that already have a latest video.",
    )
    parser.add_argument(
        "--force-redetect",
        action="store_true",
        help="Reset each chunk to panel detection, rerun detection, and auto-accept the refreshed panels before continuing.",
    )
    parser.add_argument(
        "--force-script",
        action="store_true",
        help="Regenerate the narration script even if the project already has saved script lines.",
    )
    args = parser.parse_args()

    store = ProjectStore()
    queue = QueueService()
    verifier = VideoVerifier()

    project_ids: list[str] = list(args.project_ids)
    for parent_id in args.manifest_parent:
        manifest_path = Path(store._project_dir(parent_id)) / "output" / "chunk_projects.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        project_ids.extend(chunk["project_id"] for chunk in manifest.get("chunks") or [])

    seen: set[str] = set()
    ordered_ids = [project_id for project_id in project_ids if not (project_id in seen or seen.add(project_id))]
    results: list[dict[str, object]] = []

    for project_id in ordered_ids:
        try:
            _acquire_project_lock(project_id)
        except RuntimeError as exc:
            print(json.dumps({"project_id": project_id, "skipped": True, "reason": str(exc)}), flush=True)
            continue
        _cancel_stale_jobs(store, project_id)
        project = store.get_project(project_id)
        record: dict[str, object] = {"project_id": project_id}

        if args.force_redetect:
            store.reset_pipeline_from_stage(project_id, PipelineStage.PANEL_DETECTION)
            project = store.get_project(project_id)

        if args.skip_existing_video and project.latest_video:
            video_path = _resolve_video_path(store, project_id)
            verification = verifier.verify_project_video(Path(store._project_dir(project_id)), video_path, None)
            record.update(
                {
                    "skipped": True,
                    "latest_video": str(video_path),
                    "verification": verification.to_dict(),
                }
            )
            results.append(record)
            print(json.dumps(record), flush=True)
            continue

        if not store.list_page_paths(project_id):
            record["ingestion_job_id"] = _run_stage(store, queue, project_id, PipelineStage.INGESTION)
            project = store.get_project(project_id)

        panel_detection_state = project.stage_states.get(PipelineStage.PANEL_DETECTION)
        panel_review_state = project.stage_states.get(PipelineStage.PANEL_REVIEW)
        if not project.panels or panel_detection_state is None or panel_detection_state.status != StageStatus.COMPLETED:
            record["panel_detection_job_id"] = _run_panel_detection_stage(store, queue, project_id)
            _auto_accept_detected_panels(store, project_id)
            project = store.get_project(project_id)
        elif panel_review_state is None or panel_review_state.status != StageStatus.COMPLETED:
            _auto_accept_detected_panels(store, project_id)
            project = store.get_project(project_id)

        if args.force_script or not project.script_lines:
            record["script_job_id"] = _run_stage(store, queue, project_id, PipelineStage.SCRIPT_GENERATION)

        project = store.get_project(project_id)
        if not project.audio_files:
            record["narration_job_id"] = _run_stage(store, queue, project_id, PipelineStage.NARRATION_GENERATION)

        project = store.get_project(project_id)
        if not project.latest_video:
            record["video_job_id"] = _run_stage(store, queue, project_id, PipelineStage.VIDEO_RENDERING)

        video_path = _resolve_video_path(store, project_id)
        verification = verifier.verify_project_video(Path(store._project_dir(project_id)), video_path, None)
        record.update(
            {
                "latest_video": str(video_path),
                "verification": verification.to_dict(),
            }
        )
        results.append(record)
        print(json.dumps(record), flush=True)

    print(json.dumps({"count": len(results), "results": results}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
