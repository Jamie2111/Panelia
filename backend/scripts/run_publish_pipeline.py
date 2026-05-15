"""
Run a project's downstream stages (narration → video → bundle) directly,
bypassing the FastAPI queue. Used to QA the publish pipeline end-to-end
without standing up the full worker process.

Usage:
    python scripts/run_publish_pipeline.py <project_id>

Runs in sequence:
  • script_generation (only if no script.txt exists yet)
  • narration_generation
  • video_rendering
  • youtube_bundle
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
BACKEND_ROOT = REPO_ROOT / "backend"
sys.path.insert(0, str(BACKEND_ROOT))

import os
ENV_PATH = REPO_ROOT / ".env"
if ENV_PATH.exists():
    for line in ENV_PATH.read_text().splitlines():
        if "=" in line and not line.strip().startswith("#"):
            k, _, v = line.partition("=")
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))

from app.pipeline.context import PipelineContext  # noqa: E402
from app.pipeline.stages import (  # noqa: E402
    run_narration_generation,
    run_script_generation,
    run_video_rendering,
    run_youtube_bundle,
)
from app.schemas.project import JobRecord, JobStatus, PipelineStage  # noqa: E402
from app.services.project_store import ProjectStore  # noqa: E402
from app.services.queue_service import QueueService  # noqa: E402


def _progress(stage: str):
    def cb(pct: float, msg: str) -> None:
        bar = "█" * int(pct / 2) + "·" * (50 - int(pct / 2))
        print(f"\r[{stage}] [{bar}] {pct:5.1f}%  {msg[:60]:<60}", end="", flush=True)
    return cb


def _run_stage(handler, project_id: str, stage: PipelineStage, store: ProjectStore, queue: QueueService) -> bool:
    """Create a job, invoke the handler with a real context, report timing."""
    print(f"\n────── {stage.value} ──────")
    started = time.perf_counter()
    job = store.create_job(project_id, stage, payload={})
    ctx = PipelineContext(store=store, queue=queue, project_id=project_id, job_id=job.id, stage=stage)
    try:
        handler(ctx)
    except Exception as exc:  # noqa: BLE001
        print(f"\n✗ {stage.value} failed: {exc}")
        return False
    print(f"\n✓ {stage.value} done in {time.perf_counter() - started:.1f}s")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("project_id")
    parser.add_argument("--skip-script", action="store_true", help="Skip script_generation even if no script exists")
    parser.add_argument("--skip-audio", action="store_true")
    parser.add_argument("--skip-video", action="store_true")
    parser.add_argument("--skip-bundle", action="store_true")
    args = parser.parse_args()

    store = ProjectStore()
    queue = QueueService()

    project_dir = REPO_ROOT / "backend" / "data" / "projects" / args.project_id
    has_script = (project_dir / "script_manifest.json").exists()
    if not has_script and not args.skip_script:
        if not _run_stage(run_script_generation, args.project_id, PipelineStage.SCRIPT_GENERATION, store, queue):
            return
    if not args.skip_audio:
        if not _run_stage(run_narration_generation, args.project_id, PipelineStage.NARRATION_GENERATION, store, queue):
            return
    if not args.skip_video:
        if not _run_stage(run_video_rendering, args.project_id, PipelineStage.VIDEO_RENDERING, store, queue):
            return
    if not args.skip_bundle:
        if not _run_stage(run_youtube_bundle, args.project_id, PipelineStage.YOUTUBE_BUNDLE, store, queue):
            return

    print("\n✓ Pipeline complete.")


if __name__ == "__main__":
    main()
