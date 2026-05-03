from __future__ import annotations

import atexit
import fcntl
import os
import sys
import traceback
from datetime import datetime
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = BACKEND_ROOT.parent

sys.path.insert(0, str(BACKEND_ROOT))

from tools.merge_chunk_projects import main as merge_chunk_projects_main
from tools.run_chunk_pipeline import main as run_chunk_pipeline_main

_LOCK_HANDLE = None


def _acquire_parent_lock(parent_project_id: str) -> None:
    global _LOCK_HANDLE
    lock_path = Path("/tmp") / f"panelia-parent-{parent_project_id}.lock"
    handle = lock_path.open("w")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        handle.close()
        raise RuntimeError("Another long-form daemon is already running for this parent project.")
    handle.write(str(os.getpid()))
    handle.flush()
    _LOCK_HANDLE = handle

    def _release() -> None:
        global _LOCK_HANDLE
        if _LOCK_HANDLE is None:
            return
        try:
            fcntl.flock(_LOCK_HANDLE.fileno(), fcntl.LOCK_UN)
        except Exception:
            pass
        try:
            _LOCK_HANDLE.close()
        except Exception:
            pass
        _LOCK_HANDLE = None

    atexit.register(_release)


def main() -> int:
    if len(sys.argv) < 3:
        print("usage: run_chunk_pipeline_daemon.py <parent_project_id> <log_path> [--force-redetect] [--skip-existing-video]", file=sys.stderr)
        return 64

    parent_project_id = sys.argv[1]
    log_path = Path(sys.argv[2]).expanduser().resolve()
    extra_flags = [value for value in sys.argv[3:] if value in {"--force-redetect", "--skip-existing-video"}]
    log_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        _acquire_parent_lock(parent_project_id)
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 0

    with log_path.open("a", encoding="utf-8", buffering=1) as handle:
        def log(message: str) -> None:
            handle.write(f"{message}\n")
            handle.flush()

        try:
            log(f"[{datetime.now().isoformat(timespec='seconds')}] starting parent {parent_project_id}")
            os.chdir(REPO_ROOT)
            sys.argv = [
                "run_chunk_pipeline.py",
                "--manifest-parent",
                parent_project_id,
                *extra_flags,
            ]
            run_result = int(run_chunk_pipeline_main() or 0)
            if run_result != 0:
                log(f"[{datetime.now().isoformat(timespec='seconds')}] chunk pipeline failed for {parent_project_id} with code {run_result}")
                return run_result
            log(f"[{datetime.now().isoformat(timespec='seconds')}] chunk pipeline completed for {parent_project_id}; starting merge")
            sys.argv = [
                "merge_chunk_projects.py",
                parent_project_id,
                "--output-name",
                "merged_longform",
            ]
            merge_result = int(merge_chunk_projects_main() or 0)
            log(f"[{datetime.now().isoformat(timespec='seconds')}] merge finished for {parent_project_id} with code {merge_result}")
            return merge_result
        except Exception:
            log(f"[{datetime.now().isoformat(timespec='seconds')}] unhandled exception for {parent_project_id}")
            log(traceback.format_exc())
            return 1


if __name__ == "__main__":
    raise SystemExit(main())
