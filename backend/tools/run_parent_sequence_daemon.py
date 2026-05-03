from __future__ import annotations

import atexit
import fcntl
import os
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = BACKEND_ROOT.parent

sys.path.insert(0, str(BACKEND_ROOT))

from tools.merge_chunk_projects import main as merge_chunk_projects_main
from tools.run_chunk_pipeline import main as run_chunk_pipeline_main

_LOCK_HANDLE = None


def _acquire_parent_sequence_lock(parent_ids: list[str]) -> None:
    global _LOCK_HANDLE
    lock_name = "_".join(parent_ids)[:180] or "parent_sequence"
    lock_path = Path("/tmp") / f"panelia-parent-sequence-{lock_name}.lock"
    handle = lock_path.open("w")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        handle.close()
        raise RuntimeError("Another long-form parent sequence is already running for this project set.")
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


def _is_transient_network_failure(error: BaseException, trace: str) -> bool:
    text = f"{error}\n{trace}".lower()
    markers = (
        "requests.exceptions.connectionerror",
        "name resolution",
        "maxretryerror",
        "failed to resolve",
        "temporary failure",
        "timed out",
        "read timeout",
        "connect timeout",
        "connection aborted",
        "connection reset",
    )
    return any(marker in text for marker in markers)


def main() -> int:
    if len(sys.argv) < 4:
        print(
            "usage: run_parent_sequence_daemon.py <log_path> <parent_project_id> <parent_project_id> [...] [--force-redetect] [--skip-existing-video]",
            file=sys.stderr,
        )
        return 64

    log_path = Path(sys.argv[1]).expanduser().resolve()
    extra_flags = [value for value in sys.argv[2:] if value in {"--force-redetect", "--skip-existing-video"}]
    parent_ids = [
        str(value).strip()
        for value in sys.argv[2:]
        if str(value).strip() and str(value).strip() not in {"--force-redetect", "--skip-existing-video"}
    ]
    if not parent_ids:
        print("No parent project ids were supplied.", file=sys.stderr)
        return 64

    try:
        _acquire_parent_sequence_lock(parent_ids)
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 0

    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8", buffering=1) as handle:
        def log(message: str) -> None:
            handle.write(f"{message}\n")
            handle.flush()

        try:
            os.chdir(REPO_ROOT)
            for parent_id in parent_ids:
                log(f"[{datetime.now().isoformat(timespec='seconds')}] starting parent {parent_id}")
                run_result = None
                last_error: BaseException | None = None
                for attempt in range(1, 5):
                    try:
                        sys.argv = [
                            "run_chunk_pipeline.py",
                            "--manifest-parent",
                            parent_id,
                            *extra_flags,
                        ]
                        run_result = int(run_chunk_pipeline_main() or 0)
                        last_error = None
                        break
                    except Exception as exc:
                        trace = traceback.format_exc()
                        if attempt >= 4 or not _is_transient_network_failure(exc, trace):
                            log(f"[{datetime.now().isoformat(timespec='seconds')}] chunk pipeline crashed for {parent_id} on attempt {attempt}")
                            log(trace)
                            raise
                        delay = min(30 * (2 ** (attempt - 1)), 180)
                        last_error = exc
                        log(
                            f"[{datetime.now().isoformat(timespec='seconds')}] transient failure for {parent_id} on attempt {attempt}; retrying in {delay}s"
                        )
                        log(trace)
                        time.sleep(delay)
                if run_result is None and last_error is not None:
                    raise last_error
                if run_result != 0:
                    log(f"[{datetime.now().isoformat(timespec='seconds')}] chunk pipeline failed for {parent_id} with code {run_result}")
                    return run_result

                log(f"[{datetime.now().isoformat(timespec='seconds')}] chunk pipeline completed for {parent_id}; starting merge")
                sys.argv = [
                    "merge_chunk_projects.py",
                    parent_id,
                    "--output-name",
                    "merged_longform",
                ]
                merge_result = int(merge_chunk_projects_main() or 0)
                log(f"[{datetime.now().isoformat(timespec='seconds')}] merge finished for {parent_id} with code {merge_result}")
                if merge_result != 0:
                    return merge_result

            log(f"[{datetime.now().isoformat(timespec='seconds')}] finished parent sequence")
            return 0
        except Exception:
            log(f"[{datetime.now().isoformat(timespec='seconds')}] unhandled exception during parent sequence")
            log(traceback.format_exc())
            return 1


if __name__ == "__main__":
    raise SystemExit(main())
