from __future__ import annotations

import json
import sys
import time
from datetime import datetime
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = BACKEND_ROOT.parent
sys.path.insert(0, str(BACKEND_ROOT))

from app.services.project_store import ProjectStore
from app.services.video_verifier import VideoVerifier
from tools.merge_chunk_projects import main as merge_main


def main() -> int:
    if len(sys.argv) < 3:
        print("usage: merge_when_ready_daemon.py <parent_project_id> <log_path>", file=sys.stderr)
        return 64

    parent_project_id = sys.argv[1]
    log_path = Path(sys.argv[2]).expanduser().resolve()
    log_path.parent.mkdir(parents=True, exist_ok=True)

    store = ProjectStore()
    verifier = VideoVerifier()
    parent_dir = Path(store._project_dir(parent_project_id))
    manifest_path = parent_dir / "output" / "chunk_projects.json"

    with log_path.open("a", encoding="utf-8", buffering=1) as handle:
        sys.stdout = handle
        sys.stderr = handle
        print(f"[{datetime.now().isoformat(timespec='seconds')}] waiting to merge parent {parent_project_id}")
        while True:
            if not manifest_path.exists():
                print(f"[{datetime.now().isoformat(timespec='seconds')}] manifest missing: {manifest_path}")
                time.sleep(120)
                continue

            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            chunks = manifest.get("chunks") or []
            ready = True
            for chunk in chunks:
                project = store.get_project(chunk["project_id"])
                if not project.latest_video:
                    ready = False
                    break
                relative = project.latest_video.path.replace(f"/media/projects/{project.id}/", "")
                video_path = Path(store._project_dir(project.id)) / relative
                verification = verifier.verify_project_video(Path(store._project_dir(project.id)), video_path, None)
                if not verification.ok:
                    ready = False
                    break

            if not ready:
                print(f"[{datetime.now().isoformat(timespec='seconds')}] chunks not ready yet for {parent_project_id}")
                time.sleep(120)
                continue

            print(f"[{datetime.now().isoformat(timespec='seconds')}] all chunks ready for {parent_project_id}; merging")
            sys.argv = ["merge_chunk_projects.py", parent_project_id, "--output-name", "merged_longform"]
            result = int(merge_main() or 0)
            if result == 0:
                print(f"[{datetime.now().isoformat(timespec='seconds')}] merge completed for {parent_project_id}")
                return 0
            print(f"[{datetime.now().isoformat(timespec='seconds')}] merge failed for {parent_project_id}; retrying")
            time.sleep(120)


if __name__ == "__main__":
    raise SystemExit(main())
