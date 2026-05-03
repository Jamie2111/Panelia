from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.services.project_store import ProjectStore
from app.services.video_service import VideoRenderService
from app.services.video_verifier import VideoVerifier


def main() -> int:
    parser = argparse.ArgumentParser(description="Merge verified chunk-project videos into a parent project export.")
    parser.add_argument("parent_project_id", help="Parent project id that contains output/chunk_projects.json")
    parser.add_argument("--output-name", default="merged_longform", help="Base name for the merged output video")
    args = parser.parse_args()

    store = ProjectStore()
    verifier = VideoVerifier()
    video_service = VideoRenderService()
    parent_dir = Path(store._project_dir(args.parent_project_id))
    manifest_path = parent_dir / "output" / "chunk_projects.json"
    if not manifest_path.exists():
        raise SystemExit(f"Chunk manifest not found: {manifest_path}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    chunks = manifest.get("chunks") or []
    if not chunks:
        raise SystemExit("Chunk manifest is empty.")

    video_paths: list[Path] = []
    video_config = None

    for chunk in chunks:
        chunk_project = store.get_project(chunk["project_id"])
        if not chunk_project.latest_video:
            raise SystemExit(f"Chunk project {chunk_project.id} has no latest video yet.")
        chunk_dir = Path(store._project_dir(chunk_project.id))
        relative = chunk_project.latest_video.path.replace(f"/media/projects/{chunk_project.id}/", "")
        video_path = chunk_dir / relative
        verification = verifier.verify_project_video(chunk_dir, video_path, None)
        if not verification.ok:
            raise SystemExit(
                f"Chunk project {chunk_project.id} failed verification: {'; '.join(verification.issues)}"
            )
        video_paths.append(video_path)
        video_config = chunk_project.video_config

    if video_config is None:
        raise SystemExit("Unable to resolve a video config from chunk projects.")

    output_path = video_service.merge_videos(parent_dir / "video", video_paths, args.output_name, video_config)
    verification = verifier.verify_project_video(parent_dir, output_path, None)
    print(json.dumps({"output_path": str(output_path), "verification": verification.to_dict()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
