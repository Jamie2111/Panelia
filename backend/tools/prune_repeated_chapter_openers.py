from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.schemas.project import JobStatus, PipelineStage
from app.services.project_store import ProjectStore
from app.utils.files import ensure_dir


def _file_sha1(path: Path) -> str:
    digest = hashlib.sha1()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _rewrite_pages(pages_dir: Path, kept_paths: list[Path]) -> None:
    staging_dir = ensure_dir(pages_dir.parent / "temp" / "pruned_pages")
    shutil.rmtree(staging_dir, ignore_errors=True)
    ensure_dir(staging_dir)

    for index, source_path in enumerate(kept_paths, start=1):
        shutil.copy2(source_path, staging_dir / f"{index:04d}.png")

    backup_dir = pages_dir.parent / "temp" / "pages_backup_before_prune"
    shutil.rmtree(backup_dir, ignore_errors=True)
    if pages_dir.exists():
        shutil.move(str(pages_dir), str(backup_dir))
    shutil.move(str(staging_dir), str(pages_dir))
    shutil.rmtree(backup_dir, ignore_errors=True)


def prune_project(store: ProjectStore, project_id: str) -> dict[str, object]:
    project = store.get_project(project_id)
    raw = project.chapter_metadata.raw if isinstance(project.chapter_metadata.raw, dict) else {}
    chapters = raw.get("chapters")
    if not isinstance(chapters, list) or len(chapters) < 2:
        return {"project_id": project_id, "skipped": True, "reason": "no-multi-chapter-metadata"}

    pages_dir = Path(store._project_dir(project_id)) / "pages"
    page_paths = sorted(path for path in pages_dir.glob("*.png"))
    if not page_paths:
        return {"project_id": project_id, "skipped": True, "reason": "no-pages"}

    expected_total = sum(int((chapter or {}).get("page_count") or 0) for chapter in chapters if isinstance(chapter, dict))
    if expected_total != len(page_paths):
        return {
            "project_id": project_id,
            "skipped": True,
            "reason": "page-count-mismatch",
            "expected_total": expected_total,
            "actual_total": len(page_paths),
        }

    seen_opener_hashes: dict[str, dict[str, object]] = {}
    updated_chapters: list[dict[str, object]] = []
    kept_paths: list[Path] = []
    removed: list[dict[str, object]] = []
    cursor = 0

    for chapter_index, chapter in enumerate(chapters, start=1):
        chapter_dict = dict(chapter or {})
        page_count = int(chapter_dict.get("page_count") or 0)
        chapter_paths = page_paths[cursor : cursor + page_count]
        cursor += page_count
        if not chapter_paths:
            chapter_dict["page_count"] = 0
            updated_chapters.append(chapter_dict)
            continue

        opener_hash = _file_sha1(chapter_paths[0])
        opener_seen = seen_opener_hashes.get(opener_hash)
        if opener_seen is None:
            seen_opener_hashes[opener_hash] = {
                "chapter_index": chapter_index,
                "chapter_number": chapter_dict.get("chapter_number"),
                "source_url": chapter_dict.get("source_url"),
            }
        elif len(chapter_paths) > 1:
            removed.append(
                {
                    "chapter_index": chapter_index,
                    "chapter_number": chapter_dict.get("chapter_number"),
                    "source_url": chapter_dict.get("source_url"),
                    "removed_page": chapter_paths[0].name,
                    "matched_opener": opener_seen,
                }
            )
            chapter_paths = chapter_paths[1:]

        chapter_dict["page_count"] = len(chapter_paths)
        kept_paths.extend(chapter_paths)
        updated_chapters.append(chapter_dict)

    if not removed:
        return {"project_id": project_id, "skipped": True, "reason": "no-repeated-openers"}

    _rewrite_pages(pages_dir, kept_paths)

    updated_metadata = project.chapter_metadata.model_copy(deep=True)
    updated_raw = dict(raw)
    updated_raw["chapters"] = updated_chapters
    dedupe_payload = dict(updated_raw.get("ingestion_dedupe", {}))
    dedupe_payload["repeated_chapter_openers_removed"] = removed
    updated_raw["ingestion_dedupe"] = dedupe_payload
    updated_metadata = updated_metadata.model_copy(
        update={
            "page_count": len(kept_paths),
            "raw": updated_raw,
        }
    )
    store.update_project_metadata(project_id, chapter_metadata=updated_metadata.model_dump(mode="json"))

    for job in store.list_jobs(project_id):
        if job.status in {JobStatus.QUEUED, JobStatus.RUNNING}:
            store.update_job(project_id, job.id, status=JobStatus.CANCELLED.value)

    store.reset_pipeline_from_stage(project_id, PipelineStage.PANEL_DETECTION)

    return {
        "project_id": project_id,
        "removed_count": len(removed),
        "page_count_before": len(page_paths),
        "page_count_after": len(kept_paths),
        "removed": removed,
    }


def update_parent_manifest(store: ProjectStore, parent_project_id: str) -> dict[str, object]:
    manifest_path = Path(store._project_dir(parent_project_id)) / "output" / "chunk_projects.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    for chunk in payload.get("chunks", []):
        chunk_project_id = str(chunk.get("project_id") or "").strip()
        if not chunk_project_id:
            continue
        project = store.get_project(chunk_project_id)
        chunk["page_total"] = int(project.page_count or 0)
    manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return {"parent_project_id": parent_project_id, "chunk_count": len(payload.get("chunks", []))}


def main() -> int:
    parser = argparse.ArgumentParser(description="Prune repeated chapter opener pages from combined chunk projects.")
    parser.add_argument("project_ids", nargs="+", help="Project ids to repair.")
    parser.add_argument("--update-parent", help="Parent project id whose chunk manifest should be refreshed after pruning.")
    args = parser.parse_args()

    store = ProjectStore()
    results = [prune_project(store, project_id) for project_id in args.project_ids]
    parent_result = update_parent_manifest(store, args.update_parent) if args.update_parent else None
    print(json.dumps({"results": results, "parent": parent_result}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
