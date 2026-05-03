from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.services.project_store import ProjectStore


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark saved script quality across Panelia projects.")
    parser.add_argument("project_ids", nargs="*", help="Project ids to score.")
    parser.add_argument(
        "--manifest-parent",
        action="append",
        default=[],
        help="Parent project id whose chunk manifest should be expanded into project ids.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON only.",
    )
    args = parser.parse_args()

    store = ProjectStore()
    project_ids: list[str] = list(args.project_ids)
    for parent_id in args.manifest_parent:
        manifest_path = Path(store._project_dir(parent_id)) / "output" / "chunk_projects.json"
        if not manifest_path.exists():
            raise SystemExit(f"Missing chunk manifest: {manifest_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        project_ids.extend(chunk["project_id"] for chunk in manifest.get("chunks") or [])

    seen: set[str] = set()
    ordered_ids = [project_id for project_id in project_ids if not (project_id in seen or seen.add(project_id))]
    reports: list[dict[str, object]] = []

    for project_id in ordered_ids:
        project = store.get_project(project_id)
        report = store.load_script_quality_report(project_id)
        entry = {
            "project_id": project_id,
            "project_name": project.name,
            "kept_panels": sum(1 for panel in project.panels if panel.keep),
            "script_lines": len(project.script_lines),
            "quality_score": report.get("quality_score"),
            "should_block_tts": report.get("should_block_tts"),
            "blank_lines": report.get("blank_lines"),
            "duplicate_lines": report.get("duplicate_lines"),
            "raw_ocr_echo_lines": report.get("raw_ocr_echo_lines"),
            "fact_mismatch_lines": report.get("fact_mismatch_lines"),
            "generic_lines": report.get("generic_lines"),
            "visual_lines": report.get("visual_lines"),
            "summary": report.get("summary"),
        }
        reports.append(entry)

    payload = {"count": len(reports), "projects": reports}
    if args.json:
        print(json.dumps(payload, indent=2))
        return 0

    for item in reports:
        print(
            f"{item['project_id']}: score={item['quality_score']} "
            f"block_tts={item['should_block_tts']} kept={item['kept_panels']} lines={item['script_lines']} "
            f"blanks={item['blank_lines']} dupes={item['duplicate_lines']} "
            f"raw={item['raw_ocr_echo_lines']} mismatch={item['fact_mismatch_lines']} "
            f"generic={item['generic_lines']} visual={item['visual_lines']}"
        )
        print(f"  {item['summary']}")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
