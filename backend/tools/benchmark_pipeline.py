from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.services.project_store import ProjectStore


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark panel + script quality across Panelia projects.")
    parser.add_argument("project_ids", nargs="*", help="Project ids to score.")
    parser.add_argument(
        "--manifest-parent",
        action="append",
        default=[],
        help="Parent project id whose chunk manifest should be expanded into project ids.",
    )
    parser.add_argument("--json", action="store_true", help="Emit JSON only.")
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
        panel_report = store.load_panel_quality_report(project_id)
        script_report = store.load_script_quality_report(project_id)
        reports.append(
            {
                "project_id": project_id,
                "project_name": project.name,
                "panels_total": panel_report.get("total_panels"),
                "panels_kept": panel_report.get("kept_panels"),
                "panel_score": panel_report.get("quality_score"),
                "block_script": panel_report.get("should_block_script"),
                "whitespace_panels": panel_report.get("whitespace_panels"),
                "full_page_like_panels": panel_report.get("full_page_like_panels"),
                "composite_like_panels": panel_report.get("composite_like_panels"),
                "suspicious_auto_skips": panel_report.get("suspicious_auto_skips"),
                "script_score": script_report.get("quality_score"),
                "block_tts": script_report.get("should_block_tts"),
                "blank_lines": script_report.get("blank_lines"),
                "duplicate_lines": script_report.get("duplicate_lines"),
                "raw_ocr_echo_lines": script_report.get("raw_ocr_echo_lines"),
                "generic_lines": script_report.get("generic_lines"),
                "fact_mismatch_lines": script_report.get("fact_mismatch_lines"),
                "panel_summary": panel_report.get("summary"),
                "script_summary": script_report.get("summary"),
            }
        )

    payload = {"count": len(reports), "projects": reports}
    if args.json:
        print(json.dumps(payload, indent=2))
        return 0

    for item in reports:
        print(
            f"{item['project_id']}: panel_score={item['panel_score']} block_script={item['block_script']} "
            f"script_score={item['script_score']} block_tts={item['block_tts']}"
        )
        print(
            f"  panel: kept={item['panels_kept']}/{item['panels_total']} whitespace={item['whitespace_panels']} "
            f"fullpage={item['full_page_like_panels']} composite={item['composite_like_panels']} "
            f"auto_skip={item['suspicious_auto_skips']}"
        )
        print(
            f"  script: blanks={item['blank_lines']} dupes={item['duplicate_lines']} raw={item['raw_ocr_echo_lines']} "
            f"generic={item['generic_lines']} mismatch={item['fact_mismatch_lines']}"
        )
        print(f"  {item['panel_summary']}")
        print(f"  {item['script_summary']}")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
