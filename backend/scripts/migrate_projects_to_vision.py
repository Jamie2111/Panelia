"""
Migrate existing projects to the vision-grounded narration pipeline.

Updates each project's `metadata.json` so that:
    pipeline_config.script_pipeline_version == "vision"

This does NOT re-run script generation. It only flips the flag so that the
NEXT time you click "Generate script" in the UI, the new vision pipeline
runs instead of the legacy cascade.

Usage:
    python scripts/migrate_projects_to_vision.py            # migrate all
    python scripts/migrate_projects_to_vision.py --dry-run  # show only
    python scripts/migrate_projects_to_vision.py PROJECT_ID # one project
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
PROJECTS_DIR = REPO_ROOT / "backend" / "data" / "projects"


def migrate(project_id: str, *, dry_run: bool) -> tuple[str, str]:
    """Flip a project's pipeline_config to use the vision pipeline +
    end-to-end auto-run. Existing legacy values are preserved if a key
    is missing entirely."""
    meta_path = PROJECTS_DIR / project_id / "metadata.json"
    if not meta_path.exists():
        return ("missing", "metadata.json not found")
    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    pipeline = dict(metadata.get("pipeline_config") or {})
    current_version = str(pipeline.get("script_pipeline_version") or "legacy").strip()
    current_auto = bool(pipeline.get("auto_run_end_to_end", False))
    if current_version.casefold() == "vision" and current_auto:
        return ("already", "vision + auto_run on")
    pipeline["script_pipeline_version"] = "vision"
    pipeline["auto_run_end_to_end"] = True
    metadata["pipeline_config"] = pipeline
    if not dry_run:
        meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return ("migrated", f"{current_version}+auto={current_auto} → vision+auto=True")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "project_id", nargs="?", default=None,
        help="Specific project ID to migrate (defaults to all)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show what would be changed without writing.",
    )
    args = parser.parse_args()

    if not PROJECTS_DIR.exists():
        print(f"ERROR: {PROJECTS_DIR} does not exist.", file=sys.stderr)
        return 1

    targets: list[str]
    if args.project_id:
        targets = [args.project_id]
    else:
        targets = sorted(p.name for p in PROJECTS_DIR.iterdir() if p.is_dir())

    print(f"{'[dry-run] ' if args.dry_run else ''}Inspecting {len(targets)} project(s)\n")
    migrated = already = missing = 0
    for project_id in targets:
        status, detail = migrate(project_id, dry_run=args.dry_run)
        icon = {"migrated": "→", "already": "✓", "missing": "?"}.get(status, "•")
        print(f"  {icon} {project_id}: {status} ({detail})")
        if status == "migrated":
            migrated += 1
        elif status == "already":
            already += 1
        else:
            missing += 1

    print(f"\nSummary: {migrated} migrated, {already} already on vision, {missing} missing/invalid.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
