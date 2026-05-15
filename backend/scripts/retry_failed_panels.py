"""
Retry only the panels that failed during a vision narration run.

Identifies panels whose `narration_source` starts with "vision_failed" or
"vision_needs_regenerate", or whose review_flags contain a vision_*
marker, and re-narrates just those — keeping all already-good narrations
intact.

Usage:
    python scripts/retry_failed_panels.py <project_id>
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
BACKEND_ROOT = REPO_ROOT / "backend"
sys.path.insert(0, str(BACKEND_ROOT))

ENV_PATH = REPO_ROOT / ".env"
if ENV_PATH.exists():
    import os
    for line in ENV_PATH.read_text().splitlines():
        if "=" in line and not line.strip().startswith("#"):
            k, _, v = line.partition("=")
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))

from app.services.panel_vision_narrator import (  # noqa: E402
    PanelInput,
    PanelVisionNarrator,
)

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s :: %(message)s")


def _panel_needs_retry(panel: dict) -> bool:
    src = str(panel.get("narration_source") or "")
    if src.startswith(("vision_failed", "vision_needs_regenerate")):
        return True
    for flag in panel.get("review_flags") or []:
        if str(flag).startswith("vision_"):
            return True
    if panel.get("keep") and not (panel.get("narration") or "").strip():
        return True
    return False


async def _run(project_id: str) -> None:
    project_dir = REPO_ROOT / "backend" / "data" / "projects" / project_id
    panels_path = project_dir / "panels.json"
    if not panels_path.exists():
        raise SystemExit(f"panels.json not found at {panels_path}")

    panels_json = json.loads(panels_path.read_text(encoding="utf-8"))
    kept_sorted = sorted(
        [p for p in panels_json if p.get("keep")],
        key=lambda p: (int(p.get("page", 0)), int(p.get("panel", 0))),
    )
    failed = [p for p in kept_sorted if _panel_needs_retry(p)]
    if not failed:
        print("No panels need retry — all narrations are healthy.")
        return
    print(f"Retrying {len(failed)} panels…")

    narrator = PanelVisionNarrator()

    # Index of each panel in the visual order list, for continuity context.
    index_by_id = {p["id"]: i for i, p in enumerate(kept_sorted)}

    succeeded = 0
    failures_still: list[tuple[str, str]] = []
    for panel in failed:
        order = int(panel.get("order", 0))
        image_path = project_dir / "panels" / f"panel_{order:03d}.png"
        if not image_path.exists():
            failures_still.append((panel["id"], "image missing"))
            continue
        pos = index_by_id.get(panel["id"], 0)
        prior = kept_sorted[max(0, pos - 4):pos]
        context_lines = [
            (p.get("narration") or "").strip()
            for p in prior
            if (p.get("narration") or "").strip()
            and not _panel_needs_retry(p)
        ]
        context_str = "\n".join(f"  • {line}" for line in context_lines) or "  (this is the opening panel)"

        panel_input = PanelInput(
            panel_id=panel["id"],
            order=order,
            page=int(panel.get("page", 0)),
            panel=int(panel.get("panel", 0)),
            image_path=image_path,
            ocr_text=str(panel.get("ocr_text") or ""),
            character_hints=[],
        )

        semaphore = asyncio.Semaphore(1)
        result = await narrator._narrate_one(panel_input, context_str, semaphore)

        if result.status == "ok" and result.narration:
            panel["narration"] = result.narration
            panel["narration_source"] = "panel_vision_narrator"
            panel["review_flags"] = [
                f for f in (panel.get("review_flags") or [])
                if not str(f).startswith("vision_")
            ]
            succeeded += 1
            print(f"  ✓ {panel['id']}: {result.narration[:80]}")
        else:
            failures_still.append((panel["id"], result.reason or result.status))
            print(f"  ✗ {panel['id']}: {result.reason}")

    panels_path.write_text(json.dumps(panels_json, indent=2), encoding="utf-8")

    # Mirror to manifest + script.json + script.txt so all surfaces stay in sync.
    manifest_path = project_dir / "script_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        narration_by_panel = {p["id"]: (p.get("narration") or "") for p in panels_json if p.get("keep")}
        retry_ids = {panel_id for panel_id, _ in failures_still}
        for seg in manifest.get("story_segments", []):
            for pid in seg.get("panel_ids") or []:
                if pid in narration_by_panel:
                    seg["text"] = narration_by_panel[pid]
                    seg["narration"] = narration_by_panel[pid]
                    seg["needs_regenerate"] = pid in retry_ids
                    if pid not in retry_ids:
                        seg["regenerate_reason"] = ""
                    break
        manifest["script_lines"] = [s.get("text", "") for s in manifest.get("story_segments", [])]
        manifest["script_story"] = "\n".join(line for line in manifest["script_lines"] if line)
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        # Mirror to legacy script.json + script.txt
        script_json_path = project_dir / "script.json"
        if script_json_path.exists():
            legacy = json.loads(script_json_path.read_text(encoding="utf-8"))
            legacy["script_lines"] = manifest["script_lines"]
            legacy["script_story"] = manifest["script_story"]
            legacy["story_segments"] = manifest["story_segments"]
            script_json_path.write_text(json.dumps(legacy, indent=2), encoding="utf-8")
        (project_dir / "script.txt").write_text("\n".join(manifest["script_lines"]), encoding="utf-8")

    print(f"\nRetry complete: {succeeded}/{len(failed)} succeeded. {len(failures_still)} still need review.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("project_id")
    args = parser.parse_args()
    asyncio.run(_run(args.project_id))


if __name__ == "__main__":
    main()
