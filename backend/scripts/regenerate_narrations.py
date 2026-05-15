"""
Regenerate narrations for an entire project using PanelVisionNarrator.

Usage:
    python -m backend.scripts.regenerate_narrations <project_id> [--limit N]

This is the replacement for the legacy script_generation cascade. It loads
the project's kept panels in visual order, sends each panel image to
Gemini Vision with rolling continuity context, and writes the results to:
    • panels.json            (panel.narration field per kept panel)
    • script_manifest.json   (canonical script + story_segments)
    • script.txt             (plaintext, for inspection)

Failed/weak panels are flagged with `needs_regenerate=True` in segments
and `vision_failed`/`vision_needs_regenerate` in panel.review_flags.
They are NOT silently filled with garbage.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import time
from pathlib import Path

# Ensure the backend `app` package is importable when run as a script.
REPO_ROOT = Path(__file__).resolve().parents[2]
BACKEND_ROOT = REPO_ROOT / "backend"
sys.path.insert(0, str(BACKEND_ROOT))

# Load .env so GEMINI_API_KEY is available.
ENV_PATH = REPO_ROOT / ".env"
if ENV_PATH.exists():
    import os
    for line in ENV_PATH.read_text().splitlines():
        if "=" in line and not line.strip().startswith("#"):
            k, _, v = line.partition("=")
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))

from app.services.panel_vision_narrator import (  # noqa: E402
    PanelVisionNarrator,
    panels_from_store,
    write_narration_outputs,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s :: %(message)s",
)
logger = logging.getLogger("regenerate_narrations")


def _progress(pct: float, msg: str) -> None:
    bar = "█" * int(pct / 2) + "·" * (50 - int(pct / 2))
    print(f"\r[{bar}] {pct:5.1f}%  {msg}", end="", flush=True)


async def _run(project_id: str, limit: int | None) -> None:
    project_dir = REPO_ROOT / "backend" / "data" / "projects" / project_id
    panels_path = project_dir / "panels.json"
    if not panels_path.exists():
        raise SystemExit(f"panels.json not found at {panels_path}")

    panels_json = json.loads(panels_path.read_text())
    panel_inputs = panels_from_store(project_dir, panels_json)
    if limit:
        panel_inputs = panel_inputs[:limit]
        print(f"⚠ Limiting to first {limit} panels (test mode)")

    print(f"Loaded {len(panel_inputs)} kept panels from {project_id}")
    missing_images = [p for p in panel_inputs if not p.image_path.exists()]
    if missing_images:
        print(f"⚠ {len(missing_images)} panel images are missing; they will be marked failed.")

    # Build / load the cast bible so the narrator can refer to characters
    # by name. Reads the manga / chapter title from metadata.json.
    from app.services.cast_bible_service import CastBibleService
    cast_block = ""
    try:
        metadata_path = project_dir / "metadata.json"
        manga_title = ""
        chapter_title = ""
        if metadata_path.exists():
            meta = json.loads(metadata_path.read_text())
            chapter_meta = meta.get("chapter_metadata") or {}
            manga_title = str(chapter_meta.get("manga_title") or meta.get("name") or "").strip()
            chapter_title = str(chapter_meta.get("chapter_title") or "").strip()
        bible = CastBibleService().ensure_bible(
            project_dir,
            manga_title=manga_title or "(unknown)",
            chapter_title=chapter_title or "(unknown)",
        )
        cast_block = CastBibleService.format_for_prompt(bible)
        print(f"Cast bible: {len(bible.members)} characters ({bible.source})")
    except Exception as exc:  # noqa: BLE001
        print(f"⚠ Cast bible step skipped: {exc}")

    narrator = PanelVisionNarrator()
    started = time.perf_counter()
    batch = await narrator.narrate_chapter(
        panel_inputs,
        cast_block=cast_block,
        progress_callback=_progress,
    )
    elapsed = time.perf_counter() - started
    print()  # newline after progress bar

    print(f"\nDone in {elapsed:.1f}s")
    print(f"  ✓ Successful: {batch.successful}")
    print(f"  ⚠ Needs review / failed: {batch.failed}")

    summary = write_narration_outputs(
        project_dir, panel_inputs, batch.results, panels_json
    )
    print(f"  Wrote panels.json, script_manifest.json, script.txt")
    print(f"  Stats: {summary}")

    # Print 5 sample narrations for spot-checking quality
    print("\n── Sample narrations ──")
    ok = [r for r in batch.results if r.status == "ok"]
    for r in ok[: min(5, len(ok))]:
        print(f"  • {r.narration}")
    if batch.failed:
        print("\n── Panels needing regeneration ──")
        for r in batch.results:
            if r.status != "ok":
                print(f"  • {r.panel_id} ({r.status}): {r.reason}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("project_id", help="Project ID under backend/data/projects/")
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Limit to first N panels (for fast smoke tests)",
    )
    args = parser.parse_args()
    asyncio.run(_run(args.project_id, args.limit))


if __name__ == "__main__":
    main()
