"""Backfill canonical_characters.json for darling from successful panel extractions.

Gemini blocked the portrait pass on most darling pages, so the canonical roster
ended up with only "Zero Two". But the per-panel vision extractions DID identify
other characters (Hiro, Ichigo, Mitsuru, Zorome) on the panels Gemini didn't
refuse. This script mines those panel-level character_names and merges them
into canonical_characters.json so the script_generation prompts have the full
roster.

Usage:
    .venv/bin/python tools/backfill_darling_canonical.py darling-in-the-franxx-6f2b8388
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from uuid import uuid4

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))

# Match placeholders we never want as canonical names.
_PLACEHOLDER_TOKENS = {
    "speaker_1",
    "speaker_2",
    "speaker_3",
    "narrator",
    "unknown",
    "pilot",
    "official",
    "off-screen speaker",
    "off-camera speaker",
    "unseen speaker",
    "bystander",
    "neighbor",
    "ich",  # spurious German pronoun caught from a refused panel
}


def _is_placeholder(name: str) -> bool:
    n = (name or "").strip().casefold()
    if not n:
        return True
    if n in _PLACEHOLDER_TOKENS:
        return True
    if n.startswith("unknown ") or n.startswith("unnamed "):
        return True
    return False


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("project_id")
    parser.add_argument(
        "--min-mentions",
        type=int,
        default=2,
        help="Minimum panel mentions before a name is added (default: 2)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show planned roster but do not write canonical_characters.json",
    )
    args = parser.parse_args()

    project_dir = BACKEND / "data" / "projects" / args.project_id
    output_dir = project_dir / "output"
    canonical_path = output_dir / "canonical_characters.json"
    panel_vision_path = output_dir / "panel_vision_final.json"
    if not panel_vision_path.exists():
        # Fall back to raw vision if final hasn't been written yet.
        panel_vision_path = output_dir / "panel_vision.json"
    if not panel_vision_path.exists():
        print(f"No panel_vision*.json found in {output_dir}", file=sys.stderr)
        return 2

    panels = json.loads(panel_vision_path.read_text())

    # Mine character_names + speaker fields. Track which panels each name
    # appeared in so we can populate portrait_panel_ids without a separate pass.
    mentions: Counter[str] = Counter()
    panel_ids: defaultdict[str, list[str]] = defaultdict(list)
    pages: defaultdict[str, set[int]] = defaultdict(set)

    for panel in panels:
        panel_id = panel.get("panel_id") or ""
        page = int(panel.get("page") or 0)
        names_seen = set()
        for raw in (panel.get("character_names") or []):
            n = (raw or "").strip()
            if not n or _is_placeholder(n):
                continue
            names_seen.add(n)
        speaker = (panel.get("speaker") or "").strip()
        if speaker and not _is_placeholder(speaker):
            names_seen.add(speaker)
        for n in names_seen:
            mentions[n] += 1
            if panel_id and panel_id not in panel_ids[n]:
                panel_ids[n].append(panel_id)
            if page:
                pages[n].add(page)

    # Load existing canonical so we don't duplicate entries.
    existing: list[dict] = []
    if canonical_path.exists():
        existing = json.loads(canonical_path.read_text())
        if not isinstance(existing, list):
            existing = []

    existing_names = {(entry.get("name") or "").strip().casefold() for entry in existing}
    existing_aliases = set()
    for entry in existing:
        for alias in entry.get("aliases") or []:
            existing_aliases.add((alias or "").strip().casefold())

    # Build new roster entries for names not already canonical.
    new_entries: list[dict] = []
    for name, count in mentions.most_common():
        if count < args.min_mentions:
            continue
        norm = name.casefold()
        if norm in existing_names or norm in existing_aliases:
            continue
        # Pick a stable id slot that doesn't collide with existing.
        next_idx = len(existing) + len(new_entries) + 1
        new_entries.append(
            {
                "stable_id": f"char-{next_idx:03d}",
                "name": name,
                "role": "supporting",
                "visual_description": (
                    f"Character identified in {count} panel(s) by the panel vision extractor. "
                    "Gemini refused the page-level portrait pass for most pages, so no "
                    "page-thumbnail-derived visual_description is available."
                ),
                "portrait_panel_ids": panel_ids[name][:3],
                "portrait_pages": sorted(pages[name])[:8],
                "confidence": None,
                "aliases": [],
            }
        )

    final_roster = existing + new_entries

    print(f"Existing canonical entries: {len(existing)}")
    print(f"Mined unique non-placeholder names: {len(mentions)}")
    print(f"New entries to add (>= {args.min_mentions} mentions): {len(new_entries)}")
    for entry in new_entries:
        print(
            f"  + {entry['name']:<25} stable_id={entry['stable_id']} "
            f"panels={len(entry['portrait_panel_ids'])} pages={entry['portrait_pages']}"
        )

    if args.dry_run:
        print("\n[dry-run] canonical_characters.json NOT written")
        return 0

    canonical_path.write_text(json.dumps(final_roster, indent=2, ensure_ascii=False))
    print(f"\nWrote {canonical_path} with {len(final_roster)} canonical entries")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
