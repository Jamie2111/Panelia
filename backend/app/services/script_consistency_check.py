"""
Script ↔ panels consistency check.

A single guard rail that detects the silent-desync class of bugs we hit
multiple times in the legacy pipeline:
    • panels.json says N kept panels, script_manifest.json has M segments
    • A segment's panel_id refers to a panel that no longer exists
    • A kept panel has no narration AND no segment claims it
    • Two segments claim the same panel
    • Segment order doesn't match visual reading order

This is meant to be called:
    • At the end of every script-generation stage (sanity gate)
    • From an admin endpoint the UI can hit before TTS/render
    • From CI tests over fixture projects

It returns a structured report (no side effects). Callers decide whether
to surface it as a warning, hard-block downstream stages, or auto-repair.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class ConsistencyIssue:
    code: str          # Stable identifier for downstream handling
    severity: str      # "info" | "warning" | "error"
    message: str
    panel_id: str | None = None
    segment_id: str | None = None


@dataclass
class ConsistencyReport:
    project_id: str
    panel_count: int
    kept_panel_count: int
    segment_count: int
    issues: list[ConsistencyIssue] = field(default_factory=list)

    @property
    def has_errors(self) -> bool:
        return any(issue.severity == "error" for issue in self.issues)

    def summary(self) -> str:
        if not self.issues:
            return f"OK - {self.kept_panel_count} kept panels match {self.segment_count} segments."
        counts = {"error": 0, "warning": 0, "info": 0}
        for issue in self.issues:
            counts[issue.severity] = counts.get(issue.severity, 0) + 1
        parts = [f"{n} {sev}" for sev, n in counts.items() if n]
        return f"Issues: {', '.join(parts)}"


def check_project(project_dir: Path, project_id: str | None = None) -> ConsistencyReport:
    project_id = project_id or project_dir.name

    # --- Load both sides of the contract ---
    panels_path = project_dir / "panels.json"
    manifest_path = project_dir / "script_manifest.json"

    if not panels_path.exists():
        return ConsistencyReport(
            project_id=project_id,
            panel_count=0,
            kept_panel_count=0,
            segment_count=0,
            issues=[ConsistencyIssue("missing_panels", "error", "panels.json not found")],
        )

    panels: list[dict[str, Any]] = json.loads(panels_path.read_text(encoding="utf-8"))
    kept = [p for p in panels if p.get("keep")]
    kept_ids: set[str] = {str(p["id"]) for p in kept}
    panels_by_id = {str(p["id"]): p for p in panels}

    if not manifest_path.exists():
        return ConsistencyReport(
            project_id=project_id,
            panel_count=len(panels),
            kept_panel_count=len(kept),
            segment_count=0,
            issues=[
                ConsistencyIssue(
                    "missing_manifest",
                    "warning",
                    "script_manifest.json not found - narration stage has not run.",
                )
            ],
        )

    manifest: dict[str, Any] = json.loads(manifest_path.read_text(encoding="utf-8"))
    segments: list[dict[str, Any]] = manifest.get("story_segments") or []
    script_lines: list[str] = manifest.get("script_lines") or []

    issues: list[ConsistencyIssue] = []

    # --- Cross-reference checks ---
    seen_panel_ids: dict[str, str] = {}  # panel_id → segment_id that claimed it
    for seg in segments:
        seg_id = str(seg.get("id") or seg.get("segment_id") or "")
        for pid in seg.get("panel_ids") or []:
            pid = str(pid)
            if pid not in kept_ids:
                issues.append(ConsistencyIssue(
                    "segment_references_missing_panel",
                    "error",
                    f"Segment {seg_id} references panel {pid} which is not a kept panel.",
                    panel_id=pid,
                    segment_id=seg_id,
                ))
                continue
            if pid in seen_panel_ids:
                issues.append(ConsistencyIssue(
                    "panel_claimed_twice",
                    "error",
                    f"Panel {pid} is claimed by both segments {seen_panel_ids[pid]} and {seg_id}.",
                    panel_id=pid,
                    segment_id=seg_id,
                ))
            else:
                seen_panel_ids[pid] = seg_id

    # Any kept panel with no segment? That's a coverage gap.
    uncovered = kept_ids - set(seen_panel_ids)
    if uncovered:
        # Cap at first 10 in the report to avoid spam; record the count separately.
        for pid in sorted(uncovered)[:10]:
            issues.append(ConsistencyIssue(
                "panel_has_no_segment",
                "warning",
                f"Kept panel {pid} is not referenced by any segment.",
                panel_id=pid,
            ))
        if len(uncovered) > 10:
            issues.append(ConsistencyIssue(
                "panel_has_no_segment_truncated",
                "info",
                f"... and {len(uncovered) - 10} more uncovered panels.",
            ))

    # script_lines length should equal segment count.
    if len(script_lines) != len(segments):
        issues.append(ConsistencyIssue(
            "script_lines_segment_mismatch",
            "error",
            f"script_lines has {len(script_lines)} entries but story_segments has {len(segments)}.",
        ))

    # Segment ordering should match visual reading order of their panels.
    expected_order_keys: list[tuple[int, int]] = []
    actual_order_keys: list[tuple[int, int]] = []
    for seg in segments:
        pids = [pid for pid in (seg.get("panel_ids") or []) if pid in panels_by_id]
        if not pids:
            continue
        # Use the first-claimed panel's coordinate.
        first_panel = panels_by_id[str(pids[0])]
        key = (int(first_panel.get("page", 0)), int(first_panel.get("panel", 0)))
        actual_order_keys.append(key)
        expected_order_keys.append(key)
    if sorted(expected_order_keys) != actual_order_keys:
        issues.append(ConsistencyIssue(
            "segments_out_of_visual_order",
            "warning",
            "Story segments are not ordered by visual reading order (page, panel).",
        ))

    # Narration sync: panels.json.narration should equal segment.text for claimed panels.
    for seg in segments:
        seg_text = (seg.get("text") or seg.get("narration") or "").strip()
        for pid in seg.get("panel_ids") or []:
            panel = panels_by_id.get(str(pid))
            if panel is None:
                continue
            panel_text = (panel.get("narration") or "").strip()
            if seg_text and panel_text and seg_text != panel_text:
                issues.append(ConsistencyIssue(
                    "narration_text_mismatch",
                    "warning",
                    f"Panel {pid} narration text differs from segment text - last write wins on next render.",
                    panel_id=str(pid),
                    segment_id=str(seg.get("id") or seg.get("segment_id") or ""),
                ))

    return ConsistencyReport(
        project_id=project_id,
        panel_count=len(panels),
        kept_panel_count=len(kept),
        segment_count=len(segments),
        issues=issues,
    )


def format_report(report: ConsistencyReport) -> str:
    """Human-readable formatter for CLI / logs."""
    lines = [
        f"Project: {report.project_id}",
        f"  Panels: {report.panel_count} ({report.kept_panel_count} kept)",
        f"  Segments: {report.segment_count}",
        f"  Status: {report.summary()}",
    ]
    if report.issues:
        lines.append("")
        for issue in report.issues:
            badge = {"error": "✗", "warning": "⚠", "info": "·"}.get(issue.severity, "•")
            lines.append(f"  {badge} [{issue.code}] {issue.message}")
    return "\n".join(lines)
