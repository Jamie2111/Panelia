"""
YouTubeBundleService - generates the "publish to YouTube" bundle.

What this service produces after a project's video has rendered:

  1. youtube_bundle/title.txt          - single best YouTube title (≤100 chars)
  2. youtube_bundle/title_variants.json - three alternative titles ranked
  3. youtube_bundle/description.md     - description with hook, summary,
                                         chapters (if multiple), and a
                                         caption-friendly tags line
  4. youtube_bundle/thumbnail_source.png - the panel image we selected
                                           as the visual base for the
                                           thumbnail
  5. youtube_bundle/thumbnail.png      - viral-style thumbnail with text
                                         overlay, 1280×720, ready to upload
  6. youtube_bundle/manifest.json      - paths + metadata for the API

The whole bundle is intended to be drag-and-drop into YouTube Studio.
Title goes in the title field, description.md content goes in the
description field, thumbnail.png is the custom thumbnail.

Best-panel selection heuristic:
  • Skip the first 10% of panels (almost always title/credits boring)
  • Prefer panels whose vision narration matches "shocked", "revealed",
    "explodes", "appears", "kiss", "screams", "destroyed", "transforms"
    etc. - the moments that pop on a thumbnail
  • Prefer larger original-image panels (more visual surface area)
  • Final tie-breaker: panel close to the climax (60-80% through the
    chapter)

Viral thumbnail composition:
  • Take the selected panel, fit to 1280×720 with a slight zoom-in
  • Add a soft mint glow vignette in one corner
  • Add a bold uppercase title overlay (≤4 words) bottom-left with
    white fill + heavy stroke, plus a colored highlight word
  • Optional: add a small red circle/arrow accent pointing at the
    subject if we can detect a focal region (heuristic: brightest
    quadrant)
"""

from __future__ import annotations

import json
import logging
import math
import random
import re
import textwrap
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont

from app.core.config import get_settings
from app.utils.files import write_json

try:
    import google.generativeai as genai
    _GEMINI_AVAILABLE = True
except ImportError:
    _GEMINI_AVAILABLE = False

logger = logging.getLogger(__name__)


# ── Tuning constants ──────────────────────────────────────────────────────
THUMBNAIL_SIZE = (1280, 720)
TITLE_CHAR_BUDGET = 70           # Aim short; YouTube caps at 100 chars.
DESCRIPTION_CHAR_BUDGET = 4500   # YouTube max is 5000; leave headroom.

# Keywords that strongly correlate with thumbnail-worthy moments.
_THUMBNAIL_KEYWORDS = (
    "shock", "shouts", "screams", "explodes", "explosion", "destroyed",
    "reveal", "appears", "transforms", "kisses", "kiss", "punch", "fall",
    "dies", "monster", "giant", "huge", "massive", "tears", "blood",
    "weapon", "sword", "burning", "fire", "lightning",
)


def _normalize_chapter_label(raw: str | None) -> str:
    """Coerce a chapter title into 'Chapters X-Y' or 'Chapter X' form.

    Accepts whatever the upstream pipeline saved (often 'Combined chapters
    1-10', 'chapter 4', 'Chapters 1 to 10', etc.) and returns a clean
    label suitable for a YouTube title. Falls back to the input verbatim
    if no number can be found, so series with non-numbered chapters
    ('Epilogue', 'Side story') survive untouched.
    """
    if not raw:
        return ""
    text = str(raw).strip()
    if not text:
        return ""
    # Range like "1-10", "1 to 10", "1 - 10" → "Chapters 1-10"
    range_match = re.search(r"(\d+)\s*(?:-|-|-|to)\s*(\d+)", text, flags=re.IGNORECASE)
    if range_match:
        start, end = int(range_match.group(1)), int(range_match.group(2))
        if start == end:
            return f"Chapter {start}"
        return f"Chapters {start}-{end}"
    single_match = re.search(r"chapter\s*(\d+)", text, flags=re.IGNORECASE)
    if single_match:
        return f"Chapter {int(single_match.group(1))}"
    bare_num = re.search(r"\b(\d+)\b", text)
    if bare_num:
        return f"Chapter {int(bare_num.group(1))}"
    # No number? Strip dash separators and return as-is.
    return text.replace("-", "-").replace("-", "-").strip()


@dataclass
class BundleResult:
    """What the runner returns once the bundle is written to disk."""
    title: str
    description: str
    title_variants: list[str]
    thumbnail_source_panel_id: str | None
    thumbnail_source_path: str
    thumbnail_path: str
    bundle_dir: str


# ── The bundle generator ─────────────────────────────────────────────────

class YouTubeBundleService:
    """Build the YouTube publish bundle for one project."""

    def __init__(self, settings: Any | None = None) -> None:
        self.settings = settings or get_settings()
        self._model = self._init_gemini()

    def _init_gemini(self):
        if not _GEMINI_AVAILABLE or not self.settings.gemini_api_key:
            return None
        try:
            genai.configure(api_key=self.settings.gemini_api_key)
            preferred = (self.settings.gemini_model or "gemini-2.5-flash").strip()
            if preferred in {"gemini-2.0-flash", "gemini-2.0-flash-exp"}:
                preferred = "gemini-2.5-flash"
            return genai.GenerativeModel(preferred)
        except Exception as exc:  # noqa: BLE001
            logger.warning("YouTubeBundleService Gemini init failed: %s", exc)
            return None

    # ── Public API ────────────────────────────────────────────────────────

    def build(
        self,
        project_dir: Path,
        *,
        project_name: str,
        chapter_title: str | None,
        manga_title: str | None,
        panels_json: list[dict[str, Any]],
        script_lines: list[str],
        audio_manifest: dict[str, Any] | None = None,
        voice_config: Any = None,
        video_config: Any = None,
        main_video_path: Path | None = None,
        progress_callback=None,
    ) -> BundleResult:
        """Generate the full bundle. Returns paths + metadata.

        New optional inputs power the YouTuber-grade additions:
          • audio_manifest - used to compute chapter timestamps
          • voice_config + video_config - used by the finishing renderer
          • main_video_path - points at the rendered final.mp4 that we
            prepend the cold-open / append the outro to.
        Each is optional so legacy callers without finishing still work.
        """
        from app.services.channel_preset_service import ChannelPresetService

        bundle_dir = project_dir / "youtube_bundle"
        bundle_dir.mkdir(parents=True, exist_ok=True)

        preset = ChannelPresetService(self.settings).load()

        if progress_callback:
            progress_callback(8, "Selecting the best panel for your thumbnail")
        thumb_panel = self._select_thumbnail_panel(panels_json, script_lines)
        if thumb_panel is None:
            raise RuntimeError("No kept panels available to use as a thumbnail.")

        thumb_source_path = self._copy_thumbnail_source(
            project_dir, thumb_panel, bundle_dir,
        )

        if progress_callback:
            progress_callback(20, "Drafting your title and description")
        title, variants, description = self._generate_text_metadata(
            project_name=project_name,
            chapter_title=chapter_title,
            manga_title=manga_title,
            script_lines=script_lines,
            thumb_panel_narration=self._panel_narration(thumb_panel, script_lines, panels_json),
        )

        # ── Cold open + chapter markers planning ──────────────────────
        # Only attempt these if we have an audio manifest. The finishing
        # planner needs per-panel durations to place chapter markers.
        chapter_markers: list[Any] = []
        cold_open_plan = None
        offset_seconds = 0.0
        if audio_manifest is not None:
            try:
                from app.services.video_finishing_service import (
                    VideoFinishingService,
                    format_chapter_markers_for_description,
                )
                if progress_callback:
                    progress_callback(35, "Planning cold open + chapter markers")
                finishing_plan = VideoFinishingService(self.settings).plan(
                    panels_json=panels_json,
                    script_lines=script_lines,
                    audio_manifest=audio_manifest,
                    preset=preset,
                    project_dir=project_dir,
                    manga_title=manga_title,
                    chapter_label=_normalize_chapter_label(chapter_title),
                )
                cold_open_plan = finishing_plan.cold_open
                chapter_markers = finishing_plan.chapter_markers
                if preset.cold_open_enabled and cold_open_plan is not None:
                    offset_seconds += float(cold_open_plan.hold_seconds)
                if preset.title_card_enabled:
                    offset_seconds += float(preset.title_card_duration_seconds)
                # NOTE: chapter markers are still computed (used for video
                # bookmarks + future Shorts splitting) but we no longer
                # inject a "## Chapters" timestamp block into the
                # description. The previous output bloated the description
                # with low-quality auto-labels like "Hiro Covers Mouth" at
                # 13:54 that read like a slow recap rather than a hook.
                # Users who want timestamps can re-add them by hand from
                # the bundle manifest in the publish studio.
            except Exception as exc:  # noqa: BLE001
                logger.warning("Cold-open / chapter-marker planning failed: %s", exc)

        if progress_callback:
            progress_callback(55, "Painting the viral thumbnail")
        # Default thumbnail overlay text is the manga name. Short,
        # recognizable, looks bold on the panel. User can edit per-variant
        # from the publish studio if they want.
        default_overlay = (manga_title or self._shorten_for_overlay(title)).strip()
        thumbnail_path = self._compose_thumbnail(
            base_image=thumb_source_path,
            title=title,
            output_path=bundle_dir / "thumbnail.png",
            preset=preset,
            overlay_text=default_overlay,
        )

        # ── 5 thumbnail variants ──────────────────────────────────────
        # The user picks which one to ship from the publish studio UI.
        # Variant 0 is the canonical pick (same as thumbnail.png above);
        # variants 1-4 are progressively-lower-scoring candidates with a
        # 5%-of-chapter spread so they aren't five frames of the same
        # shot. Each gets its own thumbnail PNG so the carousel can show
        # them inline.
        thumbnail_variants: list[dict[str, str]] = []
        top_panels = self._select_top_thumbnail_panels(panels_json, script_lines, n=5)
        # Ensure the canonical pick is variant 0 even if scoring jitter
        # would have placed it elsewhere.
        canonical_id = str(thumb_panel.get("id"))
        top_panels = [p for p in top_panels if str(p.get("id")) != canonical_id]
        top_panels.insert(0, thumb_panel)
        top_panels = top_panels[:5]

        variants_dir = bundle_dir / "variants"
        variants_dir.mkdir(parents=True, exist_ok=True)
        variant_labels = ["Top pick", "Climax shot", "Character beat", "Stakes shot", "Quiet beat"]
        for v_idx, v_panel in enumerate(top_panels):
            try:
                v_source = self._copy_thumbnail_source(
                    project_dir, v_panel, variants_dir, suffix=f"_v{v_idx}",
                )
                v_path = self._compose_thumbnail(
                    base_image=v_source,
                    title=title,
                    output_path=variants_dir / f"variant_{v_idx}.png",
                    preset=preset,
                    overlay_text=default_overlay,
                )
                thumbnail_variants.append({
                    "style_id": f"v{v_idx}",
                    "style_label": variant_labels[v_idx] if v_idx < len(variant_labels) else f"Variant {v_idx + 1}",
                    "path": str(v_path.relative_to(project_dir)),
                    "source_panel_id": str(v_panel.get("id") or ""),
                    # Default overlay text is just the manga name. It's
                    # short, instantly recognizable, and lets the user
                    # type a custom hook per-variant if they want.
                    # Falls back to a shortened title when manga_title
                    # isn't set.
                    "overlay_text": (manga_title or self._shorten_for_overlay(title)).strip(),
                })
            except Exception as exc:  # noqa: BLE001
                logger.warning("Thumbnail variant %d failed: %s", v_idx, exc)

        # ── Finishing render: cold-open + outro stitched onto main video
        publish_video_path: Path | None = None
        if (
            main_video_path is not None
            and main_video_path.exists()
            and voice_config is not None
            and video_config is not None
            and (preset.cold_open_enabled or preset.outro_enabled)
        ):
            try:
                from app.services.video_finishing_renderer import VideoFinishingRenderer
                if progress_callback:
                    progress_callback(70, "Adding cold open + outro to the final video")
                publish_video_path = VideoFinishingRenderer(self.settings).finalize(
                    project_dir=project_dir,
                    main_video_path=main_video_path,
                    cold_open_plan=cold_open_plan,
                    preset=preset,
                    video_config=video_config,
                    voice_config=voice_config,
                    project_name=project_name,
                )
                # Register final_publish.mp4 in the video manifest so the
                # preview page picks it up as the "latest video". Without
                # this, the manifest only knows about final_music.mp4
                # (the pre-finishing output) and the user sees the
                # version without the cold-open / title-card / outro.
                if publish_video_path is not None and publish_video_path.exists():
                    self._register_publish_video(
                        project_dir=project_dir,
                        publish_path=publish_video_path,
                        video_config=video_config,
                    )
            except Exception as exc:  # noqa: BLE001
                logger.warning("Finishing render failed: %s", exc)

        # ── Shorts auto-cut ───────────────────────────────────────────
        short_meta: dict[str, Any] | None = None
        short_thumbnail_variants: list[dict[str, str]] = []
        if audio_manifest is not None and voice_config is not None:
            try:
                from app.services.shorts_service import ShortsService
                if progress_callback:
                    progress_callback(82, "Cutting a 60-second Shorts version")
                short_meta = ShortsService(self.settings).build(
                    project_dir=project_dir,
                    panels_json=panels_json,
                    script_lines=script_lines,
                    audio_manifest=audio_manifest,
                    voice_config=voice_config,
                    manga_title=manga_title,
                    chapter_title=chapter_title,
                    preset=preset,
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("Shorts render failed: %s", exc)

            # ── 5 Shorts cover thumbnail variants (vertical 1080x1920) ──
            # Same scoring + spread rules as the main thumbnail, but
            # rendered at vertical aspect. The user can swap which one
            # they upload as the Shorts cover image.
            try:
                if progress_callback:
                    progress_callback(88, "Painting Shorts cover variants")
                short_variants_dir = bundle_dir / "short_variants"
                short_variants_dir.mkdir(parents=True, exist_ok=True)
                short_top = self._select_top_thumbnail_panels(panels_json, script_lines, n=5)
                short_variant_labels = ["Top pick", "Climax shot", "Character beat", "Stakes shot", "Quiet beat"]
                for s_idx, s_panel in enumerate(short_top):
                    try:
                        s_source = self._copy_thumbnail_source(
                            project_dir, s_panel, short_variants_dir, suffix=f"_v{s_idx}",
                        )
                        s_path = self._compose_thumbnail(
                            base_image=s_source,
                            title=title,
                            output_path=short_variants_dir / f"variant_{s_idx}.png",
                            preset=preset,
                            target_size=(1080, 1920),
                            overlay_text=default_overlay,
                        )
                        short_thumbnail_variants.append({
                            "style_id": f"sv{s_idx}",
                            "style_label": short_variant_labels[s_idx] if s_idx < len(short_variant_labels) else f"Variant {s_idx + 1}",
                            "path": str(s_path.relative_to(project_dir)),
                            "source_panel_id": str(s_panel.get("id") or ""),
                            "overlay_text": default_overlay,
                        })
                    except Exception as exc:  # noqa: BLE001
                        logger.warning("Short thumbnail variant %d failed: %s", s_idx, exc)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Shorts thumbnail generation failed: %s", exc)

        # Persist text + manifest
        (bundle_dir / "title.txt").write_text(title.strip() + "\n", encoding="utf-8")
        (bundle_dir / "title_variants.json").write_text(
            json.dumps(variants, indent=2), encoding="utf-8",
        )
        (bundle_dir / "description.md").write_text(description.strip() + "\n", encoding="utf-8")

        manifest = {
            "version": "youtube_bundle_v2",
            "title": title,
            "title_variants": variants,
            "description": description,
            "thumbnail_source_panel_id": thumb_panel.get("id"),
            "thumbnail_source_path": str(thumb_source_path.relative_to(project_dir)),
            "thumbnail_path": str((bundle_dir / "thumbnail.png").relative_to(project_dir)),
            "thumbnail_variants": thumbnail_variants,
            "chosen_thumbnail_index": 0,
            "bundle_dir": str(bundle_dir.relative_to(project_dir)),
            "chapter_markers": [
                {"timecode_seconds": m.timecode_seconds, "label": m.label}
                for m in chapter_markers
            ],
            "publish_video_path": (
                str(publish_video_path.relative_to(project_dir))
                if publish_video_path else None
            ),
            "short": short_meta,
            "short_thumbnail_variants": short_thumbnail_variants,
            "short_chosen_thumbnail_index": 0 if short_thumbnail_variants else None,
            "short_thumbnail_path": (
                short_thumbnail_variants[0]["path"]
                if short_thumbnail_variants else None
            ),
            "channel_preset": preset.to_dict(),
        }
        write_json(bundle_dir / "manifest.json", manifest)

        # Auto-sync the chosen Shorts cover into the project's
        # `thumbnails/video_intro.jpg` so the video lead-in card uses
        # the same visual the user is shipping to YouTube as their
        # Shorts cover. Skipped if the user has uploaded a custom
        # thumbnail (marker file present) - their explicit choice wins.
        try:
            from app.services.project_store import ProjectStore
            project_id_for_sync = project_dir.name
            if short_thumbnail_variants:
                chosen_idx = int(manifest.get("short_chosen_thumbnail_index") or 0)
                if 0 <= chosen_idx < len(short_thumbnail_variants):
                    chosen_rel = short_thumbnail_variants[chosen_idx].get("path")
                    if chosen_rel:
                        ProjectStore().sync_video_thumbnail_from_short_cover(
                            project_id_for_sync,
                            project_dir / chosen_rel,
                        )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Auto-sync video intro thumbnail failed: %s", exc)

        if progress_callback:
            progress_callback(100, "Bundle is ready")

        return BundleResult(
            title=title,
            description=description,
            title_variants=variants,
            thumbnail_source_panel_id=thumb_panel.get("id"),
            thumbnail_source_path=str(thumb_source_path),
            thumbnail_path=str(thumbnail_path),
            bundle_dir=str(bundle_dir),
        )

    def _register_publish_video(
        self,
        *,
        project_dir: Path,
        publish_path: Path,
        video_config: Any,
    ) -> None:
        """Add final_publish.mp4 to the project's video manifest.

        The manifest is what `list_videos` (and the preview page's
        "latest video" picker) reads, so we need to update it with the
        finishing-rendered file or the preview will keep showing the
        pre-cold-open version.
        """
        from datetime import datetime

        video_dir = publish_path.parent
        manifest_path = video_dir / "manifest.json"
        try:
            existing = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
        except Exception:  # noqa: BLE001
            existing = {}
        if not isinstance(existing, dict):
            existing = {}

        # Probe duration so the preview header can label it correctly.
        duration_seconds: float | None = None
        try:
            import subprocess
            result = subprocess.run(
                [
                    self.settings.ffprobe_binary if hasattr(self.settings, "ffprobe_binary") else "ffprobe",
                    "-v", "error",
                    "-show_entries", "format=duration",
                    "-of", "default=noprint_wrappers=1:nokey=1",
                    str(publish_path),
                ],
                capture_output=True, text=True, check=False, timeout=10,
            )
            if result.returncode == 0:
                duration_seconds = round(float((result.stdout or "0").strip()), 2)
        except Exception:  # noqa: BLE001
            pass

        existing[publish_path.name] = {
            "width": int(getattr(video_config, "width", 1920) or 1920),
            "height": int(getattr(video_config, "height", 1080) or 1080),
            "output_format": publish_path.suffix.lstrip(".") or "mp4",
            "created_at": datetime.utcnow().isoformat(),
            "duration_seconds": duration_seconds,
            "kind": "publish",  # Marker so list_videos can prioritize this one.
        }
        manifest_path.write_text(json.dumps(existing, indent=2), encoding="utf-8")

    def _inject_chapter_timestamps(
        self,
        description: str,
        markers: list[Any],
        offset_seconds: float,
    ) -> str:
        """Append the YouTube chapter-timestamps block to a description.

        We slot it just after the first paragraph (the hook) and before
        any bullet list / hashtag block - that's where every channel
        with chapters puts theirs."""
        from app.services.video_finishing_service import (
            format_chapter_markers_for_description,
        )
        block = format_chapter_markers_for_description(
            markers, cold_open_offset_seconds=offset_seconds,
        )
        if not block.strip():
            return description
        timestamps_section = "## Chapters\n" + block
        lines = description.splitlines()
        # Find the first blank line after the start; insert there.
        for i, line in enumerate(lines):
            if i > 2 and not line.strip():
                lines.insert(i + 1, timestamps_section)
                lines.insert(i + 2, "")
                return "\n".join(lines)
        return description.rstrip() + "\n\n" + timestamps_section + "\n"

    # ── Best-panel selection ─────────────────────────────────────────────

    def _select_thumbnail_panel(
        self,
        panels_json: list[dict[str, Any]],
        script_lines: list[str],
    ) -> dict[str, Any] | None:
        kept = [p for p in panels_json if p.get("keep")]
        if not kept:
            return None
        # YouTube punishes the WHOLE video for a single demonetizing
        # thumbnail. Strictly exclude any panel flagged nsfw_* (borderline
        # or explicit) or marked for blur - the thumbnail is the one
        # surface that must be safe.
        def _is_safe_for_thumbnail(panel: dict[str, Any]) -> bool:
            if panel.get("content_blur"):
                return False
            rating = (panel.get("content_rating") or "").lower()
            if rating in {"borderline", "explicit"}:
                return False
            for flag in panel.get("review_flags") or []:
                if isinstance(flag, str) and flag.startswith("nsfw_"):
                    return False
            return True

        safe_kept = [p for p in kept if _is_safe_for_thumbnail(p)]
        # If somehow every kept panel is flagged (very unusual), fall back
        # to the full kept list rather than crashing - the panel will still
        # render through the blurred crop pipeline.
        if not safe_kept:
            safe_kept = kept
        kept_sorted = sorted(
            safe_kept,
            key=lambda p: (int(p.get("page", 0)), int(p.get("panel", 0))),
        )

        # Build "kept-index → narration text" lookup using script_lines order.
        narration_by_id: dict[str, str] = {}
        for i, panel in enumerate(kept_sorted):
            narr = (panel.get("narration") or "").strip()
            if not narr and i < len(script_lines):
                narr = (script_lines[i] or "").strip()
            narration_by_id[str(panel.get("id"))] = narr

        total = len(kept_sorted)
        scores: list[tuple[float, dict[str, Any]]] = []
        skip_first = max(1, int(total * 0.08))

        for idx, panel in enumerate(kept_sorted):
            if idx < skip_first:
                continue
            score = 0.0

            # Keyword affinity
            narr = narration_by_id.get(str(panel.get("id")), "").lower()
            for keyword in _THUMBNAIL_KEYWORDS:
                if keyword in narr:
                    score += 4.0
                    break

            # Larger area = more thumbnail surface
            try:
                w = float(panel.get("width") or 0)
                h = float(panel.get("height") or 0)
                area_ratio = (w * h) / 4_000_000.0  # normalize to ~1 for big panels
                score += min(area_ratio * 2.0, 3.0)
            except (TypeError, ValueError):
                pass

            # Bias toward 60-80% through the chapter (climax region)
            t = idx / max(1, total - 1)
            climax_score = max(0.0, 1.0 - abs(t - 0.72) * 3.0)
            score += climax_score * 2.0

            # Slight penalty for very short narration (often beat/transition)
            if narr and len(narr.split()) >= 12:
                score += 1.0

            scores.append((score, panel))

        if not scores:
            # Fallback: middle-ish kept panel
            return kept_sorted[len(kept_sorted) // 2]

        scores.sort(key=lambda s: s[0], reverse=True)
        return scores[0][1]

    def _select_top_thumbnail_panels(
        self,
        panels_json: list[dict[str, Any]],
        script_lines: list[str],
        n: int = 5,
    ) -> list[dict[str, Any]]:
        """Return the top-N candidate panels for thumbnail variants.

        Same scoring as `_select_thumbnail_panel` but returns the top N
        instead of just the winner. We also enforce a "spread" rule so
        the variants aren't five near-adjacent panels of the same shot.
        Picked panels must be at least 5% of the chapter apart by index.
        """
        # We have to recompute the scored list because the single-pick
        # helper doesn't expose it. Pulling that out into a helper would
        # mean a wider refactor; copying the scoring loop is intentionally
        # contained.
        kept = [p for p in panels_json if p.get("keep")]
        if not kept:
            return []

        def _is_safe(panel: dict[str, Any]) -> bool:
            if panel.get("content_blur"):
                return False
            rating = (panel.get("content_rating") or "").lower()
            if rating in {"borderline", "explicit"}:
                return False
            for flag in panel.get("review_flags") or []:
                if isinstance(flag, str) and flag.startswith("nsfw_"):
                    return False
            return True

        safe_kept = [p for p in kept if _is_safe(p)] or kept
        kept_sorted = sorted(safe_kept, key=lambda p: (int(p.get("page", 0)), int(p.get("panel", 0))))
        total = len(kept_sorted)
        if total == 0:
            return []
        skip_first = max(1, int(total * 0.08))

        narration_by_id: dict[str, str] = {}
        for i, panel in enumerate(kept_sorted):
            narr = (panel.get("narration") or "").strip()
            if not narr and i < len(script_lines):
                narr = (script_lines[i] or "").strip()
            narration_by_id[str(panel.get("id"))] = narr

        scored: list[tuple[float, int, dict[str, Any]]] = []
        for idx, panel in enumerate(kept_sorted):
            if idx < skip_first:
                continue
            score = 0.0
            narr = narration_by_id.get(str(panel.get("id")), "").lower()
            for keyword in _THUMBNAIL_KEYWORDS:
                if keyword in narr:
                    score += 4.0
                    break
            try:
                w = float(panel.get("width") or 0)
                h = float(panel.get("height") or 0)
                score += min((w * h) / 4_000_000.0 * 2.0, 3.0)
            except (TypeError, ValueError):
                pass
            t = idx / max(1, total - 1)
            score += max(0.0, 1.0 - abs(t - 0.72) * 3.0) * 2.0
            if narr and len(narr.split()) >= 12:
                score += 1.0
            scored.append((score, idx, panel))

        if not scored:
            return [kept_sorted[len(kept_sorted) // 2]]

        # Greedy pick: take highest score, then enforce a spread so the
        # next pick is at least `min_gap` panels away in the chapter order.
        scored.sort(key=lambda s: s[0], reverse=True)
        min_gap = max(1, int(total * 0.05))
        picked: list[tuple[int, dict[str, Any]]] = []
        for _, idx, panel in scored:
            if any(abs(idx - prev_idx) < min_gap for prev_idx, _ in picked):
                continue
            picked.append((idx, panel))
            if len(picked) >= n:
                break

        # If the spread rule starved us (very short chapter), top up from
        # the next-best scores without the gap constraint.
        if len(picked) < n:
            seen_ids = {id(p) for _, p in picked}
            for _, idx, panel in scored:
                if id(panel) in seen_ids:
                    continue
                picked.append((idx, panel))
                if len(picked) >= n:
                    break

        return [panel for _, panel in picked]

    def _copy_thumbnail_source(
        self,
        project_dir: Path,
        panel: dict[str, Any],
        bundle_dir: Path,
        suffix: str = "",
    ) -> Path:
        order = int(panel.get("order", 0))
        panel_path = project_dir / "panels" / f"panel_{order:03d}.png"
        if not panel_path.exists():
            for ext in ("jpg", "jpeg", "webp"):
                alt = project_dir / "panels" / f"panel_{order:03d}.{ext}"
                if alt.exists():
                    panel_path = alt
                    break
        if not panel_path.exists():
            raise FileNotFoundError(
                f"Could not find panel image at {panel_path} for thumbnail."
            )
        dest = bundle_dir / f"thumbnail_source{suffix}.png"
        # Re-save as PNG with the panel's pristine pixels.
        with Image.open(panel_path) as im:
            im.convert("RGB").save(dest, "PNG", optimize=True)
        return dest

    def _panel_narration(
        self,
        panel: dict[str, Any],
        script_lines: list[str],
        panels_json: list[dict[str, Any]],
    ) -> str:
        kept = sorted(
            [p for p in panels_json if p.get("keep")],
            key=lambda p: (int(p.get("page", 0)), int(p.get("panel", 0))),
        )
        for i, p in enumerate(kept):
            if p.get("id") == panel.get("id"):
                if (panel.get("narration") or "").strip():
                    return str(panel["narration"]).strip()
                if i < len(script_lines):
                    return (script_lines[i] or "").strip()
        return (panel.get("narration") or "").strip()

    # ── Title + description generation ───────────────────────────────────

    def _generate_text_metadata(
        self,
        *,
        project_name: str,
        chapter_title: str | None,
        manga_title: str | None,
        script_lines: list[str],
        thumb_panel_narration: str,
    ) -> tuple[str, list[str], str]:
        # Stitch the first N narrations together for context (most chapters'
        # essential setup is in the opening 30-40% of panels).
        clean = [line.strip() for line in script_lines if line and line.strip()]
        opener = " ".join(clean[: min(40, len(clean))])
        closer = " ".join(clean[-min(20, len(clean)):])

        if self._model is not None:
            try:
                title, variants, description = self._llm_metadata(
                    project_name=project_name,
                    chapter_title=chapter_title,
                    manga_title=manga_title,
                    opener=opener,
                    closer=closer,
                    thumb_narration=thumb_panel_narration,
                )
                return title, variants, description
            except Exception as exc:  # noqa: BLE001
                logger.warning("YouTubeBundleService LLM metadata failed: %s", exc)

        # Heuristic fallback - always usable even without an API key.
        return self._fallback_metadata(
            project_name=project_name,
            chapter_title=chapter_title,
            manga_title=manga_title,
            opener=opener,
        )

    def _llm_metadata(
        self,
        *,
        project_name: str,
        chapter_title: str | None,
        manga_title: str | None,
        opener: str,
        closer: str,
        thumb_narration: str,
    ) -> tuple[str, list[str], str]:
        series = manga_title or project_name or "this chapter"
        chapter = chapter_title or "this chapter"
        # The recap script we pass in is per-panel narration. The model
        # tends to echo that verbatim into the description, which produces
        # a slow Wikipedia-style synopsis ("The world is revealed as a
        # barren wasteland..."). We pass it in only as background context
        # and explicitly forbid the model from quoting or paraphrasing
        # any single sentence of it.
        chapter_label = _normalize_chapter_label(chapter_title) or "this chapter"
        prompt = f"""You are writing YouTube metadata for a manga / manhwa / comic
recap video. You have 10 years of experience as a manga-narration YouTuber
with 1.2M subscribers. You know exactly which titles get clicked on the
"recommended" sidebar and which ones get scrolled past in a tenth of a
second. Your goal is click-through, retention, and comments, in that order.

Series: {series}
Chapter focus: {chapter_label}
Most striking panel in the chapter: "{thumb_narration}"

Background context, for understanding only. Do NOT quote, paraphrase, or
echo any sentence from this block. It is what we narrate in the video,
not what we sell in the description:
---
{opener[:1500]}
...
{closer[:600]}
---

OUTPUT FORMAT:

Return a JSON object with these three keys:

- "title": The MAIN title. This is a HOOK, not a label. Pick from the
  patterns that consistently outperform on YouTube manga channels:
    a. Specific intriguing statement that implies a story:
       "She Said She'd Kill Him. He Begged to Pilot Her Anyway."
       "The Strongest Pilot in the Plantation is a Failure. Until Her."
       "Everyone Else Stayed in Their Lane. This Boy Walked Up to a Monster."
    b. Specific question that creates a knowledge gap:
       "Why Does Zero Two Have Horns?"
       "What Is the Plantation Actually Hiding from Its Pilots?"
    c. Big-stakes statement:
       "How One Pink-Haired Pilot Restarted the Mech War"
  Rules: 50-95 chars. The series name is OPTIONAL in the title - if you
  include it, parenthesize it at the end like "... (DARLING in the FRANXX)".
  NEVER use the pattern "Series Name - Chapter X" (that reads as a
  channel-naming-convention label, not a hook). NEVER use "Recap" or
  "Explained" as the suffix. NO emoji. NO ALL CAPS for emphasis (the
  series name's natural casing is fine, e.g. "DARLING in the FRANXX").
  NEVER use em dashes or en dashes - use periods, hyphens, or commas.

- "variants": three alternative titles testing three different angles
  (NOT three rephrasings of the same idea). Mix patterns from above.
  Each 50-95 chars. Same casing/dash/emoji rules as the main title.

- "description": exactly this structure, plain text, NO markdown, NO
  bullet points, NO chapter timestamps, NO "## What happens", NO
  "in this video":

  LINE 1: A single-sentence hook that EARNS the click. Under 110 chars.
          Treat it like ad copy: one job, make the viewer hit "Show more".
          GOOD: "Hiro never wanted to pilot a Franxx. Then she walked in."
          GOOD: "A boy with no future meets a girl who eats them."
          BAD:  "The world is a barren wasteland scarred by humanity's
                 extraction of magma energy" (Wikipedia voice).
          BAD:  "Every key moment of [series], in story order" (template
                 filler).
  (blank line)
  LINES 3-5: A 2-3 sentence story synopsis. The elevator pitch for THIS
             chapter / chapter range: name the protagonist if it sells,
             set the central conflict, state what's at stake. Reads like
             a back-cover blurb. NO scene-by-scene beats. NO climax
             spoiler. Do NOT echo the panel narration from context.
  (blank line)
  LINE 7: One sentence subscribe CTA in your own voice. Examples:
          "Subscribe so the next chapter hits your feed the day it drops."
          "If chapter recaps are your thing, hit subscribe."
  (blank line)
  LINE 9: Hashtag block, space-separated, 6-10 tags. Include #manga
          #anime #mangarecap plus 3-7 SERIES-SPECIFIC tags:
            - The series name as one word (#darlinginthefranxx)
            - 1-2 main character first names (#zerotwo #hiro)
            - 1-2 genre tags (#mechaanime #sciFiAnime)
            - 1 thematic tag (#starcrossedlovers)
          One word each, no spaces, no apostrophes.

ABSOLUTE RULES:
- Never use an em dash (-) or en dash (-) anywhere. Use a hyphen, period,
  or comma instead.
- Never reproduce a sentence from the background context.
- Never use ALL CAPS for emphasis (except in tags or the series name if
  the canonical title is styled that way, e.g. "DARLING in the FRANXX").
- Total description under 1200 characters.

Return ONLY the JSON. No prose before or after, no markdown fences."""
        # Disable Gemini 2.5 "thinking" budget so the whole token allowance
        # goes to the visible answer - otherwise the JSON gets truncated
        # mid-string and parsing fails.
        gen_kwargs: dict[str, Any] = {
            "temperature": 0.7,
            "top_p": 0.9,
            "max_output_tokens": 4096,
        }
        try:
            from google.generativeai.types import ThinkingConfig  # type: ignore
            gen_kwargs["thinking_config"] = ThinkingConfig(thinking_budget=0)
        except Exception:
            pass

        response = self._model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(**gen_kwargs),
        )
        text = getattr(response, "text", "") or ""
        text = text.strip()
        # Strip code fences if present.
        if text.startswith("```"):
            text = re.sub(r"^```[a-zA-Z]*", "", text).strip()
            text = re.sub(r"```$", "", text).strip()
        # Try to extract the JSON object even if the model surrounded it
        # with prose.
        if not text.startswith("{"):
            m = re.search(r"\{[\s\S]*\}", text)
            if m:
                text = m.group(0)
        data = json.loads(text)
        title = str(data.get("title") or "").strip()
        variants = [str(v).strip() for v in (data.get("variants") or []) if str(v).strip()]
        description = str(data.get("description") or "").strip()
        if not title:
            raise ValueError("LLM returned empty title")

        # Reject template-y titles that ignore the "use a hook" instruction.
        # The model sometimes falls back to "Series - Chapters X-Y" form
        # despite the prompt rules; that label is what we used to ship as
        # the default, and it gets near-zero CTR on YouTube. If the LLM
        # produced one of those, raise so the caller falls back to the
        # heuristic synthesizer (which now also writes a hook).
        bland_patterns = (
            r"^[^-]+ - chapters?\s*\d",          # "Foo - Chapters 1-10"
            r"^[^-]+ - chapter\s*\d",            # "Foo - Chapter 27"
            r"^[^-]+ explained$",                 # "Foo explained"
            r"^everything that happened in ",     # "Everything that happened in..."
            r"^[^-]+ recap$",                     # "Foo recap"
        )
        title_lower = title.casefold()
        if any(re.match(p, title_lower) for p in bland_patterns):
            raise ValueError(
                f"LLM returned a template-y title ({title!r}); "
                "falling back to heuristic synthesizer so we never ship "
                "a 'Series - Chapter X' as the click target."
            )

        return title, variants[:3], description

    def _fallback_metadata(
        self,
        *,
        project_name: str,
        chapter_title: str | None,
        manga_title: str | None,
        opener: str,
    ) -> tuple[str, list[str], str]:
        series = (manga_title or project_name or "").strip()
        chapter = _normalize_chapter_label(chapter_title)
        base = series or "Manga"

        # Heuristic title fallback (used when LLM is unavailable or its
        # output was rejected as template-y). Even in fallback we avoid
        # the dead "Series - Chapter X" pattern - we synthesize a generic
        # hook flavored by the chapter label so the click-target reads
        # like ad copy, not a filing system. The series name is in the
        # parens so YouTube's search still indexes it.
        if chapter and base:
            chapter_intro_phrases = [
                f"The Chapter That Restarts Everything ({base})",
                f"What Actually Happens When the Story Begins ({base})",
                f"Why You Should Care About {base} (start here)",
            ]
        elif base:
            chapter_intro_phrases = [
                f"The {base} Story, From The Beginning",
                f"Why {base} Is Worth Your Hour",
                f"What Makes {base} Different",
            ]
        else:
            chapter_intro_phrases = [
                "The Story That Hooks You In One Chapter",
                "An Hour Of Story In One Sitting",
                "What You'll Wish You'd Read Sooner",
            ]
        title = chapter_intro_phrases[0][:TITLE_CHAR_BUDGET]
        variants = [phrase[:TITLE_CHAR_BUDGET] for phrase in chapter_intro_phrases]
        # Fallback description used when no LLM is configured. We
        # deliberately do NOT echo per-panel narration. Instead, we
        # extract a short setting/synopsis from the opener block using
        # a light scrubber that strips per-panel directorial language
        # ("A close-up of...", "The camera pans...") and keeps the
        # first 1-2 plot sentences.
        synopsis_preview = self._extract_short_synopsis(opener, base=base) if opener else ""
        if synopsis_preview:
            first_sentence = synopsis_preview.split('.')[0].strip()
            hook = (first_sentence + '.')[:110] if first_sentence else f"Inside {base or 'this chapter'} - the story, not the setup."
        else:
            hook = f"Inside {base or 'this chapter'} - the story, not the setup."[:110]

        synopsis = self._extract_short_synopsis(opener, base=base) if opener else ""
        if not synopsis:
            synopsis = (
                f"Follow {base} through the moments that actually move the story, "
                "not the filler between them. Recap pace, no spoilers in the title."
            )

        cta = "Subscribe so the next chapter recap hits your feed the day it drops."
        series_tag = re.sub(r"[^a-z0-9]", "", (manga_title or project_name or "manga").lower()) or "manga"
        hashtag_line = f"#manga #anime #mangarecap #{series_tag} #manhwa"
        # Strip trailing period from hook so the join below doesn't double it.
        hook_clean = hook.rstrip(".").rstrip()
        # Structure: hook line, blank, synopsis, blank, CTA, blank, hashtags.
        description = f"{hook_clean}.\n\n{synopsis}\n\n{cta}\n\n{hashtag_line}"
        return title, variants, description

    @staticmethod
    def _extract_short_synopsis(opener: str, *, base: str = "") -> str:
        """Pick a 1-2 sentence synopsis-style line from the opener.

        Filters out per-panel directorial language (camera moves,
        close-ups, "A panel shows..."). Keeps the first sentence(s)
        that read like plot text. Caps at ~280 chars so the description
        stays scannable.
        """
        # Sentences that scream "panel narration":
        bad_starts = (
            "a close-up", "close-up", "the camera", "a panel", "this panel",
            "an overhead", "a side view", "a wide shot", "a medium shot",
            "the panel", "in the panel", "we see ", "a shot of",
        )
        sentences = re.split(r"(?<=[.!?])\s+", opener.strip())
        keepers: list[str] = []
        for sentence in sentences:
            s = sentence.strip()
            if not s or len(s) < 12:
                continue
            lower = s.lower()
            if any(lower.startswith(prefix) for prefix in bad_starts):
                continue
            keepers.append(s)
            if sum(len(x) for x in keepers) >= 240:
                break
            if len(keepers) >= 2:
                break
        if not keepers:
            return ""
        synopsis = " ".join(keepers)
        if len(synopsis) > 280:
            synopsis = synopsis[:277].rstrip(", ;-") + "..."
        return synopsis

    # ── Public: regenerate a single variant's overlay text ──────────────

    def regenerate_variant_thumbnail(
        self,
        *,
        project_dir: Path,
        variant_index: int,
        overlay_text: str,
        title_for_fallback: str = "",
        group: str = "main",
    ) -> dict[str, Any]:
        """Rerender just one thumbnail variant with a new overlay text.

        `group` selects which set of variants to operate on:
          - "main"  : long-form YouTube thumbnails (1280x720)
          - "short" : vertical Shorts cover thumbnails (1080x1920)

        Reads the existing bundle manifest, locates the variant's source
        thumbnail PNG (a copy of the original panel image, before any
        styling), reruns `_compose_thumbnail` on it with the new text,
        and writes the result back to the same path so URLs stay stable.
        Returns the updated variant dict so the caller can echo it to
        the client.
        """
        from app.services.channel_preset_service import ChannelPresetService

        if group not in {"main", "short"}:
            raise ValueError(f"Unknown thumbnail group: {group!r}")
        manifest_key = "thumbnail_variants" if group == "main" else "short_thumbnail_variants"
        chosen_key = "chosen_thumbnail_index" if group == "main" else "short_chosen_thumbnail_index"
        canonical_key = "thumbnail_path" if group == "main" else "short_thumbnail_path"
        variants_subdir = "variants" if group == "main" else "short_variants"
        target_size = THUMBNAIL_SIZE if group == "main" else (1080, 1920)

        bundle_dir = project_dir / "youtube_bundle"
        manifest_path = bundle_dir / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError("No bundle manifest. Generate the bundle first.")

        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        variants = manifest.get(manifest_key) or []
        if not (0 <= variant_index < len(variants)):
            raise IndexError(f"variant_index {variant_index} is out of range (have {len(variants)} variants)")

        entry = dict(variants[variant_index]) if isinstance(variants[variant_index], dict) else {}
        rel_thumb = entry.get("path") or ""
        thumb_path = project_dir / rel_thumb
        source_candidates = [
            bundle_dir / variants_subdir / f"thumbnail_source_v{variant_index}.png",
            bundle_dir / f"thumbnail_source_v{variant_index}.png",
            bundle_dir / "thumbnail_source.png",
        ]
        source_path = next((p for p in source_candidates if p.exists()), None)
        if source_path is None:
            raise FileNotFoundError(
                f"No source image found for {group} variant {variant_index}. "
                f"Re-run the bundle stage to regenerate sources.",
            )

        thumb_path.parent.mkdir(parents=True, exist_ok=True)
        preset = ChannelPresetService(self.settings).load()
        self._compose_thumbnail(
            base_image=source_path,
            title=title_for_fallback or manifest.get("title") or "",
            output_path=thumb_path,
            preset=preset,
            overlay_text=overlay_text,
            target_size=target_size,
        )

        # Update manifest with the new overlay text so it round-trips.
        entry["overlay_text"] = overlay_text
        variants[variant_index] = entry
        manifest[manifest_key] = variants
        chosen_idx = int(manifest.get(chosen_key) or 0)
        if chosen_idx == variant_index:
            manifest[canonical_key] = rel_thumb
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        return entry

    # ── Thumbnail composition ────────────────────────────────────────────

    def _compose_thumbnail(
        self,
        *,
        base_image: Path,
        title: str,
        output_path: Path,
        preset: Any = None,
        overlay_text: str | None = None,
        target_size: tuple[int, int] | None = None,
    ) -> Path:
        """Render the final thumbnail with channel-preset branding.

        `overlay_text` is the bold text painted on top of the panel.
        If None, we derive it from `title` (the YouTube video title) by
        chopping it to fit. If a non-None string is passed (even empty),
        it overrides the derived value: the empty string suppresses the
        overlay entirely, anything else is used verbatim.

        `target_size` overrides the canvas dimensions. Defaults to the
        landscape YouTube thumbnail spec (1280x720). Pass (1080, 1920)
        to produce a vertical Shorts cover, etc.

        Preset is optional so legacy callers without a preset still work.
        """
        from app.services.channel_preset_service import ChannelPreset, ChannelPresetService
        if preset is None:
            try:
                preset = ChannelPresetService(self.settings).load()
            except Exception:
                preset = ChannelPreset()

        accent_rgb = self._hex_to_rgb(preset.accent_color, fallback=(127, 255, 212))
        # Watermark is intentionally NOT drawn on thumbnails any more -
        # YouTube's recommended-video grid renders thumbnails so small
        # (~320px wide on phones) that a corner watermark just becomes
        # noise. The channel handle now lives on the rendered video
        # instead via the per-video text overlay in video_service.
        canvas_size = target_size or THUMBNAIL_SIZE

        with Image.open(base_image) as src:
            src = src.convert("RGB")

            # Fit-cover into the chosen thumbnail aspect ratio.
            canvas = self._fit_cover(src, canvas_size)

            # Mild enhancement so the panel pops on YouTube's small previews.
            canvas = ImageEnhance.Contrast(canvas).enhance(1.08)
            canvas = ImageEnhance.Color(canvas).enhance(1.18)

            # Cinematic vignette
            canvas = self._apply_vignette(canvas)

            # Brand-accent corner glow - pulls from preset so every
            # channel feels distinct at thumbnail glance.
            self._apply_corner_glow(canvas, color=(*accent_rgb, 180))

            # Title overlay using preset accent for the highlight word.
            # Custom overlay text override path: caller can pass an
            # explicit override string per-variant (this is what the
            # publish studio uses when the user types into the
            # "Thumbnail text" input).
            if overlay_text is None:
                resolved_overlay = self._shorten_for_overlay(title)
            else:
                resolved_overlay = overlay_text.strip()
            if resolved_overlay:
                self._draw_thumbnail_text(canvas, resolved_overlay, accent_rgb=accent_rgb)

            canvas.save(output_path, "PNG", optimize=True)
        return output_path

    @staticmethod
    def _hex_to_rgb(hex_color: str, *, fallback: tuple[int, int, int]) -> tuple[int, int, int]:
        s = (hex_color or "").strip().lstrip("#")
        if len(s) == 8:
            s = s[:6]
        if len(s) != 6:
            return fallback
        try:
            return (int(s[0:2], 16), int(s[2:4], 16), int(s[4:6], 16))
        except ValueError:
            return fallback

    def _draw_watermark(self, image: Image.Image, text: str) -> None:
        w, h = image.size
        font = self._load_font(max(18, int(h * 0.028)))
        draw = ImageDraw.Draw(image)
        bbox = draw.textbbox((0, 0), text, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        margin = int(h * 0.025)
        x = w - tw - margin
        y = h - th - margin
        # subtle drop shadow + soft fill so the watermark sits without
        # competing with the title text.
        self._draw_stroked_word(
            draw, text, (x, y), font=font, fill=(255, 255, 255),
        )

    @staticmethod
    def _fit_cover(image: Image.Image, target: tuple[int, int]) -> Image.Image:
        target_w, target_h = target
        src_w, src_h = image.size
        scale = max(target_w / src_w, target_h / src_h) * 1.05  # slight zoom-in
        new_w = math.ceil(src_w * scale)
        new_h = math.ceil(src_h * scale)
        resized = image.resize((new_w, new_h), Image.LANCZOS)
        left = (new_w - target_w) // 2
        top = (new_h - target_h) // 2
        return resized.crop((left, top, left + target_w, top + target_h))

    @staticmethod
    def _apply_vignette(image: Image.Image) -> Image.Image:
        w, h = image.size
        mask = Image.new("L", (w, h), 0)
        draw = ImageDraw.Draw(mask)
        # Radial-ish vignette via two ellipses + blur.
        draw.ellipse(
            (int(w * 0.08), int(h * 0.08), int(w * 0.92), int(h * 0.92)),
            fill=255,
        )
        mask = mask.filter(ImageFilter.GaussianBlur(radius=120))
        black = Image.new("RGB", (w, h), (0, 0, 0))
        return Image.composite(image, black, mask)

    @staticmethod
    def _apply_corner_glow(image: Image.Image, *, color: tuple[int, int, int, int]) -> None:
        w, h = image.size
        glow = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        gd = ImageDraw.Draw(glow)
        gd.ellipse(
            (-int(w * 0.2), int(h * 0.55), int(w * 0.55), int(h * 1.25)),
            fill=color,
        )
        glow = glow.filter(ImageFilter.GaussianBlur(radius=140))
        image.paste(
            Image.alpha_composite(image.convert("RGBA"), glow).convert("RGB"),
            (0, 0),
        )

    @staticmethod
    def _shorten_for_overlay(title: str) -> str:
        words = re.findall(r"\S+", title)
        if len(words) <= 4:
            return " ".join(words).upper()
        # Pick the most informative 4 words (skip articles/conjunctions).
        skip = {"the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "with"}
        meaningful = [w for w in words if w.lower() not in skip]
        chosen = meaningful[:4] if len(meaningful) >= 2 else words[:4]
        return " ".join(chosen).upper()

    def _draw_thumbnail_text(
        self,
        image: Image.Image,
        text: str,
        *,
        accent_rgb: tuple[int, int, int] = (127, 255, 212),
    ) -> None:
        """Paint the bold thumbnail title overlay.

        Auto-fits the text by measuring at the chosen base font size and
        shrinking down (and wrapping to up to 3 lines) until the widest
        line fits within ~88% of the canvas width. This handles both
        landscape (1280x720) and vertical Shorts (1080x1920) canvases
        without the text running off the edge.
        """
        w, h = image.size
        draw = ImageDraw.Draw(image)
        words = text.split()
        if not words:
            return

        # Base size scales with canvas height. Vertical Shorts get a
        # slightly smaller multiplier because the canvas is so tall the
        # text would otherwise dwarf the panel underneath.
        is_portrait = h > w
        base_font_px = int((min(w, h) if is_portrait else h) * 0.13)
        min_font_px = max(28, int(base_font_px * 0.45))
        max_text_w = int(w * 0.88)

        # Try wrapping into 1, 2, or 3 lines. For each layout choice, try
        # progressively smaller fonts until the widest line fits.
        best_lines: list[str] = [text]
        best_font: ImageFont.ImageFont | None = None
        for line_count in (1, 2, 3):
            # Split into roughly equal-word chunks.
            if line_count == 1 or len(words) <= line_count:
                candidate_lines = [text] if line_count == 1 else [" ".join(words[i::line_count]) for i in range(min(line_count, len(words)))]
                if line_count > 1:
                    # Better split: keep word order, partition into N chunks.
                    chunk = max(1, (len(words) + line_count - 1) // line_count)
                    candidate_lines = [" ".join(words[i:i + chunk]) for i in range(0, len(words), chunk)]
            else:
                chunk = max(1, (len(words) + line_count - 1) // line_count)
                candidate_lines = [" ".join(words[i:i + chunk]) for i in range(0, len(words), chunk)]
            candidate_lines = [c for c in candidate_lines if c.strip()]

            # Pick the largest font where every line fits.
            for size in range(base_font_px, min_font_px - 1, -4):
                font_try = self._load_font(size)
                widths = [draw.textbbox((0, 0), line, font=font_try)[2] for line in candidate_lines]
                if not widths or max(widths) <= max_text_w:
                    best_lines = candidate_lines
                    best_font = font_try
                    break
            if best_font is not None:
                break
        if best_font is None:
            best_font = self._load_font(min_font_px)
            best_lines = candidate_lines or [text]

        # Highlight the longest single word in the accent color.
        highlight_word = max(words, key=len) if words else ""

        # Anchor at bottom-left with margins.
        margin_x = int(w * 0.05)
        margin_y = int(h * 0.07)
        font_size = getattr(best_font, "size", base_font_px)
        line_height = int(font_size * 1.08)
        total_h = line_height * len(best_lines)
        y = h - margin_y - total_h

        for line in best_lines:
            x = margin_x
            for word in line.split():
                color = accent_rgb if word == highlight_word else (255, 255, 255)
                self._draw_stroked_word(
                    draw, word + " ", (x, y),
                    font=best_font,
                    fill=color,
                )
                bbox = draw.textbbox((0, 0), word + " ", font=best_font)
                x += bbox[2] - bbox[0]
            y += line_height

    @staticmethod
    def _draw_stroked_word(
        draw: ImageDraw.ImageDraw,
        text: str,
        xy: tuple[int, int],
        *,
        font: ImageFont.ImageFont,
        fill: tuple[int, int, int],
    ) -> None:
        # Heavy black stroke so the word sticks to the panel no matter the
        # background colors underneath. YouTube thumbnails live or die by
        # legibility at 320px wide on a phone.
        stroke = max(3, getattr(font, "size", 56) // 18)
        draw.text(
            xy,
            text,
            font=font,
            fill=fill,
            stroke_width=stroke,
            stroke_fill=(0, 0, 0),
        )

    @staticmethod
    def _load_font(size: int) -> ImageFont.ImageFont:
        candidates = [
            "/System/Library/Fonts/Supplemental/Impact.ttf",
            "/System/Library/Fonts/HelveticaNeue.ttc",
            "/System/Library/Fonts/Helvetica.ttc",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        ]
        for path in candidates:
            try:
                if Path(path).exists():
                    return ImageFont.truetype(path, size)
            except Exception:
                continue
        return ImageFont.load_default()
