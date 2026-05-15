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
        thumbnail_path = self._compose_thumbnail(
            base_image=thumb_source_path,
            title=title,
            output_path=bundle_dir / "thumbnail.png",
            preset=preset,
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
                )
                thumbnail_variants.append({
                    "style_id": f"v{v_idx}",
                    "style_label": variant_labels[v_idx] if v_idx < len(variant_labels) else f"Variant {v_idx + 1}",
                    "path": str(v_path.relative_to(project_dir)),
                    "source_panel_id": str(v_panel.get("id") or ""),
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
            except Exception as exc:  # noqa: BLE001
                logger.warning("Finishing render failed: %s", exc)

        # ── Shorts auto-cut ───────────────────────────────────────────
        short_meta: dict[str, Any] | None = None
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
            "channel_preset": preset.to_dict(),
        }
        write_json(bundle_dir / "manifest.json", manifest)

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
        prompt = f"""You are writing YouTube metadata for a manga / manhwa / comic recap video.
You have 10 years of experience as a manga-narration YouTuber. Your goal
is click-through, retention, and comments, in that order.

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
- "title": EXACTLY the pattern "{{Series}} - {{Chapter label}}", e.g.
  "DARLING in the FRANXX - Chapters 1-10" or "unOrdinary - Chapter 27".
  Use a regular hyphen "-", never an em dash "-". No "Recap" or
  "Explained" suffix. No emoji. No ALL CAPS.
- "variants": three alternative titles testing three angles:
    1. Question form ("Who really controls the Plantations?")
    2. Character spotlight ("Zero Two's first mission - DARLING in the FRANXX")
    3. Stakes statement ("Humanity's last weapon meets its match")
  Each under 70 chars. No em dashes. No emoji.
- "description": exactly this structure, plain text, NO markdown, NO
  bullet points, NO chapter timestamps, NO "## What happens", NO
  "in this video":

  LINE 1: A single-sentence hook. Either a question OR a stakes
          statement. Under 110 chars. Treat it like ad copy: it has
          ONE job, to make the viewer click "Show more". GOOD:
          "Hiro never wanted to pilot a Franxx. Then she walked in."
          BAD: "The world is a barren wasteland scarred by humanity's
          extraction of magma energy".
  (blank line)
  LINES 3-5: A 2-3 sentence story tease. State the central tension
             and what's at stake for the protagonist. Do NOT recap
             scene by scene. Do NOT list locations or world-building
             details. Do NOT spoil the climax. Write like you would
             pitch the chapter to a friend in an elevator.
  (blank line)
  LINE 7: One sentence subscribe CTA in your own voice. Examples:
          "Subscribe so the next chapter hits your feed the day it drops."
          "If chapter recaps are your thing, hit subscribe."
  (blank line)
  LINE 9: Hashtag block, space-separated, 5-7 tags. Include #manga
          #anime #mangarecap plus 2-4 series-specific tags (one word,
          no spaces, derived from the series name).

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

        # Format: "Manga Name - Chapters X-Y" with a plain hyphen.
        # No em dashes anywhere. No " - Recap" suffix.
        title = f"{base} - {chapter}"[:TITLE_CHAR_BUDGET] if chapter else base[:TITLE_CHAR_BUDGET]
        chapter_suffix = f" - {chapter}" if chapter else ""
        variants = [
            f"{base}{chapter_suffix}",
            f"Everything that happened in {base}{chapter_suffix}",
            f"{base} explained{chapter_suffix}",
        ]
        # Fallback description used when no LLM is configured. We
        # deliberately do NOT echo the panel narration. The earlier
        # fallback dumped 3 narration sentences which produced the
        # "barren wasteland scarred by humanity..." Wikipedia-style
        # synopsis that we want to avoid. Instead, write a generic but
        # punchy stakes line keyed off the series name.
        hook_base = f"Every key moment of {base}{chapter_suffix}, in story order, in one sitting"
        hook = hook_base if len(hook_base) <= 110 else hook_base[:107].rstrip(", ;-") + "..."
        tease = (
            "Whether you missed the chapter or want a quick refresh before the next drop, "
            "this recap covers the beats that actually move the story. "
            "No filler, no padding, no spoilers in the title."
        )
        cta = "Subscribe so the next chapter recap hits your feed the day it drops."
        series_tag = re.sub(r"[^a-z0-9]", "", (manga_title or project_name or "manga").lower()) or "manga"
        hashtag_line = f"#manga #anime #mangarecap #{series_tag} #manhwa"
        # Note: no trailing "." on hook before the newline. The hook above
        # already ends with content (no period), and the previous version
        # added one explicitly which produced "in one sitting.." on output.
        description = f"{hook}.\n\n{tease}\n\n{cta}\n\n{hashtag_line}"
        return title, variants, description

    # ── Thumbnail composition ────────────────────────────────────────────

    def _compose_thumbnail(
        self,
        *,
        base_image: Path,
        title: str,
        output_path: Path,
        preset: Any = None,
    ) -> Path:
        """Render the final thumbnail with channel-preset branding.
        Preset is optional so legacy callers without a preset still work."""
        from app.services.channel_preset_service import ChannelPreset, ChannelPresetService
        if preset is None:
            try:
                preset = ChannelPresetService(self.settings).load()
            except Exception:
                preset = ChannelPreset()

        accent_rgb = self._hex_to_rgb(preset.accent_color, fallback=(127, 255, 212))
        watermark_text = (preset.watermark_text or "").strip() if preset.watermark_enabled else ""

        with Image.open(base_image) as src:
            src = src.convert("RGB")

            # Fit-cover into the YouTube thumbnail aspect ratio.
            canvas = self._fit_cover(src, THUMBNAIL_SIZE)

            # Mild enhancement so the panel pops on YouTube's small previews.
            canvas = ImageEnhance.Contrast(canvas).enhance(1.08)
            canvas = ImageEnhance.Color(canvas).enhance(1.18)

            # Cinematic vignette
            canvas = self._apply_vignette(canvas)

            # Brand-accent corner glow - pulls from preset so every
            # channel feels distinct at thumbnail glance.
            self._apply_corner_glow(canvas, color=(*accent_rgb, 180))

            # Title overlay using preset accent for the highlight word.
            short_title = self._shorten_for_overlay(title)
            self._draw_thumbnail_text(canvas, short_title, accent_rgb=accent_rgb)

            # Channel watermark in bottom-right (small).
            if watermark_text:
                self._draw_watermark(canvas, watermark_text)

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
        w, h = image.size
        # Try to load a strong display font; fall back to default if needed.
        font = self._load_font(int(h * 0.13))
        accent_font = self._load_font(int(h * 0.13))

        draw = ImageDraw.Draw(image)
        words = text.split()
        # Wrap into max 2 lines for readability.
        if len(words) > 2 and font.size * len(text) > w * 0.6:
            mid = len(words) // 2
            lines = [" ".join(words[:mid]), " ".join(words[mid:])]
        else:
            lines = [text]

        # Highlight the longest single word in the preset's accent color.
        highlight_word = max(words, key=len) if words else ""

        # Anchor at bottom-left with margins.
        margin_x = int(w * 0.05)
        margin_y = int(h * 0.07)
        line_height = int(font.size * 1.08)
        total_h = line_height * len(lines)
        y = h - margin_y - total_h

        for line in lines:
            x = margin_x
            for word in line.split():
                color = accent_rgb if word == highlight_word else (255, 255, 255)
                self._draw_stroked_word(
                    draw, word + " ", (x, y),
                    font=font if word != highlight_word else accent_font,
                    fill=color,
                )
                # advance x - use textbbox for accurate measurement
                bbox = draw.textbbox((0, 0), word + " ", font=font)
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
