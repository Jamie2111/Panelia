from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Callable, ClassVar

import numpy as np
import soundfile as sf
from PIL import Image, ImageColor, ImageDraw, ImageEnhance, ImageFilter, ImageFont

from app.core.config import get_settings
from app.schemas.project import MusicConfig, PanelBox, StorySegment, VideoConfig
from app.services.project_store import ProjectStore
from app.utils.files import ensure_dir, read_json, write_json

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class CameraPose:
    panel_id: str
    page: int
    x: float
    y: float
    zoom: float


@dataclass(slots=True)
class CameraSegment:
    kind: str
    panel_id: str
    script_id: str | None
    start: CameraPose
    end: CameraPose
    duration_seconds: float
    easing: str
    transition_style: str


@dataclass(slots=True)
class AudioEvent:
    panel_id: str
    path: Path
    start_seconds: float
    duration_seconds: float


@dataclass(slots=True)
class ChapterMarker:
    index: int
    title: str
    start_seconds: float


@dataclass(slots=True)
class TimelinePlan:
    segments: list[CameraSegment]
    audio_events: list[AudioEvent]
    total_duration_seconds: float
    chapter_markers: list[ChapterMarker] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.chapter_markers is None:
            self.chapter_markers = []


@dataclass(slots=True)
class PanelRenderAsset:
    panel_id: str
    image_path: Path
    card_path: Path
    narration: str


class VideoRenderService:
    def __init__(self) -> None:
        self.settings = get_settings()
        self.store = ProjectStore()
        self._shared_video_cache = ensure_dir(self.settings.data_dir / "_video_cache")

    def render_project_video(
        self,
        project_dir: Path,
        panels: list[PanelBox],
        story_segments: list[StorySegment],
        video_config: VideoConfig,
        music_config: MusicConfig,
        output_name: str = "final",
        progress_callback: Callable[[float, str], None] | None = None,
        cancel_callback: Callable[[], None] | None = None,
    ) -> Path:
        kept_panels = [panel for panel in sorted(panels, key=lambda item: item.order) if panel.keep]
        timeline_segments = [
            segment
            for segment in sorted(story_segments, key=lambda item: item.order)
            if bool(getattr(segment, "keep", True))
        ]
        spoken_segments = [segment for segment in timeline_segments if str(segment.text or "").strip()]
        if not kept_panels or not timeline_segments or not spoken_segments:
            raise ValueError("No story segments with narration are available for rendering.")

        audio_manifest = read_json(project_dir / "audio" / "manifest.json", default={})
        panel_assets = self._build_panel_assets(project_dir, kept_panels, timeline_segments)
        plan = self._build_timeline_plan(project_dir, kept_panels, timeline_segments, audio_manifest, video_config)
        if progress_callback:
            progress_callback(2, "Planned camera travel")

        output_dir = ensure_dir(project_dir / "video")
        work_dir = ensure_dir(project_dir / "temp" / "render")
        shutil.rmtree(work_dir, ignore_errors=True)
        ensure_dir(work_dir)

        self._write_camera_manifest(output_dir / f"{output_name}_camera.json", plan)
        self._write_chapter_markers(output_dir / f"{output_name}_chapters.json", plan, output_name)

        narration_path = work_dir / f"{output_name}_narration.wav"
        self._render_narration_track(plan, narration_path, music_config=music_config)
        if progress_callback:
            progress_callback(4, "Built narration timeline")

        silent_video_path = work_dir / f"{output_name}_silent.mp4"
        self._render_timeline_video(
            plan,
            project_dir,
            panel_assets,
            silent_video_path,
            video_config,
            progress_callback=progress_callback,
            cancel_callback=cancel_callback,
        )
        final_video_source = silent_video_path
        final_narration_source = narration_path

        # --- Animated title card (generated, always available when enabled) ---
        if video_config.title_card_enabled:
            tc_duration = max(1.5, min(float(video_config.title_card_seconds or 3.0), 6.0))
            tc_image_path = work_dir / f"{output_name}_title_card.jpg"
            tc_video_path = work_dir / f"{output_name}_title_card.mp4"
            tc_narration_path = work_dir / f"{output_name}_narration_tc.wav"
            tc_video_with_tc = work_dir / f"{output_name}_silent_with_tc.mp4"

            # Build text fields from project metadata stored on disk
            series_title = ""
            chapter_info = ""
            char_names: list[str] = []
            try:
                bible_path = project_dir / "output" / "story_bible.json"
                if bible_path.exists():
                    bible = read_json(bible_path, default={})
                    series_title = str(bible.get("series_title") or bible.get("manga_title") or "").strip()
                    chapter_num = str(bible.get("chapter_number") or "").strip()
                    chapter_info = f"Chapter {chapter_num}" if chapter_num else ""
                    cast = bible.get("cast") or []
                    char_names = [
                        str(c.get("name") or c.get("display_name") or "").strip()
                        for c in cast if isinstance(c, dict)
                        if str(c.get("name") or c.get("display_name") or "").strip()
                    ][:5]
            except Exception:
                pass
            # Fallback: read style_vocabulary for character names
            if not char_names:
                try:
                    sv_path = project_dir / "output" / "style_vocabulary.json"
                    if sv_path.exists():
                        sv = read_json(sv_path, default={})
                        char_names = [
                            str(n).strip() for n in (sv.get("named_characters") or [])
                            if str(n).strip()
                        ][:5]
                except Exception:
                    pass

            self._generate_title_card_image(
                tc_image_path, video_config,
                series_title=series_title or "Untitled",
                chapter_info=chapter_info,
                character_names=char_names,
            )
            self._render_title_card_clip(tc_image_path, tc_video_path, video_config, tc_duration)
            self._prepend_video_intro(tc_video_path, final_video_source, tc_video_with_tc)
            self._prepend_audio_silence(final_narration_source, tc_narration_path, tc_duration)
            final_video_source = tc_video_with_tc
            final_narration_source = tc_narration_path
            if progress_callback:
                progress_callback(93, "Added animated title card")

        # --- Static intro thumbnail (user-uploaded, legacy) ---
        intro_thumbnail_path = self._resolve_video_intro_thumbnail(project_dir)
        intro_thumbnail_enabled = bool(getattr(video_config, "intro_thumbnail_enabled", False)) and intro_thumbnail_path is not None
        if intro_thumbnail_enabled:
            intro_duration = max(0.5, min(float(getattr(video_config, "intro_thumbnail_seconds", 1.5) or 1.5), 4.0))
            intro_video_path = work_dir / f"{output_name}_intro.mp4"
            intro_narration_path = work_dir / f"{output_name}_narration_intro.wav"
            timeline_with_intro = work_dir / f"{output_name}_silent_with_intro.mp4"
            self._render_intro_thumbnail_clip(intro_thumbnail_path, intro_video_path, video_config, intro_duration)
            self._prepend_video_intro(intro_video_path, silent_video_path, timeline_with_intro)
            self._prepend_audio_silence(narration_path, intro_narration_path, intro_duration)
            final_video_source = timeline_with_intro
            final_narration_source = intro_narration_path
            if progress_callback:
                progress_callback(94, "Prepended thumbnail lead-in")
        if progress_callback:
            progress_callback(95, "Preparing final video export")

        output_path = output_dir / f"{output_name}.{video_config.output_format.value}"
        if progress_callback:
            progress_callback(96, "Muxing narration and picture")
        self._mux_video_with_audio(final_video_source, final_narration_source, output_path)
        if progress_callback:
            progress_callback(97, "Muxed narration and picture")

        final_path = output_path
        if music_config.enabled and music_config.track_name:
            final_path = output_dir / f"{output_name}_music.{video_config.output_format.value}"
            if progress_callback:
                progress_callback(98, "Mixing background music")
            self._mix_background_music(output_path, final_path, music_config)
            output_path.unlink(missing_ok=True)
            output_path = final_path
            if progress_callback:
                progress_callback(99, "Mixed background music")

        wm = video_config.watermark
        if wm.enabled and wm.image_path:
            wm_image = Path(wm.image_path) if Path(wm.image_path).is_absolute() else project_dir / wm.image_path
            if wm_image.exists():
                wm_out = output_dir / f"{output_name}_wm.{video_config.output_format.value}"
                if progress_callback:
                    progress_callback(99, "Applying watermark")
                self._apply_watermark(output_path, wm_image, wm_out, video_config, wm)
                output_path.unlink(missing_ok=True)
                output_path = wm_out
                if progress_callback:
                    progress_callback(99, "Applied watermark")

        # NOTE: The dedicated channel-watermark re-encode pass used to live
        # here. It re-encoded the entire final video (2 hours @ 1080p for
        # Darling-sized projects) just to draw the @handle in one corner.
        # Even with h264_videotoolbox that step ran ~7-8 minutes; with the
        # original libx264 path it was 80+ minutes. The watermark is now
        # baked into each per-panel clip during the cheap render pass (see
        # _render_panel_clip + _resolve_channel_watermark), so this stage
        # is gone entirely. Title card + intro thumbnail clips are NOT
        # overlaid (~3s at the very start without the corner @handle);
        # the watermark appears from the first narrated panel onward.

        # Safety net: every encoder path in this file is supposed to pass
        # `-movflags +faststart`, but if any future remux step forgets to do
        # so the resulting video opens with `mdat` first and is effectively
        # unplayable in browsers (they must download the whole file before
        # they can decode a single frame). Verify by reading the leading
        # bytes; if `moov` isn't visible early, do a fast `-c copy` remux.
        self._ensure_faststart(output_path)

        self._write_video_manifest(output_dir, output_path, video_config)
        if progress_callback:
            progress_callback(100, "Video render complete")
        return output_path

    def _ensure_faststart(self, video_path: Path) -> None:
        """Fast check + repair for moov-atom ordering.

        Reads the first 16 KB of the file. A streamable MP4 has its `moov`
        atom near the start (after `ftyp`). If we instead see `mdat` first
        with no `moov` in the head, we remux in place to relocate it.
        """
        try:
            with video_path.open("rb") as handle:
                head = handle.read(16384)
        except OSError:
            return
        if b"moov" in head:
            return
        # mdat-first or unknown ordering. Remux to fix.
        tmp = video_path.with_suffix(video_path.suffix + ".faststart")
        try:
            self._run_ffmpeg([
                self.settings.ffmpeg_binary,
                "-y",
                "-i", str(video_path),
                "-c", "copy",
                "-movflags", "+faststart",
                str(tmp),
            ])
        except Exception:
            tmp.unlink(missing_ok=True)
            return
        try:
            tmp.replace(video_path)
        except OSError:
            tmp.unlink(missing_ok=True)

    def _resolve_video_intro_thumbnail(self, project_dir: Path) -> Path | None:
        path = project_dir / "thumbnails" / "video_intro.jpg"
        return path if path.exists() else None

    def _generate_title_card_image(
        self,
        output_path: Path,
        video_config: VideoConfig,
        series_title: str,
        chapter_info: str,
        character_names: list[str],
    ) -> None:
        """Render a title card image using PIL and save to output_path.

        Layout (top-to-bottom):
          - Thin accent line (top 4 px)
          - Series title (large, white, centred)
          - Chapter info (medium, muted, centred)
          - Character roster (small, spaced, centred) - max 5 names
          - Thin accent line (bottom 4 px)
        """
        w, h = video_config.width, video_config.height
        bg = video_config.background_color or "#09090b"
        accent = getattr(video_config, "title_card_accent_color", "#e11d48")

        img = Image.new("RGB", (w, h), bg)
        draw = ImageDraw.Draw(img)

        # Accent bars (top and bottom)
        bar_h = max(4, h // 270)
        draw.rectangle([(0, 0), (w, bar_h)], fill=accent)
        draw.rectangle([(0, h - bar_h), (w, h)], fill=accent)

        # Font helpers - fall back to default if system font not found
        def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
            for candidate in (
                "/System/Library/Fonts/SFNS.ttf",
                "/System/Library/Fonts/Helvetica.ttc",
                "/Library/Fonts/Arial Unicode.ttf",
                "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
            ):
                try:
                    return ImageFont.truetype(candidate, size)
                except (OSError, IOError):
                    continue
            return ImageFont.load_default()

        title_size = max(48, h // 16)
        chapter_size = max(28, h // 28)
        roster_size = max(20, h // 38)

        title_font = _font(title_size)
        chapter_font = _font(chapter_size)
        roster_font = _font(roster_size)

        # Colours
        white = "#ffffff"
        muted = "#a1a1aa"      # zinc-400
        roster_color = "#71717a"  # zinc-500

        # Centre-Y distribution
        centre_y = h // 2
        gap = h // 22

        # Series title
        bbox = draw.textbbox((0, 0), series_title, font=title_font)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
        ty = centre_y - th - gap // 2
        draw.text(((w - tw) // 2, ty), series_title, font=title_font, fill=white)

        # Chapter info
        if chapter_info:
            cbbox = draw.textbbox((0, 0), chapter_info, font=chapter_font)
            cw = cbbox[2] - cbbox[0]
            draw.text(((w - cw) // 2, centre_y + gap // 2), chapter_info, font=chapter_font, fill=muted)

        # Character roster (up to 5 names, comma-separated)
        if character_names:
            roster = "  ·  ".join(character_names[:5])
            rbbox = draw.textbbox((0, 0), roster, font=roster_font)
            rw = rbbox[2] - rbbox[0]
            ry = centre_y + gap * 2
            draw.text(((w - rw) // 2, ry), roster, font=roster_font, fill=roster_color)

        img.save(str(output_path), format="JPEG", quality=95)

    def _render_title_card_clip(
        self,
        image_path: Path,
        output_path: Path,
        video_config: VideoConfig,
        duration_seconds: float,
    ) -> None:
        """Render title card image to video with fade-in/out.

        Uses the shared _video_encoder_args helper so the codec/profile
        matches the per-panel clips. Matching codec params lets
        _prepend_video_intro use the concat-demuxer with -c copy
        (no re-encode) instead of the slow concat-filter path.
        """
        output_path.unlink(missing_ok=True)
        fade_dur = min(0.6, duration_seconds * 0.2)
        command = [
            self.settings.ffmpeg_binary,
            "-y",
            "-loop", "1",
            "-framerate", str(video_config.fps),
            "-t", f"{duration_seconds:.3f}",
            "-i", str(image_path),
            "-vf", (
                f"scale={video_config.width}:{video_config.height}:force_original_aspect_ratio=decrease,"
                f"pad={video_config.width}:{video_config.height}:(ow-iw)/2:(oh-ih)/2:color={video_config.background_color},setsar=1,"
                f"fade=t=in:st=0:d={fade_dur:.3f},"
                f"fade=t=out:st={duration_seconds - fade_dur:.3f}:d={fade_dur:.3f}"
            ),
            "-an",
            *self._video_encoder_args(intermediate=True),
            "-movflags", "+faststart",
            str(output_path),
        ]
        self._run_ffmpeg(command)

    def _render_intro_thumbnail_clip(
        self,
        image_path: Path,
        output_path: Path,
        video_config: VideoConfig,
        duration_seconds: float,
    ) -> None:
        """Render a static intro thumbnail clip.

        Uses the shared _video_encoder_args helper so the codec/profile
        matches the per-panel clips and the concat-demuxer in
        _prepend_video_intro can stream-copy without re-encoding.
        """
        output_path.unlink(missing_ok=True)
        command = [
            self.settings.ffmpeg_binary,
            "-y",
            "-loop",
            "1",
            "-framerate",
            str(video_config.fps),
            "-t",
            f"{duration_seconds:.3f}",
            "-i",
            str(image_path),
            "-vf",
            (
                f"scale={video_config.width}:{video_config.height}:force_original_aspect_ratio=decrease,"
                f"pad={video_config.width}:{video_config.height}:(ow-iw)/2:(oh-ih)/2:color={video_config.background_color},setsar=1"
            ),
            "-an",
            *self._video_encoder_args(intermediate=True),
            "-movflags",
            "+faststart",
            str(output_path),
        ]
        self._run_ffmpeg(command)

    def _prepend_video_intro(self, intro_path: Path, video_path: Path, output_path: Path) -> None:
        """Prepend a short intro clip to a long video.

        Performance critical: a Darling-sized project has a ~2-hour silent
        timeline. The original implementation used the concat *filter*
        which re-encodes both inputs end-to-end (libx264 veryfast crf 17
        on 2 hours @ 1080p = roughly an hour wall clock). Both inputs in
        practice come from our own pipeline with matching codec / pix_fmt
        / resolution, so we can use the concat *demuxer* with -c copy and
        skip re-encoding entirely - the operation drops to seconds.

        If the codec parameters don't match (rare; would only happen if
        someone fed an intro file from an unrelated source), we fall back
        to a re-encode using h264_videotoolbox on Apple Silicon
        (~6-8 min for 2 hours) or libx264 elsewhere.
        """
        output_path.unlink(missing_ok=True)
        list_file = output_path.parent / f".{output_path.stem}_concat.txt"
        list_file.write_text(
            f"file '{intro_path.resolve()}'\nfile '{video_path.resolve()}'\n",
            encoding="utf-8",
        )
        try:
            self._run_ffmpeg([
                self.settings.ffmpeg_binary,
                "-y",
                "-f", "concat",
                "-safe", "0",
                "-i", str(list_file),
                "-c", "copy",
                "-movflags", "+faststart",
                str(output_path),
            ])
            return
        except subprocess.CalledProcessError as exc:
            logger.warning(
                "concat-demuxer prepend failed (likely mismatched codec params); "
                "falling back to filter+re-encode. err=%s",
                exc,
            )
        finally:
            list_file.unlink(missing_ok=True)

        # Fallback: concat filter with hardware encode where possible.
        is_apple = Path("/opt/homebrew/bin/ffmpeg").exists()
        if is_apple:
            codec_args = [
                "-c:v", "h264_videotoolbox",
                "-b:v", "14M",
                "-maxrate", "18M",
                "-pix_fmt", "yuv420p",
            ]
        else:
            codec_args = [
                "-c:v", "libx264",
                "-preset", "veryfast",
                "-crf", "17",
                "-pix_fmt", "yuv420p",
            ]
        self._run_ffmpeg([
            self.settings.ffmpeg_binary,
            "-y",
            "-i", str(intro_path),
            "-i", str(video_path),
            "-filter_complex", "[0:v][1:v]concat=n=2:v=1:a=0[vout]",
            "-map", "[vout]",
            *codec_args,
            "-movflags", "+faststart",
            str(output_path),
        ])

    def _resolve_channel_watermark(
        self, video_config: "VideoConfig"
    ) -> tuple[Path, str] | None:
        """Return (cached PNG path, channel handle text) for the corner
        watermark, or None if it's disabled / not configured.

        The PNG is rendered once per (text, video height) and cached under
        `_video_cache/channel_watermarks/`. The per-panel ffmpeg filter
        chain then overlays it as a tiny extra input — practically free
        compared to the old dedicated re-encode pass that took 80+ min
        on a 2-hour video.
        """
        try:
            from app.services.channel_preset_service import ChannelPresetService
            preset = ChannelPresetService(self.settings).load()
        except Exception:
            return None
        if not preset or not getattr(preset, "watermark_enabled", False):
            return None
        text = (getattr(preset, "watermark_text", "") or "").strip()
        if not text:
            return None
        cache_dir = ensure_dir(self._shared_video_cache / "channel_watermarks")
        key = hashlib.sha1(f"{text}|{video_config.height}".encode("utf-8")).hexdigest()[:12]
        png_path = cache_dir / f"{key}.png"
        if not png_path.exists():
            tmp = cache_dir / f"{key}.tmp.png"
            self._render_channel_watermark_png(text, tmp, video_config)
            tmp.replace(png_path)
        return png_path, text

    @staticmethod
    def _render_channel_watermark_png(
        text: str,
        output_path: Path,
        video_config: "VideoConfig",
    ) -> None:
        """Render the channel handle as a small transparent PNG.

        Used to overlay a subtle, non-intrusive corner watermark onto
        the rendered video. Sized to ~2.6% of canvas height (so 28 px
        on a 1080p frame), white with a 1-px black stroke for legibility
        on any background. We render only the text + stroke, NO opaque
        box - the ffmpeg overlay step controls final opacity globally.
        """
        from PIL import Image as _Image, ImageDraw as _ImageDraw, ImageFont as _ImageFont

        height = int(video_config.height)
        font_size = max(18, int(height * 0.026))
        # Sized to fit the longest reasonable handle (~20 chars).
        canvas_w = int(font_size * 14)
        canvas_h = int(font_size * 2)

        # Pick a font; falls back to default if no TrueType is present.
        font: _ImageFont.ImageFont
        for candidate in (
            "/System/Library/Fonts/HelveticaNeue.ttc",
            "/System/Library/Fonts/Helvetica.ttc",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        ):
            try:
                if Path(candidate).exists():
                    font = _ImageFont.truetype(candidate, font_size)
                    break
            except Exception:
                continue
        else:
            font = _ImageFont.load_default()

        canvas = _Image.new("RGBA", (canvas_w, canvas_h), (0, 0, 0, 0))
        draw = _ImageDraw.Draw(canvas)
        text_str = text.strip()
        bbox = draw.textbbox((0, 0), text_str, font=font)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
        # Right-pad inside the canvas so we don't clip the descender;
        # the overlay positioning is done by ffmpeg from the right edge.
        x = canvas_w - tw - 4
        y = (canvas_h - th) // 2
        # Soft drop shadow for legibility on any panel background.
        draw.text(
            (x, y),
            text_str,
            fill=(255, 255, 255, 235),
            font=font,
            stroke_width=max(1, font_size // 18),
            stroke_fill=(0, 0, 0, 200),
        )
        canvas.save(output_path, "PNG", optimize=True)

    # _apply_channel_watermark was removed in favor of baking the channel
    # @handle into each per-panel clip during the cheap encode pass. See
    # _resolve_channel_watermark + _render_panel_clip for the new path.

    def _apply_watermark(
        self,
        video_path: Path,
        wm_image: Path,
        output_path: Path,
        video_config: "VideoConfig",
        wm: "WatermarkConfig",
    ) -> None:
        """Overlay a PNG watermark onto the video using FFmpeg.

        The watermark is:
        - Scaled to ``wm.scale`` × video_width
        - Positioned at the specified corner with ``wm.margin_px`` padding
        - Blended at ``wm.opacity`` using the ``format=rgba,colorchannelmixer`` approach
        - Faded in/out at the start/end of the video
        """
        from app.schemas.project import WatermarkPosition
        output_path.unlink(missing_ok=True)

        w = video_config.width
        wm_width = max(16, int(w * wm.scale))
        margin = wm.margin_px
        h = video_config.height

        pos_map = {
            WatermarkPosition.BOTTOM_RIGHT: (f"W-w-{margin}", f"H-h-{margin}"),
            WatermarkPosition.BOTTOM_LEFT:  (f"{margin}",     f"H-h-{margin}"),
            WatermarkPosition.TOP_RIGHT:    (f"W-w-{margin}", f"{margin}"),
            WatermarkPosition.TOP_LEFT:     (f"{margin}",     f"{margin}"),
        }
        x_expr, y_expr = pos_map.get(wm.position, (f"W-w-{margin}", f"H-h-{margin}"))

        # Build fade expression: alpha = opacity unless near start or end
        fade_in = wm.fade_in_seconds
        fade_out = wm.fade_out_seconds
        total = f"(main_h/main_w*{w})"  # not used - use duration from video
        alpha = wm.opacity
        # FFmpeg enable expression for fade: linear ramp at boundaries
        enable_expr = (
            f"if(lt(t,{fade_in}),t/{fade_in},"
            f"if(gt(t,duration-{fade_out}),(duration-t)/{fade_out},1))"
        )
        alpha_expr = f"{alpha}*{enable_expr}"

        filter_complex = (
            f"[1:v]scale={wm_width}:-1,format=rgba,colorchannelmixer=aa={alpha}[wm];"
            f"[0:v][wm]overlay={x_expr}:{y_expr}:format=auto:enable='between(t,0,999999)'"
        )

        command = [
            self.settings.ffmpeg_binary,
            "-y",
            "-i", str(video_path),
            "-i", str(wm_image),
            "-filter_complex", filter_complex,
            "-c:a", "copy",
            "-c:v", "libx264" if video_config.output_format.value == "mp4" else "libx264",
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            str(output_path),
        ]
        self._run_ffmpeg(command)

    def _prepend_audio_silence(self, narration_path: Path, output_path: Path, duration_seconds: float) -> None:
        output_path.unlink(missing_ok=True)
        audio_info = sf.info(str(narration_path))
        sample_rate = int(audio_info.samplerate or 24000)
        channels = int(audio_info.channels or 1)
        channel_layout = "mono" if channels <= 1 else "stereo"
        command = [
            self.settings.ffmpeg_binary,
            "-y",
            "-f",
            "lavfi",
            "-t",
            f"{duration_seconds:.3f}",
            "-i",
            f"anullsrc=channel_layout={channel_layout}:sample_rate={sample_rate}",
            "-i",
            str(narration_path),
            "-filter_complex",
            "[0:a][1:a]concat=n=2:v=0:a=1[aout]",
            "-map",
            "[aout]",
            "-c:a",
            "pcm_s16le",
            str(output_path),
        ]
        self._run_ffmpeg(command)

    def merge_videos(
        self,
        output_dir: Path,
        video_paths: list[Path],
        output_name: str,
        video_config: VideoConfig,
    ) -> Path:
        ensure_dir(output_dir)
        work_dir = ensure_dir(output_dir / "merge_tmp")
        shutil.rmtree(work_dir, ignore_errors=True)
        ensure_dir(work_dir)
        normalized_paths: list[Path] = []

        for index, path in enumerate(video_paths, start=1):
            normalized = work_dir / f"normalized_{index:03d}.mp4"
            command = [
                self.settings.ffmpeg_binary,
                "-y",
                "-i",
                str(path),
                "-vf",
                f"scale={video_config.width}:{video_config.height}:force_original_aspect_ratio=decrease,pad={video_config.width}:{video_config.height}:(ow-iw)/2:(oh-ih)/2:color={video_config.background_color}",
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-c:a",
                "aac",
                str(normalized),
            ]
            self._run_ffmpeg(command)
            normalized_paths.append(normalized)

        concat_manifest = work_dir / "concat.txt"
        concat_manifest.write_text("\n".join(f"file '{item.as_posix()}'" for item in normalized_paths), encoding="utf-8")
        output_path = output_dir / f"{output_name}.{video_config.output_format.value}"
        self._run_ffmpeg(
            [
                self.settings.ffmpeg_binary,
                "-y",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                str(concat_manifest),
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-c:a",
                "aac",
                str(output_path),
            ]
        )
        self._write_video_manifest(output_dir, output_path, video_config)
        return output_path

    def example_commands(self) -> list[str]:
        return [
            f"{self.settings.ffmpeg_binary} -y -f rawvideo -pix_fmt rgb24 -s 1080x1920 -r 24 -i - -an -c:v libx264 -pix_fmt yuv420p camera_pass.mp4",
            f"{self.settings.ffmpeg_binary} -y -i camera_pass.mp4 -i narration.wav -c:v copy -c:a aac -shortest final.mp4",
            f"{self.settings.ffmpeg_binary} -y -i final.mp4 -stream_loop -1 -i music.mp3 -filter_complex \"[1:a]volume=0.2,afade=t=in:st=0:d=1,afade=t=out:st=18:d=2[m];[0:a][m]amix=inputs=2:duration=first[aout]\" -map 0:v -map \"[aout]\" -c:v copy -c:a aac final_with_music.mp4",
        ]

    def _build_timeline_plan(
        self,
        project_dir: Path,
        kept_panels: list[PanelBox],
        story_segments: list[StorySegment],
        audio_manifest: dict[str, dict[str, object]],
        video_config: VideoConfig,
    ) -> TimelinePlan:
        narration_events = self._build_audio_events(project_dir, audio_manifest)
        panels_by_id = {panel.id: panel for panel in kept_panels}
        audio_by_id = {event.panel_id: event for event in narration_events}
        timed_audio_events: list[AudioEvent] = []

        segments: list[CameraSegment] = []
        chapter_markers: list[ChapterMarker] = []
        timeline_seconds = 0.0
        ordered_events = list(narration_events)

        for segment_index, story_segment in enumerate(story_segments, start=1):
            covered_panels = self._resolve_story_segment_panels(story_segment, kept_panels, panels_by_id)
            if not covered_panels:
                continue

            audio_event = audio_by_id.get(story_segment.id)
            if audio_event is None and segment_index - 1 < len(ordered_events):
                audio_event = ordered_events[segment_index - 1]
            audio_start = timeline_seconds

            # Build a chapter marker for each story segment - first sentence becomes the title
            segment_text = str(story_segment.text or "").strip()
            marker_title = self._chapter_title_from_text(segment_text, segment_index)
            chapter_markers.append(ChapterMarker(
                index=segment_index,
                title=marker_title,
                start_seconds=audio_start,
            ))
            audio_duration = float(audio_event.duration_seconds) if audio_event is not None else 0.0

            text_fragments = self._distribute_segment_text(str(story_segment.text or "").strip(), len(covered_panels))
            per_panel_audio = audio_duration / max(len(covered_panels), 1) if audio_duration > 0 else 0.0

            for panel_index, panel in enumerate(covered_panels, start=1):
                fragment = text_fragments[panel_index - 1] if panel_index - 1 < len(text_fragments) else ""
                hold_duration = self._panel_time(fragment, per_panel_audio, panel.duration_seconds)
                panel_poses = self._build_panel_poses(panel, video_config, sequence_seed=segment_index * 17 + panel_index)
                hold_segments, _final_pose = self._build_panel_hold_segments(
                    panel_id=panel.id,
                    script_id=story_segment.id,
                    poses=panel_poses,
                    hold_duration=hold_duration,
                    sequence_seed=segment_index * 17 + panel_index,
                    video_config=video_config,
                )
                segments.extend(hold_segments)
                timeline_seconds += sum(segment.duration_seconds for segment in hold_segments)

            if audio_event is not None:
                timed_audio_events.append(
                    AudioEvent(
                        panel_id=story_segment.id,
                        path=audio_event.path,
                        start_seconds=audio_start,
                        duration_seconds=audio_duration,
                    )
                )

        if narration_events and not timed_audio_events:
            timeline_seconds, segments = self._stretch_segments_for_narration(segments, timeline_seconds, narration_events)
            timed_audio_events = narration_events

        return TimelinePlan(
            segments=segments,
            audio_events=timed_audio_events,
            total_duration_seconds=timeline_seconds,
            chapter_markers=chapter_markers,
        )

    def _build_audio_events(self, project_dir: Path, audio_manifest: dict[str, dict[str, object]]) -> list[AudioEvent]:
        events: list[AudioEvent] = []
        cursor = 0.0
        if audio_manifest:
            candidates = [project_dir / "audio" / name for name in sorted(audio_manifest)]
        else:
            candidates = sorted((project_dir / "audio").glob("panel_*.wav"))

        for audio_path in candidates:
            duration = float(audio_manifest.get(audio_path.name, {}).get("duration_seconds") or 0.0)
            if not audio_path.exists():
                continue
            if duration <= 0 and audio_manifest:
                continue
            events.append(
                AudioEvent(
                    panel_id=str(audio_manifest.get(audio_path.name, {}).get("panel_id") or audio_path.stem),
                    path=audio_path,
                    start_seconds=cursor,
                    duration_seconds=duration,
                )
            )
            cursor += duration
        return events

    def _resolve_story_segment_panels(
        self,
        story_segment: StorySegment,
        kept_panels: list[PanelBox],
        panels_by_id: dict[str, PanelBox],
    ) -> list[PanelBox]:
        covered: list[PanelBox] = []
        for panel_id in story_segment.panel_ids:
            panel = panels_by_id.get(panel_id)
            if panel is not None:
                covered.append(panel)
        if covered:
            return sorted(covered, key=lambda item: item.order)

        if story_segment.panel_start is not None and story_segment.panel_end is not None:
            ranged = [
                panel
                for panel in kept_panels
                if story_segment.panel_start <= int(panel.order) <= story_segment.panel_end
            ]
            if ranged:
                return ranged

        if story_segment.representative_panel_id and story_segment.representative_panel_id in panels_by_id:
            return [panels_by_id[story_segment.representative_panel_id]]

        return [kept_panels[min(max(story_segment.order - 1, 0), len(kept_panels) - 1)]] if kept_panels else []

    def _distribute_segment_text(self, text: str, panel_count: int) -> list[str]:
        if panel_count <= 0:
            return []
        normalized = " ".join(str(text or "").split()).strip()
        if not normalized:
            return ["" for _ in range(panel_count)]
        if panel_count == 1:
            return [normalized]

        sentences = [
            sentence.strip()
            for sentence in re.split(r"(?<=[.!?])\s+", normalized)
            if sentence.strip()
        ]
        if len(sentences) >= panel_count:
            return self._merge_script_lines_into_targets(sentences, panel_count)

        result = ["" for _ in range(panel_count)]
        positions = self._spread_positions(panel_count, len(sentences))
        for sentence, position in zip(sentences, positions, strict=False):
            result[position] = sentence
        if not any(result):
            result[0] = normalized
        return result

    def _spread_positions(self, total_targets: int, total_lines: int) -> list[int]:
        if total_targets <= 0 or total_lines <= 0:
            return []
        if total_lines == 1:
            return [0]
        positions = [
            round(line_index * (total_targets - 1) / max(total_lines - 1, 1))
            for line_index in range(total_lines)
        ]
        for index in range(1, len(positions)):
            positions[index] = max(positions[index], positions[index - 1] + 1)
        for index in range(len(positions) - 2, -1, -1):
            positions[index] = min(positions[index], positions[index + 1] - 1)
        return [max(0, min(total_targets - 1, position)) for position in positions]

    def _merge_script_lines_into_targets(self, script_lines: list[str], total_targets: int) -> list[str]:
        if total_targets <= 0:
            return []
        merged: list[str] = []
        total_lines = len(script_lines)
        for target_index in range(total_targets):
            start = round(target_index * total_lines / total_targets)
            end = round((target_index + 1) * total_lines / total_targets)
            chunk = [line.strip() for line in script_lines[start:end] if line.strip()]
            merged.append(" ".join(chunk))
        return merged

    def _build_panel_assets(
        self,
        project_dir: Path,
        kept_panels: list[PanelBox],
        story_segments: list[StorySegment],
    ) -> dict[str, PanelRenderAsset]:
        assets: dict[str, PanelRenderAsset] = {}
        cache_dir = ensure_dir(self._shared_video_cache / "cards")
        crop_dir = ensure_dir(self._shared_video_cache / "panel_crops")
        panels_by_id = {panel.id: panel for panel in kept_panels}
        caption_lookup: dict[str, str] = {}
        for story_segment in story_segments:
            covered_panels = self._resolve_story_segment_panels(story_segment, kept_panels, panels_by_id)
            fragments = self._distribute_segment_text(str(story_segment.text or "").strip(), len(covered_panels))
            for panel, fragment in zip(covered_panels, fragments, strict=False):
                if fragment.strip():
                    caption_lookup[panel.id] = fragment.strip()
        for panel in kept_panels:
            image_path = self._prepare_panel_crop(project_dir, panel, crop_dir)
            card_path = self._prepare_panel_card(image_path, cache_dir, panel.id)
            narration = caption_lookup.get(panel.id, "")
            assets[panel.id] = PanelRenderAsset(
                panel_id=panel.id,
                image_path=image_path,
                card_path=card_path,
                narration=narration,
            )
        return assets

    def _stretch_segments_for_narration(
        self,
        segments: list[CameraSegment],
        timeline_seconds: float,
        audio_events: list[AudioEvent],
    ) -> tuple[float, list[CameraSegment]]:
        narration_duration = sum(event.duration_seconds for event in audio_events)
        target_duration = max(narration_duration + 0.35, timeline_seconds)
        if timeline_seconds <= 0 or target_duration <= timeline_seconds:
            return timeline_seconds, segments

        scale = target_duration / timeline_seconds
        stretched = [
            CameraSegment(
                kind=segment.kind,
                panel_id=segment.panel_id,
                script_id=segment.script_id,
                start=segment.start,
                end=segment.end,
                duration_seconds=round(segment.duration_seconds * scale, 3),
                easing=segment.easing,
                transition_style=segment.transition_style,
            )
            for segment in segments
        ]
        return target_duration, stretched

    # Ken Burns camera move presets: (start_x, start_y, end_x, end_y, zoom_in)
    # Chosen by sequence_seed % len to cycle through varied movements.
    _KEN_BURNS_MOVES: tuple[tuple[float, float, float, float, bool], ...] = (
        (0.47, 0.47, 0.50, 0.50, True),   # drift to centre, zoom in
        (0.53, 0.47, 0.50, 0.50, True),   # drift from top-right to centre
        (0.47, 0.53, 0.50, 0.50, True),   # drift from bottom-left to centre
        (0.53, 0.53, 0.50, 0.50, True),   # drift from bottom-right to centre
        (0.50, 0.50, 0.47, 0.47, False),  # pull back to top-left
        (0.50, 0.50, 0.53, 0.47, False),  # pull back to top-right
        (0.50, 0.50, 0.47, 0.53, False),  # pull back to bottom-left
        (0.50, 0.50, 0.53, 0.53, False),  # pull back to bottom-right
    )

    def _build_panel_poses(
        self,
        panel: PanelBox,
        video_config: VideoConfig,
        sequence_seed: int,
    ) -> list[CameraPose]:
        """Build start/end poses for the Ken Burns effect.

        Cycles through 8 directional camera moves so adjacent panels always
        feel varied. Zoom range kept subtle (1.0-1.04) to avoid cropping
        important panel content; pan magnitude ±3 % of frame so it reads as
        cinematic motion without feeling shaky.
        """
        move = self._KEN_BURNS_MOVES[sequence_seed % len(self._KEN_BURNS_MOVES)]
        sx, sy, ex, ey, zoom_in = move

        base_zoom = 1.0 + (sequence_seed % 3) * 0.002   # 1.000, 1.002, or 1.004
        zoom_delta = 0.022 + (sequence_seed % 4) * 0.004  # 0.022-0.034

        if zoom_in:
            start_zoom = base_zoom
            end_zoom = min(1.04, base_zoom + zoom_delta)
        else:
            end_zoom = base_zoom
            start_zoom = min(1.04, base_zoom + zoom_delta)

        return [
            CameraPose(panel_id=panel.id, page=panel.page, x=sx, y=sy, zoom=start_zoom),
            CameraPose(panel_id=panel.id, page=panel.page, x=ex, y=ey, zoom=end_zoom),
        ]

    def _build_panel_hold_segments(
        self,
        panel_id: str,
        script_id: str,
        poses: list[CameraPose],
        hold_duration: float,
        sequence_seed: int,
        video_config: VideoConfig,
    ) -> tuple[list[CameraSegment], CameraPose]:
        if len(poses) == 1:
            poses = [poses[0], poses[0]]

        final_pose = poses[-1]
        segment = CameraSegment(
            kind="hold",
            panel_id=panel_id,
            script_id=script_id,
            start=poses[0],
            end=final_pose,
            duration_seconds=max(hold_duration, 3.0),
            easing="cubic",
            transition_style="panel-glide",
        )
        return [segment], final_pose

    def _render_narration_track(
        self,
        plan: TimelinePlan,
        output_path: Path,
        music_config: MusicConfig | None = None,
    ) -> Path:
        """Render the per-panel narration WAVs into one continuous track.

        Hard rule: there must be EXACTLY ONE voice playing at any
        moment. The previous version used `mix[start:end] += audio_data`
        which sums overlapping audio samples - if the timeline plan
        had any event that started before the previous one's audio
        ended, two TTS voices would speak simultaneously, and a third
        could start on top while the first two were still playing. The
        result was the "audio buildup as the video goes on" symptom.

        This implementation:
          1. Sorts events by start_seconds.
          2. As we lay each event into the mix, shifts its start to be
             >= the running max-end of all events placed so far. The
             original event.start_seconds is honored as a MINIMUM, so
             gaps stay where the timeline put them; only overlaps get
             snapped forward to the next free slot.
          3. Writes (=) instead of accumulating (+=) so the narration
             slot is exclusive. SFX still get mixed on top separately.
        """
        sample_rate = self.settings.kokoro_sample_rate
        sorted_events = sorted(plan.audio_events, key=lambda ev: ev.start_seconds)
        # First pass: compute the actual placement for each event,
        # snapping start forward to the running cursor if needed.
        placements: list[tuple[int, np.ndarray]] = []
        cursor_sec = 0.0
        max_end_sec = 0.0
        for event in sorted_events:
            audio_data, audio_rate = sf.read(event.path, dtype="float32")
            if audio_data.ndim > 1:
                audio_data = audio_data.mean(axis=1)
            if audio_rate != sample_rate:
                duration = len(audio_data) / audio_rate
                target_len = int(round(duration * sample_rate))
                audio_data = np.interp(
                    np.linspace(0, len(audio_data) - 1, target_len),
                    np.arange(len(audio_data)),
                    audio_data,
                ).astype(np.float32)
            wanted_start = float(event.start_seconds)
            actual_start = max(wanted_start, cursor_sec)
            start_index = max(int(round(actual_start * sample_rate)), 0)
            placements.append((start_index, audio_data))
            duration_sec = len(audio_data) / float(sample_rate)
            cursor_sec = actual_start + duration_sec
            max_end_sec = max(max_end_sec, cursor_sec)

        plan_end = max(
            (ev.start_seconds + ev.duration_seconds for ev in plan.audio_events),
            default=0.0,
        )
        total_duration = max(plan.total_duration_seconds, plan_end, max_end_sec, 0.5)
        total_samples = max(int(math.ceil(total_duration * sample_rate)), sample_rate // 2)
        mix = np.zeros(total_samples, dtype=np.float32)
        for start_index, audio_data in placements:
            end_index = start_index + len(audio_data)
            if end_index > len(mix):
                mix = np.pad(mix, (0, end_index - len(mix)))
            # Exclusive narration slot: replace, do not accumulate.
            # Two voices NEVER overlap. SFX still gets added below.
            mix[start_index:end_index] = audio_data

        # Mix in procedural transition whooshes at panel boundaries
        if music_config is not None and getattr(music_config, "sfx_enabled", True):
            sfx_volume = float(getattr(music_config, "sfx_volume", 0.055))
            sfx_mix = self._generate_sfx_mix(plan, total_samples, sample_rate, sfx_volume)
            if len(sfx_mix) > len(mix):
                mix = np.pad(mix, (0, len(sfx_mix) - len(mix)))
            mix[:len(sfx_mix)] += sfx_mix

        peak = float(np.max(np.abs(mix))) if mix.size else 0.0
        if peak > 0.98:
            mix = mix / peak * 0.96
        sf.write(output_path, mix, sample_rate)
        return output_path

    def _generate_sfx_mix(
        self,
        plan: TimelinePlan,
        total_samples: int,
        sample_rate: int,
        volume: float,
    ) -> np.ndarray:
        """Generate procedural panel-transition whoosh SFX mixed into a single buffer.

        Each hold→hold transition gets a short (80 ms) band-filtered noise burst
        with a quick rise-fall envelope - creates a subtle cinematic 'whoosh'
        feeling at panel cuts.
        """
        sfx_buf = np.zeros(total_samples, dtype=np.float32)
        rng = np.random.default_rng(42)  # deterministic per render

        # Collect panel-boundary timestamps (when each hold segment ends)
        cursor = 0.0
        transitions: list[float] = []
        hold_segs = [s for s in plan.segments if s.kind == "hold"]
        for seg in hold_segs:
            cursor += seg.duration_seconds
            transitions.append(cursor)
        transitions = transitions[:-1]  # skip very last (end of video)

        # Whoosh parameters
        whoosh_dur = 0.08   # 80 ms
        whoosh_samples = int(whoosh_dur * sample_rate)
        t = np.linspace(0, whoosh_dur, whoosh_samples, endpoint=False)

        # Band-pass mask indices in FFT space for 700-3000 Hz
        freqs = np.fft.rfftfreq(whoosh_samples, d=1.0 / sample_rate)
        low_cut, high_cut = 700.0, 3000.0

        for ts in transitions:
            # Place whoosh so it peaks 20 ms before the cut (feels like a lead-in)
            onset = ts - 0.02
            start_idx = int(round(onset * sample_rate))
            if start_idx < 0 or start_idx + whoosh_samples > total_samples:
                continue

            # White noise → band-pass
            noise = rng.normal(0, 1.0, whoosh_samples).astype(np.float32)
            spectrum = np.fft.rfft(noise)
            mask = (freqs >= low_cut) & (freqs <= high_cut)
            spectrum[~mask] = 0
            bp_noise = np.fft.irfft(spectrum, n=whoosh_samples).astype(np.float32)

            # Amplitude envelope: quick attack (30%), gentle tail (70%)
            env = np.where(t < whoosh_dur * 0.3, t / (whoosh_dur * 0.3),
                           1.0 - (t - whoosh_dur * 0.3) / (whoosh_dur * 0.7))
            env = np.clip(env, 0, 1).astype(np.float32)

            whoosh = bp_noise * env
            peak = float(np.max(np.abs(whoosh))) or 1.0
            whoosh = (whoosh / peak) * volume

            sfx_buf[start_idx: start_idx + whoosh_samples] += whoosh

        return sfx_buf

    def _render_timeline_video(
        self,
        plan: TimelinePlan,
        project_dir: Path,
        panel_assets: dict[str, PanelRenderAsset],
        output_path: Path,
        video_config: VideoConfig,
        progress_callback: Callable[[float, str], None] | None = None,
        cancel_callback: Callable[[], None] | None = None,
    ) -> None:
        if not plan.segments:
            raise ValueError("The camera plan did not contain any segments.")
        cache_dir = ensure_dir(self._shared_video_cache / "clips")
        hold_segments = [segment for segment in plan.segments if segment.kind == "hold"]
        transition_segments = [segment for segment in plan.segments if segment.kind == "crossfade"]
        clip_paths: list[Path] = []
        clip_durations: list[float] = []
        total_clips = max(len(hold_segments), 1)
        audio_event_by_script = {
            event.panel_id: event
            for event in plan.audio_events
            if float(event.duration_seconds) > 0
        }

        # Parallel per-panel clip render. Each ffmpeg call spawns its own
        # encoder (h264_videotoolbox on Apple Silicon, libx264 elsewhere);
        # running 4 at a time keeps the Media Engine fed without
        # oversubscribing the CPU. The serial version this replaced was
        # the bottleneck on cold-cache renders (~4 clips/min); 4-way
        # parallel pushes that to ~15-20 clips/min.
        #
        # We materialize the work list first (because we need clip_paths
        # to come out in segment order for the merge step), then dispatch
        # in parallel and gather by original index.
        worker_count = max(1, min(
            int(getattr(self.settings, "video_clip_render_workers", 4)),
            max(1, os.cpu_count() or 1),
        ))

        work_items: list[tuple[int, Any, Any, float]] = []  # (index, asset, segment, caption_dur)
        for index, segment in enumerate(hold_segments, start=1):
            asset = panel_assets.get(segment.panel_id)
            if asset is None:
                raise FileNotFoundError(f"Missing panel asset for {segment.panel_id}")
            segment_audio = audio_event_by_script.get(segment.script_id or "")
            caption_dur = min(
                float(segment_audio.duration_seconds) if segment_audio is not None else segment.duration_seconds,
                segment.duration_seconds,
            )
            work_items.append((index, asset, segment, caption_dur))

        clip_results: dict[int, tuple[Path, float]] = {}
        completed_count = 0
        completed_lock = __import__("threading").Lock()

        def _render_one(item: tuple[int, Any, Any, float]) -> tuple[int, Path, float]:
            nonlocal completed_count
            idx, asset, segment, caption_dur = item
            if cancel_callback:
                cancel_callback()
            clip_path = self._render_panel_clip(
                asset,
                segment,
                cache_dir,
                video_config,
                caption_audio_path=None,
                caption_duration_seconds=caption_dur,
            )
            with completed_lock:
                completed_count += 1
                done = completed_count
            if progress_callback:
                progress_callback(
                    8 + (done / total_clips) * 72,
                    f"Prepared panel clip {done} of {total_clips}",
                )
            return idx, clip_path, max(segment.duration_seconds, 0.1)

        if worker_count <= 1 or len(work_items) <= 1:
            for item in work_items:
                idx, path, dur = _render_one(item)
                clip_results[idx] = (path, dur)
        else:
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=worker_count) as executor:
                for idx, path, dur in executor.map(_render_one, work_items):
                    clip_results[idx] = (path, dur)

        # Reassemble in segment order.
        for idx in sorted(clip_results.keys()):
            path, dur = clip_results[idx]
            clip_paths.append(path)
            clip_durations.append(dur)

        if progress_callback:
            progress_callback(84, "Merging panel timeline")
        self._merge_panel_clips(
            clip_paths,
            clip_durations,
            transition_segments,
            output_path,
            video_config,
            progress_callback=progress_callback,
            cancel_callback=cancel_callback,
        )
        if progress_callback:
            progress_callback(92, "Panel timeline ready")

    def _render_panel_clip(
        self,
        asset: PanelRenderAsset,
        segment: CameraSegment,
        cache_dir: Path,
        video_config: VideoConfig,
        caption_audio_path: Path | None = None,
        caption_duration_seconds: float | None = None,
    ) -> Path:
        duration = max(segment.duration_seconds, 0.1)
        caption_start = 0.0
        caption_target_duration = max(min(float(caption_duration_seconds or duration), duration), 0.1)
        caption_events: list[tuple[float, float, str]] = []
        if self._captions_enabled(video_config):
            if caption_audio_path is not None:
                caption_events = self._chunked_caption_events_from_audio(
                    caption_audio_path,
                    duration,
                    asset.narration,
                )
                if caption_events:
                    caption_start = caption_events[0][0]
                    caption_target_duration = max(caption_events[-1][1] - caption_start, 0.1)
            if not caption_events:
                caption_events = self._chunked_caption_events(caption_start, caption_target_duration, asset.narration)
        card_width, card_height = self._image_size(asset.card_path)
        if self._captions_enabled(video_config):
            max_card_width = min(video_config.width - 180, 820)
            max_card_height = min(video_config.height - 620, 1240)
        else:
            max_card_width = min(video_config.width - 220, 1500)
            max_card_height = min(video_config.height - 220, 980)
        base_fit_scale = min(
            max_card_width / max(card_width, 1),
            max_card_height / max(card_height, 1),
            1.0,
        )
        # Channel watermark — baked into each per-panel clip during the
        # cheap render pass, replacing the old dedicated _apply_channel_watermark
        # step that re-encoded the entire 2-hour final video just for a
        # corner overlay (~80 min on libx264, ~7 min even with videotoolbox).
        # Per-clip overlay adds negligible time because the encoder runs
        # either way.
        channel_wm = self._resolve_channel_watermark(video_config)
        channel_wm_path: Path | None = channel_wm[0] if channel_wm else None
        channel_wm_text: str = channel_wm[1] if channel_wm else ""

        cache_payload = {
            "image_hash": self._file_content_hash(asset.image_path),
            "card_hash": self._file_content_hash(asset.card_path),
            "render_strategy": "fullscreen_cover_v2" if self._panel_layout_is_fullscreen(video_config) else "card_layout_v2_bgplate",
            # Watermark text + height go into the cache key so changing the
            # @handle or output resolution invalidates only the affected
            # clips. When disabled, the field is omitted to preserve the
            # existing cache hash for projects that never used it.
            **({"channel_watermark": channel_wm_text} if channel_wm_text else {}),
            "duration": round(duration, 3),
            "panel_layout": getattr(video_config.panel_layout, "value", str(video_config.panel_layout)),
            "caption_start": round(caption_start, 3),
            "caption_duration": round(caption_target_duration, 3),
            "caption_audio_hash": self._file_content_hash(caption_audio_path) if caption_audio_path and caption_audio_path.exists() else None,
            "start": {"x": round(segment.start.x, 4), "y": round(segment.start.y, 4), "zoom": round(segment.start.zoom, 4)},
            "end": {"x": round(segment.end.x, 4), "y": round(segment.end.y, 4), "zoom": round(segment.end.zoom, 4)},
            "video": {"width": video_config.width, "height": video_config.height, "fps": video_config.fps},
            "captions": [
                {
                    "start": round(start_seconds, 3),
                    "end": round(end_seconds, 3),
                    "text": chunk,
                }
                for start_seconds, end_seconds, chunk in caption_events
            ],
        }
        digest = hashlib.sha1(json.dumps(cache_payload, sort_keys=True).encode("utf-8")).hexdigest()[:12]
        output_path = cache_dir / f"{digest}.mp4"
        if output_path.exists() and self._cache_duration_matches(output_path, duration):
            return output_path
        output_path.unlink(missing_ok=True)
        temp_output = cache_dir / f"{digest}.tmp.mp4"
        temp_output.unlink(missing_ok=True)

        duration_expr = f"{duration:.3f}"
        start_zoom = segment.start.zoom
        end_zoom = segment.end.zoom
        zoom_delta = end_zoom - start_zoom
        x_expr = "(W-w)/2"
        y_expr = "(H-h)/2"
        zoom_expr = f"({start_zoom:.5f})+({zoom_delta:.5f})*min(t\\,{duration_expr})/{duration_expr}"
        scale_expr = f"({base_fit_scale:.5f})*({zoom_expr})"
        fullscreen_layout = self._panel_layout_is_fullscreen(video_config)
        if fullscreen_layout:
            filter_steps = [
                f"[0:v]scale={video_config.width}:{video_config.height}:force_original_aspect_ratio=increase,"
                f"crop={video_config.width}:{video_config.height},setsar=1[base]",
            ]
        else:
            # Background is now precomputed in PIL ONCE per panel (see
            # _panel_background_plate) and saved as a 1920x1080 JPG. By
            # passing that as input[0], the per-frame ffmpeg filter for
            # the bg branch collapses to setsar=1 - no scale, no blur,
            # no colormix. Prior versions ran boxblur+scale+crop+colormix
            # on every one of ~480 frames per clip on the SAME static
            # image, which dominated the per-clip encode time.
            filter_steps = [
                "[0:v]setsar=1[bg]",
                f"[1:v]scale='iw*{scale_expr}':'ih*{scale_expr}':eval=frame[fg];"
                f"[bg][fg]overlay=x='{x_expr}':y='{y_expr}':format=auto[base]",
            ]
        # For card layouts, input[0] is the precomputed blurred bg plate
        # so ffmpeg doesn't redo blur/scale/colormix on every frame. For
        # fullscreen layouts we still feed the raw panel image and let
        # ffmpeg do the scale+crop (no blur involved).
        bg_input_path: Path
        if fullscreen_layout:
            bg_input_path = asset.image_path
        else:
            bg_input_path = self._panel_background_plate(asset, video_config)
        command = [
            self.settings.ffmpeg_binary,
            "-y",
            "-loop",
            "1",
            "-framerate",
            str(video_config.fps),
            "-t",
            duration_expr,
            "-i",
            str(bg_input_path),
        ]
        if not fullscreen_layout:
            command.extend(
                [
                    "-loop",
                    "1",
                    "-framerate",
                    str(video_config.fps),
                    "-t",
                    duration_expr,
                    "-i",
                    str(asset.card_path),
                ]
            )
        output_label = "[base]"
        next_input_index = 1 if fullscreen_layout else 2
        if caption_events:
            overlay_cache_dir = ensure_dir(self._shared_video_cache / "caption_overlays")
            for event_index, (start_seconds, end_seconds, chunk) in enumerate(caption_events, start=0):
                overlay_path = self._prepare_subtitle_overlay_image(
                    overlay_cache_dir,
                    f"{asset.panel_id}_{next_input_index + event_index}",
                    chunk,
                    video_config,
                )
                command.extend(
                    [
                        "-loop",
                        "1",
                        "-framerate",
                        str(video_config.fps),
                        "-t",
                        duration_expr,
                        "-i",
                        str(overlay_path),
                    ]
                )
                previous_label = output_label
                input_index = next_input_index + event_index
                output_label = f"[cap{input_index}]"
                enable_expr = f"between(t\\,{start_seconds:.3f}\\,{end_seconds:.3f})"
                filter_steps.append(
                    f"{previous_label}[{input_index}:v]overlay=x=0:y=0:format=auto:enable='{enable_expr}'{output_label}"
                )
            next_input_index += len(caption_events)

        # Channel handle watermark, baked in here so the dedicated full-video
        # re-encode pass goes away entirely. Same opacity/margin/scale rules
        # the old _apply_channel_watermark used.
        if channel_wm_path is not None:
            wm_width = max(120, int(video_config.width * 0.18))
            margin = max(12, int(min(video_config.width, video_config.height) * 0.018))
            opacity = 0.32
            command.extend(
                [
                    "-loop",
                    "1",
                    "-framerate",
                    str(video_config.fps),
                    "-t",
                    duration_expr,
                    "-i",
                    str(channel_wm_path),
                ]
            )
            wm_input_index = next_input_index
            previous_label = output_label
            output_label = f"[chwm{wm_input_index}]"
            filter_steps.append(
                f"[{wm_input_index}:v]scale={wm_width}:-1,format=rgba,"
                f"colorchannelmixer=aa={opacity}[chwmsrc{wm_input_index}];"
                f"{previous_label}[chwmsrc{wm_input_index}]overlay="
                f"W-w-{margin}:H-h-{margin}:format=auto{output_label}"
            )
            next_input_index += 1

        filter_complex = ";".join(filter_steps)
        command.extend(
            [
                "-filter_complex",
                filter_complex,
                "-map",
                output_label,
                "-an",
                *(
                    (
                        "-c:v",
                        "libx264",
                        "-preset",
                        "veryfast",
                        "-crf",
                        "17",
                        "-pix_fmt",
                        "yuv420p",
                        "-movflags",
                        "+faststart",
                    )
                    if fullscreen_layout
                    else self._video_encoder_args(intermediate=True)
                ),
                str(temp_output),
            ]
        )
        self._run_ffmpeg(command)
        os.replace(temp_output, output_path)
        return output_path

    def _merge_panel_clips(
        self,
        clip_paths: list[Path],
        clip_durations: list[float],
        transition_segments: list[CameraSegment],
        output_path: Path,
        video_config: VideoConfig,
        progress_callback: Callable[[float, str], None] | None = None,
        cancel_callback: Callable[[], None] | None = None,
    ) -> Path:
        if not clip_paths:
            raise ValueError("No panel clips were generated for the video timeline.")
        merged_cache_dir = ensure_dir(self._shared_video_cache / "merged")
        expected_duration = max(
            sum(clip_durations) - sum(segment.duration_seconds for segment in transition_segments),
            0.1,
        )
        cache_payload = {
            "clips": [
                {
                    "hash": self._file_content_hash(path),
                    "duration": round(duration, 3),
                }
                for path, duration in zip(clip_paths, clip_durations, strict=False)
            ],
            "video": {"width": video_config.width, "height": video_config.height, "fps": video_config.fps},
        }
        digest = hashlib.sha1(json.dumps(cache_payload, sort_keys=True).encode("utf-8")).hexdigest()[:12]
        cached_output = merged_cache_dir / f"merged_{digest}.mp4"
        if cached_output.exists() and self._cache_duration_matches(cached_output, expected_duration, tolerance=1.0):
            shutil.copy2(cached_output, output_path)
            return output_path
        cached_output.unlink(missing_ok=True)
        temp_cached_output = merged_cache_dir / f"merged_{digest}.tmp.mp4"
        temp_cached_output.unlink(missing_ok=True)

        if len(clip_paths) == 1:
            shutil.copy2(clip_paths[0], temp_cached_output)
            os.replace(temp_cached_output, cached_output)
            shutil.copy2(cached_output, output_path)
            return output_path

        concat_manifest = merged_cache_dir / f"merged_{digest}.txt"
        manifest_lines: list[str] = []
        for clip_path in clip_paths:
            escaped_path = clip_path.as_posix().replace("'", "'\\''")
            manifest_lines.append(f"file '{escaped_path}'")
        concat_manifest.write_text("\n".join(manifest_lines), encoding="utf-8")
        command = [
            self.settings.ffmpeg_binary,
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(concat_manifest),
            "-c",
            "copy",
            str(temp_cached_output),
        ]
        self._run_ffmpeg_with_progress(
            command,
            expected_duration_seconds=expected_duration,
            progress_callback=progress_callback,
            cancel_callback=cancel_callback,
            start_progress=84,
            end_progress=92,
            message="Merging panel timeline",
        )
        os.replace(temp_cached_output, cached_output)
        shutil.copy2(cached_output, output_path)
        return output_path

    def _render_segment_frame(
        self,
        project_dir: Path,
        segment: CameraSegment,
        progress: float,
        video_config: VideoConfig,
        panel_assets: dict[str, PanelRenderAsset],
    ) -> Image.Image:
        eased = self._ease(progress, segment.easing)
        if segment.kind == "crossfade":
            outgoing = self._render_pose_frame(project_dir, segment.start, video_config, panel_assets)
            incoming = self._render_pose_frame(project_dir, segment.end, video_config, panel_assets)
            frame = self._crossfade_slide_frame(outgoing, incoming, eased, video_config, segment.transition_style)
            subtitle_panel_id = segment.end.panel_id if eased >= 0.5 else segment.start.panel_id
            return self._overlay_subtitle(frame, panel_assets.get(subtitle_panel_id), video_config)

        pose = self._interpolate_pose(segment.start, segment.end, eased)
        frame = self._render_pose_frame(project_dir, pose, video_config, panel_assets)
        return self._overlay_subtitle(frame, panel_assets.get(segment.panel_id), video_config)

    def _render_pose_frame(
        self,
        project_dir: Path,
        pose: CameraPose,
        video_config: VideoConfig,
        panel_assets: dict[str, PanelRenderAsset],
    ) -> Image.Image:
        asset = panel_assets.get(pose.panel_id)
        if asset is None:
            raise FileNotFoundError(f"Missing panel render asset for {pose.panel_id}")
        image = self._load_page_image(str(asset.image_path))
        return self._compose_panel_frame(image, pose, video_config)

    def _mux_video_with_audio(self, video_path: Path, narration_path: Path, output_path: Path) -> None:
        command = [
            self.settings.ffmpeg_binary,
            "-y",
            "-i",
            str(video_path),
            "-i",
            str(narration_path),
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-shortest",
            str(output_path),
        ]
        self._run_ffmpeg(command)

    def _write_camera_manifest(self, path: Path, plan: TimelinePlan) -> None:
        cursor = 0.0
        payload = []
        for segment in plan.segments:
            payload.append(
                {
                    "panel_id": segment.panel_id,
                    "script_id": segment.script_id,
                    "kind": segment.kind,
                    "transition_style": segment.transition_style,
                    "start_seconds": round(cursor, 3),
                    "duration_seconds": round(segment.duration_seconds, 3),
                    "easing": segment.easing,
                    "from": {
                        "page": segment.start.page,
                        "x": round(segment.start.x, 2),
                        "y": round(segment.start.y, 2),
                        "zoom": round(segment.start.zoom, 4),
                    },
                    "to": {
                        "page": segment.end.page,
                        "x": round(segment.end.x, 2),
                        "y": round(segment.end.y, 2),
                        "zoom": round(segment.end.zoom, 4),
                    },
                }
            )
            cursor += segment.duration_seconds
        write_json(path, payload)

    @staticmethod
    def _chapter_title_from_text(text: str, fallback_index: int) -> str:
        """Extract a short chapter title from the first sentence of narration text."""
        if not text:
            return f"Part {fallback_index}"
        # Take first sentence (up to first . ! ?)
        import re as _re
        match = _re.split(r"(?<=[.!?])\s", text.strip())
        first_sentence = match[0].strip() if match else text.strip()
        # Truncate to 60 chars for YouTube chapter readability
        if len(first_sentence) > 60:
            words = first_sentence.split()
            truncated = ""
            for word in words:
                if len(truncated) + len(word) + 1 > 57:
                    break
                truncated = (truncated + " " + word).strip()
            first_sentence = truncated + "…"
        return first_sentence or f"Part {fallback_index}"

    def _write_chapter_markers(self, path: Path, plan: TimelinePlan, output_name: str) -> None:
        """Write YouTube-ready chapter markers as chapters.json.

        Format:
        [
          {"index": 1, "title": "...", "start_seconds": 0.0, "youtube_timestamp": "0:00"},
          ...
        ]

        The ``youtube_description`` key contains the full chapter list formatted
        for pasting directly into a YouTube video description.
        """
        markers = plan.chapter_markers
        if not markers:
            return

        def _fmt_timestamp(seconds: float) -> str:
            total = max(0, int(seconds))
            h, rem = divmod(total, 3600)
            m, s = divmod(rem, 60)
            if h:
                return f"{h}:{m:02d}:{s:02d}"
            return f"{m}:{s:02d}"

        payload = [
            {
                "index": m.index,
                "title": m.title,
                "start_seconds": round(m.start_seconds, 2),
                "youtube_timestamp": _fmt_timestamp(m.start_seconds),
            }
            for m in markers
        ]

        description_lines = [f"{_fmt_timestamp(m.start_seconds)} {m.title}" for m in markers]
        result = {
            "video": output_name,
            "total_duration_seconds": round(plan.total_duration_seconds, 2),
            "markers": payload,
            "youtube_description": "\n".join(description_lines),
        }
        write_json(path, result)

    def _panel_time(self, script_line: str, audio_duration: float, manual_duration: float | None) -> float:
        """Compute the on-screen time for one panel.

        Audio is the source of truth when available: the visual must
        stay on the panel for at least as long as the audio takes to
        read aloud. Anything shorter means the next panel's narration
        starts while the previous panel is still visible (and worse:
        the audio track keeps reading the previous panel's narration
        while the visual advances - that's the "audio doesn't line up
        with the panels" symptom).

        Resolution order:
          1. If we have audio, use it + 80 ms tail. This wins even
             when a stale `manual_duration` exists - those values get
             auto-computed once at script time and don't get refreshed
             when narration is re-rendered with a different TTS voice
             (Kokoro -> Edge produces longer reads).
          2. If we don't have audio but a `manual_duration` is set
             (user dragged the inspector slider, OR an old auto-value
             is still around), honor it.
          3. Last resort: word-count heuristic.

        Floor: 0.45 s - any panel briefer than that registers as a
        flicker rather than a panel.
        """
        if audio_duration and audio_duration > 0.01:
            tail = 0.08
            target = float(audio_duration) + tail
            return round(max(target, 0.45), 2)
        if manual_duration is not None:
            return round(max(float(manual_duration), 0.45), 2)
        words = len([word for word in script_line.split() if word.strip()])
        heuristic = max(1.5, words * 0.32) if words else 1.2
        return round(heuristic, 2)

    def _same_page_transition_duration(
        self,
        previous: CameraPose,
        current: CameraPose,
        page_width: int,
        page_height: int,
    ) -> float:
        delta_y = abs(current.y - previous.y) / max(page_height, 1)
        delta_x = abs(current.x - previous.x) / max(page_width, 1)
        zoom_change = abs(math.log(max(current.zoom, 1.0) / max(previous.zoom, 1.0)))
        duration = 0.3 + (delta_y * 0.85) + (delta_x * 0.25) + (zoom_change * 0.2)
        return round(min(max(duration, 0.3), 1.1), 2)

    def _expanded_bbox(self, panel: PanelBox, page_width: int, page_height: int) -> tuple[float, float, float, float]:
        pad_x = panel.width * 0.08
        pad_y = panel.height * 0.08
        x1 = max(panel.x - pad_x, 0.0)
        y1 = max(panel.y - pad_y, 0.0)
        x2 = min(panel.x + panel.width + pad_x, float(page_width))
        y2 = min(panel.y + panel.height + pad_y, float(page_height))
        return x1, y1, x2, y2

    def _pose_from_crop(
        self,
        page: int,
        center_x: float,
        center_y: float,
        crop_width: float,
        page_width: int,
        page_height: int,
        aspect: float,
    ) -> CameraPose:
        crop_width = min(max(crop_width, page_width * 0.18), float(page_width))
        crop_height = crop_width / max(aspect, 0.001)

        half_width = crop_width / 2
        half_height = crop_height / 2

        if crop_width >= page_width:
            center_x = page_width / 2
        else:
            center_x = min(max(center_x, half_width), page_width - half_width)

        if crop_height >= page_height:
            center_y = page_height / 2
        else:
            center_y = min(max(center_y, half_height), page_height - half_height)

        zoom = page_width / crop_width
        return CameraPose(page=page, x=center_x, y=center_y, zoom=max(zoom, 1.0))

    def _hold_pose(
        self,
        pose: CameraPose,
        seed: int,
        video_config: VideoConfig,
    ) -> CameraPose:
        direction = -1.0 if seed % 2 else 1.0
        x_drift = 0.008 * direction
        y_drift = 0.01 if seed % 3 else -0.01
        zoom_factor = 1.008 + (0.006 * ((seed % 2) / 2))
        return CameraPose(
            panel_id=pose.panel_id,
            page=pose.page,
            x=min(max(pose.x + x_drift, 0.45), 0.55),
            y=min(max(pose.y + y_drift, 0.44), 0.56),
            zoom=pose.zoom * min(zoom_factor, 1.012),
        )

    def _interpolate_pose(self, start: CameraPose, end: CameraPose, progress: float) -> CameraPose:
        return CameraPose(
            panel_id=end.panel_id if progress >= 0.5 else start.panel_id,
            page=end.page if progress >= 0.5 else start.page,
            x=self._lerp(start.x, end.x, progress),
            y=self._lerp(start.y, end.y, progress),
            zoom=self._lerp(start.zoom, end.zoom, progress),
        )

    def _crossfade_slide_frame(
        self,
        outgoing: Image.Image,
        incoming: Image.Image,
        progress: float,
        video_config: VideoConfig,
        transition_style: str,
    ) -> Image.Image:
        shift_y = int(video_config.height * 0.045)
        shift_x = int(video_config.width * 0.045)
        outgoing_canvas = Image.new("RGB", (video_config.width, video_config.height), video_config.background_color)
        incoming_canvas = Image.new("RGB", (video_config.width, video_config.height), video_config.background_color)

        out_x = 0
        out_y = 0
        in_x = 0
        in_y = 0
        if transition_style == "slide-left":
            out_x = -int(shift_x * progress)
            in_x = int(shift_x * (1.0 - progress))
        elif transition_style == "slide-right":
            out_x = int(shift_x * progress)
            in_x = -int(shift_x * (1.0 - progress))
        elif transition_style == "slide-up":
            out_y = -int(shift_y * progress)
            in_y = int(shift_y * (1.0 - progress))
        elif transition_style == "slide-down":
            out_y = int(shift_y * progress)
            in_y = -int(shift_y * (1.0 - progress))

        outgoing_canvas.paste(outgoing, (out_x, out_y))
        incoming_canvas.paste(incoming, (in_x, in_y))
        return Image.blend(outgoing_canvas, incoming_canvas, progress)

    def _compose_panel_frame(self, panel_image: Image.Image, pose: CameraPose, video_config: VideoConfig) -> Image.Image:
        if self._panel_layout_is_fullscreen(video_config):
            return self._fullscreen_panel_frame(panel_image, pose, video_config).convert("RGB")
        frame = self._blurred_background(panel_image, video_config)
        panel_card = self._foreground_panel(panel_image, pose, video_config)
        panel_x = (video_config.width - panel_card.width) // 2
        if video_config.orientation.value == "vertical":
            top_padding = 110
            caption_lane = 290
            usable_height = max(video_config.height - top_padding - caption_lane, panel_card.height)
            panel_y = top_padding + max((usable_height - panel_card.height) // 2, 0)
        else:
            panel_y = (video_config.height - panel_card.height) // 2
            safe_left = (video_config.width - 1600) // 2
            safe_top = (video_config.height - 900) // 2
            safe_right = safe_left + 1600 - panel_card.width
            safe_bottom = safe_top + 900 - panel_card.height
            panel_x = min(max(panel_x, safe_left), max(safe_left, safe_right))
            panel_y = min(max(panel_y, safe_top), max(safe_top, safe_bottom))
        frame.alpha_composite(panel_card, (panel_x, panel_y))
        return frame.convert("RGB")

    def _fullscreen_panel_frame(self, panel_image: Image.Image, pose: CameraPose, video_config: VideoConfig) -> Image.Image:
        image = panel_image.convert("RGB")
        cover_width, cover_height = self._fit_cover(image.width, image.height, video_config.width, video_config.height)
        cover_scale = cover_width / max(image.width, 1)
        zoom = max(pose.zoom, 1.0)
        render_width = max(1, int(round(image.width * cover_scale * zoom)))
        render_height = max(1, int(round(image.height * cover_scale * zoom)))
        rendered = image.resize((render_width, render_height), Image.Resampling.LANCZOS)
        left = max((render_width - video_config.width) // 2, 0)
        top = max((render_height - video_config.height) // 2, 0)
        return rendered.crop((left, top, left + video_config.width, top + video_config.height)).convert("RGBA")

    def _blurred_background(self, panel_image: Image.Image, video_config: VideoConfig) -> Image.Image:
        image = panel_image.convert("RGB")
        cover = self._fit_cover(image.size[0], image.size[1], video_config.width, video_config.height)
        background = image.resize(cover, Image.Resampling.LANCZOS)
        left = max((background.width - video_config.width) // 2, 0)
        top = max((background.height - video_config.height) // 2, 0)
        background = background.crop((left, top, left + video_config.width, top + video_config.height))
        background = background.filter(ImageFilter.GaussianBlur(radius=40))
        background = ImageEnhance.Brightness(background).enhance(0.6)
        return background.convert("RGBA")

    def _panel_background_plate(self, asset: "PanelRenderAsset", video_config: VideoConfig) -> Path:
        """Precompute the blurred/darkened bg plate ONCE per panel and cache.

        Before this existed, every ffmpeg clip render ran boxblur+colormix
        on the same static panel image, 480 times in a row (24 fps x 20s).
        With h264_videotoolbox eating ~50% CPU and the blur filter eating
        the other ~40%, each 20-second clip took 90+ seconds to encode.
        Pre-rendering the bg plate to a 1920x1080 JPG and feeding that as
        a static input lets ffmpeg short-circuit the bg branch entirely:
        scale/crop/blur/colormix all collapse to one PIL call per panel.
        """
        cache_dir = ensure_dir(self._shared_video_cache / "bg_plates")
        image_hash = self._file_content_hash(asset.image_path) or "noimg"
        key = (
            f"{image_hash}_"
            f"{video_config.width}x{video_config.height}_"
            f"blur40_dark60_v1"
        )
        out_path = cache_dir / f"{key}.jpg"
        if out_path.exists():
            return out_path
        with Image.open(asset.image_path) as raw:
            plate = self._blurred_background(raw, video_config).convert("RGB")
        tmp = cache_dir / f"{key}.tmp.jpg"
        plate.save(tmp, format="JPEG", quality=82, optimize=False)
        tmp.replace(out_path)
        return out_path

    def _foreground_panel(self, panel_image: Image.Image, pose: CameraPose, video_config: VideoConfig) -> Image.Image:
        max_width = min(1400, video_config.width - 320)
        max_height = min(900, video_config.height - 180)
        fit_width, fit_height = self._fit_inside(panel_image.width, panel_image.height, max_width, max_height)
        scale = pose.zoom
        render_width = max(1, int(round(fit_width * scale)))
        render_height = max(1, int(round(fit_height * scale)))
        rendered = panel_image.resize((render_width, render_height), Image.Resampling.LANCZOS).convert("RGBA")
        radius = 6
        mask = Image.new("L", (render_width, render_height), 0)
        draw = ImageDraw.Draw(mask)
        draw.rounded_rectangle((0, 0, render_width - 1, render_height - 1), radius=radius, fill=255)
        rounded = Image.new("RGBA", (render_width, render_height), (0, 0, 0, 0))
        rounded.paste(rendered, (0, 0), mask)

        shadow_margin = 28
        shadow = Image.new("RGBA", (render_width + shadow_margin * 2, render_height + shadow_margin * 2), (0, 0, 0, 0))
        shadow_mask = Image.new("L", (render_width, render_height), 0)
        shadow_draw = ImageDraw.Draw(shadow_mask)
        shadow_draw.rounded_rectangle((0, 0, render_width - 1, render_height - 1), radius=radius, fill=180)
        shadow_layer = Image.new("RGBA", (render_width, render_height), (0, 0, 0, 110))
        shadow.paste(shadow_layer, (shadow_margin, shadow_margin), shadow_mask)
        shadow = shadow.filter(ImageFilter.GaussianBlur(radius=18))
        shadow.alpha_composite(rounded, (shadow_margin, shadow_margin))
        return shadow

    @staticmethod
    def _panel_layout_is_fullscreen(video_config: VideoConfig) -> bool:
        return getattr(video_config, "panel_layout", None) == "fullscreen" or getattr(getattr(video_config, "panel_layout", None), "value", None) == "fullscreen"

    def _overlay_subtitle(
        self,
        frame: Image.Image,
        asset: PanelRenderAsset | None,
        video_config: VideoConfig,
    ) -> Image.Image:
        text = str(asset.narration if asset else "").strip()
        if not text:
            return frame
        image = frame.convert("RGBA")
        draw = ImageDraw.Draw(image)
        font = self._subtitle_font(58)
        max_width = 1500
        lines = self._wrap_text(draw, text, font, max_width)
        if not lines:
            return image.convert("RGB")

        line_spacing = 10
        outlines = 4
        line_boxes = [draw.textbbox((0, 0), line, font=font, stroke_width=outlines) for line in lines]
        text_height = sum((box[3] - box[1]) for box in line_boxes) + line_spacing * max(len(lines) - 1, 0)
        safe_left = (video_config.width - 1600) // 2
        safe_top = (video_config.height - 900) // 2
        safe_width = 1600
        safe_height = 900
        y = safe_top + safe_height - text_height - 28

        box_padding_x = 26
        box_padding_y = 16
        widest = max((box[2] - box[0]) for box in line_boxes)
        box_width = min(safe_width - 40, widest + box_padding_x * 2)
        box_height = text_height + box_padding_y * 2
        box_x = safe_left + (safe_width - box_width) // 2
        box_y = y - box_padding_y

        overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
        overlay_draw = ImageDraw.Draw(overlay)
        overlay_draw.rounded_rectangle(
            (box_x, box_y, box_x + box_width, box_y + box_height),
            radius=18,
            fill=(0, 0, 0, 118),
        )
        image = Image.alpha_composite(image, overlay)
        draw = ImageDraw.Draw(image)
        current_y = y
        for line, box in zip(lines, line_boxes, strict=False):
            line_width = box[2] - box[0]
            x = safe_left + (safe_width - line_width) // 2
            draw.text(
                (x, current_y),
                line,
                font=font,
                fill=(255, 255, 255, 255),
                stroke_width=4,
                stroke_fill=(0, 0, 0, 255),
            )
            current_y += (box[3] - box[1]) + line_spacing
        return image.convert("RGB")

    def _wrap_text(
        self,
        draw: ImageDraw.ImageDraw,
        text: str,
        font: ImageFont.FreeTypeFont,
        max_width: int,
        *,
        max_lines: int = 3,
        stroke_width: int = 4,
    ) -> list[str]:
        words = text.split()
        if not words:
            return []
        lines: list[str] = []
        current = words[0]
        for word in words[1:]:
            candidate = f"{current} {word}"
            bbox = draw.textbbox((0, 0), candidate, font=font, stroke_width=stroke_width)
            if bbox[2] - bbox[0] <= max_width:
                current = candidate
            else:
                lines.append(current)
                current = word
        lines.append(current)
        if len(lines) <= max_lines:
            return lines
        return lines[: max_lines - 1] + [" ".join(lines[max_lines - 1 :])]

    def _fit_subtitle_layout(
        self,
        draw: ImageDraw.ImageDraw,
        text: str,
        video_config: VideoConfig,
    ) -> tuple[ImageFont.FreeTypeFont, list[str], int, int, int, int, int]:
        vertical = self._captions_enabled(video_config)
        candidate_sizes = (82, 76, 70, 64, 58, 54) if vertical else (58, 54, 50, 46)
        outline_width = 6 if vertical else 4
        line_spacing = 14 if vertical else 8
        safe_width = min(video_config.width - 96, 900) if vertical else 1600
        safe_left = (video_config.width - safe_width) // 2
        max_width = safe_width - (72 if vertical else 52)
        max_lines = 3

        best_font = self._subtitle_font(candidate_sizes[-1])
        best_lines = self._wrap_text(
            draw,
            text,
            best_font,
            max_width,
            max_lines=max_lines,
            stroke_width=outline_width,
        )
        for size in candidate_sizes:
            font = self._subtitle_font(size)
            lines = self._wrap_text(
                draw,
                text,
                font,
                max_width,
                max_lines=max_lines,
                stroke_width=outline_width,
            )
            if len(lines) <= max_lines:
                best_font = font
                best_lines = lines
                break

        return best_font, best_lines, outline_width, line_spacing, safe_width, safe_left, max_width

    @staticmethod
    @lru_cache(maxsize=4)
    def _subtitle_font(size: int) -> ImageFont.FreeTypeFont:
        candidates = (
            "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
            "/System/Library/Fonts/HelveticaNeue.ttc",
            "/System/Library/Fonts/SFNS.ttf",
            "/System/Library/Fonts/Supplemental/Arial.ttf",
            "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
        )
        for path in candidates:
            try:
                return ImageFont.truetype(path, size)
            except OSError:
                continue
        return ImageFont.load_default()

    @staticmethod
    def _fit_inside(source_width: int, source_height: int, max_width: int, max_height: int) -> tuple[int, int]:
        scale = min(max_width / max(source_width, 1), max_height / max(source_height, 1))
        return max(1, int(round(source_width * scale))), max(1, int(round(source_height * scale)))

    @staticmethod
    def _fit_cover(source_width: int, source_height: int, target_width: int, target_height: int) -> tuple[int, int]:
        scale = max(target_width / max(source_width, 1), target_height / max(source_height, 1))
        return max(1, int(round(source_width * scale))), max(1, int(round(source_height * scale)))

    @staticmethod
    def _panel_transition_style(seed: int) -> str:
        styles = ("crossfade", "slide-left", "slide-right", "slide-up", "slide-down")
        return styles[seed % len(styles)]

    def _ease(self, progress: float, easing: str) -> float:
        value = min(max(progress, 0.0), 1.0)
        if easing == "cubic":
            return value ** 3 if value < 0.5 else 1 - ((-2 * value + 2) ** 3) / 2
        if easing == "smoothstep":
            return value * value * (3 - 2 * value)
        return 4 * value * value * value if value < 0.5 else 1 - ((-2 * value + 2) ** 3) / 2

    def _lerp(self, start: float, end: float, progress: float) -> float:
        return start + (end - start) * progress

    @staticmethod
    @lru_cache(maxsize=64)
    def _image_size(path: Path) -> tuple[int, int]:
        with Image.open(path) as image:
            return image.size

    @staticmethod
    @lru_cache(maxsize=512)
    def _file_content_hash(path: Path | None) -> str | None:
        if path is None or not path.exists():
            return None
        digest = hashlib.sha1()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    @staticmethod
    @lru_cache(maxsize=8)
    def _load_page_image(path: str) -> Image.Image:
        # Be tolerant of partially-written cache entries (rare crash mid-write).
        # If the file is truncated we delete it and re-raise so the caller's
        # regenerate path takes over - better than aborting the whole render.
        try:
            with Image.open(path) as image:
                return image.convert("RGB")
        except OSError as err:
            if "truncated" in str(err).lower():
                try:
                    Path(path).unlink(missing_ok=True)
                except OSError:
                    pass
            raise

    @staticmethod
    @lru_cache(maxsize=4)
    def _vignette_overlay(width: int, height: int) -> Image.Image | None:
        x = np.linspace(-1.0, 1.0, width, dtype=np.float32)
        y = np.linspace(-1.0, 1.0, height, dtype=np.float32)
        xx, yy = np.meshgrid(x, y)
        radius = np.sqrt(xx * xx + yy * yy)
        alpha = np.clip((radius - 0.45) / 0.55, 0.0, 1.0) ** 2
        alpha = (alpha * 48).astype(np.uint8)
        if not np.any(alpha):
            return None

        overlay = np.zeros((height, width, 4), dtype=np.uint8)
        overlay[..., 3] = alpha
        return Image.fromarray(overlay, mode="RGBA")

    def _prepare_panel_card(self, image_path: Path, cache_dir: Path, panel_id: str) -> Path:
        digest = hashlib.sha1(
            json.dumps(
                {
                    "image_hash": self._file_content_hash(image_path),
                    "size": self._image_size(image_path),
                    "strategy": "panel_card_v2",
                },
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()[:12]
        output_path = cache_dir / f"{digest}.png"
        if output_path.exists():
            return output_path

        image = self._load_page_image(str(image_path)).convert("RGBA")
        fit_width, fit_height = self._fit_inside(image.width, image.height, 1400, 900)
        panel = image.resize((fit_width, fit_height), Image.Resampling.LANCZOS)
        radius = 6
        mask = Image.new("L", (fit_width, fit_height), 0)
        mask_draw = ImageDraw.Draw(mask)
        mask_draw.rounded_rectangle((0, 0, fit_width - 1, fit_height - 1), radius=radius, fill=255)
        rounded = Image.new("RGBA", (fit_width, fit_height), (0, 0, 0, 0))
        rounded.paste(panel, (0, 0), mask)

        shadow_margin = 28
        canvas = Image.new("RGBA", (fit_width + shadow_margin * 2, fit_height + shadow_margin * 2), (0, 0, 0, 0))
        shadow_mask = Image.new("L", (fit_width, fit_height), 0)
        shadow_draw = ImageDraw.Draw(shadow_mask)
        shadow_draw.rounded_rectangle((0, 0, fit_width - 1, fit_height - 1), radius=radius, fill=175)
        shadow_layer = Image.new("RGBA", (fit_width, fit_height), (0, 0, 0, 120))
        canvas.paste(shadow_layer, (shadow_margin, shadow_margin), shadow_mask)
        canvas = canvas.filter(ImageFilter.GaussianBlur(radius=18))
        canvas.alpha_composite(rounded, (shadow_margin, shadow_margin))
        canvas.save(output_path)
        return output_path

    # ── Content-safety blur intensities ──────────────────────────────────
    # "borderline" panels (partial nudity, intimate scenes) get a moderate
    # blur - the silhouette is still readable so the story beat works.
    # "explicit" panels are usually skipped via panel.keep=False; this
    # heavier sigma is the fallback for when the user manually force-keeps
    # an explicit panel so nothing demonetizing ever ships unintentionally.
    NSFW_BLUR_SIGMA: ClassVar[int] = 28
    NSFW_BLUR_SIGMA_EXPLICIT: ClassVar[int] = 56

    def _prepare_panel_crop(self, project_dir: Path, panel: PanelBox, cache_dir: Path) -> Path:
        panel = self.store.sanitize_panel_box(project_dir.name, panel)
        page_path = project_dir / "pages" / f"{panel.page:04d}.png"
        if not page_path.exists():
            raise FileNotFoundError(f"Missing page image for panel {panel.id} on page {panel.page}")

        # Compute blur sigma from the panel's content-safety state. We bake
        # this into the cache digest so changing the rating invalidates the
        # cached crop and the new render picks up the blur.
        blur_sigma = 0
        if getattr(panel, "content_blur", False):
            rating = (getattr(panel, "content_rating", None) or "").lower()
            blur_sigma = (
                self.NSFW_BLUR_SIGMA_EXPLICIT
                if rating == "explicit"
                else self.NSFW_BLUR_SIGMA
            )

        digest = hashlib.sha1(
            json.dumps(
                {
                    "page_hash": self._file_content_hash(page_path),
                    "panel": {
                        "page": panel.page,
                        "x": int(panel.x),
                        "y": int(panel.y),
                        "width": int(panel.width),
                        "height": int(panel.height),
                    },
                    "strategy": "panel_crop_v3",
                    "blur_sigma": blur_sigma,
                },
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()[:12]
        output_path = cache_dir / f"{digest}.png"
        if output_path.exists():
            return output_path

        with Image.open(page_path) as image:
            page = image.convert("RGB")
            left = max(int(panel.x), 0)
            top = max(int(panel.y), 0)
            right = min(left + max(int(panel.width), 1), page.width)
            bottom = min(top + max(int(panel.height), 1), page.height)
            if right <= left or bottom <= top:
                left, top, right, bottom = 0, 0, page.width, page.height
            crop = page.crop((left, top, right, bottom))
            if blur_sigma > 0:
                # Apply Gaussian blur in PIL so every downstream surface
                # (card, thumbnail, etc.) gets the blurred version for free.
                crop = crop.filter(ImageFilter.GaussianBlur(radius=blur_sigma))
            crop.save(output_path)
        return output_path

    def _prepare_subtitle_overlay_image(
        self,
        cache_dir: Path,
        panel_id: str,
        narration: str,
        video_config: VideoConfig,
    ) -> Path:
        digest = hashlib.sha1(
            json.dumps(
                {
                    "panel_id": panel_id,
                    "narration": narration,
                    "width": video_config.width,
                    "height": video_config.height,
                },
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()[:12]
        output_path = cache_dir / f"{panel_id}_{digest}.png"
        if output_path.exists():
            return output_path

        image = Image.new("RGBA", (video_config.width, video_config.height), (0, 0, 0, 0))
        text = str(narration or "").strip()
        if not text:
            image.save(output_path)
            return output_path

        draw = ImageDraw.Draw(image)
        vertical = self._captions_enabled(video_config)
        font, lines, outline_width, line_spacing, safe_width, safe_left, max_width = self._fit_subtitle_layout(
            draw,
            text,
            video_config,
        )
        if not lines:
            image.save(output_path)
            return output_path

        line_boxes = [draw.textbbox((0, 0), line, font=font, stroke_width=outline_width) for line in lines]
        text_height = sum((box[3] - box[1]) for box in line_boxes) + line_spacing * max(len(lines) - 1, 0)
        box_padding_x = 36 if vertical else 26
        box_padding_y = 24 if vertical else 16
        widest = min(max((box[2] - box[0]) for box in line_boxes), max_width)
        # In vertical mode, use a stable minimum box width so subtitles don't jump
        # around as chunk lengths vary.
        min_box_width = int(safe_width * 0.6) if vertical else 0
        box_width = min(safe_width, max(widest + box_padding_x * 2, min_box_width))
        box_height = text_height + box_padding_y * 2
        box_x = safe_left + (safe_width - box_width) // 2
        y = video_config.height - box_height - (180 if vertical else 28)

        box_draw = ImageDraw.Draw(image)
        box_y = y
        box_draw.rounded_rectangle(
            (box_x, box_y, box_x + box_width, box_y + box_height),
            radius=22 if vertical else 18,
            fill=(0, 0, 0, 148 if vertical else 118),
        )

        current_y = y + box_padding_y
        box_center_x = box_x + box_width // 2
        for line, box in zip(lines, line_boxes, strict=False):
            line_width = box[2] - box[0]
            x = box_center_x - line_width // 2
            box_draw.text(
                (x, current_y),
                line,
                font=font,
                fill=(255, 255, 255, 255),
                stroke_width=outline_width,
                stroke_fill=(0, 0, 0, 255),
            )
            current_y += (box[3] - box[1]) + line_spacing

        image.save(output_path)
        return output_path

    @staticmethod
    def _captions_enabled(video_config: VideoConfig) -> bool:
        return video_config.orientation.value == "vertical"

    def _subtitle_audio_window(self, audio_path: Path, duration_seconds: float) -> tuple[float, float]:
        try:
            audio_data, sample_rate = sf.read(audio_path, dtype="float32")
        except Exception:
            return (0.0, max(float(duration_seconds), 0.1))
        if audio_data.ndim > 1:
            audio_data = audio_data.mean(axis=1)
        if not len(audio_data) or sample_rate <= 0:
            return (0.0, max(float(duration_seconds), 0.1))

        frame_size = max(int(sample_rate * 0.02), 1)
        frame_count = max(int(math.ceil(len(audio_data) / frame_size)), 1)
        energy = np.zeros(frame_count, dtype=np.float32)
        for index in range(frame_count):
            start = index * frame_size
            end = min(start + frame_size, len(audio_data))
            window = audio_data[start:end]
            if len(window):
                energy[index] = float(np.sqrt(np.mean(np.square(window))))

        peak = float(np.max(energy)) if energy.size else 0.0
        if peak <= 0:
            return (0.0, max(float(duration_seconds), 0.1))
        threshold = max(peak * 0.14, 0.006)
        voiced = np.flatnonzero(energy >= threshold)
        if voiced.size == 0:
            return (0.0, max(float(duration_seconds), 0.1))

        speech_start = (int(voiced[0]) * frame_size) / float(sample_rate)
        speech_end = min(((int(voiced[-1]) + 1) * frame_size) / float(sample_rate), float(duration_seconds))
        anticipation = min(0.08, max(speech_start * 0.45, 0.02))
        release = min(0.05, max((float(duration_seconds) - speech_end) * 0.35, 0.01))
        adjusted_start = max(speech_start - anticipation, 0.0)
        adjusted_end = max(min(speech_end + release, float(duration_seconds)), adjusted_start + 0.1)
        return (round(adjusted_start, 3), round(adjusted_end, 3))

    def _chunked_caption_events_from_audio(
        self,
        audio_path: Path,
        duration_seconds: float,
        text: str,
    ) -> list[tuple[float, float, str]]:
        chunks = self._caption_chunks(text)
        if not chunks:
            return []
        try:
            audio_data, sample_rate = sf.read(audio_path, dtype="float32")
        except Exception:
            return self._chunked_caption_events(0.0, duration_seconds, text)
        if audio_data.ndim > 1:
            audio_data = audio_data.mean(axis=1)
        if not len(audio_data) or sample_rate <= 0:
            return self._chunked_caption_events(0.0, duration_seconds, text)

        envelope = np.abs(audio_data)
        smoothing_window = max(int(sample_rate * 0.012), 1)
        if smoothing_window > 1:
            kernel = np.ones(smoothing_window, dtype=np.float32) / smoothing_window
            envelope = np.convolve(envelope, kernel, mode="same")

        peak = float(np.max(envelope)) if envelope.size else 0.0
        if peak <= 0:
            return self._chunked_caption_events(0.0, duration_seconds, text)

        threshold = max(peak * 0.13, 0.0035)
        active = envelope.copy()
        active[active < threshold] = 0.0
        voiced = np.flatnonzero(active > 0.0)
        if voiced.size == 0:
            return self._chunked_caption_events(0.0, duration_seconds, text)

        speech_start = min(max((int(voiced[0]) / sample_rate) + 0.02, 0.0), max(float(duration_seconds) - 0.1, 0.0))
        speech_end = min(((int(voiced[-1]) + 1) / sample_rate) + 0.02, float(duration_seconds))
        if speech_end - speech_start < 0.1:
            return self._chunked_caption_events(speech_start, max(speech_end - speech_start, 0.1), text)

        base_index = max(int(round(speech_start * sample_rate)), 0)
        end_index = min(int(round(speech_end * sample_rate)), len(active) - 1)
        energy_slice = active[base_index : end_index + 1]
        if not len(energy_slice) or float(np.sum(energy_slice)) <= 0:
            return self._chunked_caption_events(speech_start, max(speech_end - speech_start, 0.1), text)

        cumulative_energy = np.cumsum(energy_slice)
        total_energy = float(cumulative_energy[-1])
        weights = [max(len(chunk.replace(" ", "")), 1) for chunk in chunks]
        total_weight = max(sum(weights), 1)
        minimum_chunk_seconds = 0.26
        cursor = speech_start
        events: list[tuple[float, float, str]] = []
        consumed_weight = 0

        for index, (chunk, weight) in enumerate(zip(chunks, weights, strict=False)):
            consumed_weight += weight
            if index == len(chunks) - 1:
                end_time = speech_end
            else:
                target_energy = total_energy * (consumed_weight / total_weight)
                boundary_index = int(np.searchsorted(cumulative_energy, target_energy, side="left"))
                end_time = speech_start + (boundary_index / sample_rate)
                remaining_minimum = minimum_chunk_seconds * max(len(chunks) - index - 1, 0)
                latest_end = speech_end - remaining_minimum
                end_time = min(max(end_time + 0.035, cursor + minimum_chunk_seconds), latest_end)
            events.append((round(cursor, 3), round(max(end_time, cursor + minimum_chunk_seconds), 3), chunk))
            cursor = end_time

        return events

    def _chunked_caption_events(
        self,
        start_seconds: float,
        duration_seconds: float,
        text: str,
    ) -> list[tuple[float, float, str]]:
        chunks = self._caption_chunks(text)
        if not chunks:
            return []

        total_weight = max(sum(max(len(chunk.replace(" ", "")), 1) for chunk in chunks), 1)
        minimum_chunk_seconds = 0.18
        lead_in = 0.0
        lead_out = min(0.05, max(float(duration_seconds) * 0.025, 0.01))
        active_duration = max(float(duration_seconds) - lead_in - lead_out, minimum_chunk_seconds * len(chunks))
        cursor = float(start_seconds)
        events: list[tuple[float, float, str]] = []
        cumulative_weight = 0.0
        timing_bias = 0.98

        for index, chunk in enumerate(chunks):
            cumulative_weight += max(len(chunk.replace(" ", "")), 1) / total_weight
            if index == len(chunks) - 1:
                end = start_seconds + active_duration
            else:
                biased_progress = min(max(cumulative_weight, 0.0), 1.0) ** timing_bias
                target_end = start_seconds + active_duration * biased_progress
                remaining_minimum = minimum_chunk_seconds * max(len(chunks) - index - 1, 0)
                latest_end = start_seconds + active_duration - remaining_minimum
                end = min(max(target_end, cursor + minimum_chunk_seconds), latest_end)
            events.append((round(cursor, 3), round(max(end, cursor + 0.18), 3), chunk))
            cursor = end
        return events

    def _caption_chunks(self, text: str) -> list[str]:
        normalized = " ".join(str(text).split()).strip()
        if not normalized:
            return []

        phrase_candidates = [
            part.strip()
            for part in re.split(r"(?<=[.!?,:;])\s+", normalized)
            if part.strip()
        ]
        phrases = phrase_candidates or [normalized]
        chunks: list[str] = []

        for phrase in phrases:
            words = phrase.split()
            index = 0
            while index < len(words):
                remaining = len(words) - index
                if remaining <= 3:
                    if remaining == 1 and chunks:
                        chunks[-1] = f"{chunks[-1]} {words[index]}"
                    else:
                        chunks.append(" ".join(words[index:]))
                    break
                chunk_size = 2 if remaining in {4, 5} else 3
                candidate = " ".join(words[index:index + chunk_size])
                if len(candidate) < 8 and index + chunk_size < len(words):
                    chunk_size += 1
                    candidate = " ".join(words[index:index + chunk_size])
                chunks.append(candidate)
                index += chunk_size

        return [chunk.strip() for chunk in chunks if chunk.strip()]

    @lru_cache(maxsize=1)
    def _video_encoder_args(self, intermediate: bool = False) -> tuple[str, ...]:
        encoder = "libx264"
        try:
            result = subprocess.run(
                [self.settings.ffmpeg_binary, "-hide_banner", "-encoders"],
                capture_output=True,
                text=True,
                check=False,
            )
            if "h264_videotoolbox" in result.stdout:
                encoder = "h264_videotoolbox"
        except Exception:
            encoder = "libx264"

        if encoder == "h264_videotoolbox":
            bitrate = "14M" if intermediate else "18M"
            maxrate = "18M" if intermediate else "25M"
            return (
                "-c:v",
                "h264_videotoolbox",
                "-b:v",
                bitrate,
                "-maxrate",
                maxrate,
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
            )

        preset = "veryfast" if intermediate else "medium"
        return (
            "-c:v",
            "libx264",
            "-preset",
            preset,
            "-crf",
            "17",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
        )

    @staticmethod
    def _ffmpeg_transition_name(style: str) -> str:
        mapping = {
            "crossfade": "fade",
            "slide-left": "slideleft",
            "slide-right": "slideright",
            "slide-up": "slideup",
            "slide-down": "slidedown",
        }
        return mapping.get(style, "fade")

    def _cache_duration_matches(self, path: Path, expected_duration: float, tolerance: float = 0.6) -> bool:
        if not path.exists():
            return False
        actual_duration = self._probe_duration(path)
        if actual_duration is None:
            return False
        return abs(float(actual_duration) - float(expected_duration)) <= tolerance

    def _mix_background_music(self, video_path: Path, output_path: Path, music_config: MusicConfig) -> None:
        match = self.store.resolve_music_track(music_config.track_name or "")
        if not match:
            raise ValueError(f"Unknown music track: {music_config.track_name}")

        music_path = Path(str(match["path"]))
        if not music_path.exists():
            raise FileNotFoundError(f"Music asset is missing: {music_path}")

        video_duration = self._probe_duration(video_path) or 30.0
        fade_out_start = max(video_duration - music_config.fade_out_seconds, 0)
        # Keep the music bed tucked under the narration even when older projects still
        # carry a louder saved slider value.
        effective_volume = round(max(min(music_config.volume, 1.0), 0.0) * 0.7, 3)

        command = [
            self.settings.ffmpeg_binary,
            "-y",
            "-i",
            str(video_path),
            "-stream_loop",
            "-1",
            "-i",
            str(music_path),
            "-filter_complex",
            f"[1:a]volume={effective_volume},afade=t=in:st=0:d={music_config.fade_in_seconds},afade=t=out:st={fade_out_start}:d={music_config.fade_out_seconds}[music];[0:a][music]amix=inputs=2:duration=first[aout]",
            "-map",
            "0:v",
            "-map",
            "[aout]",
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            # CRITICAL: without +faststart, the moov atom lands at the END of
            # the file. Browsers (and YouTube's web uploader) then have to
            # download the entire video before playback can start, which
            # looks like "the video doesn't work" on anything streaming.
            # This output replaces final_publish.mp4 in the pipeline, so
            # this single flag is what makes the published file watchable.
            "-movflags",
            "+faststart",
            str(output_path),
        ]
        self._run_ffmpeg(command)

    def _write_video_manifest(self, output_dir: Path, video_path: Path, video_config: VideoConfig) -> None:
        manifest_path = output_dir / "manifest.json"
        manifest = read_json(manifest_path, default={})
        manifest[video_path.name] = {
            "width": video_config.width,
            "height": video_config.height,
            "output_format": video_config.output_format.value,
            "created_at": datetime.utcnow().isoformat(),
            "duration_seconds": self._probe_duration(video_path),
        }
        write_json(manifest_path, manifest)

    def _probe_duration(self, path: Path) -> float | None:
        command = [
            self.settings.ffprobe_binary,
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "json",
            str(path),
        ]
        result = subprocess.run(command, capture_output=True, check=False, text=True)
        if result.returncode != 0:
            return None
        payload = json.loads(result.stdout)
        return round(float(payload["format"]["duration"]), 2)

    def _run_ffmpeg(self, command: list[str]) -> None:
        try:
            subprocess.run(command, check=True, capture_output=True)
        except subprocess.CalledProcessError:
            fallback_command = self._software_ffmpeg_fallback(command)
            if fallback_command is None:
                raise
            subprocess.run(fallback_command, check=True, capture_output=True)

    def _software_ffmpeg_fallback(self, command: list[str]) -> list[str] | None:
        if "h264_videotoolbox" not in command:
            return None

        fallback = list(command)
        try:
            codec_index = fallback.index("h264_videotoolbox")
        except ValueError:
            return None

        fallback[codec_index] = "libx264"

        cleaned: list[str] = []
        skip_next = False
        pairs_to_strip = {
            "-b:v",
            "-maxrate",
            "-bufsize",
            "-profile:v",
            "-level:v",
            "-allow_sw",
            "-realtime",
            "-q:v",
        }
        for index, item in enumerate(fallback):
            if skip_next:
                skip_next = False
                continue
            if item in pairs_to_strip and index + 1 < len(fallback):
                skip_next = True
                continue
            cleaned.append(item)

        output_path = cleaned[-1]
        encoder_index = cleaned.index("libx264")
        tail = cleaned[encoder_index + 1 : -1]
        if "-preset" not in tail:
            cleaned[encoder_index + 1 : encoder_index + 1] = ["-preset", "veryfast"]
            encoder_index += 2
        tail = cleaned[encoder_index + 1 : -1]
        if "-crf" not in tail:
            cleaned[encoder_index + 1 : encoder_index + 1] = ["-crf", "17"]
            encoder_index += 2
        tail = cleaned[encoder_index + 1 : -1]
        if "-pix_fmt" not in tail:
            cleaned[encoder_index + 1 : encoder_index + 1] = ["-pix_fmt", "yuv420p"]
            encoder_index += 2
        tail = cleaned[encoder_index + 1 : -1]
        if "-movflags" not in tail:
            cleaned[encoder_index + 1 : encoder_index + 1] = ["-movflags", "+faststart"]

        cleaned[-1] = output_path
        return cleaned

    def _run_ffmpeg_with_progress(
        self,
        command: list[str],
        *,
        expected_duration_seconds: float,
        progress_callback: Callable[[float, str], None] | None,
        cancel_callback: Callable[[], None] | None,
        start_progress: float,
        end_progress: float,
        message: str,
    ) -> None:
        if progress_callback is None:
            self._run_ffmpeg(command)
            return

        progress_command = list(command)
        progress_command[1:1] = ["-hide_banner", "-loglevel", "error", "-nostats", "-progress", "pipe:1"]
        process = subprocess.Popen(
            progress_command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        try:
            expected = max(float(expected_duration_seconds), 0.1)
            span = max(float(end_progress) - float(start_progress), 0.0)
            last_progress = float(start_progress)
            if process.stdout is not None:
                for raw_line in process.stdout:
                    if cancel_callback:
                        cancel_callback()
                    line = raw_line.strip()
                    if not line or "=" not in line:
                        continue
                    key, value = line.split("=", 1)
                    if key not in {"out_time_ms", "out_time_us"}:
                        continue
                    try:
                        if key == "out_time_us":
                            rendered_seconds = max(float(value) / 1_000_000.0, 0.0)
                        else:
                            rendered_seconds = max(float(value) / 1_000_000.0, 0.0)
                    except ValueError:
                        continue
                    ratio = min(max(rendered_seconds / expected, 0.0), 1.0)
                    progress = min(float(end_progress) - 0.2, float(start_progress) + span * ratio)
                    if progress > last_progress + 0.1:
                        last_progress = progress
                        progress_callback(progress, message)
            stdout_text, _ = process.communicate()
            if process.returncode != 0:
                raise subprocess.CalledProcessError(process.returncode, progress_command, output=stdout_text, stderr="")
        finally:
            if process.poll() is None:
                process.kill()
