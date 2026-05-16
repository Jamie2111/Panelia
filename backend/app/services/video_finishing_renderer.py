"""
VideoFinishingRenderer - renders the cold-open + outro clips, optionally
the watermark overlay, and concatenates everything with the main video
into the publish-ready final.

Inputs:
  • The already-rendered `final.mp4` from VideoRenderService
  • The ColdOpenPlan from VideoFinishingService
  • The active ChannelPreset

Outputs (alongside the original):
  • `final_publish.mp4` - cold-open + title-card + main + outro,
    everything stitched. This is the file the user uploads to YouTube.

Implementation notes:
  • We render cold-open and outro as standalone MP4s with FFmpeg using
    the lavfi `color` + `drawtext` filters - no PIL, no Pillow font
    pain. This keeps the dependency surface flat and runs anywhere
    FFmpeg + a TrueType font are present.
  • Cold-open uses TTS to speak the teaser line (Edge TTS by default;
    falls back to Kokoro). The audio is mixed at the same level the
    main video uses.
  • Concat uses FFmpeg's `concat` demuxer which is the only stitching
    method that works reliably across codecs at the demuxer level.
"""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
from pathlib import Path
from typing import Any

import soundfile as sf

from app.core.config import get_settings
from app.schemas.project import VideoConfig, VoiceConfig
from app.services.channel_preset_service import ChannelPreset
from app.services.edge_tts_service import EdgeTTSService, is_edge_voice
from app.services.kokoro_service import KokoroTTSService
from app.services.video_finishing_service import ColdOpenPlan

logger = logging.getLogger(__name__)


# Default fonts we try in order. We deliberately list system paths that
# exist on macOS / Linux installs. If none load FFmpeg uses its default.
_DRAWTEXT_FONTS: tuple[str, ...] = (
    "/System/Library/Fonts/Supplemental/Impact.ttf",
    "/System/Library/Fonts/HelveticaNeue.ttc",
    "/System/Library/Fonts/Helvetica.ttc",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
)


def _pick_font() -> str | None:
    for path in _DRAWTEXT_FONTS:
        if Path(path).exists():
            return path
    return None


def _drawtext_escape(text: str) -> str:
    """FFmpeg drawtext is finicky about colons, percent signs, quotes and
    backslashes. This escapes every character that breaks the filter
    expression."""
    if not text:
        return ""
    # Order matters: backslashes first.
    return (
        text.replace("\\", "\\\\")
        .replace(":", "\\:")
        .replace("'", "’")  # straight to curly apostrophe - same visual, FFmpeg-safe
        .replace("%", "\\%")
        .replace("[", "\\[")
        .replace("]", "\\]")
        .replace(",", "\\,")
    )


def _hex_to_ffmpeg(hex_color: str) -> str:
    """FFmpeg accepts `0xRRGGBB` or `RRGGBB`. Normalize a leading '#'."""
    s = hex_color.strip().lstrip("#")
    if len(s) == 8:
        # Drop alpha if present (8-digit hex like #RRGGBBAA)
        s = s[:6]
    if not all(c in "0123456789abcdefABCDEF" for c in s) or len(s) != 6:
        return "FFFFFF"
    return s


class VideoFinishingRenderer:
    """Stitch cold-open + main video + outro into a publish-ready file."""

    def __init__(self, settings: Any | None = None) -> None:
        self.settings = settings or get_settings()
        self._cached_video_codec_args: tuple[str, ...] | None = None

    def _video_codec_args(self) -> tuple[str, ...]:
        """Best-available H.264 encoder args.

        On Apple Silicon we get h264_videotoolbox at ~10x the throughput
        of libx264 for the kind of full-video re-encodes this renderer
        does. Probe `ffmpeg -encoders` once and cache the result so the
        probe doesn't run per clip.

        Critical: every full-video re-encode in this file used to be
        hardcoded libx264 default-preset, which on a 2-hour 1080p main
        video meant ~2 HOURS of wall-clock just for the concat
        normalize step. h264_videotoolbox brings that down to ~6-8 min.
        """
        if self._cached_video_codec_args is not None:
            return self._cached_video_codec_args
        encoder = "libx264"
        try:
            result = subprocess.run(
                [self.settings.ffmpeg_binary, "-hide_banner", "-encoders"],
                capture_output=True,
                text=True,
                check=False,
            )
            if "h264_videotoolbox" in (result.stdout or ""):
                encoder = "h264_videotoolbox"
        except Exception:  # noqa: BLE001
            pass
        if encoder == "h264_videotoolbox":
            args = (
                "-c:v", "h264_videotoolbox",
                "-b:v", "14M",
                "-maxrate", "18M",
                "-pix_fmt", "yuv420p",
            )
        else:
            args = (
                "-c:v", "libx264",
                "-preset", "veryfast",
                "-crf", "18",
                "-pix_fmt", "yuv420p",
            )
        self._cached_video_codec_args = args
        return args

    # ── Public entry point ────────────────────────────────────────────────

    def finalize(
        self,
        *,
        project_dir: Path,
        main_video_path: Path,
        cold_open_plan: ColdOpenPlan | None,
        preset: ChannelPreset,
        video_config: VideoConfig,
        voice_config: VoiceConfig,
        project_name: str,
        output_name: str = "final_publish",
    ) -> Path:
        """Produce `<project>/video/<output_name>.mp4` with the cold-open
        prepended and outro appended. Returns the new file path.

        If FFmpeg fails on any step we log + fall back to just copying
        the main video so the user still gets a publishable file.
        """
        video_dir = project_dir / "video"
        video_dir.mkdir(parents=True, exist_ok=True)
        work_dir = project_dir / "temp" / "finishing"
        work_dir.mkdir(parents=True, exist_ok=True)
        # Wipe stale intermediates so partial reruns don't reuse broken
        # cold-open clips.
        for stale in work_dir.glob("*.mp4"):
            stale.unlink(missing_ok=True)
        for stale in work_dir.glob("*.wav"):
            stale.unlink(missing_ok=True)

        parts: list[Path] = []

        # ── Frame-zero thumbnail card (optional) ─────────────────────
        # Prepends a brief still card of the chosen thumbnail variant at
        # t=0 of the final video. This lets the user pause the video on
        # their phone the first time they open it, screenshot the
        # thumbnail, and upload it directly to YouTube Studio without
        # needing to wrangle a separate file. Toggled via the channel
        # preset; off by default to keep legacy projects untouched.
        if getattr(preset, "thumbnail_card_enabled", False):
            try:
                thumb_card_path = self._render_thumbnail_card(
                    project_dir=project_dir,
                    work_dir=work_dir,
                    preset=preset,
                    video_config=video_config,
                )
                if thumb_card_path is not None:
                    parts.append(thumb_card_path)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Thumbnail-card render failed: %s", exc)

        # ── Cold open (optional) ─────────────────────────────────────
        if preset.cold_open_enabled and cold_open_plan is not None:
            try:
                cold_path = self._render_cold_open(
                    plan=cold_open_plan,
                    project_dir=project_dir,
                    work_dir=work_dir,
                    preset=preset,
                    video_config=video_config,
                    voice_config=voice_config,
                    project_name=project_name,
                )
                if cold_path is not None:
                    parts.append(cold_path)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Cold-open render failed: %s", exc)

        # ── Title card (optional) ─────────────────────────────────────
        if preset.title_card_enabled:
            try:
                title_path = self._render_title_card(
                    work_dir=work_dir,
                    preset=preset,
                    video_config=video_config,
                    project_name=project_name,
                )
                if title_path is not None:
                    parts.append(title_path)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Title-card render failed: %s", exc)

        parts.append(main_video_path)

        # ── Outro card (optional) ─────────────────────────────────────
        if preset.outro_enabled:
            try:
                outro_path = self._render_outro_card(
                    work_dir=work_dir,
                    preset=preset,
                    video_config=video_config,
                )
                if outro_path is not None:
                    parts.append(outro_path)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Outro-card render failed: %s", exc)

        # ── Concatenate everything ───────────────────────────────────
        output_path = video_dir / f"{output_name}.mp4"
        if len(parts) == 1:
            # Nothing to add → just copy the main video as the publish file.
            if parts[0] != output_path:
                shutil.copy2(parts[0], output_path)
            return output_path

        try:
            self._concat_clips(parts, output_path)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Concat failed (%s); copying main video as publish output.",
                exc,
            )
            shutil.copy2(main_video_path, output_path)
        return output_path

    # ── Cold-open render ─────────────────────────────────────────────────

    def _render_cold_open(
        self,
        *,
        plan: ColdOpenPlan,
        project_dir: Path,
        work_dir: Path,
        preset: ChannelPreset,
        video_config: VideoConfig,
        voice_config: VoiceConfig,
        project_name: str,
    ) -> Path | None:
        """Render a 5-7 second clip: panel image full-bleed with a slow
        zoom-in + teaser TTS over the existing audio bed."""
        panel_image = project_dir / "panels" / f"panel_{plan.panel_order:03d}.png"
        if not panel_image.exists():
            # Try alternate extensions
            for ext in ("jpg", "jpeg", "webp"):
                alt = project_dir / "panels" / f"panel_{plan.panel_order:03d}.{ext}"
                if alt.exists():
                    panel_image = alt
                    break
        if not panel_image.exists():
            logger.warning("Cold-open: panel image %s not found", panel_image)
            return None

        # Synthesize teaser audio.
        teaser_wav = work_dir / "cold_open_teaser.wav"
        try:
            self._synthesize_teaser(plan.teaser_text, teaser_wav, voice_config)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Teaser TTS failed (%s); rendering silent cold-open.", exc)
            self._write_silence(teaser_wav, plan.hold_seconds)

        teaser_duration = self._wav_duration(teaser_wav)
        # Make the visual hold ≥ teaser audio + a 0.7s breath at the end.
        hold = max(plan.hold_seconds, teaser_duration + 0.7)

        accent_hex = _hex_to_ffmpeg(preset.accent_color)
        font_path = _pick_font()
        font_clause = f":fontfile={font_path}" if font_path else ""

        # PIL-render the teaser text overlay as a transparent PNG, then
        # composite it onto the zoompan'd panel via ffmpeg's `overlay`
        # filter. This avoids ffmpeg's `drawtext` (which requires
        # libfreetype, not enabled in Homebrew's default ffmpeg build).
        width = video_config.width
        height = video_config.height
        fps = 30
        total_frames = int(hold * fps)

        overlay_png = work_dir / "cold_open_text.png"
        self._render_text_overlay_png(
            text=plan.teaser_text.upper(),
            size=(width, height),
            output_path=overlay_png,
            anchor="bottom_center",
            text_color=(255, 255, 255),
            box_color=(0, 0, 0, 115),
            font_size=int(height * 0.07),
        )

        cold_video = work_dir / "cold_open.mp4"
        ffmpeg = self.settings.ffmpeg_binary
        command = [
            ffmpeg,
            "-y",
            "-loop", "1",
            "-i", str(panel_image),
            "-i", str(overlay_png),
            "-i", str(teaser_wav),
            "-filter_complex",
            (
                f"[0:v]scale={width * 1.1:.0f}:{height * 1.1:.0f}:force_original_aspect_ratio=increase,"
                f"crop={width}:{height},setsar=1,"
                f"zoompan=z='min(zoom+0.0008,1.08)':d={total_frames}:s={width}x{height}:fps={fps},"
                f"eq=brightness=-0.08[bg];"
                f"[bg][1:v]overlay=0:0:format=auto[vout];"
                f"[2:a]apad,atrim=0:{hold}[aout]"
            ),
            "-map", "[vout]",
            "-map", "[aout]",
            *self._video_codec_args(),
            "-r", str(fps),
            "-t", f"{hold}",
            "-c:a", "aac",
            "-b:a", "192k",
            "-shortest",
            "-movflags", "+faststart",
            str(cold_video),
        ]
        self._run_ffmpeg(command)
        return cold_video if cold_video.exists() else None

    @staticmethod
    def _render_text_overlay_png(
        *,
        text: str,
        size: tuple[int, int],
        output_path: Path,
        anchor: str = "bottom_center",
        text_color: tuple[int, int, int] = (255, 255, 255),
        box_color: tuple[int, int, int, int] | None = (0, 0, 0, 115),
        font_size: int = 56,
    ) -> Path:
        """Render text into an RGBA PNG sized to the video canvas.

        ffmpeg's overlay filter is always available; we use it to
        composite this PNG onto the cold-open or title-card video.
        Eliminates the need for ffmpeg drawtext (libfreetype).
        """
        from PIL import Image, ImageDraw, ImageFont
        import textwrap as _tw

        width, height = size
        font_path = _pick_font()
        try:
            font = ImageFont.truetype(font_path, font_size) if font_path else ImageFont.load_default()
        except Exception:
            font = ImageFont.load_default()

        canvas = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(canvas)

        # Word-wrap so long teasers don't overflow.
        approx_chars = max(1, int(width * 0.85 / max(font_size * 0.45, 1)))
        wrapped = _tw.fill(text, width=approx_chars)
        lines = wrapped.split("\n")

        # Measure
        line_heights: list[int] = []
        line_widths: list[int] = []
        for line in lines:
            bbox = draw.textbbox((0, 0), line, font=font)
            line_widths.append(bbox[2] - bbox[0])
            line_heights.append(bbox[3] - bbox[1])
        text_h_total = sum(line_heights) + (len(lines) - 1) * int(font_size * 0.18)
        text_w_max = max(line_widths) if line_widths else 0

        # Anchor
        if anchor == "bottom_center":
            base_x = (width - text_w_max) // 2
            base_y = int(height * 0.78) - text_h_total // 2
        elif anchor == "center":
            base_x = (width - text_w_max) // 2
            base_y = (height - text_h_total) // 2
        else:
            base_x = 0
            base_y = 0

        # Box behind the text for legibility.
        if box_color is not None:
            pad_x = int(font_size * 0.6)
            pad_y = int(font_size * 0.35)
            box_x0 = max(0, base_x - pad_x)
            box_y0 = max(0, base_y - pad_y)
            box_x1 = min(width, base_x + text_w_max + pad_x)
            box_y1 = min(height, base_y + text_h_total + pad_y)
            draw.rectangle((box_x0, box_y0, box_x1, box_y1), fill=box_color)

        # Draw text with a thin black stroke for legibility.
        cursor_y = base_y
        for line, lw, lh in zip(lines, line_widths, line_heights):
            line_x = (width - lw) // 2  # center each line
            stroke_w = max(2, int(font_size * 0.05))
            draw.text(
                (line_x, cursor_y),
                line,
                fill=text_color,
                font=font,
                stroke_width=stroke_w,
                stroke_fill=(0, 0, 0, 220),
            )
            cursor_y += lh + int(font_size * 0.18)

        canvas.save(output_path, "PNG", optimize=True)
        return output_path

    # ── Title card render ────────────────────────────────────────────────

    def _render_thumbnail_card(
        self,
        *,
        project_dir: Path,
        work_dir: Path,
        preset: ChannelPreset,
        video_config: VideoConfig,
    ) -> Path | None:
        """Render a short still card of the chosen YouTube thumbnail.

        The user can then pause the published video on their phone at
        t=0, screenshot the frame, and upload it directly to YouTube
        Studio as the channel thumbnail. Useful when uploading from a
        device that can't easily handle the standalone PNG.

        Source is `youtube_bundle/thumbnail.png` (which the publish
        studio keeps in sync with whichever variant the user picked).
        Falls back to the canonical thumbnail if no bundle exists yet.
        """
        # Find the active thumbnail. Prefer the bundle's chosen variant.
        bundle_dir = project_dir / "youtube_bundle"
        manifest_path = bundle_dir / "manifest.json"
        thumb_path: Path | None = None
        if manifest_path.exists():
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                variants = manifest.get("thumbnail_variants") or []
                chosen_idx = int(manifest.get("chosen_thumbnail_index") or 0)
                if 0 <= chosen_idx < len(variants):
                    rel = variants[chosen_idx].get("path") if isinstance(variants[chosen_idx], dict) else None
                    if rel:
                        candidate = project_dir / rel
                        if candidate.exists():
                            thumb_path = candidate
                if thumb_path is None and manifest.get("thumbnail_path"):
                    candidate = project_dir / manifest["thumbnail_path"]
                    if candidate.exists():
                        thumb_path = candidate
            except Exception as exc:  # noqa: BLE001
                logger.warning("Reading thumbnail manifest failed: %s", exc)

        if thumb_path is None:
            return None

        width = video_config.width
        height = video_config.height
        duration = max(0.5, float(getattr(preset, "thumbnail_card_duration_seconds", 1.5)))
        clip_path = work_dir / "thumbnail_card.mp4"
        ffmpeg = self.settings.ffmpeg_binary

        # Subtle zoom-in (1.0 to 1.04 over the duration) to make the still
        # feel alive. Center-crop after scale-with-pad covers the canvas
        # without distortion.
        zoom_expr = f"min(zoom+0.0006,1.04)"
        command = [
            ffmpeg,
            "-y",
            "-loop", "1",
            "-framerate", "30",
            "-t", str(duration),
            "-i", str(thumb_path),
            "-f", "lavfi",
            "-i", f"anullsrc=channel_layout=stereo:sample_rate=48000:duration={duration}",
            "-vf",
            (
                f"scale={width}:{height}:force_original_aspect_ratio=increase,"
                f"crop={width}:{height},"
                f"zoompan=z='{zoom_expr}':d=1:s={width}x{height}:fps=30"
            ),
            *self._video_codec_args(),
            "-r", "30",
            "-c:a", "aac",
            "-b:a", "192k",
            "-shortest",
            "-movflags", "+faststart",
            str(clip_path),
        ]
        self._run_ffmpeg(command)
        return clip_path if clip_path.exists() else None

    def _render_title_card(
        self,
        *,
        work_dir: Path,
        preset: ChannelPreset,
        video_config: VideoConfig,
        project_name: str,
    ) -> Path | None:
        """A short branded slide: project name + channel name underneath,
        accent underline.

        PIL-rendered for the same libfreetype reason as the outro and
        cold-open: ffmpeg's drawtext is not available on every install,
        but PIL is a hard dependency and works everywhere.
        """
        from PIL import Image, ImageDraw, ImageFont

        width = video_config.width
        height = video_config.height
        duration = max(0.5, float(preset.title_card_duration_seconds))
        accent_rgb = self._hex_to_rgb_tuple(preset.accent_color, fallback=(127, 255, 212))
        font_path = _pick_font()

        title_size = int(height * 0.085)
        channel_size = int(height * 0.04)

        def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
            try:
                return ImageFont.truetype(font_path, size) if font_path else ImageFont.load_default()
            except Exception:
                return ImageFont.load_default()

        # PIL-render the full title card.
        card_png = work_dir / "title_card.png"
        canvas = Image.new("RGB", (width, height), (10, 10, 15))
        draw = ImageDraw.Draw(canvas)

        title_text = (project_name or "Panelia").strip()
        channel_text = preset.channel_name.strip()

        title_font = _font(title_size)
        channel_font = _font(channel_size)
        title_bbox = draw.textbbox((0, 0), title_text, font=title_font)
        title_w = title_bbox[2] - title_bbox[0]
        title_h = title_bbox[3] - title_bbox[1]
        channel_bbox = draw.textbbox((0, 0), channel_text, font=channel_font)
        channel_w = channel_bbox[2] - channel_bbox[0]

        title_x = (width - title_w) // 2
        title_y = (height - title_h) // 2 - int(title_size * 0.3)
        draw.text((title_x, title_y), title_text, fill=(255, 255, 255), font=title_font)

        # Accent underline
        underline_w = int(width * 0.18)
        underline_h = max(3, int(height * 0.006))
        underline_x = (width - underline_w) // 2
        underline_y = height // 2 + int(title_size * 0.85)
        draw.rectangle(
            (underline_x, underline_y, underline_x + underline_w, underline_y + underline_h),
            fill=accent_rgb,
        )

        channel_x = (width - channel_w) // 2
        channel_y = height // 2 + int(title_size * 1.0)
        draw.text((channel_x, channel_y), channel_text, fill=accent_rgb, font=channel_font)
        canvas.save(card_png, "PNG", optimize=True)

        title_clip = work_dir / "title_card.mp4"
        ffmpeg = self.settings.ffmpeg_binary
        command = [
            ffmpeg,
            "-y",
            "-loop", "1",
            "-framerate", "30",
            "-t", str(duration),
            "-i", str(card_png),
            "-f", "lavfi",
            "-i", f"anullsrc=channel_layout=stereo:sample_rate=48000:duration={duration}",
            *self._video_codec_args(),
            "-r", "30",
            "-c:a", "aac",
            "-b:a", "192k",
            "-shortest",
            "-movflags", "+faststart",
            str(title_clip),
        ]
        self._run_ffmpeg(command)
        return title_clip if title_clip.exists() else None

    # ── Outro render ─────────────────────────────────────────────────────

    def _render_outro_card(
        self,
        *,
        work_dir: Path,
        preset: ChannelPreset,
        video_config: VideoConfig,
    ) -> Path | None:
        """A subscribe-CTA card. Sized for YouTube's end-screen overlay,
        so the actual rendered text occupies the LEFT side only - the
        right side stays empty for end-screen placement.

        Implementation note: we render the text overlay with PIL into a
        single PNG and have ffmpeg loop it into a silent MP4. ffmpeg's
        own `drawtext` filter is only available when ffmpeg was built
        with libfreetype - which the Homebrew formula no longer enables
        by default - so this implementation works regardless of build.
        """
        from PIL import Image, ImageDraw, ImageFont

        width = video_config.width
        height = video_config.height
        duration = max(2.0, float(preset.outro_duration_seconds))
        accent_rgb = self._hex_to_rgb_tuple(preset.accent_color, fallback=(127, 255, 212))
        font_path = _pick_font()

        title_size = int(height * 0.07)
        body_size = int(height * 0.038)
        tagline_size = int(height * 0.028)

        def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
            try:
                return ImageFont.truetype(font_path, size) if font_path else ImageFont.load_default()
            except Exception:
                return ImageFont.load_default()

        card_png = work_dir / "outro_card.png"
        canvas = Image.new("RGB", (width, height), (10, 10, 15))
        draw = ImageDraw.Draw(canvas)

        # Accent stripe on the left edge.
        stripe_w = max(8, int(width * 0.012))
        draw.rectangle((0, 0, stripe_w, height), fill=accent_rgb)

        text_x = int(width * 0.06)
        draw.text((text_x, int(height * 0.32)), preset.channel_name.strip().upper(), fill=(255, 255, 255), font=_font(title_size))
        draw.text((text_x, int(height * 0.48)), preset.outro_message.strip(), fill=accent_rgb, font=_font(body_size))
        draw.text((text_x, int(height * 0.58)), preset.tagline.strip(), fill=(161, 161, 170), font=_font(tagline_size))

        canvas.save(card_png, "PNG", optimize=True)

        outro_clip = work_dir / "outro_card.mp4"
        ffmpeg = self.settings.ffmpeg_binary
        command = [
            ffmpeg,
            "-y",
            "-loop", "1",
            "-framerate", "30",
            "-t", str(duration),
            "-i", str(card_png),
            "-f", "lavfi",
            "-i", f"anullsrc=channel_layout=stereo:sample_rate=48000:duration={duration}",
            *self._video_codec_args(),
            "-r", "30",
            "-c:a", "aac",
            "-b:a", "192k",
            "-shortest",
            "-movflags", "+faststart",
            str(outro_clip),
        ]
        self._run_ffmpeg(command)
        return outro_clip if outro_clip.exists() else None

    @staticmethod
    def _hex_to_rgb_tuple(hex_color: str, *, fallback: tuple[int, int, int]) -> tuple[int, int, int]:
        s = (hex_color or "").strip().lstrip("#")
        if len(s) == 8:
            s = s[:6]
        if len(s) != 6:
            return fallback
        try:
            return (int(s[0:2], 16), int(s[2:4], 16), int(s[4:6], 16))
        except ValueError:
            return fallback

    # ── Concat ───────────────────────────────────────────────────────────

    def _concat_clips(self, parts: list[Path], output: Path) -> None:
        """Stitch the clips end-to-end. We use a two-pass approach -
        first normalize every clip to the same codec/timebase, then
        concat via the demuxer. This avoids the "first frame is black"
        bug FFmpeg's concat filter has when input streams differ."""
        if not parts:
            raise ValueError("No clips to concat.")

        work_dir = output.parent.parent / "temp" / "finishing"
        work_dir.mkdir(parents=True, exist_ok=True)

        # Normalize every part to a known format so concat is safe.
        # IMPORTANT: this step used to hardcode libx264 default-preset,
        # which on the 2-hour main video meant ~2 HOURS of wall-clock
        # encoding just for normalization. Now uses h264_videotoolbox
        # on Apple Silicon (~10x faster), libx264 -preset veryfast
        # elsewhere. Quality difference at 14M bitrate is imperceptible
        # for this kind of recap content.
        codec_args = self._video_codec_args()
        normalized: list[Path] = []
        for idx, src in enumerate(parts):
            norm = work_dir / f"concat_{idx:03d}.mp4"
            self._run_ffmpeg([
                self.settings.ffmpeg_binary,
                "-y",
                "-i", str(src),
                *codec_args,
                "-r", "30",
                "-c:a", "aac",
                "-ar", "48000",
                "-b:a", "192k",
                "-movflags", "+faststart",
                str(norm),
            ])
            normalized.append(norm)

        manifest = work_dir / "concat_manifest.txt"
        manifest.write_text(
            "\n".join(f"file '{n.as_posix()}'" for n in normalized),
            encoding="utf-8",
        )
        self._run_ffmpeg([
            self.settings.ffmpeg_binary,
            "-y",
            "-f", "concat",
            "-safe", "0",
            "-i", str(manifest),
            "-c", "copy",
            "-movflags", "+faststart",
            str(output),
        ])

    # ── Helpers ──────────────────────────────────────────────────────────

    def _synthesize_teaser(
        self,
        text: str,
        out_path: Path,
        voice_config: VoiceConfig,
    ) -> None:
        if is_edge_voice(voice_config.voice):
            EdgeTTSService().synthesize_to_file(text, out_path, voice_config)
        else:
            KokoroTTSService().synthesize_to_file(text, out_path, voice_config)

    @staticmethod
    def _write_silence(out_path: Path, duration: float) -> None:
        import numpy as np
        rate = 24_000
        samples = np.zeros(int(rate * max(0.5, duration)), dtype="float32")
        sf.write(out_path, samples, rate)

    @staticmethod
    def _wav_duration(path: Path) -> float:
        info = sf.info(str(path))
        return float(info.duration or 0.0)

    def _run_ffmpeg(self, command: list[str]) -> None:
        result = subprocess.run(command, capture_output=True, text=True, check=False)
        if result.returncode != 0:
            tail = (result.stderr or "")[-400:]
            raise RuntimeError(f"ffmpeg failed (exit {result.returncode}): {tail}")
