"""
ShortsService - generate a 50-60 second vertical recap of a chapter for
YouTube Shorts.

Why Shorts matter for a manga-recap channel:
  • Shorts feed casual viewers into the subscriber base. Long-form
    discovery is gated by your subscriber count; Shorts have their own
    feed that surfaces tiny channels.
  • One long video → multiple Shorts means each video can promote
    itself to a brand-new audience.

What this service produces:
  • `<project>/video/short.mp4` - vertical 1080×1920, ~55 seconds
  • `<project>/youtube_bundle/short_description.md` - a short-specific
    description (different hashtag pack, no chapter markers, CTA to the
    full video)
  • `<project>/youtube_bundle/short_title.txt` - a short-form title
    (under 60 chars, hook-only)

How the Short is composed:
  1. We pick the 8-12 climax panels using the same scoring as the
     thumbnail picker, but request more candidates and trim them to
     fit a 55-second budget.
  2. For each picked panel we use its existing narration (no fresh
     generation) but TIGHTEN the per-panel duration: Shorts work on
     fast cuts (~4-7 seconds each).
  3. A one-line cold-open ("In this chapter…") hooks the first 3
     seconds. The Short ends with a flash "Subscribe" overlay.
  4. The original portrait crop of each panel is fit-cover into 1080w
     × 1920h with a soft blur background of the same panel so 4:3
     panels don't have black bars.

Implementation note:
  We render this as ONE FFmpeg command per panel + a final concat,
  rather than building yet another camera-plan system. That keeps the
  code path independent of VideoRenderService's complexity - a Short
  rendering can never break the main video render and vice versa.
"""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import soundfile as sf

from app.core.config import get_settings
from app.schemas.project import VoiceConfig
from app.services.channel_preset_service import ChannelPreset, ChannelPresetService
from app.services.edge_tts_service import EdgeTTSService, is_edge_voice
from app.services.kokoro_service import KokoroTTSService
from app.services.video_finishing_renderer import _drawtext_escape, _hex_to_ffmpeg, _pick_font
from app.utils.files import write_json

logger = logging.getLogger(__name__)


SHORT_WIDTH = 1080
SHORT_HEIGHT = 1920
SHORT_TARGET_DURATION = 55.0   # YouTube allows up to 60s but the algo
                               # prefers cleaner ends a touch early.
SHORT_MAX_PANELS = 12
SHORT_MIN_PANELS = 6
SHORT_FPS = 30

_CLIMAX_KEYWORDS = (
    "shock", "shouts", "screams", "explodes", "explosion", "destroyed",
    "reveal", "appears", "transforms", "kisses", "kiss", "punch", "fall",
    "dies", "monster", "giant", "huge", "massive", "tears", "blood",
    "weapon", "sword", "burning", "fire", "lightning", "awaken", "born",
)


@dataclass
class ShortClipPlan:
    panel_id: str
    panel_order: int
    narration: str
    duration_seconds: float


@dataclass
class ShortPlan:
    clips: list[ShortClipPlan] = field(default_factory=list)
    hook_line: str = ""
    total_duration_seconds: float = 0.0


class ShortsService:
    """Renders a Shorts cut + writes its publish-bundle text artifacts."""

    def __init__(self, settings: Any | None = None) -> None:
        self.settings = settings or get_settings()

    # ── Public ───────────────────────────────────────────────────────────

    def build(
        self,
        *,
        project_dir: Path,
        panels_json: list[dict[str, Any]],
        script_lines: list[str],
        audio_manifest: dict[str, Any],
        voice_config: VoiceConfig,
        manga_title: str | None,
        chapter_title: str | None,
        preset: ChannelPreset,
    ) -> dict[str, Any] | None:
        """Compose and render the Short. Returns a metadata dict on
        success or None if no panels were available."""
        kept_sorted = sorted(
            [p for p in panels_json if p.get("keep")],
            key=lambda p: (int(p.get("page", 0)), int(p.get("panel", 0))),
        )
        if not kept_sorted:
            return None

        plan = self._plan(kept_sorted, script_lines, audio_manifest)
        if not plan.clips:
            return None

        work_dir = project_dir / "temp" / "shorts"
        if work_dir.exists():
            shutil.rmtree(work_dir, ignore_errors=True)
        work_dir.mkdir(parents=True, exist_ok=True)

        # 1. TTS the hook line - short-specific so the Short can stand
        #    alone without the long-form cold-open audio.
        hook_text = self._make_hook(manga_title, chapter_title)
        plan.hook_line = hook_text
        hook_wav = work_dir / "hook.wav"
        self._synth(hook_text, hook_wav, voice_config)
        hook_duration = self._wav_duration(hook_wav)

        # 2. For each picked clip, copy the matching narration audio
        #    out of the project (avoids re-synthesizing).
        narration_dir = project_dir / "audio"
        clip_wavs: list[Path] = []
        for idx, clip in enumerate(plan.clips):
            # Find this clip's index in the kept-panel order so we can
            # locate the matching narration audio on disk. The earlier
            # version of this code had a dead "find empty-dict in list"
            # expression above the lookup that threw before this branch
            # could run; that bug is fixed here.
            kept_index = next(
                (i for i, p in enumerate(kept_sorted) if p.get("id") == clip.panel_id),
                None,
            )
            if kept_index is None:
                continue
            src = narration_dir / f"panel_{kept_index + 1:03d}.wav"
            dst = work_dir / f"clip_{idx:03d}.wav"
            if not src.exists():
                # No audio for this panel - synthesize on the fly so the
                # Short can still ship.
                self._synth(clip.narration, dst, voice_config)
            else:
                shutil.copy2(src, dst)
            clip_wavs.append(dst)

        # 3. Render each clip's vertical video segment.
        clip_videos: list[Path] = []
        for clip_idx, (clip_plan, clip_wav) in enumerate(zip(plan.clips, clip_wavs)):
            clip_video = work_dir / f"clip_{clip_idx:03d}.mp4"
            try:
                self._render_short_clip(
                    project_dir=project_dir,
                    plan=clip_plan,
                    audio_path=clip_wav,
                    output=clip_video,
                )
                if clip_video.exists():
                    clip_videos.append(clip_video)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Short clip %d render failed: %s", clip_idx, exc)
        if not clip_videos:
            return None

        # 4. Render the hook clip (uses the first picked panel as visual).
        hook_clip = work_dir / "hook.mp4"
        try:
            hook_plan = plan.clips[0]
            self._render_short_clip(
                project_dir=project_dir,
                plan=ShortClipPlan(
                    panel_id=hook_plan.panel_id,
                    panel_order=hook_plan.panel_order,
                    narration=plan.hook_line,
                    duration_seconds=max(2.5, hook_duration + 0.4),
                ),
                audio_path=hook_wav,
                output=hook_clip,
                overlay_text=plan.hook_line,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Short hook render failed: %s", exc)
            hook_clip = None  # type: ignore[assignment]

        # 5. Render an outro flash that says "Watch the full video".
        cta_clip = work_dir / "cta.mp4"
        try:
            self._render_cta(cta_clip, preset)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Short CTA render failed: %s", exc)
            cta_clip = None  # type: ignore[assignment]

        # 6. Concat hook + clips + cta into the final Short.
        order = [p for p in [hook_clip] + clip_videos + [cta_clip] if p is not None]
        output_dir = project_dir / "video"
        output_dir.mkdir(parents=True, exist_ok=True)
        short_video = output_dir / "short.mp4"
        try:
            self._concat(order, short_video)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Shorts concat failed: %s", exc)
            return None

        # 7. Write description + title alongside the YouTube bundle.
        bundle_dir = project_dir / "youtube_bundle"
        bundle_dir.mkdir(parents=True, exist_ok=True)
        short_title = self._make_title(manga_title, chapter_title, plan.hook_line)
        short_description = self._make_description(manga_title, chapter_title, plan.hook_line)
        (bundle_dir / "short_title.txt").write_text(short_title + "\n", encoding="utf-8")
        (bundle_dir / "short_description.md").write_text(short_description, encoding="utf-8")

        return {
            "short_video": str(short_video.relative_to(project_dir)),
            "short_title": short_title,
            "short_description": short_description,
            "clip_count": len(clip_videos),
            "total_duration_seconds": plan.total_duration_seconds + hook_duration,
        }

    # ── Plan: pick climax clips inside a duration budget ─────────────────

    def _plan(
        self,
        kept_sorted: list[dict[str, Any]],
        script_lines: list[str],
        audio_manifest: dict[str, Any],
    ) -> ShortPlan:
        total = len(kept_sorted)
        scored: list[tuple[float, int, dict[str, Any], str, float]] = []
        for idx, panel in enumerate(kept_sorted):
            narr = (
                (panel.get("narration") or "").strip()
                or (script_lines[idx].strip() if idx < len(script_lines) else "")
            )
            lower = narr.lower()
            score = 0.0
            if any(kw in lower for kw in _CLIMAX_KEYWORDS):
                score += 5.0
            # Climax bias around 60-85% of the chapter.
            t = idx / max(1, total - 1)
            score += max(0.0, 1.0 - abs(t - 0.72) * 2.0) * 3.5
            try:
                w = float(panel.get("width") or 0)
                h = float(panel.get("height") or 0)
                score += min((w * h) / 4_000_000.0, 1.5)
            except (TypeError, ValueError):
                pass
            # Skip flagged content - Shorts get even more scrutiny on
            # mobile feeds than long-form videos.
            flags = panel.get("review_flags") or []
            if any(isinstance(f, str) and f.startswith("nsfw_") for f in flags):
                continue
            audio_key = f"panel_{idx + 1:03d}.wav"
            audio_dur = float((audio_manifest.get(audio_key) or {}).get("duration_seconds") or 0.0)
            scored.append((score, idx, panel, narr, audio_dur))

        scored.sort(key=lambda s: s[0], reverse=True)
        # Take top candidates by score, but then re-sort by chronology so
        # the Short still tells the story in order.
        top = scored[: max(SHORT_MAX_PANELS * 2, 16)]
        top.sort(key=lambda s: s[1])

        budget = SHORT_TARGET_DURATION - 6.0  # leave headroom for hook+CTA
        picks: list[ShortClipPlan] = []
        consumed = 0.0
        for _, idx, panel, narr, audio_dur in top:
            # Per-clip duration: prefer the panel's actual audio length
            # but tighten short ones to a 3.5s minimum so Shorts feel
            # snappy. Cap at 6s so no single panel dominates.
            clip_dur = max(3.5, min(audio_dur + 0.2 if audio_dur else 4.5, 6.0))
            if consumed + clip_dur > budget and len(picks) >= SHORT_MIN_PANELS:
                break
            consumed += clip_dur
            picks.append(
                ShortClipPlan(
                    panel_id=str(panel.get("id")),
                    panel_order=int(panel.get("order", 0)),
                    narration=narr,
                    duration_seconds=clip_dur,
                )
            )
            if len(picks) >= SHORT_MAX_PANELS:
                break

        plan = ShortPlan(clips=picks, total_duration_seconds=consumed)
        return plan

    # ── Text generation ──────────────────────────────────────────────────

    @staticmethod
    def _make_hook(manga_title: str | None, chapter_title: str | None) -> str:
        series = (manga_title or "this chapter").strip()
        if chapter_title:
            return f"In {series}, {chapter_title}..."
        return f"Here's what happened in {series}."

    @staticmethod
    def _make_title(manga_title: str | None, chapter_title: str | None, hook: str) -> str:
        """Shorts title: a single click-line, NOT a label.

        Shorts have ~60 chars of visible title in the feed. Wasting that
        on "Series Name - Chapter X in 60s" is throwing away the only
        text-side hook you get. We synthesize from the long-form HOOK
        the bundle already wrote (passed in as `hook`) - usually a
        question or shock statement - and append the series name in
        parens for searchability. If no hook, fall back to a curiosity
        opener using the series.
        """
        series = (manga_title or "").strip()
        hook = (hook or "").strip()
        SHORTS_TITLE_MAX = 100  # YouTube cap; Shorts UI truncates around 70-90

        # Prefer the long-form hook (e.g. "What if humanity's last hope
        # is also its greatest danger?") - it's already been tuned for
        # click-through. Strip a trailing period or ellipsis so we can
        # append the series tag cleanly.
        if hook:
            clean_hook = hook.rstrip(".? ")
            # The Gemini hook often ends with "?" - keep that punctuation
            # for the question variant.
            if hook.endswith("?"):
                base = clean_hook + "?"
            else:
                base = clean_hook + "."
            if series and series.casefold() not in base.casefold():
                tail = f" ({series})"
                if len(base) + len(tail) <= SHORTS_TITLE_MAX:
                    base = base + tail
            return base[:SHORTS_TITLE_MAX]

        # No hook - synthesize a curiosity-friendly opener.
        if series:
            return f"The moment {series} actually starts."[:SHORTS_TITLE_MAX]
        return "The moment the story actually starts."[:SHORTS_TITLE_MAX]

    @staticmethod
    def _make_description(manga_title: str | None, chapter_title: str | None, hook: str) -> str:
        """Shorts description: hook line + soft CTA + tags.

        Avoids the bland "60-second recap: <chapter>" template that
        signals filler. The long-form hook (if present) becomes the
        first line; chapter label only appears as a tag.
        """
        series = manga_title or "Manga"
        chapter_tag = ""
        if chapter_title:
            # "Chapters 1-10" -> "chapters110", "Chapter 27" -> "chapter27"
            chapter_tag = "#" + "".join(
                ch for ch in chapter_title.lower() if ch.isalnum()
            )
        series_tag = "#" + "".join(ch for ch in series.lower() if ch.isalnum())
        # Preserve the original terminal punctuation so question hooks
        # stay questions ("...danger?" not "...danger.").
        raw_hook = (hook or "").strip()
        if raw_hook:
            if raw_hook[-1] in "?.!":
                lead = raw_hook
            else:
                lead = raw_hook + "."
        else:
            lead = f"The moment {series} actually starts."
        return (
            f"{lead}\n\n"
            f"Watch the full chapter recap on the channel for the complete story.\n\n"
            f"#shorts #manga #manhwa #anime #recap {series_tag} {chapter_tag}\n"
        )

    # ── Rendering primitives ─────────────────────────────────────────────

    def _render_short_clip(
        self,
        *,
        project_dir: Path,
        plan: ShortClipPlan,
        audio_path: Path,
        output: Path,
        overlay_text: str | None = None,
    ) -> None:
        """Render one panel-as-vertical-clip with subtle Ken Burns + optional
        text overlay. Audio bed is the supplied WAV."""
        panel_image = project_dir / "panels" / f"panel_{plan.panel_order:03d}.png"
        if not panel_image.exists():
            for ext in ("jpg", "jpeg", "webp"):
                alt = project_dir / "panels" / f"panel_{plan.panel_order:03d}.{ext}"
                if alt.exists():
                    panel_image = alt
                    break
        if not panel_image.exists():
            raise FileNotFoundError(f"missing panel image for shorts clip {plan.panel_order}")

        duration = max(plan.duration_seconds, self._wav_duration(audio_path) + 0.2)
        total_frames = int(duration * SHORT_FPS)

        font_path = _pick_font()
        font_clause = f":fontfile={font_path}" if font_path else ""
        overlay_clause = ""
        if overlay_text:
            txt = _drawtext_escape(overlay_text)
            overlay_clause = (
                f",drawtext=text='{txt}'{font_clause}:fontsize=64"
                f":fontcolor=white:borderw=4:bordercolor=black@0.85"
                f":box=1:boxcolor=black@0.55:boxborderw=24"
                f":x=(w-text_w)/2:y=h*0.85"
            )

        # Filter graph: two passes over the panel - one blurred fit-cover
        # to fill the vertical canvas as a backdrop, one sharp panel
        # centered with subtle zoom.
        filter_complex = (
            # backdrop: blow up + heavy blur, fill 1080×1920
            f"[0:v]split=2[bg][fg];"
            f"[bg]scale={SHORT_WIDTH}:{SHORT_HEIGHT}:force_original_aspect_ratio=increase,"
            f"crop={SHORT_WIDTH}:{SHORT_HEIGHT},gblur=sigma=30:steps=2,eq=brightness=-0.18[bg2];"
            # foreground: fit-contain into 1080×1820 (leave 100px breathing top/bottom)
            f"[fg]scale={SHORT_WIDTH}:{SHORT_HEIGHT - 100}:force_original_aspect_ratio=decrease,"
            f"setsar=1,zoompan=z='min(zoom+0.0008,1.06)':d={total_frames}:s={SHORT_WIDTH}x{SHORT_HEIGHT - 100}:fps={SHORT_FPS}[fg2];"
            # composite
            f"[bg2][fg2]overlay=(W-w)/2:(H-h)/2{overlay_clause}[vout]"
        )

        command = [
            self.settings.ffmpeg_binary,
            "-y",
            "-loop", "1",
            "-i", str(panel_image),
            "-i", str(audio_path),
            "-filter_complex", filter_complex,
            "-map", "[vout]",
            "-map", "1:a:0",
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-r", str(SHORT_FPS),
            "-t", f"{duration}",
            "-c:a", "aac",
            "-b:a", "192k",
            "-shortest",
            str(output),
        ]
        self._run_ffmpeg(command)

    def _render_cta(self, output: Path, preset: ChannelPreset) -> None:
        """Final 3-second CTA card - vertical, accent-colored."""
        accent_hex = _hex_to_ffmpeg(preset.accent_color)
        font_path = _pick_font()
        font_clause = f":fontfile={font_path}" if font_path else ""
        channel = _drawtext_escape(preset.channel_name.strip().upper())
        cta = _drawtext_escape("Watch the full recap → subscribe")
        command = [
            self.settings.ffmpeg_binary,
            "-y",
            "-f", "lavfi",
            "-i", f"color=c=0x0a0a0f:size={SHORT_WIDTH}x{SHORT_HEIGHT}:duration=3:rate={SHORT_FPS}",
            "-f", "lavfi",
            "-i", "anullsrc=channel_layout=stereo:sample_rate=48000:duration=3",
            "-vf",
            (
                f"drawbox=x=0:y=0:w={int(SHORT_WIDTH * 0.02)}:h=ih:color=0x{accent_hex}:t=fill,"
                f"drawtext=text='{channel}'{font_clause}:fontsize=120"
                f":fontcolor=white:x=(w-text_w)/2:y=h*0.42,"
                f"drawtext=text='{cta}'{font_clause}:fontsize=56"
                f":fontcolor=0x{accent_hex}:x=(w-text_w)/2:y=h*0.55"
            ),
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-r", str(SHORT_FPS),
            "-c:a", "aac",
            "-b:a", "192k",
            "-shortest",
            str(output),
        ]
        self._run_ffmpeg(command)

    def _concat(self, parts: list[Path], output: Path) -> None:
        if not parts:
            raise ValueError("no parts to concat")
        work_dir = output.parent.parent / "temp" / "shorts" / "concat"
        work_dir.mkdir(parents=True, exist_ok=True)
        normalized: list[Path] = []
        for idx, src in enumerate(parts):
            norm = work_dir / f"part_{idx:03d}.mp4"
            self._run_ffmpeg([
                self.settings.ffmpeg_binary,
                "-y",
                "-i", str(src),
                "-c:v", "libx264",
                "-pix_fmt", "yuv420p",
                "-r", str(SHORT_FPS),
                "-c:a", "aac",
                "-ar", "48000",
                "-b:a", "192k",
                str(norm),
            ])
            normalized.append(norm)
        manifest = work_dir / "concat.txt"
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

    def _synth(self, text: str, out_path: Path, voice_config: VoiceConfig) -> None:
        try:
            if is_edge_voice(voice_config.voice):
                EdgeTTSService().synthesize_to_file(text, out_path, voice_config)
                return
        except Exception as exc:  # noqa: BLE001
            logger.warning("Shorts TTS via Edge failed (%s); falling back to Kokoro.", exc)
        KokoroTTSService().synthesize_to_file(text, out_path, voice_config)

    @staticmethod
    def _wav_duration(path: Path) -> float:
        info = sf.info(str(path))
        return float(info.duration or 0.0)

    def _run_ffmpeg(self, command: list[str]) -> None:
        result = subprocess.run(command, capture_output=True, text=True, check=False)
        if result.returncode != 0:
            tail = (result.stderr or "")[-400:]
            raise RuntimeError(f"ffmpeg failed (exit {result.returncode}): {tail}")
