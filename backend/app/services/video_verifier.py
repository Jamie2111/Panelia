from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageStat

from app.core.config import get_settings
from app.utils.files import ensure_dir


@dataclass
class VideoVerificationResult:
    path: Path
    width: int
    height: int
    duration_seconds: float
    audio_duration_seconds: float | None
    sample_count: int
    dark_samples: int
    low_detail_samples: int
    issues: list[str]

    @property
    def ok(self) -> bool:
        return not self.issues

    def to_dict(self) -> dict[str, object]:
        return {
            "path": str(self.path),
            "width": self.width,
            "height": self.height,
            "duration_seconds": round(self.duration_seconds, 3),
            "audio_duration_seconds": None if self.audio_duration_seconds is None else round(self.audio_duration_seconds, 3),
            "sample_count": self.sample_count,
            "dark_samples": self.dark_samples,
            "low_detail_samples": self.low_detail_samples,
            "issues": self.issues,
            "ok": self.ok,
        }


class VideoVerifier:
    def __init__(self) -> None:
        self.settings = get_settings()

    def verify_project_video(
        self,
        project_dir: Path,
        video_path: Path,
        audio_path: Path | None = None,
    ) -> VideoVerificationResult:
        width, height, duration = self._probe_video(video_path)
        audio_duration = self._resolve_audio_duration(project_dir, audio_path)
        sample_dir = ensure_dir(project_dir / "temp" / "video_verify")
        timestamps = self._sample_timestamps(duration)
        dark_samples = 0
        low_detail_samples = 0
        issues: list[str] = []

        for index, timestamp in enumerate(timestamps, start=1):
            frame_path = sample_dir / f"{video_path.stem}_{index:02d}.png"
            self._extract_frame(video_path, timestamp, frame_path)
            mean_luma, stddev_luma = self._analyze_frame(frame_path)
            if mean_luma < 10:
                dark_samples += 1
            if stddev_luma < 4:
                low_detail_samples += 1

        if width <= 0 or height <= 0:
            issues.append("Video dimensions are invalid.")
        if duration <= 1:
            issues.append("Video duration is unexpectedly short.")
        if audio_duration is not None and duration < max(audio_duration * 0.9, audio_duration - 8):
            issues.append("Video duration is much shorter than narration audio.")
        if timestamps and dark_samples == len(timestamps):
            issues.append("Sampled frames are all nearly black.")
        if timestamps and low_detail_samples == len(timestamps):
            issues.append("Sampled frames show extremely low visual detail.")

        result = VideoVerificationResult(
            path=video_path,
            width=width,
            height=height,
            duration_seconds=duration,
            audio_duration_seconds=audio_duration,
            sample_count=len(timestamps),
            dark_samples=dark_samples,
            low_detail_samples=low_detail_samples,
            issues=issues,
        )
        report_path = project_dir / "output" / f"{video_path.stem}_verification.json"
        report_path.write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")
        return result

    def _resolve_audio_duration(self, project_dir: Path, audio_path: Path | None) -> float | None:
        if audio_path and audio_path.exists():
            return self._probe_audio(audio_path)
        manifest_path = project_dir / "audio" / "manifest.json"
        if not manifest_path.exists():
            return None
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            return None
        total = 0.0
        for item in manifest.values():
            if not isinstance(item, dict):
                continue
            try:
                total += float(item.get("duration_seconds") or 0.0)
            except Exception:
                continue
        return total or None

    def _probe_video(self, path: Path) -> tuple[int, int, float]:
        payload = self._run_ffprobe(
            [
                self.settings.ffprobe_binary,
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=width,height:format=duration",
                "-of",
                "json",
                str(path),
            ]
        )
        data = json.loads(payload)
        stream = (data.get("streams") or [{}])[0]
        width = int(stream.get("width") or 0)
        height = int(stream.get("height") or 0)
        duration = float((data.get("format") or {}).get("duration") or 0.0)
        return width, height, duration

    def _probe_audio(self, path: Path) -> float:
        payload = self._run_ffprobe(
            [
                self.settings.ffprobe_binary,
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "json",
                str(path),
            ]
        )
        data = json.loads(payload)
        return float((data.get("format") or {}).get("duration") or 0.0)

    def _sample_timestamps(self, duration: float) -> list[float]:
        if duration <= 3:
            return [max(duration / 2, 0.0)] if duration > 0 else []
        fractions = [0.1, 0.35, 0.6, 0.85]
        return [max(min(duration * fraction, max(duration - 0.2, 0)), 0) for fraction in fractions]

    def _extract_frame(self, video_path: Path, timestamp: float, frame_path: Path) -> None:
        command = [
            self.settings.ffmpeg_binary,
            "-y",
            "-ss",
            f"{timestamp:.3f}",
            "-i",
            str(video_path),
            "-frames:v",
            "1",
            str(frame_path),
        ]
        subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def _analyze_frame(self, frame_path: Path) -> tuple[float, float]:
        with Image.open(frame_path) as image:
            grayscale = image.convert("L")
            stats = ImageStat.Stat(grayscale)
            return float(stats.mean[0]), float(stats.stddev[0])

    def _run_ffprobe(self, command: list[str]) -> str:
        completed = subprocess.run(command, check=True, capture_output=True, text=True)
        return completed.stdout
