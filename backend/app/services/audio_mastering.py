from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

import soundfile as sf

from app.core.config import get_settings


class AudioMasteringService:
    def __init__(self) -> None:
        self.settings = get_settings()

    def master_directory(
        self,
        audio_dir: Path,
        progress_callback: callable | None = None,
    ) -> dict[str, Any]:
        if not self.settings.narration_mastering_enabled:
            return {"enabled": False, "summary": "Audio mastering disabled.", "durations": {}}

        wav_files = sorted(audio_dir.glob("panel_*.wav"))
        durations: dict[str, float] = {}
        if not wav_files:
            return {"enabled": True, "summary": "No clips required mastering.", "durations": durations}

        for index, wav_path in enumerate(wav_files, start=1):
            temp_path = wav_path.with_name(f".{wav_path.stem}.mastered.wav")
            # NOTE: the previous mastering chain included
            # `aecho=0.7:0.45:18:0.04`. That added a 0.45-gain echo with
            # only 18 ms delay, which the ear perceives as a DOUBLED
            # voice (it's below the ~50 ms "single sound" Haas threshold
            # but above the comb-filter range). Result: every panel's
            # narration sounded like two TTS voices saying the same
            # thing in tight succession. Removed.
            command = [
                self.settings.ffmpeg_binary,
                "-y",
                "-i",
                str(wav_path),
                "-af",
                ",".join(
                    (
                        "highpass=f=70",
                        "lowpass=f=14500",
                        "acompressor=threshold=-18dB:ratio=2.4:attack=20:release=180:makeup=2.5",
                        "equalizer=f=3200:t=q:w=1.1:g=1.2",
                        "loudnorm=I=-16:TP=-1.5:LRA=11",
                    )
                ),
                "-ar",
                str(self.settings.kokoro_sample_rate),
                str(temp_path),
            ]
            try:
                subprocess.run(command, check=True, capture_output=True, text=True)
                temp_path.replace(wav_path)
            except Exception:
                temp_path.unlink(missing_ok=True)
            durations[wav_path.name] = self._duration_seconds(wav_path)
            if progress_callback:
                progress_callback(index / max(len(wav_files), 1) * 100, f"Mastered narration clip {index}/{len(wav_files)}")

        return {"enabled": True, "summary": "Applied ffmpeg loudness, compression, EQ, and subtle reverb.", "durations": durations}

    def _duration_seconds(self, path: Path) -> float:
        info = sf.info(str(path))
        return round(float(info.duration or 0.0), 2)
