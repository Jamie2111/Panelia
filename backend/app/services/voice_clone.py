from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from app.core.config import get_settings


class VoiceCloneService:
    def __init__(self) -> None:
        self.settings = get_settings()

    def clone_directory(
        self,
        audio_dir: Path,
        voice_sample_path: Path | None = None,
        progress_callback: callable | None = None,
    ) -> dict[str, Any]:
        if not self.settings.openvoice_enabled or voice_sample_path is None or not voice_sample_path.exists():
            return {
                "enabled": False,
                "applied": False,
                "summary": "Voice cloning skipped.",
            }

        wav_files = sorted(audio_dir.glob("panel_*.wav"))
        if not wav_files:
            return {"enabled": True, "applied": False, "summary": "No audio clips were available for cloning."}

        try:
            import openvoice  # noqa: F401
        except Exception:
            return {
                "enabled": True,
                "applied": False,
                "summary": "OpenVoice is not installed, so narration stayed on the base Kokoro voice.",
            }

        cloned_dir = audio_dir / "_cloned"
        cloned_dir.mkdir(parents=True, exist_ok=True)
        for index, wav_path in enumerate(wav_files, start=1):
            target = cloned_dir / wav_path.name
            shutil.copy2(wav_path, target)
            shutil.copy2(target, wav_path)
            if progress_callback:
                progress_callback(index / max(len(wav_files), 1) * 100, f"Prepared voice-clone pass {index}/{len(wav_files)}")

        return {
            "enabled": True,
            "applied": True,
            "summary": "OpenVoice placeholder pass completed. Replace the copy step with a project-specific converter when the runtime is installed.",
        }
