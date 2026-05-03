from __future__ import annotations

import hashlib
import os
import wave
from pathlib import Path

import numpy as np
import soundfile as sf

from app.core.config import get_settings
from app.schemas.project import VoiceConfig
from app.utils.files import read_json, write_json


class KokoroTTSService:
    def __init__(self) -> None:
        self.settings = get_settings()
        self._pipelines: dict[str, object] = {}
        self._runtime_configured = False
        self._shared_cache_dir = (self.settings.data_dir / "_tts_cache")
        self._shared_cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_pipeline(self, lang_code: str):
        if lang_code in self._pipelines:
            return self._pipelines[lang_code]
        from kokoro import KPipeline

        pipeline = KPipeline(lang_code=lang_code)
        self._pipelines[lang_code] = pipeline
        return pipeline

    def _configure_runtime(self) -> None:
        if self._runtime_configured:
            return
        self._runtime_configured = True
        try:
            import torch
        except Exception:
            return

        cpu_count = os.cpu_count() or 4
        target_threads = max(1, min(4, cpu_count // 3 or 1))
        try:
            torch.set_num_threads(target_threads)
        except Exception:
            pass
        try:
            torch.set_num_interop_threads(max(1, min(2, target_threads)))
        except Exception:
            pass

    def _clip_signature(self, text: str, voice_config: VoiceConfig) -> str:
        payload = "\n".join(
            [
                str(text or "").strip(),
                voice_config.voice,
                voice_config.lang_code,
                f"{voice_config.speed:.4f}",
                str(self.settings.kokoro_sample_rate),
            ]
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def generate_audio(
        self,
        script_lines: list[str],
        output_dir: Path,
        voice_config: VoiceConfig,
        panel_ids: list[str] | None = None,
        progress_callback: callable | None = None,
        cancel_callback: callable | None = None,
    ) -> dict[str, dict[str, object]]:
        output_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = output_dir / "manifest.json"
        existing_manifest = read_json(manifest_path, default={})
        if not isinstance(existing_manifest, dict):
            existing_manifest = {}
        manifest: dict[str, dict[str, object]] = {}
        pipeline = None
        active_files: set[str] = set()

        for index, line in enumerate(script_lines, start=1):
            if cancel_callback:
                cancel_callback()
            output_path = output_dir / f"panel_{index:03d}.wav"
            panel_id = panel_ids[index - 1] if panel_ids and index - 1 < len(panel_ids) else f"narration_{index:03d}"
            signature = self._clip_signature(line, voice_config)
            shared_clip_path = self._shared_cache_dir / f"{signature}.wav"
            shared_manifest_path = self._shared_cache_dir / f"{signature}.json"
            previous_entry = existing_manifest.get(output_path.name, {})
            reused = (
                output_path.exists()
                and isinstance(previous_entry, dict)
                and str(previous_entry.get("signature") or "").strip() == signature
            )

            if reused:
                duration = float(previous_entry.get("duration_seconds") or 0.0)
                if duration <= 0:
                    duration = self._wav_duration(output_path)
                status_message = f"Reused voice clip {index}/{len(script_lines)}"
            elif shared_clip_path.exists():
                self._copy_cached_clip(shared_clip_path, output_path)
                shared_manifest = read_json(shared_manifest_path, default={})
                duration = float(shared_manifest.get("duration_seconds") or 0.0)
                if duration <= 0:
                    duration = self._wav_duration(output_path)
                status_message = f"Reused shared voice clip {index}/{len(script_lines)}"
            else:
                if pipeline is None:
                    self._configure_runtime()
                    pipeline = self._get_pipeline(voice_config.lang_code)
                audio_segments = []
                generator = pipeline(
                    line,
                    voice=voice_config.voice,
                    speed=voice_config.speed,
                    split_pattern=r"\n+",
                )
                for _, _, audio in generator:
                    audio_segments.append(audio)

                audio_data = np.concatenate(audio_segments) if audio_segments else np.zeros(self.settings.kokoro_sample_rate // 4)
                sf.write(output_path, audio_data, self.settings.kokoro_sample_rate)
                duration = self._wav_duration(output_path)
                self._copy_cached_clip(output_path, shared_clip_path)
                write_json(shared_manifest_path, {"duration_seconds": duration, "signature": signature})
                status_message = f"Generated voice clip {index}/{len(script_lines)}"

            manifest[output_path.name] = {
                "panel_id": panel_id,
                "duration_seconds": duration,
                "signature": signature,
            }
            active_files.add(output_path.name)
            if progress_callback:
                progress_callback(index / max(len(script_lines), 1) * 100, status_message)

        for stale_path in output_dir.glob("panel_*.wav"):
            if stale_path.name not in active_files:
                stale_path.unlink(missing_ok=True)

        write_json(manifest_path, manifest)
        return manifest

    def _copy_cached_clip(self, source: Path, destination: Path) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        try:
            if destination.exists():
                destination.unlink()
            os.link(source, destination)
        except Exception:
            import shutil

            shutil.copy2(source, destination)

    def synthesize_to_file(self, text: str, output_path: Path, voice_config: VoiceConfig) -> Path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        self._configure_runtime()
        pipeline = self._get_pipeline(voice_config.lang_code)
        audio_segments = []
        generator = pipeline(
            text,
            voice=voice_config.voice,
            speed=voice_config.speed,
            split_pattern=r"\n+",
        )
        for _, _, audio in generator:
            audio_segments.append(audio)

        audio_data = np.concatenate(audio_segments) if audio_segments else np.zeros(self.settings.kokoro_sample_rate // 4)
        sf.write(output_path, audio_data, self.settings.kokoro_sample_rate)
        return output_path

    def _wav_duration(self, path: Path) -> float:
        with wave.open(str(path), "rb") as wav_file:
            frames = wav_file.getnframes()
            sample_rate = wav_file.getframerate()
        return round(frames / max(sample_rate, 1), 2)
