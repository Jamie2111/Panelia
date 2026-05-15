from __future__ import annotations

import hashlib
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import soundfile as sf

from app.core.config import get_settings
from app.schemas.project import VoiceConfig
from app.services.kokoro_service import KokoroTTSService
from app.services.story_preprocessor import NarrationUnit
from app.utils.files import ensure_dir, write_json


class KokoroTTSEngine:
    def __init__(self) -> None:
        self.settings = get_settings()
        self._kokoro = KokoroTTSService()
        self._sentence_cache_dir = ensure_dir(self.settings.data_dir / "_sentence_tts_cache")

    def synthesize_units(
        self,
        units: list[NarrationUnit],
        output_dir: Path,
        voice_config: VoiceConfig,
        progress_callback: callable | None = None,
        cancel_callback: callable | None = None,
    ) -> dict[str, dict[str, object]]:
        output_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = output_dir / "manifest.json"
        manifest: dict[str, dict[str, object]] = {}

        sentence_jobs = self._collect_sentence_jobs(units, voice_config)
        ordered_jobs = list(sentence_jobs.items())
        for index, (signature, payload) in enumerate(ordered_jobs, start=1):
            if cancel_callback:
                cancel_callback()
            cache_path = self._sentence_cache_dir / f"{signature}.wav"
            if progress_callback:
                start_progress = ((index - 1) / max(len(ordered_jobs), 1)) * 38
                progress_callback(start_progress, f"Preparing narration sentence cache {index}/{len(ordered_jobs)}")
            if not cache_path.exists():
                unit_voice = payload["voice_config"]
                self._kokoro.synthesize_to_file(str(payload["text"]), cache_path, unit_voice)
            if progress_callback:
                progress_callback(index / max(len(ordered_jobs), 1) * 38, f"Prepared narration sentence cache {index}/{len(ordered_jobs)}")

        assembly_payloads = [
            (
                index,
                unit,
                self._segments_for_unit(unit, voice_config),
                output_dir / f"panel_{index:03d}.wav",
            )
            for index, unit in enumerate(units, start=1)
        ]

        max_workers = max(1, min(int(self.settings.narration_sentence_cache_workers or 1), os.cpu_count() or 1, 4))
        if max_workers <= 1 or len(assembly_payloads) <= 1:
            results = []
            for index, payload in enumerate(assembly_payloads, start=1):
                if progress_callback:
                    start_progress = 38 + ((index - 1) / max(len(assembly_payloads), 1)) * 62
                    progress_callback(start_progress, f"Assembling narration clip {index}/{len(assembly_payloads)}")
                results.append(self._assemble_unit_audio(payload))
        else:
            if progress_callback:
                progress_callback(38, f"Assembling {len(assembly_payloads)} narration clips")
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                results = list(executor.map(self._assemble_unit_audio, assembly_payloads))

        active_files: set[str] = set()
        for completed_index, payload in enumerate(sorted(results, key=lambda item: item[0]), start=1):
            _, unit, output_path, duration_seconds, clip_signature = payload
            active_files.add(output_path.name)
            manifest[output_path.name] = {
                "panel_id": unit.panel_id,
                "duration_seconds": duration_seconds,
                "signature": clip_signature,
                "emotion": unit.emotion,
                "language": unit.language,
            }
            if progress_callback:
                progress_callback(38 + completed_index / max(len(results), 1) * 62, f"Synthesized narration clip {completed_index}/{len(results)}")

        for stale_path in output_dir.glob("panel_*.wav"):
            if stale_path.name not in active_files:
                stale_path.unlink(missing_ok=True)
        write_json(manifest_path, manifest)
        return manifest

    def _collect_sentence_jobs(self, units: list[NarrationUnit], voice_config: VoiceConfig) -> dict[str, dict[str, object]]:
        jobs: dict[str, dict[str, object]] = {}
        for unit in units:
            for text, sentence_voice, pause_ms in self._segments_for_unit(unit, voice_config):
                signature = self._sentence_signature(text, sentence_voice, unit.emotion, pause_ms)
                jobs.setdefault(signature, {"text": text, "voice_config": sentence_voice, "pause_ms": pause_ms})
        return jobs

    def _segments_for_unit(self, unit: NarrationUnit, voice_config: VoiceConfig) -> list[tuple[str, VoiceConfig, int]]:
        spoken_text = " ".join(str(unit.spoken_text or "").split()).strip()
        if not spoken_text:
            return []

        # Kokoro already models natural prosody within a complete sentence.
        # Keeping each panel line as one segment avoids artificial hard-silence
        # gaps in the middle of narration.
        segments = [spoken_text]
        lang_code = str(unit.metadata.get("kokoro_lang_code") or voice_config.lang_code or "a")
        segment_voice = voice_config.model_copy(update={"lang_code": lang_code, "speed": self._emotion_speed(unit.emotion, voice_config.speed)})
        payload: list[tuple[str, VoiceConfig, int]] = []
        for text in segments:
            payload.append((text, segment_voice, 0))
        return payload

    def _assemble_unit_audio(
        self,
        payload: tuple[int, NarrationUnit, list[tuple[str, VoiceConfig, int]], Path],
    ) -> tuple[int, NarrationUnit, Path, float, str]:
        index, unit, segments, output_path = payload
        audio_segments: list[np.ndarray] = []
        signature_parts: list[str] = []
        sample_rate = self.settings.kokoro_sample_rate
        for segment_text, segment_voice, pause_ms in segments:
            signature = self._sentence_signature(segment_text, segment_voice, unit.emotion, pause_ms)
            cache_path = self._sentence_cache_dir / f"{signature}.wav"
            audio, segment_rate = sf.read(cache_path, dtype="float32")
            if segment_rate != sample_rate:
                sample_rate = segment_rate
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            audio_segments.append(audio.astype(np.float32, copy=False))
            signature_parts.append(signature)
            if pause_ms > 0:
                silence = np.zeros(int(sample_rate * (pause_ms / 1000.0)), dtype=np.float32)
                audio_segments.append(silence)
        final_audio = np.concatenate(audio_segments) if audio_segments else np.zeros(sample_rate // 4, dtype=np.float32)
        sf.write(output_path, final_audio, sample_rate)
        duration_seconds = round(len(final_audio) / max(sample_rate, 1), 2)
        clip_signature = hashlib.sha256("|".join(signature_parts).encode("utf-8")).hexdigest()
        return index, unit, output_path, duration_seconds, clip_signature

    def _sentence_signature(
        self,
        text: str,
        voice_config: VoiceConfig,
        emotion: str,
        pause_ms: int,
    ) -> str:
        payload = "\n".join(
            [
                text.strip(),
                voice_config.voice,
                voice_config.lang_code,
                f"{voice_config.speed:.4f}",
                emotion,
                str(pause_ms),
            ]
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _emotion_speed(self, emotion: str, base_speed: float) -> float:
        adjustment = {
            "action": 0.08,
            "tension": -0.02,
            "mystery": -0.06,
            "shock": -0.08,
            "revenge": 0.02,
            "calm planning": -0.04,
            "neutral narration": 0.0,
        }.get(emotion, 0.0)
        return max(0.7, min(1.3, float(base_speed) + adjustment))
