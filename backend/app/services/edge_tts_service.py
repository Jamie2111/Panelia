"""
Edge TTS service - Microsoft Azure Neural voices via the free public
endpoint that Edge browser uses for "Read aloud".

Why this is the right "most human, free" TTS:
  • Zero API key required. No quota negotiation, no billing surface.
  • Powered by Microsoft Azure Neural voice models - the same engines
    Microsoft sells through Azure Speech ($16/million chars). Quality
    is on par with paid offerings; a notch below ElevenLabs but well
    above any other open-source option.
  • Dozens of preset voices in every major language. We curate a
    YouTube-recap-friendly subset in catalog_service.
  • No model files to download (Kokoro ships ~300MB; Edge TTS streams).
  • Streams MP3 (mp3-24khz-96kbitrate-mono); we transcode to WAV via
    soundfile so the rest of the pipeline (manifest, video render,
    crossfade) doesn't change.

Failure mode policy:
  • Microsoft may rate-limit or block the public endpoint. When that
    happens we fall back to Kokoro automatically (the existing engine
    stays installed). Callers don't see the difference beyond a voice
    swap.
  • The fallback is wired in `app/services/generate_narration.py` -
    this module only handles its own happy/fail path.

Implementation notes:
  • edge-tts is an async library. We expose a SYNC interface here so
    the call sites can keep their existing threaded fan-out without
    juggling event loops.
  • Output sample rate matches Kokoro's setting (24kHz) so downstream
    audio mastering doesn't have to resample.
"""

from __future__ import annotations

import asyncio
import hashlib
import io
import logging
import os
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from app.core.config import get_settings
from app.schemas.project import VoiceConfig
from app.utils.files import read_json, write_json

logger = logging.getLogger(__name__)

try:
    import edge_tts
    _EDGE_TTS_AVAILABLE = True
except ImportError:
    _EDGE_TTS_AVAILABLE = False


# ── Default voice mapping ─────────────────────────────────────────────────
# Map our short voice IDs (e.g. "edge_ava") to the full Microsoft voice
# string. We pick voices that consistently land high on YouTube recap
# channel A/B tests - natural cadence, can do drama without sounding
# theatrical, not over-processed.
EDGE_VOICE_MAP: dict[str, str] = {
    # ── English (US) ────────────────────────────────────────────────────
    "edge_ava":         "en-US-AvaMultilingualNeural",   # warm female, most human-sounding
    "edge_andrew":      "en-US-AndrewMultilingualNeural",  # confident male, recap default
    "edge_emma":        "en-US-EmmaMultilingualNeural",  # bright female, energetic recaps
    "edge_brian":       "en-US-BrianMultilingualNeural",  # neutral male, smooth flow
    "edge_jenny":       "en-US-JennyMultilingualNeural",  # versatile, casual
    "edge_aria":        "en-US-AriaNeural",              # polished female news voice
    "edge_guy":         "en-US-GuyNeural",               # mid-range male, steady
    "edge_christopher": "en-US-ChristopherNeural",       # deeper male, dramatic
    "edge_eric":        "en-US-EricNeural",              # bright male, fast pacing
    "edge_michelle":    "en-US-MichelleNeural",          # mature female, storyteller
    # ── English (UK) ────────────────────────────────────────────────────
    "edge_libby":       "en-GB-LibbyNeural",             # crisp British female
    "edge_ryan":        "en-GB-RyanNeural",              # British male editorial
    "edge_sonia":       "en-GB-SoniaNeural",             # warm British female
    # ── Japanese ────────────────────────────────────────────────────────
    "edge_nanami":      "ja-JP-NanamiNeural",            # Japanese female
    "edge_keita":       "ja-JP-KeitaNeural",             # Japanese male
}


def is_edge_voice(voice_id: str) -> bool:
    """Check whether a voice id should be routed to the Edge TTS engine."""
    return voice_id in EDGE_VOICE_MAP or voice_id.startswith("edge_")


class EdgeTTSService:
    """Synchronous wrapper around the async `edge-tts` library."""

    # Microsoft's streaming endpoint emits MP3 at 24kHz; the file conversion
    # below resamples to soundfile's preferred format. We pick 24kHz so
    # Kokoro and Edge clips can share the same audio mastering pipeline
    # without a resample step.
    SAMPLE_RATE: int = 24_000

    def __init__(self) -> None:
        self.settings = get_settings()
        self._shared_cache_dir = self.settings.data_dir / "_edge_tts_cache"
        self._shared_cache_dir.mkdir(parents=True, exist_ok=True)
        if not _EDGE_TTS_AVAILABLE:
            raise RuntimeError(
                "edge-tts is not installed. Run: pip install edge-tts"
            )

    # ── Public API (mirrors KokoroTTSService) ─────────────────────────────

    def synthesize_to_file(
        self,
        text: str,
        output_path: Path,
        voice_config: VoiceConfig,
    ) -> Path:
        """Render one sentence to disk as WAV. Used by the engine for
        sentence-cache warm-up. Raises on Microsoft endpoint failures so
        the caller can fall back to Kokoro."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        wav_bytes = self._render_to_wav_bytes(text, voice_config)
        output_path.write_bytes(wav_bytes)
        return output_path

    def generate_audio(
        self,
        script_lines: list[str],
        output_dir: Path,
        voice_config: VoiceConfig,
        panel_ids: list[str] | None = None,
        progress_callback: Any = None,
        cancel_callback: Any = None,
    ) -> dict[str, dict[str, object]]:
        """Same shape as KokoroTTSService.generate_audio so the existing
        narration pipeline can swap engines transparently. We synthesize
        per-script-line, mirror Kokoro's manifest format, and reuse the
        shared cache so a re-run is free."""
        output_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = output_dir / "manifest.json"
        existing_manifest = read_json(manifest_path, default={}) or {}
        if not isinstance(existing_manifest, dict):
            existing_manifest = {}

        manifest: dict[str, dict[str, object]] = {}
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
                shared_manifest = read_json(shared_manifest_path, default={}) or {}
                duration = float(shared_manifest.get("duration_seconds") or 0.0)
                if duration <= 0:
                    duration = self._wav_duration(output_path)
                status_message = f"Reused shared voice clip {index}/{len(script_lines)}"
            else:
                wav_bytes = self._render_to_wav_bytes(line, voice_config)
                output_path.write_bytes(wav_bytes)
                duration = self._wav_duration(output_path)
                self._copy_cached_clip(output_path, shared_clip_path)
                write_json(shared_manifest_path, {"duration_seconds": duration, "signature": signature})
                status_message = f"Generated voice clip {index}/{len(script_lines)}"

            manifest[output_path.name] = {
                "panel_id": panel_id,
                "duration_seconds": duration,
                "signature": signature,
                "tts_engine": "edge",
            }
            active_files.add(output_path.name)
            if progress_callback:
                progress_callback(index / max(len(script_lines), 1) * 100, status_message)

        for stale_path in output_dir.glob("panel_*.wav"):
            if stale_path.name not in active_files:
                stale_path.unlink(missing_ok=True)

        write_json(manifest_path, manifest)
        return manifest

    # ── Internals ─────────────────────────────────────────────────────────

    def _render_to_wav_bytes(self, text: str, voice_config: VoiceConfig) -> bytes:
        """Run the async Edge TTS pipeline and return WAV bytes.

        edge-tts emits MP3 (24kHz mono 96kbps). We decode in-memory to a
        numpy float32 array via soundfile, then write back to a WAV bytes
        buffer at our chosen sample rate. No temp files; no shell-outs.
        """
        clean_text = (text or "").strip()
        if not clean_text:
            return self._silence_wav(0.25)

        voice = self._resolve_voice(voice_config)
        rate, volume, pitch = self._format_prosody(voice_config)

        # `Communicate.save` writes to disk; we keep things in memory.
        async def _gather() -> bytes:
            communicate = edge_tts.Communicate(
                clean_text,
                voice=voice,
                rate=rate,
                volume=volume,
                pitch=pitch,
            )
            chunks: list[bytes] = []
            async for message in communicate.stream():
                if message.get("type") == "audio":
                    data = message.get("data")
                    if data:
                        chunks.append(data)
            return b"".join(chunks)

        mp3_bytes = self._run_async(_gather())
        if not mp3_bytes:
            raise RuntimeError("Edge TTS returned empty audio stream")

        # Decode MP3 → float32 array; soundfile handles MP3 via libsndfile.
        with sf.SoundFile(io.BytesIO(mp3_bytes), "r") as src:
            audio = src.read(dtype="float32", always_2d=False)
            source_rate = src.samplerate
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if source_rate != self.SAMPLE_RATE:
            audio = self._resample(audio, source_rate, self.SAMPLE_RATE)

        buf = io.BytesIO()
        sf.write(buf, audio.astype(np.float32), self.SAMPLE_RATE, format="WAV", subtype="PCM_16")
        return buf.getvalue()

    @staticmethod
    def _silence_wav(duration_seconds: float) -> bytes:
        samples = max(1, int(EdgeTTSService.SAMPLE_RATE * duration_seconds))
        silence = np.zeros(samples, dtype=np.float32)
        buf = io.BytesIO()
        sf.write(buf, silence, EdgeTTSService.SAMPLE_RATE, format="WAV", subtype="PCM_16")
        return buf.getvalue()

    @staticmethod
    def _resample(audio: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
        if source_rate == target_rate:
            return audio
        # Lightweight linear resample - Edge TTS already outputs 24kHz mono,
        # so this path is exercised only for unexpected source rates. Good
        # enough that we don't pull in scipy/librosa just for fallback.
        ratio = target_rate / float(source_rate)
        new_length = int(round(len(audio) * ratio))
        if new_length <= 0:
            return audio
        x_old = np.linspace(0.0, 1.0, num=len(audio), endpoint=False)
        x_new = np.linspace(0.0, 1.0, num=new_length, endpoint=False)
        return np.interp(x_new, x_old, audio).astype(np.float32)

    @staticmethod
    def _resolve_voice(voice_config: VoiceConfig) -> str:
        configured = (voice_config.voice or "").strip()
        if configured in EDGE_VOICE_MAP:
            return EDGE_VOICE_MAP[configured]
        # If the caller already passes a fully-qualified Microsoft voice
        # name (e.g. "en-US-AriaNeural"), honor it as-is.
        if "-" in configured and configured.endswith("Neural"):
            return configured
        # Last resort: a sensible default. Better than crashing on a
        # voice we don't have a mapping for.
        return EDGE_VOICE_MAP["edge_ava"]

    @staticmethod
    def _format_prosody(voice_config: VoiceConfig) -> tuple[str, str, str]:
        """Translate VoiceConfig.speed to edge-tts rate/volume/pitch SSML strings.

        Speed 1.0 → "+0%". 0.85 → "-15%". 1.20 → "+20%".
        Volume and pitch stay at defaults; we change them only if the
        speaker's emotion tagging requests it later.
        """
        speed = max(0.5, min(1.5, float(voice_config.speed or 1.0)))
        delta = int(round((speed - 1.0) * 100))
        rate_str = f"{delta:+d}%"
        return rate_str, "+0%", "+0Hz"

    @staticmethod
    def _run_async(coro):
        """Run an async coroutine to completion from a sync context.

        Handles the "already running event loop" case (FastAPI thread)
        by spinning up a fresh loop in a dedicated thread.
        """
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(coro)
        # We're inside a running loop - spin a worker.
        import threading
        result_box: dict[str, Any] = {}

        def _runner() -> None:
            loop = asyncio.new_event_loop()
            try:
                result_box["value"] = loop.run_until_complete(coro)
            except Exception as exc:  # noqa: BLE001
                result_box["error"] = exc
            finally:
                loop.close()

        thread = threading.Thread(target=_runner, daemon=True)
        thread.start()
        thread.join()
        if "error" in result_box:
            raise result_box["error"]
        return result_box.get("value")

    @staticmethod
    def _clip_signature(text: str, voice_config: VoiceConfig) -> str:
        payload = "\n".join(
            [
                str(text or "").strip(),
                voice_config.voice,
                voice_config.lang_code,
                f"{voice_config.speed:.4f}",
                "edge",
            ]
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    @staticmethod
    def _copy_cached_clip(source: Path, destination: Path) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        try:
            if destination.exists():
                destination.unlink()
            os.link(source, destination)
        except Exception:
            import shutil
            shutil.copy2(source, destination)

    def _wav_duration(self, path: Path) -> float:
        info = sf.info(str(path))
        return round(float(info.duration or 0.0), 2)
