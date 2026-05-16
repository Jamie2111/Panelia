"""
AzureTTSService - paid-tier Microsoft Neural TTS.

This is the same voice catalog the free Edge TTS endpoint serves, just
talking to the official Azure Speech REST API with an API key instead
of reverse-engineering the Edge browser's WebSocket. Benefits:

  • No rate limit (vs free Edge endpoint's frequent 503s)
  • No Kokoro fallback latency spikes mid-render
  • 16-concurrent friendly (vs 2-concurrent ceiling on free)
  • Same voice IDs (en-US-AvaNeural, en-US-EmmaNeural, ...) so the
    audio character is identical to your current Edge output. No
    re-narration needed; cached clips stay valid.

Configuration (in .env):
  PANELIA_AZURE_SPEECH_KEY     32-char hex from Azure Portal -> Speech
                                resource -> Keys and Endpoint
  PANELIA_AZURE_SPEECH_REGION  e.g. "eastus", "westeurope"

When the key isn't set, the service is disabled and the existing free
Edge path stays in play. When the key IS set, EdgeTTSService routes
all calls here first; this module raises on failure so EdgeTTSService
can fall back to free Edge TTS automatically. That means: if you run
out of Azure credit, the worst case is the pipeline slows back down
to free-Edge speed - it never breaks.
"""

from __future__ import annotations

import io
import logging
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from app.core.config import get_settings
from app.schemas.project import VoiceConfig
from app.services.edge_tts_service import EDGE_VOICE_MAP


def _silence_wav_bytes(duration_seconds: float, sample_rate: int = 24_000) -> bytes:
    """Generate WAV bytes for a short silent buffer (used for empty text)."""
    samples = max(1, int(sample_rate * duration_seconds))
    silence = np.zeros(samples, dtype=np.float32)
    buf = io.BytesIO()
    sf.write(buf, silence, sample_rate, format="WAV", subtype="PCM_16")
    return buf.getvalue()

try:
    import azure.cognitiveservices.speech as speechsdk  # type: ignore
    _AZURE_AVAILABLE = True
except ImportError:
    _AZURE_AVAILABLE = False

logger = logging.getLogger(__name__)


def is_azure_configured() -> bool:
    """True when both the SDK is installed AND credentials are present."""
    if not _AZURE_AVAILABLE:
        return False
    settings = get_settings()
    return bool(
        getattr(settings, "azure_speech_key", "")
        and getattr(settings, "azure_speech_region", "")
    )


class AzureQuotaExhausted(RuntimeError):
    """Raised when Azure returns a billing/auth error indicating the
    paid tier won't serve more requests this run. EdgeTTSService
    catches this and stops attempting Azure for the rest of the run,
    falling all subsequent calls back to free Edge TTS."""


class AzureTTSService:
    """Sync wrapper around the Azure Speech REST API.

    Output shape mirrors EdgeTTSService.synthesize_to_file: a 24 kHz
    mono WAV at `output_path`. Voice IDs are identical (en-US-...) so
    the rest of the pipeline (cache hashing, mastering, manifest)
    stays unchanged.
    """

    SAMPLE_RATE: int = 24_000

    def __init__(self) -> None:
        self.settings = get_settings()
        if not _AZURE_AVAILABLE:
            raise RuntimeError(
                "azure-cognitiveservices-speech is not installed. "
                "Run: pip install azure-cognitiveservices-speech"
            )
        key = getattr(self.settings, "azure_speech_key", "")
        region = getattr(self.settings, "azure_speech_region", "")
        if not key or not region:
            raise RuntimeError(
                "AZURE_SPEECH_KEY and AZURE_SPEECH_REGION must be set in .env"
            )
        self._config = speechsdk.SpeechConfig(subscription=key, region=region)
        # Request the same audio format the rest of Panelia uses internally.
        self._config.set_speech_synthesis_output_format(
            speechsdk.SpeechSynthesisOutputFormat.Raw24Khz16BitMonoPcm
        )

    def synthesize_to_file(
        self,
        text: str,
        output_path: Path,
        voice_config: VoiceConfig,
    ) -> Path:
        """Render one sentence to disk as WAV. Raises on Azure failure
        so the caller (EdgeTTSService) can fall back to free Edge TTS."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        wav_bytes = self._render_to_wav_bytes(text, voice_config)
        output_path.write_bytes(wav_bytes)
        return output_path

    def _render_to_wav_bytes(self, text: str, voice_config: VoiceConfig) -> bytes:
        clean_text = (text or "").strip()
        if not clean_text:
            return _silence_wav_bytes(0.25, sample_rate=self.SAMPLE_RATE)

        voice = self._resolve_voice(voice_config)
        ssml = self._build_ssml(clean_text, voice, voice_config)

        # NullStream sink so we don't write a file; we'll handle the
        # PCM bytes directly via the Result's audio_data.
        synth = speechsdk.SpeechSynthesizer(
            speech_config=self._config,
            audio_config=None,
        )
        result = synth.speak_ssml_async(ssml).get()

        if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
            pcm_bytes = bytes(result.audio_data)
            return self._pcm_to_wav(pcm_bytes)

        if result.reason == speechsdk.ResultReason.Canceled:
            cancellation = result.cancellation_details
            code = getattr(cancellation, "error_code", None)
            detail = getattr(cancellation, "error_details", "") or ""
            reason_str = str(code) if code is not None else "unknown"
            # Treat billing/quota/auth as terminal so the engine stops
            # retrying Azure for this run.
            quota_signals = (
                "Forbidden", "QuotaExceeded", "Unauthorized",
                "InvalidSubscription", "InsufficientBalance",
                "TooManyRequests", "AuthenticationFailure",
            )
            if any(sig.lower() in (detail or "").lower() for sig in quota_signals) \
               or any(sig in reason_str for sig in ("Forbidden", "Unauthorized", "Quota")):
                logger.warning(
                    "Azure TTS quota/auth signal (%s): %s",
                    reason_str, detail[:160],
                )
                raise AzureQuotaExhausted(detail or reason_str)
            raise RuntimeError(f"Azure TTS cancelled ({reason_str}): {detail[:160]}")

        raise RuntimeError(f"Azure TTS unexpected reason: {result.reason}")

    def _pcm_to_wav(self, pcm_bytes: bytes) -> bytes:
        """Wrap raw 24 kHz 16-bit mono PCM in a WAV container."""
        audio = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        buf = io.BytesIO()
        sf.write(buf, audio, self.SAMPLE_RATE, format="WAV", subtype="PCM_16")
        return buf.getvalue()

    @staticmethod
    def _resolve_voice(voice_config: VoiceConfig) -> str:
        """Return the Azure voice name (e.g. en-US-AvaNeural).

        Reuses Edge's voice map so the catalog UI and cached signatures
        stay consistent across the two engines.
        """
        configured = (voice_config.voice or "").strip()
        if configured in EDGE_VOICE_MAP:
            return EDGE_VOICE_MAP[configured]
        if configured.startswith("edge_"):
            return EDGE_VOICE_MAP.get(configured, EDGE_VOICE_MAP["edge_ava"])
        # If the user passed a raw Azure voice id (en-US-...), accept it.
        if "-" in configured and configured[:2].isalpha():
            return configured
        return EDGE_VOICE_MAP["edge_ava"]

    def _build_ssml(self, text: str, voice: str, voice_config: VoiceConfig) -> str:
        """Build SSML that includes the prosody rate/pitch from VoiceConfig."""
        speed = float(getattr(voice_config, "speed", 1.0) or 1.0)
        # Azure prefers relative percent strings. Cap at +/- 50%.
        rate_pct = int(round(max(-50.0, min(50.0, (speed - 1.0) * 100))))
        rate_str = f"{rate_pct:+d}%"

        # Escape the text for SSML (quoted attrs not needed here; just
        # the body needs to be safe).
        escaped = (
            text.replace("&", "&amp;")
                .replace("<", "&lt;")
                .replace(">", "&gt;")
        )
        return (
            '<speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" xml:lang="en-US">'
            f'<voice name="{voice}">'
            f'<prosody rate="{rate_str}">{escaped}</prosody>'
            '</voice></speak>'
        )
