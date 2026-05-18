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


class _AzureRESTError(RuntimeError):
    """Internal error envelope from the REST path. EdgeTTSService never
    sees this directly - _render_to_wav_bytes converts it to either
    AzureQuotaExhausted or plain RuntimeError before it leaves the
    service."""

    def __init__(self, *, code: str, detail: str, is_quota: bool) -> None:
        super().__init__(f"{code}: {detail[:160]}")
        self.code = code
        self.detail = detail
        self.is_quota = is_quota


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

        # Route via the REST API instead of the C++ Speech SDK. The SDK
        # consistently segfaults (SIGSEGV exit -11) when its internal
        # codec layer fails to start ("Codec decoding is not started
        # within 2s") - the failure path takes the worker child down
        # entirely instead of raising a clean Python exception.
        # The REST endpoint produces identical audio (same neural
        # voice models on the same servers) with no SDK state to
        # corrupt. Errors come back as standard HTTP responses we
        # can handle cleanly.
        try:
            pcm_bytes = self._render_via_rest(ssml)
            return self._pcm_to_wav(pcm_bytes)
        except _AzureRESTError as exc:
            # Quota / auth → terminal; everything else → transient.
            if exc.is_quota:
                logger.warning("Azure TTS REST quota/auth: %s", exc.detail[:160])
                raise AzureQuotaExhausted(exc.detail or exc.code) from exc
            raise RuntimeError(f"Azure TTS REST failed ({exc.code}): {exc.detail[:160]}") from exc

    def _render_via_rest(self, ssml: str) -> bytes:
        """Direct REST call to Azure Speech /cognitiveservices/v1.

        Returns raw PCM bytes (Raw24Khz16BitMonoPcm format).
        Raises _AzureRESTError on any non-200 response.
        """
        import httpx

        # Build a short-lived auth token (10-min TTL). Reuse the cached
        # token if it's still valid to avoid burning a call per synth.
        token = self._get_or_refresh_token()
        region = getattr(self.settings, "azure_speech_region", "") or "eastus"
        url = f"https://{region}.tts.speech.microsoft.com/cognitiveservices/v1"
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/ssml+xml",
            "X-Microsoft-OutputFormat": "raw-24khz-16bit-mono-pcm",
            "User-Agent": "Panelia-AzureTTS/1.0",
        }
        # Short timeout per request; the caller (EdgeTTSService) handles
        # retries with backoff if we raise transiently.
        try:
            with httpx.Client(timeout=30.0) as client:
                resp = client.post(url, headers=headers, content=ssml.encode("utf-8"))
        except httpx.HTTPError as exc:
            raise _AzureRESTError(code="network", detail=str(exc), is_quota=False) from exc

        if resp.status_code == 200:
            return resp.content

        # Map HTTP errors. 401/403 = auth/quota, 429 = rate limit,
        # everything else = transient. Token might also be expired
        # (Azure returns 401), so on 401 force a refresh + retry once.
        body = (resp.text or "")[:500]
        if resp.status_code == 401:
            # Token may have expired between fetch and call. Force a
            # refresh and retry once before declaring auth failure.
            self._auth_token = None
            self._auth_token_expires_at = 0.0
            try:
                token = self._get_or_refresh_token()
                headers["Authorization"] = f"Bearer {token}"
                with httpx.Client(timeout=30.0) as client:
                    resp2 = client.post(url, headers=headers, content=ssml.encode("utf-8"))
                if resp2.status_code == 200:
                    return resp2.content
                raise _AzureRESTError(
                    code=f"http_{resp2.status_code}",
                    detail=(resp2.text or "")[:500],
                    is_quota=True,
                )
            except _AzureRESTError:
                raise
            except Exception as exc:  # noqa: BLE001
                raise _AzureRESTError(
                    code="auth_refresh_failed",
                    detail=str(exc),
                    is_quota=True,
                ) from exc
        is_quota = resp.status_code in (402, 403, 429)
        raise _AzureRESTError(
            code=f"http_{resp.status_code}",
            detail=body,
            is_quota=is_quota,
        )

    def _get_or_refresh_token(self) -> str:
        """Cache the issued auth token for ~9 minutes (token TTL is 10).

        Cheaper than burning a token fetch on every synth call.
        """
        import time as _time
        import httpx

        now = _time.time()
        existing = getattr(self, "_auth_token", None)
        expires = getattr(self, "_auth_token_expires_at", 0.0)
        if existing and now < expires - 30:
            return existing

        region = getattr(self.settings, "azure_speech_region", "") or "eastus"
        key = getattr(self.settings, "azure_speech_key", "")
        if not key:
            raise _AzureRESTError(code="no_key", detail="azure_speech_key not configured", is_quota=True)
        token_url = f"https://{region}.api.cognitive.microsoft.com/sts/v1.0/issueToken"
        try:
            with httpx.Client(timeout=10.0) as client:
                resp = client.post(
                    token_url,
                    headers={"Ocp-Apim-Subscription-Key": key, "Content-Length": "0"},
                )
        except httpx.HTTPError as exc:
            raise _AzureRESTError(code="token_network", detail=str(exc), is_quota=False) from exc
        if resp.status_code != 200:
            is_quota = resp.status_code in (401, 403)
            raise _AzureRESTError(
                code=f"token_http_{resp.status_code}",
                detail=(resp.text or "")[:300],
                is_quota=is_quota,
            )
        self._auth_token = resp.text
        # Tokens are valid for 10 minutes; refresh after 9.
        self._auth_token_expires_at = now + 9 * 60
        return self._auth_token

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
