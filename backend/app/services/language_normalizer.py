from __future__ import annotations

import re

from app.schemas.project import VoiceConfig
from app.services.language_detector import LanguageDetector
from app.services.story_preprocessor import NarrationUnit


class LanguageNormalizer:
    _KOKORO_LANG_MAP = {
        "en": "a",
        "es": "e",
        "pt": "p",
        "ja": "j",
        "zh": "z",
        "ko": "k",
    }

    def __init__(self) -> None:
        self._detector = LanguageDetector()

    def apply(
        self,
        units: list[NarrationUnit],
        voice_config: VoiceConfig,
        language_hint: str | None = None,
    ) -> list[NarrationUnit]:
        locked_lang_code = self._locked_voice_lang_code(voice_config.lang_code)
        for unit in units:
            detected = self._detector.detect(unit.raw_text or unit.story_text, language_hint)
            unit.language = detected
            unit.spoken_text = self._normalize_text(unit.spoken_text, detected)
            unit.metadata["normalized_language"] = detected
            unit.metadata["kokoro_lang_code"] = locked_lang_code
        return units

    def _preferred_lang_code(self, detected_language: str, configured_lang_code: str) -> str:
        if configured_lang_code:
            return configured_lang_code
        return self._KOKORO_LANG_MAP.get(detected_language, configured_lang_code or "a")

    def _locked_voice_lang_code(self, configured_lang_code: str | None) -> str:
        configured = str(configured_lang_code or "").strip()
        if configured:
            return configured
        return "a"

    def _normalize_text(self, value: str, language: str) -> str:
        text = str(value or "").strip()
        text = text.replace("…", "...")
        text = text.replace("—", ", ")
        text = text.replace("–", ", ")
        if language in {"zh", "ja"}:
            text = text.replace("，", ", ").replace("。", ". ").replace("！", "! ").replace("？", "? ")
        elif language == "ko":
            text = text.replace("，", ", ").replace("。", ". ").replace("！", "! ").replace("？", "? ")
        elif language in {"pt", "es", "en"}:
            text = re.sub(r"\s*([,.!?])\s*", r"\1 ", text)
        return re.sub(r"\s+", " ", text).strip()
