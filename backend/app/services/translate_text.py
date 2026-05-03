from __future__ import annotations

import json
import logging
import re
from typing import Any

import requests

from app.core.config import get_settings
from app.services.language_detector import LanguageDetector

logger = logging.getLogger(__name__)


class TranslateTextService:
    _LEGACY_MODEL_MAP = {
        "gemini-2.0-flash": "gemini-2.5-flash-lite",
        "gemini-2.0-flash-001": "gemini-2.5-flash-lite",
        "gemini-2.0-flash-lite": "gemini-2.5-flash-lite",
        "gemini-2.0-flash-lite-001": "gemini-2.5-flash-lite",
        "gemini-1.5-flash": "gemini-2.5-flash-lite",
    }

    def __init__(self, language_detector: LanguageDetector | None = None) -> None:
        self.settings = get_settings()
        self.language_detector = language_detector or LanguageDetector()
        self._cache: dict[tuple[str, str], str] = {}
        self._gemini_disabled_reason: str | None = None

    def translate_batch(self, texts: list[str], language_code: str, context_hint: str = "") -> list[str]:
        language = self.language_detector.normalize_language_code(language_code)
        cleaned = [self._pre_clean_ocr(text.strip()) for text in texts]
        if language == "en":
            return cleaned
        if not self.settings.gemini_api_key:
            return self.language_detector.translate_batch_to_english(cleaned, language)

        unresolved: list[str] = []
        resolved: dict[str, str] = {}
        for text in cleaned:
            key = (language, text)
            cached = self._cache.get(key)
            if cached is not None:
                resolved[text] = cached
            elif text:
                unresolved.append(text)
            else:
                resolved[text] = ""

        if unresolved:
            translated = self._gemini_translate(unresolved, language, context_hint)
            for original, english in zip(unresolved, translated, strict=False):
                key = (language, original)
                self._cache[key] = english
                resolved[original] = english

        return [resolved.get(text, text) for text in cleaned]

    def translate(self, text: str, language_code: str, context_hint: str = "") -> str:
        return self.translate_batch([text], language_code, context_hint)[0]

    def _pre_clean_ocr(self, text: str) -> str:
        """Fix common OCR artifacts before translation."""
        cleaned = str(text or "")
        # Remove spurious periods inserted mid-word (common OCR error)
        cleaned = re.sub(r"(\w)\.\s+(\w)", r"\1 \2", cleaned)
        # Collapse repeated whitespace
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        return cleaned

    def _gemini_translate(self, texts: list[str], language_code: str, context_hint: str = "") -> list[str]:
        if self._gemini_disabled_reason:
            return self.language_detector.translate_batch_to_english(texts, language_code)
        context_block = f"\nContext (manga/comic title or scene): {context_hint}\n" if context_hint else ""
        prompt = (
            "You are translating OCR-extracted text from a manga comic.\n"
            "The OCR may contain errors: broken words, mixed languages, spurious punctuation.\n"
            f"{context_block}"
            "For each line:\n"
            "- First correct obvious OCR errors (spurious periods mid-word, mixed language fragments)\n"
            "- Then translate the corrected text into natural English\n"
            "- Preserve character names, place names, and proper nouns exactly\n"
            "Return valid JSON only in this format:\n"
            '{"translations":[{"index":0,"translation":"..."}]}\n\n'
            f"Source language hint: {language_code}\n"
            "Lines:\n"
            + "\n".join(f"{index}: {text}" for index, text in enumerate(texts))
        )
        try:
            payload: dict[str, Any] | None = None
            last_exc: Exception | None = None
            for model in self._gemini_models():
                try:
                    response = requests.post(
                        f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
                        params={"key": self.settings.gemini_api_key},
                        headers={"Content-Type": "application/json"},
                        json={
                            "contents": [{"parts": [{"text": prompt}]}],
                            "generationConfig": {
                                "temperature": 0.1,
                                "maxOutputTokens": 1400,
                                "responseMimeType": "application/json",
                            },
                        },
                        timeout=90,
                    )
                    response.raise_for_status()
                    payload = response.json()
                    break
                except requests.HTTPError as exc:
                    last_exc = exc
                    status_code = exc.response.status_code if exc.response is not None else None
                    if status_code != 404:
                        raise
                    continue
            if payload is None:
                if last_exc is not None:
                    raise last_exc
                raise ValueError("No usable Gemini translation model was available")
            text = self._extract_text(payload)
            parsed = self._parse_json(text)
            raw_translations = parsed.get("translations") if isinstance(parsed, dict) else None
            if not isinstance(raw_translations, list):
                raise ValueError("Gemini translation payload was missing translations")
            resolved: dict[int, str] = {}
            for item in raw_translations:
                if not isinstance(item, dict):
                    continue
                match = re.search(r"\d+", str(item.get("index") or ""))
                if not match:
                    continue
                index = int(match.group(0))
                translation = str(item.get("translation") or "").strip()
                original = texts[index] if index < len(texts) else ""
                if translation and self._translation_looks_valid(translation, original):
                    resolved[index] = translation
            return [resolved.get(index, original) for index, original in enumerate(texts)]
        except Exception as exc:
            self._gemini_disabled_reason = str(exc)
            logger.warning("Gemini translation fell back to local translator: %s", self._safe_error(exc))
            return self.language_detector.translate_batch_to_english(texts, language_code)

    def _safe_error(self, exc: Exception) -> str:
        return re.sub(r"key=[^&\s)]+", "key=<redacted>", str(exc))

    def _gemini_models(self) -> list[str]:
        candidates = [
            self._resolve_model_name(str(self.settings.llm_gemini_model or "").strip()),
            self._resolve_model_name(str(self.settings.gemini_model or "").strip()),
            "gemini-2.5-flash-lite",
            "gemini-2.5-flash",
        ]
        ordered: list[str] = []
        for model in candidates:
            if model and model not in ordered:
                ordered.append(model)
        return ordered

    def _resolve_model_name(self, model_name: str) -> str:
        stripped = model_name.strip()
        return self._LEGACY_MODEL_MAP.get(stripped, stripped or "gemini-2.5-flash-lite")

    def _extract_text(self, payload: dict[str, Any]) -> str:
        candidates = payload.get("candidates", [])
        if not candidates:
            raise ValueError("Gemini translation returned no candidates")
        parts = candidates[0].get("content", {}).get("parts", [])
        text = "\n".join(str(part.get("text") or "").strip() for part in parts if isinstance(part, dict))
        if not text.strip():
            raise ValueError("Gemini translation returned empty content")
        return text.strip()

    def _parse_json(self, raw_text: str) -> Any:
        cleaned = raw_text.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
            cleaned = re.sub(r"\s*```$", "", cleaned)
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            match = re.search(r"(\{.*\}|\[.*\])", cleaned, re.DOTALL)
            if not match:
                raise
            return json.loads(match.group(1))

    def _translation_looks_valid(self, translation: str, original: str) -> bool:
        """Return False if the translation is garbled, echoed back, or not English."""
        if not translation.strip():
            return False
        # Echo-back: translation is just the original text unchanged
        if translation.strip().casefold() == original.strip().casefold():
            return False
        # Accented character clusters: if > 25% of alpha chars are accented, likely not English
        alpha_chars = [c for c in translation if c.isalpha()]
        if alpha_chars:
            accented = sum(1 for c in alpha_chars if ord(c) > 127)
            if accented / len(alpha_chars) > 0.25:
                logger.debug("Translation rejected (accented ratio %.0f%%): %s", accented / len(alpha_chars) * 100, translation[:80])
                return False
        # Character soup: > 30% digits/punctuation
        if len(translation) > 10:
            non_alpha = sum(1 for c in translation if not c.isalpha() and not c.isspace())
            if non_alpha / len(translation) > 0.30:
                return False
        # Basic English word check: at least 40% of words should be recognizable
        _BASIC_ENGLISH = {
            "the", "a", "an", "i", "you", "he", "she", "it", "we", "they",
            "is", "are", "was", "were", "be", "been", "being", "am",
            "has", "have", "had", "do", "does", "did", "will", "would",
            "could", "should", "may", "might", "can", "shall", "must",
            "to", "of", "in", "for", "on", "with", "at", "by", "from",
            "as", "into", "through", "during", "before", "after",
            "and", "but", "or", "not", "no", "if", "so", "that", "this",
            "what", "who", "how", "when", "where", "why", "which",
            "his", "her", "its", "my", "your", "our", "their",
            "him", "them", "me", "us", "up", "out", "about", "over",
            "all", "one", "two", "new", "just", "also", "than", "more",
            "very", "too", "only", "now", "then", "here", "there",
            "said", "says", "told", "asked", "went", "came", "got",
            "made", "took", "gave", "know", "think", "see", "look",
            "want", "need", "let", "kill", "die", "fight", "run",
            "come", "go", "get", "make", "take", "give", "keep",
        }
        words = re.findall(r"[a-z']+", translation.casefold())
        if len(words) >= 5:
            english_count = sum(1 for w in words if w in _BASIC_ENGLISH)
            if english_count / len(words) < 0.40:
                logger.debug("Translation rejected (English ratio %.0f%%): %s", english_count / len(words) * 100, translation[:80])
                return False
        return True
