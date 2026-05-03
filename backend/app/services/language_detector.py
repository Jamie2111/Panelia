from __future__ import annotations

import re
from threading import Lock
from typing import Any

from app.services.dialogue_cleaner import DialogueCleaner


class LanguageDetector:
    _TRANSLATOR_CACHE: dict[str, tuple[Any, Any]] = {}
    _TRANSLATED_TEXT_CACHE: dict[tuple[str, str], str] = {}
    _LOAD_LOCK = Lock()

    def __init__(self, cleaner: DialogueCleaner | None = None) -> None:
        self.cleaner = cleaner or DialogueCleaner()

    def normalize_language_code(self, language_code: str | None) -> str:
        if not language_code:
            return "en"
        lowered = language_code.lower()
        if lowered in {"a", "en-us", "en-gb"}:
            return "en"
        if lowered.startswith("pt"):
            return "pt"
        if lowered.startswith("es"):
            return "es"
        if lowered.startswith("fr"):
            return "fr"
        if lowered.startswith("de"):
            return "de"
        if lowered.startswith("tr"):
            return "tr"
        if lowered.startswith("id"):
            return "id"
        if lowered.startswith("th"):
            return "th"
        if lowered.startswith("it"):
            return "it"
        if lowered.startswith("ro"):
            return "ro"
        if lowered.startswith("ca"):
            return "ca"
        if lowered.startswith("gl"):
            return "gl"
        if lowered.startswith("ja"):
            return "ja"
        if lowered.startswith("ko"):
            return "ko"
        if lowered.startswith("zh"):
            return "zh"
        if lowered.startswith("en"):
            return "en"
        return lowered

    def detect(self, text: str, language_hint: str | None = None) -> str:
        cleaned = self.cleaner.clean_text(text)
        hint = self.normalize_language_code(language_hint)
        if not cleaned:
            return hint

        try:
            from langdetect import detect

            detected = self.normalize_language_code(detect(cleaned))
            if self._should_prefer_language_hint(cleaned, detected, hint):
                return hint
            return detected
        except Exception:
            return self._heuristic_language(cleaned, hint)

    def translate_to_english(self, text: str, language_code: str | None) -> str:
        cleaned = self.cleaner.clean_text(text)
        if not cleaned:
            return ""
        language = self.normalize_language_code(language_code)
        if language in {"en", "a"}:
            return cleaned
        cache_key = (language, cleaned)
        cached = self._TRANSLATED_TEXT_CACHE.get(cache_key)
        if cached is not None:
            return cached
        translated = self.translate_batch_to_english([cleaned], language)[0]
        self._TRANSLATED_TEXT_CACHE[cache_key] = translated
        return translated

    def translate_batch_to_english(self, texts: list[str], language_code: str | None) -> list[str]:
        language = self.normalize_language_code(language_code)
        cleaned_texts = [self.cleaner.clean_text(text) for text in texts]
        if language in {"en", "a"}:
            return cleaned_texts
        model_name = self._translator_model_name(language)
        if not model_name:
            return cleaned_texts
        try:
            tokenizer, model = self._get_translator(model_name)
            encoded = tokenizer(cleaned_texts, return_tensors="pt", truncation=True, max_length=512, padding=True)
            try:
                import torch

                with torch.no_grad():
                    generated = model.generate(**encoded, max_length=512)
            except Exception:
                generated = model.generate(**encoded, max_length=512)
            decoded = tokenizer.batch_decode(generated, skip_special_tokens=True)
            translated = [self.cleaner.clean_text(text) or original for text, original in zip(decoded, cleaned_texts, strict=False)]
            if len(translated) < len(cleaned_texts):
                translated.extend(cleaned_texts[len(translated) :])
            return translated
        except Exception:
            return cleaned_texts

    def translation_supported(self, language_code: str | None) -> bool:
        return bool(self._translator_model_name(self.normalize_language_code(language_code)))

    def _translator_model_name(self, language: str) -> str | None:
        return {
            "ja": "Helsinki-NLP/opus-mt-ja-en",
            "ko": "Helsinki-NLP/opus-mt-ko-en",
            "zh": "Helsinki-NLP/opus-mt-zh-en",
            "pt": "Helsinki-NLP/opus-mt-ROMANCE-en",
            "es": "Helsinki-NLP/opus-mt-ROMANCE-en",
            "fr": "Helsinki-NLP/opus-mt-ROMANCE-en",
            "de": "Helsinki-NLP/opus-mt-de-en",
            "tr": "Helsinki-NLP/opus-mt-tr-en",
            "id": "Helsinki-NLP/opus-mt-id-en",
            "it": "Helsinki-NLP/opus-mt-ROMANCE-en",
            "ro": "Helsinki-NLP/opus-mt-ROMANCE-en",
            "ca": "Helsinki-NLP/opus-mt-ROMANCE-en",
            "gl": "Helsinki-NLP/opus-mt-ROMANCE-en",
        }.get(language)

    def _heuristic_language(self, text: str, language_hint: str) -> str:
        ranges = {
            "ja": r"[\u3040-\u30ff\u31f0-\u31ff]",
            "ko": r"[\uac00-\ud7af]",
            "zh": r"[\u4e00-\u9fff]",
        }
        for code, pattern in ranges.items():
            if re.search(pattern, text):
                if code == "zh" and re.search(r"[\u3040-\u30ff]", text):
                    return "ja"
                return code
        if re.search(r"[A-Za-z]", text):
            if language_hint in {"pt", "es", "fr", "de", "tr", "id", "it", "ro", "ca", "gl"}:
                return language_hint
            return "en"
        return language_hint

    def _should_prefer_language_hint(self, text: str, detected_language: str, language_hint: str) -> bool:
        if not language_hint or language_hint in {"en", "a"} or detected_language == language_hint:
            return False
        if language_hint not in {"pt", "es", "fr", "de", "tr", "id", "it", "ro", "ca", "gl"}:
            return False
        if not re.search(r"[A-Za-z]", text):
            return False
        tokens = re.findall(r"[a-z]{2,}", self.cleaner.clean_text(text).casefold())
        if not tokens:
            return False
        accent_markers = {
            "pt": {"cao", "ção", "nh", "lh", "que", "uma", "você", "voce"},
            "es": {"que", "una", "está", "esta", "por", "para"},
            "fr": {"que", "une", "vous", "avec", "pour"},
            "de": {"der", "die", "das", "und", "nicht", "mit"},
            "tr": {"bir", "ve", "icin", "için", "degil", "degil", "olan", "kadar"},
            "id": {"yang", "dan", "untuk", "tidak", "dengan"},
            "it": {"che", "una", "con", "per"},
            "ro": {"este", "pentru", "care"},
            "ca": {"que", "una", "amb"},
            "gl": {"que", "unha", "para"},
        }
        markers = accent_markers.get(language_hint, set())
        return any(marker.casefold() in text.casefold() for marker in markers)

    def _get_translator(self, model_name: str) -> tuple[Any, Any]:
        cached = self._TRANSLATOR_CACHE.get(model_name)
        if cached is not None:
            return cached
        with self._LOAD_LOCK:
            cached = self._TRANSLATOR_CACHE.get(model_name)
            if cached is not None:
                return cached
            from transformers import MarianMTModel, MarianTokenizer

            tokenizer = MarianTokenizer.from_pretrained(model_name)
            model = MarianMTModel.from_pretrained(model_name)
            try:
                model.eval()
            except Exception:
                pass
            self._TRANSLATOR_CACHE[model_name] = (tokenizer, model)
            return tokenizer, model
