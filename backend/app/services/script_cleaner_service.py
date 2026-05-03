from __future__ import annotations

import importlib.util
import re
import sys
from functools import lru_cache
from pathlib import Path

from app.services.storytelling_style_guide import strip_storytelling_meta


@lru_cache(maxsize=1)
def _script_cleaner_class():
    module_path = Path(__file__).resolve().parents[3] / "script_cleaner.py"
    spec = importlib.util.spec_from_file_location("panelia_root_script_cleaner", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load script cleaner module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.ScriptCleaner


class ScriptCleanerService:
    _PANEL_FILLER_PHRASES = (
        "questions start piling up",
        "the world still feels normal",
        "by the end of the chapter",
        "spelled out in brutal detail",
        "the scene keeps evolving",
        "the story advances",
        "the chapter opens",
        "another tense beat",
        "the situation grows harder to explain",
        "the situation grows difficult to explain",
        "the next development",
        "another crucial moment",
        "the mood turns more urgent",
        "the pressure keeps mounting",
        "the stakes become clearer",
        "the consequences grow harder to ignore",
        "the situation grows harder to contain",
        "another revealing moment changes how the conflict is unfolding",
        "the mood sharpens as the next beat pushes the scene forward",
        "a pointed question abruptly changes the tone of the conversation",
        "a fresh email notification cuts through the moment",
        "a polite thanks leaves the exchange hanging in an awkward silence",
        "before the scene can settle",
        "as everyone absorbs what just happened",
        "the scene leans into the strange santa-themed spectacle surrounding the moment",
        "finally puts a name and history to the person at the center of the story",
        "fixation on the story becomes impossible to ignore",
        "a sharp question cuts through the moment",
        "one pointed question makes it clear",
        "the panel holds for a beat",
        "the moment catches on a single unanswered question",
        "a sudden question leaves the moment hanging",
        "the unanswered question freezes the scene",
        "a sharp question hangs in the air",
        "one abrupt question is enough to stall",
        "tension builds around",
        "the pressure around survival keeps rising",
    )
    _STACKED_TRANSITION_PATTERN = re.compile(
        r"^(?:then|next|soon|after that|at that point|at this point|for a moment|"
        r"from there|by now|meanwhile|in practice|that choice means|"
        r"the plan keeps moving as|the next step is clear as)"
        r"\s*,?\s*"
        r"(?:then|next|soon|after that|at that point|at this point|for a moment|"
        r"from there|by now|meanwhile|in practice|that choice means|"
        r"the plan keeps moving as|the next step is clear as)"
        r"\s*,?\s*",
        re.IGNORECASE,
    )
    _GENERIC_SUFFIX_PATTERNS = (
        r",\s*as the stakes become clearer\.?$",
        r",\s*as the pressure keeps mounting\.?$",
        r",\s*as the consequences grow harder to ignore\.?$",
        r",\s*as the situation grows harder to contain\.?$",
        r",\s*as the mood turns more urgent\.?$",
        r",\s*before the scene can settle\.?$",
        r",\s*as everyone absorbs what just happened\.?$",
        r",\s*while the confrontation stays unresolved\.?$",
        r",\s*while the reaction keeps rippling outward\.?$",
        r",\s*as the tension keeps hanging there\.?$",
        r",\s*for a beat longer\.?$",
    )

    def __init__(self) -> None:
        self.cleaner = _script_cleaner_class()()
        self.cleaner.similarity._load_model = lambda: None

    def clean_story_block(self, text: str) -> str:
        cleaned = self.cleaner.clean_script(text or "", ai_clean=False).strip()
        if not cleaned:
            return ""
        sentences = [sentence.strip() for sentence in cleaned.splitlines() if sentence.strip()]
        paragraphs: list[str] = []
        chunk: list[str] = []
        for sentence in sentences:
            chunk.append(sentence)
            if len(chunk) >= 3:
                paragraphs.append(" ".join(chunk))
                chunk = []
        if chunk:
            paragraphs.append(" ".join(chunk))
        return "\n\n".join(paragraphs).strip()

    def _strip_stacked_transitions(self, text: str) -> str:
        """Remove garbled double/triple transitions like 'Soon, by now,' or 'Next, next,'."""
        cleaned = str(text or "")
        for _ in range(5):
            new_cleaned = self._STACKED_TRANSITION_PATTERN.sub("", cleaned).strip()
            if new_cleaned == cleaned:
                break
            cleaned = new_cleaned
        # Restore capitalization if we stripped a prefix
        if cleaned and cleaned[0].islower():
            cleaned = cleaned[0].upper() + cleaned[1:]
        return cleaned

    def clean_panel_line(
        self,
        text: str,
        *,
        extracted_text: str = "",
        previous_lines: list[str] | None = None,
    ) -> str:
        previous_lines = previous_lines or []
        base = self._strip_stacked_transitions(self.humanize_inline_speakers(str(text or "")))
        base = strip_storytelling_meta(base)
        preserved_fact_line = self._normalize_panel_sentence(base)
        if (
            preserved_fact_line
            and not self._is_panel_filler(preserved_fact_line)
            and not self.is_first_person_narration(preserved_fact_line)
            and not (extracted_text and self._looks_like_raw_ocr_echo(preserved_fact_line, extracted_text))
            and self._should_preserve_fact_sentence(preserved_fact_line, extracted_text)
            and all(self.cleaner.similarity.similarity(preserved_fact_line, previous) < 0.92 for previous in previous_lines[-5:])
        ):
            return preserved_fact_line
        sentences = self._split_panel_sentences(base)
        sentences = [self._strip_generic_suffixes(sentence) for sentence in sentences]
        sentences = [sentence for sentence in sentences if sentence and not self._is_panel_filler(sentence)]
        sentences = self.cleaner.remove_visual_descriptions(sentences)
        sentences = self.cleaner.shorten_sentences(sentences)
        sentences = self.cleaner.enforce_narration_rules(sentences)

        candidates: list[str] = []
        for sentence in sentences:
            candidate = self._normalize_panel_sentence(sentence)
            if not candidate:
                continue
            if self.is_first_person_narration(candidate):
                continue
            if extracted_text and self._looks_like_raw_ocr_echo(candidate, extracted_text):
                continue
            candidates.append(candidate)

        if not candidates:
            fallback = self._normalize_panel_sentence(base)
            if fallback and not self._is_panel_filler(fallback) and not self.is_first_person_narration(fallback):
                candidates.append(fallback)

        for candidate in candidates:
            if all(self.cleaner.similarity.similarity(candidate, previous) < 0.92 for previous in previous_lines[-5:]):
                return candidate
        return candidates[0] if candidates else ""

    def _split_panel_sentences(self, text: str) -> list[str]:
        normalized = str(text or "").replace("\r", "\n")
        normalized = re.sub(r"\n{2,}", "\n", normalized)
        normalized = re.sub(r"[•●▪■]+", "\n", normalized)
        chunks = re.split(r"(?<=[.!?])\s+|\n+", normalized)
        sentences: list[str] = []
        for chunk in chunks:
            cleaned = self.cleaner._normalize_sentence(chunk)
            if cleaned:
                sentences.append(cleaned)
        return sentences

    def humanize_speaker_label(self, label: str) -> str:
        cleaned = re.sub(r"[_-]+", " ", str(label or "")).strip()
        if not cleaned:
            return ""

        match = re.match(r"^(.*?)(?:\s+(\d+))?$", cleaned)
        base = str(match.group(1) if match else cleaned).strip()
        ordinal = int(match.group(2) or 1) if match else 1
        normalized = base.casefold()

        if normalized in {"protagonist"}:
            return "Protagonist"
        if normalized in {"other", "unknown"}:
            return ""
        if normalized == "stranger":
            return ""
        if normalized == "crowd member":
            return "A voice in the crowd" if ordinal <= 1 else ""

        sentence_case = self._sentence_case_label(base)
        if ordinal > 1 and normalized not in {"manager"} and sentence_case:
            return f"Another {sentence_case.casefold()}"
        return sentence_case or ""

    def humanize_inline_speakers(self, text: str) -> str:
        result = str(text or "")
        # Context-aware "someone nearby" replacements FIRST (before generic deletion)
        # Handle adjective/modifier usage: "someone nearby items" → "other items"
        result = re.sub(
            r"\bsomeone nearby\s+(items?|men|man|women|woman|people|persons?|things?|ones?|side|parts?|options?|members?|figures?)\b",
            r"other \1", result, flags=re.IGNORECASE,
        )
        # "the someone nearby" → "the other", "two someone nearby men" → "two other men"
        result = re.sub(r"\b(the|two|three|a|each)\s+someone nearby\b", r"\1 other", result, flags=re.IGNORECASE)
        # "someone nearby looking" → "the other person, looking"
        result = re.sub(r"\bsomeone nearby\s+looking\b", "the other person, looking", result, flags=re.IGNORECASE)
        # Speaker-action: "Someone nearby states/says/asks/shouts/calls/exclaims" → "Another person ..."
        result = re.sub(
            r"\bsomeone nearby\s+(states|says|tells|asks|shouts|calls|exclaims|questions|announces|declares|mentions|replies|responds|notes|commands)\b",
            r"Another person \1", result, flags=re.IGNORECASE,
        )
        # Generic catch-all deletions (run after context-aware replacements)
        for pattern, replacement in (
            (r"\bStranger(?:\s+\d+)?\b", ""),
            (r"\bOther\b", ""),
            (r"\bSomeone nearby\b", ""),
            (r"\bAnother nearby voice\b", ""),
            (r"\bCrowd Member(?:\s+\d+)?\b", ""),
            (r"\bNeighbor(?:\s+2|\s+3|\s+4|\s+5)?\b", "a neighbor"),
            (r"\bRestaurant Worker(?:\s+\d+)?\b", "a restaurant worker"),
            (r"\bSecurity Guard(?:\s+\d+)?\b", "a guard"),
            (r"\bLoan Shark(?:\s+\d+)?\b", "a lender"),
        ):
            result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)
        return re.sub(r"\s+", " ", result).strip()

    def is_low_specificity_speaker_label(self, label: str) -> bool:
        lowered = str(label or "").strip().casefold()
        return lowered in {
            "",
            "other",
            "unknown",
            "someone nearby",
            "another nearby voice",
            "someone in the crowd",
            "another voice in the crowd",
        }

    def is_first_person_narration(self, text: str) -> bool:
        lowered = str(text or "").strip().casefold()
        if not lowered:
            return False
        return bool(
            re.search(
                r"\b(i|i'm|i’ve|i've|i’ll|i'll|i’d|i'd|me|my|mine|myself|we|we're|we’ve|we've|we’ll|we'll|us|our|ours|ourselves)\b",
                lowered,
            )
        )

    def _normalize_panel_sentence(self, sentence: str) -> str:
        cleaned = self.humanize_inline_speakers(str(sentence or ""))
        cleaned = self._strip_generic_suffixes(cleaned)
        cleaned = self.cleaner._normalize_sentence(cleaned)
        if (
            not cleaned
            or self._is_panel_filler(cleaned)
            or self.is_first_person_narration(cleaned)
            or self._is_low_information_sentence(cleaned)
        ):
            return ""
        return cleaned

    def _strip_generic_suffixes(self, sentence: str) -> str:
        cleaned = str(sentence or "").strip()
        for pattern in self._GENERIC_SUFFIX_PATTERNS:
            cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE)
        return re.sub(r"\s+", " ", cleaned).strip(" ,")

    def _is_panel_filler(self, sentence: str) -> bool:
        lowered = str(sentence or "").strip().casefold()
        if not lowered:
            return True
        return any(phrase in lowered for phrase in self._PANEL_FILLER_PHRASES)

    def _is_low_information_sentence(self, sentence: str) -> bool:
        cleaned = str(sentence or "").strip()
        if not cleaned:
            return True
        tokens = re.findall(r"[A-Za-z']+", cleaned)
        if len(tokens) <= 2 and cleaned.endswith("."):
            lowered = " ".join(token.casefold() for token in tokens)
            if lowered not in {"he", "she", "they", "we", "i"}:
                return True
        if re.fullmatch(r"[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2}\.?", cleaned):
            return True
        return False

    def _looks_like_raw_ocr_echo(self, narration: str, extracted_text: str) -> bool:
        normalized_narration = re.sub(r"\s+", " ", str(narration or "").casefold()).strip()
        normalized_text = re.sub(r"\s+", " ", str(extracted_text or "").casefold()).strip()
        if not normalized_narration or not normalized_text:
            return False
        if normalized_narration == normalized_text:
            return True
        return self.cleaner.similarity.similarity(normalized_narration, normalized_text) >= 0.86

    def _fact_anchor_tokens(self, text: str) -> list[str]:
        lowered = re.sub(r"\s+", " ", str(text or "").casefold()).strip()
        anchors: list[str] = []
        anchors.extend(re.findall(r"\b(?:19|20)\d{2}\b", lowered)[:2])
        anchors.extend(
            [
                re.sub(r"\s+", " ", match).strip()
                for match in re.findall(r"\b\d[\d,]*(?:\.\d+)?\s*(?:light[- ]?years?|degrees?|days?|months?|years?)\b", lowered)
            ][:2]
        )
        for keyword in (
            "supernova",
            "freeze",
            "frozen apocalypse",
            "apocalypse",
            "blizzard",
            "temperature",
            "blue star",
            "world",
            "storage space",
            "vault door",
        ):
            if keyword in lowered and keyword not in anchors:
                anchors.append(keyword)
        return anchors

    def _should_preserve_fact_sentence(self, candidate: str, extracted_text: str) -> bool:
        anchors = self._fact_anchor_tokens(extracted_text)
        if len(anchors) < 2:
            return False
        lowered_candidate = re.sub(r"\s+", " ", str(candidate or "").casefold()).strip()
        matched = sum(1 for anchor in anchors if anchor in lowered_candidate)
        numeric_anchors = [anchor for anchor in anchors if any(character.isdigit() for character in anchor)]
        if numeric_anchors and any(anchor in lowered_candidate for anchor in numeric_anchors):
            return True
        return matched >= 2

    def _sentence_case_label(self, text: str) -> str:
        cleaned = " ".join(part for part in str(text or "").split() if part).strip()
        if not cleaned:
            return ""
        return cleaned[0].upper() + cleaned[1:].lower()
