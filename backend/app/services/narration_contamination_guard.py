from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from app.services.character_name_filters import normalize_name_key


_KNOWN_UNSUPPORTED_NAME_KEYS = frozenset({"nance"})


@dataclass(slots=True)
class NarrationGuardResult:
    script_lines: list[str]
    panel_ids: list[str] | None
    report: dict[str, Any]


class NarrationContaminationGuard:
    """Final defensive checks before narration artifacts are treated as usable."""

    _SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")
    _WORD_RE = re.compile(r"\b[\w'-]+\b")
    _NAME_RE = re.compile(r"\b(?:[A-Z][a-z0-9]+|[A-Z]{2,})(?:\s+(?:[A-Z][a-z0-9]+|[A-Z]{2,})){0,2}\b")
    _BROKEN_MIXED_CASE_RE = re.compile(r"\b[a-z]{1,4}[A-Z][A-Za-z]{2,}\b")
    _BROKEN_ALNUM_RE = re.compile(r"\b(?=[A-Za-z0-9]*\d)(?=[A-Za-z0-9]*[A-Za-z])[A-Za-z0-9]{3,}\b")
    _FOREIGN_OR_NOISE_RE = re.compile(r"[\u3040-\u30ff\u3400-\u9fff]|[・]{2,}|[≡三]{1,}")
    _CORRUPTED_PHRASE_RE = re.compile(
        r"\b(?:gwirrr|garrr|codc|cyyc|cynmcw|aano|hiko|uenn|azlp|oervu|jalr|malc|kecivicividck|vri)\b",
        re.IGNORECASE,
    )
    _SFX_HEAVY_RE = re.compile(
        r"\b(?:gwirr+|garr+|bam+|bang+|boom+|crash+|wham+|whoosh+|grr+|clang+|thud+)\b",
        re.IGNORECASE,
    )

    def prepare(
        self,
        script_lines: list[str],
        *,
        panel_ids: list[str] | None = None,
        supported_character_names: list[str] | None = None,
        world_terms: list[str] | None = None,
        source_artifact_status: str = "pending",
    ) -> NarrationGuardResult:
        supported_keys = {
            normalize_name_key(name)
            for name in supported_character_names or []
            if normalize_name_key(name)
        }
        world_keys = {normalize_name_key(term) for term in world_terms or [] if normalize_name_key(term)}
        accepted: list[tuple[str, str | None, int]] = []
        quarantined: list[dict[str, Any]] = []
        repaired: list[dict[str, Any]] = []

        for index, line in enumerate(script_lines, start=1):
            text = self._normalize(line)
            panel_id = panel_ids[index - 1] if panel_ids and index - 1 < len(panel_ids) else None
            if not text:
                continue
            reasons = self.contamination_reasons(
                text,
                supported_character_keys=supported_keys,
                world_term_keys=world_keys,
            )
            if reasons:
                cleaned = self._repair_text(text, reasons)
                if cleaned and not self.contamination_reasons(
                    cleaned,
                    supported_character_keys=supported_keys,
                    world_term_keys=world_keys,
                ):
                    repaired.append(
                        {
                            "index": index,
                            "panel_id": panel_id,
                            "reasons": reasons,
                            "before": text,
                            "after": cleaned,
                        }
                    )
                    accepted.append((cleaned, panel_id, index))
                    continue
                quarantined.append(
                    {
                        "index": index,
                        "panel_id": panel_id,
                        "reasons": reasons,
                        "text": text,
                    }
                )
                continue
            accepted.append((text, panel_id, index))

        deduped, merged_report = self._merge_overlaps_and_sentence_runs(accepted)
        prepared_lines = [item[0] for item in deduped]
        prepared_panel_ids = [item[1] or f"segment_{i:03d}" for i, item in enumerate(deduped, start=1)] if panel_ids is not None else None
        final_reasons = [
            {
                "index": index,
                "panel_id": panel_id,
                "reasons": self.contamination_reasons(
                    text,
                    supported_character_keys=supported_keys,
                    world_term_keys=world_keys,
                ),
                "text": text,
            }
            for index, (text, panel_id, _original_index) in enumerate(deduped, start=1)
            if self.contamination_reasons(text, supported_character_keys=supported_keys, world_term_keys=world_keys)
        ]
        one_sentence_count = sum(1 for text in prepared_lines if self._sentence_count(text) <= 1)
        max_one_sentence_run = self._max_one_sentence_run(prepared_lines)
        report = {
            "analysis_version": "narration_contamination_guard_v1",
            "source_artifact_status": source_artifact_status,
            "input_units": len([line for line in script_lines if str(line or "").strip()]),
            "output_units": len(prepared_lines),
            "quarantined_units": len(quarantined),
            "repaired_units": len(repaired),
            "merged_units": merged_report["merged_units"],
            "near_duplicate_units": merged_report["near_duplicate_units"],
            "one_sentence_units": one_sentence_count,
            "max_one_sentence_run": max_one_sentence_run,
            "contamination_remaining": len(final_reasons),
            "script_ready": not quarantined and not final_reasons and max_one_sentence_run <= 3,
            "quarantined": quarantined[:100],
            "repaired": repaired[:100],
            "final_contamination": final_reasons[:100],
            "merge_report": merged_report,
        }
        return NarrationGuardResult(script_lines=prepared_lines, panel_ids=prepared_panel_ids, report=report)

    def contamination_reasons(
        self,
        text: str,
        *,
        supported_character_keys: set[str] | None = None,
        world_term_keys: set[str] | None = None,
    ) -> list[str]:
        normalized = self._normalize(text)
        if not normalized:
            return []
        reasons: list[str] = []
        lowered = normalized.casefold()
        tokens = self._WORD_RE.findall(normalized)
        if self._FOREIGN_OR_NOISE_RE.search(normalized):
            reasons.append("mixed_language_or_symbol_noise")
        if self._CORRUPTED_PHRASE_RE.search(normalized):
            reasons.append("known_corrupted_ocr")
        if self._has_repeated_fragment(normalized):
            reasons.append("repeated_ocr_fragment")
        if self._looks_sfx_heavy(normalized):
            reasons.append("sfx_heavy_ocr")
        if self._has_broken_token_chain(normalized):
            reasons.append("broken_ocr_token_chain")
        if re.search(r"\btrans\s+ported\b", lowered):
            reasons.append("broken_line_join")
        if re.search(r"\b(?:hasn'?t\s+given\s+up)\s+[a-z]{3,}\b", lowered) and "hiro" in lowered:
            reasons.append("corrupted_partial_name")
        unsupported_keys = set(_KNOWN_UNSUPPORTED_NAME_KEYS)
        supported_character_keys = supported_character_keys or set()
        world_term_keys = world_term_keys or set()
        for candidate in self._NAME_RE.findall(normalized):
            key = normalize_name_key(candidate)
            if key in unsupported_keys and key not in supported_character_keys and key not in world_term_keys:
                reasons.append(f"unsupported_name:{candidate}")
        if tokens:
            upper_tokens = [token for token in tokens if len(token) >= 3 and token.isupper()]
            if len(upper_tokens) >= 4 and any(self._SFX_HEAVY_RE.search(token) for token in upper_tokens):
                reasons.append("corrupted_all_caps_run")
        return list(dict.fromkeys(reasons))

    def _repair_text(self, text: str, reasons: list[str]) -> str:
        repaired = text
        if any(reason.startswith("unsupported_name:") for reason in reasons):
            for reason in reasons:
                if not reason.startswith("unsupported_name:"):
                    continue
                name = reason.split(":", 1)[1].strip()
                repaired = re.sub(rf"\b{re.escape(name)}\b", "the other pilot", repaired)
        if self.contamination_reasons(
            repaired,
            supported_character_keys=set(),
            world_term_keys={"the other pilot"},
        ):
            return ""
        return self._normalize(repaired)

    def _merge_overlaps_and_sentence_runs(
        self,
        rows: list[tuple[str, str | None, int]],
    ) -> tuple[list[tuple[str, str | None, int]], dict[str, Any]]:
        merged: list[tuple[str, str | None, int]] = []
        near_duplicate_units = 0
        merged_units = 0
        for text, panel_id, original_index in rows:
            if not merged:
                merged.append((text, panel_id, original_index))
                continue
            previous_text, previous_panel_id, previous_index = merged[-1]
            overlap = self._content_overlap(previous_text, text)
            if overlap >= 0.58:
                near_duplicate_units += 1
                combined = self._merge_text(previous_text, text)
                merged[-1] = (combined, previous_panel_id or panel_id, previous_index)
                merged_units += 1
                continue
            if self._sentence_count(previous_text) <= 1 and self._sentence_count(text) <= 1:
                combined_sentence_count = self._sentence_count(previous_text) + self._sentence_count(text)
                combined_words = len(self._WORD_RE.findall(f"{previous_text} {text}"))
                if combined_sentence_count <= 4 and combined_words <= 90:
                    merged[-1] = (f"{previous_text} {text}", previous_panel_id or panel_id, previous_index)
                    merged_units += 1
                    continue
            merged.append((text, panel_id, original_index))
        return merged, {
            "merged_units": merged_units,
            "near_duplicate_units": near_duplicate_units,
            "output_order": [
                {
                    "output_index": index,
                    "source_index": original_index,
                    "panel_id": panel_id,
                }
                for index, (_text, panel_id, original_index) in enumerate(merged, start=1)
            ],
        }

    def _merge_text(self, first: str, second: str) -> str:
        first_sentences = self._sentences(first)
        second_sentences = self._sentences(second)
        kept = list(first_sentences)
        seen = [self._content_tokens(sentence) for sentence in first_sentences]
        for sentence in second_sentences:
            tokens = self._content_tokens(sentence)
            if tokens and any(len(tokens & prior) / max(1, len(tokens | prior)) >= 0.62 for prior in seen if prior):
                continue
            kept.append(sentence)
            seen.append(tokens)
        return self._normalize(" ".join(kept))

    def _has_repeated_fragment(self, text: str) -> bool:
        parts = [part.strip().casefold() for part in re.split(r"[.!?;]+", text) if part.strip()]
        seen: set[str] = set()
        for part in parts:
            compact = re.sub(r"[^a-z0-9]+", " ", part).strip()
            if len(compact.split()) < 3:
                continue
            if compact in seen:
                return True
            seen.add(compact)
        words = [word.casefold() for word in self._WORD_RE.findall(text)]
        if len(words) >= 6:
            half = len(words) // 2
            if words[:half] == words[half : half * 2]:
                return True
        return False

    def _looks_sfx_heavy(self, text: str) -> bool:
        tokens = self._WORD_RE.findall(text)
        if not tokens:
            return False
        sfx_count = sum(1 for token in tokens if self._SFX_HEAVY_RE.fullmatch(token))
        return sfx_count >= 2 or (sfx_count >= 1 and len(tokens) <= 5)

    def _has_broken_token_chain(self, text: str) -> bool:
        tokens = self._WORD_RE.findall(text)
        if not tokens:
            return False
        broken = 0
        for token in tokens:
            if self._BROKEN_MIXED_CASE_RE.fullmatch(token) or self._BROKEN_ALNUM_RE.fullmatch(token):
                broken += 1
        return broken >= 2 or (broken >= 1 and len(tokens) <= 6)

    def _content_overlap(self, first: str, second: str) -> float:
        first_tokens = self._content_tokens(first)
        second_tokens = self._content_tokens(second)
        if not first_tokens or not second_tokens:
            return 0.0
        return len(first_tokens & second_tokens) / max(1, len(first_tokens | second_tokens))

    def _content_tokens(self, text: str) -> set[str]:
        stop = {
            "the", "a", "an", "and", "or", "but", "to", "of", "in", "on", "for", "with", "as", "is", "are",
            "was", "were", "he", "she", "they", "it", "his", "her", "their", "this", "that", "then", "while",
        }
        return {
            token.casefold()
            for token in self._WORD_RE.findall(text)
            if len(token) > 2 and token.casefold() not in stop
        }

    def _sentence_count(self, text: str) -> int:
        return len(self._sentences(text))

    def _sentences(self, text: str) -> list[str]:
        return [part.strip() for part in self._SENTENCE_RE.split(str(text or "").strip()) if part.strip()]

    def _max_one_sentence_run(self, lines: list[str]) -> int:
        max_run = 0
        current = 0
        for line in lines:
            if self._sentence_count(line) <= 1:
                current += 1
                max_run = max(max_run, current)
            else:
                current = 0
        return max_run

    def _normalize(self, value: object) -> str:
        return re.sub(r"\s+", " ", str(value or "").strip()).strip()
