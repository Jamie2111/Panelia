from __future__ import annotations

import re
from typing import Iterable

from app.services.ocr_cleaner import clean_ocr_lines, clean_ocr_text, combined_dialogue_entry_lines, is_usable_ocr_text


class DialogueCleaner:
    def clean_text(self, text: str) -> str:
        cleaned = clean_ocr_text(text)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        cleaned = re.sub(r"([!?.,])\1{2,}", r"\1\1", cleaned)
        cleaned = re.sub(r"\s+([,.;:!?])", r"\1", cleaned)
        return cleaned.strip()

    def normalize_dialogue(self, text: str) -> str:
        cleaned = self.clean_text(text)
        if not cleaned:
            return ""
        if any(char.isalpha() for char in cleaned) and cleaned[-1] not in ".!?":
            cleaned += "."
        return cleaned

    def clean_lines(self, lines: Iterable[str]) -> list[str]:
        return clean_ocr_lines(self.clean_text(line) for line in lines)

    def combine_entry_lines(self, entries: Iterable[dict[str, object]]) -> list[str]:
        return clean_ocr_lines(combined_dialogue_entry_lines(entries))

    def merge_broken_lines(self, text: str) -> str:
        raw_lines = [segment.strip() for segment in str(text or "").splitlines() if segment.strip()]
        if not raw_lines:
            return self.normalize_dialogue(text)
        merged = " ".join(self.clean_lines(raw_lines)).strip()
        return self.normalize_dialogue(merged)

    def dedupe_lines(self, lines: Iterable[str]) -> list[str]:
        deduped: list[str] = []
        seen: set[str] = set()
        for line in self.clean_lines(lines):
            key = line.casefold()
            if key in seen:
                continue
            seen.add(key)
            deduped.append(line)
        return deduped

    def is_usable(self, text: str) -> bool:
        return is_usable_ocr_text(self.clean_text(text))
