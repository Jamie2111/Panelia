from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
import re


OFFICIAL_HINTS = (
    "official",
    "simulpub",
    "mangaplus",
    "manga plus",
    "viz",
    "webtoon official",
    "tappytoon",
    "tapas official",
    "yen press",
    "kodansha",
    "square enix",
)


@dataclass
class ChapterCandidate:
    source_url: str
    chapter_number_raw: str | None
    chapter_number_value: float | None
    chapter_key: str | None
    language: str | None = None
    group_name: str | None = None
    is_official: bool = False
    page_count: int | None = None
    updated_at: str | int | float | None = None
    metadata: dict[str, object] = field(default_factory=dict)


def infer_is_official(group_name: str | None) -> bool:
    if not group_name:
        return False
    lowered = group_name.casefold()
    return any(hint in lowered for hint in OFFICIAL_HINTS)


def normalize_language_code(value: str | None) -> str | None:
    cleaned = str(value or "").strip().casefold()
    return cleaned or None


def parse_chapter_number(value: str | int | float | None) -> tuple[str | None, float | None]:
    if value is None:
        return None, None
    cleaned = str(value).strip()
    if not cleaned:
        return None, None
    match = re.match(r"^\d+(?:\.\d+)?$", cleaned)
    if not match:
        return None, None
    numeric = float(cleaned)
    if numeric.is_integer():
        return str(int(numeric)), numeric
    normalized = cleaned.rstrip("0").rstrip(".")
    return normalized, numeric


def parse_range_spec(spec: str | None) -> list[tuple[float, float]] | None:
    cleaned = str(spec or "").strip()
    if not cleaned:
        return None
    ranges: list[tuple[float, float]] = []
    for part in [segment.strip() for segment in cleaned.split(",") if segment.strip()]:
        match = re.match(r"^(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)$", part)
        if match:
            start = float(match.group(1))
            end = float(match.group(2))
            if end < start:
                start, end = end, start
            ranges.append((start, end))
            continue
        match = re.match(r"^(\d+(?:\.\d+)?)$", part)
        if match:
            value = float(match.group(1))
            ranges.append((value, value))
            continue
        raise ValueError(f"Invalid chapter range segment: {part}")
    return ranges


def chapter_in_ranges(chapter_number: float | None, ranges: list[tuple[float, float]] | None) -> bool:
    if chapter_number is None:
        return False
    if not ranges:
        return True
    for start, end in ranges:
        if start <= chapter_number <= end:
            return True
    return False


def select_chapters(
    candidates: list[ChapterCandidate],
    *,
    chapter_range: str | None = None,
    preferred_language: str | None = None,
    duplicate_mode: str = "auto_pick_best",
    default_first_if_no_range: bool = False,
) -> list[ChapterCandidate]:
    ranges = parse_range_spec(chapter_range)
    filtered = [candidate for candidate in candidates if candidate.chapter_number_value is not None and candidate.chapter_key]
    if preferred_language:
        preferred = normalize_language_code(preferred_language)
        exact = [candidate for candidate in filtered if normalize_language_code(candidate.language) == preferred]
        if exact:
            filtered = exact
        elif preferred not in {None, "", "any"}:
            raise ValueError(f"No chapters were found for the requested source language: {preferred_language}")

    if ranges:
        filtered = [candidate for candidate in filtered if chapter_in_ranges(candidate.chapter_number_value, ranges)]
    elif default_first_if_no_range and filtered:
        first_value = min(candidate.chapter_number_value for candidate in filtered if candidate.chapter_number_value is not None)
        filtered = [candidate for candidate in filtered if candidate.chapter_number_value == first_value]

    grouped: dict[str, list[ChapterCandidate]] = {}
    for candidate in filtered:
        grouped.setdefault(candidate.chapter_key or "", []).append(candidate)
    group_counts: dict[str, int] = {}
    for candidate in filtered:
        key = str(candidate.group_name or "").strip().casefold()
        if key:
            group_counts[key] = group_counts.get(key, 0) + 1

    def timestamp_score(value: str | int | float | None) -> float:
        if value is None:
            return 0.0
        if isinstance(value, (int, float)):
            return float(value)
        text = str(value).strip()
        if not text:
            return 0.0
        try:
            return datetime.fromisoformat(text.replace("Z", "+00:00")).timestamp()
        except ValueError:
            return 0.0

    def candidate_score(candidate: ChapterCandidate) -> tuple[float, ...]:
        group_key = str(candidate.group_name or "").strip().casefold()
        official_score = 1.0 if candidate.is_official else 0.0
        fan_score = 0.0 if candidate.is_official else 1.0
        group_score = float(group_counts.get(group_key, 0))
        completeness = float(candidate.page_count or 0)
        recency = timestamp_score(candidate.updated_at)
        stable = candidate.source_url
        if duplicate_mode == "prefer_official":
            return (official_score, completeness, group_score, recency, stable)
        if duplicate_mode == "prefer_fan":
            return (fan_score, completeness, group_score, recency, stable)
        if duplicate_mode == "prefer_consistent_group":
            return (group_score, official_score, completeness, recency, stable)
        return (official_score, completeness, group_score, recency, stable)

    selected: list[ChapterCandidate] = []
    for group in grouped.values():
        picked = sorted(group, key=candidate_score, reverse=True)[0]
        selected.append(picked)

    return sorted(
        selected,
        key=lambda candidate: (
            candidate.chapter_number_value if candidate.chapter_number_value is not None else float("inf"),
            candidate.source_url,
        ),
    )
