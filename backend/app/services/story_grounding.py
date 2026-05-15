from __future__ import annotations

import difflib
import re
from typing import Any

from app.schemas.project import ChapterMetadata
from app.services.character_name_filters import looks_like_false_character_name, normalize_name_key

_NON_NAME_KEYS: frozenset[str] = frozenset(
    {
        "after",
        "amidst",
        "before",
        "during",
        "earth",
        "friday",
        "here",
        "however",
        "in",
        "inside",
        "later",
        "meanwhile",
        "monday",
        "morning",
        "night",
        "outside",
        "plantation",
        "saturday",
        "sunday",
        "then",
        "there",
        "thursday",
        "tuesday",
        "wednesday",
    }
)


def compact_chapter_metadata(chapter_metadata: ChapterMetadata | dict[str, Any] | Any) -> dict[str, Any]:
    raw_payload = _coerce_metadata_dict(chapter_metadata)
    raw = raw_payload.get("raw") if isinstance(raw_payload.get("raw"), dict) else {}
    manga = raw.get("manga") if isinstance(raw.get("manga"), dict) else {}
    chapter = raw.get("chapter") if isinstance(raw.get("chapter"), dict) else {}

    compact: dict[str, Any] = {
        "manga_title": str(raw_payload.get("manga_title") or manga.get("title") or "").strip(),
        "chapter_title": str(raw_payload.get("chapter_title") or chapter.get("name") or "").strip(),
        "chapter_number": str(raw_payload.get("chapter_number") or chapter.get("number") or "").strip(),
        "volume_number": str(raw_payload.get("volume_number") or chapter.get("volume") or "").strip(),
        "language": str(raw_payload.get("language") or chapter.get("language") or "").strip(),
        "page_count": _coerce_positive_int(raw_payload.get("page_count")),
        "series_type": str(manga.get("type") or "").strip(),
        "series_slug": str(manga.get("slug") or "").strip(),
        "original_language": str(manga.get("original_language") or "").strip(),
        "series_status": str(manga.get("status") or "").strip(),
        "series_synopsis": str(manga.get("synopsis") or raw_payload.get("series_synopsis") or "").strip()[:2400],
        "series_alt_titles": _unique_clean_strings(manga.get("alt_titles") or raw_payload.get("series_alt_titles") or [], limit=12),
        "source_urls": _normalize_source_urls(raw_payload.get("source_url")),
        "series_cast_hints": _unique_clean_strings(raw_payload.get("series_cast_hints") or [], limit=24),
        "canonical_name_corrections": _normalize_corrections(raw_payload.get("canonical_name_corrections") or []),
    }
    return {key: value for key, value in compact.items() if value not in ("", None, [], {})}


def build_name_grounding(
    chapter_metadata: ChapterMetadata | dict[str, Any] | Any,
    character_dictionary: dict[str, Any],
    protagonist_name: str | None,
) -> dict[str, Any]:
    metadata = compact_chapter_metadata(chapter_metadata)
    corrections = _normalize_corrections(metadata.get("canonical_name_corrections") or [])
    allowed_name_map: dict[str, str] = {}

    def register(canonical: str, *variants: str) -> None:
        canonical_clean = str(canonical or "").strip()
        if not canonical_clean or looks_like_false_character_name(canonical_clean):
            return
        canonical_key = normalize_name_key(canonical_clean)
        if not canonical_key:
            return
        allowed_name_map.setdefault(canonical_key, canonical_clean)
        for value in variants:
            variant_clean = str(value or "").strip()
            if not variant_clean or looks_like_false_character_name(variant_clean):
                continue
            variant_key = normalize_name_key(variant_clean)
            if variant_key:
                allowed_name_map.setdefault(variant_key, canonical_clean)

    for name, info in (character_dictionary or {}).items():
        display_name = (
            str(info.get("display_name") or info.get("name") or name).strip()
            if isinstance(info, dict)
            else str(name).strip()
        )
        aliases = info.get("aliases", []) if isinstance(info, dict) else []
        register(display_name, str(name), *[str(alias).strip() for alias in aliases or []])

    for hint in metadata.get("series_cast_hints", []) or []:
        register(str(hint))
    if protagonist_name:
        register(str(protagonist_name))
    for item in corrections:
        register(str(item.get("canonical") or ""), str(item.get("variant") or ""))

    allowed_names = []
    seen_names: set[str] = set()
    for canonical in allowed_name_map.values():
        key = normalize_name_key(canonical)
        if key and key not in seen_names:
            seen_names.add(key)
            allowed_names.append(canonical)

    metadata["series_cast_hints"] = _unique_clean_strings(
        [*metadata.get("series_cast_hints", []), *allowed_names],
        limit=30,
    )
    metadata["canonical_name_corrections"] = corrections

    return {
        "chapter_metadata": metadata,
        "allowed_character_names": allowed_names,
        "allowed_name_map": allowed_name_map,
        "canonical_name_corrections": corrections,
        "protagonist_name": str(protagonist_name or "").strip(),
    }


def canonicalize_character_name(value: object, grounding: dict[str, Any], *, strict: bool = False) -> str:
    raw = str(value or "").strip()
    if not raw or looks_like_false_character_name(raw):
        return ""
    corrected = raw
    for item in grounding.get("canonical_name_corrections", []) or []:
        variant = str(item.get("variant") or "").strip()
        canonical = str(item.get("canonical") or "").strip()
        if variant and canonical and normalize_name_key(corrected) == normalize_name_key(variant):
            corrected = canonical
            break

    key = normalize_name_key(corrected)
    allowed_name_map = grounding.get("allowed_name_map") or {}
    if key in allowed_name_map:
        return str(allowed_name_map[key]).strip()
    if not strict or not allowed_name_map:
        return corrected

    allowed_keys = list(allowed_name_map.keys())
    cutoff = 0.92 if len(key) <= 5 else 0.88
    matches = difflib.get_close_matches(key, allowed_keys, n=1, cutoff=cutoff)
    if matches:
        return str(allowed_name_map[matches[0]]).strip()
    return ""


def apply_name_corrections_to_text(text: str, grounding: dict[str, Any]) -> str:
    """Replace each ``variant`` with its ``canonical`` form, but never
    duplicate the canonical when a substitution would extend an existing
    match.

    Naive ``re.sub(\\bvariant\\b, canonical, ...)`` corrupts the text when
    the variant is a token *inside* the canonical (the most common case is
    a first-name alias of a multi-word canonical, e.g. ``variant="Zero"``
    for ``canonical="Zero Two"``). On a sentence that already says
    "Zero Two stands…", the substitution rewrites the leading "Zero" into
    "Zero Two", producing the gibberish "Zero Two Two stands…". To prevent
    that we build a regex with a lookbehind for the canonical tokens that
    appear before the variant and a lookahead for the tokens that appear
    after, so the substitution skips occurrences that are already part of
    the canonical phrase.
    """
    corrected = str(text or "")
    corrections = grounding.get("canonical_name_corrections", []) or []
    ordered = sorted(
        (
            (
                str(item.get("variant") or "").strip(),
                str(item.get("canonical") or "").strip(),
            )
            for item in corrections
        ),
        key=lambda item: len(item[0]),
        reverse=True,
    )
    for variant, canonical in ordered:
        if not variant or not canonical:
            continue
        if variant.casefold() == canonical.casefold():
            continue
        var_tokens = variant.split()
        can_tokens = canonical.split()
        can_lower = [token.casefold() for token in can_tokens]
        var_lower = [token.casefold() for token in var_tokens]
        var_pos: int | None = None
        if 0 < len(var_lower) <= len(can_lower):
            for start in range(len(can_lower) - len(var_lower) + 1):
                if can_lower[start : start + len(var_lower)] == var_lower:
                    var_pos = start
                    break

        pattern = rf"\b{re.escape(variant)}\b"
        if var_pos is not None:
            before = " ".join(can_tokens[:var_pos])
            after = " ".join(can_tokens[var_pos + len(var_tokens) :])
            if before:
                pattern = rf"(?<!\b{re.escape(before)}\s){pattern}"
            if after:
                pattern = rf"{pattern}(?!\s+{re.escape(after)}\b)"
        try:
            corrected = re.sub(pattern, canonical, corrected, flags=re.IGNORECASE)
        except re.error:
            # Variable-width lookbehind on some Python versions can throw -
            # fall back to the naive replacement so we still apply the
            # correction (a stray double-name is preferable to a missing
            # canonical).
            corrected = re.sub(
                rf"\b{re.escape(variant)}\b", canonical, corrected, flags=re.IGNORECASE
            )
    return corrected


def extract_proper_name_candidates(text: str) -> list[str]:
    candidates: list[str] = []
    seen: set[str] = set()
    for phrase in re.findall(r"\b(?:[A-Z][a-z0-9]+|[A-Z]{2,})(?:\s+(?:[A-Z][a-z0-9]+|[A-Z]{2,})){0,2}\b", str(text or "")):
        key = normalize_name_key(phrase)
        if not key or key in seen or key in _NON_NAME_KEYS:
            continue
        if looks_like_false_character_name(phrase):
            continue
        seen.add(key)
        candidates.append(phrase.strip())
    return candidates


def contains_unapproved_names(
    text: str,
    grounding: dict[str, Any],
    *,
    world_terms: list[str] | None = None,
    extra_allowed_names: list[str] | None = None,
) -> bool:
    approved = {
        normalize_name_key(item)
        for item in grounding.get("allowed_character_names", []) or []
        if normalize_name_key(item)
    }
    approved.update(
        normalize_name_key(item)
        for item in extra_allowed_names or []
        if normalize_name_key(item)
    )
    approved.update(
        normalize_name_key(item)
        for item in world_terms or []
        if normalize_name_key(item)
    )
    for candidate in extract_proper_name_candidates(text):
        if normalize_name_key(candidate) not in approved:
            return True
    return False


def _coerce_metadata_dict(chapter_metadata: ChapterMetadata | dict[str, Any] | Any) -> dict[str, Any]:
    if isinstance(chapter_metadata, dict):
        return dict(chapter_metadata)
    if isinstance(chapter_metadata, ChapterMetadata):
        return chapter_metadata.model_dump(mode="json")
    model_dump = getattr(chapter_metadata, "model_dump", None)
    if callable(model_dump):
        payload = model_dump(mode="json")
        if isinstance(payload, dict):
            return payload
    return {}


def _normalize_source_urls(value: object) -> list[str]:
    if isinstance(value, list):
        raw_values = [str(item).strip() for item in value if str(item).strip()]
    else:
        raw_values = [line.strip() for line in str(value or "").splitlines() if line.strip()]
    return raw_values[:8]


def _normalize_corrections(raw: object) -> list[dict[str, str]]:
    corrections: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for item in raw or []:
        if not isinstance(item, dict):
            continue
        variant = str(item.get("variant") or "").strip()
        canonical = str(item.get("canonical") or "").strip()
        if not variant or not canonical:
            continue
        if looks_like_false_character_name(variant) and looks_like_false_character_name(canonical):
            continue
        signature = (normalize_name_key(variant), normalize_name_key(canonical))
        if not all(signature) or signature in seen:
            continue
        seen.add(signature)
        corrections.append({"variant": variant, "canonical": canonical})
    return corrections[:24]


def _unique_clean_strings(values: object, *, limit: int) -> list[str]:
    cleaned: list[str] = []
    seen: set[str] = set()
    for item in values or []:
        value = str(item or "").strip()
        key = normalize_name_key(value)
        if not value or not key or key in seen:
            continue
        seen.add(key)
        cleaned.append(value)
        if len(cleaned) >= limit:
            break
    return cleaned


def _coerce_positive_int(value: object) -> int | None:
    try:
        parsed = int(value)  # type: ignore[arg-type]
    except Exception:
        return None
    return parsed if parsed > 0 else None
