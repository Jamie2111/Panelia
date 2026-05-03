from __future__ import annotations

import re
from collections import Counter
from dataclasses import asdict, dataclass
from typing import Any

from app.services.character_name_filters import normalize_name_key
from app.services.story_grounding import extract_proper_name_candidates


_PLACEHOLDER_NAME_RE = re.compile(
    r"\b(?:unknown|speaker|narrator|protagonist|character|figure|someone|"
    r"person|man|woman|boy|girl|victim)\b",
    re.IGNORECASE,
)

_TEAM_HEADS = {
    "crew",
    "crowd",
    "family",
    "group",
    "members",
    "neighbors",
    "pack",
    "pilots",
    "squad",
    "students",
    "survivors",
    "team",
    "tribe",
    "villagers",
}

_STAKE_NOUNS = {
    "argument",
    "attack",
    "bargain",
    "battle",
    "confrontation",
    "crisis",
    "decision",
    "escape",
    "fight",
    "mission",
    "move",
    "pact",
    "partner",
    "partnership",
    "pilot",
    "piloting",
    "preparation",
    "request",
    "resource",
    "resources",
    "revenge",
    "shelter",
    "supplies",
    "supply",
    "survival",
    "threat",
}

_ENEMY_NOUNS = {
    "attacker",
    "creature",
    "enemy",
    "monster",
    "opponent",
    "threat",
    "villain",
}

_STOP_TERMS = {
    "ability",
    "chapter",
    "piloting",
    "pilot",
    "scene",
    "panel",
    "story",
    "moment",
    "situation",
    "thing",
    "someone",
    "something",
    "everyone",
    "anything",
    "nothing",
}

_WEAK_STYLE_PHRASES = {
    "a female pilot",
    "a former prodigy pilot",
    "a male pilot",
    "a failed pilot",
    "a partner",
    "a pilot",
    "a capable pilot",
    "a former prodigy",
    "a formidable pilot",
    "a relationship with",
    "an elite pilot",
    "an exceptionally skilled pilot",
    "current existential crisis",
    "first piloting",
    "former prodigy pilot",
    "his current existential crisis",
    "his inability to pilot",
    "his initial piloting",
    "her partners",
    "her to pilot",
    "her previous partners",
    "his partner",
    "piloting",
    "pilot",
    "partners",
    "the ability to pilot",
    "the female pilot",
    "the former prodigy pilot",
    "the male pilot",
    "the failed pilot",
    "the pilot",
    "the piloting",
    "the purpose of pilot",
    "the purpose of piloting",
    "first pilot",
    "mock battle",
    "series",
    "an exceptional pilot",
    "the children",
    "the development",
    "the earth",
    "the first pilot",
    "the mock battle",
    "the series",
}


@dataclass(frozen=True, slots=True)
class StyleVocabulary:
    """Per-chapter narrative fill-ins used by generic bridge templates."""

    named_characters: tuple[str, ...] = ()
    protagonist: str | None = None
    antagonist_term: str | None = None
    team_term: str | None = None
    world_terms: tuple[str, ...] = ()
    stakes_phrases: tuple[str, ...] = ()
    action_verbs: tuple[str, ...] = ()
    allowed_drift_keys: frozenset[str] = frozenset()

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["allowed_drift_keys"] = sorted(self.allowed_drift_keys)
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> "StyleVocabulary | None":
        if not isinstance(payload, dict):
            return None
        return cls(
            named_characters=tuple(str(item).strip() for item in payload.get("named_characters", []) if str(item).strip()),
            protagonist=str(payload.get("protagonist") or "").strip() or None,
            antagonist_term=str(payload.get("antagonist_term") or "").strip() or None,
            team_term=str(payload.get("team_term") or "").strip() or None,
            world_terms=tuple(str(item).strip() for item in payload.get("world_terms", []) if str(item).strip()),
            stakes_phrases=tuple(str(item).strip() for item in payload.get("stakes_phrases", []) if str(item).strip()),
            action_verbs=tuple(str(item).strip() for item in payload.get("action_verbs", []) if str(item).strip()),
            allowed_drift_keys=frozenset(
                str(item).strip()
                for item in payload.get("allowed_drift_keys", [])
                if str(item).strip()
            ),
        )


def build_style_vocabulary(
    *,
    canonical_characters: list[Any] | None = None,
    character_dictionary: dict[str, Any] | None = None,
    story_bible: dict[str, Any] | None = None,
    scene_summaries: list[dict[str, Any]] | dict[str, Any] | list[str] | None = None,
    chapter_summary: str | None = None,
) -> StyleVocabulary:
    """Build project-specific wording from project artefacts only.

    The extraction is intentionally deterministic and LLM-free. It can run on
    older projects that only have a character dictionary and story bible, while
    newer vision-first projects benefit from canonical character aliases.
    """

    story_bible = story_bible or {}
    corpus = _corpus_text(story_bible, scene_summaries, chapter_summary)
    canonical_entries = _canonical_entries(canonical_characters, character_dictionary, story_bible)
    named_characters, protagonist = _rank_characters(canonical_entries, corpus)
    world_terms = _extract_world_terms(story_bible, corpus, named_characters)
    team_term = _extract_team_term(corpus, world_terms)
    stakes_phrases = _extract_stakes_phrases(corpus, world_terms)
    action_verbs = _extract_action_verbs(corpus)
    antagonist_term = _extract_antagonist_term(canonical_entries, world_terms, corpus)

    drift_values: list[str] = []
    drift_values.extend(named_characters)
    drift_values.extend(world_terms)
    drift_values.extend(stakes_phrases)
    drift_values.extend(action_verbs)
    for value in (protagonist, antagonist_term, team_term):
        if value:
            drift_values.append(value)
    allowed_drift_keys = frozenset(
        key
        for value in drift_values
        for key in _drift_keys_for_phrase(value)
        if key
    )

    return StyleVocabulary(
        named_characters=tuple(named_characters),
        protagonist=protagonist,
        antagonist_term=antagonist_term,
        team_term=team_term,
        world_terms=tuple(world_terms),
        stakes_phrases=tuple(stakes_phrases),
        action_verbs=tuple(action_verbs),
        allowed_drift_keys=allowed_drift_keys,
    )


def _canonical_entries(
    canonical_characters: list[Any] | None,
    character_dictionary: dict[str, Any] | None,
    story_bible: dict[str, Any],
) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for raw in canonical_characters or []:
        data = _as_mapping(raw)
        name = str(data.get("name") or data.get("display_name") or "").strip()
        if not name or _PLACEHOLDER_NAME_RE.search(name):
            continue
        entries.append(
            {
                "name": name,
                "role": str(data.get("role") or "").strip(),
                "aliases": [str(alias).strip() for alias in data.get("aliases", []) or [] if str(alias).strip()],
            }
        )
    for raw in (story_bible.get("cast") or []):
        data = _as_mapping(raw)
        name = str(data.get("name") or data.get("display_name") or data.get("canonical_name") or raw).strip()
        if not name or _PLACEHOLDER_NAME_RE.search(name):
            continue
        if any(normalize_name_key(entry["name"]) == normalize_name_key(name) for entry in entries):
            continue
        entries.append(
            {
                "name": name,
                "role": str(data.get("role") or "").strip(),
                "aliases": [str(alias).strip() for alias in data.get("aliases", []) or [] if str(alias).strip()],
            }
        )
    for key, raw in (character_dictionary or {}).items():
        data = _as_mapping(raw)
        name = str(data.get("display_name") or data.get("name") or key).strip()
        if not name or _PLACEHOLDER_NAME_RE.search(name):
            continue
        if any(normalize_name_key(entry["name"]) == normalize_name_key(name) for entry in entries):
            continue
        entries.append({"name": name, "role": str(data.get("role") or "").strip(), "aliases": []})
    return entries


def _rank_characters(entries: list[dict[str, Any]], corpus: str) -> tuple[list[str], str | None]:
    counts: Counter[str] = Counter()
    corpus_key = normalize_name_key(corpus)
    for entry in entries:
        name = str(entry.get("name") or "").strip()
        aliases = [name, *(entry.get("aliases") or [])]
        key = normalize_name_key(name)
        if not key:
            continue
        counts[name] += 1
        for alias in aliases:
            alias_key = normalize_name_key(alias)
            if alias_key:
                counts[name] += corpus_key.count(alias_key)
    ranked = sorted(
        (str(entry.get("name") or "").strip() for entry in entries if str(entry.get("name") or "").strip()),
        key=lambda name: (-counts[name], name.casefold()),
    )
    protagonist = next(
        (
            str(entry.get("name") or "").strip()
            for entry in entries
            if str(entry.get("role") or "").casefold() in {"protagonist", "lead", "main"}
            and str(entry.get("name") or "").strip()
        ),
        ranked[0] if ranked else None,
    )
    return ranked[:12], protagonist


def _extract_world_terms(story_bible: dict[str, Any], corpus: str, named_characters: list[str]) -> list[str]:
    terms: list[str] = []
    for raw in story_bible.get("world_terms", []) or []:
        if isinstance(raw, dict):
            value = str(raw.get("name") or raw.get("term") or "").strip()
        else:
            value = str(raw or "").strip()
        if _usable_term(value, named_characters):
            terms.append(value)

    counts: Counter[str] = Counter()
    for name in extract_proper_name_candidates(corpus):
        if _usable_term(name, named_characters):
            counts[name] += 1
    for phrase in re.findall(r"\b(?:the|a|an)\s+[a-z][a-z-]{3,}(?:\s+[a-z][a-z-]{3,})?\b", corpus, flags=re.IGNORECASE):
        normalized = phrase.strip()
        if _usable_term(normalized, named_characters):
            counts[normalized] += 1
    keyword_nouns = _STAKE_NOUNS | _ENEMY_NOUNS | {"ability", "alliance", "class", "city", "power", "rank", "school"}
    noun_pattern = "|".join(sorted(re.escape(noun) for noun in keyword_nouns))
    for match in re.finditer(rf"\b(?:[a-z-]{{4,}}\s+){{0,2}}(?:{noun_pattern})s?\b", corpus, flags=re.IGNORECASE):
        phrase = _clean_phrase(match.group(0))
        if _usable_term(phrase, named_characters):
            counts[phrase] += 1
    terms.extend(name for name, count in counts.most_common(12) if count >= 2)
    return _dedupe_phrases(terms)[:10]


def _extract_team_term(corpus: str, world_terms: list[str]) -> str | None:
    counts: Counter[str] = Counter()
    for match in re.finditer(
        r"\b(?:the|their|his|her|our|a|an)\s+(?:[a-z0-9-]+\s+){0,2}"
        r"(crew|crowd|family|group|members|neighbors|pack|pilots|squad|students|survivors|team|tribe|villagers)\b",
        corpus,
        flags=re.IGNORECASE,
    ):
        phrase = _clean_team_phrase(match.group(0))
        head = match.group(1).casefold()
        if head in _TEAM_HEADS:
            counts[phrase] += 1
    for match in re.finditer(r"\b[A-Z][A-Za-z]*(?:\s+\d{1,3})\b", corpus):
        phrase = match.group(0).strip()
        head = phrase.split()[0].casefold()
        if head in _TEAM_HEADS:
            counts[phrase] += 3
    for match in re.finditer(
        r"\b(?:other|nearby|remaining|local|frightened|surviving)\s+"
        r"(crew|crowd|family|group|members|neighbors|pack|pilots|squad|students|survivors|team|tribe|villagers)\b",
        corpus,
        flags=re.IGNORECASE,
    ):
        counts[_clean_team_phrase(match.group(0))] += 2
    for term in world_terms:
        head = str(term).split()[-1].casefold() if str(term).split() else ""
        if head in _TEAM_HEADS:
            counts[_clean_team_phrase(str(term))] += 2
    if not counts:
        return None
    return _clean_team_phrase(counts.most_common(1)[0][0])


def _extract_stakes_phrases(corpus: str, world_terms: list[str]) -> list[str]:
    counts: Counter[str] = Counter()
    noun_pattern = "|".join(sorted(_STAKE_NOUNS))
    for match in re.finditer(
        rf"\b(?:the|a|an|this|that|his|her|their|our)\s+(?:[a-z-]+\s+){{0,2}}(?:{noun_pattern})s?\b",
        corpus,
        flags=re.IGNORECASE,
    ):
        counts[_clean_phrase(match.group(0))] += 1
    for term in world_terms:
        term_text = str(term).strip()
        if term_text and any(noun in term_text.casefold() for noun in _STAKE_NOUNS):
            counts[term_text] += 2
        elif term_text and re.search(r"\b(?:enemy|monster|creature|threat|attack|fight|battle|mission)\b", term_text, flags=re.IGNORECASE):
            counts[f"the {term_text.strip()} threat"] += 1
        elif term_text and re.search(r"\b(?:pilot|partner|alliance|team|squad|crew|family)\b", term_text, flags=re.IGNORECASE):
            counts[f"the {term_text.strip()} pressure"] += 1
    for match in re.finditer(
        r"\b(?:failed|upcoming|dangerous|reckless|unresolved|nearby|immediate|next|central)\s+"
        r"(?:choice|connection|exchange|pressure|test|danger|order|battle|fight|mission|attack|request|decision)\b",
        corpus,
        flags=re.IGNORECASE,
    ):
        counts[_clean_phrase(match.group(0))] += 1
    phrases = [
        phrase
        for phrase, count in counts.most_common(12)
        if count >= 1 and _usable_stakes_phrase(phrase)
    ]
    return _dedupe_phrases(phrases)[:6]


def _extract_action_verbs(corpus: str) -> list[str]:
    counts: Counter[str] = Counter(
        match.group(0).casefold()
        for match in re.finditer(r"\b[a-z]{5,}ing\b", corpus, flags=re.IGNORECASE)
    )
    weak_verbs = {"hovering", "prompting", "resulting", "suggesting"}
    verbs = [verb for verb, count in counts.most_common(8) if count >= 1 and verb not in weak_verbs]
    if verbs:
        return verbs[:8]
    return ["pressing", "scrambling", "pushing", "facing"]


def _extract_antagonist_term(entries: list[dict[str, Any]], world_terms: list[str], corpus: str) -> str | None:
    for entry in entries:
        if str(entry.get("role") or "").casefold() == "antagonist":
            name = str(entry.get("name") or "").strip()
            if name:
                return name
    lowered = corpus.casefold()
    candidates = list(world_terms)
    candidates.extend(re.findall(r"\b(?:the|a|an)\s+(?:[a-z-]+\s+)?(?:enemy|monster|attacker|threat|villain|opponent|creature)s?\b", corpus, flags=re.IGNORECASE))
    for candidate in candidates:
        text = str(candidate).strip()
        words = set(re.findall(r"[a-z]+", text.casefold()))
        if words & _ENEMY_NOUNS:
            return _clean_phrase(text)
    if any(noun in lowered for noun in _ENEMY_NOUNS):
        return "the threat"
    return None


def _corpus_text(
    story_bible: dict[str, Any],
    scene_summaries: list[dict[str, Any]] | dict[str, Any] | list[str] | None,
    chapter_summary: str | None,
) -> str:
    parts: list[str] = []
    for key in ("chapter_premise", "series_external_context"):
        value = str(story_bible.get(key) or "").strip()
        if value:
            parts.append(value)
    for item in story_bible.get("continuity_notes", []) or []:
        if str(item).strip():
            parts.append(str(item).strip())
    for item in story_bible.get("scene_memory", []) or []:
        if isinstance(item, dict):
            parts.extend(str(item.get(key) or "").strip() for key in ("state", "location", "open_thread") if str(item.get(key) or "").strip())
    if chapter_summary:
        parts.append(str(chapter_summary))
    parts.extend(_scene_summary_strings(scene_summaries))
    return " ".join(parts)


def _scene_summary_strings(payload: list[dict[str, Any]] | dict[str, Any] | list[str] | None) -> list[str]:
    if not payload:
        return []
    if isinstance(payload, dict):
        values: list[str] = []
        for key in ("chapter_summary", "summary", "description"):
            if str(payload.get(key) or "").strip():
                values.append(str(payload.get(key)).strip())
        for key in ("scenes", "scene_seeds"):
            values.extend(_scene_summary_strings(payload.get(key) or []))
        return values
    values = []
    for item in payload:
        if isinstance(item, str):
            if item.strip():
                values.append(item.strip())
            continue
        if not isinstance(item, dict):
            continue
        for key in (
            "summary",
            "description",
            "combined_text",
            "scene_summary",
            "vision_action_beat",
            "vision_caption",
            "vision_dialogue",
        ):
            value = str(item.get(key) or "").strip()
            if value:
                values.append(value)
        chars = item.get("characters")
        if isinstance(chars, list):
            values.extend(str(name).strip() for name in chars if str(name).strip())
    return values


def _as_mapping(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if hasattr(raw, "model_dump"):
        try:
            dumped = raw.model_dump()
            return dumped if isinstance(dumped, dict) else {}
        except Exception:
            return {}
    return {}


def _usable_term(value: str, named_characters: list[str]) -> bool:
    text = _clean_phrase(value)
    if len(text) < 3:
        return False
    key = normalize_name_key(text)
    if not key or key in _STOP_TERMS or key in _WEAK_STYLE_PHRASES:
        return False
    if re.search(
        r"\brelationship\s+with\b|\b(?:male|female|former prodigy)\s+pilot\b|\bformer prodigy\b",
        text,
        flags=re.IGNORECASE,
    ):
        return False
    if re.search(r"\b(?:failed|promising)\s+pilot\b|\b(?:his|her|their)\s+to\s+pilot\b", text, flags=re.IGNORECASE):
        return False
    if re.search(r"\b(?:his|her|their)\s+partners?\b", text, flags=re.IGNORECASE):
        return False
    if re.search(
        r"\b(?:existential crisis|capable pilot|inability to pilot|the piloting|first piloting|ability to pilot|(?:elite|exceptionally skilled) .*pilot|partners?|a pilot)\b",
        text,
        flags=re.IGNORECASE,
    ):
        return False
    if re.search(r"\b(?:the\s+)?series\b|\bmock battle\b|\bfirst pilot\b|\bsquad\s+\d+\b", text, flags=re.IGNORECASE):
        return False
    if re.search(r"\b(?:the earth|the development|the children)\b", text, flags=re.IGNORECASE):
        return False
    if re.search(r"\b(?:attack|attacks|developed|prompting|resulting|suggesting)\b", text, flags=re.IGNORECASE) and len(text.split()) >= 3:
        return False
    if text.casefold().endswith(" with"):
        return False
    if any(key == normalize_name_key(name) for name in named_characters):
        return False
    if _PLACEHOLDER_NAME_RE.search(text):
        return False
    return True


def _usable_stakes_phrase(value: str) -> bool:
    phrase = _clean_phrase(value)
    key = normalize_name_key(phrase)
    if not phrase or not key or key in _WEAK_STYLE_PHRASES:
        return False
    if re.search(
        r"\b(?:purpose|next exchange|immediate problem|the exchange|relationship with|mock battle|exceptional pilot)\b",
        phrase,
        flags=re.IGNORECASE,
    ):
        return False
    if re.search(
        r"\b(?:male|female|former prodigy)\s+pilot\b|\bformer prodigy\b|\b(?:his|her|their)\s+partners?\b",
        phrase,
        flags=re.IGNORECASE,
    ):
        return False
    if re.search(r"\b(?:failed|promising)\s+pilot\b|\b(?:his|her|their)\s+to\s+pilot\b", phrase, flags=re.IGNORECASE):
        return False
    if re.search(
        r"\b(?:existential crisis|capable pilot|inability to pilot|initial piloting|the piloting|first piloting|ability to pilot|(?:elite|exceptional|exceptionally skilled|formidable) .*pilot|partners?|a pilot)\b",
        phrase,
        flags=re.IGNORECASE,
    ):
        return False
    words = re.findall(r"[A-Za-z0-9]+", phrase)
    return len(words) >= 2


def _clean_phrase(value: str) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip(" ,;:-"))


def _clean_team_phrase(value: str) -> str:
    text = _clean_phrase(value)
    if not text:
        return ""
    text = re.sub(r"^(?:his|her|their|our|a|an)\s+", "the ", text, flags=re.IGNORECASE)
    text = re.sub(r"^(?:nearby|remaining|local|frightened|surviving)\s+", "the ", text, flags=re.IGNORECASE)
    return _clean_phrase(text)


def _dedupe_phrases(values: list[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        phrase = _clean_phrase(value)
        key = normalize_name_key(phrase)
        if not phrase or not key or key in seen:
            continue
        seen.add(key)
        result.append(phrase)
    return result


def _drift_keys_for_phrase(value: str) -> set[str]:
    keys = {normalize_name_key(value)}
    keys.update(normalize_name_key(candidate) for candidate in extract_proper_name_candidates(value))
    return {key for key in keys if key}
