from __future__ import annotations

import re


FALSE_CHARACTER_NAME_PHRASES: frozenset[str] = frozenset(
    {
        "por favor",
        "de novo",
        "be dead",
        "jle trle",
        "nati a",
        "start it",
        "a shaft",
        "nc jaiv",
        "kcdikaini lass",
    }
)

FALSE_CHARACTER_NAME_TOKENS: frozenset[str] = frozenset(
    {
        "about",
        "after",
        "again",
        "agora",
        "and",
        "apocalypse",
        "be",
        "because",
        "before",
        "claro",
        "customer",
        "dead",
        "favor",
        "freeze",
        "gracias",
        "hello",
        "hipo",
        "hose",
        "jaiv",
        "jle",
        "kcdikaini",
        "lass",
        "manager",
        "nati",
        "nc",
        "none",
        "obrigada",
        "obrigado",
        "okay",
        "ola",
        "olá",
        "please",
        "roger",
        "salur",
        "sauri",
        "shaft",
        "sim",
        "sorry",
        "start",
        "thanks",
        "trle",
        "wait",
        "world",
    }
)

PLACEHOLDER_NAME_RE = re.compile(
    r"^(?:character|stranger|unidentified character)(?:[_\s-]*\d+)?$",
    re.IGNORECASE,
)


def normalize_name_key(value: object) -> str:
    return " ".join(re.findall(r"[a-záàâãéêíóôõúçñ0-9]+", str(value or "").casefold())).strip()


def looks_like_false_character_name(value: object) -> bool:
    key = normalize_name_key(value)
    if not key:
        return True
    if key in FALSE_CHARACTER_NAME_PHRASES:
        return True
    if PLACEHOLDER_NAME_RE.fullmatch(key):
        return True
    tokens = [token for token in key.split() if token]
    if not tokens:
        return True
    if tokens[0] in {"a", "an", "the"}:
        return True
    if any(token in FALSE_CHARACTER_NAME_TOKENS for token in tokens):
        return True
    if len(tokens) == 1 and len(tokens[0]) < 4:
        return True
    return False
