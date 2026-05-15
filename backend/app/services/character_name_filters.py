from __future__ import annotations

import re


FALSE_CHARACTER_NAME_PHRASES: frozenset[str] = frozenset(
    {
        "break it",
        "the other student",
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
        "break",
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
        "other",
        "please",
        "roger",
        "run",
        "salur",
        "sauri",
        "shaft",
        "sim",
        "sorry",
        "start",
        "stop",
        "thanks",
        "trle",
        "wait",
        "world",
        "idiot",
    }
)

_GENERIC_ROLE_TOKENS: frozenset[str] = frozenset(
    {
        "boy",
        "bystander",
        "child",
        "classmate",
        "crowd",
        "figure",
        "girl",
        "guy",
        "kid",
        "lady",
        "man",
        "narrator",
        "person",
        "speaker",
        "stranger",
        "student",
        "teacher",
        "voice",
        "woman",
    }
)

_SFX_NAME_TOKENS: frozenset[str] = frozenset(
    {
        "ah",
        "bam",
        "bang",
        "boom",
        "clack",
        "click",
        "crash",
        "gasp",
        "grr",
        "ha",
        "haa",
        "hah",
        "hm",
        "hmm",
        "ouch",
        "pant",
        "pow",
        "sob",
        "tap",
        "ugh",
        "whoosh",
        "woosh",
    }
)

_DIALOGUE_FRAGMENT_TOKENS: frozenset[str] = frozenset(
    {
        "attack",
        "can",
        "come",
        "die",
        "do",
        "does",
        "dont",
        "don't",
        "get",
        "give",
        "go",
        "help",
        "hit",
        "hold",
        "is",
        "just",
        "keep",
        "kill",
        "leave",
        "let",
        "move",
        "need",
        "pay",
        "run",
        "say",
        "see",
        "shut",
        "sit",
        "stand",
        "stay",
        "stop",
        "take",
        "tell",
        "use",
        "wait",
        "watching",
        "what",
        "why",
        "you",
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


def is_valid_character_name_candidate(value: object, *, allow_stable_label: bool = False) -> bool:
    raw = str(value or "").strip()
    key = normalize_name_key(raw)
    if not key:
        return False
    tokens = [token for token in key.split() if token]
    if not tokens:
        return False
    if allow_stable_label and tokens[0] in {"the", "a", "an"} and len(tokens) >= 2:
        role_tokens = set(tokens[1:])
        return bool(role_tokens & _GENERIC_ROLE_TOKENS) and len(tokens) <= 5
    if looks_like_false_character_name(raw):
        return False
    if tokens[0] in {"a", "an", "the"}:
        return False
    if any(token in _SFX_NAME_TOKENS for token in tokens):
        return False
    if len(tokens) == 1 and tokens[0] in _DIALOGUE_FRAGMENT_TOKENS:
        return False
    if len(tokens) <= 3 and any(token in _DIALOGUE_FRAGMENT_TOKENS for token in tokens):
        return False
    if all(token in _GENERIC_ROLE_TOKENS for token in tokens):
        return False
    if len(tokens) > 3:
        return False
    letters = [char for char in raw if char.isalpha()]
    punctuation = sum(char in ".,!?;:()[]{}<>/\\|_+=*" for char in raw)
    if letters and punctuation > max(1, len(letters) // 3):
        return False
    if len(raw) > 32:
        return False
    if raw[:1].islower() and len(tokens) <= 3:
        return False
    if re.search(r"[.!?]", raw) and len(tokens) > 1:
        return False
    return True
