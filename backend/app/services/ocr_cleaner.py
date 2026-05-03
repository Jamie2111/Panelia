from __future__ import annotations

import re
import unicodedata
from typing import Iterable

_LATIN_CHAR_CLASS = r"A-Za-zÀ-ÖØ-öø-ÿĀ-žƀ-ɏḀ-ỿ"


_SCANLATOR_PATTERNS = (
    r"(?i)\b[a-z0-9][a-z0-9-]{1,40}\.(?:com|net|org|gg|io|me|co|cc|xyz|to|tv)(?:/[^\s]*)?\b",
    r"(?i)\boriginal\s+novel(?:\s+and\s+script)?\b.*$",
    r"(?i)\bquality\s+control\b.*$",
    r"(?i)\bclean(?:ing)?\b.*$",
    r"(?i)\bredraw(?:n|er)?\b.*$",
    r"(?i)\btypesett(?:ing|er)?\b.*$",
    r"(?i)\bsocial\s+media\b.*$",
    r"(?i)\bfollow\s+us\b.*$",
    r"(?i)\bvisit\s+our\s+site\b.*$",
    r"(?i)\btranslated by\b.*$",
    r"(?i)\btypeset by\b.*$",
    r"(?i)\bcorrected by\b.*$",
    r"(?i)\bedited by\b.*$",
    r"(?i)\bsync\b.*$",
    r"(?i)\btl\.\s*note\b.*$",
    r"(?i)\bnote:\s*other translations.*$",
    r"(?i)\benglish translation\b.*$",
    r"(?i)\bim\s+ba\s*\d+\s*e\b.*$",
    r"(?i)^hi[!.\s]*(?:i'?m|im)\b.*\btranslation\b.*$",
    r"(?i)\bi wanted to read\b.*$",
    r"(?i)\bi tried fixing\b.*$",
    r"(?i)\bi can't redraw\b.*$",
    r"(?i)\bi'?m not claiming\b.*$",
    r"(?i)\bstraight dog poo\b.*$",
    r"(?i)\bproofread(?:er)?\b.*$",
    r"(?i)\bscan(?:lation|lator|s)?\b.*$",
    r"(?i)\bdiscord\b.*$",
    r"(?i)\bko[\s-]*fi\b.*$",
    r"(?i)\bkofi\b.*$",
    r"(?i)\blinktree\b.*$",
    r"(?i)\bgoogle spreadsheet\b.*$",
    r"(?i)\bquestions?\s+or\s+comments?\b.*$",
    r"(?i)\bif you want to message me\b.*$",
    r"(?i)\bbutterfly scans\b.*$",
    r"(?i)\bpatreon\b.*$",
    r"(?i)\bpadrim\b.*$",
    r"(?i)\bbe\s+a\s+godfather\b.*$",
    r"(?i)\bjoin\s+our\s+server\b.*$",
    r"(?i)\bsupport\s+our\s+work\b.*$",
    r"(?i)\bdestek[\s-]*ol\b.*$",
    r"(?i)\banimewho\b.*$",
    r"(?i)\bneox\b.*\battention\b.*$",
    r"(?i)\bcredit(?:s)?\b.*$",
    r"(?i)\bdocs\.google\.com\b.*$",
    r"(?i)\bko-fi\.com\b.*$",
    r"(?i)\bwww\.[^\s]+",
)

_SFX_TOKENS = {
    "ah", "aha", "bam", "bang", "boom", "burp", "clack", "click", "crash", "eh", "gah", "gasp", "grr", "growl",
    "haa", "haaa", "haaaa", "hah", "hee", "hm", "hmm", "mm", "mmm", "pant", "pow", "sob", "sniff", "tap", "ugh",
    "uh", "vroom", "wham", "whoosh", "woosh",
}
_LOW_SIGNAL_TOKENS = {"amazing", "awesome", "great", "incredible", "nice", "wow", "whoa"}

_MEANINGFUL_SHORT_TOKENS = {"go", "no", "run", "wait", "stop", "help", "why", "what", "who", "yes"}
_DANGLING_ENDINGS = {"if", "and", "or", "but", "to", "of", "for", "with", "because", "when", "than", "then"}
_COMMON_VERBS = {
    "am", "are", "be", "begins", "buy", "buys", "can", "come", "comes", "did", "do", "does", "eat", "eats",
    "feel", "feels", "find", "finds", "get", "gets", "give", "gives", "go", "goes", "gonna", "gulfing",
    "have", "has", "head", "heads", "help", "helps", "is", "kill", "know", "knows", "make", "makes", "mind",
    "need", "needs", "pay", "pays", "preparing", "promise", "realize", "realizes", "say", "says", "see",
    "sees", "share", "spend", "spending", "stay", "stays", "stock", "stockpile", "take", "takes", "throwing",
    "treat", "try", "trying", "understand", "understood", "want", "wants", "will", "won't", "would",
}


def clean_ocr_text(text: str) -> str:
    cleaned = str(text or "").replace("\u2019", "'").replace("\u2018", "'").replace("\u201c", '"').replace("\u201d", '"')
    cleaned = re.sub(r"[^\x00-\x7F\u00C0-\u024F\u1E00-\u1EFF\u3040-\u30ff\u4e00-\u9fff\uac00-\ud7af]+", " ", cleaned)
    cleaned = re.sub(r"[`~^_=+*#<>]+", " ", cleaned)
    # Preserve legitimate all-caps dialogue and captions; only strip long
    # code-like OCR artifacts that also contain digits.
    cleaned = re.sub(r"\b(?=[A-Z0-9]{8,}\b)(?=[A-Z0-9]*\d)[A-Z0-9]+\b", "", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" -_|/\\")
    cleaned = re.sub(r"(?<=\d)[oO](?=\d)", "0", cleaned)
    cleaned = re.sub(r"(?<=\d)[lI](?=\d)", "1", cleaned)
    cleaned = re.sub(r"\b(\d)[oO]{2,}\b", lambda match: match.group(1) + ("0" * (len(match.group(0)) - 1)), cleaned)
    cleaned = re.sub(
        r"\b(\d+)\s+([oO]{2,})\b",
        lambda match: match.group(1) + ("0" * len(match.group(2))),
        cleaned,
    )
    cleaned = re.sub(r"\b([oO])(?=\d)", "0", cleaned)
    cleaned = re.sub(r"(?<=\d)([oO])\b", "0", cleaned)
    cleaned = re.sub(r"\b5[oO]\b", "50", cleaned)
    cleaned = re.sub(r"([a-z])(\d)", r"\1 \2", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"(\d)([a-z])", r"\1 \2", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s+([,.;:!?])", r"\1", cleaned)
    cleaned = re.sub(rf"([,.;:!?])([{_LATIN_CHAR_CLASS}])", r"\1 \2", cleaned)
    for pattern in _SCANLATOR_PATTERNS:
        cleaned = re.sub(pattern, "", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if not cleaned:
        return ""

    letters = [char for char in cleaned if char.isalpha()]
    if letters:
        uppercase_ratio = sum(char.isupper() for char in letters) / max(len(letters), 1)
        if uppercase_ratio > 0.82 and len(cleaned.split()) > 1:
            cleaned = cleaned.lower()
            cleaned = re.sub(r"(^|[.!?]\s+)([a-z])", lambda match: match.group(1) + match.group(2).upper(), cleaned)
    return cleaned.strip()


def is_usable_ocr_text(text: str) -> bool:
    stripped = str(text or "").strip()
    if len(stripped) < 2:
        return False
    if re.search(r"[\u3040-\u30ff\u4e00-\u9fff\uac00-\ud7af]", stripped):
        return True
    if any(re.search(pattern, stripped) for pattern in _SCANLATOR_PATTERNS):
        return False
    if not re.search(rf"[{_LATIN_CHAR_CLASS}0-9]", stripped):
        return False

    letters = sum(char.isalpha() for char in stripped)
    digits = sum(char.isdigit() for char in stripped)
    spaces = stripped.count(" ")
    punctuation = sum(char in ".,!?;:'\"-" for char in stripped)

    if letters < 2 and digits:
        return False
    if digits and digits >= letters:
        return False
    if re.fullmatch(r"[A-Za-z0-9]{4,}", stripped) and digits > 0:
        return False
    if re.fullmatch(r"[A-Z0-9]{5,}", stripped):
        return False
    if digits > 0 and len(stripped) <= 4:
        return False
    if letters and digits > max(4, letters * 0.6):
        return False
    if spaces == 0 and letters >= 5 and not re.search(r"[aeiouAEIOU]", stripped):
        return False
    if letters and punctuation == 0 and spaces == 0 and len(stripped) > 22:
        return False

    latin_tokens = [_latin_token_key(token) for token in re.findall(rf"[{_LATIN_CHAR_CLASS}']+", stripped)]
    latin_tokens = [token for token in latin_tokens if token]
    if len(latin_tokens) == 1:
        token = latin_tokens[0]
        if token in _COMMON_VERBS and token not in _MEANINGFUL_SHORT_TOKENS:
            return False
        if len(token) <= 4 and token not in _MEANINGFUL_SHORT_TOKENS:
            return False
    if latin_tokens and sum(1 for token in latin_tokens if len(token) <= 2) >= max(2, len(latin_tokens) - 1):
        return False
    if latin_tokens and not any(re.search(r"[aeiouy]", token) for token in latin_tokens if len(token) >= 3):
        return False
    if latin_tokens and not any(len(token) >= 3 or token in _MEANINGFUL_SHORT_TOKENS for token in latin_tokens):
        return False
    if _looks_like_sfx_only(latin_tokens):
        return False
    if _looks_like_dangling_fragment(stripped, latin_tokens):
        return False
    if _looks_like_section_header(stripped, latin_tokens):
        return False

    if any(
        word in stripped.casefold()
        for word in (
            "translator",
            "typesetter",
            "proofreader",
            "scanlation",
            "discord",
            "patreon",
            "credits",
            "original novel",
            "quality control",
            "social media",
            "follow us",
        )
    ):
        return False
    if any(
        phrase in stripped.casefold()
        for phrase in (
            "the beginning after the end",
            "beginning after the end",
        )
    ) and len(re.findall(rf"[{_LATIN_CHAR_CLASS}']+", stripped)) <= 5:
        return False
    if re.search(r"(?i)\b[a-z0-9][a-z0-9-]{1,40}\.(?:com|net|org|gg|io|me|co|cc|xyz|to|tv)\b", stripped):
        return False
    return True


def clean_ocr_lines(lines: Iterable[str]) -> list[str]:
    unique: list[str] = []
    seen: set[str] = set()
    for raw_line in lines:
        cleaned = clean_ocr_text(str(raw_line or ""))
        if not cleaned or not is_usable_ocr_text(cleaned):
            continue
        key = cleaned.casefold()
        if key in seen:
            continue
        seen.add(key)
        unique.append(cleaned)
    return _merge_broken_lines(unique)


def combined_dialogue_entry_lines(entries: Iterable[dict[str, object]]) -> list[str]:
    grouped: list[str] = []
    current_speaker = ""
    current_parts: list[str] = []

    for raw_entry in entries:
        if not isinstance(raw_entry, dict):
            continue
        text = clean_ocr_text(str(raw_entry.get("text") or ""))
        if not text:
            continue
        speaker = str(raw_entry.get("speaker_name") or "").strip()
        if current_parts and speaker and current_speaker and speaker != current_speaker:
            grouped.append(" ".join(current_parts))
            current_parts = [text]
            current_speaker = speaker
            continue
        current_parts.append(text)
        current_speaker = speaker or current_speaker

    if current_parts:
        grouped.append(" ".join(current_parts))

    cleaned = clean_ocr_lines(grouped)
    if cleaned:
        return cleaned

    merged_text = clean_ocr_text(" ".join(str(entry.get("text") or "") for entry in entries if isinstance(entry, dict)))
    return [merged_text] if is_usable_ocr_text(merged_text) else []


def combined_ocr_text(lines: Iterable[str]) -> str:
    cleaned_lines = clean_ocr_lines(lines)
    return " ".join(cleaned_lines).strip()


def keyword_tokens(text: str) -> set[str]:
    return {
        token
        for token in (_latin_token_key(raw) for raw in re.findall(rf"[{_LATIN_CHAR_CLASS}']+", clean_ocr_text(text)))
        if token not in {"the", "and", "that", "with", "from", "into", "this", "they", "have", "will"}
        and len(token) >= 3
    }


def _latin_token_key(token: str) -> str:
    normalized = unicodedata.normalize("NFKD", token.casefold())
    ascii_token = "".join(char for char in normalized if "a" <= char <= "z")
    return ascii_token


def _merge_broken_lines(lines: list[str]) -> list[str]:
    if not lines:
        return []
    merged: list[str] = []
    buffer = ""
    for line in lines:
        if not buffer:
            buffer = line
            continue
        if _should_join(buffer, line):
            buffer = f"{buffer.rstrip(' -')} {line.lstrip()}"
        else:
            merged.append(buffer.strip())
            buffer = line
    if buffer:
        merged.append(buffer.strip())
    return merged


def _should_join(left: str, right: str) -> bool:
    if not left or not right:
        return False
    if left.endswith((".", "!", "?", ":", ";")):
        return False
    if left.endswith("-"):
        return True
    if right and right[0].islower() and not re.match(r"(?i)^(hi|hello|hey)\b", right):
        return True
    if len(left.split()) <= 3 and len(right.split()) <= 4 and right[:1].islower():
        return True
    return False


def _looks_like_sfx_only(tokens: list[str]) -> bool:
    if not tokens or len(tokens) > 3:
        return False
    for token in tokens:
        if token in _SFX_TOKENS:
            continue
        if token in _LOW_SIGNAL_TOKENS:
            continue
        if len(set(token)) <= 2 and len(token) >= 3:
            continue
        return False
    return True


def _looks_like_dangling_fragment(text: str, tokens: list[str]) -> bool:
    if not text or not tokens:
        return False
    if text.casefold().rstrip(".!?").split()[-1] in _DANGLING_ENDINGS:
        return True
    if len(tokens) <= 4 and tokens[-1] in _DANGLING_ENDINGS:
        return True
    return False


def _looks_like_section_header(text: str, tokens: list[str]) -> bool:
    if not text or not tokens:
        return False
    if len(tokens) > 4:
        return False
    if re.search(r"[.!?]", text):
        return False
    if any(token in _COMMON_VERBS for token in tokens):
        return False
    if any(token in _MEANINGFUL_SHORT_TOKENS for token in tokens):
        return False
    if "'s" in text.casefold():
        return True
    if len(tokens) <= 3:
        return True
    return False
