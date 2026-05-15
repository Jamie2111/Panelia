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
_MEANINGFUL_SINGLE_WORD_DIALOGUE = {
    "beautiful",
    "fine",
    "goodbye",
    "hello",
    "help",
    "hey",
    "hurry",
    "no",
    "okay",
    "ok",
    "please",
    "run",
    "sorry",
    "stop",
    "thanks",
    "wait",
    "what",
    "why",
    "yes",
}
_DANGLING_ENDINGS = {"if", "and", "or", "but", "to", "of", "for", "with", "because", "when", "than", "then"}
_COMMON_VERBS = {
    "am", "are", "be", "begins", "buy", "buys", "can", "come", "comes", "did", "do", "does", "eat", "eats",
    "feel", "feels", "find", "finds", "get", "gets", "give", "gives", "go", "goes", "gonna", "gulfing",
    "have", "has", "head", "heads", "help", "helps", "is", "kill", "know", "knows", "make", "makes", "mind",
    "need", "needs", "pay", "pays", "preparing", "promise", "realize", "realizes", "say", "says", "see",
    "sees", "share", "spend", "spending", "stay", "stays", "stock", "stockpile", "take", "takes", "throwing",
    "treat", "try", "trying", "understand", "understood", "want", "wants", "will", "won't", "would",
}

_KNOWN_OCR_GARBAGE_TOKENS = {
    "alis",
    "cni",
    "eoneannn",
    "fnu",
    "fop",
    "jaiv",
    "jle",
    "kcdikaini",
    "kjur",
    "lnits",
    "lnne",
    "nc",
    "snntnn",
    "sviovcnt",
    "teene",
    "trle",
    "uen",
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


def classify_ocr_text(text: str, expected_language: str = "en") -> str:
    cleaned = clean_ocr_text(text)
    if not cleaned:
        return "ocr_garbage"
    expected = str(expected_language or "en").casefold().split("-")[0]
    if _looks_like_foreign_noise_for_expected_language(cleaned, expected):
        return "foreign_text"
    sfx_tokens = [_latin_token_key(token) for token in re.findall(rf"[{_LATIN_CHAR_CLASS}']+", cleaned)]
    sfx_tokens = [token for token in sfx_tokens if token]
    if _looks_like_sfx_only(sfx_tokens):
        return "sfx"
    if _looks_like_broken_letter_sequence(cleaned):
        return "ocr_garbage"
    if _looks_like_ocr_garbage(cleaned):
        return "ocr_garbage"
    if _looks_like_isolated_name_fragment(cleaned):
        return "low_confidence"
    lowered = cleaned.casefold()
    if any(re.search(pattern, cleaned) for pattern in _SCANLATOR_PATTERNS):
        if re.search(r"(?i)\b(?:translated|typeset|proofread|scanlation|credit|quality control|cleaning|redraw)\b", cleaned):
            return "credit"
        return "source_label"
    tokens = sfx_tokens
    tokens = [token for token in tokens if token]
    if not is_usable_ocr_text(cleaned):
        return "ocr_garbage"
    if re.search(r"(?i)\b(?:tap to|episode|chapter|menu|setting|login|subscribe|follow|read more|next)\b", cleaned):
        return "ui"
    if re.search(r"(?i)\b(?:©|copyright|watermark|all rights reserved)\b", cleaned):
        return "watermark"
    if re.search(r'["“”]', cleaned) or cleaned.endswith(("!", "?", ".")) or re.search(r"\b(?:i|you|we|he|she|they|my|your|our)\b", lowered):
        return "dialogue"
    if len(tokens) >= 4:
        return "narration"
    return "dialogue"


def clean_ocr_fragment_payloads(
    entries: Iterable[dict[str, object]],
    *,
    expected_language: str = "en",
) -> list[dict[str, object]]:
    cleaned: list[dict[str, object]] = []
    seen: set[tuple[str, tuple[int, ...]]] = set()
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        raw_text = str(entry.get("raw_text") or entry.get("text") or entry.get("text_english") or entry.get("text_original") or "")
        script_text = str(
            entry.get("repaired_text")
            or entry.get("text")
            or entry.get("text_english")
            or entry.get("text_original")
            or raw_text
        )
        text = clean_ocr_text(script_text)
        category = classify_ocr_text(text, expected_language=expected_language)
        confidence = entry.get("confidence")
        if (
            isinstance(confidence, (int, float))
            and float(confidence) < 0.68
            and _looks_like_ocr_garbage(text, strict=False)
        ):
            category = "low_confidence"
        if category in {"ocr_garbage", "foreign_text", "low_confidence", "sfx", "ui", "watermark", "credit", "source_label"}:
            usable = False
        else:
            usable = is_usable_ocr_text(text)
        rejection_reason = "" if usable else category
        raw_bbox = entry.get("bbox") or entry.get("bubble_bbox") or []
        bbox = tuple(int(value) for value in raw_bbox[:4]) if isinstance(raw_bbox, (list, tuple)) and len(raw_bbox) >= 4 else ()
        key = (text.casefold(), bbox)
        if not text or key in seen:
            continue
        seen.add(key)
        output = {
            "text": text,
            "raw_text": raw_text,
            "cleaned_text": text,
            "bbox": list(bbox),
            "bubble_group_id": entry.get("bubble_id") or entry.get("bubble_group_id"),
            "bubble_bbox": entry.get("bubble_bbox"),
            "page": entry.get("page"),
            "panel": entry.get("panel"),
            "panel_id": entry.get("panel_id"),
            "panel_order": entry.get("panel_order"),
            "reading_order_index": len(cleaned) + 1,
            "confidence": confidence,
            "detector": entry.get("detector"),
            "ocr_engine": entry.get("ocr_engine"),
            "classification": category,
            "category": category,
            "accepted": bool(usable),
            "usable_for_script": bool(usable),
            "rejection_reason": rejection_reason,
        }
        cleaned.append(output)
    return sorted(
        cleaned,
        key=lambda item: (
            (item.get("bbox") or [0, 0, 0, 0])[1],
            (item.get("bbox") or [0, 0, 0, 0])[0],
        ),
    )


def is_usable_ocr_text(text: str) -> bool:
    stripped = str(text or "").strip()
    if len(stripped) < 2:
        return False
    if _looks_like_ocr_garbage(stripped):
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


def _looks_like_ocr_garbage(text: str, *, strict: bool = True) -> bool:
    cleaned = str(text or "").strip()
    if not cleaned:
        return True
    lowered = cleaned.casefold()
    latin_tokens = [_latin_token_key(token) for token in re.findall(rf"[{_LATIN_CHAR_CLASS}']+", cleaned)]
    latin_tokens = [token for token in latin_tokens if token]
    if not latin_tokens:
        return False
    if any(token in _KNOWN_OCR_GARBAGE_TOKENS for token in latin_tokens):
        return True
    if any(
        phrase in lowered
        for phrase in (
            "what a high nance girl",
            "nance girl",
            "kcdikaini lass",
            "me stay who sorry",
            "take a lhne",
            "show some self oc",
        )
    ):
        return True

    letters = sum(char.isalpha() for char in cleaned)
    digits = sum(char.isdigit() for char in cleaned)
    punctuation = sum(char in ".,!?;:|/\\[]{}()" for char in cleaned)
    short_tokens = sum(1 for token in latin_tokens if len(token) <= 2)
    long_tokens = sum(1 for token in latin_tokens if len(token) >= 4)
    no_vowel_tokens = sum(1 for token in latin_tokens if len(token) >= 4 and not re.search(r"[aeiouy]", token))
    malformed_digits = bool(re.search(r"\b(?:[a-z]\s*)?\d(?:\s+\d|[a-z])", lowered))
    repeated_tiny_clauses = cleaned.count(".") >= 4 and short_tokens >= max(3, len(latin_tokens) // 3)

    if digits >= 2 and malformed_digits:
        return True
    if no_vowel_tokens >= 2 and no_vowel_tokens >= long_tokens // 2:
        return True
    if repeated_tiny_clauses:
        return True
    if punctuation > max(6, letters // 5) and short_tokens >= 3:
        return True
    if short_tokens >= max(5, round(len(latin_tokens) * 0.55)) and not any(token in _MEANINGFUL_SHORT_TOKENS for token in latin_tokens):
        return True
    if not strict:
        if no_vowel_tokens >= 1 and short_tokens >= 3:
            return True
        if digits and short_tokens >= 4:
            return True
        if punctuation > max(4, letters // 7) and len(latin_tokens) >= 5:
            return True
    return False


def _looks_like_foreign_noise_for_expected_language(text: str, expected_language: str) -> bool:
    if expected_language not in {"en", "eng", "a", ""}:
        return False
    cleaned = str(text or "").strip()
    if not cleaned:
        return False
    kana_kanji = len(re.findall(r"[\u3040-\u30ff\u3400-\u9fff]", cleaned))
    hangul = len(re.findall(r"[\uac00-\ud7af]", cleaned))
    latin_letters = len(re.findall(rf"[{_LATIN_CHAR_CLASS}]", cleaned))
    punctuation_runs = len(re.findall(r"[.。・…]{3,}", cleaned))
    if kana_kanji + hangul == 0:
        return False
    if latin_letters and kana_kanji + hangul >= 1:
        return True
    if kana_kanji + hangul >= 3 and latin_letters <= 12:
        return True
    if kana_kanji + hangul >= 2 and punctuation_runs:
        return True
    if kana_kanji + hangul >= 1 and _looks_like_broken_letter_sequence(cleaned):
        return True
    return False


def _looks_like_broken_letter_sequence(text: str) -> bool:
    cleaned = str(text or "").strip()
    if not cleaned:
        return True
    if re.search(r"(?<![A-Za-z])[A-Za-z]\s*[.．。・]\s*[A-Za-z]\s*[.．。・]", cleaned):
        return True
    if re.search(r"(?<![A-Za-z])[A-Za-z]\s+[A-Za-z]\s+[A-Za-z](?![A-Za-z])", cleaned):
        return True
    latin_tokens = re.findall(rf"[{_LATIN_CHAR_CLASS}]+", cleaned)
    short_tokens = sum(1 for token in latin_tokens if len(token) <= 3)
    punctuation_runs = len(re.findall(r"[.．。・…]{2,}", cleaned))
    if punctuation_runs and latin_tokens and short_tokens == len(latin_tokens):
        return True
    return False


def _looks_like_isolated_name_fragment(text: str) -> bool:
    cleaned = str(text or "").strip()
    if not cleaned.endswith("."):
        return False
    body = cleaned.rstrip(".．。").strip()
    if not re.fullmatch(rf"[{_LATIN_CHAR_CLASS}][{_LATIN_CHAR_CLASS}'-]{{1,24}}", body):
        return False
    if body.casefold() in _MEANINGFUL_SINGLE_WORD_DIALOGUE:
        return False
    return body[:1].isupper() and not body.isupper()
