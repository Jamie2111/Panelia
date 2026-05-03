from __future__ import annotations

import re

# This guide is intentionally inspired by the storytelling constraints used by
# MangaNarrate: third-person prose, no panel/camera language, and strong
# continuity across sequential images.

_VIEWER_META_PATTERNS = (
    r"\bin this panel\b",
    r"\bon this page\b",
    r"\bthe panel shows\b",
    r"\bthis page shows\b",
    r"\bthe scene shifts\b",
    r"\bthe camera\b",
    r"\bcamera zooms\b",
    r"\bclose-up\b",
    r"\bwide shot\b",
    r"\bwe see\b",
    r"\bthe viewer sees\b",
    r"\bframing\b",
    r"\bangle\b",
    r"\bpanel by panel\b",
)


def immersive_recap_contract() -> str:
    return (
        "- Tell the story like natural third-person prose, closer to a dramatic short story than a caption list.\n"
        "- Treat adjacent panels as consecutive beats of one unfolding moment unless the text clearly signals a time jump.\n"
        "- Do not mention panels, frames, pages, camera moves, angles, close-ups, or what the viewer sees.\n"
        "- Fold dialogue into narration naturally. Paraphrase when helpful instead of copying OCR literally.\n"
        "- Focus on actions, reactions, motives, consequences, and emotional turns that are supported by the evidence.\n"
        "- Preserve concrete facts such as names, places, dates, money, temperatures, named events, and causal explanations.\n"
        "- If local panel evidence is weak, bridge conservatively from nearby context instead of inventing a new event.\n"
        "- Keep the narration immersive and sequential, but never let atmosphere override what the text actually says.\n"
    )


def short_recap_hint() -> str:
    return (
        "Write like a continuous English recap, not a panel caption. "
        "Prefer concrete story beats, paraphrased dialogue, and chronological flow."
    )


def strip_storytelling_meta(text: str) -> str:
    cleaned = str(text or "").strip()
    if not cleaned:
        return ""
    for pattern in _VIEWER_META_PATTERNS:
        cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" ,.;:-")
    if cleaned and cleaned[0].islower():
        cleaned = cleaned[0].upper() + cleaned[1:]
    return cleaned


def looks_like_storytelling_meta(text: str) -> bool:
    lowered = str(text or "").strip().casefold()
    if not lowered:
        return False
    return any(re.search(pattern, lowered, flags=re.IGNORECASE) for pattern in _VIEWER_META_PATTERNS)
