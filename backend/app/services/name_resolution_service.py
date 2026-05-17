"""
name_resolution_service.py

Post-vision pass that scans the full script and rewrites generic
descriptors ("the boy with dark hair", "a uniformed pilot", "the
pink-haired girl") into cast-bible names ("Hiro", "Zero Two") whenever
the description matches a known character.

Why this exists:
  The per-panel vision narrator can't always identify a character from
  a partial angle (side profile, back, distant shot). Even with a
  cast bible, it sometimes falls back to "the boy with dark hair"
  when the SAME character has already been named confidently five
  panels earlier.

  This service runs after every vision narration pass. It uses the
  cast bible + a single Gemini call to rewrite the whole script,
  swapping every descriptor that visibly matches a cast member back
  to their proper name. User rule: "if their name has never been said
  then and only then can their description be used."

When called from the pipeline (right after opening_polish), this is
the last narration-text edit before TTS runs - so all wavs come out
with consistent character names.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

from app.core.config import get_settings

logger = logging.getLogger(__name__)

try:
    import google.generativeai as genai  # type: ignore
    _GEMINI_AVAILABLE = True
except ImportError:
    _GEMINI_AVAILABLE = False


_NAME_RESOLUTION_PROMPT = """You are editing a manga-recap narration script
to enforce a strict character-naming rule the YouTuber demands:

  RULE: Every character should be referred to by their cast-bible NAME.
  Generic descriptors ("the boy with dark hair", "a uniformed pilot",
  "the pink-haired girl", "a stern officer", "a figure", "a voice")
  should be replaced with cast names WHENEVER the surrounding script
  context lets you confidently infer who is being referred to.

You will be given:
  1. The CAST BIBLE for the series, with each member's visible
     features (hair color, eye color, signature outfit, etc.).
  2. The SCRIPT as a numbered list of narration lines (one per panel).

How to assign names to descriptors:

  STRATEGY A - feature match against the bible:
    Bible: "Hiro: messy short dark blue hair, blue eyes, Squad 13
    uniform". Line: "The boy with dark blue hair stares at the device."
    → rewrite to "Hiro stares at the device."

  STRATEGY B - context inference (the most powerful one):
    A bare "a figure", "a voice", "the boy", "the girl", or "one
    character" almost always refers to the character most recently
    named in the SAME or PRECEDING lines. Read 3-5 lines of context
    before and after, ask "who has been the active subject of this
    scene?", and use that name.
    Example: line N-1: "Hiro punches the console in frustration."
             line N:   "A figure storms off the bridge."
    → rewrite line N as "Hiro storms off the bridge."

  STRATEGY C - speaker inference from dialogue context:
    A bare "A voice declares, 'X'" or "Someone shouts 'Y'" can often
    be tied to the named character whose POV the scene is in, or to
    the character whose name appears in the next 1-2 lines.

When to LEAVE the descriptor alone:
  • The descriptor refers to a generic crowd / NPC / background
    character clearly not in the cast bible.
  • Multiple cast members fit equally and no preceding context
    disambiguates.
  • The line already uses a cast name correctly.

ABSOLUTE RULES:
  - Output line K corresponds to input line K. Same number of lines.
  - One sentence per line. No multi-sentence rewrites.
  - No numbering, no preface, no commentary.
  - Never invent a cast member who isn't in the bible.
  - Keep tense, tone, word count similar to the original.

CAST BIBLE:
{cast_text}

SCRIPT TO REWRITE:
{numbered}

OUTPUT:
Return EXACTLY {n} lines, one per input line, in the same order.
Plain text. No numbering, no blank lines, no preface."""


def resolve_character_names(
    project_dir: Path,
    *,
    cast_block: str,
    batch_size: int = 80,
) -> dict[str, Any]:
    """Run a Gemini-driven pass that rewrites the script swapping
    descriptors → cast names where the bible matches.

    Updates script_manifest.json, script.json, script.txt in place.
    Non-fatal: returns a report dict even on partial failure.
    """
    settings = get_settings()
    sm_path = project_dir / "script_manifest.json"
    if not sm_path.exists():
        return {"updated": 0, "reason": "no script_manifest.json"}

    cast_text = (cast_block or "").strip()
    if not cast_text or cast_text.lower().startswith("(no character roster"):
        return {"updated": 0, "reason": "no cast bible available; nothing to resolve"}
    if not _GEMINI_AVAILABLE or not settings.gemini_api_key:
        return {"updated": 0, "reason": "gemini not configured"}

    sm = json.loads(sm_path.read_text(encoding="utf-8"))
    all_lines: list[str] = [
        line if isinstance(line, str) else line.get("text", "")
        for line in (sm.get("script_lines") or [])
    ]
    if not all_lines:
        return {"updated": 0, "reason": "empty script"}

    # If the cast block is just the header, strip it - we have our own framing.
    cleaned_cast = cast_text
    if cleaned_cast.startswith("KNOWN CAST"):
        cleaned_cast = "\n".join(cleaned_cast.splitlines()[1:]).strip()

    # Quick heuristic to skip the LLM call entirely when no line contains
    # a descriptor word - the bible isn't going to help and we don't want
    # to spend tokens regenerating identical text.
    descriptor_signal = re.compile(
        r"\b(?:the|a|an|one|another|someone)\s+(?:[a-z\-]+\s+){0,4}"
        r"(?:boy|girl|man|woman|pilot|officer|figure|character|stranger|"
        r"person|child|teen|student|guard|knight|prisoner|warrior|"
        r"voice|individual|kid|youth|adult|elder|companion|partner|"
        r"friend|enemy|soldier|fighter|warrior|leader|villain|hero)\b",
        re.IGNORECASE,
    )
    candidate_indices = [
        i for i, line in enumerate(all_lines)
        if descriptor_signal.search(line or "")
    ]
    if not candidate_indices:
        return {"updated": 0, "reason": "no descriptor candidates in script"}

    genai.configure(api_key=settings.gemini_api_key)
    primary_model = (settings.gemini_model or "gemini-2.5-flash-lite").strip()
    if primary_model in {"gemini-2.0-flash", "gemini-2.0-flash-exp"}:
        primary_model = "gemini-2.5-flash-lite"
    # Model cascade. ORDER matters: empirically gemini-2.5-flash-lite is
    # the most willing to actually REWRITE descriptors into cast names
    # for this task. gemini-2.5-flash is more conservative and tends to
    # return the input unchanged even when a descriptor clearly maps to
    # a cast member. Pro is the deepest fallback for the rare batch
    # both lite + flash refuse to rewrite. Each model has its own free-
    # tier quota bucket so per-model 429s don't kill the run; on paid
    # tier the primary essentially never 429s.
    model_cascade: list[str] = []
    seen: set[str] = set()
    for name in (primary_model, "gemini-2.5-flash-lite", "gemini-2.5-flash", "gemini-2.5-pro"):
        if name and name not in seen:
            model_cascade.append(name)
            seen.add(name)

    # Maximally permissive safety settings: this is editing manga recap
    # script text, not generating fresh content. Default Gemini safety
    # blocks the whole batch on a single PROHIBITED_CONTENT trigger
    # (a bath / intimate scene reference), losing 80 lines per blocked
    # batch even when only one line is borderline. Override the four
    # exposed harm categories to BLOCK_NONE.
    safety_settings = []
    try:
        from google.generativeai.types import HarmCategory, HarmBlockThreshold  # type: ignore
        safety_settings = [
            {"category": HarmCategory.HARM_CATEGORY_HATE_SPEECH, "threshold": HarmBlockThreshold.BLOCK_NONE},
            {"category": HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT, "threshold": HarmBlockThreshold.BLOCK_NONE},
            {"category": HarmCategory.HARM_CATEGORY_HARASSMENT, "threshold": HarmBlockThreshold.BLOCK_NONE},
            {"category": HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT, "threshold": HarmBlockThreshold.BLOCK_NONE},
        ]
    except Exception:
        pass
    # Cache the per-model GenerativeModel objects so we don't reconstruct
    # on every batch. _call_resolver will try them in order.
    models_by_name: dict[str, Any] = {
        name: genai.GenerativeModel(name, safety_settings=safety_settings or None)
        for name in model_cascade
    }

    new_lines = list(all_lines)
    total_rewrites = 0
    batches_run = 0
    # Process the entire script in fixed-size batches so the model has
    # rolling context of cast names already-used in preceding lines.
    for start in range(0, len(all_lines), batch_size):
        end = min(start + batch_size, len(all_lines))
        batch = all_lines[start:end]
        if not any(descriptor_signal.search(line or "") for line in batch):
            # Skip a batch entirely if none of its lines contain a
            # potential descriptor - saves a Gemini call.
            continue
        rewritten = _call_resolver_with_fallback(
            models_by_name, cascade=model_cascade,
            cast_text=cleaned_cast, lines=batch,
        )
        for k, (before, after) in enumerate(zip(batch, rewritten)):
            if (after or "").strip() and after.strip() != (before or "").strip():
                new_lines[start + k] = after.strip()
                total_rewrites += 1
        batches_run += 1

    if total_rewrites == 0:
        return {"updated": 0, "reason": "no replacements suggested by model"}

    sm["script_lines"] = new_lines
    # Also mirror into the manifest's story_segments[].text/narration if
    # present, since the vision-mode renderer reads from that field
    # rather than the legacy script_lines.
    sm_segments = sm.get("story_segments") or []
    for seg, new_line in zip(sm_segments, new_lines):
        if isinstance(seg, dict) and (seg.get("keep", True)) and new_line:
            seg["text"] = new_line
            seg["narration"] = new_line
    sm_path.write_text(json.dumps(sm, indent=2, ensure_ascii=False), encoding="utf-8")
    script_path = project_dir / "script.json"
    if script_path.exists():
        try:
            other = json.loads(script_path.read_text(encoding="utf-8"))
            if isinstance(other, dict):
                if "script_lines" in other:
                    other["script_lines"] = new_lines
                # Critical: update story_segments[].text + narration too.
                # The vision-mode video renderer reads each panel's
                # narration from segments here, not from script_lines.
                # Without this mirror, the resolver's rewrites never
                # reach the rendered video.
                other_segments = other.get("story_segments") or []
                for seg, new_line in zip(other_segments, new_lines):
                    if isinstance(seg, dict) and seg.get("keep", True) and new_line:
                        seg["text"] = new_line
                        seg["narration"] = new_line
                script_path.write_text(
                    json.dumps(other, indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not update script.json after name resolution: %s", exc)
    try:
        (project_dir / "script.txt").write_text(
            "\n".join(new_lines), encoding="utf-8"
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not update script.txt after name resolution: %s", exc)

    logger.info(
        "Name resolution updated %d lines across %d batches for %s",
        total_rewrites, batches_run, project_dir.name,
    )
    return {
        "updated": total_rewrites,
        "batches": batches_run,
        "total_lines": len(all_lines),
    }


def _call_resolver_with_fallback(
    models_by_name: dict[str, Any],
    *,
    cascade: list[str],
    cast_text: str,
    lines: list[str],
    _depth: int = 0,
) -> list[str]:
    """Try each model in `cascade` in order until one returns rewrites.

    Falls through to the next model in two cases:
      1. The current model raised an exception (typically a 429 quota,
         MAX_TOKENS overflow, or transient API error). Each Gemini
         model has its own free-tier quota bucket so 429s don't cascade.
      2. The current model returned the input UNCHANGED for every line
         in the batch. Empirically gemini-2.5-flash will sometimes
         refuse to rewrite even clear-cut descriptor cases; the lighter
         flash-lite model is more willing. We detect "fully identity"
         and let the next model try.

    Final fallback: if ALL models in the cascade fail with a token
    overflow (thinking-tokens + output overshooting the cap on a
    large-ish batch), we recursively split the batch in half and try
    again. Bounded by _depth to avoid infinite recursion.
    """
    def _normalised(s: str) -> str:
        return " ".join((s or "").split()).strip().casefold()

    last_exc: Exception | None = None
    last_was_max_tokens = False
    input_signature = [_normalised(L) for L in lines]
    for idx, model_name in enumerate(cascade):
        model = models_by_name.get(model_name)
        if model is None:
            continue
        try:
            rewritten = _call_resolver(model, cast_text=cast_text, lines=lines)
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            err_str = str(exc)
            err_lower = err_str.lower()
            is_quota = ("429" in err_lower or "quota" in err_lower or
                        "exhausted" in err_lower or "resourceexhausted" in err_lower)
            is_max_tokens = (
                "finish_reason" in err_lower or "MAX_TOKENS" in err_str
                or "is 2" in err_str or "valid `Part`" in err_str
            )
            last_was_max_tokens = is_max_tokens
            if is_quota:
                logger.warning(
                    "Name resolver hit quota on %s; falling to next model",
                    model_name,
                )
            else:
                logger.warning(
                    "Name resolver %s call raised %s; falling to next model",
                    model_name, exc,
                )
            continue

        rewritten_sig = [_normalised(L) for L in rewritten]
        if rewritten_sig == input_signature and idx < len(cascade) - 1:
            logger.info(
                "Name resolver %s returned identity for all %d lines; "
                "trying %s next",
                model_name, len(lines), cascade[idx + 1],
            )
            continue
        return rewritten

    # All models exhausted. If the last error was MAX_TOKENS and the
    # batch is still big enough to halve, split and recurse. Each half
    # gets its own full cascade attempt. Bounded by _depth so we don't
    # recurse forever; at _depth=3 we've gone batch_size -> /2 -> /4
    # -> /8 which is plenty of granularity.
    if last_was_max_tokens and len(lines) > 4 and _depth < 3:
        midpoint = len(lines) // 2
        logger.warning(
            "Name resolver: all models hit MAX_TOKENS on %d-line batch; "
            "splitting in half and retrying",
            len(lines),
        )
        first_half = _call_resolver_with_fallback(
            models_by_name, cascade=cascade, cast_text=cast_text,
            lines=lines[:midpoint], _depth=_depth + 1,
        )
        second_half = _call_resolver_with_fallback(
            models_by_name, cascade=cascade, cast_text=cast_text,
            lines=lines[midpoint:], _depth=_depth + 1,
        )
        return first_half + second_half

    if last_exc is not None:
        logger.warning(
            "Name resolver exhausted all %d fallback models (last error: %s); "
            "leaving batch unchanged.", len(cascade), last_exc,
        )
    return list(lines)


def _call_resolver(model: Any, *, cast_text: str, lines: list[str]) -> list[str]:
    numbered = "\n".join(f"{i + 1}. {line}" for i, line in enumerate(lines))
    prompt = _NAME_RESOLUTION_PROMPT.format(
        cast_text=cast_text, numbered=numbered, n=len(lines)
    )
    # Per-line token budget: average rewritten line is ~30 tokens but
    # Gemini 2.5 models can spend hundreds of "thinking tokens" before
    # producing visible output that count toward max_output_tokens. We
    # try to disable thinking via ThinkingConfig + the dict form (SDK
    # version differs in what it accepts) and then set a high ceiling
    # as a belt-and-suspenders fallback. Calls that still hit
    # MAX_TOKENS get caught in _call_resolver_with_fallback which
    # split-recurses the batch.
    # Per-line token budget: Gemini 2.5 models spend "thinking tokens"
    # before producing visible output, and those count against
    # max_output_tokens. SDK in use does NOT accept thinking_config in
    # GenerationConfig so we can't disable thinking explicitly; instead
    # we budget generously (200 tokens per line, 32k ceiling) and rely
    # on _call_resolver_with_fallback's auto-split path to recurse on
    # batches that still overflow.
    per_line_budget = 200
    dynamic_max = max(8192, len(lines) * per_line_budget + 2048)
    gen_kwargs: dict[str, Any] = {
        "temperature": 0.3,  # low temperature: this is editing, not creating
        "top_p": 0.9,
        "max_output_tokens": min(dynamic_max, 32768),
    }

    response = None
    try:
        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(**gen_kwargs),
        )
        text = (getattr(response, "text", "") or "").strip()
    except Exception as exc:  # noqa: BLE001
        # Specific handling for the MAX_TOKENS case: Gemini returns a
        # response object with no .text accessor (raises ValueError on
        # access) when finish_reason=MAX_TOKENS. Detect and RE-RAISE so
        # the caller's cascade can fall through to a model with bigger
        # output capacity instead of silently returning identity.
        err_str = str(exc)
        is_max_tokens = (
            "finish_reason" in err_str.lower()
            or "MAX_TOKENS" in err_str
            or "is 2" in err_str
            or "valid `Part`" in err_str
        )
        if is_max_tokens:
            logger.warning(
                "Name-resolution batch hit MAX_TOKENS (%d lines); cascade will retry",
                len(lines),
            )
            raise  # propagate so fallback caller switches models
        logger.warning("Name-resolution LLM call failed (%s); leaving batch unchanged.", exc)
        return list(lines)

    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\s*", "", text)
        text = re.sub(r"```\s*$", "", text)
    raw_lines = [L.strip() for L in text.splitlines() if L.strip()]

    cleaned: list[str] = []
    for L in raw_lines:
        m = re.match(r"^\d+[\.\)]\s*(.+)$", L)
        cleaned.append(m.group(1) if m else L)
        cleaned[-1] = cleaned[-1].strip("\"' `")

    if len(cleaned) < len(lines):
        cleaned.extend(lines[len(cleaned):])
    return cleaned[:len(lines)]
