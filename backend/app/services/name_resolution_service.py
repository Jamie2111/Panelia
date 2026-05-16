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

  RULE: Every character must be referred to by their cast-bible NAME.
  Generic descriptors ("the boy with dark hair", "a uniformed pilot",
  "the pink-haired girl", "a stern officer") are only allowed for
  characters who are NOT in the cast bible.

You will be given:
  1. The CAST BIBLE for the series, with each member's visible
     features (hair color, eye color, signature outfit, etc.).
  2. The SCRIPT as a numbered list of narration lines (one per panel).

For each line, decide:
  • If a generic descriptor in that line matches a cast member's
    visible features in the bible, REPLACE the descriptor with that
    cast member's name.
    Example: bible has "Hiro: dark blue hair, blue eyes, Squad 13
    uniform". Line says "The boy with dark blue hair stares at the
    device." → rewrite to "Hiro stares at the device."
  • If multiple cast members could plausibly fit the descriptor, pick
    the one most recently named in the preceding lines (continuity).
  • If no cast member fits the descriptor (a new background character,
    an unnamed NPC), leave the descriptor as-is.
  • If a line ALREADY uses a cast name correctly, leave it untouched.
  • Do not invent new content. Do not add or remove sentences.
  • Do not rephrase beyond the descriptor swap. Keep tense, tone,
    word count similar to the original line.

ABSOLUTE RULES:
  - Output line K corresponds to input line K. Same number of lines.
  - One sentence per line. No multi-sentence rewrites.
  - No numbering, no preface, no commentary.
  - Never invent a cast member who isn't in the bible.

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
        r"\b(the|a|an)\s+(?:[a-z\-]+\s+){0,3}"
        r"(?:boy|girl|man|woman|pilot|officer|figure|character|stranger|"
        r"person|child|teen|student|guard|knight|prisoner|warrior)\b",
        re.IGNORECASE,
    )
    candidate_indices = [
        i for i, line in enumerate(all_lines)
        if descriptor_signal.search(line or "")
    ]
    if not candidate_indices:
        return {"updated": 0, "reason": "no descriptor candidates in script"}

    genai.configure(api_key=settings.gemini_api_key)
    model_name = (settings.gemini_model or "gemini-2.5-flash").strip()
    if model_name in {"gemini-2.0-flash", "gemini-2.0-flash-exp"}:
        model_name = "gemini-2.5-flash"
    model = genai.GenerativeModel(model_name)

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
        rewritten = _call_resolver(
            model, cast_text=cleaned_cast, lines=batch
        )
        for k, (before, after) in enumerate(zip(batch, rewritten)):
            if (after or "").strip() and after.strip() != (before or "").strip():
                new_lines[start + k] = after.strip()
                total_rewrites += 1
        batches_run += 1

    if total_rewrites == 0:
        return {"updated": 0, "reason": "no replacements suggested by model"}

    sm["script_lines"] = new_lines
    sm_path.write_text(json.dumps(sm, indent=2, ensure_ascii=False), encoding="utf-8")
    script_path = project_dir / "script.json"
    if script_path.exists():
        try:
            other = json.loads(script_path.read_text(encoding="utf-8"))
            if isinstance(other, dict) and "script_lines" in other:
                other["script_lines"] = new_lines
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


def _call_resolver(model: Any, *, cast_text: str, lines: list[str]) -> list[str]:
    numbered = "\n".join(f"{i + 1}. {line}" for i, line in enumerate(lines))
    prompt = _NAME_RESOLUTION_PROMPT.format(
        cast_text=cast_text, numbered=numbered, n=len(lines)
    )
    gen_kwargs: dict[str, Any] = {
        "temperature": 0.3,  # low temperature: this is editing, not creating
        "top_p": 0.9,
        "max_output_tokens": 4096,
    }
    try:
        from google.generativeai.types import ThinkingConfig  # type: ignore
        gen_kwargs["thinking_config"] = ThinkingConfig(thinking_budget=0)
    except Exception:
        pass

    try:
        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(**gen_kwargs),
        )
        text = (getattr(response, "text", "") or "").strip()
    except Exception as exc:  # noqa: BLE001
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
