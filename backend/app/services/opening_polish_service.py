"""
opening_polish_service.py

Service-layer version of backend/scripts/polish_opening_lines.py.

Used by the vision-script generation stage to automatically polish the
first ~20 narration lines of every project. The first 20 lines are the
make-or-break retention window on YouTube: if they read as panel
descriptions ("a uniformed character with a reflective visor holds a
small orb") instead of cinematic recap narration ("In a dying world,
humanity hides inside moving fortresses called Plantations"), the
audience-retention curve drops off a cliff.

This module wraps the same Gemini prompt the original one-off script
uses, persists the rewritten lines back to script_manifest.json +
script.json + script.txt, and (when called BEFORE narration audio has
been generated) does NOT need to invalidate any WAV files.

If `polish_opening_lines.py` is run manually AFTER narration, the script
still does WAV invalidation - this service is just the pipeline-level
hook that runs polish at the right moment (between script_generation
and narration_generation) so nothing needs invalidating.
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


_POLISH_PROMPT = """You are a top manga-recap YouTuber (1.2M subscribers) polishing the
opening voiceover for a video. The current draft narration reads like
panel-description notes ("the student's brow is furrowed, a silent
ellipsis indicating internal processing") - that's how the pipeline
auto-generates lines, but it sounds awful when a TTS voice reads it
aloud as the FIRST thing a viewer hears.

Rewrite these {n} lines so they sound like cinematic recap narration,
in the same order. Each rewritten line corresponds 1:1 to the original
line - it covers the same panel, just sounds better.

SERIES: {series}
CHAPTER: {chapter}

CAST (use these names whenever you can match a character to a panel;
fall back to "a girl with green hair" / "the lead boy" / etc. when
the bible doesn't help):
{cast_text}

ABSOLUTE RULES:
  - One rewritten sentence per line. No multi-sentence rewrites.
  - Present tense. Active voice.
  - Use character names from the cast when the panel description
    matches them. NEVER invent names not in the cast.
  - Strip ALL of: panel-direction phrases ("a close-up of", "the
    camera pans", "a panel shows", "we see"), SFX onomatopoeia
    ("BOOM", "GWOOO", any all-caps repeated-letter word), and
    redundant descriptions of expressions ("a silent ellipsis
    indicating internal processing" -> just drop it).
  - The FIRST line is the opening narration after the cold-open
    hook. It should establish setting + stakes in one strong
    sentence.
  - Keep facts that are in the source line. Don't invent action.
  - 8-30 words per line. No commentary, no quotes, no numbering in
    the output.

ORIGINAL LINES TO REWRITE (numbered):
{numbered}

OUTPUT:
Return EXACTLY {n} rewritten lines, one per line, in the same order.
No numbers, no blank lines, no preface. The plain text of line K goes
on output line K."""


def polish_opening_narration(
    project_dir: Path,
    *,
    cast_block: str = "",
    manga_title: str | None = None,
    chapter_title: str | None = None,
    line_count: int = 20,
) -> dict[str, Any]:
    """Polish the first `line_count` lines of the project's script.

    Updates script_manifest.json, script.json (if present), and script.txt
    in place. Safe to call before narration runs - in that case nothing
    needs invalidating downstream.

    Returns a small report dict with counts and a list of before/after
    pairs for the first 5 lines, useful for logging.
    """
    settings = get_settings()
    if not _GEMINI_AVAILABLE:
        raise RuntimeError("google-generativeai is not installed")
    if not settings.gemini_api_key:
        raise RuntimeError("GEMINI_API_KEY not configured; cannot polish opening lines")

    sm_path = project_dir / "script_manifest.json"
    if not sm_path.exists():
        raise FileNotFoundError(f"script_manifest.json missing at {sm_path}")
    sm = json.loads(sm_path.read_text(encoding="utf-8"))
    all_lines = [
        line if isinstance(line, str) else line.get("text", "")
        for line in (sm.get("script_lines") or [])
    ]
    n = min(line_count, len(all_lines))
    if n == 0:
        return {"polished": 0, "reason": "no lines to polish"}

    # Light cast formatter: cast_block is already a "KNOWN CAST" header
    # block from CastBibleService.format_for_prompt, perfectly usable here.
    cast_text = (cast_block or "(no character roster available; use neutral descriptors)").strip()
    if cast_text.startswith("KNOWN CAST"):
        # Strip the header line, we have our own framing.
        cast_text = "\n".join(cast_text.splitlines()[1:]).strip() or cast_text

    series = (manga_title or sm.get("manga_title") or "this series").strip()
    chapter = (chapter_title or sm.get("chapter_title") or "this chapter").strip()

    polished = _call_gemini(
        settings,
        series=series,
        chapter=chapter,
        cast_text=cast_text,
        lines=all_lines[:n],
    )

    # Replace first n lines, keep tail untouched.
    new_script_lines = polished + all_lines[n:]
    sm["script_lines"] = new_script_lines
    sm_path.write_text(json.dumps(sm, indent=2, ensure_ascii=False), encoding="utf-8")

    script_path = project_dir / "script.json"
    if script_path.exists():
        try:
            other = json.loads(script_path.read_text(encoding="utf-8"))
            if isinstance(other, dict) and "script_lines" in other:
                other["script_lines"] = new_script_lines
                script_path.write_text(
                    json.dumps(other, indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not update script.json: %s", exc)
    try:
        (project_dir / "script.txt").write_text(
            "\n".join(new_script_lines), encoding="utf-8"
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not update script.txt: %s", exc)

    sample = [
        {"index": i + 1, "before": all_lines[i][:140], "after": polished[i][:140]}
        for i in range(min(5, n))
    ]
    logger.info("Polished opening %d lines for %s", n, project_dir.name)
    return {"polished": n, "sample": sample}


def _call_gemini(
    settings: Any,
    *,
    series: str,
    chapter: str,
    cast_text: str,
    lines: list[str],
) -> list[str]:
    genai.configure(api_key=settings.gemini_api_key)
    model_name = (settings.gemini_model or "gemini-2.5-flash").strip()
    if model_name in {"gemini-2.0-flash", "gemini-2.0-flash-exp"}:
        model_name = "gemini-2.5-flash"
    model = genai.GenerativeModel(model_name)

    numbered = "\n".join(f"{i + 1}. {line}" for i, line in enumerate(lines))
    prompt = _POLISH_PROMPT.format(
        n=len(lines),
        series=series,
        chapter=chapter,
        cast_text=cast_text,
        numbered=numbered,
    )

    gen_kwargs: dict[str, Any] = {
        "temperature": 0.55,
        "top_p": 0.9,
        "max_output_tokens": 4096,
    }
    try:
        from google.generativeai.types import ThinkingConfig  # type: ignore
        gen_kwargs["thinking_config"] = ThinkingConfig(thinking_budget=0)
    except Exception:
        pass

    response = model.generate_content(
        prompt,
        generation_config=genai.types.GenerationConfig(**gen_kwargs),
    )
    text = (getattr(response, "text", "") or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\s*", "", text)
        text = re.sub(r"```\s*$", "", text)
    raw_lines = [L.strip() for L in text.splitlines() if L.strip()]

    cleaned: list[str] = []
    for L in raw_lines:
        m = re.match(r"^\d+[\.\)]\s*(.+)$", L)
        cleaned.append(m.group(1) if m else L)
        cleaned[-1] = cleaned[-1].strip("\"' `")

    # Pad with originals if model returned fewer lines than expected, so
    # the caller can always swap the prefix safely.
    if len(cleaned) < len(lines):
        cleaned.extend(lines[len(cleaned):])
    return cleaned[:len(lines)]
