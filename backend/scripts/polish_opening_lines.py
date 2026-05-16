"""
polish_opening_lines.py

One-shot script-polish pass focused on the OPENING N narration lines
of a project. The first 10-20 lines are what the viewer hears in the
critical first minute - if they read as panel descriptions ("the
student's brow is furrowed, a silent ellipsis indicating internal
processing") instead of cinematic narration, the retention curve
drops off a cliff. This script rewrites them.

Usage:
    python backend/scripts/polish_opening_lines.py <project_id> [--lines 20]

What it does:
  1. Loads cast bible (for character names) + the first N lines of
     script_manifest.json.
  2. Sends them to Gemini with a prompt that demands cinematic
     present-tense narration, character names when available,
     no panel-direction language ("the camera shows", "a close-up
     of"), and no SFX onomatopoeia.
  3. Writes the polished lines back to script_manifest.json + script.json.
  4. Deletes the corresponding panel_NNN.wav files so the next
     narration run re-synthesizes those lines with the new text.

Idempotent: re-running on already-polished lines just polishes them
again. The original narration is overwritten - the project_store
backs up to history under jobs/ anyway.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
BACKEND_ROOT = REPO_ROOT / "backend"
sys.path.insert(0, str(BACKEND_ROOT))

import os
ENV_PATH = REPO_ROOT / ".env"
if ENV_PATH.exists():
    for line in ENV_PATH.read_text().splitlines():
        if "=" in line and not line.strip().startswith("#"):
            k, _, v = line.partition("=")
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))

from app.core.config import get_settings  # noqa: E402

try:
    import google.generativeai as genai  # type: ignore
    _GEMINI_AVAILABLE = True
except ImportError:
    _GEMINI_AVAILABLE = False


def load_cast_bible(project_dir: Path) -> list[dict]:
    cb_path = project_dir / "output" / "cast_bible.json"
    if not cb_path.exists():
        return []
    try:
        cb = json.loads(cb_path.read_text())
        return cb.get("members") or []
    except Exception:
        return []


def cast_block(members: list[dict]) -> str:
    """Format the cast as a prompt-friendly bullet list."""
    if not members:
        return "(no character roster available - use generic descriptors only)"
    rows = []
    for m in members[:15]:
        name = m.get("canonical_name") or m.get("name") or "?"
        desc = (m.get("appearance_summary") or m.get("description") or m.get("one_liner") or "").strip()
        if desc:
            rows.append(f"  - {name}: {desc[:120]}")
        else:
            rows.append(f"  - {name}")
    return "\n".join(rows)


def build_prompt(
    series: str,
    chapter: str,
    cast_text: str,
    lines: list[str],
) -> str:
    numbered = "\n".join(f"{i+1}. {line}" for i, line in enumerate(lines))
    return f"""You are a top manga-recap YouTuber (1.2M subscribers) polishing the
opening voiceover for a video. The current draft narration reads like
panel-description notes ("the student's brow is furrowed, a silent
ellipsis indicating internal processing") - that's how the pipeline
auto-generates lines, but it sounds awful when a TTS voice reads it
aloud as the FIRST thing a viewer hears.

Rewrite these {len(lines)} lines so they sound like cinematic recap
narration, in the same order. Each rewritten line corresponds 1:1 to
the original line - it covers the same panel, just sounds better.

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
Return EXACTLY {len(lines)} rewritten lines, one per line, in the
same order. No numbers, no blank lines, no preface. The plain text
of line K goes on output line K."""


def polish_lines(
    series: str,
    chapter: str,
    cast_text: str,
    lines: list[str],
) -> list[str]:
    if not _GEMINI_AVAILABLE:
        raise RuntimeError("google-generativeai not installed")
    settings = get_settings()
    if not settings.gemini_api_key:
        raise RuntimeError("GEMINI_API_KEY not configured")
    genai.configure(api_key=settings.gemini_api_key)
    model_name = (settings.gemini_model or "gemini-2.5-flash").strip()
    if model_name in {"gemini-2.0-flash", "gemini-2.0-flash-exp"}:
        model_name = "gemini-2.5-flash"
    model = genai.GenerativeModel(model_name)

    prompt = build_prompt(series, chapter, cast_text, lines)
    gen_kwargs = {
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
    # Clean common LLM artifacts: code fences, leading numbers, quotes.
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\s*", "", text)
        text = re.sub(r"```\s*$", "", text)
    raw_lines = [L.strip() for L in text.splitlines() if L.strip()]
    # Strip leading "N." or "N)" numbering if Gemini didn't honor "no
    # numbering" rule.
    cleaned = []
    for L in raw_lines:
        m = re.match(r"^\d+[\.\)]\s*(.+)$", L)
        cleaned.append(m.group(1) if m else L)
        cleaned[-1] = cleaned[-1].strip("\"' `")

    if len(cleaned) < len(lines):
        # Model returned fewer lines than expected. Pad with originals.
        cleaned.extend(lines[len(cleaned):])
    return cleaned[:len(lines)]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("project_id")
    parser.add_argument("--lines", type=int, default=20)
    args = parser.parse_args()

    project_dir = BACKEND_ROOT / "data" / "projects" / args.project_id
    if not project_dir.exists():
        print(f"unknown project: {args.project_id}", file=sys.stderr)
        return 1

    sm_path = project_dir / "script_manifest.json"
    sm = json.loads(sm_path.read_text())
    all_lines = [
        L if isinstance(L, str) else L.get("text", "")
        for L in sm.get("script_lines", [])
    ]
    n = min(args.lines, len(all_lines))
    if n == 0:
        print("no lines to polish", file=sys.stderr)
        return 1

    # Cast bible + project metadata
    members = load_cast_bible(project_dir)
    project_meta = json.loads((project_dir / "metadata.json").read_text())
    chapter_meta = project_meta.get("chapter_metadata", {}) or {}
    series = chapter_meta.get("manga_title") or project_meta.get("name", "this series")
    chapter = chapter_meta.get("chapter_title") or "this chapter"

    print(f"  series:    {series}")
    print(f"  chapter:   {chapter}")
    print(f"  cast size: {len(members)}")
    print(f"  polishing first {n} of {len(all_lines)} lines")
    print()

    polished = polish_lines(series, chapter, cast_block(members), all_lines[:n])

    print("BEFORE -> AFTER")
    for i, (before, after) in enumerate(zip(all_lines[:n], polished), start=1):
        print(f"  [{i:>2}] {before[:180]}")
        print(f"       -> {after[:180]}")
    print()

    # Write back to script_manifest.json + script.json.
    new_script_lines = polished + all_lines[n:]
    sm["script_lines"] = new_script_lines
    sm_path.write_text(json.dumps(sm, indent=2, ensure_ascii=False))
    # Also write script.json (project_store reads this for some paths)
    script_path = project_dir / "script.json"
    if script_path.exists():
        try:
            other = json.loads(script_path.read_text())
            if isinstance(other, dict) and "script_lines" in other:
                other["script_lines"] = new_script_lines
                script_path.write_text(json.dumps(other, indent=2, ensure_ascii=False))
        except Exception:
            pass
    # Also flat script.txt mirror so the narration stage's text-read
    # path picks up the change too.
    (project_dir / "script.txt").write_text("\n".join(new_script_lines), encoding="utf-8")

    # Invalidate the audio for the polished panels so the next narration
    # run re-synthesizes them with the new text.
    audio_dir = project_dir / "audio"
    deleted = 0
    for i in range(1, n + 1):
        wav = audio_dir / f"panel_{i:03d}.wav"
        if wav.exists():
            wav.unlink()
            deleted += 1
    print(f"polished {n} lines; deleted {deleted} stale panel WAVs for re-narration")
    return 0


if __name__ == "__main__":
    sys.exit(main())
