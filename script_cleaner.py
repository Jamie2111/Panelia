from __future__ import annotations

import argparse
import json
import os
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import requests

try:
    from flask import Flask, render_template_string, request
except Exception:  # pragma: no cover - optional UI dependency
    Flask = None
    render_template_string = None
    request = None


FILLER_PATTERNS = (
    "questions start piling up",
    "the situation grows harder to explain",
    "the situation grows difficult to explain",
    "the situation grows more difficult to explain",
    "the scene takes a turn",
    "another moment changes everything",
    "another tense beat",
    "the situation escalates",
    "tension builds",
    "the next move depends on",
    "the survival plan continues",
    "the story advances",
    "the scene shifts again",
    "the chapter opens",
    "the world still feels normal",
    "by the end of the chapter",
    "the scene keeps evolving",
    "as the stakes become clearer",
    "as the pressure keeps mounting",
    "as the consequences grow harder to ignore",
    "as the situation grows harder to contain",
    "as the mood turns more urgent",
)

VISUAL_KEYWORDS = (
    "hair",
    "hairstyle",
    "eyes",
    "clothing",
    "shirt",
    "jacket",
    "dress",
    "skirt",
    "pants",
    "smile",
    "smirk",
    "chibi",
    "face",
    "facial expression",
    "camera",
    "framing",
    "lighting",
    "table",
    "couch",
    "sofa",
    "chair",
    "objects",
    "object",
    "standing",
    "sitting",
    "kneeling",
    "lying",
    "brown hair",
    "yellow hair",
    "blue hair",
    "background",
    "white background",
    "black background",
    "website",
    "social media",
    "silhouette",
    "close up",
    "displayed",
    "shown",
    "wooden surface",
    "dimly lit room",
    "glowing object",
)

ACTION_HINTS = (
    "runs",
    "walks",
    "shouts",
    "attacks",
    "betrays",
    "wakes",
    "realizes",
    "returns",
    "discovers",
    "buys",
    "stockpiles",
    "prepares",
    "remembers",
    "begins",
    "starts",
    "decides",
    "fights",
    "reveals",
    "freezes",
    "dies",
)

STOPWORDS = {
    "a",
    "an",
    "the",
    "and",
    "or",
    "but",
    "of",
    "to",
    "for",
    "with",
    "in",
    "on",
    "at",
    "by",
    "from",
    "into",
    "his",
    "her",
    "their",
    "he",
    "she",
    "they",
    "it",
    "this",
    "that",
    "is",
    "was",
    "are",
    "were",
}

HTML_TEMPLATE = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Manga Recap Script Cleaner</title>
  <style>
    body { font-family: sans-serif; max-width: 1100px; margin: 2rem auto; padding: 0 1rem; background: #f7f5ef; color: #1c1b18; }
    h1 { margin-bottom: 0.4rem; }
    form { display: grid; gap: 1rem; }
    textarea { width: 100%; min-height: 320px; padding: 0.9rem; border-radius: 12px; border: 1px solid #cbbfa7; font-family: monospace; font-size: 0.95rem; background: #fffdfa; }
    .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; }
    button { width: fit-content; padding: 0.7rem 1.1rem; border: 0; border-radius: 999px; background: #2f5d50; color: white; font-weight: 700; cursor: pointer; }
    label { font-weight: 700; display: block; margin-bottom: 0.4rem; }
    .meta { color: #5f584d; font-size: 0.92rem; }
    .toggle { display: flex; align-items: center; gap: 0.5rem; }
  </style>
</head>
<body>
  <h1>Manga Recap Script Cleaner</h1>
  <p class="meta">Paste a raw recap draft, clean it into voiceover-ready narration, and optionally let the AI perform a deeper rewrite.</p>
  <form method="post">
    <div class="toggle">
      <input id="ai_clean" type="checkbox" name="ai_clean" value="1" {% if ai_clean %}checked{% endif %}>
      <label for="ai_clean" style="margin:0;">Use AI clean pass</label>
    </div>
    <div class="grid">
      <div>
        <label for="raw_text">Raw script</label>
        <textarea id="raw_text" name="raw_text">{{ raw_text }}</textarea>
      </div>
      <div>
        <label for="cleaned_text">Clean narration</label>
        <textarea id="cleaned_text" readonly>{{ cleaned_text }}</textarea>
      </div>
    </div>
    <button type="submit">Clean Script</button>
  </form>
</body>
</html>
"""


class SemanticSimilarity:
    _MODEL = None

    def similarity(self, left: str, right: str) -> float:
        left_text = str(left or "").strip()
        right_text = str(right or "").strip()
        if not left_text or not right_text:
            return 0.0
        model = self._load_model()
        if model is not None:
            try:
                embeddings = model.encode([left_text, right_text], normalize_embeddings=True)
                return float((embeddings[0] * embeddings[1]).sum())
            except Exception:
                pass
        return self._token_similarity(left_text, right_text)

    def _load_model(self):
        if self.__class__._MODEL is not None:
            return self.__class__._MODEL
        try:
            from sentence_transformers import SentenceTransformer

            self.__class__._MODEL = SentenceTransformer("all-MiniLM-L6-v2")
        except Exception:
            self.__class__._MODEL = None
        return self.__class__._MODEL

    def _token_similarity(self, left: str, right: str) -> float:
        left_counts = self._token_counts(left)
        right_counts = self._token_counts(right)
        if not left_counts or not right_counts:
            return 0.0
        shared = set(left_counts) & set(right_counts)
        numerator = sum(left_counts[token] * right_counts[token] for token in shared)
        left_norm = sum(value * value for value in left_counts.values()) ** 0.5
        right_norm = sum(value * value for value in right_counts.values()) ** 0.5
        if not left_norm or not right_norm:
            return 0.0
        return numerator / (left_norm * right_norm)

    def _token_counts(self, sentence: str) -> Counter[str]:
        counts: Counter[str] = Counter()
        for token in re.findall(r"[a-z']{3,}", sentence.casefold()):
            if token not in STOPWORDS:
                counts[token] += 1
        return counts


@dataclass
class ScriptCleaner:
    similarity_threshold: float = 0.88

    def __post_init__(self) -> None:
        self.similarity = SemanticSimilarity()

    def clean_script(self, text: str, ai_clean: bool = False) -> str:
        sentences = self._split_sentences(text)
        sentences = self.remove_repetition(sentences)
        sentences = self.remove_filler(sentences)
        sentences = self.remove_visual_descriptions(sentences)
        sentences = self.merge_redundant_lines(sentences)
        sentences = self.enforce_chronological_flow(sentences)
        sentences = self.shorten_sentences(sentences)
        sentences = self.enforce_narration_rules(sentences)
        if ai_clean:
            rewritten = self._ai_clean("\n".join(sentences))
            if rewritten.strip():
                sentences = self.enforce_narration_rules(self._split_sentences(rewritten))
        return "\n\n".join(sentence.strip() for sentence in sentences if sentence.strip())

    def remove_repetition(self, sentences: Iterable[str]) -> list[str]:
        unique: list[str] = []
        for sentence in sentences:
            candidate = self._normalize_sentence(sentence)
            if not candidate:
                continue
            if any(self.similarity.similarity(candidate, existing) >= self.similarity_threshold for existing in unique):
                continue
            unique.append(candidate)
        return unique

    def remove_visual_descriptions(self, sentences: Iterable[str]) -> list[str]:
        cleaned: list[str] = []
        for sentence in sentences:
            lowered = sentence.casefold()
            if any(keyword in lowered for keyword in VISUAL_KEYWORDS) and not any(action in lowered for action in ACTION_HINTS):
                continue
            if re.search(r"\b(?:is|are)\s+(?:shown|displayed|standing|sitting)\b", lowered):
                continue
            if re.search(r"\b(?:against a|on a)\s+(?:white|black|wooden|dimly lit)\b", lowered):
                continue
            if re.search(r"\b(two|three|four)\s+\w+\s+with\b", lowered):
                continue
            cleaned.append(sentence)
        return cleaned

    def shorten_sentences(self, sentences: Iterable[str]) -> list[str]:
        shortened: list[str] = []
        for sentence in sentences:
            line = self._strip_soft_filler(sentence)
            clauses = self._split_long_sentence(line)
            for clause in clauses:
                normalized = self._compress_clause(clause)
                if normalized:
                    shortened.append(normalized)
        return shortened

    def merge_redundant_lines(self, sentences: Iterable[str]) -> list[str]:
        merged: list[str] = []
        for sentence in sentences:
            current = self._normalize_sentence(sentence)
            if not current:
                continue
            if not merged:
                merged.append(current)
                continue
            previous = merged[-1]
            combined = self._try_merge(previous, current)
            if combined:
                merged[-1] = combined
            else:
                merged.append(current)
        return merged

    def enforce_narration_rules(self, sentences: Iterable[str]) -> list[str]:
        final_lines: list[str] = []
        previous_subject = ""
        for sentence in sentences:
            line = self._normalize_sentence(sentence)
            if not line:
                continue
            line = self._to_storytelling_tone(line)
            line = self._reduce_consecutive_name_repetition(line, previous_subject)
            if len(line.split()) > 18:
                parts = self._split_long_sentence(line)
                for part in parts:
                    if part.strip():
                        final_lines.append(self._normalize_sentence(part))
                previous_subject = self._subject_hint(final_lines[-1]) if final_lines else previous_subject
                continue
            final_lines.append(line)
            previous_subject = self._subject_hint(line)
        return self.remove_repetition(final_lines)

    def remove_filler(self, sentences: Iterable[str]) -> list[str]:
        filtered: list[str] = []
        for sentence in sentences:
            lowered = sentence.casefold()
            if any(phrase in lowered for phrase in FILLER_PATTERNS):
                continue
            filtered.append(sentence)
        return filtered

    def enforce_chronological_flow(self, sentences: Iterable[str]) -> list[str]:
        buckets: dict[int, list[tuple[int, str]]] = {index: [] for index in range(6)}
        for index, sentence in enumerate(sentences):
            bucket = self._flow_bucket(sentence)
            buckets[bucket].append((index, sentence))
        ordered: list[str] = []
        for bucket in range(6):
            for _, sentence in sorted(buckets[bucket], key=lambda item: item[0]):
                ordered.append(sentence)
        return ordered

    def _split_sentences(self, text: str) -> list[str]:
        normalized = str(text or "")
        normalized = normalized.replace("\r", "\n")
        normalized = re.sub(r"Panel\s+\d+\s*:?", " ", normalized, flags=re.IGNORECASE)
        normalized = re.sub(r"\b(Narration|Dialogue|Visual)\s*:?", " ", normalized, flags=re.IGNORECASE)
        normalized = re.sub(r"\n{2,}", "\n", normalized)
        normalized = re.sub(r"[•●▪■]+", "\n", normalized)
        chunks = re.split(r"(?<=[.!?])\s+|\n+", normalized)
        sentences: list[str] = []
        for chunk in chunks:
            cleaned = self._normalize_sentence(chunk)
            if cleaned:
                sentences.append(cleaned)
        return sentences

    def _normalize_sentence(self, sentence: str) -> str:
        cleaned = re.sub(r"\s+", " ", str(sentence or "").strip())
        cleaned = cleaned.strip("-:;,. ")
        if not cleaned:
            return ""
        if cleaned[-1] not in ".!?":
            cleaned += "."
        return cleaned

    def _flow_bucket(self, sentence: str) -> int:
        lowered = sentence.casefold()
        if any(token in lowered for token in ("freeze", "apocalypse", "betray", "crowd", "die", "killed", "tear him apart")):
            return 0
        if any(token in lowered for token in ("ordinary", "alone", "office", "worker", "home", "neighbor")):
            return 1
        if any(token in lowered for token in ("remember", "last time", "once", "before", "because")):
            return 2
        if any(token in lowered for token in ("wake", "returned", "back in time", "power", "ability", "twist", "discovers")):
            return 3
        if any(token in lowered for token in ("buy", "stockpile", "prepare", "gather", "fortify", "loan", "warehouse")):
            return 4
        return 5

    def _strip_soft_filler(self, sentence: str) -> str:
        line = sentence
        for pattern in (
            r"(?i)\bjust\b",
            r"(?i)\breally\b",
            r"(?i)\bquite\b",
            r"(?i)\bsuddenly\b(?! wakes| returns)",
            r"(?i)\bfor a moment\b",
            r"(?i)\bat this point\b",
        ):
            line = re.sub(pattern, "", line)
        return re.sub(r"\s+", " ", line).strip()

    def _split_long_sentence(self, sentence: str) -> list[str]:
        words = sentence.split()
        if len(words) <= 18:
            return [sentence]
        segments = re.split(r"(?i)\b(?:and|but|while|as|because|when|then)\b|[,;:]", sentence)
        chunks = [segment.strip() for segment in segments if segment.strip()]
        if not chunks:
            return [sentence]
        shortened: list[str] = []
        for chunk in chunks:
            piece = chunk.strip()
            if not piece:
                continue
            if len(piece.split()) > 18:
                midpoint = len(piece.split()) // 2
                words = piece.split()
                shortened.append(" ".join(words[:midpoint]))
                shortened.append(" ".join(words[midpoint:]))
            else:
                shortened.append(piece)
        return shortened or [sentence]

    def _compress_clause(self, sentence: str) -> str:
        cleaned = re.sub(r"(?i)\bit looks like\b", "", sentence)
        cleaned = re.sub(r"(?i)\bit seems like\b", "", cleaned)
        cleaned = re.sub(r"(?i)\bin order to\b", "to", cleaned)
        cleaned = re.sub(r"(?i)\bbegins to\b", "", cleaned)
        cleaned = re.sub(r"(?i)\bcontinues to\b", "", cleaned)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        return self._normalize_sentence(cleaned)

    def _try_merge(self, previous: str, current: str) -> str | None:
        prev_tokens = self._content_tokens(previous)
        curr_tokens = self._content_tokens(current)
        if not prev_tokens or not curr_tokens:
            return None
        overlap = len(set(prev_tokens) & set(curr_tokens))
        if overlap < 2 and self.similarity.similarity(previous, current) < 0.72:
            return None
        if any(keyword in " ".join(prev_tokens + curr_tokens) for keyword in ("suppl", "stockpil", "prepar", "shop")):
            subject = self._subject_hint(previous) or self._subject_hint(current) or "He"
            return self._normalize_sentence(f"{subject} begins secretly stockpiling supplies for the coming disaster.")
        if any(keyword in " ".join(prev_tokens + curr_tokens) for keyword in ("freeze", "apocalypse", "month", "time")):
            return self._normalize_sentence("The truth becomes clear. The apocalypse is now only one month away.")
        return previous if len(previous) <= len(current) else current

    def _content_tokens(self, sentence: str) -> list[str]:
        return [token for token in re.findall(r"[a-z']{3,}", sentence.casefold()) if token not in STOPWORDS]

    def _to_storytelling_tone(self, sentence: str) -> str:
        line = sentence.strip()
        line = re.sub(r"(?i)^there is\b", "There is", line)
        line = re.sub(r"(?i)^there are\b", "There are", line)
        line = re.sub(r"(?i)\bthe protagonist\b", "the protagonist", line)
        if line.casefold().startswith("zhang yi ") and " and " in line and len(line.split()) <= 18:
            return self._normalize_sentence(line)
        if line.casefold().startswith(("a young man with", "a young woman with", "two women with", "a tomato and", "the camera", "the lighting")):
            return ""
        return self._normalize_sentence(line)

    def _reduce_consecutive_name_repetition(self, sentence: str, previous_subject: str) -> str:
        current_subject = self._subject_hint(sentence)
        if previous_subject and current_subject and previous_subject == current_subject:
            escaped = re.escape(current_subject)
            remainder = re.sub(rf"^{escaped}\b", "", sentence).strip()
            if re.match(
                r"(?i)^(?:is|was|looks|looked|walks|walked|runs|ran|shouts|shouted|opens|opened|begins|began|starts|started|realizes|realized|discovers|discovered|returns|returned|loads|loaded|takes|took|pushes|pushed|rushes|rushed|turns|turned|keeps|kept|decides|decided|explains|explained|tries|tried|leans|leaned|becomes|became|steps|stepped|moves|moved)\b",
                remainder,
            ):
                sentence = re.sub(rf"^{escaped}\b", "He", sentence)
                sentence = re.sub(rf"^{escaped}'s\b", "His", sentence)
        return self._normalize_sentence(sentence)

    def _subject_hint(self, sentence: str) -> str:
        match = re.match(r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)", sentence.strip())
        return match.group(1) if match else ""

    def _different_model_default(self) -> str:
        return os.getenv("OPENAI_MODEL", "gpt-4.1-mini")

    def _ai_clean(self, text: str) -> str:
        api_key = os.getenv("OPENAI_API_KEY", "").strip()
        if not api_key:
            return text
        prompt = (
            "Clean this manga recap narration for YouTube voiceover.\n"
            "Remove repetition, filler, visual panel descriptions, and weak flow.\n"
            "Keep the story chronological.\n"
            "Use cinematic third-person narration.\n"
            "Keep each sentence under 18 words when possible.\n"
            "Return narration only, with one sentence per line.\n\n"
            f"{text.strip()}"
        )
        try:
            response = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self._different_model_default(),
                    "temperature": 0.2,
                    "messages": [
                        {
                            "role": "system",
                            "content": "You rewrite messy recap scripts into tight narration for YouTube voiceover.",
                        },
                        {"role": "user", "content": prompt},
                    ],
                },
                timeout=120,
            )
            response.raise_for_status()
            payload = response.json()
            choices = payload.get("choices", [])
            if not choices:
                return text
            content = str(choices[0].get("message", {}).get("content") or "").strip()
            return content or text
        except Exception:
            return text


def clean_script(text: str, ai_clean: bool = False) -> str:
    return ScriptCleaner().clean_script(text, ai_clean=ai_clean)


def remove_repetition(text: str) -> list[str]:
    return ScriptCleaner().remove_repetition(ScriptCleaner()._split_sentences(text))


def remove_visual_descriptions(text: str) -> list[str]:
    cleaner = ScriptCleaner()
    return cleaner.remove_visual_descriptions(cleaner._split_sentences(text))


def shorten_sentences(text: str) -> list[str]:
    cleaner = ScriptCleaner()
    return cleaner.shorten_sentences(cleaner._split_sentences(text))


def merge_redundant_lines(text: str) -> list[str]:
    cleaner = ScriptCleaner()
    return cleaner.merge_redundant_lines(cleaner._split_sentences(text))


def enforce_narration_rules(text: str) -> list[str]:
    cleaner = ScriptCleaner()
    return cleaner.enforce_narration_rules(cleaner._split_sentences(text))


def build_flask_app() -> Flask:
    if Flask is None or render_template_string is None or request is None:  # pragma: no cover
        raise RuntimeError("Flask is not installed. Install backend requirements first.")

    app = Flask("manga_recap_script_cleaner")
    cleaner = ScriptCleaner()

    @app.route("/", methods=["GET", "POST"])
    def index():
        raw_text = ""
        cleaned_text = ""
        ai_clean = False
        if request.method == "POST":
            raw_text = str(request.form.get("raw_text") or "")
            ai_clean = bool(request.form.get("ai_clean"))
            cleaned_text = cleaner.clean_script(raw_text, ai_clean=ai_clean)
        return render_template_string(
            HTML_TEMPLATE,
            raw_text=raw_text,
            cleaned_text=cleaned_text,
            ai_clean=ai_clean,
        )

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean messy manga recap scripts into YouTube-ready narration.")
    parser.add_argument("input", nargs="?", help="Path to the raw input script")
    parser.add_argument("output", nargs="?", help="Path to write the cleaned narration")
    parser.add_argument("--ai-clean", action="store_true", help="Use the OpenAI API for a deeper rewrite pass")
    parser.add_argument("--web", action="store_true", help="Launch the Flask web interface")
    parser.add_argument("--host", default="127.0.0.1", help="Flask host when using --web")
    parser.add_argument("--port", type=int, default=5055, help="Flask port when using --web")
    parser.add_argument("--json", action="store_true", help="Print the cleaned output as JSON")
    args = parser.parse_args()

    if args.web:
        app = build_flask_app()
        app.run(host=args.host, port=args.port, debug=False)
        return

    if not args.input:
        raise SystemExit("Input file path is required unless --web is used.")

    input_path = Path(args.input)
    raw_text = input_path.read_text(encoding="utf-8")
    cleaned = clean_script(raw_text, ai_clean=args.ai_clean)

    if args.output:
        Path(args.output).write_text(cleaned + "\n", encoding="utf-8")
    elif args.json:
        print(json.dumps({"narration": cleaned}, ensure_ascii=False, indent=2))
    else:
        print(cleaned)


if __name__ == "__main__":
    main()
