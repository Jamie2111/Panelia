from __future__ import annotations

import re
from collections import Counter
from threading import Lock
from typing import Iterable

from app.schemas.project import ChapterMetadata
from app.services.character_name_filters import is_valid_character_name_candidate, looks_like_false_character_name
from app.services.ocr_cleaner import clean_ocr_text


class CharacterNameService:
    _NLP = None
    _LOAD_LOCK = Lock()

    def discover(
        self,
        texts: Iterable[str],
        metadata: ChapterMetadata,
    ) -> tuple[dict[str, str], str | None]:
        candidates: Counter[str] = Counter()
        surface_forms: dict[str, Counter[str]] = {}

        metadata_names = self._metadata_names(metadata)
        metadata_keys = {self._name_key(name) for name in metadata_names if self._name_key(name)}
        for name in metadata_names:
            key = self._name_key(name)
            if not key:
                continue
            candidates[key] += 6
            surface_forms.setdefault(key, Counter())[name] += 6

        for text in texts:
            for name in self._names_from_text(text):
                key = self._name_key(name)
                if not key:
                    continue
                candidates[key] += 1
                surface_forms.setdefault(key, Counter())[name] += 1

        character_dictionary: dict[str, str] = {}
        for key, count in candidates.most_common():
            if key not in metadata_keys and count < 3:
                continue
            best_surface = surface_forms.get(key, Counter()).most_common(1)
            if not best_surface:
                continue
            if (
                looks_like_false_character_name(key)
                or looks_like_false_character_name(best_surface[0][0])
                or not is_valid_character_name_candidate(best_surface[0][0])
            ):
                continue
            character_dictionary[key] = best_surface[0][0]

        protagonist_name = None
        if metadata_names:
            protagonist_name = metadata_names[0]
            character_dictionary.setdefault(self._name_key(protagonist_name), protagonist_name)
        elif character_dictionary:
            protagonist_name = next(iter(character_dictionary.values()))

        return character_dictionary, protagonist_name

    def character_names_in_text(self, text: str, character_dictionary: dict[str, str]) -> list[str]:
        cleaned = clean_ocr_text(text).casefold()
        names: list[str] = []
        for key, canonical in character_dictionary.items():
            if not key:
                continue
            pattern = rf"\b{re.escape(key)}\b"
            if re.search(pattern, cleaned):
                names.append(canonical)
        return list(dict.fromkeys(names))

    def extract_names(self, text: str) -> list[str]:
        return self._names_from_text(text)

    def _names_from_text(self, text: str) -> list[str]:
        cleaned = clean_ocr_text(text).strip()
        if not cleaned:
            return []

        candidates: list[str] = []
        candidates.extend(self._spacy_person_entities(cleaned))
        candidates.extend(self._regex_names(cleaned))

        normalized: list[str] = []
        seen: set[str] = set()
        for candidate in candidates:
            name = self._normalize_name(candidate)
            key = self._name_key(name)
            if not key or key in seen:
                continue
            if not is_valid_character_name_candidate(name):
                continue
            seen.add(key)
            normalized.append(name)
        return normalized

    def _spacy_person_entities(self, text: str) -> list[str]:
        nlp = self._load_nlp()
        if nlp is None:
            return []
        try:
            doc = nlp(text)
        except Exception:
            return []
        return [ent.text.strip() for ent in doc.ents if ent.label_ == "PERSON" and ent.text.strip()]

    def _regex_names(self, text: str) -> list[str]:
        names: list[str] = []
        for pattern in (
            r"(?i)\b([a-z][a-z-]+(?:\s+[a-z][a-z-]+){1,2})(?=[!,:])",
            r"(?i)\bmr\.?\s+([a-z][a-z-]+(?:\s+[a-z][a-z-]+)?)\b",
            r"(?i)\bmy name is\s+([a-z][a-z-]+(?:\s+[a-z][a-z-]+){0,2})\b",
            r"(?i)\bi am\s+([a-z][a-z-]+(?:\s+[a-z][a-z-]+){0,2})\b",
            r"(?i)\bi'?m\s+([a-z][a-z-]+(?:\s+[a-z][a-z-]+){0,2})\b",
            r"(?i)\bthis is\s+([a-z][a-z-]+(?:\s+[a-z][a-z-]+){0,2})\b",
        ):
            for match in re.finditer(pattern, text):
                names.append(match.group(1))
        return names

    def _metadata_names(self, metadata: ChapterMetadata) -> list[str]:
        raw = metadata.raw if isinstance(metadata.raw, dict) else {}
        texts: list[str] = []

        for relation in raw.get("relationships", []) if isinstance(raw.get("relationships"), list) else []:
            if not isinstance(relation, dict) or relation.get("type") != "manga":
                continue
            attributes = relation.get("attributes")
            if not isinstance(attributes, dict):
                continue
            description = attributes.get("description")
            excerpt = self._metadata_description_excerpt(description)
            if excerpt:
                texts.append(excerpt)
        manga_payload = raw.get("manga")
        if isinstance(manga_payload, dict):
            for key in ("synopsis", "description", "summary"):
                excerpt = self._metadata_description_excerpt(manga_payload.get(key))
                if excerpt:
                    texts.append(excerpt)
        for key in ("synopsis", "description", "summary"):
            excerpt = self._metadata_description_excerpt(raw.get(key))
            if excerpt:
                texts.append(excerpt)

        names: list[str] = []
        for text in texts:
            for pattern in (
                r"\b(?:boy|girl|man|woman|student|pilot|hero|heroine|protagonist|character)\s+named\s+[\"“”']?([A-Z][A-Za-z0-9-]{1,18}(?:\s+[A-Z][A-Za-z0-9-]{1,18}){0,2})\b",
                r"\b(?:boy|girl|man|woman|student|pilot|hero|heroine|protagonist|character)\s+known\s+as\s+[\"“”']?([A-Z][A-Za-z0-9-]{1,18}(?:\s+[A-Z][A-Za-z0-9-]{1,18}){0,2})\b",
                r"\bknown\s+as\s+[\"“”']?([A-Z][A-Za-z0-9-]{1,18}(?:\s+[A-Z][A-Za-z0-9-]{1,18}){0,2})[\"“”']?\s+(?:appears|arrives|enters|joins|stands|faces)\b",
                r"\bhero\s+([A-Z][a-z]{1,14}(?:\s+[A-Z][a-z]{1,14}){0,2})\b",
                r"\b([A-Z][a-z]{1,14}(?:\s+[A-Z][a-z]{1,14}){0,2})\s+(?:was|is|has|wakes|awakens|returns|reborn|uses|must|begins)\b",
                r"\b([A-Z][a-z]{1,14}(?:\s+[A-Z][a-z]{1,14}){0,2})\s+(?:tries|finds|learns|discovers|faces|fights)\b",
            ):
                for match in re.finditer(pattern, text):
                    names.append(match.group(1))
        ordered = []
        seen = set()
        for name in names:
            normalized = self._normalize_name(name)
            key = self._name_key(normalized)
            if not key or key in seen or not is_valid_character_name_candidate(normalized):
                continue
            seen.add(key)
            ordered.append(normalized)
        return ordered

    def _normalize_name(self, raw_name: str) -> str:
        tokens = [token for token in re.findall(r"[a-z]+", str(raw_name or "").casefold()) if token]
        if not tokens:
            return ""
        if looks_like_false_character_name(" ".join(tokens)):
            return ""
        filler_leads = {"now", "hey", "wait", "sorry", "okay", "well", "so"}
        while len(tokens) > 1 and tokens[0] in filler_leads:
            tokens = tokens[1:]
        honorifics = {"lady", "lord", "sir", "madam", "mr", "mrs", "ms", "miss"}
        while len(tokens) > 1 and tokens[0] in honorifics:
            tokens = tokens[1:]
        while len(tokens) > 1 and tokens[-1] in honorifics:
            tokens = tokens[:-1]
        if not tokens:
            return ""
        stop_tokens = {
            "about", "after", "again", "am", "and", "apocalypse", "are", "be", "because", "before", "being", "break", "customer",
            "did", "do", "does", "dont", "fi", "for", "freeze", "from", "going", "google", "guys", "have", "help",
            "hello", "here", "hey", "hotel", "im", "i'm", "is", "just", "ko", "kofi", "link",
            "manager", "money", "my", "name", "need", "next", "other", "please", "questions", "really", "run", "scans",
            "spreadsheet", "staff", "thanks", "that", "the", "their", "there", "they", "this", "truck",
            "trucks", "very", "what", "world", "you", "your", "official", "translation", "translations", "website",
            "trailer", "webtoon", "youtube", "reader", "viewpoint", "omniscient", "survive", "survival", "destroyed",
            "ways", "three", "book", "original", "traditional", "simplified", "chinese", "japanese", "spanish",
            "german", "french", "english", "korean", "indonesian", "thai", "novel", "suddenly", "becomes",
            "reality", "average", "office", "worker", "sole", "interest", "favorite", "companion", "humanity",
            "course", "story", "ordinary", "support", "supporter", "monetization", "extras", "lately",
            "has", "had", "life", "previous", "people", "helped", "power", "manipulation", "fear", "problem",
            "doing", "years", "year", "them", "like", "work", "with", "ass", "promises", "promised", "est",
            "il", "mais", "une", "dans", "pois", "ele", "para", "que", "avec", "avez", "notre", "vida", "sua",
            "past", "crowd", "frozen", "global", "coming", "disaster", "shelter", "created", "an", "icy",
            "age", "through", "night", "saint", "claus",
            "sorry", "wait", "okay", "yeah", "yep", "nope", "stop", "idiot",
            "dead", "jle", "trle", "nati", "salur", "sauri",
        }
        if len(tokens) > 3:
            tokens = tokens[:3]
        if len(tokens) == 1 and tokens[0] in stop_tokens:
            return ""
        if len(tokens) > 1 and any(token in stop_tokens for token in tokens):
            return ""
        if any(
            token in {
                "world", "freeze", "apocalypse", "customer", "manager", "hotel", "link", "comments", "questions",
                "ko", "fi", "kofi", "spreadsheet", "discord", "scans", "my", "official", "translation", "website",
                "trailer", "reader", "viewpoint", "omniscient", "traditional", "simplified", "chinese", "japanese",
                "english", "spanish", "german", "french", "thai", "indonesian", "book", "original",
            }
            for token in tokens
        ):
            return ""
        if len(tokens) == 1 and len(tokens[0]) < 4:
            return ""
        normalized = " ".join(token.capitalize() for token in tokens)
        return normalized if is_valid_character_name_candidate(normalized) else ""

    def _metadata_description_excerpt(self, description: object) -> str:
        if isinstance(description, dict):
            ordered_values: list[str] = []
            for key in ("en", "en-us", "en-gb", "ja", "ko", "es", "fr", "de"):
                value = description.get(key)
                if isinstance(value, str) and value.strip():
                    ordered_values.append(value)
            if not ordered_values:
                ordered_values.extend(str(value) for value in description.values() if isinstance(value, str) and value.strip())
            text = ordered_values[0] if ordered_values else ""
        elif isinstance(description, str):
            text = description
        else:
            text = ""
        if not text:
            return ""
        text = re.split(r"\n\s*---+\s*\n", text, maxsplit=1)[0]
        text = re.sub(r"\[[^\]]+\]\([^)]+\)", " ", text)
        text = re.sub(r"https?://\S+", " ", text)
        lines: list[str] = []
        for raw_line in text.splitlines():
            cleaned = raw_line.strip()
            if not cleaned or cleaned.startswith(("-", ">", "*")):
                continue
            lines.append(cleaned)
        return re.sub(r"\s+", " ", " ".join(lines)).strip()

    def _name_key(self, name: str) -> str:
        return " ".join(re.findall(r"[a-z]+", str(name or "").casefold())).strip()

    def _load_nlp(self):
        if self.__class__._NLP is not None:
            return self.__class__._NLP
        with self.__class__._LOAD_LOCK:
            if self.__class__._NLP is not None:
                return self.__class__._NLP
            try:
                import spacy

                self.__class__._NLP = spacy.load("en_core_web_sm", disable=["tagger", "parser", "lemmatizer"])
            except Exception:
                self.__class__._NLP = None
            return self.__class__._NLP
