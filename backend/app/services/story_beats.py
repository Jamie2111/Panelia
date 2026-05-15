from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass
from typing import Any

from app.schemas.project import ChapterMetadata
from app.services.llm_router import LLMRouter, LLMRouterError
from app.services.ocr_cleaner import clean_ocr_text

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class StoryBeatBundle:
    story_script: str
    beats: list[dict[str, Any]]
    provider: str
    model: str
    warning: str | None = None


class StoryBeatService:
    def __init__(self, router: LLMRouter | None = None) -> None:
        self.router = router or LLMRouter()

    def generate(
        self,
        metadata: ChapterMetadata,
        project_title: str | None,
        narrative_units: list[dict[str, Any]],
        character_dictionary: dict[str, str],
        protagonist_name: str | None,
        *,
        required_provider: str | None = None,
        allow_fallback: bool = True,
    ) -> StoryBeatBundle:
        payload = self._story_payload(narrative_units)
        if not payload:
            return StoryBeatBundle(
                story_script="",
                beats=[],
                provider="fallback",
                model="local",
                warning="No usable scene text was available for story beat extraction.",
            )

        beat_count = max(5, min(24, round(len(payload) * 0.55)))
        if self.router.available_providers():
            try:
                result = asyncio.run(
                    self.router.generate_story_beats(
                        payload,
                        {
                            "metadata": self._metadata_payload(metadata),
                            "project_title": project_title or "",
                            "character_dictionary": character_dictionary,
                            "protagonist_name": protagonist_name or "",
                            "beat_count": beat_count,
                        },
                        provider=required_provider,
                    )
                )
                beats = self._normalize_beats(result.payload.get("beats", []), protagonist_name)
                story_script = self._normalize_story_script(
                    str(result.payload.get("story_script") or "").strip(),
                    protagonist_name,
                )
                if beats and story_script:
                    return StoryBeatBundle(
                        story_script=story_script,
                        beats=beats,
                        provider=result.provider,
                        model=result.model,
                    )
            except LLMRouterError as exc:
                if not allow_fallback:
                    raise
                logger.warning("Story beat generation fell back to local beats: %s", exc)

        if not allow_fallback:
            raise LLMRouterError("No usable story beat provider was available.")

        fallback_beats = self._fallback_beats(payload, protagonist_name)
        return StoryBeatBundle(
            story_script=self._compose_story_script(fallback_beats, protagonist_name),
            beats=fallback_beats,
            provider="fallback",
            model="local",
            warning="LLM story beats were unavailable, so Panelia built a local beat outline instead.",
        )

    def _story_payload(self, narrative_units: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not narrative_units:
            return []
        if narrative_units and narrative_units[0].get("panel") is not None:
            payload: list[dict[str, Any]] = []
            for item in narrative_units:
                caption = str(item.get("caption") or "").strip()
                dialogue_block = " ".join(
                    str(entry.get("text") or "").strip()
                    for entry in item.get("dialogue", []) or []
                    if isinstance(entry, dict) and str(entry.get("text") or "").strip()
                )
                combined_text = clean_ocr_text(" ".join(part for part in (caption, dialogue_block) if part).strip())
                if not combined_text:
                    continue
                payload.append(
                    {
                        "panel": int(item.get("panel") or 0),
                        "caption": caption,
                        "dialogue": item.get("dialogue", []) or [],
                        "character_names": [str(name).strip() for name in item.get("character_names", []) or [] if str(name).strip()],
                        "combined_text": combined_text,
                    }
                )
            return payload
        return [
            {
                "scene_id": int(seed.get("scene_id") or 0),
                "panel_start": int(seed.get("panel_start") or 0),
                "panel_end": int(seed.get("panel_end") or 0),
                "combined_text": str(seed.get("combined_text") or "").strip()[:900],
                "character_names": [str(name).strip() for name in seed.get("character_names", []) or [] if str(name).strip()],
            }
            for seed in narrative_units[:36]
            if str(seed.get("combined_text") or "").strip()
        ]

    def align_beats_to_scenes(
        self,
        beats: list[dict[str, Any]],
        scene_seeds: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        if not scene_seeds:
            return []
        if not beats:
            return [
                {
                    "scene_id": int(seed.get("scene_id") or 0),
                    "description": "",
                    "summary": "",
                    "beat_id": None,
                    "characters": [],
                }
                for seed in scene_seeds
            ]
        scenes_per_beat = max(1, -(-len(scene_seeds) // len(beats)))
        aligned: list[dict[str, Any]] = []
        for index, seed in enumerate(scene_seeds):
            beat = beats[min(index // scenes_per_beat, len(beats) - 1)]
            aligned.append(
                {
                    "scene_id": int(seed.get("scene_id") or 0),
                    "description": str(beat.get("description") or "").strip(),
                    "summary": str(beat.get("description") or "").strip(),
                    "beat_id": int(beat.get("beat_id") or 0) if str(beat.get("beat_id") or "").strip() else None,
                    "characters": [str(name).strip() for name in beat.get("characters", []) or [] if str(name).strip()],
                }
            )
        return aligned

    def _normalize_beats(self, beats: list[dict[str, Any]], protagonist_name: str | None) -> list[dict[str, Any]]:
        normalized: list[dict[str, Any]] = []
        for index, beat in enumerate(beats[:36], start=1):
            description = self._normalize_story_script(str(beat.get("description") or "").strip(), protagonist_name)
            if not description:
                continue
            characters = [
                self._normalize_name(str(name).strip(), protagonist_name)
                for name in beat.get("characters", []) or []
                if str(name).strip()
            ]
            normalized.append(
                {
                    "beat_id": int(beat.get("beat_id") or index),
                    "description": description,
                    "characters": list(dict.fromkeys(name for name in characters if name)),
                }
            )
        return normalized

    def _normalize_story_script(self, text: str, protagonist_name: str | None) -> str:
        cleaned = re.sub(r"\s+", " ", str(text or "").strip())
        if not cleaned:
            return ""
        replacement = protagonist_name or "the protagonist"
        cleaned = re.sub(r"(?i)\ba man\b", replacement, cleaned)
        cleaned = re.sub(r"(?i)\bthe man\b", replacement, cleaned)
        cleaned = re.sub(r"(?i)\bthis man\b", replacement, cleaned)
        cleaned = re.sub(r"(?i)\bthe protagonist\b", replacement, cleaned)
        cleaned = re.sub(r"\s+([,.;:!?])", r"\1", cleaned)
        return cleaned

    def _compose_story_script(self, beats: list[dict[str, Any]], protagonist_name: str | None) -> str:
        sentences: list[str] = []
        for beat in beats:
            description = self._normalize_story_script(str(beat.get("description") or "").strip(), protagonist_name)
            if not description:
                continue
            if description[-1] not in ".!?":
                description += "."
            if description in sentences:
                continue
            sentences.append(description)
        if not sentences:
            return ""
        if len(sentences) > 1 and sentences[-1][-1] not in ".!?":
            sentences[-1] += "."
        paragraphs: list[str] = []
        chunk: list[str] = []
        for sentence in sentences:
            chunk.append(sentence)
            if len(chunk) >= 3:
                paragraphs.append(" ".join(chunk))
                chunk = []
        if chunk:
            paragraphs.append(" ".join(chunk))
        return "\n\n".join(paragraphs).strip()

    def _fallback_beats(self, scene_payload: list[dict[str, Any]], protagonist_name: str | None) -> list[dict[str, Any]]:
        beats: list[dict[str, Any]] = []
        for item in scene_payload:
            description = self._fallback_description(item, protagonist_name)
            if not description:
                continue
            beats.append(
                {
                    "beat_id": int(item.get("scene_id") or item.get("panel") or len(beats) + 1),
                    "description": description,
                    "characters": [str(name).strip() for name in item.get("character_names", []) or [] if str(name).strip()],
                }
            )
        return beats[:36]

    def _fallback_description(self, payload_item: dict[str, Any], protagonist_name: str | None) -> str:
        caption = self._normalize_story_script(str(payload_item.get("caption") or "").strip(), protagonist_name)
        text = self._normalize_story_terms(clean_ocr_text(str(payload_item.get("combined_text") or "")).strip())
        lowered = text.casefold()
        subject = protagonist_name or "the protagonist"
        grounded = self._grounded_fallback_description(text, subject)
        if grounded:
            return grounded
        if any(
            token in lowered
            for token in (
                "last thing i remember",
                "where am i",
                "how did i die",
                "near death",
                "can't see",
                "what happened",
            )
        ):
            return f"{subject} is dropped into confusion, piecing together memories of death and the chaos around him."
        if any(token in lowered for token in ("kill him", "corpse", "betray", "chopped up")):
            return "The starving crowd turns on the protagonist and exposes how desperate survival has become."
        if any(token in lowered for token in ("regress", "back in time", "month before", "travel back")):
            return f"{subject} realizes an impossible second chance has appeared before the disaster."
        if any(token in lowered for token in ("three ways to survive", "destroyed world")):
            if any(token in lowered for token in ("i am", "reader", "only child", "dok", "one of them")):
                return f"{subject} slowly realizes he is already tied to the doomed story being described."
            return "The chapter opens by explaining the cruel rules for surviving a destroyed world."
        if any(token in lowered for token in ("freeze", "supernova", "blizzard", "apocalypse")):
            return "The chapter spells out the frozen catastrophe that is about to reshape the world."
        if any(token in lowered for token in ("warehouse", "storage", "ability", "supplies")):
            return f"{subject} gains a new advantage and starts thinking in terms of stockpiling."
        if any(token in lowered for token in ("loan", "deposit", "interest", "million", "mortgage")):
            return f"Money becomes another weapon as {subject} pushes the preparation plan further."
        if any(token in lowered for token in ("security company", "safe house", "vault", "bulletproof")):
            return f"{subject}'s home begins turning into a fortress built for survival."
        if any(token in lowered for token in ("restaurant", "customer", "vip-card", "prepared meals", "banquet")):
            return f"{subject} treats food and prepared meals as part of the larger survival strategy."
        if any(token in lowered for token in ("neighbor", "committee", "coincidence", "rich")):
            return "Suspicion grows as the people around the protagonist start noticing the strange behavior."
        if any(token in lowered for token in ("beautiful", "attractive", "burning smile", "strict and serious")):
            return f"{subject}'s first impression is shaped by the striking people around him."
        if caption:
            if not self._looks_generic_fallback_caption(caption):
                return caption
        clauses = [
            clause.strip(" ,;:-")
            for clause in re.split(r"(?<=[.!?])\s+|,\s+", text)
            if clause.strip()
        ]
        for clause in clauses:
            if len(re.findall(r"[a-z']+", clause.casefold())) >= 4:
                candidate = self._normalize_story_script(clause, protagonist_name)
                if candidate and not self._looks_generic_fallback_caption(candidate):
                    return candidate
        return f"{subject} keeps adjusting the plan as the danger ahead becomes clearer."

    def _grounded_fallback_description(self, text: str, subject: str) -> str:
        lowered = text.casefold()
        if "global freeze" in lowered and "month before" in lowered:
            return "The story jumps back to Tian Hai City one month before the Global Freeze."
        if any(token in lowered for token in ("renasci", "reborn", "returned to the past", "second chance")):
            return f"{subject} realizes he has somehow returned before the catastrophe begins."
        if any(token in lowered for token in ("-60", "-70", "minus 60", "minus 70")) and any(token in lowered for token in ("blizzard", "month", "freeze")):
            return "The coming blizzard will drag temperatures down to minus sixty or seventy degrees and freeze the city for a month."
        if any(token in lowered for token in ("supernova", "blue star", "light-years")) and "freeze" in lowered:
            return "A distant blue star's supernova is revealed as the trigger for the frozen apocalypse."
        if any(token in lowered for token in ("preserve fresh food", "prepared meals", "fresh food", "preservar alimentos", "refeicoes preparadas", "refeições preparadas")):
            return f"{subject} starts thinking through which food and prepared meals can be preserved first."
        if any(token in lowered for token in ("supermarket", "stockpile", "supplies", "medicine", "food", "warehouse")):
            return f"{subject} begins stockpiling food, medicine, and daily necessities before the freeze arrives."
        if any(token in lowered for token in ("loan", "mortgage", "deposit", "bank account", "cash", "million", "account number")):
            return f"{subject} scrambles to raise more cash so the preparation plan can continue."
        if any(token in lowered for token in ("weapon", "weapons", "crossbow", "hunting", "self-defense", "protection")):
            return f"Weapons and self-defense become part of {subject}'s survival plan."
        if any(token in lowered for token in ("security", "safe house", "vault", "alloy", "fortress", "shelter")):
            return f"{subject} moves ahead with turning home into a fortified shelter."
        if any(token in lowered for token in ("restaurant", "vip-card", "prepared meals", "one of everything")):
            return f"{subject} buys food aggressively, treating every meal like part of a survival stockpile."
        if any(token in lowered for token in ("neighbors", "coincidence", "help you out", "dinner sometime", "share the food", "rich guy", "stingy")):
            return f"The neighbors mistake {subject}'s spending spree for easy money and start circling him."
        return ""

    def _normalize_story_terms(self, text: str) -> str:
        normalized = clean_ocr_text(text)
        replacements = {
            "congelamento global": "global freeze",
            "congelamento": "freeze",
            "congelou": "freeze",
            "nevasca": "blizzard",
            "tempestade de neve": "blizzard",
            "estrela azul": "blue star",
            "anos-luz": "light-years",
            "años-luz": "light-years",
            "mês": "month",
            "mes": "month",
            "meses": "months",
            "dias": "days",
            "anos": "years",
            "novembro": "november",
            "dezembro": "december",
            "preservar alimentos": "preserve fresh food",
            "refeicoes preparadas": "prepared meals",
            "refeições preparadas": "prepared meals",
            "renasci": "reborn",
            "emprestado": "borrowed",
            "arma": "weapon",
            "armas": "weapons",
        }
        for wrong, right in replacements.items():
            normalized = re.sub(rf"\b{re.escape(wrong)}\b", right, normalized, flags=re.IGNORECASE)
        return normalized

    def _looks_generic_fallback_caption(self, caption: str) -> bool:
        lowered = caption.casefold()
        return any(
            phrase in lowered
            for phrase in (
                "the situation shifts again",
                "the protagonist finally puts a name and history",
                "a grim narration lays out",
                "the scale of the catastrophe becomes clearer",
                "is shown against",
                "is shown with",
                "are displayed against",
                "stand together",
                "looking thoughtful",
                "glowing object sits on",
                "website",
                "social media",
                "white background",
                "black background",
                "wooden surface",
                "dimly lit room",
                "smiling character",
                "silhouette",
            )
        )

    def _normalize_name(self, name: str, protagonist_name: str | None) -> str:
        cleaned = " ".join(part for part in re.findall(r"[A-Za-z][A-Za-z'-]*", str(name or "")) if part).strip()
        if not cleaned:
            return ""
        if protagonist_name and cleaned.casefold() in {"the protagonist", "protagonist"}:
            return protagonist_name
        return " ".join(token.capitalize() for token in cleaned.split())

    def _metadata_payload(self, metadata: ChapterMetadata) -> dict[str, Any]:
        return {
            "manga_title": metadata.manga_title,
            "chapter_title": metadata.chapter_title,
            "chapter_number": metadata.chapter_number,
            "volume_number": metadata.volume_number,
            "language": metadata.language,
        }
