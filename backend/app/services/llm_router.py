from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import random
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import httpx
import requests
import yaml

from app.core.config import get_settings
from app.services.storytelling_style_guide import immersive_recap_contract

logger = logging.getLogger(__name__)


def _safe_error(exc: object) -> str:
    text = str(exc)
    text = re.sub(r"([?&]key=)[^&\s)]+", r"\1<redacted>", text)
    text = re.sub(r"(Bearer\s+)[A-Za-z0-9._~-]+", r"\1<redacted>", text)
    return text


class LLMRouterError(RuntimeError):
    pass


class LLMValidationError(LLMRouterError):
    pass


# HTTP status codes that are worth retrying (transient server / rate-limit errors).
_RETRYABLE_HTTP_CODES: frozenset[int] = frozenset({429, 500, 502, 503, 529})
# HTTP status codes for Gemini that mean "try the next model" instead of raising.
# 400 = bad request (content/size/safety issue specific to this model config)
# 404 = model not found
_SKIP_MODEL_HTTP_CODES: frozenset[int] = frozenset({400, 404})
def _env_int(name: str, default: int, minimum: int, maximum: int) -> int:
    raw_value = os.getenv(name, "").strip()
    if raw_value:
        try:
            return max(minimum, min(maximum, int(raw_value)))
        except ValueError:
            logger.warning("Ignoring invalid %s=%r", name, raw_value)
    return default


# Maximum number of per-attempt retries before giving up on a model / provider.
_HTTP_MAX_RETRIES: int = _env_int("PANELIA_LLM_HTTP_MAX_RETRIES", 3, 1, 3)
# Ceiling for exponential-backoff sleep, in seconds.
_HTTP_MAX_BACKOFF: float = 60.0


@dataclass(slots=True)
class RoutedResult:
    provider: str
    model: str
    payload: dict[str, Any]


class LLMRouter:
    def __init__(self) -> None:
        self.settings = get_settings()
        self._timeout_seconds = 75
        self._http_client: httpx.AsyncClient | None = None

    async def generate_story_beats(
        self,
        dialogues: list[dict[str, Any]],
        context: dict[str, Any],
        *,
        provider: str | None = None,
    ) -> RoutedResult:
        prompt = self._story_beats_prompt(dialogues, context)
        beat_count = int(context.get("beat_count") or 10)
        dynamic_budget = min(4200, max(1200, 110 * beat_count + 700))
        return await self._route_json(
            task_name="story beats",
            prompt=prompt,
            validator=self._validate_story_beats_payload,
            max_output_tokens=dynamic_budget,
            provider=provider,
        )

    async def rewrite_panel_narration(
        self,
        panel: dict[str, Any],
        beat: dict[str, Any],
        context: dict[str, Any],
        *,
        provider: str | None = None,
    ) -> RoutedResult:
        prompt = self._panel_rewrite_prompt(panel, beat, context)
        return await self._route_json(
            task_name="panel narration rewrite",
            prompt=prompt,
            validator=self._validate_panel_rewrite_payload,
            max_output_tokens=320,
            provider=provider,
        )

    async def rewrite_panel_batch(
        self,
        panels: list[dict[str, Any]],
        context: dict[str, Any],
        *,
        provider: str | None = None,
        panel_image_paths: dict[int, Path] | None = None,
    ) -> RoutedResult:
        prompt = self._panel_batch_rewrite_prompt(panels, context)
        # Build multimodal parts when panel images are available (Gemini only).
        # This lets Gemini Vision read non-English text directly from the manga
        # art and produce accurate English narration.
        parts: list[dict[str, Any]] | None = None
        if panel_image_paths and provider in (None, "gemini"):
            parts = [{"text": prompt}]
            for panel in panels:
                panel_num = int(panel.get("panel") or 0)
                img_path = panel_image_paths.get(panel_num)
                if img_path and img_path.exists():
                    mime = "image/png" if img_path.suffix.lower() == ".png" else "image/jpeg"
                    parts.append({"text": f"Panel {panel_num} image:"})
                    parts.append({
                        "inlineData": {
                            "mimeType": mime,
                            "data": base64.b64encode(img_path.read_bytes()).decode("utf-8"),
                        },
                    })
        return await self._route_json(
            task_name="panel narration batch rewrite",
            prompt=prompt,
            validator=self._validate_panel_batch_rewrite_payload,
            max_output_tokens=min(2600, max(800, 220 * max(len(panels), 1))),
            provider=provider,
            parts=parts,
        )

    async def rewrite_scene_panel_sequence(
        self,
        panels: list[dict[str, Any]],
        context: dict[str, Any],
        *,
        provider: str | None = None,
    ) -> RoutedResult:
        prompt = self._scene_panel_sequence_prompt(panels, context)
        return await self._route_json(
            task_name="scene panel sequence rewrite",
            prompt=prompt,
            validator=self._validate_panel_batch_rewrite_payload,
            max_output_tokens=min(4000, max(1000, 145 * max(len(panels), 1))),
            provider=provider,
        )

    async def spread_scene_panel_beats(
        self,
        panels: list[dict[str, Any]],
        context: dict[str, Any],
        *,
        provider: str | None = None,
    ) -> RoutedResult:
        prompt = self._spread_scene_panel_beats_prompt(panels, context)
        return await self._route_json(
            task_name="scene panel beat spread",
            prompt=prompt,
            validator=self._validate_panel_batch_rewrite_payload,
            max_output_tokens=min(4000, max(800, 180 * max(len(panels), 1))),
            provider=provider,
        )

    async def resolve_character_names(self, dialogues: list[dict[str, Any]], context: dict[str, Any] | None = None) -> RoutedResult:
        prompt = self._character_names_prompt(dialogues, context or {})
        return await self._route_json(
            task_name="character name resolution",
            prompt=prompt,
            validator=self._validate_character_name_payload,
            max_output_tokens=700,
        )

    async def rewrite_full_story(
        self,
        draft_lines: list[str],
        chapter_summary: str,
        character_dictionary: dict[str, Any],
        *,
        project_title: str = "",
        chapter_metadata: dict[str, Any] | None = None,
        locked_examples: str = "",
        previous_lines: list[str] | None = None,
        next_lines: list[str] | None = None,
        chunk_index: int = 1,
        chunk_total: int = 1,
        slot_evidence: list[dict[str, Any]] | None = None,
        preserve_multi_sentence: bool = False,
        provider: str | None = None,
    ) -> RoutedResult:
        prompt = self._story_rewrite_prompt(
            draft_lines,
            chapter_summary,
            character_dictionary,
            project_title=project_title,
            chapter_metadata=chapter_metadata,
            locked_examples=locked_examples,
            previous_lines=previous_lines,
            next_lines=next_lines,
            chunk_index=chunk_index,
            chunk_total=chunk_total,
            slot_evidence=slot_evidence,
            preserve_multi_sentence=preserve_multi_sentence,
        )
        dynamic_budget = min(9000, max(1400, 70 * max(len(draft_lines), 1)))
        return await self._route_json(
            task_name="full story cohesive rewrite",
            prompt=prompt,
            validator=self._validate_story_rewrite_payload,
            max_output_tokens=dynamic_budget,
            provider=provider,
        )

    async def repair_story_lines(
        self,
        lines: list[dict[str, Any]],
        context: dict[str, Any],
        *,
        provider: str | None = None,
    ) -> RoutedResult:
        prompt = self._robotic_story_line_repair_prompt(lines, context)
        return await self._route_json(
            task_name="story line repair",
            prompt=prompt,
            validator=self._validate_indexed_line_rewrite_payload,
            max_output_tokens=min(4200, max(900, 190 * max(len(lines), 1))),
            provider=provider,
        )

    async def refine_story_segment_style(
        self,
        lines: list[dict[str, Any]],
        context: dict[str, Any],
        *,
        provider: str | None = None,
    ) -> RoutedResult:
        prompt = self._story_segment_style_prompt(lines, context)
        return await self._route_json(
            task_name="story segment style refinement",
            prompt=prompt,
            validator=self._validate_indexed_line_rewrite_payload,
            max_output_tokens=min(4200, max(900, 180 * max(len(lines), 1))),
            provider=provider,
            model_candidates=["gemini-2.5-flash-lite", "gemini-2.5-flash"],
        )

    async def expand_story_segment_details(
        self,
        lines: list[dict[str, Any]],
        context: dict[str, Any],
        *,
        provider: str | None = None,
    ) -> RoutedResult:
        prompt = self._story_segment_expansion_prompt(lines, context)
        return await self._route_json(
            task_name="story segment detail expansion",
            prompt=prompt,
            validator=self._validate_indexed_line_rewrite_payload,
            max_output_tokens=min(3200, max(900, 160 * max(len(lines), 1))),
            provider=provider,
            model_candidates=["gemini-2.5-flash-lite", "gemini-2.5-flash"],
        )

    async def suggest_series_cast_hints(
        self,
        context: dict[str, Any],
        *,
        provider: str | None = None,
    ) -> RoutedResult:
        prompt = self._series_cast_hints_prompt(context)
        return await self._route_json(
            task_name="series cast hints",
            prompt=prompt,
            validator=self._validate_series_cast_hints_payload,
            max_output_tokens=1200,
            provider=provider,
        )

    async def build_story_bible(
        self,
        scenes: list[dict[str, Any]],
        context: dict[str, Any],
        *,
        provider: str | None = None,
    ) -> RoutedResult:
        prompt = self._story_bible_prompt(scenes, context)
        return await self._route_json(
            task_name="story bible",
            prompt=prompt,
            validator=self._validate_story_bible_payload,
            max_output_tokens=min(5200, max(1400, 125 * max(len(scenes), 1))),
            provider=provider,
        )

    async def enumerate_characters_from_pages(
        self,
        pages: list[dict[str, Any]],
        context: dict[str, Any],
        *,
        provider: str | None = None,
        page_image_paths: dict[int, Path] | None = None,
    ) -> RoutedResult:
        prompt = self._character_portrait_prompt(pages, context)
        parts = self._page_image_prompt_parts(prompt, pages, page_image_paths, provider)
        return await self._route_json(
            task_name="character portrait enumeration",
            prompt=prompt,
            validator=self._validate_character_portrait_payload,
            max_output_tokens=min(2800, max(700, 260 * max(len(pages), 1))),
            provider=provider,
            parts=parts,
            model_candidates=["gemini-2.5-flash", "gemini-2.5-flash-lite"],
        )

    async def consolidate_character_portraits(
        self,
        characters: list[dict[str, Any]],
        context: dict[str, Any],
        *,
        provider: str | None = None,
    ) -> RoutedResult:
        prompt = self._character_portrait_consolidation_prompt(characters, context)
        return await self._route_json(
            task_name="character portrait consolidation",
            prompt=prompt,
            validator=self._validate_character_portrait_payload,
            max_output_tokens=min(4200, max(900, 180 * max(len(characters), 1))),
            provider=provider,
            model_candidates=["gemini-2.5-flash", "gemini-2.5-flash-lite"],
        )

    async def extract_panel_vision(
        self,
        panels: list[dict[str, Any]],
        context: dict[str, Any],
        *,
        provider: str | None = None,
        panel_image_paths: dict[str, Path] | None = None,
    ) -> RoutedResult:
        prompt = self._panel_vision_prompt(panels, context)
        parts = self._panel_image_prompt_parts(prompt, panels, panel_image_paths, provider)
        return await self._route_json(
            task_name="panel vision extraction",
            prompt=prompt,
            validator=self._validate_panel_vision_payload,
            max_output_tokens=min(4200, max(1200, 360 * max(len(panels), 1))),
            provider=provider,
            parts=parts,
            model_candidates=self._panel_vision_model_candidates(),
        )

    async def rescue_panel_vision(
        self,
        panel: dict[str, Any],
        context: dict[str, Any],
        *,
        provider: str | None = None,
        labeled_image_paths: list[tuple[str, Path]] | None = None,
    ) -> RoutedResult:
        prompt = self._panel_vision_rescue_prompt(panel, context)
        parts: list[dict[str, Any]] | None = None
        if labeled_image_paths and provider in (None, "gemini"):
            parts = [{"text": prompt}]
            for label, image_path in labeled_image_paths:
                if not image_path.exists():
                    continue
                mime = "image/png" if image_path.suffix.lower() == ".png" else "image/jpeg"
                parts.append({"text": label})
                parts.append(
                    {
                        "inlineData": {
                            "mimeType": mime,
                            "data": base64.b64encode(image_path.read_bytes()).decode("utf-8"),
                        }
                    }
                )
        return await self._route_json(
            task_name="panel vision rescue",
            prompt=prompt,
            validator=self._validate_panel_vision_payload,
            max_output_tokens=1000,
            provider=provider,
            parts=parts,
            model_candidates=["gemini-2.5-pro", "gemini-2.5-flash", "gemini-2.5-flash-lite"],
        )

    async def generate_story_segments(
        self,
        scenes: list[dict[str, Any]],
        context: dict[str, Any],
        *,
        provider: str | None = None,
        scene_image_paths: dict[str, list[Path]] | None = None,
    ) -> RoutedResult:
        prompt = self._story_segments_prompt(scenes, context)
        parts = self._story_segment_prompt_parts(prompt, scenes, scene_image_paths, provider)
        return await self._route_json(
            task_name="story segment drafting",
            prompt=prompt,
            validator=self._validate_story_segments_payload,
            max_output_tokens=min(4800, max(1400, 300 * max(len(scenes), 1))),
            provider=provider,
            parts=parts,
            model_candidates=["gemini-2.5-flash-lite", "gemini-2.5-flash"],
        )

    async def cohere_chapter_narrator(
        self,
        lines: list[dict[str, Any]],
        context: dict[str, Any],
        *,
        provider: str | None = None,
    ) -> RoutedResult:
        """Rewrite a chunk of the whole chapter into a single flowing YouTube
        recap narrator voice — same line count, same order, same indices; each
        line may grow to 2-3 sentences with natural transitions.
        """
        prompt = self._chapter_narrator_cohesion_prompt(lines, context)
        return await self._route_json(
            task_name="chapter narrator cohesion",
            prompt=prompt,
            validator=self._validate_indexed_line_rewrite_payload,
            max_output_tokens=min(4200, max(1200, 180 * max(len(lines), 1))),
            provider=provider,
            model_candidates=["gemini-2.5-flash-lite", "gemini-2.5-flash"],
        )

    async def enrich_chapter_narrator(
        self,
        lines: list[dict[str, Any]],
        context: dict[str, Any],
        *,
        provider: str | None = None,
    ) -> RoutedResult:
        """Expand thin narrator lines while preserving index/order/facts."""
        prompt = self._chapter_narrator_enrichment_prompt(lines, context)
        return await self._route_json(
            task_name="chapter narrator enrichment",
            prompt=prompt,
            validator=self._validate_indexed_line_rewrite_payload,
            max_output_tokens=min(4200, max(1200, 220 * max(len(lines), 1))),
            provider=provider,
            model_candidates=["gemini-2.5-flash-lite", "gemini-2.5-flash"],
        )

    async def critique_story_segments(
        self,
        segments: list[dict[str, Any]],
        context: dict[str, Any],
        *,
        provider: str | None = None,
    ) -> RoutedResult:
        prompt = self._story_segment_critic_prompt(segments, context)
        return await self._route_json(
            task_name="story segment critic",
            prompt=prompt,
            validator=self._validate_indexed_line_rewrite_payload,
            max_output_tokens=min(4200, max(900, 190 * max(len(segments), 1))),
            provider=provider,
        )

    async def repair_story_segments_multimodal(
        self,
        segments: list[dict[str, Any]],
        context: dict[str, Any],
        *,
        provider: str | None = None,
        scene_image_paths: dict[str, list[Path]] | None = None,
    ) -> RoutedResult:
        prompt = self._multimodal_story_segment_repair_prompt(segments, context)
        parts = self._story_segment_prompt_parts(prompt, segments, scene_image_paths, provider)
        return await self._route_json(
            task_name="multimodal story segment repair",
            prompt=prompt,
            validator=self._validate_indexed_line_rewrite_payload,
            max_output_tokens=min(4200, max(900, 220 * max(len(segments), 1))),
            provider=provider,
            parts=parts,
        )

    def available_providers(self) -> list[str]:
        providers: list[str] = []
        for provider in self._provider_order():
            if self._provider_enabled(provider):
                providers.append(provider)
        return providers

    def preferred_provider(self) -> str | None:
        providers = self.available_providers()
        return providers[0] if providers else None

    # ─── Grounded web search ─────────────────────────────────────────────────

    async def fetch_series_context(
        self,
        series_title: str,
        chapter_metadata: dict[str, Any],
    ) -> dict[str, Any]:
        """Use Gemini Google Search grounding to retrieve external context about
        the manga/comic series.  Returns a dict with ``search_context`` (plain
        text) that can be injected into the story bible before script generation.

        Falls back gracefully to ``{}`` on any error or when Gemini is unavailable.
        """
        if not series_title.strip():
            return {}
        if "gemini" not in self.available_providers():
            return {}
        try:
            chapter_str = str(chapter_metadata.get("chapter_number") or "").strip()
            volume_str = str(chapter_metadata.get("volume_number") or "").strip()
            query_parts = [f"Manga series '{series_title}'"]
            if volume_str:
                query_parts.append(f"volume {volume_str}")
            if chapter_str:
                query_parts.append(f"chapter {chapter_str}")
            query_parts.append(
                "— main characters (names, roles, relationships, abilities), "
                "story setting, world rules, and key plot events. "
                "Provide factual information that helps narrate the story accurately."
            )
            query = " ".join(query_parts)
            # Use asyncio.to_thread for grounded search since it uses the legacy
            # requests library (grounded search cannot use httpx due to streaming quirks)
            text = await asyncio.to_thread(self._grounded_text_request, query)
            if not text.strip():
                return {}
            return {"search_context": text.strip(), "series_title": series_title}
        except Exception as exc:
            logger.warning("Series context fetch failed for '%s': %s", series_title, _safe_error(exc))
            return {}

    def _grounded_text_request(self, query: str, max_output_tokens: int = 2500) -> str:
        """Make a Gemini request with Google Search grounding enabled.

        Returns plain text.  Cannot use ``responseMimeType: application/json``
        alongside search grounding, so this is a raw text call.
        """
        last_exc: Exception | None = None
        for model in self._gemini_models():
            for attempt in range(min(_HTTP_MAX_RETRIES, 3)):
                try:
                    response = requests.post(
                        f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
                        params={"key": self.settings.gemini_api_key},
                        headers={"Content-Type": "application/json"},
                        json={
                            "contents": [{"parts": [{"text": query}]}],
                            "tools": [{"google_search": {}}],
                            "generationConfig": {
                                "temperature": 0.2,
                                "maxOutputTokens": max_output_tokens,
                            },
                        },
                        timeout=self._timeout_seconds,
                    )
                    response.raise_for_status()
                    data = response.json()
                    candidates = data.get("candidates", [])
                    if not candidates:
                        return ""
                    parts = candidates[0].get("content", {}).get("parts", [])
                    return "\n".join(
                        str(part.get("text") or "").strip()
                        for part in parts
                        if isinstance(part, dict) and str(part.get("text") or "").strip()
                    ).strip()
                except requests.HTTPError as exc:
                    last_exc = exc
                    status_code = exc.response.status_code if exc.response is not None else None
                    if status_code in _SKIP_MODEL_HTTP_CODES:
                        break  # Try next model
                    if status_code in _RETRYABLE_HTTP_CODES and attempt < 2:
                        time.sleep(self._retry_after_seconds(exc.response, attempt))
                        continue
                    break
                except Exception as exc:
                    last_exc = exc
                    break
        if last_exc is not None:
            logger.debug("Grounded search failed: %s", _safe_error(last_exc))
        return ""

    async def _route_json(
        self,
        task_name: str,
        prompt: str,
        validator: Callable[[Any], dict[str, Any]],
        max_output_tokens: int,
        provider: str | None = None,
        *,
        parts: list[dict[str, Any]] | None = None,
        model_candidates: list[str] | None = None,
    ) -> RoutedResult:
        attempts: list[str] = []
        provider_order = [provider] if provider else self._provider_order()
        for candidate in provider_order:
            if not self._provider_enabled(candidate):
                continue
            try:
                model, raw_payload = await self._provider_request_async(
                    candidate,
                    prompt,
                    max_output_tokens,
                    parts=parts,
                    model_candidates=model_candidates,
                )
                raw_text = self._extract_provider_text(candidate, raw_payload)
                try:
                    parsed = self._parse_json_text(raw_text)
                except Exception as parse_exc:
                    repaired_text = await self._repair_json_with_provider_async(
                        candidate,
                        raw_text,
                        max_output_tokens,
                    )
                    if not repaired_text:
                        raise parse_exc
                    parsed = self._parse_json_text(repaired_text)
                validated = validator(parsed)
                logger.info("LLM router handled %s with provider=%s model=%s", task_name, candidate, model)
                return RoutedResult(provider=candidate, model=model, payload=validated)
            except Exception as exc:
                safe = _safe_error(exc)
                logger.warning("LLM router %s attempt failed via %s: %s", task_name, candidate, safe)
                attempts.append(f"{candidate}: {safe}")
                continue
        if provider and not attempts:
            raise LLMRouterError(f"Requested provider {provider} is not enabled for {task_name}.")
        raise LLMRouterError(f"No configured LLM provider could complete {task_name}. Attempts: {' | '.join(attempts) or 'none'}")

    def _provider_order(self) -> list[str]:
        raw = [token.strip().casefold() for token in str(self.settings.llm_provider_order or "").split(",")]
        ordered = [token for token in raw if token in {"gemini", "grok", "deepseek"}]
        return ordered or ["gemini", "grok", "deepseek"]

    def _provider_enabled(self, provider: str) -> bool:
        if provider == "gemini":
            return bool(self.settings.gemini_api_key)
        if provider == "grok":
            return bool(self.settings.grok_api_key)
        if provider == "deepseek":
            return bool(self.settings.deepseek_api_key)
        return False

    def _provider_model(self, provider: str) -> str:
        if provider == "gemini":
            return self.settings.llm_gemini_model or self.settings.gemini_model
        if provider == "grok":
            return self.settings.grok_model
        if provider == "deepseek":
            return self.settings.deepseek_model
        raise KeyError(provider)

    def _provider_request(
        self,
        provider: str,
        prompt: str,
        max_output_tokens: int,
        *,
        parts: list[dict[str, Any]] | None = None,
        model_candidates: list[str] | None = None,
    ) -> tuple[str, dict[str, Any]]:
        if provider == "gemini":
            content_parts = parts if parts else [{"text": prompt}]
            last_exc: Exception | None = None
            # Default Gemini safety blocks many manga/manhwa/anime pages as
            # "medium" harm (often HARM_CATEGORY_SEXUALLY_EXPLICIT because of
            # stylized character art, or DANGEROUS_CONTENT because of fight
            # scenes). For a fictional-art recap tool this results in empty
            # responses on a large share of stylized action pages. Lower to
            # BLOCK_ONLY_HIGH so only actually explicit or
            # dangerous content is blocked. Text responses are still strict
            # because we ask the model to produce recap prose, not raw
            # reproductions of the source material.
            safety_settings = [
                {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_ONLY_HIGH"},
                {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_ONLY_HIGH"},
                {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_ONLY_HIGH"},
                {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_ONLY_HIGH"},
            ]
            for model in self._gemini_models(model_candidates):
                for attempt in range(_HTTP_MAX_RETRIES):
                    try:
                        generation_config = self._gemini_generation_config(
                            model,
                            max_output_tokens=max_output_tokens,
                            response_mime_type="application/json",
                        )
                        response = requests.post(
                            f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
                            params={"key": self.settings.gemini_api_key},
                            headers={"Content-Type": "application/json"},
                            json={
                                "contents": [{"parts": content_parts}],
                                "generationConfig": generation_config,
                                "safetySettings": safety_settings,
                            },
                            timeout=self._timeout_seconds,
                        )
                        response.raise_for_status()
                        data = response.json()
                        # Gemini sometimes returns HTTP 200 with no candidates and
                        # ``promptFeedback.blockReason: OTHER`` — a non-category
                        # prompt-level block that bypasses safety settings. When
                        # that happens, fall through to the next model rather
                        # than returning an empty payload to ``_route_json``.
                        candidates = data.get("candidates", []) or []
                        prompt_block_reason = (data.get("promptFeedback") or {}).get("blockReason")
                        if not candidates and prompt_block_reason:
                            last_exc = LLMRouterError(
                                f"Gemini {model} blocked at prompt level "
                                f"(blockReason={prompt_block_reason})"
                            )
                            logger.warning(
                                "Gemini %s prompt-level block (%s) — trying next model",
                                model,
                                prompt_block_reason,
                            )
                            break  # try next model
                        # Pro sometimes returns a candidate whose parts have no
                        # text and ``finishReason`` is SAFETY / PROHIBITED_CONTENT
                        # / RECITATION / OTHER. That's effectively a refusal.
                        # Detect it here so we can try another model rather than
                        # returning a response that ``_extract_provider_text``
                        # will reject as "empty content".
                        if candidates:
                            first = candidates[0] or {}
                            finish_reason = str(first.get("finishReason") or "").upper()
                            parts = first.get("content", {}).get("parts", []) or []
                            text_present = any(
                                isinstance(part, dict) and str(part.get("text") or "").strip()
                                for part in parts
                            )
                            refusal_finish_reasons = {
                                "SAFETY",
                                "PROHIBITED_CONTENT",
                                "RECITATION",
                                "BLOCKLIST",
                                "OTHER",
                                "SPII",
                                "IMAGE_SAFETY",
                            }
                            if not text_present and finish_reason in refusal_finish_reasons:
                                last_exc = LLMRouterError(
                                    f"Gemini {model} returned empty content "
                                    f"(finishReason={finish_reason or 'unset'})"
                                )
                                logger.warning(
                                    "Gemini %s candidate-level refusal (finishReason=%s) — trying next model",
                                    model,
                                    finish_reason or "unset",
                                )
                                break  # try next model
                        return model, data
                    except requests.HTTPError as exc:
                        last_exc = exc
                        status_code = exc.response.status_code if exc.response is not None else None
                        if status_code in _RETRYABLE_HTTP_CODES:
                            if attempt < _HTTP_MAX_RETRIES - 1:
                                wait = self._retry_after_seconds(exc.response, attempt)
                                logger.warning(
                                    "Gemini %s HTTP %s (attempt %d/%d) — retrying in %.1fs",
                                    model, status_code, attempt + 1, _HTTP_MAX_RETRIES, wait,
                                )
                                time.sleep(wait)
                                continue
                            # Exhausted retries for this model; try the next one.
                            logger.warning(
                                "Gemini %s HTTP %s — exhausted %d retries, trying next model",
                                model, status_code, _HTTP_MAX_RETRIES,
                            )
                            break
                        if status_code in _SKIP_MODEL_HTTP_CODES:
                            # 400: bad request (content/size/safety issue for this model).
                            # 404: model not found.
                            # Either way, skip to the next model rather than retrying.
                            body = (exc.response.text[:400] if exc.response is not None else "(no body)")
                            logger.warning(
                                "Gemini %s HTTP %s — skipping to next model. Response: %s",
                                model, status_code, body,
                            )
                            break
                        # Any other HTTP error (401 Unauthorized, 403 Forbidden, …) — fatal.
                        raise
                    except requests.Timeout:
                        last_exc = requests.Timeout(f"Gemini {model} timed out after {self._timeout_seconds}s")
                        if attempt < _HTTP_MAX_RETRIES - 1:
                            wait = self._retry_after_seconds(None, attempt)
                            logger.warning(
                                "Gemini %s timeout (attempt %d/%d) — retrying in %.1fs",
                                model, attempt + 1, _HTTP_MAX_RETRIES, wait,
                            )
                            time.sleep(wait)
                            continue
                        logger.warning("Gemini %s timeout — exhausted retries, trying next model", model)
                        break
            if last_exc is not None:
                raise last_exc
            raise LLMRouterError("Gemini is enabled but no usable model was configured.")

        if provider == "grok":
            model = self._provider_model(provider)
            last_exc_grok: Exception | None = None
            for attempt in range(_HTTP_MAX_RETRIES):
                try:
                    response = requests.post(
                        f"{self.settings.grok_api_base.rstrip('/')}/chat/completions",
                        headers={
                            "Authorization": f"Bearer {self.settings.grok_api_key}",
                            "Content-Type": "application/json",
                        },
                        json={
                            "model": model,
                            "temperature": 0.35,
                            "max_tokens": max_output_tokens,
                            "messages": [{"role": "user", "content": prompt}],
                        },
                        timeout=self._timeout_seconds,
                    )
                    response.raise_for_status()
                    return model, response.json()
                except requests.HTTPError as exc:
                    last_exc_grok = exc
                    status_code = exc.response.status_code if exc.response is not None else None
                    if status_code in _RETRYABLE_HTTP_CODES and attempt < _HTTP_MAX_RETRIES - 1:
                        wait = self._retry_after_seconds(exc.response, attempt)
                        logger.warning(
                            "Grok HTTP %s (attempt %d/%d) — retrying in %.1fs",
                            status_code, attempt + 1, _HTTP_MAX_RETRIES, wait,
                        )
                        time.sleep(wait)
                        continue
                    raise
                except requests.Timeout:
                    last_exc_grok = requests.Timeout(f"Grok timed out after {self._timeout_seconds}s")
                    if attempt < _HTTP_MAX_RETRIES - 1:
                        wait = self._retry_after_seconds(None, attempt)
                        logger.warning(
                            "Grok timeout (attempt %d/%d) — retrying in %.1fs",
                            attempt + 1, _HTTP_MAX_RETRIES, wait,
                        )
                        time.sleep(wait)
                        continue
                    raise last_exc_grok
            if last_exc_grok is not None:
                raise last_exc_grok
            raise LLMRouterError("Grok request loop exited without result.")

        if provider == "deepseek":
            model = self._provider_model(provider)
            last_exc_ds: Exception | None = None
            for attempt in range(_HTTP_MAX_RETRIES):
                try:
                    response = requests.post(
                        f"{self.settings.deepseek_api_base.rstrip('/')}/chat/completions",
                        headers={
                            "Authorization": f"Bearer {self.settings.deepseek_api_key}",
                            "Content-Type": "application/json",
                        },
                        json={
                            "model": model,
                            "temperature": 0.35,
                            "max_tokens": max_output_tokens,
                            "messages": [{"role": "user", "content": prompt}],
                            "response_format": {"type": "json_object"},
                        },
                        timeout=self._timeout_seconds,
                    )
                    response.raise_for_status()
                    return model, response.json()
                except requests.HTTPError as exc:
                    last_exc_ds = exc
                    status_code = exc.response.status_code if exc.response is not None else None
                    if status_code in _RETRYABLE_HTTP_CODES and attempt < _HTTP_MAX_RETRIES - 1:
                        wait = self._retry_after_seconds(exc.response, attempt)
                        logger.warning(
                            "DeepSeek HTTP %s (attempt %d/%d) — retrying in %.1fs",
                            status_code, attempt + 1, _HTTP_MAX_RETRIES, wait,
                        )
                        time.sleep(wait)
                        continue
                    raise
                except requests.Timeout:
                    last_exc_ds = requests.Timeout(f"DeepSeek timed out after {self._timeout_seconds}s")
                    if attempt < _HTTP_MAX_RETRIES - 1:
                        wait = self._retry_after_seconds(None, attempt)
                        logger.warning(
                            "DeepSeek timeout (attempt %d/%d) — retrying in %.1fs",
                            attempt + 1, _HTTP_MAX_RETRIES, wait,
                        )
                        time.sleep(wait)
                        continue
                    raise last_exc_ds
            if last_exc_ds is not None:
                raise last_exc_ds
            raise LLMRouterError("DeepSeek request loop exited without result.")

        raise KeyError(provider)

    def _retry_after_seconds(self, response: requests.Response | None, attempt: int) -> float:
        """Return seconds to wait before the next retry attempt.

        Respects the Retry-After response header when present.  Otherwise uses
        exponential back-off (2^(attempt+1) seconds) with ±25 % jitter, capped
        at _HTTP_MAX_BACKOFF.  The +1 exponent shift means:
          attempt 0 → ~2 s, attempt 1 → ~4 s, attempt 2 → ~8 s,
          attempt 3 → ~16 s (max ~60 s via cap).
        """
        if response is not None:
            header = response.headers.get("Retry-After")
            if header:
                try:
                    return max(1.0, float(header))
                except ValueError:
                    pass
        base = min(float(2 ** (attempt + 1)), _HTTP_MAX_BACKOFF)
        jitter = random.uniform(0.75, 1.25)
        return round(base * jitter, 2)

    def _retry_after_seconds_httpx(self, response: httpx.Response | None, attempt: int) -> float:
        """Same as _retry_after_seconds but accepts an httpx.Response."""
        if response is not None:
            header = response.headers.get("Retry-After")
            if header:
                try:
                    return max(1.0, float(header))
                except ValueError:
                    pass
        base = min(float(2 ** (attempt + 1)), _HTTP_MAX_BACKOFF)
        jitter = random.uniform(0.75, 1.25)
        return round(base * jitter, 2)

    def _get_http_client(self) -> httpx.AsyncClient:
        """Return a fresh async HTTP client per call.

        A shared AsyncClient is bound to the event loop that created it.
        When asyncio.run() is called from ThreadPoolExecutor threads (e.g. the
        critic pass), each thread gets its own event loop, so reusing a shared
        client causes it to hang on the wrong loop. Creating a new client each
        time is cheap and avoids cross-loop deadlocks.
        """
        return httpx.AsyncClient(timeout=self._timeout_seconds)

    async def _provider_request_async(
        self,
        provider: str,
        prompt: str,
        max_output_tokens: int,
        *,
        parts: list[dict[str, Any]] | None = None,
        model_candidates: list[str] | None = None,
    ) -> tuple[str, dict[str, Any]]:
        """Async version of _provider_request using httpx.AsyncClient.

        Replaces asyncio.to_thread(_provider_request, ...) in _route_json,
        giving true async HTTP concurrency instead of threadpool blocking.
        """
        async with self._get_http_client() as client:
            return await self._provider_request_async_inner(
                provider, prompt, max_output_tokens, parts=parts, model_candidates=model_candidates, client=client,
            )

    async def _provider_request_async_inner(
        self,
        provider: str,
        prompt: str,
        max_output_tokens: int,
        *,
        parts: list[dict[str, Any]] | None = None,
        model_candidates: list[str] | None = None,
        client: httpx.AsyncClient,
    ) -> tuple[str, dict[str, Any]]:
        if provider == "gemini":
            content_parts = parts if parts else [{"text": prompt}]
            safety_settings = [
                {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_ONLY_HIGH"},
                {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_ONLY_HIGH"},
                {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_ONLY_HIGH"},
                {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_ONLY_HIGH"},
            ]
            last_exc: Exception | None = None
            for model in self._gemini_models(model_candidates):
                for attempt in range(_HTTP_MAX_RETRIES):
                    try:
                        generation_config = self._gemini_generation_config(
                            model,
                            max_output_tokens=max_output_tokens,
                            response_mime_type="application/json",
                        )
                        response = await client.post(
                            f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
                            params={"key": self.settings.gemini_api_key},
                            headers={"Content-Type": "application/json"},
                            json={
                                "contents": [{"parts": content_parts}],
                                "generationConfig": generation_config,
                                "safetySettings": safety_settings,
                            },
                        )
                        response.raise_for_status()
                        data = response.json()
                        candidates = data.get("candidates", []) or []
                        prompt_block_reason = (data.get("promptFeedback") or {}).get("blockReason")
                        if not candidates and prompt_block_reason:
                            last_exc = LLMRouterError(
                                f"Gemini {model} blocked at prompt level "
                                f"(blockReason={prompt_block_reason})"
                            )
                            logger.warning(
                                "Gemini %s prompt-level block (%s) — trying next model",
                                model, prompt_block_reason,
                            )
                            break
                        if candidates:
                            first = candidates[0] or {}
                            finish_reason = str(first.get("finishReason") or "").upper()
                            cand_parts = first.get("content", {}).get("parts", []) or []
                            text_present = any(
                                isinstance(part, dict) and str(part.get("text") or "").strip()
                                for part in cand_parts
                            )
                            refusal_finish_reasons = {
                                "SAFETY", "PROHIBITED_CONTENT", "RECITATION",
                                "BLOCKLIST", "OTHER", "SPII", "IMAGE_SAFETY",
                            }
                            if not text_present and finish_reason in refusal_finish_reasons:
                                last_exc = LLMRouterError(
                                    f"Gemini {model} returned empty content "
                                    f"(finishReason={finish_reason or 'unset'})"
                                )
                                logger.warning(
                                    "Gemini %s candidate-level refusal (finishReason=%s) — trying next model",
                                    model, finish_reason or "unset",
                                )
                                break
                        return model, data
                    except httpx.HTTPStatusError as exc:
                        last_exc = exc
                        status_code = exc.response.status_code
                        if status_code in _RETRYABLE_HTTP_CODES:
                            if attempt < _HTTP_MAX_RETRIES - 1:
                                wait = self._retry_after_seconds_httpx(exc.response, attempt)
                                logger.warning(
                                    "Gemini %s HTTP %s (attempt %d/%d) — retrying in %.1fs",
                                    model, status_code, attempt + 1, _HTTP_MAX_RETRIES, wait,
                                )
                                await asyncio.sleep(wait)
                                continue
                            logger.warning(
                                "Gemini %s HTTP %s — exhausted %d retries, trying next model",
                                model, status_code, _HTTP_MAX_RETRIES,
                            )
                            break
                        if status_code in _SKIP_MODEL_HTTP_CODES:
                            body = exc.response.text[:400]
                            logger.warning(
                                "Gemini %s HTTP %s — skipping to next model. Response: %s",
                                model, status_code, body,
                            )
                            break
                        raise
                    except httpx.TimeoutException:
                        last_exc = LLMRouterError(f"Gemini {model} timed out after {self._timeout_seconds}s")
                        if attempt < _HTTP_MAX_RETRIES - 1:
                            wait = self._retry_after_seconds_httpx(None, attempt)
                            logger.warning(
                                "Gemini %s timeout (attempt %d/%d) — retrying in %.1fs",
                                model, attempt + 1, _HTTP_MAX_RETRIES, wait,
                            )
                            await asyncio.sleep(wait)
                            continue
                        logger.warning("Gemini %s timeout — exhausted retries, trying next model", model)
                        break
            if last_exc is not None:
                raise last_exc
            raise LLMRouterError("Gemini is enabled but no usable model was configured.")

        if provider == "grok":
            model = self._provider_model(provider)
            last_exc_grok: Exception | None = None
            for attempt in range(_HTTP_MAX_RETRIES):
                try:
                    response = await client.post(
                        f"{self.settings.grok_api_base.rstrip('/')}/chat/completions",
                        headers={
                            "Authorization": f"Bearer {self.settings.grok_api_key}",
                            "Content-Type": "application/json",
                        },
                        json={
                            "model": model,
                            "temperature": 0.35,
                            "max_tokens": max_output_tokens,
                            "messages": [{"role": "user", "content": prompt}],
                        },
                    )
                    response.raise_for_status()
                    return model, response.json()
                except httpx.HTTPStatusError as exc:
                    last_exc_grok = exc
                    status_code = exc.response.status_code
                    if status_code in _RETRYABLE_HTTP_CODES and attempt < _HTTP_MAX_RETRIES - 1:
                        wait = self._retry_after_seconds_httpx(exc.response, attempt)
                        logger.warning(
                            "Grok HTTP %s (attempt %d/%d) — retrying in %.1fs",
                            status_code, attempt + 1, _HTTP_MAX_RETRIES, wait,
                        )
                        await asyncio.sleep(wait)
                        continue
                    raise
                except httpx.TimeoutException:
                    last_exc_grok = LLMRouterError(f"Grok timed out after {self._timeout_seconds}s")
                    if attempt < _HTTP_MAX_RETRIES - 1:
                        wait = self._retry_after_seconds_httpx(None, attempt)
                        logger.warning(
                            "Grok timeout (attempt %d/%d) — retrying in %.1fs",
                            attempt + 1, _HTTP_MAX_RETRIES, wait,
                        )
                        await asyncio.sleep(wait)
                        continue
                    raise last_exc_grok
            if last_exc_grok is not None:
                raise last_exc_grok
            raise LLMRouterError("Grok request loop exited without result.")

        if provider == "deepseek":
            model = self._provider_model(provider)
            last_exc_ds: Exception | None = None
            for attempt in range(_HTTP_MAX_RETRIES):
                try:
                    response = await client.post(
                        f"{self.settings.deepseek_api_base.rstrip('/')}/chat/completions",
                        headers={
                            "Authorization": f"Bearer {self.settings.deepseek_api_key}",
                            "Content-Type": "application/json",
                        },
                        json={
                            "model": model,
                            "temperature": 0.35,
                            "max_tokens": max_output_tokens,
                            "messages": [{"role": "user", "content": prompt}],
                            "response_format": {"type": "json_object"},
                        },
                    )
                    response.raise_for_status()
                    return model, response.json()
                except httpx.HTTPStatusError as exc:
                    last_exc_ds = exc
                    status_code = exc.response.status_code
                    if status_code in _RETRYABLE_HTTP_CODES and attempt < _HTTP_MAX_RETRIES - 1:
                        wait = self._retry_after_seconds_httpx(exc.response, attempt)
                        logger.warning(
                            "DeepSeek HTTP %s (attempt %d/%d) — retrying in %.1fs",
                            status_code, attempt + 1, _HTTP_MAX_RETRIES, wait,
                        )
                        await asyncio.sleep(wait)
                        continue
                    raise
                except httpx.TimeoutException:
                    last_exc_ds = LLMRouterError(f"DeepSeek timed out after {self._timeout_seconds}s")
                    if attempt < _HTTP_MAX_RETRIES - 1:
                        wait = self._retry_after_seconds_httpx(None, attempt)
                        logger.warning(
                            "DeepSeek timeout (attempt %d/%d) — retrying in %.1fs",
                            attempt + 1, _HTTP_MAX_RETRIES, wait,
                        )
                        await asyncio.sleep(wait)
                        continue
                    raise last_exc_ds
            if last_exc_ds is not None:
                raise last_exc_ds
            raise LLMRouterError("DeepSeek request loop exited without result.")

        raise KeyError(provider)

    async def _repair_json_with_provider_async(
        self, provider: str, raw_text: str, max_output_tokens: int
    ) -> str:
        """Async version of _repair_json_with_provider."""
        repair_prompt = (
            "Repair the following malformed JSON and return only valid JSON.\n"
            "Do not summarize, do not omit fields, do not add commentary.\n"
            "Preserve the original structure and values as closely as possible.\n\n"
            "Malformed JSON:\n"
            f"{raw_text[:16000]}"
        )
        try:
            _, repaired_payload = await self._provider_request_async(
                provider, repair_prompt, min(max_output_tokens + 400, 4000)
            )
            return self._extract_provider_text(provider, repaired_payload)
        except Exception as exc:
            logger.warning("LLM router JSON repair failed via %s: %s", provider, _safe_error(exc))
            return ""

    def _gemini_generation_config(
        self,
        model: str,
        *,
        max_output_tokens: int,
        response_mime_type: str | None = None,
    ) -> dict[str, Any]:
        config: dict[str, Any] = {
            "temperature": 0.35,
            "topP": 0.9,
            "maxOutputTokens": max_output_tokens,
        }
        if response_mime_type:
            config["responseMimeType"] = response_mime_type
        thinking_budget = self._gemini_thinking_budget(model)
        if thinking_budget is not None:
            config["thinkingConfig"] = {"thinkingBudget": thinking_budget}
        return config

    def _gemini_thinking_budget(self, model: str) -> int | None:
        """Return a valid ``thinkingBudget`` or ``None`` to omit ``thinkingConfig``.

        Gemini 2.5 Pro REQUIRES thinking mode: it rejects ``thinkingBudget: 0``
        with ``"Budget 0 is invalid. This model only works in thinking mode."``
        Valid Pro budgets are 128–32768, or ``-1`` for dynamic.

        Gemini 2.5 Flash / Flash-Lite accept ``thinkingBudget`` but we omit it
        entirely (``None``) so Gemini picks a safe default — sending ``0``
        historically has sporadically returned 400s from the API too, and we
        don't want to force-disable thinking on Flash for vision rescue calls.
        """
        normalized = str(model or "").casefold()
        if "gemini-2.5-pro" in normalized:
            # -1 = dynamic thinking; lets Gemini pick the budget. Any fixed value
            # in [128, 32768] would also work but dynamic is safer for mixed workloads.
            return -1
        return None

    # Known-deprecated Gemini model names that 404 on v1beta; filtered out so we
    # don't waste an attempt cycle on a guaranteed-fail request.
    _GEMINI_DEPRECATED_MODELS = {"gemini-2.0-flash", "gemini-1.5-pro", "gemini-1.5-flash"}

    def _gemini_models(self, preferred: list[str] | None = None) -> list[str]:
        if preferred:
            # Explicit candidates are intentional. Story drafting/cohesion uses
            # Flash-only lists for cost control; panel-vision rescue can still
            # opt into Pro by listing it explicitly.
            candidates = [*preferred]
        else:
            candidates = [
                str(self.settings.llm_gemini_model or "").strip(),
                str(self.settings.gemini_model or "").strip(),
                "gemini-2.5-flash-lite",
                "gemini-2.5-flash",
                # Final fallback: Pro has different prompt-level blocking
                # heuristics than Flash/Flash-lite, so it occasionally accepts
                # requests that both Flash variants reject. Costs more but only
                # invoked when a caller has not supplied a stricter candidate
                # list.
                "gemini-2.5-pro",
            ]
        ordered: list[str] = []
        for model in candidates:
            cleaned = model.strip() if isinstance(model, str) else ""
            if not cleaned or cleaned in ordered:
                continue
            if cleaned in self._GEMINI_DEPRECATED_MODELS:
                continue
            ordered.append(cleaned)
        return ordered

    def _panel_vision_model_candidates(self) -> list[str]:
        raw_value = os.getenv("PANELIA_PANEL_VISION_MODELS", "").strip()
        if raw_value:
            candidates = [item.strip() for item in raw_value.split(",") if item.strip()]
            if candidates:
                return candidates
        if os.getenv("PANELIA_PANEL_VISION_LITE_FIRST", "").strip().lower() in {"1", "true", "yes"}:
            return ["gemini-2.5-flash-lite", "gemini-2.5-flash"]
        return ["gemini-2.5-flash", "gemini-2.5-flash-lite"]

    def _extract_provider_json(self, provider: str, payload: dict[str, Any]) -> Any:
        return self._parse_json_text(self._extract_provider_text(provider, payload))

    def _extract_provider_text(self, provider: str, payload: dict[str, Any]) -> str:
        if provider == "gemini":
            candidates = payload.get("candidates", [])
            if not candidates:
                raise LLMValidationError("Gemini returned no candidates")
            parts = candidates[0].get("content", {}).get("parts", [])
            text = "\n".join(
                str(part.get("text") or "").strip()
                for part in parts
                if isinstance(part, dict) and str(part.get("text") or "").strip()
            ).strip()
            if not text:
                raise LLMValidationError("Gemini returned empty content")
            return text

        choices = payload.get("choices", [])
        if not choices:
            raise LLMValidationError(f"{provider} returned no choices")
        message = choices[0].get("message", {})
        content = str(message.get("content") or "").strip()
        if not content:
            raise LLMValidationError(f"{provider} returned empty content")
        return content

    def _repair_json_with_provider(self, provider: str, raw_text: str, max_output_tokens: int) -> str:
        repair_prompt = (
            "Repair the following malformed JSON and return only valid JSON.\n"
            "Do not summarize, do not omit fields, do not add commentary.\n"
            "Preserve the original structure and values as closely as possible.\n\n"
            "Malformed JSON:\n"
            f"{raw_text[:16000]}"
        )
        try:
            _, repaired_payload = self._provider_request(provider, repair_prompt, min(max_output_tokens + 400, 4000))
            return self._extract_provider_text(provider, repaired_payload)
        except Exception as exc:
            logger.warning("LLM router JSON repair failed via %s: %s", provider, _safe_error(exc))
            return ""

    def _parse_json_text(self, raw_text: str) -> Any:
        cleaned = raw_text.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
            cleaned = re.sub(r"\s*```$", "", cleaned)
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError as exc:
            candidates = [
                cleaned,
                self._extract_first_json_candidate(cleaned),
                self._repair_common_json_issues(cleaned),
            ]
            extracted = self._extract_first_json_candidate(cleaned)
            if extracted:
                candidates.append(self._repair_common_json_issues(extracted))

            seen: set[str] = set()
            for candidate in candidates:
                if not candidate:
                    continue
                if candidate in seen:
                    continue
                seen.add(candidate)
                try:
                    return json.loads(candidate)
                except json.JSONDecodeError:
                    decoded = self._raw_decode_json(candidate)
                    if decoded is not None:
                        return decoded
                    yaml_decoded = self._yaml_json_fallback(candidate)
                    if yaml_decoded is not None:
                        return yaml_decoded
                    continue
            raise LLMValidationError(f"Provider returned invalid JSON: {exc}") from exc

    def _extract_first_json_candidate(self, text: str) -> str:
        decoder = json.JSONDecoder()
        stripped = text.lstrip()
        if stripped:
            try:
                _, end = decoder.raw_decode(stripped)
                return stripped[:end]
            except json.JSONDecodeError:
                pass
        match = re.search(r"(\{.*\}|\[.*\])", text, re.DOTALL)
        return match.group(1).strip() if match else ""

    def _raw_decode_json(self, text: str) -> Any | None:
        decoder = json.JSONDecoder()
        stripped = text.lstrip()
        if not stripped:
            return None
        try:
            parsed, _ = decoder.raw_decode(stripped)
            return parsed
        except json.JSONDecodeError:
            return None

    def _yaml_json_fallback(self, text: str) -> Any | None:
        try:
            parsed = yaml.safe_load(text)
        except Exception:
            return None
        return parsed if isinstance(parsed, (dict, list)) else None

    def _repair_common_json_issues(self, text: str) -> str:
        repaired = text.strip()
        # Remove trailing commas before closing brackets
        repaired = re.sub(r",(\s*[}\]])", r"\1", repaired)
        # Fix unescaped newlines inside strings (common Gemini issue)
        repaired = re.sub(r'(?<=": ")(.*?)(?=")', lambda m: m.group(0).replace("\n", " "), repaired)
        # Fix single-quoted strings to double-quoted
        repaired = re.sub(r"(?<=[\[{,:\s])'([^']*?)'(?=[\s,\]}:])", r'"\1"', repaired)

        # Try to close truncated JSON (Gemini hits token limit mid-output)
        stripped = repaired.rstrip()
        if stripped and not stripped.endswith(('}', ']')):
            open_braces = stripped.count('{') - stripped.count('}')
            open_brackets = stripped.count('[') - stripped.count(']')
            # Truncate to last complete element
            last_complete = max(stripped.rfind('}'), stripped.rfind(']'))
            if last_complete > 0:
                candidate = stripped[:last_complete + 1]
                remaining_braces = candidate.count('{') - candidate.count('}')
                remaining_brackets = candidate.count('[') - candidate.count(']')
                candidate += ']' * max(0, remaining_brackets) + '}' * max(0, remaining_braces)
                try:
                    json.loads(candidate)
                    return candidate
                except json.JSONDecodeError:
                    pass
            # Fallback: just close all open delimiters
            if stripped[-1] == ',':
                stripped = stripped[:-1]
            repaired = stripped + ']' * max(0, open_brackets) + '}' * max(0, open_braces)

        lines = repaired.splitlines()
        if len(lines) <= 1:
            return repaired

        repaired_lines: list[str] = []
        previous_significant_index: int | None = None
        for line in lines:
            current = line.rstrip()
            repaired_lines.append(current)
            if not current.strip():
                continue
            if previous_significant_index is not None:
                previous = repaired_lines[previous_significant_index].rstrip()
                current_stripped = current.lstrip()
                if (
                    not previous.endswith((",", "{", "[", ":"))
                    and not current_stripped.startswith(("}", "]", ","))
                    and (
                        current_stripped.startswith('"')
                        or current_stripped.startswith("{")
                        or current_stripped.startswith("[")
                        or re.match(r"^(true|false|null|-?\d)", current_stripped)
                    )
                    and re.search(r'("|[0-9\]}]|true|false|null)\s*$', previous)
                ):
                    repaired_lines[previous_significant_index] = previous + ","
            previous_significant_index = len(repaired_lines) - 1

        return "\n".join(repaired_lines)

    def _validate_story_beats_payload(self, payload: Any) -> dict[str, Any]:
        if not isinstance(payload, dict):
            raise LLMValidationError("Story beat payload is not an object")
        story_script = str(payload.get("story_script") or payload.get("chapter_summary") or "").strip()
        beats_raw = payload.get("beats")
        if not story_script:
            raise LLMValidationError("Story beat payload is missing story_script")
        if not isinstance(beats_raw, list) or not beats_raw:
            raise LLMValidationError("Story beat payload is missing beats")

        beats: list[dict[str, Any]] = []
        for index, item in enumerate(beats_raw[:36], start=1):
            if not isinstance(item, dict):
                continue
            description = str(item.get("description") or item.get("summary") or "").strip()
            if not description:
                continue
            beat_id = self._coerce_int(item.get("beat_id")) or self._coerce_int(item.get("id")) or index
            characters = [
                str(name).strip()
                for name in item.get("characters", []) or []
                if str(name).strip()
            ]
            beats.append(
                {
                    "beat_id": beat_id,
                    "description": description,
                    "characters": list(dict.fromkeys(characters)),
                }
            )
        if not beats:
            raise LLMValidationError("Story beat payload had no usable beats")
        return {
            "story_script": story_script,
            "beats": beats,
        }

    def _validate_character_name_payload(self, payload: Any) -> dict[str, Any]:
        if not isinstance(payload, dict):
            raise LLMValidationError("Character payload is not an object")
        raw_characters = payload.get("characters")
        if not isinstance(raw_characters, list):
            raise LLMValidationError("Character payload is missing characters")
        characters: list[dict[str, Any]] = []
        for item in raw_characters:
            if not isinstance(item, dict):
                continue
            cluster = str(item.get("cluster") or item.get("cluster_id") or "").strip()
            name = str(item.get("name") or "").strip()
            if not cluster or not name:
                continue
            characters.append({"cluster": cluster, "name": name})
        if not characters:
            raise LLMValidationError("Character payload had no usable names")
        return {"characters": characters}

    def _validate_panel_rewrite_payload(self, payload: Any) -> dict[str, Any]:
        if isinstance(payload, str):
            narration = payload.strip()
        elif isinstance(payload, dict):
            narration = str(payload.get("narration") or payload.get("summary") or "").strip()
        else:
            narration = ""
        if not narration:
            raise LLMValidationError("Panel rewrite payload is missing narration")
        return {"narration": narration}

    def _validate_panel_batch_rewrite_payload(self, payload: Any) -> dict[str, Any]:
        if not isinstance(payload, dict):
            raise LLMValidationError("Panel batch rewrite payload is not an object")
        rewrites_raw = payload.get("rewrites")
        if not isinstance(rewrites_raw, list) or not rewrites_raw:
            raise LLMValidationError("Panel batch rewrite payload is missing rewrites")
        rewrites: list[dict[str, Any]] = []
        for item in rewrites_raw:
            if not isinstance(item, dict):
                continue
            panel_id = str(item.get("panel_id") or "").strip()
            panel = self._coerce_int(item.get("panel"))
            narration = str(item.get("narration") or item.get("summary") or "").strip()
            if not narration:
                continue
            rewrites.append(
                {
                    "panel_id": panel_id,
                    "panel": panel,
                    "narration": narration,
                }
            )
        if not rewrites:
            raise LLMValidationError("Panel batch rewrite payload had no usable rewrites")
        return {"rewrites": rewrites}

    def _validate_story_segments_payload(self, payload: Any) -> dict[str, Any]:
        if not isinstance(payload, dict):
            raise LLMValidationError("Story segment payload is not an object")
        raw_segments = payload.get("segments")
        if not isinstance(raw_segments, list) or not raw_segments:
            raise LLMValidationError("Story segment payload is missing segments")
        segments: list[dict[str, Any]] = []
        for index, item in enumerate(raw_segments, start=1):
            if not isinstance(item, dict):
                continue
            segment_id = str(item.get("segment_id") or "").strip() or f"segment_{index:03d}"
            scene_id = self._coerce_int(item.get("scene_id")) or index
            text = str(item.get("text") or item.get("narration") or item.get("summary") or "").strip()
            title = str(item.get("title") or "").strip()
            if not text:
                continue
            segments.append({"segment_id": segment_id, "scene_id": scene_id, "text": text, "title": title})
        if not segments:
            raise LLMValidationError("Story segment payload had no usable segments")
        return {"segments": segments}

    def _coerce_int(self, value: Any) -> int | None:
        if isinstance(value, int):
            return value
        match = re.search(r"\d+", str(value or ""))
        return int(match.group(0)) if match else None

    def _validate_story_rewrite_payload(self, payload: Any) -> dict[str, Any]:
        if not isinstance(payload, dict):
            raise LLMValidationError("Story rewrite payload is not an object")
        lines = payload.get("rewritten_lines")
        if not isinstance(lines, list) or not lines:
            raise LLMValidationError("Story rewrite payload is missing rewritten_lines")
        cleaned = [str(line or "").strip() for line in lines]
        return {"rewritten_lines": cleaned}

    def _validate_indexed_line_rewrite_payload(self, payload: Any) -> dict[str, Any]:
        if not isinstance(payload, dict):
            raise LLMValidationError("Indexed rewrite payload is not an object")
        rewrites_raw = payload.get("rewrites")
        if not isinstance(rewrites_raw, list) or not rewrites_raw:
            raise LLMValidationError("Indexed rewrite payload is missing rewrites")
        rewrites: list[dict[str, Any]] = []
        for item in rewrites_raw:
            if not isinstance(item, dict):
                continue
            index = self._coerce_int(item.get("index"))
            line = str(item.get("line") or item.get("narration") or "").strip()
            if index is None:
                continue
            rewrites.append({"index": index, "line": line})
        if not rewrites:
            raise LLMValidationError("Indexed rewrite payload had no usable rewrites")
        return {"rewrites": rewrites}

    def _validate_series_cast_hints_payload(self, payload: Any) -> dict[str, Any]:
        if not isinstance(payload, dict):
            raise LLMValidationError("Series cast hint payload is not an object")
        hints_raw = payload.get("series_cast_hints")
        corrections_raw = payload.get("canonical_name_corrections")
        hints = [
            str(item).strip()
            for item in hints_raw or []
            if str(item).strip()
        ][:30]
        corrections: list[dict[str, str]] = []
        for item in corrections_raw or []:
            if not isinstance(item, dict):
                continue
            variant = str(item.get("variant") or "").strip()
            canonical = str(item.get("canonical") or "").strip()
            if variant and canonical:
                corrections.append({"variant": variant, "canonical": canonical})
        return {
            "series_cast_hints": hints,
            "canonical_name_corrections": corrections[:20],
        }

    def _validate_story_bible_payload(self, payload: Any) -> dict[str, Any]:
        if not isinstance(payload, dict):
            raise LLMValidationError("Story bible payload is not an object")
        cast_raw = payload.get("cast") or payload.get("canonical_cast") or []
        cast: list[dict[str, Any]] = []
        for item in cast_raw:
            if not isinstance(item, dict):
                continue
            name = str(item.get("name") or "").strip()
            if not name:
                continue
            aliases = [
                str(alias).strip()
                for alias in item.get("aliases", []) or []
                if str(alias).strip()
            ][:8]
            cast.append(
                {
                    "name": name,
                    "aliases": aliases,
                    "role": str(item.get("role") or "").strip(),
                    "visual_cues": str(item.get("visual_cues") or item.get("appearance") or "").strip(),
                    "notes": str(item.get("notes") or "").strip(),
                }
            )

        continuity_notes = [
            str(item).strip()
            for item in payload.get("continuity_notes", []) or []
            if str(item).strip()
        ][:20]
        world_terms = [
            str(item).strip()
            for item in payload.get("world_terms", []) or payload.get("setting_terms", []) or []
            if str(item).strip()
        ][:20]
        scene_memory: list[dict[str, Any]] = []
        for item in payload.get("scene_memory", []) or []:
            if not isinstance(item, dict):
                continue
            scene_id = self._coerce_int(item.get("scene_id"))
            if scene_id is None:
                continue
            scene_memory.append(
                {
                    "scene_id": scene_id,
                    "state": str(item.get("state") or item.get("handoff") or "").strip(),
                    "location": str(item.get("location") or "").strip(),
                    "characters": [
                        str(name).strip()
                        for name in item.get("characters", []) or item.get("involved_characters", []) or []
                        if str(name).strip()
                    ][:8],
                    "open_thread": str(item.get("open_thread") or item.get("tension") or "").strip(),
                }
            )

        return {
            "chapter_premise": str(payload.get("chapter_premise") or payload.get("summary") or "").strip(),
            "cast": cast[:24],
            "world_terms": world_terms,
            "continuity_notes": continuity_notes,
            "scene_memory": scene_memory[:120],
        }

    def _validate_character_portrait_payload(self, payload: Any) -> dict[str, Any]:
        if not isinstance(payload, dict):
            raise LLMValidationError("Character portrait payload is not an object")
        raw_characters = payload.get("characters")
        if raw_characters is None:
            raw_characters = payload.get("cast")
        if not isinstance(raw_characters, list):
            raise LLMValidationError("Character portrait payload is missing characters")
        characters: list[dict[str, Any]] = []
        for item in raw_characters:
            if not isinstance(item, dict):
                continue
            name = str(item.get("name") or "").strip()
            if not name:
                continue
            portrait_pages = [
                coerced
                for page in item.get("portrait_pages", []) or item.get("pages", []) or []
                if (coerced := self._coerce_int(page)) is not None
            ][:6]
            characters.append(
                {
                    "name": name,
                    "role": str(item.get("role") or "supporting").strip() or "supporting",
                    "visual_description": str(
                        item.get("visual_description")
                        or item.get("appearance")
                        or item.get("description")
                        or ""
                    ).strip(),
                    "portrait_pages": portrait_pages,
                    "aliases": [
                        str(alias).strip()
                        for alias in item.get("aliases", []) or []
                        if str(alias).strip()
                    ][:8],
                }
            )
        return {"characters": characters}

    def _validate_panel_vision_payload(self, payload: Any) -> dict[str, Any]:
        if not isinstance(payload, dict):
            raise LLMValidationError("Panel vision payload is not an object")
        raw_panels = payload.get("panels")
        if raw_panels is None and isinstance(payload.get("panel"), dict):
            raw_panels = [payload.get("panel")]
        if not isinstance(raw_panels, list):
            raise LLMValidationError("Panel vision payload is missing panels")
        panels: list[dict[str, Any]] = []
        for item in raw_panels:
            if not isinstance(item, dict):
                continue
            panel_id = str(item.get("panel_id") or "").strip()
            if not panel_id:
                continue
            try:
                confidence = float(item.get("confidence"))
            except (TypeError, ValueError):
                confidence = 0.0
            panels.append(
                {
                    "panel_id": panel_id,
                    "speaker": str(item.get("speaker") or "unknown").strip() or "unknown",
                    "dialogue": str(item.get("dialogue") or "").strip(),
                    "caption": str(item.get("caption") or "").strip(),
                    "action_beat": str(item.get("action_beat") or item.get("summary") or "").strip(),
                    "emotion": str(item.get("emotion") or "").strip(),
                    "scene_change": bool(item.get("scene_change")),
                    "confidence": max(0.0, min(1.0, confidence)),
                    "character_names": [
                        str(name).strip()
                        for name in item.get("character_names", []) or []
                        if str(name).strip()
                    ][:8],
                }
            )
        if not panels:
            raise LLMValidationError("Panel vision payload had no usable panels")
        return {"panels": panels}

    def _load_prompt_template(self, filename: str) -> str:
        prompt_path = Path(__file__).resolve().parents[3] / "services" / "prompts" / filename
        return prompt_path.read_text(encoding="utf-8")

    def _page_image_prompt_parts(
        self,
        prompt: str,
        pages: list[dict[str, Any]],
        page_image_paths: dict[int, Path] | None,
        provider: str | None,
    ) -> list[dict[str, Any]] | None:
        if not page_image_paths or provider not in (None, "gemini"):
            return None
        parts: list[dict[str, Any]] = [{"text": prompt}]
        added_any = False
        for page in pages:
            page_number = int(page.get("page") or 0)
            image_path = page_image_paths.get(page_number)
            if image_path is None or not image_path.exists():
                continue
            mime = "image/png" if image_path.suffix.lower() == ".png" else "image/jpeg"
            parts.append({"text": f"Page {page_number}"})
            parts.append(
                {
                    "inlineData": {
                        "mimeType": mime,
                        "data": base64.b64encode(image_path.read_bytes()).decode("utf-8"),
                    }
                }
            )
            added_any = True
        return parts if added_any else None

    def _panel_image_prompt_parts(
        self,
        prompt: str,
        panels: list[dict[str, Any]],
        panel_image_paths: dict[str, Path] | None,
        provider: str | None,
    ) -> list[dict[str, Any]] | None:
        if not panel_image_paths or provider not in (None, "gemini"):
            return None
        parts: list[dict[str, Any]] = [{"text": prompt}]
        added_any = False
        for panel in panels:
            panel_id = str(panel.get("panel_id") or "").strip()
            image_path = panel_image_paths.get(panel_id)
            if image_path is None or not image_path.exists():
                continue
            mime = "image/png" if image_path.suffix.lower() == ".png" else "image/jpeg"
            parts.append({"text": f"Panel {panel.get('panel_order') or panel.get('panel') or panel_id}"})
            parts.append(
                {
                    "inlineData": {
                        "mimeType": mime,
                        "data": base64.b64encode(image_path.read_bytes()).decode("utf-8"),
                    }
                }
            )
            added_any = True
        return parts if added_any else None

    def _character_portrait_prompt(self, pages: list[dict[str, Any]], context: dict[str, Any]) -> str:
        template = self._load_prompt_template("character_portrait.md")
        payload = [
            {
                "page": int(page.get("page") or 0),
                "notes": str(page.get("notes") or "").strip(),
            }
            for page in pages
        ]
        return (
            template
            .replace("{page_count}", str(len(pages)))
            .replace("{chapter_context}", json.dumps(context.get("chapter_context") or {}, ensure_ascii=False))
            .replace("{project_context}", json.dumps(context.get("project_context") or {}, ensure_ascii=False))
        ) + f"\nRequested pages: {json.dumps(payload, ensure_ascii=False)}\n"

    def _character_portrait_consolidation_prompt(self, characters: list[dict[str, Any]], context: dict[str, Any]) -> str:
        template = self._load_prompt_template("character_portrait_consolidation.md")
        payload = [
            {
                "name": str(item.get("name") or "").strip(),
                "role": str(item.get("role") or "supporting").strip() or "supporting",
                "visual_description": str(item.get("visual_description") or "").strip(),
                "portrait_pages": [
                    coerced
                    for page in item.get("portrait_pages", []) or []
                    if (coerced := self._coerce_int(page)) is not None
                ][:8],
                "aliases": [
                    str(alias).strip()
                    for alias in item.get("aliases", []) or []
                    if str(alias).strip()
                ][:10],
            }
            for item in characters
            if str(item.get("name") or "").strip()
        ]
        return (
            template
            .replace("{chapter_context}", json.dumps(context.get("chapter_context") or {}, ensure_ascii=False))
            .replace("{project_context}", json.dumps(context.get("project_context") or {}, ensure_ascii=False))
            .replace("{provisional_characters}", json.dumps(payload, ensure_ascii=False))
        )

    def _panel_vision_prompt(self, panels: list[dict[str, Any]], context: dict[str, Any]) -> str:
        template = self._load_prompt_template("panel_vision.md")
        manifest = [
            {
                "panel_id": str(panel.get("panel_id") or "").strip(),
                "panel_order": int(panel.get("panel_order") or panel.get("panel") or 0),
                "page": int(panel.get("page") or 0),
                "existing_hint": str(panel.get("existing_hint") or "").strip(),
            }
            for panel in panels
        ]
        return (
            template
            .replace("{character_roster}", json.dumps(context.get("character_roster") or [], ensure_ascii=False))
            .replace("{chapter_context}", json.dumps(context.get("chapter_context") or {}, ensure_ascii=False))
            .replace("{panel_manifest}", json.dumps(manifest, ensure_ascii=False))
        )

    def _panel_vision_rescue_prompt(self, panel: dict[str, Any], context: dict[str, Any]) -> str:
        template = self._load_prompt_template("panel_vision.md")
        manifest = [
            {
                "panel_id": str(panel.get("panel_id") or "").strip(),
                "panel_order": int(panel.get("panel_order") or panel.get("panel") or 0),
                "page": int(panel.get("page") or 0),
                "existing_hint": str(panel.get("existing_hint") or "").strip(),
                "previous_panel_action": str(context.get("previous_panel_action") or "").strip(),
                "next_panel_action": str(context.get("next_panel_action") or "").strip(),
            }
        ]
        return (
            "Rescue pass: resolve this difficult panel with extra neighborhood context.\n"
            "Look carefully at the current panel first, then use adjacent panels only to clarify speaker identity or scene continuity.\n\n"
            + template
            .replace("{character_roster}", json.dumps(context.get("character_roster") or [], ensure_ascii=False))
            .replace("{chapter_context}", json.dumps(context.get("chapter_context") or {}, ensure_ascii=False))
            .replace("{panel_manifest}", json.dumps(manifest, ensure_ascii=False))
        )

    def _story_segments_prompt(self, scenes: list[dict[str, Any]], context: dict[str, Any]) -> str:
        metadata = context.get("chapter_metadata") or {}
        chapter_summary = str(context.get("chapter_summary") or "").strip()
        project_title = str(context.get("project_title") or "").strip()
        protagonist_name = str(context.get("protagonist_name") or "").strip()
        character_dictionary = context.get("character_dictionary") or {}
        story_bible = context.get("story_bible") or {}
        running_memory = str(context.get("running_memory") or "").strip()
        scene_memory = context.get("scene_memory") or []
        allowed_character_names = context.get("allowed_character_names") or []
        style_vocab_block = self._style_vocabulary_prompt_block(context)
        character_role_block = self._character_role_framing_block(context)
        chapter_context = str(context.get("chapter_context") or "").strip()
        payload = [
            {
                "segment_id": str(scene.get("segment_id") or "").strip(),
                "scene_id": int(scene.get("scene_id") or 0),
                "sequence_in_scene": int(scene.get("sequence_in_scene") or 1),
                "scene_unit_count": int(scene.get("scene_unit_count") or 1),
                "panel_start": int(scene.get("panel_start") or 0),
                "panel_end": int(scene.get("panel_end") or 0),
                "panel_count": int(scene.get("panel_count") or len(scene.get("panel_ids", []) or [])),
                "character_names": [
                    str(name).strip()
                    for name in scene.get("character_names", []) or []
                    if str(name).strip()
                ],
                "scene_summary": str(scene.get("scene_summary") or "").strip(),
                "combined_text": str(scene.get("combined_text") or "").strip(),
                "visual_cues": str(scene.get("visual_cues") or "").strip(),
                "vision_dialogue": str(scene.get("vision_dialogue") or "").strip(),
                "vision_caption": str(scene.get("vision_caption") or "").strip(),
                "vision_action_beat": str(scene.get("vision_action_beat") or "").strip(),
                "ocr_fallback_text": str(scene.get("ocr_fallback_text") or "").strip(),
            }
            for scene in scenes
        ]
        mode_block = (
            "PANEL MODE — each input segment represents EXACTLY ONE kept panel:\n"
            "- Write narration covering ONLY what that panel shows. Never summarise the surrounding scene or a later panel.\n"
            "- Use vision_action_beat, vision_dialogue, and vision_caption as primary evidence. Use ocr_fallback_text only when those three are all empty.\n"
            "- Write 2-3 natural English sentences for panels with clear action or dialogue; 1 tight sentence only for purely visual/transitional panels.\n"
            "  • Sentence 1: name the active subject and what they are doing or saying.\n"
            "  • Sentence 2 (when supported by evidence): the immediate reaction, consequence, or emotional beat.\n"
            "  • Sentence 3 (when supported): a stakes phrase or caption detail that carries the moment forward.\n"
            "- If vision_dialogue is non-empty, integrate what is said into narration (paraphrase — do not quote directly).\n"
            "EMOTIONAL TONE — match pacing to what the panel actually conveys:\n"
            "- Tense/action panels (vision_action_beat contains fighting, running, shouting, explosions, confrontations): use short punchy sentences, active verbs, present-tense urgency.\n"
            "- Emotional peaks (visual_cues or vision_action_beat show tears, rage, shock, laughter, embrace, sacrifice): let the sentence breathe — name the emotion or its physical expression explicitly.\n"
            "- Quiet/character panels (vision_action_beat is subdued, introspective, or conversational): slower rhythm, longer phrases, interiority allowed ('weighing her next words', 'something shifts in his gaze').\n"
            "- Revelatory panels (vision_caption or vision_dialogue contains a key fact or twist): lead with the revelation, not the setup.\n"
            "- Transitional/establishing panels (wide shots, location changes, time skips): one efficient sentence that moves the story clock forward.\n"
            f"{style_vocab_block}"
            "LENGTH TARGET per panel:\n"
            "- 50-80 words is the healthy band: 1-2 sentences, grounded and complete. Below 20 words is too thin. Above 100 words is over-extended for a single panel.\n"
            "- Ground every word in THIS panel's vision evidence — do not invent facts.\n"
            "- If local evidence is genuinely weak, write one short conservative sentence rather than padding with generic filler.\n"
        )
        return (
            "You are writing a continuous English manga/manhwa/comic recap script in the voice of a YouTube recap narrator.\n"
            "Return valid JSON only in this format:\n"
            "{\"segments\":[{\"segment_id\":\"scene_001_beat_01\",\"scene_id\":1,\"title\":\"Optional short label\",\"text\":\"Beat-level narration.\"}]}\n\n"
            "Rules:\n"
            f"{immersive_recap_contract()}"
            "- Write exactly one narrated segment for each input segment_id.\n"
            "- Keep the input order and do not merge, skip, or duplicate any segment_id.\n"
            f"{mode_block}"
            f"{character_role_block}"
            "- The segments should feel like one unfolding story, not isolated captions.\n"
            "- Use the supplied reference images to identify setting, physical action, recurring characters, and emotional pressure when OCR is sparse.\n"
            "- Printed narration boxes, on-page exposition, and embedded captions visible in the images count as strong local evidence even when OCR extraction failed.\n"
            "- For prologues, lore dumps, or montage pages, paraphrase the visible narration text from the images into clean recap prose instead of leaving the beat empty.\n"
            "- If OCR is malformed, clipped, or ungrammatical, ignore it and trust the images plus surrounding context instead.\n"
            "- Prefer concrete names from the character dictionary over vague labels like someone, a character, or a figure.\n"
            "- For segment-specific characters, use a proper name only when that name appears in this segment's character_names, vision_dialogue, vision_caption, vision_action_beat, combined_text, or local OCR evidence. The character dictionary and story bible may normalize names already present locally, but they do not let you introduce a character before the segment establishes them.\n"
            "- If a person's exact name is uncertain, use a natural role label instead of inventing a new proper name.\n"
            "- Do not invent flashbacks, motives, or time jumps unless the evidence clearly supports them.\n"
            "- If the title and chapter metadata clearly identify a known series, you may use that context to disambiguate iconic characters, factions, and setting terms that are already visually obvious in the segment; you may also reference known character roles and reputations as emotional framing when those characters appear, but never override the local evidence.\n"
            "- Use scene_summary as guidance, but ground the actual wording in the local scene evidence.\n"
            "- Treat vision_action_beat, vision_dialogue, vision_caption, visual_cues, and combined_text as the only sources for segment-specific facts.\n"
            "- Use ocr_fallback_text only when the vision fields are empty, and ignore it if it looks clipped or garbled.\n"
            "- For 2-4 panel aligned chunks, do not return a single-sentence caption when the local evidence contains both action and dialogue/caption evidence.\n"
            "- Preserve important facts, named events, places, and causal explanations.\n"
            "- Respect the running story memory so each new segment picks up naturally from the previous ones.\n"
            "- If local evidence is genuinely weak, write one short conservative bridge line that still moves with the surrounding story instead of padding with generic filler.\n"
            "- Prefer paraphrase over direct quotation. Only quote dialogue when it is genuinely essential.\n"
            "- Do not lightly rephrase raw dialogue bubbles into narration if the result still sounds like clipped OCR.\n"
            "- Avoid near-duplicate openings across adjacent scenes.\n"
            "- Do not use visual-report phrasing like 'is shown', 'is depicted', 'is displayed', 'is seen', or 'stands in front of'. Rewrite the evidence as story action, cause, pressure, or consequence.\n"
            "- Do not use empty glue phrases like 'confusion takes over', 'magic erupts without warning', 'the scene keeps evolving', or 'consequences push the story forward'.\n"
            "- Do not mention panels, pages, camera angles, or the viewer.\n\n"
            f"Project title: {project_title or '(unknown)'}\n"
            f"Chapter metadata: {json.dumps(metadata, ensure_ascii=False)}\n"
            f"Chapter summary: {chapter_summary or '(none)'}\n"
            f"Protagonist hint: {protagonist_name or '(unknown)'}\n"
            f"Character dictionary: {json.dumps(character_dictionary, ensure_ascii=False)}\n\n"
            f"Allowed character names: {json.dumps(allowed_character_names, ensure_ascii=False)}\n"
            f"Story bible: {json.dumps(story_bible, ensure_ascii=False)}\n"
            f"Running story memory: {running_memory or '(none)'}\n"
            f"Relevant scene memory: {json.dumps(scene_memory, ensure_ascii=False)}\n\n"
            f"Scenes: {json.dumps(payload, ensure_ascii=False)}\n"
        )

    @staticmethod
    def _character_role_framing_block(context: dict[str, Any]) -> str:
        """Build a tonal framing block from story_bible cast roles and series_external_context.

        Allows the narrator to colour tone with known character roles/reputations without
        inventing events — e.g. a character's known reputation (elite but feared, former prodigy
        who failed) is a role fact, not a fabricated event.
        """
        story_bible = context.get("story_bible") or {}
        if not isinstance(story_bible, dict):
            return ""
        cast = story_bible.get("cast") or []
        series_context = str(story_bible.get("series_external_context") or "").strip()
        if not cast and not series_context:
            return ""

        lines: list[str] = []
        for member in cast:
            if not isinstance(member, dict):
                continue
            name = str(member.get("name") or "").strip()
            notes = str(member.get("notes") or "").strip()
            role = str(member.get("role") or "").strip()
            if name and notes:
                role_label = f" ({role})" if role else ""
                lines.append(f"  • {name}{role_label}: {notes}")

        if not lines and not series_context:
            return ""

        block = "CHARACTER ROLE FRAMING — use these as tonal anchors, not as new events:\n"
        if lines:
            block += "\n".join(lines) + "\n"
        block += (
            "Rules for role framing:\n"
            "- You MAY reference a character's established role or reputation when they appear in a segment (e.g., note the weight of a failed pilot's desperate need to prove themselves, or the unease that surrounds a character known to be dangerous).\n"
            "- You MAY NOT invent specific past events, dialogue, or motivations that are not visible in the current segment's vision evidence.\n"
            "- Role framing colors tone — it does NOT override local evidence. If a character acts against type in this panel, describe what the panel shows.\n"
            "- For scenes showing two lead characters together: reflect the underlying tension or connection their dynamic carries, grounded in what is visually expressed.\n"
        )
        return block

    @staticmethod
    def _style_vocabulary_prompt_block(context: dict[str, Any]) -> str:
        vocab = context.get("style_vocabulary") or {}
        if not isinstance(vocab, dict) or not vocab:
            return ""

        def _items(key: str, limit: int) -> list[str]:
            raw = vocab.get(key) or []
            if isinstance(raw, str):
                raw = [raw]
            return [str(item).strip() for item in list(raw)[:limit] if str(item).strip()]

        named = _items("named_characters", 4)
        team = str(vocab.get("team_term") or "").strip()
        world = _items("world_terms", 5)
        stakes = _items("stakes_phrases", 5)
        if not any((named, team, world, stakes)):
            return ""
        return (
            "RECURRING WORLD VOCABULARY (use these names only when they fit this segment's local evidence; do not invent alternates):\n"
            f"- Named characters: {', '.join(named) if named else '(none supplied)'}\n"
            f"- Group / team term: {team or '(none supplied)'}\n"
            f"- Recurring world terms: {', '.join(world) if world else '(none supplied)'}\n"
            f"- Recurring stakes phrases: {', '.join(stakes) if stakes else '(none supplied)'}\n"
            "Important: never introduce a named character, team, or world term into a segment unless it appears in that segment's character_names, vision fields, scene_summary, or immediate neighboring context.\n"
        )

    def _story_bible_prompt(self, scenes: list[dict[str, Any]], context: dict[str, Any]) -> str:
        project_title = str(context.get("project_title") or "").strip()
        chapter_metadata = context.get("chapter_metadata") or {}
        chapter_summary = str(context.get("chapter_summary") or "").strip()
        protagonist_name = str(context.get("protagonist_name") or "").strip()
        character_dictionary = context.get("character_dictionary") or {}
        allowed_character_names = context.get("allowed_character_names") or []
        payload = [
            {
                "scene_id": int(scene.get("scene_id") or 0),
                "character_names": [
                    str(name).strip()
                    for name in scene.get("character_names", []) or []
                    if str(name).strip()
                ],
                "scene_summary": str(scene.get("scene_summary") or "").strip(),
                "combined_text": str(scene.get("combined_text") or "").strip()[:500],
            }
            for scene in scenes
        ]
        return (
            "Build a compact continuity bible for a single manga/manhwa/comic recap run.\n\n"
            "Return valid JSON only in this format:\n"
            "{\"chapter_premise\":\"...\",\"cast\":[{\"name\":\"Character A\",\"aliases\":[\"A1\"],\"role\":\"lead\",\"visual_cues\":\"dark-haired character\",\"notes\":\"...\"}],"
            "\"world_terms\":[\"Faction\",\"Threat\"],\"continuity_notes\":[\"...\"],"
            "\"scene_memory\":[{\"scene_id\":1,\"state\":\"...\",\"location\":\"...\",\"characters\":[\"Character A\"],\"open_thread\":\"...\"}]}\n\n"
            "Rules:\n"
            "- Keep this compact and practical for downstream generation.\n"
            "- Use the project title, chapter metadata, observed scene summaries, OCR snippets, and current character dictionary.\n"
            "- Soft outside series knowledge may be used only to normalize highly likely names, roles, and setting terms for this exact title.\n"
            "- Do not invent unsupported plot twists, backstory, or scene-specific facts.\n"
            "- Cast names must come only from the character dictionary, protagonist hint, allowed character names list, or highly confident title-aware normalization.\n"
            "- If a candidate name is uncertain, omit it from cast and use a role label in scene_memory instead of inventing a new person.\n"
            "- chapter_premise should be one short paragraph explaining the chapter's starting state and arc.\n"
            "- continuity_notes should be concrete rules or reminders that reduce naming and timeline drift.\n"
            "- scene_memory should contain one short memory handoff per scene_id, grounded in the supplied scene evidence.\n\n"
            f"Project title: {project_title or '(unknown)'}\n"
            f"Chapter metadata: {json.dumps(chapter_metadata, ensure_ascii=False)}\n"
            f"Chapter summary: {chapter_summary or '(none)'}\n"
            f"Protagonist hint: {protagonist_name or '(unknown)'}\n"
            f"Character dictionary: {json.dumps(character_dictionary, ensure_ascii=False)}\n"
            f"Allowed character names: {json.dumps(allowed_character_names, ensure_ascii=False)}\n"
            f"Scenes: {json.dumps(payload, ensure_ascii=False)}\n"
        )

    def _chapter_narrator_cohesion_prompt(
        self,
        lines: list[dict[str, Any]],
        context: dict[str, Any],
    ) -> str:
        project_title = str(context.get("project_title") or "").strip()
        chapter_metadata = context.get("chapter_metadata") or {}
        chapter_summary = str(context.get("chapter_summary") or "").strip()
        protagonist_name = str(context.get("protagonist_name") or "").strip()
        character_dictionary = context.get("character_dictionary") or {}
        allowed_character_names = context.get("allowed_character_names") or []
        chunk_index = int(context.get("chunk_index") or 1)
        chunk_total = int(context.get("chunk_total") or 1)
        require_multi_sentence = bool(context.get("require_multi_sentence"))
        style_vocab_block = self._style_vocabulary_prompt_block(context)
        payload = [
            {
                "index": int(item.get("index") or 0),
                "scene_id": int(item.get("scene_id") or 0),
                "panel_count": int(item.get("panel_count") or 0),
                "current": str(item.get("text") or item.get("current_line") or item.get("current") or "").strip(),
                "scene_summary": str(item.get("scene_summary") or "").strip(),
                "vision_dialogue": str(item.get("vision_dialogue") or "").strip(),
                "vision_caption": str(item.get("vision_caption") or "").strip(),
                "vision_action_beat": str(item.get("vision_action_beat") or "").strip(),
                "ocr_fallback_text": str(item.get("ocr_fallback_text") or "").strip(),
                "local_evidence": str(item.get("local_evidence") or "").strip(),
                "character_names": [
                    str(name).strip()
                    for name in item.get("character_names", []) or []
                    if str(name).strip()
                ],
                "visual_only": bool(item.get("visual_only")),
            }
            for item in lines
        ]
        multi_sentence_rule = (
            "- HARD REQUIREMENT: for every non-empty visual_only=false line, return 2-3 complete sentences when evidence supports it. One-sentence rewrites are allowed only for genuinely minimal evidence.\n"
            if require_multi_sentence
            else ""
        )
        return (
            "You are the final narrator voice for a YouTube manga/manhwa/comic recap video.\n"
            "You receive a consecutive chunk of the chapter's narration lines in order.\n"
            "Your ONLY job is to make the existing lines flow as one continuous narrator voice.\n"
            "You are NOT writing new narration. You are NOT a scriptwriter. You are an editor whose\n"
            "job is to polish wording so each line connects cleanly to its neighbours.\n\n"
            "Return valid JSON only in this format:\n"
            "{\"rewrites\":[{\"index\":0,\"line\":\"...\"}]}\n\n"
            "GROUNDING CONSTRAINTS (highest priority — violations cause rejection):\n"
            "- DO NOT introduce any proper noun (character name, place name, faction, title) that is not already in the same line's `current` text or in that line's `vision_dialogue` / `vision_caption` / `vision_action_beat` / `scene_summary` / `local_evidence`. If the original line says 'a pilot', do not promote them to a named character unless that name is in the local evidence for THAT line.\n"
            "- DO NOT swap which character is in a line. If the original line is about Character A, the rewrite is about Character A. Do not move Character B, Character C, or any other character into a line that did not name them.\n"
            "- DO NOT invent new events, motives, backstory, time labels, or causal claims. If the original says 'Character A stumbled upon Character B', do not add 'after fleeing the briefing' unless the local evidence for THAT line supports it.\n"
            "- DO NOT reorder events across lines. The order of indices is the order the viewer will see. Each index keeps its own beat. If you think the chronology is wrong, leave it alone — that is not your job.\n"
            "- DO NOT merge two lines into one or split one line into two. One input index → one output index.\n"
            "- If the original line is already accurate and you cannot improve it without breaking these rules, copy `current` to `line` unchanged.\n"
            "- If the original is genuinely too generic to flow (e.g. \"the pressure carries forward\"), rewrite ONLY from THAT line's local evidence. Never borrow facts from neighbouring lines or from chapter_summary to fill it in.\n\n"
            "Style rules (apply only after the grounding constraints above):\n"
            f"{immersive_recap_contract()}"
            f"{style_vocab_block}"
            "- Output EVERY supplied index exactly once, in the supplied order, using the same integer indices.\n"
            "- Skip lines flagged visual_only=true: keep their index but return an empty string for 'line'.\n"
            f"{multi_sentence_rule}"
            "- Each non-empty line covers exactly one panel beat. Keep it focused and panel-scale.\n"
            "- EMOTIONAL PACING: preserve the emotional register of the original line. If vision_action_beat or vision_dialogue signal a high-stakes moment (confrontation, betrayal, sacrifice, revelation), use shorter punchy sentences and active verbs. If the evidence shows a quiet intimate moment, allow longer rhythmic phrases.\n"
            "- RELATIONSHIP TENSION: when consecutive lines place two lead characters in the same scene and the dynamic is unstable or shifting (wariness, curiosity, unspoken vulnerability, reluctant trust), do NOT smooth the tension into neutral description. Let the uncertainty show — use language that implies something unresolved rather than settling prematurely into warmth or hostility.\n"
            "- KEY MOMENT PACING: for lines that represent a first meeting, a pivotal decision, or a turning point in a relationship, allow one extra sentence of pause or consequence rather than rushing to the next beat. These lines can exceed the 50-80 word target by up to 20 words when the evidence genuinely supports it.\n"
            "- Each non-empty line should be 2-3 natural sentences. Use 1 sentence only for purely visual or transitional panels with very thin evidence.\n"
            "- If a `current` line is a single very short sentence (≤ 10 words) and the surrounding evidence supports elaboration, add 1-2 sentences using only words and facts in current/evidence/prev/next. Treat recurring vocabulary terms as already-established.\n"
            "- If a `current` line is already 2-3 sentences that flow well, leave it unchanged — improving flow is the goal, not length for its own sake.\n"
            "- Absolutely NEVER repeat the same sentence or near-duplicate of an adjacent line. If two adjacent lines describe the same moment, trim the later one to a short transitional clause that points forward — do not invent content.\n"
            "- Use varied openings. Do not start more than two lines in a row with the same subject or the same verb.\n"
            "- Use natural connectors between panels (\"meanwhile\", \"that same instant\", \"elsewhere\", \"in the next moment\") only when the evidence supports the shift.\n"
            "- Name characters instead of using 'someone', 'a figure', 'another character', BUT only when that character is named in this line's evidence.\n"
            "- No visual-report phrasing (\"is shown\", \"is depicted\", \"can be seen\", \"is displayed\", \"appears to be\"). Rewrite as action, cause, or consequence.\n"
            "- No camera language, panel references, or viewer addresses.\n"
            "- No first-person narration, no second person (\"you\").\n"
            "- Keep tense consistent (present tense by default, or past tense if the chunk is clearly a flashback/recollection).\n"
            "- Aim for 50-80 words per line. Do not exceed 100 words per line during cohesion.\n\n"
            "Examples of FORBIDDEN edits (these will be rejected and the original line restored):\n"
            "  current: \"A pilot reports that the enemy is spreading out.\"\n"
            "  bad:     \"Character C reports that the enemy is spreading out, alarming Character D.\"  (invented Character C and Character D)\n"
            "  current: \"Character A stumbled upon Character B in a clearing.\"\n"
            "  bad:     \"After fleeing the briefing, Character A stumbled upon Character B in a clearing.\"  (invented 'after fleeing the briefing')\n"
            "  current: \"The squad scrambles as the enemy presses forward.\"\n"
            "  bad:     \"Character D's squad scrambles in Base 13 as the enemy presses forward.\"  (invented Base 13 and Character D's command)\n\n"
            f"Project title: {project_title or '(unknown)'}\n"
            f"Chunk: {chunk_index}/{chunk_total}\n"
            f"Chapter metadata: {json.dumps(chapter_metadata, ensure_ascii=False)}\n"
            f"Chapter summary: {chapter_summary or '(none)'}\n"
            f"Protagonist: {protagonist_name or '(unknown)'}\n"
            f"Character dictionary: {json.dumps(character_dictionary, ensure_ascii=False)}\n"
            f"Allowed character names: {json.dumps(allowed_character_names, ensure_ascii=False)}\n\n"
            f"Lines: {json.dumps(payload, ensure_ascii=False)}\n"
        )

    def _chapter_narrator_enrichment_prompt(
        self,
        lines: list[dict[str, Any]],
        context: dict[str, Any],
    ) -> str:
        style_vocab_block = self._style_vocabulary_prompt_block(context)
        chapter_context = str(context.get("chapter_context") or "").strip()
        payload = [
            {
                "index": int(item.get("index") or 0),
                "current": str(item.get("current") or item.get("text") or "").strip(),
                "evidence": str(item.get("evidence") or "").strip(),
                "scene_summary": str(item.get("scene_summary") or "").strip(),
                "previous_line": str(item.get("previous_line") or "").strip(),
                "next_line": str(item.get("next_line") or "").strip(),
                "character_names": [
                    str(name).strip()
                    for name in item.get("character_names", []) or []
                    if str(name).strip()
                ],
            }
            for item in lines
        ]
        return (
            "You are an editor repairing thin chapter narration without inflating healthy lines.\n"
            "You are NOT writing new narration. You are NOT a scriptwriter.\n"
            "Your only job is to strengthen blank, one-sentence, very short, or generic lines.\n\n"
            "Return valid JSON only in this format:\n"
            "{\"rewrites\":[{\"index\":0,\"line\":\"Expanded narration.\"}]}\n\n"
            "GROUNDING CONSTRAINTS (highest priority - violations cause rejection):\n"
            "- Never introduce a proper noun absent from current line, evidence, scene_summary, previous_line, next_line, or recurring world vocabulary.\n"
            "- Chapter context may explain world rules and stakes, but it does NOT permit adding a named character absent from current/previous/next/evidence/character_names.\n"
            "- Never swap which character is in a line.\n"
            "- Never invent events, motives, backstory, time labels, causal claims, characters, or locations not in evidence.\n"
            "- Never reorder events across lines.\n\n"
            "EXPANSION RULES:\n"
            "- Each line covers exactly one panel. Your job is to make thin lines richer — not longer for its own sake.\n"
            "- Rewrite ONLY lines that are blank, one sentence, under 40 words, or visibly generic. Lines already 60+ words that read naturally may stay unchanged.\n"
            "- Add 1-2 sentences of grounded detail to each thin line. Do not add more than 2 unless the evidence is unusually rich.\n"
            "- If the current line contains generic bridge scaffolding (for example: immediate problem, next exchange, before anyone can settle, cannot fully interpret, harder to protect), replace those phrases with concrete evidence-grounded prose instead of preserving them.\n"
            "- Pull elaboration material from evidence (vision_action_beat, vision_dialogue, vision_caption, visual_cues), scene_summary, previous_line, next_line, and recurring world vocabulary.\n"
            "- Use chapter context only to clarify established world terms or consequences; do not use it to jump ahead in the plot.\n"
            "- Add detail of one of these types: emotional valence, sensory detail, body language, character motivation, consequence, or implication.\n"
            "- Do NOT change which named characters appear, in what order, or what they do.\n"
            "- Length target: 50-80 words per panel. Above 100 words loses single-panel focus.\n"
            "- Reuse recurring world vocabulary freely as already-established context, but do not invent new vocabulary.\n"
            "- Skip an index if you cannot expand without violating grounding constraints.\n"
            "- Avoid template filler such as 'the next exchange', 'the pressure carries forward', or 'the situation tightens' unless the line's evidence says it concretely.\n\n"
            f"{style_vocab_block}"
            f"Chapter context for world/stakes only: {chapter_context[:1800]}\n\n"
            f"Lines to enrich: {json.dumps(payload, ensure_ascii=False)}\n"
        )

    def _story_segment_critic_prompt(self, segments: list[dict[str, Any]], context: dict[str, Any]) -> str:
        project_title = str(context.get("project_title") or "").strip()
        chapter_summary = str(context.get("chapter_summary") or "").strip()
        chapter_metadata = context.get("chapter_metadata") or {}
        character_dictionary = context.get("character_dictionary") or {}
        story_bible = context.get("story_bible") or {}
        allowed_character_names = context.get("allowed_character_names") or []
        style_vocab_block = self._style_vocabulary_prompt_block(context)
        return (
            "You are the verification pass for a comic recap script.\n\n"
            "Return valid JSON only in this format:\n"
            "{\"rewrites\":[{\"index\":0,\"line\":\"...\"}]}\n\n"
            "Critical rules:\n"
            "- Rewrite every supplied segment.\n"
            "- Keep the same chronological slot and the same local beat coverage.\n"
            "- Output fluent English only.\n"
            "- Treat vision_dialogue, vision_caption, vision_action_beat, and local_evidence as the trusted local source of truth.\n"
            "- Do not replace a concrete local vision beat with a chapter-level summary or neighboring-scene bridge.\n"
            "- Remove unsupported claims, wrong names, accidental time jumps, and repeated paraphrases.\n"
            "- Prefer canonical names from the story bible and character dictionary when confidence is high.\n"
            "- Never introduce a new proper name unless it is supported by the local evidence or the allowed character names list.\n"
            "- Translate any non-English dialogue or caption meaning into English instead of copying foreign-language text.\n"
            "- If the evidence is weak, write the shortest safe bridge line you can.\n"
            "- If even a conservative bridge would be speculative, return an empty string for that slot.\n"
            "- Ignore malformed OCR fragments instead of preserving them in slightly cleaner wording.\n"
            "- Never mention panels, pages, viewers, or camera language.\n"
            "- Keep each line natural, spoken, and compact.\n"
            "- Prefer paraphrase over direct quotation unless the exact quoted wording is essential.\n"
            "- Preserve concrete facts that are supported locally.\n\n"
            "- Eliminate visual-report phrasing like 'is shown', 'is depicted', 'is displayed', or 'is seen'. Rewrite it as story action or consequence.\n"
            "- Eliminate vague filler like 'confusion takes over', 'magic erupts without warning', or 'consequences push the story forward'.\n\n"
            f"Project title: {project_title or '(unknown)'}\n"
            f"Chapter metadata: {json.dumps(chapter_metadata, ensure_ascii=False)}\n"
            f"Chapter summary: {chapter_summary or '(none)'}\n"
            f"Character dictionary: {json.dumps(character_dictionary, ensure_ascii=False)}\n"
            f"Allowed character names: {json.dumps(allowed_character_names, ensure_ascii=False)}\n"
            f"Story bible: {json.dumps(story_bible, ensure_ascii=False)}\n\n"
            f"Segments to verify: {json.dumps(segments, ensure_ascii=False)}\n"
        )

    def _multimodal_story_segment_repair_prompt(self, segments: list[dict[str, Any]], context: dict[str, Any]) -> str:
        project_title = str(context.get("project_title") or "").strip()
        chapter_summary = str(context.get("chapter_summary") or "").strip()
        chapter_metadata = context.get("chapter_metadata") or {}
        character_dictionary = context.get("character_dictionary") or {}
        protagonist_name = str(context.get("protagonist_name") or "").strip()
        story_bible = context.get("story_bible") or {}
        allowed_character_names = context.get("allowed_character_names") or []
        payload = [
            {
                "index": int(item.get("index") or 0),
                "segment_id": str(item.get("segment_id") or "").strip(),
                "scene_id": int(item.get("scene_id") or 0),
                "sequence_in_scene": int(item.get("sequence_in_scene") or 1),
                "scene_unit_count": int(item.get("scene_unit_count") or 1),
                "panel_count": int(item.get("panel_count") or 0),
                "current_line": str(item.get("current_line") or "").strip(),
                "combined_text": str(item.get("combined_text") or "").strip(),
                "ocr_fallback_text": str(item.get("ocr_fallback_text") or "").strip(),
                "visual_cues": str(item.get("visual_cues") or "").strip(),
                "vision_dialogue": str(item.get("vision_dialogue") or "").strip(),
                "vision_caption": str(item.get("vision_caption") or "").strip(),
                "vision_action_beat": str(item.get("vision_action_beat") or "").strip(),
                "character_names": [
                    str(name).strip()
                    for name in item.get("character_names", []) or []
                    if str(name).strip()
                ],
                "scene_summary": str(item.get("scene_summary") or "").strip(),
                "previous_line": str(item.get("previous_line") or "").strip(),
                "next_line": str(item.get("next_line") or "").strip(),
                "weak_reason": str(item.get("weak_reason") or "").strip(),
            }
            for item in segments
        ]
        return (
            "You are repairing weak story beats in a manga/manhwa/comic recap using the actual segment images.\n\n"
            "Return valid JSON only in this format:\n"
            "{\"rewrites\":[{\"index\":0,\"line\":\"...\"}]}\n\n"
            "Critical rules:\n"
            "- Rewrite every supplied segment.\n"
            "- Keep the SAME slot order and the SAME local beat coverage.\n"
            "- Use the reference images as the primary evidence.\n"
            "- Output fluent English only.\n"
            "- Translate any visible Portuguese, Spanish, Japanese, or other non-English text into natural English recap prose instead of copying it.\n"
            "- Treat vision_dialogue, vision_caption, and vision_action_beat as trusted pre-read evidence from the panel vision pass.\n"
            "- Use ocr_fallback_text only as a secondary sidecar when vision fields are sparse; paraphrase only clean, sentence-like words and ignore broken shards.\n"
            "- Printed narration boxes, on-page exposition, and embedded captions visible in the images are reliable evidence even when OCR extraction failed.\n"
            "- If the segment is an exposition or montage beat, paraphrase the visible printed narration into natural recap prose.\n"
            "- Treat OCR as unreliable when it looks clipped, malformed, or ungrammatical.\n"
            "- Ignore garbled OCR instead of repeating it in cleaner English.\n"
            "- Preserve supported dialogue meaning, but paraphrase it into natural recap narration.\n"
            "- Do not output raw bubble fragments like \"It's an opportunity\" or \"A mental connection\" unless the full sentence is clearly visible and truly essential.\n"
            "- Keep the line grounded in what this exact segment shows, not the whole chapter.\n"
            "- Use neighboring lines only for flow and continuity, not for stealing facts.\n"
            "- Prefer canonical character names from the story bible, character dictionary, protagonist hint, and allowed name list.\n"
            "- If a name is uncertain, use a natural role label instead of inventing a new proper name.\n"
            "- Never keep placeholder labels like Unknown man, Unknown woman, Unknown child, narrator voice, or chibi-style person in the final line.\n"
            "- Each line should usually be one natural sentence, or two short sentences at most.\n"
            "- For segments covering multiple panels or sitting between clearly narrated beats, write a short conservative bridge line instead of returning empty.\n"
            "- Do not return an empty string when the segment has reference images, trusted vision evidence, dialogue, captions, or neighboring context.\n"
            "- Reserve an empty string only when there is literally no image evidence and no textual evidence.\n"
            "- If the title and chapter metadata clearly identify a known series, you may use that context to normalize obvious terms or iconic characters that are already visible in the images, but never override the local evidence.\n"
            "- If the images do not support a safe spoken line, return an empty string for that segment.\n"
            "- Remove visual-report phrasing like 'is shown', 'is depicted', 'is displayed', or 'is seen'. Rewrite with concrete story movement instead.\n"
            "- Remove vague filler like 'confusion takes over', 'magic erupts without warning', or 'consequences push the story forward'.\n"
            "- Never mention panels, pages, frames, viewers, or camera language.\n\n"
            f"Project title: {project_title or '(unknown)'}\n"
            f"Chapter metadata: {json.dumps(chapter_metadata, ensure_ascii=False)}\n"
            f"Chapter summary: {chapter_summary or '(none)'}\n"
            f"Protagonist hint: {protagonist_name or '(unknown)'}\n"
            f"Character dictionary: {json.dumps(character_dictionary, ensure_ascii=False)}\n"
            f"Allowed character names: {json.dumps(allowed_character_names, ensure_ascii=False)}\n"
            f"Story bible: {json.dumps(story_bible, ensure_ascii=False)}\n\n"
            f"Segments to repair: {json.dumps(payload, ensure_ascii=False)}\n"
        )

    def _story_segment_prompt_parts(
        self,
        prompt: str,
        scenes: list[dict[str, Any]],
        scene_image_paths: dict[str, list[Path]] | None,
        provider: str | None,
    ) -> list[dict[str, Any]] | None:
        if provider not in (None, "gemini") or not scene_image_paths:
            return None
        parts: list[dict[str, Any]] = [{"text": prompt}]
        added_any = False
        for scene in scenes:
            segment_id = str(scene.get("segment_id") or "").strip()
            scene_id = int(scene.get("scene_id") or 0)
            image_paths = scene_image_paths.get(segment_id) or scene_image_paths.get(str(scene_id)) or []
            for image_index, image_path in enumerate(image_paths[:3], start=1):
                if not image_path.exists():
                    continue
                raw = image_path.read_bytes()
                if not raw or len(raw) > 900 * 1024:
                    continue
                mime = "image/png" if image_path.suffix.lower() == ".png" else "image/jpeg"
                parts.append({"text": f"Reference image for segment_id {segment_id or scene_id}, scene_id {scene_id}, image {image_index}."})
                parts.append(
                    {
                        "inlineData": {
                            "mimeType": mime,
                            "data": base64.b64encode(raw).decode("utf-8"),
                        }
                    }
                )
                added_any = True
        return parts if added_any else None

    def _story_rewrite_prompt(
        self,
        draft_lines: list[str],
        chapter_summary: str,
        character_dictionary: dict[str, Any],
        *,
        project_title: str = "",
        chapter_metadata: dict[str, Any] | None = None,
        locked_examples: str = "",
        previous_lines: list[str] | None = None,
        next_lines: list[str] | None = None,
        chunk_index: int = 1,
        chunk_total: int = 1,
        slot_evidence: list[dict[str, Any]] | None = None,
        preserve_multi_sentence: bool = False,
    ) -> str:
        prompt_path = Path(__file__).resolve().parents[3] / "services" / "prompts" / "gemini-story-rewrite.md"
        template = prompt_path.read_text(encoding="utf-8")
        if preserve_multi_sentence:
            template = template.replace(
                "The draft was assembled panel-by-panel, so it already has a fixed 1:1 mapping to the app's script slots.",
                "The draft was assembled as story segments, so it already has a fixed 1:1 mapping to the app's script slots. Some lines intentionally contain multiple sentences for one full scene.",
            )
            template = template.replace(
                "- Each output line should usually be one sentence.",
                "- Preserve multi-sentence scene lines. If an input line has 2-4 sentences, return 2-4 sentences covering the same setup, action, and consequence beats; do not compress a full scene into one sentence.",
            )
            template = template.replace(
                "- Keep lines compact enough for TTS, but not so short that they sound choppy. Usually 10-24 words is right.",
                "- Keep each scene line compact enough for TTS, but allow up to about 90 words when a full scene needs setup, action, and consequence.",
            )

        char_block = ""
        if character_dictionary:
            char_entries = []
            for name, info in character_dictionary.items():
                aliases = info.get("aliases", []) if isinstance(info, dict) else []
                role = info.get("role", "") if isinstance(info, dict) else ""
                parts = [name]
                if aliases:
                    parts.append(f"(also: {', '.join(aliases)})")
                if role:
                    parts.append(f"— {role}")
                char_entries.append(" ".join(parts))
            char_block = "\n".join(char_entries)

        numbered_draft = "\n".join(f"{i}: {line}" for i, line in enumerate(draft_lines))
        previous_block = "\n".join(
            f"{i}: {line}"
            for i, line in enumerate(previous_lines or [])
            if str(line or "").strip()
        )
        next_block = "\n".join(
            f"{i}: {line}"
            for i, line in enumerate(next_lines or [])
            if str(line or "").strip()
        )
        slot_block = json.dumps(slot_evidence or [], ensure_ascii=False)

        return (
            template
            .replace("{line_count}", str(len(draft_lines)))
            .replace("{project_title}", project_title or "(unknown)")
            .replace("{chapter_metadata}", json.dumps(chapter_metadata or {}, ensure_ascii=False) or "{}")
            .replace("{character_dictionary}", char_block or "(none)")
            .replace("{chapter_summary}", chapter_summary or "(none)")
            .replace("{locked_examples}", locked_examples or "(none)")
            .replace("{previous_lines}", previous_block or "(none)")
            .replace("{next_lines}", next_block or "(none)")
            .replace("{chunk_index}", str(max(chunk_index, 1)))
            .replace("{chunk_total}", str(max(chunk_total, 1)))
            .replace("{slot_evidence}", slot_block)
            .replace("{draft_script}", numbered_draft)
        )

    def _robotic_story_line_repair_prompt(self, lines: list[dict[str, Any]], context: dict[str, Any]) -> str:
        project_title = str(context.get("project_title") or "").strip()
        chapter_summary = str(context.get("chapter_summary") or "").strip()
        chapter_metadata = context.get("chapter_metadata") or {}
        character_dictionary = context.get("character_dictionary") or {}
        locked_examples = str(context.get("locked_examples") or "").strip()
        payload = [
            {
                "index": int(item.get("index") or 0),
                "current_line": str(item.get("current_line") or "").strip(),
                "strict_line": str(item.get("strict_line") or "").strip(),
                "previous_line": str(item.get("previous_line") or "").strip(),
                "next_line": str(item.get("next_line") or "").strip(),
                "ocr_text": str(item.get("ocr_text") or "").strip(),
                "dialogue": item.get("dialogue", []) or [],
                "character_names": [
                    str(name).strip()
                    for name in item.get("character_names", []) or []
                    if str(name).strip()
                ],
                "preferred_subject": str(item.get("preferred_subject") or "").strip(),
                "scene_summary": str(item.get("scene_summary") or "").strip(),
            }
            for item in lines
        ]
        return (
            "You are repairing a small batch of robotic manga recap lines.\n\n"
            "Return valid JSON only in this format:\n"
            "{\"rewrites\":[{\"index\":0,\"line\":\"...\"}]}\n\n"
            "Critical rules:\n"
            "- Rewrite every supplied line.\n"
            "- Keep the SAME local story beat and the SAME slot alignment.\n"
            "- Write like a natural spoken recap, not a panel caption log.\n"
            "- Prefer action, intent, consequence, and pressure over report-style verbs like "
            "\"expresses\", \"questions\", \"states\", \"declares\", \"reacts\", or \"looks\".\n"
            "- If a line only says that somebody speaks, rewrite it to include what that line accomplishes in the scene.\n"
            "- Replace generic openers like \"Another figure\", \"A character\", \"Someone\", or \"The speaker\" "
            "with a real name or natural role label when the evidence supports it.\n"
            "- Remove speculative hedges like \"perhaps\", \"presumably\", or \"seemingly\".\n"
            "- Do not narrate raw sound effects by themselves. Translate them into the event they signal if the evidence supports it.\n"
            "- Keep the wording compact and TTS-friendly, usually 8-22 words.\n"
            "- Use previous_line and next_line only for flow; do not steal facts from neighboring slots.\n"
            "- If strict_line is already the best grounded version, stay close to it and simply make it sound smoother.\n"
            "- Never use first-person narration.\n"
            "- Never mention panels, pages, frames, or camera language.\n\n"
            "Immersive recap contract:\n"
            f"{immersive_recap_contract()}\n"
            f"Project title: {project_title or '(unknown)'}\n"
            f"Chapter metadata: {json.dumps(chapter_metadata, ensure_ascii=False)}\n"
            f"Character dictionary: {json.dumps(character_dictionary, ensure_ascii=False)}\n"
            f"Locked examples: {locked_examples or '(none)'}\n"
            f"Chapter summary: {chapter_summary or '(none)'}\n\n"
            f"Lines to repair: {json.dumps(payload, ensure_ascii=False)}\n"
        )

    def _story_segment_style_prompt(self, lines: list[dict[str, Any]], context: dict[str, Any]) -> str:
        project_title = str(context.get("project_title") or "").strip()
        chapter_summary = str(context.get("chapter_summary") or "").strip()
        chapter_metadata = context.get("chapter_metadata") or {}
        character_dictionary = context.get("character_dictionary") or {}
        story_bible = context.get("story_bible") or {}
        allowed_character_names = context.get("allowed_character_names") or []
        style_vocab_block = self._style_vocabulary_prompt_block(context)
        payload = [
            {
                "index": int(item.get("index") or 0),
                "current_line": str(item.get("current_line") or "").strip(),
                "previous_line": str(item.get("previous_line") or "").strip(),
                "next_line": str(item.get("next_line") or "").strip(),
                "ocr_text": str(item.get("ocr_text") or "").strip(),
                "vision_dialogue": str(item.get("vision_dialogue") or "").strip(),
                "vision_caption": str(item.get("vision_caption") or "").strip(),
                "vision_action_beat": str(item.get("vision_action_beat") or "").strip(),
                "scene_summary": str(item.get("scene_summary") or "").strip(),
                "visual_cues": str(item.get("visual_cues") or "").strip(),
                "character_names": [
                    str(name).strip()
                    for name in item.get("character_names", []) or []
                    if str(name).strip()
                ],
                "panel_count": int(item.get("panel_count") or 0),
            }
            for item in lines
        ]
        return (
            "You are refining already-aligned manga recap narration lines so they sound like a polished spoken story recap.\n\n"
            "Return valid JSON only in this format:\n"
            "{\"rewrites\":[{\"index\":0,\"line\":\"...\"}]}\n\n"
            "Critical rules:\n"
            "- Rewrite every supplied line.\n"
            "- Keep the SAME local story beat and the SAME segment alignment.\n"
            "- Improve cadence, cohesion, and spoken flow without changing what happens.\n"
            "- If current_line is a duplicate, generic bridge, visual inventory, or stale scene-summary fallback, ignore it and rewrite from trusted local evidence.\n"
            "- Trusted local evidence order: vision_dialogue, vision_caption, vision_action_beat, ocr_text, visual_cues.\n"
            "- If trusted local evidence is sparse but current_line is a conservative continuity bridge, keep the same limited meaning and make it more specific using only established names/world terms from current_line, previous_line, next_line, and the story bible.\n"
            "- Prefer a short safe continuity beat over returning an empty line when the neighboring context clearly establishes the same thread.\n"
            "- Treat scene_summary as continuity context only; never copy it and never let it override local evidence.\n"
            "- If local evidence is raw dialogue, paraphrase the pressure, choice, consequence, or emotional turn instead of quoting it.\n"
            "- Output fluent English only; translate any non-English evidence before narrating it.\n"
            "- Write in third-person recap narration, never as direct dialogue.\n"
            "- Remove quoted speech, question-and-answer phrasing, speaker labels, and raw bubble wording.\n"
            "- Avoid panel-description language such as 'is shown', 'looks', 'stands', 'smiles', 'visible in the background', sound-effect wording, or camera framing.\n"
            "- Replace reported-speech constructions like 'asking', 'telling', 'calling out', or 'stating' with the event, pressure, or decision they create.\n"
            "- Never output inventory captions like 'figures on an escalator', 'young women relaxing', 'two mecha suits', or 'charges forward' unless rewritten into story consequence.\n"
            "- Prefer cause, decision, pressure, consequence, or emotional turn over visual inventory.\n"
            "- Preserve reliable names, setting terms, and concrete plot facts from the evidence.\n"
            "- Do not keep placeholder labels like Unknown man, Unknown woman, Unknown child, or narrator voice in the final line.\n"
            "- Do not invent motives, flashbacks, lore, or timeline shifts.\n"
            "- Keep wording compact and TTS-friendly, usually 8-24 words.\n"
            "- previous_line and next_line are only for flow; do not steal facts from neighboring segments.\n"
            "- If the current_line is already the best grounded narration, keep it close and just smooth the phrasing.\n"
            "- Never preserve a repeated environmental fallback line if the local evidence is about a character conflict, battle action, cockpit event, or dialogue exchange.\n"
            "- Never use first-person narration.\n"
            "- Never mention panels, pages, frames, or cameras.\n\n"
            "Examples:\n"
            "- Bad: 'A voice called out, \"Get on!\" as the pilots prepared to board.'\n"
            "  Better: 'The pilots were rushed aboard the transport ship.'\n"
            "- Bad: 'She reassured him, telling him to relax and trust his partner.'\n"
            "  Better: 'She tried to steady him, confident their connection would hold.'\n"
            "- Bad: 'An adult appears, asking if he will pilot with her.'\n"
            "  Better: 'An adult dismisses him, but she still asks him to pilot at her side.'\n\n"
            "Immersive recap contract:\n"
            f"{immersive_recap_contract()}\n"
            f"Project title: {project_title or '(unknown)'}\n"
            f"Chapter metadata: {json.dumps(chapter_metadata, ensure_ascii=False)}\n"
            f"Character dictionary: {json.dumps(character_dictionary, ensure_ascii=False)}\n"
            f"Story bible: {json.dumps(story_bible, ensure_ascii=False)}\n"
            f"Allowed character names: {json.dumps(allowed_character_names, ensure_ascii=False)}\n"
            f"Chapter summary: {chapter_summary or '(none)'}\n\n"
            f"Lines to refine: {json.dumps(payload, ensure_ascii=False)}\n"
        )

    def _story_segment_expansion_prompt(self, lines: list[dict[str, Any]], context: dict[str, Any]) -> str:
        project_title = str(context.get("project_title") or "").strip()
        chapter_summary = str(context.get("chapter_summary") or "").strip()
        chapter_metadata = context.get("chapter_metadata") or {}
        character_dictionary = context.get("character_dictionary") or {}
        story_bible = context.get("story_bible") or {}
        allowed_character_names = context.get("allowed_character_names") or []
        style_vocab_block = self._style_vocabulary_prompt_block(context)
        payload = [
            {
                "index": int(item.get("index") or 0),
                "current_line": str(item.get("current_line") or "").strip(),
                "previous_line": str(item.get("previous_line") or "").strip(),
                "next_line": str(item.get("next_line") or "").strip(),
                "scene_summary": str(item.get("scene_summary") or "").strip(),
                "vision_dialogue": str(item.get("vision_dialogue") or "").strip(),
                "vision_caption": str(item.get("vision_caption") or "").strip(),
                "vision_action_beat": str(item.get("vision_action_beat") or "").strip(),
                "local_evidence": str(item.get("local_evidence") or "").strip(),
                "character_names": [
                    str(name).strip()
                    for name in item.get("character_names", []) or []
                    if str(name).strip()
                ],
                "panel_count": int(item.get("panel_count") or 0),
            }
            for item in lines
        ]
        return (
            "Expand short, blank, or fallback-heavy manga recap story segments into fuller YouTube narration.\n"
            "Return valid JSON only in this format:\n"
            "{\"rewrites\":[{\"index\":0,\"line\":\"...\"}]}\n\n"
            "Critical rules:\n"
            "- Rewrite every supplied index.\n"
            "- Each returned line must stay aligned to the same segment and panel range.\n"
            "- Each non-empty line must be 2-3 complete English sentences; use 3 sentences when panel_count is 3+.\n"
            "- Sentence 1 should preserve or refine current_line when it is useful, or establish the local subject when current_line is blank/generic. Sentence 2 should add the next concrete local beat from vision_dialogue, vision_caption, vision_action_beat, ocr_fallback_text, or local_evidence. Sentence 3 names the consequence, emotional pressure, or transition when panel_count is 3+.\n"
            "- Aim for 35-70 words per line. Do not return a thin one-sentence caption.\n"
            "- Do not invent plot points. Use scene_summary only as context, not as a source to copy.\n"
            "- Use ocr_fallback_text only when the vision fields are too sparse, and never repeat garbled OCR verbatim.\n"
            "- Do not copy generic bridge phrases like 'stays at the centre', 'next exchange takes shape', or 'pressure carries forward'. Replace them with concrete local story movement.\n"
            "- Prefer named characters from the dictionary. If the name is uncertain, use a natural role label.\n"
            "- Never output visual inventory: no camera/panel wording, no clothing/hair descriptions, no 'looks', 'stands', 'visible', 'bright light', or pose descriptions.\n"
            "- If local evidence is dialogue, narrate what the exchange causes: pressure, refusal, warning, decision, accusation, or consequence.\n"
            "- No first-person or second-person narration. No direct quotes. No raw OCR fragments.\n"
            "- Keep each line under about 90 words.\n\n"
            f"{immersive_recap_contract()}"
            f"{style_vocab_block}"
            f"Project title: {project_title or '(unknown)'}\n"
            f"Chapter metadata: {json.dumps(chapter_metadata, ensure_ascii=False)}\n"
            f"Character dictionary: {json.dumps(character_dictionary, ensure_ascii=False)}\n"
            f"Story bible: {json.dumps(story_bible, ensure_ascii=False)}\n"
            f"Allowed character names: {json.dumps(allowed_character_names, ensure_ascii=False)}\n"
            f"Chapter summary: {chapter_summary or '(none)'}\n\n"
            f"Segments to expand: {json.dumps(payload, ensure_ascii=False)}\n"
        )

    def _series_cast_hints_prompt(self, context: dict[str, Any]) -> str:
        project_title = str(context.get("project_title") or "").strip()
        chapter_metadata = context.get("chapter_metadata") or {}
        character_dictionary = context.get("character_dictionary") or {}
        observed_names = context.get("observed_names") or []
        return (
            "Infer weak canonical cast hints for this comic or manga series.\n\n"
            "Return valid JSON only in this format:\n"
            "{\"series_cast_hints\":[\"Character A\",\"Character B\"],\"canonical_name_corrections\":[{\"variant\":\"Alias C\",\"canonical\":\"Character C\"}]}\n\n"
            "Rules:\n"
            "- Your primary job is canonical character-name normalization for this exact title.\n"
            "- Use the project title, synopsis, compact chapter metadata, current character dictionary, and observed names.\n"
            "- Outside series knowledge may be used ONLY to choose likely established spellings or recurring cast names for this exact series.\n"
            "- Do not invent new plot facts, relationships, or obscure minor characters.\n"
            "- Only include names that are highly likely to help disambiguate OCR-like variants during narration.\n"
            "- canonical_name_corrections should contain high-confidence OCR-like variant -> canonical mappings when the title makes the intended cast name clear.\n"
            "- If an observed name looks like a malformed version of an established cast name for this exact series, prefer the correction instead of keeping the malformed variant.\n"
            "- Do not preserve malformed variants like separate cast members when a canonical established name is more likely.\n"
            "- If a variant could refer to multiple characters depending on context, omit the correction rather than guessing.\n"
            "- Prefer a small, very reliable cast list over a larger speculative one.\n"
            "- Keep the output compact: usually 5-20 cast hints and only a few corrections.\n\n"
            f"Project title: {project_title or '(unknown)'}\n"
            f"Chapter metadata: {json.dumps(chapter_metadata, ensure_ascii=False)}\n"
            f"Character dictionary: {json.dumps(character_dictionary, ensure_ascii=False)}\n"
            f"Observed names: {json.dumps(observed_names, ensure_ascii=False)}\n"
        )

    def _story_beats_prompt(self, dialogues: list[dict[str, Any]], context: dict[str, Any]) -> str:
        metadata = context.get("metadata") or {}
        project_title = str(context.get("project_title") or "").strip()
        protagonist_name = str(context.get("protagonist_name") or "").strip()
        character_dictionary = context.get("character_dictionary") or {}
        beat_count = int(context.get("beat_count") or max(5, min(10, len(dialogues) or 1)))
        scene_text = []
        for item in dialogues[: min(len(dialogues), 40)]:
            if item.get("panel") is not None:
                dialogue_lines = item.get("dialogue", []) or []
                dialogue_block = " | ".join(
                    f"{str(entry.get('speaker') or '').strip()}: {str(entry.get('text') or '').strip()}"
                    for entry in dialogue_lines
                    if isinstance(entry, dict) and str(entry.get("text") or "").strip()
                )
                scene_text.append(
                    "\n".join(
                        part
                        for part in [
                            f"PANEL {int(item.get('panel') or 0)}",
                            f"CAPTION: {str(item.get('caption') or '').strip()}",
                            f"DIALOGUE: {dialogue_block}" if dialogue_block else "",
                            f"CHARACTERS: {', '.join(str(name).strip() for name in item.get('character_names', []) or [] if str(name).strip())}" if item.get("character_names") else "",
                        ]
                        if part
                    )
                )
                continue

            characters = ", ".join(str(name).strip() for name in item.get("character_names", []) or [] if str(name).strip())
            scene_text.append(
                "\n".join(
                    part
                    for part in [
                        f"SCENE {int(item.get('scene_id') or 0)}",
                        f"PANELS {int(item.get('panel_start') or 0)}-{int(item.get('panel_end') or 0)}",
                        f"CHARACTERS: {characters}" if characters else "",
                        f"TEXT: {str(item.get('combined_text') or '').strip()}",
                    ]
                    if part
                )
            )
        return (
            "You are writing an English YouTube recap for a manga, manhwa, manhua, or comic chapter.\n\n"
            "Return valid JSON only in this format:\n"
            "{\n"
            '  "story_script": "Full chapter recap in English.",\n'
            '  "beats": [\n'
            '    {"beat_id": 1, "description": "Chronological story beat.", "characters": ["Name"]}\n'
            "  ]\n"
            "}\n\n"
            "Rules:\n"
            f"- Return exactly {beat_count} beats unless the scene list clearly supports fewer.\n"
            "- Write in third-person English.\n"
            "- Keep beats chronological.\n"
            "- Each beat must describe a STORY EVENT (who did what and why), not a visual description of a panel.\n"
            "- Bad: 'A man stands in a room looking worried.' Good: 'Character A discovers the storage room has been emptied overnight.'\n"
            "- Use real character names whenever they are reliable. Never write 'a young man' or 'someone' when a name is available in the character dictionary.\n"
            "- Preserve names from the supplied character dictionary.\n"
            "- Do not treat common OCR phrases as names. Examples: Por Favor, Please, Claro, De Novo, Olá, Sim, Não, Gracias, Obrigado.\n"
            "- Convert dialogue into story events instead of copying OCR fragments.\n"
            "- If a scene has weak or missing text, use surrounding scenes and captions conservatively; do not invent a new event, motive, or timeline label.\n"
            "- Do not call a scene a flashback, dream, memory, past trauma, regression, or rebirth unless that exact timeline idea is directly supported by the supplied text for that scene.\n"
            "- If a character dies and then later wakes up earlier in time, describe the death as happening live first; do not label it as a flashback unless the text explicitly says it is one.\n"
            "- Avoid starting most beat descriptions with He, His, Him, They, or Their.\n"
            "- Avoid filler phrases and repeated openings.\n"
            "- Keep the chapter recap cinematic, concise, and coherent.\n"
            "- Do not produce panel-level narration.\n"
            "- The story_script should read as a self-contained chapter summary that a viewer can follow without seeing the images.\n\n"
            "Immersive recap contract:\n"
            f"{immersive_recap_contract()}\n"
            f"Video project title: {project_title}\n"
            f"Chapter metadata: {json.dumps(metadata, ensure_ascii=False)}\n"
            f"Known character dictionary: {json.dumps(character_dictionary, ensure_ascii=False)}\n"
            f"Protagonist hint: {protagonist_name or 'the protagonist'}\n\n"
            "Ordered scenes:\n"
            + "\n\n".join(scene_text)
        )

    def _character_names_prompt(self, dialogues: list[dict[str, Any]], context: dict[str, Any]) -> str:
        metadata = context.get("metadata") or {}
        protagonist_name = str(context.get("protagonist_name") or "").strip()
        known_names = context.get("character_dictionary") or {}
        cluster_block = "\n\n".join(
            "\n".join(
                part
                for part in [
                    f"CLUSTER {str(item.get('cluster_id') or '').strip()}",
                    f"PANELS: {', '.join(str(panel) for panel in item.get('panels', [])[:12])}",
                    f"ROLE HINT: {str(item.get('role_hint') or '').strip()}" if str(item.get("role_hint") or "").strip() else "",
                    "DIALOGUE:",
                    "\n".join(f"- {str(line).strip()}" for line in item.get("dialogues", [])[:8] if str(line).strip()),
                ]
                if part
            )
            for item in dialogues[:20]
        )
        return (
            "Infer stable character names for recurring character clusters in a comic chapter.\n\n"
            "Return valid JSON only:\n"
            "{\n"
            '  "characters": [\n'
            '    {"cluster": "cluster-1", "name": "Mark"}\n'
            "  ]\n"
            "}\n\n"
            "Rules:\n"
            "- Prefer names already present in metadata or dialogue context.\n"
            "- Keep names consistent.\n"
            "- If a true name is unknown, assign a stable English role label such as Protagonist, Friend, Teacher, Villain, Girl, Boy, Guard, Worker, or Neighbor.\n"
            "- Do not invent a new proper name unless the evidence is strong.\n"
            "- Never treat OCR fragments or ordinary words as names. Bad examples: Kcdikaini Lass, Hose, Nc Jaiv, A Shaft, Start It, Roger, Be Dead, Sauri.\n"
            "- If the evidence is noisy, use a plain role label instead of a fake proper name.\n"
            "- Treat the manga title and chapter metadata as weak canon hints: use them to choose between conflicting spellings or OCR-like variants, not to invent new plot details.\n"
            "- If one candidate looks like OCR noise and another looks like the established canonical name for this series, prefer the canonical name.\n"
            "- Prefer recurring, well-supported names over rare one-off variants.\n"
            "- Preserve known names exactly.\n\n"
            f"Chapter metadata: {json.dumps(metadata, ensure_ascii=False)}\n"
            f"Known character dictionary: {json.dumps(known_names, ensure_ascii=False)}\n"
            f"Protagonist hint: {protagonist_name or 'unknown'}\n\n"
            "Character clusters:\n"
            f"{cluster_block}"
        )

    def _panel_rewrite_prompt(self, panel: dict[str, Any], beat: dict[str, Any], context: dict[str, Any]) -> str:
        subject = str(context.get("subject") or "the protagonist").strip()
        mode = str(context.get("mode") or "balanced").strip()
        current_narration = str(context.get("current_narration") or "").strip()
        previous_line = str(context.get("previous_line") or panel.get("previous_line") or "").strip()
        next_line = str(context.get("next_line") or panel.get("next_line") or "").strip()
        protagonist_name = str(context.get("protagonist_name") or "").strip()
        rescue_seed = str(context.get("rescue_seed") or "").strip()
        supporting_context = str(context.get("supporting_context") or panel.get("supporting_context") or "").strip()
        character_dictionary = context.get("character_dictionary") or {}
        mode_guidance = ""
        if mode == "caption_rescue":
            mode_guidance = (
                "Primary mode guidance: treat visual_caption and rescue_seed as the strongest evidence. "
                "Turn the visible reaction or action into a concrete story-event sentence. "
                "Do not copy the caption verbatim, but stay close to its meaning.\n"
            )
        elif mode == "bridge_context":
            mode_guidance = (
                "Primary mode guidance: OCR is weak here, so use supporting_context, previous_line, next_line, and rescue_seed "
                "to bridge this panel into the surrounding story without vague filler.\n"
            )
        elif mode == "closer_to_ocr":
            mode_guidance = (
                "Primary mode guidance: preserve the strongest factual details from the panel text and dialogue.\n"
            )
        return (
            "Rewrite this panel narration as one cinematic English sentence for a YouTube recap.\n"
            "Avoid starting with He/His/Him/They/Their. Prefer names or descriptive subjects.\n"
            "Use the panel dialogue and beat context to describe the specific event instead of generic summary language.\n"
            "Write in third person only. Never use first-person narration like I, me, my, we, or our.\n"
            "The rewrite must be meaningfully different from the current narration if one is provided.\n"
            "Do not just paraphrase the same weak line.\n"
            "If the panel contains factual details like dates, numbers, causes, or named events, preserve them.\n"
            "If the panel is highly expressive, describe the action or emotional turn rather than listing physical appearance.\n"
            "Use exact character names from the provided dictionary whenever a reliable name is known.\n"
            "Do not treat common OCR phrases as names. Examples: Por Favor, Please, Claro, De Novo, Olá, Sim, Não, Gracias, Obrigado.\n"
            "If OCR is noisy, write a short conservative bridge sentence grounded in the panel text, neighboring lines, and rescue seed instead of vague filler.\n"
            "Do not introduce timeline labels such as flashback, dream, memory, regression, or rebirth unless this panel's text or dialogue directly supports them.\n"
            "If a death or attack is followed later by a wake-up/regression scene, describe the attack as happening live first; do not retroactively call it a memory.\n"
            "When supporting_context is present, treat it as nearby panel evidence that can clarify a weak transition panel.\n"
            "Use previous_line and next_line when they help the sentence flow naturally.\n"
            "Treat the chapter as a continuous story instead of a list of image captions.\n"
            "Narrate the action in natural prose, and fold dialogue into the sentence instead of quoting OCR literally.\n"
            f"Rewrite mode: {mode}.\n"
            f"{mode_guidance}"
            "Immersive recap contract:\n"
            f"{immersive_recap_contract()}\n"
            "Return valid JSON only: {\"narration\": \"...\"}\n\n"
            f"Panel data: {json.dumps(panel, ensure_ascii=False)}\n"
            f"Beat: {json.dumps(beat, ensure_ascii=False)}\n"
            f"Preferred subject: {subject}\n"
            f"Protagonist hint: {protagonist_name or 'unknown'}\n"
            f"Character dictionary: {json.dumps(character_dictionary, ensure_ascii=False)}\n"
            f"Previous line: {previous_line or 'none'}\n"
            f"Next line: {next_line or 'none'}\n"
            f"Supporting context: {supporting_context or 'none'}\n"
            f"Rescue seed: {rescue_seed or 'none'}\n"
            f"Current narration: {current_narration}\n"
        )

    def _panel_batch_rewrite_prompt(self, panels: list[dict[str, Any]], context: dict[str, Any]) -> str:
        project_title = str(context.get("project_title") or "").strip()
        story_hint = str(context.get("story_hint") or "").strip()
        chunk_hint = str(context.get("chunk_hint") or "").strip()
        protagonist_name = str(context.get("protagonist_name") or "").strip()
        overlap_context = str(context.get("overlap_context") or "").strip()
        character_dictionary = context.get("character_dictionary") or {}
        payload = [
            {
                "panel_id": str(panel.get("panel_id") or "").strip(),
                "panel": int(panel.get("panel") or 0),
                "page": int(panel.get("page") or 0),
                "current_narration": str(panel.get("current_narration") or "").strip(),
                "text": str(panel.get("text") or "").strip(),
                "visual_caption": str(panel.get("visual_caption") or "").strip(),
                "dialogue": panel.get("dialogue", []) or [],
                "scene_hint": str(panel.get("scene_hint") or "").strip(),
                "previous_line": str(panel.get("previous_line") or "").strip(),
                "next_line": str(panel.get("next_line") or "").strip(),
                "supporting_context": str(panel.get("supporting_context") or "").strip(),
                "character_names": [
                    str(name).strip()
                    for name in panel.get("character_names", []) or []
                    if str(name).strip()
                ],
            }
            for panel in panels
        ]
        return (
            "Rewrite the weak comic recap panel lines below into stronger English YouTube narration.\n"
            "Return valid JSON only in this format:\n"
            "{\"rewrites\":[{\"panel_id\":\"...\",\"panel\":1,\"narration\":\"...\"}]}\n\n"
            "Rules:\n"
            "- Rewrite every supplied panel.\n"
            "- One sentence per panel.\n"
            "- Write in third-person English only.\n"
            "- Never use first-person narration.\n"
            "- Describe the specific story event happening in that panel, not a generic reaction.\n"
            "- Avoid vague filler such as \"the scene\", \"the moment\", \"the situation\", or \"pressure keeps mounting\" unless absolutely necessary.\n"
            "- Never output OCR garbage, broken translation fragments, or a paraphrase that ignores the strongest panel details.\n"
            "- Use dialogue, OCR fragments, visual caption, and neighboring panel context to describe the specific event.\n"
            "- If the panel contains strong dialogue or exposition, preserve the core meaning of that text.\n"
            "- If OCR is noisy or incomplete, write a conservative bridge from local context; do not invent a new event, motive, or timeline label.\n"
            "- Do not call a panel a flashback, dream, memory, past trauma, regression, or rebirth unless that exact timeline idea is directly supported by that panel's text or dialogue.\n"
            "- If a death or attack is followed later by a wake-up/regression scene, describe the attack as happening live first; do not retroactively call it a memory.\n"
            "- Preserve concrete facts like dates, numbers, distances, causes, named events, and reliable names.\n"
            "- Do not treat common OCR phrases as names. Examples: Por Favor, Please, Claro, De Novo, Olá, Sim, Não, Gracias, Obrigado.\n"
            "- If the panel is mostly visual, write one clean recap line about the story event or reveal, not a camera-style description of what the frame looks like.\n"
            "- Treat visual_caption as the weakest evidence unless OCR, dialogue, and scene_hint are sparse.\n"
            "- Avoid raw visual-description phrasing like hair, clothing, framing, close-ups, facial features, or \"a man stands\" unless that physical detail is the actual plot event.\n"
            "- Make the rewrite meaningfully better than the current narration, not a paraphrase of the same weak line.\n"
            "- Use exact character names from the dictionary whenever they are known. Never default to \"the man\" when a name is available.\n"
            "- Always refer to recurring characters by their dictionary name. Do not switch between a real name and a generic label for the same person.\n"
            "- Use the protagonist name instead of he or she for the first mention in each panel when that identity is known.\n"
            "- Use previous_line and next_line to maintain narrative flow across adjacent panels.\n"
            "- When supporting_context is present, use it to resolve weak OCR bridge panels without drifting into generic filler.\n"
            "- Prefer dialogue, OCR facts, and scene_hint over literal image description whenever those sources conflict.\n"
            "- Ensure the first panel in this batch flows naturally from the previous finalized narration context.\n"
            "- Vary sentence openings. Do not start three consecutive panels the same way.\n\n"
            "Immersive recap contract:\n"
            f"{immersive_recap_contract()}\n"
            f"Project title: {project_title}\n"
            f"Story hint: {story_hint}\n"
            f"Chunk hint: {chunk_hint}\n"
            f"Character dictionary: {json.dumps(character_dictionary, ensure_ascii=False)}\n"
            f"Protagonist hint: {protagonist_name or 'unknown'}\n"
            f"Previous finalized narration context: {overlap_context or 'none'}\n\n"
            f"Panels: {json.dumps(payload, ensure_ascii=False)}\n"
        )

    def _scene_panel_sequence_prompt(self, panels: list[dict[str, Any]], context: dict[str, Any]) -> str:
        scene_summary = str(context.get("scene_summary") or "").strip()
        previous_scene = str(context.get("previous_scene") or "").strip()
        next_scene = str(context.get("next_scene") or "").strip()
        protagonist_name = str(context.get("protagonist_name") or "").strip()
        character_dictionary = context.get("character_dictionary") or {}
        payload = [
            {
                "panel_id": str(panel.get("panel_id") or "").strip(),
                "panel": int(panel.get("panel") or 0),
                "current_narration": str(panel.get("current_narration") or "").strip(),
                "text": str(panel.get("text") or "").strip(),
                "dialogue": panel.get("dialogue", []) or [],
                "character_names": [
                    str(name).strip()
                    for name in panel.get("character_names", []) or []
                    if str(name).strip()
                ],
            }
            for panel in panels
        ]
        return (
            "Rewrite this comic scene into smooth YouTube recap narration, one line per panel.\n"
            "Return valid JSON only in this format:\n"
            "{\"rewrites\":[{\"panel_id\":\"...\",\"panel\":1,\"narration\":\"...\"}]}\n\n"
            "Critical rules:\n"
            "- Return exactly one rewrite for every supplied panel.\n"
            "- Each narration must be a complete, natural English sentence.\n"
            "- Treat panel OCR/dialogue/current_narration as the truth source for each panel.\n"
            "- Use the scene summary only as supporting context for tone and continuity; never let it override local panel evidence.\n"
            "- If scene summary and panel evidence conflict, follow the panel evidence.\n"
            "- Do not write image-caption prose. Never describe camera angle, framing, close-ups, clothes, hair, eyes, facial expressions, or what a person is visibly holding unless it changes the plot.\n"
            "- Do not start with generic subjects like 'A young man', 'Someone', or 'Amidst' if a named character or scene-level subject is known.\n"
            "- Avoid filler such as 'the scene', 'the moment', 'the pressure', 'a brief pause', or 'keeps the focus on what comes next'.\n"
            "- Avoid duplicate lines. If the same event spans multiple panels, advance the wording slightly across the sequence.\n"
            "- Use exact names from the character dictionary. Prefer the protagonist name when the panel is about the main character.\n"
            "- Do not treat common OCR phrases as names. Examples: Por Favor, Please, Claro, De Novo, Olá, Sim, Não, Gracias, Obrigado.\n"
            "- If a panel has weak OCR, write a conservative continuation of the scene instead of a visual description.\n"
            "- Do not call a panel a flashback, dream, memory, past trauma, regression, or rebirth unless that exact timeline idea is directly supported by that panel's text or dialogue.\n"
            "- If a death or attack is followed later by a wake-up/regression scene, describe the attack as happening live first; do not retroactively call it a memory.\n\n"
            "Immersive recap contract:\n"
            f"{immersive_recap_contract()}\n"
            f"Scene summary: {scene_summary}\n"
            f"Previous scene context: {previous_scene or 'none'}\n"
            f"Next scene context: {next_scene or 'none'}\n"
            f"Protagonist hint: {protagonist_name or 'unknown'}\n"
            f"Character dictionary: {json.dumps(character_dictionary, ensure_ascii=False)}\n"
            f"Panels: {json.dumps(payload, ensure_ascii=False)}\n"
        )

    def _spread_scene_panel_beats_prompt(self, panels: list[dict[str, Any]], context: dict[str, Any]) -> str:
        scene_summary = str(context.get("scene_summary") or "").strip()
        protagonist_name = str(context.get("protagonist_name") or "").strip()
        character_dictionary = context.get("character_dictionary") or {}
        existing_narration = context.get("existing_narration") or []
        panel_count = int(context.get("panel_count") or len(panels))
        payload = [
            {
                "panel_id": str(panel.get("panel_id") or "").strip(),
                "panel": int(panel.get("panel") or 0),
                "text": str(panel.get("text") or "").strip(),
                "dialogue": panel.get("dialogue", []) or [],
                "caption": str(panel.get("caption") or "").strip(),
                "character_names": [
                    str(name).strip()
                    for name in panel.get("character_names", []) or []
                    if str(name).strip()
                ],
            }
            for panel in panels
        ]
        existing_hint = (
            "Existing narration already written for other panels in this scene:\n"
            + "\n".join(f"- {line}" for line in existing_narration)
            if existing_narration
            else "No other narration written yet for this scene."
        )
        return (
            f"You are writing {panel_count} distinct recap narration lines for {panel_count} panels "
            "that all belong to the same manga/comic scene.\n"
            "Return valid JSON only in this format:\n"
            "{\"rewrites\":[{\"panel_id\":\"...\",\"panel\":1,\"narration\":\"...\"}]}\n\n"
            "Critical rules:\n"
            f"- Write exactly {panel_count} lines, one per panel.\n"
            "- Each line must cover a DIFFERENT micro-beat or aspect of the scene.\n"
            "- Use this progression when the panels are sequential: introduction, action/development, consequence/reaction.\n"
            "- Never repeat the same sentence or close paraphrase across lines.\n"
            "- Never output the scene summary verbatim.\n"
            "- Use the panel OCR text, dialogue, and caption as primary evidence for each panel.\n"
            "- If a panel has no OCR, write a conservative continuation from the scene context.\n"
            "- Write in third-person English. Never use first-person.\n"
            "- Do not describe camera angles, clothing, hair, or facial expressions.\n"
            "- Use named characters from the character dictionary when reliable. "
            "Never use 'a young man', 'someone nearby', 'a person', or 'the protagonist' when a name is available.\n"
            "- Do not treat OCR phrases as names: Por Favor, Please, Claro, Olá, Gracias, Obrigado.\n"
            "- Do not introduce timeline labels such as flashback, dream, memory, regression, or rebirth unless the panel text directly supports them.\n"
            "- Each line must be a complete, natural English sentence.\n\n"
            "Immersive recap contract:\n"
            f"{immersive_recap_contract()}\n"
            f"Scene summary (context only; do not copy verbatim): {scene_summary}\n"
            f"Protagonist hint: {protagonist_name or 'unknown'}\n"
            f"Character dictionary: {json.dumps(character_dictionary, ensure_ascii=False)}\n"
            f"{existing_hint}\n\n"
            f"Panels: {json.dumps(payload, ensure_ascii=False)}\n"
        )
