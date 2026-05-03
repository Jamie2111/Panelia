from __future__ import annotations

import base64
import hashlib
import json
import logging
import re
import time
from pathlib import Path
from typing import Any

import requests
import yaml

from app.core.config import get_settings
from app.services.dialogue_cleaner import DialogueCleaner
from app.services.panel_mapper import PanelMapper
from app.utils.files import ensure_dir
from app.utils.files import read_json, write_json

logger = logging.getLogger(__name__)


class PanelCaptioner:
    _LEGACY_MODEL_MAP = {
        "gemini-2.0-flash": "gemini-2.5-flash-lite",
        "gemini-2.0-flash-001": "gemini-2.5-flash-lite",
        "gemini-2.0-flash-lite": "gemini-2.5-flash-lite",
        "gemini-2.0-flash-lite-001": "gemini-2.5-flash-lite",
        "gemini-1.5-flash": "gemini-2.5-flash-lite",
    }

    def __init__(self) -> None:
        self.settings = get_settings()
        self.cleaner = DialogueCleaner()
        self.mapper = PanelMapper()
        self._captioning_disabled_reason: str | None = None
        self._global_cache_dir = ensure_dir(self.settings.data_dir / "_panel_caption_cache")
        self._next_gemini_request_at = 0.0
        self._caption_batch_size = max(1, int(self.settings.gemini_panel_caption_batch_size))

    def caption_panels(
        self,
        panels: list[dict[str, Any]],
        scene_lookup: dict[str, dict[str, Any]],
        cache_dir: Path | None,
    ) -> dict[str, str]:
        cache_path = cache_dir / "panel_captions_cache.json" if cache_dir else None
        cache_payload = read_json(cache_path, default={}) if cache_path else {}
        cache_entries = cache_payload.get("entries", {}) if isinstance(cache_payload, dict) else {}

        resolved: dict[str, str] = {}
        pending: list[dict[str, Any]] = []
        for panel in panels:
            panel_id = str(panel.get("panel_id") or "")
            image_path = self._panel_image_path(panel, cache_dir)
            if not panel_id or image_path is None or not image_path.exists():
                continue
            scene = scene_lookup.get(panel_id, {})
            if self._has_strong_text_signal(panel, scene):
                resolved[panel_id] = self._fallback_caption(panel, scene)
                continue
            panel_hash = self._panel_hash(image_path)
            global_cache_path = self._global_cache_dir / f"{panel_hash}.json"
            global_cached = read_json(global_cache_path, default=None)
            if isinstance(global_cached, dict) and str(global_cached.get("caption") or "").strip():
                resolved[panel_id] = str(global_cached.get("caption") or "").strip()
                continue
            cached = cache_entries.get(panel_id)
            if isinstance(cached, dict) and cached.get("hash") == panel_hash and str(cached.get("caption") or "").strip():
                resolved[panel_id] = str(cached.get("caption") or "").strip()
                continue
            pending.append({"panel": panel, "scene": scene, "image_path": image_path, "hash": panel_hash, "global_cache_path": global_cache_path})

        if pending:
            if self.settings.gemini_api_key and not self._captioning_disabled_reason:
                for chunk in self._chunks(pending, self._caption_batch_size):
                    if self._captioning_disabled_reason:
                        break
                    chunk_resolved = self._caption_chunk(chunk)
                    for panel_id, caption in chunk_resolved.items():
                        cleaned_caption = str(caption or "").strip()
                        if cleaned_caption:
                            resolved[panel_id] = cleaned_caption

                unresolved = [
                    item
                    for item in pending
                    if not str(resolved.get(str(item["panel"].get("panel_id") or "")) or "").strip()
                ]
                if not self._captioning_disabled_reason:
                    for item in unresolved:
                        panel_id = str(item["panel"].get("panel_id") or "")
                        single_caption = self._caption_single(item)
                        if single_caption:
                            resolved[panel_id] = single_caption

            for item in pending:
                panel_id = str(item["panel"].get("panel_id") or "")
                if str(resolved.get(panel_id) or "").strip():
                    continue
                fallback_caption = self._fallback_caption(item["panel"], item["scene"])
                if fallback_caption:
                    resolved[panel_id] = fallback_caption

        if cache_path:
            updated_entries = dict(cache_entries)
            for item in pending:
                panel_id = str(item["panel"].get("panel_id") or "")
                caption = str(resolved.get(panel_id) or "").strip()
                if not panel_id or not caption:
                    continue
                updated_entries[panel_id] = {"hash": item["hash"], "caption": caption}
                write_json(item["global_cache_path"], {"hash": item["hash"], "caption": caption})
            write_json(cache_path, {"entries": updated_entries})
        return resolved

    def _has_strong_text_signal(self, panel: dict[str, Any], scene: dict[str, Any]) -> bool:
        text = self.cleaner.clean_text(
            str(panel.get("text") or panel.get("gemini_text") or scene.get("detected_text") or "")
        )
        if not self.cleaner.is_usable(text):
            return False
        words = re.findall(r"[A-Za-z0-9\u00C0-\u024F\u1E00-\u1EFF\u3040-\u30ff\u4e00-\u9fff\uac00-\ud7af']+", text)
        return len(words) >= 8 or len(text) >= 48

    def _caption_chunk(self, chunk: list[dict[str, Any]]) -> dict[str, str]:
        parts: list[dict[str, Any]] = [{"text": self._caption_prompt(chunk)}]
        ordered_ids: list[str] = []
        for item in chunk:
            panel = item["panel"]
            panel_id = str(panel.get("panel_id") or "")
            ordered_ids.append(panel_id)
            image_path = item["image_path"]
            mime_type = "image/png" if image_path.suffix.lower() == ".png" else "image/jpeg"
            parts.append({"text": f"Panel {int(panel.get('panel') or 0)}"})
            parts.append({"inlineData": {"mimeType": mime_type, "data": base64.b64encode(image_path.read_bytes()).decode("utf-8")}})

        try:
            payload = self._request_gemini_payload(parts)
            parsed = self._parse_json(self._extract_text(payload))
            items = parsed.get("captions") if isinstance(parsed, dict) else None
            if not isinstance(items, list):
                raise ValueError("Gemini panel caption payload was missing captions")
            by_number: dict[int, str] = {}
            ordered_captions: list[str] = []
            for item in items:
                if not isinstance(item, dict):
                    continue
                caption = self.cleaner.normalize_dialogue(str(item.get("caption") or "").strip())
                if not caption or self._is_weak_caption(caption):
                    continue
                ordered_captions.append(caption)
                match = re.search(r"\d+", str(item.get("panel") or ""))
                if not match:
                    continue
                number = int(match.group(0))
                by_number[number] = caption
            resolved: dict[str, str] = {}
            unresolved_indexes: list[int] = []
            for index, item in enumerate(chunk):
                panel = item["panel"]
                panel_id = str(panel.get("panel_id") or "")
                panel_number = int(panel.get("panel") or 0)
                caption = by_number.get(panel_number)
                if caption:
                    resolved[panel_id] = caption
                else:
                    unresolved_indexes.append(index)
            if ordered_captions:
                if len(chunk) == 1:
                    panel_id = str(chunk[0]["panel"].get("panel_id") or "")
                    if panel_id and panel_id not in resolved:
                        resolved[panel_id] = ordered_captions[0]
                elif len(ordered_captions) == len(chunk):
                    for index in unresolved_indexes:
                        panel_id = str(chunk[index]["panel"].get("panel_id") or "")
                        if panel_id and panel_id not in resolved:
                            resolved[panel_id] = ordered_captions[index]
            return resolved
        except Exception as exc:
            if isinstance(exc, requests.HTTPError) and exc.response is not None and exc.response.status_code == 429:
                self._captioning_disabled_reason = "Gemini panel captioning is currently rate-limited."
            logger.warning("Gemini panel captioning fell back to local captions: %s", exc)
            return {}

    def _caption_single(self, item: dict[str, Any]) -> str:
        panel = item.get("panel", {})
        panel_id = str(panel.get("panel_id") or "")
        if not panel_id:
            return ""
        return str(self._caption_chunk([item]).get(panel_id) or "").strip()

    def _caption_prompt(self, chunk: list[dict[str, Any]]) -> str:
        return (
            "Describe what happens in each comic panel in one English sentence.\n"
            "Focus on characters, actions, environment, and emotion.\n"
            "Do not quote dialogue. Do not mention OCR. Keep each caption visual.\n"
            "If a panel has little or no text, still describe the visible action or mood.\n"
            "Do not leave any requested panel uncaptained if visible content exists.\n"
            "Return valid JSON only in this format:\n"
            '{"captions":[{"panel":1,"caption":"..."}]}\n'
        )

    def _fallback_caption(self, panel: dict[str, Any], scene: dict[str, Any]) -> str:
        text = str(panel.get("text") or panel.get("gemini_text") or scene.get("detected_text") or "").strip()
        if not text:
            text = str(
                scene.get("caption_hint")
                or scene.get("story_summary_hint")
                or scene.get("story_description_hint")
                or ""
            ).strip()
        line = self.mapper._panel_specific_line(text, "", 0, 1)
        cleaned = self.cleaner.normalize_dialogue(line)
        return "" if self._is_weak_caption(cleaned) else cleaned

    def _is_weak_caption(self, caption: str) -> bool:
        lowered = str(caption or "").strip().casefold()
        if not lowered:
            return True
        weak_phrases = (
            "the situation shifts again",
            "the protagonist finally puts a name and history",
            "a grim narration lays out",
            "the scale of the catastrophe becomes clearer",
            "the atmosphere stays strained",
            "the scene holds in a tense pause",
            "the confrontation stalls",
            "one pointed question makes it clear",
            "the sudden turn leaves",
            "another uneasy beat passes",
            "the next beat makes it clear",
            "the people around him misunderstand",
            "is shown against",
            "is shown with",
            "are displayed against",
            "stand together",
            "looking thoughtful",
            "glowing object",
            "white background",
            "black background",
            "dimly lit room",
            "smiling man and woman",
            "silhouette stands",
        )
        if any(phrase in lowered for phrase in weak_phrases):
            return True
        if len(re.findall(r"[a-zA-Z\u00C0-\u024F']{4,}", lowered)) < 4:
            return True
        return False

    def _request_gemini_payload(self, parts: list[dict[str, Any]]) -> dict[str, Any]:
        last_exc: Exception | None = None
        for model in self._gemini_models():
            for attempt in range(4):
                wait_for = max(0.0, self._next_gemini_request_at - time.monotonic())
                if wait_for:
                    time.sleep(min(wait_for, 30.0))
                try:
                    response = requests.post(
                        f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
                        params={"key": self.settings.gemini_api_key},
                        headers={"Content-Type": "application/json"},
                        json={
                            "contents": [{"parts": parts}],
                            "generationConfig": {
                                "temperature": 0.2,
                                "maxOutputTokens": 1600,
                                "responseMimeType": "application/json",
                            },
                        },
                        timeout=90,
                    )
                    if response.status_code == 429:
                        retry_after = self._retry_delay_from_response(response) or min(20 * (attempt + 1), 120)
                        self._next_gemini_request_at = time.monotonic() + retry_after
                        last_exc = requests.HTTPError(f"429 rate-limited for {retry_after}s", response=response)
                        logger.info("Gemini panel captioning hit a 429; retrying in %ss", retry_after)
                        if attempt < 3:
                            continue
                    response.raise_for_status()
                    self._next_gemini_request_at = time.monotonic() + 1.5
                    return response.json()
                except requests.HTTPError as exc:
                    last_exc = exc
                    status_code = exc.response.status_code if exc.response is not None else None
                    if status_code == 404:
                        break
                    if status_code == 429 and attempt < 3:
                        continue
                    raise
        if last_exc is not None:
            raise last_exc
        raise ValueError("No usable Gemini captioning model was available")

    def _retry_delay_from_response(self, response: requests.Response) -> int | None:
        retry_after = str(response.headers.get("Retry-After") or "").strip()
        if retry_after:
            try:
                return max(int(float(retry_after)), 1)
            except ValueError:
                return None
        try:
            payload = response.json()
        except ValueError:
            return None
        details = payload.get("error", {}).get("details", [])
        for detail in details if isinstance(details, list) else []:
            if not isinstance(detail, dict):
                continue
            retry_delay = str(detail.get("retryDelay") or "").strip()
            match = re.fullmatch(r"(?P<seconds>\d+)(?:\.(?P<fraction>\d+))?s", retry_delay)
            if match:
                return max(int(match.group("seconds")), 1)
        return None

    def _panel_image_path(self, panel: dict[str, Any], cache_dir: Path | None) -> Path | None:
        if cache_dir is None:
            return None
        project_dir = cache_dir.parent
        return project_dir / "panels" / f"panel_{int(panel.get('panel') or 0):03d}.png"

    def _panel_hash(self, image_path: Path) -> str:
        digest = hashlib.sha1()
        digest.update(image_path.read_bytes())
        return digest.hexdigest()

    def _chunks(self, items: list[dict[str, Any]], chunk_size: int) -> list[list[dict[str, Any]]]:
        return [items[index : index + chunk_size] for index in range(0, len(items), chunk_size)]

    def _gemini_models(self) -> list[str]:
        candidates = [
            self._resolve_model_name(str(self.settings.llm_gemini_model or "").strip()),
            self._resolve_model_name(str(self.settings.gemini_model or "").strip()),
            "gemini-2.5-flash-lite",
            "gemini-2.5-flash",
        ]
        ordered: list[str] = []
        for model in candidates:
            if model and model not in ordered:
                ordered.append(model)
        return ordered

    def _resolve_model_name(self, model_name: str) -> str:
        stripped = model_name.strip()
        return self._LEGACY_MODEL_MAP.get(stripped, stripped or "gemini-2.5-flash-lite")

    def _extract_text(self, payload: dict[str, Any]) -> str:
        candidates = payload.get("candidates", [])
        if not candidates:
            raise ValueError("Gemini captioning returned no candidates")
        parts = candidates[0].get("content", {}).get("parts", [])
        text = "\n".join(str(part.get("text") or "").strip() for part in parts if isinstance(part, dict))
        if not text.strip():
            raise ValueError("Gemini captioning returned empty content")
        return text.strip()

    def _parse_json(self, raw_text: str) -> Any:
        cleaned = raw_text.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
            cleaned = re.sub(r"\s*```$", "", cleaned)
        candidates = [cleaned]
        extracted = self._extract_first_json_candidate(cleaned)
        if extracted and extracted not in candidates:
            candidates.append(extracted)
        repaired = self._repair_common_json_issues(cleaned)
        if repaired and repaired not in candidates:
            candidates.append(repaired)
        if extracted:
            repaired_extracted = self._repair_common_json_issues(extracted)
            if repaired_extracted and repaired_extracted not in candidates:
                candidates.append(repaired_extracted)
        last_exc: Exception | None = None
        for candidate in candidates:
            if not candidate:
                continue
            try:
                return json.loads(candidate)
            except json.JSONDecodeError as exc:
                last_exc = exc
                yaml_decoded = self._yaml_json_fallback(candidate)
                if yaml_decoded is not None:
                    return yaml_decoded
        if last_exc is not None:
            raise last_exc
        raise ValueError("Gemini captioning response did not contain valid JSON")

    def _extract_first_json_candidate(self, text: str) -> str:
        start = None
        stack: list[str] = []
        in_string = False
        escape = False
        for index, char in enumerate(text):
            if in_string:
                if escape:
                    escape = False
                    continue
                if char == "\\":
                    escape = True
                    continue
                if char == '"':
                    in_string = False
                continue
            if char == '"':
                in_string = True
                continue
            if char in "{[":
                if start is None:
                    start = index
                stack.append("}" if char == "{" else "]")
                continue
            if char in "}]":
                if not stack:
                    continue
                expected = stack.pop()
                if char != expected:
                    continue
                if not stack and start is not None:
                    return text[start : index + 1]
        return ""

    def _yaml_json_fallback(self, text: str) -> Any | None:
        try:
            parsed = yaml.safe_load(text)
        except Exception:
            return None
        return parsed if isinstance(parsed, (dict, list)) else None

    def _repair_common_json_issues(self, text: str) -> str:
        repaired = text.strip()
        if not repaired:
            return repaired
        repaired = re.sub(r",(\s*[}\]])", r"\1", repaired)
        repaired = re.sub(r'(?<=": ")(.*?)(?=")', lambda m: m.group(0).replace("\n", " "), repaired)
        repaired = re.sub(r"(?<=[\[{,:\s])'([^']*?)'(?=[\s,\]}:])", r'"\1"', repaired)
        stripped = repaired.rstrip()
        open_braces = stripped.count("{") - stripped.count("}")
        open_brackets = stripped.count("[") - stripped.count("]")
        if open_braces > 0 or open_brackets > 0:
            repaired = stripped + "]" * max(0, open_brackets) + "}" * max(0, open_braces)
        lines = repaired.splitlines()
        if len(lines) <= 1:
            return repaired
        repaired_lines: list[str] = []
        previous_significant_index: int | None = None
        for line in lines:
            current = line.rstrip()
            repaired_lines.append(current)
            stripped_current = current.strip()
            if not stripped_current:
                continue
            if previous_significant_index is not None and not stripped_current.startswith(("}", "]", ",")):
                previous = repaired_lines[previous_significant_index].rstrip()
                if previous and not previous.endswith(("{", "[", ",", ":")):
                    repaired_lines[previous_significant_index] = previous + ","
            previous_significant_index = len(repaired_lines) - 1
        return "\n".join(repaired_lines)
