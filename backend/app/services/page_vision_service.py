"""Page-level Gemini Vision analysis — inspired by pashpashpash/manga-reader.

Sends batches of full manga pages (as images) to Gemini Vision, asking for
story-event summaries per page.  The results fill the gap left by OCR failures
and give the narration pipeline rich story context even for panels that have
zero extracted text.
"""

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
from app.utils.files import ensure_dir, read_json, write_json

logger = logging.getLogger(__name__)

_PROMPT_PATH = Path(__file__).resolve().parents[3] / "services" / "prompts" / "gemini-page-vision.md"

_LEGACY_MODEL_MAP = {
    "gemini-2.0-flash": "gemini-2.5-flash-lite",
    "gemini-2.0-flash-001": "gemini-2.5-flash-lite",
    "gemini-2.0-flash-lite": "gemini-2.5-flash-lite",
    "gemini-2.0-flash-lite-001": "gemini-2.5-flash-lite",
    "gemini-1.5-flash": "gemini-2.5-flash-lite",
}


class PageVisionService:
    """Analyse full manga pages via Gemini Vision for story understanding."""

    # How many pages to send in a single Gemini request.
    # Each page image is ~200-600 KB base64 — batches of 5 stay comfortably
    # under the 4 MB inline-data limit for gemini-2.5-flash.
    DEFAULT_BATCH_SIZE = 5
    # Overlap between batches so the model can connect story events.
    DEFAULT_OVERLAP = 1

    def __init__(self) -> None:
        self.settings = get_settings()
        self._next_request_at = 0.0
        self._disabled_reason: str | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyse_pages(
        self,
        page_paths: list[Path],
        *,
        cache_dir: Path | None = None,
        chapter_context: str = "",
        character_context: str = "",
        batch_size: int | None = None,
        overlap: int | None = None,
    ) -> dict[int, str]:
        """Return {page_number: story_events} for every supplied page.

        Pages are numbered 1-based matching the file names (0001.png → page 1).
        Results are cached in ``cache_dir / page_vision_cache.json``.
        """
        if not self.settings.gemini_api_key:
            logger.info("Gemini API key not configured — skipping page vision analysis")
            return {}
        if not page_paths:
            return {}

        batch_size = max(2, batch_size or self.DEFAULT_BATCH_SIZE)
        overlap = max(0, min(overlap or self.DEFAULT_OVERLAP, batch_size - 1))

        # Load cache -------------------------------------------------------
        cache_path = cache_dir / "page_vision_cache.json" if cache_dir else None
        cached_entries: dict[str, dict[str, str]] = {}
        if cache_path and cache_path.exists():
            try:
                cached_entries = read_json(cache_path, default={}).get("entries", {})
            except Exception:
                logger.warning("Could not read page_vision_cache.json, starting fresh")

        # Compute hashes & identify pending pages --------------------------
        page_items: list[dict[str, Any]] = []
        for path in page_paths:
            page_number = self._page_number(path)
            if page_number is None:
                continue
            page_hash = self._file_hash(path)
            cached = cached_entries.get(str(page_number))
            if isinstance(cached, dict) and cached.get("hash") == page_hash and str(cached.get("events") or "").strip():
                page_items.append({
                    "page": page_number,
                    "path": path,
                    "hash": page_hash,
                    "events": str(cached["events"]).strip(),
                    "cached": True,
                })
            else:
                page_items.append({
                    "page": page_number,
                    "path": path,
                    "hash": page_hash,
                    "events": "",
                    "cached": False,
                })

        page_items.sort(key=lambda item: item["page"])

        # Build batches of pending pages -----------------------------------
        pending_indexes = [i for i, item in enumerate(page_items) if not item["cached"]]
        if pending_indexes and not self._disabled_reason:
            batches = self._build_batches(page_items, pending_indexes, batch_size, overlap)
            for batch in batches:
                if self._disabled_reason:
                    break
                self._process_batch(batch, chapter_context, character_context)

        # Persist cache & build result -------------------------------------
        if cache_path:
            updated = dict(cached_entries)
            for item in page_items:
                events = str(item.get("events") or "").strip()
                if events:
                    updated[str(item["page"])] = {"hash": item["hash"], "events": events}
            write_json(cache_path, {"entries": updated})

        return {item["page"]: str(item["events"]).strip() for item in page_items if str(item.get("events") or "").strip()}

    # ------------------------------------------------------------------
    # Batch construction
    # ------------------------------------------------------------------

    def _build_batches(
        self,
        page_items: list[dict[str, Any]],
        pending_indexes: list[int],
        batch_size: int,
        overlap: int,
    ) -> list[list[dict[str, Any]]]:
        """Group pending pages into overlapping batches.

        The overlap pages provide story continuity context at batch boundaries.
        Overlap pages that are already cached won't be re-requested but are
        included in the batch images so the model can reference them.
        """
        if not pending_indexes:
            return []

        # Build contiguous runs from pending indexes
        runs: list[list[int]] = []
        current_run: list[int] = [pending_indexes[0]]
        for idx in pending_indexes[1:]:
            if idx == current_run[-1] + 1:
                current_run.append(idx)
            else:
                runs.append(current_run)
                current_run = [idx]
        runs.append(current_run)

        batches: list[list[dict[str, Any]]] = []
        for run in runs:
            # Expand the run to include overlap context
            start = max(0, run[0] - overlap)
            end = min(len(page_items), run[-1] + 1 + overlap)
            expanded = page_items[start:end]

            # Chunk into batch_size groups
            for chunk_start in range(0, len(expanded), batch_size - overlap):
                chunk_end = min(chunk_start + batch_size, len(expanded))
                batch = expanded[chunk_start:chunk_end]
                # Only include if batch has at least one pending page
                if any(not item["cached"] for item in batch):
                    batches.append(batch)
                if chunk_end >= len(expanded):
                    break

        return batches

    # ------------------------------------------------------------------
    # Gemini Vision request
    # ------------------------------------------------------------------

    def _process_batch(
        self,
        batch: list[dict[str, Any]],
        chapter_context: str,
        character_context: str,
    ) -> None:
        """Send a batch of page images to Gemini Vision and store results."""
        page_numbers = [item["page"] for item in batch]
        prompt = self._build_prompt(
            page_count=len(batch),
            page_numbers=page_numbers,
            chapter_context=chapter_context,
            character_context=character_context,
        )

        parts: list[dict[str, Any]] = [{"text": prompt}]
        for item in batch:
            path = item["path"]
            mime = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
            parts.append({"text": f"Page {item['page']}"})
            parts.append({
                "inlineData": {
                    "mimeType": mime,
                    "data": base64.b64encode(path.read_bytes()).decode("utf-8"),
                },
            })

        try:
            payload = self._gemini_request(parts, max_tokens=max(800, 120 * len(batch)))
            text = self._extract_text(payload)
            parsed = self._parse_json(text)
            items = parsed.get("page_events") if isinstance(parsed, dict) else None
            if not isinstance(items, list):
                logger.warning("Page vision response missing page_events array")
                return

            # Map results back by page number
            by_page: dict[int, str] = {}
            ordered_events: list[str] = []
            for item_data in items:
                if not isinstance(item_data, dict):
                    continue
                events = str(item_data.get("events") or "").strip()
                if not events:
                    continue
                ordered_events.append(events)
                page_match = re.search(r"\d+", str(item_data.get("page") or ""))
                if page_match:
                    by_page[int(page_match.group(0))] = events

            # Store events in batch items
            for idx, item in enumerate(batch):
                page_num = item["page"]
                if page_num in by_page:
                    item["events"] = by_page[page_num]
                elif idx < len(ordered_events):
                    # Fall back to positional matching
                    item["events"] = ordered_events[idx]

            logger.info(
                "Page vision batch analysed pages %s — got %d/%d events",
                page_numbers, len(by_page), len(batch),
            )
        except Exception as exc:
            if isinstance(exc, requests.HTTPError) and exc.response is not None and exc.response.status_code == 429:
                self._disabled_reason = "Page vision rate-limited"
            logger.warning("Page vision batch failed for pages %s: %s", page_numbers, exc)

    # ------------------------------------------------------------------
    # Prompt building
    # ------------------------------------------------------------------

    def _build_prompt(
        self,
        page_count: int,
        page_numbers: list[int],
        chapter_context: str,
        character_context: str,
    ) -> str:
        try:
            template = _PROMPT_PATH.read_text(encoding="utf-8")
        except FileNotFoundError:
            template = (
                "Describe the story events on each manga page in 1-3 sentences.\n"
                "Return JSON: {{\"page_events\":[{{\"page\":1,\"events\":\"...\"}}]}}\n"
                "Character context: {character_context}\n"
                "Chapter context: {chapter_context}\n"
                "Page count: {page_count}\n"
            )
        return template.format(
            character_context=character_context or "No character context available.",
            chapter_context=chapter_context or "No chapter context available.",
            page_count=page_count,
        )

    # ------------------------------------------------------------------
    # Gemini HTTP helpers (mirrors panel_captioner.py pattern)
    # ------------------------------------------------------------------

    def _gemini_request(self, parts: list[dict[str, Any]], max_tokens: int = 2000) -> dict[str, Any]:
        last_exc: Exception | None = None
        for model in self._model_candidates():
            for attempt in range(4):
                wait = max(0.0, self._next_request_at - time.monotonic())
                if wait:
                    time.sleep(min(wait, 30.0))
                try:
                    resp = requests.post(
                        f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
                        params={"key": self.settings.gemini_api_key},
                        headers={"Content-Type": "application/json"},
                        json={
                            "contents": [{"parts": parts}],
                            "generationConfig": {
                                "temperature": 0.3,
                                "maxOutputTokens": max_tokens,
                                "responseMimeType": "application/json",
                            },
                        },
                        timeout=120,
                    )
                    if resp.status_code == 429:
                        retry_after = self._retry_delay(resp) or min(20 * (attempt + 1), 120)
                        self._next_request_at = time.monotonic() + retry_after
                        last_exc = requests.HTTPError(f"429 rate-limited for {retry_after}s", response=resp)
                        logger.info("Page vision hit 429; retrying in %ss", retry_after)
                        if attempt < 3:
                            continue
                    resp.raise_for_status()
                    self._next_request_at = time.monotonic() + 2.0
                    return resp.json()
                except requests.HTTPError as exc:
                    last_exc = exc
                    status = exc.response.status_code if exc.response is not None else None
                    if status == 404:
                        break  # try next model
                    if status == 429 and attempt < 3:
                        continue
                    raise
        if last_exc is not None:
            raise last_exc
        raise ValueError("No usable Gemini model available for page vision")

    def _model_candidates(self) -> list[str]:
        candidates = [
            self._resolve_model(str(self.settings.llm_gemini_model or "").strip()),
            self._resolve_model(str(self.settings.gemini_model or "").strip()),
            "gemini-2.5-flash-lite",
            "gemini-2.5-flash",
        ]
        seen: list[str] = []
        for m in candidates:
            if m and m not in seen:
                seen.append(m)
        return seen

    def _resolve_model(self, name: str) -> str:
        return _LEGACY_MODEL_MAP.get(name.strip(), name.strip() or "gemini-2.5-flash-lite")

    def _retry_delay(self, resp: requests.Response) -> int | None:
        header = str(resp.headers.get("Retry-After") or "").strip()
        if header:
            try:
                return max(int(float(header)), 1)
            except ValueError:
                pass
        try:
            details = resp.json().get("error", {}).get("details", [])
        except ValueError:
            return None
        for d in (details if isinstance(details, list) else []):
            if not isinstance(d, dict):
                continue
            m = re.fullmatch(r"(?P<s>\d+)(?:\.\d+)?s", str(d.get("retryDelay") or ""))
            if m:
                return max(int(m.group("s")), 1)
        return None

    def _extract_text(self, payload: dict[str, Any]) -> str:
        candidates = payload.get("candidates", [])
        if not candidates:
            raise ValueError("Gemini page vision returned no candidates")
        parts = candidates[0].get("content", {}).get("parts", [])
        text = "\n".join(str(p.get("text") or "").strip() for p in parts if isinstance(p, dict))
        if not text.strip():
            raise ValueError("Gemini page vision returned empty content")
        return text.strip()

    def _parse_json(self, raw: str) -> Any:
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
            cleaned = re.sub(r"\s*```$", "", cleaned)
        # Try direct parse
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass
        # Try extracting JSON object
        extracted = self._extract_json(cleaned)
        if extracted:
            try:
                return json.loads(extracted)
            except json.JSONDecodeError:
                pass
        # YAML fallback
        try:
            parsed = yaml.safe_load(cleaned)
            if isinstance(parsed, (dict, list)):
                return parsed
        except Exception:
            pass
        raise ValueError("Page vision response did not contain valid JSON")

    def _extract_json(self, text: str) -> str:
        start = None
        stack: list[str] = []
        in_str = False
        esc = False
        for i, c in enumerate(text):
            if in_str:
                if esc:
                    esc = False
                    continue
                if c == "\\":
                    esc = True
                    continue
                if c == '"':
                    in_str = False
                continue
            if c == '"':
                in_str = True
                continue
            if c in "{[":
                if start is None:
                    start = i
                stack.append("}" if c == "{" else "]")
            elif c in "}]":
                if stack:
                    stack.pop()
                    if not stack and start is not None:
                        return text[start:i + 1]
        return ""

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _page_number(path: Path) -> int | None:
        match = re.search(r"(\d+)", path.stem)
        return int(match.group(1)) if match else None

    @staticmethod
    def _file_hash(path: Path) -> str:
        h = hashlib.sha1()
        h.update(path.read_bytes())
        return h.hexdigest()
