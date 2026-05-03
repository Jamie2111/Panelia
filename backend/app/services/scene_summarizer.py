from __future__ import annotations

import json
import re
from typing import Any, Callable

from app.schemas.project import ChapterMetadata


class SceneSummarizer:
    def __init__(self, request_fn: Callable[[str, int, str], Any], prompt_template: str) -> None:
        self._request = request_fn
        self._prompt_template = prompt_template

    def summarize(
        self,
        metadata: ChapterMetadata,
        project_title: str | None,
        scenes: list[dict[str, Any]],
    ) -> tuple[str, list[dict[str, Any]]]:
        prompt = self.build_prompt(metadata, project_title, scenes)
        response = self._request(prompt, 900, "application/json")
        payload = response.json()
        raw_text = self._extract_response_text(payload).strip()
        parsed = self._parse_json_response(raw_text)
        if not isinstance(parsed, dict):
            raise ValueError("Gemini returned an unexpected scene summary payload")

        chapter_summary = str(parsed.get("chapter_summary") or parsed.get("story_script") or parsed.get("summary") or "").strip()
        if not chapter_summary:
            raise ValueError("Gemini returned an empty chapter summary")

        raw_scene_summaries = parsed.get("scenes") or parsed.get("scene_summaries")
        if not isinstance(raw_scene_summaries, list):
            raise ValueError("Gemini returned no scene summaries")

        summaries: list[dict[str, Any]] = []
        for index, item in enumerate(raw_scene_summaries[:10], start=1):
            if not isinstance(item, dict):
                continue
            scene_id = self._coerce_number(item.get("scene_id")) or index
            description = self._normalize_text(str(item.get("description") or "").strip())
            summary = self._normalize_text(str(item.get("summary") or item.get("narration") or "").strip())
            if not summary:
                continue
            summaries.append(
                {
                    "scene_id": scene_id,
                    "description": description,
                    "summary": summary,
                }
            )
        if not summaries:
            raise ValueError("Gemini returned no usable scene summaries")
        return self._normalize_text(chapter_summary), summaries

    def build_prompt(
        self,
        metadata: ChapterMetadata,
        project_title: str | None,
        scenes: list[dict[str, Any]],
    ) -> str:
        rendered = self._prompt_template
        limited_scenes = scenes[:10]
        scene_budget = max(160, min(320, 2600 // max(len(limited_scenes), 1)))
        scene_text_block = "\n\n".join(
            (
                f"SCENE {int(scene['scene_id'])}\n"
                f"PANELS {int(scene['panel_start'])}-{int(scene['panel_end'])}\n"
                f"TEXT:\n{str(scene['combined_text']).strip()[:scene_budget]}"
            )
            for scene in limited_scenes
        )
        chapter_metadata = {
            "manga_title": metadata.manga_title,
            "chapter_title": metadata.chapter_title,
            "chapter_number": metadata.chapter_number,
            "volume_number": metadata.volume_number,
            "language": metadata.language,
        }
        replacements = {
            "project_title_context": project_title or "",
            "chapter_metadata": json.dumps(chapter_metadata, ensure_ascii=False, separators=(",", ":")),
            "scene_text_block": scene_text_block,
            "scene_count": str(min(len(limited_scenes), 10)),
        }
        for key, value in replacements.items():
            rendered = rendered.replace(f"{{{key}}}", value)
        return rendered

    def _extract_response_text(self, payload: dict[str, Any]) -> str:
        candidates = payload.get("candidates", [])
        if not candidates:
            raise ValueError("Gemini returned no candidates")
        parts = candidates[0].get("content", {}).get("parts", [])
        text_segments = [part.get("text", "") for part in parts if isinstance(part, dict) and part.get("text")]
        text = "\n".join(segment.strip() for segment in text_segments if segment.strip()).strip()
        if not text:
            raise ValueError("Gemini returned empty content")
        return text

    def _parse_json_response(self, raw_text: str) -> Any:
        cleaned = raw_text
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
            cleaned = re.sub(r"\s*```$", "", cleaned)
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            match = re.search(r"(\{.*\}|\[.*\])", cleaned, re.DOTALL)
            if not match:
                raise ValueError("Gemini returned invalid JSON")
            return json.loads(match.group(1))

    def _normalize_text(self, text: str) -> str:
        cleaned = re.sub(r"\s+", " ", str(text or "")).strip()
        cleaned = re.sub(r"\s+([,.;:!?])", r"\1", cleaned)
        return cleaned

    def _coerce_number(self, value: Any) -> int | None:
        if isinstance(value, int):
            return value
        match = re.search(r"\d+", str(value or ""))
        return int(match.group(0)) if match else None
