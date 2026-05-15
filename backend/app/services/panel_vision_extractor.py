from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
import json
import logging
import os
import re
from pathlib import Path
from typing import Any
from multiprocessing import cpu_count

from app.core.config import get_settings
from app.pipeline.image_loader import ImageLoader
from app.schemas.project import CanonicalCharacterRecord, ChapterMetadata, PanelBox, PanelVisionRecord
from app.services.character_name_filters import is_valid_character_name_candidate, normalize_name_key
from app.services.llm_router import LLMRouter
from app.services.panel_evidence_extractor import panel_evidence_by_id
from app.utils.files import ensure_dir, read_json, write_json

logger = logging.getLogger(__name__)

_PROMPT_VERSION = "panel_vision_v3_english_grounded"


def story_bible_canonical_fallback(project_dir: Path) -> list[CanonicalCharacterRecord]:
    """Build a conservative roster from an existing story bible.

    This is a fallback for projects where the page portrait pass is blocked or
    empty, but a previous narration run already established a stable cast. It
    intentionally reads only canonical-looking cast entries, never OCR tokens.
    """
    payload = read_json(project_dir / "output" / "story_bible.json", default={})
    if not isinstance(payload, dict):
        return []
    records: list[CanonicalCharacterRecord] = []
    seen: set[str] = set()
    for item in payload.get("cast") or []:
        if isinstance(item, dict):
            name = str(item.get("name") or item.get("display_name") or item.get("canonical_name") or "").strip()
            aliases = [str(alias).strip() for alias in item.get("aliases") or [] if str(alias).strip()]
            role = str(item.get("role") or "supporting").strip() or "supporting"
            visual_description = str(item.get("visual_cues") or item.get("visual_description") or "").strip()
        else:
            name = str(item or "").strip()
            aliases = []
            role = "supporting"
            visual_description = ""
        key = normalize_name_key(name)
        if not key or key in seen:
            continue
        if re.search(r"\b(?:unknown|protagonist|speaker|narrator|victim)\b", name, flags=re.IGNORECASE):
            continue
        seen.add(key)
        records.append(
            CanonicalCharacterRecord(
                stable_id=f"story_bible_{key}",
                name=name,
                role=role,
                visual_description=visual_description,
                aliases=aliases,
                confidence=0.55,
            )
        )
    return records


class PanelVisionExtractor:
    # Lowered from 6 → 3. Gemini's ``promptFeedback.blockReason: OTHER`` rate
    # rises sharply with the number of inline images per request and the
    # block bypasses safety settings. Smaller batches plus the binary-split
    # fallback below recover panels whose neighbours triggered the block.
    BATCH_SIZE = 3
    # Run independent batches concurrently. Each Gemini call already does its
    # own model fallback + retry; aggressive parallelism amplifies transient
    # 503s and drives up retry cost, so keep the default conservative.
    PARALLEL_WORKERS = 2

    def __init__(self, router: LLMRouter | None = None) -> None:
        self.router = router or LLMRouter()
        self.settings = get_settings()
        self.cache_dir = ensure_dir(self.settings.data_dir / "_panel_vision_cache" / "panel_vision")

    def run(
        self,
        *,
        project_dir: Path,
        page_paths: list[Path],
        panels: list[PanelBox],
        canonical_characters: list[CanonicalCharacterRecord],
        project_title: str,
        chapter_metadata: ChapterMetadata,
        force_refresh: bool = False,
        progress_callback: Any | None = None,
        cancel_callback: Any | None = None,
    ) -> list[PanelVisionRecord]:
        output_path = project_dir / "output" / "panel_vision.json"
        if output_path.exists() and not force_refresh:
            payload = read_json(output_path, default=[])
            if isinstance(payload, list):
                return [PanelVisionRecord.model_validate(item) for item in payload]

        normalized_characters = [
            character
            if isinstance(character, CanonicalCharacterRecord)
            else CanonicalCharacterRecord.model_validate(character)
            for character in canonical_characters
        ]
        if not normalized_characters:
            normalized_characters = story_bible_canonical_fallback(project_dir)
        kept_panels = [panel for panel in sorted(panels, key=lambda item: item.order) if panel.keep]
        if not kept_panels:
            write_json(output_path, [])
            return []

        loader = ImageLoader(project_dir=project_dir, page_paths=page_paths, max_edge=960)
        clean_evidence_by_id = panel_evidence_by_id(project_dir)
        roster = [character.model_dump(mode="json") for character in normalized_characters]
        alias_map = self._build_alias_map(normalized_characters)
        records: list[PanelVisionRecord] = []
        batches = [kept_panels[index : index + self.BATCH_SIZE] for index in range(0, len(kept_panels), self.BATCH_SIZE)]

        # Build per-batch metadata up-front so we can split cache hits from
        # cache misses and run misses concurrently.
        batch_specs: list[dict[str, Any]] = []
        for batch in batches:
            image_paths: dict[str, Path] = {}
            image_hashes: list[str] = []
            evidence_hints: list[str] = []
            for panel in batch:
                panel_path = loader.panel_image_path(panel)
                if panel_path is None:
                    continue
                image_paths[panel.id] = panel_path
                image_hashes.append(loader.image_payload(panel_path)[2])
                evidence_hints.append(self._panel_evidence_hint(clean_evidence_by_id.get(panel.id)))
            cache_key = loader.composite_hash(
                [
                    _PROMPT_VERSION,
                    project_title,
                    json.dumps(self._chapter_context(chapter_metadata), sort_keys=True, ensure_ascii=False),
                    json.dumps(roster, sort_keys=True, ensure_ascii=False),
                    json.dumps(evidence_hints, sort_keys=True, ensure_ascii=False),
                    *image_hashes,
                ]
            )
            cache_path = self.cache_dir / f"{cache_key}.json"
            panel_manifest = [
                {
                    "panel_id": panel.id,
                    "panel_order": int(panel.order),
                    "page": int(panel.page),
                    "existing_hint": self._panel_evidence_hint(clean_evidence_by_id.get(panel.id)),
                }
                for panel in batch
            ]
            batch_specs.append(
                {
                    "batch": batch,
                    "image_paths": image_paths,
                    "cache_path": cache_path,
                    "panel_manifest": panel_manifest,
                }
            )

        # First pass: cache hits (instant).
        results: list[dict[str, Any] | None] = [None] * len(batch_specs)
        miss_indices: list[int] = []
        for index, spec in enumerate(batch_specs):
            if spec["cache_path"].exists():
                cached_payload = read_json(spec["cache_path"], default={"panels": []})
                if not force_refresh or self._cache_payload_complete(cached_payload, spec["panel_manifest"]):
                    results[index] = cached_payload
                    continue
            miss_indices.append(index)

        # Second pass: cache misses, parallel. ``_extract_batch_with_split``
        # uses ``asyncio.run`` per call which creates a fresh event loop; the
        # thread pool is therefore safe.
        completed = 0

        def _process_miss(idx: int) -> tuple[int, dict[str, Any]]:
            spec = batch_specs[idx]
            payload = self._extract_batch_with_split(
                panel_manifest=spec["panel_manifest"],
                image_paths=spec["image_paths"],
                roster=roster,
                chapter_metadata=chapter_metadata,
                project_title=project_title,
            )
            write_json(spec["cache_path"], payload)
            return idx, payload

        if miss_indices:
            with ThreadPoolExecutor(max_workers=self._worker_count()) as pool:
                for idx, payload in pool.map(_process_miss, miss_indices):
                    results[idx] = payload
                    completed += 1
                    if cancel_callback:
                        cancel_callback()
                    if progress_callback:
                        progress_callback(
                            round(completed / max(len(miss_indices), 1) * 100, 2),
                            f"Extracting panel vision batch {completed}/{len(miss_indices)}",
                        )

        # Third pass: build PanelVisionRecord entries in deterministic order.
        for spec, payload in zip(batch_specs, results):
            if payload is None:
                continue
            batch = spec["batch"]
            by_id = {
                str(item.get("panel_id") or "").strip(): item
                for item in payload.get("panels", []) or []
                if isinstance(item, dict)
            }
            for panel in batch:
                item = by_id.get(panel.id, {})
                raw_speaker = str(item.get("speaker") or "unknown").strip() or "unknown"
                speaker = self._canonicalize_name(raw_speaker, alias_map) or raw_speaker
                action_beat = self._replace_character_aliases(
                    str(item.get("action_beat") or "").strip(),
                    alias_map,
                )
                dialogue = str(item.get("dialogue") or "").strip()
                caption = str(item.get("caption") or "").strip()
                character_names = self._resolve_character_names(
                    action_beat=action_beat,
                    dialogue=dialogue,
                    caption=caption,
                    speaker=speaker,
                    canonical_characters=normalized_characters,
                    explicit_names=item.get("character_names", []) or [],
                    alias_map=alias_map,
                )
                character_roles = self._resolve_character_roles(
                    raw_roles=item.get("character_roles"),
                    action_beat=action_beat,
                    dialogue=dialogue,
                    caption=caption,
                    speaker=speaker,
                    visible_names=character_names,
                    canonical_characters=normalized_characters,
                    alias_map=alias_map,
                )
                records.append(
                    PanelVisionRecord(
                        panel_id=panel.id,
                        panel_order=int(panel.order),
                        page=int(panel.page),
                        speaker=speaker,
                        dialogue=dialogue,
                        caption=caption,
                        action_beat=action_beat,
                        emotion=str(item.get("emotion") or "").strip(),
                        scene_change=bool(item.get("scene_change")),
                        confidence=float(item.get("confidence") or 0.0),
                        character_names=character_names,
                        character_roles=character_roles,
                    )
                )

        write_json(output_path, [record.model_dump(mode="json") for record in records])
        return records

    def _worker_count(self) -> int:
        raw_value = os.getenv("PANELIA_PANEL_VISION_WORKERS", "").strip()
        if raw_value:
            try:
                return max(1, min(8, int(raw_value)))
            except ValueError:
                logger.warning("Ignoring invalid PANELIA_PANEL_VISION_WORKERS=%r", raw_value)
        # Scale to available CPU cores, but cap at 6 to avoid overwhelming Gemini API
        # (safety block rate increases with concurrent requests)
        try:
            return min(6, max(2, cpu_count() // 2))
        except Exception:
            return self.PARALLEL_WORKERS

    def _cache_payload_complete(self, payload: dict[str, Any], panel_manifest: list[dict[str, Any]]) -> bool:
        if not isinstance(payload, dict):
            return False
        panels = payload.get("panels")
        if not isinstance(panels, list):
            return False
        expected_ids = {str(entry.get("panel_id") or "").strip() for entry in panel_manifest}
        expected_ids.discard("")
        observed_ids = {
            str(item.get("panel_id") or "").strip()
            for item in panels
            if isinstance(item, dict)
        }
        return bool(expected_ids) and expected_ids <= observed_ids

    def _chapter_context(self, chapter_metadata: ChapterMetadata) -> dict[str, Any]:
        return {
            "manga_title": chapter_metadata.manga_title,
            "chapter_title": chapter_metadata.chapter_title,
            "chapter_number": chapter_metadata.chapter_number,
            "language": chapter_metadata.language,
        }

    def _extract_batch_with_split(
        self,
        *,
        panel_manifest: list[dict[str, Any]],
        image_paths: dict[str, Path],
        roster: list[dict[str, Any]],
        chapter_metadata: ChapterMetadata,
        project_title: str,
    ) -> dict[str, Any]:
        """Call Gemini panel vision extraction with binary-split fallback.

        Returns ``{"panels": [...]}`` (possibly empty) and never raises.
        """
        if not panel_manifest:
            return {"panels": []}
        sub_image_paths = {
            entry["panel_id"]: image_paths[entry["panel_id"]]
            for entry in panel_manifest
            if entry.get("panel_id") in image_paths
        }
        try:
            result = asyncio.run(
                self.router.extract_panel_vision(
                    panel_manifest,
                    {
                        "character_roster": roster,
                        "chapter_context": {
                            **self._chapter_context(chapter_metadata),
                            "project_title": project_title,
                        },
                    },
                    provider="gemini",
                    panel_image_paths=sub_image_paths,
                )
            )
            return result.payload
        except Exception as exc:
            if len(panel_manifest) <= 1:
                logger.warning(
                    "Panel vision extraction failed for single-panel batch %s: %s. Rescue pass will retry.",
                    [entry.get("panel_id") for entry in panel_manifest],
                    exc,
                )
                return {"panels": []}
            logger.warning(
                "Panel vision extraction failed for batch (%s panels): %s. Splitting batch and retrying.",
                len(panel_manifest),
                exc,
            )
        midpoint = len(panel_manifest) // 2
        left = self._extract_batch_with_split(
            panel_manifest=panel_manifest[:midpoint],
            image_paths=image_paths,
            roster=roster,
            chapter_metadata=chapter_metadata,
            project_title=project_title,
        )
        right = self._extract_batch_with_split(
            panel_manifest=panel_manifest[midpoint:],
            image_paths=image_paths,
            roster=roster,
            chapter_metadata=chapter_metadata,
            project_title=project_title,
        )
        merged_panels: list[dict[str, Any]] = []
        for partial in (left, right):
            if isinstance(partial, dict):
                items = partial.get("panels", []) or []
                if isinstance(items, list):
                    merged_panels.extend(item for item in items if isinstance(item, dict))
        return {"panels": merged_panels}

    def _resolve_character_names(
        self,
        *,
        action_beat: str,
        dialogue: str,
        caption: str,
        speaker: str,
        canonical_characters: list[CanonicalCharacterRecord],
        explicit_names: list[Any],
        alias_map: dict[str, str],
    ) -> list[str]:
        names: list[str] = []
        for raw in explicit_names:
            value = self._canonicalize_name(str(raw).strip(), alias_map) or str(raw).strip()
            if value and is_valid_character_name_candidate(value, allow_stable_label=True):
                names.append(value)
        if speaker and speaker not in {"unknown", "narrator"} and is_valid_character_name_candidate(speaker, allow_stable_label=True):
            names.append(speaker)
        action_haystack = str(action_beat or "").casefold()
        for character in canonical_characters:
            candidate_names = [character.name, *character.aliases]
            for candidate in candidate_names:
                normalized = str(candidate or "").strip()
                if not normalized:
                    continue
                if re.search(rf"\b{re.escape(normalized.casefold())}\b", action_haystack):
                    names.append(character.name)
                    break
        deduped: list[str] = []
        seen: set[str] = set()
        for name in names:
            key = re.sub(r"\s+", " ", name.casefold()).strip()
            if key and key not in seen:
                seen.add(key)
                deduped.append(name)
        return deduped[:8]

    def _resolve_character_roles(
        self,
        *,
        raw_roles: Any,
        action_beat: str,
        dialogue: str,
        caption: str,
        speaker: str,
        visible_names: list[str],
        canonical_characters: list[CanonicalCharacterRecord],
        alias_map: dict[str, str],
    ) -> dict[str, list[str]]:
        allowed = {
            "visible_present",
            "speaker",
            "addressee",
            "mentioned_absent",
            "flashback_present",
            "memory_present",
            "imagined_present",
            "uncertain",
        }
        roles: dict[str, list[str]] = {}

        def add(name: str, role: str) -> None:
            canonical = self._canonicalize_name(str(name).strip(), alias_map) or str(name).strip()
            if role not in allowed or not is_valid_character_name_candidate(canonical, allow_stable_label=True):
                return
            bucket = roles.setdefault(canonical, [])
            if role not in bucket:
                bucket.append(role)

        if isinstance(raw_roles, dict):
            for raw_name, raw_values in raw_roles.items():
                values = raw_values if isinstance(raw_values, list) else [raw_values]
                for value in values:
                    add(str(raw_name), str(value or "").strip())

        for name in visible_names:
            add(name, "visible_present")
        if speaker and speaker not in {"", "unknown", "narrator", "off-screen speaker", "unseen speaker"}:
            add(speaker, "speaker")
            add(speaker, "visible_present")

        dialogue_text = " ".join(part for part in (dialogue, caption) if part)
        action_key = normalize_name_key(action_beat)
        dialogue_key = normalize_name_key(dialogue_text)
        for character in canonical_characters:
            names = [character.name, *character.aliases]
            canonical_name = str(character.name or "").strip()
            if not canonical_name:
                continue
            for candidate in names:
                candidate_key = normalize_name_key(candidate)
                if not candidate_key:
                    continue
                if candidate_key in action_key:
                    add(canonical_name, "visible_present")
                    break
                if candidate_key in dialogue_key and canonical_name not in roles:
                    add(canonical_name, "mentioned_absent")
                    break

        return {name: value[:4] for name, value in roles.items()}

    def _panel_evidence_hint(self, evidence: dict[str, Any] | None) -> str:
        if not isinstance(evidence, dict):
            return ""
        source_summary = evidence.get("source_summary") if isinstance(evidence.get("source_summary"), dict) else {}
        detectors = {
            str(item or "").strip()
            for item in (source_summary.get("detectors") or [])
            if str(item or "").strip()
        }
        confidence = float(evidence.get("confidence") or 0.0)
        if not detectors or detectors <= {"existing-panel-ocr"}:
            return ""
        if confidence < 0.58 and not (detectors & {"page-ocr-backfill", "apple-vision", "comic-ocr", "opencv-region"}):
            return ""
        parts: list[str] = []
        caption = str(evidence.get("caption_text") or "").strip()
        dialogue = str(evidence.get("dialogue_text") or "").strip()
        text = str(evidence.get("text_english") or "").strip()
        confidence = evidence.get("confidence")
        if caption:
            parts.append(f"clean_caption={caption[:500]}")
        if dialogue:
            parts.append(f"clean_dialogue={dialogue[:700]}")
        elif text:
            parts.append(f"clean_text={text[:900]}")
        if isinstance(confidence, (int, float)):
            parts.append(f"evidence_confidence={float(confidence):.2f}")
        return " | ".join(parts)[:1400]

    def _build_alias_map(self, canonical_characters: list[CanonicalCharacterRecord]) -> dict[str, str]:
        alias_targets: dict[str, set[str]] = {}
        for character in canonical_characters:
            canonical_name = str(character.name or "").strip()
            if not canonical_name:
                continue
            for candidate in [canonical_name, *list(character.aliases)]:
                normalized = self._normalize_person_name(candidate)
                if not normalized:
                    continue
                alias_targets.setdefault(normalized, set()).add(canonical_name)

        resolved: dict[str, str] = {}
        for alias, targets in alias_targets.items():
            if len(targets) == 1:
                resolved[alias] = next(iter(targets))
        return resolved

    def _canonicalize_name(self, raw_name: str, alias_map: dict[str, str]) -> str:
        normalized = self._normalize_person_name(raw_name)
        if not normalized:
            return ""
        return alias_map.get(normalized, raw_name.strip())

    def _replace_character_aliases(self, text: str, alias_map: dict[str, str]) -> str:
        rewritten = str(text or "").strip()
        if not rewritten:
            return ""
        replacements = sorted(
            (
                (alias, canonical)
                for alias, canonical in alias_map.items()
                if alias != self._normalize_person_name(canonical)
            ),
            key=lambda item: len(item[0]),
            reverse=True,
        )
        for alias, canonical in replacements:
            canonical_normalized = self._normalize_person_name(canonical)
            canonical_tokens = canonical_normalized.split()
            alias_tokens = alias.split()
            if (
                alias_tokens
                and canonical_tokens
                and len(alias_tokens) < len(canonical_tokens)
                and canonical_tokens[: len(alias_tokens)] == alias_tokens
            ):
                remaining = r"\s+".join(re.escape(token) for token in canonical_tokens[len(alias_tokens) :])
                pattern = re.compile(rf"\b{re.escape(alias)}\b(?!\s+{remaining}\b)", re.IGNORECASE)
            else:
                pattern = re.compile(rf"\b{re.escape(alias)}\b", re.IGNORECASE)
            rewritten = pattern.sub(canonical, rewritten)
        return rewritten

    def _normalize_person_name(self, value: str) -> str:
        cleaned = re.sub(r"[^\w\s'-]", " ", str(value or "").strip(), flags=re.UNICODE)
        cleaned = re.sub(r"\s+", " ", cleaned).strip().casefold()
        if not cleaned:
            return ""
        return cleaned
