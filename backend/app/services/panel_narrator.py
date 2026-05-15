"""Multimodal panel narration service.

Generates narration lines by sending panel images + text to Gemini Vision.
Replaces the 3,833-line panel_script_builder.py and narration methods from
gemini_service.py with a simpler multimodal-first approach.
"""
from __future__ import annotations

import asyncio
import base64
import hashlib
import io
import json
import logging
import re
from pathlib import Path
from typing import Any

try:
    from PIL import Image as _PILImage
    _PIL_AVAILABLE = True
except ImportError:
    _PIL_AVAILABLE = False

from app.core.config import get_settings
from app.schemas.project import ChapterMetadata, PanelBox
from app.services.llm_router import LLMRouter, RoutedResult
from app.services.ocr_cleaner import (
    clean_ocr_lines,
    clean_ocr_text,
    combined_dialogue_entry_lines,
    is_usable_ocr_text,
)
from app.services.scene_builder import SceneBuilder

logger = logging.getLogger(__name__)

_PROMPT_DIR = Path(__file__).resolve().parents[3] / "services" / "prompts"


class PanelNarrator:
    """Generates narration lines by sending panel images + text to Gemini Vision."""

    def __init__(self, router: LLMRouter | None = None, cache_dir: Path | None = None) -> None:
        self.router = router or LLMRouter()
        self.cache_dir = cache_dir
        self.settings = get_settings()
        self._scene_builder = SceneBuilder()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def narrate_all(
        self,
        panels: list[PanelBox],
        scenes: list[dict[str, Any]],
        chapter_summary: str,
        character_dictionary: dict[str, str],
        protagonist_name: str | None = None,
        project_title: str = "",
        metadata: ChapterMetadata | None = None,
        panel_image_dir: Path | None = None,
        scene_clusters: list[dict[str, Any]] | None = None,
        progress_callback: Any | None = None,
    ) -> list[str]:
        """Generate one narration line per kept panel using multimodal LLM."""
        kept = [p for p in sorted(panels, key=lambda p: p.order) if p.keep]
        if not kept:
            return []

        # Collect manually locked narrations - used both to preserve user edits
        # and as authoritative context fed back to the LLM for other panels.
        locked_map: dict[int, str] = {}
        for panel in kept:
            if panel.narration_locked and panel.narration:
                locked_map[panel.order] = panel.narration

        # Build ordered panel payloads with OCR text
        ordered_payloads = self._prepare_panel_payloads(kept, scenes)

        # Build scene seeds for story beat context
        scene_seeds = self._build_scene_seeds(ordered_payloads, scene_clusters or [])

        # Build full-chapter transcript so every batch has complete story context
        chapter_transcript = self._build_chapter_transcript(ordered_payloads)

        # Build locked-narration examples string.
        # These are human-verified lines shown to every batch so the LLM learns
        # the correct character names, relationships, and style from user edits.
        locked_examples = self._build_locked_examples(locked_map, ordered_payloads)

        context = {
            "chapter_summary": chapter_summary,
            "character_dictionary": character_dictionary,
            "protagonist_name": protagonist_name or "",
            "project_title": project_title,
            "metadata": metadata,
            "scene_seeds": scene_seeds,
            "chapter_transcript": chapter_transcript,
            "locked_examples": locked_examples,
        }

        # Batch panels and generate narration (increased from 8 to 16 for async parity)
        batches = self._batch_panels(kept, ordered_payloads, batch_size=16)
        results: list[str] = [""] * len(kept)
        panel_offset = 0
        for batch_index, (batch_panels, batch_payloads) in enumerate(batches):
            if progress_callback:
                progress_callback(batch_index / max(len(batches), 1), f"Narrating batch {batch_index + 1}/{len(batches)}")
            # Pass last 4 non-blank lines from previous batches. Prefer locked
            # (human-reviewed) lines over machine-generated ones when available.
            preceding_machine = [r for r in results[:panel_offset] if r.strip()]
            preceding_locked = [
                locked_map[p.order]
                for p in kept[:panel_offset]
                if p.order in locked_map
            ]
            # Merge: locked lines take priority; fill remainder with machine lines
            preceding_combined = list(dict.fromkeys(preceding_locked[-2:] + preceding_machine[-4:]))[-4:]
            batch_context = {**context, "preceding_narrations": preceding_combined}
            lines = self._narrate_batch(batch_panels, batch_payloads, batch_context, panel_image_dir)
            for j, line in enumerate(lines):
                if panel_offset + j < len(results):
                    results[panel_offset + j] = line
            panel_offset += len(batch_panels)

        # Retry any panels that came back blank (LLM returned fewer items than expected)
        blank_indices = [i for i, line in enumerate(results) if not line.strip()]
        if blank_indices:
            logger.info("Retrying %d blank panels individually", len(blank_indices))
            for idx in blank_indices:
                panel = kept[idx]
                if panel.order in locked_map:
                    continue
                payload = ordered_payloads[idx]
                img_path = self._find_panel_image(panel, panel_image_dir) if panel_image_dir else None
                retry = self._narrate_single_retry(panel, payload, context, img_path)
                if retry.strip():
                    results[idx] = retry
                    logger.debug("Retry succeeded for panel %d: %s", panel.order, retry[:80])
                else:
                    logger.warning("Retry still blank for panel %d (page %s)", panel.order, panel.page)

        # Restore locked narrations
        final: list[str] = []
        for panel, line in zip(kept, results, strict=False):
            if panel.order in locked_map:
                final.append(locked_map[panel.order])
            else:
                final.append(line)

        return final

    def narrate_single(
        self,
        panel: PanelBox,
        context: dict[str, Any],
        mode: str = "balanced",
        current_narration: str = "",
        panel_image_path: Path | None = None,
        previous_lines: list[str] | None = None,
        next_line: str = "",
    ) -> str:
        """Rewrite a single panel's narration (for the rewrite-panel API)."""
        scene_lookup = context.get("scene_lookup", {})
        scene = scene_lookup.get(panel.id, {})

        text = clean_ocr_text(panel.ocr_text or "").strip()
        dialogue_lines = combined_dialogue_entry_lines(scene.get("dialogue_entries", []) or [])
        if dialogue_lines:
            text = " ".join(dialogue_lines).strip()

        char_dict = context.get("character_dictionary", {})
        char_block = ", ".join(f"{k}: {v}" for k, v in char_dict.items() if k and v) if char_dict else "(none)"
        chapter_summary = context.get("chapter_summary", "")
        project_title = context.get("project_title", "")

        mode_instruction = {
            "balanced": "Write a balanced narration that captures the key story event.",
            "closer_to_ocr": "Stay close to the extracted text/dialogue meaning while writing clean narration.",
            "shorten": "Write a concise narration of 8-12 words.",
        }.get(mode, "Write a balanced narration that captures the key story event.")

        prompt = (
            "Rewrite the narration for a single manga panel in a YouTube recap video.\n\n"
            f"Mode: {mode_instruction}\n"
            f"Current narration: {current_narration}\n"
            f"Extracted text: {text[:500]}\n"
            f"Previous lines: {' | '.join((previous_lines or [])[-3:])}\n"
            f"Next line: {next_line}\n"
            f"Chapter summary: {chapter_summary[:500]}\n"
            f"Characters: {char_block}\n"
            f"Project: {project_title}\n\n"
            "Rules:\n"
            "- Write one complete English sentence with a subject and verb.\n"
            "- Describe the story event, not what the image looks like.\n"
            "- Use character names when known.\n"
            "- Make it meaningfully different from the current narration.\n\n"
            'Return valid JSON only: {"narration": "Your narration line here."}\n'
        )

        parts: list[dict[str, Any]] | None = None
        if panel_image_path and panel_image_path.exists():
            # Use _load_and_resize_image to cap at _IMAGE_MAX_PX - avoids
            # sending full-resolution images (can be 5+ MB each) to Gemini.
            img_data, mime = self._load_and_resize_image(panel_image_path)
            if img_data:
                parts = [{"text": prompt}]
                parts.append({"text": "Panel image:"})
                parts.append({
                    "inlineData": {
                        "mimeType": mime,
                        "data": base64.b64encode(img_data).decode("utf-8"),
                    },
                })

        try:
            result = asyncio.run(
                self.router._route_json(
                    task_name="single panel rewrite",
                    prompt=prompt,
                    validator=self._validate_single_response,
                    max_output_tokens=320,
                    parts=parts,
                )
            )
            narration = str(result.payload.get("narration") or "").strip()
            return narration if narration else current_narration
        except Exception as exc:
            logger.warning("Single panel rewrite failed: %s", exc)
            return current_narration

    def load_context_from_cache(self, output_dir: Path) -> dict[str, Any]:
        """Load saved context artifacts for single-panel rewrites."""
        context: dict[str, Any] = {
            "chapter_summary": "",
            "character_dictionary": {},
            "protagonist_name": "",
            "project_title": "",
            "scene_lookup": {},
        }
        try:
            char_path = output_dir / "character_dictionary.json"
            if char_path.exists():
                context["character_dictionary"] = json.loads(char_path.read_text(encoding="utf-8"))
        except Exception:
            pass
        try:
            scene_path = output_dir / "scene_summaries.json"
            if scene_path.exists():
                data = json.loads(scene_path.read_text(encoding="utf-8"))
                context["chapter_summary"] = str(data.get("chapter_summary") or "")
        except Exception:
            pass
        try:
            identity_path = output_dir / "character_identity_report.json"
            if identity_path.exists():
                data = json.loads(identity_path.read_text(encoding="utf-8"))
                context["protagonist_name"] = str(data.get("protagonist_name") or "")
        except Exception:
            pass
        return context

    # ------------------------------------------------------------------
    # Panel preparation (extracted from gemini_service._prepare_panel_payload)
    # ------------------------------------------------------------------

    def _prepare_panel_payloads(
        self,
        panels: list[PanelBox],
        scene_data: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Build ordered panel payloads with cleaned OCR text."""
        scene_lookup = {
            scene.get("panel_order"): scene
            for scene in scene_data
            if scene.get("panel_order") is not None
        }
        prepared: list[dict[str, Any]] = []
        for panel in panels:
            scene = scene_lookup.get(panel.order, {})
            text_lines: list[str] = []
            character_names: list[str] = []

            if panel.manual_ocr_text:
                text_lines = clean_ocr_lines([panel.ocr_text or ""])
            else:
                text_lines = combined_dialogue_entry_lines(scene.get("dialogue_entries", []) or [])
                if not text_lines:
                    text_lines = clean_ocr_lines(scene.get("dialogue", []))
                if not text_lines:
                    text_lines = clean_ocr_lines(scene.get("dialogue_original", []))
                if not text_lines and scene.get("detected_text"):
                    text_lines = clean_ocr_lines([str(scene.get("detected_text", ""))])
                if not text_lines and panel.ocr_text:
                    text_lines = clean_ocr_lines([panel.ocr_text])

            for entry in scene.get("dialogue_entries", []) or []:
                if isinstance(entry, dict):
                    for name in entry.get("character_names", []) or []:
                        name_str = str(name).strip()
                        if name_str:
                            character_names.append(name_str)

            scene_character_names = [
                str(name).strip()
                for name in scene.get("character_names", []) or []
                if str(name).strip()
            ]
            character_names = sorted(set(character_names + scene_character_names))

            combined = clean_ocr_text(" ".join(text_lines).strip())[:900]
            if combined and not is_usable_ocr_text(combined):
                combined = ""

            # Detect non-English text that translation failed to handle
            translation_failed = False
            if combined:
                alpha_chars = [c for c in combined if c.isalpha()]
                if alpha_chars:
                    accented = sum(1 for c in alpha_chars if ord(c) > 127)
                    if accented / len(alpha_chars) > 0.30:
                        logger.debug(
                            "Panel %d: non-English text detected (%.0f%% accented), marking translation_failed",
                            panel.order, accented / len(alpha_chars) * 100,
                        )
                        combined = ""
                        translation_failed = True

            prepared.append({
                "panel": int(panel.page or 0) * 10000 + int(getattr(panel, "panel", 0) or 0),
                "panel_id": panel.id,
                "page": panel.page,
                "text": combined,
                "translation_failed": translation_failed,
                "character_names": character_names,
                "visual_caption": panel.visual_caption or "",
            })
        return prepared

    # ------------------------------------------------------------------
    # Scene seed building (extracted from gemini_service._build_scene_seeds)
    # ------------------------------------------------------------------

    def _build_scene_seeds(
        self,
        ordered_payloads: list[dict[str, Any]],
        scene_clusters: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Build scene seeds from clusters or automatic grouping."""
        if scene_clusters:
            seeds = self._scene_seeds_from_clusters(ordered_payloads, scene_clusters)
            if seeds:
                return seeds
        return [seed.to_dict() for seed in self._scene_builder.build(ordered_payloads)]

    def _scene_seeds_from_clusters(
        self,
        ordered_payloads: list[dict[str, Any]],
        scene_clusters: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        if not ordered_payloads or not scene_clusters:
            return []

        panel_by_id = {str(p["panel_id"]): p for p in ordered_payloads}
        panel_by_order = {int(p["panel"]): p for p in ordered_payloads}
        groups: list[list[dict[str, Any]]] = []

        for cluster in sorted(scene_clusters, key=lambda c: int(c.get("scene", 0) or 0)):
            group: list[dict[str, Any]] = []
            for pid in cluster.get("panel_ids", []) or []:
                p = panel_by_id.get(str(pid))
                if p is not None:
                    group.append(p)
            if not group:
                for po in cluster.get("panels", []) or []:
                    p = panel_by_order.get(int(po))
                    if p is not None:
                        group.append(p)
            # Dedup
            seen: set[str] = set()
            deduped: list[dict[str, Any]] = []
            for p in sorted(group, key=lambda x: int(x["panel"])):
                pid = str(p["panel_id"])
                if pid not in seen:
                    seen.add(pid)
                    deduped.append(p)
            if deduped:
                groups.append(deduped)

        if not groups:
            return []

        return [
            self._seed_from_group(i + 1, g)
            for i, g in enumerate(groups)
            if g
        ]

    def _seed_from_group(self, scene_id: int, group: list[dict[str, Any]]) -> dict[str, Any]:
        ordered = sorted(group, key=lambda p: int(p["panel"]))
        combined_text = " ".join(
            str(p.get("text") or "").strip()
            for p in ordered
            if str(p.get("text") or "").strip()
        )
        return {
            "scene_id": scene_id,
            "panel_start": int(ordered[0]["panel"]),
            "panel_end": int(ordered[-1]["panel"]),
            "panel_ids": [str(p["panel_id"]) for p in ordered],
            "panels": [int(p["panel"]) for p in ordered],
            "combined_text": clean_ocr_text(combined_text)[:1800],
            "character_names": sorted({
                str(name).strip()
                for p in ordered
                for name in p.get("character_names", []) or []
                if str(name).strip()
            }),
        }

    # ------------------------------------------------------------------
    # Chapter transcript
    # ------------------------------------------------------------------

    def _build_chapter_transcript(self, payloads: list[dict[str, Any]]) -> str:
        """Concatenate all panel OCR texts into a single chapter-wide transcript.

        This gives every batch full story context so the LLM can resolve
        character names, track continuity, and narrate panels that have no
        OCR text by understanding the surrounding story beats.
        """
        lines: list[str] = []
        for payload in payloads:
            panel_num = payload["panel"]
            text = str(payload.get("text") or "").strip()
            if text:
                lines.append(f"Panel {panel_num}: {text}")
        if not lines:
            return "(no text extracted)"
        return "\n".join(lines)

    def _build_locked_examples(
        self,
        locked_map: dict[int, str],
        payloads: list[dict[str, Any]],
    ) -> str:
        """Format manually-locked narrations as reference examples for the LLM.

        These lines were written or corrected by a human, so they carry the
        correct character names, relationships, and story framing. Every batch
        receives them so the LLM can infer who 'the silver-haired manager' or
        'the loan shark' is, and match that style in un-reviewed panels.

        Returns an empty string when no locked narrations exist (new project).
        """
        if not locked_map:
            return ""

        # Build a panel-number → payload text map for context alongside narration
        payload_text: dict[int, str] = {
            int(p["panel"]): str(p.get("text") or "").strip()
            for p in payloads
        }

        lines: list[str] = []
        for order in sorted(locked_map):
            narration = locked_map[order].strip()
            if not narration:
                continue
            ocr = payload_text.get(order, "")
            if ocr:
                lines.append(f'Panel {order}: "{narration}"  [source text: {ocr[:80]}]')
            else:
                lines.append(f'Panel {order}: "{narration}"')

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Blank-panel retry
    # ------------------------------------------------------------------

    def _narrate_single_retry(
        self,
        panel: PanelBox,
        payload: dict[str, Any],
        context: dict[str, Any],
        img_path: Path | None,
    ) -> str:
        """Re-narrate a single panel that came back blank from its batch.

        Uses a tighter prompt focused solely on the one missing panel so the
        LLM cannot skip it by returning a short list.
        """
        panel_num = payload["panel"]
        text = str(payload.get("text") or "").strip()
        caption = str(payload.get("visual_caption") or "").strip()
        translation_failed = payload.get("translation_failed", False)

        char_dict = context.get("character_dictionary", {})
        char_block = ", ".join(f"{k}: {v}" for k, v in char_dict.items() if k and v) if char_dict else "(none)"
        transcript = context.get("chapter_transcript", "") or "(none)"
        locked_examples = context.get("locked_examples", "") or ""

        text_part = (
            "[Translation unavailable - use image and chapter transcript for context]"
            if translation_failed
            else (f'"{text}"' if text else "(no text)")
        )
        hint_part = f"\nVisual hint: {caption}" if caption else ""
        locked_part = f"\nManually reviewed narrations (use for character names):\n{locked_examples}" if locked_examples else ""

        prompt = (
            "You are writing narration for a single manga panel in a YouTube recap video.\n\n"
            f"Panel number: {panel_num}\n"
            f"Extracted text: {text_part}{hint_part}\n\n"
            "Full chapter transcript (for story context):\n"
            f"{transcript}\n\n"
            f"Characters: {char_block}\n"
            f"Chapter summary: {context.get('chapter_summary', '') or '(none)'}"
            f"{locked_part}\n\n"
            "Rules:\n"
            "- Write ONE complete English sentence with a subject and verb.\n"
            "- 8-20 words.\n"
            "- Describe the STORY EVENT - not what the image looks like.\n"
            "- Use character names when known. No generic filler.\n\n"
            f'Return JSON only: {{"panel_narrations":[{{"panel":{panel_num},"narration":"..."}}]}}\n'
        )

        parts: list[dict[str, Any]] | None = None
        if img_path and img_path.exists():
            try:
                img_data, mime = self._load_and_resize_image(img_path)
                if img_data is not None:
                    parts = [
                        {"text": prompt},
                        {"text": f"Panel {panel_num} image:"},
                        {"inlineData": {"mimeType": mime, "data": base64.b64encode(img_data).decode("utf-8")}},
                    ]
            except Exception as exc:
                logger.debug("Retry image load failed for panel %d: %s", panel_num, exc)

        try:
            result = asyncio.run(
                self.router._route_json(
                    task_name="panel narration retry",
                    prompt=prompt,
                    validator=self._validate_batch_response,
                    max_output_tokens=320,
                    parts=parts,
                )
            )
            lines = self._parse_batch_response(result, [payload])
            return lines[0] if lines else ""
        except Exception as exc:
            logger.warning("Retry narration failed for panel %d: %s", panel_num, exc)
            return ""

    # ------------------------------------------------------------------
    # Batching
    # ------------------------------------------------------------------

    def _batch_panels(
        self,
        panels: list[PanelBox],
        payloads: list[dict[str, Any]],
        batch_size: int = 15,
    ) -> list[tuple[list[PanelBox], list[dict[str, Any]]]]:
        """Split panels into batches for multimodal processing."""
        batches: list[tuple[list[PanelBox], list[dict[str, Any]]]] = []
        for i in range(0, len(panels), batch_size):
            batch_panels = panels[i:i + batch_size]
            batch_payloads = payloads[i:i + batch_size]
            batches.append((batch_panels, batch_payloads))
        return batches

    # ------------------------------------------------------------------
    # LLM narration
    # ------------------------------------------------------------------

    def _narrate_batch(
        self,
        panels: list[PanelBox],
        payloads: list[dict[str, Any]],
        context: dict[str, Any],
        image_dir: Path | None,
    ) -> list[str]:
        """Send a batch of panels with images to Gemini Vision.

        On failure, falls back to text-only prompts for the same batch,
        then to individual panel narrations as a last resort.
        """
        prompt = self._build_batch_prompt(payloads, context)
        parts = self._build_multimodal_parts(prompt, panels, image_dir)

        # Attempt 1: multimodal (images + text)
        if parts is not None:
            try:
                result = asyncio.run(
                    self.router._route_json(
                        task_name="panel narration batch",
                        prompt=prompt,
                        validator=self._validate_batch_response,
                        max_output_tokens=min(4000, max(800, 220 * len(panels))),
                        parts=parts,
                    )
                )
                return self._parse_batch_response(result, payloads)
            except Exception as exc:
                logger.warning("Panel narration batch (multimodal) failed: %s - retrying text-only", exc)

        # Attempt 2: text-only (no images)
        try:
            result = asyncio.run(
                self.router._route_json(
                    task_name="panel narration batch text-only",
                    prompt=prompt,
                    validator=self._validate_batch_response,
                    max_output_tokens=min(4000, max(800, 220 * len(panels))),
                    parts=None,
                )
            )
            lines = self._parse_batch_response(result, payloads)
            # Check if text-only produced results
            non_blank = sum(1 for l in lines if l.strip())
            if non_blank >= len(panels) // 2:
                return lines
        except Exception as exc:
            logger.warning("Panel narration batch (text-only) also failed: %s - falling back to individual", exc)

        # Attempt 3: narrate each panel individually (text-only, no images)
        logger.warning("Falling back to individual panel narration for %d panels", len(panels))
        return self._narrate_individually(panels, payloads, context)

    def _narrate_individually(
        self,
        panels: list[PanelBox],
        payloads: list[dict[str, Any]],
        context: dict[str, Any],
    ) -> list[str]:
        """Last-resort: narrate each panel as a micro-batch of 1 (text-only)."""
        lines: list[str] = []
        for panel, payload in zip(panels, payloads):
            single_prompt = self._build_batch_prompt([payload], context)
            try:
                result = asyncio.run(
                    self.router._route_json(
                        task_name="panel narration single",
                        prompt=single_prompt,
                        validator=self._validate_batch_response,
                        max_output_tokens=320,
                        parts=None,
                    )
                )
                single_lines = self._parse_batch_response(result, [payload])
                lines.append(single_lines[0] if single_lines else "")
            except Exception as exc:
                logger.warning("Individual panel narration failed for panel %s: %s", panel.order, exc)
                lines.append("")
        return lines

    def _build_batch_prompt(
        self,
        payloads: list[dict[str, Any]],
        context: dict[str, Any],
    ) -> str:
        """Build the prompt for a batch of panels."""
        template = (_PROMPT_DIR / "narrator-panel-batch.md").read_text(encoding="utf-8")

        metadata = context.get("metadata")
        chapter_metadata_str = ""
        if metadata:
            chapter_metadata_str = json.dumps({
                "manga_title": getattr(metadata, "manga_title", ""),
                "chapter_title": getattr(metadata, "chapter_title", ""),
                "chapter_number": getattr(metadata, "chapter_number", ""),
                "language": getattr(metadata, "language", ""),
            }, ensure_ascii=False)

        char_dict = context.get("character_dictionary", {})
        char_block = ""
        if char_dict:
            entries = []
            for name, info in char_dict.items():
                if isinstance(info, dict):
                    display = info.get("display_name", name)
                    aliases = info.get("aliases", [])
                    role = info.get("role", "")
                    appearance = info.get("appearance", "")
                    parts = [display]
                    if aliases:
                        parts.append(f"(also: {', '.join(aliases)})")
                    if role:
                        parts.append(f"- {role}")
                    if appearance:
                        parts.append(f"[appearance: {appearance}]")
                    entries.append(" ".join(parts))
                else:
                    entries.append(f"{name}: {info}")
            char_block = "\n".join(entries)

        # Preceding narrations: last N lines from previous batches.
        # Shown to the LLM so it knows what story beats were already covered.
        preceding_lines = context.get("preceding_narrations", [])
        preceding_block = ""
        if preceding_lines:
            preceding_block = "\n".join(f"- {line}" for line in preceding_lines)

        # Build panel block
        panel_lines: list[str] = []
        for payload in payloads:
            panel_num = int(payload["panel"])
            text = str(payload.get("text") or "").strip()
            chars = payload.get("character_names", [])
            caption = str(payload.get("visual_caption") or "").strip()
            translation_failed = payload.get("translation_failed", False)

            parts = [f"Panel {panel_num}"]
            if translation_failed:
                parts.append("Extracted text: [Translation unavailable - use panel image and scene context]")
            elif text:
                parts.append(f"Extracted text: {text[:520]}")
            else:
                parts.append("Extracted text: (none)")
            if chars:
                parts.append(f"Characters: {', '.join(chars)}")
            if caption:
                parts.append(f"Visual hint: {caption[:200]}")
            panel_lines.append("\n".join(parts))

        panel_block = "\n\n".join(panel_lines)

        locked_examples = context.get("locked_examples", "") or ""

        return (
            template
            .replace("{project_title}", context.get("project_title", ""))
            .replace("{chapter_metadata}", chapter_metadata_str or "(none)")
            .replace("{chapter_summary}", context.get("chapter_summary", "") or "(none)")
            .replace("{character_dictionary}", char_block or "(none)")
            .replace("{chapter_transcript}", context.get("chapter_transcript", "") or "(none)")
            .replace("{locked_examples}", locked_examples or "(none - this is a fresh run)")
            .replace("{preceding_narrations}", preceding_block or "(this is the first batch)")
            .replace("{panel_block}", panel_block)
        )

    # Max pixel dimension for panel images sent to Gemini.
    # 768px keeps quality high while reducing payload by ~10x vs full-res.
    _IMAGE_MAX_PX = 768

    def _build_multimodal_parts(
        self,
        prompt: str,
        panels: list[PanelBox],
        image_dir: Path | None,
    ) -> list[dict[str, Any]] | None:
        """Build text + image parts array for Gemini Vision.

        Images are resized to _IMAGE_MAX_PX on the longest side and
        re-encoded as JPEG to keep payloads small (~60-100KB each).
        """
        if not image_dir:
            return None

        parts: list[dict[str, Any]] = [{"text": prompt}]
        any_image = False

        for panel in panels:
            img_path = self._find_panel_image(panel, image_dir)
            if img_path and img_path.exists():
                try:
                    img_data, mime = self._load_and_resize_image(img_path)
                    if img_data is None:
                        continue
                    parts.append({"text": f"Panel {panel.order} image:"})
                    parts.append({
                        "inlineData": {
                            "mimeType": mime,
                            "data": base64.b64encode(img_data).decode("utf-8"),
                        },
                    })
                    any_image = True
                except Exception as exc:
                    logger.debug("Could not read panel image %s: %s", img_path, exc)

        return parts if any_image else None

    def _load_and_resize_image(self, img_path: Path) -> tuple[bytes | None, str]:
        """Load an image file, resize to _IMAGE_MAX_PX, return (bytes, mime_type).

        Uses PIL when available for quality resizing + JPEG encoding.
        Falls back to raw bytes (capped at 1MB) when PIL is not available.
        """
        if _PIL_AVAILABLE:
            try:
                with _PILImage.open(img_path) as img:
                    # Convert to RGB for JPEG encoding (strips alpha channel)
                    if img.mode not in ("RGB", "L"):
                        img = img.convert("RGB")
                    # Resize keeping aspect ratio
                    w, h = img.size
                    max_px = self._IMAGE_MAX_PX
                    if w > max_px or h > max_px:
                        scale = max_px / max(w, h)
                        new_w = max(1, int(w * scale))
                        new_h = max(1, int(h * scale))
                        img = img.resize((new_w, new_h), _PILImage.LANCZOS)
                    buf = io.BytesIO()
                    img.save(buf, format="JPEG", quality=82, optimize=True)
                    return buf.getvalue(), "image/jpeg"
            except Exception as exc:
                logger.debug("PIL resize failed for %s: %s", img_path, exc)

        # PIL not available - use raw bytes but cap at 500KB
        raw = img_path.read_bytes()
        if len(raw) > 500 * 1024:
            logger.debug("Image %s is %dKB, skipping (PIL unavailable for resize)", img_path, len(raw) // 1024)
            return None, ""
        mime = "image/png" if img_path.suffix.lower() == ".png" else "image/jpeg"
        return raw, mime

    def _find_panel_image(self, panel: PanelBox, image_dir: Path) -> Path | None:
        """Locate the panel's cropped image file."""
        # Try common naming patterns
        for pattern in [
            f"panel_{panel.order:03d}.png",
            f"panel_{panel.order:03d}.jpg",
            f"panel_{panel.order:03d}.jpeg",
            f"{panel.id}.png",
            f"{panel.id}.jpg",
        ]:
            candidate = image_dir / pattern
            if candidate.exists():
                return candidate
        return None

    # ------------------------------------------------------------------
    # Response parsing
    # ------------------------------------------------------------------

    def _validate_batch_response(self, payload: Any) -> dict[str, Any]:
        """Validate the LLM response has panel_narrations array."""
        if not isinstance(payload, dict):
            raise ValueError("Panel narration response is not a JSON object")

        narrations = payload.get("panel_narrations")
        if not isinstance(narrations, list) or not narrations:
            # Also check for "rewrites" key (LLM sometimes uses that format)
            narrations = payload.get("rewrites")
        if not isinstance(narrations, list) or not narrations:
            raise ValueError("Response missing panel_narrations array")

        cleaned: list[dict[str, Any]] = []
        for item in narrations:
            if not isinstance(item, dict):
                continue
            panel_val = item.get("panel")
            match = re.search(r"\d+", str(panel_val or ""))
            panel_num = int(match.group(0)) if match else None
            narration = str(item.get("narration") or item.get("summary") or "").strip()
            if narration:
                cleaned.append({"panel": panel_num, "narration": narration})

        if not cleaned:
            raise ValueError("No usable narrations in response")

        return {"panel_narrations": cleaned}

    def _validate_single_response(self, payload: Any) -> dict[str, Any]:
        """Validate single panel rewrite response."""
        if not isinstance(payload, dict):
            raise ValueError("Single panel response is not a JSON object")
        narration = str(payload.get("narration") or "").strip()
        if not narration:
            raise ValueError("Response missing narration field")
        return {"narration": narration}

    def _parse_batch_response(
        self,
        result: RoutedResult,
        payloads: list[dict[str, Any]],
    ) -> list[str]:
        """Parse LLM batch response into narration lines aligned to panels."""
        narrations = result.payload.get("panel_narrations", [])
        by_panel: dict[int, str] = {}
        for item in narrations:
            panel_num = item.get("panel")
            narration = str(item.get("narration") or "").strip()
            if panel_num is not None and narration:
                by_panel[panel_num] = narration

        # Align to payload order
        lines: list[str] = []
        for i, payload in enumerate(payloads):
            panel_num = int(payload["panel"])
            line = by_panel.get(panel_num, "")
            if not line and i < len(narrations):
                # Fallback: use positional alignment if panel numbers don't match
                line = str(narrations[i].get("narration") or "").strip() if i < len(narrations) else ""
            lines.append(line)

        return lines
