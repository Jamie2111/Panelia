"""Character visual profiler.

Builds appearance descriptions for named characters by asking Gemini Vision
to describe them in panels where their names appear in OCR text.

These descriptions are merged into the character_dictionary before narration
so the LLM can identify characters visually even in silent/non-OCR panels.
"""
from __future__ import annotations

import asyncio
import base64
import io
import json
import logging
from pathlib import Path
from typing import Any

try:
    from PIL import Image as _PILImage
    _PIL_AVAILABLE = True
except ImportError:
    _PIL_AVAILABLE = False

from app.services.llm_router import LLMRouter

logger = logging.getLogger(__name__)

# Cache file name inside the project output directory
_CACHE_FILE = "character_appearances.json"


class CharacterVisualProfiler:
    """Builds visual appearance profiles for named characters.

    Makes one Gemini Vision call per character to produce a concise appearance
    description (hair, clothing, distinctive features). Results are cached so
    re-runs do not repeat the LLM calls.
    """

    _IMAGE_MAX_PX = 512  # Smaller than narration images - just enough for faces

    def __init__(self, router: LLMRouter | None = None) -> None:
        self.router = router or LLMRouter()

    def enrich_character_dictionary(
        self,
        character_dictionary: dict[str, Any],
        panels: list[Any],          # list[PanelBox]
        scenes: list[dict[str, Any]],
        panel_image_dir: Path | None,
        cache_dir: Path | None = None,
    ) -> dict[str, Any]:
        """Return character_dictionary enriched with appearance descriptions.

        Looks up the best panel image for each named character, calls Gemini
        Vision once per character, and caches the result. On any failure, the
        original character_dictionary is returned unchanged.
        """
        if not character_dictionary or not panel_image_dir:
            return character_dictionary

        # Load existing cache
        cache = self._load_cache(cache_dir)
        enriched = dict(character_dictionary)
        updated = False

        # Build panel → character name lookup from scene data
        named_panels = self._find_named_panels(panels, scenes)

        for raw_name, value in character_dictionary.items():
            display_name = value if isinstance(value, str) else raw_name
            cache_key = display_name.lower().strip()

            # Already cached?
            if cache_key in cache:
                logger.debug("Using cached appearance for %s", display_name)
                appearance = cache[cache_key]
            else:
                # Find best panel for this character and profile it
                panel_obj = named_panels.get(display_name.lower()) or named_panels.get(raw_name.lower())
                if panel_obj is None:
                    continue

                img_path = self._find_panel_image(panel_obj, panel_image_dir)
                if img_path is None:
                    continue

                appearance = self._profile_character(display_name, img_path)
                if not appearance:
                    continue

                cache[cache_key] = appearance
                updated = True
                logger.info("Profiled appearance for %s: %s", display_name, appearance[:80])

            # Merge into enriched dict
            existing = enriched.get(raw_name) or {}
            if isinstance(existing, dict):
                existing = dict(existing)
                existing["appearance"] = appearance
                enriched[raw_name] = existing
            else:
                enriched[raw_name] = {"display_name": display_name, "appearance": appearance}

        if updated:
            self._save_cache(cache, cache_dir)

        return enriched

    # ------------------------------------------------------------------
    # Named-panel lookup
    # ------------------------------------------------------------------

    def _find_named_panels(
        self,
        panels: list[Any],
        scenes: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Return {character_name_lower: best_panel_object} for each character.

        Picks the panel where the character's name appears in OCR and which
        has the highest OCR text quality (longest, most English content).
        """
        # Build scene lookup by panel_id and by panel_order
        scene_by_id: dict[str, dict] = {}
        scene_by_order: dict[int, dict] = {}
        for scene in scenes:
            pid = scene.get("panel_id")
            if pid:
                scene_by_id[str(pid)] = scene
            po = scene.get("panel_order")
            if po is not None:
                scene_by_order[int(po)] = scene

        # char_name_lower → (panel_obj, score)
        best: dict[str, tuple[Any, int]] = {}

        for panel in panels:
            if not getattr(panel, "keep", False):
                continue

            scene = scene_by_id.get(str(panel.id)) or scene_by_order.get(int(panel.order), {})
            char_names: list[str] = []

            # From dialogue_entries
            for entry in scene.get("dialogue_entries", []) or []:
                if isinstance(entry, dict):
                    for name in entry.get("character_names", []) or []:
                        name_str = str(name).strip()
                        if name_str:
                            char_names.append(name_str.lower())

            # From scene-level character_names
            for name in scene.get("character_names", []) or []:
                name_str = str(name).strip()
                if name_str:
                    char_names.append(name_str.lower())

            if not char_names:
                continue

            # Score this panel by OCR quality (longer English text = better)
            ocr = str(getattr(panel, "ocr_text", "") or "").strip()
            score = sum(1 for c in ocr if c.isalpha() and ord(c) < 128)

            for name in set(char_names):
                existing = best.get(name)
                if existing is None or score > existing[1]:
                    best[name] = (panel, score)

        return {name: panel_obj for name, (panel_obj, _) in best.items()}

    # ------------------------------------------------------------------
    # Gemini Vision profiling
    # ------------------------------------------------------------------

    def _profile_character(self, character_name: str, img_path: Path) -> str:
        """Ask Gemini Vision to describe character's appearance in one panel."""
        prompt = (
            f"You are helping build a character reference for a manga narration pipeline.\n\n"
            f"The character named '{character_name}' appears in this manga panel.\n\n"
            f"Describe their visual appearance in ONE concise sentence (20-35 words) covering:\n"
            f"- Hair: color, length, style\n"
            f"- Build: age, body type\n"
            f"- Clothing: distinctive outfit or colors\n"
            f"- Any other highly distinctive visual marker (scar, eyepatch, unusual eyes, etc.)\n\n"
            f"Focus only on features that would help identify this character across many different panels.\n"
            f"Do not mention the panel content or story. Only describe physical appearance.\n\n"
            f'Return JSON only: {{"name": "{character_name}", "appearance": "..."}}\n'
        )

        try:
            img_data, mime = self._load_image(img_path)
            if img_data is None:
                return ""

            parts: list[dict[str, Any]] = [
                {"text": prompt},
                {"text": f"Panel image (find {character_name}):"},
                {"inlineData": {"mimeType": mime, "data": base64.b64encode(img_data).decode("utf-8")}},
            ]

            result = asyncio.run(
                self.router._route_json(
                    task_name="character visual profile",
                    prompt=prompt,
                    validator=self._validate_profile_response,
                    max_output_tokens=256,
                    parts=parts,
                )
            )
            return str(result.payload.get("appearance") or "").strip()
        except Exception as exc:
            logger.warning("Character visual profiling failed for %s: %s", character_name, exc)
            return ""

    def _validate_profile_response(self, payload: Any) -> dict[str, Any]:
        if not isinstance(payload, dict):
            raise ValueError("Profile response is not a JSON object")
        appearance = str(payload.get("appearance") or "").strip()
        if not appearance or len(appearance) < 10:
            raise ValueError("Profile response missing or too short")
        return {"name": str(payload.get("name") or ""), "appearance": appearance}

    # ------------------------------------------------------------------
    # Image loading
    # ------------------------------------------------------------------

    def _load_image(self, img_path: Path) -> tuple[bytes | None, str]:
        if _PIL_AVAILABLE:
            try:
                with _PILImage.open(img_path) as img:
                    if img.mode not in ("RGB", "L"):
                        img = img.convert("RGB")
                    w, h = img.size
                    max_px = self._IMAGE_MAX_PX
                    if w > max_px or h > max_px:
                        scale = max_px / max(w, h)
                        img = img.resize((max(1, int(w * scale)), max(1, int(h * scale))), _PILImage.LANCZOS)
                    buf = io.BytesIO()
                    img.save(buf, format="JPEG", quality=80, optimize=True)
                    return buf.getvalue(), "image/jpeg"
            except Exception as exc:
                logger.debug("PIL load failed for %s: %s", img_path, exc)

        raw = img_path.read_bytes()
        if len(raw) > 400 * 1024:
            return None, ""
        mime = "image/png" if img_path.suffix.lower() == ".png" else "image/jpeg"
        return raw, mime

    def _find_panel_image(self, panel: Any, image_dir: Path) -> Path | None:
        order = getattr(panel, "order", None)
        pid = getattr(panel, "id", None)
        candidates = []
        if order is not None:
            candidates += [
                image_dir / f"panel_{int(order):03d}.png",
                image_dir / f"panel_{int(order):03d}.jpg",
            ]
        if pid:
            candidates += [image_dir / f"{pid}.png", image_dir / f"{pid}.jpg"]
        for c in candidates:
            if c.exists():
                return c
        return None

    # ------------------------------------------------------------------
    # Cache
    # ------------------------------------------------------------------

    def _cache_path(self, cache_dir: Path | None) -> Path | None:
        return (cache_dir / _CACHE_FILE) if cache_dir else None

    def _load_cache(self, cache_dir: Path | None) -> dict[str, str]:
        path = self._cache_path(cache_dir)
        if path and path.exists():
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                pass
        return {}

    def _save_cache(self, cache: dict[str, str], cache_dir: Path | None) -> None:
        path = self._cache_path(cache_dir)
        if path:
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(cache, indent=2, ensure_ascii=False), encoding="utf-8")
            except Exception as exc:
                logger.warning("Failed to save character appearances cache: %s", exc)
