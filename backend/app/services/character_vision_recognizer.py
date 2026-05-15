"""Character Vision Recognizer.

Uses Gemini Vision to identify and cluster recurring characters across all
manga panels, replacing the CLIP-based approach that fails on manga art styles
and non-English text.

Key differences from CharacterClusterer:
- Scans ALL kept panels, not just dialogue panels detected by Magi
- Uses multimodal Gemini calls: the LLM SEES the actual art
- Works on any language - Gemini reads the images directly
- No CLIP model to load; falls back gracefully if Gemini is unavailable
"""
from __future__ import annotations

import asyncio
import base64
import concurrent.futures
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

from app.schemas.project import PanelBox
from app.services.llm_router import LLMRouter

logger = logging.getLogger(__name__)

_CACHE_FILE = "vision_character_cache.json"
_BATCH_SIZE = 10  # panels per Gemini Vision call
_VISION_WORKERS = 3  # parallel Gemini Vision calls
_IMAGE_MAX_PX = 640  # resize panels before encoding to keep request size manageable

# Hair colour groups for disambiguation.
# Two characters with hair colours from DIFFERENT groups cannot be the same person.
# Colours within a group are interchangeable (manga art often renders black/dark-brown
# similarly depending on panel shading).
_HAIR_COLOR_GROUPS: dict[str, str] = {
    # dark naturals - often look the same in manga ink/tone
    "black": "dark", "brown": "dark", "dark": "dark", "brunette": "dark",
    # light naturals
    "blonde": "light", "blond": "light", "golden": "light", "light": "light",
    # achromatic
    "white": "achromatic", "silver": "achromatic", "gray": "achromatic",
    "grey": "achromatic", "platinum": "achromatic",
    # vibrant - unambiguous
    "red": "vibrant-red", "blue": "vibrant-blue", "green": "vibrant-green",
    "purple": "vibrant-purple", "pink": "vibrant-pink", "orange": "vibrant-orange",
}

# Stopwords excluded from keyword-overlap similarity.
# Generic clothing items and build descriptors are excluded because they appear
# across many characters and don't help distinguish identities.
_STOPWORDS: frozenset[str] = frozenset(
    {
        # articles / prepositions
        "a", "an", "the", "and", "or", "with", "in", "on", "at", "of", "to",
        # verbs
        "is", "are", "has", "have", "wearing", "appears", "looks", "seems",
        "holds", "holding", "standing", "sitting", "lying",
        # generic descriptors
        "character", "person", "figure", "male", "female", "man", "woman",
        "young", "old", "adult", "middle", "aged", "build", "type", "style",
        "slim", "thin", "tall", "short", "long", "large", "small", "big",
        "athletic", "muscular", "stocky",
        # generic clothing items (don't distinguish characters)
        "jacket", "coat", "shirt", "pants", "jeans", "dress", "skirt",
        "hoodie", "vest", "suit", "outfit", "clothes", "clothing", "top",
        "uniform", "robe", "cape", "armor", "sweater", "jumper",
        # accessories so common they're not distinguishing
        "hair", "eyes",
    }
)


class GeminiCharacterRecognizer:
    """Identifies recurring manga characters using Gemini Vision.

    Scans all kept panel images in batches, asks Gemini to identify every
    distinct character with visual descriptions, then clusters those descriptions
    across panels to produce a ``character_clusters`` list compatible with the
    rest of the pipeline.
    """

    def __init__(self, router: LLMRouter | None = None) -> None:
        self.router = router or LLMRouter()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def is_available(self) -> bool:
        """True if at least one LLM provider is configured."""
        return bool(self.router.available_providers())

    def load_canonical_characters(self, project_dir: Path) -> list[dict[str, Any]]:
        """Vision-first source of truth for canonical characters.

        This lets newer vision-first runs ignore the legacy OCR-derived
        character dictionary while leaving the older recognize() path intact.
        """
        path = project_dir / "output" / "canonical_characters.json"
        if not path.exists():
            return []
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            logger.warning("Failed to read canonical_characters.json from %s", path)
            return []
        if not isinstance(payload, list):
            return []
        cleaned: list[dict[str, Any]] = []
        for item in payload:
            if not isinstance(item, dict):
                continue
            name = str(item.get("name") or "").strip()
            if not name:
                continue
            cleaned.append(
                {
                    "stable_id": str(item.get("stable_id") or "").strip(),
                    "name": name,
                    "role": str(item.get("role") or "").strip(),
                    "visual_description": str(item.get("visual_description") or "").strip(),
                    "portrait_panel_ids": [
                        str(panel_id).strip()
                        for panel_id in item.get("portrait_panel_ids", []) or []
                        if str(panel_id).strip()
                    ],
                }
            )
        return cleaned

    def recognize(
        self,
        page_paths: list[Path],
        page_payloads: dict[int, dict[str, Any]],
        panels: list[PanelBox],
        *,
        panel_image_dir: Path | None = None,
        cache_dir: Path | None = None,
        cancel_callback: Any = None,
    ) -> dict[str, Any]:
        """Identify and cluster characters using Gemini Vision.

        Compatible with :meth:`CharacterClusterer.cluster` - returns the same
        ``{page_payloads, clusters, character_id_map, provider}`` shape.

        Args:
            page_paths: Ordered list of page image paths (1-indexed).
            page_payloads: Magi speaker detection output.  May be empty.
            panels: All PanelBox objects (kept + dropped).
            panel_image_dir: Directory containing ``panel_NNN.png`` crops.
            cache_dir: Directory for caching Gemini responses.
            cancel_callback: Called periodically to check for cancellation.

        Returns:
            Dict with keys: page_payloads, clusters, character_id_map, provider.
        """
        kept = sorted(
            (p for p in panels if p.keep),
            key=lambda p: p.order,
        )
        if not kept:
            return self._empty_result(page_payloads)

        cache = self._load_cache(cache_dir)

        # Step 1 - per-batch panel scan (parallel when multiple batches)
        raw_appearances: list[dict[str, Any]] = []
        batches = [kept[i : i + _BATCH_SIZE] for i in range(0, len(kept), _BATCH_SIZE)]

        logger.info(
            "GeminiCharacterRecognizer: scanning %d batches across %d workers",
            len(batches),
            min(_VISION_WORKERS, len(batches)),
        )

        def _scan_one(idx_batch: tuple[int, list]) -> list[dict[str, Any]]:
            batch_index, batch = idx_batch
            if cancel_callback:
                cancel_callback()
            logger.info(
                "GeminiCharacterRecognizer: scanning batch %d/%d (%d panels)",
                batch_index + 1,
                len(batches),
                len(batch),
            )
            return self._scan_batch(batch, panel_image_dir, page_paths, cache)

        if len(batches) <= 1:
            for result in map(_scan_one, enumerate(batches)):
                raw_appearances.extend(result)
        else:
            with concurrent.futures.ThreadPoolExecutor(max_workers=_VISION_WORKERS) as executor:
                for batch_results in executor.map(_scan_one, enumerate(batches)):
                    raw_appearances.extend(batch_results)

        self._save_cache(cache, cache_dir)

        if not raw_appearances:
            logger.warning("GeminiCharacterRecognizer: no characters detected in any panel")
            return self._empty_result(page_payloads)

        # Step 2 - cluster appearances by visual description similarity
        clusters = self._cluster_appearances(raw_appearances)

        # Step 3 - sort by appearance count (protagonist first)
        clusters.sort(key=lambda c: (-c["appearance_count"], c["cluster_id"]))
        for rank, cluster in enumerate(clusters):
            cluster["cluster_id"] = f"cluster-{rank + 1:03d}"
            cluster["role_hint"] = "Protagonist" if rank == 0 else cluster.get("role_hint", "Character")

        logger.info(
            "GeminiCharacterRecognizer: found %d unique characters across %d panels",
            len(clusters),
            len(kept),
        )

        return {
            "page_payloads": page_payloads,
            "clusters": [self._serialize_cluster(c) for c in clusters],
            "character_id_map": {},
            "provider": "gemini-vision-v1",
        }

    # ------------------------------------------------------------------
    # Batch scanning
    # ------------------------------------------------------------------

    def _scan_batch(
        self,
        batch: list[PanelBox],
        panel_image_dir: Path | None,
        page_paths: list[Path],
        cache: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """Send one batch of panels to Gemini and return per-panel character lists."""
        panel_images: list[tuple[PanelBox, bytes, str]] = []  # (panel, img_bytes, mime)
        cache_keys: list[str | None] = []

        for panel in batch:
            img_bytes, mime = self._load_panel_image(panel, panel_image_dir, page_paths)
            if img_bytes:
                cache_key = hashlib.md5(img_bytes).hexdigest()
            else:
                cache_key = None
            panel_images.append((panel, img_bytes, mime))
            cache_keys.append(cache_key)

        # Check if all panels are cached
        batch_results: list[dict[str, Any] | None] = []
        all_cached = True
        for cache_key in cache_keys:
            if cache_key and cache_key in cache:
                batch_results.append(cache.get(cache_key))
            else:
                all_cached = False
                batch_results.append(None)

        if all_cached and all(r is not None for r in batch_results):
            # Merge cached results
            appearances: list[dict[str, Any]] = []
            for idx, (panel, result) in enumerate(zip(batch, batch_results)):
                if result:
                    appearances.extend(self._result_to_appearances(panel, result))
            return appearances

        # Build multimodal prompt
        prompt = self._build_batch_prompt(batch)
        parts: list[dict[str, Any]] = [{"text": prompt}]
        included_panels: list[PanelBox] = []

        for panel, img_bytes, mime in panel_images:
            if img_bytes:
                parts.append({"text": f"Panel {panel.order}:"})
                parts.append({
                    "inlineData": {
                        "mimeType": mime,
                        "data": base64.b64encode(img_bytes).decode("utf-8"),
                    },
                })
                included_panels.append(panel)

        if not included_panels:
            return []

        try:
            result = asyncio.run(
                self.router._route_json(
                    task_name="character vision scan",
                    prompt=prompt,
                    validator=self._validate_batch_response,
                    max_output_tokens=min(4096, max(512, 200 * len(included_panels))),
                    parts=parts,
                )
            )
            response = result.payload
        except Exception as exc:
            logger.warning("GeminiCharacterRecognizer batch failed: %s", exc)
            return []

        # Cache per-panel results
        panel_char_map: dict[int, list[dict]] = {}
        for entry in response.get("panel_characters", []):
            order = int(entry.get("panel_order") or 0)
            chars = entry.get("characters") or []
            if order:
                panel_char_map[order] = chars

        appearances: list[dict[str, Any]] = []
        for panel, img_bytes, mime in panel_images:
            chars = panel_char_map.get(panel.order, [])
            # Store per-panel result in cache
            if img_bytes:
                cache_key = hashlib.md5(img_bytes).hexdigest()
                cache[cache_key] = {"panel_order": panel.order, "characters": chars}
            panel_result = {"panel_order": panel.order, "characters": chars}
            appearances.extend(self._result_to_appearances(panel, panel_result))

        return appearances

    def _result_to_appearances(
        self,
        panel: PanelBox,
        result: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """Convert a cached per-panel result to appearance dicts."""
        appearances = []
        for char in result.get("characters") or []:
            desc = str(char.get("description") or "").strip()
            if not desc or len(desc) < 8:
                continue
            appearances.append(
                {
                    "panel_order": panel.order,
                    "panel_id": str(panel.id),
                    "page": int(panel.page),
                    "description": desc,
                    "is_protagonist": bool(char.get("is_protagonist", False)),
                }
            )
        return appearances

    # ------------------------------------------------------------------
    # Prompts
    # ------------------------------------------------------------------

    def _build_batch_prompt(self, panels: list[PanelBox]) -> str:
        panel_list = ", ".join(str(p.order) for p in panels)
        return f"""You are identifying recurring characters in manga/manhwa panels.

I will show you panels {panel_list}. For each panel, identify every visually distinct character whose face or distinctive features are clearly visible.

RULES:
- Skip blurry background crowd members (only silhouettes visible).
- If the SAME character appears in multiple panels in this batch, give them the SAME batch_id.
- Focus on: hair color/style, gender, build, clothing, and any unique feature (scar, glasses, mask, etc.)
- Be specific: "short black hair" not just "dark hair"
- The protagonist (main character) usually appears in the most panels.

Return JSON only - no extra text:
{{
  "panel_characters": [
    {{
      "panel_order": <integer>,
      "characters": [
        {{
          "batch_id": "A",
          "description": "young adult male, short black hair, gray school uniform, athletic build",
          "is_protagonist": true
        }}
      ]
    }}
  ]
}}

If a panel has no clearly visible characters, return an empty characters list for it.
"""

    # ------------------------------------------------------------------
    # Response validation
    # ------------------------------------------------------------------

    def _validate_batch_response(self, payload: Any) -> dict[str, Any]:
        if not isinstance(payload, dict):
            raise ValueError("Response is not a JSON object")
        panel_chars = payload.get("panel_characters")
        if not isinstance(panel_chars, list):
            raise ValueError("Missing panel_characters list")
        validated: list[dict] = []
        for entry in panel_chars:
            if not isinstance(entry, dict):
                continue
            order = entry.get("panel_order")
            chars = entry.get("characters")
            if order is None or not isinstance(chars, list):
                continue
            clean_chars = []
            for char in chars:
                if not isinstance(char, dict):
                    continue
                desc = str(char.get("description") or "").strip()
                if len(desc) < 8:
                    continue
                clean_chars.append(
                    {
                        "batch_id": str(char.get("batch_id") or ""),
                        "description": desc,
                        "is_protagonist": bool(char.get("is_protagonist", False)),
                    }
                )
            validated.append({"panel_order": int(order), "characters": clean_chars})
        return {"panel_characters": validated}

    # ------------------------------------------------------------------
    # Cross-panel clustering
    # ------------------------------------------------------------------

    def _cluster_appearances(
        self,
        appearances: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Group appearances by visual description similarity.

        Uses a greedy nearest-neighbour approach: each new appearance is added
        to the best-matching existing cluster (if score ≥ threshold) or starts a
        new one.  The threshold is intentionally loose (0.30) because manga
        characters are drawn from many angles and descriptions vary across panels.
        """
        THRESHOLD = 0.15
        clusters: list[dict[str, Any]] = []

        for app in appearances:
            desc = app["description"]
            best_idx: int | None = None
            best_score = -1.0

            for idx, cluster in enumerate(clusters):
                score = self._description_similarity(desc, cluster["canonical_description"])
                if score > best_score:
                    best_score = score
                    best_idx = idx

            if best_idx is not None and best_score >= THRESHOLD:
                cluster = clusters[best_idx]
                cluster["appearances"].append(app)
                cluster["appearance_count"] += 1
                cluster["pages"].add(int(app["page"]))
                cluster["panel_orders"].add(int(app["panel_order"]))
                cluster["panel_ids"].add(str(app["panel_id"]))
                if app.get("is_protagonist"):
                    cluster["is_protagonist"] = True
                # Prefer longer / more detailed descriptions as canonical
                if len(desc) > len(cluster["canonical_description"]):
                    cluster["canonical_description"] = desc
            else:
                clusters.append(
                    {
                        "cluster_id": f"cluster-{len(clusters) + 1:03d}",
                        "canonical_description": desc,
                        "is_protagonist": bool(app.get("is_protagonist", False)),
                        "appearances": [app],
                        "appearance_count": 1,
                        "pages": {int(app["page"])},
                        "panel_orders": {int(app["panel_order"])},
                        "panel_ids": {str(app["panel_id"])},
                    }
                )

        # Merge protagonist clusters: there's only one protagonist per manga,
        # so if the same character got split across batches both marked as
        # protagonist, collapse them into the largest cluster.
        clusters = self._merge_protagonist_clusters(clusters)

        # Remove singletons (likely background characters glimpsed once).
        # Keep if protagonist or appears on multiple pages.
        filtered = [
            c
            for c in clusters
            if c["appearance_count"] >= 2
            or c.get("is_protagonist")
            or len(c["pages"]) >= 2
        ]
        return filtered if filtered else clusters

    def _merge_protagonist_clusters(
        self, clusters: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Collapse multiple protagonist-flagged clusters into one.

        Across batches the same protagonist may get slightly different descriptions
        and end up in separate clusters both marked ``is_protagonist=True``.  Since
        a manga has exactly one protagonist, we force-merge them.
        """
        protagonist_clusters = [c for c in clusters if c.get("is_protagonist")]
        non_protagonist = [c for c in clusters if not c.get("is_protagonist")]

        if len(protagonist_clusters) <= 1:
            return clusters

        # Keep the cluster with the most detailed (longest) description as
        # the canonical one.
        primary = max(
            protagonist_clusters,
            key=lambda c: (c["appearance_count"], len(c["canonical_description"])),
        )
        for other in protagonist_clusters:
            if other is primary:
                continue
            primary["appearances"].extend(other["appearances"])
            primary["appearance_count"] += other["appearance_count"]
            primary["pages"].update(other["pages"])
            primary["panel_orders"].update(other["panel_orders"])
            primary["panel_ids"].update(other["panel_ids"])
            if len(other["canonical_description"]) > len(primary["canonical_description"]):
                primary["canonical_description"] = other["canonical_description"]

        return [primary] + non_protagonist

    def _description_similarity(self, desc1: str, desc2: str) -> float:
        """Jaccard-like similarity with a context-aware hair-colour group veto.

        Only considers hair colour when the colour word appears adjacent to "hair"
        in the description (e.g. "black hair" or "hair is silver").  Generic colour
        words in clothing phrases ("white jacket") do NOT count as hair colour.

        Returns 0.0 if the two descriptions have *different* hair colour groups.
        Otherwise returns the Jaccard index of content-word overlap (stopwords
        removed).
        """
        words1 = self._content_words(desc1)
        words2 = self._content_words(desc2)

        if not words1 or not words2:
            return 0.0

        # Context-aware hair colour extraction
        group1 = self._hair_color_group(desc1)
        group2 = self._hair_color_group(desc2)
        if group1 and group2 and group1 != group2:
            return 0.0

        intersection = len(words1 & words2)
        union = len(words1 | words2)
        return intersection / union if union else 0.0

    def _hair_color_group(self, description: str) -> str | None:
        """Return the hair-colour group mentioned near 'hair', or None."""
        desc_lower = description.casefold()
        # Patterns: "COLOR hair", "COLOR-COLOR hair", or "hair COLOR"
        for match in re.finditer(
            r"(\w+(?:-\w+)?)\s+hair|hair\s+(?:is\s+)?(\w+)",
            desc_lower,
        ):
            color_word = (match.group(1) or match.group(2) or "").lower()
            # Strip compound modifiers - take last word ("dark-brown" → "brown")
            color_word = color_word.split("-")[-1]
            if color_word in _HAIR_COLOR_GROUPS:
                return _HAIR_COLOR_GROUPS[color_word]
        return None

    def _content_words(self, text: str) -> set[str]:
        tokens = set(re.findall(r"[a-z]+", text.casefold()))
        return tokens - _STOPWORDS

    # ------------------------------------------------------------------
    # Output serialization
    # ------------------------------------------------------------------

    def _serialize_cluster(self, cluster: dict[str, Any]) -> dict[str, Any]:
        pages = sorted(cluster["pages"])
        panels = sorted(cluster["panel_orders"])
        panel_ids = sorted(cluster["panel_ids"])
        # Rich per-appearance data so downstream can build correct sample images
        # (each appearance gets its own page, not just pages[0]).
        rich_appearances = [
            {
                "page": int(app.get("page") or 0),
                "panel_order": int(app.get("panel_order") or 0),
                "panel_id": str(app.get("panel_id") or ""),
                "bbox": [],
            }
            for app in cluster.get("appearances", [])
            if app.get("panel_id")
        ]
        return {
            "cluster_id": cluster["cluster_id"],
            "pages": pages,
            "panels": panels,
            "panel_ids": panel_ids,
            "dialogues": [],
            "role_hint": cluster.get("role_hint") or ("Protagonist" if cluster.get("is_protagonist") else "Character"),
            "appearance_count": cluster["appearance_count"],
            "description": cluster.get("canonical_description", ""),
            "_appearances": rich_appearances,
        }

    def _empty_result(self, page_payloads: dict[int, dict[str, Any]]) -> dict[str, Any]:
        return {
            "page_payloads": page_payloads,
            "clusters": [],
            "character_id_map": {},
            "provider": "gemini-vision-v1",
        }

    # ------------------------------------------------------------------
    # Image loading
    # ------------------------------------------------------------------

    def _load_panel_image(
        self,
        panel: PanelBox,
        panel_image_dir: Path | None,
        page_paths: list[Path],
    ) -> tuple[bytes | None, str]:
        """Load the panel image, resize if needed, and return (bytes, mime)."""
        img_path = self._find_panel_image(panel, panel_image_dir)
        if img_path and img_path.exists():
            return self._encode_image(img_path)
        # Fall back to cropping from the source page
        page_idx = int(panel.page) - 1
        if 0 <= page_idx < len(page_paths) and page_paths[page_idx].exists():
            return self._encode_panel_crop(panel, page_paths[page_idx])
        return None, ""

    def _find_panel_image(self, panel: PanelBox, image_dir: Path | None) -> Path | None:
        if not image_dir:
            return None
        for name in (
            f"panel_{int(panel.order):03d}.png",
            f"panel_{int(panel.order):03d}.jpg",
            f"{panel.id}.png",
            f"{panel.id}.jpg",
        ):
            candidate = image_dir / name
            if candidate.exists():
                return candidate
        return None

    def _encode_image(self, path: Path) -> tuple[bytes | None, str]:
        """Resize (if needed) and encode an image file."""
        if _PIL_AVAILABLE:
            try:
                with _PILImage.open(path) as img:
                    if img.mode not in ("RGB", "L"):
                        img = img.convert("RGB")
                    w, h = img.size
                    if w > _IMAGE_MAX_PX or h > _IMAGE_MAX_PX:
                        scale = _IMAGE_MAX_PX / max(w, h)
                        img = img.resize(
                            (max(1, int(w * scale)), max(1, int(h * scale))),
                            _PILImage.LANCZOS,
                        )
                    buf = io.BytesIO()
                    img.save(buf, format="JPEG", quality=82, optimize=True)
                    return buf.getvalue(), "image/jpeg"
            except Exception as exc:
                logger.debug("PIL encode failed for %s: %s", path, exc)

        # Raw fallback (skip if too large)
        raw = path.read_bytes()
        if len(raw) > 600 * 1024:
            return None, ""
        mime = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
        return raw, mime

    def _encode_panel_crop(
        self,
        panel: PanelBox,
        page_path: Path,
    ) -> tuple[bytes | None, str]:
        """Crop a panel region from its source page and encode it."""
        if not _PIL_AVAILABLE:
            return None, ""
        try:
            with _PILImage.open(page_path) as page_img:
                page_img = page_img.convert("RGB")
                x, y, w, h = int(panel.x), int(panel.y), int(panel.width), int(panel.height)
                pw, ph = page_img.size
                x0 = max(0, min(x, pw - 1))
                y0 = max(0, min(y, ph - 1))
                x1 = max(x0 + 1, min(x + w, pw))
                y1 = max(y0 + 1, min(y + h, ph))
                crop = page_img.crop((x0, y0, x1, y1))
                cw, ch = crop.size
                if cw > _IMAGE_MAX_PX or ch > _IMAGE_MAX_PX:
                    scale = _IMAGE_MAX_PX / max(cw, ch)
                    crop = crop.resize(
                        (max(1, int(cw * scale)), max(1, int(ch * scale))),
                        _PILImage.LANCZOS,
                    )
                buf = io.BytesIO()
                crop.save(buf, format="JPEG", quality=82, optimize=True)
                return buf.getvalue(), "image/jpeg"
        except Exception as exc:
            logger.debug("Panel crop failed for panel %s: %s", panel.order, exc)
            return None, ""

    # ------------------------------------------------------------------
    # Cache
    # ------------------------------------------------------------------

    def _cache_path(self, cache_dir: Path | None) -> Path | None:
        return (cache_dir / _CACHE_FILE) if cache_dir else None

    def _load_cache(self, cache_dir: Path | None) -> dict[str, Any]:
        path = self._cache_path(cache_dir)
        if path and path.exists():
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                pass
        return {}

    def _save_cache(self, cache: dict[str, Any], cache_dir: Path | None) -> None:
        path = self._cache_path(cache_dir)
        if not path:
            return
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps(cache, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        except Exception as exc:
            logger.warning("GeminiCharacterRecognizer: failed to save cache: %s", exc)
