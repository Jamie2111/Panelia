from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from app.services.embedding_encoder import EmbeddingEncoder


class CharacterIdentityTracker:
    def __init__(self) -> None:
        self._encoder = EmbeddingEncoder()

    def refine(
        self,
        page_paths: list[Path],
        page_payloads: dict[int, dict[str, Any]],
        character_clusters: list[dict[str, Any]],
        character_memory: dict[str, Any],
        raw_regions: list[Any],
        protagonist_name: str | None = None,
    ) -> tuple[dict[str, Any], list[Any], dict[str, Any]]:
        source_to_character_id = character_memory.get("source_to_character_id", {}) if isinstance(character_memory, dict) else {}
        characters = character_memory.get("characters", {}) if isinstance(character_memory, dict) else {}
        if not source_to_character_id or not characters:
            return character_memory, raw_regions, {"identities": [], "summary": "No character identities were available to refine."}

        detections: list[dict[str, Any]] = []
        for page_number, payload in sorted(page_payloads.items(), key=lambda item: int(item[0])):
            for character in payload.get("characters", []) or []:
                character_id = str(character.get("character_id") or "").strip()
                bbox = character.get("bbox")
                if not character_id or not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
                    continue
                detections.append(
                    {
                        "source_character_id": character_id,
                        "page": int(page_number),
                        "bbox": [int(round(float(value))) for value in bbox[:4]],
                        "panel_id": "",
                    }
                )

        embeddings = self._encoder.encode(page_paths, detections)
        image_cache: dict[int, Image.Image] = {}
        signatures: dict[str, dict[str, Any]] = defaultdict(lambda: {
            "embedding_vectors": [],
            "hair_colors": [],
            "clothing_colors": [],
            "pages": [],
            "samples": 0,
        })

        for detection, embedding in zip(detections, embeddings, strict=False):
            source_id = str(detection.get("source_character_id") or "").strip()
            stable_id = str(source_to_character_id.get(source_id) or "").strip()
            if not stable_id:
                continue
            page = int(detection.get("page") or 0)
            bbox = detection.get("bbox") or [0, 0, 0, 0]
            image = image_cache.get(page)
            if image is None and 0 < page <= len(page_paths):
                image = Image.open(page_paths[page - 1]).convert("RGB")
                image_cache[page] = image
            if image is None:
                continue
            crop = self._crop_box(image, bbox)
            if crop is None:
                continue
            signature = signatures[stable_id]
            signature["embedding_vectors"].append(np.array(embedding, copy=True))
            signature["hair_colors"].append(self._dominant_band_color(crop, 0.0, 0.3))
            signature["clothing_colors"].append(self._dominant_band_color(crop, 0.45, 0.85))
            signature["pages"].append(page)
            signature["samples"] = int(signature["samples"]) + 1

        cluster_lookup = {str(cluster.get("cluster_id") or ""): cluster for cluster in character_clusters}
        identity_report: list[dict[str, Any]] = []
        for stable_id, memory in characters.items():
            source_ids = [str(value) for value in memory.get("source_character_ids", []) if str(value).strip()]
            cluster = next((cluster_lookup[source_id] for source_id in source_ids if source_id in cluster_lookup), {})
            signature = signatures.get(stable_id, {})
            explicit_name = self._explicit_name_from_regions(raw_regions, stable_id)
            refined_name = explicit_name or self._refine_name(
                current_name=str(memory.get("name") or "").strip(),
                protagonist_name=protagonist_name,
                role_hint=str(cluster.get("role_hint") or memory.get("role") or "").strip(),
                signature=signature,
            )
            narration_reference = refined_name or str(memory.get("narration_reference") or stable_id).strip()
            description = refined_name or self._descriptive_label(
                role_hint=str(cluster.get("role_hint") or memory.get("role") or "").strip(),
                signature=signature,
            )
            characters[stable_id] = {
                **memory,
                "name": refined_name or memory.get("name"),
                "display_name": refined_name or str(memory.get("display_name") or stable_id).strip(),
                "description": description,
                "narration_reference": narration_reference,
                "identity_signature": {
                    "hair_color": self._majority_color_name(signature.get("hair_colors", [])),
                    "clothing_color": self._majority_color_name(signature.get("clothing_colors", [])),
                    "appearance_confidence": self._appearance_confidence(signature),
                    "sample_count": int(signature.get("samples") or 0),
                },
            }
            identity_report.append(
                {
                    "stable_id": stable_id,
                    "display_name": characters[stable_id]["display_name"],
                    "hair_color": self._majority_color_name(signature.get("hair_colors", [])),
                    "clothing_color": self._majority_color_name(signature.get("clothing_colors", [])),
                    "sample_count": int(signature.get("samples") or 0),
                }
            )

        for region in raw_regions:
            stable_id = str(getattr(region, "stable_character_id", "") or "").strip()
            if not stable_id or stable_id not in characters:
                continue
            display_name = str(characters[stable_id].get("display_name") or stable_id).strip()
            setattr(region, "character_display_name", display_name)
            if not str(getattr(region, "speaker_name", "") or "").strip():
                setattr(region, "speaker_name", display_name)

        character_memory["characters"] = characters
        return character_memory, raw_regions, {
            "identities": identity_report,
            "summary": f"Refined {len(identity_report)} character identities with appearance continuity.",
        }

    def _explicit_name_from_regions(self, raw_regions: list[Any], stable_id: str) -> str:
        candidates: list[str] = []
        for region in raw_regions:
            if str(getattr(region, "stable_character_id", "") or "").strip() != stable_id:
                continue
            for source in (
                str(getattr(region, "speaker_name", "") or "").strip(),
                str(getattr(region, "speaker_label", "") or "").strip(),
                str(getattr(region, "character_display_name", "") or "").strip(),
            ):
                if self._is_named_identity(source):
                    candidates.append(source)
        return candidates[0] if candidates else ""

    def _refine_name(
        self,
        current_name: str,
        protagonist_name: str | None,
        role_hint: str,
        signature: dict[str, Any],
    ) -> str:
        if protagonist_name and current_name == protagonist_name:
            return protagonist_name
        if self._is_named_identity(current_name):
            return current_name
        label = self._descriptive_label(role_hint, signature)
        return "" if label.startswith("the ") else label

    def _descriptive_label(self, role_hint: str, signature: dict[str, Any]) -> str:
        role = role_hint.strip() or "character"
        hair = self._majority_color_name(signature.get("hair_colors", []))
        clothing = self._majority_color_name(signature.get("clothing_colors", []))
        role_noun = {
            "Protagonist": "protagonist",
            "Neighbor": "neighbor",
            "Friend": "friend",
            "Villain": "villain",
            "Stranger": "stranger",
            "Character": "character",
        }.get(role, role.casefold() or "character")
        if hair:
            return f"the {hair}-haired {role_noun}"
        if clothing:
            return f"the {clothing}-clad {role_noun}"
        return f"the {role_noun}"

    def _appearance_confidence(self, signature: dict[str, Any]) -> float:
        vectors = [np.array(value, copy=False) for value in signature.get("embedding_vectors", []) if isinstance(value, np.ndarray)]
        if len(vectors) < 2:
            return 0.55 if vectors else 0.0
        centroid = np.mean(vectors, axis=0)
        centroid_norm = float(np.linalg.norm(centroid))
        if centroid_norm > 0:
            centroid = centroid / centroid_norm
        similarities = [float(np.dot(vector, centroid)) for vector in vectors]
        adjacency_bonus = 0.08 if self._page_span(signature.get("pages", [])) <= 4 else 0.0
        return round(max(0.0, min(0.99, (sum(similarities) / len(similarities)) + adjacency_bonus)), 4)

    def _page_span(self, pages: list[int]) -> int:
        if not pages:
            return 0
        return max(pages) - min(pages)

    def _crop_box(self, image: Image.Image, bbox: list[int]) -> Image.Image | None:
        x, y, width, height = [int(value) for value in bbox[:4]]
        x0 = max(x, 0)
        y0 = max(y, 0)
        x1 = min(x + max(width, 1), image.size[0])
        y1 = min(y + max(height, 1), image.size[1])
        if x1 <= x0 or y1 <= y0:
            return None
        return image.crop((x0, y0, x1, y1))

    def _dominant_band_color(self, image: Image.Image, start_ratio: float, end_ratio: float) -> str:
        width, height = image.size
        if width <= 0 or height <= 0:
            return ""
        start = max(0, min(int(height * start_ratio), height - 1))
        end = max(start + 1, min(int(height * end_ratio), height))
        band = np.array(image.crop((0, start, width, end)).resize((32, 32)))
        if band.size == 0:
            return ""
        rgb = band.reshape(-1, 3).mean(axis=0)
        return self._color_name(tuple(float(value) for value in rgb))

    def _majority_color_name(self, colors: list[str]) -> str:
        filtered = [color for color in colors if color]
        if not filtered:
            return ""
        counts: dict[str, int] = defaultdict(int)
        for color in filtered:
            counts[color] += 1
        return max(counts.items(), key=lambda item: item[1])[0]

    def _color_name(self, rgb: tuple[float, float, float]) -> str:
        red, green, blue = rgb
        brightness = (red + green + blue) / 3
        if brightness < 50:
            return "dark"
        if brightness > 215:
            return "white"
        if red > 180 and green > 150 and blue < 120:
            return "blond"
        if red > 170 and green < 120 and blue < 120:
            return "red"
        if blue > red + 20 and blue > green + 20:
            return "blue"
        if green > red + 15 and green > blue + 15:
            return "green"
        if abs(red - green) < 18 and abs(green - blue) < 18:
            return "gray"
        if red > 120 and green > 90 and blue < 100:
            return "brown"
        return "black" if brightness < 95 else "silver"

    def _is_named_identity(self, value: str) -> bool:
        cleaned = " ".join(value.split()).strip()
        if not cleaned:
            return False
        if cleaned.casefold() in {"other", "protagonist", "stranger", "character"}:
            return False
        if cleaned.startswith("Character_") or cleaned.startswith("Stranger"):
            return False
        return bool(re.search(r"[A-Z][a-z]+", cleaned) or re.search(r"[\u3040-\u30ff\u3400-\u9fff\uac00-\ud7af]", cleaned))
