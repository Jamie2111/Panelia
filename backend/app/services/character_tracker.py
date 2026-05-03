from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np


class CharacterTracker:
    def __init__(self, similarity_threshold: float = 0.82) -> None:
        self.similarity_threshold = similarity_threshold

    def track(
        self,
        detections: list[dict[str, Any]],
        embeddings: list[np.ndarray],
    ) -> dict[str, Any]:
        if not detections:
            return {
                "characters": {},
                "source_to_character_id": {},
                "panel_characters": {},
                "provider": "embedding-tracker-v1",
            }

        source_to_character_id: dict[str, str] = {}
        characters: dict[str, dict[str, Any]] = {}
        panel_characters: dict[str, list[str]] = defaultdict(list)

        ordered = sorted(
            zip(detections, embeddings, strict=False),
            key=lambda item: (
                int(item[0].get("page") or 0),
                int(item[0].get("panel_order") or 0),
                int(item[0].get("bbox", [0, 0, 0, 0])[1]),
                int(item[0].get("bbox", [0, 0, 0, 0])[0]),
            ),
        )

        for detection, embedding in ordered:
            source_id = str(detection.get("source_character_id") or "").strip()
            if not source_id:
                continue

            stable_id = source_to_character_id.get(source_id)
            if not stable_id:
                stable_id = self._match_existing_character(characters, embedding, source_id)
            if not stable_id:
                stable_id = f"Character_{len(characters) + 1}"
                characters[stable_id] = {
                    "id": stable_id,
                    "name": None,
                    "description": "",
                    "first_panel": int(detection.get("panel_order") or 0),
                    "first_page": int(detection.get("page") or 0),
                    "appearances": [],
                    "source_character_ids": set(),
                    "centroid": np.array(embedding, copy=True),
                    "embedding_count": 0,
                }

            entry = characters[stable_id]
            entry["source_character_ids"].add(source_id)
            entry["appearances"].append(
                {
                    "page": int(detection.get("page") or 0),
                    "panel": int(detection.get("panel_order") or 0),
                    "panel_id": str(detection.get("panel_id") or ""),
                    "bbox": [int(value) for value in detection.get("bbox", [])[:4]],
                }
            )
            entry["embedding_count"] = int(entry.get("embedding_count") or 0) + 1
            count = entry["embedding_count"]
            centroid = np.array(entry["centroid"], copy=False)
            entry["centroid"] = self._normalize_vector(((centroid * (count - 1)) + embedding) / max(count, 1))

            source_to_character_id[source_id] = stable_id
            panel_id = str(detection.get("panel_id") or "")
            if panel_id and stable_id not in panel_characters[panel_id]:
                panel_characters[panel_id].append(stable_id)

        serialized_characters: dict[str, Any] = {}
        for stable_id, entry in characters.items():
            appearances = sorted(
                entry["appearances"],
                key=lambda item: (int(item.get("page") or 0), int(item.get("panel") or 0)),
            )
            serialized_characters[stable_id] = {
                "id": stable_id,
                "name": entry.get("name"),
                "description": str(entry.get("description") or ""),
                "first_panel": int(entry.get("first_panel") or 0),
                "first_page": int(entry.get("first_page") or 0),
                "appearances": appearances,
                "appearance_count": len(appearances),
                "source_character_ids": sorted(str(value) for value in entry.get("source_character_ids", set())),
            }

        return {
            "characters": serialized_characters,
            "source_to_character_id": source_to_character_id,
            "panel_characters": {key: value for key, value in panel_characters.items()},
            "provider": "embedding-tracker-v1",
        }

    def _match_existing_character(
        self,
        characters: dict[str, dict[str, Any]],
        embedding: np.ndarray,
        source_id: str,
    ) -> str | None:
        best_id: str | None = None
        best_score = -1.0
        for stable_id, entry in characters.items():
            if source_id in entry.get("source_character_ids", set()):
                return stable_id
            centroid = entry.get("centroid")
            if centroid is None:
                continue
            score = float(np.dot(np.array(centroid, copy=False), embedding))
            if score > best_score:
                best_score = score
                best_id = stable_id
        if best_id and best_score >= self.similarity_threshold:
            return best_id
        return None

    def _normalize_vector(self, vector: np.ndarray) -> np.ndarray:
        norm = float(np.linalg.norm(vector))
        if norm <= 0:
            return vector.astype(np.float32, copy=False)
        return (vector / norm).astype(np.float32, copy=False)
