from __future__ import annotations

import asyncio
import logging
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from threading import Lock
from typing import Any

import numpy as np
from PIL import Image

from app.core.config import get_settings
from app.schemas.project import ChapterMetadata, PanelBox
from app.services.character_name_filters import is_valid_character_name_candidate, looks_like_false_character_name
from app.services.character_name_service import CharacterNameService
from app.services.llm_router import LLMRouter, LLMRouterError
from app.services.ocr_cleaner import clean_ocr_text

logger = logging.getLogger(__name__)


class CharacterClusterer:
    _CLIP_MODEL: Any | None = None
    _CLIP_PROCESSOR: Any | None = None
    _CLIP_DEVICE = "cpu"
    _LOAD_LOCK = Lock()

    def __init__(self) -> None:
        self.settings = get_settings()
        self._name_service = CharacterNameService()
        self._similarity_threshold = 0.84
        self._active_similarity_threshold = self._similarity_threshold

    def cluster(
        self,
        page_paths: list[Path],
        page_payloads: dict[int, dict[str, Any]],
        panels: list[PanelBox],
        cancel_callback: callable | None = None,
    ) -> dict[str, Any]:
        panel_lookup: dict[int, list[PanelBox]] = defaultdict(list)
        for panel in sorted((panel for panel in panels if panel.keep), key=lambda item: item.order):
            panel_lookup[int(panel.page)].append(panel)

        samples: list[dict[str, Any]] = []
        image_cache: dict[int, Image.Image] = {}
        for page_number in sorted(page_payloads):
            page_panels = panel_lookup.get(int(page_number), [])
            if not page_panels:
                continue
            payload = page_payloads.get(page_number) or {}
            for character in payload.get("characters", []) or []:
                if cancel_callback:
                    cancel_callback()
                local_character_id = str(character.get("character_id") or "").strip()
                bbox = self._coerce_bbox(character.get("bbox"))
                if not local_character_id or bbox is None:
                    continue
                associated_panels = self._panels_for_character(bbox, page_panels)
                if not associated_panels:
                    continue
                image = image_cache.get(page_number)
                if image is None:
                    try:
                        image = Image.open(page_paths[page_number - 1]).convert("RGB")
                    except Exception:
                        continue
                    image_cache[page_number] = image
                crop = self._crop_character(image, bbox)
                if crop is None:
                    continue
                samples.append(
                    {
                        "page": int(page_number),
                        "local_character_id": local_character_id,
                        "bbox": bbox,
                        "image": crop,
                        "panel_ids": [panel.id for panel in associated_panels],
                        "panels": [int(panel.order) for panel in associated_panels],
                    }
                )

        if not samples:
            return {
                "page_payloads": page_payloads,
                "clusters": [],
                "character_id_map": {},
                "provider": "clip-greedy-v1",
            }

        embeddings = self._embed_samples(samples)
        cluster_entries: list[dict[str, Any]] = []
        character_id_map: dict[str, str] = {}

        for sample, embedding in zip(samples, embeddings, strict=False):
            cluster_index = self._assign_cluster(cluster_entries, embedding)
            if cluster_index is None:
                cluster_index = len(cluster_entries)
                cluster_entries.append(
                    {
                        "cluster_id": f"cluster-{cluster_index + 1:03d}",
                        "centroid": embedding.copy(),
                        "count": 1,
                        "pages": {int(sample["page"])},
                        "panels": set(int(panel) for panel in sample["panels"]),
                        "panel_ids": set(str(panel_id) for panel_id in sample["panel_ids"]),
                        "local_character_ids": {str(sample["local_character_id"])},
                        "appearances": [],
                        "dialogues": [],
                        "role_hint": "",
                    }
                )
            else:
                entry = cluster_entries[cluster_index]
                entry["centroid"] = self._normalize_vector((entry["centroid"] * entry["count"]) + embedding)
                entry["count"] += 1
                entry["pages"].add(int(sample["page"]))
                entry["panels"].update(int(panel) for panel in sample["panels"])
                entry["panel_ids"].update(str(panel_id) for panel_id in sample["panel_ids"])
                entry["local_character_ids"].add(str(sample["local_character_id"]))

            cluster_entries[cluster_index]["appearances"].append(
                {
                    "page": int(sample["page"]),
                    "bbox": [int(value) for value in sample["bbox"]],
                    "panel_ids": [str(panel_id) for panel_id in sample["panel_ids"]],
                    "panels": [int(panel) for panel in sample["panels"]],
                    "local_character_id": str(sample["local_character_id"]),
                }
            )

            cluster_id = cluster_entries[cluster_index]["cluster_id"]
            character_id_map[str(sample["local_character_id"])] = cluster_id

        updated_payloads: dict[int, dict[str, Any]] = {}
        for page_number, payload in page_payloads.items():
            texts = []
            for text in payload.get("texts", []) or []:
                updated = dict(text)
                local_character_id = str(updated.get("character_id") or "").strip()
                if local_character_id and local_character_id in character_id_map:
                    updated["character_id"] = character_id_map[local_character_id]
                texts.append(updated)

            characters = []
            for character in payload.get("characters", []) or []:
                updated = dict(character)
                local_character_id = str(updated.get("character_id") or "").strip()
                if local_character_id and local_character_id in character_id_map:
                    updated["character_id"] = character_id_map[local_character_id]
                characters.append(updated)

            updated_payloads[int(page_number)] = {
                **payload,
                "texts": texts,
                "characters": characters,
            }

        clusters = [
            {
                "cluster_id": entry["cluster_id"],
                "pages": sorted(entry["pages"]),
                "panels": sorted(entry["panels"]),
                "panel_ids": sorted(entry["panel_ids"]),
                "appearances": sorted(
                    entry["appearances"],
                    key=lambda item: (
                        int(item.get("page") or 0),
                        min((int(value) for value in item.get("panels", []) or [0]), default=0),
                        str(item.get("local_character_id") or ""),
                    ),
                ),
                "dialogues": [],
                "role_hint": "",
                "appearance_count": int(entry["count"]),
            }
            for entry in cluster_entries
        ]
        return {
            "page_payloads": updated_payloads,
            "clusters": clusters,
            "character_id_map": character_id_map,
            "provider": "clip-greedy-v1",
        }

    def attach_dialogues(
        self,
        clusters: list[dict[str, Any]],
        raw_regions: list[Any],
        protagonist_name: str | None = None,
    ) -> list[dict[str, Any]]:
        cluster_lookup = {str(cluster.get("cluster_id") or ""): dict(cluster) for cluster in clusters}
        for cluster in cluster_lookup.values():
            cluster["dialogues"] = []
            cluster["dialogue_count"] = 0

        for region in raw_regions:
            cluster_id = str(getattr(region, "character_id", "") or "").strip()
            if not cluster_id or cluster_id not in cluster_lookup:
                continue
            text = clean_ocr_text(str(getattr(region, "text_english", "") or getattr(region, "text_original", "") or "")).strip()
            if not text:
                continue
            cluster = cluster_lookup[cluster_id]
            if text not in cluster["dialogues"]:
                cluster["dialogues"].append(text)
            cluster["dialogue_count"] = int(cluster.get("dialogue_count") or 0) + 1

        ordered_clusters = sorted(
            cluster_lookup.values(),
            key=lambda item: (-int(item.get("dialogue_count") or 0), -int(item.get("appearance_count") or 0), str(item.get("cluster_id") or "")),
        )
        for index, cluster in enumerate(ordered_clusters, start=1):
            cluster["role_hint"] = self._role_hint(cluster, protagonist_name, index)
        return ordered_clusters

    def resolve_names(
        self,
        clusters: list[dict[str, Any]],
        metadata: ChapterMetadata,
        character_dictionary: dict[str, str],
        protagonist_name: str | None,
        router: LLMRouter | None = None,
    ) -> dict[str, str]:
        resolved: dict[str, str] = {}
        unresolved_payloads: list[dict[str, Any]] = []
        unresolved_clusters: dict[str, dict[str, Any]] = {}

        for cluster in clusters:
            cluster_id = str(cluster.get("cluster_id") or "").strip()
            if not cluster_id:
                continue
            direct_name = self._direct_cluster_name(cluster, character_dictionary, protagonist_name)
            if direct_name:
                resolved[cluster_id] = direct_name
                continue
            unresolved_clusters[cluster_id] = cluster
            unresolved_payloads.append(
                {
                    "cluster_id": cluster_id,
                    "panels": cluster.get("panels", []),
                    "dialogues": cluster.get("dialogues", [])[:8],
                    "role_hint": cluster.get("role_hint") or "",
                }
            )

        unresolved_limit = max(int(self.settings.character_name_resolution_limit or 0), 0)
        if (
            unresolved_payloads
            and router
            and router.available_providers()
            and (unresolved_limit == 0 or len(unresolved_payloads) <= unresolved_limit)
        ):
            try:
                result = asyncio.run(
                    router.resolve_character_names(
                        unresolved_payloads,
                        {
                            "metadata": self._metadata_payload(metadata),
                            "character_dictionary": character_dictionary,
                            "protagonist_name": protagonist_name or "",
                        },
                    )
                )
                for item in result.payload.get("characters", []):
                    cluster_id = str(item.get("cluster") or "").strip()
                    if not cluster_id or cluster_id not in unresolved_clusters or cluster_id in resolved:
                        continue
                    resolved_name = self._sanitize_name(
                        str(item.get("name") or "").strip(),
                        unresolved_clusters[cluster_id],
                        protagonist_name,
                        len(resolved) + 1,
                    )
                    if resolved_name:
                        resolved[cluster_id] = resolved_name
            except LLMRouterError as exc:
                logger.warning("Character cluster name resolution fell back to local labels: %s", exc)
        elif unresolved_payloads and unresolved_limit and len(unresolved_payloads) > unresolved_limit:
            logger.info(
                "Skipping LLM cluster name resolution for %s unresolved clusters because it exceeds the configured limit of %s",
                len(unresolved_payloads),
                unresolved_limit,
            )

        role_counts: Counter[str] = Counter()
        for cluster in clusters:
            cluster_id = str(cluster.get("cluster_id") or "").strip()
            if not cluster_id or cluster_id in resolved:
                continue
            fallback = self._fallback_cluster_name(cluster, protagonist_name, role_counts)
            if fallback:
                role_counts[fallback] += 1
                resolved[cluster_id] = fallback

        return resolved

    def _direct_cluster_name(
        self,
        cluster: dict[str, Any],
        character_dictionary: dict[str, str],
        protagonist_name: str | None,
    ) -> str:
        dialogue_text = " ".join(str(line).strip() for line in cluster.get("dialogues", []) if str(line).strip())
        if dialogue_text:
            extracted = []
            for line in cluster.get("dialogues", []) or []:
                extracted.extend(self._extract_self_identified_names(str(line or "")))
            if extracted:
                return extracted[0]
        if protagonist_name and str(cluster.get("role_hint") or "") == "Protagonist":
            return protagonist_name
        return ""

    def _extract_self_identified_names(self, text: str) -> list[str]:
        found: list[str] = []
        for pattern in (
            r"(?i)\bmy name is\s+([a-z][a-z-]+(?:\s+[a-z][a-z-]+){0,2})\b",
            r"(?i)\bi(?:'m| am)\s+([a-z][a-z-]+(?:\s+[a-z][a-z-]+){0,2})\b",
            r"(?i)\bthis is\s+([a-z][a-z-]+(?:\s+[a-z][a-z-]+){0,2})\b",
        ):
            for match in re.finditer(pattern, clean_ocr_text(text)):
                for candidate in self._name_service.extract_names(match.group(1)):
                    if is_valid_character_name_candidate(candidate):
                        found.append(candidate)
        return list(dict.fromkeys(found))

    def _sanitize_name(
        self,
        raw_name: str,
        cluster: dict[str, Any],
        protagonist_name: str | None,
        ordinal: int,
    ) -> str:
        cleaned = " ".join(part for part in raw_name.replace("_", " ").split() if part).strip()
        if not cleaned:
            return self._fallback_cluster_name(cluster, protagonist_name, Counter(), ordinal=ordinal)
        lowered = cleaned.casefold()
        if lowered in {"a man", "the man", "someone", "person", "other", "unknown"}:
            return self._fallback_cluster_name(cluster, protagonist_name, Counter(), ordinal=ordinal)
        if protagonist_name and lowered in {"the protagonist", "protagonist"}:
            return protagonist_name
        if lowered.startswith("speaker ") or lowered.startswith("character "):
            return self._fallback_cluster_name(cluster, protagonist_name, Counter(), ordinal=ordinal)
        if looks_like_false_character_name(cleaned) or not is_valid_character_name_candidate(cleaned):
            return self._fallback_cluster_name(cluster, protagonist_name, Counter(), ordinal=ordinal)
        return " ".join(token.capitalize() for token in cleaned.split())

    def _fallback_cluster_name(
        self,
        cluster: dict[str, Any],
        protagonist_name: str | None,
        role_counts: Counter[str],
        ordinal: int | None = None,
    ) -> str:
        base = str(cluster.get("role_hint") or "").strip() or "Stranger"
        if base == "Protagonist" and protagonist_name:
            return protagonist_name
        if ordinal is None:
            ordinal = role_counts[base] + 1
        if base == "Stranger" and ordinal > 1:
            return f"Stranger {ordinal}"
        if role_counts[base] > 0 and base not in {"Protagonist", protagonist_name or ""}:
            return f"{base} {ordinal}"
        return base

    def _role_hint(self, cluster: dict[str, Any], protagonist_name: str | None, index: int) -> str:
        dialogue_text = " ".join(str(line).casefold() for line in cluster.get("dialogues", []) if str(line).strip())
        if protagonist_name and index == 1:
            first_person_hits = len(
                [match for match in (" i ", " i'm ", " i’ll ", " my ", " me ") if match.strip() in f" {dialogue_text} "]
            )
            if first_person_hits or int(cluster.get("appearance_count") or 0) >= 3:
                return "Protagonist"
        role_keywords = (
            ("Neighbor", ("neighbor", "committee", "building", "sale going on", "share some")),
            ("Security Guard", ("guard", "security", "report", "intruder", "responsible")),
            ("Restaurant Worker", ("customer", "vip-card", "restaurant", "banquet", "menu")),
            ("Manager", ("manager", "deposit", "account number", "business department")),
            ("Loan Shark", ("interest", "loan", "million", "repay", "debt")),
            ("Teacher", ("teacher", "class", "students", "school")),
            ("Friend", ("friend", "buddy", "pal", "come with me")),
            ("Villain", ("kill him", "corpse", "traitor", "revenge")),
            ("Crowd Member", ("everyone", "crowd", "mob", "all of you")),
        )
        for label, keywords in role_keywords:
            if any(keyword in dialogue_text for keyword in keywords):
                return label
        return "Stranger"

    def _assign_cluster(self, clusters: list[dict[str, Any]], embedding: np.ndarray) -> int | None:
        best_index: int | None = None
        best_score = -1.0
        for index, cluster in enumerate(clusters):
            score = float(np.dot(cluster["centroid"], embedding))
            if score > best_score:
                best_score = score
                best_index = index
        if best_index is not None and best_score >= self._active_similarity_threshold:
            return best_index
        return None

    def _embed_samples(self, samples: list[dict[str, Any]]) -> list[np.ndarray]:
        sample_limit = max(int(self.settings.character_clip_sample_limit or 0), 0)
        if sample_limit and len(samples) > sample_limit:
            logger.info(
                "Character clustering using lightweight visual embeddings for %s samples because it exceeds the CLIP sample limit of %s",
                len(samples),
                sample_limit,
            )
            self._active_similarity_threshold = 0.75
            return [self._lightweight_image_embedding(sample["image"], sample["bbox"]) for sample in samples]

        model_bundle = self._load_clip_bundle()
        if model_bundle is None:
            self._active_similarity_threshold = 0.75
            return [self._lightweight_image_embedding(sample["image"], sample["bbox"]) for sample in samples]

        self._active_similarity_threshold = self._similarity_threshold
        processor, model, device = model_bundle
        import torch

        embeddings: list[np.ndarray] = []
        batch_size = 8
        with torch.inference_mode():
            for start in range(0, len(samples), batch_size):
                batch = samples[start : start + batch_size]
                inputs = processor(images=[sample["image"] for sample in batch], return_tensors="pt", padding=True)
                inputs = {key: value.to(device) if hasattr(value, "to") else value for key, value in inputs.items()}
                outputs = model.get_image_features(**inputs)
                normalized = outputs / outputs.norm(p=2, dim=-1, keepdim=True).clamp(min=1e-6)
                embeddings.extend(normalized.detach().cpu().numpy())
        return [self._normalize_vector(vector.astype(np.float32, copy=False)) for vector in embeddings]

    def _fallback_embedding(self, bbox: tuple[int, int, int, int]) -> np.ndarray:
        x, y, width, height = bbox
        vector = np.array([x, y, width, height, x + width / 2, y + height / 2], dtype=np.float32)
        return self._normalize_vector(vector)

    def _load_clip_bundle(self) -> tuple[Any, Any, str] | None:
        if self.__class__._CLIP_MODEL is not None and self.__class__._CLIP_PROCESSOR is not None:
            return self.__class__._CLIP_PROCESSOR, self.__class__._CLIP_MODEL, self.__class__._CLIP_DEVICE
        with self.__class__._LOAD_LOCK:
            if self.__class__._CLIP_MODEL is not None and self.__class__._CLIP_PROCESSOR is not None:
                return self.__class__._CLIP_PROCESSOR, self.__class__._CLIP_MODEL, self.__class__._CLIP_DEVICE
            try:
                from transformers import CLIPModel, CLIPProcessor
                import torch

                self.__class__._CLIP_PROCESSOR = CLIPProcessor.from_pretrained(self.settings.clip_model_id)
                self.__class__._CLIP_MODEL = CLIPModel.from_pretrained(self.settings.clip_model_id)
                self.__class__._CLIP_DEVICE = self._resolve_torch_device(torch)
                if hasattr(self.__class__._CLIP_MODEL, "to"):
                    self.__class__._CLIP_MODEL = self.__class__._CLIP_MODEL.to(self.__class__._CLIP_DEVICE)
                self.__class__._CLIP_MODEL.eval()
            except Exception as exc:
                logger.warning("Character clustering fell back to geometric grouping because CLIP could not load: %s", exc)
                self.__class__._CLIP_MODEL = None
                self.__class__._CLIP_PROCESSOR = None
                self.__class__._CLIP_DEVICE = "cpu"
                return None
        return self.__class__._CLIP_PROCESSOR, self.__class__._CLIP_MODEL, self.__class__._CLIP_DEVICE

    def _resolve_torch_device(self, torch_module: Any) -> str:
        if torch_module.cuda.is_available():
            return "cuda"
        mps_backend = getattr(torch_module.backends, "mps", None)
        if mps_backend is not None and mps_backend.is_available():
            return "mps"
        return "cpu"

    def _normalize_vector(self, vector: np.ndarray) -> np.ndarray:
        norm = float(np.linalg.norm(vector))
        if not math.isfinite(norm) or norm <= 0.0:
            return vector.astype(np.float32, copy=False)
        return (vector / norm).astype(np.float32, copy=False)

    def _lightweight_image_embedding(self, image: Image.Image, bbox: tuple[int, int, int, int]) -> np.ndarray:
        """Cheap visual fallback that clusters by crop content, not page geometry."""
        grayscale = image.convert("L").resize((16, 16), Image.Resampling.BILINEAR)
        pixels = np.asarray(grayscale, dtype=np.float32) / 255.0
        ink = 1.0 - pixels
        small = ink.flatten()
        row_projection = ink.mean(axis=1)
        column_projection = ink.mean(axis=0)
        histogram = np.histogram(pixels, bins=8, range=(0.0, 1.0), density=False)[0].astype(np.float32)
        histogram = histogram / max(float(histogram.sum()), 1.0)
        _, _, width, height = bbox
        geometry = np.array(
            [
                min(width / max(height, 1), 4.0) / 4.0,
                min(height / max(width, 1), 4.0) / 4.0,
            ],
            dtype=np.float32,
        )
        vector = np.concatenate(
            [
                small * 0.4,
                row_projection,
                column_projection,
                histogram * 0.5,
                geometry * 0.15,
            ]
        ).astype(np.float32, copy=False)
        return self._normalize_vector(vector)

    def _crop_character(self, image: Image.Image, bbox: tuple[int, int, int, int]) -> Image.Image | None:
        x, y, width, height = bbox
        page_width, page_height = image.size
        x0 = max(0, min(x, page_width - 1))
        y0 = max(0, min(y, page_height - 1))
        x1 = max(x0 + 1, min(x + width, page_width))
        y1 = max(y0 + 1, min(y + height, page_height))
        if x1 <= x0 or y1 <= y0:
            return None
        return image.crop((x0, y0, x1, y1)).resize((224, 224))

    def _panels_for_character(self, bbox: tuple[int, int, int, int], page_panels: list[PanelBox]) -> list[PanelBox]:
        matches: list[PanelBox] = []
        for panel in page_panels:
            panel_box = (int(panel.x), int(panel.y), int(panel.width), int(panel.height))
            expanded = self._expand_box(panel_box)
            if self._iou(expanded, bbox) >= 0.05 or self._center_inside(bbox, expanded):
                matches.append(panel)
        return sorted(matches, key=lambda item: item.order)

    def _expand_box(self, bbox: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
        x, y, width, height = bbox
        pad_x = max(24, int(width * 0.08))
        pad_y = max(24, int(height * 0.08))
        return (max(0, x - pad_x), max(0, y - pad_y), width + pad_x * 2, height + pad_y * 2)

    def _center_inside(self, inner: tuple[int, int, int, int], outer: tuple[int, int, int, int]) -> bool:
        center_x = inner[0] + inner[2] / 2
        center_y = inner[1] + inner[3] / 2
        return outer[0] <= center_x <= outer[0] + outer[2] and outer[1] <= center_y <= outer[1] + outer[3]

    def _iou(self, left: tuple[int, int, int, int], right: tuple[int, int, int, int]) -> float:
        intersection = self._intersection(left, right)
        if intersection <= 0:
            return 0.0
        left_area = max(left[2], 1) * max(left[3], 1)
        right_area = max(right[2], 1) * max(right[3], 1)
        return intersection / max(left_area + right_area - intersection, 1)

    def _intersection(self, left: tuple[int, int, int, int], right: tuple[int, int, int, int]) -> int:
        left_x1 = max(left[0], right[0])
        left_y1 = max(left[1], right[1])
        right_x2 = min(left[0] + left[2], right[0] + right[2])
        right_y2 = min(left[1] + left[3], right[1] + right[3])
        width = max(0, right_x2 - left_x1)
        height = max(0, right_y2 - left_y1)
        return width * height

    def _coerce_bbox(self, value: Any) -> tuple[int, int, int, int] | None:
        if not isinstance(value, (list, tuple)) or len(value) < 4:
            return None
        try:
            x, y, width, height = [int(round(float(item))) for item in value[:4]]
        except (TypeError, ValueError):
            return None
        if width <= 0 or height <= 0:
            return None
        return x, y, width, height

    def _metadata_payload(self, metadata: ChapterMetadata) -> dict[str, Any]:
        return {
            "manga_title": metadata.manga_title,
            "chapter_title": metadata.chapter_title,
            "chapter_number": metadata.chapter_number,
            "volume_number": metadata.volume_number,
            "language": metadata.language,
        }
