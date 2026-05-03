from __future__ import annotations

from datetime import datetime
import logging
import re
from pathlib import Path
from typing import Any

from PIL import Image

from app.core.config import get_settings
from app.schemas.character_identity import (
    CharacterReviewIdentity,
    CharacterReviewSample,
    CharacterReviewState,
)
from app.schemas.project import ChapterMetadata, PanelBox
from app.services.anime_face_detection_service import AnimeFaceDetectionService
from app.services.character_clusterer import CharacterClusterer
from app.services.character_name_service import CharacterNameService
from app.services.magi_service import MagiHFService
from app.utils.files import ensure_dir, read_json, slugify, write_json

logger = logging.getLogger(__name__)
_CHARACTER_REVIEW_STATE_VERSION = 5


class CharacterReviewService:
    def __init__(self) -> None:
        self.settings = get_settings()
        self._magi_service = MagiHFService()
        self._anime_face_service = AnimeFaceDetectionService()
        self._clusterer = CharacterClusterer()
        self._name_service = CharacterNameService()

    def review_state_path(self, project_dir: Path) -> Path:
        return project_dir / "output" / "character_review_state.json"

    def review_samples_dir(self, project_dir: Path) -> Path:
        return ensure_dir(project_dir / "characters" / "review")

    def series_key(self, metadata: ChapterMetadata, project_name: str) -> str:
        return slugify(str(metadata.manga_title or project_name or "unknown-series"))

    def series_memory_path(self, series_key: str) -> Path:
        return ensure_dir(self.settings.data_dir / "character_memory" / series_key) / "characters.json"

    def load_review_state(self, project_dir: Path) -> CharacterReviewState | None:
        payload = read_json(self.review_state_path(project_dir), default=None)
        if not isinstance(payload, dict):
            return None
        try:
            return CharacterReviewState.model_validate(payload)
        except Exception:
            logger.exception("Failed to load character review state from %s", self.review_state_path(project_dir))
            return None

    def load_series_memory(self, series_key: str) -> dict[str, Any]:
        payload = read_json(self.series_memory_path(series_key), default={})
        return payload if isinstance(payload, dict) else {}

    def build_review_state(
        self,
        project_id: str,
        project_dir: Path,
        project_name: str,
        metadata: ChapterMetadata,
        panels: list[PanelBox],
        artifacts: dict[str, Any],
    ) -> CharacterReviewState:
        artifacts = self._hydrate_artifacts_from_disk(project_dir, artifacts)
        series_key = self.series_key(metadata, project_name)
        series_memory = self.load_series_memory(series_key)
        existing_state = self.load_review_state(project_dir)
        characters = self._character_records_from_artifacts(artifacts)
        character_clusters = artifacts.get("character_clusters") or []
        cluster_lookup = {
            str(cluster.get("cluster_id") or "").strip(): cluster
            for cluster in character_clusters
            if isinstance(cluster, dict) and str(cluster.get("cluster_id") or "").strip()
        }
        panel_lookup = {str(panel.id): panel for panel in panels}

        existing_identities = existing_state.identities if existing_state else []
        used_stable_ids: set[str] = set()
        identities: list[CharacterReviewIdentity] = []

        for existing_identity in existing_identities:
            stable_ids = [
                stable_id
                for stable_id in existing_identity.stable_character_ids
                if stable_id in characters
            ]
            if not stable_ids:
                continue
            identities.append(
                self._build_identity(
                    project_dir,
                    stable_ids,
                    characters,
                    cluster_lookup,
                    panel_lookup,
                    series_memory,
                    existing_identity=existing_identity,
                )
            )
            used_stable_ids.update(stable_ids)

        for stable_id in sorted(characters.keys(), key=self._stable_character_sort_key):
            if stable_id in used_stable_ids:
                continue
            identities.append(
                self._build_identity(
                    project_dir,
                    [stable_id],
                    characters,
                    cluster_lookup,
                    panel_lookup,
                    series_memory,
                    existing_identity=None,
                )
            )

        protagonist_name = str(
            (existing_state.protagonist_name if existing_state else None)
            or artifacts.get("protagonist_name")
            or ""
        ).strip() or None
        identities = self._normalize_identity_names(identities, protagonist_name)

        memory_names = [
            str(item.get("canonical_name") or "").strip()
            for item in series_memory.get("characters", [])
            if isinstance(item, dict) and str(item.get("canonical_name") or "").strip()
        ]

        return CharacterReviewState(
            analysis_version=_CHARACTER_REVIEW_STATE_VERSION,
            project_id=project_id,
            series_key=series_key,
            protagonist_name=protagonist_name,
            memory_names=sorted(dict.fromkeys(memory_names)),
            identities=identities,
            generated_at=existing_state.generated_at if existing_state else datetime.utcnow(),
            updated_at=datetime.utcnow(),
        )

    def save_review_state(
        self,
        project_dir: Path,
        project_name: str,
        metadata: ChapterMetadata,
        state: CharacterReviewState,
    ) -> CharacterReviewState:
        normalized = CharacterReviewState.model_validate(
            {
                **state.model_dump(mode="json"),
                "analysis_version": _CHARACTER_REVIEW_STATE_VERSION,
                "series_key": self.series_key(metadata, project_name),
                "updated_at": datetime.utcnow().isoformat(),
            }
        )
        write_json(self.review_state_path(project_dir), normalized.model_dump(mode="json"))
        self._update_series_memory(normalized.series_key, project_name, normalized)
        return normalized

    def prepare_review_artifacts(
        self,
        project_dir: Path,
        metadata: ChapterMetadata,
        panels: list[PanelBox],
        page_paths: list[Path],
        *,
        progress_callback: callable | None = None,
        cancel_callback: callable | None = None,
    ) -> tuple[dict[str, Any], bool]:
        output_dir = project_dir / "output"
        kept_panels = [panel for panel in sorted(panels, key=lambda item: item.order) if panel.keep]
        panel_signature = self._panel_signature(kept_panels)

        cached_manifest = self._load_cached_dialogue_manifest(
            output_dir / "dialogue_pipeline_manifest.json",
            panel_signature,
        )
        if cached_manifest is not None:
            if progress_callback:
                progress_callback(32, "Reused cached dialogue analysis for character review")
            return cached_manifest, True

        page_numbers = sorted({int(panel.page) for panel in kept_panels if int(panel.page) > 0})
        if not page_numbers:
            return (
                {
                    "character_clusters": [],
                    "character_tracking": {"characters": {}, "source_to_character_id": {}, "panel_characters": {}},
                    "characters": {},
                    "character_dictionary": {},
                    "character_identity_report": {
                        "summary": "No kept panels were available for character review.",
                        "mode": "fast-character-review-v1",
                    },
                    "protagonist_name": None,
                },
                False,
            )

        cached_payloads = self._load_character_review_payloads(
            output_dir / "character_review_page_payloads.json",
            page_numbers,
        )
        generic_magi_payloads = self._load_character_review_payloads(
            output_dir / "magi_page_payloads.json",
            page_numbers,
        )
        for page_number, payload in generic_magi_payloads.items():
            cached_payloads.setdefault(page_number, payload)

        missing_pages = [page_number for page_number in page_numbers if page_number not in cached_payloads]
        if missing_pages and self._magi_service.is_available():
            if progress_callback:
                progress_callback(8, "Scanning character layouts with MAGI")
            fresh_payloads = self._magi_service.predict_page_payloads(
                page_paths,
                page_numbers=missing_pages,
                do_ocr=False,
                batch_size=int(self.settings.magi_batch_size or 1),
                cancel_callback=cancel_callback,
                progress_callback=(
                    (lambda pct, message: progress_callback(8 + pct * 0.58, message))
                    if progress_callback
                    else None
                ),
                progress_label="Scanning character layouts with MAGI",
            )
            cached_payloads.update(fresh_payloads)
            self._write_character_review_payloads(
                output_dir / "character_review_page_payloads.json",
                cached_payloads,
            )
        elif cached_payloads and progress_callback:
            progress_callback(28, "Reused cached character-layout scan")

        anime_face_detections = 0
        anime_face_pages = 0
        if self._anime_face_service.is_available():
            if progress_callback:
                progress_callback(64, "Finding manga face crops")
            anime_face_payloads = self._anime_face_service.detect_page_payloads(
                page_paths,
                page_numbers=page_numbers,
                cache_path=output_dir / "anime_face_page_payloads.json",
                cancel_callback=cancel_callback,
                progress_callback=(
                    (lambda pct, message: progress_callback(64 + pct * 0.06, message))
                    if progress_callback
                    else None
                ),
            )
            cached_payloads, anime_face_detections = self._merge_face_payloads(
                cached_payloads,
                anime_face_payloads,
            )
            anime_face_pages = sum(
                1 for payload in anime_face_payloads.values() if payload.get("characters")
            )
            if anime_face_payloads:
                self._write_character_review_payloads(
                    output_dir / "character_review_page_payloads.json",
                    cached_payloads,
                )

        if progress_callback:
            progress_callback(72, "Clustering recurring character appearances")
        cluster_payload = self._clusterer.cluster(
            page_paths,
            cached_payloads,
            kept_panels,
            cancel_callback=cancel_callback,
        )
        character_clusters = cluster_payload.get("clusters", [])
        character_tracking = self._build_tracking_from_clusters(character_clusters)

        known_texts = [
            str(panel.ocr_text or "").strip()
            for panel in kept_panels
            if str(panel.ocr_text or "").strip()
        ]
        character_dictionary, protagonist_name = self._name_service.discover(known_texts, metadata)
        artifacts = {
            "character_clusters": character_clusters,
            "character_tracking": character_tracking,
            "characters": character_tracking.get("characters", {}),
            "character_dictionary": character_dictionary,
            "character_identity_report": {
                "summary": (
                    f"Prepared {len(character_clusters)} visual character groups from cached/layout MAGI data."
                    if character_clusters
                    else "No recurring visual character groups were found."
                ),
                "mode": "fast-character-review-v1",
                "review_analysis_version": _CHARACTER_REVIEW_STATE_VERSION,
                "cluster_count": len(character_clusters),
                "scanned_pages": len(cached_payloads),
                "cached_pages": len(page_numbers) - len(missing_pages),
                "anime_face_pages": anime_face_pages,
                "anime_face_detections": anime_face_detections,
            },
            "protagonist_name": protagonist_name,
        }
        if progress_callback:
            progress_callback(88, "Building character review cards")
        return artifacts, False

    def apply_review_to_artifacts(self, project_dir: Path, artifacts: dict[str, Any]) -> dict[str, Any]:
        state = self.load_review_state(project_dir)
        if state is None:
            return artifacts

        stable_to_name: dict[str, str] = {}
        source_to_name: dict[str, str] = {}
        alias_to_name: dict[str, str] = {}
        for identity in state.identities:
            final_name = self._clean_name(identity.name)
            if not final_name or identity.status == "unknown":
                continue
            for stable_id in identity.stable_character_ids:
                stable_to_name[str(stable_id)] = final_name
            for source_id in identity.source_character_ids:
                source_to_name[str(source_id)] = final_name
            for alias in (identity.suggested_name, identity.remembered_name, identity.name):
                normalized = self._normalize_tokens(alias)
                if normalized:
                    alias_to_name[normalized] = final_name

        if not stable_to_name and not source_to_name and not alias_to_name:
            return artifacts

        characters = artifacts.get("characters")
        if isinstance(characters, dict):
            for stable_id, payload in characters.items():
                if not isinstance(payload, dict):
                    continue
                final_name = stable_to_name.get(str(stable_id))
                if not final_name:
                    continue
                payload["name"] = final_name
                payload["display_name"] = final_name
                payload["narration_reference"] = final_name
                description = str(payload.get("description") or "").strip()
                if not description or description == str(payload.get("display_name") or "").strip():
                    payload["description"] = final_name

        character_dictionary = artifacts.get("character_dictionary")
        normalized_dictionary: dict[str, str] = {}
        if isinstance(character_dictionary, dict):
            for key, value in character_dictionary.items():
                final_name = (
                    alias_to_name.get(self._normalize_tokens(value))
                    or alias_to_name.get(self._normalize_tokens(key))
                    or self._clean_name(value)
                )
                if final_name:
                    normalized_dictionary[self._normalize_tokens(final_name)] = final_name
        for final_name in stable_to_name.values():
            normalized_key = self._normalize_tokens(final_name)
            if normalized_key:
                normalized_dictionary[normalized_key] = final_name
        artifacts["character_dictionary"] = normalized_dictionary

        for region in artifacts.get("dialogue_regions", []) or []:
            if not isinstance(region, dict):
                continue
            final_name = (
                stable_to_name.get(str(region.get("stable_character_id") or ""))
                or source_to_name.get(str(region.get("character_id") or ""))
                or alias_to_name.get(self._normalize_tokens(region.get("speaker_name")))
                or alias_to_name.get(self._normalize_tokens(region.get("character_display_name")))
            )
            if final_name:
                region["speaker_name"] = final_name
                region["character_display_name"] = final_name

        for scene in artifacts.get("scenes", []) or []:
            if not isinstance(scene, dict):
                continue
            scene["character_names"] = self._dedupe_preserve_order(
                [
                    stable_to_name.get(str(character_id))
                    or alias_to_name.get(self._normalize_tokens(name))
                    or str(name).strip()
                    for character_id, name in zip(
                        scene.get("character_ids", []) or [],
                        scene.get("character_names", []) or [],
                    )
                    if str(
                        stable_to_name.get(str(character_id))
                        or alias_to_name.get(self._normalize_tokens(name))
                        or name
                    ).strip()
                ]
            )
            scene["speaker_names"] = self._dedupe_preserve_order(
                [
                    alias_to_name.get(self._normalize_tokens(name)) or str(name).strip()
                    for name in scene.get("speaker_names", []) or []
                    if str(alias_to_name.get(self._normalize_tokens(name)) or name).strip()
                ]
            )
            primary_speaker = alias_to_name.get(self._normalize_tokens(scene.get("primary_speaker_name")))
            if primary_speaker:
                scene["primary_speaker_name"] = primary_speaker
            scene_protagonist = state.protagonist_name or alias_to_name.get(self._normalize_tokens(scene.get("protagonist_name")))
            if scene_protagonist:
                scene["protagonist_name"] = scene_protagonist

        protagonist_name = state.protagonist_name or alias_to_name.get(self._normalize_tokens(artifacts.get("protagonist_name")))
        if protagonist_name:
            artifacts["protagonist_name"] = protagonist_name

        identity_report = artifacts.get("character_identity_report")
        if not isinstance(identity_report, dict):
            identity_report = {}
        identity_report["reviewed_identity_count"] = sum(
            1 for identity in state.identities if self._clean_name(identity.name) and identity.status != "unknown"
        )
        identity_report["reviewed_character_dictionary"] = normalized_dictionary
        if protagonist_name:
            identity_report["protagonist_name"] = protagonist_name
        artifacts["character_identity_report"] = identity_report
        return artifacts

    def ensure_review_state(
        self,
        project_id: str,
        project_dir: Path,
        project_name: str,
        metadata: ChapterMetadata,
        panels: list[PanelBox],
        artifacts: dict[str, Any] | None = None,
    ) -> CharacterReviewState:
        current = self.load_review_state(project_dir)
        if current is not None and int(current.analysis_version or 0) >= _CHARACTER_REVIEW_STATE_VERSION:
            return current
        source_artifacts = artifacts or read_json(project_dir / "output" / "dialogue_pipeline_manifest.json", default=None)
        if not isinstance(source_artifacts, dict):
            if current is not None:
                return current
            raise FileNotFoundError("Character candidates have not been prepared yet.")
        report = source_artifacts.get("character_identity_report")
        source_version = int(report.get("review_analysis_version") or 0) if isinstance(report, dict) else 0
        if current is not None and source_version < _CHARACTER_REVIEW_STATE_VERSION:
            return current
        state = self.build_review_state(project_id, project_dir, project_name, metadata, panels, source_artifacts)
        return self.save_review_state(project_dir, project_name, metadata, state)

    def _hydrate_artifacts_from_disk(self, project_dir: Path, artifacts: dict[str, Any]) -> dict[str, Any]:
        hydrated = dict(artifacts or {})
        output_dir = project_dir / "output"
        if not isinstance(hydrated.get("characters"), dict):
            hydrated["characters"] = read_json(output_dir / "characters.json", default={})
        if not isinstance(hydrated.get("character_clusters"), list):
            hydrated["character_clusters"] = read_json(output_dir / "character_clusters.json", default=[])
        if not hydrated.get("protagonist_name"):
            identity_report = read_json(output_dir / "character_identity_report.json", default={})
            if isinstance(identity_report, dict):
                hydrated["protagonist_name"] = identity_report.get("protagonist_name")
        return hydrated

    def _load_cached_dialogue_manifest(
        self,
        manifest_path: Path,
        panel_signature: list[dict[str, Any]],
    ) -> dict[str, Any] | None:
        cached = read_json(manifest_path, default=None)
        if not isinstance(cached, dict):
            return None
        if cached.get("panel_signature") != panel_signature:
            return None
        if "character_clusters" not in cached:
            return None
        if "characters" not in cached:
            return None
        report = cached.get("character_identity_report")
        if not isinstance(report, dict):
            return None
        if int(report.get("review_analysis_version") or 0) < _CHARACTER_REVIEW_STATE_VERSION:
            return None
        return cached

    def _load_character_review_payloads(
        self,
        cache_path: Path,
        page_numbers: list[int],
    ) -> dict[int, dict[str, Any]]:
        cached_payload = read_json(cache_path, default=None)
        if not isinstance(cached_payload, dict):
            return {}
        reusable: dict[int, dict[str, Any]] = {}
        for page_number in page_numbers:
            candidate = cached_payload.get(page_number)
            if candidate is None:
                candidate = cached_payload.get(str(page_number))
            if not isinstance(candidate, dict):
                continue
            if not isinstance(candidate.get("characters"), list):
                continue
            reusable[int(page_number)] = candidate
        return reusable

    def _write_character_review_payloads(
        self,
        cache_path: Path,
        payloads: dict[int, dict[str, Any]],
    ) -> None:
        existing = read_json(cache_path, default={})
        merged = existing if isinstance(existing, dict) else {}
        for page_number, payload in payloads.items():
            if isinstance(payload, dict):
                merged[str(int(page_number))] = payload
        write_json(cache_path, merged)

    def _merge_face_payloads(
        self,
        base_payloads: dict[int, dict[str, Any]],
        face_payloads: dict[int, dict[str, Any]],
    ) -> tuple[dict[int, dict[str, Any]], int]:
        merged = {int(page_number): dict(payload) for page_number, payload in base_payloads.items()}
        added_count = 0
        for page_number, face_payload in face_payloads.items():
            if not isinstance(face_payload, dict):
                continue
            face_characters = [
                dict(character)
                for character in face_payload.get("characters", []) or []
                if isinstance(character, dict)
            ]
            if not face_characters:
                merged.setdefault(
                    int(page_number),
                    {
                        "page": int(page_number),
                        "provider": "character-review-merged-v1",
                        "panels": [],
                        "texts": [],
                        "characters": [],
                    },
                )
                continue
            target = dict(
                merged.get(int(page_number))
                or {
                    "page": int(page_number),
                    "provider": "character-review-merged-v1",
                    "panels": [],
                    "texts": [],
                    "characters": [],
                }
            )
            existing_characters = [
                dict(character)
                for character in target.get("characters", []) or []
                if isinstance(character, dict)
            ]
            additions: list[dict[str, Any]] = []
            for face_character in face_characters:
                face_bbox = self._coerce_bbox(face_character.get("bbox"))
                if face_bbox is None:
                    continue
                if self._duplicates_existing_face(face_bbox, existing_characters + additions):
                    continue
                face_character["bbox"] = face_bbox
                face_character["source"] = str(face_character.get("source") or "animeface-lbp")
                additions.append(face_character)
            if additions:
                target["characters"] = existing_characters + additions
                providers = self._payload_providers(
                    [target.get("provider"), face_payload.get("provider")]
                )
                target["provider"] = "+".join(providers) if providers else "character-review-merged-v1"
                merged[int(page_number)] = target
                added_count += len(additions)
            else:
                merged[int(page_number)] = target
        return merged, added_count

    def _payload_providers(self, providers: list[Any]) -> list[str]:
        unique: list[str] = []
        seen: set[str] = set()
        for raw_provider in providers:
            for provider in str(raw_provider or "").split("+"):
                cleaned = provider.strip()
                if not cleaned or cleaned in seen:
                    continue
                seen.add(cleaned)
                unique.append(cleaned)
        return unique

    def _duplicates_existing_face(
        self,
        face_bbox: list[int],
        existing_characters: list[dict[str, Any]],
    ) -> bool:
        face_area = max(face_bbox[2] * face_bbox[3], 1)
        for character in existing_characters:
            existing_bbox = self._coerce_bbox(character.get("bbox"))
            if existing_bbox is None:
                continue
            existing_area = max(existing_bbox[2] * existing_bbox[3], 1)
            if self._bbox_iou(face_bbox, existing_bbox) >= 0.38 and existing_area <= face_area * 3.5:
                return True
        return False

    def _coerce_bbox(self, value: Any) -> list[int] | None:
        candidate = value.tolist() if hasattr(value, "tolist") else value
        if not isinstance(candidate, (list, tuple)) or len(candidate) < 4:
            return None
        try:
            x, y, width, height = [int(round(float(component))) for component in candidate[:4]]
        except (TypeError, ValueError):
            return None
        if width <= 0 or height <= 0:
            return None
        return [x, y, width, height]

    def _bbox_iou(self, left: list[int], right: list[int]) -> float:
        lx, ly, lw, lh = left[:4]
        rx, ry, rw, rh = right[:4]
        left_x2 = lx + lw
        left_y2 = ly + lh
        right_x2 = rx + rw
        right_y2 = ry + rh
        inter_w = max(0, min(left_x2, right_x2) - max(lx, rx))
        inter_h = max(0, min(left_y2, right_y2) - max(ly, ry))
        inter_area = inter_w * inter_h
        if inter_area <= 0:
            return 0.0
        union = (lw * lh) + (rw * rh) - inter_area
        return inter_area / float(max(union, 1))

    def _panel_signature(self, panels: list[PanelBox]) -> list[dict[str, Any]]:
        return [
            {
                "id": panel.id,
                "order": panel.order,
                "page": panel.page,
                "x": panel.x,
                "y": panel.y,
                "width": panel.width,
                "height": panel.height,
                "keep": panel.keep,
                "manual_keep": panel.manual_keep,
                "logical_panel_id": panel.logical_panel_id,
                "multi_page_panel": panel.multi_page_panel,
            }
            for panel in panels
        ]

    def _character_records_from_artifacts(self, artifacts: dict[str, Any]) -> dict[str, Any]:
        characters = artifacts.get("characters")
        if isinstance(characters, dict) and characters:
            return characters

        tracking_payload = artifacts.get("character_tracking")
        tracking_characters = tracking_payload.get("characters") if isinstance(tracking_payload, dict) else None
        if isinstance(tracking_characters, dict) and tracking_characters:
            return tracking_characters

        synthesized: dict[str, dict[str, Any]] = {}
        for index, cluster in enumerate(artifacts.get("character_clusters") or [], start=1):
            if not isinstance(cluster, dict):
                continue
            cluster_id = str(cluster.get("cluster_id") or "").strip()
            if not cluster_id:
                continue
            role_hint = self._clean_name(cluster.get("role_hint")) or "Character"
            panel_ids = [
                str(panel_id).strip()
                for panel_id in cluster.get("panel_ids", []) or []
                if str(panel_id).strip()
            ]
            appearances = self._cluster_appearances(cluster, fallback_panel_ids=panel_ids)
            synthesized[f"Character_{index}"] = {
                "id": f"Character_{index}",
                "name": None,
                "display_name": role_hint,
                "description": role_hint,
                "role": role_hint,
                "first_panel": int((cluster.get("panels") or [0])[0] or 0),
                "first_page": int((cluster.get("pages") or [0])[0] or 0),
                "appearance_count": int(cluster.get("appearance_count") or len(panel_ids)),
                "appearances": appearances,
                "source_character_ids": [cluster_id],
                "narration_reference": role_hint,
            }
        return synthesized

    def _build_tracking_from_clusters(self, character_clusters: list[dict[str, Any]]) -> dict[str, Any]:
        characters: dict[str, dict[str, Any]] = {}
        source_to_character_id: dict[str, str] = {}
        panel_characters: dict[str, list[str]] = {}

        ordered_clusters = sorted(
            (
                cluster
                for cluster in character_clusters
                if str(cluster.get("cluster_id") or "").strip()
            ),
            key=lambda item: (
                min((int(value) for value in item.get("pages", []) or [0]), default=0),
                min((int(value) for value in item.get("panels", []) or [0]), default=0),
                str(item.get("cluster_id") or ""),
            ),
        )

        for index, cluster in enumerate(ordered_clusters, start=1):
            cluster_id = str(cluster.get("cluster_id") or "").strip()
            if not cluster_id:
                continue
            stable_id = f"Character_{index}"
            source_to_character_id[cluster_id] = stable_id
            panel_ids = [
                str(panel_id).strip()
                for panel_id in cluster.get("panel_ids", []) or []
                if str(panel_id).strip()
            ]
            pages = [int(value) for value in cluster.get("pages", []) or [] if str(value).strip()]
            panels = [int(value) for value in cluster.get("panels", []) or [] if str(value).strip()]
            appearances = self._cluster_appearances(cluster, fallback_panel_ids=panel_ids)
            characters[stable_id] = {
                "id": stable_id,
                "name": None,
                "description": "",
                "first_panel": min(panels) if panels else 0,
                "first_page": min(pages) if pages else 0,
                "appearances": appearances,
                "appearance_count": max(len(panel_ids), int(cluster.get("appearance_count") or 0)),
                "source_character_ids": [cluster_id],
            }
            for panel_id in panel_ids:
                panel_characters.setdefault(panel_id, [])
                if stable_id not in panel_characters[panel_id]:
                    panel_characters[panel_id].append(stable_id)

        return {
            "characters": characters,
            "source_to_character_id": source_to_character_id,
            "panel_characters": panel_characters,
            "provider": "character-review-fast-v1",
        }

    def _cluster_appearances(
        self,
        cluster: dict[str, Any],
        *,
        fallback_panel_ids: list[str],
    ) -> list[dict[str, Any]]:
        appearances: list[dict[str, Any]] = []
        seen: set[tuple[str, int, tuple[int, ...]]] = set()
        for raw in cluster.get("appearances", []) or []:
            if not isinstance(raw, dict):
                continue
            page = int(raw.get("page") or 0)
            bbox = [int(value) for value in (raw.get("bbox") or [])[:4] if isinstance(value, (int, float))]
            panel_ids = [
                str(panel_id).strip()
                for panel_id in raw.get("panel_ids", []) or []
                if str(panel_id).strip()
            ]
            panels = [int(panel) for panel in raw.get("panels", []) or [] if str(panel).strip()]
            if not panel_ids:
                panel_ids = fallback_panel_ids[:1]
            for index, panel_id in enumerate(panel_ids):
                panel_order = panels[min(index, len(panels) - 1)] if panels else 0
                key = (panel_id, page, tuple(bbox))
                if key in seen:
                    continue
                seen.add(key)
                appearances.append(
                    {
                        "page": page,
                        "panel": panel_order,
                        "panel_id": panel_id,
                        "bbox": bbox,
                    }
                )
        if appearances:
            appearances.sort(
                key=lambda item: (
                    int(item.get("page") or 0),
                    int(item.get("panel") or 0),
                    str(item.get("panel_id") or ""),
                )
            )
            return appearances

        pages = [int(value) for value in cluster.get("pages", []) or [] if str(value).strip()]
        panels = [int(value) for value in cluster.get("panels", []) or [] if str(value).strip()]
        return [
            {
                "page": pages[0] if pages else 0,
                "panel": panels[0] if panels else 0,
                "panel_id": panel_id,
                "bbox": [],
            }
            for panel_id in fallback_panel_ids
        ]

    def _build_identity(
        self,
        project_dir: Path,
        stable_character_ids: list[str],
        characters: dict[str, Any],
        cluster_lookup: dict[str, dict[str, Any]],
        panel_lookup: dict[str, PanelBox],
        series_memory: dict[str, Any],
        *,
        existing_identity: CharacterReviewIdentity | None,
    ) -> CharacterReviewIdentity:
        character_records = [
            characters[stable_id]
            for stable_id in stable_character_ids
            if isinstance(characters.get(stable_id), dict)
        ]
        source_character_ids = self._dedupe_preserve_order(
            [
                str(source_id).strip()
                for record in character_records
                for source_id in record.get("source_character_ids", []) or []
                if str(source_id).strip()
            ]
        )
        suggested_name = (
            existing_identity.suggested_name
            if existing_identity is not None
            else self._suggested_name(character_records)
        )
        role_hint = self._role_hint(character_records, cluster_lookup)
        memory_matches = self._memory_matches(series_memory, suggested_name, role_hint)
        remembered_name = (
            existing_identity.remembered_name
            if existing_identity is not None and self._clean_name(existing_identity.remembered_name)
            else self._remembered_name(series_memory, suggested_name)
        )
        chosen_name = self._clean_name(
            (existing_identity.name if existing_identity is not None else None)
            or remembered_name
            or suggested_name
        )
        status = existing_identity.status if existing_identity is not None else ("confirmed" if remembered_name and chosen_name else "suggested")
        appearances = self._collect_appearances(character_records)
        pages = sorted({int(item.get("page", 0) or 0) for item in appearances if int(item.get("page", 0) or 0) > 0})
        panel_ids = self._dedupe_preserve_order(
            [
                str(item.get("panel_id") or "").strip()
                for item in appearances
                if str(item.get("panel_id") or "").strip()
            ]
        )
        appearance_count = max(
            len(appearances),
            sum(int(record.get("appearance_count") or 0) for record in character_records),
        )
        sample_images = self._build_samples(
            project_dir,
            existing_identity.review_id if existing_identity is not None else stable_character_ids[0],
            appearances,
            panel_lookup,
        )
        return CharacterReviewIdentity(
            review_id=existing_identity.review_id if existing_identity is not None else f"review-{slugify(stable_character_ids[0])}",
            stable_character_ids=stable_character_ids,
            source_character_ids=source_character_ids,
            suggested_name=suggested_name,
            remembered_name=remembered_name,
            memory_matches=self._dedupe_preserve_order(
                (existing_identity.memory_matches if existing_identity is not None else []) + memory_matches
            )[:4],
            name=chosen_name,
            status=status,
            role_hint=role_hint,
            appearance_count=appearance_count,
            pages=pages,
            panel_ids=panel_ids,
            sample_images=sample_images,
            notes=existing_identity.notes if existing_identity is not None else None,
        )

    def _collect_appearances(self, character_records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        appearances: list[dict[str, Any]] = []
        seen_keys: set[tuple[Any, ...]] = set()
        for record in character_records:
            for appearance in record.get("appearances", []) or []:
                if not isinstance(appearance, dict):
                    continue
                key = (
                    appearance.get("panel_id"),
                    appearance.get("page"),
                    appearance.get("panel"),
                    tuple(appearance.get("bbox", []) or []),
                )
                if key in seen_keys:
                    continue
                seen_keys.add(key)
                appearances.append(appearance)
        appearances.sort(
            key=lambda item: (
                int(item.get("page", 0) or 0),
                int(item.get("panel", 0) or 0),
                str(item.get("panel_id") or ""),
            )
        )
        return appearances

    def _build_samples(
        self,
        project_dir: Path,
        review_id: str,
        appearances: list[dict[str, Any]],
        panel_lookup: dict[str, PanelBox],
    ) -> list[CharacterReviewSample]:
        samples: list[CharacterReviewSample] = []
        if not appearances:
            return samples
        page_paths = sorted((project_dir / "pages").glob("*"))
        for index, appearance in enumerate(appearances[:4], start=1):
            sample = self._build_sample_image(
                project_dir,
                page_paths,
                review_id,
                appearance,
                panel_lookup.get(str(appearance.get("panel_id") or "").strip()),
                index,
            )
            if sample is not None:
                samples.append(sample)
        return samples

    def _build_sample_image(
        self,
        project_dir: Path,
        page_paths: list[Path],
        review_id: str,
        appearance: dict[str, Any],
        panel: PanelBox | None,
        index: int,
    ) -> CharacterReviewSample | None:
        page = int(appearance.get("page", 0) or 0)
        if page <= 0 or page > len(page_paths):
            return None
        page_path = page_paths[page - 1]
        target_dir = self.review_samples_dir(project_dir)
        sample_id = f"{slugify(review_id)}-{index:02d}"
        target_path = target_dir / f"{sample_id}.jpg"
        bbox = [int(value) for value in (appearance.get("bbox") or [])[:4] if isinstance(value, (int, float))]
        try:
            with Image.open(page_path) as source:
                image = source.convert("RGB")
                crop_box = self._appearance_crop_box(image.size, bbox, panel)
                crop = image.crop(crop_box)
                crop.thumbnail((384, 384))
                crop.save(target_path, format="JPEG", quality=88)
        except Exception:
            logger.exception("Failed to build character review sample for %s", page_path)
            return None
        return CharacterReviewSample(
            sample_id=sample_id,
            image_url=self._relative_media_url(target_path),
            image_path=str(target_path),
            panel_id=str(appearance.get("panel_id") or "").strip() or None,
            page=page,
            panel=int(appearance.get("panel", 0) or 0) or None,
            bbox=bbox,
        )

    def _appearance_crop_box(
        self,
        image_size: tuple[int, int],
        bbox: list[int],
        panel: PanelBox | None,
    ) -> tuple[int, int, int, int]:
        image_width, image_height = image_size
        if len(bbox) >= 4:
            x, y, width, height = bbox[:4]
            if width > 0 and height > 0:
                padding = max(16, int(round(max(width, height) * 0.18)))
                left = max(0, x - padding)
                top = max(0, y - padding)
                right = min(image_width, x + width + padding)
                bottom = min(image_height, y + height + padding)
                if right > left and bottom > top:
                    return (left, top, right, bottom)
        if panel is not None:
            left = max(0, int(panel.x))
            top = max(0, int(panel.y))
            right = min(image_width, int(panel.x + panel.width))
            bottom = min(image_height, int(panel.y + panel.height))
            if right > left and bottom > top:
                return (left, top, right, bottom)
        return (0, 0, image_width, image_height)

    def _update_series_memory(self, series_key: str, project_name: str, state: CharacterReviewState) -> None:
        memory = self.load_series_memory(series_key)
        entries = memory.get("characters")
        if not isinstance(entries, list):
            entries = []
        entry_by_name = {
            self._normalize_tokens(item.get("canonical_name")): item
            for item in entries
            if isinstance(item, dict) and self._normalize_tokens(item.get("canonical_name"))
        }
        for identity in state.identities:
            final_name = self._clean_name(identity.name)
            if not final_name or identity.status == "unknown":
                continue
            key = self._normalize_tokens(final_name)
            if not key:
                continue
            aliases = self._dedupe_preserve_order(
                [
                    final_name,
                    str(identity.suggested_name or "").strip(),
                    str(identity.remembered_name or "").strip(),
                    *[str(match).strip() for match in identity.memory_matches or [] if str(match).strip()],
                ]
            )
            existing = entry_by_name.get(key, {})
            merged_aliases = self._dedupe_preserve_order(
                [str(alias).strip() for alias in existing.get("aliases", []) or [] if str(alias).strip()] + aliases
            )
            merged_role_hints = self._dedupe_preserve_order(
                [str(role).strip() for role in existing.get("role_hints", []) or [] if str(role).strip()]
                + ([str(identity.role_hint).strip()] if str(identity.role_hint or "").strip() else [])
            )
            merged_suggested_names = self._dedupe_preserve_order(
                [str(name).strip() for name in existing.get("suggested_names", []) or [] if str(name).strip()]
                + ([str(identity.suggested_name).strip()] if str(identity.suggested_name or "").strip() else [])
            )
            entry_by_name[key] = {
                "canonical_name": final_name,
                "aliases": merged_aliases,
                "role_hints": merged_role_hints,
                "suggested_names": merged_suggested_names,
                "last_project": state.project_id,
                "project_name": project_name,
                "updated_at": datetime.utcnow().isoformat(),
            }
        write_json(
            self.series_memory_path(series_key),
            {
                "series_key": series_key,
                "updated_at": datetime.utcnow().isoformat(),
                "characters": sorted(entry_by_name.values(), key=lambda item: str(item.get("canonical_name") or "").casefold()),
            },
        )

    def _remembered_name(self, series_memory: dict[str, Any], suggested_name: str | None) -> str | None:
        normalized_suggested = self._normalize_tokens(suggested_name)
        if not normalized_suggested:
            return None
        for item in series_memory.get("characters", []) or []:
            if not isinstance(item, dict):
                continue
            canonical_name = self._clean_name(item.get("canonical_name"))
            aliases = {
                self._normalize_tokens(alias)
                for alias in item.get("aliases", []) or []
                if self._normalize_tokens(alias)
            }
            if canonical_name and (normalized_suggested == self._normalize_tokens(canonical_name) or normalized_suggested in aliases):
                return canonical_name
        return None

    def _memory_matches(
        self,
        series_memory: dict[str, Any],
        suggested_name: str | None,
        role_hint: str | None,
    ) -> list[str]:
        normalized_suggested = self._normalize_tokens(suggested_name)
        normalized_role = self._normalize_tokens(role_hint)
        exact: list[str] = []
        close: list[str] = []
        role_based: list[str] = []
        for item in series_memory.get("characters", []) or []:
            if not isinstance(item, dict):
                continue
            canonical_name = self._clean_name(item.get("canonical_name"))
            if not canonical_name:
                continue
            alias_keys = self._alias_keys(item)
            if normalized_suggested:
                if normalized_suggested in alias_keys:
                    exact.append(canonical_name)
                    continue
                if any(self._memory_name_is_close(normalized_suggested, alias_key) for alias_key in alias_keys):
                    close.append(canonical_name)
                    continue
            if normalized_role:
                role_hints = {
                    self._normalize_tokens(value)
                    for value in item.get("role_hints", []) or []
                    if self._normalize_tokens(value)
                }
                if normalized_role and normalized_role in role_hints:
                    role_based.append(canonical_name)
        return self._dedupe_preserve_order(exact + close + role_based)

    def _alias_keys(self, item: dict[str, Any]) -> set[str]:
        keys = {
            self._normalize_tokens(item.get("canonical_name")),
            *[
                self._normalize_tokens(alias)
                for alias in item.get("aliases", []) or []
            ],
            *[
                self._normalize_tokens(alias)
                for alias in item.get("suggested_names", []) or []
            ],
        }
        return {key for key in keys if key}

    def _memory_name_is_close(self, suggested_key: str, alias_key: str) -> bool:
        if not suggested_key or not alias_key:
            return False
        if suggested_key == alias_key:
            return True
        suggested_tokens = suggested_key.split()
        alias_tokens = alias_key.split()
        if len(suggested_tokens) >= 2 and all(token in alias_tokens for token in suggested_tokens):
            return True
        if len(suggested_tokens) == 1 and len(suggested_tokens[0]) >= 4 and suggested_tokens[0] in alias_tokens:
            return True
        if len(alias_tokens) >= 2 and (alias_key.startswith(suggested_key) or alias_key.endswith(suggested_key)):
            return True
        return False

    def _suggested_name(self, character_records: list[dict[str, Any]]) -> str | None:
        for record in character_records:
            for key in ("display_name", "name", "narration_reference", "description", "role"):
                value = self._clean_name(record.get(key))
                if value:
                    return value
        return None

    def _role_hint(self, character_records: list[dict[str, Any]], cluster_lookup: dict[str, dict[str, Any]]) -> str | None:
        hints = [
            self._clean_name(record.get("role"))
            for record in character_records
            if self._clean_name(record.get("role"))
        ]
        if hints:
            return hints[0]
        for record in character_records:
            for source_id in record.get("source_character_ids", []) or []:
                cluster = cluster_lookup.get(str(source_id))
                if not cluster:
                    continue
                role_hint = self._clean_name(cluster.get("role_hint"))
                if role_hint:
                    return role_hint
        return None

    def _normalize_identity_names(
        self,
        identities: list[CharacterReviewIdentity],
        protagonist_name: str | None,
    ) -> list[CharacterReviewIdentity]:
        if not identities:
            return identities

        base_counts: dict[str, int] = {}
        normalized: list[CharacterReviewIdentity] = []
        for identity in identities:
            if identity.status == "confirmed":
                normalized.append(identity)
                continue

            current_name = self._clean_name(identity.name or identity.suggested_name)
            if not self._identity_name_needs_cleanup(current_name, identity.role_hint, protagonist_name):
                normalized.append(identity)
                continue

            base = self._identity_fallback_base(identity.role_hint, protagonist_name)
            base_counts[base] = base_counts.get(base, 0) + 1
            ordinal = base_counts[base]
            replacement = base if base == protagonist_name else f"{base} {ordinal}" if base_counts[base] > 1 or base == "Unidentified Character" else base
            normalized.append(
                identity.model_copy(
                    update={
                        "name": replacement,
                        "suggested_name": replacement,
                    }
                )
            )
        return normalized

    def _identity_name_needs_cleanup(
        self,
        name: str | None,
        role_hint: str | None,
        protagonist_name: str | None,
    ) -> bool:
        cleaned = self._clean_name(name)
        if not cleaned:
            return True
        lowered = cleaned.casefold()
        if protagonist_name and lowered == protagonist_name.casefold():
            return False
        if re.fullmatch(r"(?:stranger|character|other|unknown)(?:\s+\d+)?", lowered):
            return True
        if role_hint and lowered == str(role_hint).casefold():
            return True
        tokens = re.findall(r"[a-z]+", lowered)
        if not tokens:
            return True
        noisy_tokens = {
            "account", "building", "card", "chapter", "city", "countdown", "day", "days",
            "district", "door", "floor", "freeze", "global", "hall", "hotel", "manager",
            "month", "number", "page", "road", "room", "shelter", "station", "street",
            "tower", "world",
        }
        if any(token in noisy_tokens for token in tokens):
            return True
        if len(tokens[0]) <= 2 and len(tokens) > 1:
            return True
        return False

    def _identity_fallback_base(self, role_hint: str | None, protagonist_name: str | None) -> str:
        base = self._clean_name(role_hint) or "Unidentified Character"
        if protagonist_name and base.casefold() == "protagonist":
            return protagonist_name
        if base.casefold() in {"stranger", "character", "other", "unknown"}:
            return "Unidentified Character"
        return base

    def _relative_media_url(self, path: Path) -> str:
        relative_path = path.resolve().relative_to(self.settings.data_dir.resolve()).as_posix()
        return f"/media/{relative_path}"

    def _stable_character_sort_key(self, value: str) -> tuple[int, str]:
        match = re.search(r"(\d+)$", str(value))
        if match:
            return (int(match.group(1)), str(value))
        return (999999, str(value))

    def _normalize_tokens(self, value: object) -> str:
        tokens = re.findall(r"[a-z0-9]+", str(value or "").casefold())
        return " ".join(tokens).strip()

    def _clean_name(self, value: object) -> str | None:
        text = str(value or "").strip()
        return text or None

    def _dedupe_preserve_order(self, values: list[str]) -> list[str]:
        seen: set[str] = set()
        ordered: list[str] = []
        for value in values:
            cleaned = str(value or "").strip()
            if not cleaned:
                continue
            key = cleaned.casefold()
            if key in seen:
                continue
            seen.add(key)
            ordered.append(cleaned)
        return ordered
