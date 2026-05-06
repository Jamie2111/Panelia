from __future__ import annotations

import logging
from pathlib import Path
from threading import Lock
from typing import Any

import numpy as np
from PIL import Image

from app.core.config import get_settings

logger = logging.getLogger(__name__)


class MagiHFService:
    """Lazy loader and normalizer for the upstream MAGI Hugging Face models."""

    _MODELS: dict[str, Any] = {}
    _MODEL_DEVICES: dict[str, str] = {}
    _LOAD_LOCK = Lock()

    def __init__(self) -> None:
        self.settings = get_settings()

    def provider_tag(self) -> str:
        if not self.settings.magi_enabled:
            return "disabled"
        return (
            f"magi-hf:{self.settings.magi_model_id}:"
            f"ocr={int(bool(self.settings.magi_dialogue_ocr_enabled))}:"
            f"webtoon_panels={int(bool(self.settings.magi_detect_webtoon_panels))}"
        )

    def is_available(self) -> bool:
        if not self.settings.magi_enabled:
            return False
        try:
            import torch  # noqa: F401
            from transformers import AutoModel  # noqa: F401
        except Exception as exc:
            logger.warning("MAGI dependencies are unavailable: %s", exc)
            return False
        return True

    def load_model(self, model_id: str | None = None) -> Any | None:
        resolved_model_id = str(model_id or self.settings.magi_model_id or "").strip()
        if not resolved_model_id or not self.is_available():
            return None

        if self.settings.magi_local_files_only and not self._has_cached_model_weights(resolved_model_id):
            logger.info(
                "Skipping MAGI model %s because local-only mode is enabled and weights are not cached",
                resolved_model_id,
            )
            return None

        with self._LOAD_LOCK:
            cached = self._MODELS.get(resolved_model_id)
            if cached is not None:
                return cached

            try:
                import torch
                from transformers import AutoModel
            except Exception as exc:
                logger.warning("Failed to import MAGI runtime dependencies: %s", exc)
                return None

            try:
                load_kwargs: dict[str, Any] = {
                    "trust_remote_code": True,
                    "local_files_only": bool(self.settings.magi_local_files_only),
                }
                if (
                    self.settings.magi_local_files_only
                    and self._cached_file_path(resolved_model_id, "pytorch_model.bin") is not None
                    and self._cached_file_path(resolved_model_id, "model.safetensors") is None
                ):
                    load_kwargs["use_safetensors"] = False
                model = AutoModel.from_pretrained(resolved_model_id, **load_kwargs)
            except Exception as exc:
                logger.warning("Failed to load MAGI model %s: %s", resolved_model_id, exc)
                return None

            last_error: Exception | None = None
            for device in self._candidate_devices(torch):
                try:
                    model = model.to(device)
                    model.eval()
                    self._MODELS[resolved_model_id] = model
                    self._MODEL_DEVICES[resolved_model_id] = device
                    logger.info("Loaded MAGI model %s on %s", resolved_model_id, device)
                    return model
                except Exception as exc:
                    last_error = exc
                    logger.warning("Failed to move MAGI model %s to %s: %s", resolved_model_id, device, exc)

            logger.warning("MAGI model %s could not be initialized on any device: %s", resolved_model_id, last_error)
            return None

    def _has_cached_model_weights(self, model_id: str) -> bool:
        """Avoid slow network retries when MAGI is configured for local-only use."""
        model_path = Path(model_id).expanduser()
        if model_path.exists():
            return True
        # The upstream MAGI remote code requests model.safetensors internally.
        # A cached pytorch_model.bin alone still causes a slow network lookup,
        # so local-only mode treats it as unavailable and falls back to CV.
        if self._cached_file_path(model_id, "model.safetensors") is None:
            return False

        for filename in ("model.safetensors", "pytorch_model.bin"):
            cached = self._cached_file_path(model_id, filename)
            if cached is not None:
                return True
        return False

    def _cached_file_path(self, model_id: str, filename: str) -> Path | None:
        try:
            from huggingface_hub import try_to_load_from_cache
        except Exception:
            return None
        try:
            cached = try_to_load_from_cache(model_id, filename)
        except Exception:
            return None
        if not isinstance(cached, str):
            return None
        path = Path(cached)
        return path if path.exists() else None

    def predict_page_payloads(
        self,
        page_paths: list[Path],
        page_numbers: list[int] | None = None,
        *,
        do_ocr: bool | None = None,
        batch_size: int | None = None,
        model: Any | None = None,
        cancel_callback: callable | None = None,
        progress_callback: callable | None = None,
        progress_label: str = "Scanning chapter structure with MAGI",
    ) -> dict[int, dict[str, Any]]:
        active_model = model or self.load_model()
        if active_model is None:
            return {}

        selected_pages = sorted(
            {
                int(page_number)
                for page_number in (page_numbers or range(1, len(page_paths) + 1))
                if 1 <= int(page_number) <= len(page_paths)
            }
        )
        if not selected_pages:
            return {}

        run_ocr = self.settings.magi_dialogue_ocr_enabled if do_ocr is None else bool(do_ocr)
        page_payloads: dict[int, dict[str, Any]] = {}
        active_batch_size = max(int(batch_size or self.settings.magi_batch_size or 1), 1)

        try:
            import torch
        except Exception as exc:
            logger.warning("MAGI inference requires torch but it could not be imported: %s", exc)
            return {}

        total_batches = max((len(selected_pages) + active_batch_size - 1) // active_batch_size, 1)
        for batch_index, start in enumerate(range(0, len(selected_pages), active_batch_size), start=1):
            if cancel_callback:
                cancel_callback()
            batch_page_numbers = selected_pages[start:start + active_batch_size]
            if progress_callback:
                processed_pages = min(start + len(batch_page_numbers), len(selected_pages))
                progress_callback(
                    ((batch_index - 1) / total_batches) * 100.0,
                    f"{progress_label} ({processed_pages}/{len(selected_pages)} pages loaded)",
                )
            batch_images = [self._read_page_as_array(page_paths[page_number - 1]) for page_number in batch_page_numbers]
            if not batch_images:
                continue

            try:
                with torch.inference_mode():
                    batch_results = active_model.predict_detections_and_associations(batch_images)
                    batch_ocr = None
                    if run_ocr:
                        text_boxes = [result.get("texts") or [] for result in batch_results]
                        batch_ocr = active_model.predict_ocr(batch_images, text_boxes, use_tqdm=False)
            except Exception as exc:
                logger.warning("MAGI inference failed for pages %s: %s", batch_page_numbers, exc)
                continue

            for offset, page_number in enumerate(batch_page_numbers):
                result = batch_results[offset] if offset < len(batch_results) else {}
                ocr_texts = batch_ocr[offset] if batch_ocr and offset < len(batch_ocr) else None
                page_payloads[page_number] = self._normalize_page_result(page_number, result, ocr_texts)

        if progress_callback:
            progress_callback(100.0, "MAGI chapter scan complete")
        return page_payloads

    def _read_page_as_array(self, page_path: Path) -> np.ndarray:
        with page_path.open("rb") as file_handle:
            return np.array(Image.open(file_handle).convert("L").convert("RGB"))

    def _normalize_page_result(
        self,
        page_number: int,
        result: dict[str, Any],
        ocr_texts: list[str] | None,
    ) -> dict[str, Any]:
        raw_panels = (
            result.get("panels")
            or result.get("panel_bboxes")
            or result.get("panel_boxes")
            or []
        )
        raw_texts = result.get("texts") or []
        raw_characters = result.get("characters") or []
        raw_text_character_pairs = result.get("text_character_associations") or []
        raw_dialogue_flags = result.get("is_essential_text") or []
        raw_cluster_labels = result.get("character_cluster_labels") or []

        character_entries: list[dict[str, Any]] = []
        index_to_character_id: dict[int, str] = {}
        cluster_labels = self._coerce_int_list(raw_cluster_labels, len(raw_characters))
        for character_index, raw_character in enumerate(raw_characters):
            bbox = self._coerce_xyxy_bbox(raw_character)
            if bbox is None:
                continue
            character_id = f"magi-p{page_number:04d}-char-{character_index + 1:03d}"
            index_to_character_id[character_index] = character_id
            character_entries.append(
                {
                    "character_index": character_index,
                    "character_id": character_id,
                    "bbox": self._xyxy_to_xywh(bbox),
                    "cluster_label_local": (
                        cluster_labels[character_index]
                        if character_index < len(cluster_labels)
                        else character_index
                    ),
                    "source": "magi-hf",
                }
            )

        text_to_character_index = {
            text_index: character_index
            for text_index, character_index in self._coerce_index_pairs(raw_text_character_pairs)
        }
        dialogue_flags = self._coerce_bool_list(raw_dialogue_flags, len(raw_texts), default=True)
        text_entries: list[dict[str, Any]] = []
        for text_index, raw_text in enumerate(raw_texts):
            bbox = self._coerce_xyxy_bbox(raw_text)
            if bbox is None:
                continue
            character_index = text_to_character_index.get(text_index)
            character_id = index_to_character_id.get(character_index) if character_index is not None else None
            text_entries.append(
                {
                    "text_index": text_index,
                    "bbox": self._xyxy_to_xywh(bbox),
                    "character_id": character_id,
                    "character_index": character_index,
                    "is_dialogue": (
                        dialogue_flags[text_index]
                        if text_index < len(dialogue_flags)
                        else True
                    ),
                    "text": str(ocr_texts[text_index]).strip() if ocr_texts and text_index < len(ocr_texts) else "",
                    "source": "magi-hf",
                }
            )

        panel_entries: list[dict[str, Any]] = []
        for panel_index, raw_panel in enumerate(raw_panels):
            bbox = self._coerce_xyxy_bbox(raw_panel)
            if bbox is None:
                continue
            panel_entries.append(
                {
                    "panel_index": panel_index,
                    "bbox": self._xyxy_to_xywh(bbox),
                    "order": panel_index + 1,
                    "source": "magi-hf",
                }
            )

        return {
            "page": page_number,
            "provider": self.provider_tag(),
            "panels": panel_entries,
            "texts": text_entries,
            "characters": character_entries,
        }

    def _candidate_devices(self, torch_module: Any) -> list[str]:
        devices: list[str] = []
        if torch_module.cuda.is_available():
            devices.append("cuda")
        mps_backend = getattr(torch_module.backends, "mps", None)
        if mps_backend and mps_backend.is_available():
            devices.append("mps")
        devices.append("cpu")
        return devices

    def _coerce_xyxy_bbox(self, value: Any) -> list[int] | None:
        candidate = value.tolist() if hasattr(value, "tolist") else value
        if not isinstance(candidate, (list, tuple)) or len(candidate) < 4:
            return None
        try:
            x1, y1, x2, y2 = [int(round(float(component))) for component in candidate[:4]]
        except (TypeError, ValueError):
            return None
        x1, x2 = sorted((x1, x2))
        y1, y2 = sorted((y1, y2))
        if x2 <= x1 or y2 <= y1:
            return None
        return [x1, y1, x2, y2]

    def _xyxy_to_xywh(self, bbox: list[int]) -> list[int]:
        x1, y1, x2, y2 = bbox
        return [x1, y1, max(1, x2 - x1), max(1, y2 - y1)]

    def _coerce_index_pairs(self, value: Any) -> list[tuple[int, int]]:
        pairs: list[tuple[int, int]] = []
        candidate = value.tolist() if hasattr(value, "tolist") else value
        if not isinstance(candidate, (list, tuple)):
            return pairs
        for item in candidate:
            pair = item.tolist() if hasattr(item, "tolist") else item
            if not isinstance(pair, (list, tuple)) or len(pair) < 2:
                continue
            try:
                pairs.append((int(pair[0]), int(pair[1])))
            except (TypeError, ValueError):
                continue
        return pairs

    def _coerce_bool_list(self, value: Any, size: int, *, default: bool) -> list[bool]:
        candidate = value.tolist() if hasattr(value, "tolist") else value
        if not isinstance(candidate, (list, tuple)):
            return [default] * size
        flags: list[bool] = []
        for item in candidate[:size]:
            if isinstance(item, (bool, int, float)):
                flags.append(bool(item))
            else:
                flags.append(default)
        while len(flags) < size:
            flags.append(default)
        return flags

    def _coerce_int_list(self, value: Any, size: int) -> list[int]:
        candidate = value.tolist() if hasattr(value, "tolist") else value
        if not isinstance(candidate, (list, tuple)):
            return list(range(size))
        numbers: list[int] = []
        for item in candidate[:size]:
            try:
                numbers.append(int(item))
            except (TypeError, ValueError):
                numbers.append(len(numbers))
        while len(numbers) < size:
            numbers.append(len(numbers))
        return numbers
