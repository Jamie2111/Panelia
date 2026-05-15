from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from app.core.config import get_settings
from app.schemas.project import PanelBox

logger = logging.getLogger(__name__)


class CrossPagePanelMerger:
    def __init__(self) -> None:
        self.settings = get_settings()

    def merge(
        self,
        page_paths: list[Path],
        panels: list[PanelBox],
        progress_callback: callable | None = None,
        cancel_callback: callable | None = None,
    ) -> tuple[list[PanelBox], dict[str, Any]]:
        if not self.settings.cross_page_panel_merging_enabled or len(page_paths) < 2 or not panels:
            return panels, {"enabled": False, "summary": "Cross-page merging disabled.", "merges": []}

        page_images: dict[int, np.ndarray] = {}
        page_sizes: dict[int, tuple[int, int]] = {}
        page_panels: dict[int, list[PanelBox]] = {}
        for panel in sorted(panels, key=lambda item: (item.page, item.order)):
            page_panels.setdefault(int(panel.page), []).append(panel)

        merges: list[dict[str, Any]] = []
        panel_updates: dict[str, dict[str, Any]] = {}
        total_pairs = max(len(page_paths) - 1, 1)

        for page_number in range(1, len(page_paths)):
            if cancel_callback:
                cancel_callback()
            current_image = page_images.get(page_number)
            if current_image is None:
                current_image = np.array(Image.open(page_paths[page_number - 1]).convert("RGB"))
                page_images[page_number] = current_image
            next_image = page_images.get(page_number + 1)
            if next_image is None:
                next_image = np.array(Image.open(page_paths[page_number]).convert("RGB"))
                page_images[page_number + 1] = next_image

            page_sizes[page_number] = (current_image.shape[1], current_image.shape[0])
            page_sizes[page_number + 1] = (next_image.shape[1], next_image.shape[0])
            bottom_candidates = self._bottom_edge_panels(page_panels.get(page_number, []), current_image.shape[0])
            top_candidates = self._top_edge_panels(page_panels.get(page_number + 1, []), next_image.shape[0])
            if not bottom_candidates or not top_candidates:
                continue

            for bottom_panel in bottom_candidates:
                for top_panel in top_candidates:
                    candidate = self._build_merge_candidate(
                        bottom_panel,
                        top_panel,
                        current_image,
                        next_image,
                    )
                    if candidate is None:
                        continue
                    logical_id = str(bottom_panel.logical_panel_id or bottom_panel.id)
                    paired_ids = sorted({*bottom_panel.continuation_panel_ids, *top_panel.continuation_panel_ids, bottom_panel.id, top_panel.id})
                    bottom_spans = sorted({*bottom_panel.spans_pages, int(bottom_panel.page), int(top_panel.page)})
                    top_spans = sorted({*top_panel.spans_pages, int(bottom_panel.page), int(top_panel.page)})
                    # Preserve keep=False if this bottom panel was already marked as a
                    # continuation in a previous merge (i.e. it spans 3+ pages).
                    existing_bottom = panel_updates.get(bottom_panel.id, {})
                    bottom_update: dict[str, Any] = {
                        **existing_bottom,
                        "logical_panel_id": logical_id,
                        "multi_page_panel": True,
                        "continuation_panel_ids": [panel_id for panel_id in paired_ids if panel_id != bottom_panel.id],
                        "spans_pages": bottom_spans,
                    }
                    panel_updates[bottom_panel.id] = bottom_update
                    panel_updates[top_panel.id] = {
                        "logical_panel_id": logical_id,
                        "multi_page_panel": True,
                        "continuation_panel_ids": [panel_id for panel_id in paired_ids if panel_id != top_panel.id],
                        "spans_pages": top_spans,
                        "keep": False,
                        "auto_skipped": True,
                        "skip_reason": (
                            f"continuation of cross-page panel "
                            f"(pages {int(bottom_panel.page)}-{int(top_panel.page)})"
                        ),
                    }
                    merges.append(candidate)
                    break

            if progress_callback:
                progress_callback(
                    page_number / total_pairs * 100,
                    f"Checked cross-page panel continuity {page_number}/{len(page_paths) - 1}",
                )

        updated: list[PanelBox] = []
        for panel in panels:
            update = panel_updates.get(panel.id)
            updated.append(panel.model_copy(update=update) if update else panel)

        summary = {
            "enabled": True,
            "merge_count": len(merges),
            "merges": merges,
            "summary": (
                f"Linked {len(merges)} cross-page panel continuations."
                if merges
                else "No cross-page continuations were linked."
            ),
        }
        return updated, summary

    def _bottom_edge_panels(self, panels: list[PanelBox], page_height: int) -> list[PanelBox]:
        edge_margin = max(int(page_height * self.settings.cross_page_merge_edge_ratio), 90)
        # Exclude full-page/splash panels - their uniform top/bottom borders are
        # visually similar across any two consecutive pages, which causes false
        # cross-page merges at our 0.65 similarity threshold.  A genuine cross-page
        # panel is always a partial-page region; it never spans nearly the full height.
        full_page_height = page_height * 0.82
        return [
            panel
            for panel in panels
            if panel.keep
            and (panel.y + panel.height) >= page_height - edge_margin
            and panel.height < full_page_height
        ]

    def _top_edge_panels(self, panels: list[PanelBox], page_height: int) -> list[PanelBox]:
        edge_margin = max(int(page_height * self.settings.cross_page_merge_edge_ratio), 90)
        full_page_height = page_height * 0.82
        return [
            panel
            for panel in panels
            if panel.keep
            and panel.y <= edge_margin
            and panel.height < full_page_height
        ]

    def _build_merge_candidate(
        self,
        bottom_panel: PanelBox,
        top_panel: PanelBox,
        current_image: np.ndarray,
        next_image: np.ndarray,
    ) -> dict[str, Any] | None:
        x_overlap = self._axis_overlap_ratio(
            bottom_panel.x,
            bottom_panel.x + bottom_panel.width,
            top_panel.x,
            top_panel.x + top_panel.width,
        )
        width_similarity = min(bottom_panel.width, top_panel.width) / max(max(bottom_panel.width, top_panel.width), 1)
        if x_overlap < 0.45 and width_similarity < 0.65:
            return None

        bottom_strip = self._edge_strip(current_image, bottom_panel, bottom=True)
        top_strip = self._edge_strip(next_image, top_panel, bottom=False)
        if bottom_strip.size == 0 or top_strip.size == 0:
            return None

        similarity = self._strip_similarity(bottom_strip, top_strip)
        if similarity < self.settings.cross_page_merge_similarity_threshold:
            return None

        return {
            "logical_panel_id": str(bottom_panel.logical_panel_id or bottom_panel.id),
            "pages": [int(bottom_panel.page), int(top_panel.page)],
            "panel_ids": [bottom_panel.id, top_panel.id],
            "background_similarity": round(similarity, 4),
        }

    def _edge_strip(self, image: np.ndarray, panel: PanelBox, *, bottom: bool) -> np.ndarray:
        x0 = max(int(panel.x), 0)
        y0 = max(int(panel.y), 0)
        x1 = min(int(panel.x + panel.width), image.shape[1])
        y1 = min(int(panel.y + panel.height), image.shape[0])
        if x1 <= x0 or y1 <= y0:
            return np.zeros((0, 0, 3), dtype=np.uint8)
        crop = image[y0:y1, x0:x1]
        strip_height = max(int(crop.shape[0] * 0.10), 30)
        return crop[-strip_height:, :, :] if bottom else crop[:strip_height, :, :]

    def _strip_similarity(self, left: np.ndarray, right: np.ndarray) -> float:
        target_width = max(min(left.shape[1], right.shape[1]), 1)
        target_height = max(min(left.shape[0], right.shape[0]), 1)
        left_resized = self._resize_strip(left, target_width, target_height)
        right_resized = self._resize_strip(right, target_width, target_height)
        if left_resized.size == 0 or right_resized.size == 0:
            return 0.0
        difference = np.mean(np.abs(left_resized.astype(np.float32) - right_resized.astype(np.float32)))
        return max(0.0, 1.0 - (difference / 255.0))

    def _resize_strip(self, strip: np.ndarray, width: int, height: int) -> np.ndarray:
        if strip.size == 0:
            return np.zeros((0, 0, 3), dtype=np.uint8)
        image = Image.fromarray(strip)
        resized = image.resize((max(width, 1), max(height, 1)))
        return np.array(resized)

    def _axis_overlap_ratio(self, start_a: int, end_a: int, start_b: int, end_b: int) -> float:
        overlap = max(0, min(end_a, end_b) - max(start_a, start_b))
        if overlap <= 0:
            return 0.0
        return overlap / max(min(end_a - start_a, end_b - start_b), 1)
