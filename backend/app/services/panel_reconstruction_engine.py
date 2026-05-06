from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
from PIL import Image

from app.core.config import get_settings
from app.schemas.project import ChapterMetadata, PanelBox
from app.services.multilingual_ocr import MultilingualOCRService

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class OCRTextBox:
    text: str
    x: int
    y: int
    width: int
    height: int
    confidence: float | None = None

    @property
    def center_x(self) -> float:
        return self.x + self.width / 2

    @property
    def center_y(self) -> float:
        return self.y + self.height / 2


class PanelReconstructionEngine:
    def __init__(self, ocr_service: MultilingualOCRService | None = None) -> None:
        self.settings = get_settings()
        self._ocr = ocr_service or MultilingualOCRService()

    def reconstruct(
        self,
        page_paths: list[Path],
        detected_panels: list[PanelBox],
        metadata: ChapterMetadata | None = None,
        detector_text_boxes_by_page: dict[int, list[tuple[int, int, int, int]]] | None = None,
        progress_callback: callable | None = None,
        cancel_callback: callable | None = None,
    ) -> tuple[list[PanelBox], dict[str, Any]]:
        if not self.settings.panel_reconstruction_enabled or not page_paths:
            return list(detected_panels), {"enabled": False, "summary": "Panel reconstruction disabled."}

        page_panels: dict[int, list[PanelBox]] = {}
        for panel in sorted(detected_panels, key=lambda item: (item.page, item.order)):
            page_panels.setdefault(int(panel.page), []).append(panel)

        reconstructed: list[PanelBox] = []
        report_pages: list[dict[str, Any]] = []
        page_text_boxes: dict[int, list[dict[str, Any]]] = {}
        total_pages = max(len(page_paths), 1)

        for page_number, path in enumerate(page_paths, start=1):
            if cancel_callback:
                cancel_callback()
            image = np.array(Image.open(path).convert("RGB"))
            page_height, page_width = image.shape[:2]
            language_hint = str((metadata.language if metadata else "") or "en").strip() or "en"
            detector_boxes = list((detector_text_boxes_by_page or {}).get(page_number, []) or [])
            ocr_source = "skipped_full_page_ocr_disabled"
            if detector_boxes:
                fragments = [
                    OCRTextBox(text="", x=int(x), y=int(y), width=int(width), height=int(height), confidence=None)
                    for x, y, width, height in detector_boxes
                    if int(width) > 0 and int(height) > 0
                ]
                ocr_source = "detector_text_boxes"
            elif self.settings.panel_reconstruction_full_page_ocr_enabled:
                fragments = self._extract_text_boxes(image, language_hint)
                ocr_source = "full_page_ocr"
            else:
                fragments = []
            # Persist page-level OCR text boxes for downstream dialogue backfill
            page_text_boxes[page_number] = [
                {
                    "text": box.text,
                    "x": box.x,
                    "y": box.y,
                    "width": box.width,
                    "height": box.height,
                    "confidence": box.confidence,
                }
                for box in fragments
                if box.text.strip()
            ]
            clusters = self._cluster_text_boxes(fragments, page_width, page_height)
            rebuilt_page_panels, page_report = self._rebuild_page_panels(
                image=image,
                page_number=page_number,
                detected_panels=page_panels.get(page_number, []),
                clusters=clusters,
            )
            page_report["ocr_source"] = ocr_source
            reconstructed.extend(rebuilt_page_panels)
            report_pages.append(page_report)
            if progress_callback:
                progress_callback(
                    page_number / total_pages * 100,
                    f"Reconstructed logical panels from OCR on page {page_number}/{total_pages}",
                )

        ordered = self._assign_reading_order(reconstructed, metadata)
        summary = {
            "enabled": True,
            "detected_panels": len(detected_panels),
            "reconstructed_panels": len(ordered),
            "ocr_cluster_panels": sum(1 for panel in ordered if (panel.reconstruction_source or "").startswith("ocr")),
            "hybrid_panels": sum(1 for panel in ordered if panel.reconstruction_source == "detector+ocr_cluster"),
            "summary": (
                f"Reconstructed {len(ordered)} panels from detector + text-box geometry "
                f"({sum(1 for panel in ordered if panel.reconstruction_source == 'ocr_cluster')} OCR-only recoveries)."
            ),
            "pages": report_pages,
            "page_text_boxes": page_text_boxes,
        }
        return ordered, summary

    def _extract_text_boxes(self, image: np.ndarray, language_hint: str) -> list[OCRTextBox]:
        fragments = self._ocr.extract(image, language_hint=language_hint)
        text_boxes: list[OCRTextBox] = []
        for fragment in fragments:
            if not fragment.text:
                continue
            x, y, width, height = [int(value) for value in fragment.bbox[:4]]
            text_boxes.append(
                OCRTextBox(
                    text=str(fragment.text).strip(),
                    x=x,
                    y=y,
                    width=width,
                    height=height,
                    confidence=fragment.confidence,
                )
            )
        return text_boxes

    def _cluster_text_boxes(
        self,
        boxes: list[OCRTextBox],
        page_width: int,
        page_height: int,
    ) -> list[list[OCRTextBox]]:
        if not boxes:
            return []
        parent = list(range(len(boxes)))

        def find(index: int) -> int:
            while parent[index] != index:
                parent[index] = parent[parent[index]]
                index = parent[index]
            return index

        def union(left: int, right: int) -> None:
            root_left = find(left)
            root_right = find(right)
            if root_left != root_right:
                parent[root_right] = root_left

        for left in range(len(boxes)):
            for right in range(left + 1, len(boxes)):
                if self._boxes_should_cluster(boxes[left], boxes[right], page_width, page_height):
                    union(left, right)

        grouped: dict[int, list[OCRTextBox]] = {}
        for index, box in enumerate(boxes):
            grouped.setdefault(find(index), []).append(box)

        clusters = sorted(
            grouped.values(),
            key=lambda items: (
                min(box.center_y for box in items),
                min(box.center_x for box in items),
            ),
        )
        return [cluster for cluster in clusters if cluster]

    def _boxes_should_cluster(
        self,
        left: OCRTextBox,
        right: OCRTextBox,
        page_width: int,
        page_height: int,
    ) -> bool:
        horizontal_gap = self._axis_gap(left.x, left.x + left.width, right.x, right.x + right.width)
        vertical_gap = self._axis_gap(left.y, left.y + left.height, right.y, right.y + right.height)
        mean_width = max((left.width + right.width) / 2, 1.0)
        mean_height = max((left.height + right.height) / 2, 1.0)
        near_x = horizontal_gap <= max(mean_width * 2.4, page_width * self.settings.panel_reconstruction_cluster_distance_ratio)
        near_y = vertical_gap <= max(mean_height * 2.2, page_height * (self.settings.panel_reconstruction_cluster_distance_ratio * 0.75))
        stacked = horizontal_gap <= max(mean_width * 1.2, 42.0) and vertical_gap <= max(mean_height * 2.8, page_height * 0.06)
        inline = vertical_gap <= max(mean_height * 0.8, 18.0) and horizontal_gap <= max(mean_width * 3.2, page_width * 0.12)
        return (near_x and near_y) or stacked or inline

    def _rebuild_page_panels(
        self,
        image: np.ndarray,
        page_number: int,
        detected_panels: list[PanelBox],
        clusters: list[list[OCRTextBox]],
    ) -> tuple[list[PanelBox], dict[str, Any]]:
        page_height, page_width = image.shape[:2]
        seeds: list[dict[str, Any]] = []
        cluster_reports: list[dict[str, Any]] = []

        for panel in detected_panels:
            seeds.append(
                {
                    "panel": panel,
                    "bbox": [int(panel.x), int(panel.y), int(panel.width), int(panel.height)],
                    "source": "detector",
                    "matched_clusters": 0,
                }
            )

        for cluster in clusters:
            cluster_bbox = self._cluster_bbox(cluster, page_width, page_height)
            overlapping_indexes = [
                index
                for index, seed in enumerate(seeds)
                if self._bbox_overlap_ratio(cluster_bbox, seed["bbox"]) >= self.settings.panel_reconstruction_overlap_threshold
            ]
            cluster_reports.append(
                {
                    "text_boxes": len(cluster),
                    "bbox": cluster_bbox,
                    "matched_detector_panels": len(overlapping_indexes),
                }
            )
            if overlapping_indexes:
                target_index = overlapping_indexes[0]
                for merge_index in reversed(overlapping_indexes[1:]):
                    seeds[target_index]["bbox"] = self._union_bbox(seeds[target_index]["bbox"], seeds[merge_index]["bbox"], page_width, page_height)
                    seeds[target_index]["panel"] = seeds[target_index]["panel"].model_copy(
                        update={
                            "merged_from": sorted(
                                {
                                    *seeds[target_index]["panel"].merged_from,
                                    seeds[target_index]["panel"].id,
                                    seeds[merge_index]["panel"].id,
                                    *seeds[merge_index]["panel"].merged_from,
                                }
                            )
                        }
                    )
                    seeds.pop(merge_index)
                # Detector-matched panels already have accurate visual bounds.
                # Only expand the bbox to include the cluster (not full content
                # expansion), so we don't overshoot whitespace gutters on tall
                # webtoon pages and accidentally merge adjacent panels.
                seeds[target_index]["bbox"] = self._union_bbox(
                    seeds[target_index]["bbox"], cluster_bbox, page_width, page_height
                )
                seeds[target_index]["source"] = "detector+ocr_cluster"
                seeds[target_index]["matched_clusters"] += 1
            else:
                expanded = self._expand_bbox_to_content(image, cluster_bbox, page_width, page_height)
                seeds.append(
                    {
                        "panel": None,
                        "bbox": expanded,
                        "source": "ocr_cluster",
                        "matched_clusters": 1,
                    }
                )

        page_panels: list[PanelBox] = []
        for seed in self._dedupe_seed_boxes(seeds, page_width, page_height):
            bbox = [int(value) for value in seed["bbox"][:4]]
            panel = seed["panel"]
            if panel is None:
                panel = PanelBox(
                    id=f"recon_{uuid4().hex[:12]}",
                    page=page_number,
                    panel=0,
                    x=bbox[0],
                    y=bbox[1],
                    width=bbox[2],
                    height=bbox[3],
                    order=0,
                    keep=True,
                    merged_from=[],
                    reconstruction_source=str(seed["source"]),
                    reconstruction_confidence=min(0.98, 0.55 + min(int(seed["matched_clusters"]), 3) * 0.12),
                    review_flags=["ocr_reconstructed"],
                    logical_panel_id=None,
                    spans_pages=[page_number],
                )
            else:
                panel = panel.model_copy(
                    update={
                        "x": bbox[0],
                        "y": bbox[1],
                        "width": bbox[2],
                        "height": bbox[3],
                        "reconstruction_source": str(seed["source"]),
                        "reconstruction_confidence": min(0.99, 0.72 + min(int(seed["matched_clusters"]), 3) * 0.08),
                        "spans_pages": sorted({*panel.spans_pages, page_number}),
                    }
                )
            page_panels.append(panel)

        page_panels = sorted(page_panels, key=lambda item: (item.y + item.height / 2, item.x + item.width / 2))
        page_report = {
            "page": page_number,
            "detected_panels": len(detected_panels),
            "ocr_clusters": len(clusters),
            "reconstructed_panels": len(page_panels),
            "clusters": cluster_reports,
        }
        return page_panels, page_report

    def _cluster_bbox(
        self,
        cluster: list[OCRTextBox],
        page_width: int,
        page_height: int,
    ) -> list[int]:
        min_x = min(box.x for box in cluster)
        min_y = min(box.y for box in cluster)
        max_x = max(box.x + box.width for box in cluster)
        max_y = max(box.y + box.height for box in cluster)
        width = max_x - min_x
        height = max_y - min_y
        pad_x = max(18, int(width * self.settings.panel_reconstruction_text_margin_ratio))
        pad_y = max(18, int(height * self.settings.panel_reconstruction_text_margin_ratio * 1.3))
        x = max(min_x - pad_x, 0)
        y = max(min_y - pad_y, 0)
        right = min(max_x + pad_x, page_width)
        bottom = min(max_y + pad_y, page_height)
        return [int(x), int(y), int(max(right - x, 1)), int(max(bottom - y, 1))]

    def _expand_bbox_to_content(
        self,
        image: np.ndarray,
        bbox: list[int],
        page_width: int,
        page_height: int,
    ) -> list[int]:
        x, y, width, height = [int(value) for value in bbox[:4]]
        pad_x = max(int(width * 0.4), 40)
        pad_y = max(int(height * 0.65), 48)
        x0 = max(x - pad_x, 0)
        y0 = max(y - pad_y, 0)
        x1 = min(x + width + pad_x, page_width)
        y1 = min(y + height + pad_y, page_height)
        region = image[y0:y1, x0:x1]
        if region.size == 0:
            return bbox

        gray = np.mean(region, axis=2)
        content_mask = gray < 242
        coords = np.argwhere(content_mask)
        if coords.size == 0:
            return bbox

        min_row = int(coords[:, 0].min())
        max_row = int(coords[:, 0].max()) + 1
        min_col = int(coords[:, 1].min())
        max_col = int(coords[:, 1].max()) + 1
        content_width = max_col - min_col
        content_height = max_row - min_row
        if content_width < width * 0.75 and content_height < height * 0.75:
            return bbox

        extra_x = max(10, int(content_width * 0.04))
        extra_y = max(10, int(content_height * 0.04))
        new_x = max(x0 + min_col - extra_x, 0)
        new_y = max(y0 + min_row - extra_y, 0)
        new_right = min(x0 + max_col + extra_x, page_width)
        new_bottom = min(y0 + max_row + extra_y, page_height)
        expanded = [int(new_x), int(new_y), int(max(new_right - new_x, 1)), int(max(new_bottom - new_y, 1))]
        return expanded

    def _dedupe_seed_boxes(
        self,
        seeds: list[dict[str, Any]],
        page_width: int,
        page_height: int,
    ) -> list[dict[str, Any]]:
        ordered = sorted(seeds, key=lambda item: ((item["bbox"][1] + item["bbox"][3] / 2), (item["bbox"][0] + item["bbox"][2] / 2)))
        deduped: list[dict[str, Any]] = []
        for seed in ordered:
            bbox = [int(value) for value in seed["bbox"][:4]]
            if bbox[2] < 32 or bbox[3] < 32:
                continue
            merged = False
            for existing in deduped:
                overlap = self._bbox_overlap_ratio(bbox, existing["bbox"])
                contains = self._bbox_contains(existing["bbox"], bbox) or self._bbox_contains(bbox, existing["bbox"])
                if overlap >= 0.76 or contains:
                    existing["bbox"] = self._union_bbox(existing["bbox"], bbox, page_width, page_height)
                    if existing["panel"] is None:
                        existing["panel"] = seed["panel"]
                    existing["source"] = self._combined_source(str(existing["source"]), str(seed["source"]))
                    existing["matched_clusters"] = int(existing["matched_clusters"]) + int(seed["matched_clusters"])
                    merged = True
                    break
            if not merged:
                deduped.append(dict(seed))
        return deduped

    def _assign_reading_order(
        self,
        panels: list[PanelBox],
        metadata: ChapterMetadata | None = None,
    ) -> list[PanelBox]:
        ordered = sorted(
            panels,
            key=lambda item: (
                int(item.page),
                item.y + item.height / 2,
                item.x + item.width / 2,
                item.order,
            ),
        )
        page_counters: dict[int, int] = {}
        normalized: list[PanelBox] = []
        for order, panel in enumerate(ordered, start=1):
            page_number = int(panel.page)
            page_counters[page_number] = page_counters.get(page_number, 0) + 1
            logical_id = panel.logical_panel_id or panel.id
            normalized.append(
                panel.model_copy(
                    update={
                        "order": order,
                        "panel": page_counters[page_number],
                        "logical_panel_id": logical_id,
                    }
                )
            )
        return normalized

    def _bbox_overlap_ratio(self, left: list[int], right: list[int]) -> float:
        intersection = self._intersection_area(left, right)
        if intersection <= 0:
            return 0.0
        left_area = max(int(left[2]), 1) * max(int(left[3]), 1)
        right_area = max(int(right[2]), 1) * max(int(right[3]), 1)
        return intersection / max(min(left_area, right_area), 1)

    def _bbox_contains(self, outer: list[int], inner: list[int]) -> bool:
        return (
            outer[0] <= inner[0]
            and outer[1] <= inner[1]
            and outer[0] + outer[2] >= inner[0] + inner[2]
            and outer[1] + outer[3] >= inner[1] + inner[3]
        )

    def _intersection_area(self, left: list[int], right: list[int]) -> int:
        x1 = max(int(left[0]), int(right[0]))
        y1 = max(int(left[1]), int(right[1]))
        x2 = min(int(left[0] + left[2]), int(right[0] + right[2]))
        y2 = min(int(left[1] + left[3]), int(right[1] + right[3]))
        if x2 <= x1 or y2 <= y1:
            return 0
        return int((x2 - x1) * (y2 - y1))

    def _union_bbox(self, left: list[int], right: list[int], page_width: int, page_height: int) -> list[int]:
        x1 = max(min(int(left[0]), int(right[0])), 0)
        y1 = max(min(int(left[1]), int(right[1])), 0)
        x2 = min(max(int(left[0] + left[2]), int(right[0] + right[2])), page_width)
        y2 = min(max(int(left[1] + left[3]), int(right[1] + right[3])), page_height)
        return [int(x1), int(y1), int(max(x2 - x1, 1)), int(max(y2 - y1, 1))]

    def _combined_source(self, left: str, right: str) -> str:
        values = {left.strip(), right.strip()} - {""}
        if values == {"detector"}:
            return "detector"
        if "detector" in values and "ocr_cluster" in values:
            return "detector+ocr_cluster"
        if "detector+ocr_cluster" in values:
            return "detector+ocr_cluster"
        return values.pop() if values else "detector"

    def _axis_gap(self, start_a: int, end_a: int, start_b: int, end_b: int) -> float:
        if end_a < start_b:
            return float(start_b - end_a)
        if end_b < start_a:
            return float(start_a - end_b)
        return 0.0
