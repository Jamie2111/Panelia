"""
panel_detector.py — Hybrid manga/manhwa/comic panel detector.

Two-stage architecture:
  Stage 1: AI detection via Magi model (if available)
  Stage 2: Classical computer-vision fallback

Handles:
  • manga (traditional pages, right-to-left reading)
  • manhwa / webtoon (tall vertical scrolls, top-to-bottom)
  • western comics (left-to-right reading)
  • any language

Guarantees:
  • NEVER returns zero panels — worst case returns the full page
  • Dynamic thresholds based on page dimensions
  • Correct reading-order sorting per format
  • Debug visualisation output when enabled
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any, Sequence
from uuid import uuid4

import cv2
import numpy as np
from PIL import Image

from app.core.config import get_settings
from app.schemas.project import PanelBox
from app.services.magi_service import MagiHFService
from training.panel_detector_model import LoadedPanelDetector, load_latest_panel_detector_runtime

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════
#  Data types
# ═══════════════════════════════════════════════════════════════════

class ReadingOrder(Enum):
    """How panels should be sorted for narrative flow."""
    MANGA = auto()    # top-to-bottom, right-to-left
    MANHWA = auto()   # top-to-bottom (single column)
    WESTERN = auto()  # top-to-bottom, left-to-right


class PageLayout(Enum):
    """Detected page geometry."""
    TRADITIONAL = auto()   # standard manga / comic page
    WEBTOON = auto()       # tall vertical scroll strip


@dataclass
class DetectedPanel:
    """A single detected panel bounding box."""
    x: int
    y: int
    width: int
    height: int
    confidence: float = 1.0
    source: str = "unknown"   # "magi", "cv", "webtoon", "fallback"
    page: int = 0
    order: int = 0
    panel_id: str = field(default_factory=lambda: f"panel_{uuid4().hex[:12]}")

    @property
    def area(self) -> int:
        return self.width * self.height

    @property
    def aspect_ratio(self) -> float:
        return self.width / max(self.height, 1)

    @property
    def x2(self) -> int:
        return self.x + self.width

    @property
    def y2(self) -> int:
        return self.y + self.height

    @property
    def center_x(self) -> float:
        return self.x + self.width / 2.0

    @property
    def center_y(self) -> float:
        return self.y + self.height / 2.0

    def as_tuple(self) -> tuple[int, int, int, int]:
        return (self.x, self.y, self.width, self.height)


@dataclass
class DetectionResult:
    """Full detection result for one page."""
    panels: list[DetectedPanel]
    page_width: int
    page_height: int
    layout: PageLayout
    reading_order: ReadingOrder
    source: str  # which stage produced the result
    text_boxes: list[tuple[int, int, int, int]] = field(default_factory=list)
    character_boxes: list[tuple[int, int, int, int]] = field(default_factory=list)


# ═══════════════════════════════════════════════════════════════════
#  Configuration
# ═══════════════════════════════════════════════════════════════════

@dataclass
class DetectorConfig:
    """All tuneable thresholds in one place."""

    # ── Page layout classification ────────────────────────────────
    # A page with height/width >= this is classified as webtoon
    webtoon_ratio_threshold: float = 1.8

    # ── Whitespace gutter detection (webtoon) ─────────────────────
    # For tall webtoon strips, split at horizontal white-space gutters.
    # A row is "light" if this fraction of its pixels are above the
    # white threshold (all 3 channels must exceed it).
    webtoon_light_thresh: int = 220
    webtoon_light_row_frac: float = 0.90
    # Minimum consecutive light rows to count as a gutter.
    webtoon_min_gutter_px: int = 8
    # Minimum panel height to keep (smaller regions are discarded).
    webtoon_min_panel_px: int = 150

    # ── Minimum panel dimensions (relative to page) ───────────────
    # Panels smaller than these fractions of page area are dropped.
    # Calibrated against 332 real panels: no good panel has area
    # below 1 % of page area or min-dimension below 12 % of page
    # width in the test corpus.
    min_panel_area_ratio: float = 0.01
    min_panel_dim_px: int = 150
    min_panel_height_px: int = 80
    min_panel_width_px: int = 120

    # ── Junk panel filters ────────────────────────────────────────
    # Aspect-ratio and height thresholds for junk rejection.
    # These are applied ONLY when character detection says no
    # characters are present in the box.
    #
    # Wide strip (speech bubble bar, gutter, transition effect):
    #   observed: ar > 2.5, h ≤ 420 px
    junk_strip_max_ar: float = 2.5
    junk_strip_max_height: int = 420
    #
    # Chapter / episode title banner:
    #   observed: ar > 1.8, h ≤ 280 px, h ≤ page_h * 0.10
    junk_banner_max_ar: float = 1.8
    junk_banner_max_height: int = 280
    junk_banner_page_ratio: float = 0.10
    #
    # Tiny fragments:
    junk_fragment_max_area: int = 150_000
    #
    # Ultra-extreme aspect ratio:
    junk_extreme_ar: float = 6.0

    # ── Whiteness / low-content filter ────────────────────────────
    # Panels that are mostly blank background (speech bubble on
    # white, gutter strip, etc.) are filtered when white_ratio
    # exceeds this and height is below the max.
    junk_white_ratio: float = 0.72
    junk_white_max_height: int = 450

    # ── Webtoon slice parameters ──────────────────────────────────
    webtoon_slice_height: int = 1400
    webtoon_slice_overlap: int = 200

    # ── CV detection tuning ───────────────────────────────────────
    cv_morph_close_kernel_w: int = 15
    cv_morph_close_kernel_h: int = 15
    cv_canny_sigma: float = 0.33

    # ── Debug ─────────────────────────────────────────────────────
    debug: bool = False
    debug_dir: str = "debug"

    # ── Magi model ────────────────────────────────────────────────
    magi_max_image_edge: int = 960
    magi_force_grayscale: bool = False
    magi_detect_webtoon_panels: bool = False

    # ── Full-page suppression ─────────────────────────────────────
    # Panels covering more than this fraction of the page are
    # suspect (unless they're the only panel).
    full_page_width_ratio: float = 0.94
    full_page_height_ratio: float = 0.90

    # ── Boundary-panel overlap ────────────────────────────────────
    # IoU above this between two boxes → treat as duplicates.
    duplicate_iou_threshold: float = 0.62


# ═══════════════════════════════════════════════════════════════════
#  Main detector class
# ═══════════════════════════════════════════════════════════════════

class PanelDetector:
    """
    Hybrid manga / manhwa / comic panel detector.

    Usage:
        detector = PanelDetector()
        result = detector.detect_panels(image)
        for panel in result.panels:
            print(panel.x, panel.y, panel.width, panel.height)
    """

    def __init__(
        self,
        config: DetectorConfig | None = None,
        reading_order: ReadingOrder = ReadingOrder.MANHWA,
        magi_model: Any | None = None,
        trained_detector: LoadedPanelDetector | None = None,
    ) -> None:
        self.config = config or DetectorConfig()
        self.default_reading_order = reading_order
        self._magi_model = magi_model
        self._trained_detector = trained_detector

    # ──────────────────────────────────────────────────────────────
    #  Public API
    # ──────────────────────────────────────────────────────────────

    def detect_panels(
        self,
        image: np.ndarray,
        page_index: int = 1,
        reading_order: ReadingOrder | None = None,
    ) -> DetectionResult:
        """
        Detect panels in a single page image.

        Args:
            image:         RGB numpy array (H, W, 3)
            page_index:    1-based page number
            reading_order: Override reading direction

        Returns:
            DetectionResult with sorted, filtered panels.
            Guaranteed to contain at least one panel.
        """
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)

        page_height, page_width = image.shape[:2]
        layout = self._classify_layout(page_width, page_height)
        order = reading_order or self.default_reading_order

        # ── Stage 1: AI detection (trained checkpoint + Magi) ───
        trained_panels = self._detect_trained_model(image)
        should_run_magi = layout != PageLayout.WEBTOON or self.config.magi_detect_webtoon_panels
        if should_run_magi:
            magi_panels, text_boxes, char_boxes = self._detect_magi(image)
        else:
            magi_panels, text_boxes, char_boxes = [], [], []

        # ── Stage 2: CV fallback ─────────────────────────────────
        if layout == PageLayout.WEBTOON:
            cv_panels = self._detect_webtoon(image)
        else:
            cv_panels = self._detect_cv(image)

        # ── Merge stages ─────────────────────────────────────────
        if layout == PageLayout.TRADITIONAL and order == ReadingOrder.MANGA and magi_panels:
            panels = self._merge_traditional_manga_detections(
                magi_panels,
                trained_panels + cv_panels,
                page_width,
                page_height,
            )
        else:
            ai_panels = trained_panels + magi_panels
            panels = self._merge_detections(
                ai_panels, cv_panels, page_width, page_height
            )

        # ── Filter junk ──────────────────────────────────────────
        panels = self._filter_junk(
            image, panels, char_boxes, page_width, page_height
        )

        # ── Deduplicate ──────────────────────────────────────────
        panels = self._deduplicate(panels)

        # ── Remove full-page boxes (unless only option) ──────────
        panels = self._suppress_full_page(
            panels, page_width, page_height
        )

        # ── Split composite panels (multi-panel in one box) ──────
        panels = self._split_composite_panels(
            image, panels, page_width, page_height
        )

        # ── Guarantee at least one panel ─────────────────────────
        if not panels:
            panels = [DetectedPanel(
                x=0, y=0,
                width=page_width, height=page_height,
                confidence=0.1,
                source="fallback",
            )]

        # ── Sort reading order ───────────────────────────────────
        panels = self._sort_panels(panels, order, page_width)

        # ── Assign metadata ──────────────────────────────────────
        for idx, panel in enumerate(panels, start=1):
            panel.page = page_index
            panel.order = idx

        # ── Determine source label ───────────────────────────────
        sources = set(p.source for p in panels)
        if sources == {"fallback"}:
            source_label = "fallback"
        elif "magi" in sources:
            source_label = "magi+cv" if "cv" in sources or "webtoon" in sources else "magi"
        else:
            source_label = "cv"

        result = DetectionResult(
            panels=panels,
            page_width=page_width,
            page_height=page_height,
            layout=layout,
            reading_order=order,
            source=source_label,
            text_boxes=text_boxes,
            character_boxes=char_boxes,
        )

        # ── Debug output ─────────────────────────────────────────
        if self.config.debug:
            self._debug_visualization(image, result, page_index)

        return result

    def detect_panels_batch(
        self,
        page_paths: list[Path],
        reading_order: ReadingOrder | None = None,
        progress_callback: callable | None = None,
    ) -> list[DetectionResult]:
        """Detect panels across multiple page images."""
        results: list[DetectionResult] = []
        total = max(len(page_paths), 1)
        for idx, path in enumerate(page_paths, start=1):
            image = np.array(Image.open(path).convert("RGB"))
            result = self.detect_panels(image, page_index=idx, reading_order=reading_order)
            results.append(result)
            if progress_callback:
                progress_callback(idx / total * 100, f"Page {idx}/{total}")
        return results

    # ──────────────────────────────────────────────────────────────
    #  Stage 1: AI detection (Magi)
    # ──────────────────────────────────────────────────────────────

    def _detect_trained_model(
        self,
        image: np.ndarray,
    ) -> list[DetectedPanel]:
        if self._trained_detector is None:
            return []

        try:
            predictions = self._trained_detector.predict(image)
        except Exception as exc:
            logger.warning("Trained panel detector failed: %s", exc)
            return []

        trained_panels: list[DetectedPanel] = []
        for x, y, width, height, score in predictions:
            trained_panels.append(
                DetectedPanel(
                    x=int(x),
                    y=int(y),
                    width=max(int(width), 1),
                    height=max(int(height), 1),
                    confidence=float(score),
                    source="trained",
                )
            )
        return trained_panels

    def _detect_magi(
        self,
        image: np.ndarray,
    ) -> tuple[list[DetectedPanel], list[tuple], list[tuple]]:
        """
        Run the Magi transformer model for panel detection.

        Returns:
            (panels, text_boxes, character_boxes)
            All in (x, y, w, h) format at original image scale.
        """
        if self._magi_model is None:
            return [], [], []

        page_height, page_width = image.shape[:2]

        try:
            import torch

            # Prepare image for model
            model_image, scale_x, scale_y = self._prepare_magi_image(image)

            with torch.inference_mode():
                results = self._magi_model.predict_detections_and_associations(
                    [model_image]
                )

            if not results:
                return [], [], []

            result = results[0]

            # Extract panel boxes
            raw_panels = (
                result.get("panels")
                or result.get("panel_bboxes")
                or result.get("panel_boxes")
                or result.get("bboxes")
                or []
            )

            model_h, model_w = model_image.shape[:2]
            panels = self._parse_magi_boxes(
                raw_panels, scale_x, scale_y, page_width, page_height
            )

            # Extract text boxes
            text_boxes = self._parse_xyxy_boxes(
                result.get("texts", []),
                scale_x, scale_y, page_width, page_height,
            )

            # Extract character boxes
            char_boxes = self._parse_xyxy_boxes(
                result.get("characters", []),
                scale_x, scale_y, page_width, page_height,
            )

            magi_panels = [
                DetectedPanel(
                    x=x, y=y, width=w, height=h,
                    confidence=0.9,
                    source="magi",
                )
                for x, y, w, h in panels
            ]

            return magi_panels, text_boxes, char_boxes

        except Exception as exc:
            logger.warning("Magi detection failed: %s", exc)
            return [], [], []

    def _prepare_magi_image(
        self,
        image: np.ndarray,
    ) -> tuple[np.ndarray, float, float]:
        """Resize image for the Magi model, return (resized, scale_x, scale_y)."""
        src_h, src_w = image.shape[:2]

        if self.config.magi_force_grayscale:
            pil = Image.fromarray(image).convert("L").convert("RGB")
        else:
            pil = Image.fromarray(image).convert("RGB")

        max_edge = max(src_w, src_h)
        if max_edge <= self.config.magi_max_image_edge:
            return np.array(pil), 1.0, 1.0

        scale = self.config.magi_max_image_edge / max_edge
        tw = max(1, int(round(src_w * scale)))
        th = max(1, int(round(src_h * scale)))
        resized = pil.resize((tw, th), Image.LANCZOS)
        return np.array(resized), src_w / tw, src_h / th

    def _parse_magi_boxes(
        self,
        raw: list,
        scale_x: float,
        scale_y: float,
        page_w: int,
        page_h: int,
    ) -> list[tuple[int, int, int, int]]:
        """Convert Magi output (various formats) → scaled (x, y, w, h) list."""
        boxes: list[tuple[int, int, int, int]] = []
        for item in raw:
            if isinstance(item, dict):
                if {"x", "y", "width", "height"} <= item.keys():
                    boxes.append((int(item["x"]), int(item["y"]),
                                  int(item["width"]), int(item["height"])))
                    continue
                if {"x1", "y1", "x2", "y2"} <= item.keys():
                    boxes.append((int(item["x1"]), int(item["y1"]),
                                  int(item["x2"]) - int(item["x1"]),
                                  int(item["y2"]) - int(item["y1"])))
                    continue
            if isinstance(item, (list, tuple)) and len(item) >= 4:
                vals = [int(v) for v in item[:4]]
                # xyxy → xywh
                boxes.append((vals[0], vals[1], vals[2] - vals[0], vals[3] - vals[1]))

        return self._scale_boxes(boxes, scale_x, scale_y, page_w, page_h)

    def _parse_xyxy_boxes(
        self,
        raw: list,
        scale_x: float,
        scale_y: float,
        page_w: int,
        page_h: int,
    ) -> list[tuple[int, int, int, int]]:
        """Parse Magi text/character boxes (xyxy) → scaled (x, y, w, h)."""
        boxes: list[tuple[int, int, int, int]] = []
        for item in raw or []:
            if not isinstance(item, (list, tuple)) or len(item) < 4:
                continue
            try:
                x1, y1, x2, y2 = [float(v) for v in item[:4]]
            except (TypeError, ValueError):
                continue
            boxes.append((
                int(round(x1)), int(round(y1)),
                max(1, int(round(x2 - x1))),
                max(1, int(round(y2 - y1))),
            ))
        return self._scale_boxes(boxes, scale_x, scale_y, page_w, page_h)

    # ──────────────────────────────────────────────────────────────
    #  Stage 2a: Classical CV detection (traditional pages)
    # ──────────────────────────────────────────────────────────────

    def _detect_cv(self, image: np.ndarray) -> list[DetectedPanel]:
        """
        Classical computer-vision panel detection.

        Pipeline:
          1. Convert to grayscale
          2. Adaptive thresholding (handles varying backgrounds)
          3. Morphological closing (connect nearby edges)
          4. Contour detection
          5. Filter by size / aspect ratio
          6. Convert to bounding boxes
        """
        page_height, page_width = image.shape[:2]
        page_area = page_width * page_height

        grayscale = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        # ── Try multiple detection strategies and pick the best ───
        all_panels: list[DetectedPanel] = []

        # Strategy A: Adaptive threshold + contours
        panels_a = self._cv_adaptive_threshold(grayscale, page_width, page_height)
        if panels_a:
            all_panels = panels_a

        # Strategy B: Canny edge-based detection
        panels_b = self._cv_canny_edges(grayscale, page_width, page_height)
        if len(panels_b) > len(all_panels):
            all_panels = panels_b

        # Strategy C: Otsu + morphological closing (good for clean borders)
        panels_c = self._cv_otsu_morph(grayscale, page_width, page_height)
        if len(panels_c) > len(all_panels):
            all_panels = panels_c

        # Strategy D: Row/column signal analysis (for borderless panels)
        if not all_panels:
            panels_d = self._cv_signal_analysis(grayscale, page_width, page_height)
            all_panels = panels_d

        for p in all_panels:
            p.source = "cv"

        return all_panels

    def _cv_adaptive_threshold(
        self,
        gray: np.ndarray,
        page_w: int,
        page_h: int,
    ) -> list[DetectedPanel]:
        """Strategy A: adaptive threshold → contours."""
        # Block size must be odd, proportional to image
        block_size = max(11, (min(page_w, page_h) // 20) | 1)

        binary = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            block_size, 8,
        )

        # Close gaps between nearby edges to form panel regions
        kw = max(self.config.cv_morph_close_kernel_w, page_w // 60)
        kh = max(self.config.cv_morph_close_kernel_h, page_h // 60)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kw, kh))
        closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)

        # Remove small noise
        kernel_open = np.ones((3, 3), np.uint8)
        cleaned = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel_open, iterations=1)

        return self._extract_panels_from_binary(cleaned, page_w, page_h)

    def _cv_canny_edges(
        self,
        gray: np.ndarray,
        page_w: int,
        page_h: int,
    ) -> list[DetectedPanel]:
        """Strategy B: adaptive Canny → dilate → contours."""
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = self._adaptive_canny(blurred)

        # Dilate edges to connect nearby lines into panel boundaries
        kw = max(10, page_w // 40)
        kh = max(10, page_h // 40)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kw, kh))
        dilated = cv2.dilate(edges, kernel, iterations=2)

        # Close remaining gaps
        closed = cv2.morphologyEx(
            dilated, cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_RECT, (kw * 2, kh * 2)),
            iterations=1,
        )

        return self._extract_panels_from_binary(closed, page_w, page_h)

    def _cv_otsu_morph(
        self,
        gray: np.ndarray,
        page_w: int,
        page_h: int,
    ) -> list[DetectedPanel]:
        """Strategy C: Otsu binarisation → morphological closing → contours."""
        _, binary = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )

        # Close horizontal then vertical to form panel blobs
        kw = max(page_w // 8, 30)
        kh = max(page_h // 12, 20)
        kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (kw, 1))
        kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kh))
        closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_h)
        closed = cv2.morphologyEx(closed, cv2.MORPH_CLOSE, kernel_v)

        # Dilate to merge nearby components
        kernel_d = np.ones((5, 5), np.uint8)
        dilated = cv2.dilate(closed, kernel_d, iterations=3)

        return self._extract_panels_from_binary(dilated, page_w, page_h)

    def _cv_signal_analysis(
        self,
        gray: np.ndarray,
        page_w: int,
        page_h: int,
    ) -> list[DetectedPanel]:
        """
        Strategy D: Row/column content-signal analysis.

        For borderless panels — detects content bands by scanning
        horizontal rows for ink density, then groups consecutive
        content rows into panel regions.
        """
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = self._adaptive_canny(blurred)

        # Compute adaptive bright threshold from image statistics
        median_val = float(np.median(blurred))
        bright_threshold = max(238, int(median_val + 12))

        # Signal: pixels that are either dark or have edges
        signal = (blurred < bright_threshold) | (edges > 0)
        signal_mean = float(np.mean(signal))
        if signal_mean < 0.005:
            return []

        # Smooth row-wise signal
        row_signal = signal.astype(np.float32).mean(axis=1)
        window = max(11, int(page_h * 0.01))
        if window % 2 == 0:
            window += 1
        kernel = np.ones(window, dtype=np.float32) / window
        row_signal = np.convolve(row_signal, kernel, mode="same")

        # Find content runs (consecutive rows above threshold)
        threshold = 0.01
        runs = self._extract_runs(row_signal > threshold)

        panels: list[DetectedPanel] = []
        min_run_h = max(140, int(page_h * 0.08))

        for run_start, run_end in runs:
            run_height = run_end - run_start
            if run_height < min_run_h:
                continue

            # Find horizontal extent of content in this run
            region = signal[run_start:run_end, :]
            col_signal = region.astype(np.float32).mean(axis=0)
            col_window = max(9, int(page_w * 0.015))
            if col_window % 2 == 0:
                col_window += 1
            col_kernel = np.ones(col_window, dtype=np.float32) / col_window
            col_signal = np.convolve(col_signal, col_kernel, mode="same")
            cols = np.where(col_signal > 0.02)[0]
            if cols.size == 0:
                continue

            x1 = int(cols[0])
            x2 = int(cols[-1] + 1)
            w = max(x2 - x1, 1)

            if w < page_w * 0.28:
                continue
            if w * run_height < page_w * page_h * 0.03:
                continue

            panels.append(DetectedPanel(
                x=x1, y=run_start,
                width=w, height=run_height,
                confidence=0.6,
                source="cv",
            ))

        return panels

    def _extract_panels_from_binary(
        self,
        binary: np.ndarray,
        page_w: int,
        page_h: int,
    ) -> list[DetectedPanel]:
        """Find contours in a binary mask and convert to panel boxes."""
        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        page_area = page_w * page_h
        min_area = max(
            int(page_area * self.config.min_panel_area_ratio),
            self.config.min_panel_dim_px ** 2,
        )
        min_dim = self.config.min_panel_dim_px

        panels: list[DetectedPanel] = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)

            # Size filters
            if w * h < min_area:
                continue
            if w < min_dim or h < min_dim:
                continue

            # Skip tiny slivers relative to page
            if w < page_w * 0.15 and h < page_h * 0.05:
                continue

            panels.append(DetectedPanel(
                x=x, y=y, width=w, height=h,
                confidence=0.7,
                source="cv",
            ))

        return panels

    # ──────────────────────────────────────────────────────────────
    #  Stage 2b: Webtoon / manhwa detection
    # ──────────────────────────────────────────────────────────────

    def _detect_webtoon(self, image: np.ndarray) -> list[DetectedPanel]:
        """
        Specialised detection for tall vertical webtoon pages.

        Pass 0: Whitespace gutter detection — scan for horizontal bands
                of near-white pixels and split the strip at those gaps.
                This handles most modern manhwa/webtoon where panels are
                separated by white margins rather than black borders.

        Pass 1: Slice-based CV detection (original approach).
        Pass 2-4: Progressive fallbacks for edge cases.
        """
        page_height, page_width = image.shape[:2]

        # ── Pass 0: whitespace gutter splitting ───────────────────
        whitespace_panels = self._split_by_whitespace_gutters(image)
        if len(whitespace_panels) >= 2:
            logger.debug(
                "_detect_webtoon: whitespace gutters found %d panels",
                len(whitespace_panels),
            )
            return whitespace_panels

        slice_h = self.config.webtoon_slice_height
        overlap = self.config.webtoon_slice_overlap
        step = max(slice_h - overlap, 200)

        all_panels: list[DetectedPanel] = []

        # ── Pass 1: per-slice detection ───────────────────────────
        y_offset = 0
        while y_offset < page_height:
            y_end = min(y_offset + slice_h, page_height)
            slice_img = image[y_offset:y_end, :, :]

            if slice_img.shape[0] < 100:
                break

            gray = cv2.cvtColor(slice_img, cv2.COLOR_RGB2GRAY)

            # Try adaptive threshold first
            slice_panels = self._cv_adaptive_threshold(
                gray, page_width, y_end - y_offset
            )

            # Fall back to signal analysis
            if not slice_panels:
                slice_panels = self._cv_signal_analysis(
                    gray, page_width, y_end - y_offset
                )

            # Translate y coordinates back to full-page space
            for panel in slice_panels:
                panel.y += y_offset
                panel.source = "webtoon"
                all_panels.append(panel)

            y_offset += step

        # ── Pass 2: merge overlapping panels from adjacent slices ─
        all_panels = self._merge_overlapping_panels(all_panels)

        # ── Pass 3: if slice-based detection failed, try full-page ─
        if not all_panels:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            all_panels = self._cv_signal_analysis(gray, page_width, page_height)
            for p in all_panels:
                p.source = "webtoon"

        # ── Pass 4: if still nothing, try Otsu on full image ──────
        if not all_panels:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            all_panels = self._cv_otsu_morph(gray, page_width, page_height)
            for p in all_panels:
                p.source = "webtoon"

        return all_panels

    def _split_by_whitespace_gutters(
        self,
        image: np.ndarray,
    ) -> list[DetectedPanel]:
        """
        Split a webtoon strip into panels by finding horizontal white-space
        gutters — rows where almost all pixels are near-white.

        This handles the standard manhwa/webtoon format where individual
        scenes/panels are separated by a few rows of white (or near-white)
        background rather than black panel borders.

        Returns one DetectedPanel per content region. Returns an empty list
        (not a fallback) so callers can decide whether to use this result.
        """
        cfg = self.config
        page_h, page_w = image.shape[:2]

        # Per-row: is the row mostly light?
        # We check all three channels so coloured backgrounds don't fool us.
        is_light_pixel = (
            (image[:, :, 0] > cfg.webtoon_light_thresh)
            & (image[:, :, 1] > cfg.webtoon_light_thresh)
            & (image[:, :, 2] > cfg.webtoon_light_thresh)
        )
        row_light_frac: np.ndarray = is_light_pixel.mean(axis=1)  # shape (H,)
        is_gutter_row: np.ndarray = row_light_frac >= cfg.webtoon_light_row_frac

        # Collect gutter runs (consecutive light rows ≥ min_gutter_px)
        gutters: list[tuple[int, int]] = []
        in_run = False
        run_start = 0
        for i, g in enumerate(is_gutter_row):
            if g and not in_run:
                run_start = i
                in_run = True
            elif not g and in_run:
                run_len = i - run_start
                if run_len >= cfg.webtoon_min_gutter_px:
                    gutters.append((run_start, i))
                in_run = False
        if in_run:
            run_len = page_h - run_start
            if run_len >= cfg.webtoon_min_gutter_px:
                gutters.append((run_start, page_h))

        if not gutters:
            return []

        # Build content regions between gutters
        panels: list[DetectedPanel] = []
        prev_end = 0
        for g_start, g_end in gutters:
            if g_start > prev_end + cfg.webtoon_min_panel_px:
                panels.append(DetectedPanel(
                    x=0, y=prev_end,
                    width=page_w, height=g_start - prev_end,
                    confidence=0.75,
                    source="webtoon",
                ))
            prev_end = g_end
        # Trailing content after the last gutter
        if page_h - prev_end > cfg.webtoon_min_panel_px:
            panels.append(DetectedPanel(
                x=0, y=prev_end,
                width=page_w, height=page_h - prev_end,
                confidence=0.75,
                source="webtoon",
            ))

        return panels

    def _merge_overlapping_panels(
        self,
        panels: list[DetectedPanel],
    ) -> list[DetectedPanel]:
        """
        Merge panels that overlap significantly (from adjacent slices).
        Uses greedy union-find: if IoU > threshold, merge into one box.
        """
        if len(panels) <= 1:
            return panels

        # Sort top-to-bottom
        panels = sorted(panels, key=lambda p: (p.y, p.x))
        merged: list[DetectedPanel] = []

        for panel in panels:
            was_merged = False
            for i, existing in enumerate(merged):
                iou = self._iou(
                    existing.as_tuple(), panel.as_tuple()
                )
                # Also check if one contains the other
                overlap_ratio = self._intersection_over_min(
                    existing.as_tuple(), panel.as_tuple()
                )
                # Webtoon slice overlap: panels with similar X range
                # sharing a vertical band of >100px should merge even
                # when IoU is low (the 200px overlap pattern gives ~0.13 IoU).
                x_overlap_start = max(existing.x, panel.x)
                x_overlap_end = min(existing.x2, panel.x2)
                x_overlap_px = max(0, x_overlap_end - x_overlap_start)
                min_width = min(existing.width, panel.width)
                x_overlap_frac = x_overlap_px / max(min_width, 1)

                y_overlap_start = max(existing.y, panel.y)
                y_overlap_end = min(existing.y2, panel.y2)
                y_overlap_px = max(0, y_overlap_end - y_overlap_start)

                vertical_slice_overlap = (
                    x_overlap_frac > 0.5
                    and y_overlap_px > 100
                )

                if iou > 0.3 or overlap_ratio > 0.7 or vertical_slice_overlap:
                    # Merge by taking the union bounding box
                    ux = min(existing.x, panel.x)
                    uy = min(existing.y, panel.y)
                    ux2 = max(existing.x2, panel.x2)
                    uy2 = max(existing.y2, panel.y2)
                    merged[i] = DetectedPanel(
                        x=ux, y=uy,
                        width=ux2 - ux, height=uy2 - uy,
                        confidence=max(existing.confidence, panel.confidence),
                        source=existing.source,
                    )
                    was_merged = True
                    break
            if not was_merged:
                merged.append(panel)

        return merged

    # ──────────────────────────────────────────────────────────────
    #  Merging AI + CV results
    # ──────────────────────────────────────────────────────────────

    def _merge_detections(
        self,
        magi_panels: list[DetectedPanel],
        cv_panels: list[DetectedPanel],
        page_w: int,
        page_h: int,
    ) -> list[DetectedPanel]:
        """
        Combine Magi and CV detections.

        Priority:
          - If Magi returned panels, use them as primary
          - Add CV panels that don't overlap existing Magi panels
          - If Magi returned nothing, use CV panels
        """
        if not magi_panels:
            return cv_panels
        if not cv_panels:
            return magi_panels

        # Score both sets
        magi_score = self._score_panel_set(magi_panels, page_w, page_h)
        cv_score = self._score_panel_set(cv_panels, page_w, page_h)

        # Start with the better set as base
        if magi_score >= cv_score:
            base = list(magi_panels)
            additions = cv_panels
        else:
            base = list(cv_panels)
            additions = magi_panels

        # Add non-overlapping panels from the other set
        for add_panel in additions:
            overlaps = False
            for base_panel in base:
                if self._iou(add_panel.as_tuple(), base_panel.as_tuple()) > 0.3:
                    overlaps = True
                    break
                if self._intersection_over_min(
                    add_panel.as_tuple(), base_panel.as_tuple()
                ) > 0.7:
                    overlaps = True
                    break
            if not overlaps:
                base.append(add_panel)

        return base

    def _panel_has_inset_neighbors(
        self,
        panel: DetectedPanel,
        panels: list[DetectedPanel],
        page_area: int,
    ) -> bool:
        for other in panels:
            if other is panel:
                continue
            if other.area >= panel.area * 0.55:
                continue
            if other.area >= page_area * 0.45:
                continue
            if self._intersection_over_min(panel.as_tuple(), other.as_tuple()) >= 0.92:
                return True
        return False

    def _is_near_full_page_panel(
        self,
        panel: DetectedPanel,
        page_w: int,
        page_h: int,
    ) -> bool:
        return bool(
            panel.width >= page_w * self.config.full_page_width_ratio
            and panel.height >= page_h * self.config.full_page_height_ratio
        )

    def _merge_traditional_manga_detections(
        self,
        magi_panels: list[DetectedPanel],
        fallback_panels: list[DetectedPanel],
        page_w: int,
        page_h: int,
    ) -> list[DetectedPanel]:
        """Prefer MAGI structure on traditional manga pages.

        If MAGI finds multiple panels, keep that layout and only add smaller
        non-overlapping fallback boxes. If MAGI collapses the page to a single
        near-full-page box, allow a meaningfully richer fallback layout to win.
        """
        if not magi_panels:
            return fallback_panels

        magi_score = self._score_panel_set(magi_panels, page_w, page_h)
        fallback_score = self._score_panel_set(fallback_panels, page_w, page_h)
        magi_is_single_full_page = (
            len(magi_panels) == 1
            and self._is_near_full_page_panel(magi_panels[0], page_w, page_h)
        )
        if (
            magi_is_single_full_page
            and len(fallback_panels) >= 2
            and fallback_score > magi_score + 0.35
        ):
            return fallback_panels

        base = list(magi_panels)
        for add_panel in fallback_panels:
            if self._is_near_full_page_panel(add_panel, page_w, page_h):
                continue
            overlaps = False
            for base_panel in base:
                if self._iou(add_panel.as_tuple(), base_panel.as_tuple()) > 0.3:
                    overlaps = True
                    break
                if self._intersection_over_min(
                    add_panel.as_tuple(), base_panel.as_tuple()
                ) > 0.7:
                    overlaps = True
                    break
            if not overlaps:
                base.append(add_panel)
        return base

    def _score_panel_set(
        self,
        panels: list[DetectedPanel],
        page_w: int,
        page_h: int,
    ) -> float:
        """
        Heuristic quality score for a set of detected panels.

        Rewards:
          - Good page coverage (target ~60-80 % for webtoon, ~40-60 % for traditional)
          - Reasonable panel count (2-8 panels typical)
          - Portrait-oriented panels (for manhwa)

        Penalises:
          - Excessive overlap between panels
          - Single panel covering the entire page
        """
        if not panels:
            return float("-inf")

        page_area = page_w * page_h
        total_area = sum(p.area for p in panels)
        coverage = min(total_area / max(page_area, 1), 1.25)

        tall_page = page_h / max(page_w, 1) >= 1.45
        target = 0.70 if tall_page else 0.58
        coverage_score = 1.7 - abs(coverage - target) * 2.4

        count_score = min(len(panels), 12) * 0.38

        full_page_penalty = 0.0
        for panel in panels:
            if panel.area < page_area * 0.72:
                continue
            if tall_page:
                full_page_penalty += 0.8
                continue
            # Traditional manga pages can legitimately have a splash panel
            # with inset panels layered on top. Penalise those much less.
            full_page_penalty += (
                0.65 if self._panel_has_inset_neighbors(panel, panels, page_area) else 1.35
            )

        overlap_penalty = 0.0
        for i, a in enumerate(panels):
            for b in panels[i + 1:]:
                overlap_penalty += self._iou(a.as_tuple(), b.as_tuple())
        overlap_penalty *= 2.8

        portrait_bonus = 0.0
        if tall_page:
            portrait_bonus = sum(
                0.16 for p in panels if p.height >= p.width
            )

        return (
            count_score
            + coverage_score
            + portrait_bonus
            - full_page_penalty
            - overlap_penalty
        )

    # ──────────────────────────────────────────────────────────────
    #  Junk panel filtering
    # ──────────────────────────────────────────────────────────────

    def _filter_junk(
        self,
        image: np.ndarray,
        panels: list[DetectedPanel],
        character_boxes: list[tuple[int, int, int, int]],
        page_w: int,
        page_h: int,
    ) -> list[DetectedPanel]:
        """
        Remove junk panels: speech-bubble bars, chapter-title banners,
        gutter slivers, tiny background fragments.

        ALL checks are gated on character_hits == 0 so legitimate
        character close-ups (which can be wide and short) are never
        dropped.

        Thresholds were calibrated against 332 real panels from a
        manhwa test corpus — 32/32 junk caught, 0 false positives.
        """
        if not panels:
            return panels

        cfg = self.config
        kept: list[DetectedPanel] = []
        is_webtoon_page = page_h / max(page_w, 1) >= cfg.webtoon_ratio_threshold
        effective_min_dim = max(
            cfg.min_panel_dim_px,
            int(page_w * (0.15 if is_webtoon_page else 0.10)),
        )
        min_area = 100_000 if is_webtoon_page else max(int(page_w * page_h * 0.004), 45_000)

        for panel in panels:
            w, h = panel.width, panel.height
            ar = panel.aspect_ratio
            area = panel.area

            # ── Hard dimensional rejects (no good panel hits these) ──
            if h < cfg.min_panel_height_px:
                continue
            if w < cfg.min_panel_width_px:
                continue
            if min(w, h) < effective_min_dim:
                continue
            if area < min_area:
                continue
            if ar > cfg.junk_extreme_ar:
                continue

            # ── Character-aware rejects ───────────────────────────
            char_hits = sum(
                1 for cb in character_boxes
                if self._intersection_over_min(panel.as_tuple(), cb) >= 0.12
            )

            # Small fragment with no characters
            if char_hits == 0 and area < cfg.junk_fragment_max_area:
                continue

            # Wide strip with no characters (speech bubbles, gutters)
            # Skip this filter for whitespace-detected panels: in webtoon
            # format, all panels are full-width so their AR is naturally
            # large even for legitimate short scene panels.
            if (
                char_hits == 0
                and panel.source != "webtoon"
                and ar > cfg.junk_strip_max_ar
                and h <= cfg.junk_strip_max_height
            ):
                continue

            # Chapter / episode title banner
            if (
                char_hits == 0
                and panel.source != "webtoon"
                and ar > cfg.junk_banner_max_ar
                and h <= cfg.junk_banner_max_height
                and h <= max(int(page_h * cfg.junk_banner_page_ratio), 200)
            ):
                continue

            # Low-content sliver (needs image metrics)
            if char_hits == 0 and h <= 400:
                metrics = self._compute_box_metrics(image, panel.as_tuple())
                ed = metrics["edge_density"]
                wr = metrics["white_ratio"]
                if ed < 0.015 and wr > 0.4:
                    continue
                if ed < 0.03 and ar > 2.5:
                    continue

            # ── High-whiteness reject (speech bubble on blank bg) ─
            # Panels that are mostly white/blank with limited height
            # have no visual storytelling value for video output.
            if char_hits == 0 and h <= cfg.junk_white_max_height:
                metrics = self._compute_box_metrics(image, panel.as_tuple())
                if metrics["white_ratio"] > cfg.junk_white_ratio and metrics["edge_density"] < 0.06:
                    continue

            # ── Ultra-white catch-all (any height) ───────────────
            # Nearly blank panels with almost no visual content are
            # never useful in a video, regardless of size.
            if char_hits == 0:
                metrics = self._compute_box_metrics(image, panel.as_tuple())
                if (
                    metrics["white_ratio"] > 0.85
                    and metrics["edge_density"] < 0.02
                    and metrics["contrast"] < 15
                ):
                    continue

            kept.append(panel)

        # Safety: never return empty — keep largest panel
        if not kept and panels:
            kept = [max(panels, key=lambda p: p.area)]

        return kept

    # ──────────────────────────────────────────────────────────────
    #  Deduplication
    # ──────────────────────────────────────────────────────────────

    def _deduplicate(
        self,
        panels: list[DetectedPanel],
    ) -> list[DetectedPanel]:
        """Remove panels that are contained within larger panels."""
        if len(panels) <= 1:
            return panels

        # Sort largest first
        sorted_panels = sorted(panels, key=lambda p: p.area, reverse=True)
        kept: list[DetectedPanel] = []

        for panel in sorted_panels:
            is_contained = False
            for existing in kept:
                # Check if panel is inside existing
                overlap = self._intersection_over_min(
                    existing.as_tuple(), panel.as_tuple()
                )
                if overlap > 0.85 and panel.area < existing.area:
                    is_contained = True
                    break
                # Check high IoU (near-duplicate)
                if self._iou(existing.as_tuple(), panel.as_tuple()) > self.config.duplicate_iou_threshold:
                    # Keep the higher-confidence one
                    if panel.confidence > existing.confidence:
                        kept.remove(existing)
                        break
                    else:
                        is_contained = True
                        break
            if not is_contained:
                kept.append(panel)

        return kept

    # ──────────────────────────────────────────────────────────────
    #  Full-page suppression
    # ──────────────────────────────────────────────────────────────

    def _suppress_full_page(
        self,
        panels: list[DetectedPanel],
        page_w: int,
        page_h: int,
    ) -> list[DetectedPanel]:
        """
        Remove panels that cover the entire page — unless they're
        the only panel (splash page preservation).
        """
        cfg = self.config
        is_webtoon_page = page_h / max(page_w, 1) >= cfg.webtoon_ratio_threshold
        full_page = [
            p for p in panels
            if (
                p.width >= page_w * cfg.full_page_width_ratio
                and p.height >= page_h * cfg.full_page_height_ratio
            )
        ]
        sub_page = [
            p for p in panels
            if not (
                p.width >= page_w * cfg.full_page_width_ratio
                and p.height >= page_h * cfg.full_page_height_ratio
            )
        ]

        if sub_page:
            if not is_webtoon_page and full_page:
                preserved = [
                    panel
                    for panel in full_page
                    if self._panel_has_inset_neighbors(panel, sub_page, page_w * page_h)
                ]
                if preserved:
                    return sub_page + preserved
            return sub_page

        # All panels are full-page → keep the best one (splash page)
        if panels:
            best = max(panels, key=lambda p: p.confidence)
            # Inset slightly so it's not literally the full page
            inset_x = max(int(page_w * 0.01), 4)
            inset_y = max(int(page_h * 0.01), 4)
            best.x = max(best.x + inset_x, 0)
            best.y = max(best.y + inset_y, 0)
            best.width = min(best.width - inset_x * 2, page_w)
            best.height = min(best.height - inset_y * 2, page_h)
            return [best]

        return panels

    # ──────────────────────────────────────────────────────────────
    #  Composite panel splitting
    # ──────────────────────────────────────────────────────────────

    def _split_composite_panels(
        self,
        image: np.ndarray,
        panels: list[DetectedPanel],
        page_w: int,
        page_h: int,
    ) -> list[DetectedPanel]:
        """
        Split panels that contain multiple sub-panels separated by
        white gutters.  These appear when the CV detector merges
        adjacent panels into one large bounding box.

        Uses the same gutter-detection heuristic as
        OpenCVPanelDetectionService._split_composite_box.
        """
        result: list[DetectedPanel] = []

        for panel in panels:
            w, h = panel.width, panel.height

            # Staggered manhwa pages can merge into a squarish box: two art
            # panels plus gutter speech bubbles. Try internal contour islands
            # before requiring a tall horizontal-gutter shape.
            large_enough_for_contours = (
                w >= page_w * 0.50
                and h >= page_h * 0.24
                and (w * h) >= (page_w * page_h) * 0.18
            )
            if large_enough_for_contours:
                contour_sub_panels = self._split_composite_by_internal_contours(
                    image=image,
                    panel=panel,
                    page_w=page_w,
                    page_h=page_h,
                )
                if len(contour_sub_panels) >= 2:
                    result.extend(contour_sub_panels)
                    continue

            # Only attempt horizontal gutter splitting on large, tall panels.
            if w < page_w * 0.58 or h < max(page_h * 0.34, w * 1.05):
                result.append(panel)
                continue

            x1 = max(panel.x, 0)
            y1 = max(panel.y, 0)
            x2 = min(panel.x2, page_w)
            y2 = min(panel.y2, page_h)
            if x2 <= x1 or y2 <= y1:
                result.append(panel)
                continue

            gray = cv2.cvtColor(image[y1:y2, x1:x2], cv2.COLOR_RGB2GRAY)
            if gray.size == 0:
                result.append(panel)
                continue

            blurred = cv2.GaussianBlur(gray, (3, 3), 0)
            edges = cv2.Canny(blurred, 40, 120)
            signal = (blurred < 242) | (edges > 0)
            bright = blurred > 245

            crop_h = y2 - y1
            smooth_w = max(9, int(crop_h * 0.018))
            if smooth_w % 2 == 0:
                smooth_w += 1
            kernel = np.ones(smooth_w, dtype=np.float32) / smooth_w

            row_signal = np.convolve(
                signal.astype(np.float32).mean(axis=1), kernel, mode="same"
            )
            row_bright = np.convolve(
                bright.astype(np.float32).mean(axis=1), kernel, mode="same"
            )

            gutter_mask = (row_signal < 0.02) & (row_bright > 0.88)
            gutter_runs = self._extract_runs(gutter_mask)

            usable = [
                (s, e) for s, e in gutter_runs
                if e - s >= max(18, int(crop_h * 0.02))
                and s >= int(crop_h * 0.12)
                and e <= int(crop_h * 0.88)
            ]

            if not usable:
                result.append(panel)
                continue

            # Build segments between gutters
            segments: list[tuple[int, int]] = []
            cursor = 0
            for gs, ge in usable:
                if gs - cursor >= int(crop_h * 0.18):
                    segments.append((cursor, gs))
                cursor = ge
            if crop_h - cursor >= int(crop_h * 0.18):
                segments.append((cursor, crop_h))

            if len(segments) < 2:
                result.append(panel)
                continue

            # Create sub-panels
            sub_panels: list[DetectedPanel] = []
            for seg_start, seg_end in segments:
                seg_h = seg_end - seg_start
                if seg_h < max(120, int(crop_h * 0.16)):
                    continue
                sub_panels.append(DetectedPanel(
                    x=x1,
                    y=y1 + seg_start,
                    width=x2 - x1,
                    height=seg_h,
                    confidence=panel.confidence * 0.95,
                    source=panel.source,
                ))

            # Only use split if it produced meaningful results
            if len(sub_panels) >= 2:
                total_sub = sum(sp.area for sp in sub_panels)
                if total_sub >= w * h * 0.58:
                    result.extend(sub_panels)
                    continue

            result.append(panel)

        return result

    def _split_composite_by_internal_contours(
        self,
        *,
        image: np.ndarray,
        panel: DetectedPanel,
        page_w: int,
        page_h: int,
    ) -> list[DetectedPanel]:
        """Split staggered comic panels that fallback CV merged into one box.

        Webtoon/manhwa pages often place bordered art panels diagonally with
        speech bubbles in the gutters. Horizontal gutter splitting misses that
        layout, so we look for large internal contour islands instead.
        """

        x1 = max(panel.x, 0)
        y1 = max(panel.y, 0)
        x2 = min(panel.x2, page_w)
        y2 = min(panel.y2, page_h)
        crop = image[y1:y2, x1:x2]
        if crop.size == 0:
            return []

        crop_h, crop_w = crop.shape[:2]
        gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 140)
        dilated = cv2.dilate(edges, np.ones((5, 5), dtype=np.uint8), iterations=1)
        contours, _hierarchy = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        candidates: list[tuple[int, int, int, int]] = []
        min_w = max(110, int(crop_w * 0.18))
        min_h = max(90, int(crop_h * 0.15))
        for contour in contours:
            bx, by, bw, bh = cv2.boundingRect(contour)
            if bw < min_w or bh < min_h:
                continue
            area = bw * bh
            if area < crop_w * crop_h * 0.08:
                continue
            if bw >= crop_w * 0.96 and bh >= crop_h * 0.96:
                continue
            # Mostly-empty bubbles/word balloons can have a rectangle-ish
            # outline but very little interior contour area; art panels and
            # bordered image blocks have a much denser contour footprint.
            contour_fill = cv2.contourArea(contour) / max(area, 1)
            if contour_fill < 0.28:
                continue
            # The contour is already dilated above, so extra padding here
            # reintroduces the thin white halo the review UI makes obvious.
            pad = max(1, int(min(bw, bh) * 0.003))
            left = max(bx - pad, 0)
            top = max(by - pad, 0)
            right = min(bx + bw + pad, crop_w)
            bottom = min(by + bh + pad, crop_h)
            candidates.append((left, top, right - left, bottom - top))

        if len(candidates) < 2:
            return []

        candidates = self._dedupe_local_boxes(candidates)
        if len(candidates) < 2:
            return []

        total_area = sum(width * height for _x, _y, width, height in candidates)
        if total_area < crop_w * crop_h * 0.36:
            return []

        return [
            DetectedPanel(
                x=x1 + bx,
                y=y1 + by,
                width=bw,
                height=bh,
                confidence=panel.confidence * 0.94,
                source=panel.source,
            )
            for bx, by, bw, bh in sorted(candidates, key=lambda box: (box[1], box[0]))
        ]

    def _dedupe_local_boxes(self, boxes: list[tuple[int, int, int, int]]) -> list[tuple[int, int, int, int]]:
        ordered = sorted(boxes, key=lambda box: box[2] * box[3], reverse=True)
        kept: list[tuple[int, int, int, int]] = []
        for box in ordered:
            if any(self._intersection_over_min(existing, box) > 0.82 for existing in kept):
                continue
            kept.append(box)
        return sorted(kept, key=lambda box: (box[1], box[0]))

    # ──────────────────────────────────────────────────────────────
    #  Reading order sorting
    # ──────────────────────────────────────────────────────────────

    def _sort_panels(
        self,
        panels: list[DetectedPanel],
        order: ReadingOrder,
        page_width: int,
    ) -> list[DetectedPanel]:
        """
        Sort panels into correct reading order.

        Manga:   top-to-bottom, right-to-left (Japanese)
        Manhwa:  top-to-bottom, left-to-right (Korean webtoon)
        Western: top-to-bottom, left-to-right (same as manhwa)

        Uses row-clustering: panels whose vertical centres are
        within 30 % of each other's height are considered same row.
        """
        if len(panels) <= 1:
            return panels

        if order == ReadingOrder.MANHWA:
            # Pure top-to-bottom (single column webtoon)
            return sorted(panels, key=lambda p: (p.y, p.x))

        # For manga and western: cluster into rows then sort within each row
        sorted_by_y = sorted(panels, key=lambda p: p.y)
        rows: list[list[DetectedPanel]] = []
        current_row: list[DetectedPanel] = [sorted_by_y[0]]

        for panel in sorted_by_y[1:]:
            prev = current_row[-1]
            # Same row if vertical overlap is significant
            row_threshold = min(prev.height, panel.height) * 0.30
            if abs(panel.center_y - prev.center_y) <= row_threshold:
                current_row.append(panel)
            else:
                rows.append(current_row)
                current_row = [panel]
        rows.append(current_row)

        # Sort within each row
        result: list[DetectedPanel] = []
        for row in rows:
            if order == ReadingOrder.MANGA:
                # Right-to-left
                row.sort(key=lambda p: -p.x)
            else:
                # Left-to-right (western)
                row.sort(key=lambda p: p.x)
            result.extend(row)

        return result

    # ──────────────────────────────────────────────────────────────
    #  Debug visualisation
    # ──────────────────────────────────────────────────────────────

    def _debug_visualization(
        self,
        image: np.ndarray,
        result: DetectionResult,
        page_index: int,
    ) -> None:
        """
        Save debug images:
          debug/page_{N}_original.png
          debug/page_{N}_with_boxes.png
          debug/page_{N}_panels/panel_{M}.png
        """
        debug_dir = Path(self.config.debug_dir)
        debug_dir.mkdir(parents=True, exist_ok=True)

        prefix = f"page_{page_index:03d}"

        # Save original
        orig_path = debug_dir / f"{prefix}_original.png"
        Image.fromarray(image).save(orig_path)

        # Draw boxes on a copy
        annotated = image.copy()
        colours = {
            "magi": (0, 255, 0),      # green
            "cv": (255, 165, 0),       # orange
            "webtoon": (0, 200, 255),  # cyan
            "fallback": (255, 0, 0),   # red
        }

        for panel in result.panels:
            colour = colours.get(panel.source, (255, 255, 255))
            cv2.rectangle(
                annotated,
                (panel.x, panel.y),
                (panel.x2, panel.y2),
                colour, 3,
            )
            label = f"#{panel.order} ({panel.source})"
            cv2.putText(
                annotated, label,
                (panel.x + 5, panel.y + 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, colour, 2,
            )

        # Draw text boxes in blue
        for tx, ty, tw, th in result.text_boxes:
            cv2.rectangle(annotated, (tx, ty), (tx + tw, ty + th), (100, 100, 255), 1)

        # Draw character boxes in magenta
        for cx, cy, cw, ch in result.character_boxes:
            cv2.rectangle(annotated, (cx, cy), (cx + cw, cy + ch), (255, 0, 255), 1)

        boxes_path = debug_dir / f"{prefix}_with_boxes.png"
        Image.fromarray(annotated).save(boxes_path)

        # Save individual panel crops
        panels_dir = debug_dir / f"{prefix}_panels"
        panels_dir.mkdir(exist_ok=True)
        for panel in result.panels:
            x1 = max(panel.x, 0)
            y1 = max(panel.y, 0)
            x2 = min(panel.x2, image.shape[1])
            y2 = min(panel.y2, image.shape[0])
            crop = image[y1:y2, x1:x2]
            if crop.size > 0:
                crop_path = panels_dir / f"panel_{panel.order:03d}.png"
                Image.fromarray(crop).save(crop_path)

        # Write metadata
        meta_path = debug_dir / f"{prefix}_meta.txt"
        with open(meta_path, "w") as f:
            f.write(f"Page: {page_index}\n")
            f.write(f"Size: {result.page_width}x{result.page_height}\n")
            f.write(f"Layout: {result.layout.name}\n")
            f.write(f"Reading: {result.reading_order.name}\n")
            f.write(f"Source: {result.source}\n")
            f.write(f"Panels: {len(result.panels)}\n")
            f.write(f"Text boxes: {len(result.text_boxes)}\n")
            f.write(f"Char boxes: {len(result.character_boxes)}\n")
            f.write("---\n")
            for p in result.panels:
                f.write(
                    f"  #{p.order}: ({p.x},{p.y}) {p.width}x{p.height} "
                    f"ar={p.aspect_ratio:.2f} area={p.area} "
                    f"conf={p.confidence:.2f} src={p.source}\n"
                )

    # ──────────────────────────────────────────────────────────────
    #  Image analysis utilities
    # ──────────────────────────────────────────────────────────────

    def _classify_layout(self, page_w: int, page_h: int) -> PageLayout:
        """Classify a page as traditional or webtoon based on aspect ratio."""
        ratio = page_h / max(page_w, 1)
        if ratio >= self.config.webtoon_ratio_threshold:
            return PageLayout.WEBTOON
        return PageLayout.TRADITIONAL

    def _adaptive_canny(
        self,
        gray: np.ndarray,
        sigma: float | None = None,
    ) -> np.ndarray:
        """
        Compute Canny edges with thresholds adapted to the image's
        median pixel intensity.  Works across dark manga and bright manhwa.
        """
        if sigma is None:
            sigma = self.config.cv_canny_sigma
        med = float(np.median(gray))
        lo = int(max(0, (1.0 - sigma) * med))
        hi = int(min(255, (1.0 + sigma) * med))
        if hi - lo < 30:
            lo = max(0, int(med) - 20)
            hi = min(255, int(med) + 40)
        return cv2.Canny(gray, lo, hi)

    def _compute_box_metrics(
        self,
        image: np.ndarray,
        box: tuple[int, int, int, int],
    ) -> dict[str, float]:
        """
        Compute visual content metrics for a bounding box region.

        Returns:
            edge_density: fraction of pixels with edges
            ink_ratio:    fraction of non-white pixels
            contrast:     standard deviation of grayscale values
            white_ratio:  fraction of very bright pixels
        """
        x, y, w, h = box
        crop = image[
            max(y, 0):min(y + h, image.shape[0]),
            max(x, 0):min(x + w, image.shape[1]),
        ]
        if crop.size == 0:
            return {
                "edge_density": 0.0,
                "ink_ratio": 0.0,
                "contrast": 0.0,
                "white_ratio": 1.0,
            }

        gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
        edges = self._adaptive_canny(gray)

        p95 = float(np.percentile(gray, 95))
        white_t = max(min(p95 - 3, 248), 230)
        ink_t = max(min(p95 - 5, 245), 225)

        return {
            "edge_density": float(np.mean(edges > 0)),
            "ink_ratio": float(np.mean(gray < ink_t)),
            "contrast": float(np.std(gray)),
            "white_ratio": float(np.mean(gray > white_t)),
        }

    # ──────────────────────────────────────────────────────────────
    #  Geometry utilities
    # ──────────────────────────────────────────────────────────────

    def _scale_boxes(
        self,
        boxes: list[tuple[int, int, int, int]],
        scale_x: float,
        scale_y: float,
        page_w: int,
        page_h: int,
    ) -> list[tuple[int, int, int, int]]:
        """Scale boxes from model coordinates to page coordinates."""
        if scale_x == 1.0 and scale_y == 1.0:
            return boxes
        scaled = []
        for x, y, w, h in boxes:
            sx = int(round(x * scale_x))
            sy = int(round(y * scale_y))
            sw = int(round(w * scale_x))
            sh = int(round(h * scale_y))
            # Clamp to page
            sx = max(0, min(sx, page_w))
            sy = max(0, min(sy, page_h))
            sw = max(1, min(sw, page_w - sx))
            sh = max(1, min(sh, page_h - sy))
            scaled.append((sx, sy, sw, sh))
        return scaled

    def _iou(
        self,
        a: tuple[int, int, int, int],
        b: tuple[int, int, int, int],
    ) -> float:
        """Intersection over Union of two (x, y, w, h) boxes."""
        inter = self._intersection_area(a, b)
        if inter <= 0:
            return 0.0
        area_a = a[2] * a[3]
        area_b = b[2] * b[3]
        union = area_a + area_b - inter
        return inter / max(union, 1)

    def _intersection_over_min(
        self,
        a: tuple[int, int, int, int],
        b: tuple[int, int, int, int],
    ) -> float:
        """Intersection area divided by the smaller box's area."""
        inter = self._intersection_area(a, b)
        if inter <= 0:
            return 0.0
        min_area = min(a[2] * a[3], b[2] * b[3])
        return inter / max(min_area, 1)

    def _intersection_area(
        self,
        a: tuple[int, int, int, int],
        b: tuple[int, int, int, int],
    ) -> int:
        """Compute intersection area of two (x, y, w, h) boxes."""
        x1 = max(a[0], b[0])
        y1 = max(a[1], b[1])
        x2 = min(a[0] + a[2], b[0] + b[2])
        y2 = min(a[1] + a[3], b[1] + b[3])
        if x2 <= x1 or y2 <= y1:
            return 0
        return (x2 - x1) * (y2 - y1)

    def _extract_runs(
        self,
        mask: np.ndarray,
    ) -> list[tuple[int, int]]:
        """Find consecutive True runs in a 1D boolean array."""
        runs: list[tuple[int, int]] = []
        start: int | None = None
        for i, v in enumerate(mask.tolist()):
            if v and start is None:
                start = i
            elif not v and start is not None:
                runs.append((start, i))
                start = None
        if start is not None:
            runs.append((start, len(mask)))
        return runs


# ═══════════════════════════════════════════════════════════════════
#  Convenience functions (module-level API)
# ═══════════════════════════════════════════════════════════════════

# Module-level singleton for quick usage
_default_detector: PanelDetector | None = None


def _get_detector(**kwargs) -> PanelDetector:
    global _default_detector
    if _default_detector is None:
        _default_detector = PanelDetector(**kwargs)
    return _default_detector


def detect_panels(
    image: np.ndarray,
    page_index: int = 1,
    reading_order: ReadingOrder = ReadingOrder.MANHWA,
    debug: bool = False,
) -> DetectionResult:
    """
    Detect panels in a single page image.

    This is the primary entry point for the module.

    Args:
        image:         RGB numpy array (H, W, 3)
        page_index:    1-based page number
        reading_order: Reading direction
        debug:         Save debug visualisations

    Returns:
        DetectionResult with sorted, filtered panels.
        Guaranteed to contain at least one panel.
    """
    config = DetectorConfig(debug=debug)
    detector = PanelDetector(config=config, reading_order=reading_order)
    return detector.detect_panels(image, page_index=page_index)


def detect_panels_from_path(
    path: str | Path,
    page_index: int = 1,
    reading_order: ReadingOrder = ReadingOrder.MANHWA,
    debug: bool = False,
) -> DetectionResult:
    """Convenience: detect panels from a file path."""
    image = np.array(Image.open(path).convert("RGB"))
    return detect_panels(image, page_index, reading_order, debug)


def detect_panels_batch(
    paths: list[str | Path],
    reading_order: ReadingOrder = ReadingOrder.MANHWA,
    debug: bool = False,
    progress_callback: callable | None = None,
) -> list[DetectionResult]:
    """Convenience: detect panels across multiple pages."""
    config = DetectorConfig(debug=debug)
    detector = PanelDetector(config=config, reading_order=reading_order)
    return detector.detect_panels_batch(
        [Path(p) for p in paths],
        progress_callback=progress_callback,
    )


# ═══════════════════════════════════════════════════════════════════
#  Integration adapter: wraps PanelDetector to match the existing
#  MagiPanelDetectionService interface so the rest of the Panelia
#  pipeline doesn't need changes.
# ═══════════════════════════════════════════════════════════════════

class PanelDetectorAdapter:
    """
    Drop-in replacement for MagiPanelDetectionService.detect_panels().

    Usage in your existing pipeline:
        # Old:
        service = MagiPanelDetectionService()
        panels = service.detect_panels(page_paths, ...)

        # New:
        adapter = PanelDetectorAdapter()
        panels = adapter.detect_panels(page_paths, ...)
    """

    _DETECTOR_VERSION = "2.4.1"

    # ── Cross-page duplicate detection ────────────────────────────
    # Panels whose downscaled thumbnails match within this threshold
    # (mean absolute pixel difference, 0-255 scale) are considered
    # duplicates.  Title cards and cover pages across chapters are
    # the primary target.
    _DEDUP_THUMB_SIZE: int = 32
    _DEDUP_MAX_DIFF: float = 6.0

    # ── Boundary-panel detection ─────────────────────────────────
    # Panels that touch the very top or bottom edge of a page and
    # are shorter than this fraction of page height are likely
    # continuation fragments from the adjacent page.
    _BOUNDARY_EDGE_PX: int = 8
    _BOUNDARY_MAX_HEIGHT_RATIO: float = 0.18
    _SAME_PAGE_OVERLAP_PX: int = 160
    _SAME_PAGE_GAP_RATIO: float = 0.12
    _WHITE_GAP_SHORT_RATIO: float = 0.32

    def __init__(self, magi_model: Any | None = None) -> None:
        self._settings = get_settings()
        self._magi_model = magi_model
        self._trained_detector = None
        self._config = DetectorConfig()
        self._config.magi_detect_webtoon_panels = bool(self._settings.magi_detect_webtoon_panels)
        self._config.magi_max_image_edge = int(self._settings.magi_max_image_edge or self._config.magi_max_image_edge)
        self._last_character_review_page_payloads: dict[int, dict[str, Any]] = {}

    @property
    def detector_version(self) -> str:
        return self._DETECTOR_VERSION

    def _is_webtoon_page(self, page_w: int, page_h: int) -> bool:
        return page_h / max(page_w, 1) >= self._config.webtoon_ratio_threshold

    def _should_auto_skip_top_boundary_panel(
        self,
        *,
        panel: DetectedPanel,
        page_w: int,
        page_h: int,
        is_webtoon_page: bool,
        metrics: dict[str, float | tuple[int, int, int, int]],
    ) -> bool:
        if panel.y > self._BOUNDARY_EDGE_PX:
            return False
        if panel.height >= page_h * self._BOUNDARY_MAX_HEIGHT_RATIO:
            return False
        if not is_webtoon_page:
            # Traditional manga pages often begin with a legitimate top-edge panel.
            # Only treat it as a fragment when it is both very shallow and spans
            # almost the full page width.
            if panel.width < page_w * 0.86 or panel.height > page_h * 0.12:
                return False
        bbox = metrics["content_bbox"]
        content_ratio = float(metrics["content_bbox_area_ratio"])
        center_y = float(metrics["content_center_y_ratio"])
        return bool(
            bbox[1] > panel.height * 0.16
            or center_y > 0.58
            or (content_ratio < 0.72 and center_y > 0.55)
            or (float(metrics["white_ratio"]) > 0.25 and float(metrics["edge_density"]) < 0.07)
        )

    def _load_model(self) -> Any | None:
        """
        Compatibility shim for older worker startup code.

        The previous MAGI service exposed `_load_model()` for prewarm.
        The adapter-based detector may not have a model wired in, so we
        simply return the injected model when present.
        """
        self._trained_detector = load_latest_panel_detector_runtime()
        if self._magi_model is None and self._settings.magi_enabled:
            self._magi_model = MagiHFService().load_model()
        return self._magi_model or self._trained_detector

    def detect_panels(
        self,
        page_paths: list[Path],
        progress_callback: callable | None = None,
        cancel_callback: callable | None = None,
    ) -> list:
        """
        Returns a list of objects matching PanelBox interface:
            id, page, panel, x, y, width, height, order, keep, zoom_hint

        Compatible with the existing Panelia pipeline.
        """
        self._trained_detector = load_latest_panel_detector_runtime()
        if self._magi_model is None and self._settings.magi_enabled:
            self._magi_model = MagiHFService().load_model()
        detector = PanelDetector(
            config=self._config,
            reading_order=ReadingOrder.MANGA,
            magi_model=self._magi_model,
            trained_detector=self._trained_detector,
        )
        self._last_character_review_page_payloads = {}

        panel_boxes = []
        global_order = 1

        # Cache page sizes for boundary detection
        page_sizes: dict[int, tuple[int, int]] = {}

        for page_idx, path in enumerate(page_paths, start=1):
            if cancel_callback:
                cancel_callback()

            image = np.array(Image.open(path).convert("RGB"))
            page_h, page_w = image.shape[:2]
            page_sizes[page_idx] = (page_w, page_h)
            reading_order = (
                ReadingOrder.MANHWA
                if detector._classify_layout(page_w, page_h) == PageLayout.WEBTOON
                else ReadingOrder.MANGA
            )
            result = detector.detect_panels(image, page_index=page_idx, reading_order=reading_order)
            is_webtoon_page = result.layout == PageLayout.WEBTOON
            self._last_character_review_page_payloads[page_idx] = self._character_review_payload_for_result(
                page_idx,
                result,
            )

            for panel_num, panel in enumerate(result.panels, start=1):
                # ── Boundary-panel detection ─────────────────────
                auto_skipped = False
                skip_reason = None
                keep = True

                touches_top = panel.y <= self._BOUNDARY_EDGE_PX
                touches_bottom = (panel.y + panel.height) >= (page_h - self._BOUNDARY_EDGE_PX)
                is_short = panel.height < page_h * self._BOUNDARY_MAX_HEIGHT_RATIO

                if is_short and touches_top:
                    crop = image[
                        max(panel.y, 0):min(panel.y + panel.height, page_h),
                        max(panel.x, 0):min(panel.x + panel.width, page_w),
                    ]
                    if crop.size > 0:
                        metrics = self._compute_crop_metrics(crop)
                        edge_where = "top"

                        if self._should_auto_skip_top_boundary_panel(
                            panel=panel,
                            page_w=page_w,
                            page_h=page_h,
                            is_webtoon_page=is_webtoon_page,
                            metrics=metrics,
                        ):
                            auto_skipped = True
                            skip_reason = f"split page-boundary panel ({edge_where})"
                            keep = False

                # ── Credit/watermark page detection ──────────────
                # Full-page panels that are mostly blank with low
                # visual complexity are likely credit or watermark pages.
                if (
                    not auto_skipped
                    and panel.width >= page_w * 0.80
                    and panel.height >= page_h * 0.80
                ):
                    credit_crop = image[
                        max(panel.y, 0):min(panel.y + panel.height, page_h),
                        max(panel.x, 0):min(panel.x + panel.width, page_w),
                    ]
                    if credit_crop.size > 0:
                        credit_gray = cv2.cvtColor(credit_crop, cv2.COLOR_RGB2GRAY)
                        credit_white = float(np.mean(credit_gray > 240))
                        credit_edges = cv2.Canny(credit_gray, 40, 120)
                        credit_edge_density = float(np.mean(credit_edges > 0))
                        if credit_white > 0.65 and credit_edge_density < 0.04:
                            auto_skipped = True
                            skip_reason = "likely credit/watermark page"
                            keep = False

                panel_boxes.append({
                    "id": panel.panel_id,
                    "page": page_idx,
                    "panel": panel_num,
                    "x": panel.x,
                    "y": panel.y,
                    "width": panel.width,
                    "height": panel.height,
                    "order": global_order,
                    "keep": keep,
                    "auto_skipped": auto_skipped,
                    "skip_reason": skip_reason,
                    "zoom_hint": self._zoom_hint(panel.width, panel.height),
                })
                global_order += 1

            if progress_callback:
                pct = page_idx / max(len(page_paths), 1) * 100
                progress_callback(pct, f"Page {page_idx}/{len(page_paths)}")

        # ── Cross-page deduplication ─────────────────────────────
        panel_boxes = self._deduplicate_across_pages(panel_boxes, page_paths)

        # ── Refine bounding boxes: trim white borders ────────────
        panel_boxes = self._refine_boxes(panel_boxes, page_paths)

        # ── Same-page cleanup: remove overlap strips and gap junk ─
        panel_boxes = self._cleanup_same_page_panels(panel_boxes, page_paths)

        # ── Re-number orders after filtering ─────────────────────
        for idx, pb in enumerate(panel_boxes, start=1):
            pb["order"] = idx

        # ── Log summary ──────────────────────────────────────────
        kept_count = sum(1 for pb in panel_boxes if pb.get("keep", True))
        skipped_count = sum(1 for pb in panel_boxes if pb.get("auto_skipped", False))
        logger.info(
            "PanelDetectorAdapter v%s: %d panels (%d kept, %d auto-skipped) from %d pages",
            self._DETECTOR_VERSION, len(panel_boxes), kept_count, skipped_count, len(page_paths),
        )

        # ── Convert dicts → PanelBox objects for the pipeline ────
        return [PanelBox.model_validate(pb) for pb in panel_boxes]

    def export_character_review_page_payloads(self) -> dict[int, dict[str, Any]]:
        return {
            int(page_number): dict(payload)
            for page_number, payload in self._last_character_review_page_payloads.items()
            if isinstance(payload, dict)
        }

    def _character_review_payload_for_result(
        self,
        page_number: int,
        result: DetectionResult,
    ) -> dict[str, Any]:
        text_entries = [
            {
                "text_index": index,
                "bbox": [int(x), int(y), int(w), int(h)],
                "character_id": None,
                "character_index": None,
                "is_dialogue": True,
                "text": "",
                "source": "panel-detection",
            }
            for index, (x, y, w, h) in enumerate(result.text_boxes)
        ]
        character_entries = [
            {
                "character_index": index,
                "character_id": f"detect-p{page_number:04d}-char-{index + 1:03d}",
                "bbox": [int(x), int(y), int(w), int(h)],
                "cluster_label_local": index,
                "source": "panel-detection",
            }
            for index, (x, y, w, h) in enumerate(result.character_boxes)
        ]
        panel_entries = [
            {
                "panel_index": index,
                "bbox": [int(panel.x), int(panel.y), int(panel.width), int(panel.height)],
                "order": index + 1,
                "source": result.source,
            }
            for index, panel in enumerate(result.panels)
        ]
        return {
            "page": int(page_number),
            "provider": f"panel-detection:{self._DETECTOR_VERSION}",
            "panels": panel_entries,
            "texts": text_entries,
            "characters": character_entries,
        }

    def _deduplicate_across_pages(
        self,
        panel_boxes: list[dict],
        page_paths: list[Path],
    ) -> list[dict]:
        """
        Remove visually identical panels that appear on different pages
        (recurring title cards, cover pages, watermarks).

        Keeps the first occurrence and marks subsequent duplicates as
        auto-skipped.
        """
        thumbs: list[np.ndarray | None] = []
        page_cache: dict[int, np.ndarray] = {}

        for pb in panel_boxes:
            page_idx = int(pb["page"])
            if page_idx not in page_cache:
                try:
                    page_cache[page_idx] = np.array(
                        Image.open(page_paths[page_idx - 1]).convert("L")
                    )
                except Exception:
                    page_cache[page_idx] = None

            page_gray = page_cache.get(page_idx)
            if page_gray is None:
                thumbs.append(None)
                continue

            x, y = int(pb["x"]), int(pb["y"])
            w, h = int(pb["width"]), int(pb["height"])
            crop = page_gray[
                max(y, 0):min(y + h, page_gray.shape[0]),
                max(x, 0):min(x + w, page_gray.shape[1]),
            ]
            if crop.size == 0:
                thumbs.append(None)
                continue

            # Resize to small thumbnail for fast comparison
            thumb = cv2.resize(
                crop,
                (self._DEDUP_THUMB_SIZE, self._DEDUP_THUMB_SIZE),
                interpolation=cv2.INTER_AREA,
            )
            thumbs.append(thumb)

        # Mark duplicates (keep first occurrence)
        seen_indices: list[int] = []
        for i, thumb_i in enumerate(thumbs):
            if thumb_i is None or panel_boxes[i].get("auto_skipped"):
                continue
            is_dup = False
            for j in seen_indices:
                thumb_j = thumbs[j]
                if thumb_j is None:
                    continue
                # Same-page panels are handled by per-page dedup already
                if panel_boxes[i]["page"] == panel_boxes[j]["page"]:
                    continue
                diff = float(np.mean(np.abs(
                    thumb_i.astype(np.float32) - thumb_j.astype(np.float32)
                )))
                if diff <= self._DEDUP_MAX_DIFF:
                    is_dup = True
                    break
            if is_dup:
                panel_boxes[i]["auto_skipped"] = True
                panel_boxes[i]["skip_reason"] = "duplicate of earlier panel"
                panel_boxes[i]["keep"] = False
            else:
                seen_indices.append(i)

        return panel_boxes

    def _should_skip_same_page_overlap_residue(
        self,
        *,
        smaller: dict,
        larger: dict,
        page_w: int,
        page_h: int,
        vertical_overlap: int,
        smaller_metrics: dict[str, float | tuple[int, int, int, int]],
        larger_metrics: dict[str, float | tuple[int, int, int, int]],
        is_webtoon_page: bool,
    ) -> bool:
        smaller_area = int(smaller["width"]) * int(smaller["height"])
        larger_area = max(int(larger["width"]) * int(larger["height"]), 1)
        smaller_width_ratio = int(smaller["width"]) / max(page_w, 1)
        larger_width_ratio = int(larger["width"]) / max(page_w, 1)
        smaller_height_ratio = int(smaller["height"]) / max(page_h, 1)
        smaller_area_ratio = smaller_area / max(page_w * page_h, 1)
        smaller_touches_edge = (
            int(smaller["y"]) <= self._BOUNDARY_EDGE_PX
            or (int(smaller["y"]) + int(smaller["height"])) >= page_h - self._BOUNDARY_EDGE_PX
        )
        overlap_slice_like = (
            vertical_overlap >= max(self._SAME_PAGE_OVERLAP_PX, int(page_h * 0.06))
            and min(smaller_width_ratio, larger_width_ratio) >= 0.82
        )
        smaller_score = float(smaller_metrics["score"])
        larger_score = float(larger_metrics["score"])
        low_content = (
            float(smaller_metrics["white_ratio"]) >= 0.20
            and float(smaller_metrics["edge_density"]) <= 0.07
        )
        if is_webtoon_page:
            return bool(
                int(smaller["height"]) <= int(larger["height"]) * 0.92
                and (
                    overlap_slice_like
                    or (smaller_touches_edge and smaller_area <= larger_area * 0.72)
                    or (
                        smaller_score <= larger_score * 0.96
                        or float(smaller_metrics["white_ratio"]) >= 0.18
                    )
                )
            )

        # Traditional manga pages often have stacked panels that legitimately
        # overlap after box refinement. Only suppress the smaller box when it
        # still clearly behaves like a shallow edge strip rather than a full beat.
        strip_like = (
            smaller_height_ratio <= 0.16
            or (
                smaller_height_ratio <= 0.22
                and smaller_width_ratio >= 0.84
            )
            or (
                smaller_height_ratio <= 0.30
                and smaller_width_ratio >= 0.88
                and (low_content or float(smaller_metrics["content_bbox_area_ratio"]) <= 0.74)
            )
            or (
                smaller_area_ratio <= 0.14
                and smaller_width_ratio >= 0.76
            )
        )
        if not strip_like:
            return False
        return bool(
            int(smaller["height"]) <= int(larger["height"]) * 0.88
            and (
                overlap_slice_like
                or (smaller_touches_edge and smaller_area <= larger_area * 0.54)
                or smaller_score <= larger_score * 0.92
                or low_content
            )
        )

    def _cleanup_same_page_panels(
        self,
        panel_boxes: list[dict],
        page_paths: list[Path],
    ) -> list[dict]:
        if not panel_boxes:
            return panel_boxes

        page_cache: dict[int, np.ndarray | None] = {}
        metrics_cache: dict[str, dict[str, float | tuple[int, int, int, int]]] = {}
        by_page: dict[int, list[dict]] = {}
        for pb in panel_boxes:
            by_page.setdefault(int(pb["page"]), []).append(pb)

        def page_character_boxes(page_idx: int) -> list[tuple[int, int, int, int]]:
            payload = self._last_character_review_page_payloads.get(int(page_idx), {})
            boxes: list[tuple[int, int, int, int]] = []
            for item in payload.get("characters", []) or []:
                try:
                    x, y, width, height = [int(value) for value in (item.get("bbox") or [])[:4]]
                except Exception:
                    continue
                if width > 0 and height > 0:
                    boxes.append((x, y, width, height))
            return boxes

        def has_character_center_inside(pb: dict, character_boxes: list[tuple[int, int, int, int]]) -> bool:
            x, y = int(pb["x"]), int(pb["y"])
            right = x + int(pb["width"])
            bottom = y + int(pb["height"])
            for cx, cy, cw, ch in character_boxes:
                center_x = cx + cw / 2
                center_y = cy + ch / 2
                if x <= center_x <= right and y <= center_y <= bottom:
                    return True
            return False

        def load_page(page_idx: int) -> np.ndarray | None:
            if page_idx not in page_cache:
                try:
                    page_cache[page_idx] = np.array(
                        Image.open(page_paths[page_idx - 1]).convert("RGB")
                    )
                except Exception:
                    page_cache[page_idx] = None
            return page_cache[page_idx]

        def metrics_for(pb: dict, page_img: np.ndarray | None) -> dict[str, float | tuple[int, int, int, int]]:
            panel_id = str(pb["id"])
            if panel_id in metrics_cache:
                return metrics_cache[panel_id]
            if page_img is None:
                metrics_cache[panel_id] = {
                    "white_ratio": 0.0,
                    "edge_density": 0.0,
                    "contrast": 0.0,
                    "content_bbox": (0, 0, int(pb["width"]), int(pb["height"])),
                    "content_bbox_area_ratio": 1.0,
                    "content_center_y_ratio": 0.5,
                    "score": 0.0,
                }
                return metrics_cache[panel_id]

            x, y = int(pb["x"]), int(pb["y"])
            w, h = int(pb["width"]), int(pb["height"])
            crop = page_img[
                max(y, 0):min(y + h, page_img.shape[0]),
                max(x, 0):min(x + w, page_img.shape[1]),
            ]
            metrics = self._compute_crop_metrics(crop)
            metrics["score"] = (
                metrics["edge_density"] * 3.2
                + metrics["contrast"] / 36.0
                - metrics["white_ratio"] * 0.9
                + metrics["content_bbox_area_ratio"] * 0.6
            )
            metrics_cache[panel_id] = metrics
            return metrics

        for page_idx, items in by_page.items():
            page_img = load_page(page_idx)
            if page_img is None:
                continue
            page_h, page_w = page_img.shape[:2]
            is_webtoon_page = self._is_webtoon_page(page_w, page_h)
            character_boxes = page_character_boxes(page_idx)
            active = sorted(
                [pb for pb in items if not pb.get("auto_skipped")],
                key=lambda pb: (int(pb["y"]), int(pb["x"]), int(pb["panel"])),
            )
            if not active:
                continue

            # Pass 1: kill same-page overlap residue from webtoon slicing.
            for first, second in zip(active, active[1:]):
                if first.get("auto_skipped") or second.get("auto_skipped"):
                    continue
                vertical_overlap = max(
                    0,
                    min(int(first["y"]) + int(first["height"]), int(second["y"]) + int(second["height"]))
                    - max(int(first["y"]), int(second["y"]))
                )
                if vertical_overlap < self._SAME_PAGE_OVERLAP_PX:
                    continue
                horizontal_overlap = max(
                    0,
                    min(int(first["x"]) + int(first["width"]), int(second["x"]) + int(second["width"]))
                    - max(int(first["x"]), int(second["x"]))
                )
                min_width = max(min(int(first["width"]), int(second["width"])), 1)
                if horizontal_overlap / min_width < 0.78:
                    continue

                first_metrics = metrics_for(first, page_img)
                second_metrics = metrics_for(second, page_img)
                first_area = int(first["width"]) * int(first["height"])
                second_area = int(second["width"]) * int(second["height"])
                smaller, larger = (
                    (first, second) if first_area <= second_area else (second, first)
                )
                smaller_metrics = first_metrics if smaller is first else second_metrics
                larger_metrics = second_metrics if smaller is first else first_metrics
                if self._should_skip_same_page_overlap_residue(
                    smaller=smaller,
                    larger=larger,
                    page_w=page_w,
                    page_h=page_h,
                    vertical_overlap=vertical_overlap,
                    smaller_metrics=smaller_metrics,
                    larger_metrics=larger_metrics,
                    is_webtoon_page=is_webtoon_page,
                ):
                    smaller["auto_skipped"] = True
                    smaller["keep"] = False
                    smaller["skip_reason"] = "overlapping strip from same-page panel split"

            # Pass 2: suppress white-gap / speech-bubble fragments between real panels.
            active = sorted(
                [pb for pb in items if not pb.get("auto_skipped")],
                key=lambda pb: (int(pb["y"]), int(pb["x"]), int(pb["panel"])),
            )
            gap_threshold = max(int(page_h * self._SAME_PAGE_GAP_RATIO), 220)
            for idx, current in enumerate(active):
                if current.get("auto_skipped"):
                    continue
                current_metrics = metrics_for(current, page_img)
                prev_panel = active[idx - 1] if idx > 0 else None
                next_panel = active[idx + 1] if idx + 1 < len(active) else None
                gap_above = (
                    int(current["y"]) - (int(prev_panel["y"]) + int(prev_panel["height"]))
                    if prev_panel is not None else int(current["y"])
                )
                gap_below = (
                    int(next_panel["y"]) - (int(current["y"]) + int(current["height"]))
                    if next_panel is not None else page_h - (int(current["y"]) + int(current["height"]))
                )

                short_panel = int(current["height"]) <= page_h * self._WHITE_GAP_SHORT_RATIO
                wide_panel = int(current["width"]) >= page_w * 0.72
                white_gap_like = (
                    float(current_metrics["white_ratio"]) >= 0.34
                    and float(current_metrics["edge_density"]) <= 0.075
                    and float(current_metrics["content_bbox_area_ratio"]) <= 0.70
                )
                top_edge_fragment_like = is_webtoon_page and (
                    int(current["y"]) <= self._BOUNDARY_EDGE_PX and gap_below >= gap_threshold
                )
                top_continuation_like = is_webtoon_page and (
                    int(current["y"]) <= self._BOUNDARY_EDGE_PX
                    and int(current["width"]) >= page_w * 0.82
                    and int(current["height"]) <= page_h * 0.28
                    and next_panel is not None
                    and int(next_panel["width"]) >= page_w * 0.82
                    and gap_below <= max(int(page_h * 0.06), 130)
                    and (int(next_panel["width"]) * int(next_panel["height"])) >= (int(current["width"]) * int(current["height"])) * 2.2
                )
                top_gap_fragment_like = is_webtoon_page and (
                    int(current["y"]) <= self._BOUNDARY_EDGE_PX
                    and int(current["width"]) >= page_w * 0.82
                    and int(current["height"]) <= page_h * 0.42
                    and gap_below >= max(gap_threshold, int(page_h * 0.18))
                    and float(current_metrics["white_ratio"]) >= 0.30
                    and float(current_metrics["edge_density"]) <= 0.07
                )
                isolated_gap_like = (
                    (gap_above >= gap_threshold or gap_below >= gap_threshold)
                    and short_panel
                    and wide_panel
                    and white_gap_like
                )
                speech_bubble_strip_like = (
                    short_panel
                    and wide_panel
                    and float(current_metrics["white_ratio"]) >= 0.55
                    and float(current_metrics["edge_density"]) <= 0.07
                    and prev_panel is not None
                    and int(prev_panel["width"]) >= page_w * 0.82
                    and int(prev_panel["height"]) >= int(current["height"]) * 1.35
                    and gap_above <= max(int(page_h * 0.05), 120)
                )
                area_ratio = (
                    (int(current["width"]) * int(current["height"]))
                    / max(page_w * page_h, 1)
                )
                source = str(current.get("reconstruction_source") or "")
                no_character_center = not has_character_center_inside(current, character_boxes)
                near_story_panel = (
                    (prev_panel is not None and gap_above <= max(int(page_h * 0.18), 220))
                    or (next_panel is not None and gap_below <= max(int(page_h * 0.18), 220))
                    or gap_above < 0
                    or gap_below < 0
                )
                floating_text_fragment_like = (
                    no_character_center
                    and near_story_panel
                    and area_ratio <= 0.16
                    and int(current["width"]) <= page_w * 0.48
                    and (
                        (
                            float(current_metrics["white_ratio"]) >= 0.40
                            and float(current_metrics["edge_density"]) <= 0.09
                        )
                        or (
                            "ocr_cluster" in source
                            and area_ratio <= 0.04
                            and float(current_metrics["white_ratio"]) >= 0.24
                            and float(current_metrics["edge_density"]) <= 0.13
                        )
                    )
                )
                if top_continuation_like:
                    current["auto_skipped"] = True
                    current["keep"] = False
                    current["skip_reason"] = "top continuation fragment"
                elif top_gap_fragment_like:
                    current["auto_skipped"] = True
                    current["keep"] = False
                    current["skip_reason"] = "low-content top gap fragment"
                elif top_edge_fragment_like and (
                    short_panel
                    or float(current_metrics["content_center_y_ratio"]) <= 0.40
                    or float(current_metrics["content_center_y_ratio"]) >= 0.60
                ):
                    current["auto_skipped"] = True
                    current["keep"] = False
                    current["skip_reason"] = "low-content boundary fragment"
                elif isolated_gap_like:
                    current["auto_skipped"] = True
                    current["keep"] = False
                    current["skip_reason"] = "low-content white-gap panel"
                elif speech_bubble_strip_like:
                    current["auto_skipped"] = True
                    current["keep"] = False
                    current["skip_reason"] = "speech-bubble strip below larger panel"
                elif floating_text_fragment_like:
                    current["auto_skipped"] = True
                    current["keep"] = False
                    current["skip_reason"] = "floating speech-bubble/text fragment"

        return panel_boxes

    def _refine_boxes(
        self,
        panel_boxes: list[dict],
        page_paths: list[Path],
    ) -> list[dict]:
        """
        Tighten bounding boxes by trimming uniform white/blank borders.

        This removes the white margins that are common when the CV
        detector overshoots panel boundaries, producing cleaner crops
        that look better in the final video.
        """
        page_cache: dict[int, np.ndarray] = {}
        refined: list[dict] = []

        for pb in panel_boxes:
            page_idx = int(pb["page"])
            if page_idx not in page_cache:
                try:
                    page_cache[page_idx] = np.array(
                        Image.open(page_paths[page_idx - 1]).convert("RGB")
                    )
                except Exception:
                    page_cache[page_idx] = None

            page_img = page_cache.get(page_idx)
            if page_img is None or pb.get("auto_skipped"):
                refined.append(pb)
                continue

            page_h, page_w = page_img.shape[:2]
            x, y = int(pb["x"]), int(pb["y"])
            w, h = int(pb["width"]), int(pb["height"])

            crop = page_img[
                max(y, 0):min(y + h, page_h),
                max(x, 0):min(x + w, page_w),
            ]
            if crop.size == 0 or min(w, h) < 60:
                refined.append(pb)
                continue

            gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
            edges = cv2.Canny(gray, 40, 120)
            bright_cutoff = max(232, int(np.percentile(gray, 92)) - 4)
            content_mask = (gray < bright_cutoff) | (edges > 0)

            # Find content bounding box
            rows = np.mean(content_mask, axis=1) > 0.01
            cols = np.mean(content_mask, axis=0) > 0.01

            if not np.any(rows) or not np.any(cols):
                refined.append(pb)
                continue

            row_indices = np.where(rows)[0]
            col_indices = np.where(cols)[0]
            cy1 = int(row_indices[0])
            cy2 = int(row_indices[-1]) + 1
            cx1 = int(col_indices[0])
            cx2 = int(col_indices[-1]) + 1

            # Only refine if we're trimming significant whitespace
            # (at least 3% of the dimension on any side)
            trim_top = cy1
            trim_bottom = h - cy2
            trim_left = cx1
            trim_right = w - cx2
            min_trim = max(int(min(w, h) * 0.03), 4)

            if max(trim_top, trim_bottom, trim_left, trim_right) >= min_trim:
                # Add a small padding (2%) so content doesn't touch the edge
                pad_x = max(int(w * 0.02), 2)
                pad_y = max(int(h * 0.02), 2)
                new_x = x + max(cx1 - pad_x, 0)
                new_y = y + max(cy1 - pad_y, 0)
                new_x2 = x + min(cx2 + pad_x, w)
                new_y2 = y + min(cy2 + pad_y, h)
                new_w = max(new_x2 - new_x, 1)
                new_h = max(new_y2 - new_y, 1)

                # Don't refine if we'd shrink by more than 40% — that
                # suggests the panel is legitimately sparse.
                if new_w * new_h >= w * h * 0.60:
                    pb = dict(pb)
                    pb["x"] = new_x
                    pb["y"] = new_y
                    pb["width"] = new_w
                    pb["height"] = new_h

            border_trimmed = self._trim_border_connected_blank_regions(page_img, pb)
            if border_trimmed is not None:
                pb = border_trimmed

            tightened = self._trim_sparse_margins(page_img, pb)
            if tightened is not None:
                pb = tightened

            occupancy_trimmed = self._trim_low_occupancy_margins(page_img, pb)
            if occupancy_trimmed is not None:
                pb = occupancy_trimmed

            mask_trimmed = self._trim_to_mask_aware_rectangle(page_img, pb)
            if mask_trimmed is not None:
                pb = mask_trimmed

            refined.append(pb)

        return refined

    def _trim_border_connected_blank_regions(self, page_img: np.ndarray, pb: dict) -> dict | None:
        x, y = int(pb["x"]), int(pb["y"])
        w, h = int(pb["width"]), int(pb["height"])
        crop = page_img[
            max(y, 0):min(y + h, page_img.shape[0]),
            max(x, 0):min(x + w, page_img.shape[1]),
        ]
        if crop.size == 0 or min(w, h) < 80:
            return None

        gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 40, 120)
        edge_guard = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
        bright_cutoff = max(238, min(248, int(np.percentile(gray, 85))))
        blank_mask = ((gray >= bright_cutoff) & (edge_guard == 0)).astype(np.uint8)
        if not np.any(blank_mask):
            return None

        num_labels, labels = cv2.connectedComponents(blank_mask)
        if num_labels <= 1:
            return None

        border_labels = set()
        border_labels.update(int(v) for v in labels[0, :])
        border_labels.update(int(v) for v in labels[-1, :])
        border_labels.update(int(v) for v in labels[:, 0])
        border_labels.update(int(v) for v in labels[:, -1])
        border_labels.discard(0)
        if not border_labels:
            return None

        border_blank = np.isin(labels, list(border_labels))
        occupied = ~border_blank
        if float(np.mean(occupied)) < 0.45:
            return None

        step_x = max(4, min(18, w // 40))
        step_y = max(4, min(18, h // 40))
        max_x_trim = int(w * 0.45)
        max_y_trim = int(h * 0.26)
        trim_left = trim_right = trim_top = trim_bottom = 0

        while trim_left + step_x < max_x_trim:
            band = border_blank[:, trim_left:trim_left + step_x]
            if float(np.mean(band)) < 0.72:
                break
            trim_left += step_x
        while trim_right + step_x < max_x_trim:
            band = border_blank[:, max(w - trim_right - step_x, 0):w - trim_right]
            if float(np.mean(band)) < 0.72:
                break
            trim_right += step_x
        while trim_top + step_y < max_y_trim:
            band = border_blank[trim_top:trim_top + step_y, :]
            if float(np.mean(band)) < 0.72:
                break
            trim_top += step_y
        while trim_bottom + step_y < max_y_trim:
            band = border_blank[max(h - trim_bottom - step_y, 0):h - trim_bottom, :]
            if float(np.mean(band)) < 0.72:
                break
            trim_bottom += step_y

        if max(trim_top, trim_bottom, trim_left, trim_right) < max(int(min(w, h) * 0.03), 6):
            rows = np.where(np.mean(occupied, axis=1) > 0.06)[0]
            cols = np.where(np.mean(occupied, axis=0) > 0.10)[0]
            if rows.size == 0 or cols.size == 0:
                return None
            y1 = int(rows[0])
            y2 = int(rows[-1]) + 1
            x1 = int(cols[0])
            x2 = int(cols[-1]) + 1
            trim_top = y1
            trim_bottom = h - y2
            trim_left = x1
            trim_right = w - x2

        if max(trim_top, trim_bottom, trim_left, trim_right) < max(int(min(w, h) * 0.03), 6):
            return None

        pad_x = max(int(w * 0.015), 2)
        pad_y = max(int(h * 0.015), 2)
        new_x = x + max(trim_left - pad_x, 0)
        new_y = y + max(trim_top - pad_y, 0)
        new_x2 = x + min(w - trim_right + pad_x, w)
        new_y2 = y + min(h - trim_bottom + pad_y, h)
        new_w = max(new_x2 - new_x, 1)
        new_h = max(new_y2 - new_y, 1)
        if new_w <= 40 or new_h <= 40:
            return None
        if new_w * new_h < w * h * 0.55:
            return None

        tightened = dict(pb)
        tightened["x"] = new_x
        tightened["y"] = new_y
        tightened["width"] = new_w
        tightened["height"] = new_h
        return tightened

    def _trim_low_occupancy_margins(self, page_img: np.ndarray, pb: dict) -> dict | None:
        x, y = int(pb["x"]), int(pb["y"])
        w, h = int(pb["width"]), int(pb["height"])
        crop = page_img[
            max(y, 0):min(y + h, page_img.shape[0]),
            max(x, 0):min(x + w, page_img.shape[1]),
        ]
        if crop.size == 0 or min(w, h) < 80:
            return None

        gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 40, 120)
        bright_cutoff = max(232, int(np.percentile(gray, 90)) - 4)
        content_mask = ((gray < bright_cutoff) | (edges > 0)).astype(np.float32)
        row_signal = content_mask.mean(axis=1)
        col_signal = content_mask.mean(axis=0)

        row_window = max(9, min(41, (h // 18) | 1))
        col_window = max(9, min(41, (w // 18) | 1))
        row_kernel = np.ones(row_window, dtype=np.float32) / row_window
        col_kernel = np.ones(col_window, dtype=np.float32) / col_window
        smooth_rows = np.convolve(row_signal, row_kernel, mode="same")
        smooth_cols = np.convolve(col_signal, col_kernel, mode="same")

        row_threshold = 0.10
        col_threshold = 0.12
        rows = np.where(smooth_rows > row_threshold)[0]
        cols = np.where(smooth_cols > col_threshold)[0]
        if rows.size == 0 or cols.size == 0:
            return None

        y1 = int(rows[0])
        y2 = int(rows[-1]) + 1
        x1 = int(cols[0])
        x2 = int(cols[-1]) + 1
        trim_top = y1
        trim_bottom = h - y2
        trim_left = x1
        trim_right = w - x2
        if max(trim_top, trim_bottom, trim_left, trim_right) < max(int(min(w, h) * 0.04), 8):
            return None

        pad_x = max(int(w * 0.01), 2)
        pad_y = max(int(h * 0.01), 2)
        new_x = x + max(x1 - pad_x, 0)
        new_y = y + max(y1 - pad_y, 0)
        new_x2 = x + min(x2 + pad_x, w)
        new_y2 = y + min(y2 + pad_y, h)
        new_w = max(new_x2 - new_x, 1)
        new_h = max(new_y2 - new_y, 1)
        if new_w <= 40 or new_h <= 40:
            return None
        if new_w * new_h < w * h * 0.50:
            return None

        tightened = dict(pb)
        tightened["x"] = new_x
        tightened["y"] = new_y
        tightened["width"] = new_w
        tightened["height"] = new_h
        return tightened

    def _trim_to_mask_aware_rectangle(self, page_img: np.ndarray, pb: dict) -> dict | None:
        x, y = int(pb["x"]), int(pb["y"])
        w, h = int(pb["width"]), int(pb["height"])
        crop = page_img[
            max(y, 0):min(y + h, page_img.shape[0]),
            max(x, 0):min(x + w, page_img.shape[1]),
        ]
        if crop.size == 0 or min(w, h) < 80:
            return None

        gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 40, 120)
        edge_guard = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
        bright_cutoff = max(238, min(248, int(np.percentile(gray, 85))))
        blank_mask = ((gray >= bright_cutoff) & (edge_guard == 0)).astype(np.uint8)
        if not np.any(blank_mask):
            return None

        labels_count, labels = cv2.connectedComponents(blank_mask)
        if labels_count <= 1:
            return None

        border_labels = set()
        border_labels.update(int(value) for value in labels[0, :])
        border_labels.update(int(value) for value in labels[-1, :])
        border_labels.update(int(value) for value in labels[:, 0])
        border_labels.update(int(value) for value in labels[:, -1])
        border_labels.discard(0)
        if not border_labels:
            return None

        border_blank = np.isin(labels, list(border_labels))
        occupied = (~border_blank).astype(np.uint8)
        border_blank_ratio = float(np.mean(border_blank))
        if float(np.mean(occupied)) <= 0.10:
            return None

        downsample = 4 if max(w, h) >= 600 else 2
        small_w = max(1, w // downsample)
        small_h = max(1, h // downsample)
        small_mask = cv2.resize(occupied, (small_w, small_h), interpolation=cv2.INTER_AREA) > 0.85
        rect = self._largest_true_rectangle(small_mask)
        if rect is None:
            return None

        left, top, rect_w, rect_h = rect
        new_x = x + left * downsample
        new_y = y + top * downsample
        new_w = min(rect_w * downsample, w - left * downsample)
        new_h = min(rect_h * downsample, h - top * downsample)
        if new_w <= 40 or new_h <= 40:
            return None

        area_ratio = float((new_w * new_h) / max(w * h, 1))
        trim_left = new_x - x
        trim_top = new_y - y
        trim_right = (x + w) - (new_x + new_w)
        trim_bottom = (y + h) - (new_y + new_h)
        largest_trim = max(trim_left, trim_top, trim_right, trim_bottom)
        if largest_trim < max(int(min(w, h) * 0.05), 8):
            return None

        band_y = max(20, int(h * 0.12))
        band_x = max(20, int(w * 0.12))
        top_band_blank = float(np.mean(border_blank[:band_y, :]))
        bottom_band_blank = float(np.mean(border_blank[-band_y:, :]))
        corner_blank = max(
            float(np.mean(border_blank[:band_y, :band_x])),
            float(np.mean(border_blank[:band_y, -band_x:])),
            float(np.mean(border_blank[-band_y:, :band_x])),
            float(np.mean(border_blank[-band_y:, -band_x:])),
        )

        if border_blank_ratio >= 0.22:
            min_area_ratio = 0.35
        elif border_blank_ratio >= 0.08:
            min_area_ratio = 0.60
        else:
            min_area_ratio = 0.78

        if top_band_blank >= 0.60 or bottom_band_blank >= 0.60 or corner_blank >= 0.42:
            min_area_ratio = min(min_area_ratio, 0.82 if border_blank_ratio < 0.08 else min_area_ratio)

        if area_ratio < min_area_ratio:
            return None

        pad_x = max(2, int(new_w * 0.006))
        pad_y = max(2, int(new_h * 0.006))
        final_x = max(x, new_x - pad_x)
        final_y = max(y, new_y - pad_y)
        final_x2 = min(x + w, new_x + new_w + pad_x)
        final_y2 = min(y + h, new_y + new_h + pad_y)
        final_w = max(final_x2 - final_x, 1)
        final_h = max(final_y2 - final_y, 1)
        if final_w * final_h < w * h * min_area_ratio:
            return None

        tightened = dict(pb)
        tightened["x"] = final_x
        tightened["y"] = final_y
        tightened["width"] = final_w
        tightened["height"] = final_h
        return tightened

    def _trim_sparse_margins(self, page_img: np.ndarray, pb: dict) -> dict | None:
        x, y = int(pb["x"]), int(pb["y"])
        w, h = int(pb["width"]), int(pb["height"])
        crop = page_img[
            max(y, 0):min(y + h, page_img.shape[0]),
            max(x, 0):min(x + w, page_img.shape[1]),
        ]
        if crop.size == 0 or min(w, h) < 80:
            return None

        gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 40, 120)

        def strip_metrics(arr: np.ndarray, edge_arr: np.ndarray) -> tuple[float, float]:
            return float(np.mean(arr > 242)), float(np.mean(edge_arr > 0))

        left = right = top = bottom = 0
        step_x = max(4, min(18, w // 40))
        step_y = max(4, min(18, h // 40))
        max_x_trim = int(w * 0.42)
        max_y_trim = int(h * 0.24)

        while left + step_x < max_x_trim:
            strip = gray[:, left:left + step_x]
            strip_edges = edges[:, left:left + step_x]
            white, edge = strip_metrics(strip, strip_edges)
            dark = float(np.mean(strip < 236))
            if white < 0.78 or edge > 0.018 or dark > 0.24:
                break
            left += step_x
        while right + step_x < max_x_trim:
            strip = gray[:, max(w - right - step_x, 0):w - right]
            strip_edges = edges[:, max(w - right - step_x, 0):w - right]
            white, edge = strip_metrics(strip, strip_edges)
            dark = float(np.mean(strip < 236))
            if white < 0.78 or edge > 0.018 or dark > 0.24:
                break
            right += step_x
        while top + step_y < max_y_trim:
            strip = gray[top:top + step_y, :]
            strip_edges = edges[top:top + step_y, :]
            white, edge = strip_metrics(strip, strip_edges)
            dark = float(np.mean(strip < 236))
            if white < 0.72 or edge > 0.04 or dark > 0.30:
                break
            top += step_y
        while bottom + step_y < max_y_trim:
            strip = gray[max(h - bottom - step_y, 0):h - bottom, :]
            strip_edges = edges[max(h - bottom - step_y, 0):h - bottom, :]
            white, edge = strip_metrics(strip, strip_edges)
            dark = float(np.mean(strip < 236))
            if white < 0.72 or edge > 0.04 or dark > 0.30:
                break
            bottom += step_y

        if max(left, right, top, bottom) < max(min(w, h) * 0.01, 4):
            return None

        new_x = x + left
        new_y = y + top
        new_w = w - left - right
        new_h = h - top - bottom
        if new_w <= 40 or new_h <= 40:
            return None
        if new_w * new_h < w * h * 0.60:
            return None

        tightened = dict(pb)
        tightened["x"] = new_x
        tightened["y"] = new_y
        tightened["width"] = new_w
        tightened["height"] = new_h
        return tightened

    def _largest_true_rectangle(self, mask: np.ndarray) -> tuple[int, int, int, int] | None:
        if mask.size == 0:
            return None

        height, width = mask.shape
        histogram = [0] * width
        best_area = 0
        best_rect: tuple[int, int, int, int] | None = None

        for row_index in range(height):
            row = mask[row_index]
            for column_index, value in enumerate(row):
                histogram[column_index] = histogram[column_index] + 1 if value else 0

            stack: list[int] = []
            column_index = 0
            while column_index <= width:
                current_height = histogram[column_index] if column_index < width else 0
                if not stack or current_height >= histogram[stack[-1]]:
                    stack.append(column_index)
                    column_index += 1
                    continue

                top_index = stack.pop()
                rect_height = histogram[top_index]
                left_index = stack[-1] + 1 if stack else 0
                rect_width = column_index - left_index
                area = rect_height * rect_width
                if area > best_area:
                    best_area = area
                    best_rect = (left_index, row_index - rect_height + 1, rect_width, rect_height)

        return best_rect

    @staticmethod
    def _compute_crop_metrics(
        crop: np.ndarray,
    ) -> dict[str, float | tuple[int, int, int, int]]:
        if crop.size == 0:
            return {
                "white_ratio": 1.0,
                "edge_density": 0.0,
                "contrast": 0.0,
                "content_bbox": (0, 0, 0, 0),
                "content_bbox_area_ratio": 0.0,
                "content_center_y_ratio": 0.5,
            }

        gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 40, 120)
        white_ratio = float(np.mean(gray > 240))
        edge_density = float(np.mean(edges > 0))
        contrast = float(np.std(gray))

        content_mask = (gray < 236) | (edges > 0)
        rows = np.where(np.mean(content_mask, axis=1) > 0.01)[0]
        cols = np.where(np.mean(content_mask, axis=0) > 0.01)[0]
        if rows.size == 0 or cols.size == 0:
            bbox = (0, 0, crop.shape[1], crop.shape[0])
            bbox_area_ratio = 1.0
            center_y_ratio = 0.5
        else:
            y1 = int(rows[0])
            y2 = int(rows[-1]) + 1
            x1 = int(cols[0])
            x2 = int(cols[-1]) + 1
            bbox = (x1, y1, x2 - x1, y2 - y1)
            bbox_area_ratio = float((bbox[2] * bbox[3]) / max(crop.shape[0] * crop.shape[1], 1))
            center_y_ratio = float((y1 + y2) / 2.0 / max(crop.shape[0], 1))

        return {
            "white_ratio": white_ratio,
            "edge_density": edge_density,
            "contrast": contrast,
            "content_bbox": bbox,
            "content_bbox_area_ratio": bbox_area_ratio,
            "content_center_y_ratio": center_y_ratio,
        }

    @staticmethod
    def _zoom_hint(w: int, h: int) -> str:
        ar = w / max(h, 1)
        if w * h > 1_100_000:
            return "pan-wide"
        if ar > 1.6:
            return "pan-horizontal"
        if ar < 0.8:
            return "pan-vertical"
        return "focus-center"


# Alias so the rest of the pipeline can import the historical name.
MagiPanelDetectionService = PanelDetectorAdapter


class MagiSpeakerAttributionService:
    """
    Uses the Magi model to detect which speech bubbles belong to which
    characters on a page.  When the model is not loaded, returns empty
    results so the pipeline degrades gracefully.
    """

    def __init__(self, magi_model: Any | None = None) -> None:
        self._magi_model = magi_model

    def detect_page_associations(
        self,
        page_paths: list[Path],
        panels: list,
        cancel_callback: callable | None = None,
    ) -> dict[int, dict[str, Any]]:
        """
        Returns {page_number: {"characters": [...], "text_boxes": [...], ...}}

        Without a loaded Magi model, returns empty dicts so downstream
        code can still function.
        """
        pages_needed = sorted({int(p.page) for p in panels})
        payloads = MagiHFService().predict_page_payloads(
            page_paths,
            page_numbers=pages_needed,
            do_ocr=False,
            model=self._magi_model,
            cancel_callback=cancel_callback,
        )
        if payloads:
            return payloads
        if self._magi_model is None:
            return {}

        result: dict[int, dict[str, Any]] = {}

        for page_num in pages_needed:
            if cancel_callback:
                cancel_callback()
            if page_num < 1 or page_num > len(page_paths):
                continue

            try:
                import torch

                image = np.array(Image.open(page_paths[page_num - 1]).convert("RGB"))
                detector = PanelDetector(
                    config=DetectorConfig(),
                    reading_order=ReadingOrder.MANHWA,
                    magi_model=self._magi_model,
                )
                det = detector.detect_panels(image, page_index=page_num)

                result[page_num] = {
                    "characters": det.character_boxes,
                    "text_boxes": det.text_boxes,
                    "panels": [(p.x, p.y, p.width, p.height) for p in det.panels],
                }
            except Exception as exc:
                logger.warning("Speaker attribution failed for page %d: %s", page_num, exc)

        return result
