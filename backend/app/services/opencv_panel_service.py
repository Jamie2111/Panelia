from __future__ import annotations

from typing import Iterable

import numpy as np


Box = tuple[int, int, int, int]


class OpenCVPanelDetectionService:
    """
    Contour-based helper inspired by adenzu/Manga-Panel-Extractor.

    The upstream project is tuned for manga rather than manhwa/webtoons, so
    Panelia uses its background/contour ideas as a secondary signal instead of
    replacing the learned detector outright.
    """

    def detect_boxes(self, image: np.ndarray) -> list[Box]:
        import cv2

        if image.ndim == 2:
            grayscale = image
        else:
            grayscale = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        page_height, page_width = grayscale.shape[:2]
        tall_page = page_height / max(page_width, 1) >= 1.45

        vertical_boxes = self._detect_vertical_segments(grayscale, page_width, page_height, tall_page)
        contour_boxes = self._detect_rectangular_contours(grayscale, page_width, page_height, tall_page)
        merged = self._merge_candidate_sets([vertical_boxes, contour_boxes], page_width, page_height)
        refined: list[Box] = []
        for box in merged:
            split_boxes = self._split_composite_box(grayscale, box, page_width, page_height)
            if split_boxes:
                refined.extend(split_boxes)
            else:
                refined.append(box)
        return self._clean_boxes(refined, page_width, page_height)

    def _detect_vertical_segments(self, grayscale: np.ndarray, page_width: int, page_height: int, tall_page: bool) -> list[Box]:
        ink_mask = grayscale < 245
        row_density = ink_mask.mean(axis=1)
        row_density = self._smooth_signal(row_density, max(9, int(page_height * 0.008)))
        row_threshold = 0.008 if tall_page else 0.015
        runs = self._extract_runs(row_density > row_threshold)

        min_run_height = max(int(page_width * (0.2 if tall_page else 0.14)), 88)
        boxes: list[Box] = []
        for y1, y2 in runs:
            run_height = y2 - y1
            if run_height < min_run_height:
                continue

            region = ink_mask[y1:y2]
            col_density = self._smooth_signal(region.mean(axis=0), max(7, int(page_width * 0.012)))
            col_threshold = 0.01 if tall_page else 0.018
            content_columns = np.where(col_density > col_threshold)[0]
            if content_columns.size == 0:
                continue

            x1 = int(content_columns[0])
            x2 = int(content_columns[-1] + 1)
            pad_x = max(int(page_width * 0.02), 12)
            pad_y = max(int(run_height * 0.06), 16)
            box = self._expand_box((x1, y1, x2 - x1, run_height), page_width, page_height, pad_x=pad_x, pad_y=pad_y)
            boxes.append(box)

        return boxes

    def _detect_rectangular_contours(self, grayscale: np.ndarray, page_width: int, page_height: int, tall_page: bool) -> list[Box]:
        import cv2

        blurred = cv2.GaussianBlur(grayscale, (5, 5), 0)
        laplacian = cv2.Laplacian(blurred, cv2.CV_8U)
        inverted = 255 - laplacian
        adaptive = cv2.adaptiveThreshold(inverted, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 2)
        ink = 255 - adaptive

        close_kernel = np.ones((5, 5), np.uint8)
        dilate_kernel = np.ones((3, 3), np.uint8)
        ink = cv2.morphologyEx(ink, cv2.MORPH_CLOSE, close_kernel, iterations=2)
        ink = cv2.dilate(ink, dilate_kernel, iterations=1)

        contours, _ = cv2.findContours(ink, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        page_area = page_width * page_height
        min_area = page_area * (0.02 if tall_page else 0.015)

        boxes: list[Box] = []
        for contour in contours:
            x, y, width, height = cv2.boundingRect(contour)
            box_area = width * height
            if box_area < min_area:
                continue
            if width < page_width * 0.14 or height < max(page_height * 0.035, 72):
                continue
            if box_area > page_area * 0.95:
                continue

            contour_area = max(cv2.contourArea(contour), 1.0)
            fill_ratio = contour_area / max(box_area, 1)
            if fill_ratio < (0.22 if tall_page else 0.18):
                continue

            boxes.append(self._expand_box((x, y, width, height), page_width, page_height))

        return boxes

    def _merge_candidate_sets(self, candidate_sets: Iterable[list[Box]], page_width: int, page_height: int) -> list[Box]:
        merged: list[Box] = []
        for candidate_set in candidate_sets:
            for box in candidate_set:
                duplicate_index = self._matching_box_index(merged, box)
                if duplicate_index is not None:
                    existing = merged[duplicate_index]
                    if self._box_area(box) < self._box_area(existing):
                        merged[duplicate_index] = box
                    continue
                merged.append(box)

        return self._clean_boxes(merged, page_width, page_height)

    def _clean_boxes(self, boxes: Iterable[Box], page_width: int, page_height: int) -> list[Box]:
        page_area = page_width * page_height
        cleaned: list[Box] = []

        for raw_box in boxes:
            x, y, width, height = [int(value) for value in raw_box]
            if width < 48 or height < 48:
                continue
            area = width * height
            if area < page_area * 0.01:
                continue
            if width >= page_width * 0.94 and height >= page_height * 0.9:
                continue
            if width >= page_width * 0.9 and height >= page_height * 0.72:
                continue
            if width > page_width or height > page_height:
                continue

            box = (max(x, 0), max(y, 0), width, height)
            duplicate_index = self._matching_box_index(cleaned, box)
            if duplicate_index is not None:
                existing = cleaned[duplicate_index]
                if self._box_area(box) < self._box_area(existing):
                    cleaned[duplicate_index] = box
                continue
            cleaned.append(box)

        ordered = sorted(cleaned, key=lambda box: (box[1], box[0], box[2] * box[3]))
        suppressed: list[Box] = []
        for box in ordered:
            if any(self._contains(other, box) for other in suppressed):
                continue
            suppressed.append(box)
        return suppressed

    def _split_composite_box(self, grayscale: np.ndarray, box: Box, page_width: int, page_height: int) -> list[Box]:
        import cv2

        x, y, width, height = [int(value) for value in box]
        if width < page_width * 0.58:
            return []
        if height < max(page_height * 0.34, width * 1.05):
            return []

        x1 = max(x, 0)
        y1 = max(y, 0)
        x2 = min(x + width, page_width)
        y2 = min(y + height, page_height)
        if x2 <= x1 or y2 <= y1:
            return []

        crop = grayscale[y1:y2, x1:x2]
        if crop.size == 0:
            return []

        blurred = cv2.GaussianBlur(crop, (3, 3), 0)
        edges = cv2.Canny(blurred, 40, 120)
        signal = (blurred < 242) | (edges > 0)
        bright = blurred > 245
        row_signal = self._smooth_signal(signal.mean(axis=1), max(9, int(height * 0.018)))
        row_bright = self._smooth_signal(bright.mean(axis=1), max(9, int(height * 0.018)))
        gutter_runs = self._extract_runs((row_signal < 0.02) & (row_bright > 0.88))

        usable_runs = [
            (start, end)
            for start, end in gutter_runs
            if end - start >= max(18, int(height * 0.02))
            and start >= int(height * 0.12)
            and end <= int(height * 0.88)
        ]
        if not usable_runs:
            return []

        segments: list[tuple[int, int]] = []
        cursor = 0
        for gutter_start, gutter_end in usable_runs:
            if gutter_start - cursor >= int(height * 0.18):
                segments.append((cursor, gutter_start))
            cursor = gutter_end
        if height - cursor >= int(height * 0.18):
            segments.append((cursor, height))
        if len(segments) < 2:
            return []

        split_boxes: list[Box] = []
        for seg_start, seg_end in segments:
            segment_height = seg_end - seg_start
            if segment_height < max(120, int(height * 0.16)):
                continue
            split_boxes.append((x1, y1 + seg_start, width, segment_height))

        if len(split_boxes) < 2:
            return []
        if sum(box_width * box_height for _, _, box_width, box_height in split_boxes) < width * height * 0.58:
            return []
        return split_boxes

    def _expand_box(self, box: Box, page_width: int, page_height: int, pad_x: int | None = None, pad_y: int | None = None) -> Box:
        x, y, width, height = box
        pad_x = pad_x if pad_x is not None else max(int(width * 0.04), 10)
        pad_y = pad_y if pad_y is not None else max(int(height * 0.04), 10)
        x1 = max(x - pad_x, 0)
        y1 = max(y - pad_y, 0)
        x2 = min(x + width + pad_x, page_width)
        y2 = min(y + height + pad_y, page_height)
        return (int(x1), int(y1), int(x2 - x1), int(y2 - y1))

    def _matching_box_index(self, boxes: list[Box], candidate: Box) -> int | None:
        for index, box in enumerate(boxes):
            if self._iou(box, candidate) >= 0.58:
                return index
            if self._contains(box, candidate) or self._contains(candidate, box):
                return index
        return None

    def _contains(self, outer: Box, inner: Box) -> bool:
        ox, oy, ow, oh = outer
        ix, iy, iw, ih = inner
        outer_area = self._box_area(outer)
        inner_area = self._box_area(inner)
        if inner_area >= outer_area:
            return False
        intersection = self._intersection_area(outer, inner)
        return intersection / max(inner_area, 1) >= 0.9

    def _iou(self, box_a: Box, box_b: Box) -> float:
        intersection = self._intersection_area(box_a, box_b)
        if intersection <= 0:
            return 0.0
        union = self._box_area(box_a) + self._box_area(box_b) - intersection
        return intersection / max(union, 1)

    def _intersection_area(self, box_a: Box, box_b: Box) -> int:
        ax, ay, aw, ah = box_a
        bx, by, bw, bh = box_b
        x1 = max(ax, bx)
        y1 = max(ay, by)
        x2 = min(ax + aw, bx + bw)
        y2 = min(ay + ah, by + bh)
        if x2 <= x1 or y2 <= y1:
            return 0
        return int((x2 - x1) * (y2 - y1))

    def _box_area(self, box: Box) -> int:
        return max(box[2], 0) * max(box[3], 0)

    def _extract_runs(self, mask: np.ndarray) -> list[tuple[int, int]]:
        runs: list[tuple[int, int]] = []
        start: int | None = None
        for index, value in enumerate(mask.tolist()):
            if value and start is None:
                start = index
            elif not value and start is not None:
                runs.append((start, index))
                start = None
        if start is not None:
            runs.append((start, len(mask)))
        return runs

    def _smooth_signal(self, signal: np.ndarray, window_size: int) -> np.ndarray:
        window_size = max(int(window_size), 3)
        if window_size % 2 == 0:
            window_size += 1
        kernel = np.ones(window_size, dtype=np.float32) / window_size
        return np.convolve(signal, kernel, mode="same")
