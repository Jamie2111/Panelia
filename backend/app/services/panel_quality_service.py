from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

from app.core.config import get_settings
from app.schemas.project import PanelBox


class PanelQualityService:
    # Aspect ratio threshold that distinguishes webtoon (tall scroll) from
    # traditional manga / comic pages.  Webtoon pages are typically 2.5× or
    # taller; manga/comic pages are ≤ 1.6× (e.g. 2500 × 1755 ≈ 1.42).
    _WEBTOON_ASPECT_RATIO: float = 2.2

    def analyze(self, project_dir: Path, panels: list[PanelBox]) -> dict[str, Any]:
        settings = get_settings()
        ordered = sorted(panels, key=lambda item: item.order)
        if not ordered:
            return {
                "analysis_version": 2,
                "total_panels": 0,
                "kept_panels": 0,
                "quality_score": 0,
                "should_block_script": True,
                "summary": "Panel quality score 0/100; no detected panels were available; blocked before script generation.",
                "risky_panels": [],
            }

        page_images = self._load_page_images(project_dir)
        is_webtoon = self._detect_webtoon_format(page_images)
        risky_panels: list[dict[str, Any]] = []
        whitespace_count = 0
        full_page_count = 0
        composite_count = 0
        suspicious_skip_count = 0

        for panel in ordered:
            page_image = page_images.get(int(panel.page))
            if page_image is None:
                continue
            page_height, page_width = page_image.shape[:2]
            crop = self._crop_panel(page_image, panel)
            if crop.size == 0:
                risky_panels.append(
                    {
                        "panel_id": panel.id,
                        "panel": panel.panel,
                        "page": panel.page,
                        "reasons": ["empty_crop"],
                    }
                )
                whitespace_count += 1
                continue

            whitespace = self._border_whitespace_ratio(crop)
            blank_stats = self._blank_border_stats(crop)
            full_page_like = panel.width >= page_width * 0.9 and panel.height >= page_height * 0.72
            composite_like = self._looks_like_composite_panel(crop, panel.width, panel.height, page_width, page_height)
            content_score = self._content_score(crop)
            suspicious_skip = (
                (not panel.keep)
                and panel.auto_skipped
                and content_score >= 0.24
                and not (panel.skip_reason or "").startswith(("duplicate", "continuation"))
            )

            # Whitespace threshold: webtoon panels should be fully inked; manga
            # pages have white gutters, wide margins, and full-page splash art
            # whose page-border crops are 40-65% white by construction - use a
            # high threshold so only genuinely blank panels are flagged.
            whitespace_threshold = 0.22 if is_webtoon else 0.65

            reasons: list[str] = []
            if whitespace >= whitespace_threshold:
                whitespace_count += 1
                reasons.append("whitespace")
            if self._looks_like_top_blank_band(panel, blank_stats, page_width, page_height):
                reasons.append("top_blank_band")
            if self._looks_like_side_void(blank_stats):
                reasons.append("side_void")
            corner_wedge = self._looks_like_corner_wedge(blank_stats)
            if corner_wedge:
                reasons.append("corner_wedge")
            # For traditional manga, full-page panels are normal (splash art,
            # chapter openers, color inserts).  Only flag them for webtoon where
            # a full-page box almost always means detection failed.
            if full_page_like and is_webtoon:
                full_page_count += 1
                reasons.append("full_page_like")
            if composite_like:
                composite_count += 1
                reasons.append("composite_like")
            if suspicious_skip:
                suspicious_skip_count += 1
                reasons.append("suspicious_auto_skip")

            if reasons:
                risky_panels.append(
                    {
                        "panel_id": panel.id,
                        "panel": panel.panel,
                        "page": panel.page,
                        "keep": panel.keep,
                        "auto_skipped": panel.auto_skipped,
                        "whitespace_ratio": round(float(whitespace), 3),
                        "border_blank_ratio": round(float(blank_stats["border_blank_ratio"]), 3),
                        "corner_wedge": bool(corner_wedge),
                        "corner_wedge_score": round(float(blank_stats.get("corner_wedge_score", 0.0)), 3),
                        "content_score": round(float(content_score), 3),
                        "reasons": reasons,
                    }
                )

        total_panels = len(ordered)
        kept_panels = sum(1 for panel in ordered if panel.keep)
        quality_score = 100

        if is_webtoon:
            # Webtoon: full-page boxes strongly indicate detection failure.
            quality_score -= round((whitespace_count / max(total_panels, 1)) * 32)
            quality_score -= round((full_page_count / max(total_panels, 1)) * 40)
            quality_score -= round((composite_count / max(total_panels, 1)) * 28)
            quality_score -= round((suspicious_skip_count / max(total_panels, 1)) * 34)
            thresholds = {
                "whitespace": max(3, round(total_panels * 0.08)),
                "full_page_like": max(1, round(total_panels * 0.02)),
                "composite_like": max(2, round(total_panels * 0.05)),
                "suspicious_auto_skip": max(2, round(total_panels * 0.05)),
                "score": settings.panel_quality_score_webtoon,
            }
        else:
            # Traditional manga / comic: full-page splash art is intentional;
            # white gutters are part of the art style - use relaxed scoring.
            quality_score -= round((whitespace_count / max(total_panels, 1)) * 20)
            quality_score -= round((composite_count / max(total_panels, 1)) * 28)
            quality_score -= round((suspicious_skip_count / max(total_panels, 1)) * 34)
            # full_page_count is always 0 here (not flagged for manga), so no
            # deduction needed; the score simply reflects composite / skip issues.
            thresholds = {
                "whitespace": max(3, round(total_panels * 0.70)),
                "full_page_like": total_panels + 1,  # never blocks for manga
                "composite_like": max(2, round(total_panels * 0.05)),
                "suspicious_auto_skip": max(2, round(total_panels * 0.05)),
                "score": settings.panel_quality_score_manga,
            }

        quality_score = max(0, min(100, quality_score))

        should_block_script = any(
            (
                whitespace_count > thresholds["whitespace"],
                full_page_count > thresholds["full_page_like"],
                composite_count > thresholds["composite_like"],
                suspicious_skip_count > thresholds["suspicious_auto_skip"],
                quality_score < thresholds["score"],
            )
        )
        problems: list[str] = []
        if whitespace_count:
            problems.append(f"{whitespace_count} whitespace-heavy")
        if full_page_count:
            problems.append(f"{full_page_count} full-page-like")
        if composite_count:
            problems.append(f"{composite_count} composite-like")
        if suspicious_skip_count:
            problems.append(f"{suspicious_skip_count} suspicious auto-skip")
        if not problems:
            problems.append("no major issues")
        format_label = "webtoon" if is_webtoon else "manga"
        summary = (
            f"Panel quality score {quality_score}/100 across {total_panels} detected panels "
            f"({format_label} format); "
            f"{', '.join(problems)}; "
            f"{'blocked before script generation' if should_block_script else 'safe for script generation'}."
        )
        return {
            "analysis_version": 2,
            "total_panels": total_panels,
            "kept_panels": kept_panels,
            "whitespace_panels": whitespace_count,
            "full_page_like_panels": full_page_count,
            "composite_like_panels": composite_count,
            "suspicious_auto_skips": suspicious_skip_count,
            "quality_score": quality_score,
            "should_block_script": should_block_script,
            "thresholds": thresholds,
            "format": format_label,
            "risky_panels": risky_panels,
            "summary": summary,
        }

    def _detect_webtoon_format(self, page_images: dict[int, np.ndarray]) -> bool:
        """
        Return True if the majority of pages look like webtoon (tall scroll)
        rather than traditional manga / comic pages.

        Webtoon pages have height/width ratio >= 2.2.  Manga pages are
        typically ~1.4 (e.g. 2500 × 1755).  Sampling up to 10 pages is
        enough to classify reliably even for short chapters.
        """
        if not page_images:
            return False
        sample_keys = sorted(page_images)[:10]
        tall_count = 0
        for key in sample_keys:
            img = page_images[key]
            h, w = img.shape[:2]
            if w > 0 and (h / w) >= self._WEBTOON_ASPECT_RATIO:
                tall_count += 1
        return tall_count > len(sample_keys) // 2

    def _load_page_images(self, project_dir: Path) -> dict[int, np.ndarray]:
        pages_dir = project_dir / "pages"
        loaded: dict[int, np.ndarray] = {}
        for index, path in enumerate(sorted(pages_dir.glob("*")), start=1):
            try:
                loaded[index] = np.array(Image.open(path).convert("RGB"))
            except Exception:
                continue
        return loaded

    def _crop_panel(self, image: np.ndarray, panel: PanelBox) -> np.ndarray:
        x = max(int(panel.x), 0)
        y = max(int(panel.y), 0)
        width = max(int(panel.width), 1)
        height = max(int(panel.height), 1)
        return image[y : y + height, x : x + width]

    def _border_whitespace_ratio(self, crop: np.ndarray) -> float:
        if crop.size == 0:
            return 1.0
        grayscale = self._grayscale(crop)
        height, width = grayscale.shape[:2]
        edge_x = max(4, int(width * 0.08))
        edge_y = max(4, int(height * 0.08))
        strips = [
            grayscale[:edge_y, :],
            grayscale[-edge_y:, :],
            grayscale[:, :edge_x],
            grayscale[:, -edge_x:],
        ]
        blank_scores = []
        for strip in strips:
            bright = float(np.mean(strip > 245))
            low_std = float(np.std(strip))
            blank_scores.append(bright if low_std < 20 else max(bright - 0.15, 0.0))
        return float(sum(blank_scores) / len(blank_scores))

    def _looks_like_composite_panel(
        self,
        crop: np.ndarray,
        width: int,
        height: int,
        page_width: int,
        page_height: int,
    ) -> bool:
        if width < page_width * 0.58:
            return False
        if height < max(page_height * 0.34, width * 1.05):
            return False
        grayscale = self._grayscale(crop)
        signal = grayscale < 242
        bright = grayscale > 246
        row_signal = self._smooth(signal.mean(axis=1), max(9, int(height * 0.018)))
        row_bright = self._smooth(bright.mean(axis=1), max(9, int(height * 0.018)))
        gutter_mask = (row_signal < 0.02) & (row_bright > 0.88)
        runs = self._extract_runs(gutter_mask)
        usable = [
            (start, end)
            for start, end in runs
            if end - start >= max(18, int(height * 0.02))
            and start >= int(height * 0.12)
            and end <= int(height * 0.88)
        ]
        return len(usable) >= 1

    def _content_score(self, crop: np.ndarray) -> float:
        grayscale = self._grayscale(crop)
        gy, gx = np.gradient(grayscale.astype(np.float32))
        edge_density = float(np.mean((np.abs(gx) + np.abs(gy)) > 28))
        ink_ratio = float(np.mean(grayscale < 240))
        contrast = float(np.std(grayscale))
        white_ratio = float(np.mean(grayscale > 245))
        return edge_density * 2.0 + ink_ratio + contrast / 120.0 - white_ratio * 0.5

    def _blank_border_stats(self, crop: np.ndarray) -> dict[str, float]:
        grayscale = self._grayscale(crop)
        edges = cv2.Canny(grayscale, 40, 120)
        edge_guard = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
        bright_cutoff = max(238, min(248, int(np.percentile(grayscale, 85))))
        blank_mask = ((grayscale >= bright_cutoff) & (edge_guard == 0)).astype(np.uint8)
        if not np.any(blank_mask):
            return {
                "border_blank_ratio": 0.0,
                "top_band_blank_ratio": 0.0,
                "bottom_band_blank_ratio": 0.0,
                "left_band_blank_ratio": 0.0,
                "right_band_blank_ratio": 0.0,
                "corner_blank_ratio": 0.0,
                "corner_wedge_score": 0.0,
            }

        labels_count, labels = cv2.connectedComponents(blank_mask)
        if labels_count <= 1:
            return {
                "border_blank_ratio": 0.0,
                "top_band_blank_ratio": 0.0,
                "bottom_band_blank_ratio": 0.0,
                "left_band_blank_ratio": 0.0,
                "right_band_blank_ratio": 0.0,
                "corner_blank_ratio": 0.0,
                "corner_wedge_score": 0.0,
            }

        border_labels = set()
        border_labels.update(int(value) for value in labels[0, :])
        border_labels.update(int(value) for value in labels[-1, :])
        border_labels.update(int(value) for value in labels[:, 0])
        border_labels.update(int(value) for value in labels[:, -1])
        border_labels.discard(0)
        if not border_labels:
            border_blank = np.zeros_like(blank_mask, dtype=bool)
        else:
            border_blank = np.isin(labels, list(border_labels))

        height, width = border_blank.shape
        band_y = max(20, int(height * 0.12))
        band_x = max(20, int(width * 0.12))
        corner_blank_ratio = max(
            float(np.mean(border_blank[:band_y, :band_x])),
            float(np.mean(border_blank[:band_y, -band_x:])),
            float(np.mean(border_blank[-band_y:, :band_x])),
            float(np.mean(border_blank[-band_y:, -band_x:])),
        )
        corner_wedge_score = self._corner_wedge_geometry_score(border_blank)
        return {
            "border_blank_ratio": float(np.mean(border_blank)),
            "top_band_blank_ratio": float(np.mean(border_blank[:band_y, :])),
            "bottom_band_blank_ratio": float(np.mean(border_blank[-band_y:, :])),
            "left_band_blank_ratio": float(np.mean(border_blank[:, :band_x])),
            "right_band_blank_ratio": float(np.mean(border_blank[:, -band_x:])),
            "corner_blank_ratio": corner_blank_ratio,
            "corner_wedge_score": corner_wedge_score,
        }

    def _corner_wedge_geometry_score(self, border_blank: np.ndarray) -> float:
        height, width = border_blank.shape[:2]
        size = max(24, min(int(min(width, height) * 0.20), 96))
        if size * 2 > min(width, height):
            size = max(12, min(width, height) // 2)
        if size < 12:
            return 0.0

        def oriented_corner(region: np.ndarray, *, flip_y: bool = False, flip_x: bool = False) -> np.ndarray:
            corner = region
            if flip_y:
                corner = np.flipud(corner)
            if flip_x:
                corner = np.fliplr(corner)
            return corner.astype(np.float32)

        corners = [
            oriented_corner(border_blank[:size, :size]),
            oriented_corner(border_blank[:size, -size:], flip_x=True),
            oriented_corner(border_blank[-size:, :size], flip_y=True),
            oriented_corner(border_blank[-size:, -size:], flip_y=True, flip_x=True),
        ]
        scores: list[float] = []
        edge = max(3, size // 5)
        inner_start = max(edge + 1, int(size * 0.58))
        for corner in corners:
            edge_top = float(np.mean(corner[:edge, :]))
            edge_left = float(np.mean(corner[:, :edge]))
            inner = float(np.mean(corner[inner_start:, inner_start:]))
            fill = float(np.mean(corner))
            diagonal_drop = min(edge_top, edge_left) - inner
            if min(edge_top, edge_left) < 0.70 or fill < 0.24:
                scores.append(0.0)
                continue
            scores.append(max(0.0, diagonal_drop))
        return float(max(scores, default=0.0))

    def _looks_like_top_blank_band(
        self,
        panel: PanelBox,
        blank_stats: dict[str, float],
        page_width: int,
        page_height: int,
    ) -> bool:
        if panel.width < page_width * 0.75:
            return False
        if panel.height > page_height * 0.60:
            return False
        return (
            blank_stats["top_band_blank_ratio"] >= 0.62
            or blank_stats["bottom_band_blank_ratio"] >= 0.62
        )

    def _looks_like_side_void(self, blank_stats: dict[str, float]) -> bool:
        return (
            blank_stats["border_blank_ratio"] >= 0.22
            and max(blank_stats["left_band_blank_ratio"], blank_stats["right_band_blank_ratio"]) >= 0.34
        )

    def _looks_like_corner_wedge(self, blank_stats: dict[str, float]) -> bool:
        return (
            blank_stats["border_blank_ratio"] >= 0.035
            and blank_stats["corner_blank_ratio"] >= 0.30
            and blank_stats.get("corner_wedge_score", 0.0) >= 0.42
        )

    def _grayscale(self, crop: np.ndarray) -> np.ndarray:
        if crop.ndim == 2:
            return crop
        return np.dot(crop[..., :3], [0.299, 0.587, 0.114]).astype(np.uint8)

    def _smooth(self, signal: np.ndarray, window: int) -> np.ndarray:
        window = max(int(window), 3)
        if window % 2 == 0:
            window += 1
        kernel = np.ones(window, dtype=np.float32) / window
        return np.convolve(signal, kernel, mode="same")

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
