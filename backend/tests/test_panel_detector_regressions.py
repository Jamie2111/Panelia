from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.pipeline.stages import _should_recover_auto_skipped_panel_with_text
from app.schemas.project import ChapterMetadata, PanelBox
from app.services.panel_detection_service import DetectedPanel, PanelDetector, PanelDetectorAdapter, ReadingOrder
from app.services.panel_reconstruction_engine import PanelReconstructionEngine
from app.services.panel_training_annotations import changed_annotation_pages_for_detector_training


def _panel(
    panel_id: str,
    *,
    page: int = 1,
    x: int = 0,
    y: int = 0,
    width: int = 300,
    height: int = 300,
    keep: bool = True,
    auto_skipped: bool = False,
) -> PanelBox:
    return PanelBox(
        id=panel_id,
        page=page,
        panel=1,
        x=x,
        y=y,
        width=width,
        height=height,
        order=1,
        keep=keep,
        auto_skipped=auto_skipped,
    )


class PanelTrainingAnnotationTests(unittest.TestCase):
    def test_keep_only_toggle_does_not_create_detector_training_change(self) -> None:
        before = [_panel("p1", keep=True)]
        after = [_panel("p1", keep=False)]

        changed = changed_annotation_pages_for_detector_training(before, after)

        self.assertEqual(changed, {})

    def test_non_auto_skipped_panels_train_even_when_excluded_from_recap(self) -> None:
        before = [_panel("p1", keep=True, x=10, y=10, width=250, height=250)]
        after = [
            _panel("p1", keep=True, x=10, y=10, width=250, height=250),
            _panel("p2", keep=False, auto_skipped=False, x=320, y=20, width=220, height=240),
            _panel("p3", keep=False, auto_skipped=True, x=600, y=30, width=120, height=120),
        ]

        changed = changed_annotation_pages_for_detector_training(before, after)

        self.assertEqual(sorted(panel.id for panel in changed[1]), ["p1", "p2"])


class PanelDetectorHeuristicTests(unittest.TestCase):
    def test_suppress_full_page_keeps_traditional_splash_with_inset(self) -> None:
        detector = PanelDetector()
        panels = [
            DetectedPanel(x=20, y=20, width=960, height=1320, source="magi"),
            DetectedPanel(x=140, y=120, width=260, height=210, source="magi"),
        ]

        kept = detector._suppress_full_page(panels, 1000, 1400)

        self.assertEqual(len(kept), 2)
        self.assertTrue(any(panel.width >= 940 for panel in kept))
        self.assertTrue(any(panel.width == 260 and panel.height == 210 for panel in kept))

    def test_boundary_skip_is_conservative_on_traditional_pages(self) -> None:
        adapter = PanelDetectorAdapter()
        panel = DetectedPanel(x=40, y=0, width=900, height=180, source="magi")
        metrics = {
            "content_bbox": (0, 50, 900, 170),
            "content_bbox_area_ratio": 0.45,
            "content_center_y_ratio": 0.65,
            "white_ratio": 0.32,
            "edge_density": 0.03,
        }

        should_skip = adapter._should_auto_skip_top_boundary_panel(
            panel=panel,
            page_w=1000,
            page_h=1400,
            is_webtoon_page=False,
            metrics=metrics,
        )

        self.assertFalse(should_skip)

    def test_cleanup_same_page_panels_preserves_traditional_top_panel(self) -> None:
        adapter = PanelDetectorAdapter()
        with tempfile.TemporaryDirectory() as tmpdir:
            page_path = Path(tmpdir) / "page.png"
            image = Image.new("RGB", (1000, 1400), "white")
            draw = ImageDraw.Draw(image)
            draw.rectangle((50, 0, 950, 260), fill="black")
            draw.rectangle((50, 320, 950, 1120), fill="black")
            image.save(page_path)

            panel_boxes = [
                {
                    "id": "top-panel",
                    "page": 1,
                    "panel": 1,
                    "x": 50,
                    "y": 0,
                    "width": 900,
                    "height": 260,
                    "order": 1,
                    "keep": True,
                    "auto_skipped": False,
                },
                {
                    "id": "main-panel",
                    "page": 1,
                    "panel": 2,
                    "x": 50,
                    "y": 320,
                    "width": 900,
                    "height": 800,
                    "order": 2,
                    "keep": True,
                    "auto_skipped": False,
                },
            ]

            cleaned = adapter._cleanup_same_page_panels(panel_boxes, [page_path])

        top_panel = next(panel for panel in cleaned if panel["id"] == "top-panel")
        self.assertFalse(top_panel.get("auto_skipped", False))
        self.assertTrue(top_panel.get("keep", True))

    def test_cleanup_same_page_panels_preserves_large_traditional_overlap_panel(self) -> None:
        adapter = PanelDetectorAdapter()
        with tempfile.TemporaryDirectory() as tmpdir:
            page_path = Path(tmpdir) / "page.png"
            image = Image.new("RGB", (1000, 1400), "white")
            draw = ImageDraw.Draw(image)
            draw.rectangle((60, 40, 940, 920), fill="black")
            draw.rectangle((80, 760, 920, 1340), fill="black")
            image.save(page_path)

            panel_boxes = [
                {
                    "id": "upper-panel",
                    "page": 1,
                    "panel": 1,
                    "x": 60,
                    "y": 40,
                    "width": 880,
                    "height": 880,
                    "order": 1,
                    "keep": True,
                    "auto_skipped": False,
                },
                {
                    "id": "lower-panel",
                    "page": 1,
                    "panel": 2,
                    "x": 80,
                    "y": 760,
                    "width": 840,
                    "height": 580,
                    "order": 2,
                    "keep": True,
                    "auto_skipped": False,
                },
            ]

            cleaned = adapter._cleanup_same_page_panels(panel_boxes, [page_path])

        lower_panel = next(panel for panel in cleaned if panel["id"] == "lower-panel")
        self.assertFalse(lower_panel.get("auto_skipped", False))
        self.assertTrue(lower_panel.get("keep", True))

    def test_cleanup_same_page_panels_still_skips_shallow_overlap_strip(self) -> None:
        adapter = PanelDetectorAdapter()
        with tempfile.TemporaryDirectory() as tmpdir:
            page_path = Path(tmpdir) / "page.png"
            image = Image.new("RGB", (1000, 1400), "white")
            draw = ImageDraw.Draw(image)
            draw.rectangle((60, 120, 940, 1180), fill="black")
            draw.rectangle((60, 1040, 940, 1260), fill="white", outline="black", width=8)
            image.save(page_path)

            panel_boxes = [
                {
                    "id": "main-panel",
                    "page": 1,
                    "panel": 1,
                    "x": 60,
                    "y": 120,
                    "width": 880,
                    "height": 1060,
                    "order": 1,
                    "keep": True,
                    "auto_skipped": False,
                },
                {
                    "id": "strip-panel",
                    "page": 1,
                    "panel": 2,
                    "x": 60,
                    "y": 1040,
                    "width": 880,
                    "height": 220,
                    "order": 2,
                    "keep": True,
                    "auto_skipped": False,
                },
            ]

            cleaned = adapter._cleanup_same_page_panels(panel_boxes, [page_path])

        strip_panel = next(panel for panel in cleaned if panel["id"] == "strip-panel")
        self.assertTrue(strip_panel.get("auto_skipped", False))
        self.assertFalse(strip_panel.get("keep", True))
        self.assertIn(
            strip_panel.get("skip_reason"),
            {
                "overlapping strip from same-page panel split",
                "speech-bubble strip below larger panel",
            },
        )

    def test_cleanup_same_page_panels_skips_floating_speech_bubble_fragment(self) -> None:
        adapter = PanelDetectorAdapter()
        adapter._last_character_review_page_payloads = {
            1: {
                "characters": [
                    {"bbox": [80, 80, 260, 260]},
                    {"bbox": [460, 760, 300, 260]},
                ]
            }
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            page_path = Path(tmpdir) / "page.png"
            image = Image.new("RGB", (900, 1200), "white")
            draw = ImageDraw.Draw(image)
            draw.rectangle((60, 60, 420, 420), fill=(70, 120, 150), outline="black", width=6)
            draw.rectangle((420, 720, 820, 1040), fill=(80, 130, 160), outline="black", width=6)
            draw.ellipse((190, 500, 330, 660), fill=(245, 245, 220), outline="black", width=4)
            draw.line((300, 645, 430, 760), fill="black", width=3)
            image.save(page_path)

            panel_boxes = [
                {
                    "id": "upper-art",
                    "page": 1,
                    "panel": 1,
                    "x": 60,
                    "y": 60,
                    "width": 360,
                    "height": 360,
                    "order": 1,
                    "keep": True,
                    "auto_skipped": False,
                    "reconstruction_source": "detector",
                },
                {
                    "id": "bubble-only",
                    "page": 1,
                    "panel": 2,
                    "x": 180,
                    "y": 490,
                    "width": 170,
                    "height": 190,
                    "order": 2,
                    "keep": True,
                    "auto_skipped": False,
                    "reconstruction_source": "detector+ocr_cluster",
                },
                {
                    "id": "lower-art",
                    "page": 1,
                    "panel": 3,
                    "x": 420,
                    "y": 720,
                    "width": 400,
                    "height": 320,
                    "order": 3,
                    "keep": True,
                    "auto_skipped": False,
                    "reconstruction_source": "detector",
                },
            ]

            cleaned = adapter._cleanup_same_page_panels(panel_boxes, [page_path])

        bubble = next(panel for panel in cleaned if panel["id"] == "bubble-only")
        self.assertTrue(bubble.get("auto_skipped", False))
        self.assertFalse(bubble.get("keep", True))
        self.assertEqual(bubble.get("skip_reason"), "floating speech-bubble/text fragment")

    def test_cleanup_same_page_panels_preserves_large_text_title_panel(self) -> None:
        adapter = PanelDetectorAdapter()
        adapter._last_character_review_page_payloads = {1: {"characters": []}}
        with tempfile.TemporaryDirectory() as tmpdir:
            page_path = Path(tmpdir) / "page.png"
            image = Image.new("RGB", (900, 1200), "white")
            draw = ImageDraw.Draw(image)
            draw.rectangle((90, 140, 810, 470), fill=(248, 248, 238), outline="black", width=5)
            draw.line((180, 240, 720, 240), fill="black", width=4)
            draw.line((220, 310, 680, 310), fill="black", width=4)
            image.save(page_path)

            panel_boxes = [
                {
                    "id": "title-card",
                    "page": 1,
                    "panel": 1,
                    "x": 90,
                    "y": 140,
                    "width": 720,
                    "height": 330,
                    "order": 1,
                    "keep": True,
                    "auto_skipped": False,
                    "reconstruction_source": "detector+ocr_cluster",
                }
            ]

            cleaned = adapter._cleanup_same_page_panels(panel_boxes, [page_path])

        title = next(panel for panel in cleaned if panel["id"] == "title-card")
        self.assertFalse(title.get("auto_skipped", False))
        self.assertTrue(title.get("keep", True))

    def test_split_composite_panels_splits_squarish_staggered_art_boxes(self) -> None:
        detector = PanelDetector(reading_order=ReadingOrder.WESTERN)
        image = np.full((1200, 900, 3), 255, dtype=np.uint8)

        # Two real bordered art panels are staggered diagonally, with a speech
        # bubble in the gutter. The old splitter skipped this because the
        # merged box was not tall enough relative to its width.
        cv2.rectangle(image, (80, 130), (560, 560), (20, 20, 20), 5)
        cv2.rectangle(image, (95, 145), (545, 545), (90, 155, 180), -1)
        cv2.circle(image, (650, 220), 95, (238, 238, 230), -1)
        cv2.ellipse(image, (645, 220), (90, 95), 0, 0, 360, (35, 35, 35), 4)
        cv2.rectangle(image, (450, 620), (830, 1060), (20, 20, 20), 5)
        cv2.rectangle(image, (465, 635), (815, 1045), (95, 165, 185), -1)

        merged = DetectedPanel(
            x=55,
            y=105,
            width=800,
            height=980,
            confidence=0.75,
            source="cv",
        )

        split = detector._split_composite_panels(image, [merged], page_w=900, page_h=1200)

        self.assertEqual(len(split), 2)
        self.assertLess(split[0].width, merged.width)
        self.assertLess(split[1].width, merged.width)
        self.assertLess(split[0].height, merged.height)
        self.assertLess(split[1].height, merged.height)
        self.assertLess(split[0].y, split[1].y)

    def test_traditional_manga_prefers_magi_layout_over_full_page_cv(self) -> None:
        detector = PanelDetector(reading_order=ReadingOrder.MANGA)
        detector._detect_trained_model = lambda image: []
        detector._detect_magi = lambda image: (
            [
                DetectedPanel(x=40, y=40, width=420, height=520, source="magi"),
                DetectedPanel(x=520, y=40, width=420, height=520, source="magi"),
                DetectedPanel(x=60, y=650, width=860, height=650, source="magi"),
            ],
            [],
            [],
        )
        detector._detect_cv = lambda image: [
            DetectedPanel(x=0, y=0, width=1000, height=1400, source="cv"),
        ]
        detector._filter_junk = lambda image, panels, character_boxes, page_w, page_h: panels
        detector._deduplicate = lambda panels: panels
        detector._suppress_full_page = lambda panels, page_w, page_h: panels
        detector._split_composite_panels = lambda image, panels, page_w, page_h: panels

        result = detector.detect_panels(
            np.full((1400, 1000, 3), 255, dtype=np.uint8),
            page_index=1,
            reading_order=ReadingOrder.MANGA,
        )

        self.assertEqual(len(result.panels), 3)
        self.assertEqual(result.source, "magi")

    def test_traditional_manga_can_fallback_when_magi_only_returns_full_page(self) -> None:
        detector = PanelDetector(reading_order=ReadingOrder.MANGA)
        detector._detect_trained_model = lambda image: []
        detector._detect_magi = lambda image: (
            [DetectedPanel(x=0, y=0, width=1000, height=1400, source="magi")],
            [],
            [],
        )
        detector._detect_cv = lambda image: [
            DetectedPanel(x=40, y=40, width=430, height=600, source="cv"),
            DetectedPanel(x=520, y=40, width=430, height=600, source="cv"),
            DetectedPanel(x=80, y=700, width=840, height=620, source="cv"),
        ]
        detector._filter_junk = lambda image, panels, character_boxes, page_w, page_h: panels
        detector._deduplicate = lambda panels: panels
        detector._suppress_full_page = lambda panels, page_w, page_h: panels
        detector._split_composite_panels = lambda image, panels, page_w, page_h: panels

        result = detector.detect_panels(
            np.full((1400, 1000, 3), 255, dtype=np.uint8),
            page_index=1,
            reading_order=ReadingOrder.MANGA,
        )

        self.assertEqual(len(result.panels), 3)
        self.assertEqual(result.source, "cv")


class ScriptPreparationRecoveryTests(unittest.TestCase):
    def test_recovery_blocks_speech_bubble_strip(self) -> None:
        panel = _panel(
            "p1",
            width=900,
            height=180,
            keep=False,
            auto_skipped=True,
        ).model_copy(update={"skip_reason": "speech-bubble strip below larger panel"})

        self.assertFalse(_should_recover_auto_skipped_panel_with_text(panel, (1000, 1400)))

    def test_recovery_blocks_shallow_overlap_strip(self) -> None:
        panel = _panel(
            "p2",
            width=920,
            height=230,
            keep=False,
            auto_skipped=True,
        ).model_copy(update={"skip_reason": "overlapping strip from same-page panel split"})

        self.assertFalse(_should_recover_auto_skipped_panel_with_text(panel, (1000, 1400)))

    def test_recovery_allows_large_overlap_panel(self) -> None:
        panel = _panel(
            "p3",
            width=900,
            height=560,
            keep=False,
            auto_skipped=True,
        ).model_copy(update={"skip_reason": "overlapping strip from same-page panel split"})

        self.assertTrue(_should_recover_auto_skipped_panel_with_text(panel, (1000, 1400)))


class PanelReconstructionEngineTests(unittest.TestCase):
    def test_reconstruction_uses_detector_text_boxes_without_full_page_ocr(self) -> None:
        class FailingOCR:
            def extract(self, *_args, **_kwargs):  # pragma: no cover - failure path is the assertion
                raise AssertionError("full-page OCR should not run when detector text boxes are available")

        engine = PanelReconstructionEngine(ocr_service=FailingOCR())
        with tempfile.TemporaryDirectory() as tmpdir:
            page_path = Path(tmpdir) / "page.png"
            image = Image.new("RGB", (600, 900), "white")
            draw = ImageDraw.Draw(image)
            draw.rectangle((40, 40, 420, 430), fill=(70, 130, 160), outline="black", width=5)
            draw.ellipse((380, 60, 530, 220), fill=(245, 245, 235), outline="black", width=4)
            image.save(page_path)

            panels = [
                _panel(
                    "art-panel",
                    x=40,
                    y=40,
                    width=380,
                    height=390,
                )
            ]

            reconstructed, report = engine.reconstruct(
                [page_path],
                panels,
                metadata=ChapterMetadata(language="en"),
                detector_text_boxes_by_page={1: [(390, 80, 120, 100)]},
            )

        self.assertEqual(report["pages"][0]["ocr_source"], "detector_text_boxes")
        self.assertEqual(len(reconstructed), 1)
        self.assertEqual(reconstructed[0].reconstruction_source, "detector+ocr_cluster")


if __name__ == "__main__":
    unittest.main()
