from __future__ import annotations

import unittest
from types import SimpleNamespace

import numpy as np

from app.services.comic_ocr_service import ComicOCRService
from app.services.dialogue_pipeline import DialogueExtractionPipeline, DialogueRegion, OCRCandidate
from app.services.multilingual_ocr import MultilingualOCRService, OCRFragment
from app.services.ocr_cleaner import clean_ocr_lines, clean_ocr_text, is_usable_ocr_text


class OCRQualityTests(unittest.TestCase):
    def test_punctuation_only_fragments_are_not_usable_ocr(self) -> None:
        self.assertFalse(is_usable_ocr_text("!!"))
        self.assertFalse(is_usable_ocr_text(".."))
        self.assertEqual(clean_ocr_lines(["!!", "..", "What happened?"]), ["What happened?"])

    def test_clean_ocr_text_preserves_legitimate_all_caps_words(self) -> None:
        self.assertEqual(clean_ocr_text("TASKED TO"), "Tasked to")
        self.assertEqual(clean_ocr_text("PROTECT HUMANITY"), "Protect humanity")
        self.assertEqual(clean_ocr_text("ABCD1234"), "")

    def test_comic_ocr_prefers_meaningful_text_over_punctuation_noise(self) -> None:
        service = ComicOCRService()

        self.assertTrue(service._is_better_candidate("What happened here?", 0.68, "!!", 0.99))
        self.assertFalse(service._is_better_candidate("!!", 0.99, "What happened here?", 0.68))

    def test_dialogue_pipeline_ignores_noise_only_page_ocr_boxes(self) -> None:
        pipeline = DialogueExtractionPipeline()

        meaningful = pipeline._meaningful_page_ocr_boxes(
            [
                {"text": "!!"},
                {"text": ".."},
                {"text": "What happened here?"},
            ]
        )

        self.assertEqual(len(meaningful), 1)
        self.assertEqual(meaningful[0]["text"], "What happened here?")

    def test_dialogue_pipeline_requires_substantial_page_ocr_signal(self) -> None:
        pipeline = DialogueExtractionPipeline()

        self.assertFalse(pipeline._page_ocr_has_substantial_signal([{"text": "WHAT"}]))
        self.assertTrue(
            pipeline._page_ocr_has_substantial_signal(
                [
                    {"text": "What happened here?"},
                ]
            )
        )

    def test_dialogue_pipeline_distinguishes_weak_and_strong_candidate_sets(self) -> None:
        pipeline = DialogueExtractionPipeline()

        self.assertFalse(
            pipeline._candidate_set_has_substantial_signal(
                [
                    OCRCandidate(bbox=[0, 0, 10, 10], text="what", confidence=0.98),
                ]
            )
        )
        self.assertTrue(
            pipeline._candidate_set_has_substantial_signal(
                [
                    OCRCandidate(bbox=[0, 0, 10, 10], text="what happened here", confidence=0.62),
                ]
            )
        )

    def test_dialogue_pipeline_distinguishes_weak_and_strong_scene_regions(self) -> None:
        pipeline = DialogueExtractionPipeline()

        self.assertFalse(
            pipeline._scene_regions_have_substantial_signal(
                [
                    DialogueRegion(
                        page=1,
                        panel=1,
                        panel_order=1,
                        bbox=[0, 0, 10, 10],
                        language="en",
                        text_original="what",
                        text_english="what",
                    )
                ]
            )
        )
        self.assertTrue(
            pipeline._scene_regions_have_substantial_signal(
                [
                    DialogueRegion(
                        page=1,
                        panel=1,
                        panel_order=1,
                        bbox=[0, 0, 10, 10],
                        language="en",
                        text_original="what happened here",
                        text_english="what happened here",
                    )
                ]
            )
        )

    def test_comic_ocr_keeps_overlapping_non_duplicate_lines(self) -> None:
        service = ComicOCRService()

        candidates = service._dedupe_candidates(
            [
                {
                    "bbox": [10, 10, 120, 28],
                    "text": "AND THEN",
                    "confidence": 0.99,
                    "ocr_engine": "apple-vision",
                },
                {
                    "bbox": [8, 8, 126, 34],
                    "text": "THERE WERE",
                    "confidence": 0.98,
                    "ocr_engine": "apple-vision",
                },
            ],
            400,
            400,
        )

        self.assertEqual(len(candidates), 2)

    def test_multilingual_ocr_keeps_overlapping_non_duplicate_lines(self) -> None:
        service = MultilingualOCRService()

        deduped = service._dedupe(
            [
                OCRFragment(bbox=[10, 10, 120, 28], text="Tasked to", confidence=0.99),
                OCRFragment(bbox=[8, 8, 126, 34], text="Protect humanity", confidence=0.98),
            ]
        )

        self.assertEqual(len(deduped), 2)

    def test_multilingual_ocr_accepts_meaningful_caption_fragments(self) -> None:
        service = MultilingualOCRService()

        self.assertTrue(service._is_usable_fragment("Protect"))
        self.assertTrue(service._is_usable_fragment("Tasked to"))
        self.assertFalse(service._is_usable_fragment(".."))

    def test_dialogue_pipeline_preserves_high_contrast_caption_panels(self) -> None:
        pipeline = DialogueExtractionPipeline()

        panel = SimpleNamespace(manual_ocr_text=False, ocr_text="")
        triage = {
            "mode": "full",
            "white_ratio": 0.02,
            "edge_density": 0.031,
            "contrast": 38.0,
        }

        self.assertFalse(
            pipeline._should_skip_expensive_panel_ocr(
                panel,
                panel_image=np.zeros((1400, 900, 3), dtype="uint8"),
                triage=triage,
                layout_scan_available=True,
                has_page_text_in_panel=False,
                has_magi_text_in_panel=False,
            )
        )
        self.assertFalse(
            pipeline._should_trust_empty_page_ocr_for_panel(
                panel_image=np.zeros((1400, 900, 3), dtype="uint8"),
                triage=triage,
                page_ocr_boxes=[],
            )
        )


if __name__ == "__main__":
    unittest.main()
