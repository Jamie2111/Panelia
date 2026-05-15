from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import numpy as np

from app.pipeline.stages import _merge_panel_evidence_records, _transcript_evidence_records, _transcript_lines_by_panel_order
from app.schemas.project import CanonicalCharacterRecord
from app.services.comic_ocr_service import ComicOCRService
from app.services.dialogue_pipeline import DialogueExtractionPipeline, DialogueRegion, OCRCandidate
from app.services.multilingual_ocr import MultilingualOCRService, OCRFragment
from app.services.ocr_cleaner import classify_ocr_text, clean_ocr_fragment_payloads, clean_ocr_lines, clean_ocr_text, is_usable_ocr_text
from app.services.panel_vision_extractor import PanelVisionExtractor


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

    def test_ocr_cleanup_classifies_sfx_and_garbage_before_script(self) -> None:
        self.assertEqual(classify_ocr_text("BAM"), "sfx")
        self.assertEqual(classify_ocr_text("ABCD1234"), "ocr_garbage")
        payloads = clean_ocr_fragment_payloads(
            [
                {"text": "BAM", "bbox": [0, 0, 20, 20], "confidence": 0.95},
                {"text": "Papa is watching.", "bbox": [0, 30, 120, 30], "confidence": 0.91},
            ]
        )

        self.assertEqual(len(payloads), 2)
        self.assertFalse(payloads[0]["usable_for_script"])
        self.assertTrue(payloads[1]["usable_for_script"])
        self.assertIn("raw_text", payloads[1])
        self.assertIn("cleaned_text", payloads[1])
        self.assertIn("classification", payloads[1])
        self.assertIn("reading_order_index", payloads[1])

    def test_english_ocr_rejects_mixed_language_garbage(self) -> None:
        text = "... Enna. ・・・・みがみ.... Kage.... S. E. Midori."

        self.assertEqual(classify_ocr_text(text, expected_language="en"), "foreign_text")
        payloads = clean_ocr_fragment_payloads(
            [{"text": text, "bbox": [5, 5, 200, 40], "confidence": 0.82}],
            expected_language="en",
        )

        self.assertEqual(payloads[0]["category"], "foreign_text")
        self.assertFalse(payloads[0]["usable_for_script"])
        self.assertEqual(payloads[0]["rejection_reason"], "foreign_text")

    def test_english_ocr_rejects_uppercase_and_single_letter_noise(self) -> None:
        self.assertEqual(classify_ocr_text("QZXVBNM"), "ocr_garbage")
        self.assertEqual(classify_ocr_text("S. E. Midori."), "ocr_garbage")
        self.assertEqual(classify_ocr_text("Midori."), "low_confidence")
        self.assertEqual(classify_ocr_text("Sorry."), "dialogue")
        self.assertEqual(classify_ocr_text("フレンフレレン move Kagami Midori.", expected_language="en"), "foreign_text")
        self.assertEqual(classify_ocr_text("・・・・・・・・みがみ....", expected_language="en"), "foreign_text")

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

    def test_fragmented_dense_ocr_triggers_bubble_recall(self) -> None:
        pipeline = DialogueExtractionPipeline()
        image = np.zeros((900, 700, 3), dtype="uint8")
        candidates = [
            OCRCandidate(bbox=[10, 10 + index * 30, 80, 24], text=text, confidence=0.9)
            for index, text in enumerate(["OCEAN.", "WON'T", "YOU", "SETTLE", "FOR", "A", "SHOWER?", "NONE."])
        ]

        self.assertTrue(pipeline._should_run_bubble_recall(image, candidates))

    def test_many_detected_regions_with_tiny_text_fails_coverage(self) -> None:
        pipeline = DialogueExtractionPipeline()
        image = np.zeros((520, 620, 3), dtype="uint8")
        image[30:180, 30:240] = 255
        image[40:190, 350:590] = 255
        image[260:450, 180:480] = 255
        image[80:88, 70:190] = 0
        image[92:100, 70:180] = 0
        image[90:98, 390:550] = 0
        image[320:328, 230:430] = 0

        coverage = pipeline._panel_ocr_coverage(
            image,
            [
                DialogueRegion(
                    page=1,
                    panel=1,
                    panel_order=1,
                    bbox=[0, 0, 10, 10],
                    language="en",
                    text_original="OCEAN.",
                    text_english="OCEAN.",
                )
            ],
            "ltr",
        )

        self.assertGreaterEqual(coverage["expected_text_region_count"], 3)
        self.assertEqual(coverage["accepted_text_region_count"], 1)
        self.assertTrue(coverage["coverage_failure"])
        self.assertTrue(pipeline._coverage_should_trigger_region_recall(coverage))

    def test_region_recall_coverage_improvement_is_preferred(self) -> None:
        pipeline = DialogueExtractionPipeline()
        image = np.zeros((520, 620, 3), dtype="uint8")
        image[30:180, 30:240] = 255
        image[40:190, 350:590] = 255
        image[260:450, 180:480] = 255
        for y, x1, x2 in [(80, 70, 190), (92, 70, 180), (90, 390, 550), (320, 230, 430)]:
            image[y:y + 8, x1:x2] = 0

        before = pipeline._panel_ocr_coverage(
            image,
            [
                DialogueRegion(
                    page=1,
                    panel=1,
                    panel_order=1,
                    bbox=[0, 0, 10, 10],
                    language="en",
                    text_original="OCEAN.",
                    text_english="OCEAN.",
                )
            ],
            "ltr",
        )
        after = pipeline._panel_ocr_coverage(
            image,
            [
                DialogueRegion(
                    page=1,
                    panel=1,
                    panel_order=1,
                    bbox=[30, 30, 210, 150],
                    language="en",
                    text_original="I wanna take a dip in a clear ocean.",
                    text_english="I wanna take a dip in a clear ocean.",
                ),
                DialogueRegion(
                    page=1,
                    panel=1,
                    panel_order=1,
                    bbox=[350, 40, 240, 150],
                    language="en",
                    text_original="Does Plantation Thirteen have one?",
                    text_english="Does Plantation Thirteen have one?",
                ),
                DialogueRegion(
                    page=1,
                    panel=1,
                    panel_order=1,
                    bbox=[180, 260, 300, 190],
                    language="en",
                    text_original="Won't you settle for a shower?",
                    text_english="Won't you settle for a shower?",
                ),
            ],
            "ltr",
        )

        self.assertTrue(before["coverage_failure"])
        self.assertFalse(after["coverage_failure"])
        self.assertTrue(pipeline._coverage_is_better(after, before))

    def test_layout_bubble_candidates_merge_partial_apple_with_region_ocr(self) -> None:
        pipeline = DialogueExtractionPipeline()
        image = np.zeros((420, 620, 3), dtype="uint8")
        image[40:260, 40:260] = 255
        image[50:270, 350:590] = 255
        image[100:108, 80:220] = 0
        image[130:138, 80:220] = 0
        image[120:128, 390:550] = 0
        pipeline._extract_apple_vision_bubble_candidates = lambda *_args, **_kwargs: [  # type: ignore[method-assign]
            OCRCandidate(bbox=[40, 40, 220, 220], text="ocean.", confidence=0.98, detector="apple-vision-bubble", ocr_engine="apple-vision")
        ]
        pipeline._extract_speech_bubble_candidates = lambda *_args, **_kwargs: [  # type: ignore[method-assign]
            OCRCandidate(
                bbox=[350, 50, 240, 220],
                text="does plantation thirteen have one?",
                confidence=0.82,
                detector="speech-bubble-region",
                ocr_engine="paddleocr",
            )
        ]

        candidates = pipeline._extract_layout_bubble_candidates(image, "en", "ltr")
        texts = [candidate.text for candidate in candidates]

        self.assertIn("ocean.", texts)
        self.assertIn("does plantation thirteen have one?", texts)

    def test_speech_bubble_candidates_reconstruct_line_broken_dialogue(self) -> None:
        pipeline = DialogueExtractionPipeline()
        image = np.zeros((420, 520, 3), dtype="uint8")
        image[40:330, 40:230] = 255
        image[60:350, 300:490] = 255
        image[120:128, 80:190] = 0
        image[150:158, 80:190] = 0
        image[170:178, 340:450] = 0
        image[200:208, 340:450] = 0
        seen = {"count": 0}

        def fake_ocr(region_image: np.ndarray, language_hint: str, **_kwargs):
            seen["count"] += 1
            if seen["count"] == 2:
                return "IS\nWHAT?", 0.81, "test-ocr"
            return "I WANNA\nTAKE A\nDIP IN A\nCLEAR\nOCEAN.", 0.82, "test-ocr"

        pipeline._ocr_region = fake_ocr  # type: ignore[method-assign]
        candidates = pipeline._extract_speech_bubble_candidates(image, "en", "ltr", limit=6)
        texts = [candidate.text for candidate in candidates]

        self.assertTrue(any("i wanna take a dip in a clear ocean" in text for text in texts))
        self.assertTrue(any("is what?" in text for text in texts))

    def test_apple_vision_lines_group_into_bubble_dialogue(self) -> None:
        pipeline = DialogueExtractionPipeline()
        image = np.zeros((420, 520, 3), dtype="uint8")
        image[40:330, 40:230] = 255
        image[60:350, 300:490] = 255
        image[120:128, 80:190] = 0
        image[150:158, 80:190] = 0
        image[170:178, 340:450] = 0
        image[200:208, 340:450] = 0
        calls = {"count": 0}

        def recognize(_image: np.ndarray, _language: str):
            calls["count"] += 1
            if calls["count"] == 1:
                return "I WANNA\nTAKE A DIP", 0.98
            return "IS\nWHAT?", 0.98

        pipeline._comic_ocr.apple_vision_ocr = SimpleNamespace(
            is_available=lambda: True,
            recognize=recognize,
        )

        candidates = pipeline._extract_apple_vision_bubble_candidates(image, "en", "ltr")
        texts = [candidate.text for candidate in candidates]

        self.assertTrue(any("i wanna take a dip" in text for text in texts))
        self.assertTrue(any("is what?" in text for text in texts))

    def test_comic_ocr_does_not_use_apple_vision_as_full_panel_replacement(self) -> None:
        service = ComicOCRService()
        image = np.zeros((240, 240, 3), dtype="uint8")
        service._extract_paddle_fragments = lambda *_args, **_kwargs: []  # type: ignore[method-assign]
        service.multilingual_ocr = SimpleNamespace(extract=lambda _image: [])  # type: ignore[assignment]
        service.apple_vision_ocr = SimpleNamespace(
            is_available=lambda: True,
            extract=lambda *_args, **_kwargs: [
                SimpleNamespace(bbox=[10, 10, 100, 40], text="FULL PAGE APPLE", confidence=0.99)
            ],
            recognize=lambda *_args, **_kwargs: ("FULL PAGE APPLE", 0.99),
        )

        candidates = service.detect_candidates(image, "en")
        text, _confidence, engine = service.recognize_panel_text(image, "en")

        self.assertEqual(candidates, [])
        self.assertEqual(text, "")
        self.assertNotEqual(engine, "apple-vision")

    def test_dialogue_candidate_dedupe_prefers_whole_bubble_text(self) -> None:
        pipeline = DialogueExtractionPipeline()
        deduped = pipeline._dedupe_candidates(
            [
                OCRCandidate(
                    bbox=[10, 10, 250, 180],
                    text="i wanna take a dip in a clear ocean. does plantation thirteen have one?",
                    confidence=0.82,
                    detector="speech-bubble-region",
                ),
                OCRCandidate(bbox=[80, 70, 80, 30], text="ocean.", confidence=0.99, detector="hybrid-comic-ocr"),
            ],
            500,
            500,
        )

        self.assertEqual(len(deduped), 1)
        self.assertEqual(deduped[0].detector, "speech-bubble-region")

    def test_empty_apple_vision_does_not_block_layout_region_recall(self) -> None:
        pipeline = DialogueExtractionPipeline()
        image = np.zeros((520, 620, 3), dtype="uint8")
        expected = OCRCandidate(
            bbox=[40, 40, 220, 180],
            text="is this okay with you?",
            confidence=0.83,
            detector="speech-bubble-region",
            ocr_engine="paddleocr",
        )

        pipeline._extract_apple_line_group_candidates = lambda *_args, **_kwargs: []  # type: ignore[method-assign]
        pipeline._detect_dialogue_region_boxes = lambda *_args, **_kwargs: [(40, 40, 220, 180)]  # type: ignore[method-assign]
        pipeline._extract_layout_bubble_candidates = lambda *_args, **_kwargs: [expected]  # type: ignore[method-assign]
        pipeline._sort_candidates = lambda candidates, _mode: list(candidates)  # type: ignore[method-assign]

        candidates = pipeline._extract_panel_candidates(image, "en", "ltr")

        self.assertEqual(candidates, [expected])

    def test_line_grouping_keeps_separate_nearby_bubbles_apart(self) -> None:
        pipeline = DialogueExtractionPipeline()
        candidates = [
            OCRCandidate(bbox=[410, 120, 95, 24], text="Hiro's", confidence=0.96),
            OCRCandidate(bbox=[405, 155, 110, 24], text="going to", confidence=0.96),
            OCRCandidate(bbox=[405, 190, 120, 24], text="leave?", confidence=0.96),
            OCRCandidate(bbox=[95, 540, 140, 24], text="Code 556", confidence=0.96),
            OCRCandidate(bbox=[92, 575, 115, 24], text="Kokoro", confidence=0.96),
        ]

        grouped = pipeline._group_line_candidates_into_bubbles(candidates, 700, 900, "manga", "en")
        grouped_texts = [candidate.text for candidate in grouped if candidate.detector == "line-group"]

        self.assertTrue(any("hiro's going to leave" in text for text in grouped_texts))
        self.assertFalse(any("hiro's" in text and "code 556" in text for text in grouped_texts))
        self.assertTrue(any(candidate.text == "Code 556" for candidate in grouped))
        self.assertTrue(any(candidate.text == "Kokoro" for candidate in grouped))

    def test_transcript_backfill_uses_panel_order_when_scene_id_is_missing(self) -> None:
        with TemporaryDirectory() as tmp:
            project_dir = Path(tmp)
            output_dir = project_dir / "output"
            output_dir.mkdir(parents=True)
            (output_dir / "transcript.json").write_text(
                """
                {
                  "fragments": [
                    {"panel_order": 64, "accepted": true, "repaired_text": "Dreaming of the day they can spread their wings and fly."},
                    {"panel_order": 64, "accepted": true, "text": "Dreaming of the day they can spread their wings and fly."},
                    {"panel_order": 65, "accepted": false, "text": "CHIIRP"}
                  ]
                }
                """,
                encoding="utf-8",
            )

            lines = _transcript_lines_by_panel_order(project_dir)

        self.assertEqual(lines[64], ["Dreaming of the day they can spread their wings and fly."])
        self.assertNotIn(65, lines)

    def test_transcript_fragments_become_story_evidence_records(self) -> None:
        with TemporaryDirectory() as tmp:
            project_dir = Path(tmp)
            output_dir = project_dir / "output"
            output_dir.mkdir(parents=True)
            (output_dir / "transcript.json").write_text(
                """
                {
                  "fragments": [
                    {"panel_id": "p47", "panel_order": 47, "accepted": true, "repaired_text": "Partner killer?"},
                    {"panel_id": "p47", "panel_order": 47, "accepted": true, "repaired_text": "Any parasite who rides with her will get his blood sucked out!"}
                  ]
                }
                """,
                encoding="utf-8",
            )

            records = _transcript_evidence_records(project_dir)

        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["panel_order"], 47)
        self.assertIn("Partner killer?", records[0]["dialogue_text"])
        self.assertIn("blood sucked out", records[0]["dialogue_text"])

    def test_transcript_evidence_merges_with_panel_evidence(self) -> None:
        merged = _merge_panel_evidence_records(
            [{"panel_id": "p1", "panel_order": 1, "dialogue_text": "Existing line.", "regions": []}],
            [{"panel_id": "p1", "panel_order": 1, "dialogue_text": "Recovered bubble.", "regions": [{"bbox": [1, 2, 3, 4]}]}],
        )

        self.assertEqual(len(merged), 1)
        self.assertIn("Existing line.", merged[0]["dialogue_text"])
        self.assertIn("Recovered bubble.", merged[0]["dialogue_text"])
        self.assertEqual(len(merged[0]["regions"]), 1)

    def test_manga_reading_order_groups_row_jitter_right_to_left(self) -> None:
        pipeline = DialogueExtractionPipeline()
        candidates = pipeline._sort_candidates(
            [
                OCRCandidate(bbox=[100, 85, 120, 80], text="left", confidence=0.9),
                OCRCandidate(bbox=[420, 97, 120, 80], text="right", confidence=0.9),
                OCRCandidate(bbox=[260, 280, 120, 80], text="lower", confidence=0.9),
            ],
            "manga",
        )

        self.assertEqual([candidate.text for candidate in candidates], ["right", "left", "lower"])

    def test_weak_existing_panel_ocr_does_not_bypass_bubble_recall(self) -> None:
        pipeline = DialogueExtractionPipeline()

        self.assertFalse(
            pipeline._existing_panel_text_is_strong(
                "malc 2 male 4 enle nnet. indeedi! and male oul /cdcc. wvnciv inc positive axn. uus 1 ive and laeae aen g. as one.."
            )
        )
        self.assertFalse(pipeline._existing_panel_text_is_strong("jalr ate it?!"))
        self.assertTrue(
            pipeline._existing_panel_text_is_strong(
                "i wanna take a dip in a clear ocean. does plantation thirteen have one?"
            )
        )

    def test_page_ocr_backfill_requires_plausible_dialogue_signal(self) -> None:
        pipeline = DialogueExtractionPipeline()

        self.assertFalse(pipeline._page_ocr_has_substantial_signal([{"text": "geous.. is truly"}]))
        self.assertFalse(pipeline._page_ocr_has_substantial_signal([{"text": "who u."}]))
        self.assertTrue(
            pipeline._page_ocr_has_substantial_signal(
                [{"text": "think of it as an old man's whims."}]
            )
        )

    def test_panel_vision_mentions_do_not_become_visible_names(self) -> None:
        extractor = PanelVisionExtractor()
        characters = [
            CanonicalCharacterRecord(stable_id="john", name="John"),
            CanonicalCharacterRecord(stable_id="papa", name="Papa"),
        ]
        alias_map = extractor._build_alias_map(characters)

        names = extractor._resolve_character_names(
            action_beat="John freezes in the hallway.",
            dialogue="Papa is watching.",
            caption="",
            speaker="unknown",
            canonical_characters=characters,
            explicit_names=[],
            alias_map=alias_map,
        )
        roles = extractor._resolve_character_roles(
            raw_roles={},
            action_beat="John freezes in the hallway.",
            dialogue="Papa is watching.",
            caption="",
            speaker="unknown",
            visible_names=names,
            canonical_characters=characters,
            alias_map=alias_map,
        )

        self.assertIn("John", names)
        self.assertNotIn("Papa", names)
        self.assertEqual(roles["Papa"], ["mentioned_absent"])


if __name__ == "__main__":
    unittest.main()
