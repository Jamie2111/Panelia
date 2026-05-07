import unittest

from app.schemas.project import PanelBox, StorySegment
from app.services.script_quality_service import ScriptQualityService


def _panel(
    panel_id: str,
    order: int,
    *,
    ocr_text: str | None = None,
    text_detected: bool | None = None,
    manual_keep: bool = False,
) -> PanelBox:
    return PanelBox(
        id=panel_id,
        page=1,
        panel=order,
        x=0,
        y=0,
        width=100,
        height=100,
        order=order,
        keep=True,
        ocr_text=ocr_text,
        text_detected=text_detected,
        manual_keep=manual_keep,
    )


class ScriptQualityServiceTests(unittest.TestCase):
    def test_visual_only_blank_lines_do_not_block_tts(self) -> None:
        service = ScriptQualityService()
        panels = [_panel(f"p{i}", i) for i in range(1, 8)]
        script = [
            "Hiro meets Zero Two.",
            "",
            "",
            "Zero Two changes Hiro's future.",
            "",
            "",
            "",
        ]

        report = service.analyze(panels, script)

        self.assertEqual(report["blank_lines"], 5)
        self.assertEqual(report["visual_only_blank_lines"], 5)
        self.assertEqual(report["blocking_blank_lines"], 0)
        self.assertFalse(report["should_block_tts"])

    def test_text_panel_blank_lines_still_block_tts(self) -> None:
        service = ScriptQualityService()
        panels = [
            _panel("p1", 1, ocr_text="Hiro, wait!", text_detected=True),
            _panel("p2", 2, ocr_text="Zero Two arrives.", text_detected=True),
            _panel("p3", 3, ocr_text="Ichigo calls out.", text_detected=True),
            _panel("p4", 4),
        ]
        script = ["", "", "", "Zero Two smiles."]

        report = service.analyze(panels, script)

        self.assertEqual(report["blank_lines"], 3)
        self.assertEqual(report["blocking_blank_lines"], 3)
        self.assertTrue(report["should_block_tts"])

    def test_excessive_visual_only_story_segments_block_tts(self) -> None:
        service = ScriptQualityService()
        segments = [
            StorySegment(
                id=f"scene_{index:03d}",
                order=index,
                text="",
                panel_ids=[f"p{index}a", f"p{index}b"],
                visual_only=True,
                suppression_reason="weak_evidence",
            )
            for index in range(1, 25)
        ]

        report = service.analyze_story_segments(segments)

        self.assertEqual(report["visual_only_blank_lines"], 24)
        self.assertEqual(report["visual_only_panel_refs"], 48)
        self.assertTrue(report["excessive_visual_only"])
        self.assertTrue(report["should_block_tts"])

    def test_generic_bridge_phrases_block_story_segments(self) -> None:
        service = ScriptQualityService()
        segments = [
            StorySegment(
                id="scene_001",
                order=1,
                scene_id=1,
                text="The beat keeps moving because the last choice still has consequences.",
                panel_ids=["p1", "p2"],
            ),
            StorySegment(
                id="scene_002",
                order=2,
                scene_id=2,
                text="John keeps the nearby choice active while the surrounding group reacts.",
                panel_ids=["p3", "p4"],
            ),
            StorySegment(
                id="scene_003",
                order=3,
                scene_id=3,
                text="The dynamic shifts once more, leaving John with few options.",
                panel_ids=["p5", "p6"],
            ),
        ]

        report = service.analyze_story_segments(segments)

        self.assertEqual(report["generic_lines"], 3)
        self.assertIn("short_segments", report["failure_codes"])
        self.assertTrue(report["should_block_tts"])

    def test_near_duplicate_and_scene_regression_block_story_segments(self) -> None:
        service = ScriptQualityService()
        segments = [
            StorySegment(
                id="scene_002_a",
                order=1,
                scene_id=2,
                text="John staggers back from the blow, clutching his arm while the hallway closes around him.",
                panel_ids=["p1"],
            ),
            StorySegment(
                id="scene_002_b",
                order=2,
                scene_id=2,
                text="John staggers back from the attack, holding his injured arm as the hallway tightens around him.",
                panel_ids=["p2"],
            ),
            StorySegment(
                id="scene_001",
                order=3,
                scene_id=1,
                text="The classroom had been quiet before the confrontation pulled everyone into the hall.",
                panel_ids=["p3"],
            ),
        ]

        report = service.analyze_story_segments(segments)

        self.assertGreaterEqual(report["semantic_near_duplicate_lines"], 1)
        self.assertEqual(report["scene_order_regressions"], 1)
        self.assertIn("scene_order_regression", report["failure_codes"])
        self.assertTrue(report["should_block_tts"])

    def test_visual_caption_leakage_blocks_story_segments(self) -> None:
        service = ScriptQualityService()
        segments = [
            StorySegment(
                id="scene_001",
                order=1,
                scene_id=1,
                text="John's face shows extreme shock, his eyes wide as sweat beads on his forehead.",
                panel_ids=["p1", "p2"],
            ),
            StorySegment(
                id="scene_002",
                order=2,
                scene_id=1,
                text="Symbols for a men's and women's restroom appear before the fight continues.",
                panel_ids=["p3"],
            ),
        ]

        report = service.analyze_story_segments(segments)

        self.assertEqual(report["visual_lines"], 2)
        self.assertIn("visual_caption_leakage", report["failure_codes"])
        self.assertTrue(report["should_block_tts"])

    def test_malformed_captiony_youtube_lines_block_story_segments(self) -> None:
        service = ScriptQualityService()
        segments = [
            StorySegment(
                id="scene_001",
                order=1,
                scene_id=1,
                text="Mathematical equations and explanations are presented in speech bubbles, suggesting a lecture is in progress.",
                panel_ids=["p1"],
            ),
            StorySegment(
                id="scene_002",
                order=2,
                scene_id=1,
                text="John sighs as the hallway erupts around him. John blocks are so exhausting.",
                panel_ids=["p2"],
            ),
        ]

        report = service.analyze_story_segments(segments)

        self.assertGreaterEqual(report["visual_lines"], 1)
        self.assertGreaterEqual(report["malformed_lines"], 1)
        self.assertIn("malformed_english", report["failure_codes"])
        self.assertTrue(report["should_block_tts"])


if __name__ == "__main__":
    unittest.main()
