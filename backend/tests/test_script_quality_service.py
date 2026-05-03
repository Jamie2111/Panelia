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


if __name__ == "__main__":
    unittest.main()
