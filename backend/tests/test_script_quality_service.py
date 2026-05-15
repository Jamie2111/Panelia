import json
import unittest
from pathlib import Path

from app.schemas.project import PanelBox, StorySegment
from app.services.script_quality_service import ScriptQualityService
from app.services.story_script_service import StoryScriptService


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

    def test_caption_like_run_and_unclear_pronouns_are_reported(self) -> None:
        service = ScriptQualityService()
        segments = [
            StorySegment(
                id="scene_001",
                order=1,
                scene_id=1,
                text="John walks into the classroom.",
                panel_ids=["p1"],
            ),
            StorySegment(
                id="scene_002",
                order=2,
                scene_id=1,
                text="He sees the teacher.",
                panel_ids=["p2"],
            ),
            StorySegment(
                id="scene_003",
                order=3,
                scene_id=1,
                text="He asks a question.",
                panel_ids=["p3"],
            ),
            StorySegment(
                id="scene_004",
                order=4,
                scene_id=1,
                text="He waits for an answer.",
                panel_ids=["p4"],
            ),
        ]

        report = service.analyze_story_segments(segments)

        self.assertGreaterEqual(report["caption_like_lines"], 3)
        self.assertGreaterEqual(report["unclear_pronoun_lines"], 2)
        self.assertGreaterEqual(report["max_one_sentence_run"], 3)
        self.assertIn("caption_like_segments", report["failure_codes"])
        self.assertIn("one_sentence_run", report["failure_codes"])
        self.assertTrue(report["should_block_tts"])

    def test_flashback_and_ability_ambiguity_are_reported(self) -> None:
        service = ScriptQualityService()
        segments = [
            StorySegment(
                id="scene_001",
                order=1,
                scene_id=1,
                text="The flashback begins as he remembers her.",
                panel_ids=["p1"],
            ),
            StorySegment(
                id="scene_002",
                order=2,
                scene_id=1,
                text="His ability awakens, and everyone braces.",
                panel_ids=["p2"],
            ),
        ]

        report = service.analyze_story_segments(segments)

        self.assertEqual(report["flashback_confusion_lines"], 1)
        self.assertEqual(report["ability_ambiguity_lines"], 1)
        self.assertIn("flashback_confusion", report["failure_codes"])
        self.assertTrue(report["should_block_tts"])

    def test_ocr_garbage_and_dialogue_fragment_name_block_story_segments(self) -> None:
        service = ScriptQualityService()
        segments = [
            StorySegment(
                id="scene_001",
                order=1,
                scene_id=1,
                text="GWIRRR GARRR CODE CODc 016 Uenn AND CYYC 7 azlp You will Soon be TRANS PORTED Back to GARDEN.",
                panel_ids=["p1"],
            ),
            StorySegment(
                id="scene_002",
                order=2,
                scene_id=1,
                text="Other says break it before Hiro reacts.",
                panel_ids=["p2"],
            ),
        ]

        report = service.analyze_story_segments(segments)

        self.assertGreaterEqual(report["ocr_contamination_lines"], 1)
        self.assertGreaterEqual(report["ocr_garbage_leak_lines"], 1)
        self.assertGreaterEqual(report["invalid_name_lines"], 1)
        self.assertIn("ocr_garbage_leakage", report["failure_codes"])
        self.assertIn("invalid_names", report["failure_codes"])
        self.assertTrue(report["should_block_tts"])

    def test_mentioned_absent_character_cannot_act_in_scene(self) -> None:
        service = ScriptQualityService()
        segments = [
            StorySegment(
                id="scene_001",
                order=1,
                text="Papa watches John from the doorway, making the threat feel immediate.",
                panel_ids=["p1"],
            )
        ]
        panel_vision = [
            {
                "panel_id": "p1",
                "character_roles": {"Papa": ["mentioned_absent"], "John": ["visible_present"]},
                "confidence": 0.9,
            }
        ]

        report = service.analyze_story_segments(segments, panel_vision_records=panel_vision)

        self.assertEqual(report["mentioned_as_present_errors"], 1)
        self.assertIn("mentioned_as_present", report["failure_codes"])

    def test_important_mentioned_character_is_not_placed_in_vehicle_without_presence(self) -> None:
        service = ScriptQualityService()
        segments = [
            StorySegment(
                id="scene_001",
                order=1,
                text="Hiro rides inside the machine with Zero Two as the battle begins.",
                panel_ids=["p1"],
            )
        ]
        panel_vision = [
            {
                "panel_id": "p1",
                "character_roles": {"Hiro": ["mentioned_absent"], "Zero Two": ["visible_present", "speaker"]},
                "confidence": 0.92,
            }
        ]

        report = service.analyze_story_segments(segments, panel_vision_records=panel_vision)

        self.assertGreaterEqual(report["mentioned_as_present_errors"], 1)
        self.assertIn("mentioned_as_present", report["failure_codes"])

    def test_final_script_flags_invalid_names_and_ocr_garbage(self) -> None:
        service = ScriptQualityService()
        segments = [
            StorySegment(
                id="scene_001",
                order=1,
                text="Other says break it before John reacts.",
                panel_ids=["p1"],
            )
        ]

        report = service.analyze_story_segments(segments)

        self.assertGreaterEqual(report["invalid_name_lines"], 1)
        self.assertEqual(report["ocr_garbage_leak_lines"], 1)
        self.assertIn("invalid_names", report["failure_codes"])

    def test_invalid_name_checker_ignores_sentence_starters_and_world_terms(self) -> None:
        service = ScriptQualityService()
        segments = [
            StorySegment(
                id="scene_001",
                order=1,
                scene_id=1,
                text=(
                    "In a future where humanity survives inside fortress cities, FRANXX pilots face Klaxosaurs. "
                    "Their choices decide whether the evacuation can continue."
                ),
                panel_ids=["p1"],
            )
        ]

        report = service.analyze_story_segments(segments)

        self.assertEqual(report["invalid_name_lines"], 0)
        self.assertNotIn("invalid_names", report["failure_codes"])

    def test_ordinary_ability_phrase_is_not_power_system_ambiguity(self) -> None:
        service = ScriptQualityService()
        segments = [
            StorySegment(
                id="scene_001",
                order=1,
                scene_id=1,
                text=(
                    "Zero Two's ability to move quickly leaves the group scrambling for an explanation. "
                    "The sudden departure changes the direction of the search."
                ),
                panel_ids=["p1"],
            )
        ]

        report = service.analyze_story_segments(segments)

        self.assertEqual(report["ability_ambiguity_lines"], 0)
        self.assertNotIn("ability_ambiguity", report["failure_codes"])

    def test_bad_story_script_scores_below_sixty(self) -> None:
        service = ScriptQualityService()
        segments = [
            StorySegment(
                id="scene_003",
                order=1,
                scene_id=3,
                panel_start=30,
                panel_end=32,
                text="John walks into the classroom.",
                panel_ids=["p30"],
            ),
            StorySegment(
                id="scene_001",
                order=2,
                scene_id=1,
                panel_start=10,
                panel_end=12,
                text="GWIRRR GARRR CODE CODc 016 Uenn AND CYYC 7 azlp You will Soon be TRANS PORTED Back to GARDEN.",
                panel_ids=["p10"],
            ),
            StorySegment(
                id="scene_002",
                order=3,
                scene_id=2,
                panel_start=13,
                panel_end=14,
                text="Other says break it before John reacts.",
                panel_ids=["p13"],
            ),
            StorySegment(
                id="scene_004",
                order=4,
                scene_id=4,
                panel_start=15,
                panel_end=16,
                text="The danger of their situation was palpable, amplifying his internal conflict. This realization underscores the gravity of the circumstances and the emotional turmoil John is experiencing as he confronts the challenges ahead.",
                panel_ids=["p15"],
            ),
            StorySegment(
                id="scene_005",
                order=5,
                scene_id=5,
                panel_start=17,
                panel_end=18,
                text="John walks into the classroom.",
                panel_ids=["p17"],
            ),
        ]

        report = service.analyze_story_segments(segments)

        self.assertLess(report["quality_score"], 60)
        self.assertIn("filler_meta_language", report["failure_codes"])
        self.assertIn("panel_order_regression", report["failure_codes"])
        self.assertTrue(report["should_block_tts"])

    def test_good_story_script_scores_above_ninety(self) -> None:
        service = ScriptQualityService()
        segments = [
            StorySegment(
                id="scene_001",
                order=1,
                scene_id=1,
                panel_start=1,
                panel_end=4,
                text="John tries to disappear into the back of the classroom while the teacher searches for a volunteer. When another student mocks the lesson, the room turns tense because everyone understands how quickly jokes become punishments here.",
                panel_ids=["p1", "p2"],
            ),
            StorySegment(
                id="scene_002",
                order=2,
                scene_id=2,
                panel_start=5,
                panel_end=9,
                text="The hallway fight pulls John out of that numb routine. A wounded student collapses nearby, and John finally has to choose whether staying invisible is worth letting the bullying continue.",
                panel_ids=["p5", "p6"],
            ),
            StorySegment(
                id="scene_003",
                order=3,
                scene_id=3,
                panel_start=10,
                panel_end=14,
                text="John steps in with one clean punch, turning the bully's performance into a warning. The fight is not over, but the balance changes because John has stopped pretending the violence has nothing to do with him.",
                panel_ids=["p10", "p11"],
            ),
        ]

        report = service.analyze_story_segments(segments)

        self.assertGreaterEqual(report["quality_score"], 90)
        self.assertFalse(report["should_block_tts"])

    def test_good_prose_that_skips_thirty_percent_of_panels_blocks_tts(self) -> None:
        service = ScriptQualityService()
        panels = [_panel(f"p{i}", i) for i in range(1, 11)]
        segments = [
            StorySegment(
                id="scene_001",
                order=1,
                scene_id=1,
                panel_start=1,
                panel_end=4,
                text="John tries to keep quiet while the classroom pressure slowly turns toward him. The teasing matters because every laugh makes it harder for him to stay invisible.",
                panel_ids=["p1", "p2", "p3", "p4"],
            ),
            StorySegment(
                id="scene_002",
                order=2,
                scene_id=2,
                panel_start=5,
                panel_end=7,
                text="The conflict follows him outside, where another student becomes the target. John realizes the problem will not stop just because he refuses to look at it.",
                panel_ids=["p5", "p6", "p7"],
            ),
        ]

        report = service.analyze_story_segments(segments, panels=panels)

        self.assertLess(report["quality_score"], 90)
        self.assertTrue(report["should_block_tts"])
        self.assertIn("insufficient_panel_coverage", report["failure_codes"])
        self.assertEqual(report["panel_coverage"]["panels_used_in_narration"], 7)

    def test_good_prose_out_of_panel_order_blocks_tts(self) -> None:
        service = ScriptQualityService()
        panels = [_panel(f"p{i}", i) for i in range(1, 9)]
        segments = [
            StorySegment(
                id="scene_002",
                order=1,
                scene_id=2,
                panel_start=5,
                panel_end=8,
                text="John finally steps into the hallway fight, and the bully loses control of the room. The choice matters because silence would have left the weaker student alone.",
                panel_ids=["p5", "p6", "p7", "p8"],
            ),
            StorySegment(
                id="scene_001",
                order=2,
                scene_id=1,
                panel_start=1,
                panel_end=4,
                text="Earlier, John only wants to survive class without attention. The teacher's question turns ordinary embarrassment into the pressure that follows him outside.",
                panel_ids=["p1", "p2", "p3", "p4"],
            ),
        ]

        report = service.analyze_story_segments(segments, panels=panels)

        self.assertLess(report["quality_score"], 90)
        self.assertTrue(report["should_block_tts"])
        self.assertIn("panel_order_regression", report["failure_codes"])
        self.assertGreaterEqual(report["panel_coverage"]["out_of_order_panel_references"], 1)

    def test_overlapping_panel_ranges_block_duplicate_events(self) -> None:
        service = ScriptQualityService()
        panels = [_panel(f"p{i}", i) for i in range(1, 9)]
        segments = [
            StorySegment(
                id="scene_001",
                order=1,
                scene_id=1,
                panel_start=1,
                panel_end=5,
                text="John tries to stay out of the fight until the bully corners a weaker student. The pressure finally makes his silence impossible.",
                panel_ids=["p1", "p2", "p3", "p4", "p5"],
            ),
            StorySegment(
                id="scene_001_repeat",
                order=2,
                scene_id=1,
                panel_start=3,
                panel_end=7,
                text="The bully corners the same weaker student, and John again realizes that doing nothing would only make the cruelty worse.",
                panel_ids=["p3", "p4", "p5", "p6", "p7"],
            ),
            StorySegment(
                id="scene_002",
                order=3,
                scene_id=2,
                panel_start=8,
                panel_end=8,
                text="When John moves, the hallway's balance finally changes.",
                panel_ids=["p8"],
            ),
        ]

        report = service.analyze_story_segments(segments, panels=panels)

        self.assertLess(report["quality_score"], 90)
        self.assertTrue(report["should_block_tts"])
        self.assertIn("duplicated_panel_ranges", report["failure_codes"])
        self.assertGreater(report["duplicated_panel_count"], 0)

    def test_late_worldbuilding_context_blocks_random_intro_at_end(self) -> None:
        service = ScriptQualityService()
        panels = [_panel(f"p{i}", i) for i in range(1, 11)]
        segments = [
            StorySegment(
                id="scene_001",
                order=1,
                scene_id=1,
                panel_start=1,
                panel_end=5,
                text="John keeps his attention low as the classroom starts turning on him. The first laugh matters because it invites everyone else to join in.",
                panel_ids=["p1", "p2", "p3", "p4", "p5"],
            ),
            StorySegment(
                id="scene_002",
                order=2,
                scene_id=2,
                panel_start=6,
                panel_end=8,
                text="The hallway fight forces him into the open. Once the bully corners someone weaker, John has to decide what kind of invisibility he is protecting.",
                panel_ids=["p6", "p7", "p8"],
            ),
            StorySegment(
                id="scene_003",
                order=3,
                scene_id=3,
                panel_start=9,
                panel_end=10,
                text="In this world, school hierarchy decides who gets protected and who becomes prey.",
                panel_ids=["p9", "p10"],
            ),
        ]

        report = service.analyze_story_segments(segments, panels=panels)

        self.assertLess(report["quality_score"], 90)
        self.assertTrue(report["should_block_tts"])
        self.assertIn("late_worldbuilding_context", report["failure_codes"])

    def test_assigned_but_unused_panel_evidence_blocks_quality(self) -> None:
        service = ScriptQualityService()
        panels = [_panel(f"p{i}", i) for i in range(1, 5)]
        evidence = [
            {"panel_id": "p1", "panel_order": 1, "dialogue_text": "Hiro calls for Zero Two."},
            {"panel_id": "p2", "panel_order": 2, "dialogue_text": "The enemy mech closes in."},
            {"panel_id": "p3", "panel_order": 3, "dialogue_text": "Ichigo warns everyone to move."},
            {"panel_id": "p4", "panel_order": 4, "dialogue_text": "The cockpit starts shaking."},
        ]
        segments = [
            StorySegment(
                id="scene_001",
                order=1,
                scene_id=1,
                panel_start=1,
                panel_end=4,
                text="The situation grows more serious as the group faces another layer of tension.",
                panel_ids=["p1", "p2", "p3", "p4"],
            )
        ]

        report = service.analyze_story_segments(segments, panels=panels, panel_evidence_records=evidence)

        self.assertLess(report["quality_score"], 90)
        self.assertTrue(report["should_block_tts"])
        self.assertIn("unused_meaningful_panels", report["failure_codes"])
        self.assertGreater(report["scene_usage"]["unused_meaningful_panel_count"], 0)

    def test_supporting_panels_must_contribute_or_be_marked_low_information(self) -> None:
        service = ScriptQualityService()
        panels = [_panel(f"p{i}", i) for i in range(1, 4)]
        evidence = [
            {"panel_id": "p1", "panel_order": 1, "dialogue_text": "Hiro calls for Zero Two."},
            {"panel_id": "p2", "panel_order": 2, "dialogue_text": ""},
            {"panel_id": "p3", "panel_order": 3, "dialogue_text": "Zero Two answers Hiro."},
        ]
        segments = [
            StorySegment(
                id="scene_001",
                order=1,
                scene_id=1,
                panel_start=1,
                panel_end=3,
                representative_panel_id="p1",
                text="Hiro calls for Zero Two, and Zero Two answers him before the scene moves on.",
                panel_ids=["p1", "p2", "p3"],
            )
        ]

        report = service.analyze_story_segments(segments, panels=panels, panel_evidence_records=evidence)
        scene = report["scene_usage"]["scenes"][0]

        self.assertIn("p2", scene["low_information_panel_ids"])
        self.assertIn("p3", scene["meaningfully_used_panel_ids"])
        self.assertEqual(scene["unused_meaningful_panel_ids"], [])

    def test_visual_only_panel_gets_compact_summary_before_low_information(self) -> None:
        service = ScriptQualityService()
        panels = [_panel("p1", 1, text_detected=False), _panel("p2", 2, text_detected=False)]
        segments = [
            StorySegment(
                id="scene_001",
                order=1,
                scene_id=1,
                panel_start=1,
                panel_end=2,
                text="John pauses before the confrontation continues.",
                panel_ids=["p1", "p2"],
            )
        ]

        report = service.analyze_story_segments(segments, panels=panels, panel_evidence_records=[])
        scene = report["scene_usage"]["scenes"][0]

        self.assertEqual(scene["unused_meaningful_panel_ids"], [])
        self.assertIn("p1", scene["low_information_panel_ids"])
        self.assertEqual(scene["panel_contribution_map"]["p1"]["visual_summary_source"], "generated_compact")
        self.assertIn("Visual-only support panel", scene["panel_contribution_map"]["p1"]["compact_visual_summary"])

    def test_out_of_order_and_distant_scene_grouping_is_flagged(self) -> None:
        service = ScriptQualityService()
        panels = [
            _panel("p1", 1),
            _panel("p2", 2),
            _panel("p40", 40),
        ]
        panels[0].page = 18
        panels[1].page = 9
        panels[2].page = 45
        segments = [
            StorySegment(
                id="scene_001",
                order=1,
                scene_id=1,
                text="Hiro tries to understand why Zero Two vanished while the facility panic continues.",
                panel_ids=["p1", "p2", "p40"],
            )
        ]

        report = service.analyze_story_segments(segments, panels=panels)

        self.assertLess(report["quality_score"], 90)
        self.assertTrue(report["should_block_tts"])
        self.assertIn("suspicious_panel_grouping", report["failure_codes"])

    def test_action_scene_without_concrete_action_is_blocked(self) -> None:
        service = ScriptQualityService()
        panels = [_panel("p1", 1), _panel("p2", 2)]
        evidence = [
            {"panel_id": "p1", "panel_order": 1, "dialogue_text": "Enemy mechs are surrounding us."},
            {"panel_id": "p2", "panel_order": 2, "dialogue_text": "The cockpit shakes as Hiro calls out."},
        ]
        segments = [
            StorySegment(
                id="scene_001",
                order=1,
                scene_id=1,
                text="Faced with an overwhelming number of new enemy mechs, the group is forced to make immediate decisions.",
                panel_ids=["p1", "p2"],
            )
        ]

        report = service.analyze_story_segments(segments, panels=panels, panel_evidence_records=evidence)

        self.assertLess(report["quality_score"], 90)
        self.assertTrue(report["should_block_tts"])
        self.assertIn("action_without_concrete_action", report["failure_codes"])

    def test_technical_coverage_cannot_pass_when_meaningful_usage_is_poor(self) -> None:
        service = ScriptQualityService()
        panels = [_panel(f"p{i}", i) for i in range(1, 7)]
        evidence = [
            {"panel_id": f"p{i}", "panel_order": i, "dialogue_text": f"Specific dialogue beat {i} changes the scene."}
            for i in range(1, 7)
        ]
        segments = [
            StorySegment(
                id="scene_001",
                order=1,
                scene_id=1,
                panel_start=1,
                panel_end=6,
                text="The group deals with a tense situation and prepares for what comes next.",
                panel_ids=[f"p{i}" for i in range(1, 7)],
            )
        ]

        report = service.analyze_story_segments(segments, panels=panels, panel_evidence_records=evidence)

        self.assertEqual(report["panel_coverage"]["coverage_percent"], 100.0)
        self.assertLess(report["meaningful_panel_usage_rate"], 0.9)
        self.assertLess(report["quality_score"], 90)
        self.assertTrue(report["should_block_tts"])

    def test_deterministic_repair_template_language_blocks_quality(self) -> None:
        service = ScriptQualityService()
        segments = [
            StorySegment(
                id="scene_001",
                order=1,
                text="The exchange gives a concrete detail: Hiro is leaving. The beat ends with a clear consequence for the group.",
                panel_ids=["p1", "p2"],
            ),
            StorySegment(
                id="scene_002",
                order=2,
                text="The immediate result pushes the group into the next choice, and that response carries directly into the following action.",
                panel_ids=["p3", "p4"],
            ),
        ]

        report = service.analyze_story_segments(segments, panels=[_panel(f"p{i}", i) for i in range(1, 5)])

        self.assertGreaterEqual(report["generic_lines"], 2)
        self.assertLess(report["quality_score"], 90)
        self.assertTrue(report["should_block_tts"])

    def test_coverage_dialogue_recap_combines_bubbles_into_story_exchange(self) -> None:
        service = StoryScriptService()

        text = service._coverage_dialogue_recap(
            [
                "Partner killer?",
                "Yeah, so I hear.",
                "Any parasite who rides with her will get his blood sucked out!",
            ],
            "The group",
        )

        self.assertIn("exchange starts", text)
        self.assertIn("Partner killer?", text)
        self.assertIn("turns serious", text)
        self.assertNotIn("Someone says", text)
        self.assertNotIn("The beat ends", text)

    def test_script_quality_regression_fixture_set(self) -> None:
        service = ScriptQualityService()
        fixture_path = Path(__file__).parent / "fixtures" / "script_quality_cases.json"
        cases = json.loads(fixture_path.read_text(encoding="utf-8"))

        for case in cases:
            with self.subTest(case=case["name"]):
                panels = [_panel(f"p{i}", i) for i in range(1, int(case["panel_count"]) + 1)]
                segments = [StorySegment(**segment) for segment in case["segments"]]
                report = service.analyze_story_segments(segments, panels=panels)
                if "expected_max_score" in case:
                    self.assertLessEqual(report["quality_score"], int(case["expected_max_score"]))
                if "expected_min_score" in case:
                    self.assertGreaterEqual(report["quality_score"], int(case["expected_min_score"]))
                self.assertEqual(report["should_block_tts"], bool(case["expected_blocked"]))


if __name__ == "__main__":
    unittest.main()
