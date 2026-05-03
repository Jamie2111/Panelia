import os
import tempfile
import unittest
from pathlib import Path

from app.core.config import get_settings
from app.schemas.project import ChapterMetadata, PanelBox, SourceType, StorySegment
from app.services.llm_router import RoutedResult
from app.services.project_store import ProjectStore
from app.services.story_segment_repair_service import StorySegmentRepairService
from app.services.story_grounding import build_name_grounding, compact_chapter_metadata
from app.services.story_script_service import StoryScriptService


class StorySegmentPersistenceTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self._previous_data_dir = os.environ.get("PANELIA_DATA_DIR")
        os.environ["PANELIA_DATA_DIR"] = self._tmpdir.name
        get_settings.cache_clear()
        self.store = ProjectStore()

    def tearDown(self) -> None:
        if self._previous_data_dir is None:
            os.environ.pop("PANELIA_DATA_DIR", None)
        else:
            os.environ["PANELIA_DATA_DIR"] = self._previous_data_dir
        get_settings.cache_clear()
        self._tmpdir.cleanup()

    def test_save_story_segments_round_trips_manifest_and_quality(self) -> None:
        project = self.store.create_project("Story Segment Test", SourceType.IMAGES)
        panels = [
            PanelBox(id="p1", page=1, panel=1, x=0, y=0, width=100, height=100, order=1, keep=True),
            PanelBox(id="p2", page=1, panel=2, x=0, y=100, width=100, height=100, order=2, keep=True),
            PanelBox(id="p3", page=1, panel=3, x=0, y=200, width=100, height=100, order=3, keep=True),
        ]
        self.store.save_panels(project.id, panels)

        segments = [
            StorySegment(
                id="scene_001",
                order=1,
                text="Hiro meets Zero Two for the first time.",
                panel_ids=["p1", "p2"],
                panel_start=1,
                panel_end=2,
                scene_id=1,
                title="The encounter",
                representative_panel_id="p1",
            ),
            StorySegment(
                id="scene_002",
                order=2,
                text="The battle pulls them into a dangerous partnership.",
                panel_ids=["p3"],
                panel_start=3,
                panel_end=3,
                scene_id=2,
                title="The partnership",
                representative_panel_id="p3",
            ),
        ]

        self.store.save_story_segments(project.id, segments, story_block="Hiro meets Zero Two.\n\nThey are pulled into battle.")

        loaded_segments = self.store.load_story_segments(project.id)
        self.assertEqual([segment.id for segment in loaded_segments], ["scene_001", "scene_002"])
        self.assertEqual(self.store.load_script(project.id), [
            "Hiro meets Zero Two for the first time.",
            "The battle pulls them into a dangerous partnership.",
        ])
        report = self.store.load_script_quality_report(project.id)
        self.assertEqual(report.get("analysis_mode"), "story_segments_v1")

    def test_invalidate_script_outputs_clears_stale_story_segments_after_panel_edits(self) -> None:
        project = self.store.create_project("Panel Edit Invalidates Story Segments", SourceType.IMAGES)
        panels = [
            PanelBox(id="p1", page=1, panel=1, x=0, y=0, width=100, height=100, order=1, keep=True, narration="Generated line."),
            PanelBox(
                id="p2",
                page=1,
                panel=2,
                x=0,
                y=100,
                width=100,
                height=100,
                order=2,
                keep=True,
                narration="Manual line.",
                narration_locked=True,
                manual_narration=True,
            ),
        ]
        self.store.save_panels(project.id, panels)
        self.store.save_story_segments(
            project.id,
            [
                StorySegment(
                    id="scene_001",
                    order=1,
                    text="Old story segment that no longer matches the panel list.",
                    panel_ids=["p1", "p2"],
                    panel_start=1,
                    panel_end=2,
                    scene_id=1,
                    title="Old segment",
                    representative_panel_id="p1",
                )
            ],
        )

        self.store.invalidate_script_outputs(project.id, clear_generated_panel_narration=True)

        project_dir = self.store._project_dir(project.id)
        self.assertEqual(self.store.load_script(project.id), [])
        self.assertEqual(self.store.load_story_segments(project.id), [])
        self.assertFalse((project_dir / "output" / "story_segments.json").exists())
        self.assertFalse((project_dir / "output" / "narration_story.txt").exists())
        saved_panels = {panel.id: panel for panel in self.store.load_panels(project.id)}
        self.assertIsNone(saved_panels["p1"].narration)
        self.assertEqual(saved_panels["p2"].narration, "Manual line.")

    def test_visual_only_blank_story_segment_is_not_blocking(self) -> None:
        project = self.store.create_project("Visual Only Segment Test", SourceType.IMAGES)
        self.store.save_panels(
            project.id,
            [
                PanelBox(id="p1", page=1, panel=1, x=0, y=0, width=100, height=100, order=1, keep=True),
            ],
        )
        self.store.save_story_segments(
            project.id,
            [
                StorySegment(
                    id="scene_001",
                    order=1,
                    text="",
                    panel_ids=["p1"],
                    panel_start=1,
                    panel_end=1,
                    scene_id=1,
                    title="Silent beat",
                    representative_panel_id="p1",
                    visual_only=True,
                    suppression_reason="weak_evidence",
                )
            ],
            story_block="",
        )
        report = self.store.load_script_quality_report(project.id)
        self.assertEqual(report.get("visual_only_blank_lines"), 1)
        self.assertEqual(report.get("blocking_blank_lines"), 0)
        self.assertFalse(bool(report.get("should_block_tts")))

    def test_incremental_story_repair_fills_visual_only_segment_and_persists_outputs(self) -> None:
        project = self.store.create_project("Incremental Repair Test", SourceType.IMAGES)
        self.store.save_panels(
            project.id,
            [
                PanelBox(
                    id="p1",
                    page=1,
                    panel=1,
                    x=0,
                    y=0,
                    width=100,
                    height=100,
                    order=1,
                    keep=True,
                    ocr_text="Hiro reaches Zero Two beside the lake.",
                    text_detected=True,
                )
            ],
        )
        self.store.save_story_segments(
            project.id,
            [
                StorySegment(
                    id="scene_001",
                    order=1,
                    text="",
                    panel_ids=["p1"],
                    panel_start=1,
                    panel_end=1,
                    scene_id=1,
                    title="Lake encounter",
                    representative_panel_id="p1",
                    visual_only=True,
                    suppression_reason="weak_evidence",
                )
            ],
            story_block="",
        )

        repair = StorySegmentRepairService(
            store=self.store,
            story_service=StoryScriptService(router=_FakeStoryDraftRouter()),
        )
        result = repair.repair_project(project.id)

        loaded_segments = self.store.load_story_segments(project.id)
        self.assertEqual(result.repaired_segments, 1)
        self.assertEqual(loaded_segments[0].text, "Hiro and Zero Two press deeper into the forest.")
        self.assertFalse(loaded_segments[0].visual_only)
        project_dir = self.store._project_dir(project.id)
        self.assertTrue((project_dir / "output" / "story_segments.json").exists())
        self.assertTrue((project_dir / "output" / "narration_story.txt").exists())


class _FakeStoryDraftRouter:
    def available_providers(self) -> list[str]:
        return ["gemini"]

    async def generate_story_segments(self, scenes, context, *, provider=None, scene_image_paths=None):
        self.last_scenes = scenes
        self.last_context = context
        self.last_scene_image_paths = scene_image_paths
        return RoutedResult(
            provider="gemini",
            model="fake",
            payload={
                "segments": [
                    {
                        "segment_id": str(scene.get("segment_id") or f"segment_{index:03d}"),
                        "scene_id": int(scene.get("scene_id") or 0),
                        "title": f"Scene {index}",
                        "text": f"Drafted {scene.get('segment_id') or f'segment_{index:03d}'}.",
                    }
                    for index, scene in enumerate(scenes, start=1)
                ]
            },
        )

    async def repair_story_segments_multimodal(self, segments, context, *, provider=None, scene_image_paths=None):
        self.last_rescue_segments = segments
        self.last_rescue_context = context
        self.last_rescue_image_paths = scene_image_paths
        return RoutedResult(
            provider="gemini",
            model="fake",
            payload={
                "rewrites": [
                    {
                        "index": int(segment.get("index") or 0),
                        "line": str(segment.get("current_line") or "").strip()
                        or "Hiro and Zero Two press deeper into the forest."
                    }
                    for segment in segments
                ]
            },
        )

    async def refine_story_segment_style(self, lines, context, *, provider=None):
        self.last_style_lines = lines
        self.last_style_context = context
        rewrites = []
        for item in lines:
            index = int(item.get("index") or 0)
            current = str(item.get("current_line") or "").strip()
            if "looked around, confused" in current:
                rewrites.append({"index": index, "line": "Hiro struggles to make sense of the chaos around him."})
            else:
                rewrites.append({"index": index, "line": current})
        return RoutedResult(
            provider="gemini",
            model="fake",
            payload={"rewrites": rewrites},
        )


class StoryScriptServiceTests(unittest.TestCase):
    def test_noisy_ocr_detector_keeps_canonical_uppercase_terms(self) -> None:
        service = StoryScriptService(router=_FakeStoryDraftRouter())

        self.assertFalse(service._text_is_noisy_ocr("APE deploys the FRANXX units."))

    def test_fallback_scene_line_can_use_visual_cues(self) -> None:
        service = StoryScriptService(router=_FakeStoryDraftRouter())

        line = service._fallback_scene_line(
            {
                "scene_unit_count": 2,
                "scene_summary": "Too broad for a split scene.",
                "combined_text": "",
                "visual_cues": "Hiro and Zero Two are rushed toward the transport ship.",
            },
            protagonist_name="Hiro",
        )

        self.assertEqual(line, "Hiro and Zero Two are rushed toward the transport ship.")

    def test_draft_scene_lines_accepts_dict_chapter_metadata(self) -> None:
        router = _FakeStoryDraftRouter()
        service = StoryScriptService(router=router)
        story_units = [
            {
                "segment_id": "scene_001_beat_01",
                "scene_id": 1,
                "sequence_in_scene": 1,
                "scene_unit_count": 2,
                "panel_start": 1,
                "panel_end": 2,
                "panel_count": 2,
                "panel_ids": ["p1", "p2"],
                "character_names": ["Hiro"],
                "combined_text": "Hiro wakes up and heads outside.",
                "scene_summary": "Hiro wakes up and moves into the next moment.",
                "visual_cues": "",
            },
            {
                "segment_id": "scene_001_beat_02",
                "scene_id": 1,
                "sequence_in_scene": 2,
                "scene_unit_count": 2,
                "panel_start": 3,
                "panel_end": 4,
                "panel_count": 2,
                "panel_ids": ["p3", "p4"],
                "character_names": ["Hiro", "Zero Two"],
                "combined_text": "He meets Zero Two by the ruined shore.",
                "scene_summary": "Hiro wakes up and moves into the next moment.",
                "visual_cues": "",
            }
        ]

        lines = service._draft_scene_lines(
            story_units,
            project_title="DARLING in the FRANXX",
            chapter_metadata=ChapterMetadata(manga_title="DARLING in the FRANXX", language="en").model_dump(mode="json"),
            chapter_summary="A boy wakes up in a broken world.",
            character_dictionary={"hiro": {"aliases": [], "role": "protagonist"}},
            protagonist_name="Hiro",
        )

        self.assertEqual(lines, ["Drafted scene_001_beat_01.", "Drafted scene_001_beat_02."])
        self.assertEqual(router.last_context.get("chapter_metadata", {}).get("manga_title"), "DARLING in the FRANXX")
        self.assertEqual([scene.get("segment_id") for scene in router.last_scenes], ["scene_001_beat_01", "scene_001_beat_02"])

    def test_chapter_metadata_payload_is_compact_and_keeps_series_grounding(self) -> None:
        metadata = ChapterMetadata(
            manga_title="DARLING in the FRANXX",
            chapter_title="Combined chapters 1-10",
            chapter_number="1-10",
            language="en",
            page_count=316,
            raw={
                "manga": {
                    "title": "DARLING in the FRANXX",
                    "alt_titles": ["ダーリン・イン・ザ・フランキス", "DarliFra"],
                    "synopsis": "Hiro meets Zero Two in a ruined future where children pilot FRANXX.",
                    "slug": "darling-in-the-franxx",
                    "type": "manga",
                    "original_language": "ja",
                }
            },
        )

        payload = compact_chapter_metadata(metadata)

        self.assertEqual(payload.get("manga_title"), "DARLING in the FRANXX")
        self.assertEqual(payload.get("series_slug"), "darling-in-the-franxx")
        self.assertIn("Zero Two", payload.get("series_synopsis", ""))
        self.assertNotIn("raw", payload)

    def test_story_bible_sanitizer_drops_unapproved_names(self) -> None:
        service = StoryScriptService(router=_FakeStoryDraftRouter())
        metadata = {
            "manga_title": "DARLING in the FRANXX",
            "series_synopsis": "Hiro meets Zero Two in a ruined future where children pilot FRANXX.",
            "series_cast_hints": ["Hiro", "Zero Two", "Naomi", "Ichigo"],
            "canonical_name_corrections": [{"variant": "ZeroTwo", "canonical": "Zero Two"}],
        }
        grounding = build_name_grounding(
            metadata,
            {
                "hiro": {"display_name": "Hiro", "aliases": ["016"]},
                "zero two": {"display_name": "Zero Two", "aliases": ["002"]},
            },
            "Hiro",
        )
        fallback = {
            "chapter_premise": "Hiro meets Zero Two in a ruined future.",
            "cast": [{"name": "Hiro", "aliases": []}, {"name": "Zero Two", "aliases": []}],
            "world_terms": ["FRANXX"],
            "continuity_notes": ["Keep Hiro and Zero Two named consistently."],
            "scene_memory": [
                {
                    "scene_id": 1,
                    "state": "Hiro meets Zero Two near the ruined shore.",
                    "location": "Ruined shore",
                    "characters": ["Hiro", "Zero Two"],
                    "open_thread": "Their first encounter changes everything.",
                }
            ],
        }
        generated = {
            "chapter_premise": "Hiro and Nance struggle inside Plantation.",
            "cast": [{"name": "Hiro"}, {"name": "Nance"}, {"name": "Zero Two"}],
            "world_terms": ["FRANXX"],
            "continuity_notes": ["Keep Hiro and Nance named consistently."],
            "scene_memory": [
                {
                    "scene_id": 1,
                    "state": "Hiro and Nance struggle inside Plantation.",
                    "location": "Plantation",
                    "characters": ["Hiro", "Nance"],
                    "open_thread": "Nance pulls Hiro toward a new fight.",
                }
            ],
        }

        sanitized = service._sanitize_story_bible(generated, fallback, grounding)

        self.assertEqual([item.get("name") for item in sanitized.get("cast", [])], ["Hiro", "Zero Two"])
        self.assertEqual(sanitized.get("scene_memory", [])[0].get("characters"), ["Hiro", "Zero Two"])
        self.assertNotIn("Nance", sanitized.get("scene_memory", [])[0].get("state", ""))

    def test_noisy_scene_seed_text_is_suppressed_before_story_generation(self) -> None:
        service = StoryScriptService(router=_FakeStoryDraftRouter())
        grounding = build_name_grounding(
            {"manga_title": "DARLING in the FRANXX", "series_cast_hints": ["Hiro", "Zero Two"]},
            {"hiro": {"display_name": "Hiro"}},
            "Hiro",
        )
        sanitized = service._sanitize_scene_seeds(
            [
                {
                    "scene_id": 1,
                    "panel_ids": ["p1"],
                    "character_names": ["Hiro", "Nance"],
                    "combined_text": "y dash! sorry.. asi can name is dropping. attempt. confirmed. three.. vy tyi positive 27 n.",
                }
            ],
            grounding,
        )
        self.assertEqual(sanitized[0].get("combined_text"), "")
        self.assertEqual(sanitized[0].get("character_names"), ["Hiro"])

    def test_expand_story_units_splits_rich_scene_into_micro_beats(self) -> None:
        service = StoryScriptService(router=_FakeStoryDraftRouter())
        grounding = build_name_grounding(
            {"manga_title": "DARLING in the FRANXX", "series_cast_hints": ["Hiro", "Zero Two"]},
            {
                "hiro": {"display_name": "Hiro"},
                "zero two": {"display_name": "Zero Two"},
            },
            "Hiro",
        )
        story_units = service._expand_story_units(
            [
                {
                    "scene_id": 1,
                    "panel_start": 1,
                    "panel_end": 3,
                    "panel_ids": ["p1", "p2", "p3"],
                    "panels": [1, 2, 3],
                    "combined_text": "Hiro wakes up. He runs outside. He sees Zero Two. They speak at the ruined shore.",
                    "character_names": ["Hiro", "Zero Two"],
                }
            ],
            [
                {"panel_id": "p1", "panel": 1, "page": 1, "text": "Hiro wakes up in a ruined room.", "character_names": ["Hiro"], "visual_caption": ""},
                {"panel_id": "p2", "panel": 2, "page": 1, "text": "He rushes outside after hearing a distant alarm.", "character_names": ["Hiro"], "visual_caption": ""},
                {"panel_id": "p3", "panel": 3, "page": 1, "text": "A horned girl appears by the water and watches him.", "character_names": ["Zero Two"], "visual_caption": ""},
                {"panel_id": "p4", "panel": 4, "page": 2, "text": "Hiro stares at Zero Two while the shore burns behind them.", "character_names": ["Hiro", "Zero Two"], "visual_caption": ""},
            ],
            [{"scene_id": 1, "description": "Hiro wakes up and meets Zero Two by the ruined shore."}],
            grounding,
        )

        self.assertGreaterEqual(len(story_units), 2)
        self.assertEqual(story_units[0].get("segment_id"), "scene_001_beat_01")
        self.assertEqual(story_units[1].get("scene_id"), 1)
        covered_panel_ids = [panel_id for unit in story_units for panel_id in unit.get("panel_ids", []) or []]
        self.assertEqual(covered_panel_ids, ["p1", "p2", "p3", "p4"])
        self.assertEqual(story_units[-1].get("panel_end"), 4)

    def test_sentence_fragment_is_treated_as_low_quality(self) -> None:
        service = StoryScriptService(router=_FakeStoryDraftRouter())
        self.assertTrue(service._line_is_sentence_fragment("A mental connection with your partner."))
        self.assertTrue(service._line_is_low_quality("A mental connection with your partner."))
        self.assertFalse(service._line_is_sentence_fragment("Zero Two explains that partners must connect mentally."))
        self.assertTrue(service._line_is_dialogue_fragment("Who are you?"))
        self.assertTrue(service._line_is_low_quality("It's an opportunity!"))

    def test_multimodal_rescue_reason_flags_vague_or_ocr_echo_lines(self) -> None:
        service = StoryScriptService(router=_FakeStoryDraftRouter())
        unit = {
            "segment_id": "scene_001_beat_05",
            "scene_id": 1,
            "panel_count": 1,
            "combined_text": "Me stay WHO SORRY..",
            "visual_cues": "",
            "character_names": [],
            "scene_summary": "",
        }
        self.assertEqual(service._multimodal_rescue_reason("Me stay WHO SORRY.", unit), "low_quality")
        self.assertEqual(
            service._multimodal_rescue_reason("Hiro asked if someone was there.", {**unit, "combined_text": "Who is there.."}),
            "generic",
        )
        self.assertEqual(
            service._multimodal_rescue_reason("Who are you?", {**unit, "combined_text": "who are you"}),
            "low_quality",
        )

    def test_style_pass_refines_spoken_segments_without_touching_visual_only(self) -> None:
        router = _FakeStoryDraftRouter()
        service = StoryScriptService(router=router)
        units = [
            {
                "segment_id": "scene_001_beat_01",
                "scene_id": 1,
                "panel_count": 2,
                "panel_ids": ["p1", "p2"],
                "combined_text": "Hiro looked around, confused by what he was seeing.",
                "scene_summary": "Hiro faces sudden chaos.",
                "visual_cues": "",
                "character_names": ["Hiro"],
            },
            {
                "segment_id": "scene_001_beat_02",
                "scene_id": 1,
                "panel_count": 1,
                "panel_ids": ["p3"],
                "combined_text": "",
                "scene_summary": "",
                "visual_cues": "",
                "character_names": [],
            },
        ]
        grounding = build_name_grounding(
            {"manga_title": "DARLING in the FRANXX", "series_cast_hints": ["Hiro", "Zero Two"]},
            {"hiro": {"display_name": "Hiro"}},
            "Hiro",
        )
        payloads = [
            {"text": "Hiro looked around, confused by what he was seeing.", "visual_only": False, "suppression_reason": None},
            {"text": "", "visual_only": True, "suppression_reason": "weak_evidence"},
        ]

        styled = service._style_spoken_segment_payloads(
            payloads,
            units,
            project_title="DARLING in the FRANXX",
            chapter_metadata=compact_chapter_metadata({"manga_title": "DARLING in the FRANXX"}),
            chapter_summary="Hiro is thrown into chaos.",
            character_dictionary={"hiro": {"display_name": "Hiro"}},
            story_bible={"chapter_premise": "Hiro is thrown into chaos."},
            name_grounding=grounding,
        )

        self.assertEqual(styled[0].get("text"), "Hiro struggles to make sense of the chaos around him.")
        self.assertEqual(styled[1].get("text"), "")
        self.assertTrue(styled[1].get("visual_only"))

    def test_line_needs_style_refinement_flags_reported_speech_shapes(self) -> None:
        service = StoryScriptService(router=_FakeStoryDraftRouter())
        self.assertTrue(service._line_needs_style_refinement('A voice called out, "Get on!" as the pilots prepared to board.'))
        self.assertTrue(service._line_needs_style_refinement("She reassured him, telling him to relax and trust his partner."))
        self.assertFalse(service._line_needs_style_refinement("Hiro rushes toward the shoreline as Zero Two follows."))

    def test_scene_id_for_missing_group_avoids_collapsing_distant_gaps(self) -> None:
        service = StoryScriptService(router=_FakeStoryDraftRouter())
        raw_units = [
            {"scene_id": 1, "panel_start": 1, "panel_end": 12},
            {"scene_id": 2, "panel_start": 30, "panel_end": 40},
        ]

        inherited = service._scene_id_for_missing_group(
            [{"panel": 13}, {"panel": 14}],
            raw_units,
            3,
        )
        new_scene = service._scene_id_for_missing_group(
            [{"panel": 20}, {"panel": 22}],
            raw_units,
            3,
        )

        self.assertEqual(inherited, 1)
        self.assertEqual(new_scene, 3)

    def test_visual_only_recovery_revives_multi_panel_blanks(self) -> None:
        router = _FakeStoryDraftRouter()
        service = StoryScriptService(router=router)
        units = [
            {
                "segment_id": "scene_001_beat_01",
                "scene_id": 1,
                "panel_count": 3,
                "panel_ids": ["p1", "p2", "p3"],
                "combined_text": "",
                "scene_summary": "Hiro and Zero Two move deeper into the forest.",
                "visual_cues": "",
                "character_names": ["Hiro", "Zero Two"],
            }
        ]
        payloads = [
            {"text": "", "visual_only": True, "suppression_reason": "weak_evidence"},
        ]
        grounding = build_name_grounding(
            {"manga_title": "DARLING in the FRANXX", "series_cast_hints": ["Hiro", "Zero Two"]},
            {
                "hiro": {"display_name": "Hiro"},
                "zero two": {"display_name": "Zero Two"},
            },
            "Hiro",
        )

        recovered = service._recover_visual_only_payloads_multimodal(
            payloads,
            units,
            project_title="DARLING in the FRANXX",
            chapter_metadata=compact_chapter_metadata({"manga_title": "DARLING in the FRANXX"}),
            chapter_summary="Hiro meets Zero Two.",
            character_dictionary={"hiro": {"display_name": "Hiro"}},
            protagonist_name="Hiro",
            story_bible={"chapter_premise": "Hiro meets Zero Two."},
            name_grounding=grounding,
            scene_visual_paths={"scene_001_beat_01": [Path("/tmp/fake-scene.jpg")]},
        )

        self.assertEqual(recovered[0].get("text"), "Hiro and Zero Two press deeper into the forest.")
        self.assertFalse(bool(recovered[0].get("visual_only")))
        self.assertIsNone(recovered[0].get("suppression_reason"))


if __name__ == "__main__":
    unittest.main()
