from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

from app.core.config import get_settings
from app.schemas.project import ChapterMetadata, PanelBox, SourceType
from app.services.project_store import ProjectStore
from app.services.script_generation_vnext import ScriptGenerationVNextService, ScriptVNextRedraftConfig


class FakeRedraftClient:
    def __init__(self, *, fail_first: bool = False) -> None:
        self.fail_first = fail_first
        self.calls: list[str] = []

    def redraft(self, *, prompt: str, config: ScriptVNextRedraftConfig) -> dict:
        self.calls.append(prompt)
        if self.fail_first and len(self.calls) == 1:
            raise RuntimeError("Gemini blocked at prompt level")
        scene_ids = []
        for marker in prompt.split('"scene_id": "')[1:]:
            scene_ids.append(marker.split('"', 1)[0])
        rewrites = [
            {
                "scene_id": scene_id,
                "text": (
                    "Rin Vale raises his hand and asks, Can I use the bathroom, trying to slip away from the class pressure. "
                    "The classroom lesson at the board keeps the moment grounded while his nervous reaction turns a simple request into the next conflict."
                ),
            }
            for scene_id in dict.fromkeys(scene_ids)
        ]
        return {"provider": "gemini", "model": "fake-gemini", "payload": {"rewrites": rewrites}}


class ScriptGenerationVNextTests(unittest.TestCase):
    # These names are synthetic fixtures. Project-specific names must come from
    # runtime artifacts in production, not from vNext code or prompts.

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

    def _project_with_panels(self) -> tuple[str, Path, list[PanelBox]]:
        project = self.store.create_project("vNext Script Test", SourceType.IMAGES)
        project_dir = Path(self._tmpdir.name) / "projects" / project.id
        panels = [
            PanelBox(id="p1", page=1, panel=1, x=0, y=0, width=300, height=260, order=1, keep=True, ocr_text="Class starts with a math lesson."),
            PanelBox(id="p2", page=1, panel=2, x=0, y=260, width=300, height=260, order=2, keep=True, ocr_text="Can I use the bathroom?"),
            PanelBox(id="p3", page=1, panel=3, x=0, y=520, width=300, height=260, order=3, keep=True, visual_caption="A student raises his hand nervously."),
            PanelBox(id="p4", page=2, panel=1, x=0, y=0, width=300, height=260, order=4, keep=True, visual_caption="Another student is punched backward."),
            PanelBox(id="p5", page=2, panel=2, x=0, y=260, width=300, height=260, order=5, keep=True, visual_caption="Rin Vale looks shocked by the sudden violence."),
            PanelBox(id="p6", page=2, panel=3, x=0, y=520, width=300, height=260, order=6, keep=True, ocr_text="Break it."),
        ]
        self.store.save_panels(project.id, panels)
        return project.id, project_dir, panels

    def test_vnext_builds_chronological_scene_plan_and_uses_supporting_panels(self) -> None:
        project_id, project_dir, panels = self._project_with_panels()
        output_dir = project_dir / "output"
        panel_vision = [
            {
                "panel_id": "p1",
                "panel_order": 1,
                "page": 1,
                "dialogue": "Class starts with a math lesson.",
                "action_beat": "The teacher opens the class at the board.",
                "character_names": ["Teacher"],
                "character_roles": {"Teacher": ["visible_present", "speaker"]},
                "confidence": 0.9,
            },
            {
                "panel_id": "p2",
                "panel_order": 2,
                "page": 1,
                "dialogue": "Can I use the bathroom?",
                "action_beat": "A student tries to avoid the lesson.",
                "character_names": ["Rin Vale"],
                "character_roles": {"Rin Vale": ["visible_present", "speaker"]},
                "confidence": 0.9,
            },
            {
                "panel_id": "p4",
                "panel_order": 4,
                "page": 2,
                "dialogue": "",
                "action_beat": "A student is punched backward in front of the class.",
                "character_names": ["Rin Vale"],
                "character_roles": {"Rin Vale": ["visible_present"]},
                "confidence": 0.9,
            },
            {
                "panel_id": "p5",
                "panel_order": 5,
                "page": 2,
                "dialogue": "",
                "action_beat": "Rin Vale reacts in shock to the sudden violence.",
                "character_names": ["Rin Vale"],
                "character_roles": {"Rin Vale": ["visible_present"]},
                "confidence": 0.9,
            },
            {
                "panel_id": "p6",
                "panel_order": 6,
                "page": 2,
                "dialogue": "Break it.",
                "action_beat": "A command is shouted during the confrontation.",
                "character_names": ["break it"],
                "character_roles": {"break it": ["speaker"]},
                "confidence": 0.9,
            },
        ]
        (output_dir / "panel_vision_final.json").write_text(json.dumps(panel_vision), encoding="utf-8")

        result = ScriptGenerationVNextService().run(
            project_id=project_id,
            project_name="vNext Script Test",
            project_dir=project_dir,
            chapter_metadata=ChapterMetadata(manga_title="Regression"),
            panels=panels,
            job_id="job-vnext",
        )

        scene_plan = result.scene_plan
        self.assertEqual(scene_plan["artifact_version"], "script_vnext_scene_plan_v1")
        starts = [scene["panel_start"] for scene in scene_plan["scenes"]]
        self.assertEqual(starts, sorted(starts))
        self.assertTrue(all(scene["representative_panel_id"] in scene["source_panel_ids"] for scene in scene_plan["scenes"]))
        self.assertTrue(any(scene["supporting_panel_ids"] for scene in scene_plan["scenes"]))
        promoted_names = [
            name.casefold()
            for scene in scene_plan["scenes"]
            for bucket in ("visible_characters", "speakers", "mentioned_characters")
            for name in scene.get(bucket, [])
        ]
        self.assertNotIn("break it", promoted_names)
        self.assertEqual(result.cost_report["gemini_calls_total"], 0)
        self.assertTrue((project_dir / "output" / "script_vnext" / "final_script.md").exists())
        self.assertTrue((project_dir / "output" / "script_vnext" / "story_context_pack.json").exists())
        context_names = [item["name"] for item in result.story_context_pack["main_characters"]]
        self.assertIn("Rin Vale", context_names)
        self.assertNotIn("break it", [name.casefold() for name in context_names])
        self.assertFalse(result.story_context_pack["rejected_invalid_aliases"] and "Rin Vale" in result.story_context_pack["rejected_invalid_aliases"])

    def test_vnext_timing_repair_uses_supporting_panel_evidence(self) -> None:
        project_id, project_dir, panels = self._project_with_panels()
        panels = [
            panel.model_copy(update={"duration_seconds": 4.0})
            for panel in panels[:4]
        ]
        result = ScriptGenerationVNextService().run(
            project_id=project_id,
            project_name="vNext Timing Test",
            project_dir=project_dir,
            chapter_metadata=ChapterMetadata(),
            panels=panels,
            job_id="job-vnext",
        )

        chunks = result.narration_chunks["chunks"]
        self.assertTrue(chunks)
        self.assertTrue(any(chunk["repair_action"] in {"expanded_with_supporting_panel_evidence", "reduced_visual_duration_for_low_information_scene", "none"} for chunk in chunks))
        self.assertTrue(all("scene_duration_seconds" in chunk for chunk in chunks))
        self.assertTrue(all("estimated_narration_duration_seconds" in chunk for chunk in chunks))

    def test_pipeline_config_accepts_vnext_flag(self) -> None:
        project = self.store.create_project("vNext Config Test", SourceType.IMAGES)
        metadata_path = Path(self._tmpdir.name) / "projects" / project.id / "metadata.json"
        metadata = json.loads(metadata_path.read_text())
        metadata["pipeline_config"]["script_pipeline_version"] = "vNext"
        metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

        loaded = self.store.get_project(project.id)

        self.assertEqual(loaded.pipeline_config.script_pipeline_version, "vNext")

    def _redraft_fixture(self) -> tuple[dict, dict, dict]:
        scene_plan = {
            "scenes": [
                {
                    "scene_id": "scene_001",
                    "source_panel_ids": ["p1", "p2"],
                    "representative_panel_id": "p1",
                    "supporting_panel_ids": ["p2"],
                    "transcript_snippets": ["Can I use the bathroom?"],
                    "visible_characters": ["Rin Vale"],
                    "speakers": ["Rin Vale"],
                    "mentioned_characters": [],
                    "character_roles": {"Rin Vale": ["visible_present", "speaker"]},
                    "target_narration_duration_seconds": 5.0,
                    "panel_contribution_map": {
                        "p1": {
                            "panel_order": 1,
                            "page": 1,
                            "contribution": "dialogue_meaning",
                            "evidence_text": "Can I use the bathroom?",
                            "visual_summary": "Rin Vale raises his hand in class.",
                            "visual_only": False,
                            "rejected_ocr": "DEBUG stale artifact raw OCR garbage",
                        },
                        "p2": {
                            "panel_order": 2,
                            "page": 1,
                            "contribution": "character_reaction",
                            "evidence_text": "",
                            "visual_summary": "debug stale artifact should not appear",
                            "visual_only": True,
                        },
                    },
                },
                {
                    "scene_id": "scene_002",
                    "source_panel_ids": ["p3"],
                    "representative_panel_id": "p3",
                    "supporting_panel_ids": [],
                    "transcript_snippets": ["Class starts normally."],
                    "visible_characters": ["Teacher"],
                    "speakers": ["Teacher"],
                    "mentioned_characters": [],
                    "character_roles": {"Teacher": ["visible_present", "speaker"]},
                    "target_narration_duration_seconds": 3.0,
                    "panel_contribution_map": {
                        "p3": {
                            "panel_order": 3,
                            "page": 1,
                            "contribution": "dialogue_meaning",
                            "evidence_text": "Class starts normally.",
                            "visual_summary": "The teacher lectures at the board.",
                            "visual_only": False,
                        }
                    },
                },
            ]
        }
        narration_chunks = {
            "chunks": [
                {
                    "chunk_id": "scene_001",
                    "scene_id": "scene_001",
                    "text": "The story moves through a short transition before the next turn.",
                    "source_panel_ids": ["p1", "p2"],
                    "panel_start": 1,
                    "panel_end": 2,
                    "representative_panel_id": "p1",
                    "supporting_panel_ids": ["p2"],
                    "scene_duration_seconds": 5.0,
                    "estimated_narration_duration_seconds": 1.8,
                    "duration_gap_seconds": 3.2,
                },
                {
                    "chunk_id": "scene_002",
                    "scene_id": "scene_002",
                    "text": (
                        "The teacher starts class with a normal lesson, and the room settles around the board before anyone interrupts. "
                        "Because the setup is already clear, the next student response can push the classroom pressure forward without needing extra explanation."
                    ),
                    "source_panel_ids": ["p3"],
                    "panel_start": 3,
                    "panel_end": 3,
                    "representative_panel_id": "p3",
                    "supporting_panel_ids": [],
                    "scene_duration_seconds": 3.0,
                    "estimated_narration_duration_seconds": 4.0,
                    "duration_gap_seconds": 0.0,
                },
            ]
        }
        qc_report = {"should_block_tts": True, "failure_codes": ["filler_meta_language"], "quality_score": 10}
        return scene_plan, narration_chunks, qc_report

    def test_redraft_dry_run_estimates_cost_without_calls(self) -> None:
        scene_plan, narration_chunks, qc_report = self._redraft_fixture()
        fake = FakeRedraftClient()
        service = ScriptGenerationVNextService(redraft_client=fake)

        result = service._run_scene_redraft_pass(
            project_dir=Path(self._tmpdir.name),
            scene_plan=scene_plan,
            narration_chunks=narration_chunks,
            qc_report=qc_report,
            config=ScriptVNextRedraftConfig(enabled=True, dry_run=True, max_estimated_cost_usd=1.0),
        )

        self.assertEqual(fake.calls, [])
        self.assertGreater(result["redraft_log"]["estimated_call_count"], 0)
        self.assertEqual(result["redraft_log"]["actual_call_count"], 0)
        self.assertEqual(result["redraft_log"]["target_scene_ids"], ["scene_001"])
        self.assertEqual(result["narration_chunks"]["chunks"][0]["text"], narration_chunks["chunks"][0]["text"])

    def test_redraft_prompt_excludes_rejected_and_debug_text(self) -> None:
        scene_plan, narration_chunks, _ = self._redraft_fixture()
        service = ScriptGenerationVNextService(redraft_client=FakeRedraftClient())
        context_pack = {
            "main_characters": [{"name": "Rin Vale", "confidence": 0.95}],
            "special_terms": ["Source-Grounded Term"],
            "preserve_terms_exact": ["Rin Vale", "Source-Grounded Term"],
            "reject_terms": ["debug stale artifact"],
            "style_tone_notes": ["Use project evidence."],
        }
        packet = service._scene_redraft_packet(scene_plan["scenes"][0], narration_chunks["chunks"][0], context_pack)
        prompt = service._redraft_prompt([packet], context_pack, sanitized=False)

        self.assertIn("source_panel_ids", prompt)
        self.assertIn("representative_panel_id", prompt)
        self.assertIn("supporting_panel_ids", prompt)
        self.assertIn("Rin Vale", prompt)
        self.assertIn("Source-Grounded Term", prompt)
        self.assertNotIn("rejected_ocr", prompt)
        self.assertNotIn("raw OCR garbage", prompt)
        self.assertNotIn("debug stale artifact should not appear", prompt)

    def test_character_name_substitution_prefers_established_name(self) -> None:
        service = ScriptGenerationVNextService()
        scene_plan, narration_chunks, _ = self._redraft_fixture()
        narration_chunks["chunks"][0]["text"] = "The other student looks shocked as the classroom pressure turns against him."
        context_pack = {"main_characters": [{"name": "Rin Vale", "confidence": 0.94}], "preserve_terms_exact": ["Rin Vale"], "reject_terms": []}

        updated = service._apply_character_name_substitutions(narration_chunks, scene_plan, context_pack)

        self.assertIn("Rin Vale looks shocked", updated["chunks"][0]["text"])
        self.assertNotIn("The other student", updated["chunks"][0]["text"])

    def test_generic_label_qc_blocks_when_known_name_exists(self) -> None:
        service = ScriptGenerationVNextService()
        scene_plan, narration_chunks, _ = self._redraft_fixture()
        narration_chunks["chunks"][0]["text"] = "The other student tries to slip away from class."
        context_pack = {"main_characters": [{"name": "Rin Vale", "confidence": 0.94}], "preserve_terms_exact": ["Rin Vale"], "reject_terms": []}

        qc = service._apply_vnext_style_qc({"should_block_tts": False, "failure_codes": [], "quality_score": 90}, scene_plan, narration_chunks, context_pack)

        self.assertTrue(qc["should_block_tts"])
        self.assertIn("vnext_generic_labels_for_known_characters", qc["failure_codes"])

    def test_forbidden_internal_phrases_are_removed_and_flagged(self) -> None:
        service = ScriptGenerationVNextService()
        scene_plan, narration_chunks, _ = self._redraft_fixture()
        context_pack = {"main_characters": [{"name": "Rin Vale", "confidence": 0.94}], "preserve_terms_exact": ["Rin Vale"], "reject_terms": []}

        updated = service._apply_character_name_substitutions(narration_chunks, scene_plan, context_pack)
        qc = service._apply_vnext_style_qc({"should_block_tts": False, "failure_codes": [], "quality_score": 90}, scene_plan, updated, context_pack)

        self.assertFalse(any("The story moves through" in chunk["text"] for chunk in updated["chunks"]))
        self.assertEqual(qc["vnext_style_qc"]["banned_template_phrase_count"], 0)

    def test_rejected_ocr_name_is_not_accepted_in_redraft(self) -> None:
        service = ScriptGenerationVNextService()
        scene_plan, narration_chunks, _ = self._redraft_fixture()
        reason = service._redraft_rejection_reason(
            scene_plan["scenes"][0],
            narration_chunks["chunks"][0],
            "Break it attacks while Rin Vale watches the classroom freeze.",
            {"main_characters": [{"name": "Rin Vale", "confidence": 0.94}], "preserve_terms_exact": ["Rin Vale"], "reject_terms": ["Break it"]},
        )

        self.assertEqual(reason, "rejected_term_leak")

    def test_generalization_audit_scans_runtime_terms_without_active_hits(self) -> None:
        service = ScriptGenerationVNextService()
        audit = service._build_generalization_audit({"preserve_terms_exact": ["Synthetic Fixture Term"]})

        self.assertEqual(audit["blocked_active_hardcoding_hits"], [])
        self.assertTrue(audit["passed"])

    def test_redraft_budget_prevents_excessive_calls(self) -> None:
        scene_plan, narration_chunks, qc_report = self._redraft_fixture()
        fake = FakeRedraftClient()
        service = ScriptGenerationVNextService(redraft_client=fake)

        result = service._run_scene_redraft_pass(
            project_dir=Path(self._tmpdir.name),
            scene_plan=scene_plan,
            narration_chunks=narration_chunks,
            qc_report=qc_report,
            config=ScriptVNextRedraftConfig(enabled=True, max_calls=0, max_estimated_cost_usd=1.0),
        )

        self.assertEqual(fake.calls, [])
        self.assertTrue(result["redraft_log"]["budget_exceeded"])
        self.assertIn("scene_001", result["redraft_log"]["unresolved_scene_ids"])

    def test_oversized_redraft_batches_are_split(self) -> None:
        scene_plan, narration_chunks, qc_report = self._redraft_fixture()
        extra = dict(scene_plan["scenes"][0])
        extra["scene_id"] = "scene_003"
        extra["source_panel_ids"] = ["p4"]
        scene_plan["scenes"].append(extra)
        narration_chunks["chunks"].append({**narration_chunks["chunks"][0], "chunk_id": "scene_003", "scene_id": "scene_003"})
        service = ScriptGenerationVNextService(redraft_client=FakeRedraftClient())

        result = service._run_scene_redraft_pass(
            project_dir=Path(self._tmpdir.name),
            scene_plan=scene_plan,
            narration_chunks=narration_chunks,
            qc_report=qc_report,
            config=ScriptVNextRedraftConfig(enabled=True, max_calls=5, max_scenes_per_batch=5, max_prompt_chars=2200, max_estimated_cost_usd=1.0),
        )

        self.assertGreaterEqual(len(result["redraft_log"]["batches"]), 2)

    def test_redraft_block_retries_sanitized_without_crashing(self) -> None:
        scene_plan, narration_chunks, qc_report = self._redraft_fixture()
        fake = FakeRedraftClient(fail_first=True)
        service = ScriptGenerationVNextService(redraft_client=fake)

        result = service._run_scene_redraft_pass(
            project_dir=Path(self._tmpdir.name),
            scene_plan=scene_plan,
            narration_chunks=narration_chunks,
            qc_report=qc_report,
            config=ScriptVNextRedraftConfig(enabled=True, max_calls=3, max_estimated_cost_usd=1.0),
        )

        self.assertGreaterEqual(len(fake.calls), 2)
        self.assertIn("This is a sanitized retry", fake.calls[-1])
        self.assertIn("scene_001", result["redraft_log"]["redrafted_scene_ids"])

    def test_successful_redraft_improves_style_without_breaking_timing(self) -> None:
        scene_plan, narration_chunks, qc_report = self._redraft_fixture()
        fake = FakeRedraftClient()
        service = ScriptGenerationVNextService(redraft_client=fake)
        before_score = service._local_style_score(narration_chunks["chunks"][0]["text"])

        result = service._run_scene_redraft_pass(
            project_dir=Path(self._tmpdir.name),
            scene_plan=scene_plan,
            narration_chunks=narration_chunks,
            qc_report=qc_report,
            config=ScriptVNextRedraftConfig(enabled=True, max_calls=2, max_estimated_cost_usd=1.0),
        )
        redrafted = result["narration_chunks"]["chunks"][0]

        self.assertGreater(service._local_style_score(redrafted["text"]), before_score)
        self.assertLessEqual(redrafted["duration_gap_seconds"], 2.0)
        self.assertEqual(result["redraft_log"]["actual_call_count"], 1)

    def test_cached_redraft_avoids_repeat_calls(self) -> None:
        scene_plan, narration_chunks, qc_report = self._redraft_fixture()
        fake = FakeRedraftClient()
        service = ScriptGenerationVNextService(redraft_client=fake)
        config = ScriptVNextRedraftConfig(enabled=True, max_calls=2, max_estimated_cost_usd=1.0)

        first = service._run_scene_redraft_pass(
            project_dir=Path(self._tmpdir.name),
            scene_plan=scene_plan,
            narration_chunks=narration_chunks,
            qc_report=qc_report,
            config=config,
        )
        second = service._run_scene_redraft_pass(
            project_dir=Path(self._tmpdir.name),
            scene_plan=scene_plan,
            narration_chunks=narration_chunks,
            qc_report=qc_report,
            config=config,
        )

        self.assertEqual(first["redraft_log"]["actual_call_count"], 1)
        self.assertEqual(second["redraft_log"]["actual_call_count"], 0)
        self.assertEqual(len(fake.calls), 1)
        self.assertTrue(any(batch.get("cache_hit") for batch in second["redraft_log"]["batches"]))

    def test_unresolved_redraft_scenes_keep_qc_blocked(self) -> None:
        service = ScriptGenerationVNextService()
        qc = {"should_block_tts": False, "failure_codes": [], "quality_score": 90}
        gated = service._apply_redraft_qc_gate(qc, {"unresolved_scene_ids": ["scene_001"]})

        self.assertTrue(gated["should_block_tts"])
        self.assertIn("unresolved_redraft_scenes", gated["failure_codes"])
