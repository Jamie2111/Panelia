import os
import tempfile
import unittest
from pathlib import Path
import json

from app.core.config import get_settings
from app.schemas.project import PanelBox, SourceType
from app.services.project_store import ProjectStore


class ProjectStoreScriptAlignmentTests(unittest.TestCase):
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

    def test_save_script_preserves_unflagged_textless_slot(self) -> None:
        project = self.store.create_project("Alignment Test", SourceType.IMAGES)
        project_id = project.id
        panels = [
            PanelBox(
                id="panel-1",
                page=1,
                panel=1,
                x=0,
                y=0,
                width=100,
                height=100,
                order=1,
                keep=True,
                narration="Old fallback line that should not keep a slot.",
            ),
            PanelBox(
                id="panel-2",
                page=1,
                panel=2,
                x=0,
                y=100,
                width=100,
                height=100,
                order=2,
                keep=True,
                ocr_text="Real dialogue one.",
            ),
            PanelBox(
                id="panel-3",
                page=1,
                panel=3,
                x=0,
                y=200,
                width=100,
                height=100,
                order=3,
                keep=True,
                ocr_text="Real dialogue two.",
            ),
        ]
        self.store.save_panels(project_id, panels)

        self.store.save_script(
            project_id,
            [
                "Old fallback line that should not keep a slot.",
                "Hiro answers the first real beat.",
                "Zero Two drives the next beat forward.",
            ],
        )

        saved_panels = self.store.load_panels(project_id)
        kept_panels = [panel for panel in sorted(saved_panels, key=lambda item: item.order) if panel.keep]
        self.assertEqual([panel.narration or "" for panel in kept_panels], [
            "Old fallback line that should not keep a slot.",
            "Hiro answers the first real beat.",
            "Zero Two drives the next beat forward.",
        ])
        self.assertFalse(bool(kept_panels[0].manual_narration))

        manifest = self.store.load_script(project_id)
        self.assertEqual(manifest, [
            "Old fallback line that should not keep a slot.",
            "Hiro answers the first real beat.",
            "Zero Two drives the next beat forward.",
        ])

    def test_locked_textless_panel_keeps_its_script_slot(self) -> None:
        project = self.store.create_project("Locked Alignment Test", SourceType.IMAGES)
        project_id = project.id
        panels = [
            PanelBox(
                id="panel-1",
                page=1,
                panel=1,
                x=0,
                y=0,
                width=100,
                height=100,
                order=1,
                keep=True,
                narration="Manual opening beat.",
                narration_locked=True,
                manual_narration=True,
            ),
            PanelBox(
                id="panel-2",
                page=1,
                panel=2,
                x=0,
                y=100,
                width=100,
                height=100,
                order=2,
                keep=True,
                ocr_text="Real dialogue.",
            ),
        ]
        self.store.save_panels(project_id, panels)

        self.store.save_script(
            project_id,
            [
                "Manual opening beat.",
                "The real dialogue lands here.",
            ],
        )

        kept_panels = [
            panel for panel in sorted(self.store.load_panels(project_id), key=lambda item: item.order) if panel.keep
        ]
        self.assertEqual([panel.narration or "" for panel in kept_panels], [
            "Manual opening beat.",
            "The real dialogue lands here.",
        ])

    def test_unflagged_textless_visual_beat_keeps_its_line(self) -> None:
        project = self.store.create_project("Visual Beat Test", SourceType.IMAGES)
        project_id = project.id
        panels = [
            PanelBox(
                id="panel-1",
                page=1,
                panel=1,
                x=0,
                y=0,
                width=100,
                height=100,
                order=1,
                keep=True,
                narration="A silent but important visual beat still needs narration.",
            ),
            PanelBox(
                id="panel-2",
                page=1,
                panel=2,
                x=0,
                y=100,
                width=100,
                height=100,
                order=2,
                keep=True,
                ocr_text="Real dialogue.",
            ),
        ]
        self.store.save_panels(project_id, panels)

        self.store.save_script(
            project_id,
            [
                "A silent but important visual beat still needs narration.",
                "The real dialogue lands here.",
            ],
        )

        kept_panels = [
            panel for panel in sorted(self.store.load_panels(project_id), key=lambda item: item.order) if panel.keep
        ]
        self.assertEqual([panel.narration or "" for panel in kept_panels], [
            "A silent but important visual beat still needs narration.",
            "The real dialogue lands here.",
        ])

    def test_flagged_textless_fragment_drops_its_slot(self) -> None:
        project = self.store.create_project("Fragment Alignment Test", SourceType.IMAGES)
        project_id = project.id
        panels = [
            PanelBox(
                id="panel-1",
                page=1,
                panel=1,
                x=0,
                y=0,
                width=100,
                height=100,
                order=1,
                keep=True,
                narration="A fragment line that should not keep a slot.",
                review_flags=["side_void", "corner_wedge"],
            ),
            PanelBox(
                id="panel-2",
                page=1,
                panel=2,
                x=0,
                y=100,
                width=100,
                height=100,
                order=2,
                keep=True,
                ocr_text="Real dialogue one.",
            ),
            PanelBox(
                id="panel-3",
                page=1,
                panel=3,
                x=0,
                y=200,
                width=100,
                height=100,
                order=3,
                keep=True,
                ocr_text="Real dialogue two.",
            ),
        ]
        self.store.save_panels(project_id, panels)

        self.store.save_script(
            project_id,
            [
                "A fragment line that should not keep a slot.",
                "Hiro answers the first real beat.",
                "Zero Two drives the next beat forward.",
            ],
        )

        kept_panels = [
            panel for panel in sorted(self.store.load_panels(project_id), key=lambda item: item.order) if panel.keep
        ]
        self.assertEqual([panel.narration or "" for panel in kept_panels], [
            "",
            "Hiro answers the first real beat.",
            "Zero Two drives the next beat forward.",
        ])

    def test_save_script_persists_strict_lines_and_slot_blocks(self) -> None:
        project = self.store.create_project("Strict Script Test", SourceType.IMAGES)
        project_id = project.id
        panels = [
            PanelBox(
                id="panel-1",
                page=1,
                panel=1,
                x=0,
                y=0,
                width=100,
                height=100,
                order=1,
                keep=True,
                ocr_text="Hiro apologizes.",
            ),
            PanelBox(
                id="panel-2",
                page=1,
                panel=2,
                x=0,
                y=100,
                width=100,
                height=100,
                order=2,
                keep=True,
                ocr_text="Zero Two drags him away.",
            ),
        ]
        self.store.save_panels(project_id, panels)

        self.store.save_script(
            project_id,
            [
                "Hiro quietly apologizes before everything falls apart.",
                "Zero Two yanks him away before anyone can stop her.",
            ],
            strict_lines=[
                "Hiro apologizes.",
                "Zero Two drags him away.",
            ],
            slot_evidence=[
                {
                    "panel_id": "panel-1",
                    "panel_order": 1,
                    "page": 1,
                    "ocr_text": "Hiro apologizes.",
                    "character_names": ["Hiro"],
                    "preferred_subject": "Hiro",
                    "scene_summary": "Hiro apologizes.",
                },
                {
                    "panel_id": "panel-2",
                    "panel_order": 2,
                    "page": 1,
                    "ocr_text": "Zero Two drags him away.",
                    "character_names": ["Zero Two", "Hiro"],
                    "preferred_subject": "Zero Two",
                    "scene_summary": "Zero Two drags Hiro away.",
                },
            ],
        )

        manifest_path = Path(self._tmpdir.name) / "projects" / project_id / "script.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.assertEqual(
            manifest["script_lines_strict"],
            [
                "Hiro apologizes.",
                "Zero Two drags him away.",
            ],
        )
        self.assertEqual(
            manifest["script_lines_cinematic"],
            [
                "Hiro quietly apologizes before everything falls apart.",
                "Zero Two yanks him away before anyone can stop her.",
            ],
        )

        blocks_path = Path(self._tmpdir.name) / "projects" / project_id / "output" / "panel_script_blocks.json"
        blocks = json.loads(blocks_path.read_text(encoding="utf-8"))
        self.assertEqual(blocks[0]["strict_narration"], "Hiro apologizes.")
        self.assertEqual(blocks[1]["preferred_subject"], "Zero Two")


if __name__ == "__main__":
    unittest.main()
