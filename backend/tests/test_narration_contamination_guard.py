import json
import tempfile
import unittest
from pathlib import Path

from app.schemas.project import VoiceConfig
from app.services.generate_narration import generate_narration
from app.services.narration_contamination_guard import NarrationContaminationGuard


class NarrationContaminationGuardTests(unittest.TestCase):
    def test_contaminated_ocr_strings_are_quarantined_before_narration(self) -> None:
        guard = NarrationContaminationGuard()

        result = guard.prepare(
            [
                "Ichigo is reminded that Papa and the Sages are watching.",
                "GWIRRR GARRR CODE CODc 016 Uenn AND CYYC 7 azlp You will Soon be TRANS PORTED Back to GARDEN.",
                "WHAT cYnmcw ARE MAN; Aano WHAT cYnmcw ARE MAN; Aano Zero Two notes the blow wasn't lethal.",
                "HIRO HASN'T Given up HIKO HIRO HASN'T Given up HIKO.",
            ],
            supported_character_names=["Hiro", "Zero Two", "Ichigo"],
            world_terms=["Garden"],
        )

        self.assertEqual(result.report["quarantined_units"], 3)
        self.assertFalse(result.report["script_ready"])
        self.assertEqual(result.script_lines, ["Ichigo is reminded that Papa and the Sages are watching."])

    def test_nance_is_rejected_unless_source_supported(self) -> None:
        guard = NarrationContaminationGuard()

        unsupported = guard.prepare(
            ["Hiro and Nance are marked as official parasites."],
            supported_character_names=["Hiro", "Zero Two"],
        )
        supported = guard.prepare(
            ["Hiro and Nance are marked as official parasites."],
            supported_character_names=["Hiro", "Nance"],
        )

        self.assertEqual(unsupported.report["repaired_units"], 1)
        self.assertIn("the other pilot", unsupported.script_lines[0])
        self.assertNotIn("Nance", unsupported.script_lines[0])
        self.assertTrue(supported.report["script_ready"])
        self.assertIn("Nance", supported.script_lines[0])

    def test_repeated_overlapping_beats_are_merged(self) -> None:
        guard = NarrationContaminationGuard()

        result = guard.prepare(
            [
                "Zorome insists the rumor is true, claiming he overheard security discussing it.",
                "Without warning, Zorome insists the rumor is real, claiming he overheard security discussing it.",
                "Ichigo calls Hiro a dummy for his actions.",
                "Mitsuru suggests that understanding Hiro's feelings might be more productive.",
            ],
            supported_character_names=["Zorome", "Ichigo", "Hiro", "Mitsuru"],
        )

        self.assertGreaterEqual(result.report["near_duplicate_units"], 1)
        self.assertLess(len(result.script_lines), 4)
        self.assertLessEqual(result.report["max_one_sentence_run"], 3)

    def test_blocked_generation_writes_partial_not_ready_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            audio_dir = Path(tmp) / "audio"

            with self.assertRaises(ValueError):
                generate_narration(
                    ["GWIRRR GARRR CODE CODc 016 Uenn AND CYYC 7 azlp You will Soon be TRANS PORTED Back to GARDEN."],
                    audio_dir,
                    VoiceConfig(),
                    supported_character_names=["Hiro"],
                    world_terms=["Garden"],
                )

            artifact = Path(tmp) / "output" / "enhanced_narration.json"
            payload = json.loads(artifact.read_text(encoding="utf-8"))
            self.assertEqual(payload["artifact_status"], "in_progress")
            self.assertFalse(payload["script_ready"])
            self.assertGreater(payload["qc_report"]["quarantined_units"], 0)


if __name__ == "__main__":
    unittest.main()
