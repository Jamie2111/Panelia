from __future__ import annotations

from tempfile import TemporaryDirectory
import unittest
from pathlib import Path

from app.schemas.project import PanelBox
from app.schemas.project import ChapterMetadata
from app.schemas.character_identity import CharacterReviewIdentity
from app.services.character_review_service import CharacterReviewService
from app.utils.files import write_json


class CharacterReviewServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.service = CharacterReviewService()

    def test_character_records_fall_back_to_tracking_payload(self) -> None:
        artifacts = {
            "characters": {},
            "character_tracking": {
                "characters": {
                    "Character_1": {
                        "id": "Character_1",
                        "display_name": "Manager",
                        "description": "Manager",
                        "role": "Manager",
                        "appearance_count": 3,
                        "appearances": [],
                        "source_character_ids": ["cluster-001"],
                    }
                }
            },
        }

        records = self.service._character_records_from_artifacts(artifacts)

        self.assertIn("Character_1", records)
        self.assertEqual(records["Character_1"]["display_name"], "Manager")

    def test_normalize_identity_names_cleans_noisy_labels(self) -> None:
        identities = [
            CharacterReviewIdentity(
                review_id="review-1",
                stable_character_ids=["Character_1"],
                source_character_ids=["cluster-001"],
                suggested_name="Th Floor",
                name="Th Floor",
                status="suggested",
                role_hint="Stranger",
                appearance_count=5,
            ),
            CharacterReviewIdentity(
                review_id="review-2",
                stable_character_ids=["Character_2"],
                source_character_ids=["cluster-002"],
                suggested_name="Stranger 2",
                name="Stranger 2",
                status="suggested",
                role_hint="Manager",
                appearance_count=2,
            ),
        ]

        normalized = self.service._normalize_identity_names(identities, protagonist_name="Zhang Yi")

        self.assertEqual(normalized[0].name, "Unidentified Character 1")
        self.assertEqual(normalized[1].name, "Manager")

    def test_build_tracking_from_clusters_creates_character_records(self) -> None:
        tracking = self.service._build_tracking_from_clusters(
            [
                {
                    "cluster_id": "cluster-001",
                    "pages": [2, 4],
                    "panels": [3, 7],
                    "panel_ids": ["p2-3", "p4-7"],
                    "appearance_count": 5,
                }
            ]
        )

        self.assertIn("Character_1", tracking["characters"])
        self.assertEqual(tracking["source_to_character_id"]["cluster-001"], "Character_1")
        self.assertEqual(tracking["characters"]["Character_1"]["appearance_count"], 5)
        self.assertIn("Character_1", tracking["panel_characters"]["p2-3"])

    def test_prepare_review_artifacts_reuses_cached_manifest(self) -> None:
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
            )
        ]
        with TemporaryDirectory() as tmp:
            project_dir = Path(tmp)
            output_dir = project_dir / "output"
            output_dir.mkdir(parents=True, exist_ok=True)
            manifest = {
                "strategy": "dialogue_pipeline_v2_page_ocr_recall",
                "panel_signature": self.service._panel_signature(panels),
                "character_clusters": [],
                "characters": {},
                "character_dictionary": {},
                "character_identity_report": {"review_analysis_version": 5},
            }
            write_json(output_dir / "dialogue_pipeline_manifest.json", manifest)

            artifacts, used_manifest = self.service.prepare_review_artifacts(
                project_dir,
                metadata=ChapterMetadata(),
                panels=panels,
                page_paths=[],
            )

        self.assertTrue(used_manifest)
        self.assertEqual(artifacts["panel_signature"], manifest["panel_signature"])

    def test_merge_face_payloads_keeps_precise_face_alongside_body_box(self) -> None:
        merged, added_count = self.service._merge_face_payloads(
            {
                1: {
                    "page": 1,
                    "provider": "magi-hf",
                    "characters": [
                        {
                            "character_id": "magi-p0001-char-001",
                            "bbox": [10, 10, 200, 500],
                            "source": "magi-hf",
                        }
                    ],
                    "texts": [],
                    "panels": [],
                }
            },
            {
                1: {
                    "page": 1,
                    "provider": "animeface-lbp-v1",
                    "characters": [
                        {
                            "character_id": "animeface-p0001-face-001",
                            "bbox": [60, 40, 70, 70],
                            "source": "animeface-lbp",
                        }
                    ],
                }
            },
        )

        self.assertEqual(added_count, 1)
        self.assertEqual(len(merged[1]["characters"]), 2)
        self.assertIn("animeface-lbp-v1", merged[1]["provider"])

    def test_merge_face_payloads_skips_duplicate_face_boxes(self) -> None:
        merged, added_count = self.service._merge_face_payloads(
            {
                1: {
                    "page": 1,
                    "provider": "character-review-merged-v1",
                    "characters": [
                        {
                            "character_id": "animeface-p0001-face-001",
                            "bbox": [60, 40, 70, 70],
                            "source": "animeface-lbp",
                        }
                    ],
                    "texts": [],
                    "panels": [],
                }
            },
            {
                1: {
                    "page": 1,
                    "provider": "animeface-lbp-v1",
                    "characters": [
                        {
                            "character_id": "animeface-p0001-face-002",
                            "bbox": [62, 42, 70, 70],
                            "source": "animeface-lbp",
                        }
                    ],
                }
            },
        )

        self.assertEqual(added_count, 0)
        self.assertEqual(len(merged[1]["characters"]), 1)


if __name__ == "__main__":
    unittest.main()
