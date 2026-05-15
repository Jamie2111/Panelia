from __future__ import annotations

import unittest

from app.schemas.project import ChapterMetadata
from app.services.character_name_service import CharacterNameService


class CharacterNameServiceTests(unittest.TestCase):
    def test_extracts_canon_names_from_manga_synopsis(self) -> None:
        service = CharacterNameService()
        metadata = ChapterMetadata(
            raw={
                "relationships": [
                    {
                        "type": "manga",
                        "attributes": {
                            "description": {
                                "en": (
                                    "A boy named Hiro, called Code:016, was once known as a prodigy. "
                                    "One day, a mysterious girl known as “Zero Two” appears before him."
                                )
                            }
                        },
                    }
                ]
            }
        )

        character_dictionary, protagonist_name = service.discover(["Be Dead!", "Sauri?"], metadata)

        self.assertEqual(protagonist_name, "Hiro")
        self.assertEqual(character_dictionary.get("hiro"), "Hiro")
        self.assertEqual(character_dictionary.get("zero two"), "Zero Two")
        self.assertNotIn("be dead", character_dictionary)
        self.assertNotIn("sauri", character_dictionary)

    def test_rejects_ocr_noise_names_from_dialogue(self) -> None:
        service = CharacterNameService()
        metadata = ChapterMetadata()

        character_dictionary, _protagonist_name = service.discover(
            [
                "Kcdikaini Lass!",
                "Start It!",
                "Nc Jaiv!",
                "Hose!",
                "A Shaft!",
                "My name is Zero Two.",
                "My name is Hiro.",
                "I am Hiro.",
                "I am Zero Two.",
                "This is Hiro.",
                "This is Zero Two.",
            ],
            metadata,
        )

        self.assertEqual(character_dictionary.get("hiro"), "Hiro")
        self.assertEqual(character_dictionary.get("zero two"), "Zero Two")
        self.assertNotIn("kcdikaini lass", character_dictionary)
        self.assertNotIn("start it", character_dictionary)
        self.assertNotIn("nc jaiv", character_dictionary)
        self.assertNotIn("hose", character_dictionary)
        self.assertNotIn("a shaft", character_dictionary)

    def test_rejects_dialogue_fragments_and_other_as_names(self) -> None:
        service = CharacterNameService()
        metadata = ChapterMetadata()

        character_dictionary, _protagonist_name = service.discover(
            [
                "Break it!",
                "Other",
                "Run!",
                "Stop!",
                "Idiot!",
                "John!",
                "My name is John.",
                "This is John.",
                "I am John.",
            ],
            metadata,
        )

        self.assertEqual(character_dictionary.get("john"), "John")
        self.assertNotIn("break it", character_dictionary)
        self.assertNotIn("other", character_dictionary)
        self.assertNotIn("run", character_dictionary)
        self.assertNotIn("stop", character_dictionary)
        self.assertNotIn("idiot", character_dictionary)

    def test_extracts_canon_names_from_comix_synopsis(self) -> None:
        service = CharacterNameService()
        metadata = ChapterMetadata(
            raw={
                "manga": {
                    "synopsis": (
                        "A boy named Hiro, called Code:016, was once known as a prodigy. "
                        "One day, a mysterious girl known as “Zero Two” appears before him."
                    )
                }
            }
        )

        character_dictionary, protagonist_name = service.discover([], metadata)

        self.assertEqual(protagonist_name, "Hiro")
        self.assertEqual(character_dictionary.get("hiro"), "Hiro")
        self.assertEqual(character_dictionary.get("zero two"), "Zero Two")


if __name__ == "__main__":
    unittest.main()
