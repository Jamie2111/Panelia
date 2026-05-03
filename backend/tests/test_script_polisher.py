import unittest

from app.schemas.project import ChapterMetadata
from app.services.llm_router import RoutedResult
from app.services.script_polisher import ScriptPolisher


class _FakeRouter:
    def __init__(
        self,
        rewritten_chunks: list[list[str]],
        line_repair_batches: list[list[dict[str, object]]] | None = None,
    ) -> None:
        self._rewritten_chunks = list(rewritten_chunks)
        self._line_repair_batches = list(line_repair_batches or [])
        self.calls: list[dict[str, object]] = []
        self.repair_calls: list[dict[str, object]] = []

    def available_providers(self) -> list[str]:
        return ["gemini"]

    async def rewrite_full_story(
        self,
        draft_lines: list[str],
        chapter_summary: str,
        character_dictionary: dict[str, object],
        *,
        project_title: str = "",
        chapter_metadata: dict[str, object] | None = None,
        locked_examples: str = "",
        previous_lines: list[str] | None = None,
        next_lines: list[str] | None = None,
        chunk_index: int = 1,
        chunk_total: int = 1,
        slot_evidence: list[dict[str, object]] | None = None,
        provider: str | None = None,
    ) -> RoutedResult:
        self.calls.append(
            {
                "draft_lines": list(draft_lines),
                "project_title": project_title,
                "chapter_metadata": dict(chapter_metadata or {}),
                "previous_lines": list(previous_lines or []),
                "next_lines": list(next_lines or []),
                "chunk_index": chunk_index,
                "chunk_total": chunk_total,
                "slot_evidence": list(slot_evidence or []),
                "provider": provider,
            }
        )
        rewritten = self._rewritten_chunks.pop(0)
        return RoutedResult(
            provider="gemini",
            model="fake-gemini",
            payload={"rewritten_lines": rewritten},
        )

    async def repair_story_lines(
        self,
        lines: list[dict[str, object]],
        context: dict[str, object],
        *,
        provider: str | None = None,
    ) -> RoutedResult:
        self.repair_calls.append(
            {
                "lines": list(lines),
                "context": dict(context),
                "provider": provider,
            }
        )
        if self._line_repair_batches:
            rewrites = self._line_repair_batches.pop(0)
        else:
            rewrites = [
                {"index": int(item.get("index") or 0), "line": str(item.get("current_line") or "").strip()}
                for item in lines
            ]
        return RoutedResult(
            provider="gemini",
            model="fake-gemini",
            payload={"rewrites": rewrites},
        )

    async def suggest_series_cast_hints(self, context, *, provider=None) -> RoutedResult:
        return RoutedResult(
            provider="gemini",
            model="fake-gemini",
            payload={"series_cast_hints": [], "canonical_name_corrections": []},
        )


class _FakeCohesiveRouter:
    def available_providers(self) -> list[str]:
        return ["gemini"]

    async def _route_json(self, **kwargs) -> RoutedResult:
        return RoutedResult(
            provider="gemini",
            model="fake-gemini",
            payload={"rewritten_lines": ["Hiro meets Zero Two."]},
        )

    async def rewrite_full_story(self, *args, **kwargs) -> RoutedResult:
        return RoutedResult(
            provider="gemini",
            model="fake-gemini",
            payload={"rewritten_lines": ["Hiro meets Zero Two."]},
        )

    async def repair_story_lines(self, lines, context, *, provider=None) -> RoutedResult:
        return RoutedResult(
            provider="gemini",
            model="fake-gemini",
            payload={
                "rewrites": [
                    {"index": int(item.get("index") or 0), "line": str(item.get("current_line") or "").strip()}
                    for item in lines
                ]
            },
        )

    async def suggest_series_cast_hints(self, context, *, provider=None) -> RoutedResult:
        return RoutedResult(
            provider="gemini",
            model="fake-gemini",
            payload={"series_cast_hints": [], "canonical_name_corrections": []},
        )


class ScriptPolisherContinuityPassTests(unittest.TestCase):
    def test_cohesive_rewrite_accepts_chapter_metadata_model(self) -> None:
        polisher = ScriptPolisher(router=_FakeCohesiveRouter())

        result = polisher._rewrite_chunk(
            ["Hiro meets Zero Two."],
            "Hiro meets Zero Two.",
            {"hiro": "Hiro", "zero two": "Zero Two"},
            project_title="DARLING in the FRANXX",
            chapter_metadata=ChapterMetadata(manga_title="DARLING in the FRANXX", language="en"),
        )

        self.assertEqual(result, ["Hiro meets Zero Two."])

    def test_final_continuity_pass_applies_mapped_rewrite(self) -> None:
        lines = [
            "Hiro feels like he has no place left in the world.",
            "Zero Two appears and turns that despair upside down.",
        ]
        rewritten = [
            [
                "Hiro can barely hide how completely he has lost faith in himself.",
                "Zero Two tears into that despair and gives his story a violent new direction.",
            ]
        ]
        router = _FakeRouter(rewritten)
        polisher = ScriptPolisher(router=router)
        polisher._cohesive_rewrite = (
            lambda draft, summary, characters, project_title="", chapter_metadata=None, locked_examples="": draft
        )

        result = polisher.polish(lines, "Hiro meets Zero Two.", {})

        self.assertEqual(
            result,
            [
                "Hiro can barely hide how completely he has lost faith in himself.",
                "Zero Two tears into that despair and gives his story a violent new direction.",
            ],
        )
        self.assertEqual(len(router.calls), 1)
        self.assertEqual(router.calls[0]["provider"], "gemini")

    def test_final_continuity_pass_falls_back_when_line_count_changes(self) -> None:
        lines = [
            "Hiro feels like he has no place left in the world.",
            "Zero Two appears and turns that despair upside down.",
        ]
        router = _FakeRouter([["Hiro loses himself when Zero Two appears."]])
        polisher = ScriptPolisher(router=router)
        polisher._cohesive_rewrite = (
            lambda draft, summary, characters, project_title="", chapter_metadata=None, locked_examples="": draft
        )

        result = polisher.polish(lines, "Hiro meets Zero Two.", {})

        self.assertEqual(result, lines)
        self.assertEqual(len(router.calls), 1)

    def test_final_pass_repairs_generic_subjects_and_wrong_local_name(self) -> None:
        lines = [
            "Hiro freezes when Zero Two suddenly appears before him.",
            "Zero Two grabs Hiro and pulls him away from the battlefield.",
        ]
        router = _FakeRouter(
            [
                [
                    "Someone freezes when the mysterious girl appears.",
                    "Sauri drags him away from the battlefield.",
                ]
            ]
        )
        polisher = ScriptPolisher(router=router)
        polisher._cohesive_rewrite = (
            lambda draft, summary, characters, project_title="", chapter_metadata=None, locked_examples="": draft
        )
        slot_evidence = [
            {
                "character_names": ["Hiro", "Zero Two"],
                "preferred_subject": "Hiro",
                "ocr_text": "Zero Two appears before Hiro.",
                "scene_summary": "Hiro meets Zero Two for the first time.",
            },
            {
                "character_names": ["Zero Two", "Hiro"],
                "preferred_subject": "Zero Two",
                "ocr_text": "Zero Two grabs Hiro and runs.",
                "scene_summary": "Zero Two drags Hiro away.",
            },
        ]

        result = polisher.polish(
            lines,
            "Hiro meets Zero Two.",
            {"Hiro": {}, "Zero Two": {}},
            slot_evidence=slot_evidence,
        )

        self.assertEqual(
            result,
            [
                "Hiro freezes when the mysterious girl appears.",
                "Zero Two drags him away from the battlefield.",
            ],
        )

    def test_final_pass_does_not_replace_normal_capitalized_sentence_opener(self) -> None:
        lines = [
            "Massive glowing structures hover over the desolate landscape.",
        ]
        router = _FakeRouter(
            [
                [
                    "Massive glowing structures hover ominously above the barren landscape.",
                ]
            ]
        )
        polisher = ScriptPolisher(router=router)
        polisher._cohesive_rewrite = (
            lambda draft, summary, characters, project_title="", chapter_metadata=None, locked_examples="": draft
        )
        slot_evidence = [
            {
                "character_names": ["Hiro"],
                "preferred_subject": "Hiro",
                "scene_summary": "Massive structures hover over a barren landscape.",
            }
        ]

        result = polisher.polish(
            lines,
            "A devastated world is revealed.",
            {"Hiro": {}},
            slot_evidence=slot_evidence,
        )

        self.assertEqual(result, ["Massive glowing structures hover ominously above the barren landscape."])

    def test_final_pass_does_not_replace_acronym_sentence_subject(self) -> None:
        lines = [
            "APE developed a desperate countermeasure against the Klaxosaurs.",
        ]
        router = _FakeRouter(
            [
                [
                    "APE, a group of gifted scientists, invented a countermeasure.",
                ]
            ]
        )
        polisher = ScriptPolisher(router=router)
        polisher._cohesive_rewrite = (
            lambda draft, summary, characters, project_title="", chapter_metadata=None, locked_examples="": draft
        )
        slot_evidence = [
            {
                "character_names": ["Hiro"],
                "preferred_subject": "Hiro",
                "scene_summary": "APE creates the Franxx countermeasure.",
            }
        ]

        result = polisher.polish(
            lines,
            "APE creates the Franxx.",
            {"Hiro": {}},
            slot_evidence=slot_evidence,
        )

        self.assertEqual(result, ["APE, a group of gifted scientists, invented a countermeasure."])

    def test_final_pass_scrubs_machine_character_labels(self) -> None:
        lines = [
            "The handler warns Hiro to stay away from Zero Two.",
        ]
        router = _FakeRouter(
            [
                [
                    "Character_10 warns Hiro not to get close to Zero Two.",
                ]
            ]
        )
        polisher = ScriptPolisher(router=router)
        polisher._cohesive_rewrite = (
            lambda draft, summary, characters, project_title="", chapter_metadata=None, locked_examples="": draft
        )

        result = polisher.polish(
            lines,
            "Hiro meets Zero Two.",
            {"Hiro": {}, "Zero Two": {}},
            slot_evidence=[{"scene_summary": "A handler warns Hiro about Zero Two."}],
        )

        self.assertEqual(result, ["The speaker warns Hiro not to get close to Zero Two."])

    def test_placeholder_scrub_preserves_idioms_and_plural_subjects(self) -> None:
        polisher = ScriptPolisher(router=_FakeRouter([[]]))

        self.assertEqual(
            polisher._replace_machine_placeholders("Other of nowhere, Klaxosaurs appear."),
            "Out of nowhere, Klaxosaurs appear.",
        )
        self.assertEqual(
            polisher._replace_machine_placeholders("Other figures in uniform discuss the situation."),
            "Other figures in uniform discuss the situation.",
        )

    def test_final_pass_falls_back_when_rewrite_drops_local_beat(self) -> None:
        lines = [
            "Hiro apologizes as he reaches out to Naomi.",
            "Naomi turns away without answering him.",
        ]
        router = _FakeRouter(
            [
                [
                    "The squad prepares for a desperate battle.",
                    "The situation becomes more intense.",
                ]
            ]
        )
        polisher = ScriptPolisher(router=router)
        polisher._cohesive_rewrite = (
            lambda draft, summary, characters, project_title="", chapter_metadata=None, locked_examples="": draft
        )
        slot_evidence = [
            {
                "character_names": ["Hiro", "Naomi"],
                "preferred_subject": "Hiro",
                "ocr_text": "sorry naomi",
                "scene_summary": "Hiro tries to apologize to Naomi.",
            },
            {
                "character_names": ["Naomi", "Hiro"],
                "preferred_subject": "Naomi",
                "ocr_text": "Naomi ignores him.",
                "scene_summary": "Naomi turns away from Hiro.",
            },
        ]

        result = polisher.polish(
            lines,
            "Hiro tries to apologize to Naomi.",
            {"Hiro": {}, "Naomi": {}},
            slot_evidence=slot_evidence,
        )

        self.assertEqual(result, lines)

    def test_robotic_line_repair_rewrites_report_style_sentence(self) -> None:
        lines = [
            "Zero Two expresses her desire to swim in the ocean.",
            "Hiro cannot tell whether she is joking.",
        ]
        router = _FakeRouter(
            [lines],
            line_repair_batches=[
                [
                    {"index": 0, "line": "Zero Two keeps talking about finding an ocean big enough to swim in."},
                ]
            ],
        )
        polisher = ScriptPolisher(router=router)
        polisher._cohesive_rewrite = (
            lambda draft, summary, characters, project_title="", chapter_metadata=None, locked_examples="": draft
        )

        result = polisher.polish(
            lines,
            "Zero Two wants to see the ocean.",
            {"Zero Two": {}, "Hiro": {}},
            slot_evidence=[
                {
                    "character_names": ["Zero Two", "Hiro"],
                    "preferred_subject": "Zero Two",
                    "ocr_text": "I want to swim in the ocean",
                    "scene_summary": "Zero Two talks about the ocean.",
                },
                {
                    "character_names": ["Hiro", "Zero Two"],
                    "preferred_subject": "Hiro",
                    "scene_summary": "Hiro is unsure how serious she is.",
                },
            ],
        )

        self.assertEqual(
            result,
            [
                "Zero Two keeps talking about finding an ocean big enough to swim in.",
                "Hiro cannot tell whether she is joking.",
            ],
        )
        self.assertEqual(len(router.repair_calls), 1)

    def test_robotic_line_repair_rejects_off_slot_candidate(self) -> None:
        lines = [
            "Ichigo questions Hiro about missing the enlistment ceremony briefing.",
        ]
        router = _FakeRouter(
            [lines],
            line_repair_batches=[
                [
                    {"index": 0, "line": "A strange explosion tears through the city."},
                ]
            ],
        )
        polisher = ScriptPolisher(router=router)
        polisher._cohesive_rewrite = (
            lambda draft, summary, characters, project_title="", chapter_metadata=None, locked_examples="": draft
        )

        result = polisher.polish(
            lines,
            "Ichigo confronts Hiro about the briefing.",
            {"Ichigo": {}, "Hiro": {}},
            slot_evidence=[
                {
                    "character_names": ["Ichigo", "Hiro"],
                    "preferred_subject": "Ichigo",
                    "ocr_text": "Why did you miss the briefing?",
                    "scene_summary": "Ichigo presses Hiro about the ceremony briefing.",
                }
            ],
        )

        self.assertEqual(result, lines)


if __name__ == "__main__":
    unittest.main()
