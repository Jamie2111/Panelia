from __future__ import annotations

import json
from pathlib import Path

from app.services.style_vocabulary import build_style_vocabulary
from app.services.llm_router import LLMRouter


PROJECT_ROOT = Path(__file__).resolve().parents[1] / "data" / "projects"


def _load(project_id: str, name: str, default):
    path = PROJECT_ROOT / project_id / "output" / name
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def _build(project_id: str):
    return build_style_vocabulary(
        canonical_characters=_load(project_id, "canonical_characters.json", []),
        character_dictionary=_load(project_id, "character_dictionary.json", {}),
        story_bible=_load(project_id, "story_bible.json", {}),
        scene_summaries=_load(project_id, "scene_summaries.json", {}),
    )


def test_darling_vocabulary_uses_project_terms() -> None:
    vocab = _build("darling-in-the-franxx-6f2b8388")

    assert vocab.protagonist in {"Hiro", "Zero Two"}
    assert "Squad 13" == vocab.team_term
    assert {"franxx", "klaxosaurs"} & {term.casefold() for term in vocab.world_terms}
    assert "zhang yi" not in {name.casefold() for name in vocab.named_characters}


def test_codex_part_two_vocabulary_uses_project_terms() -> None:
    vocab = _build("codex-global-freeze-2h-ptbr-part-02-caef52f4")

    assert vocab.protagonist == "Zhang Yi"
    assert any("shelter" in term.casefold() or "suppl" in term.casefold() for term in vocab.world_terms)
    assert "hiro" not in {name.casefold() for name in vocab.named_characters}


def test_third_project_does_not_inherit_reference_tokens() -> None:
    vocab = _build("unordinary-83a533dc")
    serialized = json.dumps(vocab.to_dict()).casefold()

    for leaked in ("hiro", "zero two", "franxx", "klaxosaur", "zhang yi", "doomsday", "snowbound"):
        assert leaked not in serialized


def test_scene_mode_prompt_includes_style_vocabulary_and_length_target() -> None:
    router = LLMRouter()
    prompt = router._story_segments_prompt(
        [{"segment_id": "scene_001_beat_01", "scene_id": 1, "panel_count": 3}],
        {
            "scene_mode": True,
            "style_vocabulary": {
                "named_characters": ["Character A", "Character B"],
                "team_term": "the crew",
                "world_terms": ["the tower"],
                "stakes_phrases": ["the escape"],
            },
        },
    )

    assert "RECURRING WORLD VOCABULARY" in prompt
    assert "50-90 words per segment" in prompt
    assert "the tower" in prompt
