from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.services.llm_router import LLMRouter
from app.services.project_store import ProjectStore
from app.services.story_segment_repair_service import StorySegmentRepairService
from app.services.story_script_service import StoryScriptService


def _load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def _scene_summary_lookup(project_dir: Path) -> dict[int, str]:
    payload = _load_json(project_dir / "output" / "scene_summaries.json")
    scenes = payload.get("scenes") or []
    lookup: dict[int, str] = {}
    for item in scenes:
        if not isinstance(item, dict):
            continue
        scene_id = int(item.get("scene_id") or 0)
        summary = str(item.get("description") or item.get("summary") or "").strip()
        if scene_id and summary:
            lookup[scene_id] = summary
    return lookup


def _build_units(store: ProjectStore, project_id: str, service: StoryScriptService) -> tuple[list[dict], dict[str, object]]:
    project = store.get_project(project_id)
    segments = sorted(project.story_segments or store.load_story_segments(project_id), key=lambda item: item.order)
    panels_by_id = {panel.id: panel for panel in project.panels if panel.keep}
    scene_summary_lookup = _scene_summary_lookup(store._project_dir(project_id))
    scene_counts = Counter(int(segment.scene_id or 0) for segment in segments)
    scene_offsets: dict[int, int] = defaultdict(int)
    units: list[dict] = []
    for segment in segments:
        scene_id = int(segment.scene_id or 0) or segment.order
        scene_offsets[scene_id] += 1
        panel_ids = [panel_id for panel_id in segment.panel_ids if panel_id in panels_by_id]
        panels = [panels_by_id[panel_id] for panel_id in panel_ids]
        combined_text = " ".join(str(panel.ocr_text or "").strip() for panel in panels if str(panel.ocr_text or "").strip()).strip()
        if service._text_is_noisy_ocr(combined_text):
            combined_text = ""
        visual_cues = " ".join(
            str(getattr(panel, "visual_caption", "") or "").strip()
            for panel in panels
            if str(getattr(panel, "visual_caption", "") or "").strip()
        ).strip()
        character_names = []
        seen_names: set[str] = set()
        for panel in panels:
            for raw_name in getattr(panel, "character_names", None) or []:
                name = str(raw_name or "").strip()
                key = name.casefold()
                if name and key not in seen_names:
                    seen_names.add(key)
                    character_names.append(name)
        units.append(
            {
                "segment_id": segment.id,
                "scene_id": scene_id,
                "sequence_in_scene": scene_offsets[scene_id],
                "scene_unit_count": scene_counts[scene_id],
                "panel_start": int(segment.panel_start or 0),
                "panel_end": int(segment.panel_end or 0),
                "panel_count": len(panel_ids),
                "panel_ids": panel_ids,
                "character_names": character_names,
                "combined_text": combined_text,
                "visual_cues": visual_cues,
                "scene_summary": scene_summary_lookup.get(scene_id, ""),
            }
        )
    return units, panels_by_id


def main() -> int:
    parser = argparse.ArgumentParser(description="Repair existing story-first segments with the multimodal rescue pass.")
    parser.add_argument("project_id")
    parser.add_argument("--mode", choices=["repair", "redraft", "cleanup"], default="redraft")
    args = parser.parse_args()

    store = ProjectStore()
    project = store.get_project(args.project_id)
    if not project.story_segments:
        raise SystemExit(f"{args.project_id} has no story_segments to repair.")

    project_dir = store._project_dir(args.project_id)
    service = StoryScriptService(router=LLMRouter())
    # Use the production repair service's vision-aware unit builder. The local
    # legacy builder only sees OCR/old captions and will wrongly blank
    # vision-backed hybrid segments whose OCR is sparse or noisy.
    repair_service = StorySegmentRepairService(store=store, story_service=service)
    units, panels_by_id = repair_service._build_units(
        args.project_id,
        sorted(project.story_segments, key=lambda item: item.order),
    )
    scene_visual_paths = service._build_scene_visual_paths(
        units,
        panels_by_id,
        project_dir / "panels",
        project_dir / "output" / "scene_visuals_repair",
    )
    story_bible = _load_json(project_dir / "output" / "story_bible.json")
    grounding_state = _load_json(project_dir / "output" / "story_grounding.json")
    character_dictionary = _load_json(project_dir / "output" / "character_dictionary.json")
    scene_summary_payload = _load_json(project_dir / "output" / "scene_summaries.json")
    existing_story_text = (project_dir / "output" / "narration_story.txt").read_text(encoding="utf-8").strip() if (project_dir / "output" / "narration_story.txt").exists() else ""
    chapter_summary = str(scene_summary_payload.get("chapter_summary") or "").strip()
    if not chapter_summary:
        chapter_summary = str(story_bible.get("chapter_premise") or "").strip()
    if not chapter_summary:
        chapter_summary = " ".join(part.strip() for part in existing_story_text.splitlines()[:2] if part.strip())[:700].strip()
    protagonist_name = str(grounding_state.get("protagonist_name") or "").strip() or None

    chapter_metadata_payload = service._chapter_metadata_payload(project.chapter_metadata)
    if args.mode == "redraft":
        draft_lines = service._draft_scene_lines(
            units,
            project_title=project.name or "",
            chapter_metadata=chapter_metadata_payload,
            chapter_summary=chapter_summary,
            character_dictionary=character_dictionary,
            protagonist_name=protagonist_name,
            story_bible=story_bible,
            scene_visual_paths=scene_visual_paths,
            name_grounding=grounding_state,
        )
        polished_lines = service.polisher.polish(
            draft_lines,
            chapter_summary,
            character_dictionary,
            project_title=project.name or "",
            chapter_metadata=chapter_metadata_payload,
            slot_evidence=service._slot_evidence(units, draft_lines),
        )
        if len(polished_lines) != len(units):
            polished_lines = list(draft_lines)
        reviewed_payloads = service._critic_scene_lines(
            polished_lines,
            units,
            project_title=project.name or "",
            chapter_metadata=chapter_metadata_payload,
            chapter_summary=chapter_summary,
            character_dictionary=character_dictionary,
            protagonist_name=protagonist_name,
            story_bible=story_bible,
            name_grounding=grounding_state,
            scene_visual_paths=scene_visual_paths,
        )
        reviewed_payloads = service._style_spoken_segment_payloads(
            reviewed_payloads,
            units,
            project_title=project.name or "",
            chapter_metadata=chapter_metadata_payload,
            chapter_summary=chapter_summary,
            character_dictionary=character_dictionary,
            story_bible=story_bible,
            name_grounding=grounding_state,
        )
        reviewed_payloads = service._stabilize_reviewed_segments(
            reviewed_payloads,
            units,
            protagonist_name,
            grounding_state,
            story_bible,
        )
        repaired_segments = service._build_story_segments(units, reviewed_payloads)
    elif args.mode == "cleanup":
        original_lines = [segment.text.strip() for segment in sorted(project.story_segments, key=lambda item: item.order)]
        payloads = service._apply_weak_scene_policy(original_lines, units, protagonist_name)
        payloads = service._recover_visual_only_payloads_multimodal(
            payloads,
            units,
            project_title=project.name or "",
            chapter_metadata=chapter_metadata_payload,
            chapter_summary=chapter_summary,
            character_dictionary=character_dictionary,
            protagonist_name=protagonist_name,
            story_bible=story_bible,
            name_grounding=grounding_state,
            scene_visual_paths=scene_visual_paths,
        )
        stabilized = service._stabilize_reviewed_segments(payloads, units, protagonist_name, grounding_state, story_bible)
        styled = service._style_spoken_segment_payloads(
            stabilized,
            units,
            project_title=project.name or "",
            chapter_metadata=chapter_metadata_payload,
            chapter_summary=chapter_summary,
            character_dictionary=character_dictionary,
            story_bible=story_bible,
            name_grounding=grounding_state,
        )
        stabilized = service._stabilize_reviewed_segments(styled, units, protagonist_name, grounding_state, story_bible)
        stabilized = service._fill_blank_story_payloads(
            stabilized,
            units,
            protagonist_name=protagonist_name,
            grounding=grounding_state,
            story_bible=story_bible,
        )
        stabilized = service._reinforce_multi_sentence_scene_payloads(
            stabilized,
            units,
            protagonist_name=protagonist_name,
            grounding=grounding_state,
            story_bible=story_bible,
        )
        stabilized = service._expand_short_scene_payloads_with_llm(
            stabilized,
            units,
            project_title=project.name or "",
            chapter_metadata=chapter_metadata_payload,
            chapter_summary=chapter_summary,
            character_dictionary=character_dictionary,
            protagonist_name=protagonist_name,
            story_bible=story_bible,
            name_grounding=grounding_state,
        )
        stabilized = service._apply_weak_scene_policy(
            [str(payload.get("text") or "") for payload in stabilized],
            units,
            protagonist_name,
        )
        stabilized = service._fill_blank_story_payloads(
            stabilized,
            units,
            protagonist_name=protagonist_name,
            grounding=grounding_state,
            story_bible=story_bible,
        )
        stabilized = service._remove_overused_generic_sentences(stabilized)
        stabilized = service._collapse_internal_duplicate_sentences(stabilized)
        stabilized = service._collapse_near_duplicate_segments(
            stabilized,
            units,
            blank_unresolved=False,
        )
        stabilized = service._fill_blank_story_payloads(
            stabilized,
            units,
            protagonist_name=protagonist_name,
            grounding=grounding_state,
            story_bible=story_bible,
        )
        stabilized = service._collapse_near_duplicate_segments(
            stabilized,
            units,
            blank_unresolved=False,
        )
        stabilized = service._fill_blank_story_payloads(
            stabilized,
            units,
            protagonist_name=protagonist_name,
            grounding=grounding_state,
            story_bible=story_bible,
        )
        repaired_segments = service._build_story_segments(units, stabilized)
    else:
        original_lines = [segment.text.strip() for segment in sorted(project.story_segments, key=lambda item: item.order)]
        repaired_lines = service._rescue_scene_lines_multimodal(
            original_lines,
            units,
            project_title=project.name or "",
            chapter_metadata=chapter_metadata_payload,
            chapter_summary=chapter_summary,
            character_dictionary=character_dictionary,
            protagonist_name=protagonist_name,
            story_bible=story_bible,
            name_grounding=grounding_state,
            scene_visual_paths=scene_visual_paths,
        )
        payloads = service._apply_weak_scene_policy(repaired_lines, units, protagonist_name)
        payloads = service._recover_visual_only_payloads_multimodal(
            payloads,
            units,
            project_title=project.name or "",
            chapter_metadata=chapter_metadata_payload,
            chapter_summary=chapter_summary,
            character_dictionary=character_dictionary,
            protagonist_name=protagonist_name,
            story_bible=story_bible,
            name_grounding=grounding_state,
            scene_visual_paths=scene_visual_paths,
        )
        stabilized = service._stabilize_reviewed_segments(payloads, units, protagonist_name, grounding_state, story_bible)
        styled = service._style_spoken_segment_payloads(
            stabilized,
            units,
            project_title=project.name or "",
            chapter_metadata=chapter_metadata_payload,
            chapter_summary=chapter_summary,
            character_dictionary=character_dictionary,
            story_bible=story_bible,
            name_grounding=grounding_state,
        )
        stabilized = service._stabilize_reviewed_segments(styled, units, protagonist_name, grounding_state, story_bible)
        repaired_segments = service._build_story_segments(units, stabilized)
    repaired_story = service._compose_story_text(repaired_segments)

    store.save_story_segments(args.project_id, repaired_segments, story_block=repaired_story)
    if repaired_story:
        (project_dir / "output" / "narration_story.txt").write_text(repaired_story.strip() + "\n", encoding="utf-8")

    report = store.load_script_quality_report(args.project_id)
    print(
        json.dumps(
            {
                "project_id": args.project_id,
                "mode": args.mode,
                "segment_count": len(repaired_segments),
                "spoken_segments": sum(1 for segment in repaired_segments if segment.text.strip()),
                "visual_only_segments": sum(1 for segment in repaired_segments if segment.visual_only),
                "quality_score": report.get("quality_score"),
                "should_block_tts": report.get("should_block_tts"),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
