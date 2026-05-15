from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime
import hashlib
import json
import re
import time
from pathlib import Path
from typing import Any

from app.schemas.project import ChapterMetadata, PanelBox, StorySegment
from app.services.character_name_filters import looks_like_false_character_name
from app.services.llm_router import LLMRouter
from app.services.script_quality_service import ScriptQualityService
from app.utils.files import ensure_dir, read_json, write_json


VNEXT_ARTIFACT_VERSION = "script_vnext_scene_plan_v1"
VNEXT_REDRAFT_PROMPT_VERSION = "vnext_scene_redraft_v1"
VNEXT_REDRAFT_CACHE_VERSION = "vnext_scene_redraft_cache_v1"

INTERNAL_TEMPLATE_PHRASES: tuple[str, ...] = (
    "The story moves through",
    "The surrounding detail is folded",
    "carries a moment through",
    "anchors the moment",
    "the important line is",
    "the next detail adds",
    "keeps the moment connected",
    "as the dialogue and visuals converge",
    "Together, the moment moves from",
    "beat built around",
    "That line pushes the exchange",
    "without adding unsupported details",
    "supporting panels stay",
    "giving the beat a clear start and consequence",
    "The same exchange also includes",
    "That evidence leads into",
    "The scene is caught in an exchange where",
    "The scene is pulled into",
    "The exchange turns on",
    "The next image shows",
)

GENERIC_CHARACTER_LABELS: tuple[str, ...] = (
    "the green-haired student",
    "the grey-haired character",
    "the gray-haired character",
    "the red-haired character",
    "another student",
    "the other student",
    "the other person",
    "a male student",
    "the aggressor",
    "the person",
    "a character",
    "someone",
)


@dataclass(frozen=True)
class ScriptVNextRedraftConfig:
    enabled: bool = False
    dry_run: bool = False
    max_calls: int = 4
    max_scenes_per_batch: int = 4
    max_prompt_chars: int = 12_000
    max_output_tokens: int = 1800
    max_estimated_cost_usd: float = 0.0
    style_threshold: int = 68
    provider: str = "gemini"


@dataclass(frozen=True)
class ScriptVNextResult:
    story_segments: list[StorySegment]
    story_text: str
    scene_plan: dict[str, Any]
    narration_chunks: dict[str, Any]
    qc_report: dict[str, Any]
    cost_report: dict[str, Any]
    final_script_path: Path
    story_context_pack: dict[str, Any]


class ScriptGenerationVNextService:
    """Scene-level script generation using existing upstream artifacts only.

    This intentionally does not run panel detection, OCR, text-region detection,
    character portraiting, or external adapters. It consumes the best artifacts
    already present on disk and writes a new script-generation artifact set.
    """

    def __init__(self, redraft_client: Any | None = None) -> None:
        self.redraft_client = redraft_client
        self.router = LLMRouter()

    def run(
        self,
        *,
        project_id: str,
        project_name: str,
        project_dir: Path,
        chapter_metadata: ChapterMetadata,
        panels: list[PanelBox],
        job_id: str | None = None,
        max_cost_usd: float = 0.0,
        redraft_config: ScriptVNextRedraftConfig | None = None,
    ) -> ScriptVNextResult:
        started = time.perf_counter()
        output_dir = ensure_dir(project_dir / "output" / "script_vnext")
        evidence = self._load_evidence(project_dir, panels)
        character_registry = self._load_character_registry(project_dir)
        kept_panels = [
            panel
            for panel in sorted(panels, key=lambda item: (int(item.order), int(item.page), int(item.panel)))
            if bool(panel.keep) and not bool(panel.auto_skipped)
        ]
        scene_plan = self._build_scene_plan(
            project_id=project_id,
            project_name=project_name,
            chapter_metadata=chapter_metadata,
            panels=kept_panels,
            evidence=evidence,
            character_registry=character_registry,
            job_id=job_id,
        )
        story_context_pack = self._build_story_context_pack(
            project_dir=project_dir,
            scene_plan=scene_plan,
            evidence=evidence,
            character_registry=character_registry,
            chapter_metadata=chapter_metadata,
        )
        narration_chunks = self._build_narration_chunks(scene_plan)
        narration_chunks = self._apply_character_name_substitutions(narration_chunks, scene_plan, story_context_pack)
        story_segments = self._story_segments_from_chunks(narration_chunks)
        story_text = "\n\n".join(chunk["text"].strip() for chunk in narration_chunks["chunks"] if chunk["text"].strip())

        panel_vision_records = self._read_panel_vision_records(project_dir)
        panel_evidence_records = [
            *self._read_panel_evidence_records(project_dir),
            *[
                self._scene_usage_evidence_record(scene, panel_id, item)
                for scene in scene_plan["scenes"]
                for panel_id, item in scene["panel_contribution_map"].items()
            ],
        ]
        base_qc = ScriptQualityService().analyze_story_segments(
            story_segments,
            panel_vision_records=panel_vision_records,
            panel_evidence_records=panel_evidence_records,
            panels=panels,
        )
        qc_report = self._augment_qc_report(base_qc, scene_plan, narration_chunks)
        qc_report = self._apply_vnext_style_qc(qc_report, scene_plan, narration_chunks, story_context_pack)
        redraft_log = self._empty_redraft_log(redraft_config)
        redraft_config = redraft_config or ScriptVNextRedraftConfig()
        if redraft_config.enabled or redraft_config.dry_run:
            redraft_result = self._run_scene_redraft_pass(
                project_dir=project_dir,
                scene_plan=scene_plan,
                narration_chunks=narration_chunks,
                qc_report=qc_report,
                config=redraft_config,
                story_context_pack=story_context_pack,
            )
            narration_chunks = redraft_result["narration_chunks"]
            narration_chunks = self._apply_character_name_substitutions(narration_chunks, scene_plan, story_context_pack)
            redraft_log = redraft_result["redraft_log"]
            story_segments = self._story_segments_from_chunks(narration_chunks)
            story_text = "\n\n".join(chunk["text"].strip() for chunk in narration_chunks["chunks"] if chunk["text"].strip())
            base_qc = ScriptQualityService().analyze_story_segments(
                story_segments,
                panel_vision_records=panel_vision_records,
                panel_evidence_records=panel_evidence_records,
                panels=panels,
            )
            qc_report = self._augment_qc_report(base_qc, scene_plan, narration_chunks)
            qc_report = self._apply_vnext_style_qc(qc_report, scene_plan, narration_chunks, story_context_pack)
            qc_report["redraft"] = self._redraft_qc_summary(redraft_log)
            qc_report = self._apply_redraft_qc_gate(qc_report, redraft_log)
        cost_report = self._build_cost_report(
            started_at=started,
            max_cost_usd=max_cost_usd,
            scene_count=len(scene_plan["scenes"]),
            redraft_log=redraft_log,
        )

        final_script_path = output_dir / "final_script.md"
        final_script_path.write_text(story_text.strip() + ("\n" if story_text.strip() else ""), encoding="utf-8")
        write_json(output_dir / "scene_plan.json", scene_plan)
        write_json(output_dir / "narration_chunks.json", narration_chunks)
        write_json(output_dir / "qc_report.json", qc_report)
        write_json(output_dir / "cost_report.json", cost_report)
        write_json(output_dir / "benchmark_report.json", self._build_benchmark_report(scene_plan, qc_report, cost_report))
        write_json(output_dir / "redraft_log.json", redraft_log)
        write_json(output_dir / "story_context_pack.json", story_context_pack)
        write_json(output_dir / "generalization_audit.json", self._build_generalization_audit(story_context_pack))

        return ScriptVNextResult(
            story_segments=story_segments,
            story_text=story_text,
            scene_plan=scene_plan,
            narration_chunks=narration_chunks,
            qc_report=qc_report,
            cost_report=cost_report,
            final_script_path=final_script_path,
            story_context_pack=story_context_pack,
        )

    def _load_evidence(self, project_dir: Path, panels: list[PanelBox]) -> dict[str, dict[str, Any]]:
        evidence: dict[str, dict[str, Any]] = {
            panel.id: {
                "panel_id": panel.id,
                "panel_order": int(panel.order),
                "page": int(panel.page),
                "dialogue": str(panel.ocr_text or "").strip(),
                "caption": "",
                "action_beat": str(panel.visual_caption or "").strip(),
                "emotion": "",
                "character_names": [],
                "character_roles": {},
                "visual_only": not bool(str(panel.ocr_text or panel.visual_caption or "").strip()),
                "source_artifacts": ["panels.json"],
            }
            for panel in panels
        }
        for record in self._read_panel_evidence_records(project_dir):
            panel_id = str(record.get("panel_id") or "").strip()
            order = self._safe_int(record.get("panel_order") or record.get("order"))
            target = evidence.get(panel_id) or self._evidence_by_order(evidence, order)
            if not target:
                continue
            for source_key, target_key in (
                ("dialogue_text", "dialogue"),
                ("text_english", "dialogue"),
                ("cleaned_text", "dialogue"),
                ("caption_text", "caption"),
                ("caption", "caption"),
            ):
                self._append_text(target, target_key, record.get(source_key))
            target.setdefault("source_artifacts", []).append("panel_evidence.json")

        transcript = read_json(project_dir / "output" / "transcript.json", default={})
        fragments = transcript.get("fragments", []) if isinstance(transcript, dict) else []
        for fragment in fragments if isinstance(fragments, list) else []:
            if not isinstance(fragment, dict) or not bool(fragment.get("accepted", True)):
                continue
            panel_id = str(fragment.get("panel_id") or "").strip()
            order = self._safe_int(fragment.get("panel_order") or fragment.get("panel"))
            target = evidence.get(panel_id) or self._evidence_by_order(evidence, order)
            if not target:
                continue
            category = str(fragment.get("classification") or fragment.get("category") or "dialogue").casefold()
            text = fragment.get("cleaned_text") or fragment.get("text") or fragment.get("raw_text")
            if category in {"sfx", "ui", "watermark", "credit", "garbage", "low_confidence"}:
                target.setdefault("rejected_text", []).append(str(text or "").strip())
                continue
            self._append_text(target, "caption" if category == "narration" else "dialogue", text)
            target.setdefault("source_artifacts", []).append("transcript.json")

        for record in self._read_panel_vision_records(project_dir):
            panel_id = str(record.get("panel_id") or "").strip()
            order = self._safe_int(record.get("panel_order"))
            target = evidence.get(panel_id) or self._evidence_by_order(evidence, order)
            if not target:
                continue
            for key in ("dialogue", "caption", "action_beat", "emotion"):
                self._append_text(target, key, record.get(key))
            names = [
                str(name).strip()
                for name in record.get("character_names", []) or []
                if str(name).strip() and not looks_like_false_character_name(name)
            ]
            if names:
                target["character_names"] = list(dict.fromkeys([*target.get("character_names", []), *names]))
            roles = record.get("character_roles") if isinstance(record.get("character_roles"), dict) else {}
            if roles:
                target["character_roles"] = roles
            target["visual_only"] = bool(record.get("visual_only")) and not bool(target.get("dialogue") or target.get("caption") or target.get("action_beat"))
            target["confidence"] = max(float(target.get("confidence") or 0.0), float(record.get("confidence") or 0.0))
            target.setdefault("source_artifacts", []).append("panel_vision_final.json")
        return evidence

    def _build_scene_plan(
        self,
        *,
        project_id: str,
        project_name: str,
        chapter_metadata: ChapterMetadata,
        panels: list[PanelBox],
        evidence: dict[str, dict[str, Any]],
        character_registry: dict[str, Any],
        job_id: str | None,
    ) -> dict[str, Any]:
        scenes: list[dict[str, Any]] = []
        current: list[PanelBox] = []
        current_kind = ""
        for panel in panels:
            panel_evidence = evidence.get(panel.id, {})
            kind = self._panel_kind(panel_evidence)
            if current and self._should_start_new_scene(current, current_kind, panel, kind, evidence):
                scenes.append(self._scene_from_panels(len(scenes) + 1, current, evidence))
                current = []
            current.append(panel)
            current_kind = kind
        if current:
            scenes.append(self._scene_from_panels(len(scenes) + 1, current, evidence))

        return {
            "artifact_version": VNEXT_ARTIFACT_VERSION,
            "project_id": project_id,
            "project_name": project_name,
            "job_id": job_id,
            "created_at": datetime.utcnow().isoformat(),
            "source_stage": "script_generation",
            "status": "completed",
            "chapter_metadata": chapter_metadata.model_dump(mode="json"),
            "input_artifacts": self._input_artifact_statuses(evidence, character_registry),
            "character_registry": character_registry,
            "scenes": scenes,
            "summary": {
                "scene_count": len(scenes),
                "source_panel_count": sum(len(scene["source_panel_ids"]) for scene in scenes),
                "meaningful_panel_count": sum(
                    1
                    for scene in scenes
                    for item in scene["panel_contribution_map"].values()
                    if item["contribution"] not in {"low_information", "redundant_near_duplicate"}
                ),
                "visual_only_panel_count": sum(
                    1
                    for scene in scenes
                    for item in scene["panel_contribution_map"].values()
                    if item.get("visual_only")
                ),
            },
        }

    def _scene_from_panels(self, scene_number: int, panels: list[PanelBox], evidence: dict[str, dict[str, Any]]) -> dict[str, Any]:
        panel_ids = [panel.id for panel in panels]
        representative = self._representative_panel(panels, evidence)
        contribution_map: dict[str, dict[str, Any]] = {}
        transcript_snippets: list[str] = []
        visible_characters: list[str] = []
        speakers: list[str] = []
        mentioned: list[str] = []
        evidence_phrases: list[str] = []

        previous_tokens: set[str] = set()
        for panel in panels:
            item = evidence.get(panel.id, {})
            text = self._panel_dialogue_text(item)
            visual = self._panel_visual_text(item)
            tokens = self._content_tokens(" ".join([text, visual]))
            contribution = self._classify_contribution(text, visual, tokens, previous_tokens)
            if contribution not in {"low_information", "redundant_near_duplicate"}:
                evidence_phrases.append(self._short_evidence_phrase(text, visual))
            if text:
                transcript_snippets.append(text)
            for name in item.get("character_names", []) or []:
                if looks_like_false_character_name(name):
                    continue
                roles = item.get("character_roles", {}).get(name, []) if isinstance(item.get("character_roles"), dict) else []
                if "mentioned_absent" in roles:
                    mentioned.append(name)
                else:
                    visible_characters.append(name)
            speaker = str(item.get("speaker") or "").strip()
            if speaker and speaker.casefold() not in {"unknown", "unseen speaker"} and not looks_like_false_character_name(speaker):
                speakers.append(speaker)
            contribution_map[panel.id] = {
                "panel_order": int(panel.order),
                "page": int(panel.page),
                "contribution": contribution,
                "reason": self._contribution_reason(contribution),
                "evidence_text": text[:300],
                "visual_summary": visual[:300],
                "visual_only": not bool(text) and bool(visual),
            }
            if tokens:
                previous_tokens = tokens

        visible_unique = list(dict.fromkeys(visible_characters))
        speaker_unique = list(dict.fromkeys(speakers))
        mentioned_unique = [name for name in dict.fromkeys(mentioned) if name not in visible_unique]
        scene_summary = self._scene_summary(
            panel_count=len(panels),
            visible_characters=visible_unique,
            transcript_snippets=transcript_snippets,
            evidence_phrases=evidence_phrases,
            fallback_kind=self._panel_kind(evidence.get(panels[0].id, {})) if panels else "scene",
        )
        visual_duration = round(sum(float(panel.duration_seconds or self._default_panel_duration(evidence.get(panel.id, {}))) for panel in panels), 2)
        target_duration = max(visual_duration * 0.90, 4.5)
        return {
            "scene_id": f"scene_{scene_number:03d}",
            "scene_number": scene_number,
            "panel_start": int(panels[0].order),
            "panel_end": int(panels[-1].order),
            "source_panel_ids": panel_ids,
            "representative_panel_id": representative.id,
            "supporting_panel_ids": [panel.id for panel in panels if panel.id != representative.id],
            "transcript_snippets": transcript_snippets[:12],
            "visible_characters": visible_unique,
            "speakers": speaker_unique,
            "mentioned_characters": mentioned_unique,
            "character_roles": self._scene_character_roles(visible_unique, speaker_unique, mentioned_unique),
            "scene_summary": scene_summary,
            "panel_contribution_map": contribution_map,
            "visual_duration_seconds": visual_duration,
            "target_narration_duration_seconds": round(target_duration, 2),
        }

    def _build_narration_chunks(self, scene_plan: dict[str, Any]) -> dict[str, Any]:
        chunks: list[dict[str, Any]] = []
        for scene in scene_plan["scenes"]:
            text = self._narrate_scene(scene)
            estimated = self._estimate_duration(text)
            scene_duration = float(scene["visual_duration_seconds"])
            gap = round(max(scene_duration - estimated, 0.0), 2)
            repair_action = "none"
            if gap > 2.0:
                expanded = self._expand_for_timing(text, scene)
                expanded_duration = self._estimate_duration(expanded)
                if expanded_duration > estimated:
                    text = expanded
                    estimated = expanded_duration
                    gap = round(max(scene_duration - estimated, 0.0), 2)
                    repair_action = "expanded_with_supporting_panel_evidence"
                if gap > 2.0 and not self._scene_has_enough_grounded_content(scene):
                    scene_duration = round(max(estimated + 1.0, 2.0), 2)
                    gap = round(max(scene_duration - estimated, 0.0), 2)
                    repair_action = "reduced_visual_duration_for_low_information_scene"
                elif gap > 2.0:
                    scene_duration = round(max(estimated + 1.5, 2.0), 2)
                    gap = round(max(scene_duration - estimated, 0.0), 2)
                    repair_action = "reduced_visual_duration_after_grounded_expansion_limit"
            chunks.append(
                {
                    "chunk_id": scene["scene_id"],
                    "scene_id": scene["scene_id"],
                    "text": text,
                    "source_panel_ids": scene["source_panel_ids"],
                    "panel_start": scene["panel_start"],
                    "panel_end": scene["panel_end"],
                    "representative_panel_id": scene["representative_panel_id"],
                    "supporting_panel_ids": scene["supporting_panel_ids"],
                    "scene_duration_seconds": scene_duration,
                    "estimated_narration_duration_seconds": estimated,
                    "tts_duration_seconds": None,
                    "duration_gap_seconds": gap,
                    "needs_narration_expansion": gap > 2.0,
                    "repair_action": repair_action if gap <= 2.0 else "unresolved_duration_gap",
                    "intentional_silence": False,
                }
            )
        return {
            "artifact_version": VNEXT_ARTIFACT_VERSION,
            "created_at": datetime.utcnow().isoformat(),
            "chunks": chunks,
            "summary": {
                "chunk_count": len(chunks),
                "long_gap_count": sum(1 for chunk in chunks if float(chunk["duration_gap_seconds"]) > 2.0),
                "largest_gap_seconds": max([float(chunk["duration_gap_seconds"]) for chunk in chunks] or [0.0]),
            },
        }

    def _augment_qc_report(self, base_qc: dict[str, Any], scene_plan: dict[str, Any], narration_chunks: dict[str, Any]) -> dict[str, Any]:
        qc = dict(base_qc)
        long_gaps = [chunk for chunk in narration_chunks["chunks"] if float(chunk["duration_gap_seconds"]) > 2.0]
        unresolved = [
            panel_id
            for scene in scene_plan["scenes"]
            for panel_id, item in scene["panel_contribution_map"].items()
            if item["contribution"] == "unresolved_unused_error"
        ]
        qc.update(
            {
                "artifact_version": VNEXT_ARTIFACT_VERSION,
                "analysis_mode": "script_vnext_scene_level",
                "technical_coverage_score": int(qc.get("story_continuity_score", 100)),
                "meaningful_panel_usage_score": int(qc.get("meaningful_usage_score", 100)),
                "ocr_quality_score": 100 - min(int(qc.get("ocr_garbage_leak_lines", 0) or 0) * 20, 80),
                "chronology_score": 100 - min(int(qc.get("panel_order_regressions", 0) or 0) * 30, 100),
                "narration_timing_score": 100 - min(len(long_gaps) * 15, 90),
                "story_quality_score": int(qc.get("quality_score", 0)),
                "cost_score": 100,
                "long_no_tts_gap_count": len(long_gaps),
                "unresolved_unused_panel_ids": unresolved[:200],
            }
        )
        qc["should_block_tts"] = bool(qc.get("should_block_tts")) or bool(long_gaps) or bool(unresolved)
        failure_codes = set(qc.get("failure_codes") or [])
        if long_gaps:
            failure_codes.add("long_no_tts_gaps")
        else:
            failure_codes.discard("long_unintentional_gaps")
            failure_codes.discard("long_no_tts_gaps")
        if unresolved:
            failure_codes.add("unresolved_unused_panels")
        qc["failure_codes"] = sorted(failure_codes)
        qc["summary"] = self._qc_summary(qc)
        return qc

    def _build_story_context_pack(
        self,
        *,
        project_dir: Path,
        scene_plan: dict[str, Any],
        evidence: dict[str, dict[str, Any]],
        character_registry: dict[str, Any],
        chapter_metadata: ChapterMetadata,
    ) -> dict[str, Any]:
        scenes = [scene for scene in scene_plan.get("scenes", []) or [] if isinstance(scene, dict)]
        supplemental_context = self._load_story_context_seed_artifacts(project_dir)
        registry_candidates = [*self._registry_character_candidates(character_registry), *supplemental_context["characters"]]
        name_counts: dict[str, int] = {}
        name_display: dict[str, str] = {}
        name_aliases: dict[str, set[str]] = {}
        name_roles: dict[str, set[str]] = {}
        name_refs: dict[str, list[dict[str, Any]]] = {}
        rejected_names: set[str] = set()

        for candidate in registry_candidates:
            name = str(candidate.get("name") or "").strip()
            if not self._is_valid_context_name(name):
                if name:
                    rejected_names.add(name)
                continue
            key = self._term_key(name)
            name_display.setdefault(key, name)
            name_counts[key] = max(name_counts.get(key, 0), 2)
            for alias in candidate.get("aliases", []) or []:
                alias_text = str(alias or "").strip()
                if self._is_valid_context_name(alias_text):
                    name_aliases.setdefault(key, set()).add(alias_text)
                elif alias_text:
                    rejected_names.add(alias_text)
            confidence = float(candidate.get("confidence") or 0.75)
            name_refs.setdefault(key, []).append({"source": character_registry.get("source", "character_registry"), "confidence": round(confidence, 3)})

        for scene in scenes:
            scene_id = str(scene.get("scene_id") or "")
            panel_refs = list(scene.get("source_panel_ids") or [])
            for bucket, role in (("visible_characters", "visible_present"), ("speakers", "speaker"), ("mentioned_characters", "mentioned_absent")):
                for raw_name in scene.get(bucket, []) or []:
                    name = str(raw_name or "").strip()
                    if not self._is_valid_context_name(name):
                        if name:
                            rejected_names.add(name)
                        continue
                    key = self._term_key(name)
                    name_display.setdefault(key, name)
                    name_counts[key] = name_counts.get(key, 0) + (2 if role != "mentioned_absent" else 1)
                    name_roles.setdefault(key, set()).add(role)
                    name_refs.setdefault(key, []).append({"scene_id": scene_id, "panel_ids": panel_refs[:4], "role": role})

        main_characters: list[dict[str, Any]] = []
        for key, count in sorted(name_counts.items(), key=lambda item: (-item[1], name_display.get(item[0], "")))[:12]:
            name = name_display[key]
            confidence = min(0.99, 0.55 + min(count, 12) * 0.035)
            registry_ref = next((ref for ref in name_refs.get(key, []) if ref.get("source")), None)
            if registry_ref:
                confidence = max(confidence, float(registry_ref.get("confidence") or 0.75))
            main_characters.append(
                {
                    "name": name,
                    "confidence": round(confidence, 3),
                    "aliases": sorted(name_aliases.get(key, set())),
                    "rejected_aliases": sorted(alias for alias in rejected_names if self._term_key(alias) == key),
                    "roles": sorted(name_roles.get(key, set()) or {"visible_present"}),
                    "evidence_refs": name_refs.get(key, [])[:8],
                }
            )

        combined_evidence_parts: list[str] = []
        rejected_terms: set[str] = set(rejected_names)
        combined_evidence_parts.extend(supplemental_context["text"])
        for item in evidence.values():
            combined_evidence_parts.extend(
                str(item.get(key) or "")
                for key in ("dialogue", "caption", "action_beat", "emotion")
                if str(item.get(key) or "").strip()
            )
            for rejected in item.get("rejected_text", []) or []:
                text = str(rejected or "").strip()
                if text:
                    rejected_terms.add(text)
        for scene in scenes:
            combined_evidence_parts.extend(str(text) for text in scene.get("transcript_snippets", []) or [])
            combined_evidence_parts.append(str(scene.get("scene_summary") or ""))
            for item in (scene.get("panel_contribution_map") or {}).values():
                combined_evidence_parts.append(str(item.get("evidence_text") or ""))
                combined_evidence_parts.append(str(item.get("visual_summary") or ""))

        combined_evidence = self._clean_text(" ".join(combined_evidence_parts))
        character_names = {self._term_key(item["name"]) for item in main_characters}
        special_terms = list(
            dict.fromkeys(
                [
                    *[
                        term
                        for term in supplemental_context["special_terms"]
                        if self._is_valid_special_term(term, curated=True)
                        and not any(name_key and name_key in self._term_key(term) for name_key in character_names)
                    ],
                    *self._extract_context_terms(combined_evidence, character_names),
                ]
            )
        )
        relationships = self._infer_relationships(scenes, main_characters)
        premise, premise_refs = self._infer_project_premise(scenes)
        current_state, state_refs = self._infer_current_story_state(scenes)
        timeline_markers = self._infer_timeline_markers(scenes)
        unresolved_questions = self._infer_unresolved_questions(scenes)
        style_notes = self._infer_style_tone_notes(combined_evidence)

        preserve_terms = list(dict.fromkeys([item["name"] for item in main_characters[:8]] + special_terms[:16]))
        rejected_sorted = sorted(
            term
            for term in rejected_terms
            if term and (looks_like_false_character_name(term) or len(term.split()) > 3 or not self._is_valid_context_name(term))
        )[:60]
        return {
            "artifact_version": "script_vnext_story_context_pack_v1",
            "created_at": datetime.utcnow().isoformat(),
            "source_artifacts": {
                "transcript": str((project_dir / "output" / "transcript.json").name),
                "character_registry": character_registry.get("source", "none"),
                "scene_plan": "output/script_vnext/scene_plan.json",
                "chapter_metadata_present": bool(chapter_metadata.model_dump(mode="json")),
            },
            "project_premise": {"text": premise, "evidence_refs": premise_refs},
            "main_characters": main_characters,
            "stable_aliases": {item["name"]: item.get("aliases", []) for item in main_characters if item.get("aliases")},
            "rejected_invalid_aliases": rejected_sorted,
            "character_roles_relationships": relationships,
            "special_terms": special_terms[:24],
            "current_story_state": {"text": current_state, "evidence_refs": state_refs},
            "timeline_flashback_markers": timeline_markers,
            "important_unresolved_questions": unresolved_questions[:12],
            "preserve_terms_exact": preserve_terms,
            "reject_terms": rejected_sorted,
            "style_tone_notes": style_notes,
            "evidence_references": {
                "scene_count": len(scenes),
                "panel_count": sum(len(scene.get("source_panel_ids") or []) for scene in scenes),
                "premise_scene_ids": [ref.get("scene_id") for ref in premise_refs],
            },
        }

    def _apply_character_name_substitutions(
        self,
        narration_chunks: dict[str, Any],
        scene_plan: dict[str, Any],
        story_context_pack: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        story_context_pack = story_context_pack or {}
        scenes_by_id = {str(scene.get("scene_id")): scene for scene in scene_plan.get("scenes", []) or [] if isinstance(scene, dict)}
        updated_chunks: list[dict[str, Any]] = []
        for chunk in narration_chunks.get("chunks", []) or []:
            if not isinstance(chunk, dict):
                continue
            scene = scenes_by_id.get(str(chunk.get("scene_id") or chunk.get("chunk_id"))) or {}
            text = str(chunk.get("text") or "")
            known_names = self._known_scene_character_names(scene, story_context_pack)
            text = self._clean_final_narration_text(text)
            if len(known_names) == 1:
                name = known_names[0]
                for label in GENERIC_CHARACTER_LABELS:
                    text = re.sub(rf"\b{re.escape(label)}\b", name, text, flags=re.I)
            elif len(known_names) > 1:
                primary = known_names[0]
                text = re.sub(r"(^|[.!?]\s+)(He|She|They)\b", lambda match: f"{match.group(1)}{primary}", text)
            updated = dict(chunk)
            updated["text"] = self._clean_text(text)
            updated_chunks.append(updated)
        return {**narration_chunks, "chunks": updated_chunks}

    def _apply_vnext_style_qc(
        self,
        qc_report: dict[str, Any],
        scene_plan: dict[str, Any],
        narration_chunks: dict[str, Any],
        story_context_pack: dict[str, Any],
    ) -> dict[str, Any]:
        qc = dict(qc_report)
        scenes_by_id = {str(scene.get("scene_id")): scene for scene in scene_plan.get("scenes", []) or [] if isinstance(scene, dict)}
        chunks = [chunk for chunk in narration_chunks.get("chunks", []) or [] if isinstance(chunk, dict)]
        full_text = "\n\n".join(str(chunk.get("text") or "") for chunk in chunks)
        banned_phrase_count = sum(len(re.findall(re.escape(phrase), full_text, flags=re.I)) for phrase in INTERNAL_TEMPLATE_PHRASES)
        one_sentence_count = sum(1 for chunk in chunks if len(re.findall(r"[.!?](?:\s|$)", str(chunk.get("text") or ""))) <= 1)
        opener_counts: dict[str, int] = {}
        known_name_usage_count = 0
        generic_label_count = 0
        ambiguous_pronoun_count = 0
        mentioned_absent_action_count = 0
        invalid_name_count = 0
        reject_terms = [str(term).strip() for term in story_context_pack.get("reject_terms", []) or [] if str(term).strip()]

        for chunk in chunks:
            text = str(chunk.get("text") or "")
            words = re.findall(r"\b[\w'-]+\b", text)
            if len(words) >= 4:
                opener = " ".join(words[:4]).casefold()
                opener_counts[opener] = opener_counts.get(opener, 0) + 1
            scene = scenes_by_id.get(str(chunk.get("scene_id") or chunk.get("chunk_id"))) or {}
            known_names = self._known_scene_character_names(scene, story_context_pack)
            known_name_usage_count += sum(len(re.findall(rf"\b{re.escape(name)}\b", text)) for name in known_names)
            if known_names:
                generic_label_count += sum(len(re.findall(rf"\b{re.escape(label)}\b", text, flags=re.I)) for label in GENERIC_CHARACTER_LABELS)
            if len(known_names) > 1:
                ambiguous_pronoun_count += len(re.findall(r"(^|[.!?]\s+)(?:he|she|they|him|her|them)\b", text, flags=re.I))
            mentioned_only = [
                str(name)
                for name in scene.get("mentioned_characters", []) or []
                if str(name) not in set(scene.get("visible_characters", []) or [])
            ]
            for name in mentioned_only:
                if re.search(rf"\b{re.escape(name)}\b[^.!?]{{0,80}}\b(?:attacks|hits|stands|watches|reacts|speaks|grabs|throws|moves|turns)\b", text, re.I):
                    mentioned_absent_action_count += 1
            for term in reject_terms:
                if len(term) >= 3 and re.search(rf"\b{re.escape(term)}\b", text, re.I):
                    invalid_name_count += 1

        repeated_sentence_opener_count = sum(count for count in opener_counts.values() if count >= 3)
        style_penalty = (
            banned_phrase_count * 25
            + generic_label_count * 8
            + invalid_name_count * 20
            + ambiguous_pronoun_count * 4
            + mentioned_absent_action_count * 20
            + repeated_sentence_opener_count * 3
            + max(one_sentence_count - max(1, len(chunks) // 4), 0) * 4
        )
        style_qc_score = max(0, 100 - style_penalty)
        qc["vnext_style_qc"] = {
            "style_qc_score": style_qc_score,
            "banned_template_phrase_count": banned_phrase_count,
            "generic_label_count_when_known_name_available": generic_label_count,
            "known_name_usage_count": known_name_usage_count,
            "ambiguous_pronoun_count": ambiguous_pronoun_count,
            "invalid_name_count": invalid_name_count,
            "mentioned_absent_action_count": mentioned_absent_action_count,
            "one_sentence_segment_count": one_sentence_count,
            "repeated_sentence_opener_count": repeated_sentence_opener_count,
        }
        failure_codes = set(qc.get("failure_codes") or [])
        if banned_phrase_count:
            failure_codes.add("vnext_internal_template_language")
        if generic_label_count:
            failure_codes.add("vnext_generic_labels_for_known_characters")
        if invalid_name_count:
            failure_codes.add("vnext_invalid_name_leak")
        if mentioned_absent_action_count:
            failure_codes.add("vnext_mentioned_absent_character_acted")
        if style_qc_score < 72:
            failure_codes.add("vnext_style_quality_below_threshold")
        qc["failure_codes"] = sorted(failure_codes)
        qc["style_qc_score"] = style_qc_score
        qc["should_block_tts"] = bool(qc.get("should_block_tts")) or bool(
            banned_phrase_count or generic_label_count or invalid_name_count or mentioned_absent_action_count or style_qc_score < 72
        )
        qc["summary"] = self._qc_summary(qc)
        return qc

    def _build_generalization_audit(self, story_context_pack: dict[str, Any]) -> dict[str, Any]:
        terms = self._generalization_audit_terms(story_context_pack)
        service_path = Path(__file__).resolve()
        backend_dir = service_path.parents[2]
        test_path = backend_dir / "tests" / "test_script_generation_vnext.py"
        active_hits = self._scan_file_for_terms(service_path, terms)
        test_hits = self._scan_file_for_terms(test_path, terms) if test_path.exists() else []
        return {
            "artifact_version": "script_vnext_generalization_audit_v1",
            "created_at": datetime.utcnow().isoformat(),
            "terms_scanned": terms,
            "code_name_hits": active_hits,
            "prompt_name_hits": active_hits,
            "test_fixture_only_hits": test_hits,
            "allowed_sample_input_hits": test_hits,
            "blocked_active_hardcoding_hits": active_hits,
            "passed": not active_hits,
            "notes": [
                "Audit terms come from this run's Story Context Pack, not from a hardcoded title list.",
                "Test hits are treated as fixture-only unless the same term appears in active vNext service code.",
            ],
        }

    def _generalization_audit_terms(self, story_context_pack: dict[str, Any]) -> list[str]:
        generic_single_word_terms = {
            "ability",
            "abilities",
            "attacker",
            "attackers",
            "confrontation",
            "fight",
            "battle",
            "school",
            "student",
            "students",
            "teacher",
            "power",
            "powers",
            "team",
            "group",
            "mission",
            "city",
            "room",
            "hallway",
            "classroom",
        }
        raw_terms = [
            *[str(item.get("name") or "") for item in story_context_pack.get("main_characters", []) or [] if isinstance(item, dict)],
            *[str(term) for term in story_context_pack.get("preserve_terms_exact", []) or []],
        ]
        terms: list[str] = []
        for term in raw_terms:
            value = term.strip()
            if len(value) < 4 or looks_like_false_character_name(value):
                continue
            if len(value.split()) == 1 and value.casefold() in generic_single_word_terms:
                continue
            terms.append(value)
        return list(dict.fromkeys(terms))[:60]

    def _registry_character_candidates(self, character_registry: dict[str, Any]) -> list[dict[str, Any]]:
        candidates: list[dict[str, Any]] = []

        def visit(value: Any) -> None:
            if isinstance(value, list):
                for item in value:
                    visit(item)
                return
            if not isinstance(value, dict):
                return
            name = (
                value.get("name")
                or value.get("canonical_name")
                or value.get("display_name")
                or value.get("primary_name")
                or value.get("character_name")
            )
            if name:
                aliases = value.get("aliases") or value.get("alternate_names") or value.get("aka") or []
                if isinstance(aliases, str):
                    aliases = [aliases]
                candidates.append(
                    {
                        "name": str(name),
                        "aliases": aliases if isinstance(aliases, list) else [],
                        "confidence": value.get("confidence") or value.get("identity_confidence") or value.get("score") or 0.75,
                    }
                )
            for key in ("characters", "canonical_characters", "entries", "items", "registry"):
                if key in value:
                    visit(value.get(key))
            if not name:
                for child in value.values():
                    if isinstance(child, (dict, list)):
                        visit(child)

        visit(character_registry.get("characters", character_registry))
        unique: dict[str, dict[str, Any]] = {}
        for candidate in candidates:
            key = self._term_key(candidate.get("name"))
            if key and key not in unique:
                unique[key] = candidate
        return list(unique.values())

    def _load_story_context_seed_artifacts(self, project_dir: Path) -> dict[str, Any]:
        characters: list[dict[str, Any]] = []
        special_terms: list[str] = []
        text_parts: list[str] = []
        story_bible = read_json(project_dir / "output" / "story_bible.json", default={})
        if isinstance(story_bible, dict):
            text_parts.extend(
                str(story_bible.get(key) or "")
                for key in ("chapter_premise", "chapter_summary", "summary")
                if str(story_bible.get(key) or "").strip()
            )
            for note in story_bible.get("continuity_notes", []) or []:
                text_parts.append(str(note))
            for scene in story_bible.get("scene_memory", []) or []:
                if isinstance(scene, dict):
                    text_parts.extend(str(scene.get(key) or "") for key in ("state", "open_thread", "location") if str(scene.get(key) or "").strip())
            for cast_item in story_bible.get("cast", []) or []:
                if isinstance(cast_item, dict) and cast_item.get("name"):
                    aliases = cast_item.get("aliases") if isinstance(cast_item.get("aliases"), list) else []
                    characters.append({"name": str(cast_item.get("name")), "aliases": aliases, "confidence": 0.85})
                    text_parts.extend(str(cast_item.get(key) or "") for key in ("role", "visual_cues", "notes") if str(cast_item.get(key) or "").strip())
            for term in story_bible.get("world_terms", []) or []:
                special_terms.append(str(term))

        style_vocab = read_json(project_dir / "output" / "style_vocabulary.json", default={})
        if isinstance(style_vocab, dict):
            for name in style_vocab.get("named_characters", []) or []:
                characters.append({"name": str(name), "aliases": [], "confidence": 0.8})
            for term in [
                *(style_vocab.get("world_terms", []) or []),
                *(style_vocab.get("stakes_phrases", []) or []),
                style_vocab.get("team_term"),
                style_vocab.get("antagonist_term"),
            ]:
                if term:
                    special_terms.append(str(term))
            text_parts.extend(str(term) for term in style_vocab.get("action_verbs", []) or [] if str(term).strip())

        return {
            "characters": characters,
            "special_terms": list(dict.fromkeys(term.strip() for term in special_terms if term and term.strip())),
            "text": [part for part in text_parts if part and str(part).strip()],
        }

    def _extract_context_terms(self, text: str, character_name_keys: set[str]) -> list[str]:
        counts: dict[str, int] = {}
        display: dict[str, str] = {}
        for value in self._proper_name_values(text):
            key = self._term_key(value)
            if not key or key in character_name_keys or not self._is_valid_special_term(value, curated=False):
                continue
            if any(name_key and name_key in key for name_key in character_name_keys):
                continue
            counts[key] = counts.get(key, 0) + 1
            display.setdefault(key, value)
        for quoted in re.findall(r"['\"]([A-Za-z][A-Za-z0-9' -]{2,40})['\"]", text or ""):
            key = self._term_key(quoted)
            if key and key not in character_name_keys and self._is_valid_special_term(quoted, curated=False):
                if any(name_key and name_key in key for name_key in character_name_keys):
                    continue
                counts[key] = counts.get(key, 0) + 2
                display.setdefault(key, quoted.strip())
        ordered = sorted(counts.items(), key=lambda item: (-item[1], display.get(item[0], "")))
        return [display[key] for key, count in ordered[:30] if count >= 2 or len(display[key].split()) > 1]

    def _infer_relationships(self, scenes: list[dict[str, Any]], main_characters: list[dict[str, Any]]) -> list[dict[str, Any]]:
        main_keys = {self._term_key(item.get("name")): item.get("name") for item in main_characters}
        pair_counts: dict[tuple[str, str], int] = {}
        pair_refs: dict[tuple[str, str], list[str]] = {}
        for scene in scenes:
            scene_people = [*(scene.get("visible_characters", []) or []), *(scene.get("speakers", []) or [])]
            names = [str(name) for name in scene_people if self._term_key(name) in main_keys]
            unique = list(dict.fromkeys(names))
            for index, left in enumerate(unique):
                for right in unique[index + 1:]:
                    pair = tuple(sorted((left, right)))
                    pair_counts[pair] = pair_counts.get(pair, 0) + 1
                    pair_refs.setdefault(pair, []).append(str(scene.get("scene_id") or ""))
        relationships: list[dict[str, Any]] = []
        for pair, count in sorted(pair_counts.items(), key=lambda item: -item[1])[:12]:
            relationships.append(
                {
                    "characters": list(pair),
                    "relationship_signal": "shared scenes and dialogue context",
                    "confidence": round(min(0.95, 0.45 + count * 0.1), 3),
                    "evidence_refs": pair_refs.get(pair, [])[:6],
                }
            )
        return relationships

    def _infer_project_premise(self, scenes: list[dict[str, Any]]) -> tuple[str, list[dict[str, Any]]]:
        phrases: list[str] = []
        refs: list[dict[str, Any]] = []
        for scene in scenes[:5]:
            summary = self._clean_sentence(scene.get("scene_summary"))
            if summary:
                phrases.append(self._trim_phrase(summary, 18))
                refs.append({"scene_id": scene.get("scene_id"), "panel_ids": list(scene.get("source_panel_ids") or [])[:4]})
        if not phrases:
            return "The project context is still evidence-thin, so each scene must stay close to its transcript and visual summaries.", refs
        return self._ensure_sentence("The chapter opens around " + "; then ".join(phrases[:3])), refs[:3]

    def _infer_current_story_state(self, scenes: list[dict[str, Any]]) -> tuple[str, list[dict[str, Any]]]:
        phrases: list[str] = []
        refs: list[dict[str, Any]] = []
        for scene in scenes[-5:]:
            summary = self._clean_sentence(scene.get("scene_summary"))
            if not summary:
                snippets = [self._clean_sentence(text) for text in scene.get("transcript_snippets", []) or [] if self._clean_sentence(text)]
                summary = snippets[0] if snippets else ""
            if summary:
                phrases.append(self._trim_phrase(summary, 16))
                refs.append({"scene_id": scene.get("scene_id"), "panel_ids": list(scene.get("source_panel_ids") or [])[:4]})
        if not phrases:
            return "No durable story state has been established from the available artifacts yet.", refs
        return self._ensure_sentence("By the latest scenes, " + "; then ".join(phrases[-3:])), refs[-3:]

    def _infer_timeline_markers(self, scenes: list[dict[str, Any]]) -> list[dict[str, Any]]:
        markers: list[dict[str, Any]] = []
        pattern = re.compile(r"\b(?:flashback|memory|earlier|later|years? ago|before|after|then|meanwhile)\b", re.I)
        for scene in scenes:
            text = " ".join([str(scene.get("scene_summary") or ""), " ".join(scene.get("transcript_snippets", []) or [])])
            if pattern.search(text):
                markers.append({"scene_id": scene.get("scene_id"), "marker_text": self._trim_phrase(text, 24)})
        return markers[:12]

    def _infer_unresolved_questions(self, scenes: list[dict[str, Any]]) -> list[dict[str, Any]]:
        questions: list[dict[str, Any]] = []
        for scene in scenes:
            for snippet in scene.get("transcript_snippets", []) or []:
                text = self._clean_sentence(snippet)
                if "?" in text:
                    questions.append({"scene_id": scene.get("scene_id"), "question": self._trim_phrase(text, 24)})
        return questions

    def _infer_style_tone_notes(self, text: str) -> list[str]:
        lowered = self._normalize(text)
        notes: list[str] = []
        if re.search(r"\b(?:fight|attack|hit|punch|blood|danger|run|weapon|battle|crash)\b", lowered):
            notes.append("Action beats should name who acts, who is affected, and what changes.")
        if re.search(r"\b(?:class|school|student|teacher|lesson|hallway)\b", lowered):
            notes.append("Social pressure and status dynamics should be explained when the evidence supports them.")
        if re.search(r"\b(?:power|ability|magic|monster|robot|energy|curse|rank|guild|squad|team)\b", lowered):
            notes.append("Source-grounded special terms should be preserved exactly and explained only from evidence.")
        if re.search(r"\b(?:cry|afraid|shock|angry|worry|smile|sad|panic)\b", lowered):
            notes.append("Emotional reactions should be tied to the concrete event that caused them.")
        return notes or ["Keep the recap grounded, chronological, and specific to the available scene evidence."]

    def _prompt_context_pack(self, story_context_pack: dict[str, Any]) -> dict[str, Any]:
        return {
            "project_premise": story_context_pack.get("project_premise"),
            "main_characters": (story_context_pack.get("main_characters") or [])[:8],
            "relationships": (story_context_pack.get("character_roles_relationships") or [])[:8],
            "special_terms": (story_context_pack.get("special_terms") or [])[:12],
            "current_story_state": story_context_pack.get("current_story_state"),
            "timeline_markers": (story_context_pack.get("timeline_flashback_markers") or [])[:8],
            "unresolved_questions": (story_context_pack.get("important_unresolved_questions") or [])[:8],
            "preserve_terms_exact": (story_context_pack.get("preserve_terms_exact") or [])[:20],
            "reject_terms": (story_context_pack.get("reject_terms") or [])[:30],
            "style_tone_notes": story_context_pack.get("style_tone_notes") or [],
        }

    def _relevant_context_for_scene(self, scene: dict[str, Any], story_context_pack: dict[str, Any]) -> dict[str, Any]:
        scene_text = self._normalize(
            " ".join(
                [
                    str(scene.get("scene_summary") or ""),
                    " ".join(scene.get("transcript_snippets", []) or []),
                    " ".join(
                        f"{item.get('evidence_text', '')} {item.get('visual_summary', '')}"
                        for item in (scene.get("panel_contribution_map") or {}).values()
                    ),
                ]
            )
        )
        scene_people = [
            *(scene.get("visible_characters", []) or []),
            *(scene.get("speakers", []) or []),
            *(scene.get("mentioned_characters", []) or []),
        ]
        scene_names = {self._term_key(name) for name in scene_people}
        characters = [
            item
            for item in story_context_pack.get("main_characters", []) or []
            if self._term_key(item.get("name")) in scene_names
        ][:6]
        terms = [
            term
            for term in story_context_pack.get("special_terms", []) or []
            if self._term_key(term) and self._term_key(term) in self._term_key(scene_text)
        ][:8]
        return {
            "characters_in_scene": characters,
            "source_grounded_terms_in_scene": terms,
            "current_story_state": story_context_pack.get("current_story_state"),
            "style_tone_notes": story_context_pack.get("style_tone_notes", [])[:3],
            "reject_terms": story_context_pack.get("reject_terms", [])[:20],
        }

    def _known_scene_character_names(self, scene: dict[str, Any], story_context_pack: dict[str, Any]) -> list[str]:
        known_by_key = {
            self._term_key(item.get("name")): str(item.get("name"))
            for item in story_context_pack.get("main_characters", []) or []
            if isinstance(item, dict) and float(item.get("confidence") or 0.0) >= 0.65 and self._is_valid_context_name(item.get("name"))
        }
        names: list[str] = []
        for raw_name in [*(scene.get("visible_characters", []) or []), *(scene.get("speakers", []) or [])]:
            key = self._term_key(raw_name)
            if key in known_by_key:
                names.append(known_by_key[key])
        return list(dict.fromkeys(names))

    def _clean_final_narration_text(self, text: str) -> str:
        replacements = {
            "The story moves through": "The scene follows",
            "The surrounding detail is folded": "The supporting detail stays",
            "carries a moment through": "pushes forward through",
            "anchors the moment": "sets the scene",
            "the important line is": "the dialogue makes clear that",
            "the next detail adds": "the next detail shows",
            "keeps the moment connected": "keeps the exchange connected",
            "as the dialogue and visuals converge": "as the exchange builds",
            "Together, the moment moves from": "Together, the scene shifts from",
            "beat built around": "scene shaped by",
            "That line pushes the exchange": "The exchange turns",
            "without adding unsupported details": "with the evidence staying grounded",
            "supporting panels stay": "nearby reactions stay",
            "giving the beat a clear start and consequence": "showing how the exchange changes",
            "The same exchange also includes": "It also shows",
            "That evidence leads into": "That leads into",
            "The scene is caught in an exchange where": "The scene turns on",
            "The scene is pulled into": "The scene turns on",
            "The exchange turns on": "The dialogue focuses on",
            "The next image shows": "Then",
        }
        cleaned = str(text or "")
        for source, replacement in replacements.items():
            cleaned = re.sub(re.escape(source), replacement, cleaned, flags=re.I)
        cleaned = re.sub(r"\b[Pp]anel\s+\d+\s+(?:shows|depicts|contains)\b", "The scene shows", cleaned)
        return self._clean_text(cleaned)

    def _scan_file_for_terms(self, path: Path, terms: list[str]) -> list[dict[str, Any]]:
        if not path.exists():
            return []
        text = path.read_text(encoding="utf-8", errors="ignore")
        hits: list[dict[str, Any]] = []
        lowered = text.casefold()
        for term in terms:
            if term.casefold() not in lowered:
                continue
            lines = [
                index
                for index, line in enumerate(text.splitlines(), start=1)
                if term.casefold() in line.casefold()
            ][:8]
            hits.append({"term": term, "path": str(path), "lines": lines})
        return hits

    def _proper_name_values(self, text: str) -> list[str]:
        values: list[str] = []
        for match in re.finditer(r"\b[A-Z][A-Za-z0-9'_-]*(?:\s+[A-Z][A-Za-z0-9'_-]*){0,3}\b", text or ""):
            value = match.group(0).strip()
            first_token = value.split()[0] if value.split() else value
            if value in {"The", "A", "An", "I"} or first_token in {"Can", "If", "When", "What", "Who", "Why", "How", "Because", "That", "This"}:
                continue
            if looks_like_false_character_name(value):
                continue
            values.append(value)
        return values

    def _term_key(self, value: Any) -> str:
        return re.sub(r"[^a-z0-9]+", "", str(value or "").casefold())

    def _is_valid_context_name(self, value: Any) -> bool:
        name = str(value or "").strip()
        if not name or looks_like_false_character_name(name):
            return False
        if len(name) > 40 or len(name.split()) > 4:
            return False
        if name.casefold().startswith(("the ", "a ", "an ")):
            return False
        if not re.search(r"[A-Z]", name):
            return False
        return bool(re.search(r"[A-Za-z]", name))

    def _is_valid_special_term(self, value: Any, *, curated: bool) -> bool:
        term = self._clean_text(str(value or "")).strip(" .,:;!?")
        if not term or looks_like_false_character_name(term):
            return False
        if len(term) > 60 or len(term) < 3:
            return False
        lowered = term.casefold()
        blocked = {
            "the",
            "this",
            "that",
            "there",
            "here",
            "someone",
            "something",
            "nothing",
            "trying",
            "keep",
            "long",
            "mathematical",
            "you're",
            "you'd",
            "would",
            "could",
            "should",
            "again",
            "because",
            "while",
            "before",
            "after",
            "angry",
            "anxious",
            "arrogant",
            "aggression",
            "pissed",
            "shock",
            "shocked",
            "nervous",
            "fearful",
            "smiling",
            "visual",
            "dialogue",
            "caption",
            "student",
            "teacher",
            "determination",
            "pay",
            "prove",
            "volunteer",
        }
        if lowered in blocked or re.search(r"\b(?:you're|you'd|can't|won't|don't|i'm|i'll|we're|they're)\b", lowered):
            return False
        if re.search(r"[!?]{2,}|[=≡·・]{1,}|[^\w\s'/-]{3,}", term):
            return False
        word_count = len(term.split())
        if any(len(part.strip("'/-")) <= 1 for part in term.split()):
            return False
        if curated:
            return bool(re.search(r"[A-Za-z]", term))
        if any(part.casefold().strip("'/-") in blocked for part in term.split()):
            return False
        if word_count >= 2:
            return not term.split()[0] in {"The", "A", "An", "This", "That", "Trying", "Keep"}
        if term.isupper() and 2 <= len(term) <= 12:
            return True
        return False

    def _empty_redraft_log(self, config: ScriptVNextRedraftConfig | None) -> dict[str, Any]:
        cfg = config or ScriptVNextRedraftConfig()
        return {
            "artifact_version": VNEXT_ARTIFACT_VERSION,
            "prompt_version": VNEXT_REDRAFT_PROMPT_VERSION,
            "created_at": datetime.utcnow().isoformat(),
            "enabled": bool(cfg.enabled),
            "dry_run": bool(cfg.dry_run),
            "config": {
                "max_calls": int(cfg.max_calls),
                "max_scenes_per_batch": int(cfg.max_scenes_per_batch),
                "max_prompt_chars": int(cfg.max_prompt_chars),
                "max_output_tokens": int(cfg.max_output_tokens),
                "max_estimated_cost_usd": float(cfg.max_estimated_cost_usd),
                "style_threshold": int(cfg.style_threshold),
                "provider": cfg.provider,
            },
            "target_scene_ids": [],
            "target_scene_count": 0,
            "redrafted_scene_ids": [],
            "redrafted_scene_count": 0,
            "unresolved_scene_ids": [],
            "actual_call_count": 0,
            "estimated_call_count": 0,
            "cache_misses": 0,
            "estimated_input_tokens": 0,
            "estimated_output_tokens": 0,
            "estimated_cost_usd": 0.0,
            "budget_exceeded": False,
            "batches": [],
        }

    def _run_scene_redraft_pass(
        self,
        *,
        project_dir: Path,
        scene_plan: dict[str, Any],
        narration_chunks: dict[str, Any],
        qc_report: dict[str, Any],
        config: ScriptVNextRedraftConfig,
        story_context_pack: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        story_context_pack = story_context_pack or {}
        redraft_log = self._empty_redraft_log(config)
        chunks_by_scene = {
            str(chunk.get("scene_id") or chunk.get("chunk_id")): dict(chunk)
            for chunk in narration_chunks.get("chunks", []) or []
            if isinstance(chunk, dict)
        }
        targets = self._select_redraft_targets(scene_plan, chunks_by_scene, qc_report, config)
        redraft_log["target_scene_ids"] = [scene["scene_id"] for scene in targets]
        redraft_log["target_scene_count"] = len(targets)
        if not targets:
            return {"narration_chunks": narration_chunks, "redraft_log": redraft_log}

        cache_path = project_dir / "output" / "script_vnext" / "redraft_cache.json"
        cache = self._load_redraft_cache(cache_path)
        cache_entries = cache.setdefault("entries", {})
        batches = self._make_redraft_batches(targets, chunks_by_scene, config, story_context_pack)
        updated_chunks_by_scene = dict(chunks_by_scene)
        spent = 0.0
        actual_calls = 0
        cache_misses = 0
        redrafted_scene_ids: list[str] = []
        unresolved_scene_ids: set[str] = set()

        for batch_number, batch in enumerate(batches, start=1):
            packets = [self._scene_redraft_packet(scene, chunks_by_scene.get(scene["scene_id"], {}), story_context_pack) for scene in batch]
            prompt = self._redraft_prompt(packets, story_context_pack, sanitized=False)
            prompt, packets = self._ensure_prompt_budget(prompt, packets, config, story_context_pack)
            estimate = self._estimate_redraft_cost(prompt, config.max_output_tokens)
            redraft_log["estimated_call_count"] += 1
            redraft_log["estimated_input_tokens"] += estimate["input_tokens_estimate"]
            redraft_log["estimated_output_tokens"] += estimate["output_tokens_estimate"]
            redraft_log["estimated_cost_usd"] = round(float(redraft_log["estimated_cost_usd"]) + estimate["estimated_cost_usd"], 6)
            batch_scene_ids = [packet["scene_id"] for packet in packets]
            cache_key = self._redraft_cache_key(prompt, config)

            if cache_key in cache_entries:
                metadata = {
                    "batch_number": batch_number,
                    "scene_ids": batch_scene_ids,
                    "status": "cache_hit",
                    "cache_hit": True,
                    "estimated_cost_usd": 0.0,
                    "prompt_chars": len(prompt),
                    "model": cache_entries[cache_key].get("model"),
                }
                accepted = self._apply_redraft_payload(
                    cache_entries[cache_key].get("payload") or {},
                    batch,
                    updated_chunks_by_scene,
                    metadata,
                    story_context_pack,
                )
                redrafted_scene_ids.extend(accepted)
                unresolved_scene_ids.update(scene_id for scene_id in batch_scene_ids if scene_id not in accepted)
                redraft_log["batches"].append(metadata)
                continue

            if config.dry_run:
                redraft_log["batches"].append(
                    {
                        "batch_number": batch_number,
                        "scene_ids": batch_scene_ids,
                        "status": "dry_run_estimate",
                        "cache_hit": False,
                        "prompt_chars": len(prompt),
                        **estimate,
                    }
                )
                continue

            if actual_calls >= max(int(config.max_calls), 0):
                unresolved_scene_ids.update(batch_scene_ids)
                redraft_log["budget_exceeded"] = True
                redraft_log["batches"].append(
                    {
                        "batch_number": batch_number,
                        "scene_ids": batch_scene_ids,
                        "status": "skipped_call_limit",
                        "cache_hit": False,
                        "prompt_chars": len(prompt),
                        **estimate,
                    }
                )
                continue
            if config.max_estimated_cost_usd <= 0 or spent + estimate["estimated_cost_usd"] > config.max_estimated_cost_usd:
                unresolved_scene_ids.update(batch_scene_ids)
                redraft_log["budget_exceeded"] = True
                redraft_log["batches"].append(
                    {
                        "batch_number": batch_number,
                        "scene_ids": batch_scene_ids,
                        "status": "skipped_cost_budget",
                        "cache_hit": False,
                        "prompt_chars": len(prompt),
                        **estimate,
                    }
                )
                continue

            cache_misses += 1
            metadata = {
                "batch_number": batch_number,
                "scene_ids": batch_scene_ids,
                "status": "pending",
                "cache_hit": False,
                "prompt_chars": len(prompt),
                **estimate,
            }
            try:
                response = self._call_redraft_model(prompt, config)
                actual_calls += 1
                spent = round(spent + estimate["estimated_cost_usd"], 6)
                metadata.update(
                    {
                        "status": "completed",
                        "provider": response.get("provider"),
                        "model": response.get("model"),
                    }
                )
                accepted = self._apply_redraft_payload(response.get("payload") or {}, batch, updated_chunks_by_scene, metadata, story_context_pack)
                redrafted_scene_ids.extend(accepted)
                unresolved_scene_ids.update(scene_id for scene_id in batch_scene_ids if scene_id not in accepted)
                cache_entries[cache_key] = {
                    "cache_version": VNEXT_REDRAFT_CACHE_VERSION,
                    "created_at": datetime.utcnow().isoformat(),
                    "model": response.get("model"),
                    "provider": response.get("provider"),
                    "payload": response.get("payload") or {},
                }
            except Exception as exc:
                metadata["first_error"] = self._safe_error_text(exc)
                retry_metadata = self._retry_redraft_batch_safely(
                    batch=batch,
                    batch_number=batch_number,
                    config=config,
                    story_context_pack=story_context_pack,
                    updated_chunks_by_scene=updated_chunks_by_scene,
                    cache=cache_entries,
                    remaining_call_budget=max(int(config.max_calls) - actual_calls, 0),
                    remaining_cost_budget=max(float(config.max_estimated_cost_usd) - spent, 0.0),
                )
                actual_calls += int(retry_metadata.pop("actual_call_count", 0))
                spent = round(spent + float(retry_metadata.pop("spent_usd", 0.0)), 6)
                retry_accepted = retry_metadata.pop("accepted_scene_ids", [])
                redrafted_scene_ids.extend(retry_accepted)
                unresolved_scene_ids.update(scene_id for scene_id in batch_scene_ids if scene_id not in retry_accepted)
                metadata.update(retry_metadata)
            redraft_log["batches"].append(metadata)

        if not config.dry_run:
            self._save_redraft_cache(cache_path, cache)
        redraft_log["actual_call_count"] = actual_calls
        redraft_log["cache_misses"] = cache_misses
        redraft_log["redrafted_scene_ids"] = list(dict.fromkeys(redrafted_scene_ids))
        redraft_log["redrafted_scene_count"] = len(redraft_log["redrafted_scene_ids"])
        redraft_log["unresolved_scene_ids"] = sorted(unresolved_scene_ids)
        if actual_calls:
            redraft_log["estimated_cost_usd"] = round(spent, 6)
        elif not config.dry_run:
            redraft_log["estimated_cost_usd"] = 0.0

        updated_chunks = []
        for chunk in narration_chunks.get("chunks", []) or []:
            scene_id = str(chunk.get("scene_id") or chunk.get("chunk_id"))
            updated_chunks.append(updated_chunks_by_scene.get(scene_id, chunk))
        updated_narration = {
            **narration_chunks,
            "chunks": updated_chunks,
            "summary": {
                "chunk_count": len(updated_chunks),
                "long_gap_count": sum(1 for chunk in updated_chunks if float(chunk.get("duration_gap_seconds") or 0.0) > 2.0),
                "largest_gap_seconds": max([float(chunk.get("duration_gap_seconds") or 0.0) for chunk in updated_chunks] or [0.0]),
                "redrafted_scene_count": len(redraft_log["redrafted_scene_ids"]),
            },
        }
        return {"narration_chunks": updated_narration, "redraft_log": redraft_log}

    def _select_redraft_targets(
        self,
        scene_plan: dict[str, Any],
        chunks_by_scene: dict[str, dict[str, Any]],
        qc_report: dict[str, Any],
        config: ScriptVNextRedraftConfig,
    ) -> list[dict[str, Any]]:
        failure_codes = set(qc_report.get("failure_codes") or [])
        should_target_broadly = bool(qc_report.get("should_block_tts")) or bool(failure_codes)
        targets: list[dict[str, Any]] = []
        for scene in scene_plan.get("scenes", []) or []:
            scene_id = str(scene.get("scene_id") or "")
            chunk = chunks_by_scene.get(scene_id, {})
            score = self._local_style_score(str(chunk.get("text") or ""))
            chunk["local_style_score"] = score
            weak = score < config.style_threshold or float(chunk.get("duration_gap_seconds") or 0.0) > 1.5
            underused = any(
                item.get("contribution") not in {"low_information", "redundant_near_duplicate"}
                and not self._text_mentions_evidence(str(chunk.get("text") or ""), item)
                for item in (scene.get("panel_contribution_map") or {}).values()
            )
            if weak or (should_target_broadly and underused):
                targets.append(scene)
        return targets

    def _make_redraft_batches(
        self,
        targets: list[dict[str, Any]],
        chunks_by_scene: dict[str, dict[str, Any]],
        config: ScriptVNextRedraftConfig,
        story_context_pack: dict[str, Any],
    ) -> list[list[dict[str, Any]]]:
        batches: list[list[dict[str, Any]]] = []
        current: list[dict[str, Any]] = []
        for scene in targets:
            trial = [*current, scene]
            packets = [self._scene_redraft_packet(item, chunks_by_scene.get(item["scene_id"], {}), story_context_pack) for item in trial]
            prompt_len = len(self._redraft_prompt(packets, story_context_pack, sanitized=False))
            if current and (len(trial) > max(1, config.max_scenes_per_batch) or prompt_len > config.max_prompt_chars):
                batches.append(current)
                current = [scene]
            else:
                current = trial
        if current:
            batches.append(current)
        return batches

    def _scene_redraft_packet(
        self,
        scene: dict[str, Any],
        chunk: dict[str, Any],
        story_context_pack: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        visual_evidence: list[dict[str, Any]] = []
        for panel_id, item in (scene.get("panel_contribution_map") or {}).items():
            contribution = str(item.get("contribution") or "")
            if contribution in {"low_information", "redundant_near_duplicate"}:
                continue
            visual_evidence.append(
                {
                    "panel_id": panel_id,
                    "contribution": contribution,
                    "dialogue_or_caption": self._clean_prompt_text(item.get("evidence_text")),
                    "visual_summary": self._clean_prompt_text(item.get("visual_summary")),
                    "visual_only": bool(item.get("visual_only")),
                }
            )
        return {
            "scene_id": scene.get("scene_id"),
            "source_panel_ids": list(scene.get("source_panel_ids") or []),
            "representative_panel_id": scene.get("representative_panel_id"),
            "supporting_panel_ids": list(scene.get("supporting_panel_ids") or []),
            "transcript_snippets": [
                self._clean_prompt_text(text)
                for text in scene.get("transcript_snippets", []) or []
                if self._clean_prompt_text(text)
            ][:8],
            "visual_evidence": visual_evidence[:10],
            "character_roles": scene.get("character_roles") or {},
            "visible_characters": list(scene.get("visible_characters") or []),
            "mentioned_characters": list(scene.get("mentioned_characters") or []),
            "current_narration": self._clean_prompt_text(chunk.get("text")),
            "qc_failures": self._scene_qc_failures(chunk),
            "target_narration_duration_seconds": float(chunk.get("scene_duration_seconds") or scene.get("target_narration_duration_seconds") or 0.0),
            "relevant_story_context": self._relevant_context_for_scene(scene, story_context_pack or {}),
        }

    def _redraft_prompt(
        self,
        packets: list[dict[str, Any]],
        story_context_pack: dict[str, Any] | None = None,
        *,
        sanitized: bool,
    ) -> str:
        payload = json.dumps(packets, ensure_ascii=False, indent=2)
        context_payload = json.dumps(self._prompt_context_pack(story_context_pack or {}), ensure_ascii=False, indent=2)
        safety_note = (
            "This is a sanitized retry. If any evidence is uncomfortable or too thin, keep the line concise and grounded.\n"
            if sanitized
            else ""
        )
        return (
            "Turn structured scene evidence into natural YouTube recap narration.\n"
            "Use only the project context and scene evidence below. Do not ask for images. Do not invent names, lore, motives, locations, or events.\n"
            f"{safety_note}"
            "Write polished narration that sounds like a human explaining the story to a viewer, not captions, analysis notes, or pipeline metadata.\n"
            "For each scene: establish the setup, name the conflict, describe the escalation, and explain the consequence when evidence supports it.\n"
            "Default to 2-3 natural sentences per scene. Use a single sentence only when the packet has too little evidence for more.\n"
            "Use established character names naturally when confidence is high. If a real name is unknown, use the stable visual label from context.\n"
            "Avoid ambiguous pronouns when multiple characters are active. Repeat names when clarity matters.\n"
            "Mention characters only if they are visible/speaking in character roles, or clearly treat mentioned-only names as absent.\n"
            "Preserve source-grounded special terms exactly when they are listed in the context pack.\n"
            "If evidence is thin, write one concise grounded sentence instead of hallucinating.\n"
            "Aim for narration that can cover the target duration, but never pad with generic or meta language.\n"
            "Do not mention scene mechanics, evidence mechanics, ids, panels, prompts, or internal planning phrases.\n\n"
            "Never write these internal phrases in final narration: "
            f"{'; '.join(INTERNAL_TEMPLATE_PHRASES)}.\n\n"
            "PROJECT STORY CONTEXT PACK:\n"
            f"{context_payload}\n\n"
            "SCENE PACKETS JSON:\n"
            f"{payload}\n\n"
            "OUTPUT JSON ONLY:\n"
            '{"rewrites":[{"scene_id":"scene_001","text":"polished narration"}]}\n'
        )

    def _ensure_prompt_budget(
        self,
        prompt: str,
        packets: list[dict[str, Any]],
        config: ScriptVNextRedraftConfig,
        story_context_pack: dict[str, Any],
    ) -> tuple[str, list[dict[str, Any]]]:
        if len(prompt) <= config.max_prompt_chars:
            return prompt, packets
        compacted: list[dict[str, Any]] = []
        for packet in packets:
            compact = dict(packet)
            compact["transcript_snippets"] = compact.get("transcript_snippets", [])[:4]
            compact["visual_evidence"] = [
                {
                    **item,
                    "dialogue_or_caption": self._trim_phrase(str(item.get("dialogue_or_caption") or ""), 18),
                    "visual_summary": self._trim_phrase(str(item.get("visual_summary") or ""), 18),
                }
                for item in compact.get("visual_evidence", [])[:6]
            ]
            compact["current_narration"] = self._trim_phrase(str(compact.get("current_narration") or ""), 45)
            compacted.append(compact)
        return self._redraft_prompt(compacted, story_context_pack, sanitized=False), compacted

    def _call_redraft_model(self, prompt: str, config: ScriptVNextRedraftConfig) -> dict[str, Any]:
        if self.redraft_client is not None:
            return self.redraft_client.redraft(prompt=prompt, config=config)
        result = asyncio.run(
            self.router._route_json(
                task_name="vNext scene-level redraft",
                prompt=prompt,
                validator=self._validate_redraft_payload,
                max_output_tokens=config.max_output_tokens,
                provider=config.provider,
                model_candidates=["gemini-2.5-flash-lite", "gemini-2.5-flash"],
            )
        )
        return {"provider": result.provider, "model": result.model, "payload": result.payload}

    def _retry_redraft_batch_safely(
        self,
        *,
        batch: list[dict[str, Any]],
        batch_number: int,
        config: ScriptVNextRedraftConfig,
        updated_chunks_by_scene: dict[str, dict[str, Any]],
        cache: dict[str, Any],
        remaining_call_budget: int,
        remaining_cost_budget: float,
        story_context_pack: dict[str, Any],
    ) -> dict[str, Any]:
        accepted_scene_ids: list[str] = []
        attempts: list[dict[str, Any]] = []
        spent = 0.0
        actual_calls = 0
        retry_batches = [[scene] for scene in batch] if len(batch) > 1 else [batch]
        for retry_index, retry_batch in enumerate(retry_batches, start=1):
            if remaining_call_budget <= 0 or remaining_cost_budget <= 0:
                attempts.append({"retry_index": retry_index, "status": "skipped_retry_budget"})
                continue
            packets = [
                self._sanitize_packet_for_retry(
                    self._scene_redraft_packet(scene, updated_chunks_by_scene.get(scene["scene_id"], {}), story_context_pack)
                )
                for scene in retry_batch
            ]
            prompt = self._redraft_prompt(packets, story_context_pack, sanitized=True)
            prompt, packets = self._ensure_prompt_budget(prompt, packets, config, story_context_pack)
            estimate = self._estimate_redraft_cost(prompt, config.max_output_tokens)
            if estimate["estimated_cost_usd"] > remaining_cost_budget:
                attempts.append({"retry_index": retry_index, "status": "skipped_retry_cost_budget", **estimate})
                continue
            cache_key = self._redraft_cache_key(prompt, config)
            try:
                if cache_key in cache:
                    response = {"provider": cache[cache_key].get("provider"), "model": cache[cache_key].get("model"), "payload": cache[cache_key].get("payload") or {}}
                    status = "retry_cache_hit"
                else:
                    response = self._call_redraft_model(prompt, config)
                    actual_calls += 1
                    spent = round(spent + estimate["estimated_cost_usd"], 6)
                    remaining_call_budget -= 1
                    remaining_cost_budget = round(remaining_cost_budget - estimate["estimated_cost_usd"], 6)
                    cache[cache_key] = {
                        "cache_version": VNEXT_REDRAFT_CACHE_VERSION,
                        "created_at": datetime.utcnow().isoformat(),
                        "model": response.get("model"),
                        "provider": response.get("provider"),
                        "payload": response.get("payload") or {},
                    }
                    status = "retry_completed"
                metadata = {"status": status, "scene_ids": [packet["scene_id"] for packet in packets]}
                accepted = self._apply_redraft_payload(response.get("payload") or {}, retry_batch, updated_chunks_by_scene, metadata, story_context_pack)
                accepted_scene_ids.extend(accepted)
                attempts.append({**metadata, "retry_index": retry_index, **estimate})
            except Exception as exc:
                attempts.append({"retry_index": retry_index, "status": "retry_failed", "error": self._safe_error_text(exc), **estimate})
        status = "retry_completed" if accepted_scene_ids else "blocked"
        return {
            "status": status,
            "retry_attempts": attempts,
            "actual_call_count": actual_calls,
            "spent_usd": spent,
            "accepted_scene_ids": accepted_scene_ids,
        }

    def _apply_redraft_payload(
        self,
        payload: dict[str, Any],
        batch: list[dict[str, Any]],
        updated_chunks_by_scene: dict[str, dict[str, Any]],
        metadata: dict[str, Any],
        story_context_pack: dict[str, Any] | None = None,
    ) -> list[str]:
        allowed = {str(scene.get("scene_id")): scene for scene in batch}
        accepted: list[str] = []
        rejected: list[dict[str, str]] = []
        rewrites = payload.get("rewrites", []) if isinstance(payload, dict) else []
        for item in rewrites if isinstance(rewrites, list) else []:
            if not isinstance(item, dict):
                continue
            scene_id = str(item.get("scene_id") or "").strip()
            text = self._clean_text(str(item.get("text") or ""))
            scene = allowed.get(scene_id)
            chunk = updated_chunks_by_scene.get(scene_id)
            if not scene or not chunk or not text:
                continue
            rejection = self._redraft_rejection_reason(scene, chunk, text, story_context_pack or {})
            if rejection:
                rejected.append({"scene_id": scene_id, "reason": rejection})
                continue
            repaired = self._repair_chunk_after_text_update(chunk, text)
            repaired["redrafted"] = True
            repaired["redraft_status"] = metadata.get("status", "completed")
            updated_chunks_by_scene[scene_id] = repaired
            accepted.append(scene_id)
        metadata["accepted_scene_ids"] = accepted
        metadata["rejected_rewrites"] = rejected
        return accepted

    def _repair_chunk_after_text_update(self, chunk: dict[str, Any], text: str) -> dict[str, Any]:
        updated = dict(chunk)
        updated["text"] = text.strip()
        estimated = self._estimate_duration(text)
        scene_duration = float(updated.get("scene_duration_seconds") or 0.0)
        gap = round(max(scene_duration - estimated, 0.0), 2)
        repair_action = "redrafted"
        if gap > 2.0:
            scene_duration = round(max(estimated + 1.5, 2.0), 2)
            gap = round(max(scene_duration - estimated, 0.0), 2)
            repair_action = "redrafted_then_reduced_visual_duration"
        updated["scene_duration_seconds"] = scene_duration
        updated["estimated_narration_duration_seconds"] = estimated
        updated["duration_gap_seconds"] = gap
        updated["needs_narration_expansion"] = gap > 2.0
        updated["repair_action"] = repair_action if gap <= 2.0 else "unresolved_duration_gap_after_redraft"
        return updated

    def _redraft_rejection_reason(self, scene: dict[str, Any], chunk: dict[str, Any], text: str, story_context_pack: dict[str, Any] | None = None) -> str:
        if len(text.split()) < 8 and len(str(chunk.get("text") or "").split()) >= 8:
            return "rewrite_too_short"
        context = story_context_pack or {}
        for rejected_term in context.get("reject_terms", []) or []:
            term = str(rejected_term or "").strip()
            if len(term) >= 3 and re.search(rf"\b{re.escape(term)}\b", text, re.I):
                return "rejected_term_leak"
        meaningful_count = sum(
            1
            for item in (scene.get("panel_contribution_map") or {}).values()
            if item.get("contribution") not in {"low_information", "redundant_near_duplicate"}
        )
        if meaningful_count >= 2 and len(re.findall(r"[.!?](?:\s|$)", text)) < 2:
            return "rewrite_too_fragmented"
        lowered = text.casefold()
        banned = (
            "source text",
            "visual evidence",
            "supporting beat",
            "drift into silence",
            "no new event",
            "scene packet",
        )
        if any(phrase.casefold() in lowered for phrase in INTERNAL_TEMPLATE_PHRASES):
            return "meta_language"
        if any(phrase in lowered for phrase in banned):
            return "meta_language"
        context_terms = " ".join(
            [
                " ".join(str(item.get("name") or "") for item in context.get("main_characters", []) or [] if isinstance(item, dict)),
                " ".join(str(term) for term in context.get("preserve_terms_exact", []) or []),
                " ".join(str(term) for term in context.get("special_terms", []) or []),
            ]
        )
        allowed_names = set(self._proper_name_keys(" ".join([
            str(chunk.get("text") or ""),
            " ".join(scene.get("visible_characters") or []),
            " ".join(scene.get("speakers") or []),
            " ".join(scene.get("mentioned_characters") or []),
            " ".join(scene.get("transcript_snippets") or []),
            context_terms,
        ])))
        for name in [*(scene.get("visible_characters", []) or []), *(scene.get("speakers", []) or []), *(scene.get("mentioned_characters", []) or [])]:
            key = self._term_key(name)
            if key:
                allowed_names.add(key)
        for item in context.get("main_characters", []) or []:
            key = self._term_key(item.get("name") if isinstance(item, dict) else item)
            if key:
                allowed_names.add(key)
        rewrite_names = set(self._proper_name_keys(text))
        unexpected = rewrite_names - allowed_names
        if unexpected:
            return f"proper_noun_drift:{','.join(sorted(unexpected))}"
        evidence_text = " ".join(
            str(item.get("evidence_text") or item.get("visual_summary") or "")
            for item in (scene.get("panel_contribution_map") or {}).values()
        )
        if self._content_overlap_ratio(text, " ".join([evidence_text, str(chunk.get("text") or "")])) < 0.08:
            return "insufficient_evidence_overlap"
        return ""

    def _local_style_score(self, text: str) -> int:
        cleaned = self._clean_text(text)
        if not cleaned:
            return 0
        words = len(re.findall(r"\b[\w'-]+\b", cleaned))
        sentences = max(1, len(re.findall(r"[.!?](?:\s|$)", cleaned)))
        score = 55
        if words >= 28:
            score += 12
        if sentences >= 2:
            score += 10
        if re.search(r"\b(?:because|but|while|before|after|until|so|then|as)\b", cleaned, re.I):
            score += 8
        if re.search(r"\b(?:decides|realizes|tries|forces|warns|attacks|reacts|refuses|protects|confronts|reveals)\b", cleaned, re.I):
            score += 8
        meta_patterns = (
            "the story moves through",
            "anchors the moment",
            "carries a moment",
            "important line",
            "visual detail",
            "supporting detail",
            "panel",
            "source text",
        )
        score -= 12 * sum(1 for pattern in meta_patterns if pattern in cleaned.casefold())
        if re.match(r"^(the story|the scene)\b", cleaned, re.I):
            score -= 8
        if words < 16:
            score -= 16
        return max(0, min(100, score))

    def _text_mentions_evidence(self, text: str, item: dict[str, Any]) -> bool:
        evidence = " ".join([str(item.get("evidence_text") or ""), str(item.get("visual_summary") or "")])
        return self._content_overlap_ratio(text, evidence) >= 0.12

    def _clean_prompt_text(self, value: Any) -> str:
        text = self._clean_text(str(value or ""))
        if not text:
            return ""
        blocked = (
            "debug",
            "stale artifact",
            "rejected ocr",
            "watermark",
            "source label",
            "credit",
            "ui element",
        )
        if any(term in text.casefold() for term in blocked):
            return ""
        return self._trim_phrase(text, 70)

    def _scene_qc_failures(self, chunk: dict[str, Any]) -> list[str]:
        failures: list[str] = []
        text = str(chunk.get("text") or "")
        if self._local_style_score(text) < 68:
            failures.append("weak_youtube_style")
        if float(chunk.get("duration_gap_seconds") or 0.0) > 1.5:
            failures.append("narration_too_short_for_scene_duration")
        if re.search(r"\b(?:the story moves through|anchors the moment|important line|visual detail|panel)\b", text, re.I):
            failures.append("meta_or_caption_language")
        if len(re.findall(r"\b[\w'-]+\b", text)) < 18:
            failures.append("too_short")
        return failures

    def _sanitize_packet_for_retry(self, packet: dict[str, Any]) -> dict[str, Any]:
        sanitized = dict(packet)
        sanitized["transcript_snippets"] = [
            self._trim_phrase(text, 18)
            for text in sanitized.get("transcript_snippets", [])[:3]
            if self._trim_phrase(text, 18)
        ]
        sanitized["visual_evidence"] = [
            {
                "panel_id": item.get("panel_id"),
                "contribution": item.get("contribution"),
                "dialogue_or_caption": self._trim_phrase(str(item.get("dialogue_or_caption") or ""), 16),
                "visual_summary": self._trim_phrase(str(item.get("visual_summary") or ""), 16),
                "visual_only": bool(item.get("visual_only")),
            }
            for item in sanitized.get("visual_evidence", [])[:4]
        ]
        sanitized["current_narration"] = self._trim_phrase(str(sanitized.get("current_narration") or ""), 28)
        return sanitized

    def _proper_name_keys(self, text: str) -> list[str]:
        keys: list[str] = []
        for match in re.finditer(r"\b[A-Z][A-Za-z0-9'_-]*(?:\s+[A-Z][A-Za-z0-9'_-]*){0,3}\b", text or ""):
            value = match.group(0).strip()
            first_token = value.split()[0] if value.split() else value
            if value in {"The", "A", "An", "I"} or first_token in {"Can", "If", "When", "What", "Who", "Why", "How", "Because", "That", "This"}:
                continue
            if looks_like_false_character_name(value):
                continue
            keys.append(re.sub(r"[^a-z0-9]+", "", value.casefold()))
            for token in value.split():
                if token not in {"The", "A", "An", "I"} and not looks_like_false_character_name(token):
                    keys.append(re.sub(r"[^a-z0-9]+", "", token.casefold()))
        return [key for key in keys if key]

    def _content_overlap_ratio(self, left: str, right: str) -> float:
        left_tokens = self._content_tokens(left)
        right_tokens = self._content_tokens(right)
        if not left_tokens or not right_tokens:
            return 0.0
        return len(left_tokens & right_tokens) / max(len(left_tokens), 1)

    def _safe_error_text(self, exc: object) -> str:
        text = str(exc)
        text = re.sub(r"([?&]key=)[^&\s)]+", r"\1<redacted>", text)
        text = re.sub(r"(Bearer\s+)[A-Za-z0-9._~-]+", r"\1<redacted>", text)
        return text[:500]

    def _validate_redraft_payload(self, payload: Any) -> dict[str, Any]:
        if not isinstance(payload, dict):
            raise ValueError("redraft payload must be an object")
        rewrites = payload.get("rewrites", [])
        if not isinstance(rewrites, list):
            raise ValueError("redraft payload rewrites must be a list")
        cleaned: list[dict[str, str]] = []
        for item in rewrites:
            if not isinstance(item, dict):
                continue
            scene_id = str(item.get("scene_id") or "").strip()
            text = str(item.get("text") or "").strip()
            if scene_id and text:
                cleaned.append({"scene_id": scene_id, "text": text})
        return {"rewrites": cleaned}

    def _load_redraft_cache(self, path: Path) -> dict[str, Any]:
        payload = read_json(path, default={})
        if not isinstance(payload, dict) or payload.get("cache_version") != VNEXT_REDRAFT_CACHE_VERSION:
            return {"cache_version": VNEXT_REDRAFT_CACHE_VERSION, "entries": {}}
        entries = payload.get("entries")
        return payload if isinstance(entries, dict) else {"cache_version": VNEXT_REDRAFT_CACHE_VERSION, "entries": {}}

    def _save_redraft_cache(self, path: Path, cache: dict[str, Any]) -> None:
        ensure_dir(path.parent)
        write_json(path, {"cache_version": VNEXT_REDRAFT_CACHE_VERSION, "entries": cache.get("entries", cache)})

    def _redraft_cache_key(self, prompt: str, config: ScriptVNextRedraftConfig) -> str:
        payload = {
            "prompt_version": VNEXT_REDRAFT_PROMPT_VERSION,
            "provider": config.provider,
            "max_output_tokens": config.max_output_tokens,
            "prompt": prompt,
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()

    def _estimate_redraft_cost(self, prompt: str, max_output_tokens: int) -> dict[str, Any]:
        input_tokens = max(1, int(len(prompt) / 4))
        output_tokens = max(1, int(max_output_tokens))
        # Conservative Gemini Flash/Lite text-only estimate. Exact API usage is
        # not available from the shared router, so cost reports label this as an estimate.
        estimated = (input_tokens / 1_000_000 * 0.10) + (output_tokens / 1_000_000 * 0.40)
        return {
            "input_tokens_estimate": input_tokens,
            "output_tokens_estimate": output_tokens,
            "estimated_cost_usd": round(estimated, 6),
        }

    def _redraft_qc_summary(self, redraft_log: dict[str, Any]) -> dict[str, Any]:
        return {
            "enabled": bool(redraft_log.get("enabled")),
            "dry_run": bool(redraft_log.get("dry_run")),
            "target_scene_count": int(redraft_log.get("target_scene_count") or 0),
            "redrafted_scene_count": int(redraft_log.get("redrafted_scene_count") or 0),
            "unresolved_scene_ids": list(redraft_log.get("unresolved_scene_ids") or []),
            "actual_call_count": int(redraft_log.get("actual_call_count") or 0),
            "estimated_cost_usd": float(redraft_log.get("estimated_cost_usd") or 0.0),
        }

    def _apply_redraft_qc_gate(self, qc_report: dict[str, Any], redraft_log: dict[str, Any]) -> dict[str, Any]:
        unresolved = list(redraft_log.get("unresolved_scene_ids") or [])
        if not unresolved:
            return qc_report
        qc = dict(qc_report)
        failure_codes = set(qc.get("failure_codes") or [])
        failure_codes.add("unresolved_redraft_scenes")
        qc["failure_codes"] = sorted(failure_codes)
        qc["should_block_tts"] = True
        qc["summary"] = self._qc_summary(qc)
        return qc

    def _build_cost_report(
        self,
        *,
        started_at: float,
        max_cost_usd: float,
        scene_count: int,
        redraft_log: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        runtime = round(time.perf_counter() - started_at, 3)
        redraft = redraft_log or self._empty_redraft_log(None)
        call_records = [
            item for item in redraft.get("batches", []) or []
            if item.get("status") in {"completed", "failed", "blocked", "retry_completed"}
        ]
        cache_hits = sum(1 for item in redraft.get("batches", []) or [] if item.get("cache_hit"))
        estimated_cost = round(float(redraft.get("estimated_cost_usd") or 0.0), 6)
        return {
            "artifact_version": VNEXT_ARTIFACT_VERSION,
            "created_at": datetime.utcnow().isoformat(),
            "runtime_seconds": runtime,
            "max_cost_usd": max_cost_usd,
            "estimated_cost_usd": estimated_cost,
            "gemini_calls_total": int(redraft.get("actual_call_count") or 0),
            "gemini_calls_by_stage": {"vnext_scene_redraft": int(redraft.get("actual_call_count") or 0)},
            "cache_hits": cache_hits,
            "cache_misses": int(redraft.get("cache_misses") or 0),
            "scene_count": scene_count,
            "budget_exceeded": bool(redraft.get("budget_exceeded")),
            "redraft": {
                "enabled": bool(redraft.get("enabled")),
                "dry_run": bool(redraft.get("dry_run")),
                "target_scene_count": int(redraft.get("target_scene_count") or 0),
                "redrafted_scene_count": int(redraft.get("redrafted_scene_count") or 0),
                "unresolved_scene_ids": list(redraft.get("unresolved_scene_ids") or []),
                "batch_count": len(redraft.get("batches", []) or []),
                "call_record_count": len(call_records),
            },
            "notes": [
                "vNext script generation uses existing local artifacts only.",
                "Scene redraft prompts contain text evidence only; no raw panel images are sent.",
            ],
        }

    def _build_benchmark_report(self, scene_plan: dict[str, Any], qc_report: dict[str, Any], cost_report: dict[str, Any]) -> dict[str, Any]:
        return {
            "artifact_version": VNEXT_ARTIFACT_VERSION,
            "created_at": datetime.utcnow().isoformat(),
            "script_pipeline_version": "vNext",
            "scene_count": len(scene_plan.get("scenes", []) or []),
            "quality_score": qc_report.get("quality_score"),
            "should_block_tts": qc_report.get("should_block_tts"),
            "meaningful_panel_usage_rate": qc_report.get("meaningful_panel_usage_rate"),
            "long_no_tts_gap_count": qc_report.get("long_no_tts_gap_count"),
            "estimated_cost_usd": cost_report.get("estimated_cost_usd"),
            "gemini_calls_total": cost_report.get("gemini_calls_total"),
        }

    def _narrate_scene(self, scene: dict[str, Any]) -> str:
        subject = self._scene_subject(scene)
        snippets = [self._clean_sentence(snippet) for snippet in scene.get("transcript_snippets", []) if self._clean_sentence(snippet)]
        contribution_values = list((scene.get("panel_contribution_map") or {}).values())
        actions = [self._clean_sentence(item.get("visual_summary") or item.get("evidence_text")) for item in contribution_values]
        actions = [item for item in actions if item]
        summary = self._clean_sentence(scene.get("scene_summary"))
        sentences: list[str] = []
        if summary:
            sentences.append(summary)
        elif snippets:
            sentences.append(f"{subject} is pulled into the exchange when {self._trim_phrase(snippets[0], 22)}.")
        elif actions:
            sentences.append(f"{subject} faces {self._trim_phrase(actions[0], 22)}.")
        else:
            sentences.append("The chapter pauses briefly before the next turn takes over.")

        if snippets:
            sentences.append(f"The dialogue keeps attention on {self._trim_phrase(snippets[0], 18)}.")
        elif len(actions) > 1:
            sentences.append(f"Then {self._trim_phrase(actions[1], 22)}, keeping the action moving in the same direction.")

        meaningful = [
            item for item in contribution_values
            if item.get("contribution") not in {"low_information", "redundant_near_duplicate"}
        ]
        if len(meaningful) >= 2:
            first = self._clean_sentence(meaningful[0].get("visual_summary") or meaningful[0].get("evidence_text"))
            last = self._clean_sentence(meaningful[-1].get("visual_summary") or meaningful[-1].get("evidence_text"))
            if first and last and first != last:
                sentences.append(f"What begins with {self._trim_phrase(first, 14)} ends with {self._trim_phrase(last, 14)}.")
            else:
                sentences.append("The repeated details stay in one continuous turn of action and reaction.")
        return " ".join(self._ensure_sentence(sentence) for sentence in sentences[:3])

    def _expand_for_timing(self, text: str, scene: dict[str, Any]) -> str:
        supporting = [
            item for panel_id, item in (scene.get("panel_contribution_map") or {}).items()
            if panel_id in set(scene.get("supporting_panel_ids") or [])
            and item.get("contribution") not in {"low_information", "redundant_near_duplicate"}
        ]
        additions: list[str] = []
        if supporting:
            addition_text = self._clean_sentence(supporting[0].get("visual_summary") or supporting[0].get("evidence_text"))
            if addition_text:
                additions.append(f"It also shows {self._trim_phrase(addition_text, 24)}.")
        if len(supporting) > 1:
            second = self._clean_sentence(supporting[1].get("visual_summary") or supporting[1].get("evidence_text"))
            if second:
                additions.append(f"That leads into {self._trim_phrase(second, 24)}.")
        return " ".join([text, *additions])

    def _scene_summary(
        self,
        *,
        panel_count: int,
        visible_characters: list[str],
        transcript_snippets: list[str],
        evidence_phrases: list[str],
        fallback_kind: str,
    ) -> str:
        subject = visible_characters[0] if visible_characters else "The scene"
        if transcript_snippets and evidence_phrases:
            phrase = self._trim_phrase(evidence_phrases[0], 20)
            return self._ensure_sentence(f"{subject} faces {phrase}") if phrase else self._ensure_sentence(f"{subject} pauses briefly before the next turn")
        if transcript_snippets:
            return self._ensure_sentence(f"{subject} responds as the exchange turns on {self._trim_phrase(self._clean_sentence(transcript_snippets[0]), 20)}")
        if evidence_phrases:
            return self._ensure_sentence(f"{subject} faces {self._trim_phrase(evidence_phrases[0], 20)}")
        return self._ensure_sentence(f"{subject} pauses briefly before the next turn")

    def _should_start_new_scene(
        self,
        current: list[PanelBox],
        current_kind: str,
        panel: PanelBox,
        next_kind: str,
        evidence: dict[str, dict[str, Any]],
    ) -> bool:
        previous = current[-1]
        if int(panel.page) < int(previous.page) or int(panel.order) <= int(previous.order):
            return True
        if int(panel.page) - int(previous.page) > 1:
            return True
        if len(current) >= 6:
            return True
        if len(current) >= 4 and next_kind != current_kind:
            return True
        current_has_dialogue = any(self._panel_dialogue_text(evidence.get(item.id, {})) for item in current)
        next_has_dialogue = bool(self._panel_dialogue_text(evidence.get(panel.id, {})))
        if current_has_dialogue != next_has_dialogue and len(current) >= 3:
            return True
        if next_kind == "transition" and len(current) >= 2:
            return True
        return False

    def _representative_panel(self, panels: list[PanelBox], evidence: dict[str, dict[str, Any]]) -> PanelBox:
        def score(panel: PanelBox) -> float:
            item = evidence.get(panel.id, {})
            value = 0.0
            value += 4.0 if self._panel_visual_text(item) else 0.0
            value += 3.0 if self._panel_dialogue_text(item) else 0.0
            value += min(float(panel.width * panel.height) / 200_000.0, 3.0)
            value -= 2.0 if item.get("visual_only") and not self._panel_visual_text(item) else 0.0
            return value
        return max(panels, key=score)

    def _panel_kind(self, item: dict[str, Any]) -> str:
        combined = self._normalize(" ".join([self._panel_dialogue_text(item), self._panel_visual_text(item)]))
        if not combined:
            return "transition"
        if re.search(r"\b(?:hit|attack|punch|kick|blood|crash|fight|weapon|danger|run|grab|throw|slam|charge|battle)\b", combined):
            return "action"
        if self._panel_dialogue_text(item):
            return "dialogue"
        if re.search(r"\b(?:shock|angry|sad|cry|stunned|afraid|smile|panic|worry)\b", combined):
            return "reaction"
        return "visual"

    def _classify_contribution(self, text: str, visual: str, tokens: set[str], previous_tokens: set[str]) -> str:
        if not tokens:
            return "low_information"
        if previous_tokens and len(tokens & previous_tokens) / max(len(tokens | previous_tokens), 1) >= 0.78:
            return "redundant_near_duplicate"
        combined = self._normalize(" ".join([text, visual]))
        if text:
            return "dialogue_meaning"
        if re.search(r"\b(?:hit|attack|punch|kick|fight|run|grab|throw|slam|charge|blood|crash|weapon|danger)\b", combined):
            return "concrete_action"
        if re.search(r"\b(?:shock|angry|sad|cry|stunned|afraid|panic|worry|smile)\b", combined):
            return "character_reaction"
        if re.search(r"\b(?:classroom|hallway|city|room|building|battlefield|school|outside|inside)\b", combined):
            return "setting_context"
        if re.search(r"\b(?:reveal|realize|turn|change|decide|choice|consequence)\b", combined):
            return "plot_consequence"
        return "visual_escalation"

    def _contribution_reason(self, contribution: str) -> str:
        return {
            "dialogue_meaning": "clean dialogue or narration evidence contributes to the scene",
            "concrete_action": "visual evidence describes a concrete action beat",
            "character_reaction": "visual evidence captures a character reaction",
            "setting_context": "panel establishes location or context",
            "plot_consequence": "panel changes the story state",
            "visual_escalation": "visual-only beat changes intensity or focus",
            "redundant_near_duplicate": "near-duplicate of adjacent evidence",
            "low_information": "no reliable dialogue or semantic visual evidence",
        }.get(contribution, "unresolved panel contribution")

    def _scene_character_roles(self, visible: list[str], speakers: list[str], mentioned: list[str]) -> dict[str, list[str]]:
        roles: dict[str, list[str]] = {}
        for name in visible:
            roles.setdefault(name, []).append("visible_present")
        for name in speakers:
            roles.setdefault(name, []).append("speaker")
        for name in mentioned:
            roles.setdefault(name, []).append("mentioned_absent")
        return roles

    def _load_character_registry(self, project_dir: Path) -> dict[str, Any]:
        for filename in ("character_registry.json", "canonical_characters.json", "character_dictionary.json"):
            payload = read_json(project_dir / "output" / filename, default=None)
            if payload:
                return {"source": filename, "characters": payload}
        return {"source": "none", "characters": []}

    def _read_panel_vision_records(self, project_dir: Path) -> list[dict[str, Any]]:
        payload = read_json(project_dir / "output" / "panel_vision_final.json", default=[])
        records = payload.get("records", []) if isinstance(payload, dict) else payload
        return [item for item in records if isinstance(item, dict)] if isinstance(records, list) else []

    def _read_panel_evidence_records(self, project_dir: Path) -> list[dict[str, Any]]:
        payload = read_json(project_dir / "output" / "panel_evidence.json", default=[])
        records = payload.get("panels", []) if isinstance(payload, dict) else payload
        return [item for item in records if isinstance(item, dict)] if isinstance(records, list) else []

    def _scene_usage_evidence_record(self, scene: dict[str, Any], panel_id: str, item: dict[str, Any]) -> dict[str, Any]:
        text = " ".join(
            part
            for part in [str(item.get("evidence_text") or "").strip(), str(item.get("visual_summary") or "").strip()]
            if part
        )
        return {
            "panel_id": panel_id,
            "panel_order": item.get("panel_order"),
            "page": item.get("page"),
            "dialogue_text": text if item.get("contribution") == "dialogue_meaning" else "",
            "text_english": text,
            "visual_summary": str(item.get("visual_summary") or "").strip(),
            "action_beat": str(item.get("visual_summary") or "").strip(),
            "summary": scene.get("scene_summary"),
            "source": "script_vnext_scene_plan",
        }

    def _input_artifact_statuses(self, evidence: dict[str, dict[str, Any]], character_registry: dict[str, Any]) -> dict[str, Any]:
        source_counts: dict[str, int] = {}
        for item in evidence.values():
            for source in item.get("source_artifacts", []) or []:
                source_counts[source] = source_counts.get(source, 0) + 1
        return {
            "evidence_source_counts": source_counts,
            "character_registry_source": character_registry.get("source", "none"),
        }

    def _story_segments_from_plan_orders(self, scene: dict[str, Any]) -> tuple[int | None, int | None]:
        orders = [
            self._safe_int(item.get("panel_order"))
            for item in (scene.get("panel_contribution_map") or {}).values()
            if self._safe_int(item.get("panel_order"))
        ]
        return (min(orders), max(orders)) if orders else (None, None)

    def _story_segments_from_chunks(self, narration_chunks: dict[str, Any]) -> list[StorySegment]:
        segments: list[StorySegment] = []
        for index, chunk in enumerate(narration_chunks.get("chunks", []) or [], start=1):
            segment = StorySegment(
                id=str(chunk["chunk_id"]),
                order=index,
                text=str(chunk["text"]).strip(),
                keep=True,
                panel_ids=list(chunk["source_panel_ids"]),
                panel_start=self._safe_int(chunk.get("panel_start")) or None,
                panel_end=self._safe_int(chunk.get("panel_end")) or None,
                scene_id=index,
                title=f"Scene {index}",
                representative_panel_id=str(chunk["representative_panel_id"]),
                visual_only=False,
                suppression_reason=None,
            )
            segments.append(segment)
        return segments

    def _panel_dialogue_text(self, item: dict[str, Any]) -> str:
        return self._clean_text(" ".join(str(item.get(key) or "") for key in ("dialogue", "caption") if str(item.get(key) or "").strip()))

    def _panel_visual_text(self, item: dict[str, Any]) -> str:
        action = self._clean_text(str(item.get("action_beat") or ""))
        if action:
            return action
        emotion = self._clean_text(str(item.get("emotion") or ""))
        if not emotion or self._is_weak_emotion_label(emotion):
            return ""
        names = [
            str(name).strip()
            for name in item.get("character_names", []) or []
            if str(name).strip() and not looks_like_false_character_name(name)
        ]
        if not names:
            return ""
        return f"{names[0]} looks {emotion.casefold()}"

    def _short_evidence_phrase(self, text: str, visual: str) -> str:
        phrase = self._clean_sentence(text or visual)
        words = phrase.split()
        return " ".join(words[:18]).rstrip(",.;") + ("..." if len(words) > 18 else "")

    def _scene_subject(self, scene: dict[str, Any]) -> str:
        for key in ("visible_characters", "speakers"):
            values = [str(value).strip() for value in scene.get(key, []) or [] if str(value).strip()]
            if values:
                return values[0]
        return "The scene"

    def _scene_has_enough_grounded_content(self, scene: dict[str, Any]) -> bool:
        return any(
            item.get("contribution") not in {"low_information", "redundant_near_duplicate"}
            for item in (scene.get("panel_contribution_map") or {}).values()
        )

    def _estimate_duration(self, text: str) -> float:
        words = len(re.findall(r"\b[\w'-]+\b", text))
        return round(words / 2.7, 2)

    def _default_panel_duration(self, item: dict[str, Any]) -> float:
        if self._panel_dialogue_text(item):
            return 2.2
        if self._panel_visual_text(item):
            return 1.8
        return 1.0

    def _append_text(self, target: dict[str, Any], key: str, value: Any) -> None:
        incoming = self._clean_text(str(value or ""))
        if not incoming:
            return
        current = str(target.get(key) or "").strip()
        if incoming.casefold() in current.casefold():
            return
        target[key] = f"{current} {incoming}".strip()

    def _evidence_by_order(self, evidence: dict[str, dict[str, Any]], order: int) -> dict[str, Any] | None:
        if not order:
            return None
        return next((item for item in evidence.values() if int(item.get("panel_order") or 0) == int(order)), None)

    def _content_tokens(self, text: str) -> set[str]:
        stop = {"the", "and", "that", "this", "with", "from", "into", "scene", "panel", "moment", "beat", "their", "there", "they"}
        return {token for token in re.findall(r"[a-z']+", self._normalize(text)) if len(token) > 2 and token not in stop}

    def _clean_text(self, text: str) -> str:
        text = re.sub(r"\s+", " ", str(text or "")).strip()
        if not text:
            return ""
        lowered = text.casefold()
        if re.search(r"(?:[.·・]{2,}|[=≡]{2,}|[^\w\s,.!?'\-:;]{4,})", text):
            return ""
        if len(set(re.findall(r"[A-Za-z]+", lowered))) <= 2 and len(text.split()) > 10:
            return ""
        return text

    def _clean_sentence(self, text: Any) -> str:
        cleaned = self._clean_text(str(text or ""))
        return cleaned[:1].upper() + cleaned[1:] if cleaned else ""

    def _ensure_sentence(self, text: str) -> str:
        text = self._clean_sentence(text).strip()
        if not text:
            return ""
        return text if text.endswith((".", "!", "?")) else f"{text}."

    def _normalize(self, text: str) -> str:
        return re.sub(r"\s+", " ", str(text or "").casefold()).strip()

    def _trim_phrase(self, text: str, max_words: int) -> str:
        words = self._clean_sentence(text).split()
        if not words:
            return ""
        trimmed = " ".join(words[:max_words]).rstrip(",.;:")
        return f"{trimmed}..." if len(words) > max_words else trimmed.rstrip(".")

    def _is_weak_emotion_label(self, text: str) -> bool:
        value = self._normalize(text).strip(" .!?")
        return value in {
            "neutral",
            "calm",
            "normal",
            "unknown",
            "none",
            "blank",
            "unreadable",
            "surprised",
            "scared",
            "angry",
            "sad",
            "smug",
            "injured",
            "ominous",
            "shock",
            "shocked",
            "impact",
        }

    def _safe_int(self, value: Any) -> int:
        try:
            return int(value or 0)
        except Exception:
            return 0

    def _qc_summary(self, qc: dict[str, Any]) -> str:
        status = "blocked before TTS" if qc.get("should_block_tts") else "safe for TTS"
        return (
            f"vNext script quality score {qc.get('quality_score', 0)}/100; "
            f"meaningful usage {qc.get('meaningful_panel_usage_rate', 1.0)}; "
            f"long gaps {qc.get('long_no_tts_gap_count', 0)}; {status}."
        )
