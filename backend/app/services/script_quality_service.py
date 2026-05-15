"""DEPRECATED - see app/services/DEPRECATED.md.

ScriptQualityService is a 2,200-line quality gate for the legacy narration
cascade. The vision pipeline (PanelVisionNarrator) does not need it -
panels that fail are flagged for in-place regeneration. Kept on disk for
projects that still use script_pipeline_version="legacy".
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Any

from app.schemas.project import PanelBox, StorySegment
from app.services.character_name_filters import looks_like_false_character_name, normalize_name_key
from app.services.script_cleaner_service import ScriptCleanerService


class ScriptQualityService:
    def __init__(self) -> None:
        self.cleaner = ScriptCleanerService()

    def analyze_story_segments(
        self,
        story_segments: list[StorySegment],
        *,
        panel_vision_records: list[dict[str, Any]] | None = None,
        panel_evidence_records: list[dict[str, Any]] | None = None,
        panels: list[PanelBox] | None = None,
    ) -> dict[str, Any]:
        ordered_segments = [
            segment
            for segment in sorted(story_segments, key=lambda item: item.order)
            if bool(getattr(segment, "keep", True))
        ]
        lines = [str(segment.text or "").strip() for segment in ordered_segments]
        # Detect panel mode: ≥85% of segments have exactly 1 panel_id.
        # Panel mode produces structurally different output (1 sentence per segment,
        # short lines, adjacent-panel disconnects) so many thresholds must be relaxed.
        _single_panel_count = sum(
            1 for s in ordered_segments
            if len(getattr(s, "panel_ids", None) or []) == 1
        )
        is_panel_mode = len(ordered_segments) > 2 and _single_panel_count / max(len(ordered_segments), 1) >= 0.85
        duplicate_count = self._duplicate_count(lines)
        blank_count = sum(1 for line in lines if not line)
        visual_only_blank_count = sum(
            1 for segment, line in zip(ordered_segments, lines, strict=False)
            if not line and bool(getattr(segment, "visual_only", False))
        )
        total_panel_refs = sum(max(len(getattr(segment, "panel_ids", []) or []), 1) for segment in ordered_segments)
        visual_only_panel_refs = sum(
            max(len(getattr(segment, "panel_ids", []) or []), 1)
            for segment, line in zip(ordered_segments, lines, strict=False)
            if not line and bool(getattr(segment, "visual_only", False))
        )
        blocking_blank_count = max(blank_count - visual_only_blank_count, 0)
        non_empty_count = len(lines) - blank_count
        first_person_count = sum(1 for line in lines if line and self.cleaner.is_first_person_narration(line))
        generic_count = sum(1 for line in lines if line and self._looks_generic(line))
        filler_meta_count = sum(1 for line in lines if line and self._has_filler_meta_language(line))
        visual_count = sum(1 for line in lines if line and self._looks_visual(line))
        ocr_contamination_count = sum(1 for line in lines if line and self._looks_ocr_contaminated(line))
        malformed_count = sum(1 for line in lines if line and self._looks_malformed(line))
        repetitive_template_count = self._repetitive_template_count(lines)
        semantic_duplicate_count = self._semantic_near_duplicate_count(lines)
        disconnected_count = self._disconnected_pair_count(lines)
        scene_order_regression_count = self._scene_order_regression_count(ordered_segments)
        panel_order_regression_count = self._panel_order_regression_count(ordered_segments)
        panel_coverage_report = self._panel_coverage_report(ordered_segments, panels or [])
        scene_usage_report = self._scene_usage_report(
            ordered_segments,
            panels or [],
            panel_evidence_records=panel_evidence_records or [],
            panel_vision_records=panel_vision_records or [],
            is_panel_mode=is_panel_mode,
        )
        meaningful_usage_score = int(scene_usage_report.get("meaningful_usage_score", 100))
        timing_alignment_score = int(scene_usage_report.get("timing_alignment_score", 100))
        unused_meaningful_panel_count = int(scene_usage_report.get("unused_meaningful_panel_count", 0) or 0)
        overcompressed_scene_count = int(scene_usage_report.get("overcompressed_scene_count", 0) or 0)
        suspicious_grouping_count = int(scene_usage_report.get("suspicious_grouping_count", 0) or 0)
        action_without_concrete_count = int(scene_usage_report.get("action_scene_without_concrete_action_count", 0) or 0)
        vague_scene_count = int(scene_usage_report.get("abstract_or_vague_scene_count", 0) or 0)
        one_sentence_multipanel_count = int(scene_usage_report.get("one_sentence_multipanel_scene_count", 0) or 0)
        long_gap_count = int(scene_usage_report.get("long_unintentional_gap_count", 0) or 0)
        story_continuity_score = int(panel_coverage_report.get("story_continuity_score", 100))
        duplicated_panel_count = int(panel_coverage_report.get("duplicated_panel_count", 0))
        largest_skipped_panel_gap = int(panel_coverage_report.get("largest_skipped_panel_gap", 0))
        underexplained_panel_ranges = self._underexplained_panel_ranges(ordered_segments, panel_coverage_report)
        underexplained_panel_range_count = len(underexplained_panel_ranges)
        late_worldbuilding_context_count = self._late_worldbuilding_context_count(ordered_segments)
        avg_sentences_per_line = self._avg_sentences_per_line(lines)
        avg_sentences_per_spoken_paragraph = avg_sentences_per_line
        sentence_counts = self._sentence_counts(lines)
        word_counts = self._word_counts(lines)
        short_line_word_threshold = 18
        one_sentence_count = sum(1 for count in sentence_counts if count <= 1)
        max_one_sentence_run = self._max_one_sentence_run(lines)
        short_line_count = sum(1 for count in word_counts if count < short_line_word_threshold)
        short_line_under_30_count = sum(1 for count in word_counts if count < 30)
        caption_like_count = sum(1 for line in lines if line and self._looks_caption_like(line))
        caption_like_sentence_count = sum(self._caption_like_sentence_count(line) for line in lines if line)
        weak_transition_count = sum(1 for line in lines if line and self._has_weak_transition(line))
        unclear_pronoun_count = sum(1 for line in lines if line and self._has_unclear_pronouns(line))
        vague_subject_count = sum(1 for line in lines if line and self._has_vague_subject(line))
        speculation_count = sum(1 for line in lines if line and self._sounds_speculative(line))
        flashback_confusion_count = sum(1 for line in lines if line and self._has_confusing_flashback_label(line))
        ability_ambiguity_count = sum(1 for line in lines if line and self._has_unexplained_ability_reference(line))
        invalid_name_count = sum(self._invalid_name_uses(line) for line in lines if line)
        ocr_garbage_leak_count = sum(1 for line in lines if line and self._has_ocr_garbage_leak(line))
        role_errors = self._character_role_grounding_errors(ordered_segments, panel_vision_records or [])
        mentioned_as_present_count = sum(1 for item in role_errors if item["reason"] == "mentioned_absent_acts")
        flashback_present_error_count = sum(1 for item in role_errors if item["reason"] == "flashback_as_current")
        unsupported_action_count = sum(1 for item in role_errors if item["reason"] == "action_without_visible_evidence")
        median_words_per_line = self._median(word_counts)
        p10_words_per_line = self._percentile(word_counts, 0.10)
        disconnected_penalty_count = disconnected_count
        if is_panel_mode:
            # In panel mode each panel is its own slot; adjacent segments from different
            # scenes are inherently disconnected. Charge only the excess beyond 30%.
            disconnected_penalty_count = max(0, disconnected_count - round(max(len(lines), 1) * 0.30))
        elif avg_sentences_per_line >= 1.75:
            # Multi-sentence story segments naturally jump between scene beats.
            # Penalize only excess zero-overlap transitions instead of treating
            # every scene boundary like a panel-by-panel continuity failure.
            disconnected_penalty_count = max(0, disconnected_count - round(max(len(lines), 1) * 0.25))
        repetitive_penalty_weight = 24
        vision_low_confidence_count = sum(
            1
            for item in panel_vision_records or []
            if float(item.get("confidence") or 0.0) < 0.55 or bool(item.get("visual_only"))
        )
        unknown_speaker_count = sum(
            1
            for item in panel_vision_records or []
            if str(item.get("speaker") or "").strip().casefold() == "unknown"
            and bool(str(item.get("dialogue") or "").strip())
        )

        risky_segments: list[dict[str, Any]] = []
        for segment in ordered_segments:
            text = str(segment.text or "").strip()
            reasons: list[str] = []
            if not text:
                if bool(getattr(segment, "visual_only", False)):
                    reasons.append("visual_only_blank")
                else:
                    reasons.append("blank")
            if text and self.cleaner.is_first_person_narration(text):
                reasons.append("first_person")
            if text and self._looks_generic(text):
                reasons.append("generic")
            if text and self._has_filler_meta_language(text):
                reasons.append("filler_meta")
            if text and self._looks_visual(text):
                reasons.append("visual")
            if text and self._looks_ocr_contaminated(text):
                reasons.append("ocr_contamination")
            if text and self._looks_malformed(text):
                reasons.append("malformed")
            if text:
                sentence_count = len([part for part in re.split(r"(?<=[.!?])\s+(?=[A-Z])", text) if part.strip()])
                word_count = len(re.findall(r"\b[\w'-]+\b", text))
                if sentence_count <= 1:
                    reasons.append("one_sentence")
                if word_count < short_line_word_threshold:
                    reasons.append("short_line")
                if self._looks_caption_like(text):
                    reasons.append("caption_like")
                if self._caption_like_sentence_count(text):
                    reasons.append("caption_like_sentence")
                if self._has_weak_transition(text):
                    reasons.append("weak_transition")
                if self._has_unclear_pronouns(text):
                    reasons.append("unclear_pronouns")
                if self._has_vague_subject(text):
                    reasons.append("vague_subject")
                if self._sounds_speculative(text):
                    reasons.append("speculation")
                if self._has_confusing_flashback_label(text):
                    reasons.append("flashback_confusion")
                if self._has_unexplained_ability_reference(text):
                    reasons.append("ability_ambiguity")
                if self._segment_has_panel_order_regression(segment, ordered_segments):
                    reasons.append("panel_order_regression")
                if self._segment_has_panel_duplication(segment, panel_coverage_report):
                    reasons.append("duplicated_panel_range")
                if any(str(item.get("segment_id")) == segment.id for item in underexplained_panel_ranges):
                    reasons.append("underexplained_panel_range")
                if self._looks_like_late_worldbuilding(segment, ordered_segments):
                    reasons.append("late_worldbuilding_context")
                scene_usage = scene_usage_report.get("scenes_by_segment_id", {}).get(str(segment.id), {})
                if scene_usage:
                    if scene_usage.get("unused_meaningful_panel_ids"):
                        reasons.append("unused_meaningful_panels")
                    if scene_usage.get("is_overcompressed"):
                        reasons.append("overcompressed_scene")
                    if scene_usage.get("suspicious_grouping_reasons"):
                        reasons.append("suspicious_panel_grouping")
                    if scene_usage.get("needs_narration_expansion"):
                        reasons.append("timing_gap")
                    if scene_usage.get("action_scene_without_concrete_action"):
                        reasons.append("action_without_concrete_action")
                    if scene_usage.get("abstract_or_vague_narration"):
                        reasons.append("abstract_vague_narration")
                if self._invalid_name_uses(text):
                    reasons.append("invalid_name")
                if self._has_ocr_garbage_leak(text):
                    reasons.append("ocr_garbage")
                if any(int(item.get("segment_order") or 0) == int(segment.order) for item in role_errors):
                    reasons.append("character_role_grounding")
            if reasons:
                risky_segments.append(
                    {
                        "segment_id": segment.id,
                        "order": segment.order,
                        "scene_id": segment.scene_id,
                        "panel_start": segment.panel_start,
                        "panel_end": segment.panel_end,
                        "reasons": reasons,
                        "text": text,
                    }
                )

        total_segments = len(lines)
        quality_score = 100
        if total_segments:
            quality_score -= round((blocking_blank_count / total_segments) * 36)
            quality_score -= round((duplicate_count / total_segments) * 35)
            quality_score -= round((generic_count / total_segments) * 28)
            quality_score -= round((filler_meta_count / total_segments) * 36)
            quality_score -= round((visual_count / total_segments) * 22)
            quality_score -= round((ocr_contamination_count / total_segments) * 34)
            quality_score -= round((malformed_count / total_segments) * 42)
            quality_score -= round((repetitive_template_count / total_segments) * repetitive_penalty_weight)
            quality_score -= round((semantic_duplicate_count / total_segments) * 26)
            quality_score -= round((disconnected_penalty_count / total_segments) * 18)
            quality_score -= min(scene_order_regression_count * 14, 28)
            quality_score -= min(panel_order_regression_count * 18, 36)
            quality_score -= round((visual_only_blank_count / total_segments) * 18)
            # In panel mode every segment is 1 sentence by design; don't penalise that.
            if not is_panel_mode:
                quality_score -= round((one_sentence_count / total_segments) * 20)
                quality_score -= round((short_line_count / total_segments) * 16)
                quality_score -= max(0, max_one_sentence_run - 2) * 3
            quality_score -= round((caption_like_count / total_segments) * 18)
            quality_score -= round((caption_like_sentence_count / total_segments) * 12)
            quality_score -= round((weak_transition_count / total_segments) * 12)
            quality_score -= round((unclear_pronoun_count / total_segments) * 10)
            quality_score -= round((vague_subject_count / total_segments) * 10)
            quality_score -= round((speculation_count / total_segments) * 10)
            quality_score -= round((flashback_confusion_count / total_segments) * 12)
            quality_score -= round((ability_ambiguity_count / total_segments) * 12)
            quality_score -= round((invalid_name_count / total_segments) * 24)
            quality_score -= round((ocr_garbage_leak_count / total_segments) * 24)
            quality_score -= round((mentioned_as_present_count / total_segments) * 30)
            quality_score -= round((unsupported_action_count / total_segments) * 22)
            quality_score -= round((flashback_present_error_count / total_segments) * 18)
            quality_score -= round(((100 - story_continuity_score) / 100) * 32)
            # In panel mode, token-overlap measurement for meaningful usage is less reliable
            # (single-sentence paraphrases don't always share tokens with OCR); halve the weight.
            _meaningful_usage_weight = 18 if is_panel_mode else 34
            quality_score -= round(((100 - meaningful_usage_score) / 100) * _meaningful_usage_weight)
            # In panel mode, single-sentence segments are inherently short vs. panel duration;
            # reduce the timing penalty weight to avoid false failures.
            _timing_weight = 8 if is_panel_mode else 18
            quality_score -= round(((100 - timing_alignment_score) / 100) * _timing_weight)
            quality_score -= round((duplicated_panel_count / max(int(panel_coverage_report.get("meaningful_input_panels", 0) or 0), 1)) * 28)
            quality_score -= min(underexplained_panel_range_count * 12, 36)
            quality_score -= min(late_worldbuilding_context_count * 16, 32)
            quality_score -= min(overcompressed_scene_count * 8, 32)
            quality_score -= min(suspicious_grouping_count * 8, 32)
            quality_score -= min(action_without_concrete_count * 10, 30)
            quality_score -= min(vague_scene_count * 6, 30)
            # one_sentence_multipanel fires when a segment covers multiple panels but has only 1 sentence;
            # that can't happen in panel mode (1 panel per segment), so this is always 0.
            quality_score -= min(one_sentence_multipanel_count * 5, 25)
        visual_only_panel_ratio = visual_only_panel_refs / max(total_panel_refs, 1)
        if total_panel_refs:
            # In panel mode, visual_only_blank_count already penalizes blank panels;
            # halve this secondary ratio-based penalty to avoid double-counting.
            _visual_ratio_weight = 6 if is_panel_mode else 12
            quality_score -= round(visual_only_panel_ratio * _visual_ratio_weight)
        vision_quality_affects_tts = bool(panel_vision_records and (blocking_blank_count or visual_only_blank_count))
        if vision_quality_affects_tts:
            quality_score -= round((vision_low_confidence_count / max(len(panel_vision_records), 1)) * 12)
            quality_score -= round((unknown_speaker_count / max(len(panel_vision_records), 1)) * 8)
        if first_person_count:
            quality_score -= min(first_person_count * 8, 24)
        if panel_coverage_report.get("has_panel_source"):
            if float(panel_coverage_report.get("coverage_ratio", 1.0) or 0.0) < 0.90:
                quality_score = min(quality_score, 89)
            if bool(panel_coverage_report.get("has_large_skipped_gap")):
                quality_score = min(quality_score, 89)
            if int(panel_coverage_report.get("out_of_order_panel_references", 0) or 0) > 0:
                quality_score = min(quality_score, 84)
            if int(panel_coverage_report.get("duplicated_panel_count", 0) or 0) > max(1, round(int(panel_coverage_report.get("meaningful_input_panels", 0) or 0) * 0.02)):
                quality_score = min(quality_score, 88)
            if underexplained_panel_range_count:
                quality_score = min(quality_score, 89)
        if scene_usage_report.get("has_panel_source"):
            _min_usage = 0.65 if is_panel_mode else 0.90
            if float(scene_usage_report.get("meaningful_panel_usage_rate", 1.0) or 0.0) < _min_usage:
                quality_score = min(quality_score, 89)
            _unused_threshold = max(2, total_segments // 8) if is_panel_mode else 0
            if unused_meaningful_panel_count > _unused_threshold:
                quality_score = min(quality_score, 89)
            if overcompressed_scene_count or suspicious_grouping_count:
                quality_score = min(quality_score, 89)
            if action_without_concrete_count or vague_scene_count:
                quality_score = min(quality_score, 89)
            if long_gap_count:
                quality_score = min(quality_score, 89)
        if late_worldbuilding_context_count:
            quality_score = min(quality_score, 89)
        quality_score = max(0, min(100, quality_score))

        thresholds = {
            "blank": max(1, total_segments // 8),
            "duplicate": max(1, total_segments // 10),
            "generic": max(2, total_segments // 7),
            # filler_meta: block only when the script is pervasively contaminated
            # (>10% of segments use hedging or meta-narrative filler phrases).
            # Score already penalizes lower rates.  True structural fillers like
            # "the beat keeps moving" should be caught by repair before TTS.
            # In panel mode single-sentence segments are more prone to occasional hedging;
            # raise to 15% before blocking (was 10%).
            "filler_meta": max(3, round(total_segments * (0.15 if is_panel_mode else 0.10))),
            # visual: "expression", "eyes wide" etc. appear legitimately in anime narration.
            # Block only when the script contains camera/panel direction leakage (>5% of
            # segments); the score still penalizes lower counts.
            "visual": max(3, round(total_segments * 0.05)),
            # Story-first narration intentionally leaves some panels silent so
            # the edit can breathe. Treat visual-only blanks as blocking only
            # when they dominate the script; the score still penalizes them.
            "visual_only_segments": max(8, round(max(total_segments, 1) * 0.45)),
            "visual_only_panels": max(12, round(max(total_panel_refs, 1) * 0.50)),
            # In panel mode every segment is intentionally 1 sentence and often short;
            # never block on those structural properties.
            "one_sentence": total_segments + 1 if is_panel_mode else max(2, total_segments // 6),
            "one_sentence_run": total_segments + 1 if is_panel_mode else 3,
            "short_line": total_segments + 1 if is_panel_mode else max(2, total_segments // 6),
            # In panel mode single-sentence segments are structural; allow more caption-like lines.
            "caption_like": max(4, round(total_segments * (0.30 if is_panel_mode else 0.20))),
            "caption_like_sentence": max(2, total_segments // 8),
            "weak_transition": max(3, total_segments // 7),
            # unclear_pronoun: gender-swap errors (he↔she on named characters) are
            # hard blockers at 0.  Ambiguous pronoun starters are common in anime narration;
            # only block if they're unusually dense.
            # In panel mode adjacent panels lack context for anaphora resolution; allow 1-2.
            "unclear_pronoun": max(1 if is_panel_mode else 0, round(total_segments * 0.04)),
            # vague_subject, flashback_confusion, ability_ambiguity: these heuristics
            # over-fire on world-building exposition and internal monologue which are
            # legitimate in anime scripts.  Penalize the score but only block when
            # pervasive enough to meaningfully harm narration.
            "vague_subject": max(2, total_segments // 20),
            # In panel mode hedging phrases ("appears to", "seems to") are more common in
            # single-sentence paraphrases; allow a slightly higher count before blocking.
            "speculation": max(3 if is_panel_mode else 2, total_segments // 12),
            "flashback_confusion": max(2, total_segments // 25),
            "ability_ambiguity": max(3, total_segments // 20),
            "invalid_names": 0,
            "ocr_garbage": 0,
            "mentioned_as_present": 0,
            "unsupported_character_action": max(1, total_segments // 18),
            "flashback_present_error": 0,
            "minimum_panel_coverage": 90,
            "large_skipped_panel_gap": int(panel_coverage_report.get("large_gap_threshold", 0) or 0),
            "duplicated_panels": max(1, round(int(panel_coverage_report.get("meaningful_input_panels", 0) or 0) * 0.02)),
            "underexplained_panel_ranges": max(0, round(total_segments * 0.10)),
            "late_worldbuilding_context": 0,
            # In panel mode token-overlap measurement is less reliable; use 65% threshold.
            "minimum_meaningful_panel_usage": 65 if is_panel_mode else 90,
            # In panel mode some panels have no OCR and produce blank; allow a small number.
            "unused_meaningful_panels": max(2, total_segments // 8) if is_panel_mode else 0,
            "overcompressed_scenes": 0,
            "suspicious_panel_grouping": 0,
            # In panel mode a single action sentence may not include all specific verbs;
            # allow 1 miss before blocking.
            "action_without_concrete_action": 1 if is_panel_mode else 0,
            # In panel mode single-sentence narration can be brief/generic without full context;
            # allow a slightly higher rate before blocking.
            "abstract_or_vague_scenes": max(2 if is_panel_mode else 1, total_segments // 10),
            # In panel mode every segment is 1 sentence, so timing gaps are structural.
            "long_unintentional_gaps": total_segments if is_panel_mode else 0,
            # In panel mode adjacent panels often cover the same scene beat; allow more duplicates.
            "semantic_duplicate": max(2, round(total_segments * (0.22 if is_panel_mode else 0.08))),
            # In panel mode every panel is its own narration slot; adjacent segments are from
            # different scenes and therefore naturally disconnected. Use 50% as the blocker threshold.
            "disconnected": max(3, round(max(total_segments, 1) * (0.50 if is_panel_mode else 0.15))),
            "scene_order_regression": 0,
            "panel_order_regression": 0,
            "score": 68,
        }
        excessive_visual_only = (
            visual_only_blank_count > thresholds["visual_only_segments"]
            or visual_only_panel_refs > thresholds["visual_only_panels"]
        )
        should_block_tts = any(
            (
                blocking_blank_count > thresholds["blank"],
                duplicate_count > thresholds["duplicate"],
                generic_count > thresholds["generic"],
                filler_meta_count > thresholds["filler_meta"],
                visual_count > thresholds["visual"],
                ocr_contamination_count > max(1, total_segments // 12),
                malformed_count > 0,
                semantic_duplicate_count > thresholds["semantic_duplicate"],
                disconnected_count > thresholds["disconnected"],
                scene_order_regression_count > thresholds["scene_order_regression"],
                panel_order_regression_count > thresholds["panel_order_regression"],
                excessive_visual_only,
                # In panel mode every segment is 1 sentence and short by design; never block on those.
                (not is_panel_mode) and one_sentence_count > max(thresholds["one_sentence"], round(total_segments * 0.30)),
                (not is_panel_mode) and max_one_sentence_run > thresholds["one_sentence_run"],
                (not is_panel_mode) and short_line_count > max(thresholds["short_line"], round(total_segments * 0.30)),
                caption_like_count > thresholds["caption_like"],
                caption_like_sentence_count > thresholds["caption_like_sentence"],
                unclear_pronoun_count > thresholds["unclear_pronoun"],
                vague_subject_count > thresholds["vague_subject"],
                speculation_count > thresholds["speculation"],
                flashback_confusion_count > thresholds["flashback_confusion"],
                ability_ambiguity_count > thresholds["ability_ambiguity"],
                invalid_name_count > thresholds["invalid_names"],
                ocr_garbage_leak_count > thresholds["ocr_garbage"],
                mentioned_as_present_count > thresholds["mentioned_as_present"],
                unsupported_action_count > thresholds["unsupported_character_action"],
                flashback_present_error_count > thresholds["flashback_present_error"],
                bool(panel_coverage_report.get("insufficient_panel_coverage")),
                bool(panel_coverage_report.get("has_large_skipped_gap")),
                int(panel_coverage_report.get("out_of_order_panel_references", 0) or 0) > 0,
                int(panel_coverage_report.get("duplicated_panel_count", 0) or 0) > thresholds["duplicated_panels"],
                underexplained_panel_range_count > thresholds["underexplained_panel_ranges"],
                late_worldbuilding_context_count > thresholds["late_worldbuilding_context"],
                # In panel mode use the relaxed usage threshold (65%) instead of the default 90%.
                (
                    bool(scene_usage_report.get("insufficient_meaningful_panel_usage"))
                    if not is_panel_mode
                    else float(scene_usage_report.get("meaningful_panel_usage_rate", 1.0) or 0.0) < 0.65
                ),
                unused_meaningful_panel_count > thresholds["unused_meaningful_panels"],
                overcompressed_scene_count > thresholds["overcompressed_scenes"],
                suspicious_grouping_count > thresholds["suspicious_panel_grouping"],
                action_without_concrete_count > thresholds["action_without_concrete_action"],
                vague_scene_count > thresholds["abstract_or_vague_scenes"],
                long_gap_count > thresholds["long_unintentional_gaps"],
                first_person_count > 0,
                quality_score < thresholds["score"],
            )
        )

        return {
            "analysis_mode": "story_segments_v1",
            "analysis_version": 3,
            "total_segments": total_segments,
            "total_script_lines": total_segments,
            "non_empty_lines": non_empty_count,
            "blank_lines": blank_count,
            "blocking_blank_lines": blocking_blank_count,
            "visual_only_blank_lines": visual_only_blank_count,
            "total_panel_refs": total_panel_refs,
            "spoken_panel_refs": max(total_panel_refs - visual_only_panel_refs, 0),
            "visual_only_panel_refs": visual_only_panel_refs,
            "visual_only_segment_ratio": round(visual_only_blank_count / max(total_segments, 1), 4),
            "visual_only_panel_ratio": round(visual_only_panel_ratio, 4),
            "excessive_visual_only": excessive_visual_only,
            "duplicate_lines": duplicate_count,
            "first_person_lines": first_person_count,
            "generic_lines": generic_count,
            "filler_meta_lines": filler_meta_count,
            "visual_lines": visual_count,
            "raw_ocr_echo_lines": 0,
            "ocr_contamination_lines": ocr_contamination_count,
            "malformed_lines": malformed_count,
            "fact_mismatch_lines": 0,
            "repetitive_template_lines": repetitive_template_count,
            "semantic_near_duplicate_lines": semantic_duplicate_count,
            "disconnected_pairs": disconnected_count,
            "scene_order_regressions": scene_order_regression_count,
            "panel_order_regressions": panel_order_regression_count,
            "panel_coverage": panel_coverage_report,
            "scene_usage": scene_usage_report,
            "meaningful_usage_score": meaningful_usage_score,
            "timing_alignment_score": timing_alignment_score,
            "meaningful_panel_usage_rate": scene_usage_report.get("meaningful_panel_usage_rate", 1.0),
            "meaningfully_used_panel_count": scene_usage_report.get("meaningfully_used_panel_count", 0),
            "unused_meaningful_panel_count": unused_meaningful_panel_count,
            "overcompressed_scene_count": overcompressed_scene_count,
            "suspicious_grouping_count": suspicious_grouping_count,
            "action_scene_without_concrete_action_count": action_without_concrete_count,
            "abstract_or_vague_scene_count": vague_scene_count,
            "long_unintentional_gap_count": long_gap_count,
            "story_continuity_score": story_continuity_score,
            "duplicated_panel_count": duplicated_panel_count,
            "duplicated_panel_ranges": panel_coverage_report.get("duplicated_panel_ranges", []),
            "underexplained_panel_ranges": underexplained_panel_ranges,
            "underexplained_panel_range_count": underexplained_panel_range_count,
            "largest_skipped_panel_gap": largest_skipped_panel_gap,
            "skipped_panel_ranges": panel_coverage_report.get("skipped_panel_ranges", []),
            "late_worldbuilding_context_lines": late_worldbuilding_context_count,
            "avg_sentences_per_line": round(avg_sentences_per_line, 3),
            "avg_sentences_per_spoken_paragraph": round(avg_sentences_per_spoken_paragraph, 3),
            "one_sentence_lines": one_sentence_count,
            "max_one_sentence_run": max_one_sentence_run,
            "short_line_word_threshold": short_line_word_threshold,
            "short_lines_under_threshold": short_line_count,
            "short_lines_under_30_words": short_line_under_30_count,
            "caption_like_lines": caption_like_count,
            "caption_like_sentences": caption_like_sentence_count,
            "weak_transition_lines": weak_transition_count,
            "unclear_pronoun_lines": unclear_pronoun_count,
            "vague_subject_lines": vague_subject_count,
            "speculation_lines": speculation_count,
            "flashback_confusion_lines": flashback_confusion_count,
            "ability_ambiguity_lines": ability_ambiguity_count,
            "invalid_name_lines": invalid_name_count,
            "ocr_garbage_leak_lines": ocr_garbage_leak_count,
            "mentioned_as_present_errors": mentioned_as_present_count,
            "flashback_present_as_current_errors": flashback_present_error_count,
            "unsupported_character_action_errors": unsupported_action_count,
            "median_words_per_line": round(median_words_per_line, 2),
            "p10_words_per_line": round(p10_words_per_line, 2),
            "vision_low_confidence_panels": vision_low_confidence_count,
            "unknown_speaker_panels": unknown_speaker_count,
            "quality_score": quality_score,
            "should_block_tts": should_block_tts,
            "thresholds": thresholds,
            "penalty_weights": {
                "repetitive_template": repetitive_penalty_weight,
            },
            "failure_codes": [
                code
                for code, count in (
                    ("vision_low_confidence", vision_low_confidence_count if vision_quality_affects_tts else 0),
                    ("unknown_speaker", unknown_speaker_count if vision_quality_affects_tts else 0),
                    ("ocr_contamination", ocr_contamination_count),
                    ("malformed_english", malformed_count),
                    ("filler_meta_language", filler_meta_count),
                    ("visual_caption_leakage", visual_count),
                    ("semantic_near_duplicates", semantic_duplicate_count if semantic_duplicate_count > thresholds["semantic_duplicate"] else 0),
                    ("disconnected_transitions", disconnected_count if disconnected_count > thresholds["disconnected"] else 0),
                    ("scene_order_regression", scene_order_regression_count),
                    ("panel_order_regression", panel_order_regression_count),
                    # In panel mode these are structural, not quality failures; suppress from codes.
                    ("one_sentence_segments", 0 if is_panel_mode else (one_sentence_count if one_sentence_count > max(thresholds["one_sentence"], round(total_segments * 0.30)) else 0)),
                    ("one_sentence_run", 0 if is_panel_mode else (max_one_sentence_run if max_one_sentence_run > thresholds["one_sentence_run"] else 0)),
                    ("short_segments", 0 if is_panel_mode else (short_line_count if short_line_count > thresholds["short_line"] else 0)),
                    ("caption_like_segments", caption_like_count if caption_like_count > thresholds["caption_like"] else 0),
                    ("caption_like_sentences", caption_like_sentence_count if caption_like_sentence_count > thresholds["caption_like_sentence"] else 0),
                    ("weak_transitions", weak_transition_count if weak_transition_count > thresholds["weak_transition"] else 0),
                    ("unclear_pronouns", unclear_pronoun_count if unclear_pronoun_count > thresholds["unclear_pronoun"] else 0),
                    ("vague_subjects", vague_subject_count if vague_subject_count > thresholds["vague_subject"] else 0),
                    ("speculative_lines", speculation_count if speculation_count > thresholds["speculation"] else 0),
                    ("flashback_confusion", flashback_confusion_count),
                    ("ability_ambiguity", ability_ambiguity_count if ability_ambiguity_count > thresholds["ability_ambiguity"] else 0),
                    ("invalid_names", invalid_name_count if invalid_name_count > thresholds["invalid_names"] else 0),
                    ("ocr_garbage_leakage", ocr_garbage_leak_count if ocr_garbage_leak_count > thresholds["ocr_garbage"] else 0),
                    ("mentioned_as_present", mentioned_as_present_count if mentioned_as_present_count > thresholds["mentioned_as_present"] else 0),
                    ("unsupported_character_action", unsupported_action_count if unsupported_action_count > thresholds["unsupported_character_action"] else 0),
                    ("flashback_as_current", flashback_present_error_count if flashback_present_error_count > thresholds["flashback_present_error"] else 0),
                    ("insufficient_panel_coverage", 1 if panel_coverage_report.get("insufficient_panel_coverage") else 0),
                    ("large_skipped_panel_gap", 1 if panel_coverage_report.get("has_large_skipped_gap") else 0),
                    ("duplicated_panel_ranges", duplicated_panel_count if duplicated_panel_count > thresholds["duplicated_panels"] else 0),
                    ("underexplained_panel_ranges", underexplained_panel_range_count if underexplained_panel_range_count > thresholds["underexplained_panel_ranges"] else 0),
                    ("late_worldbuilding_context", late_worldbuilding_context_count if late_worldbuilding_context_count > thresholds["late_worldbuilding_context"] else 0),
                    ("insufficient_meaningful_panel_usage", 1 if (
                        float(scene_usage_report.get("meaningful_panel_usage_rate", 1.0) or 0.0) < (0.65 if is_panel_mode else 0.90)
                    ) else 0),
                    ("unused_meaningful_panels", unused_meaningful_panel_count if unused_meaningful_panel_count > thresholds["unused_meaningful_panels"] else 0),
                    ("overcompressed_scenes", overcompressed_scene_count if overcompressed_scene_count > thresholds["overcompressed_scenes"] else 0),
                    ("suspicious_panel_grouping", suspicious_grouping_count if suspicious_grouping_count > thresholds["suspicious_panel_grouping"] else 0),
                    ("action_without_concrete_action", action_without_concrete_count if action_without_concrete_count > thresholds["action_without_concrete_action"] else 0),
                    ("abstract_or_vague_scenes", vague_scene_count if vague_scene_count > thresholds["abstract_or_vague_scenes"] else 0),
                    ("long_unintentional_gaps", long_gap_count if long_gap_count > thresholds["long_unintentional_gaps"] else 0),
                )
                if count > 0
            ],
            "character_role_errors": role_errors[:50],
            "risky_segments": risky_segments[:50],
            "summary": self._story_segment_summary(
                total_segments=total_segments,
                quality_score=quality_score,
                blank_count=blocking_blank_count,
                visual_only_blank_count=visual_only_blank_count,
                visual_only_panel_refs=visual_only_panel_refs,
                total_panel_refs=total_panel_refs,
                duplicate_count=duplicate_count,
                generic_count=generic_count,
                filler_meta_count=filler_meta_count,
                repetitive_template_count=repetitive_template_count,
                malformed_count=malformed_count,
                semantic_duplicate_count=semantic_duplicate_count,
                scene_order_regression_count=scene_order_regression_count,
                panel_order_regression_count=panel_order_regression_count,
                disconnected_count=disconnected_count,
                caption_like_count=caption_like_count,
                caption_like_sentence_count=caption_like_sentence_count,
                weak_transition_count=weak_transition_count,
                unclear_pronoun_count=unclear_pronoun_count,
                vague_subject_count=vague_subject_count,
                speculation_count=speculation_count,
                flashback_confusion_count=flashback_confusion_count,
                ability_ambiguity_count=ability_ambiguity_count,
                max_one_sentence_run=max_one_sentence_run,
                panel_coverage_report=panel_coverage_report,
                story_continuity_score=story_continuity_score,
                underexplained_panel_range_count=underexplained_panel_range_count,
                late_worldbuilding_context_count=late_worldbuilding_context_count,
                should_block_tts=should_block_tts,
            ),
        }

    def analyze(self, panels: list[PanelBox], script_lines: list[str]) -> dict[str, Any]:
        kept_panels = [panel for panel in sorted(panels, key=lambda item: item.order) if panel.keep]
        slot_count = max(len(kept_panels), len(script_lines))
        aligned_lines = list(script_lines) + [""] * max(slot_count - len(script_lines), 0)

        duplicate_count = self._duplicate_count(aligned_lines)
        non_empty_count = 0
        blank_count = 0
        blocking_blank_count = 0
        visual_only_blank_count = 0
        first_person_count = 0
        generic_count = 0
        visual_count = 0
        raw_echo_count = 0
        mismatch_count = 0
        risky_panels: list[dict[str, Any]] = []

        for index, panel in enumerate(kept_panels):
            narration = (aligned_lines[index] if index < len(aligned_lines) else "").strip()
            extracted_text = (panel.ocr_text or "").strip()
            reasons: list[str] = []
            if narration:
                non_empty_count += 1
            else:
                blank_count += 1
                if self._is_visual_only_blank(panel, extracted_text):
                    visual_only_blank_count += 1
                else:
                    blocking_blank_count += 1
                    reasons.append("blank")

            if narration and self.cleaner.is_first_person_narration(narration):
                first_person_count += 1
                reasons.append("first_person")
            if narration and self._looks_generic(narration):
                generic_count += 1
                reasons.append("generic")
            if narration and self._looks_visual(narration):
                visual_count += 1
                reasons.append("visual")
            if narration and extracted_text and self.cleaner._looks_like_raw_ocr_echo(narration, extracted_text):
                raw_echo_count += 1
                reasons.append("raw_echo")
            if narration and extracted_text and self._drops_fact_anchors(narration, extracted_text):
                mismatch_count += 1
                reasons.append("fact_mismatch")

            if reasons:
                risky_panels.append(
                    {
                        "panel_id": panel.id,
                        "panel": panel.panel,
                        "page": panel.page,
                        "reasons": reasons,
                        "narration": narration,
                        "extracted_text": extracted_text[:500],
                    }
                )

        repetitive_template_count = self._repetitive_template_count(aligned_lines)
        disconnected_count = self._disconnected_pair_count(aligned_lines)

        total_lines = len(kept_panels)
        quality_score = 100
        if total_lines:
            quality_score -= round((blocking_blank_count / total_lines) * 40)
            quality_score -= round((duplicate_count / total_lines) * 35)
            quality_score -= round((raw_echo_count / total_lines) * 35)
            quality_score -= round((generic_count / total_lines) * 30)
            quality_score -= round((visual_count / total_lines) * 25)
            quality_score -= round((mismatch_count / total_lines) * 28)
            quality_score -= round((repetitive_template_count / total_lines) * 25)
            quality_score -= round((disconnected_count / total_lines) * 20)
        if first_person_count:
            quality_score -= min(first_person_count * 8, 24)
        quality_score = max(0, min(100, quality_score))

        thresholds = self._thresholds(total_lines)
        should_block_tts = any(
            (
                blocking_blank_count > thresholds["blank"],
                duplicate_count > thresholds["duplicate"],
                generic_count > thresholds["generic"],
                raw_echo_count > thresholds["raw_echo"],
                visual_count > thresholds["visual"],
                first_person_count > 0,
                mismatch_count > thresholds["fact_mismatch"],
                quality_score < thresholds["score"],
            )
        )

        return {
            "analysis_version": 3,
            "total_panels": total_lines,
            "total_script_lines": len(script_lines),
            "non_empty_lines": non_empty_count,
            "blank_lines": blank_count,
            "blocking_blank_lines": blocking_blank_count,
            "visual_only_blank_lines": visual_only_blank_count,
            "duplicate_lines": duplicate_count,
            "first_person_lines": first_person_count,
            "generic_lines": generic_count,
            "visual_lines": visual_count,
            "raw_ocr_echo_lines": raw_echo_count,
            "fact_mismatch_lines": mismatch_count,
            "repetitive_template_lines": repetitive_template_count,
            "disconnected_pairs": disconnected_count,
            "quality_score": quality_score,
            "should_block_tts": should_block_tts,
            "thresholds": thresholds,
            "risky_panels": risky_panels[:50],
            "summary": self._summary(
                total_lines=total_lines,
                quality_score=quality_score,
                blank_count=blank_count,
                blocking_blank_count=blocking_blank_count,
                visual_only_blank_count=visual_only_blank_count,
                duplicate_count=duplicate_count,
                generic_count=generic_count,
                raw_echo_count=raw_echo_count,
                mismatch_count=mismatch_count,
                repetitive_template_count=repetitive_template_count,
                disconnected_count=disconnected_count,
                should_block_tts=should_block_tts,
            ),
        }

    def _is_visual_only_blank(self, panel: PanelBox, extracted_text: str) -> bool:
        return (
            not extracted_text
            and not bool(panel.text_detected)
            and not bool(panel.manual_keep)
            and not bool(panel.manual_ocr_text)
        )

    def _duplicate_count(self, script_lines: list[str]) -> int:
        normalized = [self._normalize_line(line) for line in script_lines if self._normalize_line(line)]
        counts = Counter(normalized)
        return sum(max(count - 1, 0) for count in counts.values())

    def _normalize_line(self, text: str) -> str:
        return re.sub(r"\s+", " ", str(text or "").strip().casefold())

    def _looks_generic(self, narration: str) -> bool:
        lowered = self._normalize_line(narration)
        generic_phrases = (
            "the scene",
            "the moment",
            "the situation",
            "the conflict",
            "the pressure",
            "the stakes",
            "the story",
            "another tense beat",
            "the mood",
            "the chapter",
            "the narration",
            "the next development",
            "keeps evolving",
            "becomes impossible to ignore",
            "a sharp question cuts through",
            "one pointed question makes it clear",
            "the panel holds for a beat",
            "the moment catches on",
            "tension builds",
            "the pressure keeps rising",
            "the world still feels normal",
            "questions start piling up",
            "another crucial moment",
            "the consequences grow harder to ignore",
            "the situation grows harder to contain",
            "before the scene can settle",
            "as everyone absorbs what just happened",
            "a sudden question leaves the moment hanging",
            "the unanswered question freezes the scene",
            "nearby choice",
            "next choice",
            "last choice",
            "the beat keeps",
            "the beat shifts",
            "the beat moves",
            "surrounding group reacts",
            "matter of survival",
            "the dynamic shifts",
            "fewer options",
            "few options",
            "menacing posture signals",
            "imminent conflict",
            "precarious position",
            "the overall apathy",
            "creating a dull atmosphere",
            "promises further complications",
            "persistent challenges",
            "the beat ends",
            "the immediate result",
            "following action",
            "the exchange gives",
            "the follow-up question",
            "another nearby line",
            "the scene grounds",
            "the dialogue makes",
            "the reply adds",
            "practical detail",
            "clear consequence",
            "the visuals keep the focus",
            "moves through this moment",
            "connected to the next exchange",
        )
        return any(phrase in lowered for phrase in generic_phrases)

    def _has_filler_meta_language(self, narration: str) -> bool:
        lowered = self._normalize_line(narration)
        filler_patterns = (
            r"\bthe danger of (?:their|his|her|the) situation was palpable\b",
            r"\bthis realization underscores\b",
            r"\bgravity of the circumstances\b",
            r"\bemotional turmoil\b",
            r"\b(?:highlighting|underscoring|emphasizing|signaling|indicating|suggesting)\b",
            r"\b(?:highlights|underscores|emphasizes|signals|indicates|suggests|signifies)\b",
            r"\braises questions about\b",
            r"\bcreates? (?:a|an) (?:sense|atmosphere) of\b",
            r"\bforeshadows? (?:a|an|the)\b",
            r"\btestament to (?:his|her|their|the)\b",
            r"\bdetermination evident\b",
            r"\bpalpable\b",
            r"\b(?:express concern over|express their astonishment|leaving onlookers to wonder)\b",
            r"\b(?:complex|true|full) nature of\b",
            r"\badds another layer of uncertainty\b",
            r"\b(?:post-storm era|useful recruits)\b",
            r"\bthe stakes (?:rise|grow|become)\b",
            r"\b(?:the|this) moment (?:signals|marks|represents) (?:a|the)\b",
            # "perhaps" is a hedge - flag it as filler. But "seems", "appears", and "clearly"
            # appear legitimately in direct narration ("She seems exhausted"), so omit them.
            r"\bperhaps\b",
            # Story-pacing filler injected by LLMs that meta-narrates instead of narrates
            r"\bthe beat (?:keeps?|kept) moving\b",
            r"\bnearby choice active\b",
            r"\bkeep(?:ing)? the nearby choice\b",
            r"\bthe (?:next|last) choice (?:still has|becomes?|became|is) (?:consequences|harder|active)\b",
            r"\bchoice still has consequences\b",
            r"\bmachinery (?:of war )?turns personal trust\b",
            r"\broom (?:moves?|shifts?|moved|shifted) from (?:mere )?explanation\b",
            r"\broom from (?:mere )?explanation\b",
            r"\bbecomes? the next instruction\b",
            r"\bwhat had been a (?:pause|moment) (?:becomes?|became) the next instruction\b",
            r"\bthe threat stops? being theoretical\b",
            r"\bfight stops? being distant\b",
            r"\btreat the (?:next|last) choice as\b",
            r"\bless room to pretend the danger\b",
            r"\bleaves? (?:the group|them|everyone) with little room to (?:ignore|pretend)\b",
            r"\bno longer any room for hesitation\b",
            r"\bno room for hesitation\b",
            r"\bno longer (?:a|any) (?:room|place) for\b",
            r"\bthe pretense of safety has vanished\b",
            r"\bsurvival (?:now )?depends on more than (?:just )?strength\b",
        )
        return any(re.search(pattern, lowered) for pattern in filler_patterns)

    def _looks_ocr_contaminated(self, narration: str) -> bool:
        raw = str(narration or "")
        lowered = self._normalize_line(narration)
        patterns = (
            r"\basi\s+can\b",
            r"\btion\s+ment\b",
            r"\bhes\s+is\b",
            r"\bis\s+is\b",
            r"\bcan\s+name\s+is\b",
            r"\b(?:gwirrr|garrr|codc|cyyc|cynmcw|aano|hiko|uenn|azlp)\b",
            r"\btrans\s+ported\b",
            r"\bwhat\s+[a-z]{4,}\s+are\s+man\b",
        )
        return any(re.search(pattern, lowered) for pattern in patterns) or bool(
            re.search(r"\b[a-z]{1,4}[A-Z][A-Za-z]{2,}\b", raw)
        )

    def _scene_order_regression_count(self, segments: list[StorySegment]) -> int:
        regressions = 0
        previous_scene_id: int | None = None
        for segment in segments:
            try:
                scene_id = int(segment.scene_id or 0)
            except Exception:
                continue
            if scene_id <= 0:
                continue
            if previous_scene_id is not None and scene_id < previous_scene_id:
                regressions += 1
            previous_scene_id = scene_id
        return regressions

    def _panel_order_regression_count(self, segments: list[StorySegment]) -> int:
        regressions = 0
        previous_panel: int | None = None
        for segment in segments:
            panel_start = int(segment.panel_start or 0)
            if panel_start <= 0:
                continue
            if previous_panel is not None and panel_start < previous_panel:
                regressions += 1
            previous_panel = panel_start
        return regressions

    def _segment_has_panel_order_regression(self, segment: StorySegment, segments: list[StorySegment]) -> bool:
        previous_panel: int | None = None
        for item in segments:
            panel_start = int(item.panel_start or 0)
            if panel_start > 0 and previous_panel is not None and panel_start < previous_panel:
                if item.order == segment.order:
                    return True
            if panel_start > 0:
                previous_panel = panel_start
        return False

    def _panel_coverage_report(self, segments: list[StorySegment], panels: list[PanelBox]) -> dict[str, Any]:
        kept_panels = [
            panel
            for panel in sorted(panels, key=lambda item: item.order)
            if bool(getattr(panel, "keep", True)) and not bool(getattr(panel, "auto_skipped", False))
        ]
        if not kept_panels:
            return {
                "has_panel_source": False,
                "total_input_panels": 0,
                "meaningful_input_panels": 0,
                "panels_used_in_narration": 0,
                "coverage_ratio": 1.0,
                "skipped_panel_ranges": [],
                "largest_skipped_panel_gap": 0,
                "large_gap_threshold": 0,
                "has_large_skipped_gap": False,
                "duplicated_panel_count": 0,
                "duplicated_panel_ranges": [],
                "duplicated_panel_refs": [],
                "out_of_order_panel_references": self._panel_order_regression_count(segments),
                "segment_panel_ranges": [],
                "story_continuity_score": 100,
                "insufficient_panel_coverage": False,
            }

        order_set = {int(panel.order) for panel in kept_panels}
        panel_by_id = {str(panel.id): panel for panel in kept_panels}
        covered_counts: Counter[int] = Counter()
        segment_ranges: list[dict[str, Any]] = []
        out_of_order_count = 0
        previous_start: int | None = None

        for segment in sorted(segments, key=lambda item: item.order):
            if not bool(getattr(segment, "keep", True)):
                continue
            text = str(getattr(segment, "text", "") or "").strip()
            if not text and not bool(getattr(segment, "visual_only", False)):
                continue
            segment_orders: set[int] = set()
            for panel_id in getattr(segment, "panel_ids", []) or []:
                panel = panel_by_id.get(str(panel_id))
                if panel is not None:
                    segment_orders.add(int(panel.order))
            try:
                start = int(getattr(segment, "panel_start", 0) or 0)
                end = int(getattr(segment, "panel_end", 0) or 0)
            except Exception:
                start = 0
                end = 0
            if start > 0 and end > 0:
                low, high = sorted((start, end))
                segment_orders.update(order for order in order_set if low <= order <= high)
            if not segment_orders:
                continue
            range_start = min(segment_orders)
            range_end = max(segment_orders)
            if previous_start is not None and range_start < previous_start:
                out_of_order_count += 1
            previous_start = range_start
            for order in segment_orders:
                covered_counts[order] += 1
            segment_ranges.append(
                {
                    "segment_id": segment.id,
                    "order": segment.order,
                    "panel_start": range_start,
                    "panel_end": range_end,
                    "panel_count": len(segment_orders),
                }
            )

        covered_orders = {order for order, count in covered_counts.items() if count > 0}
        skipped_orders = sorted(order_set - covered_orders)
        duplicated_orders = sorted(order for order, count in covered_counts.items() if count > 1)
        skipped_ranges = self._compress_ranges(skipped_orders)
        duplicated_ranges = self._compress_ranges(duplicated_orders)
        meaningful_count = len(order_set)
        coverage_ratio = len(covered_orders) / max(meaningful_count, 1)
        largest_gap = max((item["count"] for item in skipped_ranges), default=0)
        large_gap_threshold = max(6, round(meaningful_count * 0.04))
        duplicate_ratio = len(duplicated_orders) / max(meaningful_count, 1)
        continuity_score = 100
        continuity_score -= round((1.0 - coverage_ratio) * 100)
        continuity_score -= min(out_of_order_count * 18, 36)
        continuity_score -= round(min(duplicate_ratio, 0.3) * 80)
        if largest_gap >= large_gap_threshold:
            continuity_score -= min(24, round((largest_gap / max(meaningful_count, 1)) * 100))
        continuity_score = max(0, min(100, continuity_score))

        return {
            "has_panel_source": True,
            "total_input_panels": len(kept_panels),
            "meaningful_input_panels": meaningful_count,
            "panels_used_in_narration": len(covered_orders),
            "coverage_ratio": round(coverage_ratio, 4),
            "coverage_percent": int(round(coverage_ratio * 100)),
            "skipped_panel_count": len(skipped_orders),
            "skipped_panel_ranges": skipped_ranges,
            "largest_skipped_panel_gap": largest_gap,
            "large_gap_threshold": large_gap_threshold,
            "has_large_skipped_gap": largest_gap >= large_gap_threshold,
            "duplicated_panel_count": len(duplicated_orders),
            "duplicated_panel_ranges": duplicated_ranges,
            "duplicated_panel_refs": duplicated_orders[:200],
            "out_of_order_panel_references": out_of_order_count,
            "segment_panel_ranges": segment_ranges,
            "story_continuity_score": continuity_score,
            "insufficient_panel_coverage": coverage_ratio < 0.90,
        }

    def _scene_usage_report(
        self,
        segments: list[StorySegment],
        panels: list[PanelBox],
        *,
        panel_evidence_records: list[dict[str, Any]],
        panel_vision_records: list[dict[str, Any]],
        is_panel_mode: bool = False,
    ) -> dict[str, Any]:
        kept_panels = [
            panel
            for panel in sorted(panels, key=lambda item: int(getattr(item, "order", 0) or 0))
            if bool(getattr(panel, "keep", True)) and not bool(getattr(panel, "auto_skipped", False))
        ]
        if not kept_panels:
            return {
                "has_panel_source": False,
                "meaningful_usage_score": 100,
                "timing_alignment_score": 100,
                "meaningful_panel_usage_rate": 1.0,
                "meaningfully_used_panel_count": 0,
                "unused_meaningful_panel_count": 0,
                "scenes": [],
                "scenes_by_segment_id": {},
            }

        panel_by_id = {str(panel.id): panel for panel in kept_panels}
        panel_by_order = {int(panel.order): panel for panel in kept_panels}
        evidence_by_id: dict[str, dict[str, Any]] = {}
        evidence_by_order: dict[int, dict[str, Any]] = {}
        for item in [*panel_evidence_records, *panel_vision_records]:
            if not isinstance(item, dict):
                continue
            panel_id = str(item.get("panel_id") or "").strip()
            if panel_id:
                evidence_by_id[panel_id] = self._merge_panel_evidence(evidence_by_id.get(panel_id, {}), item)
            try:
                order = int(item.get("panel_order") or item.get("order") or 0)
            except Exception:
                order = 0
            if order:
                evidence_by_order[order] = self._merge_panel_evidence(evidence_by_order.get(order, {}), item)

        scenes: list[dict[str, Any]] = []
        meaningfully_used_orders: set[int] = set()
        unused_meaningful_orders: set[int] = set()
        justified_orders: set[int] = set()
        previous_start: int | None = None
        overcompressed = 0
        suspicious = 0
        action_without_concrete = 0
        vague = 0
        one_sentence_multipanel = 0
        long_gaps = 0
        total_gap = 0.0
        largest_gap = 0.0

        for segment in sorted(segments, key=lambda item: item.order):
            if not bool(getattr(segment, "keep", True)):
                continue
            source_panels = self._source_panels_for_segment(segment, panel_by_id, panel_by_order)
            if not source_panels:
                continue
            source_panel_ids = [panel.id for panel in source_panels]
            source_orders = [int(panel.order) for panel in source_panels]
            text = str(getattr(segment, "text", "") or "").strip()
            text_tokens = self._content_token_set(text)
            # In panel mode any blank segment is an intentional silent panel;
            # count its evidence as low_information rather than unused_error.
            segment_is_visual_only_blank = not text and (
                bool(getattr(segment, "visual_only", False)) or is_panel_mode
            )
            contributions: dict[str, dict[str, Any]] = {}
            redundant_ids: list[str] = []
            low_info_ids: list[str] = []
            used_ids: list[str] = []
            unused_ids: list[str] = []
            previous_evidence_tokens: set[str] | None = None

            for panel in source_panels:
                evidence = self._merge_panel_evidence(
                    evidence_by_order.get(int(panel.order), {}),
                    evidence_by_id.get(panel.id, {}),
                )
                evidence_text = self._panel_evidence_text(panel, evidence)
                evidence_tokens = self._content_token_set(evidence_text)
                visual_text = self._panel_visual_text(panel, evidence)
                generated_visual_summary = ""
                if not evidence_text and not visual_text:
                    generated_visual_summary = self._compact_visual_panel_summary(panel)
                    visual_text = generated_visual_summary
                visual_tokens = self._content_token_set(visual_text)
                candidate_tokens = evidence_tokens | visual_tokens
                contribution = "low_information"
                reason = "no readable dialogue or visual summary"
                is_meaningful_candidate = bool(candidate_tokens)
                if segment_is_visual_only_blank:
                    # Visual-only blank segment: no narration was produced on purpose;
                    # treat as low_information so it doesn't penalise usage rate.
                    contribution = "low_information"
                    reason = "visual-only blank segment - narration intentionally omitted"
                    is_meaningful_candidate = False
                elif generated_visual_summary:
                    contribution = "low_information"
                    reason = "visual-only panel has no cached semantic summary; compact visual placeholder generated"
                    is_meaningful_candidate = False
                elif candidate_tokens:
                    if previous_evidence_tokens:
                        overlap = len(candidate_tokens & previous_evidence_tokens) / max(1, len(candidate_tokens | previous_evidence_tokens))
                        if overlap >= 0.72:
                            contribution = "redundant_near_duplicate"
                            reason = "near-duplicate OCR or visual evidence from adjacent panel"
                    if contribution != "redundant_near_duplicate":
                        overlap = len(candidate_tokens & text_tokens)
                        overlap_ratio = overlap / max(1, min(len(candidate_tokens), 12))
                        # In panel mode narration is a paraphrase - even 1 shared content word
                        # is enough to count as used (vs. 2 for multi-panel segments).
                        # If there are no content tokens at all in the narration but the
                        # segment IS narrated (text not empty), treat as meaningfully used;
                        # the segment exists specifically to narrate this panel.
                        _min_overlap = 1 if is_panel_mode else 2
                        _panel_mode_direct = is_panel_mode and len(source_panels) == 1 and bool(text)
                        if overlap >= _min_overlap or overlap_ratio >= 0.18 or self._evidence_phrase_appears_in_text(evidence_text, text) or _panel_mode_direct:
                            contribution = self._classify_panel_contribution(evidence_text, visual_text, text)
                            reason = "evidence reflected in narration" if not _panel_mode_direct else "panel-mode 1:1 narration slot"
                        elif len(candidate_tokens) <= 2 and (candidate_tokens & text_tokens):
                            contribution = "dialogue_meaning"
                            reason = "short dialogue cue reflected in narration"
                        else:
                            contribution = "unused_error"
                            reason = "readable panel evidence was assigned but not narrated"
                contributions[panel.id] = {
                    "panel_order": int(panel.order),
                    "page": getattr(panel, "page", None),
                    "panel_index": getattr(panel, "panel", None),
                    "contribution": contribution,
                    "reason": reason,
                    "evidence_text": evidence_text[:220],
                    "compact_visual_summary": generated_visual_summary[:220],
                    "visual_summary_source": "generated_compact" if generated_visual_summary else ("cached" if visual_text else ""),
                }
                if contribution in {
                    "concrete_action",
                    "character_reaction",
                    "dialogue_meaning",
                    "emotional_shift",
                    "plot_consequence",
                    "setting_context",
                    "transition",
                    "visual_escalation",
                }:
                    used_ids.append(panel.id)
                    meaningfully_used_orders.add(int(panel.order))
                    justified_orders.add(int(panel.order))
                elif contribution == "redundant_near_duplicate":
                    redundant_ids.append(panel.id)
                    justified_orders.add(int(panel.order))
                elif contribution == "low_information":
                    low_info_ids.append(panel.id)
                    justified_orders.add(int(panel.order))
                elif contribution == "unused_error":
                    unused_ids.append(panel.id)
                    if is_meaningful_candidate:
                        unused_meaningful_orders.add(int(panel.order))
                if candidate_tokens:
                    previous_evidence_tokens = candidate_tokens

            representative_panel_id = str(getattr(segment, "representative_panel_id", "") or "")
            if not representative_panel_id and source_panel_ids:
                representative_panel_id = source_panel_ids[len(source_panel_ids) // 2]
            suspicious_reasons = self._suspicious_grouping_reasons(source_panels, previous_start=previous_start)
            previous_start = min(source_orders)
            sentences = self._sentence_counts([text])[0] if text else 0
            words = len(re.findall(r"\b[\w'-]+\b", text))
            estimated_narration_duration = round(words / 2.6, 2)
            scene_duration = round(sum(float(getattr(panel, "duration_seconds", 0.0) or 1.2) for panel in source_panels), 2)
            duration_gap = round(max(scene_duration - estimated_narration_duration, 0.0), 2)
            repair_action = "intentional_silence"
            needs_expansion = False
            if duration_gap > 2.0:
                needs_expansion = True
                repair_action = "expanded_narration" if unused_ids or len(source_panels) <= 18 else "redistributed_duration"
                long_gaps += 1
            if duration_gap > largest_gap:
                largest_gap = duration_gap
            total_gap += duration_gap
            is_overcompressed = self._scene_is_overcompressed(source_panels, text)
            is_action = self._scene_looks_like_action(text, contributions)
            has_concrete_action = bool(re.search(
                r"\b(?:attack|strik|hit|slam|crash|charge|fire|shoot|launch|grab|pull|block|dodge|run|fight|confront|corner|punish|steps? in|explode|collapse|close in|surround|pilot|sync|connect|enter|vanish|escort|warn|call|rush|brace|kick|punch|throw|leap|lash|clash|swing|push|shove|pin|force|strike|intercept|parry|wound|injur|stab|tear|rip|smash|crush|spin|dash|fling|hurl|deflect|press|overwhelm)\b",
                self._normalize_line(text),
            ))
            action_missing = is_action and not has_concrete_action
            abstract_vague = self._scene_has_abstract_vague_narration(text)
            if is_overcompressed:
                overcompressed += 1
            if suspicious_reasons:
                suspicious += 1
            if action_missing:
                action_without_concrete += 1
            if abstract_vague:
                vague += 1
            if sentences <= 1 and len(source_panels) > 1:
                one_sentence_multipanel += 1
            scene_usage = {
                "segment_id": segment.id,
                "order": segment.order,
                "source_panel_ids": source_panel_ids,
                "source_panel_orders": source_orders,
                "representative_panel_id": representative_panel_id,
                "supporting_panel_ids": [panel_id for panel_id in source_panel_ids if panel_id != representative_panel_id],
                "meaningfully_used_panel_ids": used_ids,
                "redundant_panel_ids": redundant_ids,
                "low_information_panel_ids": low_info_ids,
                "unused_meaningful_panel_ids": unused_ids,
                "panel_contribution_map": contributions,
                "meaningful_panel_usage_rate": round(len(used_ids) / max(len(source_panels) - len(redundant_ids) - len(low_info_ids), 1), 4),
                "is_overcompressed": is_overcompressed,
                "suspicious_grouping_reasons": suspicious_reasons,
                "action_scene_without_concrete_action": action_missing,
                "abstract_or_vague_narration": abstract_vague,
                "one_sentence_multipanel_scene": sentences <= 1 and len(source_panels) > 1,
                "scene_duration_seconds": scene_duration,
                "estimated_narration_duration_seconds": estimated_narration_duration,
                "tts_duration_seconds": None,
                "duration_gap_seconds": duration_gap,
                "needs_narration_expansion": needs_expansion,
                "repair_action": repair_action,
                "text": text[:500],
            }
            scenes.append(scene_usage)

        meaningful_candidates = {
            int(panel.order)
            for panel in kept_panels
            if self._content_token_set(self._panel_evidence_text(panel, self._merge_panel_evidence(evidence_by_order.get(int(panel.order), {}), evidence_by_id.get(panel.id, {}))))
            or self._content_token_set(self._panel_visual_text(panel, self._merge_panel_evidence(evidence_by_order.get(int(panel.order), {}), evidence_by_id.get(panel.id, {}))))
        }
        # In panel mode blank segments are intentionally silent; exclude their panels
        # from the meaningful_candidates denominator so they don't penalise usage_rate.
        if is_panel_mode:
            blank_panel_orders: set[int] = set()
            for scene in scenes:
                if not str(scene.get("text") or "").strip():
                    blank_panel_orders.update(scene.get("source_panel_orders", []) or [])
            meaningful_candidates -= blank_panel_orders
        has_meaningful_evidence = bool(meaningful_candidates)
        usage_denominator = max(len(meaningful_candidates), 1)
        usage_rate = len(meaningfully_used_orders & meaningful_candidates) / usage_denominator if has_meaningful_evidence else 1.0
        justified_rate = len(justified_orders) / max(len(kept_panels), 1)
        meaningful_score = 100
        # In panel mode, the token-overlap measurement is less reliable (1-sentence
        # paraphrases don't echo OCR tokens precisely), so use 65% as the baseline
        # for the score rather than 90%.
        _usage_baseline = 0.65 if is_panel_mode else 0.90
        if has_meaningful_evidence:
            meaningful_score -= round(max(0.0, _usage_baseline - usage_rate) * 100)
        meaningful_score -= min(overcompressed * 7, 28)
        meaningful_score -= min(suspicious * 9, 36)
        meaningful_score -= min(action_without_concrete * 10, 30)
        meaningful_score -= min(vague * 5, 30)
        meaningful_score = max(0, min(100, meaningful_score))
        timing_score = max(0, min(100, 100 - min(long_gaps * 6, 42) - round(min(total_gap / max(len(scenes), 1), 12) * 2)))
        return {
            "has_panel_source": True,
            "scene_count": len(scenes),
            "technically_assigned_panel_count": len(justified_orders | unused_meaningful_orders),
            "meaningfully_used_panel_count": len(meaningfully_used_orders),
            "unused_meaningful_panel_count": len(unused_meaningful_orders) if has_meaningful_evidence else 0,
            "unused_meaningful_panel_ids": sorted(unused_meaningful_orders)[:500],
            "meaningful_panel_usage_rate": round(usage_rate, 4),
            "justified_panel_assignment_rate": round(justified_rate, 4),
            "meaningful_usage_score": meaningful_score,
            "timing_alignment_score": timing_score,
            "insufficient_meaningful_panel_usage": has_meaningful_evidence and usage_rate < (0.65 if is_panel_mode else 0.90),
            "overcompressed_scene_count": overcompressed,
            "suspicious_grouping_count": suspicious,
            "action_scene_without_concrete_action_count": action_without_concrete,
            "abstract_or_vague_scene_count": vague,
            "one_sentence_multipanel_scene_count": one_sentence_multipanel,
            "long_unintentional_gap_count": long_gaps,
            "total_duration_gap_seconds": round(total_gap, 2),
            "largest_duration_gap_seconds": round(largest_gap, 2),
            "scenes": scenes[:200],
            "scenes_by_segment_id": {str(scene["segment_id"]): scene for scene in scenes},
        }

    def _source_panels_for_segment(
        self,
        segment: StorySegment,
        panel_by_id: dict[str, PanelBox],
        panel_by_order: dict[int, PanelBox],
    ) -> list[PanelBox]:
        ordered: dict[int, PanelBox] = {}
        for panel_id in getattr(segment, "panel_ids", []) or []:
            panel = panel_by_id.get(str(panel_id))
            if panel is not None:
                ordered[int(panel.order)] = panel
        try:
            start = int(getattr(segment, "panel_start", 0) or 0)
            end = int(getattr(segment, "panel_end", 0) or 0)
        except Exception:
            start = 0
            end = 0
        if start > 0 and end > 0:
            low, high = sorted((start, end))
            for order, panel in panel_by_order.items():
                if low <= order <= high:
                    ordered[order] = panel
        return [ordered[order] for order in sorted(ordered)]

    def _panel_evidence_text(self, panel: PanelBox, evidence: dict[str, Any]) -> str:
        parts = [
            getattr(panel, "manual_ocr_text", None),
            getattr(panel, "ocr_text", None),
            evidence.get("dialogue_text"),
            evidence.get("repaired_text"),
            evidence.get("text_english"),
            evidence.get("cleaned_text"),
            evidence.get("caption"),
            evidence.get("dialogue"),
        ]
        text = " ".join(str(part).strip() for part in parts if str(part or "").strip())
        if self._looks_ocr_contaminated(text) or self._has_ocr_garbage_leak(text):
            return ""
        return re.sub(r"\s+", " ", text).strip()

    def _merge_panel_evidence(self, existing: dict[str, Any], item: dict[str, Any]) -> dict[str, Any]:
        merged = dict(existing or {})
        text_keys = (
            "dialogue_text",
            "repaired_text",
            "text_english",
            "cleaned_text",
            "text",
            "caption",
            "dialogue",
            "visual_summary",
            "action_beat",
            "emotion",
            "summary",
        )
        for key, value in item.items():
            if key in text_keys and str(value or "").strip():
                current = str(merged.get(key) or "").strip()
                incoming = str(value or "").strip()
                if current:
                    if incoming.casefold() not in current.casefold():
                        merged[key] = f"{current} {incoming}"
                else:
                    merged[key] = incoming
            elif key not in merged or merged.get(key) in (None, ""):
                merged[key] = value
        return merged

    def _panel_visual_text(self, panel: PanelBox, evidence: dict[str, Any]) -> str:
        parts = [
            getattr(panel, "visual_caption", None),
            evidence.get("visual_summary"),
            evidence.get("action_beat"),
            evidence.get("emotion"),
            evidence.get("summary"),
        ]
        return re.sub(r"\s+", " ", " ".join(str(part).strip() for part in parts if str(part or "").strip())).strip()

    def _compact_visual_panel_summary(self, panel: PanelBox) -> str:
        page = int(getattr(panel, "page", 0) or 0)
        order = int(getattr(panel, "order", 0) or 0)
        flags = [str(flag) for flag in getattr(panel, "review_flags", []) or [] if str(flag).strip()]
        flag_text = f"; flags: {', '.join(flags[:3])}" if flags else ""
        return f"Visual-only support panel at story order {order} on page {page}{flag_text}."

    def _evidence_phrase_appears_in_text(self, evidence_text: str, narration: str) -> bool:
        evidence_tokens = [token for token in re.findall(r"[A-Za-z][A-Za-z']+", evidence_text.casefold()) if len(token) > 3]
        narration_norm = self._normalize_line(narration)
        if not evidence_tokens:
            return False
        windows = [" ".join(evidence_tokens[index:index + 3]) for index in range(0, max(len(evidence_tokens) - 2, 1))]
        return any(window and window in narration_norm for window in windows)

    def _classify_panel_contribution(self, evidence_text: str, visual_text: str, narration: str) -> str:
        text = self._normalize_line(" ".join([evidence_text, visual_text, narration]))
        if re.search(r"\b(?:says|asks|tells|calls|warns|answers|replies|mentions|explains|thinks|dialogue|question)\b", text):
            return "dialogue_meaning"
        if re.search(r"\b(?:shock|afraid|angry|sad|panic|worry|smile|cry|lonely|hesitat|stunned|confused)\b", text):
            return "character_reaction"
        if re.search(r"\b(?:attack|hit|strike|fire|launch|crash|explode|fight|battle|charge|surround|enemy)\b", text):
            return "concrete_action"
        if re.search(r"\b(?:choice|decision|consequence|changes|turns|forces|leaves|reveals|realizes)\b", text):
            return "plot_consequence"
        if re.search(r"\b(?:meanwhile|afterward|then|before|later|back|arrives|leaves|vanishes|returns)\b", text):
            return "transition"
        if re.search(r"\b(?:facility|garden|city|classroom|battlefield|forest|ocean|hangar|building|structure|compound|base|outpost|district|zone)\b", text):
            return "setting_context"
        if re.search(r"\b(?:flare|glow|massive|close in|swarm|erupts|collapse|tremor)\b", text):
            return "visual_escalation"
        return "emotional_shift"

    def _suspicious_grouping_reasons(self, panels: list[PanelBox], *, previous_start: int | None) -> list[str]:
        if not panels:
            return []
        reasons: list[str] = []
        orders = [int(panel.order) for panel in panels]
        pages = [int(getattr(panel, "page", 0) or 0) for panel in panels]
        if any(right < left for left, right in zip(orders, orders[1:], strict=False)):
            reasons.append("panel_ids_out_of_order")
        if previous_start is not None and min(orders) < previous_start:
            reasons.append("scene_order_regression")
        page_jumps = [abs(right - left) for left, right in zip(pages, pages[1:], strict=False) if left and right]
        if any(jump > 8 for jump in page_jumps):
            reasons.append("distant_page_jump")
        if pages and max(pages) - min(pages) > 35 and len(panels) > 18:
            reasons.append("wide_page_span")
        order_gaps = [right - left for left, right in zip(orders, orders[1:], strict=False)]
        if any(gap > 12 for gap in order_gaps):
            reasons.append("noncontiguous_panel_gap")
        if len(panels) > 36:
            reasons.append("too_many_panels_for_one_beat")
        return reasons

    def _scene_is_overcompressed(self, panels: list[PanelBox], text: str) -> bool:
        if len(panels) <= 8:
            return False
        words = len(re.findall(r"\b[\w'-]+\b", text))
        sentences = self._sentence_counts([text])[0] if text else 0
        words_per_panel = words / max(len(panels), 1)
        return len(panels) > 24 or words_per_panel < 3.0 or (len(panels) > 12 and sentences < 3)

    def _scene_looks_like_action(self, text: str, contributions: dict[str, dict[str, Any]]) -> bool:
        combined = self._normalize_line(" ".join([text, *[str(item.get("evidence_text", "")) for item in contributions.values()]]))
        return bool(re.search(r"\b(?:enemy|mech|battle|attack|fight|weapon|fire|crash|explod|tremor|combat|danger|pilot)\b", combined))

    def _scene_has_abstract_vague_narration(self, text: str) -> bool:
        lowered = self._normalize_line(text)
        vague_patterns = (
            r"\bseveral individuals\b",
            r"\bsubsequent actions\b",
            r"\bthe group is forced to make (?:immediate )?decisions\b",
            r"\banother layer of tension\b",
            r"\bthe situation escalates\b",
            r"\bthe threat grows\b",
            r"\bthings become more serious\b",
            r"\bimmediate danger\b",
            r"\bcombat readiness\b",
            r"\bquick coordination\b",
            r"\bfull attention\b",
            r"\bthe fate of .* rests on their actions\b",
            r"\bsource order\b",
            r"\bchronology\b",
            r"\bhandoff\b",
            r"\bthe paragraph\b",
        )
        # Note: do NOT include _has_filler_meta_language here - filler_meta is already
        # penalized separately in the global score and should not double-count as a vague scene.
        return any(re.search(pattern, lowered) for pattern in vague_patterns)

    def _compress_ranges(self, values: list[int]) -> list[dict[str, int]]:
        if not values:
            return []
        ranges: list[dict[str, int]] = []
        start = previous = values[0]
        for value in values[1:]:
            if value == previous + 1:
                previous = value
                continue
            ranges.append({"start": start, "end": previous, "count": previous - start + 1})
            start = previous = value
        ranges.append({"start": start, "end": previous, "count": previous - start + 1})
        return ranges

    def _segment_has_panel_duplication(self, segment: StorySegment, panel_coverage_report: dict[str, Any]) -> bool:
        duplicated = panel_coverage_report.get("duplicated_panel_refs", [])
        if not isinstance(duplicated, list) or not duplicated:
            return False
        duplicate_orders = {int(value) for value in duplicated if isinstance(value, int)}
        if not duplicate_orders:
            return False
        try:
            start = int(getattr(segment, "panel_start", 0) or 0)
            end = int(getattr(segment, "panel_end", 0) or 0)
        except Exception:
            return False
        if start <= 0 or end <= 0:
            return False
        low, high = sorted((start, end))
        return any(low <= order <= high for order in duplicate_orders)

    def _underexplained_panel_ranges(
        self,
        segments: list[StorySegment],
        panel_coverage_report: dict[str, Any],
    ) -> list[dict[str, Any]]:
        if not panel_coverage_report.get("has_panel_source"):
            return []
        ranges = panel_coverage_report.get("segment_panel_ranges", [])
        if not isinstance(ranges, list):
            return []
        text_by_id = {
            segment.id: str(getattr(segment, "text", "") or "").strip()
            for segment in segments
            if str(getattr(segment, "text", "") or "").strip()
        }
        flagged: list[dict[str, Any]] = []
        for item in ranges:
            if not isinstance(item, dict):
                continue
            panel_count = int(item.get("panel_count", 0) or 0)
            if panel_count < 40:
                continue
            text = text_by_id.get(str(item.get("segment_id", "")), "")
            word_count = len(re.findall(r"\b[\w'-]+\b", text))
            sentences = len([part for part in re.split(r"(?<=[.!?])\s+(?=[A-Z])", text) if part.strip()])
            words_per_panel = word_count / max(panel_count, 1)
            if words_per_panel >= 2.0 and sentences >= 3:
                continue
            flagged.append(
                {
                    "segment_id": item.get("segment_id"),
                    "order": item.get("order"),
                    "panel_start": item.get("panel_start"),
                    "panel_end": item.get("panel_end"),
                    "panel_count": panel_count,
                    "word_count": word_count,
                    "sentence_count": sentences,
                    "words_per_panel": round(words_per_panel, 3),
                }
            )
        return flagged

    def _late_worldbuilding_context_count(self, segments: list[StorySegment]) -> int:
        return sum(1 for segment in segments if self._looks_like_late_worldbuilding(segment, segments))

    def _looks_like_late_worldbuilding(self, segment: StorySegment, segments: list[StorySegment]) -> bool:
        text = str(getattr(segment, "text", "") or "").strip()
        if not text:
            return False
        ordered = [item for item in sorted(segments, key=lambda item: item.order) if str(getattr(item, "text", "") or "").strip()]
        if not ordered:
            return False
        index_by_id = {item.id: index for index, item in enumerate(ordered)}
        index = index_by_id.get(segment.id, 0)
        if index <= max(1, round(len(ordered) * 0.2)):
            return False
        lowered = self._normalize_line(text)
        worldbuilding_openers = (
            "in this world",
            "in a world",
            "in the future",
            "within the confines",
            "humanity",
            "children known as",
            "parasites are",
            "young pilots are",
            "the world is",
            "the setting",
            "society",
            "long ago",
            "years before",
            "the system",
            "the rules of",
        )
        return lowered.startswith(worldbuilding_openers)

    def _semantic_near_duplicate_count(self, script_lines: list[str]) -> int:
        """Count lines that are near-duplicates of any of the 3 preceding non-empty lines.

        Using a window of 3 instead of just adjacent pairs catches cases where an
        identical recap sentence slips in after a bridging one-liner.  The 34%
        Jaccard threshold is kept - below that the overlap is coincidental vocabulary,
        not an actual duplicate.
        """
        duplicate_count = 0
        recent_token_sets: list[set[str]] = []  # up to 3 previous non-empty token sets
        for line in script_lines:
            tokens = self._content_token_set(line)
            if tokens:
                is_dup = any(
                    len(tokens & prev) / max(1, len(tokens | prev)) >= 0.34
                    for prev in recent_token_sets
                )
                if is_dup:
                    duplicate_count += 1
                # Keep at most 3 recent non-empty sets
                recent_token_sets.append(tokens)
                if len(recent_token_sets) > 3:
                    recent_token_sets.pop(0)
        return duplicate_count

    def _max_one_sentence_run(self, lines: list[str]) -> int:
        current_run = 0
        max_run = 0
        for line in lines:
            text = str(line or "").strip()
            if not text:
                current_run = 0
                continue
            sentence_count = len([part for part in re.split(r"(?<=[.!?])\s+(?=[A-Z])", text) if part.strip()])
            if sentence_count <= 1:
                current_run += 1
                max_run = max(max_run, current_run)
            else:
                current_run = 0
        return max_run

    def _looks_caption_like(self, narration: str) -> bool:
        lowered = self._normalize_line(narration)
        word_count = len(re.findall(r"\b[\w'-]+\b", narration))
        sentence_count = len([part for part in re.split(r"(?<=[.!?])\s+(?=[A-Z])", narration.strip()) if part.strip()])
        if word_count > 26 or sentence_count > 2:
            return False
        causal_markers = (
            "because", "so ", "therefore", "forcing", "leaving", "which", "after", "before",
            "while", "when", "until", "instead", "as ", "but ", "so that", "why it matters",
        )
        if any(marker in lowered for marker in causal_markers):
            return False
        if self._looks_visual(narration):
            return False
        return bool(
            sentence_count <= 1
            or re.match(
                r"^(?:[A-Z][A-Za-z0-9'-]+(?:\s+[A-Z][A-Za-z0-9'-]+){0,2}|He|She|They|The\s+[A-Za-z][A-Za-z'-]+)\s+\b(?:moves|turns|heads|walks|runs|charges|arrives|finds|sees|spots|hears|says|asks|tells|admits|warns|explains|looks|stares|watches|raises|hits|strikes|blocks)\b",
                narration,
            )
        )

    def _caption_like_sentence_count(self, narration: str) -> int:
        count = 0
        for sentence in re.split(r"(?<=[.!?])\s+(?=[A-Z])", str(narration or "").strip()):
            sentence = sentence.strip()
            if not sentence:
                continue
            words = re.findall(r"\b[\w'-]+\b", sentence)
            if len(words) > 24:
                continue
            if re.match(
                r"^(?:[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?|The\s+[A-Za-z][A-Za-z'-]+|A\s+[A-Za-z][A-Za-z'-]+)\s+(?:holds|stands|sits|walks|looks|watches|drinks|relaxes|surveys|smiles|types|reaches|poses)\b",
                sentence,
            ):
                count += 1
        return count

    def _has_weak_transition(self, narration: str) -> bool:
        lowered = self._normalize_line(narration)
        weak_openers = (
            "meanwhile", "at the same time", "after that", "from there", "in the next moment",
            "before long", "soon after", "the scene shifts", "elsewhere", "then ",
        )
        return any(lowered.startswith(phrase) for phrase in weak_openers)

    def _has_unclear_pronouns(self, narration: str) -> bool:
        lowered = self._normalize_line(narration)
        # Sentence starts with an ambiguous pronoun with no named antecedent established.
        if re.match(r"^(?:he|she|they|his|her|their)\b", lowered):
            return True
        # Generic gender-inconsistency: within a single sentence the subject pronoun
        # and a reflexive pronoun disagree in gender.  This fires without knowing any
        # character names, purely from grammatical structure.
        #   "he/him/his … herself"   →  male subject + female reflexive
        #   "she/her/hers … himself" →  female subject + male reflexive
        has_male_subject = bool(re.search(r"\b(?:he|him|his)\b", lowered))
        has_female_subject = bool(re.search(r"\b(?:she|her|hers)\b", lowered))
        # Only flag when both a directional subject pronoun and the wrong-gender
        # reflexive appear inside the SAME sentence (split on sentence boundaries).
        for sentence in re.split(r"(?<=[.!?])\s+(?=[A-Z])", narration):
            s = self._normalize_line(sentence)
            if not s:
                continue
            s_male = bool(re.search(r"\b(?:he|him|his)\b", s))
            s_female = bool(re.search(r"\b(?:she|her|hers)\b", s))
            s_self_m = bool(re.search(r"\bhimself\b", s))
            s_self_f = bool(re.search(r"\bherself\b", s))
            if s_male and s_self_f and not s_female:
                # "he/him/his … herself" with no female pronoun to explain herself
                return True
            if s_female and s_self_m and not s_male:
                # "she/her … himself" with no male pronoun to explain himself
                return True
        return False

    def _has_vague_subject(self, narration: str) -> bool:
        lowered = self._normalize_line(narration)
        vague_patterns = (
            r"\bsomeone\b",
            r"\bsomebody\b",
            r"\ba figure\b",
            r"\bthe figure\b",
            r"\banother character\b",
            r"\ba character\b",
            r"\banother person\b",
            r"\ba person\b",
            r"\bthe person\b",
            r"\bthe young man\b",
            r"\bthe young woman\b",
        )
        return any(re.search(pattern, lowered) for pattern in vague_patterns)

    def _sounds_speculative(self, narration: str) -> bool:
        lowered = self._normalize_line(narration)
        speculative_markers = (
            "perhaps", "presumably", "seemingly", "apparently", "appears to",
            "seems to", "maybe", "might be",
            # "as if" is idiomatic in narration ("she passes her belongings as if
            # it's over") - do not flag it as speculation.
        )
        return any(marker in lowered for marker in speculative_markers)

    def _has_confusing_flashback_label(self, narration: str) -> bool:
        lowered = self._normalize_line(narration)
        if not re.search(r"\b(?:flashback|dream|memory|regression|rebirth)\b", lowered):
            return False
        if re.search(r"\bdream(?:s|ing)?\s+of\b", lowered) and not re.search(r"\b(?:flashback|memory|remember|remembered|ago|before|back then)\b", lowered):
            return False
        anchors = ("earlier", "ago", "before", "back then", "once", "used to", "returns to the present", "present day")
        return not any(anchor in lowered for anchor in anchors)

    def _has_unexplained_ability_reference(self, narration: str) -> bool:
        lowered = self._normalize_line(narration)
        if not re.search(r"\b(?:ability|abilities|power|powers|magic|spell|technique|skill|aura|energy|mana|curse|gift|transformation)\b", lowered):
            return False
        if re.search(r"\b(?:predatory|commanding|dangerous|uneasy|intense)\s+aura\b", lowered):
            return False
        if re.search(r"\bability\s+to\s+(?:move|pilot|speak|see|hear|run|walk|think|choose|leave|act|fight|help|understand)\b", lowered):
            return False
        effect_markers = (
            "lets", "allows", "stops", "freezes", "hits", "heals", "blocks", "copies",
            "turns", "forces", "gives", "drains", "breaks", "counter", "amplifies",
            "slows", "strengthens", "protects", "reveals", "moves", "pilots", "pilot",
            "synchronizes", "synchronize", "combines", "transforms", "revealed", "reveal",
            "crackling", "combines", "combined",
        )
        return not any(marker in lowered for marker in effect_markers)

    def _content_token_set(self, text: str) -> set[str]:
        stop_words = {
            "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
            "of", "with", "by", "from", "is", "are", "was", "were", "be", "been",
            "being", "have", "has", "had", "do", "does", "did", "will", "would",
            "could", "should", "may", "might", "shall", "can", "not", "no", "so",
            "if", "then", "than", "that", "this", "it", "its", "as", "up", "out",
            "into", "over", "just", "also", "very", "too", "still", "even", "only",
            "about", "after", "before", "between", "each", "every", "all", "both",
            "few", "more", "most", "other", "some", "such", "one", "two", "three",
            "he", "she", "they", "him", "her", "his", "their", "them", "who", "what",
            "where", "when", "while", "because", "around", "near", "through",
            "scene", "moment", "beat", "choice", "pressure", "situation", "dynamic",
            "conflict", "consequence", "consequences", "response", "group",
        }
        return {
            token
            for token in re.findall(r"[a-z']+", self._normalize_line(text))
            if len(token) >= 3 and token not in stop_words
        }

    def _avg_sentences_per_line(self, lines: list[str]) -> float:
        counts = self._sentence_counts(lines)
        return sum(counts) / max(len(counts), 1)

    def _sentence_counts(self, lines: list[str]) -> list[int]:
        return [
            len([part for part in re.split(r"(?<=[.!?])\s+(?=[A-Z])", line.strip()) if part.strip()])
            for line in lines
            if str(line or "").strip()
        ]

    def _paired_story_lines(self, lines: list[str]) -> list[str]:
        """Mirror narration_story.txt's two-segment paragraph composition."""
        paired: list[str] = []
        buffer: list[str] = []
        for line in lines:
            text = str(line or "").strip()
            if not text:
                continue
            buffer.append(text)
            if len(buffer) >= 2:
                paired.append(" ".join(buffer))
                buffer = []
        if buffer:
            paired.append(" ".join(buffer))
        return paired

    def _word_counts(self, lines: list[str]) -> list[int]:
        return [
            len(re.findall(r"\b[\w'-]+\b", str(line or "")))
            for line in lines
            if str(line or "").strip()
        ]

    def _median(self, values: list[int]) -> float:
        if not values:
            return 0.0
        ordered = sorted(values)
        middle = len(ordered) // 2
        if len(ordered) % 2:
            return float(ordered[middle])
        return (ordered[middle - 1] + ordered[middle]) / 2

    def _percentile(self, values: list[int], percentile: float) -> float:
        if not values:
            return 0.0
        ordered = sorted(values)
        if len(ordered) == 1:
            return float(ordered[0])
        index = max(0, min(len(ordered) - 1, round((len(ordered) - 1) * percentile)))
        return float(ordered[index])

    def _looks_visual(self, narration: str) -> bool:
        lowered = self._normalize_line(narration)
        visual_phrases = (
            "a close-up reveals",
            "the camera",
            "camera pans",
            "panel shows",
            "frame shows",
            "speech bubbles",
            "presented in speech bubbles",
            "face shows",
            "expression shows",
            "symbols for",
            "young man with",
            "young woman with",
            "with wide eyes",
            "expression",
            "eyes wide",
            "sweat beading",
            "sweat drops",
            "sweat on",
            "with blood on his face",
            "with blood on her face",
            "stared in shock",
            "shocked expression",
            "startled expression",
            "with a pained expression",
            "with a determined expression",
            "glared with",
            "body coiled",
            "light blue tiled floor",
            "appears distressed",
            "appearing distressed",
            "the injury turns the confrontation",
            "the barrier turns protection into leverage",
            "bloodied and disfigured",
            "is shown",
            "are shown",
            "is displayed",
            "are displayed",
            "is visible",
            "are visible",
            "visible on a wall",
            "stands against",
            "sits on",
            "white background",
            "black background",
            "wooden surface",
            "dimly lit room",
            "bright light",
            "blinding red light",
            "blurry figure",
            "with text below",
            "visible in the darkness",
            "glows in the dark",
            "social media",
            "website",
            "silhouette",
        )
        if any(phrase in lowered for phrase in visual_phrases):
            return True
        return bool(
            re.search(
                r"^(?:a|an|the|two|three|several)\s+(?:young\s+|older\s+|middle-aged\s+)?(?:man|woman|boy|girl|person|people|character|figure|waiter|bystander|group)\b.*\b(?:stands|sits|looks|looking|stares|wearing|appears)\b",
                lowered,
            )
        )

    def _looks_malformed(self, narration: str) -> bool:
        lowered = self._normalize_line(narration)
        malformed_patterns = (
            r"\b[a-z]+\s+blocks\s+are\s+so\s+exhausting\b",
            r"\b(?:he|she|they)\s+(?:blocks|are|is)\s+are\b",
            r"\b(?:he|she|they)\s+(?:retaliates|retaliated).{0,120}\b(?:he|she)\s+(?:retaliates|punches)\b",
            r"\bmathematical equations and explanations are presented\b",
            r"\btwo elevator buttons\b",
            r"\bone for male and one for female\b",
            r"\b(?:the|a)\s+panel\b",
            r"\b(?:the|a)\s+frame\b",
        )
        return any(re.search(pattern, lowered) for pattern in malformed_patterns)

    def _invalid_name_uses(self, narration: str) -> int:
        count = 0
        for match in re.finditer(r"\b(?:[A-Z][a-z0-9]+|[A-Z]{2,})(?:\s+(?:[A-Z][a-z0-9]+|[A-Z]{2,})){0,2}\b", str(narration or "")):
            candidate = match.group(0)
            key = normalize_name_key(candidate)
            if key in {"a", "an", "the", "he", "she", "they", "his", "her", "their", "this", "that", "these", "those", "in", "as"}:
                continue
            if not looks_like_false_character_name(candidate):
                continue
            prefix = str(narration or "")[max(0, match.start() - 48):match.start()].casefold()
            suffix = str(narration or "")[match.end():match.end() + 48].casefold()
            name_context = bool(
                re.search(r"(?:called|named|known as|code name|codename|name is|as)\s+$", prefix)
                or re.match(r"\s+(?:says|asks|warns|tells|admits|watches|stands|attacks|reacts|enters|leaves|pilots|fights)\b", suffix)
            )
            if name_context:
                count += 1
        return count

    def _has_ocr_garbage_leak(self, narration: str) -> bool:
        raw = str(narration or "")
        lowered = self._normalize_line(narration)
        patterns = (
            r"\bbreak it\b",
            r"\b(?:sfx|sound effect)\b",
            r"\b(?:other|unknown|protagonist)\s+(?:says|asks|watches|stands|attacks|reacts)\b",
            r"\b[a-z]{2,}\s+[a-z]{2,}\s+are\s+so\s+exhausting\b",
            r"\b(?:kcdikaini|jaiv|jle|trle|sauri|salur|nati|gwirrr|garrr|codc|cyyc|cynmcw|aano|hiko|uenn|azlp)\b",
            r"\b(?=[A-Za-z0-9]*\d)(?=[A-Za-z0-9]*[A-Za-z])[A-Za-z0-9]{3,}\b",
        )
        return any(re.search(pattern, lowered) for pattern in patterns) or bool(
            re.search(r"\b[a-z]{1,4}[A-Z][A-Za-z]{2,}\b", raw)
        )

    def _character_role_grounding_errors(
        self,
        segments: list[StorySegment],
        panel_vision_records: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        records_by_panel = {
            str(item.get("panel_id") or "").strip(): item
            for item in panel_vision_records
            if isinstance(item, dict) and str(item.get("panel_id") or "").strip()
        }
        errors: list[dict[str, Any]] = []
        for segment in segments:
            text = str(segment.text or "").strip()
            if not text:
                continue
            segment_roles: dict[str, set[str]] = {}
            for panel_id in getattr(segment, "panel_ids", []) or []:
                record = records_by_panel.get(str(panel_id).strip())
                if not record:
                    continue
                for name, roles in (record.get("character_roles") or {}).items():
                    bucket = segment_roles.setdefault(str(name or "").strip(), set())
                    for role in roles or []:
                        bucket.add(str(role or "").strip())
            for name, roles in segment_roles.items():
                if not name or not re.search(rf"\b{re.escape(name)}\b", text, flags=re.IGNORECASE):
                    continue
                physical = self._name_has_physical_action(text, name)
                if "mentioned_absent" in roles and not (roles & {"visible_present", "speaker", "flashback_present", "memory_present", "imagined_present"}) and physical:
                    errors.append(
                        {
                            "segment_id": segment.id,
                            "segment_order": segment.order,
                            "name": name,
                            "reason": "mentioned_absent_acts",
                            "text": text,
                        }
                    )
                    continue
                if roles & {"flashback_present", "memory_present"} and physical and not re.search(r"\b(?:flashback|memory|remembers|remembered|past|back then|years earlier)\b", text, flags=re.IGNORECASE):
                    errors.append(
                        {
                            "segment_id": segment.id,
                            "segment_order": segment.order,
                            "name": name,
                            "reason": "flashback_as_current",
                            "text": text,
                        }
                    )
                    continue
                if roles and not (roles & {"visible_present", "speaker", "flashback_present", "memory_present", "imagined_present"}) and physical:
                    errors.append(
                        {
                            "segment_id": segment.id,
                            "segment_order": segment.order,
                            "name": name,
                            "reason": "action_without_visible_evidence",
                            "text": text,
                        }
                    )
        return errors

    def _name_has_physical_action(self, narration: str, name: str) -> bool:
        action_verbs = (
            "attack", "attacks", "block", "blocks", "charge", "charges", "enter", "enters", "face", "faces",
            "fall", "falls", "fight", "fights", "glare", "glares", "grab", "grabs", "hit", "hits",
            "kick", "kicks", "look", "looks", "move", "moves", "punch", "punches", "react", "reacts",
            "ride", "rides", "run", "runs", "sit", "sits", "stand", "stands", "stare", "stares", "strike", "strikes", "watch", "watches",
        )
        verb_group = "|".join(action_verbs)
        patterns = (
            rf"\b{re.escape(name)}\b[^.!?]{{0,80}}\b(?:{verb_group})\b",
            rf"\b{re.escape(name)}\b[^.!?]{{0,80}}\b(?:in|inside|on|aboard)\s+(?:the|a|an)\s+(?:ship|machine|vehicle|room|hallway|classroom|doorway|cockpit)\b",
        )
        return any(re.search(pattern, narration, flags=re.IGNORECASE) for pattern in patterns)

    def _drops_fact_anchors(self, narration: str, extracted_text: str) -> bool:
        anchors = self.cleaner._fact_anchor_tokens(extracted_text)
        if len(anchors) < 2:
            return False
        lowered = self._normalize_line(narration)
        matched = sum(1 for anchor in anchors if anchor in lowered)
        numeric_anchors = [anchor for anchor in anchors if any(character.isdigit() for character in anchor)]
        if numeric_anchors and not any(anchor in lowered for anchor in numeric_anchors):
            return True
        return matched == 0

    def _thresholds(self, total_lines: int) -> dict[str, int]:
        base = max(total_lines, 1)
        return {
            "blank": max(2, round(base * 0.05)),
            "duplicate": max(3, round(base * 0.08)),
            "generic": max(12, round(base * 0.15)),
            "raw_echo": max(2, round(base * 0.05)),
            "visual": max(2, round(base * 0.05)),
            "fact_mismatch": max(8, round(base * 0.12)),
            "score": 72,
        }

    def _summary(
        self,
        *,
        total_lines: int,
        quality_score: int,
        blank_count: int,
        blocking_blank_count: int,
        visual_only_blank_count: int,
        duplicate_count: int,
        generic_count: int,
        raw_echo_count: int,
        mismatch_count: int,
        repetitive_template_count: int = 0,
        disconnected_count: int = 0,
        should_block_tts: bool,
    ) -> str:
        problems: list[str] = []
        if blocking_blank_count:
            problems.append(f"{blocking_blank_count} blank")
        if visual_only_blank_count:
            problems.append(f"{visual_only_blank_count} visual-only blank")
        elif blank_count:
            problems.append(f"{blank_count} blank")
        if duplicate_count:
            problems.append(f"{duplicate_count} duplicate")
        if generic_count:
            problems.append(f"{generic_count} generic")
        if raw_echo_count:
            problems.append(f"{raw_echo_count} OCR-echo")
        if mismatch_count:
            problems.append(f"{mismatch_count} fact-mismatch")
        if repetitive_template_count:
            problems.append(f"{repetitive_template_count} repetitive-template")
        if disconnected_count:
            problems.append(f"{disconnected_count} disconnected-pairs")
        if not problems:
            problems.append("no major issues")
        status = "blocked before TTS" if should_block_tts else "safe for TTS"
        return f"Script quality score {quality_score}/100 across {total_lines} kept panels; {', '.join(problems)}; {status}."

    def _story_segment_summary(
        self,
        *,
        total_segments: int,
        quality_score: int,
        blank_count: int,
        visual_only_blank_count: int,
        visual_only_panel_refs: int,
        total_panel_refs: int,
        duplicate_count: int,
        generic_count: int,
        filler_meta_count: int,
        repetitive_template_count: int,
        malformed_count: int,
        semantic_duplicate_count: int,
        scene_order_regression_count: int,
        panel_order_regression_count: int,
        disconnected_count: int,
        caption_like_count: int,
        weak_transition_count: int,
        caption_like_sentence_count: int,
        unclear_pronoun_count: int,
        vague_subject_count: int,
        speculation_count: int,
        flashback_confusion_count: int,
        ability_ambiguity_count: int,
        max_one_sentence_run: int,
        panel_coverage_report: dict[str, Any],
        story_continuity_score: int,
        underexplained_panel_range_count: int,
        late_worldbuilding_context_count: int,
        should_block_tts: bool,
    ) -> str:
        problems: list[str] = []
        if blank_count:
            problems.append(f"{blank_count} blank")
        if visual_only_blank_count:
            if total_panel_refs:
                problems.append(f"{visual_only_blank_count} visual-only blank covering {visual_only_panel_refs}/{total_panel_refs} panels")
            else:
                problems.append(f"{visual_only_blank_count} visual-only blank")
        if duplicate_count:
            problems.append(f"{duplicate_count} duplicate")
        if generic_count:
            problems.append(f"{generic_count} generic")
        if filler_meta_count:
            problems.append(f"{filler_meta_count} filler/meta")
        if repetitive_template_count:
            problems.append(f"{repetitive_template_count} repetitive-template")
        if malformed_count:
            problems.append(f"{malformed_count} malformed")
        if semantic_duplicate_count:
            problems.append(f"{semantic_duplicate_count} near-duplicate")
        if scene_order_regression_count:
            problems.append(f"{scene_order_regression_count} scene-order regression")
        if panel_order_regression_count:
            problems.append(f"{panel_order_regression_count} panel-order regression")
        if disconnected_count:
            problems.append(f"{disconnected_count} disconnected-transitions")
        if caption_like_count:
            problems.append(f"{caption_like_count} caption-like")
        if caption_like_sentence_count:
            problems.append(f"{caption_like_sentence_count} caption-like sentence")
        if weak_transition_count:
            problems.append(f"{weak_transition_count} weak-transitions")
        if unclear_pronoun_count:
            problems.append(f"{unclear_pronoun_count} unclear-pronoun")
        if vague_subject_count:
            problems.append(f"{vague_subject_count} vague-subject")
        if speculation_count:
            problems.append(f"{speculation_count} speculative")
        if flashback_confusion_count:
            problems.append(f"{flashback_confusion_count} flashback-confusion")
        if ability_ambiguity_count:
            problems.append(f"{ability_ambiguity_count} ability-ambiguity")
        if panel_coverage_report.get("has_panel_source"):
            coverage_percent = int(panel_coverage_report.get("coverage_percent", 100) or 100)
            skipped_count = int(panel_coverage_report.get("skipped_panel_count", 0) or 0)
            duplicate_count = int(panel_coverage_report.get("duplicated_panel_count", 0) or 0)
            out_of_order_count = int(panel_coverage_report.get("out_of_order_panel_references", 0) or 0)
            if coverage_percent < 100 or skipped_count:
                problems.append(f"{coverage_percent}% panel coverage with {skipped_count} skipped")
            if int(panel_coverage_report.get("largest_skipped_panel_gap", 0) or 0):
                problems.append(f"largest skipped panel gap {panel_coverage_report.get('largest_skipped_panel_gap')}")
            if duplicate_count:
                problems.append(f"{duplicate_count} duplicated panel refs")
            if out_of_order_count:
                problems.append(f"{out_of_order_count} panel-order continuity error")
            if story_continuity_score < 90:
                problems.append(f"continuity score {story_continuity_score}")
        if underexplained_panel_range_count:
            problems.append(f"{underexplained_panel_range_count} underexplained large panel range")
        if late_worldbuilding_context_count:
            problems.append(f"{late_worldbuilding_context_count} late worldbuilding/context")
        if max_one_sentence_run >= 3:
            problems.append(f"max one-sentence run {max_one_sentence_run}")
        if not problems:
            problems.append("no major issues")
        status = "blocked before TTS" if should_block_tts else "safe for TTS"
        return f"Story script quality score {quality_score}/100 across {total_segments} segments; {', '.join(problems)}; {status}."

    def _repetitive_template_count(self, script_lines: list[str]) -> int:
        """Count lines with repeated content-token phrase templates.

        Recap scripts naturally reuse project names and connective grammar
        ("Squad 13 has to", "Hiro and Zero Two"). Treat those as normal
        continuity rather than template repetition. What we want to catch here
        is a repeated content cluster that recurs across a meaningful slice of
        the chapter.
        """
        stop_words = {
            "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
            "of", "with", "by", "from", "is", "are", "was", "were", "be", "been",
            "being", "have", "has", "had", "do", "does", "did", "will", "would",
            "could", "should", "may", "might", "shall", "can", "not", "no", "so",
            "if", "then", "than", "that", "this", "it", "its", "as", "up", "out",
            "into", "over", "just", "also", "very", "too", "still", "even", "only",
            "about", "after", "before", "between", "each", "every", "all", "both",
            "few", "more", "most", "other", "some", "such", "one", "two", "three",
            "he", "she", "they", "him", "her", "his", "their", "them", "who", "what",
            "where", "when", "while", "because", "around", "near", "through",
        }
        ngram_lines: dict[tuple[str, ...], int] = Counter()
        line_ngrams: list[set[tuple[str, ...]]] = []
        for line in script_lines:
            tokens = [
                token
                for token in re.findall(r"[a-z']+", self._normalize_line(line))
                if token not in stop_words and len(token) >= 3
            ]
            grams: set[tuple[str, ...]] = set()
            for n in range(3, min(len(tokens) + 1, 6)):
                for i in range(len(tokens) - n + 1):
                    gram = tuple(tokens[i:i + n])
                    grams.add(gram)
            line_ngrams.append(grams)
            for gram in grams:
                ngram_lines[gram] += 1

        threshold = max(4, round(max(len([line for line in script_lines if str(line).strip()]), 1) * 0.06))
        shared_grams = {gram for gram, count in ngram_lines.items() if count >= threshold}
        if not shared_grams:
            return 0

        flagged = 0
        for grams in line_ngrams:
            if grams & shared_grams:
                flagged += 1
        return flagged

    def _disconnected_pair_count(self, script_lines: list[str]) -> int:
        """Count consecutive line pairs that share zero content words."""
        stop_words = {
            "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
            "of", "with", "by", "from", "is", "are", "was", "were", "be", "been",
            "being", "have", "has", "had", "do", "does", "did", "will", "would",
            "could", "should", "may", "might", "shall", "can", "not", "no", "so",
            "if", "then", "than", "that", "this", "it", "its", "as", "up", "out",
            "into", "over", "just", "also", "very", "too", "still", "even", "only",
            "about", "after", "before", "between", "each", "every", "all", "both",
            "few", "more", "most", "other", "some", "such", "one", "two", "three",
            "he", "she", "they", "him", "her", "his", "their", "them", "who", "what",
        }
        disconnected = 0
        prev_words: set[str] | None = None
        for line in script_lines:
            tokens = set(re.findall(r"[a-z']+", self._normalize_line(line))) - stop_words
            if prev_words is not None and tokens and prev_words:
                if not tokens & prev_words:
                    disconnected += 1
            prev_words = tokens
        return disconnected
