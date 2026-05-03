from __future__ import annotations

import re
from collections import Counter
from typing import Any

from app.schemas.project import PanelBox, StorySegment
from app.services.script_cleaner_service import ScriptCleanerService


class ScriptQualityService:
    def __init__(self) -> None:
        self.cleaner = ScriptCleanerService()

    def analyze_story_segments(
        self,
        story_segments: list[StorySegment],
        *,
        panel_vision_records: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        ordered_segments = [
            segment
            for segment in sorted(story_segments, key=lambda item: item.order)
            if bool(getattr(segment, "keep", True))
        ]
        lines = [str(segment.text or "").strip() for segment in ordered_segments]
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
        visual_count = sum(1 for line in lines if line and self._looks_visual(line))
        ocr_contamination_count = sum(1 for line in lines if line and self._looks_ocr_contaminated(line))
        repetitive_template_count = self._repetitive_template_count(lines)
        disconnected_count = self._disconnected_pair_count(lines)
        avg_sentences_per_line = self._avg_sentences_per_line(lines)
        avg_sentences_per_spoken_paragraph = avg_sentences_per_line
        sentence_counts = self._sentence_counts(lines)
        word_counts = self._word_counts(lines)
        short_line_word_threshold = 18
        one_sentence_count = sum(1 for count in sentence_counts if count <= 1)
        short_line_count = sum(1 for count in word_counts if count < short_line_word_threshold)
        short_line_under_30_count = sum(1 for count in word_counts if count < 30)
        median_words_per_line = self._median(word_counts)
        p10_words_per_line = self._percentile(word_counts, 0.10)
        disconnected_penalty_count = disconnected_count
        if avg_sentences_per_line >= 1.75:
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
            if text and self._looks_visual(text):
                reasons.append("visual")
            if text and self._looks_ocr_contaminated(text):
                reasons.append("ocr_contamination")
            if text:
                sentence_count = len([part for part in re.split(r"(?<=[.!?])\s+(?=[A-Z])", text) if part.strip()])
                word_count = len(re.findall(r"\b[\w'-]+\b", text))
                if sentence_count <= 1:
                    reasons.append("one_sentence")
                if word_count < short_line_word_threshold:
                    reasons.append("short_line")
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
            quality_score -= round((visual_count / total_segments) * 22)
            quality_score -= round((ocr_contamination_count / total_segments) * 34)
            quality_score -= round((repetitive_template_count / total_segments) * repetitive_penalty_weight)
            quality_score -= round((disconnected_penalty_count / total_segments) * 18)
            quality_score -= round((visual_only_blank_count / total_segments) * 18)
            quality_score -= round((one_sentence_count / total_segments) * 20)
            quality_score -= round((short_line_count / total_segments) * 16)
        visual_only_panel_ratio = visual_only_panel_refs / max(total_panel_refs, 1)
        if total_panel_refs:
            quality_score -= round(visual_only_panel_ratio * 12)
        vision_quality_affects_tts = bool(panel_vision_records and (blocking_blank_count or visual_only_blank_count))
        if vision_quality_affects_tts:
            quality_score -= round((vision_low_confidence_count / max(len(panel_vision_records), 1)) * 12)
            quality_score -= round((unknown_speaker_count / max(len(panel_vision_records), 1)) * 8)
        if first_person_count:
            quality_score -= min(first_person_count * 8, 24)
        quality_score = max(0, min(100, quality_score))

        thresholds = {
            "blank": max(1, total_segments // 8),
            "duplicate": max(1, total_segments // 10),
            "generic": max(2, total_segments // 7),
            "visual": max(2, total_segments // 9),
            # Story-first narration intentionally leaves some panels silent so
            # the edit can breathe. Treat visual-only blanks as blocking only
            # when they dominate the script; the score still penalizes them.
            "visual_only_segments": max(8, round(max(total_segments, 1) * 0.45)),
            "visual_only_panels": max(12, round(max(total_panel_refs, 1) * 0.50)),
            "one_sentence": max(2, total_segments // 6),
            "short_line": max(2, total_segments // 6),
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
                visual_count > thresholds["visual"],
                ocr_contamination_count > max(1, total_segments // 12),
                excessive_visual_only,
                one_sentence_count > thresholds["one_sentence"],
                short_line_count > thresholds["short_line"],
                first_person_count > 0,
                quality_score < thresholds["score"],
            )
        )

        return {
            "analysis_mode": "story_segments_v1",
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
            "visual_lines": visual_count,
            "raw_ocr_echo_lines": 0,
            "ocr_contamination_lines": ocr_contamination_count,
            "fact_mismatch_lines": 0,
            "repetitive_template_lines": repetitive_template_count,
            "disconnected_pairs": disconnected_count,
            "avg_sentences_per_line": round(avg_sentences_per_line, 3),
            "avg_sentences_per_spoken_paragraph": round(avg_sentences_per_spoken_paragraph, 3),
            "one_sentence_lines": one_sentence_count,
            "short_line_word_threshold": short_line_word_threshold,
            "short_lines_under_threshold": short_line_count,
            "short_lines_under_30_words": short_line_under_30_count,
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
                    ("one_sentence_segments", one_sentence_count if one_sentence_count > thresholds["one_sentence"] else 0),
                    ("short_segments", short_line_count if short_line_count > thresholds["short_line"] else 0),
                )
                if count > 0
            ],
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
                repetitive_template_count=repetitive_template_count,
                disconnected_count=disconnected_count,
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
        )
        return any(phrase in lowered for phrase in generic_phrases)

    def _looks_ocr_contaminated(self, narration: str) -> bool:
        lowered = self._normalize_line(narration)
        patterns = (
            r"\basi\s+can\b",
            r"\btion\s+ment\b",
            r"\bhes\s+is\b",
            r"\bis\s+is\b",
            r"\bcan\s+name\s+is\b",
        )
        return any(re.search(pattern, lowered) for pattern in patterns)

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
            "young man with",
            "young woman with",
            "with wide eyes",
            "with a pained expression",
            "with a determined expression",
            "bloodied and disfigured",
            "is shown",
            "are shown",
            "is displayed",
            "are displayed",
            "stands against",
            "sits on",
            "white background",
            "black background",
            "wooden surface",
            "dimly lit room",
            "bright light",
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
                r"^(?:a|an|the|two|three|several)\s+(?:young\s+|older\s+|middle-aged\s+)?(?:man|woman|boy|girl|person|people|character|figure|waiter|bystander|group)\b.*\b(?:stands|sits|holds|holding|looks|looking|stares|wearing|walks|faces|smiles|grins|clenches|floats)\b",
                lowered,
            )
        )

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
        repetitive_template_count: int,
        disconnected_count: int,
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
        if repetitive_template_count:
            problems.append(f"{repetitive_template_count} repetitive-template")
        if disconnected_count:
            problems.append(f"{disconnected_count} disconnected-transitions")
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
