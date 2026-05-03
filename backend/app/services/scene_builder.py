from __future__ import annotations

from dataclasses import asdict, dataclass
from math import ceil
from typing import Any

from app.services.ocr_cleaner import keyword_tokens


@dataclass(slots=True)
class SceneSeed:
    scene_id: int
    panel_start: int
    panel_end: int
    panel_ids: list[str]
    panels: list[int]
    combined_text: str
    character_names: list[str]
    protagonist_name: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class SceneBuilder:
    def __init__(self, min_scene_size: int = 2, max_scene_size: int = 8, target_min: int = 6, target_max: int = 10) -> None:
        self.min_scene_size = min_scene_size
        self.max_scene_size = max_scene_size
        self.target_min = target_min
        self.target_max = target_max

    def build(self, panels: list[dict[str, Any]]) -> list[SceneSeed]:
        if not panels:
            return []

        raw_groups: list[list[dict[str, Any]]] = []
        current: list[dict[str, Any]] = []

        for panel in panels:
            if not current:
                current = [panel]
                continue

            if self._should_merge(current, panel):
                current.append(panel)
                continue

            raw_groups.append(current)
            current = [panel]

        if current:
            raw_groups.append(current)

        raw_groups = self._merge_small_groups(raw_groups)
        raw_groups = self._rebalance_group_count(raw_groups)
        return [self._build_scene_seed(index + 1, group) for index, group in enumerate(raw_groups)]

    def _should_merge(self, current: list[dict[str, Any]], next_panel: dict[str, Any]) -> bool:
        if len(current) < self.min_scene_size:
            return True
        if len(current) >= self.max_scene_size:
            return False

        current_text = " ".join(str(panel.get("text", "")) for panel in current)
        current_tokens = keyword_tokens(current_text)
        next_tokens = keyword_tokens(str(next_panel.get("text", "")))
        overlap = self._token_overlap(current_tokens, next_tokens)
        next_text = str(next_panel.get("text", "")).casefold()
        current_names = {
            str(name).strip()
            for panel in current
            for name in panel.get("character_names", []) or []
            if str(name).strip()
        }
        next_names = {
            str(name).strip()
            for name in next_panel.get("character_names", []) or []
            if str(name).strip()
        }
        transition_hit = any(marker in next_text for marker in ("later", "meanwhile", "suddenly", "the next day", "next day"))
        if transition_hit and len(current) >= self.min_scene_size:
            return False
        if current_names and next_names and not (current_names & next_names) and len(current) >= self.min_scene_size:
            return False
        if overlap >= 0.12:
            return True
        if len(next_tokens) <= 3:
            return True
        return len(current) < ceil(self.max_scene_size / 2)

    def _merge_small_groups(self, groups: list[list[dict[str, Any]]]) -> list[list[dict[str, Any]]]:
        if not groups:
            return []
        merged: list[list[dict[str, Any]]] = []
        for group in groups:
            if not merged:
                merged.append(group)
                continue
            if len(group) < self.min_scene_size:
                if len(merged[-1]) + len(group) <= self.max_scene_size:
                    merged[-1].extend(group)
                else:
                    group[:0] = merged.pop()
                    merged.append(group)
            else:
                merged.append(group)
        return merged

    def _rebalance_group_count(self, groups: list[list[dict[str, Any]]]) -> list[list[dict[str, Any]]]:
        rebalanced = [list(group) for group in groups if group]
        while len(rebalanced) > self.target_max:
            index = self._best_merge_index(rebalanced)
            rebalanced[index].extend(rebalanced.pop(index + 1))

        while len(rebalanced) < self.target_min:
            split_index = self._best_split_index(rebalanced)
            if split_index is None:
                break
            group = rebalanced.pop(split_index)
            midpoint = max(self.min_scene_size, min(len(group) - self.min_scene_size, len(group) // 2))
            left = group[:midpoint]
            right = group[midpoint:]
            if len(left) < self.min_scene_size or len(right) < self.min_scene_size:
                rebalanced.insert(split_index, group)
                break
            rebalanced.insert(split_index, right)
            rebalanced.insert(split_index, left)
        return rebalanced

    def _best_merge_index(self, groups: list[list[dict[str, Any]]]) -> int:
        scores: list[tuple[float, int]] = []
        for index in range(len(groups) - 1):
            left = keyword_tokens(" ".join(str(panel.get("text", "")) for panel in groups[index]))
            right = keyword_tokens(" ".join(str(panel.get("text", "")) for panel in groups[index + 1]))
            overlap = self._token_overlap(left, right)
            size_penalty = len(groups[index]) + len(groups[index + 1])
            scores.append((overlap - (size_penalty / 1000), index))
        return max(scores, key=lambda item: item[0])[1] if scores else 0

    def _best_split_index(self, groups: list[list[dict[str, Any]]]) -> int | None:
        candidates = [
            (len(group), index)
            for index, group in enumerate(groups)
            if len(group) >= self.min_scene_size * 2
        ]
        if not candidates:
            return None
        return max(candidates)[1]

    def _build_scene_seed(self, scene_id: int, group: list[dict[str, Any]]) -> SceneSeed:
        character_names = sorted(
            {
                str(name).strip()
                for panel in group
                for name in panel.get("character_names", []) or []
                if str(name).strip()
            }
        )
        protagonist_name = next(
            (
                str(panel.get("protagonist_name") or "").strip()
                for panel in group
                if str(panel.get("protagonist_name") or "").strip()
            ),
            None,
        )
        return SceneSeed(
            scene_id=scene_id,
            panel_start=int(group[0]["panel"]),
            panel_end=int(group[-1]["panel"]),
            panel_ids=[str(panel["panel_id"]) for panel in group],
            panels=[int(panel["panel"]) for panel in group],
            combined_text=" ".join(
                str(panel.get("gemini_text") or panel.get("text", "")).strip()
                for panel in group
                if str(panel.get("gemini_text") or panel.get("text", "")).strip()
            )[:1400],
            character_names=character_names,
            protagonist_name=protagonist_name,
        )

    def _token_overlap(self, left: set[str], right: set[str]) -> float:
        if not left or not right:
            return 0.0
        shared = left & right
        return len(shared) / max(min(len(left), len(right)), 1)
