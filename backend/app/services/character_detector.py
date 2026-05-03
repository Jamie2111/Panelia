from __future__ import annotations

from collections import defaultdict
from typing import Any

from app.schemas.project import PanelBox


class CharacterDetector:
    def detect(
        self,
        page_payloads: dict[int, dict[str, Any]],
        panels: list[PanelBox],
    ) -> list[dict[str, Any]]:
        panel_lookup: dict[int, list[PanelBox]] = defaultdict(list)
        for panel in sorted((item for item in panels if item.keep), key=lambda item: item.order):
            panel_lookup[int(panel.page)].append(panel)

        detections: list[dict[str, Any]] = []
        for page_number, payload in sorted(page_payloads.items(), key=lambda item: int(item[0])):
            page_panels = panel_lookup.get(int(page_number), [])
            if not page_panels:
                continue
            for character in payload.get("characters", []) or []:
                character_id = str(character.get("character_id") or "").strip()
                bbox = self._coerce_bbox(character.get("bbox"))
                if not character_id or bbox is None:
                    continue
                matched_panels = self._panels_for_character(bbox, page_panels)
                for panel in matched_panels:
                    detections.append(
                        {
                            "source_character_id": character_id,
                            "page": int(page_number),
                            "panel_id": panel.id,
                            "panel_order": int(panel.order),
                            "panel_number": int(panel.panel),
                            "bbox": list(bbox),
                            "panel_bbox": [int(panel.x), int(panel.y), int(panel.width), int(panel.height)],
                        }
                    )
        return detections

    def _panels_for_character(self, bbox: tuple[int, int, int, int], page_panels: list[PanelBox]) -> list[PanelBox]:
        matches: list[PanelBox] = []
        for panel in page_panels:
            panel_box = (int(panel.x), int(panel.y), int(panel.width), int(panel.height))
            expanded = self._expand_box(panel_box)
            if self._iou(expanded, bbox) >= 0.05 or self._center_inside(bbox, expanded):
                matches.append(panel)
        return sorted(matches, key=lambda item: item.order)

    def _expand_box(self, bbox: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
        x, y, width, height = bbox
        pad_x = max(24, int(width * 0.08))
        pad_y = max(24, int(height * 0.08))
        return (max(0, x - pad_x), max(0, y - pad_y), width + pad_x * 2, height + pad_y * 2)

    def _center_inside(self, inner: tuple[int, int, int, int], outer: tuple[int, int, int, int]) -> bool:
        center_x = inner[0] + inner[2] / 2
        center_y = inner[1] + inner[3] / 2
        return outer[0] <= center_x <= outer[0] + outer[2] and outer[1] <= center_y <= outer[1] + outer[3]

    def _iou(self, left: tuple[int, int, int, int], right: tuple[int, int, int, int]) -> float:
        intersection = self._intersection(left, right)
        if intersection <= 0:
            return 0.0
        left_area = max(left[2], 1) * max(left[3], 1)
        right_area = max(right[2], 1) * max(right[3], 1)
        return intersection / max(left_area + right_area - intersection, 1)

    def _intersection(self, left: tuple[int, int, int, int], right: tuple[int, int, int, int]) -> int:
        left_x1 = max(left[0], right[0])
        left_y1 = max(left[1], right[1])
        right_x2 = min(left[0] + left[2], right[0] + right[2])
        right_y2 = min(left[1] + left[3], right[1] + right[3])
        width = max(0, right_x2 - left_x1)
        height = max(0, right_y2 - left_y1)
        return width * height

    def _coerce_bbox(self, value: Any) -> tuple[int, int, int, int] | None:
        if not isinstance(value, (list, tuple)) or len(value) < 4:
            return None
        try:
            x, y, width, height = [int(round(float(item))) for item in value[:4]]
        except (TypeError, ValueError):
            return None
        if width <= 0 or height <= 0:
            return None
        return x, y, width, height
