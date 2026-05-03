from __future__ import annotations

from typing import Any


class ScriptGenerator:
    def apply_character_memory_to_regions(
        self,
        regions: list[Any],
        memory_payload: dict[str, Any],
    ) -> list[Any]:
        source_to_character_id = memory_payload.get("source_to_character_id", {}) if isinstance(memory_payload, dict) else {}
        characters = memory_payload.get("characters", {}) if isinstance(memory_payload, dict) else {}
        for region in regions:
            source_id = str(getattr(region, "character_id", "") or "").strip()
            stable_id = str(source_to_character_id.get(source_id) or "").strip()
            if not stable_id:
                continue
            memory = characters.get(stable_id, {})
            setattr(region, "stable_character_id", stable_id)
            display_name = str(memory.get("display_name") or stable_id).strip()
            setattr(region, "character_display_name", display_name)
        return regions

    def annotate_scene(
        self,
        scene: Any,
        scene_regions: list[Any],
        memory_payload: dict[str, Any],
    ) -> Any:
        characters = memory_payload.get("characters", {}) if isinstance(memory_payload, dict) else {}
        character_ids: list[str] = []
        character_labels: list[str] = []
        character_names = list(getattr(scene, "character_names", []) or [])

        for region in scene_regions:
            stable_id = str(getattr(region, "stable_character_id", "") or "").strip()
            if not stable_id or stable_id in character_ids:
                continue
            character_ids.append(stable_id)
            memory = characters.get(stable_id, {})
            label = str(memory.get("display_name") or stable_id).strip()
            if label and label not in character_labels:
                character_labels.append(label)
            if label and label not in {"Other", "Protagonist"} and label not in character_names:
                character_names.append(label)

        setattr(scene, "character_ids", character_ids)
        setattr(scene, "character_labels", character_labels)
        setattr(scene, "character_names", character_names)
        return scene

    def display_name_for_region(
        self,
        region: Any,
        memory_payload: dict[str, Any],
    ) -> str:
        stable_id = str(getattr(region, "stable_character_id", "") or "").strip()
        if not stable_id:
            return ""
        characters = memory_payload.get("characters", {}) if isinstance(memory_payload, dict) else {}
        memory = characters.get(stable_id, {})
        return str(memory.get("display_name") or stable_id).strip()
