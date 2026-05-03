from __future__ import annotations

from collections import Counter
from typing import Any

from app.services.character_name_filters import looks_like_false_character_name


class CharacterMemory:
    def build(
        self,
        tracking_payload: dict[str, Any],
        character_clusters: list[dict[str, Any]],
        cluster_name_map: dict[str, str],
        protagonist_name: str | None = None,
    ) -> dict[str, Any]:
        characters = tracking_payload.get("characters", {}) if isinstance(tracking_payload, dict) else {}
        source_to_character_id = tracking_payload.get("source_to_character_id", {}) if isinstance(tracking_payload, dict) else {}
        panel_characters = tracking_payload.get("panel_characters", {}) if isinstance(tracking_payload, dict) else {}

        cluster_lookup = {str(cluster.get("cluster_id") or ""): cluster for cluster in character_clusters}
        memory: dict[str, dict[str, Any]] = {}

        for stable_id, entry in sorted(
            characters.items(),
            key=lambda item: (
                int(item[1].get("first_page") or 0),
                int(item[1].get("first_panel") or 0),
                str(item[0]),
            ),
        ):
            source_ids = [str(value) for value in entry.get("source_character_ids", [])]
            names = [
                str(cluster_name_map.get(source_id) or "").strip()
                for source_id in source_ids
                if str(cluster_name_map.get(source_id) or "").strip()
            ]
            role_hints = Counter(
                str(cluster_lookup.get(source_id, {}).get("role_hint") or "").strip()
                for source_id in source_ids
                if str(cluster_lookup.get(source_id, {}).get("role_hint") or "").strip()
            )
            role_hint = role_hints.most_common(1)[0][0] if role_hints else "Character"
            name = self._resolve_name(names, protagonist_name, role_hint)
            description = self._description_for(role_hint, stable_id, name, protagonist_name)
            memory[stable_id] = {
                "id": stable_id,
                "name": name,
                "description": description,
                "role": role_hint,
                "first_panel": int(entry.get("first_panel") or 0),
                "first_page": int(entry.get("first_page") or 0),
                "appearance_count": int(entry.get("appearance_count") or len(entry.get("appearances", []))),
                "appearances": entry.get("appearances", []),
                "source_character_ids": source_ids,
                "display_name": name or stable_id,
                "narration_reference": name or self._reference_for_unknown(role_hint, stable_id, protagonist_name),
            }

        return {
            "characters": memory,
            "source_to_character_id": source_to_character_id,
            "panel_characters": panel_characters,
        }

    def _resolve_name(self, names: list[str], protagonist_name: str | None, role_hint: str) -> str | None:
        candidates = [
            name
            for name in names
            if name and name not in {"Other", "Stranger", "Character"} and not looks_like_false_character_name(name)
        ]
        if protagonist_name and role_hint == "Protagonist":
            return protagonist_name
        if candidates:
            return candidates[0]
        return protagonist_name if protagonist_name and role_hint == "Protagonist" else None

    def _description_for(
        self,
        role_hint: str,
        stable_id: str,
        name: str | None,
        protagonist_name: str | None,
    ) -> str:
        if name:
            return name
        if role_hint == "Protagonist":
            return protagonist_name or "the protagonist"
        role_map = {
            "Neighbor": "the neighbor",
            "Friend": "the friend",
            "Teacher": "the teacher",
            "Villain": "the villain",
            "Security Guard": "the guard",
            "Restaurant Worker": "the worker",
            "Manager": "the manager",
            "Loan Shark": "the loan shark",
            "Crowd Member": "the crowd member",
            "Stranger": "the other character",
            "Character": "the other character",
        }
        return role_map.get(role_hint, stable_id)

    def _reference_for_unknown(self, role_hint: str, stable_id: str, protagonist_name: str | None) -> str:
        if role_hint == "Protagonist":
            return protagonist_name or "the protagonist"
        if role_hint in {"Neighbor", "Friend", "Teacher", "Villain", "Security Guard", "Restaurant Worker", "Manager", "Loan Shark"}:
            return self._description_for(role_hint, stable_id, None, protagonist_name)
        return "the other character"
