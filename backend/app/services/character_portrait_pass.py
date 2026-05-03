from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from difflib import SequenceMatcher
import json
import logging
import re
from pathlib import Path
from typing import Any

from app.core.config import get_settings
from app.pipeline.image_loader import ImageLoader
from app.schemas.project import CanonicalCharacterRecord, ChapterMetadata, PanelBox
from app.services.llm_router import LLMRouter
from app.utils.files import ensure_dir, read_json, write_json

logger = logging.getLogger(__name__)

_PROMPT_VERSION = "character_portrait_v2"
_ROLE_PRIORITY = {"protagonist": 0, "antagonist": 1, "supporting": 2, "cameo": 3}
_PLACEHOLDER_NAME_PATTERN = re.compile(
    r"\b(protagonist|unknown|victim|speaker|narrator|man|woman|boy|girl|person|figure|child|manager|delivery man|old woman|elderly woman)\b",
    re.IGNORECASE,
)
_TITLE_PREFIXES = {
    "mr", "mrs", "ms", "miss", "sr", "sra", "senhor", "senhora",
    "dr", "doctor", "captain", "boss", "chefe", "gerente", "old", "elderly",
}
_DESCRIPTION_STOPWORDS = {
    "a", "an", "the", "with", "and", "or", "of", "in", "on", "at", "to", "from",
    "man", "woman", "boy", "girl", "person", "young", "old", "unknown", "often",
    "seen", "wearing", "wears", "wear", "short", "long", "dark", "light", "colored",
    "hair", "eyes", "face", "looks", "looking", "standing", "sitting",
}
_HAIR_COLORS = {"black", "white", "blue", "brown", "blonde", "red", "gray", "grey", "silver", "purple", "pink", "green"}
_GENDER_MARKERS = {"man", "woman", "boy", "girl", "child"}


class CharacterPortraitPass:
    # Lowered from 4 → 2 because Gemini blocks 4-page anime batches with
    # ``promptFeedback.blockReason: OTHER`` (a prompt-level non-category block
    # that bypasses safety settings). Single-page requests on the same
    # content succeed; halving the batch sharply reduces OTHER blocks.
    BATCH_SIZE = 2
    # Run independent batches concurrently. Each Gemini call already does its
    # own model fallback + retry; parallelism just hides the per-batch latency
    # behind other batches. Keep modest so we don't trigger rate limits.
    PARALLEL_WORKERS = 4

    def __init__(self, router: LLMRouter | None = None) -> None:
        self.router = router or LLMRouter()
        self.settings = get_settings()
        self.cache_dir = ensure_dir(self.settings.data_dir / "_panel_vision_cache" / "character_portrait")

    def run(
        self,
        *,
        project_dir: Path,
        page_paths: list[Path],
        panels: list[PanelBox],
        project_title: str,
        chapter_metadata: ChapterMetadata,
        force_refresh: bool = False,
        progress_callback: Any | None = None,
        cancel_callback: Any | None = None,
    ) -> list[CanonicalCharacterRecord]:
        output_path = project_dir / "output" / "canonical_characters.json"
        if output_path.exists() and not force_refresh:
            payload = read_json(output_path, default=[])
            if isinstance(payload, list):
                return [CanonicalCharacterRecord.model_validate(item) for item in payload]

        loader = ImageLoader(project_dir=project_dir, page_paths=page_paths, max_edge=960)
        kept_panels = [panel for panel in sorted(panels, key=lambda item: item.order) if panel.keep]
        batches = [
            list(range(start, min(start + self.BATCH_SIZE - 1, len(page_paths)) + 1))
            for start in range(1, len(page_paths) + 1, self.BATCH_SIZE)
        ]
        merged: dict[str, dict[str, Any]] = {}
        project_context = self._project_context(project_title, chapter_metadata)

        # Build per-batch metadata up-front so we can split cache hits from
        # cache misses and run misses concurrently.
        batch_specs: list[dict[str, Any]] = []
        for page_numbers in batches:
            image_paths = {
                page_number: loader.page_thumbnail_path(page_number, max_edge=960)
                for page_number in page_numbers
            }
            cache_key = loader.composite_hash(
                [
                    _PROMPT_VERSION,
                    project_title,
                    json.dumps(self._chapter_context(chapter_metadata), sort_keys=True, ensure_ascii=False),
                    *[
                        loader.image_payload(path)[2]
                        for path in image_paths.values()
                        if path is not None and path.exists()
                    ],
                ]
            )
            cache_path = self.cache_dir / f"{cache_key}.json"
            batch_specs.append(
                {
                    "page_numbers": page_numbers,
                    "image_paths": image_paths,
                    "cache_path": cache_path,
                }
            )

        # First pass: cache hits, sequential and instant.
        results: list[dict[str, Any] | None] = [None] * len(batch_specs)
        miss_indices: list[int] = []
        for index, spec in enumerate(batch_specs):
            if spec["cache_path"].exists() and not force_refresh:
                results[index] = read_json(spec["cache_path"], default={"characters": []})
            else:
                miss_indices.append(index)

        # Second pass: cache misses, parallel. Each worker calls
        # ``_enumerate_pages_with_split`` which internally uses ``asyncio.run``
        # — that creates a fresh event loop per call so threads are safe.
        completed = 0

        def _process_miss(idx: int) -> tuple[int, dict[str, Any]]:
            spec = batch_specs[idx]
            payload = self._enumerate_pages_with_split(
                page_numbers=spec["page_numbers"],
                image_paths=spec["image_paths"],
                chapter_metadata=chapter_metadata,
                project_context=project_context,
            )
            write_json(spec["cache_path"], payload)
            return idx, payload

        if miss_indices:
            with ThreadPoolExecutor(max_workers=max(1, self.PARALLEL_WORKERS)) as pool:
                for idx, payload in pool.map(_process_miss, miss_indices):
                    results[idx] = payload
                    completed += 1
                    if cancel_callback:
                        cancel_callback()
                    if progress_callback:
                        spec = batch_specs[idx]
                        progress_callback(
                            round(completed / max(len(miss_indices), 1) * 100, 2),
                            f"Enumerating characters from pages "
                            f"{spec['page_numbers'][0]}-{spec['page_numbers'][-1]}",
                        )

        # Third pass: merge all per-batch results in deterministic page order.
        for spec, payload in zip(batch_specs, results):
            if payload is None:
                continue
            page_numbers = spec["page_numbers"]
            for item in payload.get("characters", []) or []:
                name = self._normalize_name(str(item.get("name") or ""))
                if not name:
                    continue
                key = name.casefold()
                current = merged.get(key, {})
                current_pages = {
                    int(page)
                    for page in current.get("portrait_pages", []) or []
                    if isinstance(page, int) or str(page).isdigit()
                }
                next_pages = {
                    int(page)
                    for page in item.get("portrait_pages", []) or []
                    if isinstance(page, int) or str(page).isdigit()
                }
                merged[key] = {
                    "name": name,
                    "role": self._preferred_role(str(current.get("role") or ""), str(item.get("role") or "")),
                    "visual_description": self._preferred_description(
                        str(current.get("visual_description") or ""),
                        str(item.get("visual_description") or ""),
                    ),
                    "portrait_pages": sorted(current_pages | next_pages)[:8],
                    "aliases": sorted(
                        {
                            *[str(alias).strip() for alias in current.get("aliases", []) or [] if str(alias).strip()],
                            *[str(alias).strip() for alias in item.get("aliases", []) or [] if str(alias).strip()],
                        }
                    )[:8],
                }

        records: list[CanonicalCharacterRecord] = []
        consolidated = self._consolidate_characters(list(merged.values()))
        consolidation_cache_key = loader.composite_hash(
            [
                _PROMPT_VERSION,
                "character_portrait_consolidation_v1",
                project_title,
                json.dumps(self._chapter_context(chapter_metadata), sort_keys=True, ensure_ascii=False),
                json.dumps(consolidated, sort_keys=True, ensure_ascii=False),
            ]
        )
        consolidation_cache_path = self.cache_dir / f"{consolidation_cache_key}_consolidated.json"
        if consolidation_cache_path.exists() and not force_refresh:
            payload = read_json(consolidation_cache_path, default={"characters": consolidated})
        else:
            try:
                result = asyncio.run(
                    self.router.consolidate_character_portraits(
                        consolidated,
                        {
                            "chapter_context": self._chapter_context(chapter_metadata),
                            "project_context": project_context,
                        },
                        provider="gemini",
                    )
                )
                payload = result.payload
                write_json(consolidation_cache_path, payload)
            except Exception as exc:
                logger.warning("Character portrait consolidation failed; using local merge only: %s", exc)
                payload = {"characters": consolidated}
        consolidated = self._normalize_consolidated_payload(payload.get("characters", consolidated))
        consolidated = self._apply_series_name_hints(consolidated, chapter_metadata)
        consolidated = self._absorb_placeholder_singletons(consolidated)
        # Final cleanup: the LLM consolidation may have folded placeholder
        # labels like "Unknown man with curly hair" into the wrong named
        # canonical. Drop any alias whose own gender/hair markers disagree
        # with the canonical it was merged into.
        consolidated = self._prune_mismatched_aliases(consolidated)
        ordered = sorted(
            consolidated,
            key=lambda item: (_ROLE_PRIORITY.get(str(item.get("role") or "supporting"), 9), str(item.get("name") or "").casefold()),
        )
        for index, item in enumerate(ordered, start=1):
            portrait_pages = [int(page) for page in item.get("portrait_pages", []) or [] if int(page) > 0]
            portrait_panel_ids = self._portrait_panel_ids_for_pages(kept_panels, portrait_pages)
            records.append(
                CanonicalCharacterRecord(
                    stable_id=f"char-{index:03d}",
                    name=str(item.get("name") or "").strip(),
                    role=str(item.get("role") or "supporting").strip() or "supporting",
                    visual_description=str(item.get("visual_description") or "").strip(),
                    portrait_panel_ids=portrait_panel_ids[:3],
                    portrait_pages=portrait_pages[:8],
                    aliases=[str(alias).strip() for alias in item.get("aliases", []) or [] if str(alias).strip()][:8],
                )
            )

        write_json(output_path, [record.model_dump(mode="json") for record in records])
        return records

    def _chapter_context(self, chapter_metadata: ChapterMetadata) -> dict[str, Any]:
        return {
            "manga_title": chapter_metadata.manga_title,
            "chapter_title": chapter_metadata.chapter_title,
            "chapter_number": chapter_metadata.chapter_number,
            "language": chapter_metadata.language,
        }

    def _enumerate_pages_with_split(
        self,
        *,
        page_numbers: list[int],
        image_paths: dict[int, Path],
        chapter_metadata: ChapterMetadata,
        project_context: dict[str, Any],
    ) -> dict[str, Any]:
        """Call Gemini character enumeration with binary-split fallback.

        Gemini occasionally rejects the prompt with ``promptFeedback.blockReason
        OTHER`` even when individual images are fine. Splitting the batch and
        retrying the halves recovers character data we'd otherwise lose.
        Returns ``{"characters": [...]}`` (possibly empty) and never raises.
        """
        if not page_numbers:
            return {"characters": []}
        pages = [{"page": page_number} for page_number in page_numbers]
        request_paths = {page: path for page, path in image_paths.items() if path is not None}
        try:
            result = asyncio.run(
                self.router.enumerate_characters_from_pages(
                    pages,
                    {
                        "chapter_context": self._chapter_context(chapter_metadata),
                        "project_context": project_context,
                    },
                    provider="gemini",
                    page_image_paths=request_paths,
                )
            )
            return result.payload
        except Exception as exc:
            if len(page_numbers) <= 1:
                logger.warning(
                    "Character portrait enumeration failed for page %s: %s. Continuing with empty batch.",
                    page_numbers,
                    exc,
                )
                return {"characters": []}
            logger.warning(
                "Character portrait enumeration failed for pages %s: %s. Splitting batch and retrying.",
                page_numbers,
                exc,
            )
        midpoint = len(page_numbers) // 2
        left = self._enumerate_pages_with_split(
            page_numbers=page_numbers[:midpoint],
            image_paths=image_paths,
            chapter_metadata=chapter_metadata,
            project_context=project_context,
        )
        right = self._enumerate_pages_with_split(
            page_numbers=page_numbers[midpoint:],
            image_paths=image_paths,
            chapter_metadata=chapter_metadata,
            project_context=project_context,
        )
        merged_characters: list[dict[str, Any]] = []
        for partial in (left, right):
            if isinstance(partial, dict):
                items = partial.get("characters", []) or []
                if isinstance(items, list):
                    merged_characters.extend(item for item in items if isinstance(item, dict))
        return {"characters": merged_characters}

    def _project_context(self, project_title: str, chapter_metadata: ChapterMetadata) -> dict[str, Any]:
        synopsis = self._series_synopsis_excerpt(chapter_metadata)
        name_hints = self._known_name_hints(chapter_metadata)
        return {
            "project_title": project_title,
            "series_synopsis_excerpt": synopsis,
            "known_name_hints": name_hints,
            "main_protagonist_hint": name_hints[0] if name_hints else "",
        }

    def _series_synopsis_excerpt(self, chapter_metadata: ChapterMetadata) -> str:
        raw = chapter_metadata.raw or {}
        if not isinstance(raw, dict):
            return ""
        for relationship in raw.get("relationships", []) or []:
            if not isinstance(relationship, dict) or relationship.get("type") != "manga":
                continue
            attributes = relationship.get("attributes") or {}
            if not isinstance(attributes, dict):
                continue
            description = attributes.get("description") or {}
            if isinstance(description, dict):
                for key in ("en", "pt-br", "es", "fr"):
                    value = str(description.get(key) or "").strip()
                    if value:
                        return value[:900]
        return ""

    def _known_name_hints(self, chapter_metadata: ChapterMetadata) -> list[str]:
        synopsis = self._series_synopsis_excerpt(chapter_metadata)
        if not synopsis:
            return []
        counts: dict[str, int] = {}
        for match in re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2}\b", synopsis):
            cleaned = self._normalize_name(match)
            if not cleaned or len(cleaned) < 3:
                continue
            if cleaned in {"But", "The", "In", "Global Freeze", "I Created", "Apocalypse Shelter"}:
                continue
            counts[cleaned] = counts.get(cleaned, 0) + 1
        ordered = sorted(counts.items(), key=lambda item: (item[1], len(item[0].split())), reverse=True)
        return [name for name, _ in ordered[:8]]

    def _normalize_name(self, raw: str) -> str:
        cleaned = re.sub(r"\s+", " ", str(raw or "").strip())
        if not cleaned:
            return ""
        if len(cleaned.split()) > 5:
            cleaned = " ".join(cleaned.split()[:5])
        return cleaned

    def _normalize_name_key(self, raw: str) -> str:
        cleaned = re.sub(r"[^\w\s'-]", " ", str(raw or "").strip(), flags=re.UNICODE)
        cleaned = re.sub(r"\s+", " ", cleaned).strip().casefold()
        if not cleaned:
            return ""
        return cleaned

    def _name_without_title(self, raw: str) -> str:
        tokens = self._normalize_name(raw).split()
        while tokens and tokens[0].casefold().rstrip(".") in _TITLE_PREFIXES:
            tokens = tokens[1:]
        return " ".join(tokens).strip()

    def _is_placeholder_name(self, raw: str) -> bool:
        name = self._normalize_name(raw)
        if not name:
            return True
        if _PLACEHOLDER_NAME_PATTERN.search(name):
            return True
        return False

    def _name_quality(self, raw: str) -> tuple[int, int]:
        name = self._normalize_name(raw)
        if not name:
            return (0, 0)
        base = self._name_without_title(name) or name
        tokens = [token for token in base.split() if token]
        if self._is_placeholder_name(name):
            return (1, len(tokens))
        if len(tokens) >= 2:
            return (5, len(tokens))
        if name != base and tokens:
            return (3, len(tokens))
        return (4, len(tokens))

    def _prefer_name(self, current: str, incoming: str) -> str:
        current_quality = self._name_quality(current)
        incoming_quality = self._name_quality(incoming)
        if incoming_quality > current_quality:
            return self._normalize_name(incoming)
        if incoming_quality == current_quality and len(self._normalize_name(incoming)) > len(self._normalize_name(current)):
            return self._normalize_name(incoming)
        return self._normalize_name(current)

    def _description_tokens(self, raw: str) -> set[str]:
        return {
            token
            for token in re.findall(r"[a-zA-Z][a-zA-Z'-]+", str(raw or "").casefold())
            if token not in _DESCRIPTION_STOPWORDS
        }

    def _visual_similarity(self, current: dict[str, Any], incoming: dict[str, Any]) -> float:
        current_tokens = self._description_tokens(str(current.get("visual_description") or ""))
        incoming_tokens = self._description_tokens(str(incoming.get("visual_description") or ""))
        if not current_tokens or not incoming_tokens:
            return 0.0
        overlap = len(current_tokens & incoming_tokens)
        denominator = max(len(current_tokens), len(incoming_tokens), 1)
        return overlap / denominator

    def _appearance_traits(
        self,
        item: dict[str, Any],
        *,
        include_aliases: bool = True,
    ) -> dict[str, set[str]]:
        # Read traits from both `visual_description` AND the name / aliases.
        # Placeholder names like "Unknown red-haired man" carry the only hair
        # color + gender signal because the LLM often strips redundant words
        # from the description field. Without this, a female named canonical
        # (gender="woman") can wrongly absorb "Unknown man with curly hair"
        # because the placeholder's description has no "man" token.
        #
        # ``include_aliases=False`` is used by the alias-pruning pass, where
        # the canonical's own traits must not be contaminated by the very
        # aliases we are about to audit. Otherwise a wrongly-absorbed male
        # placeholder alias would teach the canonical it is also "male" and
        # then no alias could ever fail the gender check.
        parts: list[str] = [str(item.get("visual_description") or "")]
        name = str(item.get("name") or "")
        if name:
            parts.append(name)
        if include_aliases:
            for alias in item.get("aliases", []) or []:
                parts.append(str(alias))
        text = " ".join(parts).casefold()
        tokens = {
            token
            for token in re.findall(r"[a-zA-Z][a-zA-Z'-]+", text)
        }
        hair_colors = {token for token in tokens if token in _HAIR_COLORS}
        gender_markers = {token for token in tokens if token in _GENDER_MARKERS}
        # Treat "<color>-haired" as a hair color too (e.g. "red-haired man").
        for compound in re.findall(r"([a-z]+)-haired", text):
            if compound in _HAIR_COLORS:
                hair_colors.add(compound)
        return {
            "hair_colors": hair_colors,
            "gender_markers": gender_markers,
        }

    def _appearance_conflict(self, current: dict[str, Any], incoming: dict[str, Any]) -> bool:
        current_traits = self._appearance_traits(current)
        incoming_traits = self._appearance_traits(incoming)
        if (
            current_traits["gender_markers"]
            and incoming_traits["gender_markers"]
            and current_traits["gender_markers"].isdisjoint(incoming_traits["gender_markers"])
        ):
            return True
        if (
            current_traits["hair_colors"]
            and incoming_traits["hair_colors"]
            and current_traits["hair_colors"].isdisjoint(incoming_traits["hair_colors"])
        ):
            return True
        return False

    def _names_equivalent(self, current: dict[str, Any], incoming: dict[str, Any]) -> bool:
        current_candidates = {
            self._normalize_name_key(value)
            for value in [current.get("name"), *(current.get("aliases", []) or [])]
            if self._normalize_name_key(str(value or ""))
        }
        incoming_candidates = {
            self._normalize_name_key(value)
            for value in [incoming.get("name"), *(incoming.get("aliases", []) or [])]
            if self._normalize_name_key(str(value or ""))
        }
        if current_candidates & incoming_candidates:
            return True

        current_base = self._normalize_name_key(self._name_without_title(str(current.get("name") or "")))
        incoming_base = self._normalize_name_key(self._name_without_title(str(incoming.get("name") or "")))
        if current_base and incoming_base:
            if current_base == incoming_base:
                return True
            current_tokens = current_base.split()
            incoming_tokens = incoming_base.split()
            if len(current_tokens) == 1 and current_tokens[0] in incoming_tokens:
                return True
            if len(incoming_tokens) == 1 and incoming_tokens[0] in current_tokens:
                return True
            similarity = SequenceMatcher(None, current_base, incoming_base).ratio()
            if similarity >= 0.88:
                return True
            if len(current_tokens) >= 2 and len(incoming_tokens) >= 2:
                if current_tokens[-1] == incoming_tokens[-1] and SequenceMatcher(None, current_tokens[0], incoming_tokens[0]).ratio() >= 0.72:
                    return True
        return False

    def _should_merge_character(self, current: dict[str, Any], incoming: dict[str, Any]) -> bool:
        if self._names_equivalent(current, incoming):
            return True
        if self._appearance_conflict(current, incoming):
            return False
        role_a = str(current.get("role") or "supporting").casefold()
        role_b = str(incoming.get("role") or "supporting").casefold()
        visual_similarity = self._visual_similarity(current, incoming)
        if self._is_placeholder_name(str(current.get("name") or "")) or self._is_placeholder_name(str(incoming.get("name") or "")):
            if role_a == role_b and role_a == "protagonist" and visual_similarity >= 0.34:
                return True
            if role_a == role_b and visual_similarity >= 0.48:
                return True
        if role_a == role_b == "protagonist" and visual_similarity >= 0.32:
            return True
        return False

    def _merge_character(self, current: dict[str, Any], incoming: dict[str, Any]) -> dict[str, Any]:
        canonical_name = self._prefer_name(str(current.get("name") or ""), str(incoming.get("name") or ""))
        aliases = {
            *[str(alias).strip() for alias in current.get("aliases", []) or [] if str(alias).strip()],
            *[str(alias).strip() for alias in incoming.get("aliases", []) or [] if str(alias).strip()],
            str(current.get("name") or "").strip(),
            str(incoming.get("name") or "").strip(),
        }
        aliases = {alias for alias in aliases if alias and alias != canonical_name}
        return {
            "name": canonical_name,
            "role": self._preferred_role(str(current.get("role") or ""), str(incoming.get("role") or "")),
            "visual_description": self._preferred_description(
                str(current.get("visual_description") or ""),
                str(incoming.get("visual_description") or ""),
            ),
            "portrait_pages": sorted(
                {
                    *[int(page) for page in current.get("portrait_pages", []) or [] if int(page) > 0],
                    *[int(page) for page in incoming.get("portrait_pages", []) or [] if int(page) > 0],
                }
            )[:8],
            "aliases": sorted(aliases)[:12],
        }

    def _collapse_primary_protagonist(self, items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        protagonists = [item for item in items if str(item.get("role") or "").casefold() == "protagonist"]
        others = [item for item in items if str(item.get("role") or "").casefold() != "protagonist"]
        if len(protagonists) <= 1:
            return items
        primary = max(
            protagonists,
            key=lambda item: (
                self._name_quality(str(item.get("name") or "")),
                len(item.get("portrait_pages", []) or []),
                len(str(item.get("visual_description") or "")),
            ),
        )
        kept: list[dict[str, Any]] = []
        for item in protagonists:
            if item is primary:
                continue
            if (
                self._should_merge_character(primary, item)
                or self._names_equivalent(primary, item)
                or (
                    self._is_placeholder_name(str(item.get("name") or ""))
                    and not self._appearance_conflict(primary, item)
                    and self._visual_similarity(primary, item) >= 0.38
                )
            ):
                primary = self._merge_character(primary, item)
            else:
                kept.append(item)
        return [primary, *kept, *others]

    def _consolidate_characters(self, items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        clusters: list[dict[str, Any]] = []
        ordered = sorted(
            items,
            key=lambda item: (
                self._name_quality(str(item.get("name") or "")),
                len(item.get("portrait_pages", []) or []),
                len(str(item.get("visual_description") or "")),
            ),
            reverse=True,
        )
        for item in ordered:
            merged = False
            for index, current in enumerate(clusters):
                if self._should_merge_character(current, item):
                    clusters[index] = self._merge_character(current, item)
                    merged = True
                    break
            if not merged:
                clusters.append(
                    {
                        "name": self._normalize_name(str(item.get("name") or "")),
                        "role": str(item.get("role") or "supporting").strip() or "supporting",
                        "visual_description": str(item.get("visual_description") or "").strip(),
                        "portrait_pages": [int(page) for page in item.get("portrait_pages", []) or [] if int(page) > 0][:8],
                        "aliases": [str(alias).strip() for alias in item.get("aliases", []) or [] if str(alias).strip()][:12],
                    }
                )
        return self._collapse_primary_protagonist(clusters)

    def _normalize_consolidated_payload(self, items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        normalized: list[dict[str, Any]] = []
        merged_by_name: dict[str, dict[str, Any]] = {}
        for item in items:
            if not isinstance(item, dict):
                continue
            name = self._normalize_name(str(item.get("name") or ""))
            if not name:
                continue
            aliases = [
                self._normalize_name(str(alias or ""))
                for alias in item.get("aliases", []) or []
                if self._normalize_name(str(alias or ""))
            ]
            key = self._normalize_name_key(name)
            current = merged_by_name.get(key)
            normalized_item = {
                "name": name,
                "role": str(item.get("role") or "supporting").strip() or "supporting",
                "visual_description": str(item.get("visual_description") or "").strip(),
                "portrait_pages": [
                    int(page)
                    for page in item.get("portrait_pages", []) or []
                    if isinstance(page, int) or str(page).isdigit()
                ][:8],
                "aliases": sorted({alias for alias in aliases if alias and alias != name})[:12],
            }
            if current is None:
                merged_by_name[key] = normalized_item
                continue
            merged_by_name[key] = {
                "name": self._prefer_name(str(current.get("name") or ""), name),
                "role": self._preferred_role(str(current.get("role") or ""), normalized_item["role"]),
                "visual_description": self._preferred_description(
                    str(current.get("visual_description") or ""),
                    normalized_item["visual_description"],
                ),
                "portrait_pages": sorted(
                    {
                        *[int(page) for page in current.get("portrait_pages", []) or [] if int(page) > 0],
                        *normalized_item["portrait_pages"],
                    }
                )[:8],
                "aliases": sorted(
                    {
                        *[str(alias).strip() for alias in current.get("aliases", []) or [] if str(alias).strip()],
                        *normalized_item["aliases"],
                    }
                )[:12],
            }
        normalized.extend(merged_by_name.values())
        return normalized

    def _apply_series_name_hints(
        self,
        items: list[dict[str, Any]],
        chapter_metadata: ChapterMetadata,
    ) -> list[dict[str, Any]]:
        hints = self._known_name_hints(chapter_metadata)
        if not hints:
            return items
        protagonist_hint = hints[0]
        protagonist_hint_key = self._normalize_name_key(protagonist_hint)
        if any(self._normalize_name_key(str(item.get("name") or "")) == protagonist_hint_key for item in items):
            return items

        hinted_tokens = set(protagonist_hint_key.split())
        best_index: int | None = None
        best_score: tuple[int, int] | None = None
        for index, item in enumerate(items):
            if str(item.get("role") or "").casefold() != "protagonist":
                continue
            candidate_names = [str(item.get("name") or ""), *(item.get("aliases", []) or [])]
            match_score = 0
            for candidate in candidate_names:
                candidate_key = self._normalize_name_key(candidate)
                if not candidate_key:
                    continue
                candidate_tokens = set(candidate_key.split())
                if hinted_tokens & candidate_tokens:
                    match_score = max(match_score, len(hinted_tokens & candidate_tokens))
                elif SequenceMatcher(None, candidate_key, protagonist_hint_key).ratio() >= 0.72:
                    match_score = max(match_score, 1)
            if match_score <= 0:
                continue
            score = (match_score, self._name_quality(str(item.get("name") or ""))[0])
            if best_score is None or score > best_score:
                best_index = index
                best_score = score

        if best_index is None:
            return items

        updated = [dict(item) for item in items]
        chosen = dict(updated[best_index])
        aliases = {
            *[str(alias).strip() for alias in chosen.get("aliases", []) or [] if str(alias).strip()],
            str(chosen.get("name") or "").strip(),
        }
        aliases.discard(protagonist_hint)
        chosen["name"] = protagonist_hint
        chosen["aliases"] = sorted(aliases)[:12]
        updated[best_index] = chosen
        return updated

    def _absorb_placeholder_singletons(
        self,
        items: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Merge placeholder-named canonicals into the closest named canonical.

        After the LLM consolidation pass there are sometimes leftover entries
        whose ``name`` is a descriptive placeholder ("Unknown red-haired man",
        "Protagonist", "Delivery man", …). These poison narration prompts
        because the downstream character_dictionary forwards them verbatim, so
        the narrator ends up saying things like "the unknown red-haired man
        walks inside" instead of a real character name or just "someone".

        Strategy per placeholder:
          1. Find every named canonical whose appearance is compatible
             (non-conflicting gender + overlapping hair color, and a
             non-trivial visual-description overlap).
          2. If at least one match exists, merge into the best one (highest
             visual similarity, then highest name quality).
          3. Otherwise demote the placeholder to ``role="cameo"`` so the
             narration layer can filter it out while still keeping it in
             ``canonical_characters.json`` as an audit trail.

        Named canonicals are left untouched — this only promotes/absorbs
        placeholder orphans.
        """
        if not items:
            return items

        named = [item for item in items if not self._is_placeholder_name(str(item.get("name") or ""))]
        placeholders = [item for item in items if self._is_placeholder_name(str(item.get("name") or ""))]
        if not placeholders:
            return items

        result: list[dict[str, Any]] = [dict(item) for item in named]
        for placeholder in placeholders:
            best_index: int | None = None
            best_score: tuple[float, tuple[int, int]] | None = None
            placeholder_traits = self._appearance_traits(placeholder)
            for index, named_item in enumerate(result):
                if self._appearance_conflict(named_item, placeholder):
                    continue
                named_traits = self._appearance_traits(named_item)
                # Require at least one shared trait (gender or hair) to avoid
                # absorbing random placeholders into arbitrary named characters.
                gender_overlap = bool(
                    placeholder_traits["gender_markers"]
                    and named_traits["gender_markers"]
                    and placeholder_traits["gender_markers"] & named_traits["gender_markers"]
                )
                hair_overlap = bool(
                    placeholder_traits["hair_colors"]
                    and named_traits["hair_colors"]
                    and placeholder_traits["hair_colors"] & named_traits["hair_colors"]
                )
                if not (gender_overlap or hair_overlap):
                    continue
                visual = self._visual_similarity(named_item, placeholder)
                # Gate — need at least a modest appearance overlap or a gender+hair
                # match before we absorb, so distinct people don't collapse.
                if visual < 0.2 and not (gender_overlap and hair_overlap):
                    continue
                score = (
                    visual + (0.1 if gender_overlap else 0.0) + (0.1 if hair_overlap else 0.0),
                    self._name_quality(str(named_item.get("name") or "")),
                )
                if best_score is None or score > best_score:
                    best_score = score
                    best_index = index
            if best_index is not None:
                result[best_index] = self._merge_character(result[best_index], placeholder)
                continue
            # No named match — keep the placeholder but demote it so the
            # narration layer knows to skip it.
            demoted = dict(placeholder)
            demoted["role"] = "cameo"
            result.append(demoted)
        return result

    def _prune_mismatched_aliases(
        self,
        items: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Drop aliases whose own text clearly conflicts with the canonical.

        The LLM consolidation pass sometimes folds placeholder labels into the
        wrong canonical (e.g. "Unknown man with curly hair" absorbed under a
        female canonical named "Fang Yu Qing"). Our appearance-conflict check
        can only catch these when the pre-consolidation entries arrive as
        separate canonicals — by the time they're already aliases, we need to
        re-check each one and drop the bad ones.

        An alias is "bad" if the gender/hair markers extracted from its own
        text disagree with the canonical's own markers (e.g. "Unknown man..."
        alias on a canonical whose description says "woman"). We only prune
        placeholder-style aliases — real-name aliases ("Mr. Zhang", "Tio",
        "Ning Ning") are left alone even if their text happens to share a
        gender token.
        """
        if not items:
            return items
        cleaned: list[dict[str, Any]] = []
        for item in items:
            current = dict(item)
            aliases = list(current.get("aliases", []) or [])
            if not aliases:
                cleaned.append(current)
                continue
            # Use only the canonical's own name + visual_description — not
            # its aliases — so a wrongly-absorbed alias can't justify itself.
            canonical_traits = self._appearance_traits(current, include_aliases=False)
            kept_aliases: list[str] = []
            for alias in aliases:
                alias_str = str(alias).strip()
                if not alias_str:
                    continue
                # Only audit placeholder-style aliases — a real-name alias
                # (e.g. "Mr. Zhang", "Tio", "Ning Ning") should stick even if
                # we can't verify its gender.
                if not self._is_placeholder_name(alias_str):
                    kept_aliases.append(alias_str)
                    continue
                alias_proxy = {"name": alias_str, "visual_description": ""}
                alias_traits = self._appearance_traits(alias_proxy)
                gender_conflict = (
                    canonical_traits["gender_markers"]
                    and alias_traits["gender_markers"]
                    and canonical_traits["gender_markers"].isdisjoint(alias_traits["gender_markers"])
                )
                hair_conflict = (
                    canonical_traits["hair_colors"]
                    and alias_traits["hair_colors"]
                    and canonical_traits["hair_colors"].isdisjoint(alias_traits["hair_colors"])
                )
                if gender_conflict or hair_conflict:
                    logger.debug(
                        "Pruning alias %r from canonical %r (gender_conflict=%s hair_conflict=%s)",
                        alias_str,
                        current.get("name"),
                        bool(gender_conflict),
                        bool(hair_conflict),
                    )
                    continue
                kept_aliases.append(alias_str)
            current["aliases"] = kept_aliases
            cleaned.append(current)
        return cleaned

    def _preferred_role(self, current: str, incoming: str) -> str:
        current_role = (current or "supporting").strip().casefold()
        incoming_role = (incoming or "supporting").strip().casefold()
        if _ROLE_PRIORITY.get(incoming_role, 9) < _ROLE_PRIORITY.get(current_role, 9):
            return incoming_role
        return current_role or incoming_role or "supporting"

    def _preferred_description(self, current: str, incoming: str) -> str:
        current = str(current or "").strip()
        incoming = str(incoming or "").strip()
        if not current:
            return incoming
        if len(incoming) > len(current):
            return incoming
        return current

    def _portrait_panel_ids_for_pages(self, panels: list[PanelBox], portrait_pages: list[int]) -> list[str]:
        page_set = {int(page) for page in portrait_pages}
        return [
            panel.id
            for panel in panels
            if int(panel.page) in page_set
        ]
