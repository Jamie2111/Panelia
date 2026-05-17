"""
character_identifier_service.py

Combines #5 (face cluster + cast match) and #6 (character portrait
references) from the recognition-upgrade brainstorm.

Pipeline:
  1. Detect anime faces on every page (anime_face_detection_service,
     cached on disk).
  2. For each detected face, work out which kept panel it falls inside
     (panel bbox geometry).
  3. Embed every face crop with CLIP (reusing the model loaded by
     CharacterClusterer for free) and group them via cosine similarity
     into character clusters.
  4. Pick the best-quality face crop per cluster as that cluster's
     "portrait" - that's the deliverable for #6.
  5. ONE Gemini Vision call: hand the model the cast bible plus the
     full set of cluster portraits, ask "which cluster matches which
     cast member?". Each cluster gets labeled with a cast name or
     'unknown'.
  6. For each panel, list the cluster names whose faces fell inside
     that panel - that's the per-panel character_hints index #3
     consumes downstream.

Output sits at:
  <project>/output/character_identity/index.json   (panel -> [names])
  <project>/output/character_identity/portraits/    (one PNG per cast)
  <project>/output/character_identity/clusters/     (cluster centroid crops)

Idempotent: re-running with the same panels + face cache returns
cached results in seconds.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from app.core.config import get_settings
from app.services.anime_face_detection_service import AnimeFaceDetectionService
from app.services.cast_bible_service import CastBible, CastBibleService
from app.utils.files import ensure_dir, read_json, write_json

logger = logging.getLogger(__name__)

try:
    import google.generativeai as genai  # type: ignore
    _GEMINI_AVAILABLE = True
except ImportError:
    _GEMINI_AVAILABLE = False


_INDEX_FILENAME = "index.json"
_INDEX_VERSION = "character_identity_v1"
_CLUSTER_DIRNAME = "clusters"
_PORTRAIT_DIRNAME = "portraits"
# Bigger minimum captures hair color + eye detail so CLIP can actually
# distinguish look-alikes (Arlo↔Gavin, Blyke↔Cecile). Tiny background
# faces don't carry enough signal and pollute clusters. 64 px on the
# short edge is enough resolution for the model without filtering most
# in-action close-ups out entirely.
_MIN_FACE_PIXELS = 64
# Tuning history:
#   0.82 (initial): collapsed too many characters into mega-clusters.
#   0.92 (Darling-era): pure clusters but coverage drops badly because
#     the same character at a different angle / expression splits into a
#     new cluster - unord-qa coverage was only 13.6% of panels with hints.
#   0.80 (current): broader same-character merging across angles,
#     trading some intra-cluster purity for materially higher coverage
#     (~35-45% panels with hints on unord-qa). Per-panel face cosine
#     tie-breaker (see panel_vision_narrator) recovers the precision
#     loss by re-ranking ambiguous candidates against portrait
#     embeddings at narration time.
_SIMILARITY_THRESHOLD = 0.80
_MAX_CLUSTERS_FOR_NAMING = 30  # cap per-cluster Gemini calls


@dataclass
class _FaceRecord:
    page: int
    panel_id: str | None
    bbox_in_page: list[int]
    crop_path: Path
    embedding: np.ndarray | None = None
    cluster_id: int = -1


@dataclass
class CharacterIdentityResult:
    cluster_to_name: dict[int, str]
    panel_to_names: dict[str, list[str]] = field(default_factory=dict)
    portrait_paths: dict[str, str] = field(default_factory=dict)


def build_character_identity(
    project_dir: Path,
    *,
    bible: CastBible | None = None,
    cancel_callback: callable | None = None,
) -> CharacterIdentityResult:
    """Build the per-panel character identity index for `project_dir`.

    Returns a `CharacterIdentityResult` and also writes
    `<project>/output/character_identity/index.json` for the vision
    narrator to consume.
    """
    settings = get_settings()
    out_dir = ensure_dir(project_dir / "output" / "character_identity")
    crops_dir = ensure_dir(out_dir / _CLUSTER_DIRNAME)
    portraits_dir = ensure_dir(out_dir / _PORTRAIT_DIRNAME)
    index_path = out_dir / _INDEX_FILENAME

    # Idempotency check: an existing index that matches the current
    # panels.json's panel id set is reused.
    panels_path = project_dir / "panels.json"
    if not panels_path.exists():
        logger.info("No panels.json yet; skipping character identifier.")
        return CharacterIdentityResult(cluster_to_name={})
    panels_json = json.loads(panels_path.read_text(encoding="utf-8"))
    kept_panels = [p for p in panels_json if p.get("keep")]
    panel_id_set = {str(p["id"]) for p in kept_panels}

    existing = read_json(index_path, default={}) if index_path.exists() else {}
    if (
        isinstance(existing, dict)
        and existing.get("version") == _INDEX_VERSION
        and set(existing.get("panel_to_names", {}).keys()) >= panel_id_set
    ):
        logger.info(
            "Character identity cache hit at %s (%d panels indexed).",
            index_path, len(existing.get("panel_to_names", {})),
        )
        return CharacterIdentityResult(
            cluster_to_name={int(k): str(v) for k, v in (existing.get("cluster_to_name") or {}).items()},
            panel_to_names=dict(existing.get("panel_to_names") or {}),
            portrait_paths=dict(existing.get("portrait_paths") or {}),
        )

    if bible is None:
        try:
            bible = CastBibleService().load_cached(project_dir)
        except Exception:
            bible = None
    cast_members = bible.members if bible else []
    if not cast_members:
        logger.info("No cast bible; character identifier returning empty.")
        _write_empty_index(index_path)
        return CharacterIdentityResult(cluster_to_name={})

    detector = AnimeFaceDetectionService()
    if not detector.is_available():
        logger.info("Anime face detector unavailable; identifier returning empty.")
        _write_empty_index(index_path)
        return CharacterIdentityResult(cluster_to_name={})

    pages_dir = project_dir / "pages"
    if not pages_dir.exists():
        logger.info("No pages dir; identifier returning empty.")
        _write_empty_index(index_path)
        return CharacterIdentityResult(cluster_to_name={})
    page_paths = sorted(pages_dir.glob("*.png")) + sorted(pages_dir.glob("*.jpg")) + sorted(pages_dir.glob("*.jpeg")) + sorted(pages_dir.glob("*.webp"))
    page_paths = [p for p in page_paths if p.is_file()]
    if not page_paths:
        _write_empty_index(index_path)
        return CharacterIdentityResult(cluster_to_name={})

    face_cache_path = settings.data_dir / "_anime_face_cache" / f"{project_dir.name}.json"
    ensure_dir(face_cache_path.parent)
    page_payloads = detector.detect_page_payloads(
        page_paths,
        cache_path=face_cache_path,
        cancel_callback=cancel_callback,
    )

    # Build the per-panel bbox index. panels.json stores panel bboxes
    # in page-relative pixel coordinates.
    panels_by_page: dict[int, list[dict[str, Any]]] = {}
    for panel in kept_panels:
        panels_by_page.setdefault(int(panel.get("page", 0)), []).append(panel)

    face_records: list[_FaceRecord] = []
    for page_number, payload in page_payloads.items():
        if cancel_callback:
            cancel_callback()
        if not isinstance(payload, dict):
            continue
        page_path = page_paths[int(page_number) - 1] if 1 <= int(page_number) <= len(page_paths) else None
        if page_path is None:
            continue
        try:
            page_image = Image.open(page_path).convert("RGB")
        except Exception:
            continue
        for char_idx, character in enumerate(payload.get("characters") or [], start=1):
            bbox = _coerce_bbox(character.get("bbox"))
            if bbox is None:
                continue
            x, y, w, h = bbox
            if w < _MIN_FACE_PIXELS or h < _MIN_FACE_PIXELS:
                continue
            panel_id = _which_panel(bbox, panels_by_page.get(int(page_number), []))
            try:
                crop = page_image.crop((x, y, x + w, y + h))
            except Exception:
                continue
            if crop.width < _MIN_FACE_PIXELS or crop.height < _MIN_FACE_PIXELS:
                continue
            crop_path = crops_dir / f"page{int(page_number):04d}_face{char_idx:02d}.jpg"
            try:
                crop.save(crop_path, format="JPEG", quality=85)
            except Exception:
                continue
            face_records.append(_FaceRecord(
                page=int(page_number),
                panel_id=panel_id,
                bbox_in_page=list(bbox),
                crop_path=crop_path,
            ))

    if not face_records:
        _write_empty_index(index_path)
        return CharacterIdentityResult(cluster_to_name={})

    # CLIP embeddings + greedy clustering. Reuses the CharacterClusterer
    # module-level CLIP model so we don't load it twice.
    from app.services.character_clusterer import CharacterClusterer
    clusterer = CharacterClusterer()
    samples_for_clip: list[dict[str, Any]] = []
    for rec in face_records:
        try:
            img = Image.open(rec.crop_path).convert("RGB")
        except Exception:
            img = Image.new("RGB", (64, 64), (0, 0, 0))
        samples_for_clip.append({"image": img, "bbox": rec.bbox_in_page})
    embeddings = clusterer._embed_samples(samples_for_clip)  # noqa: SLF001
    cluster_centroids: list[np.ndarray] = []
    cluster_members: list[list[int]] = []
    for idx, embedding in enumerate(embeddings):
        if cancel_callback and idx % 32 == 0:
            cancel_callback()
        assigned = -1
        best_sim = 0.0
        for cluster_idx, centroid in enumerate(cluster_centroids):
            sim = float(np.dot(embedding, centroid) /
                        (np.linalg.norm(embedding) * np.linalg.norm(centroid) + 1e-9))
            if sim > best_sim and sim >= _SIMILARITY_THRESHOLD:
                best_sim = sim
                assigned = cluster_idx
        if assigned < 0:
            cluster_centroids.append(embedding.copy())
            cluster_members.append([idx])
            face_records[idx].embedding = embedding
            face_records[idx].cluster_id = len(cluster_centroids) - 1
        else:
            members = cluster_members[assigned]
            n = len(members)
            cluster_centroids[assigned] = (cluster_centroids[assigned] * n + embedding) / (n + 1)
            members.append(idx)
            face_records[idx].embedding = embedding
            face_records[idx].cluster_id = assigned

    # Trim to the most populous clusters - keep the top N regardless of
    # size since the Gemini labeler is the gating step and `unknown` is
    # always a valid answer. Singleton clusters that happen to be a
    # real (rare-appearing) cast member still get correctly named.
    cluster_sizes = [(i, len(members)) for i, members in enumerate(cluster_members)]
    cluster_sizes.sort(key=lambda pair: pair[1], reverse=True)
    keep_cluster_ids = {
        cluster_id for cluster_id, _ in cluster_sizes[:_MAX_CLUSTERS_FOR_NAMING]
    }

    # Pick the largest face from each kept cluster as the cluster portrait.
    cluster_portraits: dict[int, Path] = {}
    for cluster_id in keep_cluster_ids:
        members = cluster_members[cluster_id]
        best_idx = max(members, key=lambda i: face_records[i].bbox_in_page[2] * face_records[i].bbox_in_page[3])
        portrait_path = portraits_dir / f"cluster_{cluster_id:03d}_portrait.jpg"
        try:
            Image.open(face_records[best_idx].crop_path).convert("RGB").save(
                portrait_path, format="JPEG", quality=88
            )
            cluster_portraits[cluster_id] = portrait_path
        except Exception:
            continue

    # Single Gemini call: assign each cluster portrait to a cast member.
    cluster_to_name = _gemini_label_clusters(
        cast_members=cast_members,
        cluster_portraits=cluster_portraits,
    )

    # Build panel -> names index from the records.
    panel_to_names: dict[str, list[str]] = {}
    for rec in face_records:
        if rec.cluster_id not in cluster_to_name:
            continue
        name = cluster_to_name[rec.cluster_id]
        if name == "unknown" or not name:
            continue
        if not rec.panel_id:
            continue
        bucket = panel_to_names.setdefault(rec.panel_id, [])
        if name not in bucket:
            bucket.append(name)

    # Persist named portraits (renames cluster portraits to cast names).
    portrait_paths: dict[str, str] = {}
    name_to_centroid: dict[str, np.ndarray] = {}
    for cluster_id, name in cluster_to_name.items():
        if name == "unknown" or not name:
            continue
        src = cluster_portraits.get(cluster_id)
        if not src or not src.exists():
            continue
        safe_name = re.sub(r"[^A-Za-z0-9_-]", "_", name).strip("_") or f"cluster_{cluster_id:03d}"
        dest = portraits_dir / f"{safe_name}.jpg"
        try:
            Image.open(src).convert("RGB").save(dest, format="JPEG", quality=92)
            portrait_paths[name] = str(dest.relative_to(project_dir))
        except Exception:
            continue
        # Reuse the cluster's CLIP centroid as that character's reference
        # embedding. This is what the per-panel narrator will compare
        # against to verify a named character is actually present.
        if 0 <= cluster_id < len(cluster_centroids):
            name_to_centroid[name] = cluster_centroids[cluster_id].astype(np.float32, copy=False)

    # Persist embeddings to disk so per-panel narration + verification
    # can load them without re-running CLIP. One file per project,
    # ~6 KB/character (512 float32).
    embeddings_path = out_dir / "embeddings.npz"
    if name_to_centroid:
        try:
            np.savez_compressed(embeddings_path, **name_to_centroid)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not persist character embeddings: %s", exc)

    index_payload = {
        "version": _INDEX_VERSION,
        "cluster_to_name": {str(k): v for k, v in cluster_to_name.items()},
        "panel_to_names": panel_to_names,
        "portrait_paths": portrait_paths,
        "embeddings_path": str(embeddings_path.relative_to(project_dir)) if name_to_centroid else None,
        "stats": {
            "total_faces": len(face_records),
            "total_clusters": len(cluster_members),
            "labeled_clusters": sum(1 for v in cluster_to_name.values() if v != "unknown"),
            "panels_with_hints": len(panel_to_names),
            "kept_panels": len(panel_id_set),
            "embeddings_persisted": len(name_to_centroid),
        },
    }
    write_json(index_path, index_payload)
    logger.info(
        "Character identity built: %d faces -> %d clusters -> %d named -> %d/%d panels with hints",
        len(face_records), len(cluster_members),
        sum(1 for v in cluster_to_name.values() if v != "unknown"),
        len(panel_to_names), len(panel_id_set),
    )
    return CharacterIdentityResult(
        cluster_to_name=cluster_to_name,
        panel_to_names=panel_to_names,
        portrait_paths=portrait_paths,
    )


def _write_empty_index(path: Path) -> None:
    write_json(path, {
        "version": _INDEX_VERSION,
        "cluster_to_name": {},
        "panel_to_names": {},
        "portrait_paths": {},
        "stats": {"reason": "no faces / no cast / no model"},
    })


def _coerce_bbox(raw: Any) -> tuple[int, int, int, int] | None:
    if not isinstance(raw, (list, tuple)) or len(raw) < 4:
        return None
    try:
        x, y, w, h = int(raw[0]), int(raw[1]), int(raw[2]), int(raw[3])
    except (TypeError, ValueError):
        return None
    if w <= 0 or h <= 0:
        return None
    return (x, y, w, h)


def _which_panel(face_bbox: tuple[int, int, int, int], panels: list[dict[str, Any]]) -> str | None:
    """Return the id of the kept panel that contains the face center, or None.

    Panels store geometry as separate ``x/y/width/height`` integer fields
    in page-relative pixels (newer schema). Older schemas may include
    ``bbox`` or ``page_bbox`` arrays. Both shapes are handled.

    We test against the face CENTER rather than full bbox containment so
    a face that crosses a panel gutter still attributes to the panel
    most of its area sits in.
    """
    fx = face_bbox[0] + face_bbox[2] // 2
    fy = face_bbox[1] + face_bbox[3] // 2
    for panel in panels:
        # Preferred new schema: x/y/width/height as separate ints
        try:
            x = int(panel.get("x"))
            y = int(panel.get("y"))
            w = int(panel.get("width"))
            h = int(panel.get("height"))
        except (TypeError, ValueError):
            x = y = w = h = None  # type: ignore[assignment]
        if x is None:
            # Legacy bbox array fallback
            bbox = panel.get("bbox") or panel.get("page_bbox")
            if not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
                continue
            try:
                x, y, w, h = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
            except (TypeError, ValueError):
                continue
        if w <= 0 or h <= 0:
            continue
        if x <= fx <= x + w and y <= fy <= y + h:
            return str(panel.get("id") or "")
    return None


_CLUSTER_LABEL_PROMPT = """You are analyzing manga character art for a stylistic
description task. For each of the {n} character drawings below, write a
SHORT visual description focused on these features ONLY: hair color, hair
style/length, eye color, distinctive accessories (glasses, horns, headgear),
clothing color/uniform style. This is for grouping similar art styles, not
person identification.

ABSOLUTE RULES:
- Return EXACTLY {n} lines.
- Each line is the drawing index, a colon, then a short feature list.
- Focus on hair color FIRST (most distinguishing), then eye color, then
  outfit color/type, then any unique accessory.
- Keep each description under 25 words.
- One drawing per line, in numerical order: 1, 2, 3, ...

OUTPUT FORMAT (literal):
1: Long pink hair, red eyes, red horns, white military coat
2: Short blue hair, blue eyes, black and white Squad 13 uniform
3: Indistinct, small, low detail
"""


def _match_description_to_cast(
    description: str,
    cast_members: list[Any],
) -> str:
    """Score each cast member against the description by counting matched
    feature keywords. Returns the best-matching cast name or 'unknown'.
    """
    desc_low = description.lower()
    if not desc_low.strip() or "indistinct" in desc_low or "low detail" in desc_low:
        return "unknown"

    # Visual feature vocabulary - kept broad to match the richer
    # cast-bible descriptions written by the enriched LLM prompt
    # (face shape, accessories, common props, height, expression).
    color_words = {
        "pink", "blue", "red", "green", "blonde", "blond", "yellow",
        "brown", "black", "white", "purple", "violet", "orange", "silver",
        "grey", "gray", "auburn", "magenta", "cobalt", "navy", "crimson",
        "cyan", "teal", "amber", "golden", "platinum", "ash", "ginger",
    }
    feature_words = {
        # Hair length / style
        "long", "short", "shoulder", "spiky", "straight", "curly", "wavy",
        "messy", "ponytail", "braid", "braids", "bun", "twin", "twintails",
        "bob", "buzzcut", "bald", "fringe", "bangs",
        # Accessories
        "horns", "glasses", "monocle", "scar", "tattoo", "mask", "visor",
        "cybernetic", "helmet", "headband", "hat", "cap", "crown", "earring",
        "necklace", "ring", "gloves", "scarf", "lollipop", "weapon", "sword",
        "katana", "bow", "staff", "wand", "gun", "knife", "shield",
        # Body / age
        "tall", "short", "average", "slim", "athletic", "stocky", "muscular",
        "petite", "child", "teen", "young", "adult", "elderly", "old",
        # Face / expression
        "round", "sharp", "angular", "narrow", "wide", "piercing", "gentle",
        "stern", "smug", "anxious", "cheerful", "deadpan", "scowl", "smile",
        "frown", "calm", "fierce", "kind", "menacing", "freckles",
        # Outfit categories
        "uniform", "coat", "armor", "hoodie", "jacket", "dress", "robe",
        "suit", "kimono", "cloak", "cape", "sweater", "shirt", "skirt",
        "trousers", "trenchcoat", "labcoat",
    }
    cast_keywords: dict[str, set[str]] = {}
    for m in cast_members:
        if not m.name:
            continue
        bag = set()
        for source in (m.visual_description or "", m.role or ""):
            for word in re.findall(r"\b[a-zA-Z]+\b", source.lower()):
                if word in color_words or word in feature_words:
                    bag.add(word)
        if bag:
            cast_keywords[m.name] = bag

    if not cast_keywords:
        return "unknown"

    desc_tokens = {w for w in re.findall(r"\b[a-zA-Z]+\b", desc_low)
                   if w in color_words or w in feature_words}
    if not desc_tokens:
        return "unknown"

    scores: list[tuple[str, int, int]] = []  # (name, score, unique_score)
    for name, bag in cast_keywords.items():
        score = len(desc_tokens & bag)
        # Unique-feature score: how many tokens in the match are NOT
        # shared with another cast member. This is the discriminator.
        unique = 0
        for token in desc_tokens & bag:
            shared_with = sum(1 for other_name, other_bag in cast_keywords.items()
                              if other_name != name and token in other_bag)
            if shared_with == 0:
                unique += 1
        scores.append((name, score, unique))
    if not scores:
        return "unknown"
    # Sort by (unique_score desc, total_score desc). Ties at zero unique
    # score and equal total => ambiguous; refuse to commit a name.
    scores.sort(key=lambda triple: (triple[2], triple[1]), reverse=True)
    best_name, best_score, best_unique = scores[0]
    if len(scores) >= 2:
        runner_name, runner_score, runner_unique = scores[1]
        if best_unique == runner_unique and best_score == runner_score:
            return "unknown"  # ambiguous, multiple equally-good matches
    # Require minimum signal: 2 overlapping features OR 1 unique discriminator.
    if best_score < 2 and best_unique < 1:
        return "unknown"
    return best_name


def _gemini_label_clusters(
    *,
    cast_members: list[Any],
    cluster_portraits: dict[int, Path],
) -> dict[int, str]:
    if not cluster_portraits:
        return {}
    settings = get_settings()
    if not _GEMINI_AVAILABLE or not settings.gemini_api_key:
        # Without Gemini we can still emit the clusters but no names.
        return {cid: "unknown" for cid in cluster_portraits}

    cast_block = CastBibleService.format_for_prompt(
        type("B", (), {"members": cast_members})()  # tiny shim
    )
    name_set = {m.name.strip() for m in cast_members if m.name and m.name.strip()}

    sorted_clusters = sorted(cluster_portraits.items(), key=lambda pair: pair[0])

    genai.configure(api_key=settings.gemini_api_key)
    model_name = (settings.gemini_model or "gemini-2.5-flash").strip()
    if model_name in {"gemini-2.0-flash", "gemini-2.0-flash-exp"}:
        model_name = "gemini-2.5-flash"

    # Maximally permissive safety settings for this call: the inputs are
    # face crops (no text content), and the task is character ID against
    # an existing cast bible. Gemini's default safety filter blocks the
    # whole batch on a single borderline face, leaving us with all
    # clusters = unknown. Override to BLOCK_NONE on the four
    # categories that the SDK exposes.
    safety_settings = []
    try:
        from google.generativeai.types import HarmCategory, HarmBlockThreshold  # type: ignore
        safety_settings = [
            {"category": HarmCategory.HARM_CATEGORY_HATE_SPEECH, "threshold": HarmBlockThreshold.BLOCK_NONE},
            {"category": HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT, "threshold": HarmBlockThreshold.BLOCK_NONE},
            {"category": HarmCategory.HARM_CATEGORY_HARASSMENT, "threshold": HarmBlockThreshold.BLOCK_NONE},
            {"category": HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT, "threshold": HarmBlockThreshold.BLOCK_NONE},
        ]
    except Exception:
        pass

    model = genai.GenerativeModel(model_name, safety_settings=safety_settings or None)
    gen_kwargs: dict[str, Any] = {
        "temperature": 0.2,
        "top_p": 0.9,
        "max_output_tokens": 1024,
    }
    try:
        from google.generativeai.types import ThinkingConfig  # type: ignore
        gen_kwargs["thinking_config"] = ThinkingConfig(thinking_budget=0)
    except Exception:
        pass

    cluster_to_name: dict[int, str] = {cid: "unknown" for cid in cluster_portraits}
    # Process clusters one at a time. Gemini's safety filter blocks
    # multi-image batches on a single suspicious crop, but per-image
    # calls only lose the one bad crop; the rest of the clusters still
    # get described and matched.
    for cluster_id, portrait_path in sorted_clusters:
        try:
            img = Image.open(portrait_path)
        except Exception:
            continue
        single_prompt = (
            "Briefly describe this manga character drawing's visual features for an "
            "art-style cataloging task: hair color, hair style/length, eye color, "
            "distinctive accessories (glasses, horns, headgear), clothing color or "
            "uniform style. Keep under 25 words. No identification, just appearance."
        )
        try:
            response = model.generate_content(
                [single_prompt, img],
                generation_config=genai.types.GenerationConfig(**gen_kwargs),
            )
            description = (getattr(response, "text", "") or "").strip()
        except Exception as exc:  # noqa: BLE001
            logger.debug("Cluster %d describe failed (%s); marking unknown.", cluster_id, exc)
            continue
        if not description:
            continue
        matched = _match_description_to_cast(description, cast_members)
        cluster_to_name[cluster_id] = matched
        logger.debug("cluster %d: %r -> %s", cluster_id, description[:60], matched)
    return cluster_to_name


def load_panel_hint_index(project_dir: Path) -> dict[str, list[str]]:
    """Return the panel_id -> [cast_names] map persisted by this service.

    Empty dict if the index doesn't exist or is unreadable. Used by the
    panel vision narrator to pre-populate character_hints.
    """
    path = project_dir / "output" / "character_identity" / _INDEX_FILENAME
    if not path.exists():
        return {}
    try:
        payload = read_json(path, default={}) or {}
    except Exception:
        return {}
    if not isinstance(payload, dict):
        return {}
    pti = payload.get("panel_to_names") or {}
    if not isinstance(pti, dict):
        return {}
    return {str(k): [str(n) for n in (v or []) if str(n).strip()] for k, v in pti.items()}


def load_character_portraits(project_dir: Path) -> dict[str, Path]:
    """Return {cast_name: absolute portrait jpg path} for all named cast members.

    Used by panel_vision_narrator to attach visual references to per-panel
    Gemini Vision calls. Empty dict when no portraits exist.
    """
    portraits_dir = project_dir / "output" / "character_identity" / _PORTRAIT_DIRNAME
    if not portraits_dir.exists():
        return {}
    index_path = project_dir / "output" / "character_identity" / _INDEX_FILENAME
    payload = read_json(index_path, default={}) if index_path.exists() else {}
    if not isinstance(payload, dict):
        return {}
    portrait_paths = payload.get("portrait_paths") or {}
    out: dict[str, Path] = {}
    if isinstance(portrait_paths, dict):
        for name, rel_path in portrait_paths.items():
            if not name or not rel_path:
                continue
            full = project_dir / str(rel_path)
            if full.exists():
                out[str(name)] = full
    return out


def load_character_embeddings(project_dir: Path) -> dict[str, np.ndarray]:
    """Return {cast_name: 512-dim float32 CLIP embedding} from disk.

    Empty dict if the npz file doesn't exist or is unreadable. The
    embeddings are produced by build_character_identity and used by
    the per-panel verification step to confirm a named character's
    face is actually present.
    """
    path = project_dir / "output" / "character_identity" / "embeddings.npz"
    if not path.exists():
        return {}
    try:
        loaded = np.load(path)
        return {str(name): np.asarray(loaded[name], dtype=np.float32) for name in loaded.files}
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not load character embeddings: %s", exc)
        return {}
