"""
PanelVisionNarrator - Vision-grounded panel narration service.

This is the SINGLE source of truth for generating panel narrations.
Replaces the legacy cascade (script_generator → script_polisher →
script_quality_service → script_cleaner_service → script_narrative_polish
→ story_script_service → story_segment_repair_service).

Design principles:
  • Vision-first: every narration is generated with the panel IMAGE in context.
  • Sequential with continuity: each call carries the prior N narrations.
  • One pass, one source: no polish/repair cascade - bad output gets
    regenerated per-panel by the user, never batch-polished.
  • Schema-stable: writes a single canonical script_manifest.json plus
    syncs panel.narration in panels.json. Nothing else.
  • Fail loud: if a panel can't be narrated, the slot is marked
    "needs_regenerate" with a reason, not silently filled with garbage.
"""

from __future__ import annotations

import asyncio
import io
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from PIL import Image

from app.core.config import get_settings
from app.utils.files import write_json

try:
    import google.generativeai as genai
    _GEMINI_AVAILABLE = True
except ImportError:
    _GEMINI_AVAILABLE = False

logger = logging.getLogger(__name__)


# ── Tuning constants ──────────────────────────────────────────────────────
# Number of preceding narrations to include as continuity context.
_CONTEXT_WINDOW = 4
# Max concurrent Gemini calls (Gemini Flash tier handles ~60 RPS safely).
_MAX_CONCURRENCY = 8
# Per-panel timeout (seconds). Vision calls can spike.
_PER_PANEL_TIMEOUT = 30.0
# Max image edge (Gemini Vision auto-resizes but smaller = faster + cheaper).
_MAX_IMAGE_EDGE = 1024


@dataclass
class PanelInput:
    """One panel's data for narration."""
    panel_id: str
    order: int
    page: int
    panel: int
    image_path: Path
    ocr_text: str = ""
    character_hints: list[str] = field(default_factory=list)


@dataclass
class NarrationResult:
    """Output for one panel."""
    panel_id: str
    narration: str
    status: str  # "ok" | "needs_regenerate" | "failed"
    reason: str = ""
    duration_seconds: float = 0.0
    # ── Content-safety classification ────────────────────────────────────
    # `rating` is one of "safe" | "borderline" | "explicit". Populated by
    # the vision call when content_safety_enabled is True on the project.
    # The downstream writer translates this into panel.keep / content_blur.
    rating: str = "safe"
    rating_reason: str = ""


@dataclass
class NarrationBatch:
    """Result of narrating a full chapter."""
    results: list[NarrationResult]
    elapsed_seconds: float
    successful: int
    failed: int


# ── The single narration prompt ───────────────────────────────────────────
# This is the ONLY prompt in the whole pipeline. If output quality is bad,
# we tune this - we don't add another service.
_NARRATION_PROMPT = """You are narrating a single panel from a sequential-art chapter
(manga, manhwa, webtoon, or comic) for a YouTube recap video. You are ALSO
classifying the panel for YouTube monetization safety.

You can see the panel image. Look at it carefully.

PART 1 - NARRATION (15-30 words):
Describe the SPECIFIC visible action, expression, or moment in this panel.
The narration will be read aloud over this exact panel.

GOOD narrations:
• Specific to what's visible in THIS panel
• Character names: use a name from the KNOWN CAST ONLY when this panel
  shows a clear, unambiguous match. Required: face visible at a normal
  angle, with the cast's distinguishing visible features (hair color,
  eye color, signature outfit) clearly readable. If the angle is partial
  (3/4 profile, back of head, distant shot, face cropped or obscured,
  shadowed, motion-blurred, or rendered as a silhouette), DO NOT GUESS
  a name - use a neutral descriptor instead ("the pilot", "the boy
  with dark hair", "a uniformed figure"). It is much better to under-
  identify than to wrongly call a side-profile of one character by
  another character's name. Repeat: when in doubt, no name.
• Flow naturally from the previous narration (continuity, not repetition)
• Convey emotion when the panel shows emotion
• Past or present tense - match the established voice

AVOID:
• Generic filler ("a powerful moment unfolds", "tension rises")
• Repeating the previous narration's wording with a tense change
• Inventing details not visible in the panel
• Describing what's about to happen - describe THIS panel only
• Describing visible sound effects or onomatopoeia. NEVER write things
  like 'A massive "GWOOO!" reverberates', 'The sound of "BOOM!"
  echoes', 'A loud "ZAP!" rings out'. The narration is read by a
  TTS voice; it pronounces those tokens as random letters. If a
  panel's main content is a stylized SFX, describe the IMPACT in the
  story (who, what, the emotional beat) instead.

PART 2 - CONTENT RATING (one of: "safe", "borderline", "explicit"):
Classify by YouTube's Advertiser-Friendly Content Guidelines.

"safe" - render normally. Includes:
  • Kissing, romantic embraces, hand-holding, hugging
  • Characters in swimwear or normal revealing clothes
  • Combat, fighting, action sequences, impact effects
  • Mild blood (a punch drawing blood, a scratch, a bruise)
  • Crying, anger, surprise, fear, any facial expression
  • Suggestive poses where characters are fully clothed
  → Be GENEROUS with "safe". Most romance and action belongs here.

"borderline" - blur in the final video. Includes:
  • Partial / implied nudity (silhouettes, covered nudity, side- or
    back-nudity NOT shown graphically)
  • Bed / intimate scenes past kissing (touching, undressing partially)
    where anatomy isn't directly visible
  • Heavier blood spray, deep wounds shown, broken bones, bruises
    covering most of a body
  • Stylized horror imagery that isn't viscera

"explicit" - skip the panel entirely. Includes:
  • Visible genitalia, exposed nipples, fully nude bodies shown directly
  • Exposed buttocks shown as the panel's focal subject (a clear glute
    close-up, not an incidental side angle of a clothed character)
  • On-panel depiction of sex acts, simulated or otherwise
  • Crotch-focused or panty-shot close-ups where that is the panel's
    main subject (incidental short skirts in action poses stay "safe";
    a deliberately-framed underwear close-up does not)
  • Lingerie / undressed shots that linger on a single character's
    body as the panel's focal subject (not a quick action beat)
  • Decapitations, dismemberment, exposed viscera, graphic body horror
  • Torture, mutilation, or self-harm shown explicitly

When you're on the fence between "borderline" and "explicit", default
to "explicit". Demonetization risk is asymmetric: a falsely-flagged
borderline panel just gets blurred; a missed explicit panel kills the
entire video's monetization. Be cautious.

If the panel is borderline OR explicit, give a short factual reason
(under 12 words) explaining what triggered the rating, so the user can
review.

{cast_block}CONTINUITY CONTEXT (last few narrations leading into this panel):
{context}

THIS PANEL:
• Position: page {page}, panel {panel_num} of the chapter
• Detected text in the panel (may be incomplete or noisy): {ocr_text}
{character_hint_line}

Return ONLY a JSON object on a single line with these exact keys:
{{"narration": "...", "rating": "safe|borderline|explicit", "rating_reason": "..."}}
No prose before or after. No code fences. No quotes around your answer
beyond what JSON requires."""


class PanelVisionNarrator:
    """Vision-grounded narration engine. The only one you need."""

    def __init__(self, settings: Any = None):
        self.settings = settings or get_settings()
        if not _GEMINI_AVAILABLE:
            raise RuntimeError(
                "google-generativeai is not installed. "
                "Run: pip install google-generativeai"
            )
        if not self.settings.gemini_api_key:
            raise RuntimeError(
                "GEMINI_API_KEY is not configured. "
                "Set it in .env to enable PanelVisionNarrator."
            )
        genai.configure(api_key=self.settings.gemini_api_key)
        # Primary model + fallback. When safety filtering blocks the primary
        # model on a specific panel (common for fight scenes / dramatic art),
        # we retry once with a different model that has different safety
        # priors. This recovers the long tail of panels at trivial cost.
        self._model_name = self._select_vision_model()
        self._model = genai.GenerativeModel(self._model_name)
        fallback_name = self._select_fallback_model(self._model_name)
        self._fallback_name = fallback_name
        self._fallback_model = (
            genai.GenerativeModel(fallback_name)
            if fallback_name and fallback_name != self._model_name
            else None
        )
        logger.info(
            "PanelVisionNarrator initialized with model=%s (fallback=%s)",
            self._model_name, self._fallback_name or "none",
        )

    def _select_fallback_model(self, primary: str) -> str | None:
        # Pick a vision-capable alternate that differs from the primary so
        # safety-filter recoveries actually use a different policy.
        # Order picked for: current availability on free tier, multimodal
        # support, and divergent safety priors from gemini-2.5-flash.
        candidates = [
            "gemini-2.5-flash-lite",
            "gemini-2.5-pro",
            "gemini-1.5-flash",
            "gemini-1.5-pro",
        ]
        for name in candidates:
            if name != primary:
                return name
        return None

    def _select_vision_model(self) -> str:
        configured = (self.settings.gemini_model or "").strip()
        # Retired or unavailable model identifiers - auto-upgrade to current.
        retired = {
            "gemini-2.0-flash",
            "gemini-2.0-flash-exp",
            "gemini-pro-vision",
        }
        # Current vision-capable model identifiers.
        vision_models = {
            "gemini-2.5-flash",
            "gemini-2.5-flash-lite",
            "gemini-2.5-pro",
            "gemini-1.5-flash",
            "gemini-1.5-flash-8b",
            "gemini-1.5-pro",
        }
        if configured in retired:
            upgraded = "gemini-2.5-flash"
            logger.warning(
                "Configured Gemini model %r is retired; using %r instead.",
                configured, upgraded,
            )
            return upgraded
        if configured in vision_models:
            return configured
        # Conservative default.
        return "gemini-2.5-flash"

    # ── Public API ────────────────────────────────────────────────────────

    async def narrate_chapter(
        self,
        panels: list[PanelInput],
        *,
        cast_block: str = "",
        progress_callback: Callable[[float, str], None] | None = None,
        cancel_callback: Callable[[], None] | None = None,
    ) -> NarrationBatch:
        """Narrate a full ordered list of panels with continuity.

        Panels MUST be supplied in visual reading order (sorted by page,
        then by panel-within-page). When `cast_block` is non-empty it is
        prepended to every panel's prompt - that's how the cast bible
        threads into character recognition.
        """
        started = time.perf_counter()
        results: list[NarrationResult] = [None] * len(panels)  # type: ignore[list-item]
        cast_prefix = (cast_block + "\n\n") if cast_block else ""

        # Build a rolling context window. To keep continuity meaningful we
        # process in small sequential chunks of size _MAX_CONCURRENCY: each
        # chunk runs in parallel, but later chunks see earlier ones' output.
        semaphore = asyncio.Semaphore(_MAX_CONCURRENCY)

        for chunk_start in range(0, len(panels), _MAX_CONCURRENCY):
            if cancel_callback:
                cancel_callback()
            chunk_end = min(chunk_start + _MAX_CONCURRENCY, len(panels))
            chunk = panels[chunk_start:chunk_end]

            # Context = last _CONTEXT_WINDOW completed narrations.
            context_lines = [
                results[i].narration
                for i in range(max(0, chunk_start - _CONTEXT_WINDOW), chunk_start)
                if results[i] is not None and results[i].status == "ok"
            ]
            context_str = "\n".join(f"  • {line}" for line in context_lines) or "  (this is the opening panel)"

            tasks = [
                self._narrate_one(panel, context_str, semaphore, cast_prefix=cast_prefix)
                for panel in chunk
            ]
            chunk_results = await asyncio.gather(*tasks, return_exceptions=False)

            for offset, result in enumerate(chunk_results):
                results[chunk_start + offset] = result

            done = chunk_end
            if progress_callback:
                pct = 100.0 * done / max(len(panels), 1)
                progress_callback(pct, f"Narrated {done}/{len(panels)} panels")

        elapsed = time.perf_counter() - started
        successful = sum(1 for r in results if r and r.status == "ok")
        failed = len(panels) - successful

        return NarrationBatch(
            results=list(results),
            elapsed_seconds=elapsed,
            successful=successful,
            failed=failed,
        )

    # ── Internals ─────────────────────────────────────────────────────────

    async def _narrate_one(
        self,
        panel: PanelInput,
        context_str: str,
        semaphore: asyncio.Semaphore,
        cast_prefix: str = "",
    ) -> NarrationResult:
        async with semaphore:
            start = time.perf_counter()
            try:
                # Load + optionally downscale the panel image. Gemini accepts
                # PIL images directly via the SDK.
                if not panel.image_path.exists():
                    return NarrationResult(
                        panel_id=panel.panel_id,
                        narration="",
                        status="failed",
                        reason=f"Image not found: {panel.image_path.name}",
                        duration_seconds=time.perf_counter() - start,
                    )
                image = self._load_image(panel.image_path)

                char_hint_line = ""
                if panel.character_hints:
                    char_hint_line = (
                        f"• Likely on-panel characters: {', '.join(panel.character_hints)}"
                    )

                prompt = _NARRATION_PROMPT.format(
                    cast_block=cast_prefix,
                    context=context_str,
                    page=panel.page,
                    panel_num=panel.panel,
                    ocr_text=panel.ocr_text.strip() or "(none detected)",
                    character_hint_line=char_hint_line,
                )

                # Run the blocking SDK call in a thread to keep the event
                # loop responsive across concurrent panels.
                try:
                    response_text = await asyncio.wait_for(
                        asyncio.to_thread(self._invoke_gemini, prompt, image, self._model),
                        timeout=_PER_PANEL_TIMEOUT,
                    )
                except RuntimeError as primary_err:
                    # Safety-filter blocks (block_reason=SAFETY) and similar
                    # policy rejections often pass on a different model with
                    # different safety priors. Try the fallback once.
                    err_text = str(primary_err)
                    is_safety = "blocked" in err_text.lower() or "finish_reason=2" in err_text
                    if not (is_safety and self._fallback_model is not None):
                        raise
                    logger.info(
                        "Panel %s primary model blocked (%s); trying fallback %s",
                        panel.panel_id, err_text[:80], self._fallback_name,
                    )
                    response_text = await asyncio.wait_for(
                        asyncio.to_thread(
                            self._invoke_gemini, prompt, image, self._fallback_model
                        ),
                        timeout=_PER_PANEL_TIMEOUT,
                    )

                cleaned, rating, rating_reason = self._parse_response(response_text)
                if not cleaned or len(cleaned.split()) < 4:
                    return NarrationResult(
                        panel_id=panel.panel_id,
                        narration=cleaned,
                        status="needs_regenerate",
                        reason="Output too short or empty",
                        duration_seconds=time.perf_counter() - start,
                        rating=rating,
                        rating_reason=rating_reason,
                    )

                return NarrationResult(
                    panel_id=panel.panel_id,
                    narration=cleaned,
                    status="ok",
                    duration_seconds=time.perf_counter() - start,
                    rating=rating,
                    rating_reason=rating_reason,
                )

            except asyncio.TimeoutError:
                return NarrationResult(
                    panel_id=panel.panel_id,
                    narration="",
                    status="failed",
                    reason="Gemini call timed out",
                    duration_seconds=time.perf_counter() - start,
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Panel %s vision narration failed: %s",
                    panel.panel_id, exc,
                )
                return NarrationResult(
                    panel_id=panel.panel_id,
                    narration="",
                    status="failed",
                    reason=str(exc)[:120],
                    duration_seconds=time.perf_counter() - start,
                )

    def _load_image(self, path: Path) -> Image.Image:
        image = Image.open(path).convert("RGB")
        if max(image.size) > _MAX_IMAGE_EDGE:
            scale = _MAX_IMAGE_EDGE / max(image.size)
            new_size = (int(image.size[0] * scale), int(image.size[1] * scale))
            image = image.resize(new_size, Image.LANCZOS)
        return image

    def _invoke_gemini(self, prompt: str, image: Image.Image, model: Any = None) -> str:
        model = model or self._model
        # Relaxed safety settings: manga content can trigger harmless flags on
        # violent/dramatic panels and silently produce empty output otherwise.
        safety = [
            {"category": c, "threshold": "BLOCK_ONLY_HIGH"}
            for c in (
                "HARM_CATEGORY_HARASSMENT",
                "HARM_CATEGORY_HATE_SPEECH",
                "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                "HARM_CATEGORY_DANGEROUS_CONTENT",
            )
        ]
        # Gemini 2.5 models burn output-token budget on internal "thinking"
        # tokens by default, which truncates the visible answer. We allocate
        # a budget large enough that even with thinking we get a full line,
        # and try to disable thinking entirely where the SDK supports it.
        gen_kwargs: dict[str, Any] = {
            "temperature": 0.55,
            "top_p": 0.9,
            "max_output_tokens": 1024,
        }
        try:
            from google.generativeai.types import ThinkingConfig  # type: ignore
            gen_kwargs["thinking_config"] = ThinkingConfig(thinking_budget=0)
        except Exception:
            # Older SDK versions don't expose ThinkingConfig - the larger
            # token budget alone is enough to avoid truncation.
            pass

        # Encode the panel as WebP and hand Gemini the bytes directly
        # instead of letting the SDK serialize PIL.Image (which defaults
        # to PNG). WebP q80 cuts manga line-art panel sizes by 60-80%
        # with no perceptible quality loss, which trims upload time on
        # every vision call. The Gemini API natively accepts image/webp.
        # Falls back to handing the PIL Image to the SDK if WebP encode
        # fails for any reason (e.g. exotic mode).
        image_part: Any
        try:
            buf = io.BytesIO()
            image.save(buf, format="WEBP", quality=80, method=4)
            image_part = {"mime_type": "image/webp", "data": buf.getvalue()}
        except Exception as exc:  # noqa: BLE001
            logger.warning("WebP encode failed (%s); falling back to PIL handoff", exc)
            image_part = image

        response = model.generate_content(
            [prompt, image_part],
            generation_config=genai.types.GenerationConfig(**gen_kwargs),
            safety_settings=safety,
        )
        # `response.text` is a property that raises ValueError when the
        # candidates list is empty (e.g. when safety filtered the response).
        # We have to introspect candidates manually to avoid that.
        try:
            candidates = list(getattr(response, "candidates", None) or [])
        except Exception:
            candidates = []
        if not candidates:
            feedback = getattr(response, "prompt_feedback", None)
            block_reason = getattr(feedback, "block_reason", None) if feedback else None
            if block_reason:
                raise RuntimeError(f"Gemini blocked: {block_reason}")
            raise RuntimeError("Gemini returned no candidates")

        candidate = candidates[0]
        finish_reason = getattr(candidate, "finish_reason", None)
        # Try to extract text from candidate.content.parts manually so we
        # don't depend on the .text quick accessor (which itself raises).
        text_parts: list[str] = []
        content = getattr(candidate, "content", None)
        for part in getattr(content, "parts", None) or []:
            piece = getattr(part, "text", None)
            if piece:
                text_parts.append(piece)
        text = "".join(text_parts).strip()
        if text:
            return text
        # No usable text - surface the finish_reason so the caller's
        # post-processor flags this panel for regeneration.
        if finish_reason and str(finish_reason) not in (
            "STOP", "FinishReason.STOP", "1",
        ):
            raise RuntimeError(f"Gemini finish_reason={finish_reason}")
        return ""

    @staticmethod
    def _post_process(raw: str) -> str:
        """Strip common LLM artifacts: numbering, quotes, headers."""
        text = (raw or "").strip()
        # Remove leading numbering like "1. " or "1) "
        if text and text[0].isdigit():
            for sep in (". ", ") ", "- "):
                idx = text.find(sep)
                if 0 < idx <= 4:
                    text = text[idx + len(sep):]
                    break
        # Strip surrounding quotes
        if len(text) >= 2 and text[0] in ("'", '"', "“", "‘") and text[-1] in ("'", '"', "”", "’"):
            text = text[1:-1]
        # Collapse internal whitespace
        text = " ".join(text.split())
        # Strip trailing/leading bullets
        text = text.lstrip("•·-* ").rstrip()
        return text

    @classmethod
    def _parse_response(cls, raw: str) -> tuple[str, str, str]:
        """Parse the Gemini response into (narration, rating, rating_reason).

        The new prompt asks for a JSON object on a single line. We handle:
          1. Well-formed JSON (the happy path)
          2. JSON wrapped in code fences (```json ... ```)
          3. JSON with extra prose around it (extract via regex)
          4. Bare narration string (legacy / fallback model)

        On any parse failure we default to {rating: "safe"} so we never
        block a panel because of malformed output - the user can always
        re-classify by re-running the panel.
        """
        import json as _json
        import re as _re

        text = (raw or "").strip()
        if not text:
            return "", "safe", ""

        # Strip code fences if present.
        if text.startswith("```"):
            text = _re.sub(r"^```[a-zA-Z]*\s*", "", text)
            text = _re.sub(r"```\s*$", "", text).strip()

        # Try to locate the JSON object even if surrounded by prose.
        json_text: str | None = None
        if text.startswith("{"):
            json_text = text
        else:
            m = _re.search(r"\{[\s\S]*\}", text)
            if m:
                json_text = m.group(0)

        if json_text is not None:
            try:
                data = _json.loads(json_text)
                if isinstance(data, dict):
                    narration = cls._post_process(str(data.get("narration") or ""))
                    rating_raw = str(data.get("rating") or "safe").strip().lower()
                    if rating_raw not in {"safe", "borderline", "explicit"}:
                        rating_raw = "safe"
                    reason = str(data.get("rating_reason") or "").strip()
                    return narration, rating_raw, reason
            except (_json.JSONDecodeError, ValueError):
                # JSON broke (typically because the model put unescaped
                # inner quotes inside the narration string). Fall through
                # to the regex extractor - we'd rather salvage a good
                # narration than throw the whole panel away.
                pass

            # Regex-based salvage: pull out the narration + rating fields
            # tolerantly so malformed JSON still yields a usable result.
            narr_match = _re.search(
                r'"narration"\s*:\s*"((?:[^"\\]|\\.)*?)"\s*,\s*"rating"',
                json_text,
                _re.DOTALL,
            )
            if not narr_match:
                # Looser pattern: take everything up to the rating field
                # even if there are unescaped inner quotes.
                narr_match = _re.search(
                    r'"narration"\s*:\s*"(.*?)"\s*,\s*"rating"',
                    json_text,
                    _re.DOTALL,
                )
            rating_match = _re.search(
                r'"rating"\s*:\s*"(safe|borderline|explicit)"',
                json_text,
            )
            reason_match = _re.search(
                r'"rating_reason"\s*:\s*"((?:[^"\\]|\\.)*?)"',
                json_text,
                _re.DOTALL,
            )
            if narr_match:
                narration = cls._post_process(
                    narr_match.group(1).replace('\\"', '"').replace("\\n", " ")
                )
                rating_raw = rating_match.group(1) if rating_match else "safe"
                reason = reason_match.group(1) if reason_match else ""
                return narration, rating_raw, reason

        # Fallback: treat whole response as a bare narration line.
        return cls._post_process(text), "safe", ""


# ── Convenience: load panels from project_store, narrate, persist ─────────

def write_narration_outputs(
    project_dir: Path,
    panel_order: list[PanelInput],
    results: list[NarrationResult],
    panels_json: list[dict[str, Any]],
) -> dict[str, Any]:
    """Persist narration results to canonical project files.

    Writes:
      • panels.json - each kept panel's narration field
      • script_manifest.json - single source of truth for script_lines + segments
      • script.txt - flat plaintext for quick inspection
    """
    # Build lookup
    result_by_id = {r.panel_id: r for r in results}

    # Sync panels.json
    for panel in panels_json:
        if not panel.get("keep"):
            continue
        r = result_by_id.get(panel["id"])
        if r is None:
            continue
        panel["narration"] = r.narration if r.status == "ok" else (r.narration or "")
        panel["narration_source"] = (
            "panel_vision_narrator" if r.status == "ok" else f"vision_{r.status}"
        )
        # Drop any stale vision_* / nsfw_* flags from previous runs before
        # re-applying - that keeps the UI in lockstep with the latest pass.
        flags = [
            f for f in (panel.get("review_flags") or [])
            if not (str(f).startswith("vision_") or str(f).startswith("nsfw_"))
        ]
        if r.status != "ok":
            flag = f"vision_{r.status}: {r.reason}"
            if flag not in flags:
                flags.append(flag)

        # ── Apply content-safety rating ──────────────────────────────────
        # The vision call classified this panel. Translate that into the
        # downstream effects:
        #   safe       → unchanged
        #   borderline → kept + content_blur=True + flag
        #   explicit   → keep=False + auto_skipped + content_blur=True
        #                (so a manual force-keep still ships blurred)
        rating = (r.rating or "safe").lower()
        reason = (r.rating_reason or "").strip()
        panel["content_rating"] = rating
        panel["content_rating_reason"] = reason or None

        if rating == "borderline":
            panel["content_blur"] = True
            tag = f"nsfw_borderline: {reason}" if reason else "nsfw_borderline"
            if tag not in flags:
                flags.append(tag)
        elif rating == "explicit":
            panel["content_blur"] = True  # safety-net for force-keep
            # Only auto-skip if the user hasn't manually kept the panel.
            if not bool(panel.get("manual_keep")):
                panel["keep"] = False
                panel["auto_skipped"] = True
                panel["skip_reason"] = "nsfw_explicit"
            tag = f"nsfw_explicit: {reason}" if reason else "nsfw_explicit"
            if tag not in flags:
                flags.append(tag)
        else:
            # safe - clear any prior nsfw blur unless the user manually set it.
            if not bool(panel.get("manual_keep")):
                panel["content_blur"] = False

        panel["review_flags"] = flags

    write_json(project_dir / "panels.json", panels_json)

    # Build manifest in visual order (panel_order is already sorted)
    script_lines: list[str] = []
    story_segments: list[dict[str, Any]] = []
    for idx, panel in enumerate(panel_order):
        r = result_by_id.get(panel.panel_id)
        narration_text = r.narration if (r and r.status == "ok") else ""
        script_lines.append(narration_text)
        story_segments.append({
            "id": f"segment_{idx + 1:04d}",
            "segment_id": f"segment_{idx + 1:04d}",
            "order": idx + 1,
            "text": narration_text,
            "narration": narration_text,
            "keep": True,
            "panel_count": 1,
            "panel_ids": [panel.panel_id],
            "needs_regenerate": bool(r and r.status != "ok"),
            "regenerate_reason": (r.reason if (r and r.status != "ok") else ""),
        })

    manifest = {
        "version": "panel_vision_v1",
        "script_lines": script_lines,
        "story_segments": story_segments,
        "script_story": "\n".join(line for line in script_lines if line),
    }
    write_json(project_dir / "script_manifest.json", manifest)

    # Mirror to legacy script.json for backwards compatibility with the
    # frontend Narration tab and any API consumers that still read it.
    legacy_script = {
        "script_lines": script_lines,
        "script_lines_strict": [],
        "script_lines_cinematic": [],
        "script_story": manifest["script_story"],
        "story_segments": story_segments,
        "script_mode": "panel_vision_v1",
    }
    write_json(project_dir / "script.json", legacy_script)

    # Also keep script.txt for quick human inspection
    (project_dir / "script.txt").write_text(
        "\n".join(script_lines), encoding="utf-8"
    )

    return {
        "panels_with_narration": sum(1 for line in script_lines if line),
        "panels_needing_review": sum(
            1 for s in story_segments if s["needs_regenerate"]
        ),
        "total_segments": len(story_segments),
    }


def panels_from_store(
    project_dir: Path,
    panels_json: list[dict[str, Any]],
) -> list[PanelInput]:
    """Build the ordered PanelInput list from raw panels.json data."""
    kept = [p for p in panels_json if p.get("keep")]
    kept_sorted = sorted(
        kept, key=lambda p: (int(p.get("page", 0)), int(p.get("panel", 0)))
    )
    panel_dir = project_dir / "panels"

    inputs: list[PanelInput] = []
    for p in kept_sorted:
        order = int(p.get("order", 0))
        image_path = panel_dir / f"panel_{order:03d}.png"
        inputs.append(PanelInput(
            panel_id=str(p["id"]),
            order=order,
            page=int(p.get("page", 0)),
            panel=int(p.get("panel", 0)),
            image_path=image_path,
            ocr_text=str(p.get("ocr_text") or ""),
            character_hints=[],  # populated later if character tracker results exist
        ))
    return inputs
