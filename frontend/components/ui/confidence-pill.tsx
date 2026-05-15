"use client";

/**
 * ConfidencePill — ambient feedback for a panel's narration.
 *
 * Looks at panel.narration_source + panel.review_flags and produces a
 * single, glanceable indicator the user can scan dozens of at a time.
 *
 *   ● high · vision narrator        (mint, no animation)
 *   ● needs review · safety blocked (amber, soft pulse)
 *   ● failed · regenerate           (rose)
 *   ● edited by you                 (no color, just outline)
 *
 * Pair with the `confidenceEdge()` helper to add a left-border glow on
 * the parent card. Same data, two surfaces: ambient (the card edge) and
 * explicit (this pill).
 */

import * as React from "react";
import type { PanelBox } from "@/lib/types";

export type Confidence =
  | "ok"
  | "warn"
  | "fail"
  | "manual"
  | "nsfw-blur"
  | "nsfw-skip"
  | "unknown";

interface Resolved {
  level: Confidence;
  label: string;
  /** Short reason — shown on hover via title attribute. */
  reason?: string;
}

/**
 * Resolve a panel into its confidence state using narration_source +
 * review_flags + content_rating. This is the single source of truth —
 * every UI surface that displays panel confidence should go through here.
 */
export function resolveConfidence(
  panel: Pick<
    PanelBox,
    | "narration_source"
    | "review_flags"
    | "narration"
    | "content_rating"
    | "content_rating_reason"
    | "content_blur"
    | "keep"
  >,
): Resolved {
  const flags = panel.review_flags ?? [];
  const source = (panel.narration_source ?? "") as string;
  const visionFlag = flags.find((f) => typeof f === "string" && f.startsWith("vision_"));
  const nsfwFlag = flags.find((f) => typeof f === "string" && f.startsWith("nsfw_"));

  // ── Content safety wins over narration status: a flagged panel needs
  // visible feedback even if the narration succeeded. Borderline = blur,
  // explicit = skipped from final video.
  if (panel.content_rating === "explicit" || (nsfwFlag && nsfwFlag.includes("explicit"))) {
    return {
      level: "nsfw-skip",
      label: panel.keep === false ? "Skipped (explicit)" : "Forced (will blur)",
      reason:
        (panel.content_rating_reason as string | null | undefined) ||
        nsfwFlag ||
        "Flagged as explicit — excluded from the final video.",
    };
  }
  if (panel.content_rating === "borderline" || (nsfwFlag && nsfwFlag.includes("borderline"))) {
    return {
      level: "nsfw-blur",
      label: "Will blur",
      reason:
        (panel.content_rating_reason as string | null | undefined) ||
        nsfwFlag ||
        "Flagged as borderline — the final video will blur this panel.",
    };
  }

  if (source === "vision_failed" || (visionFlag && visionFlag.includes("failed"))) {
    return {
      level: "fail",
      label: "Regenerate",
      reason: visionFlag || "Vision narration failed for this panel.",
    };
  }
  if (
    source === "vision_needs_regenerate" ||
    (visionFlag && visionFlag.includes("needs_regenerate"))
  ) {
    return {
      level: "warn",
      label: "Needs review",
      reason: visionFlag || "Auto-narration was uncertain on this panel.",
    };
  }
  if (source === "manual") {
    return {
      level: "manual",
      label: "Edited by you",
    };
  }
  if (source === "panel_vision_narrator") {
    return {
      level: "ok",
      label: "Vision narrator",
    };
  }
  if (
    source === "ocr" ||
    source === "vision_caption" ||
    source === "fallback" ||
    source === "aligned_visual_order" ||
    source === "aligned_to_visual_order" ||
    source === "restored_backup_20260430" ||
    source === "restored_and_generated"
  ) {
    return {
      level: "ok",
      label: "Auto",
    };
  }
  if (!panel.narration || !panel.narration.trim()) {
    return {
      level: "warn",
      label: "No narration",
      reason: "This panel has no narration yet.",
    };
  }
  return { level: "unknown", label: "Unknown source" };
}

interface ConfidencePillProps {
  panel: Pick<
    PanelBox,
    | "narration_source"
    | "review_flags"
    | "narration"
    | "content_rating"
    | "content_rating_reason"
    | "content_blur"
    | "keep"
  >;
  className?: string;
  /** Hide the dot ornament — useful in dense lists. */
  compact?: boolean;
}

export function ConfidencePill({ panel, className, compact }: ConfidencePillProps) {
  const c = resolveConfidence(panel);
  const pillClass = (() => {
    switch (c.level) {
      case "ok":         return "p-pill p-pill-ok";
      case "warn":       return "p-pill p-pill-warn";
      case "fail":       return "p-pill p-pill-fail";
      case "manual":     return "p-pill p-pill-info";
      // Both NSFW states render rose — the action ("Will blur" vs "Skipped")
      // is in the label text, the color signals "this is monetization
      // risk territory" regardless.
      case "nsfw-blur":
      case "nsfw-skip":  return "p-pill p-pill-fail";
      default:           return "p-pill";
    }
  })();
  return (
    <span className={`${pillClass} ${className ?? ""}`} title={c.reason}>
      {!compact && (
        <span
          className={[
            "inline-block h-1.5 w-1.5 rounded-full bg-current",
            c.level === "warn" || c.level === "fail" ? "p-anim-breathe" : "",
          ].join(" ")}
        />
      )}
      {c.label}
    </span>
  );
}

/**
 * Helper to compute the matching card-edge class for a panel.
 * Use on the wrapper card so the panel "glows" with its confidence color.
 * Combine with `.p-glass`.
 */
export function confidenceEdge(
  panel: Pick<
    PanelBox,
    | "narration_source"
    | "review_flags"
    | "narration"
    | "content_rating"
    | "content_rating_reason"
    | "content_blur"
    | "keep"
  >,
): string {
  const c = resolveConfidence(panel);
  switch (c.level) {
    case "ok":         return "p-edge-ok";
    case "warn":       return "p-edge-warn";
    case "fail":       return "p-edge-fail";
    case "manual":     return "p-edge-info";
    case "nsfw-blur":
    case "nsfw-skip":  return "p-edge-fail";
    default:           return "";
  }
}
