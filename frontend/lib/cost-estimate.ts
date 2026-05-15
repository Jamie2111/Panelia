/**
 * cost-estimate.ts
 *
 * Single source of truth for "how much will this project cost to complete?"
 * The numbers are kept conservative and slightly pessimistic - better to
 * show $0.18 and bill the user $0.05 than the inverse.
 *
 * Cost components we track:
 *   • Vision narration (Gemini 2.5 Flash multimodal)
 *   • TTS  (Kokoro is local → $0; cloud TTS would slot in here)
 *   • Music (built-in tracks → $0; licensed tracks would slot in here)
 *   • Render (FFmpeg local → $0; cloud render would slot in here)
 *
 * All prices are USD. Update GEMINI_PRICING when Google revises rates.
 */

import type { PanelBox, ProjectDetail, StageState } from "@/lib/types";

// Gemini 2.5 Flash pricing as of 2025-2026.
// Multimodal input includes image tokens; we use a flat per-panel estimate
// rather than counting tokens because images dominate and counting per-image
// tokens server-side adds latency without changing the result much.
const GEMINI_PRICING = {
  /** Estimated cost in USD per panel narration call. */
  perPanel: 0.00018,
  /** Cost when a panel needs regenerate via /regenerate-panel-vision. */
  perRegenerate: 0.00018,
} as const;

// Future-proofing slots for paid services. Keep at 0 while local-only.
const TTS_PER_PANEL_USD = 0;        // Kokoro = free local
const MUSIC_PER_PROJECT_USD = 0;    // Built-in tracks = free
const RENDER_PER_MINUTE_USD = 0;    // FFmpeg local = free

export interface CostBreakdown {
  /** Total estimated USD to complete the project from its current state. */
  remainingUsd: number;
  /** Total estimated USD if the project were generated from scratch. */
  fullProjectUsd: number;
  /** Per-component breakdown for the tooltip / details view. */
  parts: {
    label: string;
    usd: number;
    note?: string;
  }[];
  /** Friendly one-line summary suitable for the pipeline block. */
  sentence: string;
}

function isVisionNarrated(panel: PanelBox): boolean {
  return panel.narration_source === "panel_vision_narrator";
}

function isFlagged(panel: PanelBox): boolean {
  const flags = panel.review_flags ?? [];
  return flags.some((f) => typeof f === "string" && f.startsWith("vision_"));
}

function isStageComplete(state: StageState | undefined): boolean {
  return Boolean(state && state.status === "completed");
}

/**
 * Estimate the cost to complete a project from its current state.
 *
 * "Remaining" means: panels not yet vision-narrated (or flagged for retry),
 * plus stages that haven't run yet. The full-project cost is what you'd
 * pay to run everything end-to-end on a fresh project of the same size.
 */
export function estimateProjectCost(
  project: Pick<ProjectDetail, "panels" | "stage_states" | "kept_panel_count">,
): CostBreakdown {
  const panels = project.panels ?? [];
  const kept = panels.filter((p) => p.keep);
  const totalKept = kept.length || project.kept_panel_count || 0;
  const narrated = kept.filter(isVisionNarrated).length;
  const flagged = kept.filter(isFlagged).length;

  // Panels still needing first-pass narration.
  const unnarrated = Math.max(0, totalKept - narrated);
  // Panels we expect the user to retry via the regen endpoint.
  const willRetry = flagged;

  const narrationFullCost = totalKept * GEMINI_PRICING.perPanel;
  const narrationRemainingCost =
    unnarrated * GEMINI_PRICING.perPanel +
    willRetry * GEMINI_PRICING.perRegenerate;

  // TTS / music / render run after script.
  const scriptDone = isStageComplete(project.stage_states?.script_generation);
  const audioDone = isStageComplete(project.stage_states?.narration_generation);
  const videoDone = isStageComplete(project.stage_states?.video_rendering);

  const ttsRemaining = audioDone ? 0 : totalKept * TTS_PER_PANEL_USD;
  const musicRemaining = audioDone ? 0 : MUSIC_PER_PROJECT_USD;
  // Estimate video minutes from total kept-panel duration (default 4.5s).
  const estVideoMinutes = (totalKept * 4.5) / 60;
  const renderRemaining = videoDone ? 0 : estVideoMinutes * RENDER_PER_MINUTE_USD;

  const remainingUsd =
    (scriptDone ? 0 : narrationRemainingCost) +
    ttsRemaining +
    musicRemaining +
    renderRemaining;

  const fullProjectUsd =
    narrationFullCost +
    totalKept * TTS_PER_PANEL_USD +
    MUSIC_PER_PROJECT_USD +
    estVideoMinutes * RENDER_PER_MINUTE_USD;

  const parts: CostBreakdown["parts"] = [
    {
      label: "Vision narration",
      usd: scriptDone ? 0 : narrationRemainingCost,
      note: scriptDone
        ? `${narrated} of ${totalKept} panels already narrated`
        : `${unnarrated} panels to narrate${willRetry ? ` + ${willRetry} to retry` : ""}`,
    },
    {
      label: "Voice (Kokoro, local)",
      usd: ttsRemaining,
      note: "Local synthesis - free",
    },
    {
      label: "Music",
      usd: musicRemaining,
      note: "Built-in tracks - free",
    },
    {
      label: "Render",
      usd: renderRemaining,
      note: "Local FFmpeg - free",
    },
  ];

  const sentence = formatCostSentence(remainingUsd, fullProjectUsd);

  return { remainingUsd, fullProjectUsd, parts, sentence };
}

/** Format a USD figure with sensible precision. */
export function formatUsd(n: number): string {
  if (n <= 0) return "$0.00";
  if (n < 0.01) return "<$0.01";
  if (n < 1) return `$${n.toFixed(2)}`;
  if (n < 10) return `$${n.toFixed(2)}`;
  return `$${n.toFixed(0)}`;
}

function formatCostSentence(remaining: number, full: number): string {
  if (remaining <= 0) {
    return `Already paid - full run would have cost about ${formatUsd(full)}.`;
  }
  if (remaining < 0.05) {
    return `About ${formatUsd(remaining)} to finish - basically free.`;
  }
  if (remaining < 1) {
    return `About ${formatUsd(remaining)} in API calls to finish.`;
  }
  return `Roughly ${formatUsd(remaining)} to complete this project.`;
}
