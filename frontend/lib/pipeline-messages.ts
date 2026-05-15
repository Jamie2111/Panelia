/**
 * pipeline-messages.ts
 *
 * Converts opaque numeric pipeline state into the human-readable sentences
 * that the design language demands. The rule is simple:
 *
 *   "97% — Pass 1: repaired 15/399"  ❌
 *   "Polishing your script — about a minute left"  ✓
 *
 * Why this exists: percentages without context cause anxiety. Sentences
 * with a verb and a timeframe activate the same trust mechanism as a
 * progress callout from a thoughtful human. (Reference: narrative
 * transportation theory; Zeigarnik effect on unfinished tasks.)
 *
 * Every status displayed in the UI should pass through this helper.
 */

import type {
  PipelineStage,
  StageState,
  StageStatus,
} from "@/lib/types";

/** What the UI should show for a stage at a glance. */
export interface PipelineDisplay {
  /** A short past/present tense sentence — never a percentage. */
  sentence: string;
  /** Optional secondary line — only shown when expanded. */
  detail?: string;
  /** Pill tone — "ok" | "warn" | "fail" | "info" | "accent" | "muted". */
  tone: "ok" | "warn" | "fail" | "info" | "accent" | "muted";
  /** True when stage is actively running (drives shimmer/breathe motion). */
  active: boolean;
}

const STAGE_LABELS: Record<PipelineStage, string> = {
  ingestion: "pages",
  panel_detection: "panel detection",
  panel_review: "panel review",
  character_review: "character review",
  character_portrait: "character portraits",
  panel_vision_extraction: "panel vision",
  panel_vision_quality: "vision quality",
  script_generation: "script",
  narration_generation: "audio",
  video_rendering: "video",
  youtube_bundle: "publish bundle",
};

/**
 * Friendly verb for each stage — what's _happening_ when it's running.
 * Each entry is a present-tense gerund phrase you can drop after "We're".
 */
const STAGE_RUNNING_VERBS: Record<PipelineStage, string> = {
  ingestion: "preparing your pages",
  panel_detection: "finding panels on every page",
  panel_review: "saving your panel review",
  character_review: "saving your character review",
  character_portrait: "rendering character portraits",
  panel_vision_extraction: "looking at every panel",
  panel_vision_quality: "double-checking the vision pass",
  script_generation: "writing your script",
  narration_generation: "recording the narration",
  video_rendering: "rendering your video",
  youtube_bundle: "writing your title, description, and thumbnail",
};

/** Verb used after the stage completes. Past tense, single line. */
const STAGE_DONE_VERBS: Record<PipelineStage, string> = {
  ingestion: "Pages are ready.",
  panel_detection: "Panels are detected.",
  panel_review: "Panel review is saved.",
  character_review: "Character review is saved.",
  character_portrait: "Character portraits are ready.",
  panel_vision_extraction: "Every panel has been seen.",
  panel_vision_quality: "Vision pass is verified.",
  script_generation: "Your script is ready.",
  narration_generation: "Your narration is ready.",
  video_rendering: "Your video is ready.",
  youtube_bundle: "Your YouTube bundle is ready to publish.",
};

/**
 * Stages that the vision pipeline doesn't run — they're auto-completed
 * by the backend with a "skipped" message. Hide them from the canonical
 * pipeline display so the user never has to think about them.
 */
export const LEGACY_STAGES_HIDDEN_IN_VISION: ReadonlySet<PipelineStage> = new Set([
  "character_review",
  "character_portrait",
  "panel_vision_extraction",
  "panel_vision_quality",
]);

/**
 * The canonical visible-stage order for a vision-mode project.
 * Use this when rendering the pipeline header so the layout always
 * matches the actual six-step flow.
 */
export const VISION_STAGE_ORDER: readonly PipelineStage[] = [
  "ingestion",
  "panel_detection",
  "script_generation",
  "narration_generation",
  "video_rendering",
  "youtube_bundle",
] as const;

/** Short, friendly label for the stage chip. */
export function shortStageLabel(stage: PipelineStage): string {
  return STAGE_LABELS[stage] ?? stage;
}

/** Capitalize the first letter without touching the rest. */
function cap(s: string): string {
  return s ? s.charAt(0).toUpperCase() + s.slice(1) : s;
}

/**
 * Roughly map a percent + stage to a human-readable time estimate.
 * We deliberately avoid false precision — users prefer "about a minute"
 * over "47 seconds" for trust reasons.
 */
function estimateTimeLeft(stage: PipelineStage, progress: number): string {
  if (progress >= 99) return "almost done";
  if (progress <= 0) return "starting up";

  // Rough seconds-per-percent estimates per stage. These exist only to
  // give the user a "feel" of pacing, not a stopwatch. Tune over time.
  const SECONDS_PER_PCT: Partial<Record<PipelineStage, number>> = {
    script_generation: 12,
    narration_generation: 5,
    video_rendering: 8,
    panel_vision_extraction: 6,
  };
  const secPerPct = SECONDS_PER_PCT[stage] ?? 3;
  const remainingSec = Math.max(1, Math.round((100 - progress) * secPerPct));
  if (remainingSec < 30) return "less than a minute left";
  if (remainingSec < 90) return "about a minute left";
  const remainingMin = Math.round(remainingSec / 60);
  if (remainingMin <= 4) return `about ${remainingMin} minutes left`;
  if (remainingMin <= 9) return "a few more minutes";
  return "a little while yet";
}

/**
 * Turn a raw StageState into the sentence form the UI displays.
 *
 * This is the function every component should use — no other place in
 * the codebase should format pipeline status text directly.
 */
export function toPipelineDisplay(state: StageState): PipelineDisplay {
  const { stage, status, progress, message } = state;
  const stageLabel = shortStageLabel(stage);

  switch (status as StageStatus) {
    case "running": {
      const verb = STAGE_RUNNING_VERBS[stage] ?? `working on ${stageLabel}`;
      const timeLeft = estimateTimeLeft(stage, progress);
      return {
        sentence: `${cap(verb)} — ${timeLeft}`,
        detail: message || undefined,
        tone: "accent",
        active: true,
      };
    }
    case "completed": {
      return {
        sentence: STAGE_DONE_VERBS[stage] ?? `${cap(stageLabel)} is done.`,
        detail: message || undefined,
        tone: "ok",
        active: false,
      };
    }
    case "failed": {
      return {
        sentence: `${cap(stageLabel)} ran into a problem.`,
        detail: message || undefined,
        tone: "fail",
        active: false,
      };
    }
    case "cancelled": {
      return {
        sentence: `${cap(stageLabel)} was cancelled.`,
        detail: message || undefined,
        tone: "muted",
        active: false,
      };
    }
    case "needs_review": {
      return {
        sentence: `${cap(stageLabel)} needs your review.`,
        detail: message || undefined,
        tone: "warn",
        active: false,
      };
    }
    case "ready": {
      return {
        sentence: `${cap(stageLabel)} is ready when you are.`,
        detail: message || undefined,
        tone: "info",
        active: false,
      };
    }
    case "pending":
    default: {
      return {
        sentence: `${cap(stageLabel)} is up next.`,
        detail: undefined,
        tone: "muted",
        active: false,
      };
    }
  }
}

/**
 * Pick the single most actionable stage from the full pipeline.
 * This drives the "what should I do next?" callout at the top of the
 * project page — the answer is always exactly one thing.
 */
export function pickFocusStage(
  states: Record<PipelineStage, StageState> | undefined,
): { stage: PipelineStage; display: PipelineDisplay } | null {
  if (!states) return null;
  // Prefer running > failed > needs_review > ready > pending, in stage order.
  const orderedStages: PipelineStage[] = [
    "ingestion",
    "panel_detection",
    "panel_review",
    "character_review",
    "character_portrait",
    "panel_vision_extraction",
    "panel_vision_quality",
    "script_generation",
    "narration_generation",
    "video_rendering",
    "youtube_bundle",
  ];
  const priority: Record<StageStatus, number> = {
    running: 0,
    failed: 1,
    needs_review: 2,
    ready: 3,
    pending: 4,
    completed: 5,
    cancelled: 5,
  };
  let best: { stage: PipelineStage; rank: number } | null = null;
  for (const stage of orderedStages) {
    const state = states[stage];
    if (!state) continue;
    const rank = priority[state.status as StageStatus] ?? 99;
    if (best === null || rank < best.rank) {
      best = { stage, rank };
    }
  }
  if (best === null) return null;
  const state = states[best.stage];
  return { stage: best.stage, display: toPipelineDisplay(state) };
}
