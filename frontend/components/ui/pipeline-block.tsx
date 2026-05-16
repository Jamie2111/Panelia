"use client";

/**
 * PipelineBlock - the single most important UI element of the new design.
 *
 * Replaces the horizontal tab strip ("Overview / Panel Editor / Character
 * Review / ...") with a vertical, glanceable, narrative pipeline. At any
 * moment a user can answer:
 *   1. Where am I in the pipeline?
 *   2. What's happening right now?
 *   3. What should I do next?
 *
 * Visual design (matches globals.css design tokens):
 *   ◉ ── ◉ ── ◉ ── ●═══─ ○ ── ○
 *   Pages  Panels  Vision  Script  Audio  Video
 *                          ↑
 *               "Writing your script..."
 *               [ Pause ]
 *
 * Filled circles = complete. Half-filled with shimmer = running. Empty =
 * upcoming. The currently-focused stage glows mint and shows the active
 * sentence beneath it.
 */

import * as React from "react";
import type { PipelineStage, StageState, StageStatus } from "@/lib/types";
import {
  LEGACY_STAGES_HIDDEN_IN_VISION,
  pickFocusStage,
  shortStageLabel,
  toPipelineDisplay,
  VISION_STAGE_ORDER,
} from "@/lib/pipeline-messages";
import type { CostBreakdown } from "@/lib/cost-estimate";
import { formatUsd } from "@/lib/cost-estimate";

const DEFAULT_ORDER: PipelineStage[] = [...VISION_STAGE_ORDER];

const TONE_TO_PILL_CLASS: Record<string, string> = {
  ok: "p-pill p-pill-ok",
  warn: "p-pill p-pill-warn",
  fail: "p-pill p-pill-fail",
  info: "p-pill p-pill-info",
  accent: "p-pill p-pill-accent",
  muted: "p-pill",
};

interface PipelineBlockProps {
  stageStates: Record<PipelineStage, StageState> | undefined;
  /** Subset of stages to show, in order. Defaults to a curated 6-stage view. */
  order?: PipelineStage[];
  /** Called when a user clicks a stage chip. Optional. */
  onSelectStage?: (stage: PipelineStage) => void;
  /** Primary call-to-action shown beside the focus sentence. */
  primaryAction?: {
    label: string;
    onClick: () => void;
    disabled?: boolean;
  };
  /** Optional secondary action (e.g. "Pause", "Cancel"). */
  secondaryAction?: {
    label: string;
    onClick: () => void;
  };
  /** Optional cost breakdown - surfaces "$X to finish" in the block. */
  cost?: CostBreakdown;
  className?: string;
}

function dotStateClasses(status: StageStatus): string {
  switch (status) {
    case "completed":
      return "bg-[rgb(var(--p-accent))] shadow-[0_0_12px_-2px_rgb(var(--p-accent)/0.7)]";
    case "running":
      return "bg-[rgb(var(--p-accent))] p-anim-breathe";
    case "failed":
      return "bg-[rgb(var(--p-fail))]";
    case "needs_review":
      return "bg-[rgb(var(--p-warn))]";
    case "ready":
      return "border border-[rgb(var(--p-accent-ring))] bg-[rgb(var(--p-accent-soft))]";
    case "cancelled":
    case "pending":
    default:
      return "border border-[rgb(var(--p-hairline))] bg-[rgb(var(--p-surface-2))]";
  }
}

function connectorClasses(prev: StageStatus | undefined): string {
  // Connector between dot N-1 and dot N. Show as "filled" once the prior
  // stage is complete; otherwise muted.
  if (prev === "completed") {
    return "bg-[rgb(var(--p-accent)/0.45)]";
  }
  return "bg-[rgb(var(--p-hairline))]";
}

export function PipelineBlock({
  stageStates,
  order = DEFAULT_ORDER,
  onSelectStage,
  primaryAction,
  secondaryAction,
  cost,
  className,
}: PipelineBlockProps) {
  const focus = pickFocusStage(stageStates);
  const focusStage = focus?.stage;
  const focusDisplay = focus?.display;

  return (
    <section
      className={`p-glass p-6 md:p-7 ${className ?? ""}`}
      aria-label="Pipeline status"
    >
      {/* Stage chip rail */}
      <ol
        className="relative flex w-full items-center justify-between gap-2 md:gap-4 mb-6"
        aria-label="Pipeline stages"
      >
        {order.map((stage, i) => {
          const state = stageStates?.[stage];
          const status = (state?.status ?? "pending") as StageStatus;
          const prev = i > 0 ? (stageStates?.[order[i - 1]]?.status as StageStatus | undefined) : undefined;
          const isFocus = stage === focusStage;
          const label = shortStageLabel(stage);

          return (
            <React.Fragment key={stage}>
              {i > 0 && (
                <li
                  aria-hidden
                  className={`h-px flex-1 transition-colors duration-[var(--p-mid)] ${connectorClasses(prev)}`}
                />
              )}
              <li className="flex flex-col items-center gap-2 min-w-[64px]">
                <button
                  type="button"
                  onClick={onSelectStage ? () => onSelectStage(stage) : undefined}
                  disabled={!onSelectStage}
                  aria-current={isFocus ? "step" : undefined}
                  aria-label={`${label} - ${status}`}
                  className={[
                    "relative h-5 w-5 rounded-full transition-all duration-[var(--p-mid)] ease-[var(--p-ease)]",
                    "focus:outline-none focus:ring-2 focus:ring-[rgb(var(--p-accent-ring))]",
                    onSelectStage ? "cursor-pointer hover:scale-110" : "cursor-default",
                    dotStateClasses(status),
                    isFocus ? "scale-110 ring-2 ring-[rgb(var(--p-accent-ring))]" : "",
                  ].join(" ")}
                />
                <span
                  className={[
                    "text-[11px] tracking-wide select-none transition-colors duration-[var(--p-mid)]",
                    isFocus
                      ? "text-[rgb(var(--p-text))]"
                      : status === "completed"
                        ? "text-[rgb(var(--p-muted))]"
                        : "text-[rgb(var(--p-hint))]",
                  ].join(" ")}
                >
                  {label}
                </span>
              </li>
            </React.Fragment>
          );
        })}
      </ol>

      {/* Focus sentence + primary action */}
      <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-4">
        <div className="min-w-0">
          <p className="text-[rgb(var(--p-muted))] text-xs uppercase tracking-wider mb-1">
            What's happening
          </p>
          <p className="text-[rgb(var(--p-text))] text-lg md:text-xl font-medium leading-snug">
            {focusDisplay?.sentence ?? "Ready when you are."}
          </p>
          {focusDisplay?.detail && (
            <p className="text-[rgb(var(--p-muted))] text-sm mt-1 line-clamp-2">
              {focusDisplay.detail}
            </p>
          )}
          {cost && (
            <p
              className="text-[rgb(var(--p-muted))] text-xs mt-2 flex items-center gap-2"
              title={cost.parts
                .filter((p) => p.usd > 0 || p.note)
                .map((p) => `${p.label}: ${formatUsd(p.usd)}${p.note ? ` (${p.note})` : ""}`)
                .join("\n")}
            >
              <span className="p-pill p-pill-info">
                <span className="inline-block h-1.5 w-1.5 rounded-full bg-current" />
                {formatUsd(cost.remainingUsd)} to finish
              </span>
              <span className="hidden md:inline">{cost.sentence}</span>
            </p>
          )}
        </div>
        <div className="flex items-center gap-3 shrink-0">
          {focusDisplay && (
            <span className={TONE_TO_PILL_CLASS[focusDisplay.tone] ?? "p-pill"}>
              {focusDisplay.active ? (
                <span className="inline-block h-1.5 w-1.5 rounded-full bg-current p-anim-breathe" />
              ) : (
                <span className="inline-block h-1.5 w-1.5 rounded-full bg-current opacity-60" />
              )}
              {focusDisplay.active ? "in progress" : "ready"}
            </span>
          )}
          {secondaryAction && (
            <button
              type="button"
              className="p-btn-ghost"
              onClick={secondaryAction.onClick}
            >
              {secondaryAction.label}
            </button>
          )}
          {primaryAction && (
            <button
              type="button"
              className="p-btn-primary"
              onClick={primaryAction.onClick}
              disabled={primaryAction.disabled}
            >
              {primaryAction.label}
            </button>
          )}
        </div>
      </div>
    </section>
  );
}
