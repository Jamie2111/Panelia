"use client";

/**
 * WhileYouWereAway - proactive "what changed" callout.
 *
 * The fix for the most common UX failure I observed: the user closes
 * the browser tab while a long stage is running, comes back, and has
 * to mentally piece together whether anything completed. Refresh
 * anxiety. Trust collapse.
 *
 * This component compares the last snapshot of stage states the user
 * saw (persisted in localStorage) with the current state, and renders
 * a calm, one-liner banner summarizing what completed and what now
 * needs their attention. The user can dismiss; the snapshot updates
 * either way so the banner only fires when something actually new
 * happened.
 *
 * It is intentionally subtle - a one-line glass banner with a single
 * action - not a modal. The goal is awareness, not interruption.
 */

import * as React from "react";
import type {
  PipelineStage,
  ProjectDetail,
  StageStatus,
  StageState,
} from "@/lib/types";
import { shortStageLabel } from "@/lib/pipeline-messages";

interface Snapshot {
  /** Map of stage → "status@updated_at" - small string we can diff cheaply. */
  states: Partial<Record<PipelineStage, string>>;
  /** Map of stage → progress (0-100) at last view. */
  progress: Partial<Record<PipelineStage, number>>;
  /** Timestamp of the snapshot. */
  savedAt: number;
}

function snapshotKey(projectId: string) {
  return `panelia.snapshot.${projectId}`;
}

function loadSnapshot(projectId: string): Snapshot | null {
  if (typeof window === "undefined") return null;
  try {
    const raw = window.localStorage.getItem(snapshotKey(projectId));
    return raw ? (JSON.parse(raw) as Snapshot) : null;
  } catch {
    return null;
  }
}

function saveSnapshot(projectId: string, snap: Snapshot) {
  if (typeof window === "undefined") return;
  try {
    window.localStorage.setItem(snapshotKey(projectId), JSON.stringify(snap));
  } catch {
    /* ignore */
  }
}

function buildSnapshot(stageStates: Record<PipelineStage, StageState>): Snapshot {
  const states: Snapshot["states"] = {};
  const progress: Snapshot["progress"] = {};
  for (const [stage, state] of Object.entries(stageStates ?? {})) {
    states[stage as PipelineStage] = `${state.status}@${state.updated_at}`;
    progress[stage as PipelineStage] = state.progress;
  }
  return { states, progress, savedAt: Date.now() };
}

interface DiffOutcome {
  completed: PipelineStage[];
  needsReview: PipelineStage[];
  failed: PipelineStage[];
  hasChange: boolean;
}

function diff(prev: Snapshot | null, now: Record<PipelineStage, StageState>): DiffOutcome {
  const completed: PipelineStage[] = [];
  const needsReview: PipelineStage[] = [];
  const failed: PipelineStage[] = [];
  for (const [stage, state] of Object.entries(now ?? {})) {
    const stageKey = stage as PipelineStage;
    const prevKey = prev?.states?.[stageKey];
    const nowKey = `${state.status}@${state.updated_at}`;
    if (prevKey === nowKey) continue; // no change for this stage
    const status = state.status as StageStatus;
    if (status === "completed") completed.push(stageKey);
    else if (status === "needs_review") needsReview.push(stageKey);
    else if (status === "failed") failed.push(stageKey);
  }
  return {
    completed,
    needsReview,
    failed,
    hasChange: completed.length + needsReview.length + failed.length > 0,
  };
}

function describe(diffOut: DiffOutcome): string {
  const parts: string[] = [];
  if (diffOut.completed.length > 0) {
    const names = diffOut.completed.map(shortStageLabel).join(", ");
    parts.push(`${diffOut.completed.length === 1 ? names : `${diffOut.completed.length} stages (${names})`} finished`);
  }
  if (diffOut.needsReview.length > 0) {
    const names = diffOut.needsReview.map(shortStageLabel).join(", ");
    parts.push(`${names} ${diffOut.needsReview.length === 1 ? "needs" : "need"} your review`);
  }
  if (diffOut.failed.length > 0) {
    const names = diffOut.failed.map(shortStageLabel).join(", ");
    parts.push(`${names} ran into a problem`);
  }
  return parts.join(" · ");
}

interface WhileYouWereAwayProps {
  project: Pick<ProjectDetail, "id" | "stage_states">;
  /** Optional handler when user clicks the primary action. */
  onSeeDetails?: () => void;
  /** Custom CTA label. Defaults to "See what changed". */
  actionLabel?: string;
  /** Override how long ago counts as "while you were away" (ms). */
  minQuietMs?: number;
  className?: string;
}

export function WhileYouWereAway({
  project,
  onSeeDetails,
  actionLabel,
  minQuietMs = 30_000,
  className,
}: WhileYouWereAwayProps) {
  const [snapshot, setSnapshot] = React.useState<Snapshot | null>(null);
  const [dismissed, setDismissed] = React.useState(false);

  // Load the prior snapshot ONCE on mount so we diff against the user's
  // last view, not against every re-render.
  React.useEffect(() => {
    setSnapshot(loadSnapshot(project.id));
  }, [project.id]);

  // After we've shown the diff (or there's nothing to show), persist a
  // fresh snapshot so the banner won't reappear for unchanged state.
  React.useEffect(() => {
    if (!project?.stage_states) return;
    const fresh = buildSnapshot(project.stage_states as Record<PipelineStage, StageState>);
    // Only overwrite the snapshot when we've actually rendered (or
    // dismissed) the banner - otherwise we'd clobber unread changes
    // mid-render.
    if (dismissed) {
      saveSnapshot(project.id, fresh);
    }
  }, [project, dismissed]);

  // Persist a snapshot if there's nothing to diff yet (first-ever load).
  React.useEffect(() => {
    if (snapshot === null && project?.stage_states) {
      saveSnapshot(project.id, buildSnapshot(project.stage_states as Record<PipelineStage, StageState>));
    }
  }, [snapshot, project]);

  if (!project?.stage_states) return null;
  if (dismissed) return null;
  if (!snapshot) return null;

  // Don't fire for users who literally just left and came right back.
  if (Date.now() - snapshot.savedAt < minQuietMs) return null;

  const diffOut = diff(snapshot, project.stage_states as Record<PipelineStage, StageState>);
  if (!diffOut.hasChange) return null;

  const description = describe(diffOut);
  const tone =
    diffOut.failed.length > 0
      ? "fail"
      : diffOut.needsReview.length > 0
        ? "warn"
        : "ok";
  const pillClass =
    tone === "fail" ? "p-pill p-pill-fail"
      : tone === "warn" ? "p-pill p-pill-warn"
      : "p-pill p-pill-ok";

  return (
    <section
      className={`p-glass flex items-center justify-between gap-4 px-5 py-3 ${className ?? ""}`}
      role="status"
      aria-live="polite"
    >
      <div className="flex items-center gap-3 min-w-0">
        <span className={pillClass}>
          <span className="inline-block h-1.5 w-1.5 rounded-full bg-current" />
          While you were away
        </span>
        <p className="text-[rgb(var(--p-text))] text-sm md:text-base truncate">
          {description}.
        </p>
      </div>
      <div className="flex items-center gap-2 shrink-0">
        {onSeeDetails && (
          <button type="button" className="p-btn-ghost" onClick={onSeeDetails}>
            {actionLabel ?? "See what changed"}
          </button>
        )}
        <button
          type="button"
          aria-label="Dismiss"
          className="text-[rgb(var(--p-hint))] hover:text-[rgb(var(--p-text))] transition-colors duration-[var(--p-fast)] px-2"
          onClick={() => setDismissed(true)}
        >
          ✕
        </button>
      </div>
    </section>
  );
}
