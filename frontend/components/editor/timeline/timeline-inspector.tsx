"use client";

/**
 * TimelineInspector — right-rail "this clip" panel (Resolve-style).
 *
 * Shows the selected clip's editable properties. Designed to be the
 * single place where per-panel actions live, so the timeline itself
 * stays clean. Inline-edits the narration via a textarea (uncontrolled
 * commits on blur so we don't fire a state update on every keystroke).
 */

import * as React from "react";
import type { TimelineClip } from "./use-timeline-state";

interface TimelineInspectorProps {
  clip: TimelineClip | null;
  onChangeNarration: (clipId: string, value: string) => void;
  onTrim: (clipId: string, newDurationSec: number) => void;
  onRegenerate?: (clipId: string) => void;
  onJumpInTimeline?: (clipId: string) => void;
  className?: string;
}

export function TimelineInspector({
  clip,
  onChangeNarration,
  onTrim,
  onRegenerate,
  onJumpInTimeline,
  className,
}: TimelineInspectorProps) {
  // Local mirror so typing feels instant; commit on blur.
  const [draft, setDraft] = React.useState(clip?.narration ?? "");
  const [duration, setDuration] = React.useState(clip?.durationSec ?? 0);

  React.useEffect(() => {
    setDraft(clip?.narration ?? "");
    setDuration(clip?.durationSec ?? 0);
  }, [clip?.id, clip?.narration, clip?.durationSec]);

  if (!clip) {
    return (
      <aside className={`p-glass p-6 ${className ?? ""}`} aria-label="Inspector">
        <p className="text-[rgb(var(--p-muted))] text-sm">
          Select a panel to edit it.
        </p>
      </aside>
    );
  }

  return (
    <aside
      className={`p-glass p-5 flex flex-col gap-4 ${className ?? ""}`}
      aria-label="Inspector"
    >
      <header className="flex items-start justify-between gap-2">
        <div>
          <p className="text-[10px] uppercase tracking-wider text-[rgb(var(--p-hint))]">
            Inspector
          </p>
          <h3 className="text-[rgb(var(--p-text))] text-base font-medium">
            Page {clip.page} · Panel {clip.panel}
          </h3>
          <p className="text-xs text-[rgb(var(--p-muted))]">
            #{clip.order} · source: {clip.source || "—"}
          </p>
        </div>
        {clip.needsReview && <span className="p-pill p-pill-warn">Needs review</span>}
      </header>

      {/* Thumbnail */}
      {clip.thumbnailUrl && (
        <div
          className="aspect-[3/4] w-full rounded-[16px] border border-[rgb(var(--p-hairline))] bg-center bg-cover"
          style={{ backgroundImage: `url(${clip.thumbnailUrl})` }}
          role="img"
          aria-label={`Thumbnail for panel ${clip.order}`}
        />
      )}

      {/* Narration editor */}
      <label className="flex flex-col gap-1">
        <span className="text-[10px] uppercase tracking-wider text-[rgb(var(--p-hint))]">
          Narration
        </span>
        <textarea
          value={draft}
          onChange={(e) => setDraft(e.target.value)}
          onBlur={() => {
            if (draft !== clip.narration) onChangeNarration(clip.id, draft);
          }}
          rows={5}
          placeholder="Write the narration for this panel."
          className="w-full resize-y bg-[rgb(var(--p-surface-2))] border border-[rgb(var(--p-hairline))] focus:border-[rgb(var(--p-accent-ring))] focus:outline-none rounded-[var(--p-r-md)] p-3 text-sm text-[rgb(var(--p-text))] placeholder:text-[rgb(var(--p-hint))]"
        />
      </label>

      {/* Duration */}
      <label className="flex items-center justify-between gap-3">
        <span className="text-[10px] uppercase tracking-wider text-[rgb(var(--p-hint))]">
          Duration
        </span>
        <div className="flex items-center gap-2">
          <input
            type="number"
            min={0.4}
            step={0.1}
            value={Number(duration.toFixed(2))}
            onChange={(e) => setDuration(parseFloat(e.target.value))}
            onBlur={() => {
              if (!isNaN(duration) && duration !== clip.durationSec) {
                onTrim(clip.id, duration);
              }
            }}
            className="w-20 text-right bg-[rgb(var(--p-surface-2))] border border-[rgb(var(--p-hairline))] rounded-[var(--p-r-sm)] px-2 py-1 text-sm"
          />
          <span className="text-xs text-[rgb(var(--p-muted))]">sec</span>
        </div>
      </label>

      {/* Zoom hint */}
      <div className="flex items-center justify-between">
        <span className="text-[10px] uppercase tracking-wider text-[rgb(var(--p-hint))]">
          Motion
        </span>
        <span className="p-pill">{clip.zoomHint ?? "pan-wide"}</span>
      </div>

      {/* Actions */}
      <div className="flex flex-col gap-2 pt-2 border-t border-[rgb(var(--p-hairline))]">
        {onRegenerate && (
          <button type="button" className="p-btn-ghost justify-center" onClick={() => onRegenerate(clip.id)}>
            ↻ Regenerate with vision
          </button>
        )}
        {onJumpInTimeline && (
          <button type="button" className="p-btn-ghost justify-center" onClick={() => onJumpInTimeline(clip.id)}>
            ↦ Jump playhead here
          </button>
        )}
      </div>
    </aside>
  );
}
