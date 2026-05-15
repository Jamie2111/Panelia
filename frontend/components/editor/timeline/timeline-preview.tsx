"use client";

/**
 * TimelinePreview - top-of-editor "what plays right now" surface.
 *
 * Shows the selected panel image, its narration, and a transport readout
 * (playhead timecode / total / I-O range). Will eventually overlay Ken
 * Burns motion, transitions, and TTS preview - but those are downstream
 * features; the data flow is already in place for them.
 */

import * as React from "react";
import type { TimelineClip } from "./use-timeline-state";
import { formatTimecode } from "./use-timeline-state";

interface TimelinePreviewProps {
  clip: TimelineClip | null;
  playheadSec: number;
  totalDurationSec: number;
  inMarkSec: number | null;
  outMarkSec: number | null;
  playbackRate: number;
  playing: boolean;
  className?: string;
}

export function TimelinePreview({
  clip,
  playheadSec,
  totalDurationSec,
  inMarkSec,
  outMarkSec,
  playbackRate,
  playing,
  className,
}: TimelinePreviewProps) {
  return (
    <section
      aria-label="Preview"
      className={`p-glass overflow-hidden flex flex-col ${className ?? ""}`}
    >
      <div className="flex-1 relative bg-black/40 min-h-[280px]">
        {clip?.thumbnailUrl ? (
          <div
            className="absolute inset-0 bg-center bg-contain bg-no-repeat"
            style={{ backgroundImage: `url(${clip.thumbnailUrl})` }}
            role="img"
            aria-label={`Preview of panel ${clip.order}`}
          />
        ) : (
          <div className="absolute inset-0 grid place-items-center text-[rgb(var(--p-hint))] text-sm">
            Select a clip to preview.
          </div>
        )}
        {/* Narration overlay - subtitle style */}
        {clip?.narration && (
          <div className="absolute inset-x-0 bottom-0 p-6 text-center text-[rgb(var(--p-text))] text-base md:text-lg leading-snug bg-gradient-to-t from-black/70 via-black/40 to-transparent">
            {clip.narration}
          </div>
        )}
      </div>

      {/* Transport bar */}
      <div className="px-4 py-2 flex items-center justify-between border-t border-[rgb(var(--p-hairline))] text-xs text-[rgb(var(--p-muted))]">
        <div className="flex items-center gap-3 font-mono">
          <span>{formatTimecode(playheadSec)}</span>
          <span className="text-[rgb(var(--p-hint))]">/</span>
          <span>{formatTimecode(totalDurationSec)}</span>
        </div>
        <div className="flex items-center gap-2">
          {inMarkSec !== null && (
            <span className="p-pill p-pill-accent">
              I {formatTimecode(inMarkSec)}
            </span>
          )}
          {outMarkSec !== null && (
            <span className="p-pill p-pill-accent">
              O {formatTimecode(outMarkSec)}
            </span>
          )}
          <span className="p-pill">
            {playing ? `${playbackRate.toFixed(0)}× ${playbackRate < 0 ? "rev" : "play"}` : "paused"}
          </span>
        </div>
      </div>
    </section>
  );
}
