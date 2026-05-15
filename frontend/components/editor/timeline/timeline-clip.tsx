"use client";

/**
 * TimelineClip - a single panel clip on the video track.
 *
 * Responsible for rendering a clip's thumbnail + label and exposing the
 * right-edge drag handle that lets the user trim duration. Drag-to-trim
 * uses pointer events so it works with mouse + trackpad + touch.
 */

import * as React from "react";
import type { TimelineClip as Clip } from "./use-timeline-state";

interface TimelineClipProps {
  clip: Clip;
  pixelsPerSecond: number;
  selected: boolean;
  onSelect: (id: string) => void;
  onTrim: (id: string, newDurationSec: number) => void;
}

export function TimelineClip({
  clip,
  pixelsPerSecond,
  selected,
  onSelect,
  onTrim,
}: TimelineClipProps) {
  const left = clip.startSec * pixelsPerSecond;
  const width = Math.max(8, clip.durationSec * pixelsPerSecond);

  const dragRef = React.useRef<{ originX: number; originDuration: number } | null>(null);

  const handlePointerDown = (e: React.PointerEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
    (e.target as Element).setPointerCapture(e.pointerId);
    dragRef.current = { originX: e.clientX, originDuration: clip.durationSec };
  };
  const handlePointerMove = (e: React.PointerEvent<HTMLDivElement>) => {
    if (!dragRef.current) return;
    const dx = e.clientX - dragRef.current.originX;
    const next = dragRef.current.originDuration + dx / pixelsPerSecond;
    onTrim(clip.id, next);
  };
  const handlePointerUp = (e: React.PointerEvent<HTMLDivElement>) => {
    dragRef.current = null;
    (e.target as Element).releasePointerCapture(e.pointerId);
  };

  const edgeClass = clip.needsReview ? "p-edge-warn" : "";
  const selectedClass = selected
    ? "ring-2 ring-[rgb(var(--p-accent-ring))] z-10"
    : "ring-1 ring-[rgb(var(--p-hairline))] hover:ring-[rgb(var(--p-accent-soft))]";

  return (
    <div
      role="button"
      aria-label={`Panel ${clip.order} on page ${clip.page}`}
      onClick={(e) => {
        e.stopPropagation();
        onSelect(clip.id);
      }}
      className={[
        "absolute top-2 bottom-2 cursor-pointer rounded-[12px] overflow-hidden",
        "transition-shadow duration-[var(--p-fast)] ease-[var(--p-ease)]",
        edgeClass,
        selectedClass,
      ].join(" ")}
      style={{
        left,
        width,
        background:
          clip.thumbnailUrl
            ? `linear-gradient(180deg, rgb(0 0 0 / 0.05), rgb(0 0 0 / 0.35)), url(${clip.thumbnailUrl}) center/cover`
            : "rgb(var(--p-surface-2))",
      }}
    >
      {/* Caption + status */}
      <div className="absolute inset-0 flex flex-col justify-between p-1.5 pointer-events-none">
        <div className="flex items-center gap-1 text-[10px] text-white/95 drop-shadow">
          <span className="px-1 rounded bg-black/40 backdrop-blur-sm">p{clip.page}·{clip.panel}</span>
          {clip.needsReview && (
            <span className="px-1 rounded bg-[rgb(var(--p-warn)/0.85)] text-[rgb(var(--p-bg-base))]">
              review
            </span>
          )}
        </div>
        <div className="text-[10px] text-white/85 truncate drop-shadow">
          {clip.durationSec.toFixed(1)}s
        </div>
      </div>

      {/* Right-edge trim handle */}
      <div
        role="separator"
        aria-orientation="vertical"
        aria-label="Trim clip end"
        onPointerDown={handlePointerDown}
        onPointerMove={handlePointerMove}
        onPointerUp={handlePointerUp}
        onClick={(e) => e.stopPropagation()}
        className={[
          "absolute right-0 top-0 bottom-0 w-2 cursor-ew-resize",
          "bg-gradient-to-l from-[rgb(var(--p-accent))]/0 to-transparent",
          "hover:from-[rgb(var(--p-accent))]/40",
        ].join(" ")}
      />
    </div>
  );
}
