"use client";

/**
 * TimelinePlayhead - vertical line + grabbable handle marking current time.
 * Renders over all tracks; drag to scrub.
 */

import * as React from "react";

interface TimelinePlayheadProps {
  /** Seconds */
  positionSec: number;
  pixelsPerSecond: number;
  /** Called as user drags the playhead handle. */
  onSeek: (sec: number) => void;
  /** Height of the playhead line - defaults to full container. */
  className?: string;
}

export function TimelinePlayhead({
  positionSec,
  pixelsPerSecond,
  onSeek,
  className,
}: TimelinePlayheadProps) {
  const left = positionSec * pixelsPerSecond;

  const drag = React.useRef<{ startX: number; startSec: number } | null>(null);
  const onDown = (e: React.PointerEvent) => {
    e.stopPropagation();
    (e.target as Element).setPointerCapture(e.pointerId);
    drag.current = { startX: e.clientX, startSec: positionSec };
  };
  const onMove = (e: React.PointerEvent) => {
    if (!drag.current) return;
    const dx = e.clientX - drag.current.startX;
    onSeek(drag.current.startSec + dx / pixelsPerSecond);
  };
  const onUp = (e: React.PointerEvent) => {
    drag.current = null;
    (e.target as Element).releasePointerCapture(e.pointerId);
  };

  return (
    <div
      aria-hidden
      className={`pointer-events-none absolute top-0 bottom-0 z-30 ${className ?? ""}`}
      style={{ left }}
    >
      {/* Vertical line */}
      <div className="absolute top-0 bottom-0 w-px bg-[rgb(var(--p-accent))] shadow-[0_0_12px_-2px_rgb(var(--p-accent)/0.7)]" />
      {/* Drag handle (head) */}
      <div
        role="slider"
        aria-label="Playhead"
        onPointerDown={onDown}
        onPointerMove={onMove}
        onPointerUp={onUp}
        className="pointer-events-auto absolute -top-1 -translate-x-1/2 h-3 w-3 rotate-45 bg-[rgb(var(--p-accent))] shadow-[0_0_8px_rgb(var(--p-accent)/0.8)] cursor-ew-resize"
        style={{ left: 0 }}
      />
    </div>
  );
}
