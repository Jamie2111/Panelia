"use client";

/**
 * TimelineRuler - time markers at the top of the timeline.
 *
 * Adapts tick density to zoom level so we never crowd the ruler: at low
 * pixels-per-second we step in 5s/10s, at high zoom we step in 0.5s.
 */

import * as React from "react";
import { formatTimecode } from "./use-timeline-state";

interface TimelineRulerProps {
  totalDurationSec: number;
  pixelsPerSecond: number;
  className?: string;
}

function pickTickStep(pixelsPerSecond: number): number {
  // Goal: a major tick roughly every 60-120 px.
  const targetPx = 90;
  const raw = targetPx / pixelsPerSecond;
  // Snap to a friendly value.
  const choices = [0.25, 0.5, 1, 2, 5, 10, 30, 60];
  for (const c of choices) if (c >= raw) return c;
  return 120;
}

export function TimelineRuler({
  totalDurationSec,
  pixelsPerSecond,
  className,
}: TimelineRulerProps) {
  const step = pickTickStep(pixelsPerSecond);
  const ticks = React.useMemo(() => {
    const result: number[] = [];
    for (let t = 0; t <= totalDurationSec + 0.0001; t += step) {
      result.push(Math.round(t * 1000) / 1000);
    }
    return result;
  }, [totalDurationSec, step]);

  return (
    <div
      className={`relative h-6 border-b border-[rgb(var(--p-hairline))] text-[10px] text-[rgb(var(--p-hint))] select-none ${className ?? ""}`}
      style={{ width: totalDurationSec * pixelsPerSecond }}
      aria-hidden
    >
      {ticks.map((t) => {
        const x = t * pixelsPerSecond;
        const isMajor = Math.round(t / step) % 2 === 0;
        return (
          <div
            key={t}
            className="absolute top-0 bottom-0 flex items-end"
            style={{ left: x }}
          >
            <div
              className={`w-px ${isMajor ? "h-3 bg-[rgb(var(--p-muted)/0.5)]" : "h-2 bg-[rgb(var(--p-hairline))]"}`}
            />
            {isMajor && (
              <span className="ml-1 mb-0.5 text-[rgb(var(--p-muted))]">{formatTimecode(t)}</span>
            )}
          </div>
        );
      })}
    </div>
  );
}
