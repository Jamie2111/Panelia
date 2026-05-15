"use client";

/**
 * waveform-strip.tsx
 *
 * Renders an inline SVG waveform for the narration / music tracks. When a
 * real waveform PNG is available (waveformUrl), it's used as the background
 * image. Otherwise we draw a *procedural* waveform that mimics natural
 * speech / music energy: smooth low-frequency envelope × high-frequency
 * detail, plus a long-period beat for music.
 *
 * Why procedural and not just stripes:
 *   • Stripes read as "loading" or "broken", not "audio"
 *   • A waveform shape teaches the user that this strip represents sound
 *   • Once real audio is rendered, the same component plugs the PNG in
 *     without any layout change
 */

import * as React from "react";

interface WaveformStripProps {
  /** Total seconds the strip represents - drives sample count. */
  totalDurationSec: number;
  pixelsPerSecond: number;
  /** Optional URL for a real waveform image; if present, supersedes the SVG. */
  waveformUrl?: string | null;
  /** Tint of the waveform (matches the track tone). */
  tone: "accent" | "info" | "muted";
  /** Random-seed style integer to keep narration/music distinguishable. */
  seed?: number;
}

const PIXELS_PER_BAR = 3;
const BAR_GAP = 1;

function pseudoRandom(seed: number) {
  // Simple xorshift - fine for visuals, deterministic per seed.
  let x = (seed + 1) | 0;
  return () => {
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    return ((x >>> 0) % 1000) / 1000;
  };
}

function generateBars(
  width: number,
  height: number,
  seed: number,
  pattern: "speech" | "music",
): number[] {
  const stride = PIXELS_PER_BAR + BAR_GAP;
  const count = Math.max(1, Math.floor(width / stride));
  const rnd = pseudoRandom(seed);
  const bars: number[] = [];
  for (let i = 0; i < count; i++) {
    const t = i / count;
    // Slow envelope (the "shape" of the audio) plus high-frequency detail.
    const envelopeA = 0.55 + 0.4 * Math.sin(t * Math.PI * 2.4 + seed);
    const envelopeB = 0.6 + 0.3 * Math.sin(t * Math.PI * 11.7 + seed * 0.7);
    const detail = pattern === "speech"
      // Speech: pause gaps, abrupt energy bursts
      ? (rnd() < 0.07 ? rnd() * 0.15 : 0.35 + rnd() * 0.55)
      // Music: more uniform, with a slow beat
      : 0.45 + 0.45 * Math.abs(Math.sin(t * Math.PI * 18 + seed * 1.3));
    const amp = Math.max(0.06, envelopeA * envelopeB * detail);
    bars.push(Math.min(1, amp) * (height * 0.85));
  }
  return bars;
}

export function WaveformStrip({
  totalDurationSec,
  pixelsPerSecond,
  waveformUrl,
  tone,
  seed = 1337,
}: WaveformStripProps) {
  const width = Math.max(120, totalDurationSec * pixelsPerSecond);
  const height = 48; // matches track row inner height after padding

  const color =
    tone === "accent" ? "rgb(127, 255, 212)"
      : tone === "info" ? "rgb(147, 197, 253)"
      : "rgb(200, 200, 210)";

  // Real-waveform PNG path - used when the consumer provides one.
  if (waveformUrl) {
    return (
      <div
        aria-hidden
        className="absolute inset-y-2 left-0 rounded-[14px] border border-[rgb(var(--p-hairline))] overflow-hidden"
        style={{
          width,
          background: `url(${waveformUrl}) left center / cover no-repeat`,
        }}
      />
    );
  }

  const bars = React.useMemo(
    () => generateBars(width, height, seed, tone === "info" ? "music" : "speech"),
    [width, height, seed, tone],
  );

  return (
    <svg
      aria-hidden
      className="absolute inset-y-2 left-0 rounded-[14px] border border-[rgb(var(--p-hairline))]"
      width={width}
      height={height}
      style={{
        background: `linear-gradient(180deg,
            rgb(var(--p-surface-1)) 0%,
            rgb(var(--p-surface-2)) 100%)`,
      }}
    >
      <defs>
        <linearGradient id={`p-wf-grad-${seed}`} x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor={color} stopOpacity="0.85" />
          <stop offset="100%" stopColor={color} stopOpacity="0.35" />
        </linearGradient>
      </defs>
      <g>
        {bars.map((amp, i) => {
          const x = i * (PIXELS_PER_BAR + BAR_GAP);
          const y = (height - amp) / 2;
          return (
            <rect
              key={i}
              x={x}
              y={y}
              width={PIXELS_PER_BAR}
              height={amp}
              rx={1}
              fill={`url(#p-wf-grad-${seed})`}
            />
          );
        })}
      </g>
    </svg>
  );
}
