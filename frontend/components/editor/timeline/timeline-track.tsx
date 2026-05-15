"use client";

/**
 * TimelineTrack — a single horizontal row.
 *
 * Two variants:
 *   • "video"  — renders TimelineClip children (panel thumbnails)
 *   • "audio"  — renders a single continuous strip with optional inline
 *                waveform image, sized to total duration. (Real waveform
 *                rendering plugs in later; the placeholder strip is enough
 *                to teach the user "this is the narration track".)
 *
 * The track does NOT own clip selection or trim logic — that lives on the
 * child clips. The track's job is layout + click-on-empty to seek.
 */

import * as React from "react";
import { TimelineClip } from "./timeline-clip";
import type { TimelineClip as Clip } from "./use-timeline-state";

interface BaseProps {
  label: string;
  totalDurationSec: number;
  pixelsPerSecond: number;
  onSeek: (sec: number) => void;
  className?: string;
}

interface VideoTrackProps extends BaseProps {
  variant: "video";
  clips: Clip[];
  selectedClipId: string | null;
  onSelectClip: (id: string) => void;
  onTrimClip: (id: string, newDurationSec: number) => void;
}

interface AudioTrackProps extends BaseProps {
  variant: "audio";
  /** Optional URL of a pre-rendered waveform PNG. */
  waveformUrl?: string | null;
  /** Tone of the track strip (e.g. accent for narration, info for music). */
  tone?: "accent" | "info" | "muted";
}

type TimelineTrackProps = VideoTrackProps | AudioTrackProps;

export function TimelineTrack(props: TimelineTrackProps) {
  const handleBackgroundClick = (e: React.MouseEvent<HTMLDivElement>) => {
    const rect = (e.currentTarget as HTMLDivElement).getBoundingClientRect();
    const x = e.clientX - rect.left;
    const sec = x / props.pixelsPerSecond;
    props.onSeek(sec);
  };

  const widthPx = Math.max(120, props.totalDurationSec * props.pixelsPerSecond);

  return (
    <div className={`flex items-stretch ${props.className ?? ""}`}>
      {/* Track header */}
      <div className="w-24 shrink-0 px-3 py-2 border-r border-[rgb(var(--p-hairline))] bg-[rgb(var(--p-surface-1))] sticky left-0 z-20 backdrop-blur-md">
        <p className="text-[10px] uppercase tracking-wider text-[rgb(var(--p-hint))]">
          {props.label}
        </p>
      </div>

      {/* Track body */}
      <div
        className="relative h-16 flex-1 bg-[rgb(var(--p-surface-1))] border-b border-[rgb(var(--p-hairline))] cursor-pointer"
        style={{ width: widthPx }}
        onClick={handleBackgroundClick}
      >
        {props.variant === "video" ? (
          <>
            {props.clips.map((clip) => (
              <TimelineClip
                key={clip.id}
                clip={clip}
                pixelsPerSecond={props.pixelsPerSecond}
                selected={props.selectedClipId === clip.id}
                onSelect={props.onSelectClip}
                onTrim={props.onTrimClip}
              />
            ))}
          </>
        ) : (
          <AudioStrip
            widthPx={widthPx}
            waveformUrl={props.waveformUrl}
            tone={props.tone ?? "muted"}
          />
        )}
      </div>
    </div>
  );
}

function AudioStrip({
  widthPx,
  waveformUrl,
  tone,
}: {
  widthPx: number;
  waveformUrl?: string | null;
  tone: "accent" | "info" | "muted";
}) {
  // Background gradient that hints at "audio energy" even without a real
  // waveform. Replace with `<img src={waveformUrl}>` when audio is rendered.
  const fill =
    tone === "accent"
      ? "rgb(var(--p-accent) / 0.18)"
      : tone === "info"
      ? "rgb(var(--p-info) / 0.18)"
      : "rgb(255 255 255 / 0.06)";
  return (
    <div
      aria-hidden
      className="absolute inset-y-2 rounded-[12px] border border-[rgb(var(--p-hairline))]"
      style={{
        left: 0,
        width: widthPx,
        background:
          waveformUrl
            ? `url(${waveformUrl}) center/100% 100% no-repeat`
            : `repeating-linear-gradient(
                90deg,
                ${fill} 0px, ${fill} 6px,
                transparent 6px, transparent 14px
              )`,
      }}
    />
  );
}
