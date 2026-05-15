"use client";

/**
 * use-timeline-state - central state for the Resolve-style timeline.
 *
 * What it owns:
 *   • clips           - ordered list of TimelineClip derived from panels
 *   • playheadSec     - current scrubber position in seconds
 *   • selectedClipId  - single-clip selection (the Inspector reads this)
 *   • inMarkSec / outMarkSec - I/O range for partial export & loop
 *   • pixelsPerSecond - zoom level (drives clip widths)
 *   • playing         - JKL/playback state
 *   • playbackRate    - JKL speed (J=-1, K=0, L=+1; double-tap = ±2)
 *
 * What it doesn't own:
 *   • The audio engine itself (Web Audio is plugged in by the consumer).
 *   • Persistence - the consumer decides when to write back to panels.json.
 *
 * Why a custom hook (not Zustand): this state is local to one editor
 * instance, doesn't need to survive route changes, and stays simple
 * with reducer + selectors.
 */

import * as React from "react";
import type { PanelBox } from "@/lib/types";

/** One clip on the timeline (always a kept panel). */
export interface TimelineClip {
  id: string;             // matches PanelBox.id
  order: number;
  page: number;
  panel: number;
  startSec: number;       // computed running total
  durationSec: number;    // editable
  narration: string;
  zoomHint: string | null;
  thumbnailUrl: string | null;
  needsReview: boolean;
  source: string;         // narration_source raw value
  // Content-safety bridge so the inspector can show "this clip will blur
  // in the final video" without having to look the panel up again.
  contentRating: "safe" | "borderline" | "explicit" | null;
  contentRatingReason: string | null;
  contentBlur: boolean;
}

export interface TimelineState {
  clips: TimelineClip[];
  selectedClipId: string | null;
  playheadSec: number;
  inMarkSec: number | null;
  outMarkSec: number | null;
  pixelsPerSecond: number;
  playing: boolean;
  playbackRate: number;
  totalDurationSec: number;
}

export interface TimelineActions {
  selectClip: (clipId: string | null) => void;
  selectNext: () => void;
  selectPrev: () => void;
  seek: (sec: number) => void;
  nudge: (delta: number) => void;
  play: () => void;
  pause: () => void;
  setRate: (rate: number) => void;
  /** JKL standard: J reverses, K pauses, L forwards (double-tap = ×2). */
  bumpRate: (direction: -1 | 0 | 1) => void;
  setInMark: () => void;
  setOutMark: () => void;
  clearMarks: () => void;
  zoomBy: (factor: number) => void;
  setZoom: (pixelsPerSecond: number) => void;
  trimClip: (clipId: string, newDurationSec: number) => void;
  /** Replace a clip's narration in-place (called from inline edit). */
  setNarration: (clipId: string, narration: string) => void;
}

function buildClipsFromPanels(panels: PanelBox[], thumbBaseUrl?: string): TimelineClip[] {
  const kept = panels
    .filter((p) => p.keep)
    .slice()
    .sort((a, b) => (a.page - b.page) || (a.panel - b.panel));
  let cursor = 0;
  return kept.map((p) => {
    const duration = Math.max(0.4, Number(p.duration_seconds) || 4.0);
    const startSec = cursor;
    cursor += duration;
    const flags = p.review_flags ?? [];
    const visionFlag = flags.find((f) => typeof f === "string" && f.startsWith("vision_"));
    const nsfwFlag = flags.find((f) => typeof f === "string" && f.startsWith("nsfw_"));
    return {
      id: p.id,
      order: p.order,
      page: p.page,
      panel: p.panel,
      startSec,
      durationSec: duration,
      narration: p.narration ?? "",
      zoomHint: p.zoom_hint ?? null,
      thumbnailUrl: thumbBaseUrl ? `${thumbBaseUrl}/panel_${String(p.order).padStart(3, "0")}.png` : null,
      needsReview:
        Boolean(visionFlag) ||
        Boolean(nsfwFlag) ||
        !p.narration ||
        !p.narration.trim(),
      source: p.narration_source ?? "",
      contentRating: (p.content_rating ?? null) as TimelineClip["contentRating"],
      contentRatingReason: p.content_rating_reason ?? null,
      contentBlur: Boolean(p.content_blur),
    };
  });
}

const MIN_PPS = 6;       // very zoomed-out
const MAX_PPS = 240;     // ~4 sec across the visible viewport on most screens
const DEFAULT_PPS = 32;  // sensible mid-zoom

interface UseTimelineStateOptions {
  panels: PanelBox[];
  /** Base URL used to construct each panel's thumbnail (panels/panel_NNN.png). */
  thumbnailBaseUrl?: string;
}

export function useTimelineState({
  panels,
  thumbnailBaseUrl,
}: UseTimelineStateOptions): {
  state: TimelineState;
  actions: TimelineActions;
} {
  // Rebuild clips whenever panels change. This is intentional -
  // panels.json is the source of truth; the timeline reflects it.
  const clips = React.useMemo(
    () => buildClipsFromPanels(panels, thumbnailBaseUrl),
    [panels, thumbnailBaseUrl],
  );

  // Override map for live edits (narration text, durations) that we
  // haven't persisted yet. Keyed by clip.id.
  const [overrides, setOverrides] = React.useState<
    Record<string, Partial<TimelineClip>>
  >({});

  const liveClips = React.useMemo<TimelineClip[]>(() => {
    let cursor = 0;
    return clips.map((c) => {
      const ov = overrides[c.id];
      const dur = ov?.durationSec ?? c.durationSec;
      const result: TimelineClip = {
        ...c,
        ...(ov ?? {}),
        durationSec: dur,
        startSec: cursor,
      };
      cursor += dur;
      return result;
    });
  }, [clips, overrides]);

  const totalDurationSec = React.useMemo(
    () => liveClips.reduce((acc, c) => acc + c.durationSec, 0),
    [liveClips],
  );

  const [selectedClipId, setSelectedClipId] = React.useState<string | null>(
    () => liveClips[0]?.id ?? null,
  );
  const [playheadSec, setPlayheadSec] = React.useState<number>(0);
  const [inMarkSec, setInMarkSec] = React.useState<number | null>(null);
  const [outMarkSec, setOutMarkSec] = React.useState<number | null>(null);
  const [pixelsPerSecond, setPixelsPerSecond] = React.useState<number>(DEFAULT_PPS);
  const [playing, setPlaying] = React.useState<boolean>(false);
  const [playbackRate, setPlaybackRate] = React.useState<number>(1);

  // Clamp playhead when total duration shrinks.
  React.useEffect(() => {
    if (playheadSec > totalDurationSec) setPlayheadSec(totalDurationSec);
  }, [totalDurationSec, playheadSec]);

  // Sync selected clip to playhead position when playing.
  React.useEffect(() => {
    if (!playing) return;
    // Find clip that contains the current playhead and update selection.
    const clip = liveClips.find(
      (c) => playheadSec >= c.startSec && playheadSec < c.startSec + c.durationSec,
    );
    if (clip && clip.id !== selectedClipId) setSelectedClipId(clip.id);
  }, [playing, playheadSec, liveClips, selectedClipId]);

  // Drive the playhead during play.
  React.useEffect(() => {
    if (!playing || playbackRate === 0) return;
    let raf = 0;
    let lastTs = performance.now();
    const tick = (ts: number) => {
      const dt = (ts - lastTs) / 1000;
      lastTs = ts;
      setPlayheadSec((prev) => {
        const next = prev + dt * playbackRate;
        if (next <= 0) return 0;
        if (next >= totalDurationSec) {
          setPlaying(false);
          return totalDurationSec;
        }
        return next;
      });
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [playing, playbackRate, totalDurationSec]);

  const actions: TimelineActions = {
    selectClip: (clipId) => {
      setSelectedClipId(clipId);
      if (clipId) {
        const clip = liveClips.find((c) => c.id === clipId);
        if (clip) setPlayheadSec(clip.startSec);
      }
    },
    selectNext: () => {
      const idx = liveClips.findIndex((c) => c.id === selectedClipId);
      if (idx >= 0 && idx < liveClips.length - 1) {
        const next = liveClips[idx + 1];
        setSelectedClipId(next.id);
        setPlayheadSec(next.startSec);
      }
    },
    selectPrev: () => {
      const idx = liveClips.findIndex((c) => c.id === selectedClipId);
      if (idx > 0) {
        const prev = liveClips[idx - 1];
        setSelectedClipId(prev.id);
        setPlayheadSec(prev.startSec);
      }
    },
    seek: (sec) => {
      const clamped = Math.max(0, Math.min(totalDurationSec, sec));
      setPlayheadSec(clamped);
      const clip = liveClips.find(
        (c) => clamped >= c.startSec && clamped < c.startSec + c.durationSec,
      );
      if (clip && clip.id !== selectedClipId) setSelectedClipId(clip.id);
    },
    nudge: (delta) => actions.seek(playheadSec + delta),
    play: () => {
      if (!playing) setPlaybackRate(1);
      setPlaying(true);
    },
    pause: () => {
      setPlaying(false);
      setPlaybackRate(0);
    },
    setRate: (rate) => {
      setPlaybackRate(rate);
      setPlaying(rate !== 0);
    },
    bumpRate: (direction) => {
      // Standard JKL semantics. K (0) always pauses.
      if (direction === 0) {
        setPlaying(false);
        setPlaybackRate(0);
        return;
      }
      // Pressing L while paused starts forward; pressing again ramps up.
      setPlaybackRate((prev) => {
        if (Math.sign(prev) !== direction) return direction; // change direction → 1×
        // same direction; bump up the ladder
        const abs = Math.min(Math.abs(prev) * 2, 8);
        return direction * Math.max(1, abs);
      });
      setPlaying(true);
    },
    setInMark: () => setInMarkSec(playheadSec),
    setOutMark: () => setOutMarkSec(playheadSec),
    clearMarks: () => {
      setInMarkSec(null);
      setOutMarkSec(null);
    },
    zoomBy: (factor) =>
      setPixelsPerSecond((prev) => {
        const next = Math.max(MIN_PPS, Math.min(MAX_PPS, prev * factor));
        return next;
      }),
    setZoom: (pps) =>
      setPixelsPerSecond(Math.max(MIN_PPS, Math.min(MAX_PPS, pps))),
    trimClip: (clipId, newDurationSec) => {
      const dur = Math.max(0.4, newDurationSec);
      setOverrides((prev) => ({
        ...prev,
        [clipId]: { ...(prev[clipId] ?? {}), durationSec: dur },
      }));
    },
    setNarration: (clipId, narration) => {
      setOverrides((prev) => ({
        ...prev,
        [clipId]: { ...(prev[clipId] ?? {}), narration },
      }));
    },
  };

  const state: TimelineState = {
    clips: liveClips,
    selectedClipId,
    playheadSec,
    inMarkSec,
    outMarkSec,
    pixelsPerSecond,
    playing,
    playbackRate,
    totalDurationSec,
  };

  return { state, actions };
}

/** Format a duration in seconds as "1:23.4" for the ruler / readouts. */
export function formatTimecode(sec: number): string {
  if (!isFinite(sec) || sec < 0) return "0:00.0";
  const m = Math.floor(sec / 60);
  const s = sec - m * 60;
  return `${m}:${s.toFixed(1).padStart(4, "0")}`;
}
