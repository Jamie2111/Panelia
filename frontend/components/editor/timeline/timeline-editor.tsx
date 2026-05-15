"use client";

/**
 * TimelineEditor — the Resolve-inspired editing surface.
 *
 * Composes:
 *   ┌── Preview ─────────────────────────┐ ┌── Inspector ───┐
 *   │ panel image + narration subtitle   │ │ selected clip  │
 *   │ transport / timecode / I-O pills   │ │ duration + edit│
 *   └────────────────────────────────────┘ │ regenerate     │
 *   ┌── Toolbar (zoom, I/O, play) ──────────────────────────┐
 *   │                                                       │
 *   ┌── Ruler ──────────────────────────────────────────────┐
 *   ┌── Video track (panels) ───────────────────────────────┐
 *   ┌── Narration track (audio strip) ──────────────────────┐
 *   ┌── Music track (audio strip) ──────────────────────────┐
 *
 * Keyboard: J/K/L scrub, Space play, I/O mark, ←→ nudge, ↑↓ select.
 * Click empty track area = seek. Click clip = select. Drag right edge = trim.
 *
 * Stateless from the consumer's POV — pass in panels, get a fully-wired
 * editor. Persistence (save edits back to backend) is the consumer's job.
 */

import * as React from "react";
import type { PanelBox } from "@/lib/types";
import { useTimelineState, formatTimecode } from "./use-timeline-state";
import { useTimelineKeyboard } from "./use-timeline-keyboard";
import { TimelineRuler } from "./timeline-ruler";
import { TimelineTrack } from "./timeline-track";
import { TimelinePlayhead } from "./timeline-playhead";
import { TimelineInspector } from "./timeline-inspector";
import { TimelinePreview } from "./timeline-preview";
import {
  IconRewind,
  IconPause,
  IconPlay,
  IconZoomIn,
  IconZoomOut,
  IconMarkIn,
  IconMarkOut,
  IconSave,
} from "./transport-icons";

export interface TimelineEditorProps {
  panels: PanelBox[];
  thumbnailBaseUrl?: string;
  /** Optional narration audio waveform URL (one PNG per project). */
  narrationWaveformUrl?: string | null;
  /** Optional music audio waveform URL. */
  musicWaveformUrl?: string | null;
  /** Called when user clicks "regenerate" in the inspector. */
  onRegeneratePanel?: (panelId: string) => void;
  /** Called when user wants to persist current edits to the backend. */
  onSaveEdits?: (edits: PanelEdits) => void;
  className?: string;
}

export interface PanelEdits {
  /** Per-panel changes the user has made but not yet saved. */
  durations: Record<string, number>;
  narrations: Record<string, string>;
}

export function TimelineEditor({
  panels,
  thumbnailBaseUrl,
  narrationWaveformUrl,
  musicWaveformUrl,
  onRegeneratePanel,
  onSaveEdits,
  className,
}: TimelineEditorProps) {
  const { state, actions } = useTimelineState({ panels, thumbnailBaseUrl });
  useTimelineKeyboard(actions, true);

  const scrollRef = React.useRef<HTMLDivElement | null>(null);
  const selectedClip =
    state.clips.find((c) => c.id === state.selectedClipId) ?? null;

  // Auto-scroll the timeline to keep the playhead in view.
  React.useEffect(() => {
    const el = scrollRef.current;
    if (!el) return;
    const playheadX = state.playheadSec * state.pixelsPerSecond + 96; /* track header width */
    const view = el.scrollLeft;
    const right = view + el.clientWidth;
    if (playheadX < view + 64) el.scrollLeft = Math.max(0, playheadX - 96);
    else if (playheadX > right - 64) el.scrollLeft = playheadX - el.clientWidth + 96;
  }, [state.playheadSec, state.pixelsPerSecond]);

  // Save edits — derive durations/narrations diff from clip overrides.
  const handleSave = () => {
    if (!onSaveEdits) return;
    const edits: PanelEdits = { durations: {}, narrations: {} };
    for (const clip of state.clips) {
      const original = panels.find((p) => p.id === clip.id);
      if (!original) continue;
      const origDur = Number(original.duration_seconds) || 4.0;
      if (Math.abs(origDur - clip.durationSec) > 0.001) {
        edits.durations[clip.id] = clip.durationSec;
      }
      if ((original.narration ?? "") !== clip.narration) {
        edits.narrations[clip.id] = clip.narration;
      }
    }
    onSaveEdits(edits);
  };

  return (
    <div
      className={`flex flex-col gap-4 ${className ?? ""}`}
      data-testid="timeline-editor"
    >
      {/* Top: preview + inspector */}
      <div className="grid grid-cols-1 lg:grid-cols-[1fr_360px] gap-4">
        <TimelinePreview
          clip={selectedClip}
          playheadSec={state.playheadSec}
          totalDurationSec={state.totalDurationSec}
          inMarkSec={state.inMarkSec}
          outMarkSec={state.outMarkSec}
          playbackRate={state.playbackRate}
          playing={state.playing}
        />
        <TimelineInspector
          clip={selectedClip}
          onChangeNarration={(id, value) => actions.setNarration(id, value)}
          onTrim={(id, sec) => actions.trimClip(id, sec)}
          onRegenerate={onRegeneratePanel}
          onJumpInTimeline={(id) => actions.selectClip(id)}
        />
      </div>

      {/* Toolbar */}
      <section className="p-glass flex flex-wrap items-center justify-between gap-3 px-4 py-2">
        <div className="flex items-center gap-1 text-xs">
          <button
            type="button"
            className="p-btn-ghost !px-2.5 !py-2"
            onClick={() => actions.bumpRate(-1)}
            aria-label="Play backwards (J)"
            title="Play backwards (J)"
          >
            <IconRewind size={16} />
          </button>
          <button
            type="button"
            className="p-btn-ghost !px-2.5 !py-2"
            onClick={() => actions.bumpRate(0)}
            aria-label="Pause (K)"
            title="Pause (K)"
          >
            <IconPause size={16} />
          </button>
          <button
            type="button"
            className="p-btn-ghost !px-2.5 !py-2"
            onClick={() => actions.bumpRate(1)}
            aria-label="Play (L)"
            title="Play (L)"
          >
            <IconPlay size={16} />
          </button>
          <span className="mx-2 h-5 w-px bg-[rgb(var(--p-hairline))]" />
          <button
            type="button"
            className="p-btn-ghost"
            onClick={() => actions.setInMark()}
            aria-label="Set in mark (I)"
            title="Set in mark (I)"
          >
            <IconMarkIn size={14} /> In
          </button>
          <button
            type="button"
            className="p-btn-ghost"
            onClick={() => actions.setOutMark()}
            aria-label="Set out mark (O)"
            title="Set out mark (O)"
          >
            <IconMarkOut size={14} /> Out
          </button>
          <button type="button" className="p-btn-ghost" onClick={() => actions.clearMarks()}>
            Clear
          </button>
        </div>
        <div className="flex items-center gap-2 text-xs">
          <button
            type="button"
            className="p-btn-ghost !px-2.5 !py-2"
            onClick={() => actions.zoomBy(1 / 1.4)}
            aria-label="Zoom out"
            title="Zoom out (⌘−)"
          >
            <IconZoomOut size={16} />
          </button>
          <span className="p-pill min-w-[78px] justify-center">
            {state.pixelsPerSecond.toFixed(0)} px/s
          </span>
          <button
            type="button"
            className="p-btn-ghost !px-2.5 !py-2"
            onClick={() => actions.zoomBy(1.4)}
            aria-label="Zoom in"
            title="Zoom in (⌘+)"
          >
            <IconZoomIn size={16} />
          </button>
          <span className="mx-2 h-5 w-px bg-[rgb(var(--p-hairline))]" />
          <span className="text-[rgb(var(--p-muted))] font-mono">
            {formatTimecode(state.playheadSec)} / {formatTimecode(state.totalDurationSec)}
          </span>
          {onSaveEdits && (
            <button type="button" className="p-btn-primary ml-2" onClick={handleSave}>
              <IconSave size={14} /> Save edits
            </button>
          )}
        </div>
      </section>

      {/* Timeline scroller */}
      <section className="p-glass overflow-hidden">
        <div
          ref={scrollRef}
          className="overflow-x-auto overflow-y-hidden relative"
          style={{ scrollbarGutter: "stable" }}
        >
          <div
            className="relative"
            style={{ width: Math.max(800, 96 + state.totalDurationSec * state.pixelsPerSecond) }}
          >
            {/* Header row: track-label gutter + ruler */}
            <div className="flex items-stretch">
              <div className="w-24 shrink-0 border-r border-[rgb(var(--p-hairline))] bg-[rgb(var(--p-surface-1))] sticky left-0 z-20 backdrop-blur-md">
                {/* Empty gutter to align with track headers below */}
              </div>
              <TimelineRuler
                totalDurationSec={state.totalDurationSec}
                pixelsPerSecond={state.pixelsPerSecond}
              />
            </div>

            {/* Tracks */}
            <TimelineTrack
              variant="video"
              label="Panels"
              totalDurationSec={state.totalDurationSec}
              pixelsPerSecond={state.pixelsPerSecond}
              clips={state.clips}
              selectedClipId={state.selectedClipId}
              onSeek={actions.seek}
              onSelectClip={actions.selectClip}
              onTrimClip={actions.trimClip}
            />
            <TimelineTrack
              variant="audio"
              label="Narration"
              tone="accent"
              waveformUrl={narrationWaveformUrl}
              totalDurationSec={state.totalDurationSec}
              pixelsPerSecond={state.pixelsPerSecond}
              onSeek={actions.seek}
            />
            <TimelineTrack
              variant="audio"
              label="Music"
              tone="info"
              waveformUrl={musicWaveformUrl}
              totalDurationSec={state.totalDurationSec}
              pixelsPerSecond={state.pixelsPerSecond}
              onSeek={actions.seek}
            />

            {/* Playhead overlay (positioned over tracks only, not header gutter) */}
            <div className="absolute top-6 left-24 right-0 bottom-0 pointer-events-none">
              <TimelinePlayhead
                positionSec={state.playheadSec}
                pixelsPerSecond={state.pixelsPerSecond}
                onSeek={actions.seek}
                className="pointer-events-auto"
              />
              {/* I/O range shading */}
              {state.inMarkSec !== null && state.outMarkSec !== null && (
                <div
                  aria-hidden
                  className="absolute top-0 bottom-0 bg-[rgb(var(--p-accent)/0.06)] border-l border-r border-[rgb(var(--p-accent)/0.4)]"
                  style={{
                    left: state.inMarkSec * state.pixelsPerSecond,
                    width: Math.max(0, (state.outMarkSec - state.inMarkSec) * state.pixelsPerSecond),
                  }}
                />
              )}
            </div>
          </div>
        </div>
      </section>
    </div>
  );
}
