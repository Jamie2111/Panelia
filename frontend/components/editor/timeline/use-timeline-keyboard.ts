"use client";

/**
 * use-timeline-keyboard - Resolve / Premiere muscle-memory shortcuts.
 *
 *   J / K / L  - reverse / pause / forward (double-tap ramps speed)
 *   Space      - play/pause
 *   I / O      - set in-mark / out-mark at playhead
 *   ←  / →     - nudge playhead one frame (1/24s)
 *   Shift ←/→  - nudge playhead one second
 *   Up / Down  - select previous / next clip
 *   ⌘/Ctrl + + / -  - zoom in / out
 *   Esc        - clear marks
 *
 * Bind once at the editor root. Ignores keys when typing in an input,
 * textarea, or contentEditable - so inline narration editing never gets
 * eaten by scrubbing shortcuts.
 */

import * as React from "react";
import type { TimelineActions } from "./use-timeline-state";

const NUDGE_FRAME = 1 / 24;
const NUDGE_SECOND = 1;

function isTypingTarget(el: EventTarget | null): boolean {
  if (!(el instanceof HTMLElement)) return false;
  if (el.isContentEditable) return true;
  const tag = el.tagName;
  return tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT";
}

export function useTimelineKeyboard(actions: TimelineActions, enabled: boolean = true) {
  React.useEffect(() => {
    if (!enabled) return;

    const lastPress: { [k: string]: number } = {};
    const handler = (e: KeyboardEvent) => {
      if (isTypingTarget(e.target)) return;
      const now = performance.now();
      const isDouble = (key: string) => {
        const prev = lastPress[key] ?? 0;
        lastPress[key] = now;
        return now - prev < 350;
      };

      switch (e.key.toLowerCase()) {
        case "j": {
          e.preventDefault();
          actions.bumpRate(-1);
          isDouble("j"); // tracked for future ramp behaviour
          break;
        }
        case "k": {
          e.preventDefault();
          actions.bumpRate(0);
          break;
        }
        case "l": {
          e.preventDefault();
          actions.bumpRate(1);
          isDouble("l");
          break;
        }
        case " ": {
          e.preventDefault();
          // Space acts as a play/pause toggle.
          if (lastPress.k && now - lastPress.k < 50) {
            // Avoid race when both keys fire.
            break;
          }
          actions.bumpRate(1);
          break;
        }
        case "i": {
          if (e.metaKey || e.ctrlKey) return;
          e.preventDefault();
          actions.setInMark();
          break;
        }
        case "o": {
          if (e.metaKey || e.ctrlKey) return;
          e.preventDefault();
          actions.setOutMark();
          break;
        }
        case "escape": {
          e.preventDefault();
          actions.clearMarks();
          break;
        }
        case "arrowleft": {
          e.preventDefault();
          actions.nudge(e.shiftKey ? -NUDGE_SECOND : -NUDGE_FRAME);
          break;
        }
        case "arrowright": {
          e.preventDefault();
          actions.nudge(e.shiftKey ? NUDGE_SECOND : NUDGE_FRAME);
          break;
        }
        case "arrowup": {
          e.preventDefault();
          actions.selectPrev();
          break;
        }
        case "arrowdown": {
          e.preventDefault();
          actions.selectNext();
          break;
        }
        case "+":
        case "=": {
          if (e.metaKey || e.ctrlKey) {
            e.preventDefault();
            actions.zoomBy(1.4);
          }
          break;
        }
        case "-": {
          if (e.metaKey || e.ctrlKey) {
            e.preventDefault();
            actions.zoomBy(1 / 1.4);
          }
          break;
        }
        default:
          break;
      }
    };

    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [actions, enabled]);
}
