"use client";

import { useEffect, useRef } from "react";

interface AdaptivePollingOptions {
  enabled?: boolean;
  active?: boolean;
  activeMs?: number;
  idleMs?: number;
  hiddenMs?: number;
  deps?: readonly unknown[];
}

export function useAdaptivePolling(
  callback: () => Promise<void> | void,
  { enabled = true, active = false, activeMs = 8000, idleMs = 30000, hiddenMs = 120000, deps = [] }: AdaptivePollingOptions = {}
) {
  const callbackRef = useRef(callback);

  useEffect(() => {
    callbackRef.current = callback;
  }, [callback]);

  useEffect(() => {
    if (!enabled || typeof window === "undefined") {
      return;
    }

    let cancelled = false;
    let timeoutId: number | null = null;

    const clear = () => {
      if (timeoutId !== null) {
        window.clearTimeout(timeoutId);
        timeoutId = null;
      }
    };

    const schedule = () => {
      if (cancelled) {
        return;
      }
      const delay = document.visibilityState === "hidden" ? hiddenMs : active ? activeMs : idleMs;
      timeoutId = window.setTimeout(() => {
        void tick();
      }, delay);
    };

    const tick = async () => {
      try {
        await callbackRef.current();
      } catch (error) {
        console.error("Adaptive polling callback failed", error);
      } finally {
        clear();
        schedule();
      }
    };

    const handleVisibilityChange = () => {
      if (document.visibilityState !== "visible") {
        return;
      }
      clear();
      timeoutId = window.setTimeout(() => {
        void tick();
      }, active ? 1200 : 2200);
    };

    schedule();
    document.addEventListener("visibilitychange", handleVisibilityChange);

    return () => {
      cancelled = true;
      clear();
      document.removeEventListener("visibilitychange", handleVisibilityChange);
    };
  }, [enabled, active, activeMs, idleMs, hiddenMs, ...deps]);
}
