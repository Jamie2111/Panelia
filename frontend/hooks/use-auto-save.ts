"use client";

import { useCallback, useEffect, useRef, useState } from "react";

export type SaveStatus = "saved" | "saving" | "unsaved" | "error";

interface AutoSaveOptions {
  /** The data to watch for changes (e.g. panels array). */
  data: unknown;
  /** Stable reference data from server (to compare against). */
  serverData: unknown;
  /** Async save function. */
  save: () => Promise<unknown>;
  /** Debounce delay in ms. */
  delayMs?: number;
}

export function useAutoSave({ data, serverData, save, delayMs = 3000 }: AutoSaveOptions) {
  const [status, setStatus] = useState<SaveStatus>("saved");
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const saveRef = useRef(save);
  saveRef.current = save;
  const isMountedRef = useRef(true);
  const inFlightRef = useRef<Promise<void> | null>(null);

  const hasChanges = JSON.stringify(data) !== JSON.stringify(serverData);

  const runSave = useCallback(async () => {
    if (inFlightRef.current) {
      return inFlightRef.current;
    }

    setStatus("saving");
    const savePromise = (async () => {
      try {
        await saveRef.current();
        if (isMountedRef.current) {
          setStatus("saved");
        }
      } catch (error) {
        if (isMountedRef.current) {
          setStatus("error");
        }
        throw error;
      } finally {
        inFlightRef.current = null;
      }
    })();

    inFlightRef.current = savePromise;
    return savePromise;
  }, []);

  // Expose an immediate save (for Cmd+S)
  const saveNow = useCallback(async () => {
    if (timerRef.current) {
      clearTimeout(timerRef.current);
      timerRef.current = null;
    }
    await runSave();
  }, [runSave]);

  // Debounced auto-save
  useEffect(() => {
    if (!hasChanges) {
      setStatus((prev) => (prev === "saving" ? prev : "saved"));
      return;
    }
    setStatus("unsaved");

    if (timerRef.current) clearTimeout(timerRef.current);
    timerRef.current = setTimeout(() => {
      void runSave().catch(() => {
        // Auto-save updates the shared save status. Manual saves handle the surfaced error.
      });
    }, delayMs);

    return () => {
      if (timerRef.current) clearTimeout(timerRef.current);
    };
  }, [data, hasChanges, delayMs, runSave]);

  // beforeunload guard
  useEffect(() => {
    function handler(e: BeforeUnloadEvent) {
      if (hasChanges) {
        e.preventDefault();
      }
    }
    window.addEventListener("beforeunload", handler);
    return () => window.removeEventListener("beforeunload", handler);
  }, [hasChanges]);

  // Cleanup
  useEffect(() => {
    isMountedRef.current = true;
    return () => {
      isMountedRef.current = false;
    };
  }, []);

  return { status, hasChanges, saveNow };
}
