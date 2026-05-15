"use client";

import { useEffect, useRef } from "react";

import { cn } from "@/lib/utils";
import type { PanelBox } from "@/lib/types";
import { PanelThumbnail } from "./panel-thumbnail";

interface PanelStripProps {
  panels: PanelBox[];
  selectedIds: string[];
  imageUrl: string;
  naturalWidth: number;
  naturalHeight: number;
  onSelect: (id: string, additive?: boolean) => void;
  onToggleKeep: (id: string) => void;
}

export function PanelStrip({
  panels,
  selectedIds,
  imageUrl,
  naturalWidth,
  naturalHeight,
  onSelect,
  onToggleKeep
}: PanelStripProps) {
  const scrollRef = useRef<HTMLDivElement>(null);
  const selectedSet = new Set(selectedIds);

  // Scroll selected tile into view
  useEffect(() => {
    const scrollEl = scrollRef.current;
    if (!selectedIds[0] || !scrollEl) return;
    const el = scrollEl.querySelector<HTMLElement>(`[data-panel-id="${selectedIds[0]}"]`);
    if (!el) return;
    const targetLeft = el.offsetLeft - (scrollEl.clientWidth - el.clientWidth) / 2;
    scrollEl.scrollTo({
      left: Math.max(0, targetLeft),
      behavior: "smooth"
    });
  }, [selectedIds]);

  if (!panels.length) {
    return (
      <div className="flex h-full items-center justify-center text-sm text-mutedForeground">
        No panels on this page
      </div>
    );
  }

  return (
    <div
      ref={scrollRef}
      className="flex h-full items-center gap-2 overflow-x-auto px-3 py-2"
    >
      {panels.map((panel) => {
        const selected = selectedSet.has(panel.id);
        return (
          <button
            key={panel.id}
            type="button"
            data-panel-id={panel.id}
            onClick={(e) => onSelect(panel.id, e.shiftKey)}
            onDoubleClick={(e) => {
              e.preventDefault();
              onSelect(panel.id, false);
              onToggleKeep(panel.id);
            }}
            title={panel.keep ? "Double-click to exclude this panel" : "Double-click to include this panel"}
            className={cn(
              "group relative flex h-[98px] shrink-0 flex-col overflow-hidden rounded-2xl border-2 bg-black/20 transition",
              selected
                ? "border-accent shadow-[0_0_0_2px_rgba(34,211,238,0.25)]"
                : panel.keep
                  ? "border-white/[0.08] hover:border-white/20"
                  : "border-red-500/30 hover:border-red-500/50"
            )}
          >
            <div className="flex h-[72px] w-[92px] items-center justify-center px-1 py-1">
              <PanelThumbnail
                panel={panel}
                imageUrl={imageUrl}
                naturalWidth={naturalWidth}
                naturalHeight={naturalHeight}
                className="h-full w-full"
                contain
              />
            </div>
            <div className="flex h-[24px] w-full items-center justify-between bg-black/75 px-2 text-[10px] backdrop-blur-sm">
              <span className="min-w-10 text-left font-mono font-medium text-white">#{panel.order}</span>
              <span
                className={cn(
                  "rounded-full px-1.5 py-px",
                  panel.keep ? "bg-accent/20 text-accent" : "bg-red-500/20 text-red-300"
                )}
              >
                {panel.keep ? "keep" : panel.auto_skipped ? "skip" : "out"}
              </span>
            </div>
          </button>
        );
      })}
    </div>
  );
}
