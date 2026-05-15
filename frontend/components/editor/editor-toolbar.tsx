"use client";

import { useEffect, useRef, useState } from "react";
import Link from "next/link";
import { ChevronRight, Flag, Keyboard, List, MousePointerClick, Redo2, Undo2 } from "lucide-react";

import { cn } from "@/lib/utils";
import type { PanelBox } from "@/lib/types";
import type { SaveStatus } from "@/hooks/use-auto-save";
import { PageStatusPopover } from "./page-status-popover";

interface EditorToolbarProps {
  projectName: string;
  projectId: string;
  selectedPanelNumber: number;
  panelCount: number;
  selectedPage: number;
  pageCount: number;
  onPageJump: (page: number) => void;
  onPanelJump: (panelNumber: number) => void;
  saveStatus: SaveStatus;
  onSaveNow: () => void;
  drawMode: boolean;
  onToggleDrawMode: () => void;
  flaggedOnlyMode: boolean;
  onToggleFlaggedOnly: () => void;
  flaggedCount: number;
  canUndo: boolean;
  canRedo: boolean;
  onUndo: () => void;
  onRedo: () => void;
  panels: PanelBox[];
  flaggedPanelIds: Set<string>;
}

const saveStatusConfig: Record<SaveStatus, { dot: string; label: string }> = {
  saved: { dot: "bg-ok", label: "Saved" },
  saving: { dot: "bg-amber-400 animate-pulse", label: "Saving..." },
  unsaved: { dot: "bg-amber-400", label: "Unsaved" },
  error: { dot: "bg-red-400", label: "Save failed" }
};

export function EditorToolbar({
  projectName,
  projectId,
  selectedPanelNumber,
  panelCount,
  selectedPage,
  pageCount,
  onPageJump,
  onPanelJump,
  saveStatus,
  onSaveNow,
  drawMode,
  onToggleDrawMode,
  flaggedOnlyMode,
  onToggleFlaggedOnly,
  flaggedCount,
  canUndo,
  canRedo,
  onUndo,
  onRedo,
  panels,
  flaggedPanelIds
}: EditorToolbarProps) {
  const [panelInput, setPanelInput] = useState(String(selectedPanelNumber));
  const [showShortcuts, setShowShortcuts] = useState(false);
  const [showPageStatus, setShowPageStatus] = useState(false);
  const [pageStatusAnchor, setPageStatusAnchor] = useState<DOMRect | null>(null);
  const pageStatusButtonRef = useRef<HTMLButtonElement>(null);
  const statusCfg = saveStatusConfig[saveStatus];

  useEffect(() => {
    setPanelInput(String(selectedPanelNumber));
  }, [selectedPanelNumber]);

  return (
    <div className="flex h-14 shrink-0 items-center gap-3 border-b border-white/[0.08] bg-[linear-gradient(180deg,rgba(255,255,255,0.06),rgba(255,255,255,0.03))] px-4 backdrop-blur">
      {/* Breadcrumb */}
      <div className="flex items-center gap-1.5 text-sm">
        <Link href={`/projects/${projectId}`} className="rounded-full px-2 py-1 text-mutedForeground transition hover:bg-white/[0.08] hover:text-white">
          {projectName || "Project"}
        </Link>
        <ChevronRight className="h-3.5 w-3.5 text-mutedForeground/50" />
        <span className="rounded-full border border-white/[0.08] bg-white/5 px-2.5 py-1 font-medium text-white">Panel Editor</span>
      </div>

      {/* Separator */}
      <div className="h-5 w-px bg-white/10" />

      {/* Panel jump */}
      <div className="flex items-center gap-1.5 rounded-full border border-white/[0.08] bg-black/20 px-2.5 py-1 text-sm text-mutedForeground">
        <span>Panel</span>
        <input
          type="number"
          min={1}
          max={panelCount}
          value={panelInput}
          onChange={(e) => setPanelInput(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter") {
              const n = parseInt(panelInput, 10);
              if (n >= 1 && n <= panelCount) {
                onPanelJump(n);
              }
            }
          }}
          onBlur={() => setPanelInput(String(selectedPanelNumber))}
          className="w-20 rounded-lg border border-white/[0.08] bg-white/5 px-2 py-0.5 text-center text-sm text-white outline-none focus:border-accent"
        />
        <span>of {panelCount}</span>
        <span className="rounded-full bg-white/5 px-2 py-0.5 text-[11px] text-mutedForeground">Page {selectedPage}</span>
        <div className="relative">
          <button
            ref={pageStatusButtonRef}
            type="button"
            onClick={() => {
              setPageStatusAnchor(pageStatusButtonRef.current?.getBoundingClientRect() ?? null);
              setShowPageStatus((v) => !v);
            }}
            title="Page overview"
            className="flex h-7 w-7 items-center justify-center rounded-lg text-mutedForeground transition hover:bg-white/[0.08] hover:text-white"
          >
            <List className="h-3.5 w-3.5" />
          </button>
          {showPageStatus && (
            <PageStatusPopover
              pageCount={pageCount}
              panels={panels}
              flaggedPanelIds={flaggedPanelIds}
              anchorRect={pageStatusAnchor}
              onJumpToPage={onPageJump}
              onClose={() => setShowPageStatus(false)}
            />
          )}
        </div>
      </div>

      {/* Separator */}
      <div className="h-5 w-px bg-white/10" />

      {/* Save status */}
      <button
        type="button"
        onClick={onSaveNow}
        className="flex items-center gap-1.5 rounded-full border border-white/[0.08] bg-black/20 px-3 py-1.5 text-sm text-mutedForeground transition hover:text-white"
        title={saveStatus === "error" ? "Click to retry" : "Click to save now"}
      >
        <div className={cn("h-2 w-2 rounded-full", statusCfg.dot)} />
        <span>{statusCfg.label}</span>
      </button>

      {/* Spacer */}
      <div className="flex-1" />

      {/* Undo / Redo */}
      <div className="flex items-center gap-1">
        <button
          type="button"
          onClick={onUndo}
          disabled={!canUndo}
          title="Undo (Cmd+Z)"
          className="flex h-8 w-8 items-center justify-center rounded-xl border border-white/[0.08] bg-white/[0.03] text-mutedForeground transition hover:bg-white/[0.08] hover:text-white disabled:opacity-30"
        >
          <Undo2 className="h-4 w-4" />
        </button>
        <button
          type="button"
          onClick={onRedo}
          disabled={!canRedo}
          title="Redo (Cmd+Shift+Z)"
          className="flex h-8 w-8 items-center justify-center rounded-xl border border-white/[0.08] bg-white/[0.03] text-mutedForeground transition hover:bg-white/[0.08] hover:text-white disabled:opacity-30"
        >
          <Redo2 className="h-4 w-4" />
        </button>
      </div>

      {/* Draw mode */}
      <button
        type="button"
        onClick={onToggleDrawMode}
        title="Toggle draw mode (D)"
        className={cn(
          "flex h-8 items-center gap-1.5 rounded-xl border px-3 text-sm transition",
          drawMode
            ? "border-accent bg-accent/15 text-accent"
            : "border-white/[0.08] text-mutedForeground hover:bg-white/[0.08] hover:text-white"
        )}
      >
        <MousePointerClick className="h-3.5 w-3.5" />
        {drawMode ? "Drawing" : "Draw"}
      </button>

      {/* Flagged only */}
      <button
        type="button"
        onClick={onToggleFlaggedOnly}
        disabled={flaggedCount === 0}
        title="Toggle flagged-only view (F)"
        className={cn(
          "flex h-8 items-center gap-1.5 rounded-xl border px-3 text-sm transition",
          flaggedOnlyMode
            ? "border-amber-400/40 bg-amber-400/15 text-amber-200"
            : "border-white/[0.08] text-mutedForeground hover:bg-white/[0.08] hover:text-white disabled:opacity-30"
        )}
      >
        <Flag className="h-3.5 w-3.5" />
        {flaggedCount}
      </button>

      {/* Shortcuts help */}
      <div className="relative">
        <button
          type="button"
          onClick={() => setShowShortcuts((v) => !v)}
          title="Keyboard shortcuts"
          className="flex h-8 w-8 items-center justify-center rounded-xl border border-white/[0.08] bg-white/[0.03] text-mutedForeground transition hover:bg-white/[0.08] hover:text-white"
        >
          <Keyboard className="h-4 w-4" />
        </button>
        {showShortcuts && (
          <>
            <div className="fixed inset-0 z-40" onClick={() => setShowShortcuts(false)} />
            <div className="absolute right-0 top-full z-50 mt-2 w-64 rounded-xl border border-white/[0.08] bg-zinc-900 p-3 text-xs shadow-xl">
              <p className="mb-2 font-medium text-white">Keyboard shortcuts</p>
              <div className="space-y-1 text-mutedForeground">
                {[
                  ["[ / ]", "Prev / next page"],
                  ["↑ / ↓", "Prev / next panel"],
                  ["K", "Toggle keep/remove"],
                  ["Space", "Keep + advance"],
                  ["Del", "Delete selected"],
                  ["D", "Draw mode"],
                  ["H / V", "Split H / V"],
                  ["M", "Merge selected"],
                  ["A", "Select all on page"],
                  ["F", "Flagged only"],
                  ["N", "Next flagged"],
                  ["Esc", "Deselect"],
                  ["⌘S", "Save"],
                  ["⌘Z / ⌘⇧Z", "Undo / Redo"]
                ].map(([key, desc]) => (
                  <div key={key} className="flex justify-between">
                    <kbd className="rounded bg-white/10 px-1.5 py-0.5 font-mono text-[10px] text-white">{key}</kbd>
                    <span>{desc}</span>
                  </div>
                ))}
              </div>
            </div>
          </>
        )}
      </div>
    </div>
  );
}
