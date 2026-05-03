"use client";

import type { PanelBox } from "@/lib/types";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { Button } from "@/components/ui/button";
import { ArrowDown, ArrowUp, Scissors, Merge, Trash2, Eye, EyeOff, MessageSquareText, Sparkles } from "lucide-react";

function formatReviewFlag(flag: string) {
  return flag.replaceAll("_", " ");
}

interface PanelInspectorProps {
  panel: PanelBox | undefined;
  selectedCount: number;
  onToggleKeep: (id: string) => void;
  onDelete: () => void;
  onMoveOrder: (id: string, direction: "up" | "down") => void;
  onSplit: (id: string, axis: "horizontal" | "vertical") => void;
  onMerge: () => void;
  onUpdatePanel: (id: string, updates: Partial<PanelBox>) => void;
  onDetectedTextChange: (id: string, value: string) => void;
}

export function PanelInspector({
  panel,
  selectedCount,
  onToggleKeep,
  onDelete,
  onMoveOrder,
  onSplit,
  onMerge,
  onUpdatePanel,
  onDetectedTextChange
}: PanelInspectorProps) {
  if (!panel) {
    return (
      <div className="flex h-full flex-col items-center justify-center gap-2 p-4 text-center">
        <p className="text-sm text-mutedForeground">Select a panel to inspect</p>
        <p className="text-xs text-mutedForeground/60">Click on the canvas or the strip below</p>
      </div>
    );
  }

  const stateLabel = panel.keep ? "Included" : panel.auto_skipped ? "Auto-skipped" : "Removed";

  return (
    <div className="flex h-full flex-col gap-3 overflow-y-auto p-3">
      {/* Header */}
      <div className="rounded-[22px] border border-white/10 bg-[linear-gradient(180deg,rgba(255,255,255,0.06),rgba(255,255,255,0.02))] p-4">
      <div className="flex items-center justify-between">
        <div>
          <p className="text-sm font-medium text-white">
            Page {panel.page} / Panel {panel.panel}
          </p>
          <p className="text-xs text-mutedForeground">Order #{panel.order}</p>
        </div>
        <button
          type="button"
          onClick={() => onToggleKeep(panel.id)}
          className="flex items-center gap-1.5 rounded-full border border-white/10 px-3 py-1 text-xs transition hover:bg-white/10"
        >
          {panel.keep ? <Eye className="h-3 w-3 text-accent" /> : <EyeOff className="h-3 w-3 text-red-400" />}
          <span className={panel.keep ? "text-accent" : "text-red-300"}>{stateLabel}</span>
        </button>
      </div>
      <div className="mt-3 grid grid-cols-2 gap-2 text-[11px] text-mutedForeground">
        <div className="rounded-2xl border border-white/10 bg-black/20 px-3 py-2">
          <p className="uppercase tracking-[0.18em]">Order</p>
          <p className="mt-1 text-sm text-white">#{panel.order}</p>
        </div>
        <div className="rounded-2xl border border-white/10 bg-black/20 px-3 py-2">
          <p className="uppercase tracking-[0.18em]">Text mode</p>
          <p className="mt-1 text-sm text-white">{panel.manual_ocr_text ? "Manual" : "Auto"}</p>
        </div>
      </div>
      </div>

      {/* Review flags */}
      {panel.review_flags.length > 0 && (
        <div className="rounded-[22px] border border-amber-400/20 bg-amber-400/5 px-4 py-3">
          <p className="text-[10px] uppercase tracking-wider text-amber-300">Review flags</p>
          <p className="mt-1 text-xs text-amber-200">
            {panel.review_flags.map(formatReviewFlag).join(", ")}
          </p>
        </div>
      )}

      {/* Quick actions */}
      <div className="grid grid-cols-2 gap-1.5 rounded-[22px] border border-white/10 bg-white/[0.03] p-3">
        <Button variant="secondary" size="sm" onClick={() => onMoveOrder(panel.id, "up")} className="h-8 text-xs">
          <ArrowUp className="mr-1 h-3 w-3" /> Up
        </Button>
        <Button variant="secondary" size="sm" onClick={() => onMoveOrder(panel.id, "down")} className="h-8 text-xs">
          <ArrowDown className="mr-1 h-3 w-3" /> Down
        </Button>
        <Button variant="secondary" size="sm" onClick={() => onSplit(panel.id, "vertical")} className="h-8 text-xs">
          <Scissors className="mr-1 h-3 w-3" /> Split H
        </Button>
        <Button variant="secondary" size="sm" onClick={() => onSplit(panel.id, "horizontal")} className="h-8 text-xs">
          <Scissors className="mr-1 h-3 w-3" /> Split V
        </Button>
        <Button
          variant="secondary"
          size="sm"
          onClick={onMerge}
          disabled={selectedCount < 2}
          className="h-8 text-xs"
        >
          <Merge className="mr-1 h-3 w-3" /> Merge ({selectedCount})
        </Button>
        <Button variant="secondary" size="sm" onClick={onDelete} className="h-8 text-xs text-red-300 hover:text-red-200">
          <Trash2 className="mr-1 h-3 w-3" /> Delete
        </Button>
      </div>

      {/* Duration */}
      <label className="space-y-1.5 rounded-[22px] border border-white/10 bg-white/[0.03] p-3">
        <span className="text-xs text-mutedForeground">Duration (sec)</span>
        <Input
          type="number"
          step="0.1"
          value={panel.duration_seconds ?? 2.8}
          onChange={(e) => onUpdatePanel(panel.id, { duration_seconds: Number(e.target.value) })}
          className="h-8 text-sm"
        />
      </label>

      {/* Text mode */}
      <div className="rounded-[22px] border border-white/10 bg-white/[0.03] p-4">
        <div className="flex items-center justify-between">
          <p className="flex items-center gap-2 text-xs font-medium text-white">
            <MessageSquareText className="h-3.5 w-3.5 text-accent" />
            Extracted dialogue
          </p>
          <span className="text-[10px] text-mutedForeground">
            {panel.manual_ocr_text ? "Manual" : "Auto"}
          </span>
        </div>
        {!(panel.ocr_text ?? "").trim() ? (
          <div className="mt-3 rounded-2xl border border-white/10 bg-black/20 px-3 py-2 text-[11px] text-mutedForeground">
            <div className="flex items-center gap-2 text-white/90">
              <Sparkles className="h-3.5 w-3.5 text-brand-amber" />
              No extracted dialogue saved yet
            </div>
            <p className="mt-1">
              That can be normal here. Auto OCR usually lands later in the pipeline, and some panels genuinely have no readable dialogue. You can still type a manual override here or double-click the panel on canvas.
            </p>
          </div>
        ) : null}
        <Textarea
          className="mt-2 min-h-[100px] text-sm"
          value={panel.ocr_text ?? ""}
          placeholder="Edit detected text or leave blank to skip"
          onChange={(e) => onDetectedTextChange(panel.id, e.target.value)}
        />
        {panel.skip_reason && (
          <p className="mt-2 text-[10px] text-mutedForeground">{panel.skip_reason}</p>
        )}
      </div>

      {/* Zoom hint */}
      <div className="rounded-[22px] border border-white/10 bg-white/[0.03] p-3">
        <p className="text-xs text-mutedForeground">Zoom hint</p>
        <p className="mt-1 text-sm text-white">{panel.zoom_hint ?? "focus-center"}</p>
      </div>
    </div>
  );
}
