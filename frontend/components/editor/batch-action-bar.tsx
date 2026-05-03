"use client";

import { useState } from "react";
import { Eye, EyeOff, Merge, Timer, Trash2 } from "lucide-react";

import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";

interface BatchActionBarProps {
  selectedCount: number;
  allOnSamePage: boolean;
  onKeepAll: () => void;
  onRemoveAll: () => void;
  onDeleteAll: () => void;
  onMerge: () => void;
  onSetDuration: (seconds: number) => void;
}

export function BatchActionBar({
  selectedCount,
  allOnSamePage,
  onKeepAll,
  onRemoveAll,
  onDeleteAll,
  onMerge,
  onSetDuration
}: BatchActionBarProps) {
  const [durationInput, setDurationInput] = useState("");

  if (selectedCount < 2) return null;

  return (
    <div className="mx-3 mt-2 flex items-center gap-2 rounded-2xl border border-white/10 bg-zinc-900/90 px-4 py-2 backdrop-blur">
      <span className="rounded-full bg-white/5 px-2 py-1 text-[11px] font-medium text-mutedForeground">
        {selectedCount} selected
      </span>

      <div className="h-4 w-px bg-white/10" />

      <Button variant="secondary" size="sm" onClick={onKeepAll} className="h-7 gap-1 text-xs">
        <Eye className="h-3 w-3" /> Keep all
      </Button>
      <Button variant="secondary" size="sm" onClick={onRemoveAll} className="h-7 gap-1 text-xs">
        <EyeOff className="h-3 w-3" /> Remove all
      </Button>

      {allOnSamePage && (
        <Button variant="secondary" size="sm" onClick={onMerge} className="h-7 gap-1 text-xs">
          <Merge className="h-3 w-3" /> Merge
        </Button>
      )}

      <div className="flex items-center gap-1">
        <Timer className="h-3 w-3 text-mutedForeground" />
        <Input
          type="number"
          step="0.1"
          placeholder="sec"
          value={durationInput}
          onChange={(e) => setDurationInput(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter") {
              const n = parseFloat(durationInput);
              if (n > 0) { onSetDuration(n); setDurationInput(""); }
            }
          }}
          className="h-7 w-16 text-xs"
        />
      </div>

      <div className="flex-1" />

      <Button variant="secondary" size="sm" onClick={onDeleteAll} className="h-7 gap-1 text-xs text-red-300 hover:text-red-200">
        <Trash2 className="h-3 w-3" /> Delete ({selectedCount})
      </Button>
    </div>
  );
}
