"use client";

import { useEffect, useRef } from "react";
import { Eye, EyeOff, Merge, Scissors, Trash2 } from "lucide-react";

interface PanelContextMenuProps {
  x: number;
  y: number;
  panelId: string;
  isKept: boolean;
  selectedCount: number;
  onClose: () => void;
  onToggleKeep: (id: string) => void;
  onSplitH: (id: string) => void;
  onSplitV: (id: string) => void;
  onMerge: () => void;
  onDelete: () => void;
}

export function PanelContextMenu({
  x,
  y,
  panelId,
  isKept,
  selectedCount,
  onClose,
  onToggleKeep,
  onSplitH,
  onSplitV,
  onMerge,
  onDelete
}: PanelContextMenuProps) {
  const menuRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    function handleClickOutside(e: MouseEvent) {
      if (menuRef.current && !menuRef.current.contains(e.target as Node)) onClose();
    }
    function handleEscape(e: KeyboardEvent) {
      if (e.key === "Escape") onClose();
    }
    window.addEventListener("mousedown", handleClickOutside);
    window.addEventListener("keydown", handleEscape);
    return () => {
      window.removeEventListener("mousedown", handleClickOutside);
      window.removeEventListener("keydown", handleEscape);
    };
  }, [onClose]);

  const items = [
    {
      label: isKept ? "Remove" : "Keep",
      icon: isKept ? EyeOff : Eye,
      action: () => { onToggleKeep(panelId); onClose(); }
    },
    ...(selectedCount < 2
      ? [
          { label: "Split horizontal", icon: Scissors, action: () => { onSplitH(panelId); onClose(); } },
          { label: "Split vertical", icon: Scissors, action: () => { onSplitV(panelId); onClose(); } }
        ]
      : []),
    ...(selectedCount >= 2
      ? [{ label: `Merge (${selectedCount})`, icon: Merge, action: () => { onMerge(); onClose(); } }]
      : []),
    { label: selectedCount > 1 ? `Delete (${selectedCount})` : "Delete", icon: Trash2, action: () => { onDelete(); onClose(); }, danger: true }
  ];

  return (
    <div
      ref={menuRef}
      className="fixed z-50 min-w-[160px] rounded-lg border border-white/[0.08] bg-zinc-900 py-1 shadow-xl"
      style={{ left: x, top: y }}
    >
      {items.map((item) => {
        const Icon = item.icon;
        return (
          <button
            key={item.label}
            type="button"
            onClick={item.action}
            className={`flex w-full items-center gap-2 px-3 py-1.5 text-left text-sm transition hover:bg-white/[0.08] ${
              "danger" in item ? "text-red-300 hover:text-red-200" : "text-white"
            }`}
          >
            <Icon className="h-3.5 w-3.5 shrink-0 text-mutedForeground" />
            {item.label}
          </button>
        );
      })}
    </div>
  );
}
