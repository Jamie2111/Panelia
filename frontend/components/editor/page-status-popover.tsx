"use client";

import { useEffect, useMemo, useState } from "react";
import { createPortal } from "react-dom";
import { ChevronDown, ChevronUp } from "lucide-react";

import type { PanelBox } from "@/lib/types";

type SortKey = "page" | "panels" | "kept" | "flagged";

interface PageStatusPopoverProps {
  pageCount: number;
  panels: PanelBox[];
  flaggedPanelIds: Set<string>;
  anchorRect: DOMRect | null;
  onJumpToPage: (page: number) => void;
  onClose: () => void;
}

export function PageStatusPopover({
  pageCount,
  panels,
  flaggedPanelIds,
  anchorRect,
  onJumpToPage,
  onClose
}: PageStatusPopoverProps) {
  const [sortKey, setSortKey] = useState<SortKey>("page");
  const [sortAsc, setSortAsc] = useState(true);
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    setMounted(true);
  }, []);

  const rows = useMemo(() => {
    const map = new Map<number, { page: number; panels: number; kept: number; flagged: number }>();
    for (let i = 1; i <= pageCount; i++) {
      map.set(i, { page: i, panels: 0, kept: 0, flagged: 0 });
    }
    for (const p of panels) {
      const row = map.get(p.page);
      if (!row) continue;
      row.panels++;
      if (p.keep) row.kept++;
      if (flaggedPanelIds.has(p.id)) row.flagged++;
    }
    const arr = Array.from(map.values());
    arr.sort((a, b) => {
      const diff = a[sortKey] - b[sortKey];
      return sortAsc ? diff : -diff;
    });
    return arr;
  }, [pageCount, panels, flaggedPanelIds, sortKey, sortAsc]);

  function toggleSort(key: SortKey) {
    if (sortKey === key) {
      setSortAsc((v) => !v);
    } else {
      setSortKey(key);
      setSortAsc(key === "page");
    }
  }

  const SortIcon = sortAsc ? ChevronUp : ChevronDown;
  const width = 360;
  const viewportWidth = mounted ? window.innerWidth : width;
  const viewportHeight = mounted ? window.innerHeight : 720;
  const top = Math.min((anchorRect?.bottom ?? 88) + 8, viewportHeight - 120);
  const left = Math.max(12, Math.min(anchorRect?.left ?? 360, viewportWidth - width - 12));
  const maxHeight = Math.max(220, viewportHeight - top - 16);

  if (!mounted) return null;

  return createPortal(
    <>
      <div className="fixed inset-0 z-[9998]" onClick={onClose} />
      <div
        className="fixed z-[9999] rounded-xl border border-white/[0.08] bg-zinc-900 shadow-2xl shadow-black/60"
        style={{ left, top, width }}
      >
        <div className="overflow-y-auto p-2" style={{ maxHeight }}>
          <table className="w-full text-xs">
            <thead>
              <tr className="text-mutedForeground">
                {(["page", "panels", "kept", "flagged"] as const).map((key) => (
                  <th
                    key={key}
                    className="cursor-pointer px-2 py-1.5 text-left font-medium hover:text-white"
                    onClick={() => toggleSort(key)}
                  >
                    <span className="flex items-center gap-0.5">
                      {key.charAt(0).toUpperCase() + key.slice(1)}
                      {sortKey === key && <SortIcon className="h-3 w-3" />}
                    </span>
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {rows.map((row) => (
                <tr
                  key={row.page}
                  className="cursor-pointer rounded transition hover:bg-white/[0.08]"
                  onClick={() => { onJumpToPage(row.page); onClose(); }}
                >
                  <td className="rounded-l px-2 py-1 font-mono text-white">{row.page}</td>
                  <td className="px-2 py-1 text-mutedForeground">{row.panels}</td>
                  <td className="px-2 py-1 text-ok">{row.kept}</td>
                  <td className="rounded-r px-2 py-1">
                    {row.flagged > 0 ? (
                      <span className="text-amber-300">{row.flagged}</span>
                    ) : (
                      <span className="text-mutedForeground/40">0</span>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </>,
    document.body
  );
}
