"use client";

import { useEffect, useRef } from "react";

import { buildMediaUrl, cn } from "@/lib/utils";
import type { PanelBox } from "@/lib/types";

interface PageFilmstripProps {
  projectId: string;
  pageCount: number;
  selectedPage: number;
  onSelectPage: (page: number) => void;
  panels: PanelBox[];
  flaggedPages: Set<number>;
}

export function PageFilmstrip({
  projectId,
  pageCount,
  selectedPage,
  onSelectPage,
  panels,
  flaggedPages
}: PageFilmstripProps) {
  const scrollRef = useRef<HTMLDivElement>(null);
  const activeRef = useRef<HTMLButtonElement>(null);

  // Scroll active page into view
  useEffect(() => {
    const scrollEl = scrollRef.current;
    const activeEl = activeRef.current;
    if (!scrollEl || !activeEl) return;
    const targetTop = activeEl.offsetTop - (scrollEl.clientHeight - activeEl.clientHeight) / 2;
    scrollEl.scrollTo({
      top: Math.max(0, targetTop),
      behavior: "smooth"
    });
  }, [selectedPage]);

  // Compute page stats
  const pageStats = new Map<number, { total: number; kept: number }>();
  for (const p of panels) {
    const s = pageStats.get(p.page) ?? { total: 0, kept: 0 };
    s.total++;
    if (p.keep) s.kept++;
    pageStats.set(p.page, s);
  }

  return (
    <div
      ref={scrollRef}
      className="hidden h-full w-[118px] shrink-0 flex-col gap-2 overflow-y-auto border-r border-white/10 bg-[linear-gradient(180deg,rgba(255,255,255,0.05),rgba(255,255,255,0.02))] p-2.5 md:flex"
    >
      {Array.from({ length: pageCount }, (_, i) => i + 1).map((page) => {
        const stats = pageStats.get(page);
        const isActive = page === selectedPage;
        const isFullyKept = !!stats && stats.total > 0 && stats.kept === stats.total;
        const isMuted = !!stats && stats.kept === 0;
        const isPartial = !!stats && stats.kept > 0 && stats.kept < stats.total;
        const hasFlagged = flaggedPages.has(page);

        const borderColor = isActive
          ? "border-accent"
          : isMuted
            ? "border-red-500/40"
            : isPartial
              ? "border-amber-400/40"
              : isFullyKept
                ? "border-emerald-400/40"
                : "border-white/10";

        return (
          <button
            key={page}
            ref={isActive ? activeRef : undefined}
            type="button"
            onClick={() => onSelectPage(page)}
            className={cn(
              "group relative shrink-0 overflow-hidden rounded-2xl border-2 transition",
              borderColor,
              isActive && "scale-[1.01] ring-2 ring-accent/30",
              hasFlagged && !isActive && "animate-pulse"
            )}
          >
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img
              src={buildMediaUrl(`/media/projects/${projectId}/pages/${String(page).padStart(4, "0")}.png`)}
              alt={`Page ${page}`}
              className="block w-full object-cover"
              loading="lazy"
              draggable={false}
            />
            <div className="absolute inset-x-0 bottom-0 bg-gradient-to-t from-black/85 to-transparent px-2 pb-1.5 pt-4">
              <span className="rounded-full bg-black/50 px-1.5 py-0.5 text-[10px] font-medium text-white">{page}</span>
              {stats && (
                <span className="ml-1 text-[9px] text-mutedForeground">
                  {stats.kept}/{stats.total}
                </span>
              )}
            </div>
          </button>
        );
      })}
    </div>
  );
}
