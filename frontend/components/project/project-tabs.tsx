"use client";

import type { Route } from "next";
import Link from "next/link";
import { usePathname } from "next/navigation";

import { cn } from "@/lib/utils";

/**
 * ProjectTabs — pill-style tab rail, Notion-inspired.
 *
 * The active tab is a soft mint-tinted pill (not an underline) so it
 * reads as "you are here" rather than "this is the selected option of
 * many". A liquid glass strip wraps the row to anchor it visually.
 */

const tabs = [
  { label: "Overview", segment: "" },
  { label: "Panels", segment: "/editor" },
  { label: "Characters", segment: "/characters" },
  { label: "Portraits", segment: "/portraits" },
  { label: "Dictionary", segment: "/dictionary" },
  { label: "Narration", segment: "/narration" },
  { label: "Timeline", segment: "/timeline" },
  { label: "Preview", segment: "/preview" }
] as const;

export function ProjectTabs({ projectId }: { projectId: string }) {
  const pathname = usePathname();
  const base = `/projects/${projectId}`;

  return (
    <div className="p-glass overflow-x-auto px-2 py-2">
      <div className="flex items-center gap-1 min-w-max">
        {tabs.map((tab) => {
          const href = `${base}${tab.segment}`;
          const isActive =
            tab.segment === ""
              ? pathname === base || pathname === `${base}/`
              : pathname.startsWith(href);

          return (
            <Link
              key={tab.segment}
              href={href as Route}
              className={cn(
                "whitespace-nowrap rounded-full px-3 py-1.5 text-sm",
                "transition-all duration-fast ease-liquid",
                isActive
                  ? "bg-accent/[0.12] text-accent shadow-[inset_0_0_0_1px_rgb(var(--p-accent)/0.25)]"
                  : "text-mutedForeground hover:text-foreground hover:bg-white/[0.05]"
              )}
            >
              {tab.label}
            </Link>
          );
        })}
      </div>
    </div>
  );
}
