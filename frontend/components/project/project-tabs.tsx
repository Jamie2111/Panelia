"use client";

/**
 * DEPRECATED - project tabs are now rendered inline inside PageHeader
 * via the `views` prop. See `lib/project-views.ts#buildProjectViews`.
 *
 * This component is kept as a thin shim so EditorShell and any unknown
 * caller doesn't break. It renders the same pill-rail UI as before but
 * built on top of the new `PageHeader` view-switcher tokens so the
 * pixels match. New code should not import this; use PageHeader's
 * `views` prop instead.
 */

import type { Route } from "next";
import Link from "next/link";
import { usePathname } from "next/navigation";

import { cn } from "@/lib/utils";

const tabs = [
  { label: "Overview", segment: "" },
  { label: "Panels", segment: "/editor" },
  { label: "Narration", segment: "/narration" },
  { label: "Timeline", segment: "/timeline" },
  { label: "Preview", segment: "/preview" },
] as const;

export function ProjectTabs({ projectId }: { projectId: string }) {
  const pathname = usePathname();
  const base = `/projects/${projectId}`;

  return (
    <nav
      aria-label="Project views"
      className="p-glass flex items-center gap-1 overflow-x-auto px-2 py-1.5"
    >
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
            aria-current={isActive ? "page" : undefined}
            className={cn(
              "whitespace-nowrap rounded-full px-3 py-1.5 text-sm",
              "transition-all duration-fast ease-liquid",
              isActive
                ? "bg-accent/[0.12] text-accent shadow-[inset_0_0_0_1px_rgb(var(--p-accent)/0.25)]"
                : "text-mutedForeground hover:text-foreground hover:bg-white/[0.05]",
            )}
          >
            {tab.label}
          </Link>
        );
      })}
    </nav>
  );
}
