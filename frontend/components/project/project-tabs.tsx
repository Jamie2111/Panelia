"use client";

import type { Route } from "next";
import Link from "next/link";
import { usePathname } from "next/navigation";

import { cn } from "@/lib/utils";

const tabs = [
  { label: "Overview", segment: "" },
  { label: "Panel Editor", segment: "/editor" },
  { label: "Characters", segment: "/characters" },
  { label: "Narration", segment: "/narration" },
  { label: "Preview & Exports", segment: "/preview" }
] as const;

export function ProjectTabs({ projectId }: { projectId: string }) {
  const pathname = usePathname();
  const base = `/projects/${projectId}`;

  return (
    <div className="flex gap-1 border-b border-white/10 px-4">
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
              "relative px-3 py-2 text-sm transition",
              isActive
                ? "text-white"
                : "text-mutedForeground hover:text-white"
            )}
          >
            {tab.label}
            {isActive && (
              <span className="absolute inset-x-0 -bottom-px h-0.5 rounded-full bg-accent" />
            )}
          </Link>
        );
      })}
    </div>
  );
}
