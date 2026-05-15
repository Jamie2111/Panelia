import type { ReactNode } from "react";
import Link from "next/link";
import { ArrowLeft, PenSquare } from "lucide-react";

import { ProjectTabs } from "@/components/project/project-tabs";

/**
 * EditorShell — full-bleed wrapper for the panel/timeline editing screens.
 * Keeps the icon rail + project tabs anchored while inner content scrolls.
 */
export function EditorShell({
  projectId,
  children,
}: {
  projectId: string;
  children: ReactNode;
}) {
  return (
    <div className="flex h-screen w-screen overflow-hidden text-foreground">
      {/* Icon rail */}
      <aside className="hidden w-14 shrink-0 flex-col items-center gap-3 border-r border-white/[0.06] bg-[rgb(var(--p-bg-base)/0.45)] backdrop-blur-liquid py-4 md:flex">
        <Link
          href="/"
          title="Dashboard"
          className="flex h-9 w-9 items-center justify-center rounded-xl bg-accent text-accent-foreground shadow-[0_0_18px_-3px_rgb(var(--p-accent)/0.7)] transition-transform duration-fast hover:-translate-y-px"
        >
          <PenSquare className="h-4 w-4" strokeWidth={2.4} />
        </Link>
        <Link
          href={`/projects/${projectId}`}
          title="Back to project"
          className="flex h-9 w-9 items-center justify-center rounded-xl text-mutedForeground transition-colors duration-fast hover:bg-white/[0.06] hover:text-foreground"
        >
          <ArrowLeft className="h-4 w-4" />
        </Link>
      </aside>

      {/* Main area */}
      <div className="flex min-w-0 flex-1 flex-col overflow-hidden px-4 pt-4 pb-2 lg:px-6">
        <ProjectTabs projectId={projectId} />
        <div className="mt-4 flex-1 min-h-0 overflow-hidden">{children}</div>
      </div>
    </div>
  );
}
