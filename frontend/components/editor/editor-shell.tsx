import type { ReactNode } from "react";
import Link from "next/link";
import { ArrowLeft, PenSquare } from "lucide-react";

import { ProjectTabs } from "@/components/project/project-tabs";

export function EditorShell({
  projectId,
  children
}: {
  projectId: string;
  children: ReactNode;
}) {
  return (
    <div className="flex h-screen w-screen overflow-hidden bg-background text-white">
      <div className="absolute inset-0 -z-10 bg-[radial-gradient(circle_at_top_left,_rgba(34,211,238,0.14),_transparent_28%),radial-gradient(circle_at_bottom_right,_rgba(251,191,36,0.12),_transparent_28%),linear-gradient(180deg,_rgba(15,23,42,0.1),_transparent)]" />

      {/* Icon rail */}
      <aside className="hidden w-12 shrink-0 flex-col items-center gap-4 border-r border-white/10 bg-white/5 py-3 md:flex">
        <Link href="/" title="Dashboard" className="flex h-9 w-9 items-center justify-center rounded-xl bg-accent text-accent-foreground transition hover:opacity-80">
          <PenSquare className="h-4 w-4" />
        </Link>
        <Link href={`/projects/${projectId}`} title="Back to project" className="flex h-9 w-9 items-center justify-center rounded-xl text-mutedForeground transition hover:bg-white/10 hover:text-white">
          <ArrowLeft className="h-4 w-4" />
        </Link>
      </aside>

      {/* Main area */}
      <div className="flex min-w-0 flex-1 flex-col overflow-hidden">
        <ProjectTabs projectId={projectId} />
        {children}
      </div>
    </div>
  );
}
