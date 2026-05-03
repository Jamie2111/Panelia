import type { ReactNode } from "react";
import Link from "next/link";
import { Film, LayoutDashboard, PenSquare, Sparkles } from "lucide-react";

import { cn } from "@/lib/utils";
import { ProjectTabs } from "./project-tabs";

const navigation = [
  { href: "/", label: "Dashboard", icon: LayoutDashboard },
  { href: "/projects/new", label: "Create", icon: Sparkles },
  { href: "/exports", label: "Exports", icon: Film }
] as const;

export function AppShell({
  title,
  description,
  projectId,
  children
}: {
  title: string;
  description: string;
  projectId?: string;
  children: ReactNode;
}) {
  return (
    <div className="min-h-screen bg-background text-white">
      <div className="absolute inset-0 -z-10 bg-[radial-gradient(circle_at_top_left,_rgba(34,211,238,0.08),_transparent_28%),radial-gradient(circle_at_bottom_right,_rgba(251,191,36,0.06),_transparent_28%)]" />

      {/* Top bar */}
      <header className="sticky top-0 z-30 border-b border-white/8 bg-background/80 backdrop-blur-xl">
        <div className="mx-auto flex h-14 max-w-[1440px] items-center gap-6 px-4 lg:px-6">
          <Link href="/" className="flex items-center gap-2.5">
            <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-accent text-accent-foreground">
              <PenSquare className="h-3.5 w-3.5" />
            </div>
            <span className="font-display text-[15px]">Panelia</span>
          </Link>

          <nav className="hidden items-center gap-1 md:flex">
            {navigation.map((item) => {
              const Icon = item.icon;
              return (
                <Link
                  key={item.href}
                  href={item.href}
                  className={cn(
                    "flex items-center gap-2 rounded-lg px-3 py-1.5 text-sm text-mutedForeground transition hover:bg-white/8 hover:text-white"
                  )}
                >
                  <Icon className="h-3.5 w-3.5" />
                  {item.label}
                </Link>
              );
            })}
          </nav>
        </div>
      </header>

      <main className="mx-auto max-w-[1440px] px-4 py-6 lg:px-6">
        {/* Page header */}
        <div className="mb-1">
          <h1 className="font-display text-2xl tracking-tight">{title}</h1>
          <p className="mt-1 max-w-3xl text-sm text-mutedForeground">{description}</p>
        </div>

        {/* Project tabs */}
        {projectId && (
          <div className="mb-6 mt-4">
            <ProjectTabs projectId={projectId} />
          </div>
        )}
        {!projectId && <div className="mb-6" />}

        {children}
      </main>
    </div>
  );
}
