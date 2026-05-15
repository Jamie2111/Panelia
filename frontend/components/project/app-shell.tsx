import type { ReactNode } from "react";
import Link from "next/link";
import { Film, LayoutDashboard, PenSquare, Sparkles } from "lucide-react";

import { cn } from "@/lib/utils";
import { ProjectTabs } from "./project-tabs";

/**
 * AppShell — the persistent chrome around every page.
 *
 * Notion-style top bar (compact, sticky, blurred) + ambient orbs from
 * globals.css drifting behind everything. The shell never carries content
 * of its own — page content lives in `children`.
 */

const navigation = [
  { href: "/", label: "Studio", icon: LayoutDashboard },
  { href: "/projects/new", label: "Create", icon: Sparkles },
  { href: "/exports", label: "Exports", icon: Film }
] as const;

export function AppShell({
  title,
  description,
  projectId,
  children,
  contentClassName,
}: {
  title: string;
  description?: string;
  projectId?: string;
  children: ReactNode;
  contentClassName?: string;
}) {
  return (
    <div className="min-h-screen text-foreground">
      {/* Sticky top bar — heavy backdrop blur, hairline divider */}
      <header className="sticky top-0 z-40 border-b border-white/[0.06] bg-[rgb(var(--p-bg-base)/0.65)] backdrop-blur-liquid">
        <div className="mx-auto flex h-14 max-w-[1680px] items-center gap-6 px-6 lg:px-10">
          <Link
            href="/"
            className="flex items-center gap-2.5 transition-opacity duration-fast hover:opacity-90"
          >
            <div className="flex h-7 w-7 items-center justify-center rounded-[10px] bg-accent text-accent-foreground shadow-[0_0_18px_-3px_rgb(var(--p-accent)/0.7)]">
              <PenSquare className="h-3.5 w-3.5" strokeWidth={2.4} />
            </div>
            <span className="font-display text-[15px] tracking-tightish">Panelia</span>
          </Link>

          <nav className="hidden items-center gap-1 md:flex">
            {navigation.map((item) => {
              const Icon = item.icon;
              return (
                <Link
                  key={item.href}
                  href={item.href}
                  className={cn(
                    "flex items-center gap-2 rounded-full px-3 py-1.5 text-sm",
                    "text-mutedForeground transition-colors duration-fast ease-liquid",
                    "hover:bg-white/[0.06] hover:text-foreground"
                  )}
                >
                  <Icon className="h-3.5 w-3.5" strokeWidth={2} />
                  {item.label}
                </Link>
              );
            })}
          </nav>
        </div>
      </header>

      <main
        className={cn(
          "mx-auto max-w-[1680px] px-6 lg:px-10 py-8 lg:py-10",
          contentClassName
        )}
      >
        {/* Page heading — display weight, gradient text, generous margin */}
        <div className="mb-2">
          <h1
            className="font-display text-3xl md:text-[40px] leading-tight tracking-tight"
            style={{
              backgroundImage:
                "linear-gradient(180deg, rgb(var(--p-text)) 0%, rgb(var(--p-muted)) 110%)",
              WebkitBackgroundClip: "text",
              backgroundClip: "text",
              color: "transparent",
            }}
          >
            {title}
          </h1>
          {description && (
            <p className="mt-2 max-w-2xl text-sm text-mutedForeground leading-relaxed">
              {description}
            </p>
          )}
        </div>

        {projectId && (
          <div className="mt-6 mb-7">
            <ProjectTabs projectId={projectId} />
          </div>
        )}
        {!projectId && <div className="mb-7" />}

        {children}
      </main>
    </div>
  );
}
