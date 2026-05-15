import type { ReactNode } from "react";
import Link from "next/link";
import { Film, LayoutDashboard, PenSquare, Settings, Sparkles } from "lucide-react";

import { cn } from "@/lib/utils";
import { PageHeader, type PageHeaderViewLink } from "@/components/ui/page-header";

/**
 * AppShell - the persistent chrome around every page.
 *
 * Top bar (compact, sticky, blurred) is unchanged. The page-content
 * heading is now driven by props that map 1:1 to PageHeader, so every
 * page in the app uses the same hero pattern the timeline editor
 * established. Pages can also pass `hero` to slot in a fully custom
 * header - useful for the timeline page which already had its own
 * rich header before this refactor.
 */

const navigation = [
  { href: "/", label: "Studio", icon: LayoutDashboard },
  { href: "/projects/new", label: "Create", icon: Sparkles },
  { href: "/exports", label: "Exports", icon: Film },
  { href: "/settings/channel", label: "Channel", icon: Settings },
] as const;

export function AppShell({
  title,
  description,
  projectId,
  children,
  contentClassName,
  // ── Hero customization ─────────────────────────────────────────────
  hero,
  breadcrumb,
  meta,
  views,
  actions,
}: {
  title: ReactNode;
  description?: ReactNode;
  projectId?: string;
  children?: ReactNode;
  contentClassName?: string;
  /** Replace the auto-generated PageHeader with arbitrary markup. */
  hero?: ReactNode;
  /** Forwarded to PageHeader. */
  breadcrumb?: { href: string; label: string };
  meta?: ReactNode;
  views?: PageHeaderViewLink[];
  actions?: ReactNode;
}) {
  return (
    <div className="min-h-screen text-foreground">
      {/* Sticky top bar - heavy backdrop blur, hairline divider */}
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
          "mx-auto max-w-[1680px] px-6 lg:px-10 py-8 lg:py-10 space-y-6",
          contentClassName,
        )}
      >
        {hero ?? (
          <PageHeader
            breadcrumb={breadcrumb}
            title={title}
            subtitle={description}
            meta={meta}
            views={projectId ? views : undefined}
            actions={actions}
          />
        )}

        {children}
      </main>
    </div>
  );
}
