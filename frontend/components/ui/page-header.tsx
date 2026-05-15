"use client";

/**
 * PageHeader - the one hero header pattern used by every page in the app.
 *
 * Distilled from the timeline editor page so that the whole product
 * speaks the same visual language. Every page in the pipeline (project
 * list, project detail, narration, editor, timeline, exports, etc.)
 * renders this at the top instead of AppShell's basic title block.
 *
 * Layout, top-to-bottom:
 *
 *   ← Breadcrumb caps  (tracked, 11px, hover → accent)
 *   Hero title         (gradient-clipped 32-40px display weight)
 *   Status pill row    (kept count · saved time · cost · auto-run, etc)
 *
 *   ────────────────── (right-aligned on same line as title)
 *                                                    [View switcher]
 *
 * The view-switcher is the "Narration / Panels / Timeline / Preview"
 * link row that used to live in ProjectTabs. Folding it into the page
 * header brings every project page in line with the timeline page (which
 * already had this pattern) and removes the visual mode-switch when you
 * navigate between project sub-pages.
 *
 * The whole component is presentational - it owns no state, and any
 * dynamic content (pills, links) comes in as props.
 */

import * as React from "react";
import Link from "next/link";
import { cn } from "@/lib/utils";

export interface PageHeaderBreadcrumb {
  href: string;
  label: string;
}

export interface PageHeaderViewLink {
  href: string;
  label: string;
  /** If true, the link renders as the active/current view. */
  active?: boolean;
}

interface PageHeaderProps {
  /** Optional breadcrumb above the title (e.g. "← All projects"). */
  breadcrumb?: PageHeaderBreadcrumb;
  /** The headline. Renders with the display gradient. */
  title: React.ReactNode;
  /** Optional subtitle under the title (the "manga title · created N ago" line). */
  subtitle?: React.ReactNode;
  /**
   * Status pills displayed in a flex-wrap row under the title.
   * Pass `Badge` components or raw .p-pill spans.
   */
  meta?: React.ReactNode;
  /**
   * Right-side action area. Typical: a view-switcher (next-page links)
   * + a primary CTA. Aligns to the bottom of the title block on wide
   * screens, wraps under on narrow.
   */
  actions?: React.ReactNode;
  /**
   * View-switcher links rendered as a pill row in the actions area.
   * Convenience over `actions`; you can use both.
   */
  views?: PageHeaderViewLink[];
  /** Optional className passthrough. */
  className?: string;
}

export function PageHeader({
  breadcrumb,
  title,
  subtitle,
  meta,
  actions,
  views,
  className,
}: PageHeaderProps) {
  return (
    <header
      className={cn(
        "flex flex-wrap items-end justify-between gap-4",
        className,
      )}
    >
      <div className="min-w-0">
        {breadcrumb && (
          <Link
            href={breadcrumb.href as never}
            className="text-[11px] uppercase tracking-[0.18em] text-[rgb(var(--p-hint))] hover:text-[rgb(var(--p-accent))] transition-colors duration-fast inline-flex items-center gap-1"
          >
            <span aria-hidden>←</span> {breadcrumb.label}
          </Link>
        )}
        <h1
          className={cn(
            "leading-tight tracking-tight font-semibold",
            // Slightly smaller than the timeline page's 48px so it still
            // breathes on narrower content like the Studio list. Hero feel,
            // not billboard.
            "mt-2 text-3xl md:text-[40px]",
          )}
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
        {subtitle && (
          <p className="mt-2 text-sm text-mutedForeground leading-relaxed max-w-2xl">
            {subtitle}
          </p>
        )}
        {meta && (
          <div className="mt-3 flex flex-wrap items-center gap-x-3 gap-y-1.5">
            {meta}
          </div>
        )}
      </div>

      {(views || actions) && (
        <div className="flex flex-wrap items-center gap-2 shrink-0">
          {views && views.length > 0 && (
            <nav
              aria-label="Views"
              className="p-glass flex items-center gap-1 px-1.5 py-1.5"
            >
              {views.map((view) => (
                <Link
                  key={view.href}
                  href={view.href as never}
                  aria-current={view.active ? "page" : undefined}
                  className={cn(
                    "whitespace-nowrap rounded-full px-3 py-1.5 text-sm",
                    "transition-all duration-fast ease-liquid",
                    view.active
                      ? "bg-accent/[0.12] text-accent shadow-[inset_0_0_0_1px_rgb(var(--p-accent)/0.25)]"
                      : "text-mutedForeground hover:text-foreground hover:bg-white/[0.05]",
                  )}
                >
                  {view.label}
                </Link>
              ))}
            </nav>
          )}
          {actions}
        </div>
      )}
    </header>
  );
}
