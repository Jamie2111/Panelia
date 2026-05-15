/**
 * project-views — the canonical list of project sub-pages and how they
 * render in the page-header view-switcher.
 *
 * Used by every project page so the navigation feels identical no matter
 * which sub-page the user is on. The previous ProjectTabs strip rendered
 * the same idea as a separate row; folding it into the header keeps the
 * timeline page's IA consistent across the whole app.
 */

import type { PageHeaderViewLink } from "@/components/ui/page-header";

/** Internal route segment + display label for one project sub-view. */
interface ProjectView {
  /** "" for the overview page; "/timeline", "/narration", etc. for others. */
  segment: string;
  label: string;
}

/**
 * The default canonical view list. We deliberately keep it short — only
 * the surfaces a user opens during a real edit pass:
 *   Overview · Panels · Narration · Timeline · Preview
 *
 * Characters / Dictionary / Portraits are still reachable from the
 * project detail page's stage controls but they're not first-class enough
 * to fight for header space.
 */
const DEFAULT_VIEWS: ProjectView[] = [
  { segment: "", label: "Overview" },
  { segment: "/editor", label: "Panels" },
  { segment: "/narration", label: "Narration" },
  { segment: "/timeline", label: "Timeline" },
  { segment: "/preview", label: "Preview" },
];

/**
 * Build the PageHeader-ready `views` array for a given project page.
 *
 * @param projectId   The current project's id.
 * @param activeSegment The current page's segment ("" for overview,
 *                      "/timeline" etc.). Used to mark the active pill.
 */
export function buildProjectViews(
  projectId: string,
  activeSegment: string,
): PageHeaderViewLink[] {
  const base = `/projects/${projectId}`;
  return DEFAULT_VIEWS.map((view) => ({
    href: `${base}${view.segment}`,
    label: view.label,
    active: view.segment === activeSegment,
  }));
}
