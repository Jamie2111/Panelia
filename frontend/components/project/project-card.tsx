import Link from "next/link";
import { ArrowRight, Clock3, Layers3, PlayCircle, Trash2, XCircle } from "lucide-react";

import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Card, CardDescription, CardTitle } from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import { getStageProgressMeta } from "@/lib/progress";
import { buildMediaUrl, formatRelativeDate, stageLabel } from "@/lib/utils";
import { ProjectSummary } from "@/lib/types";

export function ProjectCard({
  project,
  onCancelProject,
  onDeleteProject,
  actionBusy
}: {
  project: ProjectSummary;
  onCancelProject?: (project: ProjectSummary) => void;
  onDeleteProject?: (project: ProjectSummary) => void;
  actionBusy?: boolean;
}) {
  const stageStates = Object.values(project.stage_states ?? {}).filter(
    (stage): stage is NonNullable<(typeof project.stage_states)[keyof typeof project.stage_states]> =>
      Boolean(stage && typeof stage === "object")
  );
  const activeStage = stageStates.find((stage) => stage.status === "running" || stage.status === "needs_review");
  const activeStageMeta = activeStage ? getStageProgressMeta(project, activeStage.stage) : null;
  const sourceTypeLabel = String(project.source_type ?? "source").replace(/_/g, " ");
  const mangaTitle = project.chapter_metadata?.manga_title || "Untitled manga";
  const panelCount = Number(project.panel_count ?? 0);
  const keptPanelCount = Number(project.kept_panel_count ?? 0);
  const activeJobsCount = Array.isArray(project.active_jobs) ? project.active_jobs.length : 0;

  return (
    <Card className="h-full overflow-hidden p-0">
      <div className="grid min-h-[15rem] grid-cols-[7.25rem_minmax(0,1fr)]">
        <div className="relative overflow-hidden border-r border-white/8 bg-white/5">
          {project.thumbnail_url ? (
            // eslint-disable-next-line @next/next/no-img-element
            <img
              src={buildMediaUrl(project.thumbnail_url)}
              alt={project.name}
              loading="lazy"
              decoding="async"
              className="h-full w-full object-cover object-top"
            />
          ) : (
            <div className="flex h-full items-center justify-center bg-[radial-gradient(circle_at_center,_rgba(34,211,238,0.15),_transparent_55%)]">
              <PlayCircle className="h-8 w-8 text-accent/70" />
            </div>
          )}
        </div>
        <div className="flex min-w-0 flex-col gap-3 p-4">
          <div className="flex items-start justify-between gap-3">
            <div className="min-w-0">
              <CardTitle className="line-clamp-2 text-lg leading-tight">{project.name}</CardTitle>
              <CardDescription className="mt-1 line-clamp-2 text-sm">
                {mangaTitle} • Updated {formatRelativeDate(project.updated_at)}
              </CardDescription>
            </div>
            <Badge className="shrink-0">{sourceTypeLabel}</Badge>
          </div>

          <div className="flex flex-wrap gap-2">
            <Badge className="bg-brand-cyan/10 text-brand-cyan">{project.page_count} pages</Badge>
            <Badge className="bg-brand-amber/10 text-brand-amber">{keptPanelCount} kept</Badge>
            {activeStage ? <Badge className="bg-brand-rose/10 text-brand-rose">{stageLabel(activeStage.stage)}</Badge> : null}
          </div>

          <div className="grid grid-cols-3 gap-2 text-xs text-mutedForeground">
            <div className="rounded-2xl border border-white/8 bg-white/5 px-3 py-2">
              <p className="flex items-center gap-1 text-white">
                <Layers3 className="h-3.5 w-3.5 text-accent" />
                Panels
              </p>
              <p className="mt-1 text-sm">{panelCount}</p>
            </div>
            <div className="rounded-2xl border border-white/8 bg-white/5 px-3 py-2">
              <p className="flex items-center gap-1 text-white">
                <Clock3 className="h-3.5 w-3.5 text-accent" />
                Jobs
              </p>
              <p className="mt-1 text-sm">{activeJobsCount}</p>
            </div>
            <div className="rounded-2xl border border-white/8 bg-white/5 px-3 py-2">
              <p className="text-white">Export</p>
              <p className="mt-1 truncate text-sm" title={project.latest_video?.name ?? "Not rendered"}>
                {project.latest_video?.name ?? "Not rendered"}
              </p>
            </div>
          </div>

          {activeStage ? (
            <div>
              <div className="mb-2 flex items-center justify-between text-[10px] uppercase tracking-[0.18em] text-mutedForeground">
                <span>{stageLabel(activeStage.stage)}</span>
                <span>{activeStageMeta?.progress ?? Math.round(activeStage.progress)}%</span>
              </div>
              <Progress value={activeStageMeta?.progress ?? activeStage.progress} />
              {activeStageMeta?.message ? (
                <p className="mt-1 line-clamp-2 text-[11px] text-mutedForeground">
                  {activeStageMeta.message}
                </p>
              ) : activeStageMeta?.stateLabel ? (
                <p className="mt-1 line-clamp-1 text-[11px] text-mutedForeground">{activeStageMeta.stateLabel}</p>
              ) : null}
            </div>
          ) : null}

          <div className="mt-auto grid grid-cols-2 gap-2">
            <Link
              href={`/projects/${project.id}`}
              className="inline-flex items-center justify-center gap-2 rounded-full bg-accent px-3 py-2 text-sm font-medium text-accent-foreground"
            >
              Open
              <ArrowRight className="h-4 w-4" />
            </Link>
            <Link
              href={`/projects/${project.id}/editor`}
              className="inline-flex items-center justify-center gap-2 rounded-full border border-white/10 px-3 py-2 text-sm text-white"
            >
              Panels
            </Link>
          </div>
          <div className="flex flex-wrap gap-2">
            {activeJobsCount ? (
              <Button variant="outline" size="sm" onClick={() => onCancelProject?.(project)} disabled={actionBusy}>
                <XCircle className="h-4 w-4" />
                Cancel
              </Button>
            ) : null}
            <Button variant="ghost" size="sm" onClick={() => onDeleteProject?.(project)} disabled={actionBusy || activeJobsCount > 0}>
              <Trash2 className="h-4 w-4" />
              Delete
            </Button>
          </div>
        </div>
      </div>
    </Card>
  );
}
