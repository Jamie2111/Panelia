"use client";

import Link from "next/link";
import type { Route } from "next";
import { useParams } from "next/navigation";
import { FormEvent, useEffect, useState } from "react";
import { Check, Edit3, LoaderCircle, RefreshCw, X } from "lucide-react";

import { AppShell } from "@/components/project/app-shell";
import { StageTimeline } from "@/components/project/stage-timeline";
import { Button } from "@/components/ui/button";
import { Card, CardDescription, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Input } from "@/components/ui/input";
import { Progress } from "@/components/ui/progress";
import { PipelineBlock } from "@/components/ui/pipeline-block";
import { WhileYouWereAway } from "@/components/ui/while-you-were-away";
import { PublishBundleCard, type PublishBundle } from "@/components/ui/publish-bundle";
import { api } from "@/lib/api";
import { buildProjectViews } from "@/lib/project-views";
import { formatProgressPercent, getStageProgressMeta } from "@/lib/progress";
import { shortStageLabel, toPipelineDisplay } from "@/lib/pipeline-messages";
import { estimateProjectCost } from "@/lib/cost-estimate";
import type { PipelineStage, ProjectSummary } from "@/lib/types";
import { useAdaptivePolling } from "@/lib/use-adaptive-polling";
import { buildMediaUrl, formatRelativeDate } from "@/lib/utils";

function safeDimension(value: number | string | null | undefined, fallback: number) {
  const numeric = Number(value);
  return Number.isFinite(numeric) && numeric > 0 ? Math.round(numeric) : fallback;
}

function baseResolution(width: number, height: number) {
  const safeWidth = safeDimension(width, 1920);
  const safeHeight = safeDimension(height, 1080);
  return `${Math.max(safeWidth, safeHeight)}×${Math.min(safeWidth, safeHeight)}`;
}

/**
 * Project overview page.
 *
 * Three vertical sections:
 *   1. Hero - project image, name (inline-editable), key metadata,
 *      and the canonical PipelineBlock so users always see "what now".
 *   2. Operations - pipeline controls (rerun/cancel/rewind), and the
 *      auto-run toggle.
 *   3. Sidebar - vertical StageTimeline, active jobs list.
 *
 * Everything pulls from the same liquid-glass primitives so the page
 * feels like one continuous workspace rather than a CRUD form.
 */
export default function ProjectOverviewPage() {
  const params = useParams<{ projectId: string }>();
  const projectId = Array.isArray(params.projectId) ? params.projectId[0] : params.projectId;
  const [project, setProject] = useState<ProjectSummary | null>(null);
  const [bundle, setBundle] = useState<PublishBundle | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [stageActionBusy, setStageActionBusy] = useState<PipelineStage | null>(null);
  const [rewindBusy, setRewindBusy] = useState<PipelineStage | null>(null);
  const [cancelJobBusy, setCancelJobBusy] = useState<string | null>(null);
  const [refreshBusy, setRefreshBusy] = useState(false);
  const [pipelineToggleBusy, setPipelineToggleBusy] = useState(false);
  const [editingName, setEditingName] = useState(false);
  const [draftName, setDraftName] = useState("");
  const [renameBusy, setRenameBusy] = useState(false);

  async function load() {
    if (!projectId) return;
    try {
      const next = await api.getProjectSummary(projectId);
      setProject(next);
      setError(null);
      // Fetch the publish bundle in parallel; it'll be null until the
      // youtube_bundle stage finishes.
      try {
        const fetched = await api.getYouTubeBundle(projectId);
        if (fetched) setBundle(fetched as PublishBundle);
      } catch {
        /* non-fatal - bundle hasn't been generated yet */
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unable to load project.");
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    if (!projectId) return;
    load();
  }, [projectId]);

  useEffect(() => {
    if (project && !editingName) {
      setDraftName(project.name);
    }
  }, [editingName, project]);

  useAdaptivePolling(load, {
    enabled: Boolean(projectId),
    active: Boolean(project?.active_jobs.length),
    activeMs: 10000,
    idleMs: 30000,
    hiddenMs: 120000,
    deps: [projectId]
  });

  if (loading) {
    return (
      <AppShell
        title="Loading project"
        description="Fetching project details and queue status."
        projectId={projectId}
      >
        <Card padded="md" className="flex items-center gap-3">
          <LoaderCircle className="h-4 w-4 animate-spin text-accent" />
          <span className="text-sm text-mutedForeground">Loading project details…</span>
        </Card>
      </AppShell>
    );
  }

  if (!project || error) {
    return (
      <AppShell
        title="Project unavailable"
        description="Panelia couldn't load the requested project."
        projectId={projectId}
      >
        <Card padded="lg" className="p-edge-fail">
          <CardTitle>Unable to load project</CardTitle>
          <CardDescription className="mt-2">
            {error ?? "The project may have been moved or deleted."}
          </CardDescription>
        </Card>
      </AppShell>
    );
  }

  const activeJobs = project.active_jobs;
  const activeStageJobs = (stage: PipelineStage) =>
    activeJobs.filter((job) => job.stage === stage && (job.status === "queued" || job.status === "running"));

  async function toggleStage(stage: PipelineStage) {
    if (!project) return;
    setStageActionBusy(stage);
    try {
      const jobs = activeStageJobs(stage);
      if (jobs.length) {
        await Promise.all(jobs.map((job) => api.cancelJob(project.id, job.id)));
      } else {
        await api.queueStage(
          project.id,
          stage,
          stage === "narration_generation" ? { force_quality_bypass: true } : {}
        );
      }
      setProject(await api.getProjectSummary(project.id));
    } finally {
      setStageActionBusy(null);
    }
  }

  async function submitRename(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const currentProject = project;
    const name = draftName.trim();
    if (!currentProject || !name || name === currentProject.name || renameBusy) {
      return;
    }
    setRenameBusy(true);
    try {
      const renamed = await api.renameProject(currentProject.id, name);
      setProject(renamed);
      setEditingName(false);
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unable to rename the project.");
    } finally {
      setRenameBusy(false);
    }
  }

  function cancelRename() {
    if (project) setDraftName(project.name);
    setEditingName(false);
  }

  return (
    <AppShell
      title={project.name}
      description={`${project.chapter_metadata.manga_title || "Sequential-art project"} · created ${formatRelativeDate(project.created_at)}`}
      projectId={projectId}
      breadcrumb={{ href: "/", label: "All projects" }}
      views={buildProjectViews(projectId, "")}
      meta={(
        <>
          <Badge>{project.page_count} pages</Badge>
          <Badge>{project.kept_panel_count} panels</Badge>
          {project.latest_video ? <Badge tone="info">Rendered</Badge> : null}
          {project.active_jobs.length ? (
            <Badge tone="accent" dot pulse>
              {project.active_jobs.length} running
            </Badge>
          ) : null}
        </>
      )}
    >
      <div className="space-y-6">
        {/* While-you-were-away - only fires when state changed */}
        <WhileYouWereAway project={project as any} />

        {/* Pipeline block - answers "what's happening / what next" */}
        <PipelineBlock
          stageStates={project.stage_states}
          cost={estimateProjectCost(project as any)}
        />

        {/* The Publish Studio (title / description / thumbnail editor)
            now lives on the Preview tab so it sits next to the actual
            rendered video. The overview keeps a compact hint that the
            bundle is ready and links across. */}
        {project.stage_states.youtube_bundle?.status === "completed" ? (
          <Card>
            <div className="flex flex-wrap items-center justify-between gap-3">
              <div>
                <CardTitle className="text-base">Your YouTube bundle is ready</CardTitle>
                <CardDescription className="mt-1">
                  Edit the title, description, and thumbnails in the Preview tab
                  alongside the final video.
                </CardDescription>
              </div>
              <Link
                href={`/projects/${projectId}/preview` as Route}
                className="inline-flex items-center gap-2 rounded-full border border-white/[0.08] bg-white/[0.04] px-4 py-2 text-sm font-medium text-foreground transition hover:bg-white/[0.08]"
              >
                Open publish studio
              </Link>
            </div>
          </Card>
        ) : null}

        <div className="grid gap-6 xl:grid-cols-[1.05fr_0.95fr]">
          <div className="space-y-6">
            {/* Hero card */}
            <Card padded="none" className="overflow-hidden">
              <div className="grid gap-0 sm:grid-cols-[220px_minmax(0,1fr)]">
                <div className="self-stretch border-b border-white/[0.06] bg-white/[0.04] sm:border-b-0 sm:border-r">
                  <div className="h-full min-h-[260px] w-full overflow-hidden">
                    {project.thumbnail_url ? (
                      // eslint-disable-next-line @next/next/no-img-element
                      <img
                        src={buildMediaUrl(project.thumbnail_url)}
                        alt={project.name}
                        loading="lazy"
                        decoding="async"
                        className="h-full w-full object-cover object-top"
                      />
                    ) : null}
                  </div>
                </div>
                <div className="min-w-0 space-y-4 p-6">
                  <div>
                    {editingName ? (
                      <form onSubmit={submitRename} className="flex items-center gap-2">
                        <Input
                          value={draftName}
                          autoFocus
                          maxLength={160}
                          onChange={(event) => setDraftName(event.target.value)}
                          onKeyDown={(event) => {
                            if (event.key === "Escape") {
                              event.preventDefault();
                              cancelRename();
                            }
                          }}
                          className="h-10 max-w-lg rounded-xl px-3 text-base"
                          aria-label="Project name"
                        />
                        <Button
                          type="submit"
                          size="sm"
                          className="h-10 w-10 shrink-0 px-0"
                          disabled={!draftName.trim() || draftName.trim() === project.name || renameBusy}
                          title="Save project name"
                        >
                          {renameBusy ? (
                            <LoaderCircle className="h-4 w-4 animate-spin" />
                          ) : (
                            <Check className="h-4 w-4" />
                          )}
                        </Button>
                        <Button
                          type="button"
                          variant="ghost"
                          size="sm"
                          className="h-10 w-10 shrink-0 px-0"
                          onClick={cancelRename}
                          disabled={renameBusy}
                          title="Cancel rename"
                        >
                          <X className="h-4 w-4" />
                        </Button>
                      </form>
                    ) : (
                      <div className="flex min-w-0 items-center gap-2 group">
                        <CardTitle className="truncate text-xl md:text-2xl">{project.name}</CardTitle>
                        <Button
                          type="button"
                          variant="ghost"
                          size="sm"
                          className="h-8 w-8 shrink-0 px-0 opacity-0 transition-opacity duration-fast group-hover:opacity-100"
                          onClick={() => setEditingName(true)}
                          title="Rename project"
                        >
                          <Edit3 className="h-4 w-4" />
                        </Button>
                      </div>
                    )}
                  </div>
                  <div>
                    <p className="text-sm text-mutedForeground leading-relaxed">
                      <span className="text-foreground">
                        {project.chapter_metadata.chapter_title || "Imported chapter"}
                      </span>{" "}
                      · {project.chapter_metadata.manga_title || "Unknown series"}
                    </p>
                    <div className="mt-3 flex flex-wrap gap-2">
                      <Badge>{project.page_count} pages</Badge>
                      <Badge>{project.kept_panel_count} panels kept</Badge>
                      {project.latest_video ? <Badge tone="info">Rendered</Badge> : null}
                    </div>
                  </div>
                  <div className="grid gap-2 sm:grid-cols-3 pt-2">
                    {[
                      { label: "Voice", value: project.voice_config.voice },
                      {
                        label: "Render",
                        value: `${baseResolution(project.video_config.width, project.video_config.height)} · ${project.video_config.orientation}`,
                      },
                      { label: "Exports", value: project.latest_video ? "1+" : "0" }
                    ].map((item) => (
                      <div
                        key={item.label}
                        className="rounded-2xl border border-white/[0.06] bg-white/[0.03] px-3 py-2.5"
                      >
                        <p className="text-[10px] uppercase tracking-track text-mutedForeground">
                          {item.label}
                        </p>
                        <p className="mt-1 truncate text-[13px] font-medium text-foreground">
                          {item.value}
                        </p>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            </Card>

            {/* Pipeline controls */}
            <Card padded="md">
              <div className="flex items-center justify-between gap-3">
                <div>
                  <CardTitle className="text-base">Pipeline controls</CardTitle>
                  <CardDescription className="mt-1">
                    Rerun a stage or rewind to an earlier step.
                  </CardDescription>
                </div>
                <Button
                  variant="ghost"
                  size="sm"
                  disabled={refreshBusy}
                  onClick={async () => {
                    setRefreshBusy(true);
                    try {
                      setProject(await api.getProjectSummary(project.id));
                    } finally {
                      setRefreshBusy(false);
                    }
                  }}
                >
                  <RefreshCw className={`h-3.5 w-3.5 ${refreshBusy ? "animate-spin" : ""}`} />
                  Refresh
                </Button>
              </div>

              {/* Auto-run toggle */}
              <label className="mt-4 flex cursor-pointer items-start justify-between gap-4 rounded-2xl border border-white/[0.06] bg-white/[0.03] px-4 py-3 transition-colors duration-fast hover:bg-white/[0.05]">
                <div className="space-y-1">
                  <p className="text-sm font-medium text-foreground">Run end to end automatically</p>
                  <p className="text-xs text-mutedForeground leading-relaxed">
                    Continue from import to video export whenever Panelia can proceed without a manual fix.
                  </p>
                </div>
                <span className="flex items-center gap-2 pt-1">
                  {pipelineToggleBusy ? (
                    <LoaderCircle className="h-4 w-4 animate-spin text-accent" />
                  ) : null}
                  <input
                    type="checkbox"
                    className="h-4 w-4 rounded border border-white/[0.18] bg-transparent accent-[rgb(var(--p-accent))]"
                    checked={Boolean(project.pipeline_config.auto_run_end_to_end)}
                    disabled={pipelineToggleBusy}
                    onChange={async (event) => {
                      setPipelineToggleBusy(true);
                      try {
                        setProject(
                          await api.updateProjectSettings(project.id, {
                            pipeline_config: {
                              ...project.pipeline_config,
                              auto_run_end_to_end: event.target.checked
                            }
                          })
                        );
                      } finally {
                        setPipelineToggleBusy(false);
                      }
                    }}
                  />
                </span>
              </label>

              {/* Content-safety toggle */}
              <label className="mt-3 flex cursor-pointer items-start justify-between gap-4 rounded-2xl border border-white/[0.06] bg-white/[0.03] px-4 py-3 transition-colors duration-fast hover:bg-white/[0.05]">
                <div className="space-y-1">
                  <p className="text-sm font-medium text-foreground">YouTube content safety</p>
                  <p className="text-xs text-mutedForeground leading-relaxed">
                    Auto-blur partial nudity / intimate scenes and skip explicit panels so the rendered video stays monetizable. Turn off for adult-only channels.
                  </p>
                </div>
                <span className="flex items-center gap-2 pt-1">
                  {pipelineToggleBusy ? (
                    <LoaderCircle className="h-4 w-4 animate-spin text-accent" />
                  ) : null}
                  <input
                    type="checkbox"
                    className="h-4 w-4 rounded border border-white/[0.18] bg-transparent accent-[rgb(var(--p-accent))]"
                    checked={Boolean(
                      (project.pipeline_config as any).content_safety_enabled ?? true,
                    )}
                    disabled={pipelineToggleBusy}
                    onChange={async (event) => {
                      setPipelineToggleBusy(true);
                      try {
                        setProject(
                          await api.updateProjectSettings(project.id, {
                            pipeline_config: {
                              ...project.pipeline_config,
                              content_safety_enabled: event.target.checked,
                            } as any,
                          })
                        );
                      } finally {
                        setPipelineToggleBusy(false);
                      }
                    }}
                  />
                </span>
              </label>

              <div className="mt-4 grid gap-2 sm:grid-cols-2 lg:grid-cols-3">
                {[
                  { label: "Detect panels", stage: "panel_detection" as const },
                  { label: "Prepare characters", stage: "character_review" as const },
                  { label: "Generate script", stage: "script_generation" as const },
                  { label: "Generate audio", stage: "narration_generation" as const },
                  { label: "Render video", stage: "video_rendering" as const }
                ].map((item) => (
                  <Button
                    key={item.label}
                    variant="secondary"
                    size="sm"
                    className="w-full text-xs"
                    onClick={() => toggleStage(item.stage)}
                    disabled={stageActionBusy !== null && stageActionBusy !== item.stage}
                  >
                    {stageActionBusy === item.stage ? (
                      <>
                        <LoaderCircle className="h-3.5 w-3.5 animate-spin" />
                        Working…
                      </>
                    ) : activeStageJobs(item.stage).length ? (
                      `Cancel ${item.label.toLowerCase()}`
                    ) : (
                      item.label
                    )}
                  </Button>
                ))}
              </div>

              <div className="mt-3 grid gap-2 sm:grid-cols-2 lg:grid-cols-3">
                {[
                  { label: "Rewind to import", stage: "ingestion" as const },
                  { label: "Rewind to detection", stage: "panel_detection" as const },
                  { label: "Rewind to review", stage: "panel_review" as const },
                  { label: "Rewind to characters", stage: "character_review" as const },
                  { label: "Rewind to script", stage: "script_generation" as const },
                  { label: "Rewind to audio", stage: "narration_generation" as const }
                ].map((item) => (
                  <Button
                    key={item.label}
                    variant="outline"
                    size="sm"
                    className="w-full text-xs"
                    disabled={rewindBusy !== null}
                    onClick={async () => {
                      setRewindBusy(item.stage);
                      try {
                        setProject(await api.rewindProject(project.id, item.stage));
                        setError(null);
                      } catch (err) {
                        setError(err instanceof Error ? err.message : "Unable to rewind this project.");
                      } finally {
                        setRewindBusy(null);
                      }
                    }}
                  >
                    {rewindBusy === item.stage ? <LoaderCircle className="h-3 w-3 animate-spin" /> : null}
                    {rewindBusy === item.stage ? "Rewinding…" : item.label}
                  </Button>
                ))}
              </div>
            </Card>

            {/* Active jobs */}
            {activeJobs.length ? (
              <Card padded="md">
                <CardTitle className="text-base">Active jobs</CardTitle>
                <div className="mt-4 space-y-3">
                  {activeJobs.map((job) => {
                    const progressMeta = getStageProgressMeta(project, job.stage);
                    const stageState = project.stage_states[job.stage];
                    const display = stageState ? toPipelineDisplay(stageState) : null;
                    return (
                      <div
                        key={job.id}
                        className="space-y-2.5 rounded-2xl border border-white/[0.06] bg-white/[0.03] px-4 py-3"
                      >
                        <div className="flex items-center justify-between gap-3">
                          <div className="flex min-w-0 items-center gap-2.5">
                            <LoaderCircle className="h-4 w-4 shrink-0 animate-spin text-accent" />
                            <p className="text-sm font-medium text-foreground">
                              {shortStageLabel(job.stage)}
                            </p>
                          </div>
                          <Button
                            variant="outline"
                            size="sm"
                            disabled={cancelJobBusy !== null}
                            onClick={async () => {
                              setCancelJobBusy(job.id);
                              try {
                                await api.cancelJob(project.id, job.id);
                                setProject(await api.getProjectSummary(project.id));
                              } finally {
                                setCancelJobBusy(null);
                              }
                            }}
                          >
                            {cancelJobBusy === job.id ? <LoaderCircle className="h-3.5 w-3.5 animate-spin" /> : null}
                            {cancelJobBusy === job.id ? "Cancelling…" : "Cancel"}
                          </Button>
                        </div>
                        <p className="text-xs text-mutedForeground leading-relaxed">
                          {display?.sentence || progressMeta.message || job.message || "Waiting in the queue."}
                        </p>
                        <Progress value={progressMeta.progress} shimmer />
                      </div>
                    );
                  })}
                </div>
              </Card>
            ) : null}
          </div>

          {/* Right rail */}
          <div>
            <h2 className="font-display text-base text-foreground mb-3">All stages</h2>
            <StageTimeline project={project} />
          </div>
        </div>
      </div>
    </AppShell>
  );
}
