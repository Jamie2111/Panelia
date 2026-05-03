"use client";

import Link from "next/link";
import { useParams } from "next/navigation";
import { useEffect, useState } from "react";
import { LoaderCircle, RefreshCw, Layers3 } from "lucide-react";

import { AppShell } from "@/components/project/app-shell";
import { StageTimeline } from "@/components/project/stage-timeline";
import { Button } from "@/components/ui/button";
import { Card, CardDescription, CardTitle } from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import { api } from "@/lib/api";
import { PipelineStage, ProjectDetail } from "@/lib/types";
import { useAdaptivePolling } from "@/lib/use-adaptive-polling";
import { buildMediaUrl, formatRelativeDate, stageLabel } from "@/lib/utils";

function safeDimension(value: number | string | null | undefined, fallback: number) {
  const numeric = Number(value);
  return Number.isFinite(numeric) && numeric > 0 ? Math.round(numeric) : fallback;
}

function baseResolution(width: number, height: number) {
  const safeWidth = safeDimension(width, 1920);
  const safeHeight = safeDimension(height, 1080);
  return `${Math.max(safeWidth, safeHeight)}×${Math.min(safeWidth, safeHeight)}`;
}

export default function ProjectOverviewPage() {
  const params = useParams<{ projectId: string }>();
  const projectId = Array.isArray(params.projectId) ? params.projectId[0] : params.projectId;
  const [project, setProject] = useState<ProjectDetail | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [stageActionBusy, setStageActionBusy] = useState<PipelineStage | null>(null);
  const [rewindBusy, setRewindBusy] = useState<PipelineStage | null>(null);
  const [cancelJobBusy, setCancelJobBusy] = useState<string | null>(null);
  const [refreshBusy, setRefreshBusy] = useState(false);
  const [pipelineToggleBusy, setPipelineToggleBusy] = useState(false);

  async function load() {
    if (!projectId) return;
    try {
      setProject(await api.getProject(projectId));
      setError(null);
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
      <AppShell title="Loading project" description="Fetching project details and queue status." projectId={projectId}>
        <div className="flex items-center gap-3 rounded-xl border border-white/10 bg-white/5 p-6 text-sm text-mutedForeground">
          <LoaderCircle className="h-4 w-4 animate-spin text-accent" />
          Loading project details...
        </div>
      </AppShell>
    );
  }

  if (!project || error) {
    return (
      <AppShell title="Project unavailable" description="Panelia couldn't load the requested project." projectId={projectId}>
        <Card>
          <CardTitle>Unable to load project</CardTitle>
          <CardDescription className="mt-2">{error ?? "The project may have been moved or deleted."}</CardDescription>
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
      setProject(await api.getProject(project.id));
    } finally {
      setStageActionBusy(null);
    }
  }

  return (
    <AppShell
      title={project.name}
      description={`Created ${formatRelativeDate(project.created_at)} • ${project.chapter_metadata.manga_title || "Narrated manga project"}`}
      projectId={projectId}
    >
      <div className="grid gap-6 xl:grid-cols-[1.1fr_0.9fr]">
        <div className="space-y-6">
          {/* Project info card */}
          <Card className="overflow-hidden p-0">
            <div className="grid gap-0 sm:grid-cols-[200px_minmax(0,1fr)]">
              <div className="self-start border-b border-white/8 bg-white/5 sm:border-b-0 sm:border-r">
                <div className="h-[300px] w-full overflow-hidden bg-white/5">
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
              <div className="min-w-0 space-y-4 p-5">
                <div>
                  <CardTitle className="text-balance break-normal">{project.chapter_metadata.chapter_title || "Chapter import"}</CardTitle>
                  <CardDescription className="mt-1.5 break-normal text-pretty">
                    {project.chapter_metadata.manga_title || "Unknown series"} • {project.page_count} pages • {project.kept_panel_count} kept panels
                  </CardDescription>
                </div>
                <div className="grid gap-2 sm:grid-cols-3">
                  {[
                    { label: "Voice", value: project.voice_config.voice },
                    { label: "Render", value: `${baseResolution(project.video_config.width, project.video_config.height)} • ${project.video_config.orientation}` },
                    { label: "Exports", value: String(project.videos.length) }
                  ].map((item) => (
                    <div key={item.label} className="rounded-lg border border-white/10 bg-white/5 px-3 py-2.5">
                      <p className="text-xs text-mutedForeground">{item.label}</p>
                      <p className="mt-1 truncate text-[13px] font-medium text-white">{item.value}</p>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </Card>

          {/* Pipeline controls */}
          <Card>
            <div className="flex items-center justify-between gap-3">
              <div>
                <CardTitle className="text-base">Pipeline controls</CardTitle>
                <CardDescription className="mt-1">Rerun specific stages or rewind the project to an earlier step.</CardDescription>
              </div>
              <Button
                variant="ghost"
                size="sm"
                disabled={refreshBusy}
                onClick={async () => {
                  setRefreshBusy(true);
                  try { setProject(await api.getProject(project.id)); } finally { setRefreshBusy(false); }
                }}
              >
                <RefreshCw className={`h-3.5 w-3.5 ${refreshBusy ? "animate-spin" : ""}`} />
                Refresh
              </Button>
            </div>
            <div className="mt-4 rounded-xl border border-white/10 bg-white/5 px-4 py-3">
              <label className="flex cursor-pointer items-start justify-between gap-4">
                <div className="space-y-1">
                  <p className="text-sm font-medium text-white">Run end to end automatically</p>
                  <p className="text-xs text-mutedForeground">
                    Continue from import through video export on this project whenever Panelia can proceed without a manual fix.
                  </p>
                </div>
                <span className="flex items-center gap-2">
                  {pipelineToggleBusy ? <LoaderCircle className="h-4 w-4 animate-spin text-accent" /> : null}
                  <input
                    type="checkbox"
                    className="mt-0.5 h-4 w-4 rounded border border-white/15 bg-transparent accent-[var(--accent)]"
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
            </div>
            <div className="mt-4 grid gap-2 sm:grid-cols-2 lg:grid-cols-4">
              {[
                { label: "Prepare characters", stage: "character_review" as const },
                { label: "Generate script", stage: "script_generation" as const },
                { label: "Generate audio", stage: "narration_generation" as const },
                { label: "Render video", stage: "video_rendering" as const },
                { label: "Detect panels", stage: "panel_detection" as const }
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
                      Working...
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
                    try { setProject(await api.rewindProject(project.id, item.stage)); } finally { setRewindBusy(null); }
                  }}
                >
                  {rewindBusy === item.stage ? <LoaderCircle className="h-3 w-3 animate-spin" /> : null}
                  {rewindBusy === item.stage ? "Rewinding..." : item.label}
                </Button>
              ))}
            </div>
          </Card>

          {/* Active jobs */}
          {activeJobs.length ? (
            <Card>
              <CardTitle className="text-base">Active jobs</CardTitle>
              <div className="mt-4 space-y-3">
                {activeJobs.map((job) => {
                  const progress = Math.max(0, Math.min(100, Math.round(job.progress ?? 0)));
                  return (
                    <div key={job.id} className="space-y-2.5 rounded-xl border border-white/10 bg-white/5 px-4 py-3">
                      <div className="flex items-center justify-between gap-3">
                        <div className="flex min-w-0 items-center gap-2.5">
                          <LoaderCircle className="h-4 w-4 shrink-0 animate-spin text-accent" />
                          <p className="text-sm font-medium text-white">{stageLabel(job.stage)}</p>
                        </div>
                        <Button
                          variant="outline"
                          size="sm"
                          disabled={cancelJobBusy !== null}
                          onClick={async () => {
                            setCancelJobBusy(job.id);
                            try {
                              await api.cancelJob(project.id, job.id);
                              setProject(await api.getProject(project.id));
                            } finally { setCancelJobBusy(null); }
                          }}
                        >
                          {cancelJobBusy === job.id ? <LoaderCircle className="h-3.5 w-3.5 animate-spin" /> : null}
                          {cancelJobBusy === job.id ? "Cancelling..." : "Cancel"}
                        </Button>
                      </div>
                      <div className="flex items-center justify-between gap-3 text-[10px] uppercase tracking-[0.18em] text-mutedForeground">
                        <span className="truncate">{job.message || "Waiting in the queue"}</span>
                        <span className="shrink-0 tabular-nums">{progress}%</span>
                      </div>
                      <Progress value={progress} />
                    </div>
                  );
                })}
              </div>
            </Card>
          ) : null}
        </div>

        <StageTimeline project={project} />
      </div>
    </AppShell>
  );
}
