"use client";

import Link from "next/link";
import { Suspense, useEffect, useState } from "react";
import { useRouter, useSearchParams } from "next/navigation";
import { LoaderCircle, Plus, RefreshCw } from "lucide-react";

import { AppShell } from "@/components/project/app-shell";
import { ProjectCard } from "@/components/project/project-card";
import { Button } from "@/components/ui/button";
import { Card, CardDescription, CardTitle } from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import { api } from "@/lib/api";
import { formatProgressPercent } from "@/lib/progress";
import { DetectorTrainingStatus, ProjectSummary } from "@/lib/types";
import { useAdaptivePolling } from "@/lib/use-adaptive-polling";

function DashboardContent() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const [projects, setProjects] = useState<ProjectSummary[]>([]);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [actionProjectId, setActionProjectId] = useState<string | null>(null);
  const [trainingStatus, setTrainingStatus] = useState<DetectorTrainingStatus | null>(null);
  const [trainingBusy, setTrainingBusy] = useState(false);

  async function load(options?: { background?: boolean }) {
    const background = options?.background ?? false;
    try {
      if (background) {
        setRefreshing(true);
      } else {
        setLoading(true);
      }
      setProjects(await api.listProjects());
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unable to load projects.");
    } finally {
      if (background) {
        setRefreshing(false);
      } else {
        setLoading(false);
      }
    }
  }

  async function loadTrainingStatus() {
    try {
      setTrainingStatus(await api.getPanelDetectorTrainingStatus());
    } catch {
      // Keep the dashboard usable even if the training-status endpoint is unavailable.
    }
  }

  useEffect(() => {
    load();
    loadTrainingStatus();
  }, []);

  useAdaptivePolling(
    () => load({ background: true }),
    {
      active: projects.some((project) => (project.active_jobs?.length ?? 0) > 0),
      activeMs: 10000,
      idleMs: 30000,
      hiddenMs: 120000
    }
  );

  useAdaptivePolling(
    () => loadTrainingStatus(),
    {
      active: Boolean(trainingStatus?.is_training),
      activeMs: 15000,
      idleMs: 45000,
      hiddenMs: 120000,
      deps: [trainingStatus?.is_training]
    }
  );

  async function handleCancelProject(project: ProjectSummary) {
    const confirmed = window.confirm(`Cancel all active jobs for "${project.name}"?`);
    if (!confirmed) {
      return;
    }
    try {
      setActionProjectId(project.id);
      await api.cancelProject(project.id);
      await load();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unable to cancel the project jobs.");
    } finally {
      setActionProjectId(null);
    }
  }

  async function handleDeleteProject(project: ProjectSummary) {
    const confirmed = window.confirm(`Delete "${project.name}" permanently? This removes pages, audio, videos, and saved edits.`);
    if (!confirmed) {
      return;
    }
    try {
      setActionProjectId(project.id);
      await api.deleteProject(project.id);
      await load();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unable to delete the project.");
    } finally {
      setActionProjectId(null);
    }
  }

  async function handleDuplicateProject(project: ProjectSummary) {
    try {
      setActionProjectId(project.id);
      const duplicated = await api.duplicateProject(project.id, {
        name: `${project.name} Copy`,
        copy_all_videos: false
      });
      setError(null);
      router.push(`/projects/${duplicated.id}`);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unable to duplicate the project.");
      await load({ background: true });
    } finally {
      setActionProjectId(null);
    }
  }

  async function handleRenameProject(project: ProjectSummary, name: string) {
    try {
      setActionProjectId(project.id);
      const renamed = await api.renameProject(project.id, name);
      setProjects((current) =>
        current.map((item) => (item.id === project.id ? { ...item, name: renamed.name, updated_at: renamed.updated_at } : item))
      );
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unable to rename the project.");
      await load({ background: true });
      throw err;
    } finally {
      setActionProjectId(null);
    }
  }

  async function handleTrainDetectorToggle() {
    if (!trainingStatus) return;
    try {
      setTrainingBusy(true);
      const nextStatus = trainingStatus.is_training
        ? await api.cancelPanelDetectorTraining()
        : await api.startPanelDetectorTraining();
      setTrainingStatus(nextStatus);
      setError(null);
    } catch (err) {
      setError(
        err instanceof Error
          ? err.message
          : trainingStatus.is_training
            ? "Unable to cancel detector training."
            : "Unable to start detector training."
      );
    } finally {
      setTrainingBusy(false);
    }
  }

  const [filter, setFilter] = useState<"all" | "active" | "completed" | "review">("all");

  const runningJobs = projects.reduce((total, project) => total + (project.active_jobs?.length ?? 0), 0);
  const completedVideos = projects.filter((project) => project.latest_video).length;
  const batchCreated = Number(searchParams.get("batchCreated") ?? "0");

  const filteredProjects = projects.filter((project) => {
    if (filter === "active") return (project.active_jobs?.length ?? 0) > 0;
    if (filter === "completed") return Boolean(project.latest_video);
    if (filter === "review") {
      return (
        project.stage_states.panel_review?.status === "needs_review"
        || project.stage_states.panel_review?.status === "ready"
        || project.stage_states.character_review?.status === "needs_review"
        || project.stage_states.character_review?.status === "ready"
      );
    }
    return true;
  });

  return (
    <AppShell
      title="Studio"
      description="Import chapters, detect panels, generate narration, render recap videos."
      meta={(
        <>
          <span className="p-pill">{projects.length} projects</span>
          <span className={`p-pill ${runningJobs > 0 ? "p-pill-accent" : ""}`}>
            <span className={`inline-block h-1.5 w-1.5 rounded-full bg-current ${runningJobs > 0 ? "p-anim-breathe" : "opacity-60"}`} />
            {runningJobs} running
          </span>
          <span className="p-pill p-pill-ok">{completedVideos} rendered</span>
        </>
      )}
      actions={(
        <Link href="/projects/new">
          <Button>
            <Plus className="h-4 w-4" />
            New project
          </Button>
        </Link>
      )}
    >

      {/* Filter pills + refresh */}
      <div className="flex items-center justify-between gap-3 flex-wrap">
        <div className="flex flex-wrap gap-1">
        {([
          { key: "all", label: "All" },
          { key: "active", label: "Active" },
          { key: "completed", label: "Rendered" },
          { key: "review", label: "Needs review" }
        ] as const).map((f) => (
          <button
            key={f.key}
            type="button"
            onClick={() => setFilter(f.key)}
            className={`rounded-full px-3 py-1.5 text-xs font-medium transition-all duration-fast ease-liquid ${
              filter === f.key
                ? "bg-accent/[0.12] text-accent shadow-[inset_0_0_0_1px_rgb(var(--p-accent)/0.25)]"
                : "text-mutedForeground hover:bg-white/[0.06] hover:text-foreground"
            }`}
          >
            {f.label}
          </button>
        ))}
        </div>
        <Button variant="ghost" onClick={() => load()} disabled={loading || refreshing}>
          <RefreshCw className={`h-4 w-4 ${refreshing ? "animate-spin" : ""}`} />
          {refreshing ? "Updating…" : "Refresh"}
        </Button>
      </div>

      {batchCreated > 1 ? (
        <Card padded="md" className="p-edge-ok">
          <p className="text-sm">
            Created {batchCreated} projects from your URL list. They&apos;ll continue through the pipeline automatically.
          </p>
        </Card>
      ) : null}

      {trainingStatus ? (
        <Card padded="md">
          <div className="flex flex-wrap items-start justify-between gap-4">
            <div className="min-w-0 space-y-2">
              <CardTitle>
                {trainingStatus.ready_to_train
                  ? "New training data available"
                  : trainingStatus.is_training
                    ? "Training detector now"
                    : "Detector learning loop"}
              </CardTitle>
              <CardDescription>
                {trainingStatus.message ||
                  "Panelia saves corrected panel boxes and OCR overrides so the detector keeps improving."}
              </CardDescription>
              <p className="text-xs text-mutedForeground">
                Panels corrected: <span className="text-foreground tabular-nums">{trainingStatus.new_panel_annotations}</span> new of {trainingStatus.panel_annotations_total} total
                {" · "}
                OCR corrected: <span className="text-foreground tabular-nums">{trainingStatus.new_ocr_annotations}</span> new of {trainingStatus.ocr_annotations_total} total
              </p>
              {trainingStatus.is_training ? (
                <div className="space-y-2 pt-1">
                  <div className="flex items-center justify-between gap-3 text-[10px] uppercase tracking-track text-mutedForeground">
                    <span>
                      Epoch {trainingStatus.current_epoch}/{trainingStatus.total_epochs || "?"}
                    </span>
                    <span className="tabular-nums">{formatProgressPercent(trainingStatus.progress_percent)}</span>
                  </div>
                  <Progress value={trainingStatus.progress_percent || 0} shimmer />
                  {trainingStatus.train_loss != null || trainingStatus.val_loss != null ? (
                    <p className="text-xs text-mutedForeground font-mono">
                      {trainingStatus.train_loss != null ? `train ${trainingStatus.train_loss.toFixed(4)}` : null}
                      {trainingStatus.train_loss != null && trainingStatus.val_loss != null ? " · " : null}
                      {trainingStatus.val_loss != null ? `val ${trainingStatus.val_loss.toFixed(4)}` : null}
                    </p>
                  ) : null}
                </div>
              ) : null}
            </div>
            <div className="flex shrink-0 flex-col items-end gap-2">
              {trainingStatus.is_training ? (
                <Button variant="secondary" onClick={handleTrainDetectorToggle} disabled={trainingBusy}>
                  <LoaderCircle className={`h-4 w-4 ${trainingBusy ? "animate-spin" : ""}`} />
                  {trainingBusy ? "Cancelling…" : "Cancel training"}
                </Button>
              ) : trainingStatus.ready_to_train ? (
                <Button onClick={handleTrainDetectorToggle} disabled={trainingBusy}>
                  {trainingBusy ? <LoaderCircle className="h-4 w-4 animate-spin" /> : null}
                  Train detector now
                </Button>
              ) : (
                <span className="text-xs text-mutedForeground text-right max-w-[16rem]">
                  {trainingStatus.remaining_annotations_until_ready > 0
                    ? `${trainingStatus.remaining_annotations_until_ready} more corrected page${trainingStatus.remaining_annotations_until_ready === 1 ? "" : "s"} until the next recommended training run`
                    : "Keep correcting panels to build more training signal"}
                </span>
              )}
            </div>
          </div>
        </Card>
      ) : null}

      {loading ? (
        <Card padded="lg" className="flex items-center gap-3">
          <LoaderCircle className="h-4 w-4 animate-spin text-accent" />
          <span className="text-sm text-mutedForeground">Loading your studio projects…</span>
        </Card>
      ) : error ? (
        <Card padded="lg" className="p-edge-fail">
          <p className="text-sm text-fail">{error}</p>
        </Card>
      ) : filteredProjects.length ? (
        <div className="grid gap-4 md:grid-cols-2 2xl:grid-cols-3">
          {filteredProjects.map((project) => (
            <ProjectCard
              key={project.id}
              project={project}
              onCancelProject={handleCancelProject}
              onDeleteProject={handleDeleteProject}
              onDuplicateProject={handleDuplicateProject}
              onRenameProject={handleRenameProject}
              actionBusy={actionProjectId === project.id}
            />
          ))}
        </div>
      ) : (
        <Card padded="lg" >
          <CardTitle>No projects yet</CardTitle>
          <CardDescription className="mt-2">
            Create your first project to import pages, detect panels, and start building your recap.
          </CardDescription>
          <Link href="/projects/new" className="inline-block mt-4">
            <Button>
              <Plus className="h-4 w-4" />
              New project
            </Button>
          </Link>
        </Card>
      )}
    </AppShell>
  );
}

export default function DashboardPage() {
  return (
    <Suspense
      fallback={
        <AppShell
          title="Studio"
          description="Import chapters, detect panels, generate narration, render recap videos."
        >
          <Card padded="md" className="flex items-center gap-3">
            <LoaderCircle className="h-4 w-4 animate-spin text-accent" />
            <span className="text-sm text-mutedForeground">Loading your studio projects…</span>
          </Card>
        </AppShell>
      }
    >
      <DashboardContent />
    </Suspense>
  );
}
