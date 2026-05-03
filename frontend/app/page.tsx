"use client";

import Link from "next/link";
import { Suspense, useEffect, useState } from "react";
import { useSearchParams } from "next/navigation";
import { LoaderCircle, Plus, RefreshCw } from "lucide-react";

import { AppShell } from "@/components/project/app-shell";
import { ProjectCard } from "@/components/project/project-card";
import { Button } from "@/components/ui/button";
import { Card, CardDescription, CardTitle } from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import { api } from "@/lib/api";
import { DetectorTrainingStatus, ProjectSummary } from "@/lib/types";
import { useAdaptivePolling } from "@/lib/use-adaptive-polling";

function DashboardContent() {
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
      description="Import chapters, detect panels, generate narration, and render recap videos."
    >
      <div className="flex flex-wrap items-center gap-3">
        <Link href="/projects/new">
          <Button>
            <Plus className="h-4 w-4" />
            New Project
          </Button>
        </Link>
        <div className="flex items-center gap-4 rounded-lg border border-white/10 bg-white/5 px-4 py-2 text-sm">
          <span className="text-mutedForeground">{projects.length} projects</span>
          <span className="h-4 w-px bg-white/10" />
          <span className={runningJobs > 0 ? "text-accent" : "text-mutedForeground"}>{runningJobs} running</span>
          <span className="h-4 w-px bg-white/10" />
          <span className="text-mutedForeground">{completedVideos} videos</span>
        </div>
      </div>

      <div className="mt-6 flex items-center justify-between gap-3">
        <div>
          <h2 className="font-display text-2xl">Projects</h2>
          <p className="text-sm text-mutedForeground">
            Live queue progress refreshes automatically, with faster updates while work is active.
            {refreshing ? " Updating now..." : ""}
          </p>
        </div>
        <Button variant="ghost" onClick={() => load()} disabled={loading || refreshing}>
          <RefreshCw className={`h-4 w-4 ${refreshing ? "animate-spin" : ""}`} />
          Refresh
        </Button>
      </div>

      <div className="mt-3 flex gap-1">
        {([
          { key: "all", label: "All" },
          { key: "active", label: "Active" },
          { key: "completed", label: "Completed" },
          { key: "review", label: "Needs Review" }
        ] as const).map((f) => (
          <button
            key={f.key}
            type="button"
            onClick={() => setFilter(f.key)}
            className={`rounded-full px-3 py-1 text-xs font-medium transition ${
              filter === f.key
                ? "bg-accent/15 text-accent"
                : "text-mutedForeground hover:bg-white/10 hover:text-white"
            }`}
          >
            {f.label}
          </button>
        ))}
      </div>

      {batchCreated > 1 ? (
        <div className="mt-4 rounded-[24px] border border-emerald-400/20 bg-emerald-400/10 p-4 text-sm text-emerald-50">
          Created {batchCreated} projects from your URL list. They will appear below and continue through the pipeline automatically.
        </div>
      ) : null}

      {trainingStatus ? (
        <Card className="mt-4 border-white/10 bg-white/[0.04]">
          <div className="flex flex-wrap items-start justify-between gap-4">
            <div className="space-y-2">
              <CardTitle>
                {trainingStatus.ready_to_train
                  ? "New training data available"
                  : trainingStatus.is_training
                    ? "Training detector now"
                    : "Detector learning loop"}
              </CardTitle>
              <CardDescription>
                {trainingStatus.message ||
                  "Panelia saves corrected panel boxes and OCR overrides so we can improve the detector without retraining blindly after every project."}
              </CardDescription>
              <p className="text-xs text-mutedForeground">
                Panel corrections: {trainingStatus.new_panel_annotations} new / {trainingStatus.panel_annotations_total} total
                {" · "}
                OCR corrections: {trainingStatus.new_ocr_annotations} new / {trainingStatus.ocr_annotations_total} total
              </p>
              {trainingStatus.is_training ? (
                <div className="space-y-2 pt-1">
                  <div className="flex items-center justify-between gap-3 text-xs uppercase tracking-[0.18em] text-mutedForeground">
                    <span>
                      Epoch {trainingStatus.current_epoch}/{trainingStatus.total_epochs || "?"}
                    </span>
                    <span>{Math.round(trainingStatus.progress_percent || 0)}%</span>
                  </div>
                  <Progress value={trainingStatus.progress_percent || 0} />
                  {trainingStatus.train_loss != null || trainingStatus.val_loss != null ? (
                    <p className="text-xs text-mutedForeground">
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
                  {trainingBusy ? "Cancelling..." : "Cancel training"}
                </Button>
              ) : trainingStatus.ready_to_train ? (
                <Button onClick={handleTrainDetectorToggle} disabled={trainingBusy}>
                  {trainingBusy ? <LoaderCircle className="h-4 w-4 animate-spin" /> : null}
                  Train detector now
                </Button>
              ) : (
                <span className="text-xs text-mutedForeground">
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
        <div className="mt-6 flex items-center gap-3 rounded-[28px] border border-white/10 bg-white/5 p-6 text-sm text-mutedForeground">
          <LoaderCircle className="h-4 w-4 animate-spin text-accent" />
          Loading your studio projects...
        </div>
      ) : error ? (
        <div className="mt-6 rounded-[28px] border border-red-500/25 bg-red-500/10 p-6 text-sm text-red-200">{error}</div>
      ) : filteredProjects.length ? (
        <div className="mt-6 grid gap-4 md:grid-cols-2 2xl:grid-cols-3">
          {filteredProjects.map((project) => (
            <ProjectCard
              key={project.id}
              project={project}
              onCancelProject={handleCancelProject}
              onDeleteProject={handleDeleteProject}
              actionBusy={actionProjectId === project.id}
            />
          ))}
        </div>
      ) : (
        <Card className="mt-6">
          <CardTitle>No projects yet</CardTitle>
          <CardDescription className="mt-2">
            Create your first project to import manga pages, detect panels, and start building your recap workflow.
          </CardDescription>
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
          description="Import chapters, detect panels, generate narration, and render recap videos."
        >
          <div className="flex items-center gap-3 rounded-xl border border-white/10 bg-white/5 p-6 text-sm text-mutedForeground">
            <LoaderCircle className="h-4 w-4 animate-spin text-accent" />
            Loading your studio projects...
          </div>
        </AppShell>
      }
    >
      <DashboardContent />
    </Suspense>
  );
}
