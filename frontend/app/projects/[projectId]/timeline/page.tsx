"use client";

/**
 * /projects/:projectId/timeline — the new DaVinci-inspired editor route.
 *
 * Loads the project, hands the panels to <TimelineEditor>, and wires up:
 *   • Save → POST updated durations + narrations to the backend
 *   • Regenerate single panel → POST to the vision regenerate endpoint
 *   • PipelineBlock at the top, WhileYouWereAway banner above that
 */

import * as React from "react";
import Link from "next/link";
import { useParams, useRouter } from "next/navigation";
import { api } from "@/lib/api";
import { buildMediaUrl } from "@/lib/utils";
import type { PipelineStage, ProjectDetail } from "@/lib/types";

import { PipelineBlock } from "@/components/ui/pipeline-block";
import { WhileYouWereAway } from "@/components/ui/while-you-were-away";
import {
  TimelineEditor,
  type PanelEdits,
} from "@/components/editor/timeline/timeline-editor";
import { estimateProjectCost } from "@/lib/cost-estimate";

const POLL_INTERVAL_MS = 5_000;

export default function TimelinePage() {
  const params = useParams<{ projectId: string }>();
  const router = useRouter();
  const projectId = params?.projectId;

  const [project, setProject] = React.useState<ProjectDetail | null>(null);
  const [loadError, setLoadError] = React.useState<string | null>(null);
  const [saving, setSaving] = React.useState(false);
  const [savedAt, setSavedAt] = React.useState<number | null>(null);
  const [regenStatus, setRegenStatus] = React.useState<string | null>(null);

  const load = React.useCallback(async () => {
    if (!projectId) return;
    try {
      const next = await api.getProject(projectId);
      setProject(next);
      setLoadError(null);
    } catch (err) {
      setLoadError(err instanceof Error ? err.message : "Failed to load project");
    }
  }, [projectId]);

  React.useEffect(() => {
    void load();
  }, [load]);

  // Adaptive polling — every 5s while any active job is running.
  React.useEffect(() => {
    if (!project) return;
    const hasActive = (project.active_jobs ?? []).some(
      (job) => job.status === "running" || job.status === "queued",
    );
    if (!hasActive) return;
    const handle = setInterval(load, POLL_INTERVAL_MS);
    return () => clearInterval(handle);
  }, [project, load]);

  if (!projectId) {
    return <main className="p-8">Missing project id.</main>;
  }
  if (loadError) {
    return (
      <main className="p-8">
        <p className="text-[rgb(var(--p-fail))]">Could not load project: {loadError}</p>
      </main>
    );
  }
  if (!project) {
    return (
      <main className="p-8">
        <div className="p-glass px-6 py-10 max-w-md mx-auto text-center">
          <p className="text-[rgb(var(--p-muted))]">Loading project…</p>
        </div>
      </main>
    );
  }

  const thumbnailBaseUrl = buildMediaUrl(`/media/projects/${project.id}/panels`);

  const handleSaveEdits = async (edits: PanelEdits) => {
    setSaving(true);
    try {
      // Persist duration changes via PUT /panels (panels have duration_seconds).
      const hasDurations = Object.keys(edits.durations).length > 0;
      const hasNarrations = Object.keys(edits.narrations).length > 0;
      if (hasDurations) {
        const nextPanels = (project.panels ?? []).map((p) => {
          const nextDur = edits.durations[p.id];
          return nextDur !== undefined
            ? { ...p, duration_seconds: nextDur }
            : p;
        });
        await api.updatePanels(project.id, nextPanels);
      }
      if (hasNarrations) {
        const kept = (project.panels ?? [])
          .filter((p) => p.keep)
          .sort((a, b) => a.page - b.page || a.panel - b.panel);
        const script_lines = kept.map((p) => edits.narrations[p.id] ?? p.narration ?? "");
        await api.updateScript(
          project.id,
          script_lines,
          {},
          edits.narrations,
          {},
        );
      }
      setSavedAt(Date.now());
      await load();
    } catch (err) {
      setLoadError(err instanceof Error ? err.message : "Save failed");
    } finally {
      setSaving(false);
    }
  };

  const handleRegenerate = async (panelId: string) => {
    setRegenStatus("Regenerating…");
    try {
      const res = await fetch(
        `${process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://127.0.0.1:8010/api"}/projects/${project.id}/script/regenerate-panel-vision`,
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ panel_id: panelId, mode: "balanced" }),
        },
      );
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      setRegenStatus("Regenerated.");
      await load();
      setTimeout(() => setRegenStatus(null), 2500);
    } catch (err) {
      setRegenStatus(err instanceof Error ? `Failed: ${err.message}` : "Failed");
    }
  };

  return (
    <main className="mx-auto max-w-[1680px] px-6 md:px-12 py-10 space-y-6">
      {/* Hero header — title in display weight on accent gradient,
          breadcrumb + meta sit above and below for hierarchy. */}
      <header className="flex flex-wrap items-end justify-between gap-4">
        <div className="min-w-0">
          <Link
            href="/"
            className="text-[11px] uppercase tracking-[0.18em] text-[rgb(var(--p-hint))] hover:text-[rgb(var(--p-accent))] transition-colors duration-[var(--p-fast)] inline-flex items-center gap-1"
          >
            <span aria-hidden>←</span> All projects
          </Link>
          <h1
            className="mt-2 text-3xl md:text-5xl font-semibold leading-tight tracking-tight"
            style={{
              backgroundImage:
                "linear-gradient(180deg, rgb(var(--p-text)) 0%, rgb(var(--p-muted)) 110%)",
              WebkitBackgroundClip: "text",
              backgroundClip: "text",
              color: "transparent",
            }}
          >
            {project.name}
          </h1>
          <p className="mt-2 text-sm text-[rgb(var(--p-muted))] flex flex-wrap items-center gap-x-3 gap-y-1">
            <span className="p-pill">
              {project.kept_panel_count}/{project.panel_count} panels kept
            </span>
            {savedAt && (
              <span className="p-pill p-pill-ok">
                saved {new Date(savedAt).toLocaleTimeString()}
              </span>
            )}
            {saving && (
              <span className="p-pill p-pill-info">
                <span className="inline-block h-1.5 w-1.5 rounded-full bg-current p-anim-breathe" />
                saving…
              </span>
            )}
            {regenStatus && (
              <span className="p-pill p-pill-accent">{regenStatus}</span>
            )}
          </p>
        </div>
        <nav className="flex items-center gap-2" aria-label="Other views">
          <Link href={`/projects/${project.id}/narration`} className="p-btn-ghost">
            Narration
          </Link>
          <Link href={`/projects/${project.id}/editor`} className="p-btn-ghost">
            Panels
          </Link>
        </nav>
      </header>

      {/* "While you were away" — only shows when state has changed since
          the user's last view. */}
      <WhileYouWereAway project={project} />

      {/* Pipeline at a glance — sentences, not numbers. */}
      <PipelineBlock
        stageStates={project.stage_states}
        cost={estimateProjectCost(project)}
        primaryAction={undefined}
      />

      {/* The editor */}
      <TimelineEditor
        panels={project.panels ?? []}
        thumbnailBaseUrl={thumbnailBaseUrl}
        onRegeneratePanel={handleRegenerate}
        onSaveEdits={handleSaveEdits}
      />
    </main>
  );
}
