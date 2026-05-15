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
import { useParams } from "next/navigation";
import { api } from "@/lib/api";
import { buildMediaUrl } from "@/lib/utils";
import type { ProjectDetail } from "@/lib/types";

import { AppShell } from "@/components/project/app-shell";
import { Badge } from "@/components/ui/badge";
import { PipelineBlock } from "@/components/ui/pipeline-block";
import { WhileYouWereAway } from "@/components/ui/while-you-were-away";
import {
  TimelineEditor,
  type PanelEdits,
} from "@/components/editor/timeline/timeline-editor";
import { estimateProjectCost } from "@/lib/cost-estimate";
import { buildProjectViews } from "@/lib/project-views";

const POLL_INTERVAL_MS = 5_000;

export default function TimelinePage() {
  const params = useParams<{ projectId: string }>();
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
    return (
      <AppShell title="Timeline" description="Missing project id." />
    );
  }
  if (loadError) {
    return (
      <AppShell
        title="Timeline"
        description="Panelia couldn't load the timeline for this project."
      >
        <div className="p-glass p-edge-fail px-6 py-5">
          <p className="text-sm text-fail">Could not load project: {loadError}</p>
        </div>
      </AppShell>
    );
  }
  if (!project) {
    return (
      <AppShell title="Timeline" description="Loading project details and panel data…">
        <div className="p-glass px-6 py-10 max-w-md mx-auto text-center text-mutedForeground text-sm">
          Loading project…
        </div>
      </AppShell>
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
    <AppShell
      title={project.name}
      projectId={project.id}
      breadcrumb={{ href: "/", label: "All projects" }}
      views={buildProjectViews(project.id, "/timeline")}
      meta={(
        <>
          <Badge>{project.kept_panel_count}/{project.panel_count} panels kept</Badge>
          {savedAt && (
            <Badge tone="ok">saved {new Date(savedAt).toLocaleTimeString()}</Badge>
          )}
          {saving && <Badge tone="info" dot pulse>saving…</Badge>}
          {regenStatus && <Badge tone="accent">{regenStatus}</Badge>}
        </>
      )}
    >
      {/* "While you were away" — only fires when state has changed. */}
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
    </AppShell>
  );
}
