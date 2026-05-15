"use client";

import Link from "next/link";
import { FormEvent, useEffect, useState } from "react";
import {
  ArrowRight,
  Check,
  Copy,
  Edit3,
  PlayCircle,
  Trash2,
  X,
  XCircle,
} from "lucide-react";

import { Button } from "@/components/ui/button";
import { Badge, type BadgeTone } from "@/components/ui/badge";
import { Card, CardDescription, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Progress } from "@/components/ui/progress";
import { getStageProgressMeta } from "@/lib/progress";
import { buildMediaUrl, formatRelativeDate } from "@/lib/utils";
import { ProjectSummary, StageStatus } from "@/lib/types";
import { toPipelineDisplay, shortStageLabel } from "@/lib/pipeline-messages";

/**
 * ProjectCard - the entry surface for a project in the Studio grid.
 *
 * Notion-influenced layout: cover image on the left, content on the right,
 * minimal divider, hover-reveal action row. Liquid-glass surface inherited
 * from <Card>.
 *
 * The card answers three questions at a glance:
 *   1. What is this project?     (name + manga title + cover)
 *   2. Where is it in the pipeline?   (single sentence, not a percent)
 *   3. What's the next action?   (Open primary CTA, secondary panels link)
 */
export function ProjectCard({
  project,
  onCancelProject,
  onDeleteProject,
  onDuplicateProject,
  onRenameProject,
  actionBusy,
}: {
  project: ProjectSummary;
  onCancelProject?: (project: ProjectSummary) => void;
  onDeleteProject?: (project: ProjectSummary) => void;
  onDuplicateProject?: (project: ProjectSummary) => void;
  onRenameProject?: (project: ProjectSummary, name: string) => Promise<void> | void;
  actionBusy?: boolean;
}) {
  const [editingName, setEditingName] = useState(false);
  const [draftName, setDraftName] = useState(project.name);

  const stageStates = Object.values(project.stage_states ?? {}).filter(
    (stage): stage is NonNullable<(typeof project.stage_states)[keyof typeof project.stage_states]> =>
      Boolean(stage && typeof stage === "object")
  );
  const activeStage =
    stageStates.find(
      (stage) =>
        (stage.status as StageStatus) === "running" ||
        (stage.status as StageStatus) === "needs_review"
    ) ??
    stageStates.find((stage) => (stage.status as StageStatus) === "failed") ??
    null;
  const activeStageMeta = activeStage ? getStageProgressMeta(project, activeStage.stage) : null;
  const activeDisplay = activeStage ? toPipelineDisplay(activeStage) : null;

  const mangaTitle = project.chapter_metadata?.manga_title || "Untitled chapter";
  const keptPanelCount = Number(project.kept_panel_count ?? 0);
  const activeJobsCount = Array.isArray(project.active_jobs) ? project.active_jobs.length : 0;
  const canSaveName = draftName.trim().length > 0 && draftName.trim() !== project.name;

  // Border-edge confidence: ok if completed, warn if needs_review, fail if failed.
  const finalStage = stageStates.find((s) => s.stage === "video_rendering");
  const edgeClass = (() => {
    if (finalStage && finalStage.status === "completed") return "p-edge-ok";
    const flagged = stageStates.find((s) => s.status === "failed");
    if (flagged) return "p-edge-fail";
    const review = stageStates.find((s) => s.status === "needs_review");
    if (review) return "p-edge-warn";
    return "";
  })();

  useEffect(() => {
    if (!editingName) {
      setDraftName(project.name);
    }
  }, [editingName, project.name]);

  async function submitRename(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!canSaveName || actionBusy) return;
    await onRenameProject?.(project, draftName.trim());
    setEditingName(false);
  }

  function cancelRename() {
    setDraftName(project.name);
    setEditingName(false);
  }

  // Map our friendly tones to Badge tones.
  const toneFromDisplay = (tone: ReturnType<typeof toPipelineDisplay>["tone"]): BadgeTone => {
    if (tone === "accent" || tone === "info" || tone === "warn" || tone === "fail" || tone === "ok")
      return tone;
    return "neutral";
  };

  return (
    <Card padded="none" className={`group h-full overflow-hidden ${edgeClass}`}>
      <div className="grid h-full grid-cols-[8rem_minmax(0,1fr)] md:grid-cols-[9rem_minmax(0,1fr)]">
        {/* Cover */}
        <div className="relative overflow-hidden border-r border-white/[0.06] bg-white/[0.04]">
          {project.thumbnail_url ? (
            // eslint-disable-next-line @next/next/no-img-element
            <img
              src={buildMediaUrl(project.thumbnail_url)}
              alt={project.name}
              loading="lazy"
              decoding="async"
              className="h-full w-full object-cover object-top transition-transform duration-mid ease-liquid group-hover:scale-[1.03]"
            />
          ) : (
            <div className="flex h-full items-center justify-center bg-[radial-gradient(circle_at_center,_rgb(var(--p-accent)/0.15),_transparent_55%)]">
              <PlayCircle className="h-8 w-8 text-accent/70" strokeWidth={1.4} />
            </div>
          )}
        </div>

        {/* Content */}
        <div className="flex min-w-0 flex-col gap-3 p-5">
          {/* Title + rename */}
          <div className="flex items-start justify-between gap-3">
            <div className="min-w-0 flex-1">
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
                    className="h-9 min-w-0 rounded-xl px-3 text-base"
                    aria-label="Project name"
                  />
                  <Button
                    type="submit"
                    size="sm"
                    className="h-9 w-9 shrink-0 px-0"
                    disabled={!canSaveName || actionBusy}
                    title="Save project name"
                  >
                    <Check className="h-4 w-4" />
                  </Button>
                  <Button
                    type="button"
                    variant="ghost"
                    size="sm"
                    className="h-9 w-9 shrink-0 px-0"
                    onClick={cancelRename}
                    disabled={actionBusy}
                    title="Cancel rename"
                  >
                    <X className="h-4 w-4" />
                  </Button>
                </form>
              ) : (
                <div className="flex min-w-0 items-start gap-1.5">
                  <CardTitle className="line-clamp-2 leading-tight">{project.name}</CardTitle>
                  <Button
                    type="button"
                    variant="ghost"
                    size="sm"
                    className="h-7 w-7 shrink-0 px-0 opacity-0 transition-opacity duration-fast group-hover:opacity-100"
                    onClick={() => setEditingName(true)}
                    disabled={actionBusy}
                    title="Rename project"
                  >
                    <Edit3 className="h-3.5 w-3.5" />
                  </Button>
                </div>
              )}
              <CardDescription className="mt-1 line-clamp-1 text-xs">
                {mangaTitle} · {formatRelativeDate(project.updated_at)}
              </CardDescription>
            </div>
          </div>

          {/* Status sentence - replaces the percent + "Script Generation" badge */}
          <div className="flex items-center gap-2 flex-wrap">
            {activeDisplay ? (
              <Badge tone={toneFromDisplay(activeDisplay.tone)} dot pulse={activeDisplay.active}>
                {shortStageLabel(activeStage!.stage)}
              </Badge>
            ) : (
              <Badge tone="ok" dot>
                Ready
              </Badge>
            )}
            <Badge>{project.page_count} pages</Badge>
            <Badge>{keptPanelCount} panels</Badge>
            {project.latest_video ? <Badge tone="info">Rendered</Badge> : null}
          </div>

          {/* Sentence-form status line */}
          {activeDisplay && (
            <p className="text-sm text-mutedForeground leading-relaxed line-clamp-2">
              {activeDisplay.sentence}
            </p>
          )}

          {/* Progress for running stage */}
          {activeStage && (activeStage.status as StageStatus) === "running" ? (
            <Progress
              value={activeStageMeta?.progress ?? activeStage.progress}
              shimmer
              className="mt-1"
            />
          ) : null}

          {/* Primary + secondary actions */}
          <div className="mt-auto flex flex-wrap items-center gap-2 pt-2">
            <Link
              href={`/projects/${project.id}`}
              className="inline-flex items-center justify-center gap-2 rounded-full bg-accent px-4 py-2 text-sm font-medium text-accent-foreground shadow-[0_0_24px_-6px_rgb(var(--p-accent)/0.6)] transition-all duration-fast ease-liquid hover:-translate-y-px hover:shadow-[0_0_32px_-4px_rgb(var(--p-accent)/0.8)]"
            >
              Open
              <ArrowRight className="h-4 w-4" />
            </Link>
            <Link
              href={`/projects/${project.id}/timeline`}
              className="inline-flex items-center justify-center gap-2 rounded-full border border-white/[0.10] px-3 py-2 text-sm text-foreground transition-colors duration-fast hover:bg-white/[0.06]"
            >
              Timeline
            </Link>

            {/* Hover-reveal secondary actions */}
            <div className="ml-auto flex items-center gap-1 opacity-60 transition-opacity duration-fast group-hover:opacity-100">
              <Button
                variant="ghost"
                size="sm"
                onClick={() => onDuplicateProject?.(project)}
                disabled={actionBusy}
                title="Duplicate"
              >
                <Copy className="h-3.5 w-3.5" />
              </Button>
              {activeJobsCount ? (
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={() => onCancelProject?.(project)}
                  disabled={actionBusy}
                  title="Cancel running jobs"
                >
                  <XCircle className="h-3.5 w-3.5" />
                </Button>
              ) : null}
              <Button
                variant="ghost"
                size="sm"
                onClick={() => onDeleteProject?.(project)}
                disabled={actionBusy || activeJobsCount > 0}
                title="Delete project"
              >
                <Trash2 className="h-3.5 w-3.5" />
              </Button>
            </div>
          </div>
        </div>
      </div>
    </Card>
  );
}
