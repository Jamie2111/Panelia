import { CheckCircle2, Clock3, LoaderCircle, Sparkles, XCircle } from "lucide-react";

import { Badge, type BadgeTone } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import { getStageProgressMeta } from "@/lib/progress";
import {
  LEGACY_STAGES_HIDDEN_IN_VISION,
  toPipelineDisplay,
  shortStageLabel,
} from "@/lib/pipeline-messages";
import type { PipelineStage, ProjectSummary, StageStatus } from "@/lib/types";

const FULL_ORDER: PipelineStage[] = [
  "ingestion",
  "panel_detection",
  "panel_review",
  "character_review",
  "character_portrait",
  "panel_vision_extraction",
  "panel_vision_quality",
  "script_generation",
  "narration_generation",
  "video_rendering",
  "youtube_bundle",
];

function statusIcon(status: StageStatus) {
  const cls = "h-4 w-4";
  if (status === "completed") return <CheckCircle2 className={`${cls} text-ok`} />;
  if (status === "running") return <LoaderCircle className={`${cls} animate-spin text-accent`} />;
  if (status === "failed" || status === "cancelled") return <XCircle className={`${cls} text-fail`} />;
  if (status === "needs_review") return <Sparkles className={`${cls} text-warn`} />;
  return <Clock3 className={`${cls} text-mutedForeground`} />;
}

function toneFor(status: StageStatus): BadgeTone {
  if (status === "completed") return "ok";
  if (status === "running") return "accent";
  if (status === "failed" || status === "cancelled") return "fail";
  if (status === "needs_review") return "warn";
  if (status === "ready") return "info";
  return "neutral";
}

/**
 * StageTimeline — vertical, glanceable pipeline state.
 *
 * Replaces the percent + cryptic message layout with:
 *   • Status icon (color-coded)
 *   • Stage label
 *   • One sentence from `toPipelineDisplay()` instead of raw API message
 *   • Progress bar only when actively running (no fake 100% bars)
 *   • Tone-aware Badge
 */
export function StageTimeline({ project }: { project: ProjectSummary }) {
  const usingVision =
    (project.pipeline_config as any)?.script_pipeline_version === "vision";
  const visibleStages = usingVision
    ? FULL_ORDER.filter((s) => !LEGACY_STAGES_HIDDEN_IN_VISION.has(s))
    : FULL_ORDER;

  return (
    <div className="space-y-2">
      {visibleStages.map((stage) => {
        const state = project.stage_states[stage];
        if (!state) return null;
        const status = state.status as StageStatus;
        const progressMeta = getStageProgressMeta(project, stage);
        const display = toPipelineDisplay(state);

        const isActive = status === "running" || status === "needs_review";
        const containerClass =
          isActive
            ? "p-glass p-edge-info p-4"
            : status === "completed"
            ? "p-glass p-edge-ok p-4 opacity-90"
            : status === "failed" || status === "cancelled"
            ? "p-glass p-edge-fail p-4"
            : "p-glass p-4 opacity-80";

        return (
          <div key={stage} className={`${containerClass} transition-all duration-mid ease-liquid`}>
            <div className="flex items-start justify-between gap-3">
              <div className="flex min-w-0 items-start gap-3">
                <div
                  className={`rounded-full p-1.5 ${
                    isActive ? "bg-accent/10" : "bg-white/[0.05]"
                  }`}
                >
                  {statusIcon(status)}
                </div>
                <div className="min-w-0">
                  <p
                    className={`text-sm font-medium ${
                      isActive ? "text-foreground" : "text-foreground"
                    }`}
                  >
                    {shortStageLabel(stage)}
                  </p>
                  <p className="mt-0.5 text-xs text-mutedForeground leading-relaxed">
                    {display.sentence}
                  </p>
                </div>
              </div>
              <Badge tone={toneFor(status)} dot={isActive} pulse={isActive}>
                {status === "needs_review" ? "review" : status}
              </Badge>
            </div>

            {status === "running" ? (
              <div className="mt-3 pl-9">
                <Progress value={progressMeta.progress} shimmer />
              </div>
            ) : null}
          </div>
        );
      })}
    </div>
  );
}
