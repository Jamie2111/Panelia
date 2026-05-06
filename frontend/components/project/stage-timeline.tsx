import { CheckCircle2, Clock3, LoaderCircle, Sparkles, XCircle } from "lucide-react";

import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import { getStageProgressMeta } from "@/lib/progress";
import { stageLabel } from "@/lib/utils";
import { ProjectSummary } from "@/lib/types";

const order = [
  "ingestion",
  "panel_detection",
  "panel_review",
  "character_review",
  "character_portrait",
  "panel_vision_extraction",
  "panel_vision_quality",
  "script_generation",
  "narration_generation",
  "video_rendering"
] as const;

function statusIcon(status: string) {
  if (status === "completed") return <CheckCircle2 className="h-4 w-4 text-accent" />;
  if (status === "running") return <LoaderCircle className="h-4 w-4 animate-spin text-brand-cyan" />;
  if (status === "failed" || status === "cancelled") return <XCircle className="h-4 w-4 text-brand-rose" />;
  if (status === "needs_review") return <Sparkles className="h-4 w-4 text-brand-amber" />;
  return <Clock3 className="h-4 w-4 text-mutedForeground" />;
}

export function StageTimeline({ project }: { project: ProjectSummary }) {
  return (
    <div className="space-y-2">
      {order.map((stage) => {
        const state = project.stage_states[stage];
        const progressMeta = getStageProgressMeta(project, stage);
        const isActive = state.status === "running" || state.status === "needs_review";
        const isCompleted = state.status === "completed";
        const isFailed = state.status === "failed" || state.status === "cancelled";

        return (
          <div
            key={stage}
            className={`rounded-2xl border p-4 transition-all ${
              isActive
                ? "border-accent/30 bg-accent/5"
                : "border-white/8 bg-white/[0.03]"
            }`}
          >
            <div className="flex items-center justify-between gap-3">
              <div className="flex min-w-0 items-center gap-3">
                <div className={`rounded-full p-1.5 ${isActive ? "bg-accent/10" : "bg-white/5"}`}>
                  {statusIcon(state.status)}
                </div>
                <div className="min-w-0">
                  <p className={`text-sm font-medium ${isActive ? "text-accent" : "text-white"}`}>{stageLabel(stage)}</p>
                  {(isActive || isFailed) && state.message ? (
                    <p className="mt-0.5 break-words text-xs text-mutedForeground">{state.message}</p>
                  ) : null}
                </div>
              </div>
              <Badge>{state.status.replace("_", " ")}</Badge>
            </div>

            {isActive ? (
              <div className="mt-3">
                <div className="mb-1.5 flex items-center justify-between gap-3 text-[10px] uppercase tracking-[0.18em] text-mutedForeground">
                  <span>{progressMeta.stateLabel ?? "Running"}</span>
                  <span className="tabular-nums">{progressMeta.progress}%</span>
                </div>
                <Progress value={progressMeta.progress} />
              </div>
            ) : isCompleted ? (
              <div className="mt-2">
                <div className="h-1.5 w-full overflow-hidden rounded-full bg-white/8">
                  <div className="h-full rounded-full bg-accent/50" style={{ width: "100%" }} />
                </div>
              </div>
            ) : null}
          </div>
        );
      })}
    </div>
  );
}
