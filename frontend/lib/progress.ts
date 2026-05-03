import type { JobRecord, PipelineStage, ProjectDetail, ProjectSummary } from "@/lib/types";

type ProjectWithProgress = Pick<ProjectSummary | ProjectDetail, "active_jobs" | "stage_states" | "page_count" | "panel_count" | "kept_panel_count">;
type PhaseEstimate = {
  startProgress: number;
  endProgress: number;
  expectedDurationMs: number;
};

type CountEstimate = {
  startProgress: number;
  endProgress: number;
  current: number;
  total: number;
};

function clampProgress(progress: number) {
  return Math.max(0, Math.min(100, Math.round(progress)));
}

function parseTimestamp(value?: string | null) {
  if (!value) return 0;
  const timestamp = Date.parse(value);
  return Number.isFinite(timestamp) ? timestamp : 0;
}

function normalizeMessage(message?: string | null) {
  return (message ?? "").trim().toLowerCase();
}

function activeStageMessage(job?: JobRecord, stateMessage?: string | null) {
  const stateText = String(stateMessage ?? "").trim();
  const jobText = String(job?.message ?? "").trim();
  return stateText || jobText || "";
}

function parseCountPair(message: string, patterns: RegExp[]) {
  for (const pattern of patterns) {
    const match = message.match(pattern);
    if (!match) continue;
    const current = Number(match[1]);
    const total = Number(match[2]);
    if (Number.isFinite(current) && Number.isFinite(total) && total > 0) {
      return { current, total };
    }
  }
  return null;
}

function estimateCountProgress(stage: PipelineStage, message?: string | null): CountEstimate | null {
  const normalized = normalizeMessage(message);
  if (!normalized) return null;

  if (stage === "panel_detection") {
    const pageMatch = parseCountPair(normalized, [
      /detected panels on page\s+(\d+)\s*\/\s*(\d+)/,
      /fast contour pass on tall page\s+(\d+)\s*\/\s*(\d+)/,
      /page\s+(\d+)\s*\/\s*(\d+)/
    ]);
    if (pageMatch) {
      return { startProgress: 8, endProgress: 70, ...pageMatch };
    }
  }

  if (stage === "script_generation") {
    const scanMatch = parseCountPair(normalized, [
      /page\s+(\d+)\s*\/\s*(\d+)/
    ]);
    if (scanMatch && normalized.includes("page")) {
      return { startProgress: 14, endProgress: 70, ...scanMatch };
    }
  }

  if (stage === "character_portrait") {
    const pageMatch = parseCountPair(normalized, [
      /pages?\s+(\d+)\s*-\s*(\d+)/,
      /page\s+(\d+)\s*\/\s*(\d+)/
    ]);
    if (pageMatch) {
      return { startProgress: 8, endProgress: 92, ...pageMatch };
    }
  }

  if (stage === "panel_vision_extraction") {
    const batchMatch = parseCountPair(normalized, [
      /batch\s+(\d+)\s*\/\s*(\d+)/
    ]);
    if (batchMatch) {
      return { startProgress: 8, endProgress: 94, ...batchMatch };
    }
  }

  if (stage === "panel_vision_quality") {
    const panelMatch = parseCountPair(normalized, [
      /panel\s+(\d+)\s*\/\s*(\d+)/
    ]);
    if (panelMatch) {
      return { startProgress: 12, endProgress: 96, ...panelMatch };
    }
  }

  if (stage === "character_review") {
    const scanMatch = parseCountPair(normalized, [
      /page\s+(\d+)\s*\/\s*(\d+)/
    ]);
    if (scanMatch && normalized.includes("page")) {
      return { startProgress: 10, endProgress: 88, ...scanMatch };
    }
  }

  if (stage === "narration_generation") {
    const cacheMatch = parseCountPair(normalized, [
      /prepared narration sentence cache\s+(\d+)\s*\/\s*(\d+)/,
      /prepared narration sentence cache\s+(\d+)\s+of\s+(\d+)/
    ]);
    if (cacheMatch) {
      return { startProgress: 10, endProgress: 36, ...cacheMatch };
    }

    const synthMatch = parseCountPair(normalized, [
      /synthesized narration clip\s+(\d+)\s*\/\s*(\d+)/,
      /generated voice clip\s+(\d+)\s*\/\s*(\d+)/,
      /reused voice clip\s+(\d+)\s*\/\s*(\d+)/,
      /reused shared voice clip\s+(\d+)\s*\/\s*(\d+)/
    ]);
    if (synthMatch) {
      return { startProgress: 36, endProgress: 80, ...synthMatch };
    }

    const cloneMatch = parseCountPair(normalized, [
      /prepared voice-clone pass\s+(\d+)\s*\/\s*(\d+)/,
      /prepared voice-clone pass\s+(\d+)\s+of\s+(\d+)/
    ]);
    if (cloneMatch) {
      return { startProgress: 80, endProgress: 88, ...cloneMatch };
    }

    const masterMatch = parseCountPair(normalized, [
      /mastered narration clip\s+(\d+)\s*\/\s*(\d+)/,
      /mastered narration clip\s+(\d+)\s+of\s+(\d+)/
    ]);
    if (masterMatch) {
      return { startProgress: 88, endProgress: 100, ...masterMatch };
    }
  }

  if (stage === "video_rendering") {
    const clipMatch = parseCountPair(normalized, [
      /prepared panel clip\s+(\d+)\s+of\s+(\d+)/,
      /prepared panel clip\s+(\d+)\s*\/\s*(\d+)/
    ]);
    if (clipMatch) {
      return { startProgress: 8, endProgress: 84, ...clipMatch };
    }
  }

  return null;
}

function findStageJob(project: ProjectWithProgress, stage: PipelineStage): JobRecord | undefined {
  return project.active_jobs
    .filter((job) => job.stage === stage && (job.status === "running" || job.status === "queued"))
    .sort((left, right) => {
      const leftRank = left.status === "running" ? 2 : 1;
      const rightRank = right.status === "running" ? 2 : 1;
      if (leftRank !== rightRank) return rightRank - leftRank;
      if (left.progress !== right.progress) return right.progress - left.progress;
      return Math.max(parseTimestamp(right.started_at), parseTimestamp(right.created_at))
        - Math.max(parseTimestamp(left.started_at), parseTimestamp(left.created_at));
    })[0];
}

function estimateStageDurationMs(project: ProjectWithProgress, stage: PipelineStage) {
  const pageCount = Math.max(1, Number(project.page_count ?? 0));
  const panelCount = Math.max(1, Number(project.panel_count ?? 0));
  const keptPanelCount = Math.max(1, Number(project.kept_panel_count ?? 0));

  switch (stage) {
    case "ingestion":
      return 6_000 + pageCount * 1_400;
    case "panel_detection":
      return 10_000 + pageCount * 2_200;
    case "character_review":
      return 7_000 + keptPanelCount * 300;
    case "character_portrait":
      return 5_000 + pageCount * 650;
    case "panel_vision_extraction":
      return 8_000 + keptPanelCount * 420;
    case "panel_vision_quality":
      return 7_000 + keptPanelCount * 220;
    case "script_generation":
      return 8_000 + pageCount * 1_200 + keptPanelCount * 950;
    case "narration_generation":
      return 8_000 + keptPanelCount * 1_500;
    case "video_rendering":
      return 18_000 + keptPanelCount * 2_200 + panelCount * 400;
    default:
      return 20_000;
  }
}

function estimatePhaseFromMessage(project: ProjectWithProgress, stage: PipelineStage, message?: string | null): PhaseEstimate | null {
  const normalized = normalizeMessage(message);
  const pageCount = Math.max(1, Number(project.page_count ?? 0));
  const panelCount = Math.max(1, Number(project.panel_count ?? 0));
  const keptPanelCount = Math.max(1, Number(project.kept_panel_count ?? 0));

  if (!normalized) return null;

  if (stage === "ingestion") {
    if (normalized.includes("fetching mangadex chapter metadata")) {
      return { startProgress: 2, endProgress: 14, expectedDurationMs: 4_000 };
    }
    if (normalized.includes("scanning uploaded files")) {
      return { startProgress: 3, endProgress: 18, expectedDurationMs: 4_000 + pageCount * 90 };
    }
    if (normalized.includes("prepared source")) {
      return { startProgress: 18, endProgress: 32, expectedDurationMs: 3_500 + pageCount * 80 };
    }
    if (normalized.includes("normalizing downloaded pages") || normalized.includes("normalizing pages")) {
      return { startProgress: 32, endProgress: 96, expectedDurationMs: 5_000 + pageCount * 850 };
    }
    if (normalized.includes("normalised page")) {
      return { startProgress: 32, endProgress: 96, expectedDurationMs: 5_000 + pageCount * 850 };
    }
  }

  if (stage === "panel_detection") {
    if (normalized.includes("preparing panel detector")) {
      return { startProgress: 4, endProgress: 8, expectedDurationMs: 3_000 };
    }
    if (normalized.includes("detected panels on page") || normalized.includes("fast contour pass")) {
      return { startProgress: 8, endProgress: 70, expectedDurationMs: 6_000 + pageCount * 1_100 };
    }
    if (normalized.includes("reconstructing panels from full-page ocr")) {
      return { startProgress: 70, endProgress: 90, expectedDurationMs: 4_000 + pageCount * 220 };
    }
    if (normalized.includes("linking cross-page continuation panels")) {
      return { startProgress: 90, endProgress: 96, expectedDurationMs: 2_000 + pageCount * 120 };
    }
    if (normalized.includes("saving detected panels")) {
      return { startProgress: 96, endProgress: 99, expectedDurationMs: 2_500 };
    }
  }

  if (stage === "script_generation") {
    if (normalized.includes("extracting dialogue and scene context")) {
      return { startProgress: 12, endProgress: 74, expectedDurationMs: 7_000 + keptPanelCount * 260 };
    }
    if (normalized.includes("scanning panels for dialogue candidates")) {
      return { startProgress: 14, endProgress: 46, expectedDurationMs: 5_000 + keptPanelCount * 120 };
    }
    if (normalized.includes("analyzing speaker and character layout") || normalized.includes("analysing speaker and character layout")) {
      return { startProgress: 54, endProgress: 60, expectedDurationMs: 3_000 + keptPanelCount * 50 };
    }
    if (normalized.includes("linking recurring characters across dialogue panels")) {
      return { startProgress: 60, endProgress: 64, expectedDurationMs: 2_500 + keptPanelCount * 40 };
    }
    if (normalized.includes("resolving character names and speakers")) {
      return { startProgress: 64, endProgress: 68, expectedDurationMs: 2_000 + keptPanelCount * 35 };
    }
    if (normalized.includes("building panel-by-panel dialogue scenes")) {
      return { startProgress: 68, endProgress: 70, expectedDurationMs: 2_000 + keptPanelCount * 25 };
    }
    if (normalized.includes("dialogue and scene context ready")) {
      return { startProgress: 70, endProgress: 74, expectedDurationMs: 1_500 };
    }
    if (normalized.includes("reused cached dialogue extraction")) {
      return { startProgress: 70, endProgress: 74, expectedDurationMs: 2_500 + keptPanelCount * 25 };
    }
    if (normalized.includes("generating story beats and narration lines")) {
      return { startProgress: 74, endProgress: 90, expectedDurationMs: 11_000 + keptPanelCount * 165 };
    }
    if (normalized.includes("saving narration draft")) {
      return { startProgress: 90, endProgress: 92, expectedDurationMs: 3_000 + keptPanelCount * 20 };
    }
    if (normalized.includes("analysing pages with gemini vision for story context")) {
      return { startProgress: 92, endProgress: 99, expectedDurationMs: 6_500 + pageCount * 420 };
    }
  }

  if (stage === "character_portrait") {
    if (normalized.includes("enumerating canonical characters")) {
      return { startProgress: 4, endProgress: 92, expectedDurationMs: 4_000 + pageCount * 500 };
    }
  }

  if (stage === "panel_vision_extraction") {
    if (normalized.includes("extracting panel vision batch")) {
      return { startProgress: 8, endProgress: 94, expectedDurationMs: 6_000 + keptPanelCount * 360 };
    }
  }

  if (stage === "panel_vision_quality") {
    if (normalized.includes("rescuing low-confidence panel")) {
      return { startProgress: 10, endProgress: 95, expectedDurationMs: 6_000 + keptPanelCount * 200 };
    }
  }

  if (stage === "character_review") {
    if (normalized.includes("preparing character review suggestions")) {
      return { startProgress: 0, endProgress: 6, expectedDurationMs: 2_000 };
    }
    if (normalized.includes("extracting dialogue and character context")) {
      return { startProgress: 6, endProgress: 90, expectedDurationMs: 6_000 + keptPanelCount * 220 };
    }
    if (normalized.includes("scanning panels for dialogue candidates")) {
      return { startProgress: 10, endProgress: 58, expectedDurationMs: 4_000 + keptPanelCount * 120 };
    }
    if (normalized.includes("analyzing speaker and character layout") || normalized.includes("analysing speaker and character layout")) {
      return { startProgress: 58, endProgress: 68, expectedDurationMs: 2_500 + keptPanelCount * 45 };
    }
    if (normalized.includes("linking recurring characters across dialogue panels")) {
      return { startProgress: 68, endProgress: 76, expectedDurationMs: 2_000 + keptPanelCount * 35 };
    }
    if (normalized.includes("resolving character names and speakers")) {
      return { startProgress: 76, endProgress: 84, expectedDurationMs: 2_000 + keptPanelCount * 30 };
    }
    if (normalized.includes("building panel-by-panel dialogue scenes")) {
      return { startProgress: 84, endProgress: 90, expectedDurationMs: 1_500 + keptPanelCount * 20 };
    }
    if (normalized.includes("building character review cards")) {
      return { startProgress: 90, endProgress: 98, expectedDurationMs: 2_000 + keptPanelCount * 15 };
    }
    if (normalized.includes("character review suggestions ready") || normalized.includes("no recurring characters needed review")) {
      return { startProgress: 98, endProgress: 100, expectedDurationMs: 1_000 };
    }
  }

  if (stage === "narration_generation") {
    if (normalized.includes("generating voice narration")) {
      return { startProgress: 0, endProgress: 4, expectedDurationMs: 2_000 };
    }
    if (normalized.includes("preparing cinematic narration lines")) {
      return { startProgress: 4, endProgress: 10, expectedDurationMs: 2_500 + keptPanelCount * 30 };
    }
    if (normalized.includes("prepared narration sentence cache")) {
      return { startProgress: 10, endProgress: 36, expectedDurationMs: 4_000 + keptPanelCount * 80 };
    }
    if (
      normalized.includes("synthesized narration clip")
      || normalized.includes("generated voice clip")
      || normalized.includes("reused voice clip")
      || normalized.includes("reused shared voice clip")
    ) {
      return { startProgress: 36, endProgress: 80, expectedDurationMs: 6_000 + keptPanelCount * 850 };
    }
    if (normalized.includes("prepared voice-clone pass")) {
      return { startProgress: 80, endProgress: 88, expectedDurationMs: 2_000 + keptPanelCount * 40 };
    }
    if (normalized.includes("mastered narration clip")) {
      return { startProgress: 88, endProgress: 100, expectedDurationMs: 2_500 + keptPanelCount * 55 };
    }
  }

  if (stage === "video_rendering") {
    if (normalized.includes("rendering video with ffmpeg")) {
      return { startProgress: 0, endProgress: 2, expectedDurationMs: 2_000 };
    }
    if (normalized.includes("planned camera travel")) {
      return { startProgress: 2, endProgress: 4, expectedDurationMs: 2_500 + keptPanelCount * 20 };
    }
    if (normalized.includes("built narration timeline")) {
      return { startProgress: 4, endProgress: 8, expectedDurationMs: 2_000 + keptPanelCount * 25 };
    }
    if (normalized.includes("prepared panel clip")) {
      return { startProgress: 8, endProgress: 84, expectedDurationMs: 10_000 + keptPanelCount * 1_350 + panelCount * 180 };
    }
    if (normalized.includes("merging panel timeline")) {
      return { startProgress: 84, endProgress: 92, expectedDurationMs: 5_500 + keptPanelCount * 35 };
    }
    if (normalized.includes("panel timeline ready")) {
      return { startProgress: 92, endProgress: 94, expectedDurationMs: 1_500 };
    }
    if (normalized.includes("prepended thumbnail lead-in")) {
      return { startProgress: 94, endProgress: 95, expectedDurationMs: 1_500 };
    }
    if (normalized.includes("preparing final video export")) {
      return { startProgress: 95, endProgress: 97, expectedDurationMs: 3_500 };
    }
    if (normalized.includes("muxed narration and picture")) {
      return { startProgress: 97, endProgress: 99, expectedDurationMs: 3_000 };
    }
    if (normalized.includes("mixed background music")) {
      return { startProgress: 99, endProgress: 100, expectedDurationMs: 2_000 };
    }
    if (normalized.includes("render")) {
      return { startProgress: 0, endProgress: 99, expectedDurationMs: 14_000 + keptPanelCount * 1_900 + panelCount * 240 };
    }
  }

  return null;
}

function estimateTimeBasedProgress(
  project: ProjectWithProgress,
  stage: PipelineStage,
  message?: string | null,
  job?: JobRecord,
  stateUpdatedAt?: string
) {
  const phaseEstimate = estimatePhaseFromMessage(project, stage, message);
  const startedAt = Math.max(
    parseTimestamp(job?.started_at),
    parseTimestamp(job?.created_at),
    parseTimestamp(stateUpdatedAt)
  );
  if (!startedAt) return 0;
  const elapsedMs = Date.now() - startedAt;
  if (elapsedMs <= 0) return 0;

  const countEstimate = estimateCountProgress(stage, message);
  if (countEstimate) {
    const span = Math.max(1, countEstimate.endProgress - countEstimate.startProgress);
    const ratio = Math.max(0, Math.min(1, countEstimate.current / Math.max(countEstimate.total, 1)));
    return clampProgress(Math.min(99, countEstimate.startProgress + span * ratio));
  }

  if (phaseEstimate) {
    const phaseSpan = Math.max(1, phaseEstimate.endProgress - phaseEstimate.startProgress);
    const phaseRatio = Math.min(0.985, elapsedMs / Math.max(phaseEstimate.expectedDurationMs, 1));
    const estimated = phaseEstimate.startProgress + phaseSpan * phaseRatio;
    return clampProgress(Math.min(99, estimated));
  }

  const expectedDurationMs = estimateStageDurationMs(project, stage);
  if (expectedDurationMs <= 0) return 0;
  const estimated = (elapsedMs / expectedDurationMs) * 100;
  return clampProgress(Math.min(99, estimated));
}

export function getStageProgressMeta(project: ProjectWithProgress, stage: PipelineStage) {
  const state = project.stage_states?.[stage];
  const job = findStageJob(project, stage);
  const message = activeStageMessage(job, state?.message);
  const stateProgress = Number(state?.progress ?? 0);
  const jobProgress = Number(job?.progress ?? 0);
  const isRunning = job?.status === "running" || state?.status === "running";
  // Only treat a stage as "Queued" when there is an actual queued job record.
  // A stage in "ready" state without a job just means it's ready to be started —
  // showing it as "Queued" causes the progress bar to misleadingly show unrelated
  // stages (e.g. "Script generation • 0%") after an adjacent stage fails.
  const isQueued = !isRunning && job?.status === "queued";
  const rawProgress = clampProgress(Math.max(stateProgress, jobProgress));
  const estimatedProgress = isRunning ? estimateTimeBasedProgress(project, stage, message, job, state?.updated_at) : 0;
  const progress = clampProgress(Math.max(rawProgress, estimatedProgress));

  return {
    progress,
    stateLabel: isRunning ? "Running" : isQueued ? "Queued" : null,
    job,
    estimatedProgress: estimatedProgress || null,
    remainingMs: null,
    etaLabel: null,
    message: message || null
  };
}
