"use client";

import Link from "next/link";
import { useParams, useRouter } from "next/navigation";
import { useEffect, useMemo, useRef, useState } from "react";
import { ArrowRight, Download, LoaderCircle, Lock, Pencil, RefreshCcw, ScanSearch, Scissors, Unlock, UploadCloud, X } from "lucide-react";

import { AppShell } from "@/components/project/app-shell";
import { buildProjectViews } from "@/lib/project-views";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardDescription, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { api } from "@/lib/api";
import { formatProgressPercent, getStageProgressMeta } from "@/lib/progress";
import { CatalogOptions, PanelRewriteMode, PipelineStage, ProjectDetail } from "@/lib/types";
import { useAdaptivePolling } from "@/lib/use-adaptive-polling";
import { buildMediaUrl } from "@/lib/utils";


const SCRIPT_PIPELINE_STAGES = [
  "character_portrait",
  "panel_vision_extraction",
  "panel_vision_quality",
  "script_generation"
] as const satisfies readonly PipelineStage[];

const PROGRESS_LABELS: Partial<Record<PipelineStage, string>> = {
  character_portrait: "Character portraits",
  panel_vision_extraction: "Panel vision",
  panel_vision_quality: "Vision rescue",
  script_generation: "Script generation",
  narration_generation: "Audio generation",
  video_rendering: "Video rendering"
};

interface NarrationLineItem {
  panelId: string;
  order: number;
  page: number;
  panelNumber: number;
  hasExtractedText: boolean;
  keep: boolean;
  value: string;
  extractedText: string;
  visualCaption: string;
  lineSource: "ocr" | "vision_caption" | "fallback" | "";
  previewUrl: string;
  locked: boolean;
}

interface StorySegmentItem {
  id: string;
  order: number;
  title: string;
  value: string;
  keep: boolean;
  panelIds: string[];
  panelStart?: number | null;
  panelEnd?: number | null;
  representativePanelId?: string | null;
  previewPanels: Array<{
    panelId: string;
    page: number;
    panelNumber: number;
    previewUrl: string;
  }>;
  visualOnly?: boolean;
  suppressionReason?: string | null;
}

function sourceLabel(source: NarrationLineItem["lineSource"]) {
  if (source === "ocr") return "Source: OCR";
  if (source === "vision_caption") return "Source: vision";
  if (source === "fallback") return "Source: fallback";
  return "";
}

function extractFactAnchors(text: string) {
  const normalized = text.toLowerCase();
  const anchors = new Set<string>();
  const years = normalized.match(/\b(?:19|20)\d{2}\b/g) ?? [];
  years.slice(0, 2).forEach((year) => anchors.add(year));
  const numericUnits = normalized.match(/\b\d[\d,]*(?:\.\d+)?\s*(?:light-?years?|degrees?|days?|months?|years?)\b/g) ?? [];
  numericUnits.slice(0, 2).forEach((item) => anchors.add(item.replace(/\s+/g, " ").trim()));
  for (const keyword of [
    "supernova",
    "apocalypse",
    "freeze",
    "frozen apocalypse",
    "blizzard",
    "temperature",
    "blue star",
    "storage space",
    "vault door",
    "december",
    "november",
    "january"
  ]) {
    if (normalized.includes(keyword)) anchors.add(keyword);
  }
  return [...anchors];
}

function isFactualPanel(text: string) {
  const normalized = text.toLowerCase();
  if (!normalized.trim()) return false;
  const anchors = extractFactAnchors(normalized);
  if (anchors.length >= 3) return true;
  if (/(causing|caused|leading to|resulting in|which caused)/i.test(normalized)) return true;
  return /\b(?:19|20)\d{2}\b/.test(normalized) && /(supernova|freeze|blizzard|apocalypse|temperature)/i.test(normalized);
}

function isGenericNarration(text: string) {
  const normalized = text.trim().toLowerCase();
  if (!normalized) return true;
  return [
    "the world still feels normal",
    "by the end of the chapter",
    "questions start piling up",
    "another moment changes everything",
    "the situation grows harder to explain",
    "the scene takes a turn"
  ].some((phrase) => normalized.includes(phrase));
}

function hasNarrationMismatch(extractedText: string, narration: string) {
  if (!isFactualPanel(extractedText) || !narration.trim()) return false;
  if (isGenericNarration(narration)) return true;
  const anchors = extractFactAnchors(extractedText);
  if (!anchors.length) return false;
  const loweredNarration = narration.toLowerCase();
  const matched = anchors.filter((anchor) => loweredNarration.includes(anchor));
  const hasYear = anchors.some((anchor) => /^(19|20)\d{2}$/.test(anchor));
  const missingYear = hasYear && !matched.some((anchor) => /^(19|20)\d{2}$/.test(anchor));
  const numericAnchors = anchors.filter((anchor) => /\d/.test(anchor));
  const missingNumeric = numericAnchors.length > 0 && !matched.some((anchor) => /\d/.test(anchor));
  return missingYear || missingNumeric || matched.length === 0;
}

function hasMeaningfulExtractedText(text?: string | null) {
  const cleaned = text?.trim() ?? "";
  if (!cleaned) return false;
  if (/[\u3040-\u30ff\u4e00-\u9fff\uac00-\ud7af]/.test(cleaned)) return true;
  if (/(translator|typesetter|proofreader|scanlation|discord|patreon|credits)/i.test(cleaned)) return false;
  const letters = [...cleaned].filter((char) => /[A-Za-z]/.test(char)).length;
  const digits = [...cleaned].filter((char) => /\d/.test(char)).length;
  if (letters < 2) return false;
  if (digits && digits >= letters) return false;
  if (/^[A-Za-z0-9]{4,}$/.test(cleaned) && digits > 0) return false;
  if (/^[A-Z0-9]{5,}$/.test(cleaned)) return false;
  if (!cleaned.includes(" ") && letters >= 5 && !/[aeiou]/i.test(cleaned)) return false;
  const latinTokens = (cleaned.match(/[A-Za-z']+/g) ?? []).map((token) => token.toLowerCase());
  if (latinTokens.length && !latinTokens.some((token) => token.length >= 3 || ["go", "no", "run", "wait", "stop", "help", "why", "what", "who", "yes"].includes(token))) {
    return false;
  }
  return true;
}

function buildNarrationLineItems(project: ProjectDetail): NarrationLineItem[] {
  const orderedPanels = [...project.panels].sort((a, b) => a.order - b.order);
  const keptPanelScriptLookup = new Map<string, string>();
  let keptIndex = 0;
  for (const panel of orderedPanels) {
    if (!panel.keep) continue;
    keptPanelScriptLookup.set(panel.id, project.script_lines[keptIndex] ?? "");
    keptIndex += 1;
  }

  return orderedPanels.map((panel) => {
    const extractedText = (panel.ocr_text ?? "").trim();
    const savedNarration = panel.narration ?? keptPanelScriptLookup.get(panel.id) ?? "";
    const hasExtractedText =
      Boolean(panel.manual_ocr_text ? extractedText : undefined) ||
      Boolean(panel.text_detected) ||
      hasMeaningfulExtractedText(extractedText);

    return {
      panelId: panel.id,
      order: panel.order,
      page: panel.page,
      panelNumber: panel.panel,
      hasExtractedText,
      keep: panel.keep,
      value: savedNarration,
      extractedText,
      visualCaption: (panel.visual_caption ?? "").trim(),
      lineSource: (panel.narration_source ?? "") as NarrationLineItem["lineSource"],
      previewUrl: `/api/projects/${project.id}/panels/${panel.id}/preview?v=${encodeURIComponent(project.updated_at)}`,
      locked: panel.narration_locked
    };
  });
}

function buildStorySegmentItems(project: ProjectDetail): StorySegmentItem[] {
  const panelsById = new Map(project.panels.map((panel) => [panel.id, panel]));
  return (project.story_segments ?? []).map((segment) => {
    const representativePanelId = segment.representative_panel_id ?? segment.panel_ids[0] ?? null;
    // Sort panel_ids by their global order so previews are always chronological
    const orderedPanelIds = [...segment.panel_ids].sort((a, b) => {
      const pa = panelsById.get(a);
      const pb = panelsById.get(b);
      return (pa?.order ?? 0) - (pb?.order ?? 0);
    });
    const distributedPanelIds = orderedPanelIds.length <= 3
      ? orderedPanelIds
      : [orderedPanelIds[0], orderedPanelIds[Math.floor(orderedPanelIds.length / 2)], orderedPanelIds[orderedPanelIds.length - 1]];
    const previewPanelIds = representativePanelId
      ? [representativePanelId, ...distributedPanelIds]
      : distributedPanelIds;
    const uniquePreviewPanelIds = [...new Set(previewPanelIds)].filter((panelId) => panelsById.has(panelId)).slice(0, 3);
    const previewPanels = uniquePreviewPanelIds.map((panelId) => {
      const panel = panelsById.get(panelId);
      return {
        panelId,
        page: panel?.page ?? 0,
        panelNumber: panel?.panel ?? 0,
        previewUrl: `/api/projects/${project.id}/panels/${panelId}/preview?v=${encodeURIComponent(project.updated_at)}`
      };
    });
    return {
      id: segment.id,
      order: segment.order,
      title: (segment.title ?? `Panel ${segment.order}`).trim() || `Panel ${segment.order}`,
      value: segment.text ?? "",
      keep: segment.keep ?? true,
      panelIds: segment.panel_ids ?? [],
      panelStart: segment.panel_start,
      panelEnd: segment.panel_end,
      representativePanelId,
      previewPanels,
      visualOnly: Boolean(segment.visual_only),
      suppressionReason: segment.suppression_reason ?? null
    };
  });
}

function composeStoryFromLines(items: NarrationLineItem[]) {
  const nonEmpty = items.filter((item) => item.keep).map((item) => item.value.trim()).filter(Boolean);
  if (!nonEmpty.length) return "";
  const paragraphs: string[] = [];
  for (let index = 0; index < nonEmpty.length; index += 3) {
    paragraphs.push(nonEmpty.slice(index, index + 3).join(" "));
  }
  return paragraphs.join("\n\n");
}

function composeStoryFromSegments(items: StorySegmentItem[]) {
  const nonEmpty = items.filter((item) => item.keep).map((item) => item.value.trim()).filter(Boolean);
  if (!nonEmpty.length) return "";
  const paragraphs: string[] = [];
  for (let index = 0; index < nonEmpty.length; index += 2) {
    paragraphs.push(nonEmpty.slice(index, index + 2).join(" "));
  }
  return paragraphs.join("\n\n");
}

export default function NarrationPage() {
  const params = useParams<{ projectId: string }>();
  const router = useRouter();
  const projectId = Array.isArray(params.projectId) ? params.projectId[0] : params.projectId;
  const [project, setProject] = useState<ProjectDetail | null>(null);
  const [catalog, setCatalog] = useState<CatalogOptions | null>(null);
  const [lineItems, setLineItems] = useState<NarrationLineItem[]>([]);
  const [segmentItems, setSegmentItems] = useState<StorySegmentItem[]>([]);
  const [voice, setVoice] = useState("af_bella");
  const [langCode, setLangCode] = useState("a");
  const [speed, setSpeed] = useState(1);
  const [musicEnabled, setMusicEnabled] = useState(false);
  const [musicTrack, setMusicTrack] = useState("");
  const [musicVolume, setMusicVolume] = useState(0.14);
  const [musicUploadFile, setMusicUploadFile] = useState<File | null>(null);
  const [uploadingMusic, setUploadingMusic] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [statusMessage, setStatusMessage] = useState<string | null>(null);
  const [savingSettings, setSavingSettings] = useState(false);
  const [scriptDirty, setScriptDirty] = useState(false);
  const [queueingStage, setQueueingStage] = useState<"script_generation" | "narration_generation" | "video_rendering" | null>(null);
  const [cancellingStage, setCancellingStage] = useState<"script_generation" | "narration_generation" | "video_rendering" | null>(null);
  const [rewritingPanels, setRewritingPanels] = useState<Record<string, PanelRewriteMode | null>>({});
  const [savingScript, setSavingScript] = useState(false);
  const [rewindingStage, setRewindingStage] = useState(false);
  const [storyEditMode, setStoryEditMode] = useState(false);
  const [editableStory, setEditableStory] = useState("");
  const lineItemsRef = useRef<NarrationLineItem[]>([]);
  const segmentItemsRef = useRef<StorySegmentItem[]>([]);
  const scriptDirtyRef = useRef(false);

  async function loadProject(initial = false) {
    if (!projectId) return;
    try {
      // Race the heavy /api/projects/{id} call against a 6-second
      // timeout. When the worker is mid-write on TTS/render the slow
      // endpoint can take 8-15s, which on the initial page load leaves
      // the editor blank. On timeout we fall back to the lightweight
      // summary + script-only endpoints (both <200ms) so the page
      // populates immediately with cached state. The full payload
      // resolves in the background later (next poll tick).
      const TIMEOUT_MS = 6000;
      const slowProjectPromise = api.getProject(projectId);
      let projectPayload: ProjectDetail | null = null;
      try {
        projectPayload = await Promise.race([
          slowProjectPromise,
          new Promise<ProjectDetail>((_, reject) =>
            setTimeout(() => reject(new Error("getProject timeout")), TIMEOUT_MS),
          ),
        ]);
      } catch (raceErr) {
        // Timed out. Try summary + script-only as a fast path.
        try {
          const summary = await api.getProjectSummary(projectId);
          // Merge summary fields with empty defaults for the deeper
          // fields the page expects. The polling tick will overwrite
          // with the real payload when the slow call eventually returns.
          projectPayload = {
            ...summary,
            panels: [],
            story_segments: [],
            voice_config: summary.voice_config,
            video_config: summary.video_config,
            music_config: summary.music_config,
          } as unknown as ProjectDetail;
        } catch (summaryErr) {
          throw raceErr;
        }
      }
      setProject(projectPayload);
      if (initial || !scriptDirtyRef.current) {
        const nextLineItems = buildNarrationLineItems(projectPayload);
        let nextSegmentItems = buildStorySegmentItems(projectPayload);
        // Fallback: when the worker is mid-write on TTS/render the heavy
        // /api/projects/{id} endpoint sometimes returns an older project
        // snapshot whose story_segments are empty or stale, OR we fell
        // through above with the summary-only payload. Pull the
        // lightweight script-only endpoint and merge segments in.
        if (nextSegmentItems.length === 0) {
          try {
            const scriptOnly = await api.getStoryScript(projectId);
            const merged: ProjectDetail = {
              ...projectPayload,
              story_segments: (scriptOnly.story_segments ?? []) as ProjectDetail["story_segments"],
            };
            nextSegmentItems = buildStorySegmentItems(merged);
            if (nextSegmentItems.length > 0) {
              setProject(merged);
            }
          } catch (fallbackErr) {
            // ignore, leave nextSegmentItems empty
          }
        }
        setLineItems(nextLineItems);
        setSegmentItems(nextSegmentItems);
        lineItemsRef.current = nextLineItems;
        segmentItemsRef.current = nextSegmentItems;
      }
      if (initial) {
        setVoice(projectPayload.voice_config.voice);
        setLangCode(projectPayload.voice_config.lang_code);
        setSpeed(projectPayload.voice_config.speed);
        setMusicEnabled(projectPayload.music_config.enabled);
        setMusicTrack(projectPayload.music_config.track_name ?? "");
        setMusicVolume(projectPayload.music_config.volume);
      }
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unable to load project.");
    }
  }

  useEffect(() => {
    if (!projectId) return;
    void loadProject(true);
  }, [projectId]);

  useEffect(() => {
    let cancelled = false;

    async function loadCatalog() {
      try {
        const catalogPayload = await api.getCatalogOptions();
        if (cancelled) return;
        setCatalog(catalogPayload);
      } catch (err) {
        if (cancelled) return;
        setError(err instanceof Error ? err.message : "Unable to load narrator and music options.");
      }
    }

    void loadCatalog();

    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    const validVoices = (catalog?.voices ?? []).filter((option) => option.lang_code === langCode);
    if (validVoices.length && !validVoices.some((option) => option.id === voice)) {
      setVoice(validVoices[0].id);
    }
  }, [catalog, langCode, voice]);

  const usingStorySegments = segmentItems.length > 0;
  const scriptLines = useMemo(
    () => (
      usingStorySegments
        ? segmentItems.filter((item) => item.keep).map((item) => item.value)
        : lineItems.filter((item) => item.keep).map((item) => item.value)
    ),
    [lineItems, segmentItems, usingStorySegments]
  );
  const storyPreview = useMemo(
    () => (usingStorySegments ? composeStoryFromSegments(segmentItems) : composeStoryFromLines(lineItems)),
    [lineItems, segmentItems, usingStorySegments]
  );
  const hasGeneratedScript = scriptLines.some((line) => line.trim().length > 0);
  const keptPanelCount = useMemo(
    () => (usingStorySegments ? segmentItems.filter((item) => item.keep).length : lineItems.filter((item) => item.keep).length),
    [lineItems, segmentItems, usingStorySegments]
  );
  const visualOnlySegmentCount = useMemo(
    () => (usingStorySegments ? segmentItems.filter((item) => item.keep && item.visualOnly).length : 0),
    [segmentItems, usingStorySegments]
  );
  const spokenSegmentCount = useMemo(
    () => (usingStorySegments ? segmentItems.filter((item) => item.keep && item.value.trim()).length : 0),
    [segmentItems, usingStorySegments]
  );
  const blankKeptLineCount = useMemo(
    () => (usingStorySegments ? segmentItems.filter((item) => item.keep && !item.value.trim()).length : lineItems.filter((item) => item.keep && !item.value.trim()).length),
    [lineItems, segmentItems, usingStorySegments]
  );
  const voiceOptions = (catalog?.voices ?? []).filter((option) => option.lang_code === langCode);
  const selectedVoice = voiceOptions.find((option) => option.id === voice);
  const selectedLanguage = (catalog?.languages ?? []).find((option) => option.code === langCode);
  const selectedTrack = (catalog?.music_tracks ?? []).find((track) => track.name === musicTrack);
  const scriptStageStatus = project?.stage_states.script_generation.status;
  const audioStageStatus = project?.stage_states.narration_generation.status;
  const videoStageStatus = project?.stage_states.video_rendering.status;
  const scriptDisplayMetadata = project?.script_display_metadata;
  const activeJobs = project?.active_jobs ?? [];
  const activeStageJobs = (stage: PipelineStage) =>
    activeJobs.filter((job) => job.stage === stage && (job.status === "queued" || job.status === "running"));
  const characterPortraitProgressMeta = project ? getStageProgressMeta(project, "character_portrait") : null;
  const panelVisionProgressMeta = project ? getStageProgressMeta(project, "panel_vision_extraction") : null;
  const panelVisionQualityProgressMeta = project ? getStageProgressMeta(project, "panel_vision_quality") : null;
  const scriptProgressMeta = project ? getStageProgressMeta(project, "script_generation") : null;
  const audioProgressMeta = project ? getStageProgressMeta(project, "narration_generation") : null;
  const videoProgressMeta = project ? getStageProgressMeta(project, "video_rendering") : null;
  const scriptBusy =
    queueingStage === "script_generation" ||
    scriptStageStatus === "running" ||
    SCRIPT_PIPELINE_STAGES.some((stage) => activeStageJobs(stage).length > 0 || project?.stage_states[stage]?.status === "running");
  const audioBusy =
    queueingStage === "narration_generation" ||
    audioStageStatus === "running" ||
    activeStageJobs("narration_generation").length > 0;
  const videoBusy =
    queueingStage === "video_rendering" ||
    videoStageStatus === "running" ||
    activeStageJobs("video_rendering").length > 0;
  const activeProgressStage = useMemo(() => {
    if (queueingStage) return queueingStage;
    const stages = [
      { stage: "character_portrait" as const, meta: characterPortraitProgressMeta },
      { stage: "panel_vision_extraction" as const, meta: panelVisionProgressMeta },
      { stage: "panel_vision_quality" as const, meta: panelVisionQualityProgressMeta },
      { stage: "script_generation" as const, meta: scriptProgressMeta },
      { stage: "narration_generation" as const, meta: audioProgressMeta },
      { stage: "video_rendering" as const, meta: videoProgressMeta }
    ];
    const running = stages.find((item) => item.meta?.stateLabel === "Running");
    if (running) return running.stage;
    const queued = stages.find((item) => item.meta?.stateLabel === "Queued");
    return queued?.stage ?? null;
  }, [
    audioProgressMeta,
    characterPortraitProgressMeta,
    panelVisionProgressMeta,
    panelVisionQualityProgressMeta,
    queueingStage,
    scriptProgressMeta,
    videoProgressMeta
  ]);
  const activeProgressMeta =
    activeProgressStage === "character_portrait"
      ? characterPortraitProgressMeta
      : activeProgressStage === "panel_vision_extraction"
        ? panelVisionProgressMeta
        : activeProgressStage === "panel_vision_quality"
          ? panelVisionQualityProgressMeta
          : activeProgressStage === "script_generation"
            ? scriptProgressMeta
            : activeProgressStage === "narration_generation"
              ? audioProgressMeta
              : activeProgressStage === "video_rendering"
                ? videoProgressMeta
                : null;
  const activeProgressMessage = activeProgressStage
    ? `${PROGRESS_LABELS[activeProgressStage] ?? "Pipeline"} • ${formatProgressPercent(activeProgressMeta?.progress ?? 0)}`
    : null;

  useAdaptivePolling(() => loadProject(false), {
    enabled: Boolean(projectId),
    active: Boolean(project?.active_jobs.length),
    activeMs: 8000,
    idleMs: 30000,
    hiddenMs: 120000,
    deps: [projectId]
  });

  function exportScript() {
    const blob = new Blob([storyPreview], { type: "text/plain;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = `${project?.name ?? "panelia-script"}.txt`;
    anchor.click();
    URL.revokeObjectURL(url);
  }

  async function saveScript() {
    if (!project) return;
    setSavingScript(true);
    try {
      let updated: ProjectDetail;
      if (usingStorySegments) {
        const currentSegments = segmentItemsRef.current;
        const currentScriptLines = currentSegments.map((item) => item.value);
        updated = await api.updateScript(
          project.id,
          currentScriptLines,
          {},
          {},
          {},
          currentSegments.map((item) => ({
            id: item.id,
            order: item.order,
            text: item.value,
            keep: item.keep,
            panel_ids: item.panelIds,
            panel_start: item.panelStart ?? null,
            panel_end: item.panelEnd ?? null,
            title: item.title,
            representative_panel_id: item.representativePanelId ?? null,
            visual_only: Boolean(item.visualOnly),
            suppression_reason: item.suppressionReason ?? null
          }))
        );
      } else {
        const currentLineItems = lineItemsRef.current;
        const currentScriptLines = currentLineItems.filter((item) => item.keep).map((item) => item.value);
        const panelKeeps = Object.fromEntries(currentLineItems.map((item) => [item.panelId, item.keep]));
        const panelNarrations = Object.fromEntries(currentLineItems.map((item) => [item.panelId, item.value]));
        const panelLocks = Object.fromEntries(currentLineItems.map((item) => [item.panelId, item.locked]));
        updated = await api.updateScript(project.id, currentScriptLines, panelKeeps, panelNarrations, panelLocks);
      }
      setProject(updated);
      const nextLineItems = buildNarrationLineItems(updated);
      const nextSegmentItems = buildStorySegmentItems(updated);
      setLineItems(nextLineItems);
      setSegmentItems(nextSegmentItems);
      lineItemsRef.current = nextLineItems;
      segmentItemsRef.current = nextSegmentItems;
      setScriptDirty(false);
      scriptDirtyRef.current = false;
      setStatusMessage(usingStorySegments ? "Story script saved." : "Script saved. Locked panel lines will stay in place during future regenerations.");
    } finally {
      setSavingScript(false);
    }
  }

  async function ensureScriptSavedBeforeContinue() {
    if (!project || !scriptDirtyRef.current) return;
    await saveScript();
  }

  /** Convert kept panel lines to a one-line-per-panel edit format. */
  function linesToEditText(items: NarrationLineItem[]): string {
    return items
      .filter((item) => item.keep)
      .map((item) => item.value.trim())
      .join("\n");
  }

  /** Enter edit mode: snapshot the current story into the editable textarea. */
  function enterStoryEditMode() {
    if (usingStorySegments) {
      setEditableStory(segmentItemsRef.current.filter((item) => item.keep).map((item) => item.value.trim()).join("\n"));
    } else {
      setEditableStory(linesToEditText(lineItemsRef.current));
    }
    setStoryEditMode(true);
  }

  /** Save the bulk-edited text back to individual panel line items and persist. */
  async function saveStoryEdit() {
    const rawLines = editableStory.split("\n");
    if (usingStorySegments) {
      const keptItems = segmentItemsRef.current.filter((item) => item.keep);
      const updatedSegments = segmentItemsRef.current.map((item, index) => {
        if (!item.keep) return item;
        const keptIndex = keptItems.indexOf(item);
        const edited = rawLines[keptIndex];
        return edited !== undefined ? { ...item, value: edited } : item;
      });
      segmentItemsRef.current = updatedSegments;
      setSegmentItems(updatedSegments);
    } else {
      const keptItems = lineItemsRef.current.filter((item) => item.keep);
      const updatedItems = lineItemsRef.current.map((item) => {
        if (!item.keep) return item;
        const keptIndex = keptItems.indexOf(item);
        const edited = rawLines[keptIndex];
        return edited !== undefined ? { ...item, value: edited } : item;
      });
      lineItemsRef.current = updatedItems;
      setLineItems(updatedItems);
    }
    setScriptDirty(true);
    scriptDirtyRef.current = true;
    setStoryEditMode(false);
    await saveScript();
  }

  async function saveSettings() {
    if (!project) return;
    try {
      setSavingSettings(true);
      const updated = await api.updateProjectSettings(project.id, {
        voice_config: {
          voice,
          lang_code: langCode,
          speed
        },
        music_config: {
          enabled: musicEnabled,
          track_name: musicTrack || null,
          volume: musicVolume,
          fade_in_seconds: project.music_config.fade_in_seconds,
          fade_out_seconds: project.music_config.fade_out_seconds
        }
      });
      setProject(updated);
      setVoice(updated.voice_config.voice);
      setLangCode(updated.voice_config.lang_code);
      setSpeed(updated.voice_config.speed);
      setMusicEnabled(updated.music_config.enabled);
      setMusicTrack(updated.music_config.track_name ?? "");
      setMusicVolume(updated.music_config.volume);
      setStatusMessage("Narration settings saved. Downstream stages will refresh automatically.");
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unable to save narration settings.");
    } finally {
      setSavingSettings(false);
    }
  }

  async function handleMusicUpload() {
    if (!musicUploadFile) {
      return;
    }

    try {
      setUploadingMusic(true);
      const uploaded = await api.uploadMusicTrack(musicUploadFile);
      const refreshedCatalog = await api.getCatalogOptions();
      setCatalog(refreshedCatalog);
      setMusicTrack(uploaded.name);
      setMusicEnabled(true);
      setMusicUploadFile(null);
      setStatusMessage("Custom music imported. Save narration settings to use it on this project.");
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unable to import the MP3 track.");
    } finally {
      setUploadingMusic(false);
    }
  }

  async function rewindTo(stage: "panel_review" | "script_generation" | "narration_generation") {
    if (!project) return;
    setRewindingStage(true);
    try {
      const updated = await api.rewindProject(project.id, stage);
      setProject(updated);
      if (stage === "panel_review") {
        router.push(`/projects/${project.id}/editor`);
        return;
      }
      if (stage === "script_generation") {
        setStatusMessage("Script review reopened.");
        return;
      }
      setStatusMessage("Audio step reopened.");
    } finally {
      setRewindingStage(false);
    }
  }

  async function queueStage(stage: "script_generation" | "narration_generation" | "video_rendering") {
    if (!project) return;
    setError(null);
    setQueueingStage(stage);
    try {
      if (stage !== "script_generation") {
        await ensureScriptSavedBeforeContinue();
      } else {
        setScriptDirty(false);
        scriptDirtyRef.current = false;
      }
      await api.queueStage(
        project.id,
        stage,
        stage === "script_generation"
          ? { stop_after_stage: true, force_refresh: true }
          : stage === "narration_generation"
            ? { force_quality_bypass: true }
            : {}
      );
      const refreshed = await api.getProject(project.id);
      setProject(refreshed);
      if (!scriptDirtyRef.current) {
        const nextLineItems = buildNarrationLineItems(refreshed);
        const nextSegmentItems = buildStorySegmentItems(refreshed);
        setLineItems(nextLineItems);
        setSegmentItems(nextSegmentItems);
        lineItemsRef.current = nextLineItems;
        segmentItemsRef.current = nextSegmentItems;
      }
      if (stage === "script_generation") {
        setStatusMessage("Fresh script regeneration queued. Panelia will ignore the previous script caches and stop after the script stage so your Mac is not forced straight into audio generation.");
      } else if (stage === "narration_generation") {
        setStatusMessage("Audio regeneration queued.");
      } else {
        setStatusMessage("Video rendering queued. Open Preview to watch the export update.");
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unable to queue this stage.");
    } finally {
      setQueueingStage(null);
    }
  }

  async function repairWeakStorySegments() {
    if (!project || !usingStorySegments) return;
    setError(null);
    setQueueingStage("script_generation");
    try {
      await ensureScriptSavedBeforeContinue();
      await api.queueStage(project.id, "script_generation", {
        repair_weak_segments: true,
        stop_after_stage: true
      });
      const refreshed = await api.getProject(project.id);
      setProject(refreshed);
      if (!scriptDirtyRef.current) {
        const nextLineItems = buildNarrationLineItems(refreshed);
        const nextSegmentItems = buildStorySegmentItems(refreshed);
        setLineItems(nextLineItems);
        setSegmentItems(nextSegmentItems);
        lineItemsRef.current = nextLineItems;
        segmentItemsRef.current = nextSegmentItems;
      }
      setStatusMessage("Weak segment repair queued. Panelia will improve blank and visual-only story beats without redrafting the whole script.");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unable to queue script repair.");
    } finally {
      setQueueingStage(null);
    }
  }

  async function cancelStage(stage: "script_generation" | "narration_generation" | "video_rendering") {
    if (!project) return;
    const jobs = stage === "script_generation"
      ? SCRIPT_PIPELINE_STAGES.flatMap((pipelineStage) => activeStageJobs(pipelineStage))
      : activeStageJobs(stage);
    if (!jobs.length) return;
    setError(null);
    setCancellingStage(stage);
    try {
      await Promise.all(jobs.map((job) => api.cancelJob(project.id, job.id)));
      const refreshed = await api.getProject(project.id);
      setProject(refreshed);
      setStatusMessage(
        stage === "script_generation"
          ? "Script generation cancellation requested."
          : stage === "narration_generation"
            ? "Audio generation cancellation requested."
            : "Video rendering cancellation requested."
      );
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unable to cancel this stage.");
    } finally {
      setCancellingStage(null);
    }
  }

  async function toggleStage(stage: "script_generation" | "narration_generation" | "video_rendering") {
    const stageIsBusy =
      (stage === "script_generation" && scriptBusy) ||
      (stage === "narration_generation" && audioBusy) ||
      (stage === "video_rendering" && videoBusy);
    if (stageIsBusy) {
      await cancelStage(stage);
      return;
    }
    await queueStage(stage);
  }

  function updateLineValue(index: number, value: string) {
    setLineItems((current) => {
      const nextItems = current.map((item, itemIndex) => (itemIndex === index ? { ...item, value, locked: true } : item));
      lineItemsRef.current = nextItems;
      return nextItems;
    });
    setScriptDirty(true);
    scriptDirtyRef.current = true;
  }

  function updateSegmentValue(index: number, value: string) {
    setSegmentItems((current) => {
      const nextItems = current.map((item, itemIndex) => (itemIndex === index ? { ...item, value } : item));
      segmentItemsRef.current = nextItems;
      return nextItems;
    });
    setScriptDirty(true);
    scriptDirtyRef.current = true;
  }

  function toggleSegmentKeep(index: number) {
    setSegmentItems((current) => {
      const nextItems = current.map((item, itemIndex) => (itemIndex === index ? { ...item, keep: !item.keep } : item));
      segmentItemsRef.current = nextItems;
      return nextItems;
    });
    setScriptDirty(true);
    scriptDirtyRef.current = true;
  }

  function toggleLineKeep(index: number) {
    setLineItems((current) => {
      const nextItems = current.map((item, itemIndex) => (itemIndex === index ? { ...item, keep: !item.keep } : item));
      lineItemsRef.current = nextItems;
      return nextItems;
    });
    setScriptDirty(true);
    scriptDirtyRef.current = true;
  }

  function excludePanelsWithoutScript() {
    const excludedCount = lineItemsRef.current.filter((item) => item.keep && !item.value.trim()).length;
    setLineItems((current) => {
      const nextItems = current.map((item) => (item.keep && !item.value.trim() ? { ...item, keep: false, locked: false } : item));
      lineItemsRef.current = nextItems;
      return nextItems;
    });
    if (excludedCount > 0) {
      setScriptDirty(true);
      scriptDirtyRef.current = true;
      setStatusMessage(`Marked ${excludedCount} blank-script panel${excludedCount === 1 ? "" : "s"} for exclusion. Save script to apply.`);
    } else {
      setStatusMessage("No kept panels without script were found.");
    }
  }

  function excludeSegmentsWithoutScript() {
    const excludedCount = segmentItemsRef.current.filter((item) => item.keep && !item.value.trim()).length;
    setSegmentItems((current) => {
      const nextItems = current.map((item) => (item.keep && !item.value.trim() ? { ...item, keep: false } : item));
      segmentItemsRef.current = nextItems;
      return nextItems;
    });
    if (excludedCount > 0) {
      setScriptDirty(true);
      scriptDirtyRef.current = true;
      setStatusMessage(`Marked ${excludedCount} blank segment${excludedCount === 1 ? "" : "s"} for exclusion. Save script to apply.`);
    } else {
      setStatusMessage("No included segments without script were found.");
    }
  }

  function toggleLineLock(index: number) {
    setLineItems((current) => {
      const nextItems = current.map((item, itemIndex) => (itemIndex === index ? { ...item, locked: !item.locked } : item));
      lineItemsRef.current = nextItems;
      return nextItems;
    });
    setScriptDirty(true);
    scriptDirtyRef.current = true;
  }

  async function rewriteLine(index: number, mode: PanelRewriteMode) {
    if (!project) return;
    const item = lineItems[index];
    setRewritingPanels((current) => ({ ...current, [item.panelId]: mode }));
    setError(null);
    try {
      const rewritten = await api.rewritePanelNarration(project.id, item.panelId, mode, item.value);
      setLineItems((current) =>
        {
          const nextItems = current.map((entry, itemIndex) =>
          itemIndex === index
            ? {
                ...entry,
                value: rewritten.narration,
                locked: true
              }
            : entry
          );
          lineItemsRef.current = nextItems;
          return nextItems;
        }
      );
      setScriptDirty(true);
      scriptDirtyRef.current = true;
      setStatusMessage(mode === "closer_to_ocr" ? "Panel line regenerated with a fact-preserving rewrite." : "Panel line rewritten.");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unable to rewrite this panel line.");
    } finally {
      setRewritingPanels((current) => ({ ...current, [item.panelId]: null }));
    }
  }

  return (
    <AppShell
      title={project?.name || "Narration"}
      description={project?.chapter_metadata?.manga_title ? `${project.chapter_metadata.manga_title} · narration` : "Generate a recap script, tune the narrator and soundtrack with live previews."}
      projectId={projectId}
      breadcrumb={{ href: project ? `/projects/${project.id}` : "/", label: project ? "Overview" : "All projects" }}
      views={buildProjectViews(projectId, "/narration")}
    >
      {error ? (
        <Card className="mb-6 p-edge-fail">
          <CardDescription className="text-fail">{error}</CardDescription>
        </Card>
      ) : null}
      {project?.stage_states.script_generation.message?.toLowerCase().includes("rate-limited") ? (
        <Card className="mb-6 p-edge-warn">
          <CardDescription className="text-warn">{project.stage_states.script_generation.message}</CardDescription>
        </Card>
      ) : null}
      {scriptDisplayMetadata?.is_displaying_stale_script ? (
        <Card className="mb-6 p-edge-warn">
          <CardTitle className="text-sm text-warn">Previous script shown while the new run is in progress</CardTitle>
          <CardDescription className="mt-1 text-warn/80">
            Latest job: {scriptDisplayMetadata.latest_job_id ?? "unknown"} ({scriptDisplayMetadata.latest_job_status ?? "unknown"}). Displayed script: {scriptDisplayMetadata.displayed_script_created_at ?? "unknown time"}.
          </CardDescription>
        </Card>
      ) : null}
      {audioStageStatus === "failed" && project?.stage_states.narration_generation.message ? (
        <Card className="mb-6 p-edge-fail">
          <CardDescription className="text-fail">{project.stage_states.narration_generation.message}</CardDescription>
        </Card>
      ) : null}
      {videoStageStatus === "failed" && project?.stage_states.video_rendering.message ? (
        <Card className="mb-6 p-edge-fail">
          <CardDescription className="text-fail">{project.stage_states.video_rendering.message}</CardDescription>
        </Card>
      ) : null}

      <div className="max-w-full space-y-6 overflow-x-hidden">
        {/* Action bar */}
        <div className="flex flex-wrap items-center gap-2">
          <Button
            size="sm"
            onClick={() => toggleStage("script_generation")}
            disabled={!project || (queueingStage !== null && queueingStage !== "script_generation") || (cancellingStage !== null && cancellingStage !== "script_generation")}
          >
            {scriptBusy || cancellingStage === "script_generation" ? <LoaderCircle className="h-3.5 w-3.5 animate-spin" /> : null}
            {cancellingStage === "script_generation"
              ? "Cancelling..."
              : scriptBusy
                ? "Cancel script generation"
                : hasGeneratedScript
                  ? "Regenerate script"
                  : "Generate script"}
          </Button>
          {usingStorySegments ? (
            <Button
              size="sm"
              variant="secondary"
              onClick={repairWeakStorySegments}
              disabled={
                !project ||
                scriptBusy ||
                blankKeptLineCount === 0 ||
                queueingStage !== null ||
                cancellingStage !== null
              }
            >
              {queueingStage === "script_generation" || scriptBusy ? <LoaderCircle className="h-3.5 w-3.5 animate-spin" /> : null}
              Repair weak segments
            </Button>
          ) : null}
          <Button size="sm" variant="secondary" onClick={saveScript} disabled={savingScript}>
            {savingScript ? <LoaderCircle className="h-3.5 w-3.5 animate-spin" /> : null}
            {savingScript ? "Saving..." : "Save script"}
          </Button>
          <Button
            size="sm"
            variant="secondary"
            onClick={() => toggleStage("narration_generation")}
            disabled={!project || (queueingStage !== null && queueingStage !== "narration_generation") || (cancellingStage !== null && cancellingStage !== "narration_generation")}
          >
            {audioBusy || cancellingStage === "narration_generation" ? <LoaderCircle className="h-3.5 w-3.5 animate-spin" /> : null}
            {cancellingStage === "narration_generation" ? "Cancelling..." : audioBusy ? "Cancel audio generation" : "Generate audio"}
          </Button>
          <Button
            size="sm"
            variant="secondary"
            onClick={() => toggleStage("video_rendering")}
            disabled={!project || (queueingStage !== null && queueingStage !== "video_rendering") || (cancellingStage !== null && cancellingStage !== "video_rendering")}
          >
            {videoBusy || cancellingStage === "video_rendering" ? <LoaderCircle className="h-3.5 w-3.5 animate-spin" /> : null}
            {cancellingStage === "video_rendering" ? "Cancelling..." : videoBusy ? "Cancel video render" : "Render video"}
          </Button>
          <Button size="sm" variant="outline" onClick={exportScript}>
            <Download className="h-3.5 w-3.5" />
            Export
          </Button>
          <div className="flex-1" />
          {activeProgressMessage ? <span className="text-xs text-mutedForeground">{activeProgressMessage}</span> : null}
          {statusMessage ? <span className="text-xs text-accent">{statusMessage}</span> : null}
        </div>

        <div className="grid min-w-0 gap-6 xl:grid-cols-[1fr_0.85fr]">
          <Card className="min-w-0">
            <div className="flex items-start justify-between gap-3">
              <div>
                <CardTitle className="text-base">Chapter recap</CardTitle>
                <CardDescription className="mt-1">
                  {storyEditMode
                    ? "Edit narration directly in the textarea, then save."
                    : "Live preview - stays in sync with panel lines below."}
                </CardDescription>
              </div>
              {!storyEditMode ? (
                <Button size="sm" variant="outline" onClick={enterStoryEditMode} className="shrink-0">
                  <Pencil className="h-3.5 w-3.5" />
                  Edit
                </Button>
              ) : (
                <div className="flex shrink-0 gap-2">
                  <Button size="sm" variant="outline" onClick={() => setStoryEditMode(false)}>
                    <X className="h-3.5 w-3.5" />
                    Cancel
                  </Button>
                  <Button size="sm" onClick={saveStoryEdit} disabled={savingScript}>
                    {savingScript ? <LoaderCircle className="h-3.5 w-3.5 animate-spin" /> : null}
                    Save
                  </Button>
                </div>
              )}
            </div>
            <Textarea
              className="mt-4 h-[440px] overflow-y-auto resize-none font-mono text-sm leading-relaxed"
              value={storyEditMode ? editableStory : storyPreview}
              readOnly={!storyEditMode}
              onChange={storyEditMode ? (e) => setEditableStory(e.target.value) : undefined}
            />
          </Card>

          <Card className="min-w-0">
            <CardTitle className="text-base">Narrator and soundtrack</CardTitle>
            <CardDescription className="mt-1">Saved settings for all future audio and video regeneration.</CardDescription>
            <div className="mt-6 grid gap-4 md:grid-cols-2">
              <label className="space-y-2">
                <span className="text-sm text-mutedForeground">Language</span>
                <select className="h-11 w-full rounded-2xl border border-white/[0.08] bg-white/[0.04] px-4 text-sm text-foreground focus:outline-none focus:border-accent/40 focus:ring-2 focus:ring-accent/30 transition-colors duration-fast" value={langCode} onChange={(event) => setLangCode(event.target.value)}>
                  {(catalog?.languages ?? []).map((option) => (
                    <option key={option.code} value={option.code}>
                      {option.label}
                    </option>
                  ))}
                </select>
              </label>
              <label className="space-y-2">
                <span className="text-sm text-mutedForeground">Speech speed</span>
                <Input type="number" min="0.7" max="1.3" step="0.05" value={speed} onChange={(event) => setSpeed(Number(event.target.value))} />
              </label>
              <label className="space-y-2">
                <span className="text-sm text-mutedForeground">Music volume</span>
                <Input type="number" min="0" max="1" step="0.05" value={musicVolume} onChange={(event) => setMusicVolume(Number(event.target.value))} />
              </label>
            </div>
            <div className="mt-4 rounded-2xl border border-white/[0.08] bg-white/[0.04] p-4">
              <div className="space-y-3">
                <div className="flex flex-wrap items-start justify-between gap-3">
                  <div>
                    <p className="text-sm font-medium text-white">{selectedVoice?.label ?? "Selected voice"}</p>
                    <p className="mt-1 text-sm text-mutedForeground">{selectedVoice?.description ?? "Choose a voice above."}</p>
                  </div>
                  {selectedVoice?.quality_note ? <Badge>{selectedVoice.quality_note}</Badge> : null}
                </div>
                <label className="block space-y-2">
                  <span className="text-sm text-mutedForeground">Voice</span>
                  <select className="h-11 w-full rounded-2xl border border-white/[0.08] bg-white/[0.04] px-4 text-sm text-foreground focus:outline-none focus:border-accent/40 focus:ring-2 focus:ring-accent/30 transition-colors duration-fast" value={voice} onChange={(event) => setVoice(event.target.value)}>
                    {voiceOptions.map((option) => (
                      <option key={option.id} value={option.id}>
                        {option.label}
                      </option>
                    ))}
                  </select>
                </label>
              </div>
              {selectedLanguage ? <p className="mt-3 text-xs text-mutedForeground">{selectedLanguage.description}</p> : null}
            </div>
            <div className="mt-4 space-y-4">
              <label className="flex items-center justify-between rounded-2xl border border-white/[0.08] bg-white/[0.04] px-4 py-3">
                <span className="text-sm text-white">Enable music in final video</span>
                <input type="checkbox" checked={musicEnabled} onChange={(event) => setMusicEnabled(event.target.checked)} className="h-4 w-4 accent-cyan-400" />
              </label>
              <label className="space-y-2">
                <span className="text-sm text-mutedForeground">Track</span>
                <select className="h-11 w-full rounded-2xl border border-white/[0.08] bg-white/[0.04] px-4 text-sm text-foreground focus:outline-none focus:border-accent/40 focus:ring-2 focus:ring-accent/30 transition-colors duration-fast" value={musicTrack} onChange={(event) => setMusicTrack(event.target.value)}>
                  <option value="">No track</option>
                  {(catalog?.music_tracks ?? []).map((track) => (
                    <option key={`${track.source ?? "builtin"}-${track.file}`} value={track.name}>
                      {track.name} {track.source === "uploaded" ? "(uploaded)" : track.available ? "" : "(missing file)"}
                    </option>
                  ))}
                </select>
              </label>
              <div className="rounded-2xl border border-dashed border-white/[0.14] bg-white/4 p-4">
                <p className="text-sm font-medium text-white">Import MP3 soundtrack</p>
                <p className="mt-1 text-sm text-mutedForeground">Upload a custom music bed and use it immediately on this project.</p>
                <div className="mt-4 flex flex-col gap-3 sm:flex-row sm:items-center">
                  <label className="inline-flex cursor-pointer items-center rounded-full bg-white/8 px-4 py-2 text-sm text-white transition hover:bg-white/[0.10]">
                    Choose MP3
                    <input type="file" accept=".mp3,audio/mpeg" className="hidden" onChange={(event) => setMusicUploadFile(event.target.files?.[0] ?? null)} />
                  </label>
                  <p className="text-sm text-mutedForeground">{musicUploadFile?.name ?? "No MP3 selected yet."}</p>
                </div>
                <Button className="mt-4" variant="secondary" onClick={handleMusicUpload} disabled={!musicUploadFile || uploadingMusic}>
                  {uploadingMusic ? <LoaderCircle className="h-4 w-4 animate-spin" /> : <UploadCloud className="h-4 w-4" />}
                  Import MP3
                </Button>
              </div>
              <div className="rounded-2xl border border-white/[0.08] bg-white/[0.04] p-4">
                <p className="text-sm font-medium text-white">{selectedTrack?.name ?? "Music preview"}</p>
                <p className="mt-1 text-sm text-mutedForeground">
                  {selectedTrack?.available
                    ? `${selectedTrack.source === "uploaded" ? "Uploaded track" : "Built-in preset"}${selectedTrack.mood ? ` • Mood: ${selectedTrack.mood}` : ""}`
                    : "Add the matching music file or upload your own MP3 to enable this preview."}
                </p>
                {selectedTrack?.available && selectedTrack.url ? (
                  <audio key={selectedTrack.file} controls preload="none" className="mt-4 w-full" src={buildMediaUrl(selectedTrack.url)} />
                ) : null}
              </div>
            </div>
            <Button className="mt-6 w-full" onClick={saveSettings} disabled={savingSettings}>
              {savingSettings ? <LoaderCircle className="h-4 w-4 animate-spin" /> : null}
              Save narration settings
            </Button>
            <Button className="mt-3 w-full" variant="outline" onClick={() => rewindTo("narration_generation")} disabled={!project || rewindingStage}>
              {rewindingStage ? <LoaderCircle className="h-4 w-4 animate-spin" /> : null}
              {rewindingStage ? "Reopening..." : "Reopen audio step"}
            </Button>
          </Card>
        </div>

        <Card className="min-w-0 overflow-hidden">
          <div className="flex flex-wrap items-center justify-between gap-3">
              <div>
                <CardTitle className="text-base">{usingStorySegments ? "Story segments" : "Panel narration lines"}</CardTitle>
                <CardDescription className="mt-1">
                  {usingStorySegments
                    ? "Review and edit grouped recap beats. Excluded segments stay visible so you can restore them."
                    : "Scroll through panels to review and edit narration. Excluded panels stay visible so you can restore them."}
                </CardDescription>
              </div>
            <div className="flex flex-wrap items-center justify-end gap-2">
              <Button
                size="sm"
                variant="outline"
                onClick={usingStorySegments ? excludeSegmentsWithoutScript : excludePanelsWithoutScript}
                disabled={savingScript || blankKeptLineCount === 0}
              >
                {usingStorySegments ? "Exclude blank segments" : "Exclude blank panels"}
              </Button>
              <span className="rounded-full bg-white/8 px-2.5 py-1 text-[11px] font-medium text-mutedForeground">
                {`${keptPanelCount}/${usingStorySegments ? segmentItems.length : lineItems.length} ${usingStorySegments ? "segments" : "panels"} kept`}
              </span>
              {blankKeptLineCount > 0 ? (
                <span className="rounded-full bg-warn/[0.10] px-2.5 py-1 text-[11px] font-medium text-warn">
                  {blankKeptLineCount} without script
                </span>
              ) : null}
            </div>
          </div>
          <div className="mt-6 max-w-full overflow-x-auto overscroll-x-contain pb-4">
            {usingStorySegments ? (
              <div className="grid gap-4 pr-2">
                {segmentItems.map((item, index) => (
                  <div
                    key={item.id}
                    className={`overflow-hidden rounded-2xl border p-5 shadow-[inset_0_1px_0_rgba(255,255,255,0.03)] transition ${
                      item.keep ? "border-white/[0.08] bg-white/[0.06]" : "border-white/[0.06] bg-white/[0.03] opacity-70"
                    }`}
                  >
                    <div className="flex flex-wrap items-start justify-between gap-3">
                      <div>
                        <p className="text-xs uppercase tracking-[0.22em] text-accent">Segment {item.order}</p>
                        <p className="mt-1 text-sm text-white">{item.title}</p>
                      </div>
                      <div className="flex flex-wrap gap-2">
                        {!item.keep ? <Badge className="bg-white/10 text-white">Excluded</Badge> : null}
                        {item.visualOnly ? <Badge className="bg-white/10 text-white">Visual only</Badge> : null}
                        {item.panelStart && item.panelEnd ? (
                          <Badge>
                            Panels {item.panelStart}-{item.panelEnd}
                          </Badge>
                        ) : null}
                        <Badge className="bg-white/10 text-white">
                          {item.panelIds.length} panel{item.panelIds.length === 1 ? "" : "s"}
                        </Badge>
                      </div>
                    </div>
                    <div className="mt-4 flex flex-wrap gap-2">
                      <Button variant={item.keep ? "outline" : "secondary"} size="sm" onClick={() => toggleSegmentKeep(index)}>
                        {item.keep ? "Exclude" : "Include"}
                      </Button>
                    </div>
                    {item.previewPanels.length ? (
                      <div className="mt-4">
                        <p className="mb-3 text-[11px] uppercase tracking-[0.18em] text-mutedForeground">Panel preview</p>
                        <div className="grid gap-3 sm:grid-cols-3">
                        {item.previewPanels.map((preview, previewIndex) => (
                          <div key={`${item.id}-${preview.panelId}-${previewIndex}`} className="overflow-hidden rounded-2xl border border-white/[0.08] bg-black/25">
                            {/* eslint-disable-next-line @next/next/no-img-element */}
                            <img
                              src={buildMediaUrl(preview.previewUrl)}
                              alt={`Segment ${item.order} preview ${previewIndex + 1}`}
                              loading="lazy"
                              decoding="async"
                              className="h-40 w-full bg-black/30 object-contain"
                              onError={(event) => {
                                if (!project || !preview.page) return;
                                event.currentTarget.onerror = null;
                                event.currentTarget.src = buildMediaUrl(
                                  `/media/projects/${project.id}/pages/${preview.page.toString().padStart(4, "0")}.png?v=${encodeURIComponent(project.updated_at)}`
                                );
                              }}
                            />
                            <p className="border-t border-white/[0.08] px-3 py-2 text-[11px] text-mutedForeground">
                              Page {preview.page}{preview.panelNumber ? ` • Panel ${preview.panelNumber}` : ""}
                            </p>
                          </div>
                        ))}
                        </div>
                      </div>
                    ) : null}
                    <Textarea
                      className="mt-4 h-[190px] overflow-y-auto resize-none rounded-2xl bg-white/[0.04]"
                      value={item.value}
                      placeholder={item.visualOnly ? "This segment is currently visual-only. Type here to add spoken narration." : "Write the narration for this story segment."}
                      onChange={(event) => updateSegmentValue(index, event.target.value)}
                    />
                    <p className="mt-3 text-xs text-mutedForeground">
                      {!item.keep
                        ? "This segment will be excluded from audio and video after you save."
                        : item.visualOnly
                        ? "Panelia will keep this beat visual-only unless you add narration here."
                        : "This segment's narration will be spoken across its assigned panels."}
                    </p>
                    {item.suppressionReason ? (
                      <p className="mt-2 text-xs text-mutedForeground">
                        Suppression reason: {item.suppressionReason.replace(/_/g, " ")}.
                      </p>
                    ) : null}
                  </div>
                ))}
              </div>
            ) : (
            <div className="inline-grid grid-flow-col auto-cols-[minmax(20rem,24rem)] gap-4 pr-2">
              {lineItems.map((item, index) => (
                (() => {
                  const waitingForDialogue =
                    !item.hasExtractedText &&
                    !item.value.trim() &&
                    scriptStageStatus !== "completed" &&
                    scriptStageStatus !== "failed" &&
                    scriptStageStatus !== "cancelled";
                  const factAnchors = extractFactAnchors(item.extractedText);
                  const factualPanel = isFactualPanel(item.extractedText);
                  const possibleMismatch = item.keep && hasNarrationMismatch(item.extractedText, item.value);
                  const rewriteBusy = Boolean(rewritingPanels[item.panelId]);

                  return (
                    <div
                      key={item.panelId}
                      className={`w-[min(24rem,calc(100vw-6rem))] max-w-full overflow-hidden rounded-2xl border p-5 shadow-[inset_0_1px_0_rgba(255,255,255,0.03)] transition ${
                        item.keep ? "border-white/[0.08] bg-white/[0.06]" : "border-white/[0.06] bg-white/[0.03] opacity-70"
                      }`}
                    >
                      <div className="overflow-hidden rounded-2xl border border-white/[0.08] bg-black/25">
                        {/* eslint-disable-next-line @next/next/no-img-element */}
                        <img
                          src={buildMediaUrl(item.previewUrl)}
                          alt={`Panel ${item.order}`}
                          loading="lazy"
                          decoding="async"
                          className="h-44 w-full bg-black/30 object-contain"
                          onError={(event) => {
                            if (!project) return;
                            event.currentTarget.onerror = null;
                            event.currentTarget.src = buildMediaUrl(
                              `/media/projects/${project.id}/pages/${item.page.toString().padStart(4, "0")}.png?v=${encodeURIComponent(project.updated_at)}`
                            );
                          }}
                        />
                      </div>
                      <div className="mt-4 flex items-start justify-between gap-3">
                        <div>
                          <p className="text-xs uppercase tracking-[0.22em] text-accent">Panel {item.order}</p>
                          <p className="mt-1 text-sm text-white">
                            Page {item.page} • Panel {item.panelNumber}
                          </p>
                        </div>
                        <div className="flex flex-wrap justify-end gap-2">
                          <Badge>{waitingForDialogue ? "Waiting for OCR" : item.hasExtractedText ? "Narrated" : "Empty by default"}</Badge>
                          {item.lineSource ? <Badge className="bg-white/10 text-white">{sourceLabel(item.lineSource)}</Badge> : null}
                          {item.locked ? <Badge className="bg-white/10 text-white">Locked</Badge> : null}
                          {factualPanel ? <Badge className="bg-white/10 text-white">Fact-heavy</Badge> : null}
                          {possibleMismatch ? <Badge className="bg-warn/[0.10] text-warn">Possible mismatch</Badge> : null}
                        </div>
                      </div>
                      <div className="mt-4 flex flex-wrap gap-2">
                        <Button variant="outline" size="sm" onClick={() => rewriteLine(index, "balanced")} disabled={rewriteBusy}>
                          {rewritingPanels[item.panelId] === "balanced" ? <LoaderCircle className="h-4 w-4 animate-spin" /> : <RefreshCcw className="h-4 w-4" />}
                          Rewrite
                        </Button>
                        <Button variant="outline" size="sm" onClick={() => rewriteLine(index, "closer_to_ocr")} disabled={rewriteBusy || !item.extractedText.trim()}>
                          {rewritingPanels[item.panelId] === "closer_to_ocr" ? <LoaderCircle className="h-4 w-4 animate-spin" /> : <ScanSearch className="h-4 w-4" />}
                          Keep facts
                        </Button>
                        <Button variant="outline" size="sm" onClick={() => rewriteLine(index, "shorten")} disabled={rewriteBusy || !item.value.trim()}>
                          {rewritingPanels[item.panelId] === "shorten" ? <LoaderCircle className="h-4 w-4 animate-spin" /> : <Scissors className="h-4 w-4" />}
                          Shorten
                        </Button>
                        <Button variant="outline" size="sm" onClick={() => toggleLineLock(index)}>
                          {item.locked ? <Unlock className="h-4 w-4" /> : <Lock className="h-4 w-4" />}
                          {item.locked ? "Unlock" : "Lock"}
                        </Button>
                        <Button variant={item.keep ? "outline" : "secondary"} size="sm" onClick={() => toggleLineKeep(index)}>
                          {item.keep ? "Exclude panel" : "Include panel"}
                        </Button>
                      </div>
                      <Textarea
                        className="mt-4 h-[190px] overflow-y-auto resize-none rounded-2xl bg-white/[0.04]"
                        value={item.value}
                        placeholder={
                          waitingForDialogue
                            ? "Dialogue extraction is still running for this panel."
                            : item.hasExtractedText
                              ? "Narration for this panel"
                              : "No usable extracted text on this panel. Leave blank to skip it, or type your own narration."
                        }
                        onChange={(event) => updateLineValue(index, event.target.value)}
                      />
                      {possibleMismatch ? (
                        <p className="mt-3 text-xs text-warn">
                          This line may be dropping important factual details from the extracted text. Try <span className="font-semibold">Keep facts</span>.
                        </p>
                      ) : null}
                      {item.visualCaption ? (
                        <div className="mt-4 max-h-[9rem] overflow-y-auto rounded-2xl border border-white/[0.08] bg-black/20 p-3">
                          <p className="text-[11px] uppercase tracking-[0.18em] text-mutedForeground">Visual caption</p>
                          <p className="mt-2 text-sm text-mutedForeground">{item.visualCaption}</p>
                        </div>
                      ) : null}
                      {item.extractedText ? (
                        <div className="mt-4 max-h-[9rem] overflow-y-auto rounded-2xl border border-white/[0.08] bg-black/20 p-3">
                          <p className="text-[11px] uppercase tracking-[0.18em] text-mutedForeground">Extracted text</p>
                          <p className="mt-2 text-sm text-mutedForeground">{item.extractedText}</p>
                        </div>
                      ) : null}
                      {factAnchors.length ? (
                        <div className="mt-3 flex flex-wrap gap-2">
                          {factAnchors.slice(0, 6).map((anchor) => (
                            <Badge key={`${item.panelId}-${anchor}`} className="bg-white/10 text-white">
                              {anchor}
                            </Badge>
                          ))}
                        </div>
                      ) : null}
                      {waitingForDialogue ? <p className="mt-3 text-xs text-mutedForeground">Panelia is still extracting dialogue for this panel. The line will update automatically when that finishes.</p> : null}
                      {!waitingForDialogue && !item.keep ? <p className="mt-3 text-xs text-mutedForeground">This panel will be excluded from audio and video after you save.</p> : null}
                      {item.locked ? <p className="mt-3 text-xs text-mutedForeground">This line is locked and will stay in place during future script regeneration until you unlock it.</p> : null}
                    </div>
                  );
                })()
              ))}
            </div>
            )}
          </div>
        </Card>

        <div className="flex justify-end">
          <Link href={project ? `/projects/${project.id}/preview` : "#"}>
            <Button size="lg">
              Open preview
              <ArrowRight className="h-4 w-4" />
            </Button>
          </Link>
        </div>
      </div>
    </AppShell>
  );
}
