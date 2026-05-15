"use client";

import { useParams, useRouter } from "next/navigation";
import { useEffect, useMemo, useRef, useState } from "react";
import { ArrowDown, ArrowUp, Download, LoaderCircle, Plus, X } from "lucide-react";

import { AppShell } from "@/components/project/app-shell";
import { Button } from "@/components/ui/button";
import { Card, CardDescription, CardTitle } from "@/components/ui/card";
import { api } from "@/lib/api";
import { formatProgressPercent, getStageProgressMeta } from "@/lib/progress";
import { PanelLayout, ProjectDetail } from "@/lib/types";
import { useAdaptivePolling } from "@/lib/use-adaptive-polling";
import { buildMediaUrl, formatRelativeDate } from "@/lib/utils";

function safeDimension(value: number | string | null | undefined, fallback: number) {
  const numeric = Number(value);
  return Number.isFinite(numeric) && numeric > 0 ? Math.round(numeric) : fallback;
}

function baseResolution(width: number, height: number) {
  const safeWidth = safeDimension(width, 1920);
  const safeHeight = safeDimension(height, 1080);
  return `${Math.max(safeWidth, safeHeight)}x${Math.min(safeWidth, safeHeight)}`;
}

function resolvedDimensions(resolution: string, orientation: "landscape" | "vertical") {
  const [rawWidth, rawHeight] = resolution.split("x").map((value) => safeDimension(value, 0));
  const width = rawWidth > 0 ? rawWidth : 1920;
  const height = rawHeight > 0 ? rawHeight : 1080;
  if (orientation === "vertical") {
    return { width: Math.min(width, height), height: Math.max(width, height) };
  }
  return { width: Math.max(width, height), height: Math.min(width, height) };
}

function formatCanvas(width: number | string | null | undefined, height: number | string | null | undefined) {
  return `${safeDimension(width, 1920)}×${safeDimension(height, 1080)}`;
}

function mediaAspect(width: number | string | null | undefined, height: number | string | null | undefined) {
  const safeWidth = safeDimension(width, 1920);
  const safeHeight = safeDimension(height, 1080);
  return `${safeWidth} / ${safeHeight}`;
}

export default function PreviewPage() {
  const params = useParams<{ projectId: string }>();
  const router = useRouter();
  const projectId = Array.isArray(params.projectId) ? params.projectId[0] : params.projectId;
  const [project, setProject] = useState<ProjectDetail | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [outputName, setOutputName] = useState("merged-cut");
  const [selectedPaths, setSelectedPaths] = useState<string[]>([]);
  const [resolution, setResolution] = useState("1920x1080");
  const [orientation, setOrientation] = useState<"landscape" | "vertical">("vertical");
  const [panelLayout, setPanelLayout] = useState<PanelLayout>("card");
  const [introThumbnailEnabled, setIntroThumbnailEnabled] = useState(false);
  const [introThumbnailSeconds, setIntroThumbnailSeconds] = useState(1.5);
  const [outputFormat, setOutputFormat] = useState<"mp4" | "mov">("mp4");
  const [savingVideoSettings, setSavingVideoSettings] = useState(false);
  const [uploadingThumbnail, setUploadingThumbnail] = useState(false);
  const [duplicatingVideoName, setDuplicatingVideoName] = useState<string | null>(null);
  const [renderActionBusy, setRenderActionBusy] = useState(false);
  const [audioActionBusy, setAudioActionBusy] = useState(false);
  const [rewindingToNarration, setRewindingToNarration] = useState(false);
  const [deletingVideoName, setDeletingVideoName] = useState<string | null>(null);
  const [mergingVideos, setMergingVideos] = useState(false);
  const initializedSettings = useRef(false);
  const thumbnailInputRef = useRef<HTMLInputElement | null>(null);

  useEffect(() => {
    initializedSettings.current = false;
  }, [projectId]);

  async function load() {
    if (!projectId) return;
    try {
      const nextProject = await api.getProject(projectId);
      setProject(nextProject);
      setError(null);
      if (!initializedSettings.current) {
        setResolution(baseResolution(nextProject.video_config.width, nextProject.video_config.height));
        setOrientation(nextProject.video_config.orientation);
        setPanelLayout((nextProject.video_config.panel_layout ?? "card") as PanelLayout);
        setIntroThumbnailEnabled(Boolean(nextProject.video_config.intro_thumbnail_enabled));
        setIntroThumbnailSeconds(Number(nextProject.video_config.intro_thumbnail_seconds ?? 1.5));
        setOutputFormat(nextProject.video_config.output_format);
        initializedSettings.current = true;
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unable to load the preview workspace.");
    }
  }

  useEffect(() => {
    if (!projectId) return;
    void load();
  }, [projectId]);

  const latestVideo = project?.latest_video ?? project?.videos.at(-1) ?? null;
  const latestVideoSrc = latestVideo
    ? buildMediaUrl(
        latestVideo.url,
        latestVideo.created_at ?? latestVideo.duration_seconds ?? `${latestVideo.name}-${project?.updated_at ?? ""}`
      )
    : "";
  const latestVideoDownloadUrl = project ? api.downloadLatestVideoUrl(project.id) : "";
  const videoThumbnailSrc = project?.video_thumbnail_url
    ? buildMediaUrl(project.video_thumbnail_url, `${project.updated_at}-video-thumb`)
    : "";
  const latestVideoAspect = mediaAspect(latestVideo?.width, latestVideo?.height);
  const latestVideoPortrait = safeDimension(latestVideo?.height, 1080) > safeDimension(latestVideo?.width, 1920);
  const videoStage = project?.stage_states.video_rendering;
  const audioStage = project?.stage_states.narration_generation;
  const activeJobs = project?.active_jobs ?? [];
  const activeStageJobs = (stage: "narration_generation" | "video_rendering") =>
    activeJobs.filter((job) => job.stage === stage && (job.status === "queued" || job.status === "running"));
  const videoProgressMeta = project ? getStageProgressMeta(project, "video_rendering") : null;
  const renderInFlight = videoStage?.status === "running" || activeStageJobs("video_rendering").length > 0;
  const audioInFlight = audioStage?.status === "running" || activeStageJobs("narration_generation").length > 0;
  const renderProgress = videoProgressMeta?.progress ?? Math.ceil(Math.max(0, Math.min(100, Number(videoStage?.progress ?? 0))));
  const targetDimensions = resolvedDimensions(resolution, orientation);
  const videoSettingsDirty = Boolean(
    project &&
      (
        safeDimension(project.video_config.width, 0) !== targetDimensions.width ||
        safeDimension(project.video_config.height, 0) !== targetDimensions.height ||
        project.video_config.orientation !== orientation ||
        (project.video_config.panel_layout ?? "card") !== panelLayout ||
        Boolean(project.video_config.intro_thumbnail_enabled) !== introThumbnailEnabled ||
        Math.abs(Number(project.video_config.intro_thumbnail_seconds ?? 1.5) - introThumbnailSeconds) > 0.01 ||
        project.video_config.output_format !== outputFormat
      )
  );
  const queuedVideos = useMemo(
    () => selectedPaths.map((path) => project?.videos.find((video) => video.path === path)).filter((video): video is NonNullable<typeof video> => Boolean(video)),
    [project, selectedPaths]
  );

  useAdaptivePolling(load, {
    enabled: Boolean(projectId),
    active: renderInFlight || Boolean(project?.active_jobs.length),
    activeMs: 7000,
    idleMs: 30000,
    hiddenMs: 120000,
    deps: [projectId]
  });

  async function persistVideoSettings() {
    if (!project) return null;
    const { width, height } = resolvedDimensions(resolution, orientation);
    setSavingVideoSettings(true);
    try {
      const updated = await api.updateProjectSettings(project.id, {
        video_config: {
          ...project.video_config,
          width,
          height,
          orientation,
          panel_layout: panelLayout,
          intro_thumbnail_enabled: introThumbnailEnabled,
          intro_thumbnail_seconds: introThumbnailSeconds,
          output_format: outputFormat
        }
      });
      setProject(updated);
      initializedSettings.current = true;
      return updated;
    } finally {
      setSavingVideoSettings(false);
    }
  }

  async function ensureCurrentVideoSettings() {
    if (!project) return null;
    if (!videoSettingsDirty) return project;
    return persistVideoSettings();
  }

  async function mergeSelectedVideos() {
    if (!project || selectedPaths.length < 2) return;
    setMergingVideos(true);
    try {
      const activeProject = (await ensureCurrentVideoSettings()) ?? project;
      await api.mergeVideos(activeProject.id, {
        video_paths: selectedPaths,
        output_name: outputName,
        video_config: activeProject.video_config
      });
      setProject(await api.getProject(activeProject.id));
      setSelectedPaths([]);
    } finally {
      setMergingVideos(false);
    }
  }

  function toggleVideoInQueue(path: string) {
    setSelectedPaths((current) => (current.includes(path) ? current.filter((item) => item !== path) : [...current, path]));
  }

  function moveQueuedVideo(path: string, direction: "up" | "down") {
    setSelectedPaths((current) => {
      const index = current.indexOf(path);
      if (index < 0) return current;
      const targetIndex = direction === "up" ? index - 1 : index + 1;
      if (targetIndex < 0 || targetIndex >= current.length) return current;
      const next = [...current];
      [next[index], next[targetIndex]] = [next[targetIndex], next[index]];
      return next;
    });
  }

  async function saveVideoSettings() {
    if (!project) return;
    await persistVideoSettings();
  }

  async function handleThumbnailUpload(file: File | null) {
    if (!project || !file) return;
    setUploadingThumbnail(true);
    try {
      const updated = await api.uploadVideoThumbnail(project.id, file);
      setProject(updated);
      initializedSettings.current = true;
    } finally {
      setUploadingThumbnail(false);
      if (thumbnailInputRef.current) {
        thumbnailInputRef.current.value = "";
      }
    }
  }

  async function removeVideoThumbnail() {
    if (!project || !project.video_thumbnail_url) return;
    setUploadingThumbnail(true);
    try {
      const updated = await api.deleteVideoThumbnail(project.id);
      setProject(updated);
      initializedSettings.current = true;
      setIntroThumbnailEnabled(false);
    } finally {
      setUploadingThumbnail(false);
    }
  }

  async function queueVideoRender() {
    if (!project) return;
    const activeProject = (await ensureCurrentVideoSettings()) ?? project;
    await api.queueStage(activeProject.id, "video_rendering");
    setProject(await api.getProject(activeProject.id));
  }

  async function cancelStage(stage: "narration_generation" | "video_rendering") {
    if (!project) return;
    const jobs = activeStageJobs(stage);
    if (!jobs.length) return;
    setError(null);
    if (stage === "video_rendering") {
      setRenderActionBusy(true);
    } else {
      setAudioActionBusy(true);
    }
    try {
      await Promise.all(jobs.map((job) => api.cancelJob(project.id, job.id)));
      setProject(await api.getProject(project.id));
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unable to cancel the active job.");
    } finally {
      if (stage === "video_rendering") {
        setRenderActionBusy(false);
      } else {
        setAudioActionBusy(false);
      }
    }
  }

  async function toggleVideoRender() {
    if (renderInFlight) {
      await cancelStage("video_rendering");
      return;
    }
    setRenderActionBusy(true);
    try {
      await queueVideoRender();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unable to queue video rendering.");
    } finally {
      setRenderActionBusy(false);
    }
  }

  async function toggleAudioGeneration() {
    if (!project) return;
    if (audioInFlight) {
      await cancelStage("narration_generation");
      return;
    }
    setAudioActionBusy(true);
    try {
      await api.queueStage(project.id, "narration_generation", { force_quality_bypass: true });
      setProject(await api.getProject(project.id));
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unable to queue audio generation.");
    } finally {
      setAudioActionBusy(false);
    }
  }

  async function rewindToNarration() {
    if (!project) return;
    setRewindingToNarration(true);
    try {
      await api.rewindProject(project.id, "script_generation");
      router.push(`/projects/${project.id}/narration`);
    } finally {
      setRewindingToNarration(false);
    }
  }

  async function duplicateForReedit(videoName?: string) {
    if (!project) return;
    setDuplicatingVideoName(videoName ?? "__latest__");
    try {
      const duplicated = await api.duplicateProject(project.id, {
        name: `${project.name} Alternate Cut`,
        video_name: videoName,
        copy_all_videos: !videoName
      });
      router.push(`/projects/${duplicated.id}/preview`);
    } finally {
      setDuplicatingVideoName(null);
    }
  }

  if (!project) {
    return (
      <AppShell title="Loading preview" description="Fetching generated exports and audio assets." projectId={projectId}>
        <div className="flex items-center gap-3 rounded-2xl border border-white/[0.08] bg-white/[0.04] p-6 text-sm text-mutedForeground">
          <LoaderCircle className="h-4 w-4 animate-spin text-accent" />
          Loading preview workspace...
        </div>
      </AppShell>
    );
  }

  return (
    <AppShell
      title="Preview and exports"
      description="Review the latest rendered video, re-render after panel or narration edits, and merge completed cuts into a longer recap."
      projectId={projectId}
    >
      {error ? (
        <Card className="mb-6 border-fail/[0.25] bg-fail/[0.08]">
          <CardDescription className="text-fail">{error}</CardDescription>
        </Card>
      ) : null}
      {videoStage?.status === "failed" ? (
        <Card className="mb-6 border-fail/[0.25] bg-fail/[0.08]">
          <CardDescription className="text-fail">
            The latest render failed, so the preview below is still showing the last completed export.
            {videoStage.message ? ` ${videoStage.message}` : ""}
          </CardDescription>
        </Card>
      ) : null}
      <div className="grid gap-6 xl:grid-cols-[1.15fr_0.85fr]">
        <Card>
          <CardTitle className="text-base">Latest export</CardTitle>
          <CardDescription className="mt-1">Your newest render appears when the video stage finishes.</CardDescription>
          <div className="mt-6 overflow-hidden rounded-2xl border border-white/[0.08] bg-black/30">
            {renderInFlight ? (
              <div
                className="flex w-full flex-col items-center justify-center gap-5 px-6 text-sm text-mutedForeground"
                style={{ aspectRatio: latestVideoAspect }}
              >
                <LoaderCircle className="h-6 w-6 animate-spin text-accent" />
                <div className="space-y-2 text-center">
                  <p className="text-base font-medium text-white">{videoStage?.message || "Rendering your latest export..."}</p>
                  <p>{formatProgressPercent(renderProgress)} complete</p>
                  {latestVideo ? <p className="text-xs text-mutedForeground">Your last completed export will stay available until this new render finishes.</p> : null}
                </div>
                <div className="h-2 w-full max-w-md overflow-hidden rounded-full bg-white/10">
                  <div className="h-full rounded-full bg-accent transition-all" style={{ width: `${renderProgress}%` }} />
                </div>
              </div>
            ) : latestVideo ? (
              <div className={`flex justify-center bg-black/50 ${latestVideoPortrait ? "p-4" : ""}`}>
                <video
                  key={latestVideoSrc}
                  controls
                  controlsList="nodownload"
                  className={`bg-black ${latestVideoPortrait ? "max-h-[75vh] w-auto max-w-full" : "w-full"}`}
                  style={{ aspectRatio: latestVideoAspect }}
                  src={latestVideoSrc}
                />
              </div>
            ) : (
              <div
                className="flex w-full items-center justify-center text-sm text-mutedForeground"
                style={{ aspectRatio: latestVideoAspect }}
              >
                Render a video to preview it here.
              </div>
            )}
          </div>
          <div className="mt-6 flex flex-wrap gap-3">
            <Button onClick={toggleVideoRender} disabled={savingVideoSettings || (audioActionBusy && !renderInFlight)}>
              {renderInFlight || renderActionBusy ? <LoaderCircle className="h-4 w-4 animate-spin" /> : null}
              {renderActionBusy
                ? "Working..."
                : renderInFlight
                  ? "Cancel render"
                  : videoSettingsDirty
                    ? "Save settings and render"
                    : "Re-render video"}
            </Button>
            <Button variant="secondary" onClick={toggleAudioGeneration} disabled={renderActionBusy || savingVideoSettings}>
              {audioInFlight || audioActionBusy ? <LoaderCircle className="h-4 w-4 animate-spin" /> : null}
              {audioActionBusy ? "Working..." : audioInFlight ? "Cancel audio generation" : "Regenerate audio"}
            </Button>
            <Button variant="outline" onClick={() => duplicateForReedit(latestVideo?.name)} disabled={duplicatingVideoName !== null || !latestVideo}>
              {duplicatingVideoName === (latestVideo?.name ?? "__latest__") ? <LoaderCircle className="h-4 w-4 animate-spin" /> : null}
              {duplicatingVideoName === (latestVideo?.name ?? "__latest__") ? "Duplicating..." : "Duplicate for re-edit"}
            </Button>
            <Button variant="outline" onClick={rewindToNarration} disabled={rewindingToNarration}>
              {rewindingToNarration ? <LoaderCircle className="h-4 w-4 animate-spin" /> : null}
              {rewindingToNarration ? "Rewinding..." : "Back to narration"}
            </Button>
            {latestVideo ? (
              <a
                href={latestVideoDownloadUrl}
                className="inline-flex items-center gap-2 rounded-full border border-white/[0.08] px-5 py-2.5 text-sm font-medium text-white transition hover:bg-white/5"
              >
                <Download className="h-4 w-4" />
                Download latest video
              </a>
            ) : null}
          </div>
        </Card>

        <div className="space-y-6">
          <Card>
            <CardTitle className="text-base">Render settings</CardTitle>
            <CardDescription className="mt-1">Base resolution auto-flips for vertical phone-first exports.</CardDescription>
            <div className="mt-6 grid gap-4 md:grid-cols-2">
              <label className="space-y-2">
                <span className="text-sm text-mutedForeground">Resolution</span>
                <select className="h-11 w-full rounded-2xl border border-white/[0.08] bg-white/[0.04] px-4 text-sm text-foreground focus:outline-none focus:border-accent/40 focus:ring-2 focus:ring-accent/30 transition-colors duration-fast" value={resolution} onChange={(event) => setResolution(event.target.value)}>
                  <option value="1920x1080">1920 x 1080</option>
                  <option value="1280x720">1280 x 720</option>
                </select>
              </label>
              <label className="space-y-2">
                <span className="text-sm text-mutedForeground">Orientation</span>
                <select className="h-11 w-full rounded-2xl border border-white/[0.08] bg-white/[0.04] px-4 text-sm text-foreground focus:outline-none focus:border-accent/40 focus:ring-2 focus:ring-accent/30 transition-colors duration-fast" value={orientation} onChange={(event) => setOrientation(event.target.value as "landscape" | "vertical")}>
                  <option value="landscape">Landscape</option>
                  <option value="vertical">Vertical</option>
                </select>
              </label>
              <label className="space-y-2">
                <span className="text-sm text-mutedForeground">Panel layout</span>
                <select className="h-11 w-full rounded-2xl border border-white/[0.08] bg-white/[0.04] px-4 text-sm text-foreground focus:outline-none focus:border-accent/40 focus:ring-2 focus:ring-accent/30 transition-colors duration-fast" value={panelLayout} onChange={(event) => setPanelLayout(event.target.value as PanelLayout)}>
                  <option value="card">Centered card with blur</option>
                  <option value="fullscreen">Fullscreen panel</option>
                </select>
              </label>
              <label className="space-y-2">
                <span className="text-sm text-mutedForeground">Output format</span>
                <select className="h-11 w-full rounded-2xl border border-white/[0.08] bg-white/[0.04] px-4 text-sm text-foreground focus:outline-none focus:border-accent/40 focus:ring-2 focus:ring-accent/30 transition-colors duration-fast" value={outputFormat} onChange={(event) => setOutputFormat(event.target.value as "mp4" | "mov")}>
                  <option value="mp4">MP4</option>
                  <option value="mov">MOV</option>
                </select>
              </label>
              <div className="rounded-2xl border border-white/[0.08] bg-white/[0.04] p-4 md:col-span-2">
                <div className="flex flex-wrap items-start justify-between gap-4">
                  <div className="space-y-1">
                    <p className="text-sm font-medium text-white">YouTube thumbnail lead-in</p>
                    <p className="text-xs text-mutedForeground">
                      Upload a custom image and Panelia will place it in the first seconds of the final video so YouTube can pick it up.
                    </p>
                  </div>
                  <input
                    ref={thumbnailInputRef}
                    type="file"
                    accept="image/*"
                    className="hidden"
                    onChange={(event) => handleThumbnailUpload(event.target.files?.[0] ?? null)}
                  />
                  <div className="flex flex-wrap gap-2">
                    <Button type="button" variant="secondary" onClick={() => thumbnailInputRef.current?.click()} disabled={uploadingThumbnail}>
                      {uploadingThumbnail ? "Uploading..." : project.video_thumbnail_url ? "Replace thumbnail" : "Upload thumbnail"}
                    </Button>
                    {project.video_thumbnail_url ? (
                      <Button type="button" variant="outline" onClick={removeVideoThumbnail} disabled={uploadingThumbnail}>
                        Remove
                      </Button>
                    ) : null}
                  </div>
                </div>
                <div className="mt-4 grid gap-4 md:grid-cols-[180px_1fr]">
                  <div className="overflow-hidden rounded-2xl border border-white/[0.08] bg-black/30">
                    {project.video_thumbnail_url ? (
                      // eslint-disable-next-line @next/next/no-img-element
                      <img src={videoThumbnailSrc} alt="Video thumbnail" className="aspect-[9/16] h-full w-full object-cover" />
                    ) : (
                      <div className="flex aspect-[9/16] items-center justify-center px-4 text-center text-xs text-mutedForeground">
                        No custom thumbnail uploaded yet.
                      </div>
                    )}
                  </div>
                  <div className="space-y-4">
                    <label className="flex items-center gap-3 rounded-2xl border border-white/[0.08] bg-white/[0.03] px-4 py-3 text-sm text-white">
                      <input
                        type="checkbox"
                        className="h-4 w-4 rounded border-white/20 bg-white/5"
                        checked={introThumbnailEnabled}
                        onChange={(event) => setIntroThumbnailEnabled(event.target.checked)}
                      />
                      Use thumbnail in the opening seconds of the rendered video
                    </label>
                    <label className="space-y-2">
                      <span className="text-sm text-mutedForeground">Thumbnail duration</span>
                      <input
                        type="number"
                        min={0.5}
                        max={4}
                        step={0.1}
                        value={introThumbnailSeconds}
                        onChange={(event) => setIntroThumbnailSeconds(Math.max(0.5, Math.min(4, Number(event.target.value) || 1.5)))}
                        className="h-11 w-full rounded-2xl border border-white/[0.08] bg-white/[0.04] px-4 text-sm text-foreground focus:outline-none focus:border-accent/40 focus:ring-2 focus:ring-accent/30 transition-colors duration-fast"
                      />
                    </label>
                  </div>
                </div>
              </div>
              <div className="rounded-2xl border border-white/[0.08] bg-white/[0.04] p-4 md:col-span-2">
                <p className="text-sm text-mutedForeground">Final canvas</p>
                <p className="mt-2 text-sm font-semibold text-white">
                  {targetDimensions.width} × {targetDimensions.height} • {orientation === "vertical" ? "Vertical" : "Landscape"}
                </p>
                <p className="mt-1 text-xs text-mutedForeground">
                  {panelLayout === "fullscreen" ? "Panels fill the frame" : "Panels sit on a blurred background card"}
                  {introThumbnailEnabled ? ` • thumbnail lead-in ${introThumbnailSeconds.toFixed(1)}s` : ""}
                </p>
              </div>
            </div>
            <Button className="mt-4 w-full" onClick={saveVideoSettings} disabled={savingVideoSettings || !videoSettingsDirty}>
              {savingVideoSettings ? "Saving..." : "Save render settings"}
            </Button>
          </Card>

          <Card>
            <CardTitle className="text-base">Finished videos</CardTitle>
            <CardDescription className="mt-1">Add videos to the merge queue, then arrange their order.</CardDescription>
            <div className="mt-6 space-y-3">
              {project.videos.length ? (
                project.videos.map((video) => {
                  const queuedIndex = selectedPaths.indexOf(video.path);
                  const queued = queuedIndex >= 0;
                  return (
                    <div key={video.path} className="rounded-2xl border border-white/[0.08] bg-white/[0.04] p-4">
                      <div className="flex items-center justify-between gap-3">
                      <div className="flex flex-1 items-center justify-between gap-3">
                        <div className="min-w-0">
                          <p className="truncate font-medium text-white" title={video.name}>{video.name}</p>
                          <p className="mt-1 text-sm text-mutedForeground">
                            {formatCanvas(video.width, video.height)} • {formatRelativeDate(video.created_at)}
                          </p>
                        </div>
                          <div className="flex items-center gap-2">
                            {queued ? <span className="rounded-full bg-accent/15 px-3 py-1 text-xs text-accent">#{queuedIndex + 1} in queue</span> : null}
                            <Button variant={queued ? "outline" : "secondary"} onClick={() => toggleVideoInQueue(video.path)}>
                              {queued ? <X className="h-4 w-4" /> : <Plus className="h-4 w-4" />}
                              {queued ? "Remove" : "Add to merge"}
                            </Button>
                          </div>
                        </div>
                        <Button
                          variant="ghost"
                          onClick={() => duplicateForReedit(video.name)}
                          disabled={duplicatingVideoName !== null}
                        >
                          {duplicatingVideoName === video.name ? <LoaderCircle className="h-4 w-4 animate-spin" /> : null}
                          Duplicate
                        </Button>
                        <Button
                          variant="ghost"
                          disabled={deletingVideoName !== null}
                          onClick={async () => {
                            setDeletingVideoName(video.name);
                            try {
                              await api.deleteVideo(project.id, video.name);
                              setProject(await api.getProject(project.id));
                            } catch (err) {
                              console.error("Failed to delete video:", err);
                            } finally {
                              setDeletingVideoName(null);
                            }
                          }}
                        >
                          {deletingVideoName === video.name ? <LoaderCircle className="h-4 w-4 animate-spin" /> : null}
                          {deletingVideoName === video.name ? "Deleting..." : "Delete"}
                        </Button>
                      </div>
                    </div>
                  );
                })
              ) : (
                <p className="text-sm text-mutedForeground">No finished videos yet.</p>
              )}
            </div>
          </Card>

          <Card>
            <CardTitle className="text-base">Merge queue</CardTitle>
            <CardDescription className="mt-1">Arrange clips in order — Panelia normalizes resolution before merging.</CardDescription>
            <div className="mt-6 space-y-3">
              {queuedVideos.length ? (
                queuedVideos.map((video, index) => (
                  <div key={video!.path} className="flex items-center justify-between gap-3 rounded-2xl border border-white/[0.08] bg-white/[0.04] p-4">
                    <div>
                      <p className="text-sm font-medium text-white">
                        {index + 1}. {video!.name}
                      </p>
                      <p className="mt-1 text-xs text-mutedForeground">
                        {formatCanvas(video!.width, video!.height)} • {formatRelativeDate(video!.created_at)}
                      </p>
                    </div>
                    <div className="flex items-center gap-2">
                      <Button variant="ghost" onClick={() => moveQueuedVideo(video!.path, "up")} disabled={index === 0}>
                        <ArrowUp className="h-4 w-4" />
                      </Button>
                      <Button variant="ghost" onClick={() => moveQueuedVideo(video!.path, "down")} disabled={index === queuedVideos.length - 1}>
                        <ArrowDown className="h-4 w-4" />
                      </Button>
                      <Button variant="ghost" onClick={() => toggleVideoInQueue(video!.path)}>
                        <X className="h-4 w-4" />
                      </Button>
                    </div>
                  </div>
                ))
              ) : (
                <p className="text-sm text-mutedForeground">Add at least two finished videos above to build a merge queue.</p>
              )}
            </div>
            <input
              className="mt-6 h-11 w-full rounded-2xl border border-white/[0.08] bg-white/[0.04] px-4 text-sm text-foreground focus:outline-none focus:border-accent/40 focus:ring-2 focus:ring-accent/30 transition-colors duration-fast"
              value={outputName}
              onChange={(event) => setOutputName(event.target.value)}
              placeholder="merged-cut"
            />
            <Button className="mt-4 w-full" onClick={mergeSelectedVideos} disabled={selectedPaths.length < 2 || mergingVideos}>
              {mergingVideos ? <LoaderCircle className="h-4 w-4 animate-spin" /> : null}
              {mergingVideos ? "Merging..." : "Merge queued videos"}
            </Button>
          </Card>
        </div>
      </div>
    </AppShell>
  );
}
