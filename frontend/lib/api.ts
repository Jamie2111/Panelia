import { CatalogOptions, ChannelPreset, CharacterDictionaryResponse, CharacterPortraitsResponse, CharacterReviewState, DetectorTrainingStatus, DuplicateHandlingMode, JobRecord, MusicTrack, PanelRewriteMode, PanelRewriteResponse, PipelineStage, ProjectDetail, ProjectSummary, SourceType, StorySegment } from "@/lib/types";

const API_BASE = process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://127.0.0.1:8010/api";

function asArray<T>(value: unknown): T[] {
  return Array.isArray(value) ? value as T[] : [];
}

function asRecord(value: unknown): Record<string, unknown> {
  return value && typeof value === "object" && !Array.isArray(value) ? (value as Record<string, unknown>) : {};
}

function normalizeCatalogOptions(payload: CatalogOptions): CatalogOptions {
  return {
    languages: asArray<CatalogOptions["languages"][number]>(payload?.languages),
    voices: asArray<CatalogOptions["voices"][number]>(payload?.voices).map((voice) => ({
      ...voice,
      style_tags: asArray<string>(voice?.style_tags)
    })),
    music_tracks: asArray<MusicTrack>(payload?.music_tracks)
  };
}

function normalizeProjectSummary<T extends ProjectSummary | ProjectDetail>(project: T): T {
  const chapterMetadata = asRecord(project?.chapter_metadata);
  const stageStates = asRecord(project?.stage_states) as T["stage_states"];
  const activeJobs = asArray<JobRecord>(project?.active_jobs);
  const latestVideo =
    project?.latest_video && typeof project.latest_video === "object" && !Array.isArray(project.latest_video)
      ? project.latest_video
      : null;
  const normalizedSourceType = typeof project?.source_type === "string" ? project.source_type : "images";
  const createdAt = typeof project?.created_at === "string" ? project.created_at : new Date().toISOString();
  const updatedAt = typeof project?.updated_at === "string" ? project.updated_at : createdAt;

  const normalized = {
    ...project,
    source_type: normalizedSourceType,
    created_at: createdAt,
    updated_at: updatedAt,
    chapter_metadata: chapterMetadata,
    stage_states: stageStates,
    active_jobs: activeJobs,
    latest_video: latestVideo,
    thumbnail_url: typeof project?.thumbnail_url === "string" ? project.thumbnail_url : null,
    video_thumbnail_url: typeof project?.video_thumbnail_url === "string" ? project.video_thumbnail_url : null,
    pipeline_config: {
      auto_run_end_to_end: Boolean(project?.pipeline_config?.auto_run_end_to_end)
    }
  };

  if ("panels" in normalized) {
    return {
      ...normalized,
      panels: asArray<ProjectDetail["panels"][number]>((normalized as ProjectDetail).panels),
      script_lines: asArray<string>((normalized as ProjectDetail).script_lines),
      script_display_metadata: asRecord((normalized as ProjectDetail).script_display_metadata) as ProjectDetail["script_display_metadata"],
      story_segments: asArray<StorySegment>((normalized as ProjectDetail).story_segments),
      audio_files: asArray<ProjectDetail["audio_files"][number]>((normalized as ProjectDetail).audio_files),
      videos: asArray<ProjectDetail["videos"][number]>((normalized as ProjectDetail).videos),
      available_music_tracks: asArray<MusicTrack>((normalized as ProjectDetail).available_music_tracks)
    } as T;
  }

  return normalized as T;
}

function normalizePanelForSave(panel: ProjectDetail["panels"][number]) {
  return {
    ...panel,
    x: Math.round(panel.x),
    y: Math.round(panel.y),
    width: Math.max(1, Math.round(panel.width)),
    height: Math.max(1, Math.round(panel.height))
  };
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response;
  try {
    response = await fetch(`${API_BASE}${path}`, {
      cache: "no-store",
      ...init
    });
  } catch (error) {
    const message = error instanceof Error ? error.message : "Network request failed";
    throw new Error(
      message.includes("Failed to fetch")
        ? "Could not reach the backend service. Make sure the FastAPI server and worker are running, then refresh the page."
        : message
    );
  }

  if (!response.ok) {
    const message = (await response.text()).trim();
    if (response.status === 404 && message.toLowerCase() === "not found") {
      throw new Error(
        "The frontend reached a server, but it is not the Panelia backend API. Make sure Panelia's FastAPI server is running on the configured backend target, then refresh the page."
      );
    }
    throw new Error(message || `Request failed: ${response.status}`);
  }

  return response.json() as Promise<T>;
}

export type YouTubeBundleVariant = {
  index: number;
  style_id: string;
  style_label: string;
  url: string | null;
  overlay_text?: string;
};

export type YouTubeBundleResponse = {
  project_id: string;
  title: string | null;
  title_variants: string[];
  description: string | null;
  thumbnail_url: string | null;
  thumbnail_source_url: string | null;
  thumbnail_source_panel_id: string | null;
  thumbnail_variants?: YouTubeBundleVariant[];
  chosen_thumbnail_index?: number;
  short_thumbnail_url?: string | null;
  short_thumbnail_variants?: YouTubeBundleVariant[];
  short_chosen_thumbnail_index?: number;
  short_video_url?: string | null;
  short_title?: string | null;
  short_description?: string | null;
  bundle_dir: string | null;
};

export const api = {
  getCatalogOptions: () => request<CatalogOptions>("/catalog/options").then(normalizeCatalogOptions),
  getPanelDetectorTrainingStatus: () => request<DetectorTrainingStatus>("/training/panel-detector"),
  startPanelDetectorTraining: () =>
    request<DetectorTrainingStatus>("/training/panel-detector/train", {
      method: "POST"
    }),
  cancelPanelDetectorTraining: () =>
    request<DetectorTrainingStatus>("/training/panel-detector/cancel", {
      method: "POST"
    }),
  uploadMusicTrack: async (file: File, trackName?: string, mood?: string) => {
    const form = new FormData();
    form.append("file", file);
    if (trackName?.trim()) {
      form.append("track_name", trackName.trim());
    }
    if (mood?.trim()) {
      form.append("mood", mood.trim());
    }

    return request<MusicTrack>("/catalog/music-upload", {
      method: "POST",
      body: form
    });
  },
  listProjects: () => request<ProjectSummary[]>("/projects").then((projects) => asArray<ProjectSummary>(projects).map(normalizeProjectSummary)),
  getProjectSummary: (projectId: string) => request<ProjectSummary>(`/projects/${projectId}/summary`).then(normalizeProjectSummary),
  getProject: (projectId: string) => request<ProjectDetail>(`/projects/${projectId}`).then(normalizeProjectSummary),
  // Lightweight script-only reader. Used as a fallback by the narration
  // page when the full project payload stalls (e.g. while the worker is
  // mid-write on TTS/render). Reads only script.json on the backend.
  getStoryScript: (projectId: string) =>
    request<{
      chapter_summary: string;
      story_segments: Array<Record<string, unknown>>;
      script_lines: string[];
      script_mode?: string;
    }>(`/projects/${projectId}/story-script`),
  renameProject: (projectId: string, name: string) =>
    request<ProjectDetail>(`/projects/${projectId}/name`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name })
    }).then(normalizeProjectSummary),
  downloadLatestVideoUrl: (projectId: string) => `${API_BASE}/projects/${projectId}/video/latest-download`,
  getChannelPreset: () =>
    request<ChannelPreset>("/channel/preset"),
  updateChannelPreset: (patch: Partial<ChannelPreset>) =>
    request<ChannelPreset>("/channel/preset", {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(patch),
    }),
  getYouTubeBundle: (projectId: string) =>
    request<YouTubeBundleResponse>(`/projects/${projectId}/youtube-bundle`).catch((err) => {
      // Bundle hasn't been generated yet → return null so the UI can show
      // a "preparing" placeholder instead of a hard error.
      if (err instanceof Error && /404/.test(err.message)) return null;
      throw err;
    }),
  updateYouTubeBundle: (
    projectId: string,
    patch: {
      title?: string;
      description?: string;
      chosen_thumbnail_index?: number;
      short_chosen_thumbnail_index?: number;
    },
  ) =>
    request<YouTubeBundleResponse>(`/projects/${projectId}/youtube-bundle`, {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(patch),
    }),
  regenerateThumbnailText: (
    projectId: string,
    payload: { variant_index: number; overlay_text: string; group?: "main" | "short" },
  ) =>
    request<YouTubeBundleResponse>(`/projects/${projectId}/youtube-bundle/thumbnail`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    }),
  uploadVideoThumbnail: async (projectId: string, file: File) => {
    const form = new FormData();
    form.append("file", file);
    return request<ProjectDetail>(`/projects/${projectId}/video-thumbnail`, {
      method: "POST",
      body: form
    }).then(normalizeProjectSummary);
  },
  deleteVideoThumbnail: (projectId: string) =>
    request<ProjectDetail>(`/projects/${projectId}/video-thumbnail`, {
      method: "DELETE"
    }),
  listJobs: (projectId: string) => request<JobRecord[]>(`/projects/${projectId}/jobs`),
  updatePanels: (projectId: string, panels: ProjectDetail["panels"]) =>
    request<{ status: string; project_id: string }>(`/projects/${projectId}/panels`, {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ panels: panels.map(normalizePanelForSave) })
    }),
  updateScript: (
    projectId: string,
    script_lines: string[],
    panel_keeps: Record<string, boolean> = {},
    panel_narrations: Record<string, string> = {},
    panel_locks: Record<string, boolean> = {},
    story_segments: StorySegment[] = []
  ) =>
    request<ProjectDetail>(`/projects/${projectId}/script`, {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ script_lines, panel_keeps, panel_narrations, panel_locks, story_segments })
    }).then(normalizeProjectSummary),
  getCharacterReview: (projectId: string) =>
    request<CharacterReviewState>(`/projects/${projectId}/characters`),
  getCharacterDictionary: (projectId: string) =>
    request<CharacterDictionaryResponse>(`/projects/${projectId}/character-dictionary`),
  getCharacterPortraits: (projectId: string) =>
    request<CharacterPortraitsResponse>(`/projects/${projectId}/character-portraits`),
  updateCharacterPortraits: (projectId: string, characters: CharacterPortraitsResponse["characters"]) =>
    request<CharacterPortraitsResponse>(`/projects/${projectId}/character-portraits`, {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ characters })
    }),
  updateCharacterReview: (
    projectId: string,
    payload: Pick<CharacterReviewState, "protagonist_name" | "identities">
  ) =>
    request<CharacterReviewState>(`/projects/${projectId}/characters`, {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    }),
  rewritePanelNarration: (projectId: string, panel_id: string, mode: PanelRewriteMode, current_narration = "") =>
    request<PanelRewriteResponse>(`/projects/${projectId}/script/rewrite-panel`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ panel_id, mode, current_narration })
    }),
  updateProjectSettings: (
    projectId: string,
    payload: {
      voice_config?: ProjectDetail["voice_config"];
      music_config?: ProjectDetail["music_config"];
      video_config?: ProjectDetail["video_config"];
      pipeline_config?: ProjectDetail["pipeline_config"];
    }
  ) =>
    request<ProjectDetail>(`/projects/${projectId}/settings`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    }).then(normalizeProjectSummary),
  queueStage: (projectId: string, stage: PipelineStage, payload: Record<string, unknown> = {}) =>
    request<JobRecord>(`/projects/${projectId}/jobs`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ stage, payload })
    }),
  rewindProject: (projectId: string, stage: PipelineStage) =>
    request<ProjectDetail>(`/projects/${projectId}/rewind`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ stage })
    }).then(normalizeProjectSummary),
  cancelJob: (projectId: string, jobId: string) =>
    request<{ status: string; job_id: string }>(`/projects/${projectId}/jobs/${jobId}/cancel`, {
      method: "POST"
    }),
  cancelProject: (projectId: string) =>
    request<{ status: string; job_ids: string[] }>(`/projects/${projectId}/cancel`, {
      method: "POST"
    }),
  createProject: async (payload: {
    name: string;
    sourceType: SourceType;
    mangadexUrl?: string;
    comixUrl?: string;
    chapterRange?: string;
    sourceLanguage?: string;
    duplicateMode?: DuplicateHandlingMode;
    files?: File[];
    voice: string;
    langCode: string;
    speed: number;
    resolution: string;
    orientation: string;
    outputFormat: string;
    musicEnabled: boolean;
    musicTrack?: string;
    musicVolume: number;
  }) => {
    const form = new FormData();
    form.append("name", payload.name);
    form.append("source_type", payload.sourceType);
    if (payload.mangadexUrl) {
      form.append("mangadex_url", payload.mangadexUrl);
    }
    if (payload.comixUrl) {
      form.append("comix_url", payload.comixUrl);
    }
    if (payload.chapterRange?.trim()) {
      form.append("chapter_range", payload.chapterRange.trim());
    }
    if (payload.sourceLanguage?.trim()) {
      form.append("source_language", payload.sourceLanguage.trim());
    }
    if (payload.duplicateMode) {
      form.append("duplicate_mode", payload.duplicateMode);
    }
    for (const file of payload.files ?? []) {
      form.append("files", file);
    }
    form.append("voice", payload.voice);
    form.append("lang_code", payload.langCode);
    form.append("speed", String(payload.speed));
    form.append("resolution", payload.resolution);
    form.append("orientation", payload.orientation);
    form.append("output_format", payload.outputFormat);
    form.append("music_enabled", String(payload.musicEnabled));
    if (payload.musicTrack) {
      form.append("music_track", payload.musicTrack);
    }
    form.append("music_volume", String(payload.musicVolume));

    return request<ProjectDetail>("/projects", {
      method: "POST",
      body: form
    });
  },
  createBatchProjects: (urls: string[], base_name?: string) =>
    request<ProjectDetail[]>("/projects/batch", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ urls, base_name })
    }),
  duplicateProject: (
    projectId: string,
    payload: {
      name?: string;
      video_name?: string;
      copy_all_videos?: boolean;
    } = {}
  ) =>
    request<ProjectDetail>(`/projects/${projectId}/duplicate`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    }),
  mergeVideos: (
    projectId: string,
    payload: {
      video_paths: string[];
      output_name: string;
      video_config: ProjectDetail["video_config"];
    }
  ) =>
    request<{ path: string; url: string }>(`/projects/${projectId}/merge`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ project_id: projectId, ...payload })
    }),
  deleteVideo: (projectId: string, videoName: string) =>
    request<{ status: string; video_name: string }>(`/projects/${projectId}/video/${encodeURIComponent(videoName)}`, {
      method: "DELETE"
    }),
  deleteProject: (projectId: string) =>
    request<{ status: string; project_id: string }>(`/projects/${projectId}`, {
      method: "DELETE"
    }),
  voicePreviewUrl: (voice: string, langCode: string, speed: number) =>
    `${API_BASE}/catalog/voice-preview?voice=${encodeURIComponent(voice)}&lang_code=${encodeURIComponent(langCode)}&speed=${encodeURIComponent(String(speed))}`
};
