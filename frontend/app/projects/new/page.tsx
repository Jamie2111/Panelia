"use client";

import { ChangeEvent, DragEvent, useEffect, useMemo, useState } from "react";
import { useRouter } from "next/navigation";
import { LoaderCircle, PlayCircle, UploadCloud } from "lucide-react";

import { AppShell } from "@/components/project/app-shell";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardDescription, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { api } from "@/lib/api";
import { buildMediaUrl } from "@/lib/utils";
import { CatalogOptions, DuplicateHandlingMode, SourceType } from "@/lib/types";

type CreateSourceType = SourceType | "url";

const sourceOptions: { value: CreateSourceType; label: string; description: string }[] = [
  { value: "url", label: "URL", description: "Paste a MangaDex or comix.to chapter URL, or a title URL with a chapter range." },
  { value: "zip", label: "ZIP archive", description: "Extract page images from a compressed folder." },
  { value: "pdf", label: "PDF file", description: "Render each PDF page into a normalized image." },
  { value: "images", label: "Individual images", description: "Drag in JPG, PNG, or WEBP files directly." },
  { value: "folder", label: "Folder upload", description: "Bring in an entire page directory from your machine." }
];

const sourceLanguageOptions = [
  { value: "", label: "Any language" },
  { value: "en", label: "English" },
  { value: "ja", label: "Japanese" },
  { value: "ko", label: "Korean" },
  { value: "zh", label: "Chinese" },
  { value: "es", label: "Spanish" },
  { value: "pt-br", label: "Portuguese (BR)" },
  { value: "pt", label: "Portuguese" },
  { value: "fr", label: "French" },
  { value: "de", label: "German" }
];

const duplicateModeOptions: { value: DuplicateHandlingMode; label: string; description: string }[] = [
  { value: "auto_pick_best", label: "Auto-pick best", description: "Choose the strongest chapter match automatically." },
  { value: "prefer_official", label: "Prefer official", description: "Favor official releases when duplicates exist." },
  { value: "prefer_fan", label: "Prefer fan", description: "Favor non-official scanlations when duplicates exist." },
  { value: "prefer_consistent_group", label: "Prefer consistent group", description: "Bias toward one consistent scanlation group across the range." }
];

function resolvedDimensions(resolution: string, orientation: string) {
  const [rawWidth, rawHeight] = resolution.split("x").map(Number);
  if (orientation === "vertical") {
    return { width: Math.min(rawWidth, rawHeight), height: Math.max(rawWidth, rawHeight) };
  }
  return { width: Math.max(rawWidth, rawHeight), height: Math.min(rawWidth, rawHeight) };
}

function detectUrlImportSource(urls: string[]): SourceType {
  if (!urls.length) {
    throw new Error("Paste at least one MangaDex or comix.to URL.");
  }

  const parsedHosts = urls.map((url) => {
    try {
      return new URL(url).hostname.toLowerCase();
    } catch {
      throw new Error(`"${url}" is not a valid URL.`);
    }
  });

  const isMangadex = parsedHosts.every((host) => host === "mangadex.org" || host.endsWith(".mangadex.org"));
  if (isMangadex) {
    return "mangadex_url";
  }

  const isComix = parsedHosts.every((host) => host === "comix.to" || host === "www.comix.to");
  if (isComix) {
    return "comix_to_url";
  }

  const supportedHosts = parsedHosts.every(
    (host) => host === "mangadex.org" || host.endsWith(".mangadex.org") || host === "comix.to" || host === "www.comix.to"
  );
  if (supportedHosts) {
    throw new Error("Use one site per project for URL imports. Panelia cannot mix MangaDex and comix.to links in the same import yet.");
  }

  throw new Error("Only MangaDex and comix.to links are supported in the URL importer right now.");
}

export default function CreateProjectPage() {
  const router = useRouter();
  const [catalog, setCatalog] = useState<CatalogOptions | null>(null);
  const [sourceType, setSourceType] = useState<CreateSourceType>("url");
  const [name, setName] = useState("New manga recap");
  const [sourceUrlInput, setSourceUrlInput] = useState("");
  const [chapterRange, setChapterRange] = useState("");
  const [sourceLanguage, setSourceLanguage] = useState("en");
  const [duplicateMode, setDuplicateMode] = useState<DuplicateHandlingMode>("auto_pick_best");
  const [files, setFiles] = useState<File[]>([]);
  const [voice, setVoice] = useState("af_bella");
  const [langCode, setLangCode] = useState("a");
  const [speed, setSpeed] = useState(1);
  const [resolution, setResolution] = useState("1920x1080");
  const [orientation, setOrientation] = useState("landscape");
  const [outputFormat, setOutputFormat] = useState("mp4");
  const [musicEnabled, setMusicEnabled] = useState(true);
  const [musicTrack, setMusicTrack] = useState("");
  const [musicVolume, setMusicVolume] = useState(0.14);
  const [musicUploadFile, setMusicUploadFile] = useState<File | null>(null);
  const [uploadingMusic, setUploadingMusic] = useState(false);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [loadingOptions, setLoadingOptions] = useState(true);

  const urls = useMemo(
    () =>
      sourceUrlInput
        .split("\n")
        .map((entry) => entry.trim())
        .filter(Boolean),
    [sourceUrlInput]
  );

  useEffect(() => {
    async function loadCatalog() {
      try {
        setCatalog(await api.getCatalogOptions());
        setError(null);
      } catch (err) {
        setError(err instanceof Error ? err.message : "Unable to load narrator and music options.");
      } finally {
        setLoadingOptions(false);
      }
    }
    loadCatalog();
  }, []);

  useEffect(() => {
    const validVoices = catalog?.voices.filter((option) => option.lang_code === langCode) ?? [];
    if (!validVoices.length) {
      return;
    }
    if (!validVoices.some((option) => option.id === voice)) {
      setVoice(validVoices[0].id);
    }
  }, [catalog, langCode, voice]);

  useEffect(() => {
    if (!catalog) {
      return;
    }
    if (!musicTrack) {
      const defaultTrack = catalog.music_tracks.find((track) => track.available);
      if (defaultTrack) {
        setMusicTrack(defaultTrack.name);
      }
    }
  }, [catalog, musicTrack]);

  function handleFileInput(event: ChangeEvent<HTMLInputElement>) {
    setFiles(Array.from(event.target.files ?? []));
  }

  async function handleMusicUpload() {
    if (!musicUploadFile) {
      return;
    }

    try {
      setUploadingMusic(true);
      const uploaded = await api.uploadMusicTrack(musicUploadFile);
      const refreshed = await api.getCatalogOptions();
      setCatalog(refreshed);
      setMusicTrack(uploaded.name);
      setMusicEnabled(true);
      setMusicUploadFile(null);
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unable to import the MP3 track.");
    } finally {
      setUploadingMusic(false);
    }
  }

  function handleDrop(event: DragEvent<HTMLDivElement>) {
    event.preventDefault();
    setFiles(Array.from(event.dataTransfer.files ?? []));
  }

  async function handleSubmit() {
    try {
      setSubmitting(true);
      setError(null);

      const resolvedSourceType =
        sourceType === "url"
          ? detectUrlImportSource(urls)
          : sourceType;

      const project = await api.createProject({
        name,
        sourceType: resolvedSourceType,
        mangadexUrl: resolvedSourceType === "mangadex_url" ? urls.join("\n") : undefined,
        comixUrl: resolvedSourceType === "comix_to_url" ? urls.join("\n") : undefined,
        chapterRange: sourceType === "url" ? chapterRange : undefined,
        sourceLanguage: sourceType === "url" ? sourceLanguage : undefined,
        duplicateMode: sourceType === "url" ? duplicateMode : undefined,
        files,
        voice,
        langCode,
        speed,
        resolution,
        orientation,
        outputFormat,
        musicEnabled,
        musicTrack: musicTrack || undefined,
        musicVolume
      });
      router.push(`/projects/${project.id}`);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unable to create project.");
    } finally {
      setSubmitting(false);
    }
  }

  const languageOptions = catalog?.languages ?? [];
  const voiceOptions = (catalog?.voices ?? []).filter((option) => option.lang_code === langCode);
  const selectedVoice = voiceOptions.find((option) => option.id === voice);
  const selectedLanguage = languageOptions.find((option) => option.code === langCode);
  const musicOptions = catalog?.music_tracks ?? [];
  const selectedTrack = musicOptions.find((track) => track.name === musicTrack);
  const voicePreviewUrl = api.voicePreviewUrl(voice, langCode, speed);
  const targetDimensions = resolvedDimensions(resolution, orientation);

  return (
    <AppShell
      title="Create a new project"
      description="Pull in a MangaDex or comix.to chapter, a ZIP, a PDF, or raw page images. Pick a narrator and soundtrack with live previews - then we run the rest."
    >
      <div className="grid gap-6 xl:grid-cols-[1.2fr_0.8fr]">
        <div className="space-y-6">
          <Card padded="md">
            <CardTitle>Source</CardTitle>
            <CardDescription className="mt-2">
              Choose how this project should ingest its pages.
            </CardDescription>
            <div className="mt-6 grid gap-3 md:grid-cols-2">
              {sourceOptions.map((option) => (
                <button
                  key={option.value}
                  type="button"
                  onClick={() => {
                    setSourceType(option.value);
                    setFiles([]);
                    setSourceUrlInput("");
                    if (!name.trim()) {
                      setName(option.value === "url" ? "New recap" : `New ${option.label.toLowerCase()} import`);
                    }
                  }}
                  className={`rounded-2xl border p-4 text-left transition-all duration-fast ease-liquid ${
                    sourceType === option.value
                      ? "border-accent/40 bg-accent/[0.10] shadow-[0_0_24px_-12px_rgb(var(--p-accent)/0.6)]"
                      : "border-white/[0.08] bg-white/[0.03] hover:bg-white/[0.06] hover:border-white/[0.14]"
                  }`}
                >
                  <p className="font-medium text-foreground">{option.label}</p>
                  <p className="mt-2 text-sm text-mutedForeground leading-relaxed">
                    {option.description}
                  </p>
                </button>
              ))}
            </div>
            <div className="mt-6 grid gap-4">
              <label className="space-y-2">
                <span className="text-sm text-mutedForeground">Project name</span>
                <Input value={name} onChange={(event) => setName(event.target.value)} placeholder="Chapter 42 recap" />
              </label>

              {sourceType === "url" ? (
                <>
                  <label className="space-y-2">
                    <span className="text-sm text-mutedForeground">MangaDex or comix.to URL</span>
                    <Textarea
                      value={sourceUrlInput}
                      onChange={(event) => setSourceUrlInput(event.target.value)}
                      placeholder={"https://mangadex.org/title/...\n\nhttps://comix.to/title/3yz2-global-freeze-i-created-an-apocalypse-shelter"}
                    />
                    <p className="text-xs text-mutedForeground">
                      Paste one or more MangaDex or comix.to links, one per line. Panelia will detect the site automatically. Title URLs can use the chapter range controls below.
                    </p>
                  </label>
                  <div className="grid gap-4 md:grid-cols-3">
                    <label className="space-y-2">
                      <span className="text-sm text-mutedForeground">Chapter range</span>
                      <Input
                        value={chapterRange}
                        onChange={(event) => setChapterRange(event.target.value)}
                        placeholder="1-20 or 12, 13, 15-18"
                      />
                      <p className="text-xs text-mutedForeground">Leave blank to import the first matching chapter from a title URL, or use direct chapter URLs.</p>
                    </label>
                    <label className="space-y-2">
                      <span className="text-sm text-mutedForeground">Source language</span>
                      <select
                        className="h-11 w-full rounded-2xl border border-white/[0.08] bg-white/[0.04] px-4 text-sm text-foreground focus:outline-none focus:border-accent/40 focus:ring-2 focus:ring-accent/30 transition-colors duration-fast"
                        value={sourceLanguage}
                        onChange={(event) => setSourceLanguage(event.target.value)}
                      >
                        {sourceLanguageOptions.map((option) => (
                          <option key={option.value || "any"} value={option.value}>
                            {option.label}
                          </option>
                        ))}
                      </select>
                    </label>
                    <label className="space-y-2">
                      <span className="text-sm text-mutedForeground">Duplicate handling</span>
                      <select
                        className="h-11 w-full rounded-2xl border border-white/[0.08] bg-white/[0.04] px-4 text-sm text-foreground focus:outline-none focus:border-accent/40 focus:ring-2 focus:ring-accent/30 transition-colors duration-fast"
                        value={duplicateMode}
                        onChange={(event) => setDuplicateMode(event.target.value as DuplicateHandlingMode)}
                      >
                        {duplicateModeOptions.map((option) => (
                          <option key={option.value} value={option.value}>
                            {option.label}
                          </option>
                        ))}
                      </select>
                      <p className="text-xs text-mutedForeground">
                        {duplicateModeOptions.find((option) => option.value === duplicateMode)?.description}
                      </p>
                    </label>
                  </div>
                </>
              ) : (
                <div
                  onDragOver={(event) => event.preventDefault()}
                  onDrop={handleDrop}
                  className="rounded-2xl border border-dashed border-white/[0.14] bg-white/[0.03] p-10 text-center transition-colors duration-fast hover:bg-white/[0.05]"
                >
                  <UploadCloud className="mx-auto h-10 w-10 text-accent" />
                  <p className="mt-4 font-medium text-foreground">Drag and drop pages here</p>
                  <p className="mt-2 text-sm text-mutedForeground">
                    ZIP · PDF · JPG · PNG · WEBP - or an entire folder.
                  </p>
                  <label className="mt-6 inline-flex cursor-pointer items-center gap-2 rounded-full bg-accent px-4 py-2 text-sm font-medium text-accent-foreground shadow-[0_0_24px_-6px_rgb(var(--p-accent)/0.6)] transition-transform duration-fast hover:-translate-y-px">
                    Choose files
                    <input
                      type="file"
                      multiple
                      className="hidden"
                      onChange={handleFileInput}
                      {...(sourceType === "folder" ? ({ webkitdirectory: "true", directory: "true" } as Record<string, string>) : {})}
                    />
                  </label>
                  {files.length ? (
                    <p className="mt-4 text-sm text-mutedForeground">
                      {files.length} file{files.length === 1 ? "" : "s"} selected
                    </p>
                  ) : null}
                </div>
              )}
            </div>
          </Card>

          <Card>
            <CardTitle>Voice and output</CardTitle>
            <CardDescription className="mt-2">Use dropdowns and previews so the narrator choice feels obvious instead of technical.</CardDescription>
            <div className="mt-6 grid gap-4 md:grid-cols-2">
              <label className="space-y-2">
                <span className="text-sm text-mutedForeground">Narration language</span>
                <select className="h-11 w-full rounded-2xl border border-white/[0.08] bg-white/[0.04] px-4 text-sm text-foreground focus:outline-none focus:border-accent/40 focus:ring-2 focus:ring-accent/30 transition-colors duration-fast" value={langCode} onChange={(event) => setLangCode(event.target.value)} disabled={loadingOptions}>
                  {languageOptions.map((option) => (
                    <option key={option.code} value={option.code}>
                      {option.label}
                    </option>
                  ))}
                </select>
              </label>
              <label className="space-y-2">
                <span className="text-sm text-mutedForeground">Speech speed</span>
                <Input type="number" step="0.05" min="0.7" max="1.3" value={speed} onChange={(event) => setSpeed(Number(event.target.value))} />
              </label>
              <label className="space-y-2">
                <span className="text-sm text-mutedForeground">Resolution</span>
                <select className="h-11 w-full rounded-2xl border border-white/[0.08] bg-white/[0.04] px-4 text-sm text-foreground focus:outline-none focus:border-accent/40 focus:ring-2 focus:ring-accent/30 transition-colors duration-fast" value={resolution} onChange={(event) => setResolution(event.target.value)}>
                  <option value="1920x1080">1920 × 1080</option>
                  <option value="1280x720">1280 × 720</option>
                </select>
              </label>
              <label className="space-y-2">
                <span className="text-sm text-mutedForeground">Orientation</span>
                <select className="h-11 w-full rounded-2xl border border-white/[0.08] bg-white/[0.04] px-4 text-sm text-foreground focus:outline-none focus:border-accent/40 focus:ring-2 focus:ring-accent/30 transition-colors duration-fast" value={orientation} onChange={(event) => setOrientation(event.target.value)}>
                  <option value="landscape">Landscape</option>
                  <option value="vertical">Vertical</option>
                </select>
              </label>
              <label className="space-y-2">
                <span className="text-sm text-mutedForeground">Output format</span>
                <select className="h-11 w-full rounded-2xl border border-white/[0.08] bg-white/[0.04] px-4 text-sm text-foreground focus:outline-none focus:border-accent/40 focus:ring-2 focus:ring-accent/30 transition-colors duration-fast" value={outputFormat} onChange={(event) => setOutputFormat(event.target.value)}>
                  <option value="mp4">MP4</option>
                  <option value="mov">MOV</option>
                </select>
              </label>
              <div className="rounded-2xl border border-white/[0.08] bg-white/[0.04] p-4 md:col-span-2">
                <p className="text-sm text-mutedForeground">Final canvas</p>
                <p className="mt-2 text-sm font-semibold text-white">
                  {targetDimensions.width} × {targetDimensions.height} • {orientation === "vertical" ? "Vertical" : "Landscape"}
                </p>
              </div>
            </div>

            <div className="mt-6 rounded-2xl border border-white/[0.08] bg-white/[0.04] p-4">
              <div className="space-y-3">
                <div className="flex flex-wrap items-start justify-between gap-3">
                  <div>
                    <p className="text-sm font-medium text-white">{selectedVoice?.label ?? "Voice preview"}</p>
                    <p className="mt-1 text-sm text-mutedForeground">{selectedVoice?.description ?? "Choose a voice to preview it."}</p>
                  </div>
                  {selectedVoice?.quality_note ? <Badge>{selectedVoice.quality_note}</Badge> : null}
                </div>
                <label className="block space-y-2">
                  <span className="text-sm text-mutedForeground">Voice</span>
                  <select className="h-11 w-full rounded-2xl border border-white/[0.08] bg-white/[0.04] px-4 text-sm text-foreground focus:outline-none focus:border-accent/40 focus:ring-2 focus:ring-accent/30 transition-colors duration-fast" value={voice} onChange={(event) => setVoice(event.target.value)} disabled={loadingOptions}>
                    {voiceOptions.map((option) => (
                      <option key={option.id} value={option.id}>
                        {option.label}
                      </option>
                    ))}
                  </select>
                </label>
              </div>
              {selectedLanguage ? <p className="mt-3 text-xs text-mutedForeground">{selectedLanguage.description}</p> : null}
              {selectedVoice?.style_tags?.length ? (
                <div className="mt-3 flex flex-wrap gap-2">
                  {selectedVoice.style_tags.map((tag) => (
                    <Badge key={tag} className="bg-white/8">
                      {tag}
                    </Badge>
                  ))}
                </div>
              ) : null}
              <div className="mt-4">
                <p className="mb-2 flex items-center gap-2 text-xs uppercase tracking-[0.22em] text-accent">
                  <PlayCircle className="h-3.5 w-3.5" />
                  Preview selected voice
                </p>
                <audio key={`${voice}-${langCode}-${speed}`} controls preload="none" className="w-full" src={voicePreviewUrl} />
              </div>
            </div>
          </Card>
        </div>

        <div className="space-y-6">
          <Card>
            <CardTitle>Music bed</CardTitle>
            <CardDescription className="mt-2">Preview the soundtrack before you commit to a final render.</CardDescription>
            <div className="mt-6 space-y-4">
              <label className="flex items-center justify-between rounded-2xl border border-white/[0.08] bg-white/[0.04] px-4 py-3">
                <span className="text-sm text-white">Enable music</span>
                <input type="checkbox" checked={musicEnabled} onChange={(event) => setMusicEnabled(event.target.checked)} className="h-4 w-4 accent-cyan-400" />
              </label>
              <label className="space-y-2">
                <span className="text-sm text-mutedForeground">Track preset</span>
                <select className="h-11 w-full rounded-2xl border border-white/[0.08] bg-white/[0.04] px-4 text-sm text-foreground focus:outline-none focus:border-accent/40 focus:ring-2 focus:ring-accent/30 transition-colors duration-fast" value={musicTrack} onChange={(event) => setMusicTrack(event.target.value)} disabled={loadingOptions}>
                  <option value="">No track</option>
                  {musicOptions.map((track) => (
                    <option key={`${track.source ?? "builtin"}-${track.file}`} value={track.name}>
                      {track.name} {track.source === "uploaded" ? "(uploaded)" : track.available ? "" : "(add file to enable)"}
                    </option>
                  ))}
                </select>
              </label>
              <div className="rounded-2xl border border-dashed border-white/[0.14] bg-white/4 p-4">
                <p className="text-sm font-medium text-white">Add your own MP3</p>
                <p className="mt-1 text-sm text-mutedForeground">Upload a soundtrack once and it becomes available everywhere in the app.</p>
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
              <label className="space-y-2">
                <span className="text-sm text-mutedForeground">Music volume</span>
                <Input type="number" min="0" max="1" step="0.05" value={musicVolume} onChange={(event) => setMusicVolume(Number(event.target.value))} />
              </label>
              <div className="rounded-2xl border border-white/[0.08] bg-white/[0.04] p-4">
                <p className="text-sm font-medium text-white">{selectedTrack?.name ?? "Soundtrack preview"}</p>
                <p className="mt-1 text-sm text-mutedForeground">
                  {selectedTrack?.available
                    ? `${selectedTrack.source === "uploaded" ? "Uploaded track" : "Built-in preset"}${selectedTrack.mood ? ` • Mood: ${selectedTrack.mood}` : ""}`
                    : "Add the matching MP3 file or upload your own track to enable this preview."}
                </p>
                {selectedTrack?.available && selectedTrack.url ? (
                  <audio key={selectedTrack.file} controls preload="none" className="mt-4 w-full" src={buildMediaUrl(selectedTrack.url)} />
                ) : null}
              </div>
            </div>
          </Card>

          <Card>
            <CardTitle>What happens next</CardTitle>
            <CardDescription className="mt-2">
              Project creation saves the source, normalizes all pages into a `pages/` folder, and automatically advances the background pipeline until a review step needs you.
            </CardDescription>
            <ul className="mt-6 space-y-3 text-sm text-mutedForeground">
              <li>1. Ingest chapter assets into a normalized page sequence.</li>
              <li>2. Run MAGI automatically to detect panel boxes and write `panels.json`.</li>
              <li>3. Pause for panel review when human cleanup is needed.</li>
              <li>4. After you save review changes, script, audio, and video continue automatically.</li>
            </ul>
            {error ? <p className="mt-4 rounded-2xl border border-red-500/20 bg-red-500/10 px-4 py-3 text-sm text-red-200">{error}</p> : null}
            <Button className="mt-6 w-full" size="lg" onClick={handleSubmit} disabled={submitting || loadingOptions}>
              {submitting || loadingOptions ? <LoaderCircle className="h-4 w-4 animate-spin" /> : null}
              Create project
            </Button>
          </Card>
        </div>
      </div>
    </AppShell>
  );
}
