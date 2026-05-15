"use client";

import { useEffect, useMemo, useState } from "react";
import { useParams } from "next/navigation";
import { ArrowRight, LoaderCircle, RefreshCw, RotateCcw, Save, Users } from "lucide-react";

import { AppShell } from "@/components/project/app-shell";
import { buildProjectViews } from "@/lib/project-views";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardDescription, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Progress } from "@/components/ui/progress";
import { Textarea } from "@/components/ui/textarea";
import { api } from "@/lib/api";
import { formatProgressPercent, getStageProgressMeta } from "@/lib/progress";
import { CharacterReviewIdentity, CharacterReviewState, JobRecord, ProjectDetail } from "@/lib/types";
import { useAdaptivePolling } from "@/lib/use-adaptive-polling";
import { buildMediaUrl, formatRelativeDate } from "@/lib/utils";

function dedupeStrings(values: Array<string | null | undefined>) {
  const seen = new Set<string>();
  const result: string[] = [];
  for (const value of values) {
    const cleaned = String(value ?? "").trim();
    if (!cleaned) continue;
    const key = cleaned.toLowerCase();
    if (seen.has(key)) continue;
    seen.add(key);
    result.push(cleaned);
  }
  return result;
}

function dedupeNumbers(values: Array<number | null | undefined>) {
  return Array.from(new Set(values.filter((value): value is number => Number.isFinite(value)))).sort((left, right) => left - right);
}

function activeStageJobs(project: ProjectDetail | null, stage: JobRecord["stage"]) {
  if (!project) return [];
  return project.active_jobs.filter((job) => job.stage === stage && (job.status === "queued" || job.status === "running"));
}

function mergeIdentities(review: CharacterReviewState, selectedIds: string[], targetId: string) {
  if (selectedIds.length < 2) return review;
  const selectedSet = new Set(selectedIds);
  const target = review.identities.find((identity) => identity.review_id === targetId);
  if (!target) return review;

  const sources = review.identities.filter((identity) => selectedSet.has(identity.review_id) && identity.review_id !== targetId);
  const merged: CharacterReviewIdentity = {
    ...target,
    stable_character_ids: dedupeStrings([
      ...target.stable_character_ids,
      ...sources.flatMap((identity) => identity.stable_character_ids)
    ]),
    source_character_ids: dedupeStrings([
      ...target.source_character_ids,
      ...sources.flatMap((identity) => identity.source_character_ids)
    ]),
    suggested_name: target.suggested_name || sources.map((identity) => identity.suggested_name).find(Boolean) || null,
    remembered_name: target.remembered_name || sources.map((identity) => identity.remembered_name).find(Boolean) || null,
    name: target.name || sources.map((identity) => identity.name).find(Boolean) || null,
    status:
      target.status === "confirmed" || sources.some((identity) => identity.status === "confirmed")
        ? "confirmed"
        : target.status,
    appearance_count: target.appearance_count + sources.reduce((total, identity) => total + identity.appearance_count, 0),
    pages: dedupeNumbers([...target.pages, ...sources.flatMap((identity) => identity.pages)]),
    panel_ids: dedupeStrings([...target.panel_ids, ...sources.flatMap((identity) => identity.panel_ids)]),
    sample_images: [
      ...target.sample_images,
      ...sources.flatMap((identity) => identity.sample_images)
    ].filter((sample, index, items) => items.findIndex((candidate) => candidate.sample_id === sample.sample_id) === index).slice(0, 4),
    notes: dedupeStrings([target.notes, ...sources.map((identity) => identity.notes)]).join("\n") || null
  };

  return {
    ...review,
    identities: review.identities
      .filter((identity) => !selectedSet.has(identity.review_id) || identity.review_id === targetId)
      .map((identity) => (identity.review_id === targetId ? merged : identity))
  };
}

export default function CharacterReviewPage() {
  const params = useParams<{ projectId: string }>();
  const projectId = Array.isArray(params.projectId) ? params.projectId[0] : params.projectId;
  const [project, setProject] = useState<ProjectDetail | null>(null);
  const [review, setReview] = useState<CharacterReviewState | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [reviewError, setReviewError] = useState<string | null>(null);
  const [reviewMissing, setReviewMissing] = useState(false);
  const [loading, setLoading] = useState(true);
  const [refreshBusy, setRefreshBusy] = useState(false);
  const [prepareBusy, setPrepareBusy] = useState(false);
  const [saveBusy, setSaveBusy] = useState(false);
  const [scriptBusy, setScriptBusy] = useState(false);
  const [selectedIds, setSelectedIds] = useState<string[]>([]);
  const [mergeTargetId, setMergeTargetId] = useState<string>("");
  const [searchQuery, setSearchQuery] = useState("");
  const [statusFilter, setStatusFilter] = useState<"all" | "suggested" | "confirmed" | "unknown">("all");
  const [dirty, setDirty] = useState(false);

  async function load() {
    if (!projectId) return;
    try {
      const nextProject = await api.getProject(projectId);
      setProject(nextProject);
      setError(null);
      try {
        const nextReview = await api.getCharacterReview(projectId);
        setReview(nextReview);
        setReviewMissing(false);
        setReviewError(null);
      } catch (err) {
        const message = err instanceof Error ? err.message : "Unable to load character review.";
        if (message.toLowerCase().includes("character suggestions have not been prepared")) {
          setReview(null);
          setReviewMissing(true);
          setReviewError(null);
        } else {
          setReview(null);
          setReviewMissing(false);
          setReviewError(message);
        }
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unable to load project.");
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    if (!projectId) return;
    load();
  }, [projectId]);

  useAdaptivePolling(load, {
    enabled: Boolean(projectId),
    active: Boolean(project?.active_jobs.length),
    activeMs: 8000,
    idleMs: 30000,
    hiddenMs: 120000,
    deps: [projectId]
  });

  const characterStage = project?.stage_states.character_review;
  const progressMeta = project ? getStageProgressMeta(project, "character_review") : null;
  const characterJobs = activeStageJobs(project, "character_review");
  const stageBusy = Boolean(characterStage && (characterStage.status === "running" || characterJobs.length > 0));
  const scriptJobs = activeStageJobs(project, "script_generation");
  const reviewCount = review?.identities.length ?? 0;
  const selectedCount = selectedIds.length;

  const sortedIdentities = useMemo(
    () =>
      [...(review?.identities ?? [])].sort((left, right) => {
        if (left.status !== right.status) {
          if (left.status === "confirmed") return -1;
          if (right.status === "confirmed") return 1;
        }
        return right.appearance_count - left.appearance_count;
      }),
    [review]
  );

  const visibleIdentities = useMemo(() => {
    const normalizedQuery = searchQuery.trim().toLowerCase();
    return sortedIdentities.filter((identity) => {
      if (statusFilter !== "all" && identity.status !== statusFilter) {
        return false;
      }
      if (!normalizedQuery) {
        return true;
      }
      const haystack = [
        identity.name,
        identity.suggested_name,
        identity.remembered_name,
        ...(identity.memory_matches ?? []),
        identity.role_hint,
        identity.notes,
        ...identity.pages.map((page) => `page ${page}`)
      ]
        .map((value) => String(value ?? "").toLowerCase())
        .join(" ");
      return haystack.includes(normalizedQuery);
    });
  }, [searchQuery, sortedIdentities, statusFilter]);

  useEffect(() => {
    const validSelectedIds = selectedIds.filter((reviewId) => review?.identities.some((identity) => identity.review_id === reviewId));
    if (validSelectedIds.length !== selectedIds.length) {
      setSelectedIds(validSelectedIds);
    }
    if (validSelectedIds.length < 2) {
      setMergeTargetId(validSelectedIds[0] ?? "");
      return;
    }
    if (!validSelectedIds.includes(mergeTargetId)) {
      setMergeTargetId(validSelectedIds[0]);
    }
  }, [mergeTargetId, review, selectedIds]);

  function updateIdentity(reviewId: string, patch: Partial<CharacterReviewIdentity>) {
    setReview((current) => {
      if (!current) return current;
      return {
        ...current,
        identities: current.identities.map((identity) =>
          identity.review_id === reviewId
            ? { ...identity, ...patch }
            : identity
        )
      };
    });
    setDirty(true);
  }

  async function saveReview(currentReview = review) {
    if (!project || !currentReview) return null;
    setSaveBusy(true);
    try {
      const saved = await api.updateCharacterReview(project.id, {
        protagonist_name: currentReview.protagonist_name ?? null,
        identities: currentReview.identities
      });
      setReview(saved);
      setDirty(false);
      setSelectedIds([]);
      setProject(await api.getProject(project.id));
      return saved;
    } finally {
      setSaveBusy(false);
    }
  }

  async function prepareCharacters(options: { forceRefresh?: boolean } = {}) {
    if (!project) return;
    if (options.forceRefresh && dirty) {
      const shouldContinue = window.confirm(
        "Rerunning character review will replace any unsaved character edits on this page. Continue?"
      );
      if (!shouldContinue) return;
    }
    setPrepareBusy(true);
    try {
      await api.queueStage(project.id, "character_review", options.forceRefresh ? { force_refresh: true } : {});
      if (options.forceRefresh) {
        setDirty(false);
        setSelectedIds([]);
      }
      await load();
    } finally {
      setPrepareBusy(false);
    }
  }

  async function generateScript() {
    if (!project) return;
    setScriptBusy(true);
    try {
      if (dirty && review) {
        const saved = await saveReview(review);
        if (saved) {
          setReview(saved);
        }
      }
      await api.queueStage(project.id, "script_generation");
      await load();
    } finally {
      setScriptBusy(false);
    }
  }

  if (loading) {
    return (
      <AppShell title="Characters" description="Loading the chapter cast and saved character review." projectId={projectId}>
        <div className="flex items-center gap-3 rounded-xl border border-white/[0.08] bg-white/[0.04] p-6 text-sm text-mutedForeground">
          <LoaderCircle className="h-4 w-4 animate-spin text-accent" />
          Loading character review...
        </div>
      </AppShell>
    );
  }

  if (!project || error) {
    return (
      <AppShell title="Characters unavailable" description="Panelia couldn't load the requested character review." projectId={projectId}>
        <Card>
          <CardTitle>Unable to load character review</CardTitle>
          <CardDescription className="mt-2">{error ?? "The project may have been moved or deleted."}</CardDescription>
        </Card>
      </AppShell>
    );
  }

  return (
    <AppShell
      title={project.name}
      description={`Characters · ${project.chapter_metadata.manga_title || "Sequential-art project"}`}
      projectId={projectId}
      breadcrumb={{ href: `/projects/${project.id}`, label: "Overview" }}
      views={buildProjectViews(projectId, "/editor")}
    >
      <div className="space-y-6">
        <Card>
          <div className="flex flex-col gap-4 lg:flex-row lg:items-start lg:justify-between">
            <div className="space-y-2">
              <div className="flex items-center gap-2 text-accent">
                <Users className="h-4 w-4" />
                <span className="text-sm font-medium">Cast review</span>
              </div>
              <CardTitle className="text-balance">Confirm names before script generation</CardTitle>
              <CardDescription className="max-w-3xl">
                Panelia grouped recurring characters from the dialogue and speaker pipeline. You can rename, merge, or mark groups as unknown here,
                and script generation will reuse your confirmed names instead of generic labels.
              </CardDescription>
              <div className="flex flex-wrap items-center gap-2 text-xs text-mutedForeground">
                <Badge>{reviewCount} groups</Badge>
                <Badge>{project.kept_panel_count} kept panels</Badge>
                <Badge>{project.chapter_metadata.language || "unknown source language"}</Badge>
                {review?.memory_names.length ? <Badge>{review.memory_names.length} remembered names</Badge> : null}
                <span>Project created {formatRelativeDate(project.created_at)}</span>
              </div>
            </div>

            <div className="flex flex-wrap items-center gap-2">
              <Button
                variant="ghost"
                size="sm"
                disabled={refreshBusy}
                onClick={async () => {
                  setRefreshBusy(true);
                  try {
                    await load();
                  } finally {
                    setRefreshBusy(false);
                  }
                }}
              >
                <RefreshCw className={`h-3.5 w-3.5 ${refreshBusy ? "animate-spin" : ""}`} />
                Refresh
              </Button>
              {review ? (
                <Button
                  variant="outline"
                  size="sm"
                  disabled={prepareBusy || stageBusy || saveBusy || scriptBusy}
                  onClick={() => void prepareCharacters({ forceRefresh: true })}
                >
                  {prepareBusy ? <LoaderCircle className="h-3.5 w-3.5 animate-spin" /> : <RotateCcw className="h-3.5 w-3.5" />}
                  Rerun character review
                </Button>
              ) : null}
              <Button
                variant="outline"
                size="sm"
                disabled={!review || selectedCount < 2 || !mergeTargetId || saveBusy || stageBusy}
                onClick={() => {
                  if (!review) return;
                  setReview(mergeIdentities(review, selectedIds, mergeTargetId));
                  setSelectedIds([mergeTargetId]);
                  setDirty(true);
                }}
              >
                Merge selected
              </Button>
              <Button
                variant="secondary"
                size="sm"
                disabled={!review || saveBusy || stageBusy}
                onClick={() => void saveReview()}
              >
                {saveBusy ? <LoaderCircle className="h-3.5 w-3.5 animate-spin" /> : <Save className="h-3.5 w-3.5" />}
                Save review
              </Button>
              <Button
                size="sm"
                disabled={prepareBusy || stageBusy || scriptBusy}
                onClick={() => {
                  if (reviewMissing) {
                    void prepareCharacters();
                    return;
                  }
                  void generateScript();
                }}
              >
                {prepareBusy || scriptBusy ? <LoaderCircle className="h-3.5 w-3.5 animate-spin" /> : <ArrowRight className="h-3.5 w-3.5" />}
                {reviewMissing ? "Prepare characters" : scriptJobs.length ? "Script queued" : "Save and generate script"}
              </Button>
            </div>
          </div>
        </Card>

        {stageBusy && characterStage && progressMeta ? (
          <Card>
            <CardTitle className="text-base">Preparing character suggestions</CardTitle>
            <CardDescription className="mt-1">{progressMeta.message || characterStage.message || "Running character review..."}</CardDescription>
            <div className="mt-4 space-y-2">
              <div className="flex items-center justify-between gap-3 text-[10px] uppercase tracking-[0.18em] text-mutedForeground">
                <span>{progressMeta.stateLabel ?? "Running"}</span>
                <span className="tabular-nums">{formatProgressPercent(progressMeta.progress)}</span>
              </div>
              <Progress value={progressMeta.progress} />
            </div>
          </Card>
        ) : null}

        {reviewError ? (
          <Card className="border border-fail/[0.25] bg-fail/[0.08]">
            <CardTitle className="text-base">Couldn&apos;t load the saved review</CardTitle>
            <CardDescription className="mt-1 text-fail">{reviewError}</CardDescription>
          </Card>
        ) : null}

        {reviewMissing && !stageBusy ? (
          <Card>
            <CardTitle className="text-base">Character suggestions haven&apos;t been prepared yet</CardTitle>
            <CardDescription className="mt-1">
              Save the panel review first, then Panelia can cluster recurring characters and build review cards from those suggestions.
            </CardDescription>
            <div className="mt-4 flex flex-wrap gap-2">
              <Button onClick={() => void prepareCharacters()} disabled={prepareBusy}>
                {prepareBusy ? <LoaderCircle className="h-3.5 w-3.5 animate-spin" /> : null}
                {prepareBusy ? "Preparing..." : "Prepare characters"}
              </Button>
            </div>
          </Card>
        ) : null}

        {review ? (
          <Card>
            <div className="grid gap-4 lg:grid-cols-[minmax(0,1fr)_260px]">
              <div>
                <CardTitle className="text-base">Protagonist hint</CardTitle>
                <CardDescription className="mt-1">
                  This name is reused when the script needs a primary reference and helps keep the narration consistent.
                </CardDescription>
                <Input
                  className="mt-3 max-w-xl"
                  value={review.protagonist_name ?? ""}
                  onChange={(event) => {
                    setReview((current) => (current ? { ...current, protagonist_name: event.target.value } : current));
                    setDirty(true);
                  }}
                  placeholder="Leave blank if the protagonist is still unknown"
                />
              </div>
              <div className="rounded-2xl border border-white/[0.06] bg-white/[0.03] p-4">
                <p className="text-xs uppercase tracking-[0.18em] text-mutedForeground">Remembered names</p>
                {review.memory_names.length ? (
                  <div className="mt-3 flex flex-wrap gap-2">
                    {review.memory_names.map((name) => (
                      <Badge key={name}>{name}</Badge>
                    ))}
                  </div>
                ) : (
                  <p className="mt-3 text-sm text-mutedForeground">This is the first saved character review for this series key.</p>
                )}
              </div>
            </div>
          </Card>
        ) : null}

        {review && review.identities.length > 0 ? (
          <Card>
            <div className="grid gap-4 lg:grid-cols-[minmax(0,1fr)_240px_260px]">
              <div>
                <p className="mb-1 text-xs uppercase tracking-[0.18em] text-mutedForeground">Search character groups</p>
                <Input
                  value={searchQuery}
                  onChange={(event) => setSearchQuery(event.target.value)}
                  placeholder="Search by confirmed name, suggested name, remembered name, role, note, or page"
                />
              </div>
              <div>
                <p className="mb-1 text-xs uppercase tracking-[0.18em] text-mutedForeground">Filter status</p>
                <select
                  value={statusFilter}
                  onChange={(event) => setStatusFilter(event.target.value as "all" | "suggested" | "confirmed" | "unknown")}
                  className="h-11 w-full rounded-2xl border border-white/[0.08] bg-white/[0.04] px-4 text-sm text-foreground focus:outline-none focus:border-accent/40 focus:ring-2 focus:ring-accent/30 transition-colors duration-fast"
                >
                  <option value="all">All groups</option>
                  <option value="suggested">Suggested only</option>
                  <option value="confirmed">Confirmed only</option>
                  <option value="unknown">Unknown only</option>
                </select>
              </div>
              <div>
                <p className="mb-1 text-xs uppercase tracking-[0.18em] text-mutedForeground">Merge target</p>
                <select
                  value={mergeTargetId}
                  onChange={(event) => setMergeTargetId(event.target.value)}
                  disabled={selectedCount < 2}
                  className="h-11 w-full rounded-2xl border border-white/[0.08] bg-white/[0.04] px-4 text-sm text-foreground focus:outline-none focus:border-accent/40 focus:ring-2 focus:ring-accent/30 transition-colors duration-fast disabled:opacity-50"
                >
                  <option value="">{selectedCount < 2 ? "Select at least 2 groups" : "Choose target group"}</option>
                  {selectedIds.map((reviewId) => {
                    const identity = review.identities.find((item) => item.review_id === reviewId);
                    if (!identity) return null;
                    const label = identity.name || identity.remembered_name || identity.suggested_name || identity.review_id;
                    return (
                      <option key={reviewId} value={reviewId}>
                        {label}
                      </option>
                    );
                  })}
                </select>
              </div>
            </div>
            <div className="mt-3 flex flex-wrap items-center gap-2 text-xs text-mutedForeground">
              <Badge>{visibleIdentities.length} visible</Badge>
              <Badge>{selectedCount} selected</Badge>
              {selectedCount >= 2 && mergeTargetId ? (
                <span>Merging selected groups into {review.identities.find((identity) => identity.review_id === mergeTargetId)?.name || review.identities.find((identity) => identity.review_id === mergeTargetId)?.suggested_name || "selected target"}.</span>
              ) : (
                <span>Select two or more groups to merge them into a chosen target.</span>
              )}
            </div>
          </Card>
        ) : null}

        {review && sortedIdentities.length === 0 ? (
          <Card>
            <CardTitle className="text-base">No recurring characters needed review</CardTitle>
            <CardDescription className="mt-1">
              This chapter didn&apos;t produce reusable character groups. You can move straight to script generation whenever you&apos;re ready.
            </CardDescription>
          </Card>
        ) : null}

        {review && sortedIdentities.length > 0 ? (
          <div className="grid gap-4 xl:grid-cols-2">
            {visibleIdentities.map((identity) => (
              <Card key={identity.review_id} className="p-5">
                <div className="flex items-start justify-between gap-3">
                  <label className="flex items-center gap-2 text-xs text-mutedForeground">
                    <input
                      type="checkbox"
                      className="h-4 w-4 rounded border border-white/20 bg-transparent"
                      checked={selectedIds.includes(identity.review_id)}
                      onChange={(event) => {
                        setSelectedIds((current) =>
                          event.target.checked
                            ? [...current, identity.review_id]
                            : current.filter((value) => value !== identity.review_id)
                        );
                      }}
                    />
                    Select for merge
                  </label>
                  <Badge>{identity.status}</Badge>
                </div>

                {identity.sample_images.length ? (
                  <div className="mt-4 grid grid-cols-2 gap-2 sm:grid-cols-4">
                    {identity.sample_images.map((sample) => (
                      <div key={sample.sample_id} className="overflow-hidden rounded-2xl border border-white/[0.08] bg-white/[0.04]">
                        {/* eslint-disable-next-line @next/next/no-img-element */}
                        <img
                          src={buildMediaUrl(sample.image_url ?? null, review.updated_at)}
                          alt={identity.name || identity.suggested_name || "Character sample"}
                          className="h-32 w-full object-cover"
                          loading="lazy"
                          decoding="async"
                        />
                        <div className="px-2 py-1 text-[11px] text-mutedForeground">
                          Page {sample.page ?? "?"}
                          {sample.panel ? ` • Panel ${sample.panel}` : ""}
                        </div>
                      </div>
                    ))}
                  </div>
                ) : null}

                <div className="mt-4 grid gap-3 sm:grid-cols-[minmax(0,1fr)_140px]">
                  <div>
                    <p className="mb-1 text-xs uppercase tracking-[0.18em] text-mutedForeground">Character name</p>
                    <Input
                      value={identity.name ?? ""}
                      onChange={(event) => updateIdentity(identity.review_id, { name: event.target.value })}
                      placeholder={identity.suggested_name || "Enter a confirmed name"}
                    />
                  </div>
                  <div>
                    <p className="mb-1 text-xs uppercase tracking-[0.18em] text-mutedForeground">Status</p>
                    <select
                      value={identity.status}
                      onChange={(event) => updateIdentity(identity.review_id, { status: event.target.value as CharacterReviewIdentity["status"] })}
                      className="h-11 w-full rounded-2xl border border-white/[0.08] bg-white/[0.04] px-4 text-sm text-foreground focus:outline-none focus:border-accent/40 focus:ring-2 focus:ring-accent/30 transition-colors duration-fast"
                    >
                      <option value="suggested">Suggested</option>
                      <option value="confirmed">Confirmed</option>
                      <option value="unknown">Unknown / background</option>
                    </select>
                  </div>
                </div>

                <div className="mt-3 flex flex-wrap gap-2 text-xs text-mutedForeground">
                  {identity.suggested_name ? <Badge>Suggested: {identity.suggested_name}</Badge> : null}
                  {identity.remembered_name ? <Badge>Remembered: {identity.remembered_name}</Badge> : null}
                  {(identity.memory_matches ?? []).filter((name) => name !== identity.remembered_name).slice(0, 3).map((name) => (
                    <Badge key={name}>Memory: {name}</Badge>
                  ))}
                  {identity.role_hint ? <Badge>{identity.role_hint}</Badge> : null}
                  <Badge>{identity.appearance_count} appearances</Badge>
                  <Badge>{identity.pages.length} pages</Badge>
                </div>

                <div className="mt-4 flex flex-wrap gap-2">
                  {identity.suggested_name && identity.name !== identity.suggested_name ? (
                    <Button
                      size="sm"
                      variant="outline"
                      onClick={() => updateIdentity(identity.review_id, { name: identity.suggested_name, status: "confirmed" })}
                    >
                      Use suggested
                    </Button>
                  ) : null}
                  {identity.remembered_name && identity.name !== identity.remembered_name ? (
                    <Button
                      size="sm"
                      variant="outline"
                      onClick={() => updateIdentity(identity.review_id, { name: identity.remembered_name, status: "confirmed" })}
                    >
                      Use remembered
                    </Button>
                  ) : null}
                  {(identity.memory_matches ?? [])
                    .filter((name) => name !== identity.remembered_name && name !== identity.name)
                    .slice(0, 2)
                    .map((name) => (
                      <Button
                        key={name}
                        size="sm"
                        variant="outline"
                        onClick={() => updateIdentity(identity.review_id, { name, status: "confirmed" })}
                      >
                        Use {name}
                      </Button>
                    ))}
                  <Button
                    size="sm"
                    variant="secondary"
                    disabled={!String(identity.name ?? "").trim()}
                    onClick={() => updateIdentity(identity.review_id, { status: "confirmed" })}
                  >
                    Confirm
                  </Button>
                  <Button
                    size="sm"
                    variant="ghost"
                    onClick={() => updateIdentity(identity.review_id, { status: "unknown" })}
                  >
                    Mark unknown
                  </Button>
                </div>

                <div className="mt-4">
                  <p className="mb-1 text-xs uppercase tracking-[0.18em] text-mutedForeground">Notes</p>
                  <Textarea
                    className="min-h-[90px]"
                    value={identity.notes ?? ""}
                    onChange={(event) => updateIdentity(identity.review_id, { notes: event.target.value })}
                    placeholder="Optional reminder for later chapters or manual script review."
                  />
                </div>
              </Card>
            ))}
          </div>
        ) : null}

        {dirty ? (
          <Card className="border border-amber-400/25 bg-amber-400/10">
            <CardTitle className="text-base">Unsaved character changes</CardTitle>
            <CardDescription className="mt-1 text-warn">
              Save the character review so script generation can reuse the latest confirmed names and merges.
            </CardDescription>
          </Card>
        ) : null}
      </div>
    </AppShell>
  );
}
