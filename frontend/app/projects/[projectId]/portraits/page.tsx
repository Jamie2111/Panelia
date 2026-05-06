"use client";

import { useEffect, useMemo, useState } from "react";
import { useParams } from "next/navigation";
import { LoaderCircle, RefreshCw, ScanFace } from "lucide-react";

import { AppShell } from "@/components/project/app-shell";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardDescription, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { api } from "@/lib/api";
import { CanonicalCharacter, ProjectSummary } from "@/lib/types";

function confidenceLabel(value?: number | null) {
  if (typeof value !== "number" || !Number.isFinite(value)) return "Unscored";
  return `${Math.round(value * 100)}%`;
}

export default function CharacterPortraitsPage() {
  const params = useParams<{ projectId: string }>();
  const projectId = Array.isArray(params.projectId) ? params.projectId[0] : params.projectId;
  const [project, setProject] = useState<ProjectSummary | null>(null);
  const [characters, setCharacters] = useState<CanonicalCharacter[]>([]);
  const [query, setQuery] = useState("");
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [queueing, setQueueing] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function load() {
    if (!projectId) return;
    try {
      const [nextProject, portraits] = await Promise.all([
        api.getProjectSummary(projectId),
        api.getCharacterPortraits(projectId)
      ]);
      setProject(nextProject);
      setCharacters(portraits.characters);
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unable to load character portraits.");
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  }

  useEffect(() => {
    load();
  }, [projectId]);

  const filteredCharacters = useMemo(() => {
    const normalized = query.trim().toLowerCase();
    if (!normalized) return characters;
    return characters.filter((character) =>
      [
        character.name,
        character.role,
        character.visual_description,
        ...(character.aliases ?? []),
        ...(character.portrait_pages ?? []).map((page) => `page ${page}`)
      ].join(" ").toLowerCase().includes(normalized)
    );
  }, [characters, query]);

  const portraitStage = project?.stage_states.character_portrait;
  const portraitBusy = portraitStage?.status === "running" || project?.active_jobs.some((job) => job.stage === "character_portrait");

  async function queuePortraits() {
    if (!projectId) return;
    setQueueing(true);
    try {
      await api.queueStage(projectId, "character_portrait");
      await load();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unable to queue character portraits.");
    } finally {
      setQueueing(false);
    }
  }

  return (
    <AppShell
      title="Character Portraits"
      description="Canonical character records gathered before panel vision and script generation."
      projectId={projectId}
    >
      {loading ? (
        <div className="flex items-center gap-2 text-sm text-mutedForeground">
          <LoaderCircle className="h-4 w-4 animate-spin" /> Loading portraits...
        </div>
      ) : error ? (
        <Card className="border-red-500/30 bg-red-500/10 p-6 text-sm text-red-100">{error}</Card>
      ) : (
        <div className="space-y-5">
          <div className="flex flex-col gap-3 lg:flex-row lg:items-center lg:justify-between">
            <div className="flex items-center gap-3">
              <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-white/8">
                <ScanFace className="h-4 w-4 text-accent" />
              </div>
              <div>
                <CardTitle>{characters.length} canonical characters</CardTitle>
                <CardDescription>
                  {portraitStage?.message || "Portrait records appear here after the character portrait stage runs."}
                </CardDescription>
              </div>
            </div>
            <div className="flex flex-wrap gap-2">
              <Button
                variant="secondary"
                onClick={() => {
                  setRefreshing(true);
                  load();
                }}
                disabled={refreshing}
              >
                {refreshing ? <LoaderCircle className="h-4 w-4 animate-spin" /> : <RefreshCw className="h-4 w-4" />}
                Refresh
              </Button>
              <Button onClick={queuePortraits} disabled={Boolean(queueing || portraitBusy)}>
                {queueing || portraitBusy ? <LoaderCircle className="h-4 w-4 animate-spin" /> : <ScanFace className="h-4 w-4" />}
                Run Portraits
              </Button>
            </div>
          </div>

          <Input
            value={query}
            onChange={(event) => setQuery(event.target.value)}
            placeholder="Search characters, roles, or pages"
            className="max-w-md"
          />

          {filteredCharacters.length ? (
            <div className="grid gap-3 md:grid-cols-2 xl:grid-cols-3">
              {filteredCharacters.map((character) => (
                <Card key={character.stable_id} className="p-4">
                  <div className="flex items-start justify-between gap-3">
                    <div className="min-w-0">
                      <CardTitle className="truncate text-base">{character.name || character.stable_id}</CardTitle>
                      <CardDescription className="mt-1 truncate">{character.role || "supporting"}</CardDescription>
                    </div>
                    <Badge>{confidenceLabel(character.confidence)}</Badge>
                  </div>
                  {character.visual_description ? (
                    <p className="mt-3 line-clamp-3 text-sm text-mutedForeground">{character.visual_description}</p>
                  ) : null}
                  <div className="mt-4 flex flex-wrap gap-2 text-xs text-mutedForeground">
                    {(character.portrait_pages ?? []).slice(0, 6).map((page) => (
                      <span key={page} className="rounded-md bg-white/8 px-2 py-1">Page {page}</span>
                    ))}
                    {(character.aliases ?? []).slice(0, 4).map((alias) => (
                      <span key={alias} className="rounded-md bg-white/8 px-2 py-1">{alias}</span>
                    ))}
                  </div>
                </Card>
              ))}
            </div>
          ) : (
            <Card className="p-6">
              <CardTitle>No portrait records yet</CardTitle>
              <CardDescription className="mt-2">
                Run character portraits after character review to populate canonical character records.
              </CardDescription>
            </Card>
          )}
        </div>
      )}
    </AppShell>
  );
}
