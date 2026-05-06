"use client";

import { useEffect, useMemo, useState } from "react";
import { useParams } from "next/navigation";
import { BookOpen, LoaderCircle, RefreshCw } from "lucide-react";

import { AppShell } from "@/components/project/app-shell";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardDescription, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { api } from "@/lib/api";
import { CharacterDictionaryEntry, ProjectSummary } from "@/lib/types";

export default function CharacterDictionaryPage() {
  const params = useParams<{ projectId: string }>();
  const projectId = Array.isArray(params.projectId) ? params.projectId[0] : params.projectId;
  const [project, setProject] = useState<ProjectSummary | null>(null);
  const [entries, setEntries] = useState<CharacterDictionaryEntry[]>([]);
  const [query, setQuery] = useState("");
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function load() {
    if (!projectId) return;
    try {
      const [nextProject, dictionary] = await Promise.all([
        api.getProjectSummary(projectId),
        api.getCharacterDictionary(projectId)
      ]);
      setProject(nextProject);
      setEntries(dictionary.entries);
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unable to load character dictionary.");
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  }

  useEffect(() => {
    load();
  }, [projectId]);

  const filteredEntries = useMemo(() => {
    const normalized = query.trim().toLowerCase();
    if (!normalized) return entries;
    return entries.filter((entry) =>
      `${entry.name} ${entry.key}`.toLowerCase().includes(normalized)
    );
  }, [entries, query]);

  return (
    <AppShell
      title="Character Dictionary"
      description="Names the script and narration stages can reuse consistently."
      projectId={projectId}
    >
      {loading ? (
        <div className="flex items-center gap-2 text-sm text-mutedForeground">
          <LoaderCircle className="h-4 w-4 animate-spin" /> Loading dictionary...
        </div>
      ) : error ? (
        <Card className="border-red-500/30 bg-red-500/10 p-6 text-sm text-red-100">{error}</Card>
      ) : (
        <div className="space-y-5">
          <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
            <div className="flex items-center gap-3">
              <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-white/8">
                <BookOpen className="h-4 w-4 text-accent" />
              </div>
              <div>
                <CardTitle>{entries.length} entries</CardTitle>
                <CardDescription>
                  {project?.stage_states.script_generation?.status === "completed"
                    ? "Script generation has used the latest available dictionary."
                    : "The dictionary fills in after character review and script preparation."}
                </CardDescription>
              </div>
            </div>
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
          </div>

          <Input
            value={query}
            onChange={(event) => setQuery(event.target.value)}
            placeholder="Search names"
            className="max-w-md"
          />

          {filteredEntries.length ? (
            <div className="grid gap-3 md:grid-cols-2 xl:grid-cols-3">
              {filteredEntries.map((entry) => (
                <Card key={entry.key} className="p-4">
                  <div className="flex items-start justify-between gap-3">
                    <div className="min-w-0">
                      <CardTitle className="truncate text-base">{entry.name}</CardTitle>
                      <CardDescription className="mt-1 truncate">{entry.key}</CardDescription>
                    </div>
                    <Badge>Known</Badge>
                  </div>
                </Card>
              ))}
            </div>
          ) : (
            <Card className="p-6">
              <CardTitle>No dictionary entries yet</CardTitle>
              <CardDescription className="mt-2">
                Save character review, then run the script preparation stages to populate this view.
              </CardDescription>
            </Card>
          )}
        </div>
      )}
    </AppShell>
  );
}
