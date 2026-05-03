"use client";

import Link from "next/link";
import { useEffect, useMemo, useState } from "react";
import { useRouter } from "next/navigation";
import { LoaderCircle } from "lucide-react";

import { AppShell } from "@/components/project/app-shell";
import { Button } from "@/components/ui/button";
import { Card, CardDescription, CardTitle } from "@/components/ui/card";
import { api } from "@/lib/api";
import { ProjectSummary } from "@/lib/types";
import { formatRelativeDate } from "@/lib/utils";

function formatCanvas(width: number | string | null | undefined, height: number | string | null | undefined) {
  const safeWidth = Number(width);
  const safeHeight = Number(height);
  const finalWidth = Number.isFinite(safeWidth) && safeWidth > 0 ? Math.round(safeWidth) : 1920;
  const finalHeight = Number.isFinite(safeHeight) && safeHeight > 0 ? Math.round(safeHeight) : 1080;
  return `${finalWidth}×${finalHeight}`;
}

export default function ExportsPage() {
  const router = useRouter();
  const [projects, setProjects] = useState<ProjectSummary[]>([]);
  const [duplicatingProjectId, setDuplicatingProjectId] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    api.listProjects()
      .then((payload) => {
        if (cancelled) return;
        setProjects(payload);
        setError(null);
      })
      .catch((err) => {
        if (cancelled) return;
        setError(err instanceof Error ? err.message : "Unable to load exported videos.");
      });
    return () => {
      cancelled = true;
    };
  }, []);

  const exports = useMemo(
    () =>
      projects
        .filter((project) => project.latest_video)
        .map((project) => ({ project, video: project.latest_video! })),
    [projects]
  );

  async function duplicateExport(projectId: string, name?: string) {
    setDuplicatingProjectId(projectId);
    try {
      const duplicated = await api.duplicateProject(projectId, {
        name: name ? `${name} Alternate Cut` : undefined,
        copy_all_videos: true
      });
      router.push(`/projects/${duplicated.id}/preview`);
    } finally {
      setDuplicatingProjectId(null);
    }
  }

  return (
    <AppShell
      title="Exports library"
      description="Review the latest rendered output across projects, jump back into narration or panel editing, and keep the finishing pass focused on the projects that already have exports."
    >
      {error ? (
        <Card className="mb-6 border-red-500/20 bg-red-500/10">
          <CardDescription className="text-red-200">{error}</CardDescription>
        </Card>
      ) : null}
      {exports.length ? (
        <div className="grid gap-5">
          {exports.map(({ project, video }) => (
            <Card key={project.id}>
              <div className="flex flex-col gap-4 lg:flex-row lg:items-center lg:justify-between">
                <div>
                  <CardTitle>{project.name}</CardTitle>
                  <CardDescription className="mt-2">
                    {video.name} • {formatCanvas(video.width, video.height)} • {formatRelativeDate(video.created_at)}
                  </CardDescription>
                </div>
                <div className="flex flex-wrap gap-3">
                  <Link href={`/projects/${project.id}/preview`}>
                    <Button>Open preview</Button>
                  </Link>
                  <Link href={`/projects/${project.id}/editor`}>
                    <Button variant="secondary">Edit panels</Button>
                  </Link>
                  <Button variant="outline" onClick={() => duplicateExport(project.id, project.name)} disabled={duplicatingProjectId !== null}>
                    {duplicatingProjectId === project.id ? <LoaderCircle className="h-4 w-4 animate-spin" /> : null}
                    Duplicate for re-edit
                  </Button>
                </div>
              </div>
            </Card>
          ))}
        </div>
      ) : (
        <Card>
          <CardTitle>No exports yet</CardTitle>
          <CardDescription className="mt-2">Render a project first and this page becomes your quick-access export shelf.</CardDescription>
        </Card>
      )}
    </AppShell>
  );
}
