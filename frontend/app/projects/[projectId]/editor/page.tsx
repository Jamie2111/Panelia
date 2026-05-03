"use client";

import type { Route } from "next";
import { useParams, useRouter } from "next/navigation";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { ArrowRight, LoaderCircle } from "lucide-react";

import { EditorShell } from "@/components/editor/editor-shell";
import { EditorToolbar } from "@/components/editor/editor-toolbar";
import { PageFilmstrip } from "@/components/editor/page-filmstrip";
import { PanelStrip } from "@/components/editor/panel-strip";
import { PanelInspector } from "@/components/editor/panel-inspector";
import { BatchActionBar } from "@/components/editor/batch-action-bar";
import { PanelContextMenu } from "@/components/editor/panel-context-menu";
import { PanelCanvas } from "@/components/project/panel-canvas";
import { Button } from "@/components/ui/button";
import { Card, CardDescription, CardTitle } from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import { api } from "@/lib/api";
import { useAdaptivePolling } from "@/lib/use-adaptive-polling";
import { useEditorHotkeys } from "@/hooks/use-editor-hotkeys";
import { useAutoSave } from "@/hooks/use-auto-save";
import { usePanelEditorStore } from "@/store/panel-editor-store";

export default function PanelEditorPage() {
  const params = useParams<{ projectId: string }>();
  const router = useRouter();
  const projectId = Array.isArray(params.projectId) ? params.projectId[0] : params.projectId;
  const [error, setError] = useState<string | null>(null);
  const [currentPageBounds, setCurrentPageBounds] = useState({ width: 1, height: 1 });
  const [contextMenu, setContextMenu] = useState<{ panelId: string; x: number; y: number } | null>(null);
  const [continuingToNarration, setContinuingToNarration] = useState(false);

  const store = usePanelEditorStore();
  const { project, panels, selectedIds, selectedPage, drawMode, flaggedOnlyMode } = store;

  // ── Load project ───────────────────────────────────────
  const loadProject = useCallback(
    async (preserveLocalPanels = true) => {
      if (!projectId) return;
      try {
        const nextProject = await api.getProject(projectId);
        store.setProject(nextProject, {
          preserveLocalPanels: preserveLocalPanels && autoSaveRef.current,
          preserveSelection: true
        });
        setError(null);
      } catch (err) {
        setError(err instanceof Error ? err.message : "Unable to load project.");
      }
    },
    [projectId, store]
  );

  useEffect(() => {
    if (!projectId) return;
    void loadProject(false);
  }, [projectId]);

  // ── Derived data ───────────────────────────────────────
  const pageCount = project?.page_count ?? 0;

  const flaggedPanelIds = useMemo(
    () => new Set(panels.filter((p) => p.auto_skipped || (p.review_flags?.length ?? 0) > 0).map((p) => p.id)),
    [panels]
  );
  const flaggedPages = useMemo(
    () => new Set(panels.filter((p) => flaggedPanelIds.has(p.id)).map((p) => p.page)),
    [flaggedPanelIds, panels]
  );
  const flaggedCount = flaggedPanelIds.size;

  const currentPanels = useMemo(
    () =>
      panels
        .filter((p) => p.page === selectedPage && (!flaggedOnlyMode || flaggedPanelIds.has(p.id)))
        .sort((a, b) => a.order - b.order),
    [flaggedOnlyMode, flaggedPanelIds, panels, selectedPage]
  );

  const currentImage = project
    ? `/media/projects/${project.id}/pages/${selectedPage.toString().padStart(4, "0")}.png`
    : null;

  const selectedPanel = panels.find((p) => p.id === selectedIds[0]);
  const hasAnyExtractedDialogue = useMemo(
    () => panels.some((panel) => Boolean((panel.ocr_text ?? "").trim())),
    [panels]
  );

  // Clear selection when navigating to a page where selected panels aren't visible
  useEffect(() => {
    const visibleIds = new Set(currentPanels.map((p) => p.id));
    if (selectedIds.some((id) => !visibleIds.has(id))) {
      store.clearSelection();
    }
  }, [currentPanels, selectedIds, store]);

  // ── Panel detection polling ────────────────────────────
  const panelDetectionBusy =
    project?.stage_states.panel_detection.status === "ready" ||
    project?.stage_states.panel_detection.status === "running" ||
    project?.active_jobs.some((j) => j.stage === "panel_detection") ||
    false;
  const panelDetectionProgress = Math.max(0, Math.min(100, Math.round(project?.stage_states.panel_detection.progress ?? 0)));

  // ── Auto-save ──────────────────────────────────────────
  const persistPanels = useCallback(async () => {
    if (!project) return;
    const updated = await api.updatePanels(project.id, panels);
    store.setProject(updated, {
      preserveLocalPanels: true,
      preserveSelection: true
    });
    return updated;
  }, [project, panels, store]);

  const { status: saveStatus, hasChanges, saveNow } = useAutoSave({
    data: panels,
    serverData: project?.panels,
    save: persistPanels,
    delayMs: 3000
  });

  // Ref so polling callback can check without re-subscribing
  const autoSaveRef = useRef(hasChanges);
  useEffect(() => { autoSaveRef.current = hasChanges; }, [hasChanges]);

  useAdaptivePolling(
    async () => {
      if (hasChanges) return;
      await loadProject(true);
    },
    {
      enabled: Boolean(projectId),
      active: panelDetectionBusy || (!panels.length && Boolean(project)),
      activeMs: 7000,
      idleMs: 30000,
      hiddenMs: 120000,
      deps: [projectId, hasChanges, panels.length]
    }
  );

  // ── Keyboard shortcuts ─────────────────────────────────
  useEditorHotkeys({
    pageCount,
    onSave: () => {
      void handleManualSave();
    },
    currentPanels
  });

  // ── OCR text handler ───────────────────────────────────
  function handleDetectedTextChange(panelId: string, value: string) {
    store.updatePanel(panelId, {
      ocr_text: value,
      text_detected: Boolean(value.trim()),
      manual_ocr_text: true,
      manual_keep: true,
      auto_skipped: false,
      skip_reason: value.trim() ? null : "No usable OCR text. Leave blank to skip."
    });
  }

  async function handleManualSave() {
    if (!project) return false;
    try {
      await saveNow();
      setError(null);
      return true;
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unable to save panel edits.");
      return false;
    }
  }

  // ── Continue to narration ──────────────────────────────
  async function handleContinueToNarration() {
    if (!project) return;
    setContinuingToNarration(true);
    try {
      const destination = `/projects/${project.id}/narration` as Route;
      if (hasChanges) {
        const saveResult = await handleManualSave();
        if (saveResult === false) return;
      }
      router.push(destination);
    } finally {
      setContinuingToNarration(false);
    }
  }

  // ── Batch helpers ──────────────────────────────────────
  const selectedPanelsOnSamePage = useMemo(() => {
    if (selectedIds.length < 2) return false;
    const pages = new Set(panels.filter((p) => selectedIds.includes(p.id)).map((p) => p.page));
    return pages.size === 1;
  }, [selectedIds, panels]);

  // ── Render ─────────────────────────────────────────────
  return (
    <EditorShell projectId={projectId}>
      {/* Toolbar */}
      <EditorToolbar
        projectName={project?.chapter_metadata?.manga_title || project?.name || ""}
        projectId={projectId}
        selectedPage={selectedPage}
        pageCount={pageCount}
        onPageJump={(p) => store.jumpToPage(p, pageCount)}
        saveStatus={saveStatus}
        onSaveNow={() => {
          void handleManualSave();
        }}
        drawMode={drawMode}
        onToggleDrawMode={() => store.toggleDrawMode()}
        flaggedOnlyMode={flaggedOnlyMode}
        onToggleFlaggedOnly={() => store.toggleFlaggedOnlyMode()}
        flaggedCount={flaggedCount}
        canUndo={store.canUndo}
        canRedo={store.canRedo}
        onUndo={store.undo}
        onRedo={store.redo}
        panels={panels}
        flaggedPanelIds={flaggedPanelIds}
      />

      {error && (
        <Card className="mx-4 mt-2 border-red-500/20 bg-red-500/10">
          <CardDescription className="text-red-200">{error}</CardDescription>
        </Card>
      )}

      {/* Main content area */}
      <div className="flex min-h-0 flex-1 overflow-hidden">
        {/* Left: Page filmstrip */}
        <PageFilmstrip
          projectId={projectId}
          pageCount={pageCount}
          selectedPage={selectedPage}
          onSelectPage={(p) => store.selectPage(p)}
          panels={panels}
          flaggedPages={flaggedPages}
        />

        {/* Center: Canvas */}
        <div className="flex min-w-0 flex-1 flex-col overflow-hidden bg-[linear-gradient(180deg,rgba(255,255,255,0.02),rgba(255,255,255,0))]">
          {panelDetectionBusy && (
            <Card className="mx-4 mt-3 border-white/10 bg-white/[0.04]">
              <CardTitle className="flex items-center gap-2 text-sm">
                <LoaderCircle className="h-4 w-4 animate-spin text-accent" />
                Detecting panels
              </CardTitle>
              <CardDescription className="mt-1 text-xs">
                {project?.stage_states.panel_detection.message || "Panel detection in progress..."}
              </CardDescription>
              <Progress value={panelDetectionProgress} className="mt-2" />
            </Card>
          )}

          {!panelDetectionBusy && !hasAnyExtractedDialogue ? (
            <Card className="mx-4 mt-3 border-white/10 bg-[linear-gradient(180deg,rgba(255,255,255,0.05),rgba(255,255,255,0.02))]">
              <CardTitle className="text-sm">Dialogue isn&apos;t missing, it just may not exist yet</CardTitle>
              <CardDescription className="mt-1 text-xs">
                This editor can save manual dialogue overrides, but auto-extracted dialogue usually appears after the OCR/script stage runs. If a panel truly has no readable text, it can also stay blank on purpose.
              </CardDescription>
            </Card>
          ) : null}

          <div className="min-h-0 flex-1 px-2 pb-2 pt-2 md:px-3">
            <PanelCanvas
              imageUrl={currentImage}
              panels={currentPanels}
              selectedIds={selectedIds}
              drawMode={drawMode}
              onSelect={(id, additive) => store.selectPanel(id, additive)}
              onMove={(id, dx, dy) => store.movePanel(id, dx, dy, currentPageBounds)}
              onResize={(id, handle, dx, dy) => store.resizePanel(id, handle, dx, dy, currentPageBounds)}
              onAddPanel={(box) => store.addPanel(selectedPage, box, currentPageBounds)}
              onDrawComplete={() => store.setDrawMode(false)}
              onNaturalSizeChange={(size) => setCurrentPageBounds(size)}
              onPanelContextMenu={(panelId, x, y) => setContextMenu({ panelId, x, y })}
              onDetectedTextChange={handleDetectedTextChange}
            />
          </div>
        </div>

        {/* Right: Inspector */}
        <div className="hidden w-[300px] shrink-0 border-l border-white/10 bg-[linear-gradient(180deg,rgba(255,255,255,0.03),rgba(255,255,255,0.015))] lg:block">
          <PanelInspector
            panel={selectedPanel}
            selectedCount={selectedIds.length}
            onToggleKeep={(id) => store.toggleKeep(id)}
            onDelete={() => store.deleteSelected()}
            onMoveOrder={(id, dir) => store.moveOrder(id, dir)}
            onSplit={(id, axis) => store.splitPanel(id, axis)}
            onMerge={() => store.mergeSelected()}
            onUpdatePanel={(id, updates) => store.updatePanel(id, updates)}
            onDetectedTextChange={handleDetectedTextChange}
          />
        </div>
      </div>

      {/* Context menu */}
      {contextMenu && (() => {
        const panel = panels.find((p) => p.id === contextMenu.panelId);
        if (!panel) return null;
        return (
          <PanelContextMenu
            x={contextMenu.x}
            y={contextMenu.y}
            panelId={contextMenu.panelId}
            isKept={panel.keep}
            selectedCount={selectedIds.length}
            onClose={() => setContextMenu(null)}
            onToggleKeep={(id) => store.toggleKeep(id)}
            onSplitH={(id) => store.splitPanel(id, "horizontal")}
            onSplitV={(id) => store.splitPanel(id, "vertical")}
            onMerge={() => store.mergeSelected()}
            onDelete={() => store.deleteSelected()}
          />
        );
      })()}

      {/* Bottom: Panel strip + batch bar */}
      <div className="flex shrink-0 flex-col border-t border-white/10 bg-[linear-gradient(180deg,rgba(255,255,255,0.04),rgba(255,255,255,0.02))]">
        {/* Batch action bar (only shows when 2+ selected) */}
        <BatchActionBar
          selectedCount={selectedIds.length}
          allOnSamePage={selectedPanelsOnSamePage}
          onKeepAll={() => store.batchSetKeep(selectedIds, true)}
          onRemoveAll={() => store.batchSetKeep(selectedIds, false)}
          onDeleteAll={() => store.batchDelete(selectedIds)}
          onMerge={() => store.mergeSelected()}
          onSetDuration={(sec) => store.batchSetDuration(selectedIds, sec)}
        />

        {/* Panel strip */}
        <div className="flex h-[126px] items-center px-1">
          <PanelStrip
            panels={currentPanels}
            selectedIds={selectedIds}
            imageUrl={currentImage ?? ""}
            naturalWidth={currentPageBounds.width}
            naturalHeight={currentPageBounds.height}
            onSelect={(id, additive) => store.selectPanel(id, additive)}
          />

          {/* Continue button */}
          <div className="flex shrink-0 items-center gap-2 border-l border-white/10 px-4">
            <Button
              size="sm"
              variant="secondary"
              className="rounded-xl"
              onClick={handleManualSave}
              disabled={!project || !hasChanges}
            >
              Save
            </Button>
            <Button size="sm" className="rounded-xl" onClick={handleContinueToNarration} disabled={!project || continuingToNarration}>
              {continuingToNarration
                ? <LoaderCircle className="h-3.5 w-3.5 animate-spin" />
                : null}
              {continuingToNarration ? "Saving..." : hasChanges ? "Save & continue" : "Narration"}
              {!continuingToNarration ? <ArrowRight className="ml-1 h-3.5 w-3.5" /> : null}
            </Button>
          </div>
        </div>
      </div>
    </EditorShell>
  );
}
