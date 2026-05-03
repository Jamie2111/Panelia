"use client";

import { create } from "zustand";

import { PanelBox, ProjectDetail } from "@/lib/types";

type Handle = "nw" | "ne" | "sw" | "se";
type PageBounds = { width: number; height: number };

const MAX_HISTORY = 50;

interface EditorState {
  project: ProjectDetail | null;
  panels: PanelBox[];
  selectedPage: number;
  selectedIds: string[];
  drawMode: boolean;
  flaggedOnlyMode: boolean;

  // Undo / redo stacks
  _history: PanelBox[][];
  _future: PanelBox[][];
  canUndo: boolean;
  canRedo: boolean;

  // Actions — project
  setProject: (
    project: ProjectDetail,
    options?: {
      preserveLocalPanels?: boolean;
      preserveSelection?: boolean;
    }
  ) => void;

  // Actions — navigation
  selectPage: (page: number) => void;
  nextPage: (pageCount: number) => void;
  prevPage: () => void;
  jumpToPage: (page: number, pageCount: number) => void;
  selectPanel: (id: string, additive?: boolean) => void;
  selectPanels: (ids: string[]) => void;
  clearSelection: () => void;
  selectNextPanel: () => void;
  selectPrevPanel: () => void;
  nextFlaggedPanel: (pageCount: number) => void;

  // Actions — modes
  setDrawMode: (on: boolean) => void;
  toggleDrawMode: () => void;
  setFlaggedOnlyMode: (on: boolean) => void;
  toggleFlaggedOnlyMode: () => void;

  // Actions — panel mutations (all push undo history)
  setPageKeep: (page: number, keep: boolean) => void;
  setAllPagesKeep: (keep: boolean) => void;
  updatePanel: (id: string, updates: Partial<PanelBox>) => void;
  movePanel: (id: string, dx: number, dy: number, bounds?: PageBounds) => void;
  resizePanel: (id: string, handle: Handle, dx: number, dy: number, bounds?: PageBounds) => void;
  addPanel: (page: number, box: Pick<PanelBox, "x" | "y" | "width" | "height">, bounds?: PageBounds) => void;
  deleteSelected: () => void;
  toggleKeep: (id: string) => void;
  moveOrder: (id: string, direction: "up" | "down") => void;
  splitPanel: (id: string, axis: "horizontal" | "vertical") => void;
  mergeSelected: () => void;

  // Actions — batch
  batchSetKeep: (ids: string[], keep: boolean) => void;
  batchSetDuration: (ids: string[], seconds: number) => void;
  batchDelete: (ids: string[]) => void;

  // Actions — undo / redo
  undo: () => void;
  redo: () => void;
}

// ── helpers ────────────────────────────────────────────────

function comparePanelsByReadingOrder(left: PanelBox, right: PanelBox) {
  return (
    left.page - right.page ||
    left.y - right.y ||
    left.x - right.x ||
    left.height - right.height ||
    left.width - right.width ||
    left.order - right.order ||
    left.panel - right.panel ||
    left.id.localeCompare(right.id)
  );
}

function reindex(panels: PanelBox[]) {
  const pageCounters = new Map<number, number>();
  return [...panels]
    .sort(comparePanelsByReadingOrder)
    .map((panel, index) => {
      const nextPanelNumber = (pageCounters.get(panel.page) ?? 0) + 1;
      pageCounters.set(panel.page, nextPanelNumber);
      return { ...panel, order: index + 1, panel: nextPanelNumber };
    });
}

function clampPanelToBounds(panel: PanelBox, bounds?: PageBounds, minSize = 40): PanelBox {
  let width = Math.max(minSize, Math.round(panel.width));
  let height = Math.max(minSize, Math.round(panel.height));
  let x = Math.max(0, Math.round(panel.x));
  let y = Math.max(0, Math.round(panel.y));

  if (!bounds || bounds.width <= 0 || bounds.height <= 0) {
    return { ...panel, x, y, width, height };
  }

  width = Math.min(width, Math.max(bounds.width, minSize));
  height = Math.min(height, Math.max(bounds.height, minSize));
  x = Math.min(x, Math.max(bounds.width - width, 0));
  y = Math.min(y, Math.max(bounds.height - height, 0));

  return { ...panel, x, y, width, height };
}

/** Push current panels onto undo stack, clear redo. */
function pushHistory(state: EditorState): Pick<EditorState, "_history" | "_future" | "canUndo" | "canRedo"> {
  const next = [...state._history, structuredClone(state.panels)];
  if (next.length > MAX_HISTORY) next.shift();
  return { _history: next, _future: [], canUndo: true, canRedo: false };
}

function flaggedIds(panels: PanelBox[]): Set<string> {
  return new Set(
    panels
      .filter((p) => p.auto_skipped || (p.review_flags?.length ?? 0) > 0)
      .map((p) => p.id)
  );
}

// ── store ──────────────────────────────────────────────────

export const usePanelEditorStore = create<EditorState>((set, get) => ({
  project: null,
  panels: [],
  selectedPage: 1,
  selectedIds: [],
  drawMode: false,
  flaggedOnlyMode: false,
  _history: [],
  _future: [],
  canUndo: false,
  canRedo: false,

  // ── project ────────────────────────────────────────────
  setProject: (project, options) =>
    set((state) => {
      const preserveLocalPanels = options?.preserveLocalPanels ?? false;
      const preserveSelection = options?.preserveSelection ?? preserveLocalPanels;
      const normalizedPanels = reindex(project.panels);
      const nextPanels = preserveLocalPanels ? state.panels : normalizedPanels;
      const knownIds = new Set(nextPanels.map((p) => p.id));
      const nextSelectedIds = preserveSelection ? state.selectedIds.filter((id) => knownIds.has(id)) : [];
      const projectPageCount = Math.max(project.page_count || normalizedPanels[0]?.page || 1, 1);
      const preferredPage = preserveSelection ? state.selectedPage : normalizedPanels[0]?.page ?? state.selectedPage ?? 1;
      const nextSelectedPage = Math.min(Math.max(preferredPage, 1), projectPageCount);

      return {
        project: preserveLocalPanels ? project : { ...project, panels: normalizedPanels },
        panels: nextPanels,
        selectedPage: nextSelectedPage,
        selectedIds: nextSelectedIds,
        // Reset history on fresh load (not on preserveLocal polling)
        ...(!preserveLocalPanels ? { _history: [], _future: [], canUndo: false, canRedo: false } : {})
      };
    }),

  // ── navigation ─────────────────────────────────────────
  selectPage: (page) => set({ selectedPage: page, selectedIds: [] }),

  nextPage: (pageCount) =>
    set((state) => {
      const next = Math.min(state.selectedPage + 1, pageCount);
      return next !== state.selectedPage ? { selectedPage: next, selectedIds: [] } : {};
    }),

  prevPage: () =>
    set((state) => {
      const prev = Math.max(state.selectedPage - 1, 1);
      return prev !== state.selectedPage ? { selectedPage: prev, selectedIds: [] } : {};
    }),

  jumpToPage: (page, pageCount) =>
    set(() => ({
      selectedPage: Math.max(1, Math.min(page, pageCount)),
      selectedIds: []
    })),

  selectPanel: (id, additive = false) =>
    set((state) => ({
      selectedIds: additive ? [...new Set([...state.selectedIds, id])] : [id]
    })),

  selectPanels: (ids) => set({ selectedIds: [...new Set(ids)] }),
  clearSelection: () => set({ selectedIds: [] }),

  selectNextPanel: () =>
    set((state) => {
      const pagePanels = state.panels
        .filter((p) => p.page === state.selectedPage)
        .sort((a, b) => a.order - b.order);
      if (!pagePanels.length) return {};
      const currentIdx = pagePanels.findIndex((p) => p.id === state.selectedIds[0]);
      const nextIdx = currentIdx < 0 ? 0 : Math.min(currentIdx + 1, pagePanels.length - 1);
      return { selectedIds: [pagePanels[nextIdx].id] };
    }),

  selectPrevPanel: () =>
    set((state) => {
      const pagePanels = state.panels
        .filter((p) => p.page === state.selectedPage)
        .sort((a, b) => a.order - b.order);
      if (!pagePanels.length) return {};
      const currentIdx = pagePanels.findIndex((p) => p.id === state.selectedIds[0]);
      const prevIdx = currentIdx < 0 ? pagePanels.length - 1 : Math.max(currentIdx - 1, 0);
      return { selectedIds: [pagePanels[prevIdx].id] };
    }),

  nextFlaggedPanel: (pageCount) =>
    set((state) => {
      const fIds = flaggedIds(state.panels);
      if (!fIds.size) return {};
      const sorted = state.panels
        .filter((p) => fIds.has(p.id))
        .sort((a, b) => a.order - b.order);
      const currentOrder = state.panels.find((p) => p.id === state.selectedIds[0])?.order ?? 0;
      const next = sorted.find((p) => p.order > currentOrder) ?? sorted[0];
      if (!next) return {};
      return {
        selectedPage: next.page,
        selectedIds: [next.id]
      };
    }),

  // ── modes ──────────────────────────────────────────────
  setDrawMode: (on) => set({ drawMode: on }),
  toggleDrawMode: () => set((state) => ({ drawMode: !state.drawMode })),
  setFlaggedOnlyMode: (on) => set({ flaggedOnlyMode: on }),
  toggleFlaggedOnlyMode: () => set((state) => ({ flaggedOnlyMode: !state.flaggedOnlyMode })),

  // ── panel mutations ────────────────────────────────────
  setPageKeep: (page, keep) =>
    set((state) => ({
      ...pushHistory(state),
      panels: state.panels.map((panel) =>
        panel.page === page
          ? {
              ...panel,
              keep,
              manual_keep: keep ? true : panel.manual_keep,
              auto_skipped: keep ? false : panel.auto_skipped,
              skip_reason: keep ? null : panel.skip_reason
            }
          : panel
      ),
      selectedIds: keep ? state.selectedIds : state.selectedIds.filter((id) => state.panels.find((p) => p.id === id)?.page !== page)
    })),

  setAllPagesKeep: (keep) =>
    set((state) => ({
      ...pushHistory(state),
      panels: state.panels.map((panel) => ({
        ...panel,
        keep,
        manual_keep: keep ? true : panel.manual_keep,
        auto_skipped: keep ? false : panel.auto_skipped,
        skip_reason: keep ? null : panel.skip_reason
      })),
      selectedIds: keep ? state.selectedIds : []
    })),

  updatePanel: (id, updates) =>
    set((state) => ({
      ...pushHistory(state),
      panels: reindex(state.panels.map((p) => (p.id === id ? { ...p, ...updates } : p)))
    })),

  movePanel: (id, dx, dy, bounds) =>
    set((state) => ({
      panels: reindex(
        state.panels.map((panel) =>
          panel.id === id
            ? clampPanelToBounds({ ...panel, x: panel.x + dx, y: panel.y + dy }, bounds)
            : panel
        )
      )
    })),

  resizePanel: (id, handle, dx, dy, bounds) =>
    set((state) => ({
      panels: reindex(
        state.panels.map((panel) => {
          if (panel.id !== id) return panel;
          const minSize = 40;
          const maxWidth = bounds?.width ?? Number.POSITIVE_INFINITY;
          const maxHeight = bounds?.height ?? Number.POSITIVE_INFINITY;

          const left0 = panel.x;
          const top0 = panel.y;
          const right0 = panel.x + panel.width;
          const bottom0 = panel.y + panel.height;

          let left = left0;
          let top = top0;
          let right = right0;
          let bottom = bottom0;

          if (handle.includes("w")) {
            left = left0 + dx;
            left = Math.min(left, right0 - minSize);
            left = Math.max(0, left);
          }
          if (handle.includes("e")) {
            right = right0 + dx;
            right = Math.max(right, left0 + minSize);
            if (Number.isFinite(maxWidth)) {
              right = Math.min(maxWidth, right);
            }
          }
          if (handle.includes("n")) {
            top = top0 + dy;
            top = Math.min(top, bottom0 - minSize);
            top = Math.max(0, top);
          }
          if (handle.includes("s")) {
            bottom = bottom0 + dy;
            bottom = Math.max(bottom, top0 + minSize);
            if (Number.isFinite(maxHeight)) {
              bottom = Math.min(maxHeight, bottom);
            }
          }

          const nextPanel = {
            ...panel,
            x: Math.round(left),
            y: Math.round(top),
            width: Math.max(minSize, Math.round(right - left)),
            height: Math.max(minSize, Math.round(bottom - top))
          };
          return clampPanelToBounds(nextPanel, bounds, minSize);
        })
      )
    })),

  addPanel: (page, box, bounds) =>
    set((state) => {
      const panelCountOnPage = state.panels.filter((p) => p.page === page).length;
      return {
        ...pushHistory(state),
        panels: reindex([
          ...state.panels,
          clampPanelToBounds(
            {
              id: `panel-${crypto.randomUUID().slice(0, 8)}`,
              page,
              panel: panelCountOnPage + 1,
              order: state.panels.length + 1,
              keep: true,
              merged_from: [],
              zoom_hint: "focus-center",
              manual_keep: true,
              manual_narration: false,
              narration_locked: false,
              manual_ocr_text: false,
              auto_skipped: false,
              text_detected: true,
              review_flags: [],
              ...box
            },
            bounds
          )
        ])
      };
    }),

  deleteSelected: () =>
    set((state) => ({
      ...pushHistory(state),
      panels: reindex(state.panels.filter((p) => !state.selectedIds.includes(p.id))),
      selectedIds: []
    })),

  toggleKeep: (id) =>
    set((state) => ({
      ...pushHistory(state),
      panels: state.panels.map((panel) => {
        if (panel.id !== id) return panel;
        const keep = !panel.keep;
        return {
          ...panel,
          keep,
          manual_keep: keep ? true : panel.manual_keep,
          auto_skipped: keep ? false : panel.auto_skipped,
          skip_reason: keep ? null : panel.skip_reason
        };
      })
    })),

  moveOrder: (id, direction) =>
    set((state) => {
      const panels = [...state.panels].sort((a, b) => a.order - b.order);
      const index = panels.findIndex((p) => p.id === id);
      const target = direction === "up" ? index - 1 : index + 1;
      if (index < 0 || target < 0 || target >= panels.length) return state;
      [panels[index], panels[target]] = [panels[target], panels[index]];
      return { ...pushHistory(state), panels: reindex(panels) };
    }),

  splitPanel: (id, axis) =>
    set((state) => {
      const source = state.panels.find((p) => p.id === id);
      if (!source) return state;
      const firstHalf =
        axis === "horizontal"
          ? { x: source.x, y: source.y, width: Math.round(source.width / 2), height: source.height }
          : { x: source.x, y: source.y, width: source.width, height: Math.round(source.height / 2) };
      const secondHalf =
        axis === "horizontal"
          ? { x: source.x + Math.round(source.width / 2), y: source.y, width: Math.round(source.width / 2), height: source.height }
          : { x: source.x, y: source.y + Math.round(source.height / 2), width: source.width, height: Math.round(source.height / 2) };
      const remaining = state.panels.filter((p) => p.id !== id);
      return {
        ...pushHistory(state),
        panels: reindex([
          ...remaining,
          { ...source, id: `panel-${crypto.randomUUID().slice(0, 8)}`, ...firstHalf, merged_from: [...source.merged_from], manual_keep: true, auto_skipped: false },
          { ...source, id: `panel-${crypto.randomUUID().slice(0, 8)}`, ...secondHalf, merged_from: [...source.merged_from], manual_keep: true, auto_skipped: false }
        ]),
        selectedIds: []
      };
    }),

  mergeSelected: () =>
    set((state) => {
      if (state.selectedIds.length < 2) return state;
      const selected = state.panels.filter((p) => state.selectedIds.includes(p.id));
      const remaining = state.panels.filter((p) => !state.selectedIds.includes(p.id));
      const x = Math.min(...selected.map((p) => p.x));
      const y = Math.min(...selected.map((p) => p.y));
      const maxX = Math.max(...selected.map((p) => p.x + p.width));
      const maxY = Math.max(...selected.map((p) => p.y + p.height));
      const base = selected.sort((a, b) => a.order - b.order)[0];
      return {
        ...pushHistory(state),
        panels: reindex([
          ...remaining,
          {
            ...base,
            id: `panel-${crypto.randomUUID().slice(0, 8)}`,
            x,
            y,
            width: maxX - x,
            height: maxY - y,
            merged_from: selected.map((p) => p.id),
            manual_keep: true,
            auto_skipped: false
          }
        ]),
        selectedIds: []
      };
    }),

  // ── batch ──────────────────────────────────────────────
  batchSetKeep: (ids, keep) =>
    set((state) => {
      const idSet = new Set(ids);
      return {
        ...pushHistory(state),
        panels: state.panels.map((p) =>
          idSet.has(p.id)
            ? {
                ...p,
                keep,
                manual_keep: keep ? true : p.manual_keep,
                auto_skipped: keep ? false : p.auto_skipped,
                skip_reason: keep ? null : p.skip_reason
              }
            : p
        )
      };
    }),

  batchSetDuration: (ids, seconds) =>
    set((state) => {
      const idSet = new Set(ids);
      return {
        ...pushHistory(state),
        panels: state.panels.map((p) => (idSet.has(p.id) ? { ...p, duration_seconds: seconds } : p))
      };
    }),

  batchDelete: (ids) =>
    set((state) => {
      const idSet = new Set(ids);
      return {
        ...pushHistory(state),
        panels: reindex(state.panels.filter((p) => !idSet.has(p.id))),
        selectedIds: state.selectedIds.filter((id) => !idSet.has(id))
      };
    }),

  // ── undo / redo ────────────────────────────────────────
  undo: () =>
    set((state) => {
      if (!state._history.length) return {};
      const previous = state._history[state._history.length - 1];
      const newHistory = state._history.slice(0, -1);
      const newFuture = [...state._future, structuredClone(state.panels)];
      return {
        panels: previous,
        _history: newHistory,
        _future: newFuture,
        canUndo: newHistory.length > 0,
        canRedo: true,
        selectedIds: []
      };
    }),

  redo: () =>
    set((state) => {
      if (!state._future.length) return {};
      const next = state._future[state._future.length - 1];
      const newFuture = state._future.slice(0, -1);
      const newHistory = [...state._history, structuredClone(state.panels)];
      return {
        panels: next,
        _history: newHistory,
        _future: newFuture,
        canUndo: true,
        canRedo: newFuture.length > 0,
        selectedIds: []
      };
    })
}));
