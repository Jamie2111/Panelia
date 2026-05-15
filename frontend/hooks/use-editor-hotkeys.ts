"use client";

import { useEffect } from "react";

import { usePanelEditorStore } from "@/store/panel-editor-store";

interface HotkeyOptions {
  pageCount: number;
  onSave: () => void;
  currentPanels: { id: string }[];
}

export function useEditorHotkeys({ pageCount, onSave, currentPanels }: HotkeyOptions) {
  const store = usePanelEditorStore();

  useEffect(() => {
    function handler(e: KeyboardEvent) {
      const tag = (e.target as HTMLElement)?.tagName;
      if (tag === "INPUT" || tag === "TEXTAREA" || (e.target as HTMLElement)?.isContentEditable) return;

      const meta = e.metaKey || e.ctrlKey;

      // Cmd+Z / Cmd+Shift+Z - undo / redo
      if (meta && e.key === "z") {
        e.preventDefault();
        if (e.shiftKey) store.redo();
        else store.undo();
        return;
      }

      // Cmd+S - save
      if (meta && e.key === "s") {
        e.preventDefault();
        onSave();
        return;
      }

      // Skip all other shortcuts when meta is held
      if (meta) return;

      switch (e.key) {
        case "[":
          e.preventDefault();
          store.prevPage();
          break;
        case "]":
          e.preventDefault();
          store.nextPage(pageCount);
          break;
        case "ArrowUp":
          if (store.selectedIds.length) {
            e.preventDefault();
            store.selectPrevPanel();
          }
          break;
        case "ArrowDown":
          if (store.selectedIds.length) {
            e.preventDefault();
            store.selectNextPanel();
          }
          break;
        case "k":
        case "K":
          if (store.selectedIds[0]) {
            e.preventDefault();
            store.toggleKeep(store.selectedIds[0]);
          }
          break;
        case "Delete":
        case "Backspace":
          if (store.selectedIds.length) {
            e.preventDefault();
            store.deleteSelected();
          }
          break;
        case "d":
        case "D":
          e.preventDefault();
          store.toggleDrawMode();
          break;
        case "Escape":
          e.preventDefault();
          if (store.drawMode) store.setDrawMode(false);
          else store.clearSelection();
          break;
        case "a":
        case "A":
          e.preventDefault();
          store.selectPanels(currentPanels.map((p) => p.id));
          break;
        case " ":
          if (store.selectedIds[0]) {
            e.preventDefault();
            store.toggleKeep(store.selectedIds[0]);
            store.selectNextPanel();
          }
          break;
        case "h":
        case "H":
          if (store.selectedIds[0]) {
            e.preventDefault();
            store.splitPanel(store.selectedIds[0], "vertical");
          }
          break;
        case "v":
        case "V":
          if (store.selectedIds[0]) {
            e.preventDefault();
            store.splitPanel(store.selectedIds[0], "horizontal");
          }
          break;
        case "m":
        case "M":
          if (store.selectedIds.length >= 2) {
            e.preventDefault();
            store.mergeSelected();
          }
          break;
        case "f":
        case "F":
          e.preventDefault();
          store.toggleFlaggedOnlyMode();
          break;
        case "n":
        case "N":
          e.preventDefault();
          store.nextFlaggedPanel(pageCount);
          break;
      }
    }

    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [store, pageCount, onSave, currentPanels]);
}
