"use client";

import { useEffect, useMemo, useRef, useState } from "react";

import { buildMediaUrl, cn } from "@/lib/utils";
import { PanelBox } from "@/lib/types";

type DragState =
  | { type: "move"; panelId: string; startX: number; startY: number }
  | { type: "resize"; panelId: string; handle: "nw" | "ne" | "sw" | "se"; startX: number; startY: number }
  | { type: "draw"; startX: number; startY: number; currentX: number; currentY: number }
  | null;

export function PanelCanvas({
  imageUrl,
  panels,
  selectedIds,
  drawMode = false,
  onSelect,
  onMove,
  onResize,
  onAddPanel,
  onDrawComplete,
  onNaturalSizeChange,
  onPanelContextMenu,
  onToggleKeep
}: {
  imageUrl?: string | null;
  panels: PanelBox[];
  selectedIds: string[];
  drawMode?: boolean;
  onSelect: (id: string, additive?: boolean) => void;
  onMove: (id: string, dx: number, dy: number) => void;
  onResize: (id: string, handle: "nw" | "ne" | "sw" | "se", dx: number, dy: number) => void;
  onAddPanel: (box: { x: number; y: number; width: number; height: number }) => void;
  onDrawComplete?: () => void;
  onNaturalSizeChange?: (size: { width: number; height: number }) => void;
  onPanelContextMenu?: (panelId: string, x: number, y: number) => void;
  onToggleKeep: (panelId: string) => void;
}) {
  const containerRef = useRef<HTMLDivElement>(null);
  const viewportRef = useRef<HTMLDivElement>(null);
  const frameRef = useRef<HTMLDivElement>(null);
  const imageRef = useRef<HTMLImageElement>(null);
  const lockedScrollRef = useRef<{ left: number; top: number } | null>(null);
  const [naturalSize, setNaturalSize] = useState({ width: 1, height: 1 });
  const [viewportSize, setViewportSize] = useState({ width: 1, height: 1 });
  const [renderSize, setRenderSize] = useState({ width: 1, height: 1 });
  const [dragState, setDragState] = useState<DragState>(null);
  const [zoom, setZoom] = useState(1);
  const interacting = Boolean(dragState);

  /** Lock scroll position synchronously before any state update to prevent jump */
  function lockScroll() {
    const viewport = viewportRef.current;
    if (viewport) {
      lockedScrollRef.current = {
        left: viewport.scrollLeft,
        top: viewport.scrollTop
      };
    }
  }

  useEffect(() => {
    function syncSizes() {
      if (interacting) {
        return;
      }
      const viewport = viewportRef.current;
      const frame = frameRef.current;
      if (viewport) {
        setViewportSize({
          width: Math.max(viewport.clientWidth, 1),
          height: Math.max(viewport.clientHeight, 1)
        });
      }
      if (frame) {
        setRenderSize({
          width: Math.max(frame.clientWidth, 1),
          height: Math.max(frame.clientHeight, 1)
        });
      }
    }

    const resizeObserver = new ResizeObserver(() => {
      syncSizes();
    });
    if (viewportRef.current) {
      resizeObserver.observe(viewportRef.current);
    }
    if (frameRef.current) {
      resizeObserver.observe(frameRef.current);
    }
    window.addEventListener("resize", syncSizes);
    syncSizes();
    return () => {
      resizeObserver.disconnect();
      window.removeEventListener("resize", syncSizes);
    };
  }, [imageUrl, interacting]);

  useEffect(() => {
    setZoom(1);
  }, [imageUrl]);

  const scale = useMemo(
    () => ({
      x: naturalSize.width / Math.max(renderSize.width, 1),
      y: naturalSize.height / Math.max(renderSize.height, 1)
    }),
    [naturalSize, renderSize]
  );

  const fittedSize = useMemo(() => {
    const availableWidth = Math.max(viewportSize.width - 96, 200);
    const availableHeight = Math.max(viewportSize.height - 96, 200);
    const widthScale = availableWidth / Math.max(naturalSize.width, 1);
    const heightScale = availableHeight / Math.max(naturalSize.height, 1);
    const boundedScale = Number.isFinite(Math.min(widthScale, heightScale))
      ? Math.max(Math.min(widthScale, heightScale), 0.05)
      : 1;
    return {
      width: Math.max(Math.round(naturalSize.width * boundedScale), 1),
      height: Math.max(Math.round(naturalSize.height * boundedScale), 1)
    };
  }, [naturalSize, viewportSize]);

  // Clear scroll lock when drag ends (lock is set synchronously in mousedown handlers)
  useEffect(() => {
    if (!dragState) {
      lockedScrollRef.current = null;
    }
  }, [dragState]);

  useEffect(() => {
    function handleMove(event: MouseEvent) {
      if (!dragState || !frameRef.current) return;
      event.preventDefault();
      const rect = frameRef.current.getBoundingClientRect();
      const currentX = Math.min(Math.max(event.clientX - rect.left, 0), rect.width);
      const currentY = Math.min(Math.max(event.clientY - rect.top, 0), rect.height);
      if (dragState.type === "move") {
        onMove(dragState.panelId, (currentX - dragState.startX) * scale.x, (currentY - dragState.startY) * scale.y);
        setDragState({ ...dragState, startX: currentX, startY: currentY });
      }
      if (dragState.type === "resize") {
        onResize(dragState.panelId, dragState.handle, (currentX - dragState.startX) * scale.x, (currentY - dragState.startY) * scale.y);
        setDragState({ ...dragState, startX: currentX, startY: currentY });
      }
      if (dragState.type === "draw") {
        setDragState({ ...dragState, currentX, currentY });
      }

      const viewport = viewportRef.current;
      const locked = lockedScrollRef.current;
      if (viewport && locked) {
        viewport.scrollLeft = locked.left;
        viewport.scrollTop = locked.top;
      }
    }

    function handleUp() {
      if (dragState?.type === "draw") {
        const x = Math.min(dragState.startX, dragState.currentX) * scale.x;
        const y = Math.min(dragState.startY, dragState.currentY) * scale.y;
        const width = Math.abs(dragState.currentX - dragState.startX) * scale.x;
        const height = Math.abs(dragState.currentY - dragState.startY) * scale.y;
        if (width > 32 && height > 32) {
          onAddPanel({ x, y, width, height });
          onDrawComplete?.();
        }
      }
      setDragState(null);
    }

    function blockWheel(event: WheelEvent) {
      if (!dragState) return;
      event.preventDefault();
    }

    function blockTouchMove(event: TouchEvent) {
      if (!dragState) return;
      event.preventDefault();
    }

    function blockGesture(event: Event) {
      if (!dragState) return;
      event.preventDefault();
    }

    function handleViewportScroll() {
      if (!dragState) return;
      const viewport = viewportRef.current;
      const locked = lockedScrollRef.current;
      if (!viewport || !locked) return;
      if (viewport.scrollLeft === locked.left && viewport.scrollTop === locked.top) return;
      requestAnimationFrame(() => {
        if (!viewportRef.current || !lockedScrollRef.current) return;
        viewportRef.current.scrollLeft = lockedScrollRef.current.left;
        viewportRef.current.scrollTop = lockedScrollRef.current.top;
      });
    }

    window.addEventListener("mousemove", handleMove);
    window.addEventListener("mouseup", handleUp);
    window.addEventListener("wheel", blockWheel, { passive: false });
    window.addEventListener("touchmove", blockTouchMove, { passive: false });
    window.addEventListener("gesturestart", blockGesture as EventListener, { passive: false } as AddEventListenerOptions);
    window.addEventListener("gesturechange", blockGesture as EventListener, { passive: false } as AddEventListenerOptions);
    const viewport = viewportRef.current;
    viewport?.addEventListener("scroll", handleViewportScroll, { passive: true });
    if (dragState) {
      document.body.style.userSelect = "none";
    }
    return () => {
      window.removeEventListener("mousemove", handleMove);
      window.removeEventListener("mouseup", handleUp);
      window.removeEventListener("wheel", blockWheel as EventListener);
      window.removeEventListener("touchmove", blockTouchMove as EventListener);
      window.removeEventListener("gesturestart", blockGesture as EventListener);
      window.removeEventListener("gesturechange", blockGesture as EventListener);
      viewport?.removeEventListener("scroll", handleViewportScroll as EventListener);
      document.body.style.userSelect = "";
    };
  }, [dragState, onAddPanel, onDrawComplete, onMove, onResize, scale.x, scale.y]);

  function clampZoom(nextZoom: number) {
    return Math.min(4, Math.max(1, nextZoom));
  }

  function applyZoom(nextZoom: number, clientX?: number, clientY?: number) {
    const viewport = viewportRef.current;
    if (!viewport) {
      setZoom(clampZoom(nextZoom));
      return;
    }

    const bounded = clampZoom(nextZoom);
    const previous = zoom;
    if (Math.abs(previous - bounded) < 0.001) return;

    const rect = viewport.getBoundingClientRect();
    const anchorX = clientX != null ? clientX - rect.left : rect.width / 2;
    const anchorY = clientY != null ? clientY - rect.top : rect.height / 2;
    const worldX = (viewport.scrollLeft + anchorX) / previous;
    const worldY = (viewport.scrollTop + anchorY) / previous;

    setZoom(bounded);

    requestAnimationFrame(() => {
      viewport.scrollLeft = Math.max(0, worldX * bounded - anchorX);
      viewport.scrollTop = Math.max(0, worldY * bounded - anchorY);
    });
  }

  function handleWheel(event: React.WheelEvent<HTMLDivElement>) {
    if (!event.ctrlKey && !event.metaKey) return;
    event.preventDefault();
    const nextZoom = zoom * (event.deltaY < 0 ? 1.08 : 0.92);
    applyZoom(nextZoom, event.clientX, event.clientY);
  }

  return (
    <div
      ref={containerRef}
      className={cn(
        "grid h-full w-full place-items-center overflow-hidden overscroll-none bg-[radial-gradient(circle_at_top,_rgba(34,211,238,0.06),_transparent_32%),linear-gradient(180deg,rgba(255,255,255,0.03),rgba(255,255,255,0))] p-3 md:p-4",
        drawMode && "cursor-crosshair"
      )}
    >
      {imageUrl ? (
        <div className="flex h-full w-full flex-col overflow-hidden">
          <div className="mb-2 flex items-center justify-between px-1 text-[11px] text-mutedForeground">
            <div className="rounded-full border border-white/[0.08] bg-black/20 px-2.5 py-1">
              Pinch on the trackpad or hold Ctrl/Cmd and scroll to zoom
            </div>
            <div className="flex items-center gap-2">
              <button
                type="button"
                className="rounded-full border border-white/[0.08] px-2.5 py-1 transition hover:bg-white/[0.08] hover:text-white"
                onClick={() => applyZoom(1)}
              >
                Fit
              </button>
              <span className="rounded-full border border-white/[0.08] bg-black/20 px-2.5 py-1 text-white">
                {Math.round(zoom * 100)}%
              </span>
            </div>
          </div>
          <div
            ref={viewportRef}
            className="flex-1 overflow-auto overscroll-contain rounded-[28px] border border-white/[0.08] bg-black/20 shadow-[0_24px_80px_rgba(0,0,0,0.45)]"
            onWheel={handleWheel}
            style={{ scrollbarGutter: "stable both-edges" }}
          >
            <div className="flex min-h-full min-w-full items-start justify-center px-6 pb-24 pt-6">
              <div
                ref={frameRef}
                className={cn(
                  "relative overflow-visible rounded-[28px] border border-white/[0.08] bg-black/30 shadow-[0_24px_80px_rgba(0,0,0,0.45)]",
                  dragState && "cursor-grabbing"
                )}
                style={{
                  width: `${Math.max(fittedSize.width * zoom, 1)}px`,
                  height: `${Math.max(fittedSize.height * zoom, 1)}px`
                }}
                onMouseDown={(event) => {
                  if (!drawMode || !frameRef.current) return;
                  event.preventDefault();
                  lockScroll();
                  const rect = frameRef.current.getBoundingClientRect();
                  const startX = Math.min(Math.max(event.clientX - rect.left, 0), rect.width);
                  const startY = Math.min(Math.max(event.clientY - rect.top, 0), rect.height);
                  setDragState({
                    type: "draw",
                    startX,
                    startY,
                    currentX: startX,
                    currentY: startY
                  });
                }}
              >
                {/* eslint-disable-next-line @next/next/no-img-element */}
                <img
                  ref={imageRef}
                  src={buildMediaUrl(imageUrl)}
                  alt="Manga page"
                  className="block h-full w-full select-none object-contain"
                  draggable={false}
                  onLoad={(event) => {
                    const image = event.currentTarget;
                    const size = { width: image.naturalWidth, height: image.naturalHeight };
                    setNaturalSize(size);
                    onNaturalSizeChange?.(size);
                  }}
                />

                <div className="pointer-events-none absolute inset-0">
                  {panels.map((panel) => {
                    const left = `${(panel.x / naturalSize.width) * 100}%`;
                    const top = `${(panel.y / naturalSize.height) * 100}%`;
                    const width = `${(panel.width / naturalSize.width) * 100}%`;
                    const height = `${(panel.height / naturalSize.height) * 100}%`;
                    const selected = selectedIds.includes(panel.id);

                    return (
                      <div
                        key={panel.id}
                        data-canvas-panel-id={panel.id}
                        className={cn(
                          "pointer-events-auto absolute border-2 bg-accent/10 shadow-[0_0_0_1px_rgba(0,0,0,0.85),0_0_0_3px_rgba(34,211,238,0.22)] transition",
                          drawMode && "pointer-events-none",
                          panel.keep ? "border-accent" : "border-brand-rose/70 bg-brand-rose/10 shadow-[0_0_0_1px_rgba(0,0,0,0.85),0_0_0_3px_rgba(251,113,133,0.24)]",
                          selected && "shadow-[0_0_0_1px_rgba(0,0,0,0.95),0_0_0_5px_rgba(34,211,238,0.36)]"
                        )}
                        style={{ left, top, width, height, touchAction: "none" }}
                        onContextMenu={(event) => {
                          event.preventDefault();
                          event.stopPropagation();
                          if (!selectedIds.includes(panel.id)) onSelect(panel.id, false);
                          onPanelContextMenu?.(panel.id, event.clientX, event.clientY);
                        }}
                        onDoubleClick={(event) => {
                          event.preventDefault();
                          event.stopPropagation();
                          onSelect(panel.id, false);
                          onToggleKeep(panel.id);
                        }}
                        onMouseDown={(event) => {
                          if (event.button !== 0) return;
                          event.preventDefault();
                          event.stopPropagation();
                          lockScroll();
                          onSelect(panel.id, event.shiftKey);
                          const rect = frameRef.current?.getBoundingClientRect();
                          if (!rect) return;
                          setDragState({
                            type: "move",
                            panelId: panel.id,
                            startX: Math.min(Math.max(event.clientX - rect.left, 0), rect.width),
                            startY: Math.min(Math.max(event.clientY - rect.top, 0), rect.height)
                          });
                        }}
                      >
                        <div className="absolute left-1 top-1 min-w-9 rounded-md bg-black/85 px-2 py-0.5 text-center text-[10px] font-medium text-white shadow-sm">
                          {panel.order}
                        </div>
                        {(panel.ocr_text ?? "").trim() ? (
                          <div className="absolute bottom-1 left-1 right-1 rounded-md bg-black/70 px-2 py-1 text-[10px] leading-tight text-white/80 line-clamp-2 backdrop-blur-sm">
                            {panel.ocr_text}
                          </div>
                        ) : null}
                        {(["nw", "ne", "sw", "se"] as const).map((handle) => (
                          <div
                            key={handle}
                            aria-hidden="true"
                            className={cn(
                              "absolute h-3 w-3 border border-black bg-white",
                              handle === "nw" && "-left-1.5 -top-1.5",
                              handle === "ne" && "-right-1.5 -top-1.5",
                              handle === "sw" && "-bottom-1.5 -left-1.5",
                              handle === "se" && "-bottom-1.5 -right-1.5"
                            )}
                            style={{ touchAction: "none" }}
                            onMouseDown={(event) => {
                              if (event.button !== 0) return;
                              event.preventDefault();
                              event.stopPropagation();
                              lockScroll();
                              if (!selectedIds.includes(panel.id)) {
                                onSelect(panel.id, false);
                              }
                              const rect = frameRef.current?.getBoundingClientRect();
                              if (!rect) return;
                              setDragState({
                                type: "resize",
                                panelId: panel.id,
                                handle,
                                startX: Math.min(Math.max(event.clientX - rect.left, 0), rect.width),
                                startY: Math.min(Math.max(event.clientY - rect.top, 0), rect.height)
                              });
                            }}
                          />
                        ))}

                      </div>
                    );
                  })}

                  {dragState?.type === "draw" ? (
                    <div
                      className="absolute border-2 border-dashed border-brand-amber bg-brand-amber/10"
                      style={{
                        left: Math.min(dragState.startX, dragState.currentX),
                        top: Math.min(dragState.startY, dragState.currentY),
                        width: Math.abs(dragState.currentX - dragState.startX),
                        height: Math.abs(dragState.currentY - dragState.startY)
                      }}
                    />
                  ) : null}
                </div>
              </div>
            </div>
          </div>
        </div>
      ) : (
        <div className="flex h-[480px] items-center justify-center text-sm text-mutedForeground">No page available yet.</div>
      )}
    </div>
  );
}
