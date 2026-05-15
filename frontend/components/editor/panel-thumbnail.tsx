"use client";

import { buildMediaUrl, cn } from "@/lib/utils";
import type { PanelBox } from "@/lib/types";

interface PanelThumbnailProps {
  panel: PanelBox;
  imageUrl: string;
  naturalWidth: number;
  naturalHeight: number;
  className?: string;
  contain?: boolean;
}

/**
 * Renders a cropped thumbnail of a single panel using CSS clipping.
 * No canvas API or extra backend calls needed - uses the full page image
 * with object-position and object-fit to show only the panel region.
 */
export function PanelThumbnail({
  panel,
  imageUrl,
  naturalWidth,
  naturalHeight,
  className,
  contain = false
}: PanelThumbnailProps) {
  if (!naturalWidth || !naturalHeight) return null;

  // Aspect ratio of the panel
  const aspect = panel.width / Math.max(panel.height, 1);
  const scaledWidth = (naturalWidth / Math.max(panel.width, 1)) * 100;
  const scaledHeight = (naturalHeight / Math.max(panel.height, 1)) * 100;
  const offsetX = (panel.x / Math.max(naturalWidth, 1)) * 100;
  const offsetY = (panel.y / Math.max(naturalHeight, 1)) * 100;
  const frameStyle = contain
    ? {
        aspectRatio: aspect,
        width: aspect >= 1 ? "100%" : "auto",
        height: aspect >= 1 ? "auto" : "100%"
      }
    : {
        aspectRatio: aspect
      };

  return (
    <div className={cn("flex h-full w-full items-center justify-center", className)}>
      <div
        className="relative max-h-full max-w-full overflow-hidden rounded-md bg-zinc-800"
        style={frameStyle}
      >
        {/* eslint-disable-next-line @next/next/no-img-element */}
        <img
          src={buildMediaUrl(imageUrl)}
          alt={`Panel ${panel.order}`}
          className="absolute left-0 top-0 block select-none"
          style={{
            width: `${scaledWidth}%`,
            height: `${scaledHeight}%`,
            transform: `translate(-${offsetX}%, -${offsetY}%)`,
            maxWidth: "none",
            maxHeight: "none",
            pointerEvents: "none"
          }}
          loading="lazy"
          draggable={false}
        />
      </div>
    </div>
  );
}
