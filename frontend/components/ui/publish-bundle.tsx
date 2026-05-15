"use client";

/**
 * PublishBundle — "Ready to publish on YouTube" card.
 *
 * After the youtube_bundle stage finishes the backend has:
 *   • title + 3 title variants
 *   • a markdown description
 *   • a 1280×720 viral-style thumbnail PNG (with the source panel
 *     preserved separately if the user wants to start over)
 *
 * This card shows everything inline with one-click copy buttons for the
 * fields the user pastes into YouTube Studio, plus a download button for
 * the thumbnail PNG.
 *
 * Drag-and-drop the thumbnail from the preview directly into the YouTube
 * Studio thumbnail uploader — that's the whole flow.
 */

import * as React from "react";
import { Card, CardDescription, CardTitle } from "./card";
import { Badge } from "./badge";

export interface PublishBundle {
  project_id: string;
  title: string | null;
  title_variants: string[];
  description: string | null;
  thumbnail_url: string | null;
  thumbnail_source_url: string | null;
  thumbnail_source_panel_id: string | null;
  bundle_dir: string | null;
}

interface PublishBundleCardProps {
  bundle: PublishBundle | null;
  loading?: boolean;
  /** Optional override for media-prefix when the frontend runs separately. */
  mediaPrefix?: string;
  className?: string;
}

function CopyButton({ value, label }: { value: string; label?: string }) {
  const [copied, setCopied] = React.useState(false);
  return (
    <button
      type="button"
      className="p-btn-ghost text-xs"
      onClick={async () => {
        try {
          await navigator.clipboard.writeText(value);
          setCopied(true);
          setTimeout(() => setCopied(false), 1400);
        } catch {
          /* swallow */
        }
      }}
    >
      {copied ? "Copied!" : label ?? "Copy"}
    </button>
  );
}

export function PublishBundleCard({
  bundle,
  loading,
  mediaPrefix,
  className,
}: PublishBundleCardProps) {
  if (loading) {
    return (
      <Card padded="md" className={className}>
        <CardTitle>Preparing your publish bundle</CardTitle>
        <CardDescription className="mt-2">
          We&apos;re writing a title, description, and viral-style thumbnail. Hang tight.
        </CardDescription>
      </Card>
    );
  }
  if (!bundle) {
    return (
      <Card padded="md" className={className}>
        <CardTitle>Publish bundle</CardTitle>
        <CardDescription className="mt-2">
          Once the video finishes rendering, we&apos;ll auto-generate a YouTube
          title, description, and custom thumbnail here — ready to drag into
          YouTube Studio.
        </CardDescription>
      </Card>
    );
  }

  const thumbUrl = (() => {
    if (!bundle.thumbnail_url) return null;
    if (mediaPrefix && bundle.thumbnail_url.startsWith("/media/")) {
      return `${mediaPrefix}${bundle.thumbnail_url}`;
    }
    return bundle.thumbnail_url;
  })();

  return (
    <Card padded="md" className={`p-edge-ok ${className ?? ""}`}>
      <div className="flex items-start justify-between gap-3">
        <div>
          <CardTitle>Ready to publish on YouTube</CardTitle>
          <CardDescription className="mt-1">
            Drag the thumbnail into YouTube Studio, paste the title and
            description, and you&apos;re live.
          </CardDescription>
        </div>
        <Badge tone="ok" dot>
          Bundle ready
        </Badge>
      </div>

      <div className="mt-5 grid gap-5 lg:grid-cols-[460px_minmax(0,1fr)]">
        {/* Thumbnail preview */}
        <div>
          {thumbUrl ? (
            <a
              href={thumbUrl}
              download="thumbnail.png"
              draggable
              title="Drag onto YouTube Studio, or click to download"
              className="block aspect-video w-full overflow-hidden rounded-2xl border border-white/[0.10] bg-black/30 transition-transform duration-fast ease-liquid hover:-translate-y-0.5 hover:border-accent/40 hover:shadow-[0_0_32px_-8px_rgb(var(--p-accent)/0.55)]"
            >
              {/* eslint-disable-next-line @next/next/no-img-element */}
              <img
                src={thumbUrl}
                alt="YouTube thumbnail"
                className="h-full w-full object-cover"
                loading="lazy"
                draggable
              />
            </a>
          ) : (
            <div className="aspect-video w-full rounded-2xl border border-white/[0.08] bg-white/[0.04]" />
          )}
          {thumbUrl ? (
            <p className="mt-2 text-[11px] text-mutedForeground text-center">
              Drag to YouTube Studio · or click to download
            </p>
          ) : null}
        </div>

        {/* Text fields */}
        <div className="min-w-0 space-y-4">
          {/* Title */}
          <div>
            <div className="flex items-center justify-between gap-2">
              <p className="text-[10px] uppercase tracking-track text-mutedForeground">
                Title
              </p>
              {bundle.title ? <CopyButton value={bundle.title} /> : null}
            </div>
            <p className="mt-1 text-base text-foreground font-medium leading-snug">
              {bundle.title || "—"}
            </p>
            {bundle.title_variants?.length ? (
              <details className="group mt-2">
                <summary className="cursor-pointer text-xs text-mutedForeground hover:text-foreground">
                  See {bundle.title_variants.length} alternative title{bundle.title_variants.length === 1 ? "" : "s"}
                </summary>
                <ul className="mt-2 space-y-1.5 text-sm">
                  {bundle.title_variants.map((variant) => (
                    <li
                      key={variant}
                      className="flex items-start justify-between gap-2 rounded-2xl border border-white/[0.06] bg-white/[0.03] px-3 py-2"
                    >
                      <span className="min-w-0 break-words">{variant}</span>
                      <CopyButton value={variant} />
                    </li>
                  ))}
                </ul>
              </details>
            ) : null}
          </div>

          {/* Description */}
          {bundle.description ? (
            <div>
              <div className="flex items-center justify-between gap-2">
                <p className="text-[10px] uppercase tracking-track text-mutedForeground">
                  Description (Markdown)
                </p>
                <CopyButton value={bundle.description} label="Copy all" />
              </div>
              <pre className="mt-1 max-h-64 overflow-auto whitespace-pre-wrap rounded-2xl border border-white/[0.08] bg-white/[0.03] px-4 py-3 text-xs leading-relaxed text-foreground font-sans">
                {bundle.description}
              </pre>
            </div>
          ) : null}
        </div>
      </div>
    </Card>
  );
}
