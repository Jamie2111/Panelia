"use client";

/**
 * PublishBundle: "Ready to publish on YouTube" editable studio.
 *
 * The bundle stage produces an opinionated first pass (title + alternates,
 * a viral plain-text description, and 5 thumbnail variants). This card is
 * where the user FINISHES that pass, the way a publish-checklist tab in
 * YouTube Studio would feel: edit the title inline, rewrite or tweak the
 * description in a textarea, and pick whichever thumbnail variant lands
 * best for the channel voice. Edits persist via PUT /youtube-bundle so
 * the manifest/title.txt/description.md on disk always match what the
 * user sees on screen.
 *
 * The carousel of thumbnail variants is the biggest UX win. The backend
 * generates 5 candidates spread across the chapter (climax shot,
 * character beat, stakes shot, etc.); the user picks one with a click.
 * The chosen variant becomes the canonical thumbnail.png that the
 * "drag-and-drop into YouTube Studio" affordance points at.
 */

import * as React from "react";
import { Card, CardDescription, CardTitle } from "./card";
import { Badge } from "./badge";
import { api } from "@/lib/api";

export interface PublishBundleThumbnailVariant {
  index: number;
  style_id: string;
  style_label: string;
  url: string | null;
}

export interface PublishBundle {
  project_id: string;
  title: string | null;
  title_variants: string[];
  description: string | null;
  thumbnail_url: string | null;
  thumbnail_source_url: string | null;
  thumbnail_source_panel_id: string | null;
  thumbnail_variants?: PublishBundleThumbnailVariant[];
  chosen_thumbnail_index?: number;
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

function withMediaPrefix(url: string | null, mediaPrefix?: string): string | null {
  if (!url) return null;
  if (mediaPrefix && url.startsWith("/media/")) return `${mediaPrefix}${url}`;
  return url;
}

export function PublishBundleCard({
  bundle,
  loading,
  mediaPrefix,
  className,
}: PublishBundleCardProps) {
  // ── Local edit state. Mirrors the bundle as it changes on disk, and
  // takes over once the user starts typing. Each save round-trips
  // through the PUT endpoint and the returned manifest re-syncs us.
  const [draftTitle, setDraftTitle] = React.useState<string>("");
  const [draftDescription, setDraftDescription] = React.useState<string>("");
  const [chosenIndex, setChosenIndex] = React.useState<number>(0);
  const [savingField, setSavingField] = React.useState<string | null>(null);
  const [saveError, setSaveError] = React.useState<string | null>(null);
  const [showAltTitles, setShowAltTitles] = React.useState(false);

  React.useEffect(() => {
    if (!bundle) return;
    setDraftTitle(bundle.title ?? "");
    setDraftDescription(bundle.description ?? "");
    setChosenIndex(bundle.chosen_thumbnail_index ?? 0);
  }, [bundle?.project_id, bundle?.title, bundle?.description, bundle?.chosen_thumbnail_index]);

  if (loading) {
    return (
      <Card padded="md" className={className}>
        <CardTitle>Preparing your publish bundle</CardTitle>
        <CardDescription className="mt-2">
          We&apos;re writing a title, description, and viral-style thumbnails. Hang tight.
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
          title, description, and a row of thumbnail variants here. Ready to
          drag into YouTube Studio.
        </CardDescription>
      </Card>
    );
  }

  const projectId = bundle.project_id;
  const variants = bundle.thumbnail_variants ?? [];
  const activeThumbUrl = (() => {
    if (variants.length > 0) {
      const v = variants[Math.min(chosenIndex, variants.length - 1)];
      return withMediaPrefix(v?.url ?? null, mediaPrefix);
    }
    return withMediaPrefix(bundle.thumbnail_url, mediaPrefix);
  })();

  async function persist(patch: { title?: string; description?: string; chosen_thumbnail_index?: number }, field: string) {
    setSavingField(field);
    setSaveError(null);
    try {
      await api.updateYouTubeBundle(projectId, patch);
    } catch (err) {
      setSaveError(err instanceof Error ? err.message : "Save failed");
    } finally {
      setSavingField(null);
    }
  }

  // Debounce title + description on change. Variant pick saves immediately.
  React.useEffect(() => {
    if (!bundle) return;
    if (draftTitle === (bundle.title ?? "")) return;
    const handle = window.setTimeout(() => {
      void persist({ title: draftTitle }, "title");
    }, 700);
    return () => window.clearTimeout(handle);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [draftTitle]);

  React.useEffect(() => {
    if (!bundle) return;
    if (draftDescription === (bundle.description ?? "")) return;
    const handle = window.setTimeout(() => {
      void persist({ description: draftDescription }, "description");
    }, 900);
    return () => window.clearTimeout(handle);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [draftDescription]);

  function pickVariant(idx: number) {
    setChosenIndex(idx);
    void persist({ chosen_thumbnail_index: idx }, "thumbnail");
  }

  return (
    <Card padded="md" className={`p-edge-ok ${className ?? ""}`}>
      <div className="flex items-start justify-between gap-3">
        <div>
          <CardTitle>Ready to publish on YouTube</CardTitle>
          <CardDescription className="mt-1">
            Edit any field. Changes save automatically. Drag the chosen
            thumbnail to YouTube Studio, paste the title and description, and
            you&apos;re live.
          </CardDescription>
        </div>
        <Badge tone={saveError ? "warn" : "ok"} dot>
          {savingField ? `Saving ${savingField}...` : saveError ? "Save failed" : "Bundle ready"}
        </Badge>
      </div>

      <div className="mt-5 grid gap-5 lg:grid-cols-[460px_minmax(0,1fr)]">
        {/* ── Thumbnail column ──────────────────────────────── */}
        <div>
          {activeThumbUrl ? (
            <a
              href={activeThumbUrl}
              download={`thumbnail_v${chosenIndex}.png`}
              draggable
              title="Drag onto YouTube Studio, or click to download"
              className="block aspect-video w-full overflow-hidden rounded-2xl border border-white/[0.10] bg-black/30 transition-transform duration-fast ease-liquid hover:-translate-y-0.5 hover:border-accent/40 hover:shadow-[0_0_32px_-8px_rgb(var(--p-accent)/0.55)]"
            >
              {/* eslint-disable-next-line @next/next/no-img-element */}
              <img
                src={activeThumbUrl}
                alt="YouTube thumbnail"
                className="h-full w-full object-cover"
                loading="lazy"
                draggable
              />
            </a>
          ) : (
            <div className="aspect-video w-full rounded-2xl border border-white/[0.08] bg-white/[0.04]" />
          )}
          {activeThumbUrl ? (
            <p className="mt-2 text-[11px] text-mutedForeground text-center">
              Drag to YouTube Studio, or click to download
            </p>
          ) : null}

          {/* Variant carousel */}
          {variants.length > 1 ? (
            <div className="mt-3">
              <p className="text-[10px] uppercase tracking-track text-mutedForeground">
                Thumbnail variants ({variants.length})
              </p>
              <div className="mt-2 flex gap-2 overflow-x-auto pb-1">
                {variants.map((variant) => {
                  const vUrl = withMediaPrefix(variant.url, mediaPrefix);
                  const active = variant.index === chosenIndex;
                  return (
                    <button
                      key={variant.index}
                      type="button"
                      onClick={() => pickVariant(variant.index)}
                      title={variant.style_label}
                      className={`relative shrink-0 overflow-hidden rounded-xl border transition-all duration-fast ${
                        active
                          ? "border-accent shadow-[0_0_24px_-4px_rgb(var(--p-accent)/0.6)] ring-1 ring-accent/40"
                          : "border-white/[0.10] opacity-70 hover:opacity-100 hover:border-accent/50"
                      }`}
                      style={{ width: 120, aspectRatio: "16 / 9" }}
                    >
                      {vUrl ? (
                        /* eslint-disable-next-line @next/next/no-img-element */
                        <img
                          src={vUrl}
                          alt={variant.style_label}
                          className="h-full w-full object-cover"
                          loading="lazy"
                        />
                      ) : (
                        <div className="h-full w-full bg-white/[0.04]" />
                      )}
                      <span
                        className={`absolute bottom-0 left-0 right-0 truncate px-1.5 py-0.5 text-[9px] font-medium ${
                          active
                            ? "bg-accent text-accent-foreground"
                            : "bg-black/65 text-white/85"
                        }`}
                      >
                        {variant.style_label}
                      </span>
                    </button>
                  );
                })}
              </div>
            </div>
          ) : null}
        </div>

        {/* ── Text column ────────────────────────────────── */}
        <div className="min-w-0 space-y-4">
          {/* Title */}
          <div>
            <div className="flex items-center justify-between gap-2">
              <p className="text-[10px] uppercase tracking-track text-mutedForeground">
                Title
              </p>
              {draftTitle ? <CopyButton value={draftTitle} /> : null}
            </div>
            <input
              type="text"
              value={draftTitle}
              onChange={(e) => setDraftTitle(e.target.value)}
              maxLength={100}
              placeholder="Title (50-70 chars works best on YouTube)"
              className="mt-1 w-full rounded-2xl border border-white/[0.10] bg-white/[0.04] px-3 py-2.5 text-base font-medium leading-snug text-foreground outline-none transition-colors focus:border-accent/60 focus:bg-white/[0.06]"
            />
            <div className="mt-1 flex items-center justify-between gap-2">
              <span className="text-[10px] text-mutedForeground">
                {draftTitle.length} / 100
              </span>
              {bundle.title_variants?.length ? (
                <button
                  type="button"
                  onClick={() => setShowAltTitles((v) => !v)}
                  className="text-xs text-mutedForeground hover:text-foreground"
                >
                  {showAltTitles ? "Hide" : "See"} {bundle.title_variants.length} alternative title{bundle.title_variants.length === 1 ? "" : "s"}
                </button>
              ) : null}
            </div>
            {showAltTitles && bundle.title_variants?.length ? (
              <ul className="mt-2 space-y-1.5 text-sm">
                {bundle.title_variants.map((variant) => (
                  <li
                    key={variant}
                    className="flex items-start justify-between gap-2 rounded-2xl border border-white/[0.06] bg-white/[0.03] px-3 py-2"
                  >
                    <button
                      type="button"
                      onClick={() => setDraftTitle(variant)}
                      className="min-w-0 break-words text-left hover:text-accent"
                      title="Use this title"
                    >
                      {variant}
                    </button>
                    <CopyButton value={variant} />
                  </li>
                ))}
              </ul>
            ) : null}
          </div>

          {/* Description */}
          <div>
            <div className="flex items-center justify-between gap-2">
              <p className="text-[10px] uppercase tracking-track text-mutedForeground">
                Description
              </p>
              {draftDescription ? <CopyButton value={draftDescription} label="Copy all" /> : null}
            </div>
            <textarea
              value={draftDescription}
              onChange={(e) => setDraftDescription(e.target.value)}
              rows={10}
              maxLength={5000}
              placeholder="Hook line. Then a 2-3 sentence story tease. End with a subscribe nudge and 5-7 hashtags."
              className="mt-1 w-full resize-y rounded-2xl border border-white/[0.10] bg-white/[0.04] px-4 py-3 font-sans text-xs leading-relaxed text-foreground outline-none transition-colors focus:border-accent/60 focus:bg-white/[0.06]"
            />
            <div className="mt-1 flex items-center justify-between gap-2">
              <span className="text-[10px] text-mutedForeground">
                {draftDescription.length} / 5000 (YouTube cap)
              </span>
              <span className="text-[10px] text-mutedForeground">
                Plain text. YouTube ignores Markdown formatting.
              </span>
            </div>
          </div>

          {saveError ? (
            <p className="text-xs text-fail">{saveError}</p>
          ) : null}
        </div>
      </div>
    </Card>
  );
}
