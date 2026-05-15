"use client";

/**
 * /settings/channel — the YouTuber's brand kit.
 *
 * Edits the single ChannelPreset that's applied automatically to every
 * thumbnail, cold-open title card, outro slide, and Shorts CTA in the
 * pipeline. Updates persist to backend/data/channel_preset.json.
 */

import * as React from "react";
import { AppShell } from "@/components/project/app-shell";
import { Card, CardDescription, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { Button } from "@/components/ui/button";
import { api } from "@/lib/api";
import type { ChannelPreset } from "@/lib/types";

export default function ChannelSettingsPage() {
  const [preset, setPreset] = React.useState<ChannelPreset | null>(null);
  const [saving, setSaving] = React.useState(false);
  const [savedAt, setSavedAt] = React.useState<number | null>(null);
  const [error, setError] = React.useState<string | null>(null);

  React.useEffect(() => {
    api.getChannelPreset()
      .then((data) => setPreset(data))
      .catch((err) => setError(err instanceof Error ? err.message : "Failed to load preset"));
  }, []);

  if (!preset) {
    return (
      <AppShell title="Channel" description="Loading your channel preset…">
        <Card padded="lg">
          <CardDescription>{error || "Loading…"}</CardDescription>
        </Card>
      </AppShell>
    );
  }

  const update = (patch: Partial<ChannelPreset>) =>
    setPreset((prev) => (prev ? { ...prev, ...patch } : prev));

  const save = async () => {
    if (!preset) return;
    setSaving(true);
    setError(null);
    try {
      const next = await api.updateChannelPreset(preset);
      setPreset(next);
      setSavedAt(Date.now());
    } catch (err) {
      setError(err instanceof Error ? err.message : "Save failed");
    } finally {
      setSaving(false);
    }
  };

  return (
    <AppShell
      title="Channel"
      description="Your branding applied to every thumbnail, cold open, outro, and Shorts CTA — across all projects."
      breadcrumb={{ href: "/", label: "Studio" }}
      meta={(
        <>
          {savedAt && <span className="p-pill p-pill-ok">saved {new Date(savedAt).toLocaleTimeString()}</span>}
          {saving && <span className="p-pill p-pill-info"><span className="inline-block h-1.5 w-1.5 rounded-full bg-current p-anim-breathe" />saving…</span>}
        </>
      )}
      actions={<Button onClick={save} disabled={saving}>Save preset</Button>}
    >
      {error && (
        <Card padded="md" className="p-edge-fail">
          <CardDescription className="text-fail">{error}</CardDescription>
        </Card>
      )}

      <div className="grid gap-6 lg:grid-cols-2">
        {/* Identity */}
        <Card padded="md">
          <CardTitle>Identity</CardTitle>
          <CardDescription className="mt-1">
            How viewers see your channel. The name is rendered on the title
            card + outro; the watermark sits in the corner of every video
            and thumbnail.
          </CardDescription>
          <div className="mt-5 space-y-4">
            <label className="block space-y-2">
              <span className="text-xs uppercase tracking-track text-mutedForeground">
                Channel name
              </span>
              <Input
                value={preset.channel_name}
                onChange={(e) => update({ channel_name: e.target.value })}
                placeholder="Your channel name"
              />
            </label>
            <label className="block space-y-2">
              <span className="text-xs uppercase tracking-track text-mutedForeground">
                Tagline
              </span>
              <Input
                value={preset.tagline}
                onChange={(e) => update({ tagline: e.target.value })}
                placeholder="Subscribe so you don't miss…"
              />
            </label>
            <label className="block space-y-2">
              <span className="text-xs uppercase tracking-track text-mutedForeground">
                Watermark text
              </span>
              <div className="flex gap-2">
                <Input
                  value={preset.watermark_text}
                  onChange={(e) => update({ watermark_text: e.target.value })}
                  placeholder="@yourhandle"
                  disabled={!preset.watermark_enabled}
                />
                <label className="flex items-center gap-2 text-xs text-mutedForeground shrink-0">
                  <input
                    type="checkbox"
                    className="h-4 w-4 accent-[rgb(var(--p-accent))]"
                    checked={preset.watermark_enabled}
                    onChange={(e) => update({ watermark_enabled: e.target.checked })}
                  />
                  show
                </label>
              </div>
            </label>
          </div>
        </Card>

        {/* Look */}
        <Card padded="md">
          <CardTitle>Look</CardTitle>
          <CardDescription className="mt-1">
            The accent color is your channel's signature — applied to the
            thumbnail's highlight word, the outro accent strip, and the
            Shorts CTA.
          </CardDescription>
          <div className="mt-5 space-y-4">
            <label className="block space-y-2">
              <span className="text-xs uppercase tracking-track text-mutedForeground">
                Accent color
              </span>
              <div className="flex items-center gap-3">
                <input
                  type="color"
                  value={preset.accent_color}
                  onChange={(e) => update({ accent_color: e.target.value })}
                  className="h-10 w-12 rounded-xl border border-white/[0.10] bg-transparent cursor-pointer"
                />
                <Input
                  value={preset.accent_color}
                  onChange={(e) => update({ accent_color: e.target.value })}
                  className="font-mono"
                />
              </div>
              <p className="text-xs text-mutedForeground">
                Pick something distinctive — mint, rose, electric blue, gold. This
                is what makes viewers say "that's the channel with the X thumbnails."
              </p>
            </label>
          </div>
        </Card>

        {/* Cold open */}
        <Card padded="md">
          <CardTitle>Cold open</CardTitle>
          <CardDescription className="mt-1">
            A 5-7 second teaser from the chapter's climax panel + a punchy
            one-liner before the title card.
          </CardDescription>
          <div className="mt-5 space-y-4">
            <label className="flex items-center justify-between gap-3 rounded-2xl border border-white/[0.08] bg-white/[0.03] px-4 py-3">
              <span className="text-sm">Enable cold open</span>
              <input
                type="checkbox"
                className="h-4 w-4 accent-[rgb(var(--p-accent))]"
                checked={preset.cold_open_enabled}
                onChange={(e) => update({ cold_open_enabled: e.target.checked })}
              />
            </label>
            <label className="block space-y-2">
              <span className="text-xs uppercase tracking-track text-mutedForeground">
                Hold duration (seconds)
              </span>
              <Input
                type="number"
                step="0.5"
                min="2"
                max="12"
                value={preset.cold_open_duration_seconds}
                onChange={(e) => update({ cold_open_duration_seconds: parseFloat(e.target.value) })}
                disabled={!preset.cold_open_enabled}
              />
            </label>
            <label className="flex items-center justify-between gap-3 rounded-2xl border border-white/[0.08] bg-white/[0.03] px-4 py-3">
              <span className="text-sm">Show title card after cold open</span>
              <input
                type="checkbox"
                className="h-4 w-4 accent-[rgb(var(--p-accent))]"
                checked={preset.title_card_enabled}
                onChange={(e) => update({ title_card_enabled: e.target.checked })}
              />
            </label>
            <label className="block space-y-2">
              <span className="text-xs uppercase tracking-track text-mutedForeground">
                Title card duration (seconds)
              </span>
              <Input
                type="number"
                step="0.5"
                min="1"
                max="6"
                value={preset.title_card_duration_seconds}
                onChange={(e) => update({ title_card_duration_seconds: parseFloat(e.target.value) })}
                disabled={!preset.title_card_enabled}
              />
            </label>
          </div>
        </Card>

        {/* Outro */}
        <Card padded="md">
          <CardTitle>Outro</CardTitle>
          <CardDescription className="mt-1">
            Subscribe-CTA card at the end of every video. The right half is
            kept empty so YouTube's end-screen overlay can sit there.
          </CardDescription>
          <div className="mt-5 space-y-4">
            <label className="flex items-center justify-between gap-3 rounded-2xl border border-white/[0.08] bg-white/[0.03] px-4 py-3">
              <span className="text-sm">Enable outro</span>
              <input
                type="checkbox"
                className="h-4 w-4 accent-[rgb(var(--p-accent))]"
                checked={preset.outro_enabled}
                onChange={(e) => update({ outro_enabled: e.target.checked })}
              />
            </label>
            <label className="block space-y-2">
              <span className="text-xs uppercase tracking-track text-mutedForeground">
                Outro message
              </span>
              <Textarea
                value={preset.outro_message}
                onChange={(e) => update({ outro_message: e.target.value })}
                disabled={!preset.outro_enabled}
                rows={3}
              />
            </label>
            <label className="block space-y-2">
              <span className="text-xs uppercase tracking-track text-mutedForeground">
                Outro duration (seconds)
              </span>
              <Input
                type="number"
                step="0.5"
                min="3"
                max="12"
                value={preset.outro_duration_seconds}
                onChange={(e) => update({ outro_duration_seconds: parseFloat(e.target.value) })}
                disabled={!preset.outro_enabled}
              />
            </label>
          </div>
        </Card>
      </div>

      <div className="flex justify-end">
        <Button onClick={save} disabled={saving} size="lg">
          {saving ? "Saving…" : "Save preset"}
        </Button>
      </div>
    </AppShell>
  );
}
