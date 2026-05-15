import type { HTMLAttributes, ReactNode } from "react";

import { cn } from "@/lib/utils";

/**
 * Badge - the universal status pill.
 *
 * Tone variants map 1:1 to the .p-pill-* helpers in globals.css so
 * Badge looks identical to ConfidencePill and the in-component pills
 * used throughout the timeline editor. One pill design across the app.
 */
export type BadgeTone = "neutral" | "ok" | "warn" | "fail" | "info" | "accent";

const TONE_CLASS: Record<BadgeTone, string> = {
  neutral: "p-pill",
  ok: "p-pill p-pill-ok",
  warn: "p-pill p-pill-warn",
  fail: "p-pill p-pill-fail",
  info: "p-pill p-pill-info",
  accent: "p-pill p-pill-accent"
};

interface BadgeProps extends HTMLAttributes<HTMLSpanElement> {
  tone?: BadgeTone;
  dot?: boolean;
  pulse?: boolean;
  children: ReactNode;
}

export function Badge({
  className,
  tone = "neutral",
  dot = false,
  pulse = false,
  children,
  ...props
}: BadgeProps) {
  return (
    <span className={cn(TONE_CLASS[tone], className)} {...props}>
      {dot && (
        <span
          aria-hidden
          className={cn(
            "inline-block h-1.5 w-1.5 rounded-full bg-current",
            pulse && "p-anim-breathe"
          )}
        />
      )}
      {children}
    </span>
  );
}
