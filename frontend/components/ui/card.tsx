import type { HTMLAttributes } from "react";

import { cn } from "@/lib/utils";

/**
 * Card — the liquid-glass surface every page composes against.
 * Uses the `.p-glass` primitive from globals.css so every Card gets the
 * top hairline highlight + layered shadow + backdrop blur, identical to
 * the timeline editor surfaces.
 *
 * Adjust padding via the `padded` prop:
 *   • md (default) = 24px — pages, sections
 *   • lg            = 32px — hero cards, primary content
 *   • sm            = 16px — tight metadata blocks
 */
type Padding = "sm" | "md" | "lg" | "none";

const PADDING: Record<Padding, string> = {
  none: "p-0",
  sm: "p-4",
  md: "p-6",
  lg: "p-8"
};

export function Card({
  className,
  padded = "md",
  ...props
}: HTMLAttributes<HTMLDivElement> & { padded?: Padding }) {
  return (
    <div
      className={cn("p-glass", PADDING[padded], className)}
      {...props}
    />
  );
}

export function CardTitle({ className, ...props }: HTMLAttributes<HTMLHeadingElement>) {
  return (
    <h3
      className={cn(
        "font-display text-lg md:text-xl text-foreground tracking-tightish",
        className
      )}
      {...props}
    />
  );
}

export function CardDescription({
  className,
  ...props
}: HTMLAttributes<HTMLParagraphElement>) {
  return (
    <p
      className={cn("text-sm text-mutedForeground leading-relaxed", className)}
      {...props}
    />
  );
}

export function CardSection({
  className,
  ...props
}: HTMLAttributes<HTMLDivElement>) {
  return <div className={cn("space-y-3", className)} {...props} />;
}
