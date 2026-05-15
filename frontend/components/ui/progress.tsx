import { cn } from "@/lib/utils";

function wholeProgress(value: number) {
  const numeric = Number.isFinite(value) ? value : 0;
  const clamped = Math.max(0, Math.min(100, numeric));
  if (clamped <= 0) return 0;
  if (clamped >= 100) return 100;
  return Math.ceil(clamped);
}

/**
 * Progress — slim bar with a soft accent fill + glow.
 * When `shimmer` is true, the fill gets a slow shimmering sweep — used
 * for active stages where we want to convey "this is moving" without
 * relying on a percentage number.
 */
export function Progress({
  value,
  className,
  shimmer = false
}: {
  value: number;
  className?: string;
  shimmer?: boolean;
}) {
  const width = wholeProgress(value);
  return (
    <div
      className={cn(
        "relative h-1.5 w-full overflow-hidden rounded-full",
        "bg-white/[0.06] border border-white/[0.04]",
        className
      )}
    >
      <div
        className={cn(
          "h-full rounded-full bg-accent transition-all duration-mid ease-liquid",
          "shadow-[0_0_12px_-2px_rgb(var(--p-accent)/0.65)]",
          shimmer && "p-anim-shimmer-track"
        )}
        style={{ width: `${width}%` }}
      />
    </div>
  );
}
