import * as React from "react";

import { cn } from "@/lib/utils";

/**
 * Input — translucent glass with an accent focus ring.
 * Notion-style: nearly invisible at rest, lights up on focus.
 */
export const Input = React.forwardRef<
  HTMLInputElement,
  React.InputHTMLAttributes<HTMLInputElement>
>(({ className, ...props }, ref) => (
  <input
    ref={ref}
    className={cn(
      "h-11 w-full rounded-2xl px-4",
      "bg-white/[0.04] text-foreground",
      "border border-white/[0.08]",
      "placeholder:text-mutedForeground/70",
      "transition-colors duration-fast ease-liquid",
      "focus:outline-none focus:bg-white/[0.06] focus:border-accent/40",
      "focus:ring-2 focus:ring-accent/30",
      "disabled:opacity-50 disabled:cursor-not-allowed",
      className
    )}
    {...props}
  />
));
Input.displayName = "Input";
