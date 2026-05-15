import * as React from "react";

import { cn } from "@/lib/utils";

/**
 * Textarea — glass surface, accent focus ring, comfortable reading width.
 * Resize vertical only to keep the page rhythm intact.
 */
export const Textarea = React.forwardRef<
  HTMLTextAreaElement,
  React.TextareaHTMLAttributes<HTMLTextAreaElement>
>(({ className, ...props }, ref) => (
  <textarea
    ref={ref}
    className={cn(
      "min-h-[120px] w-full resize-y rounded-2xl px-4 py-3",
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
Textarea.displayName = "Textarea";
