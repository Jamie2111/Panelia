import * as React from "react";

import { cn } from "@/lib/utils";

export const Textarea = React.forwardRef<HTMLTextAreaElement, React.TextareaHTMLAttributes<HTMLTextAreaElement>>(
  ({ className, ...props }, ref) => (
    <textarea
      ref={ref}
      className={cn(
        "min-h-[120px] w-full rounded-[24px] border border-border bg-white/5 px-4 py-3 text-sm text-white outline-none placeholder:text-mutedForeground focus:ring-2 focus:ring-accent/50",
        className
      )}
      {...props}
    />
  )
);
Textarea.displayName = "Textarea";

