import * as React from "react";

import { cn } from "@/lib/utils";

export const Input = React.forwardRef<HTMLInputElement, React.InputHTMLAttributes<HTMLInputElement>>(
  ({ className, ...props }, ref) => (
    <input
      ref={ref}
      className={cn(
        "h-11 w-full rounded-2xl border border-border bg-white/5 px-4 text-sm text-white outline-none placeholder:text-mutedForeground focus:ring-2 focus:ring-accent/50",
        className
      )}
      {...props}
    />
  )
);
Input.displayName = "Input";

