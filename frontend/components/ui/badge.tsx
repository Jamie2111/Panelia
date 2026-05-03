import type { ReactNode } from "react";

import { cn } from "@/lib/utils";

export function Badge({
  className,
  children
}: {
  className?: string;
  children: ReactNode;
}) {
  return (
    <span className={cn("inline-flex items-center whitespace-nowrap rounded-full border border-white/10 bg-white/5 px-3 py-1 text-xs text-white", className)}>
      {children}
    </span>
  );
}
