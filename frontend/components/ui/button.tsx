"use client";

import * as React from "react";
import { cva, type VariantProps } from "class-variance-authority";

import { cn } from "@/lib/utils";

/**
 * Button - Liquid Memory + Notion blend.
 *
 * • Default = mint accent with soft glow halo (the only "loud" element)
 * • Secondary = glass pill (Notion-style ghost with a hint of depth)
 * • Ghost = invisible until hovered (Notion's hover-reveal pattern)
 * • Outline = hairline border, transparent fill
 *
 * All variants share the same easing, focus ring, and disabled handling
 * so the row always feels like a coherent family.
 */
const buttonVariants = cva(
  [
    "inline-flex items-center justify-center gap-2 whitespace-nowrap",
    "rounded-full px-4 py-2 text-sm font-medium",
    "transition-all duration-fast ease-liquid",
    "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent/40 focus-visible:ring-offset-0",
    "disabled:pointer-events-none disabled:opacity-50"
  ].join(" "),
  {
    variants: {
      variant: {
        default: [
          "bg-accent text-accent-foreground",
          "shadow-[0_0_24px_-4px_rgb(var(--p-accent)/0.5)]",
          "hover:shadow-[0_0_32px_-2px_rgb(var(--p-accent)/0.75)]",
          "hover:-translate-y-px"
        ].join(" "),
        secondary: [
          "bg-white/[0.06] text-foreground border border-white/[0.08]",
          "backdrop-blur-liquid",
          "hover:bg-white/[0.10] hover:border-white/[0.14]"
        ].join(" "),
        ghost: [
          "text-mutedForeground",
          "hover:bg-white/[0.06] hover:text-foreground"
        ].join(" "),
        outline: [
          "border border-white/[0.10] bg-transparent text-foreground",
          "hover:bg-white/[0.05] hover:border-white/[0.18]"
        ].join(" "),
        destructive: [
          "bg-fail/15 text-fail border border-fail/30",
          "hover:bg-fail/22 hover:border-fail/45"
        ].join(" ")
      },
      size: {
        default: "h-10",
        sm: "h-8 px-3 text-xs",
        lg: "h-12 px-6 text-[15px]",
        icon: "h-10 w-10 p-0"
      }
    },
    defaultVariants: {
      variant: "default",
      size: "default"
    }
  }
);

export interface ButtonProps
  extends React.ButtonHTMLAttributes<HTMLButtonElement>,
    VariantProps<typeof buttonVariants> {}

const Button = React.forwardRef<HTMLButtonElement, ButtonProps>(
  ({ className, variant, size, ...props }, ref) => (
    <button
      ref={ref}
      className={cn(buttonVariants({ variant, size, className }))}
      {...props}
    />
  )
);
Button.displayName = "Button";

export { Button, buttonVariants };
