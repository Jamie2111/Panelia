import type { Config } from "tailwindcss";

/**
 * Tailwind theme aligned to the Panelia "liquid memory" design tokens.
 *
 * The CSS custom properties (--p-*) defined in app/globals.css are the
 * single source of truth for colors. Tailwind tokens here are thin
 * adapters that reference those vars so every Tailwind class
 * (e.g. `bg-card`, `text-mutedForeground`, `border-border`) renders the
 * same glass-aware values as the raw .p-glass / .p-pill primitives.
 *
 * This is what lets existing pages adopt the new aesthetic without
 * touching every className — they already use `bg-card`, `text-foreground`,
 * etc., which now map to the liquid surfaces.
 */
const config: Config = {
  content: [
    "./app/**/*.{ts,tsx}",
    "./components/**/*.{ts,tsx}",
    "./lib/**/*.{ts,tsx}",
    "./store/**/*.{ts,tsx}"
  ],
  theme: {
    extend: {
      colors: {
        // Foundation — readable but never harsh.
        background: "rgb(var(--p-bg-base) / <alpha-value>)",
        foreground: "rgb(var(--p-text) / <alpha-value>)",

        // Surfaces — translucent glass layers.
        card: "rgb(var(--p-surface-1) / <alpha-value>)",
        cardForeground: "rgb(var(--p-text) / <alpha-value>)",
        muted: "rgb(var(--p-surface-2) / <alpha-value>)",
        mutedForeground: "rgb(var(--p-muted) / <alpha-value>)",

        // Hairlines.
        border: "rgb(var(--p-hairline) / <alpha-value>)",
        input: "rgb(var(--p-hairline) / <alpha-value>)",

        // Focus ring + accent — single mint family.
        ring: "rgb(var(--p-accent) / <alpha-value>)",
        accent: {
          DEFAULT: "rgb(var(--p-accent) / <alpha-value>)",
          foreground: "rgb(var(--p-bg-base) / <alpha-value>)",
          soft: "rgb(var(--p-accent) / 0.12)",
          ring: "rgb(var(--p-accent) / 0.35)"
        },

        // Semantic tones for status + brand callouts.
        ok: "rgb(var(--p-ok) / <alpha-value>)",
        warn: "rgb(var(--p-warn) / <alpha-value>)",
        fail: "rgb(var(--p-fail) / <alpha-value>)",
        info: "rgb(var(--p-info) / <alpha-value>)",

        brand: {
          amber: "rgb(var(--p-warn) / <alpha-value>)",
          cyan: "rgb(var(--p-info) / <alpha-value>)",
          rose: "rgb(var(--p-fail) / <alpha-value>)"
        }
      },
      borderRadius: {
        xl: "1.25rem",
        "2xl": "1.75rem",
        liquid: "var(--p-r-lg)"
      },
      boxShadow: {
        // Replaces the legacy "glow" with the layered liquid shadow.
        glow: "0 1px 0 0 rgb(255 255 255 / 0.06) inset, 0 24px 60px -20px rgb(0 0 0 / 0.55)",
        liquid:
          "0 1px 0 0 rgb(255 255 255 / 0.06) inset, 0 0 0 1px rgb(255 255 255 / 0.02) inset, 0 24px 60px -20px rgb(0 0 0 / 0.55)",
        "liquid-glow":
          "0 0 0 1px rgb(var(--p-accent) / 0.35), 0 0 32px -4px rgb(var(--p-accent) / 0.6)"
      },
      fontFamily: {
        display: [
          "Geist",
          "Inter",
          "var(--font-sora)",
          "ui-sans-serif",
          "system-ui",
          "sans-serif"
        ],
        sans: [
          "Geist",
          "Inter",
          "var(--font-dm-sans)",
          "ui-sans-serif",
          "system-ui",
          "sans-serif"
        ],
        mono: [
          "Geist Mono",
          "JetBrains Mono",
          "ui-monospace",
          "SFMono-Regular",
          "Menlo",
          "monospace"
        ]
      },
      letterSpacing: {
        tightish: "-0.005em",
        track: "0.18em"
      },
      transitionTimingFunction: {
        liquid: "cubic-bezier(0.32, 0.72, 0, 1)"
      },
      transitionDuration: {
        fast: "120ms",
        mid: "260ms",
        slow: "520ms"
      },
      backdropBlur: {
        liquid: "28px"
      }
    }
  },
  plugins: []
};

export default config;
