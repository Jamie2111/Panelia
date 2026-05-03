import type { Config } from "tailwindcss";

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
        border: "hsl(240 4% 18%)",
        input: "hsl(240 4% 18%)",
        ring: "hsl(174 72% 54%)",
        background: "hsl(240 17% 7%)",
        foreground: "hsl(0 0% 98%)",
        muted: "hsl(240 5% 14%)",
        mutedForeground: "hsl(240 5% 65%)",
        card: "hsl(240 10% 10%)",
        cardForeground: "hsl(0 0% 98%)",
        accent: {
          DEFAULT: "hsl(174 72% 54%)",
          foreground: "hsl(240 17% 7%)"
        },
        brand: {
          amber: "hsl(35 100% 61%)",
          cyan: "hsl(186 100% 62%)",
          rose: "hsl(346 100% 70%)"
        }
      },
      borderRadius: {
        xl: "1.25rem",
        "2xl": "1.75rem"
      },
      boxShadow: {
        glow: "0 0 0 1px rgba(255,255,255,0.05), 0 20px 80px rgba(0,0,0,0.45)"
      },
      fontFamily: {
        display: ["var(--font-sora)"],
        sans: ["var(--font-dm-sans)"]
      }
    }
  },
  plugins: []
};

export default config;

