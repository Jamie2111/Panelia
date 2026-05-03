import type { Metadata } from "next";
import type { CSSProperties, ReactNode } from "react";
import Script from "next/script";

import "./globals.css";

export const metadata: Metadata = {
  title: "Panelia",
  description: "Create narrated manga recap videos from supported URLs or uploaded pages."
};

export default function RootLayout({ children }: Readonly<{ children: ReactNode }>) {
  return (
    <html lang="en" className="dark">
      <head>
        <Script id="panelia-client-error-trap" strategy="beforeInteractive">{`
          (function () {
            var chunkReloadKey = "__panelia_chunk_reload_at__";
            function toMessage(payload) {
              if (!payload) return "Unknown client error";
              if (typeof payload === "string") return payload;
              if (payload && typeof payload.message === "string" && payload.message) return payload.message;
              try { return JSON.stringify(payload); } catch (error) { return String(payload); }
            }
            function maybeRecoverFromChunkError(message) {
              if (!/ChunkLoadError|Loading chunk .* failed/i.test(String(message || ""))) return false;
              try {
                var lastAttempt = Number(sessionStorage.getItem(chunkReloadKey) || "0");
                var now = Date.now();
                if (!lastAttempt || now - lastAttempt > 15000) {
                  sessionStorage.setItem(chunkReloadKey, String(now));
                  var url = new URL(window.location.href);
                  url.searchParams.set("__chunk_reload__", String(now));
                  window.location.replace(url.toString());
                  return true;
                }
              } catch (error) {}
              return false;
            }
            function show(details) {
              try {
                var message = toMessage(details);
                if (maybeRecoverFromChunkError(message)) return;
                var stack = details && details.stack ? String(details.stack) : "";
                var extra = details && details.filename ? "\\n" + details.filename + (details.lineno ? ":" + details.lineno : "") + (details.colno ? ":" + details.colno : "") : "";
                var text = "Panelia client error\\n\\n" + message + extra + (stack ? "\\n\\n" + stack : "");
                window.__paneliaClientError = text;
                try { sessionStorage.setItem("__panelia_client_error__", text); } catch (error) {}
                var existing = document.getElementById("__panelia_client_error__");
                if (!existing) {
                  existing = document.createElement("pre");
                  existing.id = "__panelia_client_error__";
                  existing.style.position = "fixed";
                  existing.style.left = "16px";
                  existing.style.right = "16px";
                  existing.style.bottom = "16px";
                  existing.style.zIndex = "999999";
                  existing.style.maxHeight = "45vh";
                  existing.style.overflow = "auto";
                  existing.style.whiteSpace = "pre-wrap";
                  existing.style.padding = "12px 14px";
                  existing.style.borderRadius = "12px";
                  existing.style.border = "1px solid rgba(239,68,68,0.35)";
                  existing.style.background = "rgba(24,24,27,0.96)";
                  existing.style.color = "#fecaca";
                  existing.style.font = "12px/1.45 ui-monospace, SFMono-Regular, Menlo, monospace";
                  existing.style.boxShadow = "0 12px 40px rgba(0,0,0,0.45)";
                  document.addEventListener("DOMContentLoaded", function () {
                    if (!document.body.contains(existing)) {
                      document.body.appendChild(existing);
                    }
                  });
                  if (document.body) {
                    document.body.appendChild(existing);
                  }
                }
                existing.textContent = text;
              } catch (error) {}
            }
            window.addEventListener("error", function (event) {
              show({
                message: event && event.message ? event.message : "Unhandled error event",
                filename: event && event.filename,
                lineno: event && event.lineno,
                colno: event && event.colno,
                stack: event && event.error && event.error.stack ? event.error.stack : ""
              });
            });
            window.addEventListener("unhandledrejection", function (event) {
              var reason = event ? event.reason : null;
              show({
                message: reason && reason.message ? reason.message : toMessage(reason),
                stack: reason && reason.stack ? reason.stack : ""
              });
            });
            window.addEventListener("load", function () {
              try {
                sessionStorage.removeItem(chunkReloadKey);
              } catch (error) {}
            });
          })();
        `}</Script>
      </head>
      <body
        suppressHydrationWarning
        className="font-sans antialiased"
        style={
          {
            "--font-sora": "\"Avenir Next\", \"Segoe UI\", sans-serif",
            "--font-dm-sans": "\"SF Pro Display\", \"Avenir Next\", \"Segoe UI\", sans-serif"
          } as CSSProperties
        }
      >
        {children}
      </body>
    </html>
  );
}
