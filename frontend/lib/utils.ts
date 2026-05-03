import { clsx, type ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

export function formatRelativeDate(value: string) {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) {
    return "Unknown time";
  }
  return new Intl.DateTimeFormat("en-US", {
    month: "short",
    day: "numeric",
    hour: "numeric",
    minute: "2-digit"
  }).format(date);
}

export function stageLabel(stage: string) {
  return stage
    .split("_")
    .map((chunk) => chunk[0]?.toUpperCase() + chunk.slice(1))
    .join(" ");
}

export function buildMediaUrl(url?: string | null, cacheKey?: string | number | null) {
  if (!url) {
    return "";
  }
  const baseUrl = url.startsWith("http")
    ? url
    : process.env.NEXT_PUBLIC_MEDIA_BASE_URL
      ? `${process.env.NEXT_PUBLIC_MEDIA_BASE_URL}${url}`
      : `/backend${url}`;
  if (cacheKey === undefined || cacheKey === null || cacheKey === "") {
    return baseUrl;
  }
  const separator = baseUrl.includes("?") ? "&" : "?";
  return `${baseUrl}${separator}v=${encodeURIComponent(String(cacheKey))}`;
}
