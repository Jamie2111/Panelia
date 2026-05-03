"use client";

import { useEffect } from "react";

import { AppShell } from "@/components/project/app-shell";
import { Button } from "@/components/ui/button";
import { Card, CardDescription, CardTitle } from "@/components/ui/card";

export default function Error({
  error,
  reset
}: {
  error: Error & { digest?: string };
  reset: () => void;
}) {
  useEffect(() => {
    console.error("Panelia route error", error);
  }, [error]);

  return (
    <AppShell
      title="Something went wrong"
      description="Panelia hit a frontend error while loading this page, but the app is still running."
    >
      <Card>
        <CardTitle>Page load failed</CardTitle>
        <CardDescription className="mt-2">
          We hit an unexpected client-side error while rendering this screen. Try reloading the route first. If it happens again, the backend and worker are still available.
        </CardDescription>
        <div className="mt-6 flex flex-wrap gap-3">
          <Button onClick={reset}>Try again</Button>
          <Button variant="secondary" onClick={() => window.location.assign("/")}>
            Go to dashboard
          </Button>
        </div>
        {error?.message ? (
          <p className="mt-4 rounded-2xl border border-red-500/20 bg-red-500/10 px-4 py-3 text-sm text-red-200">
            {error.message}
          </p>
        ) : null}
      </Card>
    </AppShell>
  );
}
