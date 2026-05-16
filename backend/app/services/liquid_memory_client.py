"""
LiquidMemoryClient

Thin client for the Liquid Memory service. Used as the cross-project
"channel intelligence" layer: stores narration/character/thumbnail
patterns across every video you publish, so the next project can
recall what worked last time instead of starting from scratch.

The actual Liquid Memory API surface is filled in below the TODO
markers. Today this file ships with a NO-OP fallback so Panelia
works whether or not LM credentials are configured. When the
LIQUID_MEMORY_API_URL and LIQUID_MEMORY_API_KEY env vars are set,
all calls route to the real service; when they're not, the client
quietly returns empty results and Panelia behaves exactly as it
did before.

Configuration:
  LIQUID_MEMORY_API_URL   e.g. "https://api.liquid-memory.io/v1"
  LIQUID_MEMORY_API_KEY   API key from the Liquid Memory dashboard
  LIQUID_MEMORY_NAMESPACE (optional) per-channel namespace, default
                          falls back to channel_name from preset
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from typing import Any, Iterable

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MemoryRecord:
    """Lightweight record shape returned by recall()."""
    id: str
    content: str
    metadata: dict[str, Any]
    score: float = 0.0


class LiquidMemoryClient:
    """Pluggable client. Acts as a no-op when no credentials are set."""

    def __init__(
        self,
        api_url: str | None = None,
        api_key: str | None = None,
        namespace: str | None = None,
        timeout_seconds: float = 5.0,
    ) -> None:
        self.api_url = (api_url or os.environ.get("LIQUID_MEMORY_API_URL") or "").rstrip("/")
        self.api_key = api_key or os.environ.get("LIQUID_MEMORY_API_KEY") or ""
        self.namespace = namespace or os.environ.get("LIQUID_MEMORY_NAMESPACE") or "panelia-default"
        self.timeout_seconds = float(timeout_seconds)
        self._session: Any = None

    @property
    def enabled(self) -> bool:
        """True only when both URL and key are configured."""
        return bool(self.api_url) and bool(self.api_key)

    def _client(self):
        """Lazy HTTP session. Falls back to a stub when requests isn't
        available so the rest of Panelia still imports cleanly."""
        if self._session is not None:
            return self._session
        try:
            import requests  # type: ignore
        except ImportError:
            logger.warning("`requests` not installed; LiquidMemoryClient is no-op.")
            return None
        sess = requests.Session()
        sess.headers.update({
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "X-Panelia-Namespace": self.namespace,
        })
        self._session = sess
        return sess

    # ── Public surface (all no-op safe) ───────────────────────────────────

    def store(
        self,
        *,
        kind: str,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> str | None:
        """Store a single memory. Returns the new memory id or None
        if storage was skipped (client disabled, network failure, etc.).

        `kind` is a free-form category we use to organize:
          - "character_voice"   character speech patterns + voice id
          - "thumbnail_choice"  which variant the user picked + context
          - "hook_history"      cold-open lines that shipped, with CTR
          - "viewer_feedback"   pinned-comment topics that drove engagement
        """
        if not self.enabled:
            return None
        client = self._client()
        if client is None:
            return None
        payload = {
            "namespace": self.namespace,
            "kind": kind,
            "content": content,
            "metadata": dict(metadata or {}),
        }
        try:
            # TODO: confirm the exact endpoint path against your LM docs.
            response = client.post(
                f"{self.api_url}/memories",
                data=json.dumps(payload),
                timeout=self.timeout_seconds,
            )
            response.raise_for_status()
            data = response.json()
            return str(data.get("id") or data.get("memory_id") or "")
        except Exception as exc:  # noqa: BLE001
            logger.warning("LiquidMemory store(%s) failed: %s", kind, exc)
            return None

    def recall(
        self,
        *,
        query: str,
        kind: str | None = None,
        limit: int = 5,
    ) -> list[MemoryRecord]:
        """Retrieve relevant memories by semantic similarity to `query`.
        Returns [] if disabled or the call fails."""
        if not self.enabled:
            return []
        client = self._client()
        if client is None:
            return []
        params = {
            "namespace": self.namespace,
            "query": query,
            "limit": max(1, int(limit)),
        }
        if kind:
            params["kind"] = kind
        try:
            # TODO: confirm "search" vs "recall" vs "query" endpoint.
            response = client.get(
                f"{self.api_url}/memories/search",
                params=params,
                timeout=self.timeout_seconds,
            )
            response.raise_for_status()
            raw = response.json()
            items = raw if isinstance(raw, list) else raw.get("results", [])
            results: list[MemoryRecord] = []
            for item in items[:limit]:
                if not isinstance(item, dict):
                    continue
                results.append(MemoryRecord(
                    id=str(item.get("id") or item.get("memory_id") or ""),
                    content=str(item.get("content") or ""),
                    metadata=dict(item.get("metadata") or {}),
                    score=float(item.get("score") or item.get("similarity") or 0.0),
                ))
            return results
        except Exception as exc:  # noqa: BLE001
            logger.warning("LiquidMemory recall(%s) failed: %s", kind, exc)
            return []

    def bulk_store(self, records: Iterable[dict[str, Any]]) -> int:
        """Optional convenience: batch insert. Returns count stored."""
        if not self.enabled:
            return 0
        count = 0
        for rec in records:
            kind = rec.get("kind") or "uncategorized"
            content = rec.get("content") or ""
            if not content:
                continue
            if self.store(kind=kind, content=content, metadata=rec.get("metadata")):
                count += 1
        return count
