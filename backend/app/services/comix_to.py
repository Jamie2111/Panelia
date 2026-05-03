from __future__ import annotations

import math
import time
from pathlib import Path
from urllib.parse import urlparse

import requests
from requests import Response
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from app.core.config import get_settings
from app.schemas.project import ChapterMetadata
from app.services.chapter_selection import (
    ChapterCandidate,
    chapter_in_ranges,
    infer_is_official,
    parse_chapter_number,
    parse_range_spec,
    select_chapters,
)
from app.utils.files import ensure_dir


class ComixToService:
    def __init__(self) -> None:
        self.settings = get_settings()
        self.session = requests.Session()
        retry = Retry(
            total=self.settings.comix_retry_count,
            connect=self.settings.comix_retry_count,
            read=self.settings.comix_retry_count,
            backoff_factor=1,
            status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods=frozenset({"GET", "HEAD"}),
        )
        adapter = HTTPAdapter(max_retries=retry)
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)
        self.session.headers.update(
            {
                "Accept": "application/json, text/html;q=0.9,*/*;q=0.8",
                "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/135.0 Safari/537.36",
            }
        )

    def is_supported_url(self, url: str) -> bool:
        netloc = urlparse(url).netloc.casefold()
        return netloc in {"comix.to", "www.comix.to"}

    def parse_url(self, source_url: str) -> dict[str, str]:
        parsed = urlparse(source_url)
        parts = [part for part in parsed.path.split("/") if part]
        if len(parts) < 2 or parts[0] != "title":
            raise ValueError("Expected a comix.to title or chapter URL.")

        title_token = parts[1]
        if "-" not in title_token:
            raise ValueError("Unable to parse the comix.to title token from the provided URL.")
        hash_id, slug = title_token.split("-", 1)
        payload = {
            "hash_id": hash_id,
            "slug": slug,
            "title_path": f"/title/{hash_id}-{slug}",
            "source_url": source_url,
        }
        if len(parts) >= 3 and "-chapter-" in parts[2]:
            chapter_token = parts[2]
            chapter_id, chapter_label = chapter_token.split("-chapter-", 1)
            payload.update(
                {
                    "chapter_id": chapter_id,
                    "chapter_label": chapter_label,
                    "chapter_path": f"{payload['title_path']}/{chapter_token}",
                }
            )
        return payload

    def is_chapter_url(self, url: str) -> bool:
        try:
            return "chapter_id" in self.parse_url(url)
        except ValueError:
            return False

    def is_title_url(self, url: str) -> bool:
        try:
            return "chapter_id" not in self.parse_url(url)
        except ValueError:
            return False

    def chapter_metadata(self, source_url: str) -> ChapterMetadata:
        resolved = self.resolve_source_url(source_url)
        chapter_payload = self._get_json(f"{self.settings.comix_api_base}/chapters/{resolved['chapter_id']}")
        manga_payload = self._get_json(f"{self.settings.comix_api_base}/manga/{resolved['hash_id']}")
        return self._build_metadata(
            resolved_url=resolved["resolved_url"],
            chapter_payload=chapter_payload,
            manga_payload=manga_payload,
        )

    def download_chapter_pages(
        self,
        source_url: str,
        target_dir: Path,
        progress_callback: callable | None = None,
        cancel_callback: callable | None = None,
    ) -> ChapterMetadata:
        resolved = self.resolve_source_url(source_url)
        chapter_payload = self._get_json(f"{self.settings.comix_api_base}/chapters/{resolved['chapter_id']}")
        manga_payload = self._get_json(f"{self.settings.comix_api_base}/manga/{resolved['hash_id']}")
        image_entries = chapter_payload.get("images") or []
        if not image_entries:
            raise ValueError("This comix.to chapter did not expose any downloadable page images.")

        ensure_dir(target_dir)
        for index, image_entry in enumerate(image_entries, start=1):
            if cancel_callback:
                cancel_callback()
            image_url = str(image_entry.get("url") or "").strip()
            if not image_url:
                raise ValueError(f"Missing page URL for comix.to image {index}.")
            suffix = Path(urlparse(image_url).path).suffix or ".webp"
            destination = target_dir / f"{index:04d}{suffix}"
            self._download_with_retry(image_url, destination)
            if progress_callback:
                progress_callback(index / max(len(image_entries), 1) * 100, f"Downloaded page {index}/{len(image_entries)}")

        metadata = self._build_metadata(
            resolved_url=resolved["resolved_url"],
            chapter_payload=chapter_payload,
            manga_payload=manga_payload,
        )
        metadata.page_count = len(image_entries)
        return metadata

    def resolve_source_url(self, source_url: str) -> dict[str, str]:
        parsed = self.parse_url(source_url)
        if "chapter_id" in parsed:
            parsed["resolved_url"] = self._absolute_url(parsed["chapter_path"])
            return parsed

        candidate_numbers = self._candidate_chapter_numbers(parsed["hash_id"])
        title_url = self._absolute_url(parsed["title_path"])
        for chapter_number in candidate_numbers:
            chapter_label = self._chapter_number_label(chapter_number)
            guessed_url = f"{title_url}/{chapter_label}-chapter-{chapter_label}"
            final_url = self._follow_redirects(guessed_url)
            if not final_url or final_url.rstrip("/") == title_url.rstrip("/"):
                continue
            resolved = self.parse_url(final_url)
            if "chapter_id" in resolved:
                resolved["resolved_url"] = final_url
                return resolved

        raise ValueError(
            "This comix.to title URL could not be resolved into a chapter automatically. "
            "Please paste a direct comix.to chapter URL from the reader page."
        )

    def resolve_import_urls(
        self,
        urls: list[str],
        *,
        chapter_range: str | None = None,
        preferred_language: str | None = None,
        duplicate_mode: str = "auto_pick_best",
    ) -> list[str]:
        resolved_urls: list[str] = []
        for url in urls:
            cleaned = url.strip()
            if not cleaned:
                continue
            if self.is_chapter_url(cleaned):
                resolved_urls.append(self.resolve_source_url(cleaned)["resolved_url"])
                continue
            if self.is_title_url(cleaned):
                resolved_urls.extend(
                    self._resolve_title_urls(
                        cleaned,
                        chapter_range=chapter_range,
                        preferred_language=preferred_language,
                        duplicate_mode=duplicate_mode,
                    )
                )
                continue
            raise ValueError(f"Unsupported comix.to URL: {cleaned}")
        return resolved_urls

    def _candidate_chapter_numbers(self, hash_id: str) -> list[float]:
        chapter_indexes = self._get_json(f"{self.settings.comix_api_base}/manga/{hash_id}/chapter-indexes")
        values: list[float] = []
        for item in chapter_indexes:
            try:
                number = float(item.get("number"))
            except (TypeError, ValueError):
                continue
            if number <= 0 or math.isnan(number) or math.isinf(number):
                continue
            values.append(number)
        if not values:
            raise ValueError("No readable chapter indexes were found for this comix.to title.")
        return sorted(set(values))

    def _chapter_number_label(self, value: float) -> str:
        if float(value).is_integer():
            return str(int(value))
        text = f"{value}".rstrip("0").rstrip(".")
        return text

    def _chapter_route_token(self, chapter_label: str) -> str:
        return chapter_label.replace(".", "")

    def _build_metadata(
        self,
        *,
        resolved_url: str,
        chapter_payload: dict[str, object],
        manga_payload: dict[str, object],
    ) -> ChapterMetadata:
        chapter_number = chapter_payload.get("number")
        volume_number = chapter_payload.get("volume")
        chapter_name = str(chapter_payload.get("name") or "").strip() or None
        return ChapterMetadata(
            chapter_id=str(chapter_payload.get("chapter_id") or ""),
            source_url=resolved_url,
            manga_title=str(manga_payload.get("title") or "").strip() or None,
            chapter_title=chapter_name or (f"Chapter {chapter_number}" if chapter_number is not None else None),
            chapter_number=self._chapter_number_label(float(chapter_number)) if chapter_number is not None else None,
            volume_number=str(int(volume_number)) if isinstance(volume_number, (int, float)) and float(volume_number).is_integer() and int(volume_number) > 0 else None,
            language=str(chapter_payload.get("language") or "").strip() or None,
            page_count=len(chapter_payload.get("images") or []),
            raw={
                "manga": manga_payload,
                "chapter": chapter_payload,
            },
        )

    def _resolve_title_urls(
        self,
        title_url: str,
        *,
        chapter_range: str | None = None,
        preferred_language: str | None = None,
        duplicate_mode: str = "auto_pick_best",
    ) -> list[str]:
        parsed = self.parse_url(title_url)
        title_absolute = self._absolute_url(parsed["title_path"])
        chapter_numbers = self._candidate_chapter_numbers(parsed["hash_id"])
        ranges = parse_range_spec(chapter_range)
        if ranges:
            chapter_numbers = [number for number in chapter_numbers if chapter_in_ranges(number, ranges)]
        elif chapter_numbers:
            chapter_numbers = [chapter_numbers[0]]
        candidates: list[ChapterCandidate] = []
        for chapter_number in chapter_numbers:
            chapter_label = self._chapter_number_label(chapter_number)
            route_token = self._chapter_route_token(chapter_label)
            guessed_url = f"{title_absolute}/{route_token}-chapter-{chapter_label}"
            final_url = self._follow_redirects(guessed_url)
            if not final_url or final_url.rstrip("/") == title_absolute.rstrip("/"):
                continue
            resolved = self.parse_url(final_url)
            chapter_payload = self._get_json(f"{self.settings.comix_api_base}/chapters/{resolved['chapter_id']}")
            chapter_key, chapter_value = parse_chapter_number(chapter_payload.get("number"))
            if chapter_key is None or chapter_value is None:
                continue
            group_payload = chapter_payload.get("scanlation_group") or {}
            group_name = str(group_payload.get("name") or "").strip() or None
            candidates.append(
                ChapterCandidate(
                    source_url=final_url,
                    chapter_number_raw=str(chapter_payload.get("number") or chapter_label),
                    chapter_number_value=chapter_value,
                    chapter_key=chapter_key,
                    language=str(chapter_payload.get("language") or "").strip() or None,
                    group_name=group_name,
                    is_official=infer_is_official(group_name),
                    page_count=len(chapter_payload.get("images") or []),
                    updated_at=chapter_payload.get("updated_at") or chapter_payload.get("created_at"),
                    metadata=chapter_payload,
                )
            )

        selected = select_chapters(
            candidates,
            chapter_range=chapter_range,
            preferred_language=preferred_language,
            duplicate_mode=duplicate_mode,
            default_first_if_no_range=True,
        )
        if not selected:
            raise ValueError("No comix.to chapters matched the requested range and language.")
        return [candidate.source_url for candidate in selected]

    def _get_json(self, url: str) -> dict[str, object] | list[dict[str, object]]:
        response = self._request("GET", url)
        payload = response.json()
        if payload.get("status", 500) >= 400:
            raise ValueError(payload.get("message") or f"Comix request failed for {url}")
        return payload.get("result")

    def _download_with_retry(self, url: str, destination: Path) -> None:
        last_error: Exception | None = None
        for attempt in range(1, self.settings.comix_retry_count + 1):
            try:
                response = self._request("GET", url)
                destination.write_bytes(response.content)
                return
            except Exception as exc:  # pragma: no cover - network failures are environment-dependent
                last_error = exc
                time.sleep(min(float(2 ** (attempt - 1)), 10.0))
        raise RuntimeError(f"Failed to download comix.to page after retries: {url}") from last_error

    def _follow_redirects(self, url: str) -> str | None:
        try:
            response = self._request("GET", url)
        except Exception:
            return None
        return str(response.url)

    def _request(self, method: str, url: str) -> Response:
        response = self.session.request(
            method,
            url,
            timeout=self.settings.comix_timeout_seconds,
            allow_redirects=True,
        )
        response.raise_for_status()
        return response

    def _absolute_url(self, path: str) -> str:
        return f"{self.settings.comix_public_base.rstrip('/')}/{path.lstrip('/')}"
