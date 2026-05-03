from __future__ import annotations

import json
import re
import subprocess
import time
from urllib.parse import quote
from pathlib import Path
from urllib.parse import urlparse

import requests
from requests import Response
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from app.core.config import get_settings
from app.schemas.project import ChapterMetadata
from app.services.chapter_selection import ChapterCandidate, infer_is_official, parse_chapter_number, select_chapters
from app.utils.files import ensure_dir


class MangaDexService:
    CHAPTER_ID_PATTERN = re.compile(r"/chapter/([0-9a-fA-F-]{32,36})")
    TITLE_ID_PATTERN = re.compile(r"/title/([0-9a-fA-F-]{32,36})")

    def __init__(self) -> None:
        self.settings = get_settings()
        self.session = requests.Session()
        retry = Retry(
            total=self.settings.mangadex_retry_count,
            connect=self.settings.mangadex_retry_count,
            read=self.settings.mangadex_retry_count,
            backoff_factor=1,
            status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods=frozenset({"GET"}),
        )
        adapter = HTTPAdapter(max_retries=retry)
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)

    def parse_chapter_id(self, chapter_url: str) -> str:
        match = self.CHAPTER_ID_PATTERN.search(chapter_url)
        if not match:
            raise ValueError("Unable to parse a MangaDex chapter id from the provided URL.")
        return match.group(1)

    def parse_title_id(self, title_url: str) -> str:
        match = self.TITLE_ID_PATTERN.search(title_url)
        if not match:
            raise ValueError("Unable to parse a MangaDex title id from the provided URL.")
        return match.group(1)

    def chapter_metadata(self, chapter_url: str) -> ChapterMetadata:
        chapter_id = self.parse_chapter_id(chapter_url)
        payload = self._get_json(
            f"{self.settings.mangadex_api_base}/chapter/{chapter_id}",
            params={"includes[]": ["manga", "scanlation_group", "user"]},
        )["data"]
        manga_title = None
        for relationship in payload.get("relationships", []):
            if relationship.get("type") == "manga":
                attributes = relationship.get("attributes", {})
                titles = attributes.get("title", {})
                manga_title = next(iter(titles.values()), None) if titles else None

        attributes = payload.get("attributes", {})
        return ChapterMetadata(
            chapter_id=chapter_id,
            source_url=chapter_url,
            manga_title=manga_title,
            chapter_title=attributes.get("title"),
            chapter_number=attributes.get("chapter"),
            volume_number=attributes.get("volume"),
            language=attributes.get("translatedLanguage"),
            raw=payload,
        )

    def download_chapter_pages(
        self,
        chapter_url: str,
        target_dir: Path,
        progress_callback: callable | None = None,
        cancel_callback: callable | None = None,
    ) -> ChapterMetadata:
        metadata = self.chapter_metadata(chapter_url)
        payload = self._get_json(f"{self.settings.mangadex_api_base}/at-home/server/{metadata.chapter_id}")
        base_url = payload["baseUrl"]
        chapter = payload["chapter"]
        page_files = chapter.get("data", [])
        metadata.page_count = len(page_files)

        ensure_dir(target_dir)
        for index, page_file in enumerate(page_files, start=1):
            if cancel_callback:
                cancel_callback()

            page_url = f"{base_url}/data/{chapter['hash']}/{page_file}"
            suffix = Path(page_file).suffix or ".jpg"
            destination = target_dir / f"{index:04d}{suffix}"

            self._download_with_retry(page_url, destination)
            if progress_callback:
                progress_callback(index / max(len(page_files), 1) * 100, f"Downloaded page {index}/{len(page_files)}")

        return metadata

    def _download_with_retry(self, url: str, destination: Path) -> None:
        last_error: Exception | None = None
        for attempt in range(1, self.settings.mangadex_retry_count + 1):
            try:
                response = self.session.get(url, timeout=self.settings.mangadex_timeout_seconds)
                response.raise_for_status()
                destination.write_bytes(response.content)
                return
            except Exception as exc:  # pragma: no cover - network failures are environment-dependent
                last_error = exc
                if self._download_with_curl(url, destination):
                    return
                time.sleep(self._retry_delay_seconds(attempt))
        if self._download_with_curl(url, destination):
            return
        raise RuntimeError(f"Failed to download MangaDex page after retries: {url}") from last_error

    def is_supported_url(self, url: str) -> bool:
        parsed = urlparse(url)
        return "mangadex" in parsed.netloc or parsed.netloc == urlparse(self.settings.mangadex_public_base).netloc

    def is_chapter_url(self, url: str) -> bool:
        return bool(self.CHAPTER_ID_PATTERN.search(url))

    def is_title_url(self, url: str) -> bool:
        return bool(self.TITLE_ID_PATTERN.search(url))

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
                resolved_urls.append(cleaned)
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
            raise ValueError(f"Unsupported MangaDex URL: {cleaned}")
        return resolved_urls

    def _resolve_title_urls(
        self,
        title_url: str,
        *,
        chapter_range: str | None = None,
        preferred_language: str | None = None,
        duplicate_mode: str = "auto_pick_best",
    ) -> list[str]:
        title_id = self.parse_title_id(title_url)
        candidates: list[ChapterCandidate] = []
        offset = 0
        limit = 500
        total = None
        while total is None or offset < total:
            params: dict[str, object] = {
                "limit": limit,
                "offset": offset,
                "includes[]": ["scanlation_group"],
                "order[chapter]": "asc",
                "contentRating[]": ["safe", "suggestive", "erotica", "pornographic"],
            }
            if preferred_language and preferred_language.strip().casefold() not in {"", "any"}:
                params["translatedLanguage[]"] = [preferred_language.strip().casefold()]
            payload = self._get_json(f"{self.settings.mangadex_api_base}/manga/{title_id}/feed", params=params)
            total = int(payload.get("total", 0))
            for item in payload.get("data", []):
                attributes = item.get("attributes", {})
                chapter_raw = attributes.get("chapter")
                chapter_key, chapter_value = parse_chapter_number(chapter_raw)
                if chapter_key is None or chapter_value is None:
                    continue
                group_name = None
                for relationship in item.get("relationships", []):
                    if relationship.get("type") == "scanlation_group":
                        group_name = (
                            relationship.get("attributes", {}).get("name")
                            or relationship.get("attributes", {}).get("altName")
                            or group_name
                        )
                        break
                candidates.append(
                    ChapterCandidate(
                        source_url=f"{self.settings.mangadex_public_base.rstrip('/')}/chapter/{quote(str(item.get('id') or ''))}",
                        chapter_number_raw=str(chapter_raw),
                        chapter_number_value=chapter_value,
                        chapter_key=chapter_key,
                        language=str(attributes.get("translatedLanguage") or "").strip() or None,
                        group_name=str(group_name or "").strip() or None,
                        is_official=infer_is_official(group_name),
                        page_count=int(attributes.get("pages") or 0) or None,
                        updated_at=attributes.get("updatedAt") or attributes.get("publishAt") or attributes.get("createdAt"),
                        metadata=item,
                    )
                )
            offset += limit
            if not payload.get("data"):
                break

        selected = select_chapters(
            candidates,
            chapter_range=chapter_range,
            preferred_language=preferred_language,
            duplicate_mode=duplicate_mode,
            default_first_if_no_range=True,
        )
        if not selected:
            raise ValueError("No MangaDex chapters matched the requested range and language.")
        return [candidate.source_url for candidate in selected]

    def _get_json(self, url: str, params: dict[str, object] | None = None) -> dict[str, object]:
        last_error: Exception | None = None
        for attempt in range(1, self.settings.mangadex_retry_count + 1):
            response: Response | None = None
            try:
                response = self.session.get(url, params=params, timeout=self.settings.mangadex_timeout_seconds)
                response.raise_for_status()
                return response.json()
            except Exception as exc:  # pragma: no cover - runtime network conditions vary
                last_error = exc
                curl_payload = self._fetch_with_curl(url, params=params)
                if curl_payload is not None:
                    return json.loads(curl_payload)
                time.sleep(self._retry_delay_seconds(attempt, response))
        curl_payload = self._fetch_with_curl(url, params=params)
        if curl_payload is not None:
            return json.loads(curl_payload)
        raise last_error if last_error is not None else RuntimeError(f"Failed to fetch MangaDex JSON: {url}")

    def _retry_delay_seconds(self, attempt: int, response: Response | None = None) -> float:
        if response is not None:
            retry_after = response.headers.get("Retry-After")
            if retry_after:
                try:
                    return max(1.0, float(retry_after))
                except ValueError:
                    pass
        return min(float(2 ** (attempt - 1)), 12.0)

    def _fetch_with_curl(self, url: str, params: dict[str, object] | None = None) -> str | None:
        prepared = requests.Request("GET", url, params=params).prepare()
        command = [
            "curl",
            "-fsSL",
            "--retry",
            str(max(self.settings.mangadex_retry_count, 2)),
            "--retry-delay",
            "2",
            "--connect-timeout",
            str(self.settings.mangadex_timeout_seconds),
            "--max-time",
            str(max(self.settings.mangadex_timeout_seconds * 2, 30)),
            prepared.url,
        ]
        try:
            completed = subprocess.run(command, check=True, capture_output=True, text=True)
        except Exception:
            return None
        return completed.stdout

    def _download_with_curl(self, url: str, destination: Path) -> bool:
        command = [
            "curl",
            "-fsSL",
            "--retry",
            str(max(self.settings.mangadex_retry_count, 2)),
            "--retry-delay",
            "2",
            "--connect-timeout",
            str(self.settings.mangadex_timeout_seconds),
            "--max-time",
            str(max(self.settings.mangadex_timeout_seconds * 2, 30)),
            "-o",
            str(destination),
            url,
        ]
        try:
            subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception:
            return False
        return destination.exists() and destination.stat().st_size > 0
