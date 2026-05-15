from __future__ import annotations

import hashlib
import io
import re
import shutil
import uuid
import zipfile
from pathlib import Path

import fitz
from fastapi import UploadFile
from PIL import Image

from app.schemas.project import ChapterMetadata, SourceType
from app.services.comix_to import ComixToService
from app.services.mangadex import MangaDexService
from app.services.project_store import ProjectStore
from app.utils.files import ensure_dir


class PageIngestionService:
    IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
    ARCHIVE_EXTENSIONS = {".zip", ".cbz"}

    def __init__(self, store: ProjectStore | None = None) -> None:
        self.store = store or ProjectStore()
        self.mangadex = MangaDexService()
        self.comix_to = ComixToService()

    # Maximum upload size: 200 MB per file
    _MAX_UPLOAD_BYTES = 200 * 1024 * 1024
    # Allowed file extensions for uploads
    _ALLOWED_UPLOAD_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".zip", ".cbz", ".pdf"}

    def save_upload_sources(self, project_id: str, files: list[UploadFile]) -> list[str]:
        """Save uploaded files to the project input directory.

        Security hardening:
        - Filename is replaced with a UUID to prevent path traversal attacks
        - Original extension is preserved but validated against an allowlist
        - File size is checked before writing to prevent disk exhaustion
        """
        source_dir = ensure_dir(self.store._project_dir(project_id) / "input")
        saved_paths: list[str] = []
        for index, upload in enumerate(files, start=1):
            original_name = Path(upload.filename or f"upload-{index}")
            # Only keep the suffix - the stem is replaced with a UUID
            suffix = original_name.suffix.lower()
            if suffix not in self._ALLOWED_UPLOAD_EXTENSIONS:
                suffix = ".bin"
            safe_name = f"{uuid.uuid4().hex}{suffix}"
            target = source_dir / safe_name
            upload.file.seek(0)
            written = 0
            with target.open("wb") as handle:
                for chunk in iter(lambda: upload.file.read(65536), b""):
                    written += len(chunk)
                    if written > self._MAX_UPLOAD_BYTES:
                        handle.close()
                        target.unlink(missing_ok=True)
                        raise ValueError(
                            f"Upload '{original_name.name}' exceeds {self._MAX_UPLOAD_BYTES // (1024*1024)} MB limit."
                        )
                    handle.write(chunk)
            saved_paths.append(str(target))
        return saved_paths

    def ingest(
        self,
        project_id: str,
        source_type: SourceType,
        source_reference: str | None = None,
        progress_callback: callable | None = None,
        cancel_callback: callable | None = None,
    ) -> ChapterMetadata:
        pages_dir = ensure_dir(self.store._project_dir(project_id) / "pages")
        shutil.rmtree(pages_dir, ignore_errors=True)
        ensure_dir(pages_dir)

        if source_type == SourceType.MANGADEX_URL and source_reference:
            chapter_urls = self._parse_mangadex_urls(source_reference)
            if not chapter_urls:
                raise ValueError("At least one MangaDex chapter URL is required.")
            if progress_callback:
                progress_callback(2, "Fetching MangaDex chapter metadata")
            raw_pages_dir = ensure_dir(self.store._project_dir(project_id) / "temp" / "raw_pages")
            shutil.rmtree(raw_pages_dir, ignore_errors=True)
            ensure_dir(raw_pages_dir)
            metadata, raw_page_paths = self._download_mangadex_urls(
                chapter_urls,
                raw_pages_dir,
                progress_callback=self._subprogress(progress_callback, 5, 72),
                cancel_callback=cancel_callback,
            )
            if progress_callback:
                progress_callback(74, "Normalizing downloaded pages")
            self._normalise_images(
                raw_page_paths,
                pages_dir,
                self._subprogress(progress_callback, 74, 100),
                cancel_callback,
            )
            shutil.rmtree(raw_pages_dir, ignore_errors=True)
            metadata.page_count = len(list(pages_dir.glob("*")))
            return metadata

        if source_type == SourceType.COMIX_TO_URL and source_reference:
            chapter_urls = self._parse_comix_urls(source_reference)
            if not chapter_urls:
                raise ValueError("At least one comix.to title or chapter URL is required.")
            if progress_callback:
                progress_callback(2, "Fetching comix.to chapter metadata")
            raw_pages_dir = ensure_dir(self.store._project_dir(project_id) / "temp" / "raw_pages")
            shutil.rmtree(raw_pages_dir, ignore_errors=True)
            ensure_dir(raw_pages_dir)
            metadata, raw_page_paths = self._download_comix_urls(
                chapter_urls,
                raw_pages_dir,
                progress_callback=self._subprogress(progress_callback, 5, 72),
                cancel_callback=cancel_callback,
            )
            if progress_callback:
                progress_callback(74, "Normalizing downloaded pages")
            self._normalise_images(
                raw_page_paths,
                pages_dir,
                self._subprogress(progress_callback, 74, 100),
                cancel_callback,
            )
            shutil.rmtree(raw_pages_dir, ignore_errors=True)
            metadata.page_count = len(list(pages_dir.glob("*")))
            return metadata

        metadata = ChapterMetadata(source_url=source_reference)
        source_dir = self.store._project_dir(project_id) / "input"
        if not source_dir.exists() or not any(source_dir.glob("**/*")):
            source_dir = self.store._project_dir(project_id) / "source"
        files = sorted(path for path in source_dir.glob("**/*") if path.is_file())
        if progress_callback:
            progress_callback(3, "Scanning uploaded files")
        self._normalise_uploaded_sources(files, pages_dir, progress_callback, cancel_callback)
        metadata.page_count = len(list(pages_dir.glob("*")))
        return metadata

    def _normalise_uploaded_sources(
        self,
        files: list[Path],
        pages_dir: Path,
        progress_callback: callable | None = None,
        cancel_callback: callable | None = None,
    ) -> None:
        extracted_images: list[Path] = []
        temp_root = ensure_dir(pages_dir.parent / "temp" / "extracted")
        shutil.rmtree(temp_root, ignore_errors=True)
        ensure_dir(temp_root)

        total_files = max(len(files), 1)
        for index, file_path in enumerate(files, start=1):
            if cancel_callback:
                cancel_callback()
            suffix = file_path.suffix.lower()
            if suffix in self.IMAGE_EXTENSIONS:
                extracted_images.append(file_path)
            elif suffix in self.ARCHIVE_EXTENSIONS:
                with zipfile.ZipFile(file_path) as archive:
                    archive.extractall(temp_root / file_path.stem)
                extracted_images.extend(
                    sorted(path for path in (temp_root / file_path.stem).glob("**/*") if path.suffix.lower() in self.IMAGE_EXTENSIONS)
                )
            elif suffix == ".pdf":
                extracted_images.extend(self._extract_pdf(file_path, temp_root / file_path.stem))
            if progress_callback:
                progress = 5 + index / total_files * 25
                progress_callback(progress, f"Prepared source {index}/{len(files)}")

        if progress_callback:
            progress_callback(32, "Normalizing pages")
        self._normalise_images(
            extracted_images,
            pages_dir,
            self._subprogress(progress_callback, 32, 100),
            cancel_callback,
        )

    def _extract_pdf(self, pdf_path: Path, target_dir: Path) -> list[Path]:
        ensure_dir(target_dir)
        document = fitz.open(pdf_path)
        extracted: list[Path] = []
        for page_index in range(document.page_count):
            page = document.load_page(page_index)
            pixmap = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
            output_path = target_dir / f"{page_index + 1:04d}.png"
            pixmap.save(output_path)
            extracted.append(output_path)
        return extracted

    def _normalise_images(
        self,
        source: list[Path] | Path,
        pages_dir: Path,
        progress_callback: callable | None = None,
        cancel_callback: callable | None = None,
    ) -> None:
        if isinstance(source, Path):
            files = sorted(path for path in source.glob("*") if path.suffix.lower() in self.IMAGE_EXTENSIONS)
        else:
            files = [path for path in source if path.exists()]

        if not files:
            raise ValueError("No supported page images were found in the provided source.")

        for index, file_path in enumerate(files, start=1):
            if cancel_callback:
                cancel_callback()
            with Image.open(file_path) as source_image:
                image = source_image.convert("RGB")
                output_path = pages_dir / f"{index:04d}.png"
                image.save(output_path, format="PNG", compress_level=1)
                if index == 1:
                    self.store.write_thumbnail(output_path.parent.parent.name, self._thumbnail_bytes(image))
            if progress_callback:
                progress_callback(index / max(len(files), 1) * 100, f"Normalised page {index}/{len(files)}")

    def _thumbnail_bytes(self, image: Image.Image) -> bytes:
        thumb = image.copy()
        thumb.thumbnail((640, 640))
        buffer = io.BytesIO()
        thumb.save(buffer, format="JPEG", quality=88)
        return buffer.getvalue()

    def _subprogress(self, progress_callback: callable | None, start: float, end: float) -> callable | None:
        if progress_callback is None:
            return None

        span = max(end - start, 0)

        def reporter(progress: float, message: str) -> None:
            clamped = min(max(progress, 0), 100)
            progress_callback(start + span * (clamped / 100), message)

        return reporter

    def _parse_mangadex_urls(self, source_reference: str) -> list[str]:
        return self._parse_url_lines(source_reference)

    def _parse_comix_urls(self, source_reference: str) -> list[str]:
        return self._parse_url_lines(source_reference)

    def _parse_url_lines(self, source_reference: str) -> list[str]:
        parts = re.split(r"[\n,]+", source_reference)
        urls: list[str] = []
        seen: set[str] = set()
        for part in parts:
            cleaned = part.strip()
            if not cleaned or cleaned in seen:
                continue
            seen.add(cleaned)
            urls.append(cleaned)
        return urls

    def _download_mangadex_urls(
        self,
        chapter_urls: list[str],
        raw_pages_dir: Path,
        progress_callback: callable | None = None,
        cancel_callback: callable | None = None,
    ) -> tuple[ChapterMetadata, list[Path]]:
        collected_paths: list[Path] = []
        chapter_metadata: list[ChapterMetadata] = []
        seen_chapter_opener_hashes: dict[str, str] = {}
        total_urls = max(len(chapter_urls), 1)

        for index, chapter_url in enumerate(chapter_urls, start=1):
            if cancel_callback:
                cancel_callback()
            chapter_dir = ensure_dir(raw_pages_dir / f"chapter_{index:03d}")
            metadata = self.mangadex.download_chapter_pages(
                chapter_url,
                chapter_dir,
                progress_callback=None,
                cancel_callback=cancel_callback,
            )
            chapter_paths = sorted(
                (path for path in chapter_dir.glob("*") if path.suffix.lower() in self.IMAGE_EXTENSIONS),
                key=lambda path: path.name.lower(),
            )
            chapter_paths, opener_removed = self._prune_repeated_chapter_opener_pages(
                chapter_paths,
                seen_chapter_opener_hashes,
            )
            if opener_removed:
                raw_payload = dict(metadata.raw) if isinstance(metadata.raw, dict) else {}
                dedupe_payload = dict(raw_payload.get("ingestion_dedupe", {}))
                dedupe_payload["repeated_chapter_opener_removed"] = True
                raw_payload["ingestion_dedupe"] = dedupe_payload
                metadata = metadata.model_copy(
                    update={
                        "page_count": len(chapter_paths),
                        "raw": raw_payload,
                    }
                )
            chapter_metadata.append(metadata)
            collected_paths.extend(chapter_paths)
            if progress_callback:
                progress_callback(index / total_urls * 100, f"Downloaded chapter {index}/{len(chapter_urls)}")

        return self._combine_chapter_metadata(chapter_metadata, chapter_urls), collected_paths

    def _download_comix_urls(
        self,
        chapter_urls: list[str],
        raw_pages_dir: Path,
        progress_callback: callable | None = None,
        cancel_callback: callable | None = None,
    ) -> tuple[ChapterMetadata, list[Path]]:
        collected_paths: list[Path] = []
        chapter_metadata: list[ChapterMetadata] = []
        total_urls = max(len(chapter_urls), 1)

        for index, chapter_url in enumerate(chapter_urls, start=1):
            if cancel_callback:
                cancel_callback()
            chapter_dir = ensure_dir(raw_pages_dir / f"chapter_{index:03d}")
            metadata = self.comix_to.download_chapter_pages(
                chapter_url,
                chapter_dir,
                progress_callback=None,
                cancel_callback=cancel_callback,
            )
            chapter_paths = sorted(
                (path for path in chapter_dir.glob("*") if path.suffix.lower() in self.IMAGE_EXTENSIONS),
                key=lambda path: path.name.lower(),
            )
            chapter_metadata.append(metadata)
            collected_paths.extend(chapter_paths)
            if progress_callback:
                progress_callback(index / total_urls * 100, f"Downloaded chapter {index}/{len(chapter_urls)}")

        return self._combine_chapter_metadata(chapter_metadata, chapter_urls, source_label="comix.to"), collected_paths

    def _prune_repeated_chapter_opener_pages(
        self,
        chapter_paths: list[Path],
        seen_opener_hashes: dict[str, str],
    ) -> tuple[list[Path], bool]:
        if not chapter_paths:
            return chapter_paths, False

        opener_hash = self._file_sha1(chapter_paths[0])
        if opener_hash not in seen_opener_hashes:
            seen_opener_hashes[opener_hash] = str(chapter_paths[0])
            return chapter_paths, False

        if len(chapter_paths) <= 1:
            return chapter_paths, False
        return chapter_paths[1:], True

    def _file_sha1(self, path: Path) -> str:
        digest = hashlib.sha1()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def _combine_chapter_metadata(
        self,
        chapters: list[ChapterMetadata],
        chapter_urls: list[str],
        *,
        source_label: str = "MangaDex",
    ) -> ChapterMetadata:
        if not chapters:
            return ChapterMetadata(source_url="\n".join(chapter_urls))

        first = chapters[0]
        total_pages = sum(chapter.page_count or 0 for chapter in chapters)
        same_series = len({chapter.manga_title for chapter in chapters if chapter.manga_title}) <= 1
        chapter_numbers = [chapter.chapter_number for chapter in chapters if chapter.chapter_number]
        chapter_titles = [chapter.chapter_title for chapter in chapters if chapter.chapter_title]

        if len(chapters) == 1:
            return first.model_copy(update={"page_count": total_pages, "source_url": chapter_urls[0]})

        if chapter_numbers:
            if len(chapter_numbers) == 1:
                combined_chapter_title = f"Combined chapter import ({chapter_numbers[0]})"
            else:
                combined_chapter_title = f"Combined chapters {chapter_numbers[0]}-{chapter_numbers[-1]}"
        elif chapter_titles:
            combined_chapter_title = f"Combined import ({len(chapters)} chapters)"
        else:
            combined_chapter_title = f"Combined import ({len(chapters)} {source_label} URLs)"

        return ChapterMetadata(
            chapter_id=first.chapter_id,
            source_url="\n".join(chapter_urls),
            manga_title=first.manga_title if same_series else f"Combined {source_label} import",
            chapter_title=combined_chapter_title,
            chapter_number=f"{chapter_numbers[0]}-{chapter_numbers[-1]}" if len(chapter_numbers) >= 2 else (chapter_numbers[0] if chapter_numbers else None),
            volume_number=first.volume_number if same_series else None,
            language=first.language,
            page_count=total_pages,
            raw={
                **(first.raw if isinstance(first.raw, dict) else {}),
                "chapters": [chapter.model_dump(mode="json") for chapter in chapters],
                "relationships": first.raw.get("relationships", []) if isinstance(first.raw, dict) else [],
            },
        )
