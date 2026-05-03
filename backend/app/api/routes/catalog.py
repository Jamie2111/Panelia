from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse

from app.services.catalog_service import CatalogService, LANGUAGE_SAMPLE_TEXT, VOICE_OPTIONS

router = APIRouter(prefix="/api/catalog", tags=["catalog"])

catalog = CatalogService()


@router.get("/options")
def get_catalog_options():
    return catalog.get_options()


@router.get("/voice-preview")
def get_voice_preview(
    voice: str = Query(...),
    lang_code: str = Query(...),
    speed: float = Query(default=1.0),
):
    if not any(option.id == voice and option.lang_code == lang_code for option in VOICE_OPTIONS):
        raise HTTPException(status_code=404, detail="Unknown voice or language combination.")
    if lang_code not in LANGUAGE_SAMPLE_TEXT:
        raise HTTPException(status_code=404, detail="Unknown language code.")

    try:
        preview_path = catalog.ensure_voice_preview(voice=voice, lang_code=lang_code, speed=speed)
    except Exception as exc:  # pragma: no cover - depends on local model/runtime setup
        raise HTTPException(status_code=500, detail=f"Unable to generate the Kokoro preview: {exc}") from exc

    return FileResponse(preview_path, media_type="audio/wav", filename=preview_path.name)


@router.post("/music-upload")
async def upload_music_track(
    file: UploadFile = File(...),
    track_name: str | None = Form(default=None),
    mood: str | None = Form(default=None),
):
    filename = file.filename or "uploaded-track.mp3"
    if Path(filename).suffix.lower() != ".mp3":
        raise HTTPException(status_code=400, detail="Only MP3 files are supported for uploaded music.")

    payload = await file.read()
    if not payload:
        raise HTTPException(status_code=400, detail="The uploaded MP3 file was empty.")

    try:
        return catalog.store.add_uploaded_music_track(filename, payload, track_name=track_name, mood=mood)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Unable to import the MP3 track: {exc}") from exc
