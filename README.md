# Panelia

Panelia is a full-stack web app for turning MangaDex chapter URLs or uploaded manga pages into narrated recap videos. It combines a Next.js + Tailwind + shadcn-style frontend with a FastAPI backend, a Redis-backed worker queue, MAGI panel detection, Gemini narration generation, Kokoro TTS, and FFmpeg video rendering.

## Stack

- Frontend: Next.js, React, TailwindCSS, reusable shadcn-style UI primitives, Zustand editor state
- Backend: FastAPI, Pydantic, filesystem project store
- Jobs: Redis queue + Python worker
- AI/media pipeline: MAGI, Gemini API, Kokoro, FFmpeg

## Project structure

```text
frontend/              Next.js UI
backend/               FastAPI app, services, assets, Dockerfile
workers/               Background worker entrypoint
models/                Shared JSON schemas
services/              Prompt templates and service-level references
pipeline/              Stage docs and FFmpeg reference
```

Inside each generated project:

```text
project/
  pages/
  panels.json
  script.txt
  audio/
  video/
  jobs/
  source/
  thumbnails/
```

## Key features

- Create projects from a MangaDex chapter URL, ZIP, PDF, raw images, or folder uploads
- Normalize all inputs into ordered `pages/` images
- Detect panels with MAGI and store editable boxes in `panels.json`
- Review panels with drag, resize, add, delete, keep/remove, reorder, split, and merge tools
- Generate per-panel recap narration with Gemini
- Generate per-panel WAV narration with Kokoro
- Choose language and narrator from curated Kokoro dropdowns with instant voice previews
- Upload your own MP3 background tracks and preview them before rendering
- Render vertical motion-comic exports with a virtual camera, eased panel travel, and optional background music
- Re-open projects to edit panels, tweak narration, regenerate audio, or re-render video
- Merge finished videos into a single normalized export
- Cancel long-running background jobs

## Environment

Copy `.env.example` to `.env` and set at least:

```bash
cp .env.example .env
```

Important variables:

- `GEMINI_API_KEY`: required for live Gemini narration generation
- `GEMINI_MODEL`: defaults to `gemini-2.5-flash-lite`
- `REDIS_URL`: Redis queue connection string
- `FRONTEND_ORIGIN`: CORS origin for the UI
- `PANELIA_DATA_DIR`: where project data is written inside the backend container

## Docker setup

```bash
docker compose up --build
```

Services:

- Frontend: [http://localhost:3000](http://localhost:3000)
- Backend API: [http://localhost:8000](http://localhost:8000)
- Redis: `localhost:6379`

The frontend proxies backend and media requests through `/backend/*` by default, which avoids the browser-side `Failed to fetch` issues that show up when the UI and API are not on the exact same hostname.

## Local development without Docker

Backend:

```bash
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Worker:

```bash
python -m workers.worker
```

Frontend:

```bash
cd frontend
npm install
npm run dev
```

If your backend runs somewhere other than `http://127.0.0.1:8000`, set:

```bash
PANELIA_BACKEND_PROXY_TARGET=http://your-backend-host:8000
```

before starting the Next.js dev server.

## Pipeline overview

1. Ingestion saves the raw source into `project/source/` and normalizes images into `project/pages/`.
2. MAGI detects panel boxes and writes them into `project/panels.json`.
3. The editor lets you fix panel order and geometry before downstream stages.
4. Gemini receives chapter metadata plus kept panel order and writes `project/script.txt`.
5. Kokoro turns each narration line into a panel WAV file under `project/audio/`.
6. FFmpeg encodes a page-aware camera timeline with eased pans, tall-panel travel, synced narration, optional music, and saves the final file in `project/video/`.

## Example Gemini prompt

See [services/prompts/gemini-narration.md](/Users/jamieobala/Documents/Panelia/services/prompts/gemini-narration.md).

## FFmpeg commands

See [pipeline/ffmpeg-reference.md](/Users/jamieobala/Documents/Panelia/pipeline/ffmpeg-reference.md).

## Notes on MAGI and Kokoro integration

- MAGI is loaded through `transformers` with `trust_remote_code=True` using the upstream `ragavsachdeva/magi` model identifier.
- Kokoro is installed directly from the upstream GitHub repository in `backend/requirements.txt`.
- Built-in music preset slots are defined in [backend/assets/music/manifest.json](/Users/jamieobala/Documents/Panelia/backend/assets/music/manifest.json).
- Custom uploaded MP3 tracks are stored under [backend/data/music](/Users/jamieobala/Documents/Panelia/backend/data/music) and appear automatically in the music dropdowns.

## Recommended next steps

- Add authentication if you want multi-user project ownership
- Switch the queue to a managed worker system if you need horizontal scaling
- Add waveform editing or subtitles on top of the current project model
