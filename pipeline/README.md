# Pipeline Stages

Panelia uses a Redis-backed worker queue with project-local assets on disk.

Stages:

1. `ingestion`
   Normalizes MangaDex downloads, PDFs, ZIP archives, and uploaded images into `project/pages/`.
2. `panel_detection`
   Runs MAGI and writes normalized panel boxes into `project/panels.json`.
3. `panel_review`
   Happens in the frontend editor and saves manual edits back to `panels.json`.
4. `script_generation`
   Sends the ordered kept panels and chapter metadata to Gemini and writes `project/script.txt`.
5. `narration_generation`
   Uses Kokoro to generate one WAV file per panel under `project/audio/`.
6. `video_rendering`
   Builds a virtual camera timeline over full pages, glides between panels with eased motion, scrolls through extra-tall webtoon panels, syncs narration, optionally mixes music, and writes the final export to `project/video/`.

Cancellation:

- Each worker stage receives a `cancel_callback`.
- The API marks the job for cancellation in Redis.
- The worker checks that flag between major stage operations and stops cleanly.
