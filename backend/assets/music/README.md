# Built-in Music Slots

The app exposes three built-in background track slots through `manifest.json`:

- `Somewhere — Fuse`
- `French Fuse`
- `Somewhere Fuse`

Drop the matching audio files into this directory to make each preset available at runtime:

- `somewhere-fuse.mp3`
- `french-fuse.mp3`
- `somewhere-fuse-alt.mp3`

The manifest entries stay visible in the UI even when the files are missing so you can wire your own owned assets into those slots quickly.

The app also supports uploaded MP3 tracks directly from the UI. Those custom tracks are stored under `backend/data/music/` and appear alongside these built-in slots in the music dropdowns.
