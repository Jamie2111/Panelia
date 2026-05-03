"""Direct runner to re-verify vision-first narration end-to-end for one project.

Clears only the narration-layer caches + (optionally) canonical_characters.json
and panel_vision_final.json so the portrait pass, rescue pass, and script
generation all rebuild with the new code. Keeps the expensive raw panel
vision extraction cached.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))

from app.pipeline.context import PipelineContext
from app.pipeline.stages import run_script_generation
from app.schemas.project import PipelineStage, StageStatus
from app.services.project_store import ProjectStore
from app.services.queue_service import QueueService


ARCHIVE_BY_PROJECT = {
    "darling-in-the-franxx-6f2b8388": "darling-in-the-franxx-hardcoded-archive-20260429",
    "codex-global-freeze-2h-ptbr-part-02-caef52f4": "codex-global-freeze-part-02-hardcoded-archive-20260429",
}


def _script_metrics(path: Path) -> dict[str, float | int | bool]:
    if not path.exists():
        return {"exists": False, "chars": 0, "segments": 0, "median_words": 0, "avg_sentences": 0.0}
    text = path.read_text(encoding="utf-8", errors="ignore")
    segments = [segment.strip() for segment in re.split(r"\n\s*\n", text) if segment.strip()]
    word_counts = [len(re.findall(r"\b[\w'-]+\b", segment)) for segment in segments]
    sentence_counts = [
        len([part for part in re.split(r"(?<=[.!?])\s+(?=[A-Z])", segment) if part.strip()])
        for segment in segments
    ]
    return {
        "exists": True,
        "chars": len(text),
        "segments": len(segments),
        "median_words": statistics.median(word_counts) if word_counts else 0,
        "avg_sentences": round(sum(sentence_counts) / len(sentence_counts), 3) if sentence_counts else 0.0,
    }


def _print_length_report(store: ProjectStore, project_id: str) -> None:
    project_dir = store._project_dir(project_id)
    live_path = project_dir / "output" / "narration_story.txt"
    live = _script_metrics(live_path)
    print(f"LENGTH live={json.dumps(live, sort_keys=True)}", flush=True)
    quality_path = project_dir / "output" / "script_quality.json"
    if quality_path.exists():
        try:
            quality = json.loads(quality_path.read_text(encoding="utf-8"))
            print(
                "QUALITY "
                f"score={quality.get('quality_score')} "
                f"should_block_tts={quality.get('should_block_tts')}",
                flush=True,
            )
        except Exception as exc:
            print(f"QUALITY unavailable error={exc}", flush=True)
    archive_id = ARCHIVE_BY_PROJECT.get(project_id)
    if not archive_id:
        return
    archive_path = store._project_dir(archive_id) / "output" / "narration_story.txt"
    archive = _script_metrics(archive_path)
    if not archive.get("exists") or not archive.get("chars"):
        return
    char_ratio = float(live.get("chars") or 0) / float(archive.get("chars") or 1)
    archive_median = float(archive.get("median_words") or 0)
    median_ratio = float(live.get("median_words") or 0) / archive_median if archive_median else 0.0
    print(
        "LENGTH_COMPARE "
        f"archive={archive_id} "
        f"char_ratio={char_ratio:.3f} "
        f"median_word_ratio={median_ratio:.3f} "
        f"avg_sentences={float(live.get('avg_sentences') or 0):.3f}",
        flush=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("project_id")
    parser.add_argument("--keep-portrait", action="store_true", help="Don't delete canonical_characters.json")
    parser.add_argument("--keep-rescue", action="store_true", help="Don't delete panel_vision_final.json")
    parser.add_argument("--full-refresh", action="store_true", help="force_refresh=True (nukes panel_vision.json too)")
    parser.add_argument(
        "--disable-multimodal-rescue",
        action="store_true",
        help="Skip late image-based rescue/repair loops; useful when verifying cached vision/script behavior.",
    )
    parser.add_argument(
        "--mode",
        default="vision_first",
        choices=["vision_first", "hybrid", "story", "panel"],
        help="narration_mode_override (default: vision_first)",
    )
    args = parser.parse_args()

    store = ProjectStore()
    project_dir = store._project_dir(args.project_id)
    output_dir = project_dir / "output"

    targets = [
        "narration_story.txt",
        "story_segments.json",
        "story_bible.json",
        "story_grounding.json",
        "scene_summaries.json",
        "panel_script_blocks.json",
        "panel_evidence.json",
        "page_vision_cache.json",
        "panel_captions_cache.json",
        "gemini_summary_cache.json",
        "script_quality.json",
    ]
    if not args.keep_portrait:
        targets.append("canonical_characters.json")
    if not args.keep_rescue:
        targets.append("panel_vision_final.json")

    for rel in targets:
        path = output_dir / rel
        if path.exists():
            path.unlink()
            print(f"removed {path.relative_to(project_dir)}", flush=True)

    queue = QueueService()
    payload: dict = {
        "direct_runner": True,
        "stop_after_stage": True,
        "narration_mode_override": args.mode,
    }
    if args.full_refresh:
        payload["force_refresh"] = True
        if args.keep_portrait:
            payload["keep_cached_portrait"] = True
    if args.disable_multimodal_rescue:
        payload["disable_multimodal_rescue"] = True
        payload["disable_image_repair"] = True
    job = store.create_job(args.project_id, PipelineStage.SCRIPT_GENERATION, payload=payload)
    store.update_stage_state(
        args.project_id,
        PipelineStage.SCRIPT_GENERATION,
        StageStatus.RUNNING,
        progress=0,
        message="Direct verify",
    )
    print(f"START project={args.project_id} job={job.id}", flush=True)

    context = PipelineContext(
        store=store,
        queue=queue,
        project_id=args.project_id,
        job_id=job.id,
        stage=PipelineStage.SCRIPT_GENERATION,
    )
    run_script_generation(context)
    print(f"DONE project={args.project_id} job={job.id}", flush=True)
    _print_length_report(store, args.project_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
