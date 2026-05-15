from __future__ import annotations

from pathlib import Path
from typing import Any

import soundfile as sf

import logging

from app.schemas.project import VoiceConfig
from app.services.audio_mastering import AudioMasteringService
from app.services.edge_tts_engine import EdgeTTSEngine
from app.services.edge_tts_service import is_edge_voice
from app.services.emotion_tagger import EmotionTagger
from app.services.kokoro_tts_engine import KokoroTTSEngine
from app.services.language_detector import LanguageDetector
from app.services.language_normalizer import LanguageNormalizer
from app.services.narration_contamination_guard import NarrationContaminationGuard
from app.services.pronunciation_engine import PronunciationEngine
from app.services.story_preprocessor import StoryPreprocessor
from app.services.voice_clone import VoiceCloneService
from app.utils.files import write_json

logger = logging.getLogger(__name__)


def generate_narration(
    script: list[str],
    output_dir: Path,
    voice_config: VoiceConfig,
    panel_ids: list[str] | None = None,
    progress_callback: callable | None = None,
    cancel_callback: callable | None = None,
    language_hint: str | None = None,
    pronunciation_dictionary: dict[str, str] | None = None,
    character_names: list[str] | None = None,
    supported_character_names: list[str] | None = None,
    world_terms: list[str] | None = None,
    voice_sample_path: Path | None = None,
) -> dict[str, Any]:
    preprocessor = StoryPreprocessor()
    guard = NarrationContaminationGuard()
    pronunciation = PronunciationEngine()
    emotion_tagger = EmotionTagger()
    language_normalizer = LanguageNormalizer()
    language_detector = LanguageDetector()
    voice_clone = VoiceCloneService()
    mastering = AudioMasteringService()

    # Engine selection — Edge TTS handles every voice id we registered in
    # the catalog under the `edge_*` namespace; everything else still goes
    # through Kokoro. Kokoro is lazily constructed so projects that never
    # touch a Kokoro voice don't load its torch dependencies.
    use_edge = is_edge_voice(voice_config.voice)
    tts_engine = EdgeTTSEngine() if use_edge else KokoroTTSEngine()

    if progress_callback:
        progress_callback(3, "Checking narration contamination")
    artifact_path = output_dir.parent / "output" / "enhanced_narration.json"
    guard_result = guard.prepare(
        script,
        panel_ids=panel_ids,
        supported_character_names=supported_character_names,
        world_terms=world_terms,
        source_artifact_status="in_progress",
    )
    write_json(
        artifact_path,
        {
            "artifact_status": "in_progress",
            "script_ready": False,
            "qc_report": guard_result.report,
            "units": [],
            "manifest": {},
            "clone_report": {},
            "mastering_report": {},
        },
    )
    if guard_result.report.get("quarantined_units") or guard_result.report.get("contamination_remaining"):
        write_json(
            output_dir.parent / "output" / "enhanced_narration.qc_report.json",
            guard_result.report,
        )
        raise ValueError(
            "Narration contamination QC blocked audio generation: "
            f"{guard_result.report.get('quarantined_units', 0)} quarantined, "
            f"{guard_result.report.get('contamination_remaining', 0)} remaining."
        )

    if progress_callback:
        progress_callback(4, "Preparing cinematic narration lines")
    units = preprocessor.process(guard_result.script_lines, panel_ids=guard_result.panel_ids)
    if progress_callback:
        progress_callback(5, "Applying pronunciation rules")
    units = pronunciation.apply(units, custom_dictionary=pronunciation_dictionary, character_names=character_names)
    if progress_callback:
        progress_callback(6, "Tagging narration emotion")
    units = emotion_tagger.apply(units)
    if progress_callback:
        progress_callback(7, "Normalizing narration language")
    effective_language_hint = _narration_language_hint(language_detector, voice_config, language_hint)
    units = language_normalizer.apply(units, voice_config, language_hint=effective_language_hint)
    if progress_callback:
        progress_callback(9, "Preparing narration audio cache")

    if cancel_callback:
        cancel_callback()

    # Run the primary engine; if Edge TTS fails partway (Microsoft's
    # public endpoint can rate-limit) we transparently retry the whole
    # batch on Kokoro so the user always gets audio back.
    try:
        manifest = tts_engine.synthesize_units(
            units,
            output_dir,
            voice_config,
            progress_callback=(lambda progress, message: progress_callback(10 + progress * 0.68, message)) if progress_callback else None,
            cancel_callback=cancel_callback,
        )
    except Exception as primary_err:  # noqa: BLE001
        if not use_edge:
            raise
        logger.warning(
            "Edge TTS synthesis failed (%s); falling back to Kokoro for this run.",
            primary_err,
        )
        fallback_voice = voice_config.model_copy(update={
            "voice": "af_bella",
            "lang_code": "a",
        })
        manifest = KokoroTTSEngine().synthesize_units(
            units,
            output_dir,
            fallback_voice,
            progress_callback=(lambda progress, message: progress_callback(10 + progress * 0.68, message)) if progress_callback else None,
            cancel_callback=cancel_callback,
        )

    if cancel_callback:
        cancel_callback()

    clone_report = voice_clone.clone_directory(
        output_dir,
        voice_sample_path=voice_sample_path,
        progress_callback=(lambda progress, message: progress_callback(80 + progress * 0.08, message)) if progress_callback else None,
    )
    if progress_callback:
        progress_callback(88, "Mastering narration audio")
    mastering_report = mastering.master_directory(
        output_dir,
        progress_callback=(lambda progress, message: progress_callback(88 + progress * 0.12, message)) if progress_callback else None,
    )

    manifest_path = output_dir / "manifest.json"
    final_manifest = dict(manifest)
    for file_name, entry in final_manifest.items():
        wav_path = output_dir / file_name
        if wav_path.exists():
            entry["duration_seconds"] = _duration_seconds(wav_path)
    write_json(manifest_path, final_manifest)

    report = {
        "artifact_status": "completed",
        "script_ready": bool(guard_result.report.get("script_ready")),
        "qc_report": guard_result.report,
        "units": [
            {
                "panel_id": unit.panel_id,
                "raw_text": unit.raw_text,
                "story_text": unit.story_text,
                "spoken_text": unit.spoken_text,
                "language": unit.language,
                "emotion": unit.emotion,
                "metadata": unit.metadata,
            }
            for unit in units
        ],
        "clone_report": clone_report,
        "mastering_report": mastering_report,
        "manifest": final_manifest,
    }
    write_json(artifact_path, report)
    write_json(output_dir.parent / "output" / "enhanced_narration.qc_report.json", guard_result.report)
    return report


def _narration_language_hint(
    detector: LanguageDetector,
    voice_config: VoiceConfig,
    language_hint: str | None,
) -> str | None:
    configured = detector.normalize_language_code(voice_config.lang_code)
    # Keep narration aligned to the selected narrator language. Source-comic
    # language hints are useful for OCR/translation, but they should not cause
    # English TTS to suddenly switch accents on a few lines.
    if configured in {"en", "ja", "zh", "ko", "es", "pt", "fr", "de", "tr", "id", "it", "ro", "ca", "gl"}:
        return configured
    return detector.normalize_language_code(language_hint)


def _duration_seconds(path: Path) -> float:
    info = sf.info(str(path))
    return round(float(info.duration or 0.0), 2)
