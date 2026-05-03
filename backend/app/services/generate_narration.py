from __future__ import annotations

from pathlib import Path
from typing import Any

import soundfile as sf

from app.schemas.project import VoiceConfig
from app.services.audio_mastering import AudioMasteringService
from app.services.emotion_tagger import EmotionTagger
from app.services.kokoro_tts_engine import KokoroTTSEngine
from app.services.language_detector import LanguageDetector
from app.services.language_normalizer import LanguageNormalizer
from app.services.pronunciation_engine import PronunciationEngine
from app.services.story_preprocessor import StoryPreprocessor
from app.services.voice_clone import VoiceCloneService
from app.utils.files import write_json


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
    voice_sample_path: Path | None = None,
) -> dict[str, Any]:
    preprocessor = StoryPreprocessor()
    pronunciation = PronunciationEngine()
    emotion_tagger = EmotionTagger()
    language_normalizer = LanguageNormalizer()
    language_detector = LanguageDetector()
    kokoro = KokoroTTSEngine()
    voice_clone = VoiceCloneService()
    mastering = AudioMasteringService()

    if progress_callback:
        progress_callback(4, "Preparing cinematic narration lines")
    units = preprocessor.process(script, panel_ids=panel_ids)
    units = pronunciation.apply(units, custom_dictionary=pronunciation_dictionary, character_names=character_names)
    units = emotion_tagger.apply(units)
    effective_language_hint = _narration_language_hint(language_detector, voice_config, language_hint)
    units = language_normalizer.apply(units, voice_config, language_hint=effective_language_hint)

    if cancel_callback:
        cancel_callback()

    manifest = kokoro.synthesize_units(
        units,
        output_dir,
        voice_config,
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
    write_json(output_dir.parent / "output" / "enhanced_narration.json", report)
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
