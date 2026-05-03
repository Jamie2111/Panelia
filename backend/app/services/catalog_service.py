from __future__ import annotations

from pathlib import Path

from app.schemas.project import CatalogOptionsResponse, LanguageOption, VoiceConfig, VoiceOption
from app.services.kokoro_service import KokoroTTSService
from app.services.project_store import ProjectStore
from app.utils.files import ensure_dir


class CatalogService:
    def __init__(self, store: ProjectStore | None = None) -> None:
        self.store = store or ProjectStore()
        self.kokoro = KokoroTTSService()
        self.preview_dir = ensure_dir(self.store.settings.data_dir / "previews")

    def get_options(self) -> CatalogOptionsResponse:
        return CatalogOptionsResponse(
            languages=LANGUAGE_OPTIONS,
            voices=VOICE_OPTIONS,
            music_tracks=self.store.list_music_tracks(),
        )

    def get_preview_path(self, voice: str, lang_code: str, speed: float) -> Path:
        safe_speed = str(speed).replace(".", "_")
        return self.preview_dir / f"{lang_code}_{voice}_{safe_speed}.wav"

    def ensure_voice_preview(self, voice: str, lang_code: str, speed: float) -> Path:
        preview_path = self.get_preview_path(voice, lang_code, speed)
        if preview_path.exists():
            return preview_path

        sample_text = LANGUAGE_SAMPLE_TEXT.get(lang_code, LANGUAGE_SAMPLE_TEXT["a"])
        self.kokoro.synthesize_to_file(
            sample_text,
            preview_path,
            VoiceConfig(voice=voice, lang_code=lang_code, speed=speed),
        )
        return preview_path


LANGUAGE_OPTIONS = [
    LanguageOption(
        code="a",
        label="American English",
        description="Natural US English narration with the widest set of polished Kokoro voices.",
        sample_text="Tonight, our manga recap opens with a quiet pause before everything erupts into motion.",
    ),
    LanguageOption(
        code="b",
        label="British English",
        description="Refined UK English voices that work well for calm, documentary-style narration.",
        sample_text="In this chapter recap, the tension builds slowly before the story turns on a single decisive moment.",
    ),
    LanguageOption(
        code="f",
        label="French",
        description="French narration for recaps aimed at francophone audiences.",
        sample_text="Dans ce recap manga, la tension monte peu a peu avant un retournement decisif.",
    ),
    LanguageOption(
        code="h",
        label="Hindi",
        description="Hindi narration options for accessible spoken recaps.",
        sample_text="Is manga recap mein kahani dheere dheere tez hoti hai aur phir ek bada mod aata hai.",
    ),
    LanguageOption(
        code="j",
        label="Japanese",
        description="Japanese narration voices for manga-first storytelling workflows.",
        sample_text="Kono manga recap wa, shizukana ma kara hajimari, yagate ookina tenkai e to mukaimasu.",
    ),
    LanguageOption(
        code="z",
        label="Mandarin Chinese",
        description="Mandarin narration options for short-form recap videos.",
        sample_text="Zhe ge manga huigu cong anjing de tingdun kaishi, ranhou hen kuai zhuanru gaoneng jieduan.",
    ),
]

LANGUAGE_SAMPLE_TEXT = {item.code: item.sample_text for item in LANGUAGE_OPTIONS}

VOICE_OPTIONS = [
    VoiceOption(id="af_bella", lang_code="a", label="Bella", description="Warm and expressive storyteller.", quality_note="A- overall from the official Kokoro voice list.", style_tags=["warm", "clear", "storytelling"]),
    VoiceOption(id="af_nicole", lang_code="a", label="Nicole", description="Smooth and polished narrator for general recaps.", quality_note="B- overall.", style_tags=["balanced", "clean"]),
    VoiceOption(id="af_aoede", lang_code="a", label="Aoede", description="Measured and calm delivery for recap channels.", quality_note="C+ overall.", style_tags=["calm", "steady"]),
    VoiceOption(id="af_kore", lang_code="a", label="Kore", description="Confident delivery that fits dramatic summaries.", quality_note="C+ overall.", style_tags=["dramatic", "focused"]),
    VoiceOption(id="af_sarah", lang_code="a", label="Sarah", description="Clear narration with a conversational feel.", quality_note="C+ overall.", style_tags=["friendly", "clean"]),
    VoiceOption(id="af_nova", lang_code="a", label="Nova", description="Lighter voice for upbeat recap pacing.", quality_note="C overall.", style_tags=["bright", "upbeat"]),
    VoiceOption(id="am_michael", lang_code="a", label="Michael", description="Neutral male narrator for recap-style voiceover.", quality_note="C+ overall.", style_tags=["neutral", "steady"]),
    VoiceOption(id="am_fenrir", lang_code="a", label="Fenrir", description="A stronger male delivery for action-heavy chapters.", quality_note="C+ overall.", style_tags=["bold", "dramatic"]),
    VoiceOption(id="am_puck", lang_code="a", label="Puck", description="Fast, energetic male narrator for short-form edits.", quality_note="C+ overall.", style_tags=["energetic", "quick"]),
    VoiceOption(id="bf_emma", lang_code="b", label="Emma", description="Best British English choice for polished narration.", quality_note="B- overall.", style_tags=["british", "polished"]),
    VoiceOption(id="bf_isabella", lang_code="b", label="Isabella", description="Soft British narration for slower pacing.", quality_note="C overall.", style_tags=["british", "soft"]),
    VoiceOption(id="bm_fable", lang_code="b", label="Fable", description="British male narrator with an editorial tone.", quality_note="C overall.", style_tags=["british", "editorial"]),
    VoiceOption(id="bm_george", lang_code="b", label="George", description="Straightforward British voiceover for explainers.", quality_note="C overall.", style_tags=["british", "clear"]),
    VoiceOption(id="ff_siwis", lang_code="f", label="Siwis", description="The main French Kokoro narrator option.", quality_note="B- overall.", style_tags=["french", "natural"]),
    VoiceOption(id="hf_alpha", lang_code="h", label="Alpha", description="Hindi female narration with a clean tone.", quality_note="C overall.", style_tags=["hindi", "clear"]),
    VoiceOption(id="hf_beta", lang_code="h", label="Beta", description="Hindi female narrator with a slightly fuller delivery.", quality_note="C overall.", style_tags=["hindi", "balanced"]),
    VoiceOption(id="hm_omega", lang_code="h", label="Omega", description="Hindi male narrator for recap voiceover.", quality_note="C overall.", style_tags=["hindi", "male"]),
    VoiceOption(id="hm_psi", lang_code="h", label="Psi", description="Hindi male option for conversational pacing.", quality_note="C overall.", style_tags=["hindi", "conversational"]),
    VoiceOption(id="jf_alpha", lang_code="j", label="Alpha JP", description="Best all-around Japanese narration choice.", quality_note="C+ overall.", style_tags=["japanese", "balanced"]),
    VoiceOption(id="jf_gongitsune", lang_code="j", label="Gongitsune", description="Gentle Japanese female voice for reflective recaps.", quality_note="C overall.", style_tags=["japanese", "gentle"]),
    VoiceOption(id="jf_tebukuro", lang_code="j", label="Tebukuro", description="Japanese female narration with a storybook feel.", quality_note="C overall.", style_tags=["japanese", "storybook"]),
    VoiceOption(id="jm_kumo", lang_code="j", label="Kumo", description="Japanese male option for steady recap pacing.", quality_note="C- overall.", style_tags=["japanese", "steady"]),
    VoiceOption(id="zf_xiaoxiao", lang_code="z", label="Xiaoxiao", description="Mandarin female narrator for general recap voiceover.", quality_note="D overall.", style_tags=["mandarin", "female"]),
    VoiceOption(id="zf_xiaoyi", lang_code="z", label="Xiaoyi", description="Mandarin female voice with slightly brighter delivery.", quality_note="D overall.", style_tags=["mandarin", "bright"]),
    VoiceOption(id="zm_yunjian", lang_code="z", label="Yunjian", description="Mandarin male narration option.", quality_note="D overall.", style_tags=["mandarin", "male"]),
    VoiceOption(id="zm_yunxi", lang_code="z", label="Yunxi", description="Mandarin male option for even pacing.", quality_note="D overall.", style_tags=["mandarin", "steady"]),
]
