from __future__ import annotations

import re

from app.services.story_preprocessor import NarrationUnit


class EmotionTagger:
    def apply(self, units: list[NarrationUnit]) -> list[NarrationUnit]:
        for unit in units:
            text = f"{unit.raw_text} {unit.story_text}".casefold()
            emotion = "neutral narration"
            if self._contains(text, ("revenge", "payback", "punish", "destroy", "kill")):
                emotion = "revenge"
            elif self._contains(text, ("attack", "run", "crash", "fight", "slam", "burst", "explode")):
                emotion = "action"
            elif self._contains(text, ("frozen", "danger", "threat", "storm", "urgent", "tense", "panic")):
                emotion = "tension"
            elif self._contains(text, ("what", "impossible", "suddenly", "freeze", "stunned", "shocked")) or "!" in unit.raw_text:
                emotion = "shock"
            elif self._contains(text, ("mystery", "strange", "unknown", "shadow", "silent", "secret")):
                emotion = "mystery"
            elif self._contains(text, ("prepare", "plan", "carefully", "inventory", "calculate", "strategy")):
                emotion = "calm planning"
            unit.emotion = emotion
            unit.metadata["emotion_tag"] = f"[{emotion}]"
            unit.metadata["pause_bias_ms"] = self._pause_bias(emotion)
        return units

    def _contains(self, text: str, keywords: tuple[str, ...]) -> bool:
        return any(re.search(rf"\b{re.escape(keyword)}\b", text) for keyword in keywords)

    def _pause_bias(self, emotion: str) -> int:
        return {
            "tension": 90,
            "mystery": 120,
            "shock": 150,
            "revenge": 70,
            "calm planning": 40,
            "action": -40,
            "neutral narration": 0,
        }.get(emotion, 0)
