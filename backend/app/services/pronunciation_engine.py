from __future__ import annotations

import re
from typing import Any

from app.services.story_preprocessor import NarrationUnit


# ---------------------------------------------------------------------------
# Pinyin initial → English-phonetic approximation.
# Only includes initials that English TTS mispronounces.
# ---------------------------------------------------------------------------
_PINYIN_INITIALS: dict[str, str] = {
    "zh": "J",
    "ch": "Ch",
    "sh": "Sh",
    "q": "Ch",
    "x": "Sh",
    "j": "J",
    "z": "Dz",
    "c": "Ts",
    "r": "R",
}

# Pinyin finals → English-phonetic approximation.
# Covers finals that TTS struggles with; simple ones (a, o, e, i, u) are
# left alone when they already sound passable in English.
_PINYIN_FINALS: dict[str, str] = {
    # --- standalone vowels / common finals ---
    "i": "ee",
    "u": "oo",
    "ü": "oo",
    "v": "oo",          # common ascii substitute for ü
    "ai": "eye",
    "ei": "ay",
    "ao": "ow",
    "ou": "oh",
    "an": "an",
    "en": "un",
    "ang": "ahng",
    "eng": "ung",
    "ong": "ohng",
    "er": "ar",
    # --- i-compound finals ---
    "ia": "yah",
    "ie": "yeh",
    "iao": "yow",
    "iu": "yo",
    "ian": "yen",
    "in": "in",
    "iang": "yahng",
    "ing": "ing",
    "iong": "yohng",
    # --- u-compound finals ---
    "ua": "wah",
    "uo": "wo",
    "uai": "why",
    "ui": "way",
    "uan": "wan",
    "un": "wun",
    "uang": "wahng",
    # --- ü-compound finals ---
    "ue": "oo-eh",
    "üe": "oo-eh",
    "uan": "wan",       # also used with j/q/x where ü→u
    "un": "wun",
}

# Full-syllable overrides - for cases where the compositional approach
# doesn't produce the best result.  These take priority.
_PINYIN_OVERRIDES: dict[str, str] = {
    # Zh- initials
    "zhi": "Jir",
    "zhu": "Joo",
    "zhang": "Jahng",
    "zhao": "Jow",
    "zhen": "Jun",
    "zheng": "Jung",
    "zhong": "Johng",
    "zhou": "Joe",
    "zhe": "Juh",
    "zhan": "Jan",
    "zhuang": "Joo-ahng",
    # Q- initials
    "qi": "Chee",
    "qin": "Chin",
    "qing": "Ching",
    "qian": "Chee-en",
    "qiang": "Chee-ahng",
    "qiu": "Chee-oh",
    "qu": "Choo",
    "quan": "Choo-en",
    "que": "Choo-eh",
    # X- initials
    "xi": "Shee",
    "xin": "Shin",
    "xing": "Shing",
    "xian": "Shee-en",
    "xiang": "Shee-ahng",
    "xiao": "Shee-ow",
    "xie": "Shee-eh",
    "xiu": "Shee-oh",
    "xu": "Shoo",
    "xuan": "Shoo-en",
    "xue": "Shoo-eh",
    "xun": "Shoon",
    # J- initials
    "ji": "Jee",
    "jiu": "Jee-oh",
    "jie": "Jee-eh",
    "jian": "Jee-en",
    "jiang": "Jee-ahng",
    "jin": "Jin",
    "jing": "Jing",
    "ju": "Joo",
    "juan": "Joo-en",
    "jun": "Joon",
    # Z- initials
    "zi": "Dzuh",
    "zu": "Dzoo",
    "zai": "Dzai",
    "zao": "Dzow",
    "ze": "Dzuh",
    "zeng": "Dzung",
    "zong": "Dzohng",
    "zuo": "Dzwo",
    # C- initials
    "ci": "Tsuh",
    "cu": "Tsoo",
    "cai": "Tsai",
    "cao": "Tsow",
    "ce": "Tsuh",
    "ceng": "Tsung",
    "cong": "Tsohng",
    "cuo": "Tswo",
    "cui": "Tsway",
    # Sh- initials
    "shi": "Shir",
    "shu": "Shoo",
    "shang": "Shahng",
    "shao": "Shaow",
    "shen": "Shun",
    "sheng": "Shung",
    "shou": "Show",
    "shui": "Shway",
    "shan": "Shan",
    "she": "Shuh",
    # Ch- initials
    "chi": "Chir",
    "chu": "Choo",
    "chang": "Chahng",
    "chao": "Chaow",
    "chen": "Chun",
    "cheng": "Chung",
    "chong": "Chohng",
    "chou": "Cho",
    "chun": "Chwun",
    "chuan": "Chwan",
    # R- initial
    "ri": "Rih",
    "ru": "Roo",
    "rui": "Rway",
    "ruo": "Rwo",
    "ren": "Run",
    "rong": "Rohng",
    # Standalone vowels / y/w- initials
    "yi": "Yee",
    "yu": "Yoo",
    "ye": "Yeh",
    "yue": "Yoo-eh",
    "yuan": "Yoo-en",
    "you": "Yo",
    "yan": "Yen",
    "yang": "Yahng",
    "yin": "Yin",
    "ying": "Ying",
    "yong": "Yohng",
    # Common finals that standalone or follow simple initials
    "hao": "How",
    "peng": "Pung",
    "wei": "Way",
    "mei": "May",
    "lei": "Lay",
    "fei": "Fay",
    "bei": "Bay",
    "hui": "Hway",
    "gui": "Gway",
    "dui": "Dway",
    "sui": "Sway",
    "tui": "Tway",
    "luo": "Lwo",
    "guo": "Gwo",
    "hua": "Hwah",
    "tian": "Tee-en",
    "dian": "Dee-en",
    "lian": "Lee-en",
    "nian": "Nee-en",
    "mian": "Mee-en",
    "bian": "Bee-en",
    "pian": "Pee-en",
    "liang": "Lee-ahng",
    "niang": "Nee-ahng",
    "hai": "High",
    "tai": "Tie",
    "bai": "By",
    "lai": "Lie",
    "pai": "Pie",
    "kai": "Kye",
    "dai": "Die",
    "gai": "Guy",
    "mai": "My",
    "nai": "Nye",
    "sai": "Sigh",
    "wai": "Why",
    "kuai": "Kwhy",
    "guai": "Gwhy",
    "huai": "Hwhy",
    "liao": "Lee-ow",
    "miao": "Mee-ow",
    "biao": "Bee-ow",
    "piao": "Pee-ow",
    "tiao": "Tee-ow",
    "diao": "Dee-ow",
    "niao": "Nee-ow",
    "liu": "Lee-oh",
    "niu": "Nee-oh",
    "diu": "Dee-oh",
    "dong": "Dohng",
    "tong": "Tohng",
    "gong": "Gohng",
    "kong": "Kohng",
    "long": "Lohng",
    "nong": "Nohng",
    "song": "Sohng",
    "feng": "Fung",
    "deng": "Dung",
    "geng": "Gung",
    "heng": "Hung",
    "keng": "Kung",
    "leng": "Lung",
    "meng": "Mung",
    "neng": "Nung",
    "teng": "Tung",
    "tang": "Tahng",
    "lang": "Lahng",
    "gang": "Gahng",
    "hang": "Hahng",
    "kang": "Kahng",
    "mang": "Mahng",
    "nang": "Nahng",
    "pang": "Pahng",
    "fang": "Fahng",
    "dang": "Dahng",
    "sang": "Sahng",
    "wang": "Wahng",
    "bang": "Bahng",
    "huang": "Hwahng",
    "guang": "Gwahng",
    "kuang": "Kwahng",
    "zhuang": "Joo-ahng",
    "shuang": "Shwahng",
    "chuang": "Chwahng",
    "wan": "Wahn",
    "lan": "Lahn",
    "man": "Mahn",
    "nan": "Nahn",
    "fan": "Fahn",
    "gan": "Gahn",
    "han": "Hahn",
    "kan": "Kahn",
    "pan": "Pahn",
    "ban": "Bahn",
    "dan": "Dahn",
    "san": "Sahn",
    "tan": "Tahn",
    "guan": "Gwahn",
    "huan": "Hwahn",
    "kuan": "Kwahn",
    "duan": "Dwahn",
    "luan": "Lwahn",
    "suan": "Swahn",
    "tuan": "Twahn",
}

# Korean romanization patterns
_KOREAN_PHONETICS: dict[str, str] = {
    "hyun": "Hyoon",
    "joon": "June",
    "jun": "June",
    "yoon": "Yune",
    "eun": "Uhn",
    "seung": "Sung",
    "yeong": "Young",
    "jeong": "Jung",
    "gwang": "Kwahng",
    "cheol": "Chul",
    "ho": "Hoe",
    "hee": "Hee",
    "hyeok": "Hyuk",
    "seok": "Suk",
    "yeol": "Yul",
    "woong": "Woong",
    "gyeong": "Kyung",
    "cheon": "Chun",
    "hwan": "Hwahn",
}

# Japanese romanization patterns.
# Single-syllable entries fix vowel sounds English TTS misreads (hi→high, i→eye).
# Multi-character entries handle common long vowel spellings.
_JAPANESE_PHONETICS: dict[str, str] = {
    # --- Long-vowel spellings ---
    "ryu": "Ryoo",
    "ryuu": "Ryoo",
    "shou": "Show",
    "shuu": "Shoo",
    "jou": "Joe",
    "yuu": "Yoo",
    "kou": "Koh",
    "tou": "Toh",
    "sou": "Soh",
    "dou": "Doh",
    "mou": "Moh",
    "rou": "Roh",
    "bou": "Boh",
    "gou": "Goh",
    "hou": "Hoh",
    "nou": "Noh",
    "kyou": "Kyoh",
    "myou": "Myoh",
    "ryou": "Ryoh",
    "gyou": "Gyoh",
    "hyou": "Hyoh",
    "byou": "Byoh",
    "nyou": "Nyoh",
    # --- Consonant clusters TTS misreads ---
    "shi": "Shee",
    "chi": "Chee",
    "tsu": "Tsoo",
    "fu": "Foo",
    # --- Single syllables where the vowel is misread ---
    # "hi" → TTS says "high"; in Japanese it's "hee"
    "hi": "Hee",
    # standalone "i" at word start → TTS says "eye"; Japanese = "ee"
    "i": "Ee",
    # --- Common full-name overrides (whole-word, highest priority) ---
    # Listed here so the decomposer doesn't need to guess them.
    "hiro": "Hero",          # hi=hee + ro → sounds like "Hero"
    "ichigo": "Eecheego",    # i=ee + chi=chee + go=go
    "ichika": "Eecheeka",
    "miku": "Meekoo",
    "goro": "Goroh",
    "ikuno": "Eekoono",
    "kokoro": "Kokoroh",
    "futoshi": "Footoeshee",
    "zorome": "Zorohme",
    "nana": "Nahna",
    "naomi": "Naohmee",
    "mitsuru": "Meetsooru",
}


# ---------------------------------------------------------------------------
# Regex for extracting location-like nouns from narration
# Matches "Tian Hai City", "Mount Hua", "Bei Jing", etc.
# ---------------------------------------------------------------------------
_LOCATION_PATTERN = re.compile(
    r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)"
    r"\s+"
    r"(City|Town|Village|Mountain|Mount|River|Lake|Island|Temple|Palace|Province|District|Station|Tower|Hall|Gate|Road|Street|Bridge|Park|Forest|Bay|Port|Harbor|Harbour|Castle|Fortress|Kingdom|Empire|Academy|School|University|Sect|Clan|Guild)\b"
)


class PronunciationEngine:
    # Universal overrides for names/words English TTS consistently mispronounces.
    # Applied to every project before character-specific or custom dictionaries.
    # Key = text as it appears in narration; Value = phonetic spelling for TTS.
    DEFAULT_DICTIONARY: dict[str, str] = {
        # Japanese given names where "hi" is read as "high" by English TTS
        "Hiro": "Hero",
        "Hiroshi": "Heeroshee",
        "Hiroki": "Heerokee",
        "Hiromi": "Heeromee",
        "Hiroto": "Heeroto",
        # Japanese names where leading "I" is read as "eye"
        "Ichigo": "Eecheego",
        "Ichika": "Eecheeka",
        "Ikuno": "Eekoono",
        "Izuku": "Eezookoo",
        "Inosuke": "Eenosooke",
        # Other commonly mispronounced Japanese given names
        "Miku": "Meekoo",
        "Mikasa": "Meekahsa",
        "Goro": "Goroh",
        "Futoshi": "Footoeshee",
        "Zorome": "Zorohme",
        "Mitsuru": "Meetsooru",
        "Kokoro": "Kokoroh",
        "Naomi": "Naohmee",
        "Nezuko": "Nezookoh",
        "Tanjiro": "Tanjeeroh",
        "Zenitsu": "Zenittsoo",
        "Shinobu": "Sheenoboo",
        "Giyu": "Geeyoo",
        "Rengoku": "Rengokoo",
        "Tengen": "Tengen",
        "Muichiro": "Moo-eecheeroh",
        "Yoriichi": "Yoreeechee",
        # Darling in the FRANXX vocabulary. Without these, Edge/Azure
        # TTS spells "FRANXX" letter-by-letter ("fran ex ex"), pronounces
        # "Klaxosaur" as "klax-uh-soar" with a confused stress, and
        # mangles the squad-mech names.
        "FRANXX": "franks",
        "Franxx": "franks",
        "franxx": "franks",
        "Klaxosaur": "Klaxohsor",
        "Klaxosaurs": "Klaxohsors",
        "Strelizia": "Strelitzia",
        "Plantation": "Plantation",  # noop, but stops auto-phonetics from breaking it
        # Generic anime/sci-fi terms that English TTS misreads
        "mecha": "mekha",
        "Mecha": "Mekha",
    }

    def apply(
        self,
        units: list[NarrationUnit],
        custom_dictionary: dict[str, str] | None = None,
        character_names: list[str] | None = None,
    ) -> list[NarrationUnit]:
        # Collect names from all sources
        all_names = list(character_names or [])

        # Also extract location names from narration text
        for unit in units:
            for match in _LOCATION_PATTERN.finditer(unit.spoken_text):
                location_prefix = match.group(1).strip()
                if location_prefix and len(location_prefix) >= 2:
                    all_names.append(location_prefix)

        # Build phonetic dictionary from all collected names
        auto_phonetics = self._build_phonetic_dictionary(all_names)
        replacements = {
            **auto_phonetics,
            **self.DEFAULT_DICTIONARY,
            **{str(key): str(value) for key, value in (custom_dictionary or {}).items()
               if str(key).strip() and str(value).strip() and str(key).strip() != str(value).strip()},
        }
        ordered_terms = sorted(replacements.items(), key=lambda item: len(item[0]), reverse=True)
        for unit in units:
            spoken = unit.spoken_text
            for source, target in ordered_terms:
                spoken = re.sub(rf"\b{re.escape(source)}\b", target, spoken)
            unit.spoken_text = spoken
            unit.metadata["pronunciation_dictionary"] = replacements
        return units

    def _build_phonetic_dictionary(self, names: list[str]) -> dict[str, str]:
        """Auto-generate phonetic spellings for character/location names."""
        result: dict[str, str] = {}
        seen: set[str] = set()
        for name in names:
            clean = str(name or "").strip()
            if not clean or len(clean) < 2 or clean.lower() in seen:
                continue
            seen.add(clean.lower())
            phonetic = self._phonetic_for_name(clean)
            if phonetic and phonetic != clean:
                result[clean] = phonetic
        return result

    def _phonetic_for_name(self, name: str) -> str:
        """Convert a romanized CJK name to a TTS-friendly phonetic spelling."""
        parts = name.split()
        result_parts: list[str] = []
        any_changed = False
        for part in parts:
            # Skip common English words that appear in names like "City", "Mount"
            if part.lower() in {
                "city", "town", "village", "mountain", "mount", "river", "lake",
                "island", "temple", "palace", "province", "district", "station",
                "tower", "hall", "gate", "road", "street", "bridge", "park",
                "forest", "bay", "port", "the", "of", "and", "in", "at",
            }:
                result_parts.append(part)
                continue
            phonetic_part = self._phonetic_for_syllable(part)
            if phonetic_part != part:
                any_changed = True
            result_parts.append(phonetic_part)
        result = " ".join(result_parts)
        return result if any_changed else name

    def _phonetic_for_syllable(self, word: str) -> str:
        """Convert a single word/syllable to phonetic form."""
        lower = word.lower()

        # Check full-syllable overrides first (highest priority)
        if lower in _PINYIN_OVERRIDES:
            return self._match_case(word, _PINYIN_OVERRIDES[lower])

        # Check Korean and Japanese dictionaries
        for dictionary in (_KOREAN_PHONETICS, _JAPANESE_PHONETICS):
            if lower in dictionary:
                return self._match_case(word, dictionary[lower])

        # Try compositional pinyin: initial + final
        composed = self._compose_pinyin(lower)
        if composed:
            return self._match_case(word, composed)

        # Try decomposing multi-syllable words (e.g. "Yuqing" → "Yu" + "qing")
        decomposed = self._decompose_pinyin(lower)
        if decomposed:
            return self._match_case(word, decomposed)

        return word

    def _compose_pinyin(self, syllable: str) -> str | None:
        """Try initial+final decomposition for known problematic initials."""
        # Only convert syllables with initials that TTS mispronounces
        for initial in sorted(_PINYIN_INITIALS, key=len, reverse=True):
            if not syllable.startswith(initial):
                continue
            final = syllable[len(initial):]
            if not final:
                continue
            phonetic_initial = _PINYIN_INITIALS[initial]
            # Look up the final
            if final in _PINYIN_FINALS:
                phonetic_final = _PINYIN_FINALS[final]
                return phonetic_initial + phonetic_final
        return None

    def _decompose_pinyin(self, word: str) -> str | None:
        """Try to split a word into known pinyin syllables and convert each."""
        if len(word) < 3:
            return None

        all_syllables = {**_PINYIN_OVERRIDES}

        # Try all possible split points
        best: list[str] | None = None
        for split in range(2, len(word)):
            first = word[:split]
            second = word[split:]
            first_phonetic = all_syllables.get(first) or self._compose_pinyin(first)
            second_phonetic = all_syllables.get(second) or self._compose_pinyin(second)
            if first_phonetic and second_phonetic:
                best = [first_phonetic, second_phonetic]
                break
            # If only one half converts, still use it
            if first_phonetic and second in all_syllables:
                best = [first_phonetic, all_syllables[second]]
                break
            if first in all_syllables and second_phonetic:
                best = [all_syllables[first], second_phonetic]
                break
            # Try 3-way split for longer words
            if second and len(second) >= 4:
                for split2 in range(2, len(second)):
                    s2a = second[:split2]
                    s2b = second[split2:]
                    pa = all_syllables.get(first) or self._compose_pinyin(first)
                    pb = all_syllables.get(s2a) or self._compose_pinyin(s2a)
                    pc = all_syllables.get(s2b) or self._compose_pinyin(s2b)
                    if pa and pb and pc:
                        best = [pa, pb, pc]
                        break
                if best:
                    break

        if best:
            return "-".join(best)
        return None

    def _match_case(self, original: str, replacement: str) -> str:
        """Preserve capitalization pattern from original word."""
        if not original or not replacement:
            return replacement
        if original[0].isupper():
            return replacement[0].upper() + replacement[1:]
        return replacement
