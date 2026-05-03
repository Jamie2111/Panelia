from __future__ import annotations

import math
import re
from threading import Lock


class TextSimilarity:
    _MODEL = None
    _LOAD_LOCK = Lock()

    def similarity(self, left: str, right: str) -> float:
        left_text = str(left or "").strip()
        right_text = str(right or "").strip()
        if not left_text or not right_text:
            return 0.0
        model = self._load_model()
        if model is not None:
            try:
                embeddings = model.encode([left_text, right_text], normalize_embeddings=True)
                return float((embeddings[0] * embeddings[1]).sum())
            except Exception:
                pass
        return self._fallback_similarity(left_text, right_text)

    def too_similar(self, left: str, right: str, threshold: float = 0.85) -> bool:
        return self.similarity(left, right) > threshold

    def _load_model(self):
        if self.__class__._MODEL is not None:
            return self.__class__._MODEL
        with self.__class__._LOAD_LOCK:
            if self.__class__._MODEL is not None:
                return self.__class__._MODEL
            try:
                from sentence_transformers import SentenceTransformer

                self.__class__._MODEL = SentenceTransformer("all-MiniLM-L6-v2")
            except Exception:
                self.__class__._MODEL = None
            return self.__class__._MODEL

    def _fallback_similarity(self, left: str, right: str) -> float:
        left_counts = self._token_counts(left)
        right_counts = self._token_counts(right)
        if not left_counts or not right_counts:
            return 0.0
        shared = set(left_counts) & set(right_counts)
        numerator = sum(left_counts[token] * right_counts[token] for token in shared)
        left_norm = math.sqrt(sum(value * value for value in left_counts.values()))
        right_norm = math.sqrt(sum(value * value for value in right_counts.values()))
        if not left_norm or not right_norm:
            return 0.0
        return numerator / (left_norm * right_norm)

    def _token_counts(self, sentence: str) -> dict[str, int]:
        counts: dict[str, int] = {}
        for token in re.findall(r"[a-z']{3,}", sentence.casefold()):
            counts[token] = counts.get(token, 0) + 1
        return counts

