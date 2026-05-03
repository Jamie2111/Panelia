from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Any
from uuid import uuid4


logger = logging.getLogger(__name__)

try:
    import numpy as np
except Exception:  # pragma: no cover - numpy is expected, but keep utils import-safe
    np = None


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def slugify(value: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9]+", "-", value.strip().lower()).strip("-")
    return cleaned or "project"


def write_json(path: Path, data: Any) -> None:
    ensure_dir(path.parent)
    _write_text_atomic(path, json.dumps(_json_safe(data), indent=2, ensure_ascii=False))


def read_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    raw = path.read_text(encoding="utf-8")
    if not raw.strip():
        return default
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        decoder = json.JSONDecoder()
        try:
            parsed, end = decoder.raw_decode(raw)
        except json.JSONDecodeError:
            if default is not None:
                logger.warning("Failed to decode JSON from %s: %s", path, exc)
                return default
            raise
        trailing = raw[end:].strip()
        if trailing:
            logger.warning("Recovered truncated/duplicated JSON from %s", path)
            try:
                write_json(path, parsed)
            except Exception:
                logger.exception("Failed to rewrite recovered JSON for %s", path)
        return parsed


def _write_text_atomic(path: Path, text: str) -> None:
    ensure_dir(path.parent)
    temp_path = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    temp_path.write_text(text, encoding="utf-8")
    os.replace(temp_path, path)


def _json_safe(value: Any) -> Any:
    if np is not None:
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, np.ndarray):
            return value.tolist()

    if isinstance(value, dict):
        return {str(_json_safe(key)): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value
