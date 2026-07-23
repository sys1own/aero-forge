"""Persistent cache for LLM-derived code fixes."""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Dict, Optional

logger = logging.getLogger("aero_forge.cache")


def _code_hash(code: str) -> str:
    return hashlib.sha256(code.encode("utf-8")).hexdigest()[:16]


def _normalize_error(error: str) -> str:
    return " ".join(error.strip().split()).lower()[:500]


class FixCache:
    """Disk-backed cache mapping (error_signature, code_hash) -> fixed code."""

    def __init__(self, path: Optional[Path] = None, enabled: bool = True):
        self.enabled = enabled
        self.path = path or (Path.home() / ".cache" / "aero-forge" / "fix_cache.json")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._cache: Dict[str, str] = {}
        if self.enabled and self.path.is_file():
            self._load()

    def _key(self, error: str, code: str) -> str:
        return f"{_normalize_error(error)}::{_code_hash(code)}"

    def _load(self) -> None:
        try:
            with self.path.open("r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, dict):
                    self._cache = data
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Could not load fix cache: %s", exc)
            self._cache = {}

    def _save(self) -> None:
        if not self.enabled:
            return
        try:
            with self.path.open("w", encoding="utf-8") as f:
                json.dump(self._cache, f, indent=2)
        except OSError as exc:
            logger.warning("Could not save fix cache: %s", exc)

    def get(self, error: str, code: str) -> Optional[str]:
        if not self.enabled:
            return None
        key = self._key(error, code)
        value = self._cache.get(key)
        if value is not None:
            logger.info("Fix cache hit for error signature %s", key[:40])
        return value

    def set(self, error: str, code: str, fixed_code: str) -> None:
        if not self.enabled:
            return
        key = self._key(error, code)
        self._cache[key] = fixed_code
        self._save()
        logger.info("Stored fix in cache for error signature %s", key[:40])

    def clear(self) -> None:
        self._cache.clear()
        if self.path.is_file():
            try:
                self.path.unlink()
            except OSError as exc:
                logger.warning("Could not remove fix cache: %s", exc)


__all__ = ["FixCache"]
