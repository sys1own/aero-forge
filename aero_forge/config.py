"""Load project-level configuration and environment overrides."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional


def find_config(start: Optional[Path] = None) -> Optional[Path]:
    """Search the current directory and parents for ``accelerate.toml``."""
    directory = start or Path.cwd()
    for parent in [directory] + list(directory.parents):
        candidate = parent / "accelerate.toml"
        if candidate.is_file():
            return candidate
    return None


def load_config(path: Optional[Path] = None) -> Dict[str, Any]:
    """Parse a minimal subset of TOML used by accelerate configuration files."""
    if path is None:
        path = find_config()
    if path is None:
        return {}

    sections: Dict[str, Dict[str, Any]] = {}
    current: str | None = None
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        if line.startswith("[") and line.endswith("]"):
            current = line[1:-1].strip()
            sections.setdefault(current, {})
            continue
        if current is None or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        sections[current][key] = _parse_value(value)

    return sections


def _parse_value(value: str) -> Any:
    lowered = value.lower()
    if lowered in ("true", "yes", "on"):
        return True
    if lowered in ("false", "no", "off"):
        return False
    if (value.startswith("[") and value.endswith("]")) or (
        value.startswith("{") and value.endswith("}")
    ):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            pass
    try:
        if "." in value or "e" in value.lower():
            return float(value)
        return int(value)
    except ValueError:
        pass
    if (value.startswith('"') and value.endswith('"')) or (
        value.startswith("'") and value.endswith("'")
    ):
        return value[1:-1]
    return value


def get(config: Dict[str, Any], *keys: str, default: Any = None) -> Any:
    """Walk nested config dicts, returning ``default`` if any key is missing."""
    current = config
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


DEFAULTS: Dict[str, Any] = {
    "MODEL_PRIORITY": ["openrouter/free", "gpt-4"],
    "MAX_RETRIES": 3,
    "CACHE_ENABLED": True,
    "FALLBACK_MODEL": "openrouter/free",
    "USE_LLM": True,
    "MAX_ITERATIONS": 5,
}


def _env_list(name: str) -> Optional[List[str]]:
    value = os.getenv(name)
    if not value:
        return None
    return [part.strip() for part in value.split(",") if part.strip()]


def _env_bool(name: str) -> Optional[bool]:
    value = os.getenv(name)
    if value is None:
        return None
    return value.lower() in ("true", "1", "yes", "on")


def _env_int(name: str) -> Optional[int]:
    value = os.getenv(name)
    if value is None:
        return None
    try:
        return int(value)
    except ValueError:
        return None


def resolve_settings(
    file_config: Optional[Dict[str, Any]] = None,
    **overrides: Any,
) -> Dict[str, Any]:
    """Merge defaults, file config, environment variables, and explicit overrides."""
    file_config = file_config or {}
    settings = dict(DEFAULTS)

    # File config top-level keys
    for key in DEFAULTS:
        if key in file_config:
            settings[key] = file_config[key]

    # Environment overrides
    env_list = _env_list("AERO_FORGE_MODEL_PRIORITY")
    if env_list:
        settings["MODEL_PRIORITY"] = env_list
    env_retries = _env_int("AERO_FORGE_MAX_RETRIES")
    if env_retries is not None:
        settings["MAX_RETRIES"] = env_retries
    env_cache = _env_bool("AERO_FORGE_CACHE_ENABLED")
    if env_cache is not None:
        settings["CACHE_ENABLED"] = env_cache
    env_fallback = os.getenv("AERO_FORGE_FALLBACK_MODEL")
    if env_fallback:
        settings["FALLBACK_MODEL"] = env_fallback
    env_use_llm = _env_bool("AERO_FORGE_USE_LLM")
    if env_use_llm is not None:
        settings["USE_LLM"] = env_use_llm
    env_max_iter = _env_int("AERO_FORGE_MAX_ITERATIONS")
    if env_max_iter is not None:
        settings["MAX_ITERATIONS"] = env_max_iter

    # Explicit overrides (e.g. CLI flags)
    for key, value in overrides.items():
        if value is not None:
            settings[key] = value

    # Append fallback model to priority list if not already present.
    fallback = settings.get("FALLBACK_MODEL")
    priority = list(settings.get("MODEL_PRIORITY", []))
    if fallback and fallback not in priority:
        priority.append(fallback)
    settings["MODEL_PRIORITY"] = priority

    return settings


__all__ = [
    "DEFAULTS",
    "find_config",
    "get",
    "load_config",
    "resolve_settings",
]
