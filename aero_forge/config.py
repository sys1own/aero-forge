"""Load project-level configuration and environment overrides."""

from __future__ import annotations

import json
import os
import threading
from contextlib import contextmanager
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional


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


@dataclass
class ConfigOverride:
    """Request-scoped configuration overrides.

    Instances can be passed directly to build/generation tasks or entered as a
    context manager to make the override thread-local and request-scoped without
    mutating global environment variables.
    """

    llm_provider: Optional[str] = None
    api_key: Optional[str] = None
    model: Optional[str] = None
    max_retries: Optional[int] = None
    cache_enabled: Optional[bool] = None
    max_iterations: Optional[int] = None
    compiler_flags: Optional[List[str]] = field(default_factory=list)
    target: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Return a uppercase-keyed settings dict suitable for ``resolve_settings``."""
        result: Dict[str, Any] = {}
        for key, value in asdict(self).items():
            if value is None:
                continue
            if key == "compiler_flags" and not value:
                continue
            result[key.upper()] = value
        return result


DEFAULTS: Dict[str, Any] = {
    "LLM_PROVIDER": "none",
    "MODEL": None,
    "API_KEY": None,
    "MAX_RETRIES": 3,
    "CACHE_ENABLED": True,
    "MAX_ITERATIONS": 5,
    "COMPILER_FLAGS": [],
    "TARGET": None,
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


_tls = threading.local()


def current_override() -> Optional[ConfigOverride]:
    """Return the active request-scoped override for this thread, if any."""
    stack: List[ConfigOverride] = getattr(_tls, "override_stack", [])
    return stack[-1] if stack else None


@contextmanager
def override(
    override: Optional[ConfigOverride] = None,
    **kwargs: Any,
) -> Generator[ConfigOverride, None, None]:
    """Push a request-scoped ``ConfigOverride`` for the current thread."""
    if override is None:
        override = ConfigOverride(**kwargs)
    if not hasattr(_tls, "override_stack"):
        _tls.override_stack = []
    _tls.override_stack.append(override)
    try:
        yield override
    finally:
        _tls.override_stack.pop()


def resolve_settings(
    file_config: Optional[Dict[str, Any]] = None,
    override: Optional[ConfigOverride] = None,
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
    env_provider = os.getenv("AERO_FORGE_LLM_PROVIDER")
    if env_provider:
        settings["LLM_PROVIDER"] = env_provider
    env_model = os.getenv("AERO_FORGE_MODEL")
    if env_model:
        settings["MODEL"] = env_model
    env_retries = _env_int("AERO_FORGE_MAX_RETRIES")
    if env_retries is not None:
        settings["MAX_RETRIES"] = env_retries
    env_cache = _env_bool("AERO_FORGE_CACHE_ENABLED")
    if env_cache is not None:
        settings["CACHE_ENABLED"] = env_cache
    env_max_iter = _env_int("AERO_FORGE_MAX_ITERATIONS")
    if env_max_iter is not None:
        settings["MAX_ITERATIONS"] = env_max_iter

    # Backward compat: AERO_FORGE_USE_LLM=false forces provider to none.
    env_use_llm = _env_bool("AERO_FORGE_USE_LLM")
    if env_use_llm is False:
        settings["LLM_PROVIDER"] = "none"

    # Request-scoped override (explicit or thread-local)
    active = override or current_override()
    if active is not None:
        for key, value in active.to_dict().items():
            if value is not None:
                settings[key] = value

    # Explicit overrides (e.g. CLI flags)
    for key, value in overrides.items():
        if value is not None:
            settings[key] = value

    return settings


__all__ = [
    "DEFAULTS",
    "ConfigOverride",
    "current_override",
    "find_config",
    "get",
    "load_config",
    "override",
    "resolve_settings",
]
