"""Load project-level configuration and environment overrides."""

from __future__ import annotations

import copy
import json
import os
import threading
from contextlib import contextmanager
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional


def get_aero_forge_dir() -> Path:
    """Return the root Aero-Forge data directory, creating it if necessary."""
    path = Path(os.getenv("AERO_FORGE_HOME", Path.home() / ".aero_forge"))
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_toolchains_dir() -> Path:
    """Return the directory used for auto-bootstrapped toolchains."""
    path = Path(os.getenv("AERO_FORGE_TOOLCHAINS", get_aero_forge_dir() / "toolchains"))
    path.mkdir(parents=True, exist_ok=True)
    return path


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
        """Return a uppercase-keyed settings dict suitable for ``resolve_settings``.

        Empty strings and empty lists are treated as "not set" so they do not
        shadow environment variables or file-level defaults.
        """
        result: Dict[str, Any] = {}
        for key, value in asdict(self).items():
            if value is None:
                continue
            if isinstance(value, str) and value.strip() == "":
                continue
            if key == "compiler_flags" and not value:
                continue
            result[key.upper()] = value
        return result


DEFAULTS: Dict[str, Any] = {
    "LLM_PROVIDER": None,
    "MODEL": None,
    "API_KEY": None,
    "MAX_RETRIES": 3,
    "CACHE_ENABLED": True,
    "MAX_ITERATIONS": 5,
    "COMPILER_FLAGS": [],
    "TARGET": None,
}

# Environment variable names that identify which provider an API key belongs to.
_PROVIDER_KEY_ENVS: Dict[str, List[str]] = {
    "deepseek": ["DEEPSEEK_API_KEY", "AERO_FORGE_API_KEY"],
    "openai": ["OPENAI_API_KEY", "AERO_FORGE_API_KEY"],
    "openrouter": ["OPENROUTER_API_KEY", "AERO_FORGE_API_KEY"],
    "gemini": ["GEMINI_API_KEY", "AERO_FORGE_API_KEY"],
}


def _resolve_api_key(provider: Optional[str]) -> Optional[str]:
    """Return the best available API key for *provider* from the environment."""
    if provider and provider.lower() not in ("none", "null", ""):
        for name in _PROVIDER_KEY_ENVS.get(provider.lower(), []):
            value = os.getenv(name)
            if value:
                return value
    # If no provider is pinned, prefer a provider-specific key over the generic
    # fallback so the engine infers the right endpoint.
    for names in _PROVIDER_KEY_ENVS.values():
        for name in names:
            if name == "AERO_FORGE_API_KEY":
                continue
            value = os.getenv(name)
            if value:
                return value
    return os.getenv("AERO_FORGE_API_KEY")


def resolve_llm_provider(explicit: Optional[str] = None) -> Optional[str]:
    """Resolve the LLM provider using the configured precedence.

    Precedence:
        1. *explicit* provider argument / CLI flag.
        2. ``AERO_FORGE_LLM_PROVIDER`` environment variable.
        3. Provider-specific API key environment variable (``DEEPSEEK_API_KEY``,
           ``OPENAI_API_KEY``, etc.).
        4. ``AERO_FORGE_API_KEY`` generic fallback (defaults to deepseek).

    An explicit value of ``none``/``null``/empty disables provider inference.
    """
    if explicit is not None:
        lowered = str(explicit).lower().strip()
        if lowered in ("", "none", "null"):
            return None
        return lowered
    env_provider = os.getenv("AERO_FORGE_LLM_PROVIDER")
    if env_provider:
        return env_provider.lower().strip()
    for provider, names in _PROVIDER_KEY_ENVS.items():
        for name in names:
            if name == "AERO_FORGE_API_KEY":
                # Generic key is a last resort; do not prefer it over a
                # provider-specific key.
                continue
            if os.getenv(name):
                return provider
    if os.getenv("AERO_FORGE_API_KEY"):
        return "deepseek"
    return None


class Tier(str, Enum):
    """LLM routing tiers."""

    FAST = "fast"
    REASONING = "reasoning"


DEFAULT_TIER_MODELS: Dict[str, Dict[str, str]] = {
    "deepseek": {
        "fast": "deepseek-v4-flash",
        "reasoning": "deepseek-chat",
    },
    "openai": {
        "fast": "gpt-4o-mini",
        "reasoning": "gpt-4o",
    },
    "gemini": {
        "fast": "gemini-2.5-flash",
        "reasoning": "gemini-2.5-pro",
    },
    "openrouter": {
        "fast": "anthropic/claude-3-haiku",
        "reasoning": "anthropic/claude-3.5-sonnet",
    },
}


def _merge_tier_env(
    tier: str,
    mapping: Dict[str, Dict[str, str]],
    provider: Optional[str] = None,
) -> Dict[str, Dict[str, str]]:
    """Merge a JSON or per-provider env override into the tier mapping."""
    value = os.getenv(f"AERO_LLM_TIER_{tier.upper()}")
    if not value:
        return mapping
    value = value.strip()
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        # Treat the raw value as a model name for the requested provider.
        if provider:
            parsed = {provider: value}
        else:
            return mapping
    if not isinstance(parsed, dict):
        return mapping
    merged = {k: dict(v) for k, v in mapping.items()}
    for key, val in parsed.items():
        if isinstance(val, dict):
            merged.setdefault(key, {})
            merged[key].update({t: str(m) for t, m in val.items()})
        elif provider and key == provider:
            merged.setdefault(provider, {})
            merged[provider][tier] = str(val)
        else:
            # key is a provider, val is a model name for this tier.
            merged.setdefault(key, {})[tier] = str(val)
    return merged


def resolve_tier_model(
    provider: str,
    tier: str,
    file_config: Optional[Dict[str, Any]] = None,
    override: Optional[ConfigOverride] = None,
) -> Optional[str]:
    """Return the model name for *provider* and *tier*.

    Resolution order:
    1. Explicit override.model, if set.
    2. Environment variable ``AERO_LLM_TIER_<TIER>`` (JSON mapping or raw model name).
    3. ``llm`` config section / ``tier_models`` config section.
    4. ``DEFAULT_TIER_MODELS``.
    """
    active = override or current_override()
    if active is not None and active.model:
        return active.model

    provider = provider.lower()
    tier = tier.lower()
    mapping = copy.deepcopy(DEFAULT_TIER_MODELS)
    mapping = _merge_tier_env("fast", mapping, provider=provider)
    mapping = _merge_tier_env("reasoning", mapping, provider=provider)

    file_config = file_config or {}
    # Support [tier_models] section with keys like ``deepseek_fast`` or nested dicts.
    tier_section = file_config.get("tier_models") or {}
    if isinstance(tier_section, dict):
        for key, val in tier_section.items():
            if isinstance(val, dict):
                mapping.setdefault(key, {}).update(val)
            elif "_" in key:
                p, t = key.split("_", 1)
                mapping.setdefault(p, {})[t] = str(val)
    # Support [llm] section with ``tier_fast`` / ``tier_reasoning`` JSON mappings.
    llm_section = file_config.get("llm") or {}
    if isinstance(llm_section, dict):
        for t in ("fast", "reasoning"):
            val = llm_section.get(f"tier_{t}")
            if val:
                try:
                    parsed = json.loads(val) if isinstance(val, str) else val
                except json.JSONDecodeError:
                    parsed = None
                if isinstance(parsed, dict):
                    for p, m in parsed.items():
                        mapping.setdefault(p, {})[t] = str(m)

    return mapping.get(provider, {}).get(tier)


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


def _coalesce_str(value: Any) -> Optional[str]:
    """Treat empty/whitespace-only strings as ``None``."""
    if value is None:
        return None
    if isinstance(value, str) and value.strip() == "":
        return None
    return value


def resolve_settings(
    file_config: Optional[Dict[str, Any]] = None,
    override: Optional[ConfigOverride] = None,
    **overrides: Any,
) -> Dict[str, Any]:
    """Merge defaults, file config, environment variables, and explicit overrides.

    Precedence (highest first):
        1. Request-scoped ``ConfigOverride`` / thread-local override.
        2. Explicit keyword arguments (e.g. CLI flags).
        3. Environment variables.
        4. ``[llm]`` block / top-level keys in ``accelerate.toml``.
        5. Defaults.
    """
    file_config = file_config or {}
    settings = dict(DEFAULTS)

    # File config top-level keys
    for key in DEFAULTS:
        if key in file_config:
            settings[key] = file_config[key]

    # Honor a nested [llm] section for provider/model/api_key if no top-level key.
    llm_section = file_config.get("llm") or {}
    if isinstance(llm_section, dict):
        if settings.get("LLM_PROVIDER") is None:
            settings["LLM_PROVIDER"] = _coalesce_str(llm_section.get("provider"))
        if settings.get("API_KEY") is None:
            settings["API_KEY"] = _coalesce_str(llm_section.get("api_key"))
        if settings.get("MODEL") is None:
            settings["MODEL"] = _coalesce_str(llm_section.get("model"))

    # Environment overrides: provider is resolved from explicit flags, environment,
    # and available API keys; API key is resolved from the matching env var.
    provider_from_file = _coalesce_str(settings.get("LLM_PROVIDER"))
    env_provider = _coalesce_str(os.getenv("AERO_FORGE_LLM_PROVIDER"))
    resolved_provider = resolve_llm_provider(env_provider or provider_from_file)
    settings["LLM_PROVIDER"] = resolved_provider or "none"

    env_api_key = _resolve_api_key(settings.get("LLM_PROVIDER"))
    if env_api_key:
        settings["API_KEY"] = env_api_key
    elif settings.get("API_KEY") is None and isinstance(llm_section, dict):
        settings["API_KEY"] = _coalesce_str(llm_section.get("api_key"))

    env_model = _coalesce_str(os.getenv("AERO_FORGE_MODEL"))
    if env_model:
        settings["MODEL"] = env_model
    elif settings.get("MODEL") is None and isinstance(llm_section, dict):
        settings["MODEL"] = _coalesce_str(llm_section.get("model"))

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
        settings["API_KEY"] = None

    # Explicit overrides (e.g. CLI flags) -- applied before request-scoped
    # overrides so that ``ConfigOverride`` still wins, but after environment so
    # explicit CLI flags can override environment defaults.
    for key, value in overrides.items():
        coerced = _coalesce_str(value)
        if coerced is not None:
            settings[key] = coerced

    # Request-scoped override (explicit or thread-local) takes highest precedence.
    active = override or current_override()
    if active is not None:
        for key, value in active.to_dict().items():
            if value is not None:
                settings[key] = value

    return settings


__all__ = [
    "DEFAULTS",
    "ConfigOverride",
    "current_override",
    "find_config",
    "get",
    "get_aero_forge_dir",
    "get_toolchains_dir",
    "load_config",
    "override",
    "resolve_settings",
    "DEFAULT_TIER_MODELS",
    "Tier",
    "resolve_tier_model",
]
