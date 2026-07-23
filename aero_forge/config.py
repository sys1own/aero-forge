"""Load project-level ``accelerate.toml`` configuration."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional


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
