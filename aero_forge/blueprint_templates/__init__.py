"""Reference blueprint.aero templates for the LLM planning pass."""

from __future__ import annotations

from pathlib import Path
from typing import Dict


def _template_dir() -> Path:
    return Path(__file__).parent


def list_templates() -> Dict[str, Path]:
    """Return a mapping of template name to ``.aero`` file path."""
    return {
        p.stem: p
        for p in _template_dir().glob("*.aero")
        if p.is_file()
    }


def load_template(name: str) -> str:
    """Return the raw text of a named reference blueprint template."""
    path = list_templates().get(name)
    if not path:
        raise KeyError(f"Unknown blueprint template: {name!r}")
    return path.read_text(encoding="utf-8")


def load_all_templates() -> str:
    """Return all reference templates concatenated for LLM context."""
    parts = []
    for name, path in sorted(list_templates().items()):
        parts.append(f"--- {name}.aero ---")
        parts.append(path.read_text(encoding="utf-8"))
        parts.append("")
    return "\n".join(parts)
