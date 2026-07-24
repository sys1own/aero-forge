"""Language routing for polyglot engine generation."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

SUPPORTED_LANGUAGES = frozenset({"rust", "python", "cpp"})
DEFAULT_LANGUAGE = "rust"

_LANGUAGE_BY_EXT = {
    ".rs": "rust",
    ".py": "python",
    ".pyi": "python",
    ".cpp": "cpp",
    ".cc": "cpp",
    ".cxx": "cpp",
    ".hpp": "cpp",
    ".c": "c",
    ".h": "c",
}


def infer_language(path: Path) -> str:
    """Infer a canonical language tag from a file extension."""
    return _LANGUAGE_BY_EXT.get(path.suffix.lower(), "unknown")


def resolve_target_language(
    context: Optional[Dict[str, Any]] = None,
    *,
    source_path: Optional[Path] = None,
    source_language: Optional[str] = None,
) -> str:
    """Resolve the target language for an engine build.

    Priority:
      1. Explicit ``context["frameworks"]["language"]``.
      2. ``source_language`` hint.
      3. File-extension inference from ``source_path``.
      4. Conservative default (``rust``).
    """
    context = context or {}
    frameworks = context.get("frameworks")
    if isinstance(frameworks, dict):
        declared = str(frameworks.get("language", "")).strip().lower()
        if declared in SUPPORTED_LANGUAGES:
            return declared

    if source_language and source_language.lower() in SUPPORTED_LANGUAGES:
        return source_language.lower()

    if source_path is not None:
        inferred = infer_language(Path(source_path))
        if inferred in SUPPORTED_LANGUAGES:
            return inferred

    return DEFAULT_LANGUAGE


def is_native_crate_language(language: str) -> bool:
    """True when the language compiles through a native crate-style build."""
    return language == "rust"


def is_python(language: str) -> bool:
    return language == "python"


def is_cpp(language: str) -> bool:
    return language == "cpp"
