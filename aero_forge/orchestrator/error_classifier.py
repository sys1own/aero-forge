"""Classify build/test/API errors as transient, recoverable, or fatal."""

from __future__ import annotations

import ast
import re
import traceback
from enum import Enum
from pathlib import Path
from typing import Any, List, Optional


class ErrorClass(Enum):
    TRANSIENT = "transient"
    RECOVERABLE = "recoverable"
    FATAL = "fatal"


_TRANSIENT_PATTERNS = [
    re.compile(r"rate.?limit", re.I),
    re.compile(r"too many requests", re.I),
    re.compile(r"timeout", re.I),
    re.compile(r"connection", re.I),
    re.compile(r"network", re.I),
    re.compile(r"temporary", re.I),
    re.compile(r"service unavailable", re.I),
    re.compile(r"503|500|502|504"),
    re.compile(r"APIConnectionError|APITimeoutError|APIError", re.I),
]

_FATAL_PATTERNS = [
    re.compile(r"missing rust toolchain|cargo.*not found|rustc.*not found", re.I),
    re.compile(r"no linker found|no usable m4|m4.*not found", re.I),
    re.compile(r"out of (memory|disk)|no space left", re.I),
    re.compile(r"internal compiler error|rustc.*panicked", re.I),
    re.compile(r"permission denied", re.I),
    re.compile(r"manifest.*not found", re.I),
    re.compile(r"file not found", re.I),
    re.compile(r"AuthenticationError.*invalid.*api.*key", re.I),
]

# Standard Cargo/Maturin status lines that appear in stderr even on success.
_CARGO_STATUS_RE = re.compile(
    r"^\s*(?:Compiling|Finished|Running|Documenting|Downloading|Updating|Fresh| Packaging|Verifying|Installing)\b",
    re.MULTILINE | re.I,
)

# Real Cargo error markers; status logs with none of these are not fatal.
_CARGO_ERROR_RE = re.compile(
    r"(?:error\s*\[|error:|failed to|could not|process didn't exit successfully|exited with code|thread.*panicked|FAILED)",
    re.I,
)


def classify(text: Optional[str]) -> ErrorClass:
    """Classify an error string."""
    if text is None:
        return ErrorClass.RECOVERABLE
    # Cargo writes status lines like "Compiling ..." and "Finished ..." to
    # stderr as normal progress; they must not be flagged as errors.
    if _CARGO_STATUS_RE.search(text) and not _CARGO_ERROR_RE.search(text):
        return ErrorClass.RECOVERABLE
    lowered = text.lower()
    for pattern in _FATAL_PATTERNS:
        if pattern.search(lowered):
            return ErrorClass.FATAL
    for pattern in _TRANSIENT_PATTERNS:
        if pattern.search(lowered):
            return ErrorClass.TRANSIENT
    return ErrorClass.RECOVERABLE


def classify_exception(exc: BaseException) -> ErrorClass:
    """Classify an exception instance."""
    name = type(exc).__name__
    text = f"{name}: {exc}"
    if name in ("RateLimitError", "APIConnectionError", "APITimeoutError", "APIError"):
        return ErrorClass.TRANSIENT
    if name == "AuthenticationError":
        # Auth errors are usually fatal for a given key, but we can try another model/key.
        return ErrorClass.FATAL
    return classify(text)


def is_fatal(text: Optional[str]) -> bool:
    return classify(text) == ErrorClass.FATAL


def is_transient(text: Optional[str]) -> bool:
    return classify(text) == ErrorClass.TRANSIENT


def format_transpiler_error(
    exc: BaseException,
    source_path: Optional[Path] = None,
    source: Optional[str] = None,
) -> str:
    """Format a transpiler or unexpected exception into a concise, location-aware message.

    Output format: ``[Transpiler Error] <ExceptionType>: <Details> [File: ... Line: ...]``.
    """
    name = type(exc).__name__
    message = str(exc) or "<no details>"
    node = getattr(exc, "node", None)
    line: Optional[int] = None

    if node is None and source is not None:
        from aero_forge.errors import UnsupportedError, locate_unsupported_node

        if isinstance(exc, UnsupportedError):
            node = exc.node
        else:
            node = locate_unsupported_node(source, message)

    if node is not None:
        line = getattr(node, "lineno", None)

    location_parts = []
    if source_path:
        location_parts.append(f"File: {source_path}")
    if line:
        location_parts.append(f"Line: {line}")
    location = f" [{'; '.join(location_parts)}]" if location_parts else ""

    return f"[Transpiler Error] {name}: {message}{location}"


def format_transpiler_error_with_traceback(
    exc: BaseException,
    source_path: Optional[Path] = None,
    source: Optional[str] = None,
) -> str:
    """Return a formatted transpiler error appended with the full traceback."""
    formatted = format_transpiler_error(exc, source_path=source_path, source=source)
    tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    return f"{formatted}\n\nTraceback:\n{tb}"


# Heuristic for source-like file references in stack traces / error text.
# Captures Python traceback ``File "..."`` markers, Rust ``--> file:line:col``,
# and generic ``path/to/file.ext:line`` references including ``tests/*.py``,
# ``blueprint.aero``, and paths outside ``src/``.
_TARGET_FILE_RE = re.compile(
    r'(?:File\s+["\']?|-->\s+|at\s+)?'
    r'([\w\-/.]+\.(?:py|rs|toml|aero))'
    r'(?::(\d+))?(?::(\d+))?',
    re.MULTILINE,
)


def extract_target_files(text: Optional[str]) -> List[str]:
    """Return a sorted, unique list of source-like file paths found in *text*.

    This explicitly captures references to ``tests/*.py``, ``blueprint.aero``,
    and other non-``src/`` paths so the healing context builder can include
    them in the LLM prompt.
    """
    if not text:
        return []
    return sorted({match.group(1) for match in _TARGET_FILE_RE.finditer(text)})


__all__ = [
    "ErrorClass",
    "classify",
    "classify_exception",
    "extract_target_files",
    "format_transpiler_error",
    "format_transpiler_error_with_traceback",
    "is_fatal",
    "is_transient",
]
