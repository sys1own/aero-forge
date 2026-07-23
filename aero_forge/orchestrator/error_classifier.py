"""Classify build/test/API errors as transient, recoverable, or fatal."""

from __future__ import annotations

import re
from enum import Enum
from typing import Optional


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


def classify(text: Optional[str]) -> ErrorClass:
    """Classify an error string."""
    if text is None:
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


__all__ = ["ErrorClass", "classify", "classify_exception", "is_fatal", "is_transient"]
