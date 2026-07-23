"""Simplified self-healing router for common build/test failures."""

from __future__ import annotations

import re
from typing import Optional


def try_auto_fix(error_log: str, code: str) -> Optional[str]:
    """Apply a small set of pattern-based fixes to ``code``.

    Returns the patched source code, or ``None`` when no rule matches.
    """
    # 1. Missing Python import detected by a NameError at runtime.
    missing = re.search(r"NameError: name ['\"](\w+)['\"] is not defined", error_log)
    if missing:
        name = missing.group(1)
        # Add a standard import for common modules when referenced before definition.
        stdlib = {"math", "random", "sys", "os", "json", "time", "statistics"}
        if name in stdlib and f"import {name}" not in code:
            return f"import {name}\n{code}"

    # 2. Rust integer-vs-float mismatch: force integer division in Python source.
    if (
        "expected i64, found f64" in error_log
        or "expected `i64`, found `f64`" in error_log
    ):
        # Replace binary division with floor division to keep the function i64-typed.
        # This is intentionally naive; the LLM loop handles the remaining cases.
        patched = re.sub(r"(?<=[^/])/(?=[^/])", "//", code)
        if patched != code:
            return patched

    # 3. Missing Rust operator/function: add a `use` for common math traits.
    rust_missing = re.search(r"cannot find function [`']?(\w+)[`']?", error_log)
    if rust_missing:
        func = rust_missing.group(1)
        if func in ("sqrt", "sin", "cos", "tan", "exp", "log"):
            # Not directly patchable in Python source; signal no fix so the LLM takes over.
            pass

    return None
