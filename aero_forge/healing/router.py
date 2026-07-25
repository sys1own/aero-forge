"""Deterministic, static self-healing router for common build/test failures.

This module performs AST/pattern-based repairs only. It never calls an LLM;
LLM-based assistance is confined to upstream prompt interpretation and
human-facing diagnostics.
"""

from __future__ import annotations

import ast
import difflib
import re
from typing import Optional, Set


def _assigned_names(code: str) -> Set[str]:
    """Return all names that are assigned to anywhere in *code*."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return set()

    names: Set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        names.add(target.id)
            elif isinstance(node.target, ast.Name):
                names.add(node.target.id)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for default in node.args.defaults + node.args.kw_defaults:
                # Ignore default expressions.
                pass
        elif isinstance(node, ast.For):
            if isinstance(node.target, ast.Name):
                names.add(node.target.id)
    return names


def _closest_name(name: str, candidates: Set[str]) -> Optional[str]:
    """Return the candidate that is most likely a typo of *name*.

    Prefers an exact match after removing a single underscore, then a
    case-insensitive Levenshtein match within distance 1.
    """
    if name in candidates:
        return name

    # e.g. ``r_stats`` -> ``rstats``
    no_underscore = name.replace("_", "")
    for c in candidates:
        if c.replace("_", "") == no_underscore:
            return c

    lowered = name.lower()
    for c in candidates:
        if c.lower() == lowered:
            return c

    best: Optional[str] = None
    best_score = 0.0
    for c in candidates:
        ratio = difflib.SequenceMatcher(None, name, c).ratio()
        if ratio > best_score:
            best_score = ratio
            best = c
    if best_score >= 0.9:
        return best
    return None


def try_auto_fix(error_log: str, code: str) -> Optional[str]:
    """Apply a small set of pattern-based fixes to *code*.

    Returns the patched source code, or ``None`` when no rule matches. All
    repairs are deterministic AST rewrites; no LLM is consulted.
    """
    # 1. Missing Python import detected by a NameError at runtime.
    missing = re.search(r"NameError: name ['\"](\w+)['\"] is not defined", error_log)
    if missing:
        name = missing.group(1)
        # Add a standard import for common modules when referenced before definition.
        stdlib = {"math", "random", "sys", "os", "json", "time", "statistics"}
        if name in stdlib and f"import {name}" not in code:
            return f"import {name}\n{code}"

        # Fix local variable typos such as ``r_stats`` when ``rstats`` is defined.
        assigned = _assigned_names(code)
        if name not in assigned:
            closest = _closest_name(name, assigned)
            if closest and closest != name:
                return re.sub(rf"\b{re.escape(name)}\b", closest, code)

    # 2. Rust integer-vs-float mismatch: force integer division in Python source.
    if (
        "expected i64, found f64" in error_log
        or "expected `i64`, found `f64`" in error_log
    ):
        # Replace binary division with floor division to keep the function i64-typed.
        # This is intentionally naive; if it does not apply, the build fails cleanly
        # and the user receives a deterministic diagnostic.
        patched = re.sub(r"(?<=[^/])/(?=[^/])", "//", code)
        if patched != code:
            return patched

    # 3. Missing Rust operator/function: currently not directly patchable in Python
    # source; signal no deterministic fix so the build fails with a clear diagnostic.

    return None
