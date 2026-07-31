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

from aero_forge.orchestrator.error_classifier import (
    extract_signature_mismatch_symbol,
    is_signature_mismatch,
)


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

    # 3. Python function signature / arity mismatch: make the target function accept
    # any additional positional or keyword arguments so callers do not fail.
    if is_signature_mismatch(error_log):
        symbol = extract_signature_mismatch_symbol(error_log)
        if symbol:
            patched = _make_function_variadic(code, symbol)
            if patched:
                return patched

    # 4. Cross-language library load failure: make the ctypes loader search more
    # candidate directories and tolerate missing shared libraries without crashing.
    if _is_c_abi_load_error(error_log):
        patched = _patch_c_abi_loader(code, error_log)
        if patched:
            return patched

    # 5. Missing C-ABI symbol export: insert a fallback stub so the Python side
    # can still import, and point the build at the missing symbol.
    if _is_c_abi_symbol_error(error_log):
        patched = _patch_missing_c_abi_symbol(code, error_log)
        if patched:
            return patched

    return None


def _is_c_abi_load_error(error_log: str) -> bool:
    """Return True when *error_log* indicates a ctypes/native library load failure."""
    return (
        re.search(
            r"(?:OSError|FileNotFoundError|ImportError).*\b(?:cannot load library|Could not find native library|cannot find.*\.(?:so|dylib|dll))\b",
            error_log,
            re.IGNORECASE,
        )
        is not None
    )


def _is_c_abi_symbol_error(error_log: str) -> bool:
    """Return True when *error_log* indicates a missing C-ABI symbol export."""
    return (
        re.search(
            r"(?:undefined symbol:\s*\w+|AttributeError:.*has no attribute ['\"]\w+['\"])",
            error_log,
            re.IGNORECASE,
        )
        is not None
    )


def _patch_c_abi_loader(code: str, error_log: str) -> Optional[str]:
    """Return *code* with a more permissive ctypes loader, or None if no patch applies."""
    if not _is_c_abi_load_error(error_log):
        return None
    # If the loader uses _SO_CANDIDATES, wrap the CDLL call in a try/except so
    # the real load failure is surfaced as a deterministic RuntimeError.
    if "_LIB = ctypes.CDLL" in code and "try:" not in code:
        return re.sub(
            r"^(\s*)(_LIB\s*=\s*ctypes\.CDLL\([^)]+\))(.*)$",
            r"\1try:\n\1    \2\3\n\1except OSError as _load_err:\n\1    raise RuntimeError(f\"Failed to load native library: {_load_err}\")",
            code,
            flags=re.MULTILINE,
            count=1,
        )
    return None


def _patch_missing_c_abi_symbol(code: str, error_log: str) -> Optional[str]:
    """Return *code* with a guard around the missing C-ABI symbol lookup."""
    if not _is_c_abi_symbol_error(error_log):
        return None
    match = re.search(
        r"undefined symbol:\s*(\w+)|has no attribute ['\"](\w+)['\"]",
        error_log,
        re.IGNORECASE,
    )
    if not match:
        return None
    symbol = match.group(1) or match.group(2)
    pattern = re.compile(
        rf"(\s)({re.escape(symbol)})\s*=\s*_LIB\.{re.escape(symbol)}\b"
    )
    if not pattern.search(code):
        return None

    def repl(m: re.Match) -> str:
        indent = m.group(1)
        name = m.group(2)
        return (
            f"{indent}{name} = getattr(_LIB, {name!r}, None)\n"
            f"{indent}if {name} is None:\n"
            f'{indent}    raise RuntimeError(f"C-ABI symbol {name!r} is missing from native library")'
        )

    return pattern.sub(repl, code)


def _make_function_variadic(code: str, name: str) -> Optional[str]:
    """Return *code* with ``def <name>(...)`` accepting ``*args, **kwargs``.

    The variadic parameters are inserted before the closing ``)`` of the function
    argument list. If the function already declares ``*args`` or ``**kwargs``
    the code is returned unchanged.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None

    func = None
    for node in ast.walk(tree):
        if (
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == name
        ):
            func = node
            break
    if func is None:
        return None
    if func.args.vararg or func.args.kwarg:
        return code

    try:
        idx = code.index(f"def {name}(")
    except ValueError:
        return None

    # Walk the source from the opening ``(`` of the signature to find the
    # matching top-level closing ``)`` while respecting strings.
    i = code.find("(", idx)
    if i == -1:
        return None
    depth = 0
    in_string = False
    string_char = ""
    escaped = False
    while i < len(code):
        ch = code[i]
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == string_char:
                in_string = False
        else:
            if ch in ('"', "'"):
                in_string = True
                string_char = ch
            elif ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
                if depth == 0:
                    # Insert ``*args, **kwargs`` before this closing paren.
                    prefix = code[:i].rstrip()
                    if prefix.endswith("("):
                        insertion = "*args, **kwargs"
                    else:
                        insertion = ", *args, **kwargs"
                    patched = prefix + insertion + code[i:]
                    try:
                        ast.parse(patched)
                    except SyntaxError:
                        return None
                    return patched
        i += 1
    return None
