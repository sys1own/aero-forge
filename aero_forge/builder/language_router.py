"""Language routing for polyglot engine generation."""

from __future__ import annotations

import ast
import os
import shutil
import time
from pathlib import Path
from typing import Any, Dict, Optional


def _accel_log(level: str, message: str) -> None:
    """Append a structured line to the per-session accelerator log if set."""
    log_path = os.environ.get("AERO_FORGE_ACCEL_LOG")
    if not log_path:
        return
    try:
        timestamp = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"{timestamp} [{level}] {message}\n")
    except Exception:
        pass

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


def is_cpp_friendly(source: str) -> bool:
    """Return ``True`` when *source* is a numeric, loop-heavy function suitable for C++ acceleration.

    Lightweight control flow, string manipulation, I/O, or NumPy usage cause the
    heuristic to return ``False`` so the function stays in Python or falls back to
    the standard runtime.
    """
    _accel_log("info", "Running cpp-friendly heuristic")
    try:
        tree = ast.parse(source)
    except SyntaxError:
        _accel_log("error", "cpp-friendly heuristic failed: source parse error")
        return False

    local_functions = {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
    has_io = False
    has_numpy = False
    has_unsupported_list = False
    has_loop = False
    numeric_ops = 0
    recursive_calls = 0

    for node in ast.walk(tree):
        if isinstance(node, (ast.For, ast.While, ast.ListComp, ast.GeneratorExp)):
            has_loop = True
        if isinstance(node, ast.BinOp) and type(node.op) in (
            ast.Add,
            ast.Sub,
            ast.Mult,
            ast.Div,
            ast.Mod,
            ast.Pow,
            ast.BitOr,
            ast.BitXor,
            ast.BitAnd,
            ast.LShift,
            ast.RShift,
        ):
            numeric_ops += 1
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                if node.func.id in ("print", "input", "open"):
                    has_io = True
                if node.func.id in local_functions:
                    recursive_calls += 1
            if isinstance(node.func, ast.Attribute) and node.func.attr in ("print",):
                has_io = True
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = [alias.name for alias in node.names]
            module = getattr(node, "module", "") or ""
            if "numpy" in names or "np" in names or "numpy" in module:
                has_numpy = True
        if isinstance(node, ast.Subscript):
            # Subscript can indicate array indexing, which is fine; handled by emitter.
            pass

    if has_io or has_numpy or has_unsupported_list:
        _accel_log("info", "PASSTHROUGH: lightweight or I/O-heavy function; not C++ friendly")
        return False

    verdict = has_loop or numeric_ops >= 2 or recursive_calls >= 1
    if verdict:
        _accel_log("info", f"ACCELERATED: numeric workload detected (loops={has_loop}, numeric_ops={numeric_ops}, recursive_calls={recursive_calls})")
    else:
        _accel_log("info", "PASSTHROUGH: insufficient numeric workload for C++ acceleration")
    return verdict


def _cpp_compiler_available() -> bool:
    return any(shutil.which(name) for name in ["g++", "clang++", "c++"])


def select_native_backend(source: str, hint: Optional[str] = None) -> str:
    """Select the native backend for *source*.

    Hints:
      - ``"cpp"`` / ``"c_abi"`` -> C++
      - ``"rust"`` / ``"rust_hin"`` / ``"pyo3"`` -> Rust
      - ``"auto"`` or unset -> use :func:`is_cpp_friendly` and toolchain availability.
    """
    _accel_log("info", f"Selecting native backend (hint={hint})")
    hint = (hint or "rust_hin").lower()
    if hint in ("cpp", "c_abi"):
        _accel_log("success", "Target compilation language: C++ (extern \"C\" shared dynamic lib)")
        return "cpp"
    if hint in ("rust", "rust_hin", "pyo3"):
        _accel_log("success", "Target compilation language: Rust (cdylib / PyO3)")
        return "rust_hin"
    if is_cpp_friendly(source) and _cpp_compiler_available():
        _accel_log("success", "Target compilation language: C++ (auto-selected)")
        return "cpp"
    _accel_log("success", "Target compilation language: Rust (auto-selected)")
    return "rust_hin"


def should_accelerate_with_native(source: str, *, min_numeric_ops: int = 3) -> bool:
    """Return ``True`` when *source* has enough numeric work to justify native acceleration.

    This is a coarser gate used by the polyglot materializer to decide whether a
    generated contract should be backed by a compiled native extension or left as
    pure Python.
    """
    _accel_log("info", f"Evaluating native acceleration gate (min_numeric_ops={min_numeric_ops})")
    try:
        tree = ast.parse(source)
    except SyntaxError:
        _accel_log("error", "Native acceleration gate failed: source parse error")
        return False

    has_loop = any(isinstance(n, (ast.For, ast.While, ast.ListComp, ast.GeneratorExp)) for n in ast.walk(tree))
    numeric_ops = sum(
        1
        for n in ast.walk(tree)
        if isinstance(n, ast.BinOp)
        and type(n.op)
        in (
            ast.Add,
            ast.Sub,
            ast.Mult,
            ast.Div,
            ast.Mod,
            ast.Pow,
            ast.BitOr,
            ast.BitXor,
            ast.BitAnd,
            ast.LShift,
            ast.RShift,
        )
    )
    verdict = has_loop or numeric_ops >= min_numeric_ops
    if verdict:
        _accel_log("success", f"ACCELERATED: heavy compute detected (loops={has_loop}, numeric_ops={numeric_ops})")
    else:
        _accel_log("info", f"PASSTHROUGH: light workload (loops={has_loop}, numeric_ops={numeric_ops})")
    return verdict
