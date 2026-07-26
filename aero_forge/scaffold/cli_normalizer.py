"""Post-process generated Python packages to keep CLI and native exports consistent.

When an LLM emits a ``cli.py`` that is a REPL class (e.g. ``AeroShell``) it often
forgets to re-export the underlying accelerated functions.  This helper ensures
``__init__.py`` re-exports public functions from ``native.py``/``cli.py`` and
that ``cli.py`` always has the module-level names that tests and run_shell.py
expect.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import List, Optional, Tuple


def _public_defined_names(source: str) -> List[str]:
    """Return public top-level names defined in *source* (functions, classes, and module-level assignments)."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []
    names: List[str] = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and not node.name.startswith("_"):
            names.append(node.name)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and not target.id.startswith("_"):
                    names.append(target.id)
        elif isinstance(node, ast.AnnAssign):
            if isinstance(node.target, ast.Name) and not node.target.id.startswith("_"):
                names.append(node.target.id)
    return names


def _module_has_name(source: str, name: str) -> bool:
    """Return True if *source* defines or imports *name* at module level."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return False
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node.name == name:
            return True
        if isinstance(node, ast.ImportFrom):
            for alias in node.names:
                if (alias.asname or alias.name) == name:
                    return True
        if isinstance(node, ast.Import):
            for alias in node.names:
                if (alias.asname or alias.name.split(".")[0]) == name:
                    return True
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == name:
                    return True
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.target.id == name:
            return True
    return False


def _find_repl_class(source: str) -> Optional[Tuple[str, List[str]]]:
    """Return (class_name, do_command_names) if *source* contains a cmd.Cmd subclass."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            base_names = {
                (b.attr if isinstance(b, ast.Attribute) else b.id)
                for b in node.bases
                if isinstance(b, (ast.Name, ast.Attribute))
            }
            if "Cmd" in base_names or "cmd" in {b.attr if isinstance(b, ast.Attribute) else "" for b in node.bases}:
                do_names = [
                    m.name[3:]
                    for m in node.body
                    if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef))
                    and m.name.startswith("do_")
                ]
                return (node.name, do_names)
            for base in node.bases:
                if isinstance(base, ast.Attribute) and base.attr == "Cmd":
                    do_names = [
                        m.name[3:]
                        for m in node.body
                        if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef))
                        and m.name.startswith("do_")
                    ]
                    return (node.name, do_names)
    return None


def normalize_package_exports(workspace: Path, package_name: str) -> List[str]:
    """Ensure ``<package>/__init__.py`` re-exports public functions.

    Scans ``<package>/native.py`` and ``<package>/cli.py`` for public names and
    writes them into ``__init__.py`` so ``from package import func`` works.
    Returns a list of files modified.
"""
    pkg_dir = workspace / package_name
    if not pkg_dir.is_dir():
        return []

    init_path = pkg_dir / "__init__.py"
    init_source = init_path.read_text(encoding="utf-8") if init_path.exists() else ""

    public_names: List[str] = []
    for module_file in ("native.py", "cli.py", f"{package_name}.py", "core.py"):
        module_path = pkg_dir / module_file
        if module_path.is_file():
            public_names.extend(_public_defined_names(module_path.read_text(encoding="utf-8")))

    # De-duplicate while preserving order.
    seen: set = set()
    unique_names: List[str] = []
    for n in public_names:
        if n not in seen and not n.startswith("_") and n.isidentifier():
            seen.add(n)
            unique_names.append(n)

    if not unique_names:
        return []

    # If __init__.py already re-exports these names, do nothing.
    if all(n in init_source for n in unique_names):
        return []

    imports: List[str] = []
    # Try to group names by source module.  native.py is the canonical source.
    native_path = pkg_dir / "native.py"
    cli_path = pkg_dir / "cli.py"
    native_names = set(_public_defined_names(native_path.read_text(encoding="utf-8"))) if native_path.is_file() else set()
    cli_names = set(_public_defined_names(cli_path.read_text(encoding="utf-8"))) if cli_path.is_file() else set()

    by_module: dict = {}
    for n in unique_names:
        if n in native_names:
            by_module.setdefault("native", []).append(n)
        elif n in cli_names:
            by_module.setdefault("cli", []).append(n)
        else:
            by_module.setdefault("native", []).append(n)

    lines: List[str] = ['"""Auto-generated package exports."""', ""]
    for module, names in by_module.items():
        lines.append(f"from .{module} import {', '.join(sorted(names))}")
    lines.append("")
    all_list = ", ".join(f'"{n}"' for n in sorted(unique_names))
    lines.append(f"__all__ = [{all_list}]")
    lines.append("")

    init_path.write_text("\n".join(lines), encoding="utf-8")
    return [str(init_path.relative_to(workspace))]


def normalize_cli_reexports(workspace: Path, package_name: str) -> List[str]:
    """Ensure ``<package>/cli.py`` re-exports public functions from ``native.py``.

    If ``cli.py`` exists and does not define or import a function that is present
    in ``native.py``, a guarded ``from .native import ...`` is prepended so
    ``from package.cli import func`` and test monkey-patches work.
    Returns a list of files modified.
    """
    pkg_dir = workspace / package_name
    cli_path = pkg_dir / "cli.py"
    native_path = pkg_dir / "native.py"
    if not cli_path.is_file() or not native_path.is_file():
        return []

    cli_source = cli_path.read_text(encoding="utf-8")
    native_names = [n for n in _public_defined_names(native_path.read_text(encoding="utf-8")) if not n.startswith("_")]

    missing = [n for n in native_names if not _module_has_name(cli_source, n)]
    if not missing:
        return []

    # Avoid duplicating an existing import.
    import_line = f"from .native import {', '.join(missing)}"
    if import_line in cli_source:
        return []

    # Insert the import after the first module docstring / after the imports block.
    lines = cli_source.splitlines(keepends=True)
    insert_idx = 0
    for i, line in enumerate(lines):
        if line.strip().startswith(("\"\"\"", "\"")) and not line.strip().startswith('"' + '"'):
            insert_idx = i + 1
        elif line.startswith(("import ", "from ")):
            insert_idx = i + 1
    lines.insert(insert_idx, import_line + "\n")
    cli_path.write_text("".join(lines), encoding="utf-8")
    return [str(cli_path.relative_to(workspace))]


def _ensure_top_level_do_functions(cli_path: Path, class_name: str, do_names: List[str]) -> bool:
    """Add top-level ``do_<cmd>`` wrappers that delegate to a ``cmd.Cmd`` class.

    This lets tests and run_shell.py import ``do_status`` etc. directly even when
    the LLM generated a class-based REPL.
    """
    source = cli_path.read_text(encoding="utf-8")
    if all(_module_has_name(source, f"do_{n}") for n in do_names):
        return False
    additions = []
    for name in do_names:
        if _module_has_name(source, f"do_{name}"):
            continue
        additions.append(
            f"\ndef do_{name}(args):\n"
            f"    \"\"\"Top-level wrapper for {class_name}.do_{name}.\"\"\"\n"
            f"    return {class_name}().do_{name}(args)\n"
        )
    if not additions:
        return False
    cli_path.write_text(source + "".join(additions), encoding="utf-8")
    return True


def normalize_cli_module(workspace: Path, package_name: str) -> List[str]:
    """Normalize ``cli.py`` to expose both a REPL class and top-level ``do_*`` functions."""
    pkg_dir = workspace / package_name
    cli_path = pkg_dir / "cli.py"
    if not cli_path.is_file():
        return []

    source = cli_path.read_text(encoding="utf-8")
    repl = _find_repl_class(source)
    modified: List[str] = []
    if repl:
        class_name, do_names = repl
        if _ensure_top_level_do_functions(cli_path, class_name, do_names):
            modified.append(str(cli_path.relative_to(workspace)))

    native_path = pkg_dir / "native.py"
    if native_path.is_file():
        cli_reexports = normalize_cli_reexports(workspace, package_name)
        modified.extend(cli_reexports)

    return modified


def _candidate_package_dirs(workspace: Path) -> List[Path]:
    """Return package directories (root-level or ``src/`` layout) under *workspace*."""
    candidates: List[Path] = []
    for pkg_dir in workspace.iterdir():
        if pkg_dir.is_dir() and (pkg_dir / "__init__.py").is_file():
            candidates.append(pkg_dir)
    src_dir = workspace / "src"
    if src_dir.is_dir() and (src_dir / "__init__.py").is_file():
        candidates.append(src_dir)
    return candidates


def normalize_workspace(workspace: Path) -> List[str]:
    """Normalize all Python packages under *workspace* for CLI/native exports."""
    modified: List[str] = []
    for pkg_dir in _candidate_package_dirs(workspace):
        if (pkg_dir / "cli.py").is_file():
            modified.extend(normalize_cli_module(workspace, pkg_dir.name))
        modified.extend(normalize_package_exports(workspace, pkg_dir.name))
    return list(dict.fromkeys(modified))
