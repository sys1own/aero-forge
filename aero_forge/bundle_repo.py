"""Compact workspace bundler for LLM chat context.

Exports ``bundle_workspace`` and helpers that turn the current project tree into
a token-efficient, machine-readable payload (dictionary and serialized XML/JSON)
so the chat assistant can reason about source files, blueprints, and test status
during multi-turn sessions.
"""

from __future__ import annotations

import json
import re
import xml.etree.ElementTree as ET
from html import escape as html_escape
from pathlib import Path
from typing import Any, Dict, List, Optional

SOURCE_SUFFIXES = {
    ".rs",
    ".py",
    ".toml",
    ".aero",
    ".md",
    ".json",
    ".yaml",
    ".yml",
    ".ini",
    ".cfg",
    ".sh",
    ".txt",
}

BINARY_SUFFIXES = {
    ".so",
    ".pyd",
    ".dll",
    ".dylib",
    ".wasm",
    ".zip",
    ".tar",
    ".gz",
    ".bz2",
    ".xz",
    ".whl",
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".webp",
    ".ico",
    ".pdf",
    ".o",
    ".rlib",
    ".rmeta",
    ".d",
    ".exe",
}

SKIP_DIRS = {
    "target",
    "dist",
    "build",
    "__pycache__",
    ".pytest_cache",
    ".cargo",
    ".git",
    ".aero-forge-cache",
    ".build_cache",
    ".overlays",
    ".mypy_cache",
    ".tox",
    ".venv",
    "venv",
    "node_modules",
    ".eggs",
    "*.egg-info",
}


def _is_source_file(path: Path) -> bool:
    """Return True for human-relevant source/configuration files."""
    if path.name.startswith("."):
        return False
    ext = path.suffix.lower()
    if ext in BINARY_SUFFIXES or not ext:
        # Allow shebang scripts without an extension.
        if path.suffix == "":
            try:
                first = path.read_bytes(16)
            except OSError:
                return False
            return first.startswith(b"#!")
        return False
    return ext in SOURCE_SUFFIXES


def _should_skip_dir(rel: Path) -> bool:
    """Return True if a directory path should be skipped entirely."""
    parts = set(rel.parts)
    if parts & SKIP_DIRS:
        return True
    for part in rel.parts:
        if part.endswith(".egg-info"):
            return True
    return False


def _compact(content: str) -> str:
    """Reduce whitespace noise without altering semantics.

    - Strip trailing whitespace from each line.
    - Collapse runs of more than two consecutive blank lines to two.
    """
    lines = [line.rstrip() for line in content.splitlines()]
    compacted: List[str] = []
    blank_count = 0
    for line in lines:
        if line == "":
            blank_count += 1
            if blank_count <= 2:
                compacted.append(line)
        else:
            blank_count = 0
            compacted.append(line)
    return "\n".join(compacted).strip("\n") + "\n"


def _read_test_status(workspace_dir: Path) -> Optional[Dict[str, Any]]:
    """Return the latest test execution status if it is available on disk."""
    status_candidates = [
        workspace_dir / ".aero" / "test_status.json",
        workspace_dir / ".pytest_cache" / "v" / "cache" / "lastfailed",
        workspace_dir / "test-results.json",
    ]
    for candidate in status_candidates:
        if not candidate.is_file():
            continue
        try:
            data = candidate.read_text(encoding="utf-8")
        except OSError:
            continue
        if candidate.name == "lastfailed":
            failed = [line for line in data.splitlines() if line.strip()]
            return {"status": "failed" if failed else "passed", "failed_tests": failed}
        try:
            return json.loads(data)
        except json.JSONDecodeError:
            return {"raw": data}

    tests_dir = workspace_dir / "tests"
    if tests_dir.is_dir():
        test_files = sorted(
            p.relative_to(workspace_dir).as_posix()
            for p in tests_dir.rglob("test_*.py")
        )
        if test_files:
            return {"status": "unknown", "test_files": test_files}
    return None


def bundle_workspace(
    workspace_dir: Path,
    max_file_size_kb: int = 100,
) -> Dict[str, Any]:
    """Package *workspace_dir* into a compact, LLM-friendly dictionary.

    Returns a dictionary with:
    - ``workspace``: absolute path to the workspace.
    - ``files``: mapping from relative POSIX path to compacted source content.
    - ``blueprint``: contents of ``blueprint.aero`` if present.
    - ``test_status``: latest test execution status if available.
    """
    workspace_dir = Path(workspace_dir).resolve()
    max_bytes = max_file_size_kb * 1024

    bundle: Dict[str, Any] = {
        "workspace": str(workspace_dir),
        "files": {},
        "blueprint": None,
        "test_status": None,
    }

    for path in sorted(workspace_dir.rglob("*")):
        try:
            rel = path.relative_to(workspace_dir)
        except ValueError:
            continue
        if _should_skip_dir(rel):
            # Prune the whole directory tree by skipping its contents.
            continue
        if not path.is_file():
            continue
        if not _is_source_file(rel):
            continue
        try:
            size = path.stat().st_size
        except OSError:
            continue
        if size > max_bytes:
            continue
        try:
            content = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        bundle["files"][rel.as_posix()] = _compact(content)

    blueprint = workspace_dir / "blueprint.aero"
    if blueprint.is_file() and blueprint.stat().st_size <= max_bytes:
        try:
            bundle["blueprint"] = _compact(blueprint.read_text(encoding="utf-8"))
        except (UnicodeDecodeError, OSError):
            pass

    bundle["test_status"] = _read_test_status(workspace_dir)
    return bundle


def bundle_to_xml(bundle: Dict[str, Any]) -> str:
    """Serialize a bundle into an XML-like string with ``<file path="...">`` tags."""
    lines = [
        '<workspace path="{}">'.format(html_escape(str(bundle["workspace"]), quote=True)),
    ]
    if bundle.get("blueprint"):
        lines.append("<blueprint>{}</blueprint>".format(html_escape(bundle["blueprint"], quote=True)))
    if bundle.get("test_status"):
        status = json.dumps(bundle["test_status"], indent=2)
        lines.append("<test_status>{}</test_status>".format(html_escape(status, quote=True)))
    for path, content in bundle["files"].items():
        lines.append('<file path="{}">'.format(html_escape(path, quote=True)))
        lines.append(html_escape(content, quote=True))
        lines.append("</file>")
    lines.append("</workspace>")
    return "\n".join(lines)


def bundle_to_json(bundle: Dict[str, Any]) -> str:
    """Serialize a bundle as a compact JSON string."""
    return json.dumps(bundle, indent=2)


def format_context_block(bundle: Dict[str, Any], fmt: str = "xml") -> str:
    """Return a string suitable for injection into a system/chat prompt."""
    if fmt == "json":
        body = bundle_to_json(bundle)
    else:
        body = bundle_to_xml(bundle)
    return f"CURRENT_PROJECT_CONTEXT ({fmt.upper()} workspace bundle):\n{body}\n---END PROJECT CONTEXT---"
