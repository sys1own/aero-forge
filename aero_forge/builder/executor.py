"""Execution reporting and artifact filtering for the polyglot builder."""

from __future__ import annotations

import fnmatch
import logging
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set

logger = logging.getLogger("aero_forge.builder.executor")

ARTIFACT_DIR_PATTERNS: Set[str] = {
    ".venv",
    "venv",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".tox",
    "dist",
    "build",
    "target",
    ".git",
    ".aero",
    ".build_cache",
    ".overlays",
    ".aero_backup",
    "node_modules",
    ".cargo",
    "*.egg-info",
}

ARTIFACT_FILE_PATTERNS: Set[str] = {
    "*.so",
    "*.pyd",
    "*.dll",
    "*.dylib",
    "*.wasm",
    "*.aeroc",
    "*.aerozip",
    "*.whl",
    "*.tar",
    "*.gz",
    "*.bz2",
    "*.xz",
    "*.zip",
    "pyvenv.cfg",
    ".gitkeep",
    "Cargo.lock",
}

STANDARD_CONFIG_NAMES: Set[str] = {
    "pyproject.toml",
    "Cargo.toml",
    "Cargo.lock",
    "CMakeLists.txt",
    "blueprint.aero",
    "workspace_blueprint.yaml",
    "setup.py",
    "setup.cfg",
    "requirements.txt",
    "README.md",
    ".gitignore",
}


def is_artifact_path(rel: Path) -> bool:
    """Return True when ``rel`` points to a generated build/runtime artifact."""
    parts = rel.parts
    if not parts:
        return False

    for part in parts:
        if part in ARTIFACT_DIR_PATTERNS:
            return True
        for pat in ARTIFACT_DIR_PATTERNS:
            if "*" in pat and fnmatch.fnmatch(part, pat):
                return True

    name = rel.name
    if name in {"pyvenv.cfg", "Cargo.lock", ".gitkeep"}:
        return True

    str_rel = str(rel)
    for pattern in ARTIFACT_FILE_PATTERNS:
        if fnmatch.fnmatch(name, pattern) or fnmatch.fnmatch(str_rel, pattern):
            return True

    return False


def parse_gitignore(workspace: Path) -> List[str]:
    """Read and return non-comment patterns from ``.gitignore`` if it exists."""
    path = workspace / ".gitignore"
    if not path.is_file():
        return []
    patterns: List[str] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if line and not line.startswith("!"):
            patterns.append(line)
    return patterns


def _match_gitignore_pattern(rel: Path, pattern: str) -> bool:
    """Simple ``.gitignore`` glob matcher supporting ``*``, ``?``, ``**``, and ``/``."""
    if pattern.endswith("/"):
        pattern = pattern[:-1]

    str_rel = str(rel)
    parts = rel.parts

    if pattern.startswith("/"):
        anchored = pattern[1:]
        if fnmatch.fnmatch(str_rel, anchored) or fnmatch.fnmatch(str_rel, anchored + "/*"):
            return True
        if parts and parts[0] == anchored:
            return True
        return False

    if "**" in pattern:
        for i in range(len(parts)):
            for j in range(i + 1, len(parts) + 1):
                if fnmatch.fnmatch("/".join(parts[i:j]), pattern):
                    return True
        if fnmatch.fnmatch(str_rel, pattern):
            return True
        return False

    for part in parts:
        if fnmatch.fnmatch(part, pattern):
            return True

    if fnmatch.fnmatch(str_rel, pattern) or fnmatch.fnmatch(str_rel, "*/" + pattern):
        return True

    return False


def is_ignored_by_gitignore(rel: Path, patterns: Iterable[str]) -> bool:
    """Return True when ``rel`` matches any of the supplied ``.gitignore`` patterns."""
    for pattern in patterns:
        if _match_gitignore_pattern(rel, pattern):
            return True
    return False


def should_report_path(
    rel: Path,
    workspace_root: Optional[Path] = None,
    gitignore_patterns: Optional[List[str]] = None,
) -> bool:
    """Return True for source/config files that should appear in execution reports."""
    if is_artifact_path(rel):
        return False

    if gitignore_patterns is None and workspace_root is not None:
        gitignore_patterns = parse_gitignore(workspace_root)

    if gitignore_patterns and is_ignored_by_gitignore(rel, gitignore_patterns):
        return False

    return True


def filter_artifact_paths(paths: Iterable[str]) -> List[str]:
    """Remove artifact paths from a list of workspace-relative strings."""
    return sorted(p for p in paths if not is_artifact_path(Path(p)))


def _path_under_scope(rel: Path, scope: Iterable[str]) -> bool:
    rel_parts = rel.parts
    for scope_item in scope:
        scope_path = Path(scope_item)
        scope_parts = scope_path.parts
        if rel == scope_path:
            return True
        if len(rel_parts) >= len(scope_parts) and rel_parts[: len(scope_parts)] == scope_parts:
            return True
    return False


def filter_to_scope(
    files: Iterable[str],
    scope: Iterable[str],
    *,
    allowed_configs: Optional[Set[str]] = None,
) -> List[str]:
    """Keep only files inside the requested ``scope`` plus standard build config files.

    This is used as a defense-in-depth post-build filter: if a runaway LLM touches
    files outside the task scope, those stray diffs are hidden from the "Created or
    modified files" report.
    """
    allowed = allowed_configs or STANDARD_CONFIG_NAMES
    scope_list = [str(s) for s in scope]
    return [f for f in files if Path(f).name in allowed or _path_under_scope(Path(f), scope_list)]


class ExecutionReport:
    """Normalize builder outputs and prune build artifacts from user-facing reports."""

    def __init__(self, workspace_root: Path) -> None:
        self.workspace_root = Path(workspace_root)
        self.gitignore_patterns = parse_gitignore(self.workspace_root)

    def filter_artifacts(self, files: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Drop artifact entries from a ``_build_file_list``-style list."""
        return sorted(
            (f for f in files if should_report_path(Path(f["path"]), gitignore_patterns=self.gitignore_patterns)),
            key=lambda f: f["path"],
        )

    def filter_paths(self, paths: List[str]) -> List[str]:
        """Drop artifact paths from a list of workspace-relative paths."""
        return sorted(
            p
            for p in paths
            if should_report_path(Path(p), self.workspace_root, self.gitignore_patterns)
        )

    def filter_to_scope(
        self,
        files: List[str],
        scope: Iterable[str],
        *,
        allowed_configs: Optional[Set[str]] = None,
    ) -> List[str]:
        """Return files within the requested scope plus allowed config files."""
        return filter_to_scope(files, scope, allowed_configs=allowed_configs)
