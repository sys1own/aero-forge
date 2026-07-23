""".aeroignore pattern matching for project auto-detection.

Syntax mirrors a small subset of ``.gitignore``:

- Blank lines and lines starting with ``#`` are ignored.
- ``name`` matches any file or directory named ``name``.
- ``*.tmp`` matches files with that suffix.
- ``dir/`` matches directories (and anything under them).
- ``!pattern`` negation is not supported.
"""

from __future__ import annotations

from pathlib import Path
from typing import List


def parse_aeroignore(path: Path) -> List[str]:
    """Read ``.aeroignore`` at ``path`` and return a list of patterns."""
    if not path.is_file():
        return []
    patterns: List[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        patterns.append(stripped)
    return patterns


def _matches_pattern(rel_path: Path, pattern: str) -> bool:
    """Return True if ``rel_path`` matches a single ``.aeroignore`` pattern."""
    parts = rel_path.parts
    # Directory pattern -> match anything under that directory.
    if pattern.endswith("/"):
        dirname = pattern.rstrip("/")
        return dirname in parts
    # Exact component match anywhere in the path.
    if pattern in parts:
        return True
    # Glob match against the full relative path string.
    if rel_path.match(pattern):
        return True
    # Glob match against the file/directory name.
    if any(part for part in parts if Path(part).match(pattern)):
        return True
    return False


def is_ignored(path: Path, patterns: List[str], root: Path) -> bool:
    """Return True if ``path`` should be excluded by ``patterns`` relative to ``root``."""
    try:
        rel = path.relative_to(root)
    except ValueError:
        return False
    return any(_matches_pattern(rel, pattern) for pattern in patterns)
