"""Syntax guard for Rust and C/C++ source files.

Repairs common LLM truncation failures:

- Unbalanced braces, brackets, and parentheses.
- Dangling ``///`` / ``//!`` doc-comment lines at the end of a file.
- Unclosed ``/* ... */`` block comments at the end of a file.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import List, Tuple


# File extensions that this guard can repair.
GUARDED_EXTENSIONS = {".rs", ".cpp", ".c", ".h", ".hpp"}


def _balanced_tokens(source: str) -> Tuple[bool, List[str]]:
    """Return (ok, stack) for ``()``, ``[]``, and ``{}`` pairs.

    Strings and comments are skipped so punctuation inside them is ignored.
    """
    stack: List[str] = []
    i = 0
    n = len(source)
    in_line_comment = False
    in_block_comment = False
    in_string: str | None = None

    while i < n:
        c = source[i]
        nxt = source[i + 1] if i + 1 < n else ""

        if in_line_comment:
            if c == "\n":
                in_line_comment = False
            i += 1
            continue

        if in_block_comment:
            if c == "*" and nxt == "/":
                in_block_comment = False
                i += 2
                continue
            i += 1
            continue

        if in_string:
            if c == "\\":
                # Skip the escaped character.
                i += 2
                continue
            if c == in_string:
                in_string = None
            i += 1
            continue

        if c == "/" and nxt == "/":
            in_line_comment = True
            i += 2
            continue

        if c == "/" and nxt == "*":
            in_block_comment = True
            i += 2
            continue

        if c in ('"', "'"):
            in_string = c
            i += 1
            continue

        if c in "([{":
            stack.append(c)
        elif c in ")]}]":
            if not stack:
                return False, []
            opening = stack.pop()
            if (
                (opening == "(" and c != ")")
                or (opening == "[" and c != "]")
                or (opening == "{" and c != "}")
            ):
                return False, []

        i += 1

    return True, stack


def _is_in_block_comment(source: str) -> bool:
    """Return True if the file ends while still inside a ``/* ... */`` block."""
    i = 0
    n = len(source)
    in_line_comment = False
    in_block_comment = False
    in_string: str | None = None

    while i < n:
        c = source[i]
        nxt = source[i + 1] if i + 1 < n else ""

        if in_line_comment:
            if c == "\n":
                in_line_comment = False
            i += 1
            continue

        if in_block_comment:
            if c == "*" and nxt == "/":
                in_block_comment = False
                i += 2
                continue
            i += 1
            continue

        if in_string:
            if c == "\\":
                i += 2
                continue
            if c == in_string:
                in_string = None
            i += 1
            continue

        if c == "/" and nxt == "/":
            in_line_comment = True
            i += 2
            continue

        if c == "/" and nxt == "*":
            in_block_comment = True
            i += 2
            continue

        if c in ('"', "'"):
            in_string = c
            i += 1
            continue

        i += 1

    return in_block_comment


def _last_unclosed_block_comment_start(source: str) -> int:
    """Find the start index of the last unclosed ``/*`` block comment."""
    # Scan from the end using the same state machine to find the last unclosed /*.
    # We do a forward pass but remember the position of the most recent unclosed /*.
    i = 0
    n = len(source)
    in_line_comment = False
    in_block_comment = False
    in_string: str | None = None
    last_open = -1

    while i < n:
        c = source[i]
        nxt = source[i + 1] if i + 1 < n else ""

        if in_line_comment:
            if c == "\n":
                in_line_comment = False
            i += 1
            continue

        if in_block_comment:
            if c == "*" and nxt == "/":
                in_block_comment = False
                i += 2
                continue
            i += 1
            continue

        if in_string:
            if c == "\\":
                i += 2
                continue
            if c == in_string:
                in_string = None
            i += 1
            continue

        if c == "/" and nxt == "/":
            in_line_comment = True
            i += 2
            continue

        if c == "/" and nxt == "*":
            in_block_comment = True
            last_open = i
            i += 2
            continue

        if c in ('"', "'"):
            in_string = c
            i += 1
            continue

        i += 1

    return last_open if in_block_comment else -1


def _strip_dangling_doc_comments(source: str) -> str:
    """Remove trailing ``///`` / ``//!`` lines and unclosed ``/*`` blocks.

    Returns the original string if nothing is removed so callers can detect
    "no change".
    """
    original = source
    changed = False

    # Remove trailing blank lines and `///` / `//!` doc comments.
    lines = source.rstrip().splitlines()
    while lines:
        stripped = lines[-1].strip()
        if stripped == "" or stripped.startswith("///") or stripped.startswith("//!"):
            lines.pop()
            changed = True
        else:
            break
    source = "\n".join(lines)

    # If the file ends inside a block comment, truncate before the last unclosed /*.
    while _is_in_block_comment(source):
        idx = _last_unclosed_block_comment_start(source)
        if idx < 0:
            break
        source = source[:idx]
        changed = True
        # Re-run doc-comment stripping for the truncated source.
        lines = source.rstrip().splitlines()
        while lines:
            stripped = lines[-1].strip()
            if stripped == "" or stripped.startswith("///") or stripped.startswith("//!"):
                lines.pop()
                changed = True
            else:
                break
        source = "\n".join(lines)

    if not changed:
        return original
    return source.rstrip() + "\n"


def repair_source(source: str) -> str:
    """Repair a single Rust/C/C++ source string.

    Returns the repaired source. If no repair is needed the original content is
    returned unchanged.
    """
    repaired = _strip_dangling_doc_comments(source)
    ok, stack = _balanced_tokens(repaired)
    if not ok:
        # Mismatched closing token; we cannot safely auto-repair.
        return source

    if stack:
        closings = {"(": ")", "[": "]", "{": "}"}
        suffix = "".join(closings[op] for op in reversed(stack))
        repaired = repaired.rstrip() + "\n" + suffix + "\n"

    if repaired == source:
        return source
    return repaired


def repair_file(path: Path) -> bool:
    """Repair a single source file in place. Returns True if changed."""
    path = Path(path)
    if path.suffix.lower() not in GUARDED_EXTENSIONS:
        return False
    try:
        original = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return False
    repaired = repair_source(original)
    if repaired != original:
        path.write_text(repaired, encoding="utf-8")
        return True
    return False


def repair_workspace(workspace: Path) -> List[Path]:
    """Repair all guarded source files under *workspace*."""
    changed: List[Path] = []
    for path in Path(workspace).rglob("*"):
        if path.is_file() and path.suffix.lower() in GUARDED_EXTENSIONS:
            if repair_file(path):
                changed.append(path)
    return changed
