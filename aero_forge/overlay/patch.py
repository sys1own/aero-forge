"""Unified-diff computation for the overlay system."""

from __future__ import annotations

import difflib


def make_patch(original: str, modified: str, fromfile: str = "a", tofile: str = "b") -> str:
    """Return a unified diff turning *original* into *modified*."""
    original_lines = original.splitlines(keepends=True)
    modified_lines = modified.splitlines(keepends=True)
    diff = difflib.unified_diff(
        original_lines,
        modified_lines,
        fromfile=fromfile,
        tofile=tofile,
        n=3,
    )
    return "".join(diff)


def is_empty_patch(patch: str) -> bool:
    """True when *patch* contains no actual changes."""
    return not patch.strip()
