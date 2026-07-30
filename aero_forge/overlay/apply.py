"""Apply unified-diff patches to regenerated files."""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional, Tuple, Union


def _parse_hunks(patch: str) -> List[List[Tuple[str, str]]]:
    """Split a unified diff into hunks of ``(tag, text)`` where tag is ' ', '-', '+'."""
    hunks: List[List[Tuple[str, str]]] = []
    current: Optional[List[Tuple[str, str]]] = None
    for line in patch.splitlines():
        if line.startswith("@@"):
            current = []
            hunks.append(current)
            continue
        if current is None:
            continue
        if line.startswith("\\"):
            continue
        if line and line[0] in " -+":
            current.append((line[0], line[1:].rstrip("\n")))
        elif line == "":
            current.append((" ", ""))
    return [h for h in hunks if h]


def _find_block(haystack: List[str], needle: List[str], start: int) -> Optional[int]:
    """Index of the first occurrence of *needle* in *haystack* at/after *start*."""
    if not needle:
        return start
    last = len(haystack) - len(needle)
    for i in range(start, last + 1):
        if haystack[i : i + len(needle)] == needle:
            return i
    for i in range(0, start):
        if i + len(needle) <= len(haystack) and haystack[i : i + len(needle)] == needle:
            return i
    return None


def apply_patch(target: str, patch: str) -> Tuple[str, bool]:
    """Apply *patch* to *target*, returning ``(merged_text, conflict)``."""
    if not patch.strip():
        return target, False

    result = target.splitlines()
    conflict = False
    search_from = 0

    for hunk in _parse_hunks(patch):
        pre = [text for tag, text in hunk if tag in (" ", "-")]
        post = [text for tag, text in hunk if tag in (" ", "+")]

        idx = _find_block(result, pre, search_from)
        if idx is None:
            conflict = True
            continue
        result[idx : idx + len(pre)] = post
        search_from = idx + len(post)

    merged = "\n".join(result)
    if target.endswith("\n") and not merged.endswith("\n"):
        merged += "\n"
    return merged, conflict


def apply_patch_to_disk(target_path: Union[str, Path], patch: str) -> bool:
    """Apply *patch* to the file at *target_path* and force the change to disk.

    Returns ``True`` when a conflict was detected while applying the patch.
    The parent directory and file are fsynced so the update is visible to any
    subsequent test or runner command.
    """
    target_path = Path(target_path)
    original = target_path.read_text(encoding="utf-8") if target_path.exists() else ""
    merged, conflict = apply_patch(original, patch)
    persist_text_to_disk(target_path, merged)
    return conflict


def persist_text_to_disk(target_path: Union[str, Path], text: str) -> None:
    """Write *text* to *target_path* and fsync so it is visible to subprocesses."""
    target_path = Path(target_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_text(text, encoding="utf-8")
    if hasattr(os, "fsync"):
        with target_path.open("rb") as fh:
            os.fsync(fh.fileno())
