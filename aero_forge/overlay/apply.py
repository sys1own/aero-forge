"""Apply unified-diff patches to regenerated files."""

from __future__ import annotations

from typing import List, Optional, Tuple


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
