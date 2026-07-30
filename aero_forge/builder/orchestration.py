"""File tagging and patch orchestration for follow-up build repairs."""

from __future__ import annotations

import re
from pathlib import Path
from typing import List

from aero_forge.builder.feedback import FeedbackParser
from aero_forge.scaffold.import_pruner import TYPING_NAMES, ensure_typing_imports

_CREATE_RE = re.compile(r"\[CREATE:\s*([^\]]+)\]")
_MODIFY_RE = re.compile(r"\[MODIFY:\s*([^\]]+)\]")
_DELETE_RE = re.compile(r"\[DELETE:\s*([^\]]+)\]")


def tag_files_for_feedback(
    workspace: Path,
    error_log: str,
    user_prompt: str = "",
) -> List[str]:
    """Return structured action tags for files that need modification/creation/deletion."""
    parser = FeedbackParser(workspace)
    parsed = parser.parse_traceback(error_log)
    tags: List[str] = []

    for ref in parsed["references"]:
        file_path = ref["file"]
        tags.append(f"[MODIFY: {file_path}]")

    missing = parsed.get("missing_symbol")
    if missing in TYPING_NAMES:
        # Typing symbol missing; target the most likely Python entrypoint.
        target = _default_python_target(workspace)
        tags.append(f"[MODIFY: {target}]")

    if user_prompt:
        for match in _CREATE_RE.finditer(user_prompt):
            tags.append(f"[CREATE: {match.group(1).strip()}]")
        for match in _MODIFY_RE.finditer(user_prompt):
            tags.append(f"[MODIFY: {match.group(1).strip()}]")
        for match in _DELETE_RE.finditer(user_prompt):
            tags.append(f"[DELETE: {match.group(1).strip()}]")

    # Preserve order, remove duplicates.
    seen = set()
    unique: List[str] = []
    for tag in tags:
        if tag not in seen:
            seen.add(tag)
            unique.append(tag)
    return unique


def _default_python_target(workspace: Path) -> str:
    for candidate in ["main.py", "src/main.py", "workspace.py"]:
        if (workspace / candidate).is_file():
            return candidate
    return "main.py"
