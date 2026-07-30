"""File tagging and patch orchestration for follow-up build repairs."""

from __future__ import annotations

import re
from pathlib import Path
from typing import List

from aero_forge.builder.feedback import FeedbackParser

TYPING_NAMES = {"Any", "List", "Dict", "Optional"}
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


def ensure_typing_imports(source: str) -> str:
    """Inject missing ``typing`` imports if the source uses Any/List/Dict/Optional."""
    used = {name for name in TYPING_NAMES if re.search(rf"\b{name}\b", source)}
    if not used:
        return source

    existing: set = set()
    for match in re.finditer(r"from\s+typing\s+import\s+([^#\n]+)", source):
        for part in match.group(1).split(","):
            existing.add(part.strip().split(" ")[0])
    for match in re.finditer(r"import\s+typing(?:\s+as\s+\w+)?", source):
        existing.update(TYPING_NAMES)

    missing = used - existing
    if not missing:
        return source

    import_line = "from typing import " + ", ".join(sorted(missing))
    lines = source.splitlines(keepends=True)
    insert_at = 0
    if lines and lines[0].startswith("#!"):
        insert_at = 1
    while insert_at < len(lines) and lines[insert_at].strip().startswith("#"):
        insert_at += 1
    if insert_at < len(lines) and lines[insert_at].startswith('"""'):
        # Skip a module docstring (possibly multi-line).
        if not lines[insert_at].strip().endswith('"""') or lines[insert_at].count('"""') == 1:
            insert_at += 1
            while insert_at < len(lines) and '"""' not in lines[insert_at]:
                insert_at += 1
            insert_at += 1
        else:
            insert_at += 1
    if insert_at > 0 and insert_at < len(lines) and lines[insert_at - 1].strip() == "":
        insert_at -= 1
    lines.insert(insert_at, import_line + "\n")
    return "".join(lines)
