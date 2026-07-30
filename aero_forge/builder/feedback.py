"""Feedback/error parser for follow-up build repair."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Optional

_SANDBOX_RE = re.compile(r"/tmp/aero-forge-sandboxes/[^/\\]+/?")
_NAME_ERROR_RE = re.compile(r"NameError:\s*name\s*['\"](\w+)['\"]\s*is not defined")
_IMPORT_ERROR_RE = re.compile(r"(?:ModuleNotFoundError|ImportError):\s*(?:No module named|cannot import name)\s*['\"](\w+)['\"]")
_PYTHON_TRACE_RE = re.compile(r'File\s+"([^"]+)",\s*line\s*(\d+)')
_RUST_LOCATION_RE = re.compile(r"-->\s+([^:\s]+):(\d+):(\d+)")


class FeedbackParser:
    """Parse terminal/compiler output and map it back to workspace files."""

    def __init__(self, workspace: Path) -> None:
        self.workspace = Path(workspace).resolve()

    def normalize_paths(self, text: str) -> str:
        """Replace ephemeral sandbox absolute paths with workspace-relative markers."""

        def repl(match: re.Match) -> str:
            return "<workspace>/"

        return _SANDBOX_RE.sub(repl, text)

    def parse_traceback(self, log_text: str) -> Dict[str, Any]:
        """Extract missing symbols and file references from a failure log."""
        log_text = self.normalize_paths(log_text)
        missing_symbol: Optional[str] = None
        for match in _NAME_ERROR_RE.finditer(log_text):
            missing_symbol = match.group(1)
            break
        if not missing_symbol:
            for match in _IMPORT_ERROR_RE.finditer(log_text):
                missing_symbol = match.group(1)
                break

        references: List[Dict[str, Any]] = []
        seen: set = set()
        for match in _PYTHON_TRACE_RE.finditer(log_text):
            file_path = self._relative_file(match.group(1))
            if file_path not in seen:
                seen.add(file_path)
                references.append({"file": file_path, "line": int(match.group(2))})
        for match in _RUST_LOCATION_RE.finditer(log_text):
            file_path = self._relative_file(match.group(1))
            if file_path not in seen:
                seen.add(file_path)
                references.append(
                    {
                        "file": file_path,
                        "line": int(match.group(2)),
                        "column": int(match.group(3)),
                    }
                )

        return {"missing_symbol": missing_symbol, "references": references}

    def _relative_file(self, path_str: str) -> str:
        """Return *path_str* made relative to the workspace, if possible."""
        try:
            path = Path(path_str).resolve()
            return path.relative_to(self.workspace).as_posix()
        except (ValueError, OSError):
            return path_str


def normalize_feedback_paths(log_text: str, workspace: Optional[Path] = None) -> str:
    """Strip sandbox paths from *log_text*, optionally making them relative to *workspace*."""
    parser = FeedbackParser(workspace or Path.cwd())
    return parser.normalize_paths(log_text)
