"""Terminal log evaluator: decide whether a failure can be auto-healed."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, Optional


class LogEvaluator:
    """Analyze command output and classify failures as healable."""

    # Common Rust compile error patterns.
    RUST_ERROR_PATTERNS = [
        r"error\[?E\d*\]?:\s*(.*)",
        r"unexpected closing delimiter",
        r"could not compile",
        r"expected i64, found f64",
        r"expected `i64`, found `f64`",
    ]

    # Python runtime/parse patterns.
    PYTHON_ERROR_PATTERNS = [
        r"SyntaxError:",
        r"IndentationError:",
        r"NameError:\s*name\s*['\"](\w+)['\"]\s*is not defined",
        r"ModuleNotFoundError:\s*No module named\s*['\"](\w+)['\"]",
        r"ImportError:",
    ]

    def __init__(self) -> None:
        self._rust_re = re.compile(
            r"(?:error(?:\[E\d+\])?:\s*(.+?)(?:\n|$)|unexpected closing delimiter)"
        )
        self._location_re = re.compile(
            r"-->\s+([^:\s]+):(\d+):(\d+)"
        )
        self._python_syntax_re = re.compile(
            r'SyntaxError:\s*(.*)',
            re.IGNORECASE,
        )
        self._python_trace_re = re.compile(
            r'File\s+"([^"]+)",\s*line\s*(\d+),\s*in\s+(.+)',
        )
        self._python_name_re = re.compile(
            r"NameError:\s*name\s*['\"](\w+)['\"]\s*is not defined",
        )
        self._python_module_re = re.compile(
            r"ModuleNotFoundError:\s*No module named\s*['\"](\w+)['\"]",
        )
        self._missing_cmd_re = re.compile(
            r"(command not found|No such file or directory|exit status 127|Exit 127)",
            re.IGNORECASE,
        )

    def _extract_target_from_command(self, command: str) -> Optional[str]:
        """Best-effort target file from the command string."""
        parts = command.split()
        for i, part in enumerate(parts):
            if part.endswith(".py") and Path(part).suffix == ".py":
                return part
            if part in ("--manifest-path", "--file") and i + 1 < len(parts):
                return parts[i + 1]
        return None

    def _last_python_frame(self, log_text: str) -> Optional[Dict[str, Any]]:
        """Return the last Python traceback frame (file, line, function) if any."""
        matches = list(self._python_trace_re.finditer(log_text))
        if not matches:
            return None
        last = matches[-1]
        return {
            "file": last.group(1),
            "line_number": int(last.group(2)),
            "function": last.group(3),
        }

    def evaluate_log(
        self,
        command: str,
        exit_code: int,
        log_text: str,
    ) -> Dict[str, Any]:
        """Return a diagnosis dictionary for the given terminal output."""
        result: Dict[str, Any] = {
            "command": command,
            "exit_code": exit_code,
            "healable": False,
            "target_file": None,
            "line_number": None,
            "summary": "No error detected.",
            "error_type": None,
        }

        if exit_code == 0 and not log_text.strip():
            return result

        # Missing executable / environment issue.
        if self._missing_cmd_re.search(log_text) or exit_code == 127:
            result["summary"] = "Missing executable or environment dependency; install required toolchain."
            result["error_type"] = "missing_toolchain"
            return result

        # Python syntax error with file and line.
        if "SyntaxError" in log_text or "IndentationError" in log_text:
            syntax_match = self._python_syntax_re.search(log_text)
            summary = (
                f"Python syntax error: {syntax_match.group(1).strip()}"
                if syntax_match
                else "Python syntax error"
            )
            frame = self._last_python_frame(log_text)
            result["healable"] = True
            result["error_type"] = "python_syntax"
            result["summary"] = summary
            if frame:
                result["target_file"] = frame["file"]
                result["line_number"] = frame["line_number"]
            else:
                result["target_file"] = self._extract_target_from_command(command) or "main.py"
            return result

        # Python NameError: missing import or typo.
        name_match = self._python_name_re.search(log_text)
        if name_match:
            result["healable"] = True
            result["error_type"] = "python_name_error"
            result["summary"] = f"Missing import or typo for '{name_match.group(1)}'."
            frame = self._last_python_frame(log_text)
            if frame:
                result["target_file"] = frame["file"]
                result["line_number"] = frame["line_number"]
            else:
                result["target_file"] = self._extract_target_from_command(command) or "main.py"
            return result

        # Python ModuleNotFoundError.
        module_match = self._python_module_re.search(log_text)
        if module_match:
            result["healable"] = True
            result["error_type"] = "python_missing_module"
            result["summary"] = f"Missing Python dependency '{module_match.group(1)}'; add to requirements or install."
            result["target_file"] = "requirements.txt"
            return result

        # Rust compile error with location.
        rust_error = self._rust_re.search(log_text)
        if rust_error:
            location = self._location_re.search(log_text)
            result["healable"] = True
            result["error_type"] = "rust_compile"
            result["summary"] = (
                f"Rust compile error: {rust_error.group(1).strip() if rust_error.group(1) else 'unexpected closing delimiter'}."
            )
            if location:
                result["target_file"] = location.group(1)
                result["line_number"] = int(location.group(2))
            else:
                result["target_file"] = self._extract_target_from_command(command)
            return result

        result["summary"] = "No deterministic fix pattern matched."
        result["error_type"] = "unknown"
        return result
