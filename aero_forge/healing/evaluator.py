"""Terminal log evaluator: decide whether a failure can be auto-healed."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


class LogEvaluator:
    """Analyze command output and classify failures as AST- or LLM-healable."""

    # Structural / syntax-only failures that can be repaired by an AST overlay.
    RUST_STRUCTURAL_PATTERNS: List[Tuple[str, str, str]] = [
        (r"unexpected closing delimiter", "rust_syntax", "Unbalanced closing delimiter; structural AST patch can restore balance."),
        (r"mismatched closing delimiter", "rust_syntax", "Mismatched closing delimiter; structural AST patch can restore balance."),
        (r"unclosed delimiter", "rust_syntax", "Unclosed delimiter; structural AST patch can close it."),
        (r"expected `;`", "rust_syntax", "Missing semicolon; structural AST patch can insert it."),
        (r"expected semicolon", "rust_syntax", "Missing semicolon; structural AST patch can insert it."),
        (r"expected one of", "rust_syntax", "Syntax issue; may be repairable by structural AST patch."),
        (r"dangling.*(?:brace|bracket|paren)", "rust_syntax", "Dangling delimiter; structural AST patch can remove or balance it."),
    ]

    # Semantic / macro / linker / type errors: not AST-overlay fixable.
    RUST_SEMANTIC_PATTERNS: List[Tuple[str, str, str, str]] = [
        (r"error\[E0308\]", "type_mismatch", "E0308", "Mismatched types. Requires type refactoring rather than AST overlay."),
        (r"error\[E0425\]", "value_not_found", "E0425", "Cannot find value in scope. Requires code refactoring or symbol introduction."),
        (r"error\[E0277\]", "trait_not_satisfied", "E0277", "Trait not satisfied. Requires trait implementation or bound changes."),
        (r"error\[E0432\]", "unresolved_import", "E0432", "Unresolved import. Requires module path or dependency fix."),
        (r"error\[E0433\]", "unresolved_import", "E0433", "Failed to resolve use path. Requires module path or dependency fix."),
        (r"error\[E0061\]", "argument_mismatch", "E0061", "Method called with wrong number of arguments. Requires signature refactor."),
        (r"error\[E0599\]", "method_not_found", "E0599", "No method or field found. Requires type or trait changes."),
        (r"error\[E0609\]", "field_not_found", "E0609", "Unknown field. Requires struct definition changes."),
        (r"error\[E0381\]", "uninitialized_variable", "E0381", "Use of possibly-uninitialized variable. Requires control-flow refactor."),
        (r"error\[E0499\]", "borrow_mismatch", "E0499", "Cannot borrow mutably more than once. Requires ownership refactor."),
        (r"error\[E0502\]", "borrow_mismatch", "E0502", "Cannot borrow immutably while mutable borrow active. Requires ownership refactor."),
        (r"error\[E0507\]", "move_error", "E0507", "Cannot move out of borrowed content. Requires clone/copy changes."),
    ]

    CARGO_PATTERNS: List[Tuple[str, str, str]] = [
        (r"error: failed to select a version for `?cc`?", "cargo_dependency_conflict", None, "Cargo dependency version conflict. Requires dependency or lockfile adjustment."),
        (r"error: failed to select a version", "cargo_dependency_conflict", None, "Cargo dependency version conflict. Requires dependency or lockfile adjustment."),
        (r"error: current package believes it.*workspace", "cargo_workspace_error", None, "Cargo workspace membership mismatch. Requires manifest correction."),
        (r"error: could not find `Cargo.toml`", "cargo_workspace_error", None, "Cargo workspace path mismatch. Requires manifest correction."),
        (r"error: package collision", "cargo_workspace_error", None, "Cargo workspace package name collision. Requires manifest correction."),
        (r"error: failed to run custom build command", "cargo_build_script_error", None, "Build script or C-FFI link failure. Requires dependency or environment fix."),
        (r"ld: cannot find", "linker_error", None, "Linker cannot find symbol. Requires library or C-FFI dependency fix."),
        (r"undefined reference to", "linker_error", None, "Undefined C-FFI symbol. Requires library or binding fix."),
        (r"error: invalid array", "cargo_manifest_syntax", None, "Cargo.toml array syntax is invalid. Requires manifest rewrite."),
        (r"error: failed to parse manifest", "cargo_manifest_syntax", None, "Cargo.toml parse error. Requires manifest rewrite."),
    ]

    PYTHON_SYNTAX_PATTERNS: List[Tuple[str, str, str]] = [
        (r"SyntaxError:", "python_syntax", None, "Python syntax error; structural AST overlay can repair it."),
        (r"IndentationError:", "python_syntax", None, "Python indentation error; structural AST overlay can repair it."),
        (r"TabError:", "python_syntax", None, "Python tab/space error; structural AST overlay can repair it."),
    ]

    PYTHON_SEMANTIC_PATTERNS: List[Tuple[str, str, str]] = [
        (r"ModuleNotFoundError:", "python_missing_module", None, "Missing Python dependency. Requires installation or environment change."),
        (r"ImportError:", "python_import_error", None, "Python import error. Requires module or dependency fix."),
        (r"AttributeError:", "python_runtime_error", None, "Python attribute error. Requires logic or API fix."),
        (r"TypeError:", "python_runtime_error", None, "Python type error. Requires logic or signature fix."),
        (r"ValueError:", "python_runtime_error", None, "Python value error. Requires logic fix."),
        (r"KeyError:", "python_runtime_error", None, "Python key error. Requires logic fix."),
    ]

    def __init__(self) -> None:
        self._rust_location_re = re.compile(r"-->\s+([^:\s]+):(\d+):(\d+)")
        self._python_trace_re = re.compile(r'File\s+"([^"]+)",\s*line\s*(\d+),\s*in\s+(.+)')
        self._python_syntax_re = re.compile(r'SyntaxError:\s*(.*)', re.IGNORECASE)
        self._python_name_re = re.compile(r"NameError:\s*name\s*['\"](\w+)['\"]\s*is not defined")
        self._python_module_re = re.compile(r"(?:ModuleNotFoundError|ImportError):.*['\"](\w+)['\"]")
        self._missing_cmd_re = re.compile(
            r"(command not found|No such file or directory|exit status 127|Exit 127|not installed|is not installed)",
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
        if "native_core" in command:
            return "crates/native_core/src/lib.rs"
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

    def _find_rust_code(self, log_text: str) -> Optional[str]:
        """Extract the first Rust error code (e.g. E0308) from the log."""
        match = re.search(r"error\[(E\d{4})\]", log_text)
        if match:
            return match.group(1)
        return None

    def _extract_reason(self, code: str) -> str:
        for _pattern, _error_type, err_code, reason in self.RUST_SEMANTIC_PATTERNS:
            if err_code == code:
                return reason
        return f"Rust compiler error {code}. Requires code refactoring rather than AST overlay."

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
            "ast_healable": False,
            "llm_healable": False,
            "target_file": None,
            "line_number": None,
            "summary": "No error detected.",
            "error_type": None,
            "code": None,
            "reason": None,
        }

        if exit_code == 0 and not log_text.strip():
            return result

        # Missing executable / environment issue.
        if self._missing_cmd_re.search(log_text) or exit_code == 127:
            result.update({
                "summary": "Missing executable or environment dependency; install required toolchain.",
                "error_type": "missing_toolchain",
                "reason": "A required toolchain binary is missing from PATH. Install it and retry.",
            })
            return result

        # Python syntax errors are AST-healable.
        for pattern, error_type, _code, reason in self.PYTHON_SYNTAX_PATTERNS:
            if re.search(pattern, log_text, re.IGNORECASE):
                syntax_match = self._python_syntax_re.search(log_text)
                summary = (
                    f"Python syntax error: {syntax_match.group(1).strip()}"
                    if syntax_match
                    else "Python syntax error"
                )
                frame = self._last_python_frame(log_text)
                result.update({
                    "healable": True,
                    "ast_healable": True,
                    "llm_healable": False,
                    "error_type": error_type,
                    "reason": reason or "Structural Python syntax issue; an AST overlay can repair it.",
                    "summary": summary,
                })
                if frame:
                    result["target_file"] = frame["file"]
                    result["line_number"] = frame["line_number"]
                else:
                    result["target_file"] = self._extract_target_from_command(command) or "main.py"
                return result

        # Python NameError: missing import or typo (AST overlay can add import).
        name_match = self._python_name_re.search(log_text)
        if name_match:
            frame = self._last_python_frame(log_text)
            result.update({
                "healable": True,
                "ast_healable": True,
                "llm_healable": False,
                "error_type": "python_name_error",
                "code": None,
                "reason": f"NameError for '{name_match.group(1)}'; an AST import overlay can resolve it.",
                "summary": f"Missing import or typo for '{name_match.group(1)}'.",
            })
            if frame:
                result["target_file"] = frame["file"]
                result["line_number"] = frame["line_number"]
            else:
                result["target_file"] = self._extract_target_from_command(command) or "main.py"
            return result

        # Python semantic / runtime errors are non-healable by AST, but LLM may fix.
        for pattern, error_type, _code, reason in self.PYTHON_SEMANTIC_PATTERNS:
            if re.search(pattern, log_text, re.IGNORECASE):
                result.update({
                    "healable": False,
                    "ast_healable": False,
                    "llm_healable": True,
                    "error_type": error_type,
                    "reason": reason,
                    "summary": reason or f"Python runtime/semantic error ({error_type}).",
                })
                return result

        # Rust semantic / type / trait errors.
        rust_code = self._find_rust_code(log_text)
        if rust_code:
            for pattern, error_type, code, reason in self.RUST_SEMANTIC_PATTERNS:
                if re.search(pattern, log_text):
                    location = self._rust_location_re.search(log_text)
                    result.update({
                        "healable": False,
                        "ast_healable": False,
                        "llm_healable": True,
                        "error_type": error_type,
                        "code": code,
                        "reason": reason,
                        "summary": f"{code} in {location.group(1) if location else 'source file'}: {reason}",
                    })
                    if location:
                        result["target_file"] = location.group(1)
                        result["line_number"] = int(location.group(2))
                    else:
                        result["target_file"] = self._extract_target_from_command(command)
                    return result

        # Cargo / dependency / workspace / linker errors.
        for pattern, error_type, _code, reason in self.CARGO_PATTERNS:
            if re.search(pattern, log_text, re.IGNORECASE):
                result.update({
                    "healable": False,
                    "ast_healable": False,
                    "llm_healable": True,
                    "error_type": error_type,
                    "reason": reason or f"Cargo/dependency error ({error_type}).",
                    "summary": f"{error_type.replace('_', ' ').title()}: {reason or error_type}.",
                })
                if "native_core" in log_text:
                    result["target_file"] = "Cargo.toml"
                return result

        # Rust structural syntax errors (no compiler code).
        for pattern, error_type, reason in self.RUST_STRUCTURAL_PATTERNS:
            if re.search(pattern, log_text, re.IGNORECASE):
                location = self._rust_location_re.search(log_text)
                result.update({
                    "healable": True,
                    "ast_healable": True,
                    "llm_healable": False,
                    "error_type": error_type,
                    "reason": reason,
                    "summary": f"Rust structural syntax error: {reason}",
                })
                if location:
                    result["target_file"] = location.group(1)
                    result["line_number"] = int(location.group(2))
                else:
                    result["target_file"] = self._extract_target_from_command(command)
                return result

        # Generic Rust compile error without a recognized code.
        rust_match = re.search(r"error:\s*(.+?)(?:\n|$)", log_text)
        if rust_match:
            location = self._rust_location_re.search(log_text)
            result.update({
                "healable": False,
                "ast_healable": False,
                "llm_healable": True,
                "error_type": "rust_compile",
                "reason": "Unrecognized Rust compile error; may require code or dependency changes.",
                "summary": f"Rust compile error: {rust_match.group(1).strip()}",
            })
            if location:
                result["target_file"] = location.group(1)
                result["line_number"] = int(location.group(2))
            else:
                result["target_file"] = self._extract_target_from_command(command)
            return result

        result.update({
            "summary": "No deterministic fix pattern matched.",
            "error_type": "unknown",
            "reason": "The failure could not be classified; manual inspection is required.",
        })
        return result
