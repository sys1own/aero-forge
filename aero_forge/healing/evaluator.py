"""Terminal log evaluator: decide whether a failure can be auto-healed.

A failure is only ``healable`` when it is a structural AST/syntax error that can
be repaired by an overlay patch (unbalanced delimiters, missing semicolons,
dangling brackets, Python syntax errors, etc.). Semantic, macro, linker, type,
workspace, and dependency errors require manual intervention and are reported
with a detailed diagnostic payload.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, Optional


class LogEvaluator:
    """Analyze command output and classify failures as healable or not."""

    # Rust structural syntax errors that can be patched by structural overlay.
    RUST_STRUCTURAL_PATTERNS = [
        r"unexpected closing delimiter",
        r"unexpected (?:open|opening) delimiter",
        r"mismatched types.*expected .* found .* because of .* delimiter",
        r"cannot find macro.*in this scope",
        r"expected (?:one of |item |`;`|identifier|type)",
        r"unclosed (?:delimiter|paren|bracket|brace)",
        r"expected (?:semi[- ]?colon|`,`).*found",
        r"missing (?:semi[- ]?colon|closing delimiter)",
    ]

    # Rust semantic / macro / linker / workspace errors that must be fixed manually.
    RUST_SEMANTIC_PATTERNS = [
        (r"error\[E0308\]", "E0308", "type_mismatch", "Mismatched types; requires type refactoring rather than AST overlay."),
        (r"error\[E0425\]", "E0425", "unresolved_value", "Cannot find value; unresolved identifier or missing definition."),
        (r"error\[E0432\]", "E0432", "unresolved_import", "Unresolved import; module path or dependency is missing."),
        (r"error\[E0433\]", "E0433", "unresolved_import", "Unresolved import; module path or dependency is missing."),
        (r"error\[E0277\]", "E0277", "trait_bound", "Trait bound not satisfied; requires trait implementation or type change."),
        (r"error\[E0277\]", "E0277", "trait_bound", "Trait bound not satisfied; requires trait implementation or type change."),
        (r"error\[E0061\]", "E0061", "argument_count", "Function argument count mismatch; requires signature update."),
        (r"error\[E0599\]", "E0599", "method_not_found", "No method named ... found for the type; API mismatch."),
        (r"error\[E0609\]", "E0609", "field_not_found", "Unknown field; struct definition mismatch."),
    ]

    # Cargo / workspace / dependency issues that cannot be AST-patched.
    CARGO_WORKSPACE_PATTERNS = [
        (r"failed to select a version", "cargo_dependency_conflict", "Cargo dependency version conflict; update Cargo.toml or lockfile."),
        (r"failed to load source for dependency", "cargo_source_error", "Cargo could not load a dependency source."),
        (r"error: invalid array", "cargo_manifest_syntax", "Cargo.toml array is malformed; needs manual manifest edit."),
        (r"could not find `.*` in `.*`", "cargo_workspace_member", "Cargo workspace member or path not found."),
        (r"current package believes it's in a workspace", "cargo_workspace_context", "Cargo workspace membership mismatch; fix workspace manifest."),
        (r"linking with .* failed", "linker_error", "Native linker failed; missing libraries or FFI symbols."),
        (r"undefined symbol", "c_ffi_error", "Undefined C-FFI symbol; check extern declarations and library linking."),
        (r"cannot find .* in this scope", "rust_scope_error", "Symbol not in scope; semantic refactoring required."),
    ]

    # Python structural syntax errors.
    PYTHON_SYNTAX_PATTERNS = [
        r"SyntaxError:",
        r"IndentationError:",
        r"TabError:",
    ]

    # Python runtime/dependency errors (not healable by AST overlay alone).
    PYTHON_SEMANTIC_PATTERNS = [
        (r"ModuleNotFoundError:\s*No module named\s*['\"](\w+)['\"]", "python_missing_module", "Missing Python dependency; install it in the environment."),
        (r"ImportError:", "python_import_error", "Import failed; module or symbol not available."),
        (r"AttributeError:", "python_attribute_error", "Attribute not found; object API mismatch."),
        (r"TypeError:", "python_type_error", "Type mismatch at runtime; logic fix required."),
    ]

    def __init__(self) -> None:
        self._rust_structural_re = re.compile(
            "|".join(f"({p})" for p in self.RUST_STRUCTURAL_PATTERNS),
            re.IGNORECASE,
        )
        self._rust_error_code_re = re.compile(r"error\[(E\d{4})\]?:\s*(.*)")
        self._rust_generic_re = re.compile(
            r"error(?:\[E\d+\])?:\s*(.+?)(?:\n|-->|$)",
            re.DOTALL,
        )
        self._location_re = re.compile(r"-->\s+([^:\s]+):(\d+):(\d+)")
        self._python_syntax_re = re.compile(
            r"(?:SyntaxError|IndentationError|TabError):\s*(.*)",
            re.IGNORECASE,
        )
        self._python_trace_re = re.compile(
            r'File\s+"([^"]+)",\s*line\s+(\d+),\s*in\s+(.+)',
        )
        self._python_name_re = re.compile(
            r"NameError:\s*name\s*['\"](\w+)['\"]\s*is not defined",
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

    def _cargo_workspace_match(self, log_text: str) -> Optional[Dict[str, Any]]:
        """Detect Cargo workspace / dependency / linker / FFI errors."""
        for pattern, error_type, reason in self.CARGO_WORKSPACE_PATTERNS:
            match = re.search(pattern, log_text, re.IGNORECASE)
            if match:
                return {"error_type": error_type, "reason": reason, "match": match.group(0)}
        return None

    def _rust_semantic_match(self, log_text: str) -> Optional[Dict[str, Any]]:
        """Detect specific Rust semantic/compiler error codes."""
        for pattern, code, error_type, reason in self.RUST_SEMANTIC_PATTERNS:
            match = re.search(pattern, log_text)
            if match:
                return {"code": code, "error_type": error_type, "reason": reason}
        return None

    def _python_semantic_match(self, log_text: str) -> Optional[Dict[str, Any]]:
        """Detect Python runtime/dependency errors."""
        for pattern, error_type, reason in self.PYTHON_SEMANTIC_PATTERNS:
            match = re.search(pattern, log_text)
            if match:
                return {
                    "error_type": error_type,
                    "reason": reason,
                    "match": match.group(0),
                }
        return None

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
            "code": None,
            "reason": None,
        }

        if exit_code == 0 and not log_text.strip():
            return result

        # Missing executable / environment issue.
        if self._missing_cmd_re.search(log_text) or exit_code == 127:
            result["summary"] = "Missing executable or environment dependency; install required toolchain."
            result["error_type"] = "missing_toolchain"
            result["reason"] = "A required command or toolchain binary is not on PATH."
            return result

        # Cargo workspace / dependency / linker / FFI errors first: never AST-patchable.
        cargo_match = self._cargo_workspace_match(log_text)
        if cargo_match:
            result["error_type"] = cargo_match["error_type"]
            result["summary"] = f"Cargo/workspace error: {cargo_match['reason']}"
            result["reason"] = cargo_match["reason"]
            result["target_file"] = "Cargo.toml"
            return result

        # Python structural syntax errors: healable.
        if any(re.search(p, log_text) for p in self.PYTHON_SYNTAX_PATTERNS):
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

        # Python NameError: undefined name, often fixable by adding an import (AST overlay).
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

        # Python semantic/runtime errors: not healable by AST overlay alone.
        python_semantic = self._python_semantic_match(log_text)
        if python_semantic:
            result["error_type"] = python_semantic["error_type"]
            result["summary"] = f"Python error: {python_semantic['reason']}"
            result["reason"] = python_semantic["reason"]
            frame = self._last_python_frame(log_text)
            if frame:
                result["target_file"] = frame["file"]
                result["line_number"] = frame["line_number"]
            else:
                result["target_file"] = self._extract_target_from_command(command) or "main.py"
            return result

        # Specific Rust semantic/compiler codes.
        semantic = self._rust_semantic_match(log_text)
        if semantic:
            result["error_type"] = semantic["error_type"]
            result["code"] = semantic["code"]
            result["summary"] = f"Rust compiler error [{semantic['code']}]: {semantic['reason']}"
            result["reason"] = semantic["reason"]
            location = self._location_re.search(log_text)
            if location:
                result["target_file"] = location.group(1)
                result["line_number"] = int(location.group(2))
            else:
                result["target_file"] = self._extract_target_from_command(command)
            return result

        # Rust structural syntax errors: healable.
        structural = self._rust_structural_re.search(log_text)
        if structural:
            location = self._location_re.search(log_text)
            result["healable"] = True
            result["error_type"] = "rust_syntax"
            result["summary"] = "Rust structural syntax error; AST overlay may fix delimiters or separators."
            if location:
                result["target_file"] = location.group(1)
                result["line_number"] = int(location.group(2))
            else:
                result["target_file"] = self._extract_target_from_command(command)
            return result

        # Generic Rust compile errors without a known code: default to semantic/manual.
        rust_error = self._rust_generic_re.search(log_text)
        if rust_error:
            location = self._location_re.search(log_text)
            result["error_type"] = "rust_compile"
            result["summary"] = (
                f"Rust compile error: {rust_error.group(1).strip()}."
            )
            result["reason"] = "A Rust compiler error was detected; manual diagnosis is required."
            if location:
                result["target_file"] = location.group(1)
                result["line_number"] = int(location.group(2))
            else:
                result["target_file"] = self._extract_target_from_command(command)
            return result

        result["summary"] = "No deterministic fix pattern matched."
        result["error_type"] = "unknown"
        result["reason"] = "The failure signature is not recognized."
        return result
