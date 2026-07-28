"""Build a structured workspace context for LLM-driven healing."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from aero_forge.bundle_repo import bundle_workspace, format_context_block


class ContextBuilder:
    """Aggregate terminal failure context and workspace state into an LLM prompt."""

    # Regexes for pulling file/line/symbol references out of logs.
    _rust_location_re = re.compile(r"-->\s+([^:\s]+):(\d+):(\d+)")
    _python_trace_re = re.compile(r'File\s+"([^"]+)",\s*line\s*(\d+)')
    _plain_location_re = re.compile(r"(\S+\.(?:py|rs|toml|aero)):(\d+)(?::(\d+))?")
    _symbol_re = re.compile(r"(?:cannot find|unknown|undefined)\s+(?:value|symbol|function|method|field)\s*[`']?([^`':\s]+)[`']?", re.IGNORECASE)

    def __init__(self, workspace_path: Path) -> None:
        self.workspace_path = Path(workspace_path).resolve()

    def _relative_file(self, path_str: str) -> str:
        """Return *path_str* made relative to the workspace, if possible."""
        try:
            path = Path(path_str).resolve()
            return path.relative_to(self.workspace_path).as_posix()
        except (ValueError, OSError):
            return path_str

    def build_failure_context(
        self,
        command: str,
        exit_code: int,
        log_text: str,
        diagnosis: Optional[Dict[str, Any]] = None,
        previous_attempts: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Return a dictionary describing the failure and workspace context."""
        bundle = bundle_workspace(self.workspace_path)
        references = self._extract_references(log_text)
        affected_files = self._affected_files(references, diagnosis, command)
        context: Dict[str, Any] = {
            "workspace": str(self.workspace_path),
            "command": command,
            "exit_code": exit_code,
            "log_text": log_text,
            "diagnosis": diagnosis or {},
            "references": references,
            "affected_files": affected_files,
            "bundle": bundle,
            "previous_attempts": previous_attempts or [],
        }
        return context

    def build_prompt(
        self,
        command: str,
        exit_code: int,
        log_text: str,
        diagnosis: Optional[Dict[str, Any]] = None,
        previous_attempts: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """Return a prompt string for the LLM including the workspace bundle."""
        context = self.build_failure_context(command, exit_code, log_text, diagnosis, previous_attempts)
        bundle_block = format_context_block(context["bundle"], fmt="xml")

        previous_attempts_text = ""
        if context.get("previous_attempts"):
            previous_attempts_text = "### PREVIOUS FAILED ATTEMPTS (DO NOT REPEAT THESE SOLUTIONS):\n"
            for idx, attempt in enumerate(context["previous_attempts"], 1):
                previous_attempts_text += f"\n- Attempt {idx} ({attempt.get('strategy', 'unknown')}):\n"
                if attempt.get("target_file"):
                    previous_attempts_text += f"  Target file: {attempt['target_file']}\n"
                if attempt.get("directives"):
                    previous_attempts_text += f"  Directives: {json.dumps(attempt['directives'], indent=2)}\n"
                if attempt.get("patch"):
                    previous_attempts_text += f"  Patch:\n{attempt['patch']}\n"
                if attempt.get("error_message"):
                    previous_attempts_text += f"  Rejection/Error: {attempt['error_message']}\n"

        prompt = (
            "You are a Diagnostic Director for an automated build repair engine. "
            "Analyze the failure below and return a single JSON object. "
            "Do not write raw files; emit directives that the engine will execute.\n\n"
            f"Failed command: {context['command']}\n"
            f"Exit code: {context['exit_code']}\n"
            f"Diagnosis: {context['diagnosis']}\n\n"
            "Terminal output:\n"
            "---\n"
            f"{log_text}\n"
            "---\n\n"
            f"{bundle_block}\n\n"
            f"{previous_attempts_text}\n"
            "Return ONLY a JSON object matching this schema:\n"
            "{\n"
            '  "diagnosis": "Detailed root cause analysis...",\n'
            '  "directives": [\n'
            '    {\n'
            '      "target_file": "relative/path/in/workspace.rs",\n'
            '      "action": "rewrite" | "patch" | "update_manifest",\n'
            '      "reason": "Why this change fixes the failure",\n'
            '      "instructions": "High-level transformation to apply",\n'
            '      "content": "Full updated file content (for rewrite) or unified diff (for patch)"\n'
            '    }\n'
            '  ]\n'
            "}\n"
        )
        return prompt

    def _extract_references(self, log_text: str) -> List[Dict[str, Any]]:
        """Pull file/line and symbol references from the failure log."""
        refs: List[Dict[str, Any]] = []
        seen: set[str] = set()

        def add_ref(ref: Dict[str, Any]) -> None:
            file_path = self._relative_file(ref.get("file", ""))
            ref["file"] = file_path
            if file_path and file_path not in seen:
                seen.add(file_path)
                refs.append(ref)

        for match in self._rust_location_re.finditer(log_text):
            add_ref({
                "type": "location",
                "file": match.group(1),
                "line": int(match.group(2)),
                "column": int(match.group(3)),
            })
        for match in self._python_trace_re.finditer(log_text):
            add_ref({
                "type": "location",
                "file": match.group(1),
                "line": int(match.group(2)),
                "column": 0,
            })
        for match in self._plain_location_re.finditer(log_text):
            add_ref({
                "type": "location",
                "file": match.group(1),
                "line": int(match.group(2)),
                "column": int(match.group(3) or 0),
            })
        for match in self._symbol_re.finditer(log_text):
            refs.append({
                "type": "symbol",
                "name": match.group(1),
            })
        return refs

    def _affected_files(
        self,
        references: List[Dict[str, Any]],
        diagnosis: Optional[Dict[str, Any]],
        command: str,
    ) -> List[str]:
        """Return a list of relative paths likely affected by the failure.

        File paths are normalized relative to the workspace so stack traces
        pointing at ``tests/*.py`` or ``blueprint.aero`` are not ignored.
        """
        files: set[str] = set()
        for ref in references:
            if ref.get("type") == "location":
                files.add(self._relative_file(ref["file"]))
        if diagnosis:
            target = diagnosis.get("target_file")
            if target and isinstance(target, str):
                files.add(self._relative_file(target))
        # Best-effort target file from the command string.
        for part in command.split():
            if part.endswith(".rs") or part.endswith(".py") or part.endswith(".toml") or part.endswith(".aero"):
                files.add(self._relative_file(part))
        # Detect test files referenced in pytest/unittest summaries.
        for match in re.finditer(r"\b(tests?/[\w/]+\.py|test_[\w/]+\.py)\b", command + " " + (diagnosis or {}).get("summary", "")):
            files.add(self._relative_file(match.group(1)))
        # Ensure we include the most common native_core entrypoint when mentioned.
        if "native_core" in command or "native_core" in (diagnosis or {}).get("summary", ""):
            files.add("crates/native_core/src/lib.rs")
            files.add("Cargo.toml")
        return sorted(files)
