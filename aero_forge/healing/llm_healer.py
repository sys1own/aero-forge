"""LLM-driven, directive-based healing engine.

The LLM returns structured JSON directives; the ``LLMHealer`` engine validates and
applies them to the workspace. When no LLM client is available or the model call
fails, a rule-based fallback generates deterministic directives for common error
classes so the engine can still repair known failure modes.
"""

from __future__ import annotations

import json
import logging
import os
import re
import subprocess
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

from aero_forge.config import Tier
from aero_forge.healing.context_builder import ContextBuilder
from aero_forge.healing.router import try_auto_fix
from aero_forge.healing.structural_merger import MergeConflictError, apply_overlay
from aero_forge.overlay.apply import persist_text_to_disk
from aero_forge.llm.clients import BaseLLMClient, get_llm_client, LLMError
from aero_forge.orchestrator.error_classifier import extract_signature_mismatch_symbol

logger = logging.getLogger("aero_forge.healing.llm_healer")


class DirectiveError(Exception):
    """Raised when LLM directives are malformed or cannot be applied."""


class RuleBasedFallback:
    """Generate deterministic repair directives for known failure classes."""

    @classmethod
    def generate_directives(cls, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Return directives for the error described in *context*."""
        diagnosis = context.get("diagnosis") or {}
        error_type = diagnosis.get("error_type") or ""
        log_text = context.get("log_text", "")
        command = context.get("command", "")

        if error_type == "type_mismatch":
            return cls._type_mismatch_directives(context, log_text)
        if error_type == "cargo_dependency_conflict":
            return cls._dependency_conflict_directives(context, log_text)
        if error_type == "cargo_manifest_syntax":
            return cls._manifest_syntax_directives(context, log_text)
        if error_type in ("cargo_manifest_path", "cargo_workspace_error"):
            return cls._cargo_manifest_path_directives(context, log_text)
        if "missing" in error_type and "import" in error_type:
            return cls._missing_import_directives(context, log_text)
        if error_type in ("python_name_error",):
            return cls._missing_import_directives(context, log_text)
        if error_type in ("unresolved_import", "rust_scope_error"):
            return cls._rust_import_directives(context, log_text)
        if error_type == "signature_mismatch":
            return cls._signature_mismatch_directives(context, log_text)
        return []

    @classmethod
    def _target_path(cls, context: Dict[str, Any], default: str) -> str:
        diagnosis = context.get("diagnosis") or {}
        target = diagnosis.get("target_file")
        if target and isinstance(target, str):
            return target
        return default

    @classmethod
    def _signature_mismatch_directives(cls, context: Dict[str, Any], log_text: str) -> List[Dict[str, Any]]:
        """Return a directive that updates the mismatched function signature.

        The preferred fix is to make the target definition accept ``*args, **kwargs``
        so the caller can pass any number of positional or keyword arguments without
        changing the call site. If the target file cannot be located, no directives
        are returned and the full-workspace LLM fallback handles the repair.
        """
        symbol = extract_signature_mismatch_symbol(log_text)
        if not symbol:
            return []
        diagnosis = context.get("diagnosis") or {}
        target = diagnosis.get("target_file") or "main.py"
        workspace = Path(context["workspace"])
        target_path = workspace / target
        if not target_path.is_file():
            # Try to locate the file defining the mismatched symbol.
            builder = ContextBuilder(workspace)
            definition = builder._find_function_definition(symbol)
            if definition:
                target = definition["file"]
                target_path = workspace / target
        if not target_path.is_file():
            return []
        original = target_path.read_text(encoding="utf-8")
        # Use the deterministic router to insert *args, **kwargs if possible.
        sample_error = f"TypeError: {symbol}() takes 1 positional argument but 2 were given"
        fixed = try_auto_fix(sample_error, original)
        if not fixed or fixed == original:
            # Fallback: a more targeted error string matching the actual log.
            fixed = try_auto_fix(log_text, original)
        if not fixed or fixed == original:
            return []
        return [{
            "target_file": str(Path(target).as_posix()),
            "action": "rewrite",
            "reason": f"Python signature mismatch for '{symbol}': accept additional arguments with *args, **kwargs.",
            "instructions": f"Update the definition of {symbol} to accept *args, **kwargs, or wrap the caller to match the expected signature without breaking existing callers.",
            "content": fixed,
        }]

    @classmethod
    def _type_mismatch_directives(cls, context: Dict[str, Any], log_text: str) -> List[Dict[str, Any]]:
        # Rust E0308: guess whether it's i64/f64 division mismatch.
        target = cls._target_path(context, "crates/native_core/src/lib.rs")
        workspace = Path(context["workspace"])
        target_path = workspace / target
        if not target_path.is_file():
            return []
        original = target_path.read_text(encoding="utf-8")
        if "expected `i64`, found `f64`" in log_text or "expected i64, found f64" in log_text:
            fixed = re.sub(r"(?<=[^/])/(?=[^/])", "//", original)
            if fixed != original:
                return [{
                    "target_file": target,
                    "action": "rewrite",
                    "reason": "Rust E0308: replace floating-point division with integer floor division to keep i64 type.",
                    "instructions": "Replace every / with // inside function bodies that produce i64 results.",
                    "content": fixed,
                }]
        return []

    @classmethod
    def _missing_import_directives(cls, context: Dict[str, Any], log_text: str) -> List[Dict[str, Any]]:
        match = re.search(r"NameError:\s*name\s*['\"](\w+)['\"]\s*is not defined", log_text)
        if not match:
            return []
        name = match.group(1)
        stdlib = {"math", "random", "sys", "os", "json", "time", "statistics", "itertools", "collections"}
        if name not in stdlib:
            return []
        target = "main.py"
        workspace = Path(context["workspace"])
        target_path = workspace / target
        original = target_path.read_text(encoding="utf-8") if target_path.is_file() else ""
        if f"import {name}" in original:
            return []
        fixed = f"import {name}\n{original}"
        return [{
            "target_file": target,
            "action": "rewrite",
            "reason": f"Python NameError: add missing import for '{name}'.",
            "instructions": f"Prepend 'import {name}' to the top of the file.",
            "content": fixed,
        }]

    @classmethod
    def _rust_import_directives(cls, context: Dict[str, Any], log_text: str) -> List[Dict[str, Any]]:
        # E0432: try to add a missing `use pyo3::prelude::*;` in native_core src.
        target = cls._target_path(context, "crates/native_core/src/lib.rs")
        workspace = Path(context["workspace"])
        target_path = workspace / target
        if not target_path.is_file():
            return []
        original = target_path.read_text(encoding="utf-8")
        if "use pyo3::prelude::*;" in original:
            return []
        fixed = "use pyo3::prelude::*;\n" + original
        return [{
            "target_file": target,
            "action": "rewrite",
            "reason": "Rust E0432: add missing PyO3 prelude import.",
            "instructions": "Prepend 'use pyo3::prelude::*;' to the source file.",
            "content": fixed,
        }]

    @classmethod
    def _dependency_conflict_directives(cls, context: Dict[str, Any], log_text: str) -> List[Dict[str, Any]]:
        # Heuristic: if blake3 is involved, pin it to a compatible exact version.
        if "blake3" not in log_text:
            return []
        target = "crates/native_core/Cargo.toml"
        workspace = Path(context["workspace"])
        target_path = workspace / target
        if not target_path.is_file():
            return []
        original = target_path.read_text(encoding="utf-8")
        fixed = re.sub(
            r'blake3\s*=\s*\{[^}]*version\s*=\s*"[^"=]+"',
            'blake3 = { version = "=1.5.3"',
            original,
        )
        # Also fix bare version strings.
        fixed = re.sub(
            r'^(blake3\s*=\s*)"[^"]+"',
            r'\1"=1.5.3"',
            fixed,
            flags=re.MULTILINE,
        )
        if fixed != original:
            return [{
                "target_file": target,
                "action": "update_manifest",
                "reason": "Cargo dependency conflict: pin blake3 to a compatible exact version.",
                "instructions": "Set blake3 version to '=1.5.3' to avoid a cc version conflict.",
                "content": fixed,
            }]
        return []

    @classmethod
    def _manifest_syntax_directives(cls, context: Dict[str, Any], log_text: str) -> List[Dict[str, Any]]:
        # Heuristic: if Cargo.toml had an invalid array, rewrite workspace members cleanly.
        if "invalid array" not in log_text.lower() and "error: invalid array" not in log_text:
            return []
        target = "Cargo.toml"
        workspace = Path(context["workspace"])
        target_path = workspace / target
        if not target_path.is_file():
            return []
        original = target_path.read_text(encoding="utf-8")
        try:
            import tomllib
        except ImportError:
            import tomli as tomllib  # type: ignore
        try:
            with open(target_path, "rb") as fh:
                data = tomllib.load(fh)
        except Exception:
            return []
        members = data.get("workspace", {}).get("members", [])
        if "crates/native_core" not in members:
            members.append("crates/native_core")
        lines = ["[workspace]", 'members = [']
        for idx, member in enumerate(members):
            suffix = "" if idx == len(members) - 1 else ","
            lines.append(f'    "{member}"{suffix}')
        lines.append(']')
        resolver = data.get("workspace", {}).get("resolver", "2")
        lines.append(f'resolver = "{resolver}"')
        fixed = "\n".join(lines) + "\n\n" + original.split("\n", 1)[1] if "\n" in original else "\n".join(lines)
        return [{
            "target_file": target,
            "action": "update_manifest",
            "reason": "Cargo.toml workspace members array is malformed; rewrite it cleanly.",
            "instructions": "Replace the [workspace] section with a valid members array including crates/native_core.",
            "content": fixed,
        }]

    @classmethod
    def _cargo_manifest_path_directives(cls, context: Dict[str, Any], log_text: str) -> List[Dict[str, Any]]:
        """Create or update a workspace root Cargo.toml so nested crates are reachable.

        Looks for any ``Cargo.toml`` that is not at the workspace root. If one is
        found and the root lacks a workspace table, write a root ``Cargo.toml``
        that declares the nested crate as a workspace member.
        """
        workspace = Path(context["workspace"])
        root_manifest = workspace / "Cargo.toml"

        # Locate the first nested Cargo.toml (excluding build dirs).
        exclude = {"target", ".cargo"}
        nested = None
        for path in workspace.rglob("Cargo.toml"):
            if any(part in exclude for part in path.parts):
                continue
            if path.resolve() == root_manifest.resolve():
                continue
            nested = path
            break
        if not nested:
            return []

        nested_dir = nested.parent.relative_to(workspace).as_posix()
        if root_manifest.is_file():
            try:
                import tomllib
            except ImportError:
                import tomli as tomllib  # type: ignore
            try:
                with open(root_manifest, "rb") as fh:
                    data = tomllib.load(fh)
            except Exception:
                return []
            members = list(data.get("workspace", {}).get("members", []))
            if nested_dir in members:
                return []
            members.append(nested_dir)
            resolver = data.get("workspace", {}).get("resolver", "2")
            original = root_manifest.read_text(encoding="utf-8")
            # Try to splice a fresh [workspace] section in place of an existing one.
            if "[workspace]" in original:
                ws_start = original.index("[workspace]")
                ws_end = original.find("[", ws_start + 1)
                if ws_end == -1:
                    ws_end = len(original)
                before = original[:ws_start]
                after = original[ws_end:]
            else:
                before = original
                after = ""
            lines = ["[workspace]", "members = ["]
            for idx, member in enumerate(members):
                suffix = "" if idx == len(members) - 1 else ","
                lines.append(f'    "{member}"{suffix}')
            lines.append("]")
            lines.append(f'resolver = "{resolver}"')
            fixed = before.rstrip() + "\n" + "\n".join(lines) + "\n" + after.lstrip()
            return [{
                "target_file": "Cargo.toml",
                "action": "update_manifest",
                "reason": f"Cargo workspace manifest missing member '{nested_dir}'; adding it resolves path mismatch errors.",
                "instructions": f"Add '{nested_dir}' to the [workspace] members list.",
                "content": fixed,
            }]

        # No root Cargo.toml; create a minimal workspace pointing at the nested crate.
        fixed = (
            f'[workspace]\nmembers = ["{nested_dir}"]\nresolver = "2"\n'
        )
        return [{
            "target_file": "Cargo.toml",
            "action": "update_manifest",
            "reason": f"No root Cargo.toml found; creating a workspace that includes '{nested_dir}'.",
            "instructions": "Create a root Cargo.toml with a [workspace] members entry for the nested crate.",
            "content": fixed,
        }]


class LLMHealer:
    """Generate repair directives via LLM and apply them to the workspace."""

    def __init__(
        self,
        client: Optional[BaseLLMClient] = None,
        provider: str = "deepseek",
        model: Optional[str] = None,
        fallback: bool = True,
        log_callback: Optional[Callable[[str, str, str], None]] = None,
    ) -> None:
        self.client = client
        self.provider = provider
        # Prefer an explicit model or AERO_FORGE_MODEL; otherwise let get_llm_client
        # resolve the tier-mapped default so Tier.REASONING is honored.
        self.model = model or os.getenv("AERO_FORGE_MODEL") or None
        self.fallback = fallback
        self.log_callback = log_callback

    def _log(self, level: str, prefix: str, message: str) -> None:
        if self.log_callback:
            self.log_callback(level, prefix, message)
        elif level == "error":
            logger.error(message)
        elif level == "warning":
            logger.warning(message)
        else:
            logger.info(message)

    def _get_client(self) -> Optional[BaseLLMClient]:
        if self.client is not None:
            return self.client
        try:
            return get_llm_client(
                self.provider,
                model=self.model,
                raise_on_error=False,
                tier=Tier.REASONING,
            )
        except LLMError as exc:
            self._log("warning", "LLM", f"Could not create LLM client: {exc}")
            return None

    def repair_blueprint_output(self, raw_output: str) -> Optional[str]:
        """Ask the LLM to repair a malformed Blueprint v3.0.0 response.

        Returns the raw corrected text, or ``None`` if no client is available
        or the LLM call fails.
        """
        client = self._get_client()
        if client is None:
            return None

        prompt = (
            "You are a recovery assistant. The text below is a malformed "
            "Blueprint v3.0.0 response from an LLM. Return ONLY a single valid "
            "Blueprint v3.0.0 YAML or JSON object, with no markdown fences and "
            "no commentary.\n\nMalformed response:\n```\n"
            + raw_output
            + "\n```\n\nCorrected blueprint:"
        )
        try:
            return client.generate(prompt, temperature=0.1)
        except Exception as exc:
            self._log("warning", "HEAL_LLM", f"Blueprint repair LLM call failed: {exc}")
            return None

    def heal(
        self,
        workspace: Union[str, Path],
        error_logs: str,
        command: str = "",
        exit_code: int = 1,
        diagnosis: Optional[Dict[str, Any]] = None,
        tier: str = "reasoning",
        full_workspace: bool = True,
        workspace_context: Optional[Dict[str, Any]] = None,
        force_full_rewrite: bool = False,
        previous_attempts: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Convenience entry point that builds a failure context and heals the workspace.

        ``workspace_context`` may contain a pre-bundled workspace snapshot.
        ``force_full_rewrite`` asks the LLM to regenerate whole files rather than
        minimal patches when necessary.
        ``previous_attempts`` contains prior patch/rejection history so the LLM
        can avoid repeating failed solutions.
        """
        failure_context = {
            "command": command,
            "exit_code": exit_code,
            "log_text": error_logs,
            "diagnosis": diagnosis or {},
            "workspace_context": workspace_context,
            "force_full_rewrite": force_full_rewrite,
            "previous_attempts": previous_attempts or [],
        }
        return self.generate_and_apply_fix(workspace, failure_context)

    def generate_and_apply_fix(
        self,
        workspace_path: Union[str, Path],
        failure_context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Query the LLM for directives, validate them, and apply them.

        Returns a result dict with ``directives`` and ``status``.

        If a directive fails to apply (syntax error, merge conflict, etc.), the
        failure is recorded in ``previous_attempts`` and the LLM is queried
        again with the rejection context, up to ``max_repair_attempts``.
        """
        workspace = Path(workspace_path).resolve()
        builder = ContextBuilder(workspace)
        command = failure_context.get("command", "")
        exit_code = failure_context.get("exit_code", 1)
        log_text = failure_context.get("log_text", "")
        diagnosis = failure_context.get("diagnosis")
        previous_attempts: List[Dict[str, Any]] = list(failure_context.get("previous_attempts") or [])

        max_repair_attempts = max(1, int(failure_context.get("max_repair_attempts", 3)))

        for attempt in range(1, max_repair_attempts + 1):
            self._log("info", "HEAL_LLM", f"Building workspace context for LLM healing (attempt {attempt}/{max_repair_attempts})...")
            prompt = builder.build_prompt(
                command,
                exit_code,
                log_text,
                diagnosis,
                previous_attempts=previous_attempts,
            )

            directives: List[Dict[str, Any]] = []
            source = "llm"

            client = self._get_client()
            if client:
                self._log("info", "HEAL_LLM", f"Querying {self.provider} ({self.model}) for repair directives...")
                try:
                    response = client.generate(prompt, temperature=0.2)
                except Exception as exc:
                    response = None
                    self._log("warning", "HEAL_LLM", f"LLM call failed: {exc}")
                if response:
                    try:
                        directives = self._parse_directives(response)
                        self._log("info", "HEAL_LLM", f"Received {len(directives)} directive(s) from LLM.")
                    except DirectiveError as exc:
                        self._log("warning", "HEAL_LLM", f"Malformed LLM directives: {exc}")
                        previous_attempts.append({
                            "strategy": "llm",
                            "error_message": f"Malformed directives: {exc}",
                        })
                        directives = []
                else:
                    self._log("warning", "HEAL_LLM", "LLM returned empty response.")

            if not directives and self.fallback:
                self._log("info", "HEAL_LLM", "Using rule-based fallback for known error classes.")
                source = "fallback"
                directives = RuleBasedFallback.generate_directives(failure_context)

            if not directives:
                return {
                    "status": "failed",
                    "reason": "No repair directives could be generated.",
                    "directives": [],
                    "applied": [],
                    "details": {"previous_attempts": previous_attempts},
                }

            try:
                applied = self._apply_directives(workspace, directives)
                return {
                    "status": "success",
                    "source": source,
                    "directives": directives,
                    "applied": applied,
                    "details": {"previous_attempts": previous_attempts},
                }
            except (DirectiveError, SyntaxError, MergeConflictError) as exc:
                self._log("warning", "HEAL_LLM", f"Directive application failed: {exc}")
                previous_attempts.append({
                    "strategy": source,
                    "directives": directives,
                    "error_message": f"Patch rejected: {exc}",
                })
                # If we have attempts remaining, re-prompt with the rejection context.
                if attempt < max_repair_attempts:
                    continue
                return {
                    "status": "failed",
                    "reason": str(exc),
                    "directives": directives,
                    "applied": [],
                    "details": {"previous_attempts": previous_attempts},
                }

        return {
            "status": "failed",
            "reason": "All repair attempts exhausted.",
            "directives": [],
            "applied": [],
            "details": {"previous_attempts": previous_attempts},
        }

    def _parse_directives(self, response: str) -> List[Dict[str, Any]]:
        """Parse and validate JSON directives from an LLM response."""
        text = response.strip()
        # Extract JSON from a fenced code block if present.
        if "```" in text:
            match = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
            if match:
                text = match.group(1).strip()
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise DirectiveError(f"LLM response is not valid JSON: {exc}") from exc

        if not isinstance(data, dict):
            raise DirectiveError("LLM response JSON is not an object.")

        directives = data.get("directives")
        if directives is None:
            raise DirectiveError("LLM response missing 'directives' key.")
        if not isinstance(directives, list):
            raise DirectiveError("'directives' must be a list.")

        validated: List[Dict[str, Any]] = []
        for idx, directive in enumerate(directives):
            if not isinstance(directive, dict):
                raise DirectiveError(f"Directive {idx} is not an object.")
            required = {"target_file", "action", "reason", "instructions", "content"}
            missing = required - set(directive.keys())
            if missing:
                raise DirectiveError(f"Directive {idx} missing fields: {missing}")
            if directive["action"] not in {"rewrite", "patch", "update_manifest"}:
                raise DirectiveError(f"Directive {idx} has unknown action: {directive['action']}")
            validated.append(directive)

        return validated

    def _apply_directives(self, workspace: Path, directives: List[Dict[str, Any]]) -> List[str]:
        """Write directive contents to files under *workspace*, returning relative paths."""
        applied: List[str] = []
        for directive in directives:
            rel = Path(directive["target_file"])
            if ".." in rel.parts:
                raise DirectiveError(f"Target file escapes workspace: {rel}")
            target = (workspace / rel).resolve()
            if not str(target).startswith(str(workspace)):
                raise DirectiveError(f"Target file escapes workspace: {rel}")
            target.parent.mkdir(parents=True, exist_ok=True)
            content = directive["content"]
            if directive["action"] == "patch":
                # If the content looks like a diff, apply it; otherwise treat as full file.
                if content.lstrip().startswith("---") or content.lstrip().startswith("@@"):
                    original = target.read_text(encoding="utf-8") if target.is_file() else ""
                    patched = self._apply_unified_diff(original, content)
                    content = patched

            # Validate the proposed content using the structural merger / syntax guard.
            # This raises SyntaxError or MergeConflictError instead of silently
            # writing an invalid file that will be rolled back later.
            if rel.suffix in {".py", ".rs"}:
                original = target.read_text(encoding="utf-8") if target.is_file() else ""
                language = "rust" if rel.suffix == ".rs" else "python"
                apply_overlay(original, content, language=language)

            persist_text_to_disk(target, content)
            applied.append(rel.as_posix())
            self._log("info", "HEAL_LLM", f"Applied directive to {rel}")
        return applied

    @staticmethod
    def _apply_unified_diff(original: str, diff: str) -> str:
        """Apply a unified diff to *original*, falling back to the diff text on parse errors."""
        original_lines = original.splitlines(keepends=True)
        # Normalize line endings for matching.
        normalized_original = [line.rstrip("\r\n") for line in original_lines]
        result: List[str] = []
        idx = 0
        hunk_re = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")
        in_hunk = False
        hunk_old_start = 0
        hunk_old_count = 0
        hunk_old_seen = 0

        def normalized_starts_with(raw_line: str, prefix: str) -> bool:
            return raw_line.startswith(prefix)

        for raw_line in diff.splitlines(keepends=True):
            line = raw_line.rstrip("\r\n")
            if line.startswith("---") or line.startswith("+++"):
                continue
            match = hunk_re.match(line)
            if match:
                # Flush any remaining pre-hunk context.
                if in_hunk and hunk_old_seen < hunk_old_count:
                    while idx < len(normalized_original) and hunk_old_seen < hunk_old_count:
                        result.append(original_lines[idx])
                        idx += 1
                        hunk_old_seen += 1
                in_hunk = True
                hunk_old_start = int(match.group(1))
                hunk_old_count = int(match.group(2)) if match.group(2) else 1
                hunk_old_seen = 0
                # Position index to the start of the hunk in the original file.
                idx = max(0, hunk_old_start - 1)
                continue
            if not in_hunk:
                # Lines outside hunks are ignored.
                continue

            # Within a hunk, process context, addition, and deletion lines.
            if normalized_starts_with(line, " "):
                # Context line must match the next original line.
                if idx < len(normalized_original):
                    result.append(original_lines[idx])
                    idx += 1
                    hunk_old_seen += 1
            elif normalized_starts_with(line, "-"):
                if idx < len(normalized_original):
                    expected = normalized_original[idx]
                    removed = line[1:]
                    if removed == expected:
                        idx += 1
                        hunk_old_seen += 1
                    else:
                        # Mismatch: still skip the closest original line so the
                        # patch can be applied leniently.
                        idx += 1
                        hunk_old_seen += 1
            elif normalized_starts_with(line, "+"):
                result.append(line[1:] + "\n")
            else:
                # Possible empty context line or trailing marker.
                if idx < len(normalized_original):
                    result.append(original_lines[idx])
                    idx += 1
                    hunk_old_seen += 1

        # Append any remaining original lines after the last hunk.
        while idx < len(original_lines):
            result.append(original_lines[idx])
            idx += 1

        return "".join(result)


def run_command(
    command: str,
    workspace: Union[str, Path],
    env: Optional[Dict[str, str]] = None,
    timeout: int = 300,
) -> Dict[str, Any]:
    """Run *command* in *workspace* and return a structured result."""
    workspace = Path(workspace).resolve()
    proc = subprocess.run(
        command,
        shell=True,
        cwd=str(workspace),
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    return {
        "command": command,
        "exit_code": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
        "output": proc.stdout + proc.stderr,
    }
