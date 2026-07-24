"""Delegated pre-write validation for generated artifacts."""

from __future__ import annotations

import py_compile
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from aero_forge.scaffold.workspace import OutOfTreeWorkspace


class ValidationError(Exception):
    """Raised when a pre-write validation command fails."""

    def __init__(self, message: str, output: str = "") -> None:
        super().__init__(message)
        self.output = output


@dataclass
class ValidationResult:
    """Outcome of a delegated validation run."""

    succeeded: bool
    command: List[str]
    output: str
    return_code: int


def _default_validation_command(language: str, workspace_root: Path) -> Optional[List[str]]:
    """Return a sensible default validation command for *language*."""
    if language == "python":
        # Compile all .py files as a syntax/type-import sanity check.
        return ["python", "-m", "compileall", str(workspace_root)]
    if language == "rust":
        cargo_toml = workspace_root / "Cargo.toml"
        if cargo_toml.is_file():
            return ["cargo", "build", "--release"]
    return None


class PreWriteValidator:
    """Run validation in an isolated workspace before promoting files."""

    def __init__(self, context: Optional[Dict[str, Any]] = None, language: str = "") -> None:
        self.context: Dict[str, Any] = dict(context) if context else {}
        self.language = language

    def _resolve_command(self, language: Optional[str] = None) -> Optional[List[str]]:
        """Return the parsed validation command, or ``None`` if no command is configured."""
        validation = self.context.get("validation")
        if isinstance(validation, dict):
            cmd = validation.get("validation_cmd") or validation.get("execution_command")
            if cmd:
                return shlex.split(str(cmd))
        return None

    def _run_command(
        self,
        command: List[str],
        workspace_root: Path,
        language: Optional[str],
    ) -> ValidationResult:
        try:
            result = subprocess.run(
                command,
                cwd=workspace_root,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=300,
            )
        except FileNotFoundError as exc:
            raise ValidationError(
                f"validation command executable not found: {command[0]}",
                output=str(exc),
            ) from exc
        except subprocess.TimeoutExpired as exc:
            raise ValidationError(
                f"validation command timed out after {exc.timeout}s: {' '.join(command)}",
                output=(exc.stdout or "") + "\n" + (exc.stderr or ""),
            ) from exc

        if result.returncode != 0:
            raise ValidationError(
                f"validation command failed (exit code {result.returncode}): {' '.join(command)}\n"
                f"Captured output:\n{result.stdout}",
                output=result.stdout or "",
            )

        return ValidationResult(
            succeeded=True,
            command=command,
            output=result.stdout or "",
            return_code=result.returncode,
        )

    def validate(
        self,
        workspace_root: Path,
        *,
        language: Optional[str] = None,
    ) -> ValidationResult:
        """Run the configured validation command in *workspace_root*.

        If no command is configured, a default is chosen based on *language*.
        On failure raises :class:`ValidationError` with captured output so the
        orchestration layer can feed diagnostics into the self-healing loop.
        """
        lang = language or self.language or "rust"
        command = self._resolve_command(lang)

        if command is None:
            command = _default_validation_command(lang, Path(workspace_root))

        if not command:
            # Python target with no Cargo: do a syntax/import compile check.
            if lang == "python":
                for path in Path(workspace_root).rglob("*.py"):
                    try:
                        py_compile.compile(str(path), doraise=True)
                    except py_compile.PyCompileError as exc:
                        raise ValidationError(
                            f"Python syntax check failed for {path}: {exc}",
                            output=str(exc),
                        ) from exc
                return ValidationResult(
                    succeeded=True,
                    command=["py_compile"],
                    output="Python syntax check passed",
                    return_code=0,
                )
            return ValidationResult(
                succeeded=True,
                command=[],
                output="(no validation_cmd configured)",
                return_code=0,
            )

        return self._run_command(command, Path(workspace_root), lang)

    def validate_and_promote(
        self,
        staging_workspace: OutOfTreeWorkspace,
        *,
        language: Optional[str] = None,
    ) -> ValidationResult:
        """Run validation in the staging workspace and promote it on success."""
        result = self.validate(staging_workspace.root, language=language)
        staging_workspace.commit()
        return result
