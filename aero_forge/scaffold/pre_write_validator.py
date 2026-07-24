"""Delegated pre-write validation for generated artifacts."""

from __future__ import annotations

import ast
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


def _is_bare_dict_or_list_annotation(node: ast.AST) -> bool:
    """True when an annotation is a bare ``dict``/``list`` (or ``Dict``/``List``)."""
    if isinstance(node, ast.Name) and node.id in {"dict", "list", "Dict", "List"}:
        return True
    return False


def _annotation_is_nested_list(node: ast.AST) -> bool:
    """True when an annotation describes a nested list/matrix shape."""
    if isinstance(node, ast.Subscript):
        if isinstance(node.value, ast.Name) and node.value.id in {"list", "List"}:
            return _annotation_contains_list(node.slice)
    return False


def _annotation_contains_list(node: ast.AST) -> bool:
    if isinstance(node, ast.Name) and node.id in {"list", "List"}:
        return True
    if isinstance(node, ast.Subscript):
        if isinstance(node.value, ast.Name) and node.value.id in {"list", "List"}:
            return True
        return _annotation_contains_list(node.value) or _annotation_contains_list(node.slice)
    if isinstance(node, (ast.Tuple, ast.List)):
        return any(_annotation_contains_list(elt) for elt in node.elts)
    return False


def _collect_ann_assign_targets(tree: ast.AST) -> List[ast.AST]:
    """Return all annotation nodes from function arguments, return types, and assignments."""
    annotations: List[ast.AST] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            for arg in node.args.args + node.args.posonlyargs + node.args.kwonlyargs:
                if arg.annotation:
                    annotations.append(arg.annotation)
            if node.returns:
                annotations.append(node.returns)
        elif isinstance(node, ast.AnnAssign):
            annotations.append(node.annotation)
    return annotations


def _check_loose_annotations(tree: ast.AST) -> None:
    """Reject bare ``dict``/``list`` type annotations that lack explicit generic parameters."""
    for ann in _collect_ann_assign_targets(tree):
        if _is_bare_dict_or_list_annotation(ann):
            name = ann.id  # type: ignore[union-attr]
            raise ValidationError(
                f"Bare '{name}' type annotation is not allowed. "
                f"Use explicit generic forms such as '{name.lower()}[str, Any]' "
                "(with 'from typing import Any') or omit the annotation.",
                output="",
            )


def _check_raw_enum_state_machines(tree: ast.AST) -> None:
    """Reject classes that inherit from raw ``Enum``; prefer ``IntEnum`` or dataclasses."""
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        for base in node.bases:
            if isinstance(base, ast.Name) and base.id == "Enum":
                raise ValidationError(
                    f"Class '{node.name}' inherits from raw 'Enum'. "
                    "Use 'IntEnum' from 'enum' for serialisable state machines, "
                    "or use '@dataclass' for structured state data.",
                    output="",
                )
            if (
                isinstance(base, ast.Attribute)
                and isinstance(base.value, ast.Name)
                and base.value.id == "enum"
                and base.attr == "Enum"
            ):
                raise ValidationError(
                    f"Class '{node.name}' inherits from raw 'enum.Enum'. "
                    "Use 'enum.IntEnum' for serialisable state machines, "
                    "or use '@dataclass' for structured state data.",
                    output="",
                )


def _is_empty_list_return(node: ast.AST) -> bool:
    """True for ``return []`` / ``return list()`` / ``return list([])``."""
    if isinstance(node, ast.Return):
        if isinstance(node.value, ast.List) and not node.value.elts:
            return True
        if isinstance(node.value, ast.Call):
            if isinstance(node.value.func, ast.Name) and node.value.func.id == "list":
                return True
    return False


def _suggests_matrix_or_array(name: str) -> bool:
    return any(
        keyword in name.lower()
        for keyword in {"matrix", "mat", "array", "grid", "zero", "tensor"}
    )


def _check_empty_matrix_returns(tree: ast.AST) -> None:
    """Reject ``return []`` in matrix/array functions that would discard target dimensions."""
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef):
            continue
        matrix_like = (
            node.returns is not None
            and _annotation_is_nested_list(node.returns)
        ) or _suggests_matrix_or_array(node.name)
        if not matrix_like:
            continue
        for stmt in ast.walk(node):
            if _is_empty_list_return(stmt):
                raise ValidationError(
                    f"Function '{node.name}' returns an empty list, which discards "
                    "the expected matrix/array dimensions. Return a zero-filled structure "
                    "with the correct target shape instead (e.g. [[0] * cols for _ in range(rows)]).",
                    output="",
                )


def _run_python_static_checks(source: str) -> None:
    """Parse *source* and enforce the generator-side static-analysis rules."""
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        raise ValidationError(f"Python syntax error: {exc}", output=str(exc)) from exc
    _check_loose_annotations(tree)
    _check_raw_enum_state_machines(tree)
    _check_empty_matrix_returns(tree)


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
        workspace = Path(workspace_root)

        # Generator-side static analysis runs first so bad patterns are caught
        # before any sandboxed command or filesystem promotion.
        if lang == "python":
            for path in workspace.rglob("*.py"):
                try:
                    _run_python_static_checks(path.read_text(encoding="utf-8"))
                except ValidationError as exc:
                    exc.output = f"{path}: {exc.output}"
                    raise
                except (OSError, UnicodeDecodeError) as exc:
                    raise ValidationError(f"Could not read {path}: {exc}", output=str(exc)) from exc

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
