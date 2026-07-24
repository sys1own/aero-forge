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


class BlueprintValidationError(ValidationError):
    """Raised when a workspace is missing a file declared in ``blueprint.aero``."""

    pass


@dataclass
class ValidationResult:
    """Outcome of a delegated validation run."""

    succeeded: bool
    command: List[str]
    output: str
    return_code: int


def validate_blueprint_manifest(
    workspace_root: Path,
    blueprint_path: Optional[Path] = None,
) -> None:
    """Validate that every file declared in ``blueprint.aero`` exists.

    Raises ``BlueprintValidationError`` with the exact required message when a
    declared file is missing from the workspace.
    """
    bp_path = blueprint_path or workspace_root / "blueprint.aero"
    if not bp_path.is_file():
        return

    from aero_forge.blueprint import parse_blueprint

    try:
        blueprint = parse_blueprint(bp_path)
    except Exception as exc:
        raise BlueprintValidationError(
            f"Invalid blueprint.aero: {exc}",
            output=str(exc),
        ) from exc

    missing: List[str] = []
    for entry in blueprint.manifest:
        candidate = workspace_root / entry.path
        if not candidate.is_file():
            missing.append(entry.path)

    if missing:
        # Surface the first missing file with the contract error format.
        raise BlueprintValidationError(
            f"Missing declared file {missing[0]} from blueprint.aero",
            output=f"Missing declared files: {', '.join(missing)}",
        )


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


def _format_line(node: ast.AST) -> str:
    lineno = getattr(node, "lineno", None)
    return f" at line {lineno}" if lineno else ""


_PRIMITIVE_TYPES = frozenset(
    {"int", "float", "bool", "complex", "str", "bytes", "None", "NoneType", "Any", "object"}
)
_TYPING_ALIASES = {"List": "list", "Tuple": "tuple", "Dict": "dict", "Set": "set", "FrozenSet": "frozenset"}
_GENERIC_CONTAINERS = frozenset({"list", "tuple", "dict", "set", "frozenset", "Optional", "Union"})


def _normalize_type_name(name: str) -> str:
    """Map ``typing.List`` style names to normalized container names."""
    if name in _TYPING_ALIASES:
        return _TYPING_ALIASES[name]
    if "." in name:
        _, tail = name.rsplit(".", 1)
        return _TYPING_ALIASES.get(tail, tail)
    return name


def _type_base(node: ast.AST) -> str:
    """Return the dotted or simple name of a type node."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return f"{_type_base(node.value)}.{node.attr}"
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return ""


def _type_args(node: ast.AST) -> List[ast.AST]:
    """Return the argument nodes inside a generic subscript."""
    if isinstance(node, ast.Index):  # pragma: no cover  # Python 3.8 compatibility
        return _type_args(node.value)
    if isinstance(node, ast.Tuple):
        return list(node.elts)
    return [node]


def _annotation_str(node: ast.AST) -> str:
    """Return a human-readable string for an annotation node."""
    try:
        return ast.unparse(node)
    except Exception:  # pragma: no cover
        return ast.dump(node)


def _is_valid_primitive_annotation(node: ast.AST) -> bool:
    """Return True when *node* is a primitive builtin or a generic of primitives.

    Unknown user-defined types are allowed to pass so that scope shadows of
    standard names (e.g. a local class named ``str``) are the only thing that
    can make a standard built-in annotation fail validation.
    """
    if node is None:
        return True
    if isinstance(node, ast.Name):
        if node.id in _PRIMITIVE_TYPES:
            return True
        if _normalize_type_name(node.id) in _GENERIC_CONTAINERS:
            return False
        return True
    if isinstance(node, ast.Constant):
        if node.value is None or isinstance(node.value, bool):
            return True
        if isinstance(node.value, str):
            return True
        return False
    if isinstance(node, ast.Attribute):
        base = _normalize_type_name(_type_base(node))
        if base in _PRIMITIVE_TYPES:
            return True
        if base in _GENERIC_CONTAINERS:
            return False
        return True
    if isinstance(node, ast.Subscript):
        base = _normalize_type_name(_type_base(node.value))
        if base in _GENERIC_CONTAINERS:
            return all(_is_valid_primitive_annotation(a) for a in _type_args(node.slice))
        if base in _PRIMITIVE_TYPES:
            return False
        return True
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
        # PEP 604 union syntax: ``X | Y``.
        return _is_valid_primitive_annotation(node.left) and _is_valid_primitive_annotation(node.right)
    return True


def _check_primitive_annotations(tree: ast.AST) -> None:
    """Validate that annotations use standard primitive types or generics of primitives."""
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            for arg in node.args.args + node.args.posonlyargs + node.args.kwonlyargs:
                if arg.annotation is not None and not _is_valid_primitive_annotation(arg.annotation):
                    raise ValidationError(
                        f"Function '{node.name}' parameter '{arg.arg}' uses non-primitive type "
                        f"'{_annotation_str(arg.annotation)}'{_format_line(arg.annotation)}. "
                        "Use a standard built-in type such as 'str', 'int', 'float', 'bool', 'bytes', "
                        "'None', 'list[...]', 'dict[...]', 'Optional[...]', or 'Union[...]'.",
                        output="",
                    )
            if node.returns is not None and not _is_valid_primitive_annotation(node.returns):
                raise ValidationError(
                    f"Function '{node.name}' return type '{_annotation_str(node.returns)}' is not primitive"
                    f"{_format_line(node.returns)}. "
                    "Use a standard built-in type such as 'str', 'int', 'float', 'bool', 'bytes', "
                    "'None', 'list[...]', 'dict[...]', 'Optional[...]', or 'Union[...]'.",
                    output="",
                )
        elif isinstance(node, ast.AnnAssign):
            if node.annotation is not None and not _is_valid_primitive_annotation(node.annotation):
                raise ValidationError(
                    f"Annotated assignment uses non-primitive type '{_annotation_str(node.annotation)}'"
                    f"{_format_line(node.annotation)}. "
                    "Use a standard built-in type such as 'str', 'int', 'float', 'bool', 'bytes', "
                    "'None', 'list[...]', 'dict[...]', 'Optional[...]', or 'Union[...]'.",
                    output="",
                )


def _check_loose_annotations(tree: ast.AST) -> None:
    """Reject bare ``dict``/``list`` type annotations that lack explicit generic parameters."""
    for ann in _collect_ann_assign_targets(tree):
        if _is_bare_dict_or_list_annotation(ann):
            name = ann.id  # type: ignore[union-attr]
            raise ValidationError(
                f"Bare '{name}' type annotation is not allowed{_format_line(ann)}. "
                f"Use explicit generic forms such as '{name.lower()}[str, Any]' "
                "(with 'from typing import Any') or omit the annotation.",
                output="",
            )


def _is_allowed_enum_base(base: ast.AST) -> bool:
    """Only ``IntEnum`` / ``enum.IntEnum`` are accepted as enum bases."""
    if isinstance(base, ast.Name) and base.id == "IntEnum":
        return True
    if (
        isinstance(base, ast.Attribute)
        and isinstance(base.value, ast.Name)
        and base.value.id == "enum"
        and base.attr == "IntEnum"
    ):
        return True
    return False


def _check_raw_enum_state_machines(tree: ast.AST) -> None:
    """Reject non-IntEnum / multi-base class hierarchies for state machines."""
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        if len(node.bases) > 1:
            raise ValidationError(
                f"Class '{node.name}' has multiple base classes{_format_line(node)}. "
                "State machine enums must inherit from 'IntEnum' only, "
                "or be a plain class without complex base class trees.",
                output="",
            )
        for base in node.bases:
            if _is_allowed_enum_base(base):
                continue
            if isinstance(base, ast.Name):
                base_name = base.id
            elif isinstance(base, ast.Attribute):
                base_name = f"{base.value.id}.{base.attr}"  # type: ignore[union-attr]
            else:
                base_name = ast.unparse(base)
            raise ValidationError(
                f"Class '{node.name}' inherits from '{base_name}'{_format_line(node)}. "
                "State machine enums must use 'IntEnum' (e.g. 'from enum import IntEnum') "
                "or a plain '@dataclass' without complex base class trees.",
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
                    f"Function '{node.name}' returns an empty list{_format_line(stmt)}, which discards "
                    "the expected matrix/array dimensions. Return a zero-filled structure "
                    "with the correct target shape instead (e.g. [[0] * cols for _ in range(rows)]).",
                    output="",
                )


_DYNAMIC_REFLECTION_BUILTINS = {"hasattr", "getattr", "setattr", "eval", "exec"}


def _check_dynamic_reflection(tree: ast.AST) -> None:
    """Reject dynamic reflection builtins that break static analysis and sandboxing."""
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Name) and func.id in _DYNAMIC_REFLECTION_BUILTINS:
            raise ValidationError(
                f"Function calls dynamic reflection builtin '{func.id}'{_format_line(node)}. "
                "Use explicit type checks with 'isinstance()' or 'try...except AttributeError:' "
                "instead of 'hasattr'/'getattr'/'setattr'/'eval'/'exec'.",
                output="",
            )


def _run_python_static_checks(source: str) -> None:
    """Parse *source* and enforce the generator-side static-analysis rules."""
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        raise ValidationError(f"Python syntax error: {exc}", output=str(exc)) from exc
    _check_loose_annotations(tree)
    _check_primitive_annotations(tree)
    _check_raw_enum_state_machines(tree)
    _check_empty_matrix_returns(tree)
    _check_dynamic_reflection(tree)


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
        # Enforce the workspace blueprint before any per-file checks.
        validate_blueprint_manifest(workspace)

        if lang == "python":
            for path in workspace.rglob("*.py"):
                try:
                    _run_python_static_checks(path.read_text(encoding="utf-8"))
                except ValidationError as exc:
                    exc.output = f"{path}: {exc}"
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
