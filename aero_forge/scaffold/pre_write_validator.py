"""Delegated pre-write validation for generated artifacts."""

from __future__ import annotations

import ast
import py_compile
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from aero_forge.builder.language_router import _deduplicate_command_args
from aero_forge.orchestrator.router import classify_build_intent
from aero_forge.scaffold.cargo_runner import cargo_build, maturin_build
from aero_forge.scaffold.module_guard import ensure_package_structure
from aero_forge.scaffold.workspace import OutOfTreeWorkspace


class ValidationError(Exception):
    """Raised when a pre-write validation command fails."""

    def __init__(self, message: str, output: str = "") -> None:
        super().__init__(message)
        self.output = output


class BlueprintValidationError(ValidationError):
    """Raised when a workspace is missing a file declared in ``blueprint.aero``."""

    pass


def deduplicate_manifest_entries(entries: List[Any]) -> List[Any]:
    """Return *entries* with duplicate paths removed, preserving first occurrence.

    Works on both ``ManifestEntry`` objects and plain path dictionaries.
    """
    seen: set = set()
    unique: List[Any] = []
    for entry in entries:
        if isinstance(entry, dict):
            path = entry.get("path")
        else:
            path = getattr(entry, "path", None)
        if not path:
            unique.append(entry)
            continue
        norm = str(path).replace("\\", "/").lstrip("/")
        if norm not in seen:
            seen.add(norm)
            unique.append(entry)
    return unique


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


def _has_cpp_keywords(prompt: str, toolchains: List[str]) -> bool:
    """Detect whether a prompt/toolchain set is C++ / pybind11 oriented."""
    lower = prompt.lower()
    cpp_terms = ("c++", "cpp", "pybind11", "g++", "clang++", "cmake")
    return (
        any(term in lower for term in cpp_terms)
        or "cpp" in toolchains
        or "cmake" in toolchains
    )


def validate_blueprint_intent(prompt: str, blueprint: Any) -> None:
    """Raise ``BlueprintValidationError`` if a polyglot prompt was downgraded.

    When the prompt explicitly requests multi-language / Rust or C++ integration,
    the resulting ``blueprint.aero`` must not silently fall back to ``pure_python``.
    """
    intent = classify_build_intent(prompt)
    architecture = getattr(blueprint, "architecture", "pure_python")
    toolchains = getattr(blueprint, "toolchains", [])
    manifest = getattr(blueprint, "manifest", [])
    manifest_paths = {getattr(entry, "path", "") for entry in manifest}

    if architecture == "pure_python" and intent in (
        "hybrid_rust_python",
        "hybrid_cpp_python",
        "hybrid_cpp_rust",
        "tri_polyglot_rust_cpp_python",
    ):
        raise BlueprintValidationError(
            f"Prompt requests a {intent} build, but blueprint architecture is {architecture!r}",
            output=f"Set architecture to a hybrid/polyglot value such as '{intent}'.",
        )

    if intent == "hybrid_rust_python" or (
        intent == "hybrid_polyglot" and not _has_cpp_keywords(prompt, toolchains)
    ):
        has_rust = "rust" in toolchains or "cargo" in toolchains
        if "python" not in toolchains or not has_rust:
            raise BlueprintValidationError(
                f"Prompt requests a hybrid Rust/Python build, but toolchains {toolchains!r} are missing 'python' or 'rust'",
                output="Include both 'python' and 'rust' (or 'cargo') in toolchains.",
            )
        if "Cargo.toml" not in {Path(p).name for p in manifest_paths}:
            raise BlueprintValidationError(
                "Prompt requests a hybrid Rust/Python build, but manifest is missing Cargo.toml",
                output="Add a Cargo.toml entry to the blueprint manifest.",
            )

    if intent == "hybrid_cpp_rust" or (intent == "hybrid_polyglot" and _has_cpp_keywords(prompt, toolchains) and ("rust" in toolchains or "cargo" in toolchains) and "python" not in toolchains):
        has_rust = "rust" in toolchains or "cargo" in toolchains
        has_cpp = "cpp" in toolchains or "cmake" in toolchains or "g++" in toolchains or "clang" in toolchains
        if not has_rust or not has_cpp:
            raise BlueprintValidationError(
                f"Prompt requests a hybrid C++/Rust build, but toolchains {toolchains!r} are missing 'rust' or 'cpp'",
                output="Include both 'rust' (or 'cargo') and 'cpp' (or 'cmake') in toolchains.",
            )

    if intent == "hybrid_cpp_python" or (
        _has_cpp_keywords(prompt, toolchains)
        and intent not in ("hybrid_cpp_rust", "tri_polyglot_rust_cpp_python")
    ):
        has_cpp = "cpp" in toolchains or "cmake" in toolchains or "g++" in toolchains or "clang" in toolchains
        if "python" not in toolchains or not has_cpp:
            raise BlueprintValidationError(
                f"Prompt requests a hybrid C++/Python build, but toolchains {toolchains!r} are missing 'python' or 'cpp'",
                output="Include both 'python' and 'cpp' (or 'cmake') in toolchains.",
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


def _is_matrix_like_function(node: ast.FunctionDef) -> bool:
    """True when a function looks like it should return a structured matrix/array."""
    return (
        node.returns is not None
        and _annotation_is_nested_list(node.returns)
    ) or _suggests_matrix_or_array(node.name)


def _matrix_element_zero(returns: Optional[ast.AST]) -> Any:
    """Choose ``0`` or ``0.0`` based on the innermost element type annotation."""
    inner: Optional[ast.AST] = returns
    while isinstance(inner, ast.Subscript):
        if isinstance(inner.value, ast.Name) and inner.value.id in {"list", "List"}:
            inner = inner.slice
        else:
            break
    if isinstance(inner, ast.Name):
        if inner.id in {"int", "bool", "i32", "i64"}:
            return 0
        if inner.id in {"float", "f32", "f64", "double"}:
            return 0.0
    return 0.0


def _len_expr(name: str) -> ast.Call:
    return ast.Call(
        func=ast.Name(id="len", ctx=ast.Load()),
        args=[ast.Name(id=name, ctx=ast.Load())],
        keywords=[],
    )


def _cols_expr(name: Optional[str]) -> ast.expr:
    """Return a safe column-count expression: ``len(b[0]) if b and b[0] else 0``."""
    if not name:
        return ast.Constant(value=0)
    name_node = ast.Name(id=name, ctx=ast.Load())
    sub_zero = ast.Subscript(value=name_node, slice=ast.Constant(value=0), ctx=ast.Load())
    test = ast.BoolOp(
        op=ast.And(),
        values=[name_node, sub_zero],
    )
    body = ast.Call(
        func=ast.Name(id="len", ctx=ast.Load()),
        args=[sub_zero],
        keywords=[],
    )
    return ast.IfExp(test=test, body=body, orelse=ast.Constant(value=0))


def _zero_matrix_expr(rows_expr: ast.expr, cols_expr: ast.expr, zero: Any) -> ast.ListComp:
    """Build ``[[zero] * cols for _ in range(rows)]`` as an AST expression."""
    elt = ast.BinOp(
        left=ast.List(elts=[ast.Constant(value=zero)], ctx=ast.Load()),
        op=ast.Mult(),
        right=cols_expr,
    )
    target = ast.Name(id="_", ctx=ast.Store())
    iter_call = ast.Call(
        func=ast.Name(id="range", ctx=ast.Load()),
        args=[rows_expr],
        keywords=[],
    )
    generator = ast.comprehension(target=target, iter=iter_call, ifs=[], is_async=0)
    return ast.ListComp(elt=elt, generators=[generator])


def rewrite_empty_matrix_returns(source: str) -> str:
    """Rewrite bare ``return []`` in matrix/array functions to a zero-filled shape.

    Uses the function's argument names and return type to produce a safe
    ``[[0.0] * cols for _ in range(rows)]`` list-comprehension. If no rewrite is
    needed, the original source text is returned unchanged.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return source

    changed = False
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef):
            continue
        if not _is_matrix_like_function(node):
            continue

        args = [a.arg for a in node.args.args + node.args.posonlyargs + node.args.kwonlyargs]
        zero = _matrix_element_zero(node.returns)

        for stmt in list(ast.walk(node)):
            if _is_empty_list_return(stmt) and isinstance(stmt, ast.Return):
                rows_expr = _len_expr(args[0]) if args else ast.Constant(value=0)
                cols_expr = _cols_expr(args[1]) if len(args) > 1 else ast.Constant(value=0)
                stmt.value = _zero_matrix_expr(rows_expr, cols_expr, zero)
                changed = True

    if not changed:
        return source

    ast.fix_missing_locations(tree)
    return ast.unparse(tree)


def _check_empty_matrix_returns(tree: ast.AST) -> None:
    """Reject ``return []`` in matrix/array functions that would discard target dimensions.

    The error message advertises ``rewrite_empty_matrix_returns()`` so callers can
    auto-correct the source before writing it to the workspace.
    """
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef):
            continue
        if not _is_matrix_like_function(node):
            continue
        for stmt in ast.walk(node):
            if _is_empty_list_return(stmt):
                raise ValidationError(
                    f"Function '{node.name}' returns an empty list{_format_line(stmt)}, which discards "
                    "the expected matrix/array dimensions. Apply rewrite_empty_matrix_returns() "
                    "or return a zero-filled structure with the correct target shape "
                    "(e.g. [[0.0] * cols for _ in range(rows)]).",
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


def _check_class_initialization(tree: ast.AST) -> None:
    """Reject classes that lack an accessible ``__init__`` method.

    ``@dataclass`` and ``Enum``/``IntEnum`` classes are exempt because they
    receive an auto-generated constructor.
    """
    from aero_forge.scaffold.syntax_guard import _is_auto_initialized_class

    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        if _is_auto_initialized_class(node):
            continue
        has_init = any(
            isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef))
            and stmt.name == "__init__"
            for stmt in node.body
        )
        if not has_init:
            raise ValidationError(
                f"Class '{node.name}' is missing an __init__ method{_format_line(node)}. "
                "Every generated class must declare an explicit constructor.",
                output="",
            )


def validate_blueprint_class_contracts(
    workspace_root: Path,
    blueprint_path: Optional[Path] = None,
) -> None:
    """Verify classes and methods referenced by blueprint contracts exist in sources.

    Contract names of the form ``ClassName.__init__`` or ``ClassName.method`` are
    resolved against Python files declared in the blueprint manifest.  Missing
    classes or methods raise ``ValidationError`` before any file is promoted.
    """
    bp_path = blueprint_path or workspace_root / "blueprint.aero"
    if not bp_path.is_file():
        return

    from aero_forge.blueprint import parse_blueprint

    try:
        blueprint = parse_blueprint(bp_path)
    except Exception:
        return

    contracts = getattr(blueprint, "contracts", []) or getattr(blueprint, "abi_contracts", [])
    manifest = getattr(blueprint, "manifest", [])
    py_files = [
        workspace_root / entry.path
        for entry in manifest
        if str(getattr(entry, "path", "")).endswith(".py")
    ]

    for contract in contracts:
        name = getattr(contract, "name", "") or getattr(contract, "export_symbol", "")
        if "." not in name:
            continue
        class_name, member_name = name.split(".", 1)
        found_class = False
        found_member = False
        for path in py_files:
            if not path.is_file():
                continue
            try:
                tree = ast.parse(path.read_text(encoding="utf-8"))
            except (SyntaxError, OSError, UnicodeDecodeError):
                continue
            for class_node in ast.walk(tree):
                if not isinstance(class_node, ast.ClassDef):
                    continue
                if class_node.name != class_name:
                    continue
                found_class = True
                if member_name == "__init__":
                    found_member = any(
                        isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef))
                        and stmt.name == "__init__"
                        for stmt in class_node.body
                    )
                else:
                    found_member = any(
                        isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef))
                        and stmt.name == member_name
                        for stmt in class_node.body
                    )
                if found_member:
                    break
            if found_class and found_member:
                break

        if not found_class:
            raise ValidationError(
                f"Contract '{name}' references class '{class_name}' not found in generated sources",
                output="",
            )
        if not found_member:
            raise ValidationError(
                f"Contract '{name}' references method '{member_name}' not found in class '{class_name}'",
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
    _check_class_initialization(tree)


class BuildOutputError(ValidationError):
    """Raised when a build output directory is missing expected artifacts."""

    pass


def validate_build_outputs(
    output_dir: Path,
    blueprint: Optional[Any] = None,
) -> None:
    """Fail the build pass if the output workspace contains zero artifacts.

    Build artifacts are files produced by the compilation step (``.so``, ``.dll``,
    ``.dylib``, ``.rlib``, ``.wasm``, ``.exe``, etc.).  Source files declared in
    ``blueprint.manifest`` are validated separately by
    ``validate_blueprint_manifest`` against the workspace root.
    """
    output = Path(output_dir)
    files = [p for p in output.rglob("*") if p.is_file()]
    if not files:
        raise BuildOutputError(
            f"Build output directory {output} contains zero artifacts",
            output="",
        )

    artifact_extensions = {".so", ".dll", ".dylib", ".rlib", ".a", ".wasm", ".exe"}
    if blueprint is not None:
        manifest = getattr(blueprint, "manifest", None) or []
        missing_artifacts: List[str] = []
        for entry in manifest:
            path = getattr(entry, "path", None)
            if not path:
                continue
            # Only validate manifest entries that are declared build artifacts.
            if Path(path).suffix not in artifact_extensions:
                continue
            candidate = output / path
            if not candidate.is_file():
                missing_artifacts.append(str(path))
        if missing_artifacts:
            raise BuildOutputError(
                f"Missing declared build artifact {missing_artifacts[0]} from blueprint.aero",
                output=f"Missing build artifacts: {', '.join(missing_artifacts)}",
            )

        toolchains = getattr(blueprint, "toolchains", []) or []
        native_toolchains = {"rust", "cargo", "cpp", "cmake", "go", "npm"}
        if any(t in native_toolchains for t in toolchains):
            has_native_artifact = any(
                f.suffix in artifact_extensions for f in files
            ) or any(
                f.is_dir() and any(g.suffix in artifact_extensions for g in f.rglob("*"))
                for f in output.iterdir()
            )
            if not has_native_artifact:
                raise BuildOutputError(
                    "Build output directory contains no compiled native artifacts",
                    output=f"Files in {output}: {', '.join(str(f.name) for f in files[:10])}",
                )


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
            if command and command[0] == "cargo" and "build" in command:
                target = None
                if "--target" in command:
                    target = command[command.index("--target") + 1]
                result = cargo_build(
                    workspace_root,
                    release="--release" in command,
                    target=target,
                    timeout=300,
                )
            elif command and command[0] == "maturin" and "build" in command:
                result = maturin_build(workspace_root, timeout=300)
            else:
                result = subprocess.run(
                    _deduplicate_command_args(command),
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

        # Repair common LLM truncation in Rust/C/C++ sources before validation.
        if lang in {"rust", "c", "cpp", "c++"}:
            from aero_forge.scaffold.syntax_guard import repair_workspace

            repair_workspace(workspace)

        # Generator-side static analysis runs first so bad patterns are caught
        # before any sandboxed command or filesystem promotion.
        # Enforce the workspace blueprint before any per-file checks.
        validate_blueprint_manifest(workspace)
        validate_blueprint_class_contracts(workspace)

        if lang == "python":
            ensure_package_structure(workspace)
            from aero_forge.scaffold.syntax_guard import normalize_python_module

            for path in workspace.rglob("*.py"):
                try:
                    text = path.read_text(encoding="utf-8")
                    normalized = normalize_python_module(text)
                    if normalized != text:
                        path.write_text(normalized, encoding="utf-8")
                    _run_python_static_checks(normalized)
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
