"""Syntax guard for Rust and C/C++ source files.

Repairs common LLM truncation failures:

- Unbalanced braces, brackets, and parentheses.
- Dangling ``///`` / ``//!`` doc-comment lines at the end of a file.
- Unclosed ``/* ... */`` block comments at the end of a file.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import List, Tuple

# File extensions that this guard can repair.
GUARDED_EXTENSIONS = {".rs", ".cpp", ".c", ".h", ".hpp"}


def _balanced_tokens(source: str) -> Tuple[bool, List[str]]:
    """Return (ok, stack) for ``()``, ``[]``, and ``{}`` pairs.

    Strings and comments are skipped so punctuation inside them is ignored.
    """
    stack: List[str] = []
    i = 0
    n = len(source)
    in_line_comment = False
    in_block_comment = False
    in_string: str | None = None

    while i < n:
        c = source[i]
        nxt = source[i + 1] if i + 1 < n else ""

        if in_line_comment:
            if c == "\n":
                in_line_comment = False
            i += 1
            continue

        if in_block_comment:
            if c == "*" and nxt == "/":
                in_block_comment = False
                i += 2
                continue
            i += 1
            continue

        if in_string:
            if c == "\\":
                # Skip the escaped character.
                i += 2
                continue
            if c == in_string:
                in_string = None
            i += 1
            continue

        if c == "/" and nxt == "/":
            in_line_comment = True
            i += 2
            continue

        if c == "/" and nxt == "*":
            in_block_comment = True
            i += 2
            continue

        if c in ('"', "'"):
            # In Rust, a leading ' followed by an identifier is a lifetime (e.g. 'py, '_).
            # Treat it as a single token rather than a string delimiter.
            if c == "'" and (nxt.isalpha() or nxt == "_"):
                i += 1
                while i < n and (source[i].isalnum() or source[i] == "_"):
                    i += 1
                continue
            in_string = c
            i += 1
            continue

        if c in "([{":
            stack.append(c)
        elif c in ")]}]":
            if not stack:
                return False, []
            opening = stack.pop()
            if (
                (opening == "(" and c != ")")
                or (opening == "[" and c != "]")
                or (opening == "{" and c != "}")
            ):
                return False, []

        i += 1

    return True, stack


def _is_in_block_comment(source: str) -> bool:
    """Return True if the file ends while still inside a ``/* ... */`` block."""
    i = 0
    n = len(source)
    in_line_comment = False
    in_block_comment = False
    in_string: str | None = None

    while i < n:
        c = source[i]
        nxt = source[i + 1] if i + 1 < n else ""

        if in_line_comment:
            if c == "\n":
                in_line_comment = False
            i += 1
            continue

        if in_block_comment:
            if c == "*" and nxt == "/":
                in_block_comment = False
                i += 2
                continue
            i += 1
            continue

        if in_string:
            if c == "\\":
                i += 2
                continue
            if c == in_string:
                in_string = None
            i += 1
            continue

        if c == "/" and nxt == "/":
            in_line_comment = True
            i += 2
            continue

        if c == "/" and nxt == "*":
            in_block_comment = True
            i += 2
            continue

        if c in ('"', "'"):
            in_string = c
            i += 1
            continue

        i += 1

    return in_block_comment


def _last_unclosed_block_comment_start(source: str) -> int:
    """Find the start index of the last unclosed ``/*`` block comment."""
    # Scan from the end using the same state machine to find the last unclosed /*.
    # We do a forward pass but remember the position of the most recent unclosed /*.
    i = 0
    n = len(source)
    in_line_comment = False
    in_block_comment = False
    in_string: str | None = None
    last_open = -1

    while i < n:
        c = source[i]
        nxt = source[i + 1] if i + 1 < n else ""

        if in_line_comment:
            if c == "\n":
                in_line_comment = False
            i += 1
            continue

        if in_block_comment:
            if c == "*" and nxt == "/":
                in_block_comment = False
                i += 2
                continue
            i += 1
            continue

        if in_string:
            if c == "\\":
                i += 2
                continue
            if c == in_string:
                in_string = None
            i += 1
            continue

        if c == "/" and nxt == "/":
            in_line_comment = True
            i += 2
            continue

        if c == "/" and nxt == "*":
            in_block_comment = True
            last_open = i
            i += 2
            continue

        if c in ('"', "'"):
            in_string = c
            i += 1
            continue

        i += 1

    return last_open if in_block_comment else -1


def _strip_dangling_doc_comments(source: str) -> str:
    """Remove trailing ``///`` / ``//!`` lines and unclosed ``/*`` blocks.

    Returns the original string if nothing is removed so callers can detect
    "no change".
    """
    original = source
    changed = False

    # Remove trailing blank lines and `///` / `//!` doc comments.
    lines = source.rstrip().splitlines()
    while lines:
        stripped = lines[-1].strip()
        if stripped == "" or stripped.startswith("///") or stripped.startswith("//!"):
            lines.pop()
            changed = True
        else:
            break
    source = "\n".join(lines)

    # If the file ends inside a block comment, truncate before the last unclosed /*.
    while _is_in_block_comment(source):
        idx = _last_unclosed_block_comment_start(source)
        if idx < 0:
            break
        source = source[:idx]
        changed = True
        # Re-run doc-comment stripping for the truncated source.
        lines = source.rstrip().splitlines()
        while lines:
            stripped = lines[-1].strip()
            if (
                stripped == ""
                or stripped.startswith("///")
                or stripped.startswith("//!")
            ):
                lines.pop()
                changed = True
            else:
                break
        source = "\n".join(lines)

    if not changed:
        return original
    return source.rstrip() + "\n"


def repair_source(source: str) -> str:
    """Repair a single Rust/C/C++ source string.

    Returns the repaired source. If no repair is needed the original content is
    returned unchanged.
    """
    repaired = _strip_dangling_doc_comments(source)
    ok, stack = _balanced_tokens(repaired)
    if not ok:
        # Mismatched closing token; we cannot safely auto-repair.
        return source

    if stack:
        closings = {"(": ")", "[": "]", "{": "}"}
        suffix = "".join(closings[op] for op in reversed(stack))
        repaired = repaired.rstrip() + "\n" + suffix + "\n"

    if repaired == source:
        return source
    return repaired


def repair_file(path: Path) -> bool:
    """Repair a single source file in place. Returns True if changed."""
    path = Path(path)
    if path.suffix.lower() not in GUARDED_EXTENSIONS:
        return False
    try:
        original = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return False
    repaired = repair_source(original)
    if repaired != original:
        path.write_text(repaired, encoding="utf-8")
        return True
    return False


def repair_workspace(workspace: Path) -> List[Path]:
    """Repair all guarded source files under *workspace*."""
    changed: List[Path] = []
    for path in Path(workspace).rglob("*"):
        if path.is_file() and path.suffix.lower() in GUARDED_EXTENSIONS:
            if repair_file(path):
                changed.append(path)
    return changed


def ensure_typing_imports(source: str) -> str:
    """Python syntax guard: inject missing ``typing`` imports for annotation symbols.

    This delegates to :func:`aero_forge.scaffold.import_pruner.ensure_typing_imports`
    so the Python syntax guard and the import pruner share the same AST-based
    implementation.
    """
    from aero_forge.scaffold.import_pruner import ensure_typing_imports as _ensure

    return _ensure(source)


def _is_auto_initialized_class(node: ast.ClassDef) -> bool:
    """Classes decorated with ``@dataclass`` or inheriting ``Enum`` already get init."""
    for decorator in node.decorator_list:
        name = ""
        if isinstance(decorator, ast.Name):
            name = decorator.id
        elif isinstance(decorator, ast.Attribute):
            name = decorator.attr
        elif isinstance(decorator, ast.Call):
            func = decorator.func
            if isinstance(func, ast.Name):
                name = func.id
            elif isinstance(func, ast.Attribute):
                name = func.attr
        if name == "dataclass":
            return True

    enum_bases = {"Enum", "IntEnum", "Flag", "IntFlag"}
    for base in node.bases:
        if isinstance(base, ast.Name) and base.id in enum_bases:
            return True
        if (
            isinstance(base, ast.Attribute)
            and isinstance(base.value, ast.Name)
            and base.value.id == "enum"
            and base.attr in enum_bases
        ):
            return True
    return False


class ClassInitNormalizer(ast.NodeTransformer):
    """Guarantee every class has an explicit ``__init__`` method.

    Classes that already define ``__init__`` are left unchanged.  For classes
    without one, a default ``__init__`` is synthesized from class-level attribute
    assignments/annotations when possible, otherwise a permissive
    ``__init__(self, *args, **kwargs)`` stub is injected.

    ``@dataclass`` and ``Enum``/``IntEnum`` classes are skipped because those
    mechanisms already provide an appropriate constructor.
    """

    def visit_ClassDef(self, node: ast.ClassDef) -> ast.ClassDef:  # noqa: N802
        has_init = any(
            isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef))
            and stmt.name == "__init__"
            for stmt in node.body
        )
        if has_init or _is_auto_initialized_class(node):
            return self.generic_visit(node)

        fields: List[Tuple[str, ast.expr | None, ast.expr | None]] = []
        body: List[ast.stmt] = []

        for stmt in node.body:
            name: str | None = None
            annotation: ast.expr | None = None
            default: ast.expr | None = None

            if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
                name = stmt.target.id
                annotation = stmt.annotation
                default = stmt.value
            elif isinstance(stmt, ast.Assign):
                # Only consider simple single-target assignments to a name.
                if (
                    len(stmt.targets) == 1
                    and isinstance(stmt.targets[0], ast.Name)
                ):
                    name = stmt.targets[0].id
                    default = stmt.value

            if name is None or name.startswith("_"):
                continue

            fields.append((name, annotation, default))
            body.append(
                ast.Assign(
                    targets=[
                        ast.Attribute(
                            value=ast.Name(id="self", ctx=ast.Load()),
                            attr=name,
                            ctx=ast.Store(),
                        )
                    ],
                    value=ast.Name(id=name, ctx=ast.Load()),
                )
            )

        if not fields:
            # No meaningful fields discovered: use a permissive stub.
            args = ast.arguments(
                posonlyargs=[],
                args=[ast.arg(arg="self", annotation=None)],
                kwonlyargs=[],
                kw_defaults=[],
                defaults=[],
                vararg=ast.arg(arg="args", annotation=None),
                kwarg=ast.arg(arg="kwargs", annotation=None),
            )
            init_body: List[ast.stmt] = [ast.Pass()]
        else:
            # Build arguments with defaults from annotated/assigned class fields.
            # All parameters without defaults must precede those with defaults.
            args_without_default: List[ast.arg] = []
            args_with_default: List[ast.arg] = []
            defaults: List[ast.expr] = []
            for name, annotation, default in fields:
                arg = ast.arg(arg=name, annotation=annotation)
                if default is not None:
                    args_with_default.append(arg)
                    defaults.append(default)
                else:
                    args_without_default.append(arg)

            all_params = [ast.arg(arg="self", annotation=None)] + args_without_default + args_with_default
            args = ast.arguments(
                posonlyargs=[],
                args=all_params,
                kwonlyargs=[],
                kw_defaults=[],
                defaults=defaults,
                vararg=None,
                kwarg=None,
            )
            init_body = body

        init = ast.FunctionDef(
            name="__init__",
            args=args,
            body=init_body,
            decorator_list=[],
            returns=None,
            type_comment=None,
        )
        node.body.insert(0, init)
        return ast.fix_missing_locations(self.generic_visit(node))


def normalize_python_module(source: str) -> str:
    """Parse *source*, ensure every class has ``__init__``, and re-emit.

    If no transformation is applied, the original *source* text is returned so
    comments and formatting are preserved whenever possible.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return source

    original_dump = ast.dump(tree)
    tree = ClassInitNormalizer().visit(tree)
    ast.fix_missing_locations(tree)
    if ast.dump(tree) == original_dump:
        return source
    try:
        return ast.unparse(tree)
    except Exception:  # pragma: no cover
        return source
