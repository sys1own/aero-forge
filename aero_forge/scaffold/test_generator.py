"""Synthetic pytest generation from implementation signatures."""

from __future__ import annotations

import ast
from typing import Any, List, Optional, Tuple


def _base_name(node: ast.AST) -> str:
    """Return the final attribute/name part of a type expression."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return ""


def _type_args(node: ast.AST) -> List[ast.AST]:
    """Return the argument nodes inside a generic subscript."""
    if isinstance(node, ast.Index):  # pragma: no cover  # Python 3.8 compatibility
        return _type_args(node.value)
    if isinstance(node, ast.Tuple):
        return list(node.elts)
    return [node]


def _is_none(node: Optional[ast.AST]) -> bool:
    if node is None:
        return True
    if isinstance(node, ast.Constant) and node.value is None:
        return True
    if isinstance(node, ast.Name) and node.id == "None":
        return True
    return False


def _scalar_literal(name: str) -> str:
    """Return a valid Python literal string for a scalar type name."""
    mapping = {
        "str": '"project_name"',
        "int": "1",
        "float": "1.0",
        "bool": "True",
        "bytes": 'b"test"',
        "None": "None",
    }
    return mapping.get(name, "None")


def _value_for_annotation(node: Optional[ast.AST], *, generic_fallback: bool = True) -> str:
    """Produce a single Python literal string matching an annotation."""
    if node is None or _is_none(node):
        return "None"

    if isinstance(node, ast.Constant):
        return repr(node.value)

    if isinstance(node, ast.Name):
        name = node.id
        if name in {"list", "List", "Sequence"}:
            return "[]" if generic_fallback else '["fn_a", "fn_b"]'
        if name in {"dict", "Dict", "Mapping"}:
            return "{}"
        if name in {"tuple", "Tuple"}:
            return "(1, 2)"
        if name in {"set", "Set"}:
            return "{1, 2}"
        return _scalar_literal(name)

    if isinstance(node, ast.Attribute):
        return _value_for_annotation(ast.Name(id=_base_name(node)), generic_fallback=generic_fallback)

    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
        left = _value_for_annotation(node.left, generic_fallback=generic_fallback)
        right = _value_for_annotation(node.right, generic_fallback=generic_fallback)
        if left == "None":
            return right
        if right == "None":
            return left
        return left

    if isinstance(node, ast.Subscript):
        base = _base_name(node.value).lower()
        args = _type_args(node.slice)
        if base in {"list", "sequence"}:
            if not args:
                return "[]"
            inner = _value_for_annotation(args[0], generic_fallback=False)
            if inner.startswith("[") or inner.startswith("{") or inner.startswith("("):
                return f"[{inner}, {inner}]"
            return f"[{inner}, {inner}, {inner}]"
        if base in {"dict", "mapping"}:
            if len(args) >= 2:
                key = _value_for_annotation(args[0], generic_fallback=False)
                val = _value_for_annotation(args[1], generic_fallback=False)
                return f"{{{key}: {val}}}"
            return "{}"
        if base in {"tuple", "tuple_"}:
            parts = [_value_for_annotation(a, generic_fallback=False) for a in args]
            if len(parts) == 1:
                return f"({parts[0]},)"
            return f"({', '.join(parts)})"
        if base in {"set", "frozen_set"}:
            if args:
                inner = _value_for_annotation(args[0], generic_fallback=False)
                if inner in {"None", "[]", "{}", "()"}:
                    return "set()"
                return f"{{{inner}, {inner}}}"
            return "{1, 2}"
        if base == "optional":
            if args and not _is_none(args[0]):
                return _value_for_annotation(args[0], generic_fallback=False)
            return "None"
        if base == "union":
            for arg in args:
                if not _is_none(arg):
                    return _value_for_annotation(arg, generic_fallback=False)
            return "None"
        if base == "literal":
            for arg in args:
                if isinstance(arg, ast.Constant):
                    return repr(arg.value)
            return _value_for_annotation(args[0], generic_fallback=False)
        # Unknown generic: fall back to the base name as a scalar/container.
        return _value_for_annotation(ast.Name(id=_base_name(node.value)), generic_fallback=generic_fallback)

    return "None"


def _type_name(node: Optional[ast.AST]) -> str:
    """Return a normalized base type name for an annotation."""
    if node is None or _is_none(node):
        return "none"
    if isinstance(node, ast.Name):
        return node.id.lower()
    if isinstance(node, ast.Attribute):
        return node.attr.lower()
    if isinstance(node, ast.Subscript):
        return _type_name(node.value)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
        left = _type_name(node.left)
        right = _type_name(node.right)
        if left == "none":
            return right
        if right == "none":
            return left
        return left
    return ""


def _assertion_for_call(call: str, returns: Optional[ast.AST]) -> str:
    """Return a pytest assertion body for a function call."""
    name = _type_name(returns)
    if name in {"none"}:
        return f"    result = {call}\n    assert result is None"
    if name == "bool":
        return f"    assert {call} in (True, False)"
    if name in {"list", "sequence"}:
        return f"    result = {call}\n    assert isinstance(result, list)"
    if name in {"dict", "mapping"}:
        return f"    result = {call}\n    assert isinstance(result, dict)"
    if name in {"tuple"}:
        return f"    result = {call}\n    assert isinstance(result, tuple)"
    if name in {"set", "frozenset"}:
        return f"    result = {call}\n    assert isinstance(result, set)"
    if name in {"int", "float"}:
        return f"    result = {call}\n    assert isinstance(result, (int, float))"
    if name == "str":
        return f"    result = {call}\n    assert isinstance(result, str)"
    if name == "bytes":
        return f"    result = {call}\n    assert isinstance(result, bytes)"
    return f"    result = {call}\n    assert result is not None"


def generate_smoke_tests(implementation: str, module_name: str = "generated") -> str:
    """Generate pytest smoke tests from the implementation when none were provided.

    Uses AST inspection of parameter annotations so synthesized test arguments
    match declared types (``str`` gets a string, ``list`` gets a list, etc.).
    Integer fallbacks are never used for ``str`` or ``list`` parameters.
    """
    try:
        tree = ast.parse(implementation)
    except SyntaxError:
        return ""

    test_lines: List[str] = []
    for item in tree.body:
        if not isinstance(item, ast.FunctionDef):
            continue
        name = item.name
        if name.startswith("_"):
            continue

        args: List[str] = []
        for arg in item.args.args + item.args.posonlyargs + item.args.kwonlyargs:
            if arg.arg in {"self", "cls"}:
                continue
            args.append(_value_for_annotation(arg.annotation))

        call = f"{name}({', '.join(args)})"
        assertion = _assertion_for_call(call, item.returns)
        test_lines.append(f"def test_{name}():\n{assertion}\n")

    if not test_lines:
        return ""

    all_names = sorted(
        item.name
        for item in tree.body
        if isinstance(item, ast.FunctionDef) and not item.name.startswith("_")
    )
    imports = "\n".join(f"from {module_name} import {n}" for n in all_names)
    return imports + "\n\n" + "\n".join(test_lines)
