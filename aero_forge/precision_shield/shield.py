"""Precision shield: selects Rust types and traits from an analyzed graph."""

from __future__ import annotations

import ast
from typing import Any, Dict, List, Optional, Set

from ..errors import UnsupportedError

_FLOAT_MATH_FUNCS: Set[str] = {
    "sqrt",
    "sin",
    "cos",
    "tan",
    "exp",
    "log",
    "log10",
    "pow",
    "ceil",
    "floor",
    "trunc",
}
_BITWISE_OPS: Set[type] = {
    ast.LShift,
    ast.RShift,
    ast.BitOr,
    ast.BitXor,
    ast.BitAnd,
    ast.Invert,
}


class Shield:
    """Inspect a function/HIN graph and decide which Rust types/traits are needed."""

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.config = config or {}

    def analyze(
        self,
        graph: Any,
        func_name: Optional[str] = None,
        source: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Return required Rust traits/types for the function represented by ``graph``.

        The graph is accepted because the CLI flow passes the HIN network here,
        but concrete type decisions are made from the original Python AST.
        """
        func = None
        if source:
            try:
                tree = ast.parse(source)
            except SyntaxError as exc:
                raise ValueError(f"Could not parse source: {exc}") from exc
            func = _find_function(tree, func_name)

        if func is None:
            # Fallback to graph-level hints if the AST is unavailable.
            return {
                "function_name": func_name,
                "arg_types": ["i64"],
                "return_type": "i64",
                "function_type": "i64",
                "recursive": False,
                "traits": self._traits(["Integer"]),
            }

        arg_names = [a.arg for a in func.args.args]

        if _uses_bitwise(func):
            f64_vars = _f64_variables(func)
            if _bitwise_uses_f64(func, f64_vars):
                raise UnsupportedError(
                    "Bitwise operations are only supported on integer-typed values",
                    node=func,
                )
            # Bitwise operators constrain the whole function to integers.
            function_type = "i64"
        else:
            function_type = _infer_number_type(func)
            if (
                self.config.get("default_float") in ("double", "f64")
                and function_type == "i64"
            ):
                function_type = "f64"

        uses_float = function_type == "f64"
        recursive = _is_recursive(func)

        traits = ["Integer"]
        if uses_float:
            traits.append("Float")

        return {
            "function_name": func_name,
            "arg_types": [function_type] * len(arg_names),
            "return_type": function_type,
            "function_type": function_type,
            "arg_names": arg_names,
            "recursive": recursive,
            "traits": self._traits(traits),
        }

    def _traits(self, traits: List[str]) -> List[str]:
        if self.config.get("enable_rug") is False:
            return []
        return traits


def _find_function(tree: ast.AST, name: Optional[str]) -> Optional[ast.FunctionDef]:
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            if name is None or node.name == name:
                return node
    return None


def _infer_number_type(func: ast.FunctionDef) -> str:
    """Pick i64 or f64 for the whole function based on its literals/operators."""
    for node in ast.walk(func):
        if isinstance(node, ast.Constant) and isinstance(node.value, float):
            return "f64"
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
            return "f64"
        if isinstance(node, ast.Attribute):
            if (
                isinstance(node.value, ast.Name)
                and node.value.id == "math"
                and node.attr in ("pi", "e", "tau")
            ):
                return "f64"
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Attribute):
                base = node.func.value
                attr = node.func.attr
                if (
                    isinstance(base, ast.Name)
                    and base.id == "math"
                    and attr in _FLOAT_MATH_FUNCS
                ):
                    return "f64"
    return "i64"


def _is_recursive(func: ast.FunctionDef) -> bool:
    name = func.name
    for node in ast.walk(func):
        if isinstance(node, ast.Call):
            if _call_name(node) == name:
                return True
    return False


def _uses_bitwise(func: ast.FunctionDef) -> bool:
    """Return True if the function uses any bitwise operators or inversion."""
    for node in ast.walk(func):
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Invert):
            return True
        if isinstance(node, ast.BinOp) and type(node.op) in _BITWISE_OPS:
            return True
    return False


def _call_name(node: ast.Call) -> str:
    if isinstance(node.func, ast.Name):
        return node.func.id
    if isinstance(node.func, ast.Attribute):
        return node.func.attr
    return ""


def _is_f64_expr(expr: ast.expr, f64_vars: Set[str]) -> bool:
    """Return True if ``expr`` is definitely a floating-point expression."""
    if isinstance(expr, ast.Constant) and isinstance(expr.value, float):
        return True
    if isinstance(expr, ast.Name) and expr.id in f64_vars:
        return True
    if isinstance(expr, ast.BinOp) and isinstance(expr.op, ast.Div):
        return True
    if isinstance(expr, ast.Attribute):
        if (
            isinstance(expr.value, ast.Name)
            and expr.value.id == "math"
            and expr.attr in ("pi", "e", "tau")
        ):
            return True
    if isinstance(expr, ast.Call):
        if isinstance(expr.func, ast.Attribute):
            base = expr.func.value
            attr = expr.func.attr
            if (
                isinstance(base, ast.Name)
                and base.id == "math"
                and attr in _FLOAT_MATH_FUNCS
            ):
                return True
    if isinstance(expr, ast.UnaryOp) and isinstance(expr.op, (ast.UAdd, ast.USub)):
        return _is_f64_expr(expr.operand, f64_vars)
    if isinstance(expr, ast.BinOp):
        if type(expr.op) not in _BITWISE_OPS:
            return _is_f64_expr(expr.left, f64_vars) or _is_f64_expr(
                expr.right, f64_vars
            )
    if isinstance(expr, ast.IfExp):
        return _is_f64_expr(expr.body, f64_vars) or _is_f64_expr(expr.orelse, f64_vars)
    return False


def _f64_variables(func: ast.FunctionDef) -> Set[str]:
    """Return names that are assigned a floating-point expression."""
    f64_vars: Set[str] = set()
    changed = True
    while changed:
        changed = False
        for node in ast.walk(func):
            if isinstance(node, ast.Assign) and node.value is not None:
                for target in node.targets:
                    for name in _names_in_target(target):
                        if name not in f64_vars and _is_f64_expr(node.value, f64_vars):
                            f64_vars.add(name)
                            changed = True
            elif isinstance(node, ast.AnnAssign) and node.value is not None:
                for name in _names_in_target(node.target):
                    if name not in f64_vars and _is_f64_expr(node.value, f64_vars):
                        f64_vars.add(name)
                        changed = True
            elif isinstance(node, ast.AugAssign) and isinstance(node.target, ast.Name):
                if node.target.id not in f64_vars and _is_f64_expr(
                    node.value, f64_vars
                ):
                    f64_vars.add(node.target.id)
                    changed = True
    return f64_vars


def _names_in_target(target: ast.expr) -> List[str]:
    """Collect the simple names assigned by a target expression."""
    names: List[str] = []
    if isinstance(target, ast.Name):
        names.append(target.id)
    elif isinstance(target, (ast.Tuple, ast.List)):
        for elt in target.elts:
            names.extend(_names_in_target(elt))
    return names


def _bitwise_uses_f64(func: ast.FunctionDef, f64_vars: Set[str]) -> bool:
    """Return True if any bitwise operand is (or contains) an f64 expression."""
    for node in ast.walk(func):
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Invert):
            if _is_f64_expr(node.operand, f64_vars):
                return True
        elif isinstance(node, ast.BinOp) and type(node.op) in _BITWISE_OPS:
            if _is_f64_expr(node.left, f64_vars) or _is_f64_expr(node.right, f64_vars):
                return True
    return False
