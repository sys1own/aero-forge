"""Normalize a subset of Python source into a simple UAST module.

The UAST dialect is intentionally small: module, function_declaration,
binding, reference, literal, if, and call. It is enough to drive the
translator and the precision shield for numeric functions.
"""

from __future__ import annotations

import ast
from typing import List, Optional

from aero_forge._constants import (
    IO_MODULES,
    IO_NAMES,
    MATH_CONSTANTS,
    SAFE_BUILTINS,
    SAFE_STD_MODULES,
)
from aero_forge.errors import UnsupportedError


def python_source_to_uast(source: str) -> dict:
    """Parse Python ``source`` and return a normalized UAST ``module`` dict."""
    tree = ast.parse(source)
    _lower_expr.local_functions = {
        node.name for node in tree.body if isinstance(node, ast.FunctionDef)
    }
    children: List[dict] = []
    for stmt in tree.body:
        node = _lower_stmt(stmt)
        if node is not None:
            children.append(node)
    return {"type": "module", "children": children}


def _is_io_call(expr: ast.Call) -> bool:
    if isinstance(expr.func, ast.Name) and expr.func.id in IO_NAMES:
        return True
    if (
        isinstance(expr.func, ast.Attribute)
        and isinstance(expr.func.value, ast.Name)
        and expr.func.value.id in IO_MODULES
    ):
        return True
    return False


def _call_base_and_name(expr: ast.Call):
    """Return (base, name) for a call, e.g. ("math", "sin") for math.sin(x)."""
    if isinstance(expr.func, ast.Name):
        return (None, expr.func.id)
    if isinstance(expr.func, ast.Attribute) and isinstance(expr.func.value, ast.Name):
        return (expr.func.value.id, expr.func.attr)
    return (None, "")


def _lower_external_call(expr: ast.Call) -> dict:
    """Lower a call to a generic external function stub used by the HIN VM."""
    base, name = _call_base_and_name(expr)
    args = [_lower_expr(a) for a in expr.args]
    if base is None:
        function_name = name
    elif base in SAFE_STD_MODULES:
        function_name = f"{base}.{name}"
    else:
        # Generic object method: include the receiver as the first argument.
        function_name = f"{base}.{name}"
        args = [_lower_expr(expr.func.value)] + args
    return {
        "type": "call",
        "function": {"type": "reference", "name": function_name},
        "argument": args[0] if args else None,
        "arguments": args,
    }


def _lower_stmt(stmt: ast.stmt) -> Optional[dict]:
    # Import statements have no runtime effect in the generated numeric code;
    # skip them so safe standard library imports do not fail lowering.
    if isinstance(stmt, (ast.Import, ast.ImportFrom)):
        return None
    if isinstance(stmt, (ast.With, ast.AsyncWith)):
        raise UnsupportedError(
            "with statements / context managers are not supported", node=stmt
        )
    if isinstance(stmt, (ast.Try, getattr(ast, "TryStar", ()))):
        raise UnsupportedError(
            "try/except exception handling is not supported", node=stmt
        )
    if isinstance(stmt, (ast.Yield, ast.YieldFrom)):
        raise UnsupportedError("yield / generators are not supported", node=stmt)
    if isinstance(stmt, ast.AsyncFor):
        raise UnsupportedError("async for loops are not supported", node=stmt)
    if isinstance(stmt, ast.AsyncFunctionDef):
        raise UnsupportedError("async/await is not supported", node=stmt)
    if isinstance(stmt, ast.Match):
        raise UnsupportedError("match/case is not supported", node=stmt)
    if isinstance(stmt, ast.FunctionDef):
        params = [a.arg for a in stmt.args.args]
        body = [n for n in (_lower_stmt(s) for s in stmt.body) if n is not None]
        return {
            "type": "function_declaration",
            "name": stmt.name,
            "params": params,
            "param": params[0] if params else None,
            "body": body,
        }
    if isinstance(stmt, ast.Assign) and stmt.targets:
        target = stmt.targets[0]
        if isinstance(target, ast.Name):
            return {
                "type": "binding",
                "name": target.id,
                "value": _lower_expr(stmt.value),
            }
    if isinstance(stmt, ast.Return):
        return _lower_expr(stmt.value) if stmt.value is not None else None
    if isinstance(stmt, ast.If):
        return _lower_if(stmt)
    if isinstance(stmt, ast.Expr):
        return _lower_expr(stmt.value)
    return None


def _lower_if(stmt: ast.If) -> dict:
    then_body = [n for n in (_lower_stmt(s) for s in stmt.body) if n is not None]
    else_body = [n for n in (_lower_stmt(s) for s in stmt.orelse) if n is not None]
    return {
        "type": "if",
        "condition": _lower_expr(stmt.test),
        "then": then_body[-1] if then_body else None,
        "else": else_body[-1] if else_body else None,
    }


def _cmp_op_name(op: ast.cmpop) -> str:
    return {
        ast.Eq: "==",
        ast.NotEq: "!=",
        ast.Lt: "<",
        ast.LtE: "<=",
        ast.Gt: ">",
        ast.GtE: ">=",
    }.get(type(op), "compare")


def _lower_expr(expr: Optional[ast.expr]) -> Optional[dict]:
    if isinstance(expr, ast.NamedExpr):
        raise UnsupportedError("walrus operator (:=) is not supported", node=expr)
    if isinstance(expr, (ast.Await, ast.Yield, ast.YieldFrom)):
        raise UnsupportedError(
            "async/await and yield expressions are not supported", node=expr
        )
    if isinstance(expr, ast.ListComp):
        # The scaffold engine generates the actual Rust; the UAST frontend only
        # needs to avoid raising so the precision shield can analyze the source.
        return {
            "type": "call",
            "function": {"type": "reference", "name": "list"},
            "argument": None,
        }
    if expr is None:
        return None
    if isinstance(expr, ast.Constant):
        return {"type": "literal", "value": expr.value}
    if isinstance(expr, ast.Name):
        return {"type": "reference", "name": expr.id}
    if isinstance(expr, ast.Call):
        if _is_io_call(expr):
            base, name = _call_base_and_name(expr)
            if base in SAFE_STD_MODULES or name in SAFE_BUILTINS:
                return _lower_external_call(expr)
            raise UnsupportedError("io", node=expr)
        local_functions = getattr(_lower_expr, "local_functions", set())
        callee_name = expr.func.id if isinstance(expr.func, ast.Name) else None
        args = [_lower_expr(a) for a in expr.args]
        call_type = "user_function_call" if callee_name in local_functions else "call"
        return {
            "type": call_type,
            "function": _lower_expr(expr.func),
            "argument": args[0] if args else None,
            "arguments": args,
        }
    if isinstance(expr, ast.IfExp):
        return {
            "type": "if",
            "condition": _lower_expr(expr.test),
            "then": _lower_expr(expr.body),
            "else": _lower_expr(expr.orelse),
        }
    if isinstance(expr, ast.BinOp):
        return {
            "type": "call",
            "function": _lower_expr(expr.left),
            "argument": _lower_expr(expr.right),
        }
    if isinstance(expr, ast.UnaryOp):
        op_name = {
            ast.UAdd: "pos",
            ast.USub: "neg",
            ast.Not: "not",
            ast.Invert: "invert",
        }.get(type(expr.op), "unary")
        return {
            "type": "call",
            "function": {"type": "reference", "name": f"__unary_{op_name}__"},
            "argument": _lower_expr(expr.operand),
        }
    if isinstance(expr, ast.Compare) and len(expr.ops) == 1:
        return {
            "type": "call",
            "function": {
                "type": "reference",
                "name": f"__compare__{_cmp_op_name(expr.ops[0])}",
            },
            "argument": {
                "type": "literal",
                "value": [
                    _lower_expr(expr.left),
                    _lower_expr(expr.comparators[0]),
                ],
            },
        }
    if isinstance(expr, ast.BoolOp):
        op_name = "and" if isinstance(expr.op, ast.And) else "or"
        value = [_lower_expr(v) for v in expr.values]
        return {
            "type": "call",
            "function": {"type": "reference", "name": f"__boolop_{op_name}__"},
            "argument": {"type": "literal", "value": value},
        }
    if isinstance(expr, ast.Attribute):
        if (
            isinstance(expr.value, ast.Name)
            and expr.value.id == "math"
            and expr.attr in MATH_CONSTANTS
        ):
            return {"type": "literal", "value": MATH_CONSTANTS[expr.attr]}
        if isinstance(expr.value, ast.Name):
            # Generic attribute access becomes an external reference so that
            # object fields and safe stdlib attributes do not fail lowering.
            return {"type": "reference", "name": f"{expr.value.id}.{expr.attr}"}
        return _lower_expr(expr.value)
    if isinstance(expr, ast.Dict):
        return {
            "type": "literal",
            "value": {
                "type": "dict",
                "pairs": [
                    {"key": _lower_expr(k), "value": _lower_expr(v)}
                    for k, v in zip(expr.keys, expr.values)
                    if k is not None
                ],
            },
        }
    return None


__all__ = ["python_source_to_uast"]
