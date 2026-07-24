"""Abstract, language-agnostic engine specification for polyglot emission."""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ASTNode:
    """A single node in an abstract syntax specification.

    This is intentionally smaller than a full compiler AST: it captures only the
    constructs needed to describe computational engines (functions, types,
    bindings, control flow, and expressions) and delegates language-specific
    rendering to emitters.
    """

    kind: str
    name: Optional[str] = None
    type_hint: Optional[str] = None
    value: Any = None
    children: List[ASTNode] = field(default_factory=list)

    @property
    def params(self) -> List[ASTNode]:
        return [c for c in self.children if c.kind == "param"]

    @property
    def body(self) -> List[ASTNode]:
        return [
            c for c in self.children if c.kind not in ("param", "return_type")
        ]


@dataclass
class EngineSpec:
    """Top-level specification for a computational engine."""

    name: str
    root: ASTNode
    metadata: Dict[str, Any] = field(default_factory=dict)
    templates: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Constructors
# ---------------------------------------------------------------------------


def module(name: str = "main", children: Optional[List[ASTNode]] = None) -> ASTNode:
    return ASTNode(kind="module", name=name, children=children or [])


def function(
    name: str,
    params: Optional[List[ASTNode]] = None,
    return_type: Optional[str] = None,
    body: Optional[List[ASTNode]] = None,
) -> ASTNode:
    return ASTNode(
        kind="function",
        name=name,
        type_hint=return_type,
        children=(params or []) + (body or []),
    )


def param(name: str, type_hint: Optional[str] = None) -> ASTNode:
    return ASTNode(kind="param", name=name, type_hint=type_hint)


def binding(name: str, value: Any, type_hint: Optional[str] = None) -> ASTNode:
    node_value: Any
    if isinstance(value, ASTNode):
        node_value = None
        children = [value]
    else:
        node_value = value
        children = []
    return ASTNode(
        kind="binding",
        name=name,
        type_hint=type_hint,
        value=node_value,
        children=children,
    )


def return_node(value: Any) -> ASTNode:
    if isinstance(value, ASTNode):
        return ASTNode(kind="return", children=[value])
    return ASTNode(kind="return", value=value)


def call(name: str, args: Optional[List[Any]] = None) -> ASTNode:
    children: List[ASTNode] = []
    for arg in args or []:
        if isinstance(arg, ASTNode):
            children.append(arg)
        else:
            children.append(literal(arg))
    return ASTNode(kind="call", name=name, children=children)


def binary_op(left: Any, op: str, right: Any) -> ASTNode:
    children: List[ASTNode] = []
    for side in (left, right):
        if isinstance(side, ASTNode):
            children.append(side)
        else:
            children.append(literal(side))
    return ASTNode(kind="binary_op", name=op, children=children)


def literal(value: Any) -> ASTNode:
    return ASTNode(kind="literal", value=value)


def reference(name: str) -> ASTNode:
    return ASTNode(kind="reference", name=name)


def struct(name: str, fields: Optional[List[ASTNode]] = None) -> ASTNode:
    return ASTNode(kind="struct", name=name, children=fields or [])


def field(name: str, type_hint: Optional[str] = None) -> ASTNode:
    return ASTNode(kind="field", name=name, type_hint=type_hint)


def import_node(value: str) -> ASTNode:
    return ASTNode(kind="import", value=value)


def comment(text: str) -> ASTNode:
    return ASTNode(kind="comment", value=text)


def block(children: Optional[List[ASTNode]] = None) -> ASTNode:
    return ASTNode(kind="block", children=children or [])


def list_literal(items: List[Any]) -> ASTNode:
    children: List[ASTNode] = []
    for item in items:
        if isinstance(item, ASTNode):
            children.append(item)
        else:
            children.append(literal(item))
    return ASTNode(kind="list", children=children)


def dict_literal(pairs: Dict[Any, Any]) -> ASTNode:
    children: List[ASTNode] = []
    for key, val in pairs.items():
        key_node = key if isinstance(key, ASTNode) else literal(key)
        val_node = val if isinstance(val, ASTNode) else literal(val)
        children.append(ASTNode(kind="pair", children=[key_node, val_node]))
    return ASTNode(kind="dict", children=children)


def get_type(node: Optional[ASTNode]) -> Optional[str]:
    """Return the resolved type hint of *node*, if any."""
    return node.type_hint if node is not None else None


def set_type(node: ASTNode, type_hint: Optional[str]) -> ASTNode:
    """Return a shallow copy of *node* with its type hint set."""
    from copy import copy

    new = copy(node)
    new.type_hint = type_hint
    return new


# ---------------------------------------------------------------------------
# Python AST import
# ---------------------------------------------------------------------------


def spec_from_python(source: str, *, name: str = "generated") -> EngineSpec:
    """Parse a Python source string into an :class:`EngineSpec`.

    This is a best-effort bridge: it captures function signatures, assignments,
    returns, if statements, binary operations, calls, and literals. It is not a
    full Python front-end.
    """
    tree = ast.parse(source)
    children: List[ASTNode] = []
    for stmt in tree.body:
        node = _convert_stmt(stmt)
        if node is not None:
            children.append(node)
    return EngineSpec(name=name, root=module(name=name, children=children))


def _convert_stmt(stmt: ast.stmt) -> Optional[ASTNode]:
    if isinstance(stmt, ast.FunctionDef):
        params = [param(a.arg, _type_name(a.annotation)) for a in stmt.args.args]
        return_type: Optional[str] = _type_name(stmt.returns)
        body = [_convert_stmt(s) for s in stmt.body]
        body = [b for b in body if b is not None]
        if not body:
            body = [return_node(literal(None))]
        return function(stmt.name, params=params, return_type=return_type, body=body)
    if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1:
        target = stmt.targets[0]
        if isinstance(target, ast.Name):
            return binding(target.id, _convert_expr(stmt.value))
    if isinstance(stmt, ast.AnnAssign):
        type_hint = _type_name(stmt.annotation)
        if isinstance(stmt.target, ast.Name):
            return binding(
                stmt.target.id,
                _convert_expr(stmt.value) if stmt.value else literal(None),
                type_hint=type_hint,
            )
    if isinstance(stmt, ast.Return):
        return return_node(_convert_expr(stmt.value) if stmt.value else literal(None))
    if isinstance(stmt, ast.If):
        return _convert_if(stmt)
    if isinstance(stmt, ast.Expr):
        expr = _convert_expr(stmt.value)
        if expr.kind == "call":
            return expr
    return None


def _convert_if(stmt: ast.If) -> ASTNode:
    children = [_convert_expr(stmt.test)]
    then_body = ASTNode(kind="block", children=[b for b in (_convert_stmt(s) for s in stmt.body) if b is not None])
    children.append(then_body)
    if stmt.orelse:
        else_body = ASTNode(kind="block", children=[b for b in (_convert_stmt(s) for s in stmt.orelse) if b is not None])
        children.append(else_body)
    return ASTNode(kind="if", children=children)


def _type_name(node: Optional[ast.expr]) -> Optional[str]:
    """Convert an annotation AST node to a type string."""
    if node is None:
        return None
    if hasattr(ast, "unparse"):
        return ast.unparse(node)
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Constant):
        return str(node.value)
    return ast.dump(node)


def _convert_expr(expr: Optional[ast.expr]) -> ASTNode:
    if expr is None:
        return literal(None)
    if isinstance(expr, ast.Constant):
        return literal(expr.value)
    if isinstance(expr, ast.Name):
        return reference(expr.id)
    if isinstance(expr, ast.BinOp):
        op = _bin_op_name(expr.op)
        return binary_op(_convert_expr(expr.left), op, _convert_expr(expr.right))
    if isinstance(expr, ast.Call):
        name = _call_name(expr.func)
        args = [_convert_expr(a) for a in expr.args]
        return call(name, args)
    if isinstance(expr, ast.List):
        return list_literal([_convert_expr(e) for e in expr.elts])
    if isinstance(expr, ast.Dict):
        pairs = {}
        for k, v in zip(expr.keys, expr.values):
            pairs[_expr_key(k)] = _convert_expr(v)
        return dict_literal(pairs)
    return literal(None)


def _bin_op_name(op: ast.operator) -> str:
    mapping = {
        ast.Add: "+",
        ast.Sub: "-",
        ast.Mult: "*",
        ast.Div: "/",
        ast.Mod: "%",
        ast.Pow: "**",
        ast.LShift: "<<",
        ast.RShift: ">>",
        ast.BitOr: "|",
        ast.BitXor: "^",
        ast.BitAnd: "&",
    }
    return mapping.get(type(op), "+")


def _call_name(func: ast.expr) -> str:
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
        return f"{func.value.id}.{func.attr}"
    return "call"


def _expr_key(expr: Optional[ast.expr]) -> Any:
    if expr is None:
        return None
    if isinstance(expr, ast.Constant):
        return expr.value
    return ast.dump(expr)
