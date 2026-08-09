"""Convert a UAST JSON logic sketch into idiomatic Python source.

The UAST dialect follows Python ``ast`` node names and fields so the engine
remains the authority for emission while the LLM only supplies intent. Before
unparsing, an optional ``AttributeResolver`` (provided by the SMT engine)
rewrites known incorrect attribute names (e.g. ``conj`` -> ``conjugate``) based on
the inferred receiver type.
"""

from __future__ import annotations

import ast
from typing import Any, Callable, Optional


def uast_to_python_source(
    uast: Any,
    *,
    attribute_resolver: Optional[Callable[[Any], Any]] = None,
) -> str:
    """Return Python source code for a UAST dict/list and validate it."""
    if attribute_resolver is not None:
        uast = attribute_resolver(uast)
    py_ast = _uast_to_ast(uast)
    if not isinstance(py_ast, ast.AST):
        raise ValueError(f"UAST did not produce an ast.AST: {type(py_ast)}")
    ast.fix_missing_locations(py_ast)
    source = ast.unparse(py_ast)
    # Sanity check: emitted source must itself be syntactically valid.
    ast.parse(source)
    return source


_UAST_TYPE_KEYS = ("type", "_type", "kind")

# AST fields that are always lists of child nodes and should default to [].
_LIST_FIELDS = {
    "body",
    "orelse",
    "targets",
    "args",
    "keywords",
    "generators",
    "elts",
    "keys",
    "values",
    "comparators",
    "decorator_list",
    "defaults",
    "kw_defaults",
    "posonlyargs",
    "kwonlyargs",
    "handlers",
    "finalbody",
    "type_ignores",
    "ifs",
}


def _uast_kind(node: Any) -> Optional[str]:
    if isinstance(node, dict):
        for key in _UAST_TYPE_KEYS:
            if key in node:
                return node[key]
    return None


def _ctx(name: Optional[str]) -> ast.expr_context:
    if name in ("Store", "store"):
        return ast.Store()
    if name in ("Del", "del"):
        return ast.Del()
    return ast.Load()


def _uast_to_ast(uast: Any) -> Any:
    """Recursively convert a UAST JSON value into an ``ast.AST`` tree."""
    if isinstance(uast, list):
        return [_uast_to_ast(item) for item in uast]

    if not isinstance(uast, dict):
        # Primitive value (string, int, float, None) used for fields such as
        # ``Name.id``, ``Constant.value``, ``arg.arg`` or ``Attribute.attr``.
        return uast

    kind = _uast_kind(uast)
    if kind is None:
        return uast

    cls = getattr(ast, kind, None)
    if cls is None:
        raise ValueError(f"Unknown UAST node type: {kind!r}")

    kwargs: dict = {}
    for field in cls._fields:
        raw = uast.get(field)

        if field == "ctx":
            kwargs[field] = _ctx(uast.get("ctx"))
            continue

        if field == "op":
            kwargs[field] = _uast_to_ast(raw) if raw is not None else None
            continue

        if field == "ops":
            kwargs[field] = [_uast_to_ast(x) for x in (raw or [])]
            continue

        if field == "is_async":
            kwargs[field] = bool(raw)
            continue

        if isinstance(raw, list):
            kwargs[field] = [_uast_to_ast(item) for item in raw]
        elif isinstance(raw, dict):
            kwargs[field] = _uast_to_ast(raw)
        elif raw is None:
            # Optional list fields (e.g. Module.type_ignores) default to empty
            # lists; optional single-node fields default to None.
            kwargs[field] = [] if field in _LIST_FIELDS else raw
        else:
            kwargs[field] = raw

    return cls(**kwargs)


__all__ = ["uast_to_python_source", "_uast_to_ast"]
