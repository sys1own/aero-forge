"""Base class for polyglot source emitters."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, List, Optional

from aero_forge.builder.spec import ASTNode, EngineSpec


class EmitterError(Exception):
    """Raised when an emitter cannot render an engine spec."""


class BaseEmitter(ABC):
    """Render an :class:`EngineSpec` into source code for a target language.

    Subclasses implement language-specific syntax by overriding the
    ``_emit_*`` hooks. The public entry point is :meth:`emit`.
    """

    target_language: str = ""
    indent: str = "    "

    def __init__(self, indent: Optional[str] = None) -> None:
        if indent is not None:
            self.indent = indent
        self._lines: List[str] = []

    def emit(self, spec: EngineSpec) -> str:
        """Return the fully rendered source for *spec*."""
        self._lines = []
        self._emit_preamble(spec)
        self._emit(spec.root, 0)
        self._emit_postamble(spec)
        return "\n".join(self._lines) + "\n"

    # ------------------------------------------------------------------
    # Public hooks for pre/post-amble
    # ------------------------------------------------------------------

    def _emit_preamble(self, spec: EngineSpec) -> None:
        """Hook for file-level headers (imports, pragmas, etc.)."""

    def _emit_postamble(self, spec: EngineSpec) -> None:
        """Hook for file-level footers."""

    # ------------------------------------------------------------------
    # Core dispatch
    # ------------------------------------------------------------------

    def _emit(self, node: ASTNode, indent_level: int) -> None:
        handler = getattr(self, f"_emit_{node.kind}", None)
        if handler is None:
            raise EmitterError(
                f"{self.__class__.__name__} has no handler for node kind {node.kind!r}"
            )
        handler(node, indent_level)

    def _emit_children(self, nodes: List[ASTNode], indent_level: int) -> None:
        for child in nodes:
            self._emit(child, indent_level)

    def _emit_block(self, node: ASTNode, indent_level: int) -> None:
        self._emit_children(node.children, indent_level)

    def _write(self, line: str, indent_level: int = 0) -> None:
        self._lines.append(self.indent * indent_level + line)

    def _expr(self, node: ASTNode) -> str:
        """Render an expression node as a single string."""
        return self._emit_expression_to_string(node)

    # ------------------------------------------------------------------
    # Abstract language primitives
    # ------------------------------------------------------------------

    @abstractmethod
    def _emit_module(self, node: ASTNode, indent_level: int) -> None:
        """Render a module / translation unit."""

    @abstractmethod
    def _emit_function(self, node: ASTNode, indent_level: int) -> None:
        """Render a function declaration."""

    @abstractmethod
    def _emit_struct(self, node: ASTNode, indent_level: int) -> None:
        """Render a struct / class / record."""

    @abstractmethod
    def _emit_binding(self, node: ASTNode, indent_level: int) -> None:
        """Render a variable binding / assignment."""

    @abstractmethod
    def _emit_return(self, node: ASTNode, indent_level: int) -> None:
        """Render a return statement."""

    @abstractmethod
    def _emit_import(self, node: ASTNode, indent_level: int) -> None:
        """Render an import / use / include."""

    @abstractmethod
    def _emit_comment(self, node: ASTNode, indent_level: int) -> None:
        """Render a comment line."""

    # ------------------------------------------------------------------
    # Expression helpers (common across emitters)
    # ------------------------------------------------------------------

    def _emit_expression_to_string(self, node: ASTNode) -> str:
        if node.kind == "literal":
            return self._literal(node.value)
        if node.kind == "reference":
            return node.name or "_"
        if node.kind == "call":
            args = ", ".join(self._expr(c) for c in node.children)
            return f"{node.name}({args})"
        if node.kind == "binary_op":
            left, right = node.children
            return f"({self._expr(left)} {node.name} {self._expr(right)})"
        if node.kind == "list":
            return self._list_literal(node.children)
        if node.kind == "dict":
            return self._dict_literal(node.children)
        if node.kind == "param":
            return node.name or "_"
        raise EmitterError(
            f"Unsupported expression kind {node.kind!r} in {self.__class__.__name__}"
        )

    def _literal(self, value: Any) -> str:
        if value is None:
            return self._none_literal()
        if isinstance(value, bool):
            return self._bool_literal(value)
        if isinstance(value, (int, float)):
            return str(value)
        if isinstance(value, str):
            return self._string_literal(value)
        if isinstance(value, (list, tuple)):
            return self._list_literal([literal(v) for v in value])
        if isinstance(value, dict):
            return self._dict_literal(
                [ASTNode(kind="pair", children=[literal(k), literal(v)]) for k, v in value.items()]
            )
        return str(value)

    def _string_literal(self, value: str) -> str:
        return f'"{value}"'

    @abstractmethod
    def _bool_literal(self, value: bool) -> str:
        """Render a boolean literal."""

    @abstractmethod
    def _none_literal(self) -> str:
        """Render a None / null / unit literal."""

    @abstractmethod
    def _list_literal(self, children: List[ASTNode]) -> str:
        """Render a list / vector literal."""

    @abstractmethod
    def _dict_literal(self, pairs: List[ASTNode]) -> str:
        """Render a dict / map literal."""

    @abstractmethod
    def _map_type(self, type_hint: Optional[str]) -> str:
        """Map an abstract type hint to the target language type."""


# Re-export `literal` for emitter internal use.
from aero_forge.builder.spec import literal  # noqa: E402,F401
