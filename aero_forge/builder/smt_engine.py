"""SMT-backed type inference for structured synthesis skeletons."""

from __future__ import annotations

import ast
import copy
import re
from typing import Any, Dict, Optional

from aero_forge.precision_shield.smt_solver import SMTASTEngine

__all__ = ["SMTASTEngine", "SkeletonTypeInjector", "AttributeResolver", "SMTSaturationError"]


class SMTSaturationError(RuntimeError):
    """Raised when the SMT solver returns UNSAT or an empty model for a function body."""

    pass


class SkeletonTypeInjector:
    """Resolve typed holes in a source skeleton using the SMTASTEngine."""

    _PLACEHOLDER_RE = re.compile(r"__AERO_TYPE_(?P<name>\w+)__")

    @classmethod
    def infer_type_env(
        cls,
        source: str,
        function_name: Optional[str] = None,
        target_language: str = "rust",
    ) -> Dict[str, str]:
        """Return a mapping from variable name to a native type for *source*."""
        if not source or not source.strip():
            return {}
        try:
            return SMTASTEngine().infer_native_types(
                source, function_name, target_language=target_language
            )
        except Exception:
            return {}

    @classmethod
    def inject_types(
        cls,
        skeleton: str,
        source: Optional[str] = None,
        function_name: Optional[str] = None,
        type_env: Optional[Dict[str, str]] = None,
    ) -> str:
        """Replace ``__AERO_TYPE_<var>__`` placeholders with inferred native types."""
        env = type_env or {}
        if source is not None and not env:
            env = cls.infer_type_env(source, function_name)

        def _replace(match: re.Match) -> str:
            name = match.group("name")
            return env.get(name, "auto")

        return cls._PLACEHOLDER_RE.sub(_replace, skeleton)

    @classmethod
    def annotate_ast_nodes(
        cls,
        source: str,
        function_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Return a sparse AST annotation mapping for the top-level variables."""
        env = cls.infer_type_env(source, function_name)
        tree = ast.parse(source)
        annotations: Dict[str, Any] = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Param)):
                if node.id in env:
                    annotations[node.id] = {
                        "lineno": getattr(node, "lineno", None),
                        "type": env[node.id],
                    }
        return annotations

    @classmethod
    def infer_type_env_for_symbol(
        cls,
        source: str,
        symbol: str,
        target_language: str = "rust",
    ) -> Dict[str, str]:
        """Infer native types scoped to a single function ``symbol``.

        The SMT engine exposes the return type under ``__return__``; this helper
        renames it to ``return`` so skeleton builders can look up return types
        with a single key.
        """
        env = cls.infer_type_env(
            source, function_name=symbol, target_language=target_language
        )
        if "__return__" in env:
            env["return"] = env.pop("__return__")
        return env

    @classmethod
    def saturate(
        cls,
        source: str,
        function_name: Optional[str] = None,
        target_language: str = "rust",
    ) -> Dict[str, str]:
        """Infer and verify non-null SMT type assignments for every variable.

        Parses *source*, collects every variable that is stored or used as a
        parameter, and ensures the Z3-backed SMT solver returns a concrete native
        type for each one. If the solver is UNSAT or leaves any variable without
        a type assignment, ``SMTSaturationError`` is raised so the Proactive
        Synthesis Healing core can rewrite the intent before emission.
        """
        if not source or not source.strip():
            raise SMTSaturationError("Cannot saturate an empty logic sketch.")

        try:
            env = SMTASTEngine().infer_native_types(
                source, function_name, target_language=target_language
            )
        except Exception as exc:
            raise SMTSaturationError(
                f"SMT solver failed to infer types for {function_name or '<module>'}: {exc}"
            ) from exc

        tree = ast.parse(source)
        funcs = [
            node
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        ]
        if function_name:
            funcs = [f for f in funcs if f.name == function_name]
        target = funcs[0] if funcs else tree

        required: set = set()
        for node in ast.walk(target):
            if isinstance(node, ast.arg):
                required.add(node.arg)
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
                required.add(node.id)

        missing = sorted(n for n in required if n not in env)
        if missing:
            raise SMTSaturationError(
                f"SMT model is missing non-null type assignments for variables: {missing}"
            )

        return env


class AttributeResolver:
    """Rewrite incorrect attribute names in a UAST using SMT-inferred types.

    The resolver is deterministic: it maps known wrong attribute names (e.g.
    ``conj``) to the correct Python spelling (``conjugate``) when the receiver
    is a complex value. If a *type_env* is supplied it is used to decide whether
    the receiver is complex; otherwise a conservative set of syntactic patterns
    (``cmath.conj(x)`` -> ``x.conjugate()``, ``z.conj()`` -> ``z.conjugate()``)
    is applied.
    """

    # Attributes that are aliases in numeric libraries but not valid Python method
    # names on the built-in ``complex`` type.
    _NUMERIC_ATTR_ALIASES = {
        "conj": "conjugate",
    }

    def __init__(self, type_env: Optional[Dict[str, str]] = None) -> None:
        self.type_env = type_env or {}

    @classmethod
    def from_source(
        cls,
        source: str,
        function_name: Optional[str] = None,
    ) -> "AttributeResolver":
        """Build a resolver whose type environment comes from SMT inference."""
        env: Dict[str, str] = {}
        try:
            env = SMTASTEngine().infer_native_types(source, function_name)
        except Exception:
            env = {}
        return cls(type_env=env)

    def resolve(self, uast: Any) -> Any:
        """Return a new UAST with numeric attribute aliases corrected."""
        return self._resolve(copy.deepcopy(uast))

    def _resolve(self, node: Any) -> Any:
        if isinstance(node, list):
            return [self._resolve(item) for item in node]
        if not isinstance(node, dict):
            return node

        kind = node.get("type") or node.get("_type") or node.get("kind")

        # Call(func=Attribute(value=Name('cmath'), attr='conj'), args=[x])
        # should become x.conjugate() for the built-in complex type.
        if kind in ("call", "Call") and "func" in node:
            func = node["func"]
            attr = self._attribute_name(func)
            if attr in self._NUMERIC_ATTR_ALIASES and self._is_attribute(func, attr):
                value = func.get("value")
                if self._is_name(value, "cmath") and node.get("args"):
                    receiver = self._resolve(node["args"][0])
                    return self._make_call(
                        self._make_attribute(receiver, "conjugate"),
                        [],
                    )
                if self._should_rewrite_attr(value):
                    node["func"] = self._make_attribute(
                        self._resolve(value), "conjugate"
                    )
                    return node

        # Attribute(value=z, attr='conj') -> z.conjugate
        if kind in ("attribute", "Attribute") and "attr" in node:
            if node["attr"] in self._NUMERIC_ATTR_ALIASES:
                if self._should_rewrite_attr(node.get("value")):
                    node["attr"] = self._NUMERIC_ATTR_ALIASES[node["attr"]]

        for key, value in list(node.items()):
            if isinstance(value, (dict, list)):
                node[key] = self._resolve(value)
        return node

    def _is_name(self, node: Any, name: str) -> bool:
        if not isinstance(node, dict):
            return False
        return (
            node.get("type") in ("name", "Name", "identifier", "var", "reference")
            and node.get("id") == name
        )

    def _is_attribute(self, node: Any, attr: str) -> bool:
        if not isinstance(node, dict):
            return False
        return (
            node.get("type") in ("attribute", "Attribute")
            and node.get("attr") == attr
        )

    def _attribute_name(self, node: Any) -> str:
        if isinstance(node, dict):
            return node.get("attr", "")
        return ""

    def _should_rewrite_attr(self, node: Any) -> bool:
        # If the SMT engine has identified the receiver as complex, rewrite.
        if isinstance(node, dict):
            name = node.get("id") or node.get("name")
            if name:
                t = self.type_env.get(name, "")
                if t in ("complex", "c64", "c128"):
                    return True
                if t and t not in ("", "auto", "unknown"):
                    # Receiver has a concrete non-complex type; do not guess.
                    return False
            # Numeric libraries expose ``conj`` as an alias for ``conjugate``;
            # built-in complex objects only recognise ``conjugate``.
            if self._is_name(node, "cmath") or self._is_name(node, "np"):
                return True
            if node.get("type") in ("name", "Name", "identifier", "var", "reference", "attribute", "Attribute"):
                return True
        return False

    @staticmethod
    def _make_attribute(value: Any, attr: str) -> Dict[str, Any]:
        return {"type": "Attribute", "value": value, "attr": attr}

    @staticmethod
    def _make_call(func: Any, args: Any) -> Dict[str, Any]:
        return {"type": "Call", "func": func, "args": args}
