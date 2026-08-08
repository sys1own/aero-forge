"""SMT-backed type inference for structured synthesis skeletons."""

from __future__ import annotations

import ast
import re
from typing import Any, Dict, Optional

from aero_forge.precision_shield.smt_solver import SMTASTEngine

__all__ = ["SMTASTEngine", "SkeletonTypeInjector"]


class SkeletonTypeInjector:
    """Resolve typed holes in a source skeleton using the SMTASTEngine."""

    _PLACEHOLDER_RE = re.compile(r"__AERO_TYPE_(?P<name>\w+)__")

    @classmethod
    def infer_type_env(
        cls,
        source: str,
        function_name: Optional[str] = None,
    ) -> Dict[str, str]:
        """Return a mapping from variable name to a native type for *source*."""
        if not source or not source.strip():
            return {}
        try:
            return SMTASTEngine().infer_native_types(source, function_name)
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
