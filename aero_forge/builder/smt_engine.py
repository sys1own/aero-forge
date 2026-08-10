"""SMT-backed type inference for structured synthesis skeletons."""

from __future__ import annotations

import ast
import copy
import re
from typing import Any, Dict, Optional, Set

from aero_forge.builder.language_router import _accel_log
from aero_forge.precision_shield.smt_solver import SMTASTEngine

__all__ = ["SMTASTEngine", "SkeletonTypeInjector", "AttributeResolver", "SMTSaturationError", "BuiltinAttributeGate"]


class SMTSaturationError(RuntimeError):
    """Raised when the SMT solver returns UNSAT or an empty model for a function body."""

    pass


class BuiltinAttributeGate:
    """Static gate that validates attribute access on built-in Python types.

    The gate mirrors the SMT solver's role for attribute holes: any attribute
    accessed on ``str``, ``int``, ``list``, or ``dict`` that is not part of the
    Python 3.10+ standard runtime is rejected as if the SMT query were UNSAT.
    """

    _BUILTIN_ATTRS: Dict[str, Set[str]] = {
        "str": {
            "capitalize",
            "casefold",
            "center",
            "count",
            "encode",
            "endswith",
            "expandtabs",
            "find",
            "format",
            "format_map",
            "index",
            "isalnum",
            "isalpha",
            "isascii",
            "isdecimal",
            "isdigit",
            "isidentifier",
            "islower",
            "isnumeric",
            "isprintable",
            "isspace",
            "istitle",
            "isupper",
            "join",
            "ljust",
            "lower",
            "lstrip",
            "maketrans",
            "partition",
            "removeprefix",
            "removesuffix",
            "replace",
            "rfind",
            "rindex",
            "rjust",
            "rpartition",
            "rsplit",
            "rstrip",
            "split",
            "splitlines",
            "startswith",
            "strip",
            "swapcase",
            "title",
            "translate",
            "upper",
            "zfill",
            "__add__",
            "__contains__",
            "__eq__",
            "__ge__",
            "__getitem__",
            "__gt__",
            "__hash__",
            "__iter__",
            "__le__",
            "__len__",
            "__lt__",
            "__ne__",
            "__repr__",
            "__str__",
        },
        "int": {
            "bit_length",
            "bit_count",
            "to_bytes",
            "from_bytes",
            "as_integer_ratio",
            "is_integer",
            "__abs__",
            "__add__",
            "__and__",
            "__bool__",
            "__ceil__",
            "__divmod__",
            "__eq__",
            "__float__",
            "__floor__",
            "__floordiv__",
            "__ge__",
            "__gt__",
            "__index__",
            "__int__",
            "__invert__",
            "__le__",
            "__lshift__",
            "__lt__",
            "__mod__",
            "__mul__",
            "__ne__",
            "__neg__",
            "__or__",
            "__pos__",
            "__pow__",
            "__rshift__",
            "__sub__",
            "__truediv__",
            "__xor__",
        },
        "list": {
            "append",
            "clear",
            "copy",
            "count",
            "extend",
            "index",
            "insert",
            "pop",
            "remove",
            "reverse",
            "sort",
            "__add__",
            "__contains__",
            "__delitem__",
            "__eq__",
            "__ge__",
            "__getitem__",
            "__gt__",
            "__iadd__",
            "__imul__",
            "__iter__",
            "__le__",
            "__len__",
            "__lt__",
            "__mul__",
            "__ne__",
            "__repr__",
            "__reversed__",
            "__setitem__",
            "__str__",
        },
        "dict": {
            "clear",
            "copy",
            "fromkeys",
            "get",
            "items",
            "keys",
            "pop",
            "popitem",
            "setdefault",
            "update",
            "values",
            "__contains__",
            "__delitem__",
            "__eq__",
            "__ge__",
            "__getitem__",
            "__gt__",
            "__iter__",
            "__le__",
            "__len__",
            "__lt__",
            "__ne__",
            "__repr__",
            "__setitem__",
            "__str__",
        },
    }

    _BUILTIN_NAMES = {
        "__name__": "str",
    }

    @classmethod
    def _receiver_type(
        cls,
        value: ast.expr,
        type_env: Optional[Dict[str, str]] = None,
    ) -> Optional[str]:
        """Best-effort type of an attribute access receiver."""
        if isinstance(value, ast.Constant):
            if isinstance(value.value, str):
                return "str"
            if isinstance(value.value, int):
                return "int"
            if isinstance(value.value, list):
                return "list"
            if isinstance(value.value, dict):
                return "dict"
            return None
        if isinstance(value, ast.Name):
            name = value.id
            if name in cls._BUILTIN_NAMES:
                return cls._BUILTIN_NAMES[name]
            if type_env and name in type_env:
                t = type_env[name]
                mapping = {
                    "string": "str",
                    "i64": "int",
                    "vec_i64": "list",
                    "map": "dict",
                    "dict": "dict",
                    "list": "list",
                    "set": "set",
                }
                return mapping.get(t, None)
        if isinstance(value, ast.List):
            return "list"
        if isinstance(value, ast.Dict):
            return "dict"
        return None

    @classmethod
    def verify(
        cls,
        source: str,
        type_env: Optional[Dict[str, str]] = None,
        artifact_path: Optional[str] = None,
    ) -> None:
        """Raise SMTSaturationError if any built-in attribute is invalid."""
        if not source or not source.strip():
            return
        try:
            tree = ast.parse(source)
        except SyntaxError:
            return
        label = artifact_path or "<module>"
        for node in ast.walk(tree):
            if not isinstance(node, ast.Attribute):
                continue
            recv_type = cls._receiver_type(node.value, type_env=type_env)
            if not recv_type:
                continue
            allowed = cls._BUILTIN_ATTRS.get(recv_type, set())
            if node.attr in allowed:
                continue
            raise SMTSaturationError(
                f"Attribute Verification Failed for {label}: "
                f"'{recv_type}' object has no attribute '{node.attr}'"
            )


class SkeletonTypeInjector:
    """Resolve typed holes in a source skeleton using the SMTASTEngine."""

    _PLACEHOLDER_RE = re.compile(r"__AERO_TYPE_(?P<name>\w+)__")

    @staticmethod
    def _annotation_to_native_type(ann: Optional[ast.expr]) -> Optional[str]:
        if ann is None:
            return None
        text = ast.unparse(ann).strip().lower()
        mapping = {
            "int": "i64",
            "float": "f64",
            "bool": "bool",
            "str": "string",
            "string": "string",
            "list": "vec_i64",
            "list[int]": "vec_i64",
            "list[float]": "vec_f64",
            "dict": "map",
            "set": "set",
        }
        return mapping.get(text)

    @staticmethod
    def _value_to_native_type(value: Optional[ast.expr]) -> Optional[str]:
        if value is None:
            return None
        if isinstance(value, ast.Dict):
            return "map"
        if isinstance(value, ast.List):
            return "vec_i64"
        if isinstance(value, ast.Set):
            return "set"
        if isinstance(value, ast.Tuple):
            return "tuple"
        if isinstance(value, ast.Constant):
            if value.value is None:
                return None
            if isinstance(value.value, bool):
                return "bool"
            if isinstance(value.value, int):
                return "i64"
            if isinstance(value.value, float):
                return "f64"
            if isinstance(value.value, str):
                return "string"
        return None

    @classmethod
    def infer_type_env_for_data_constant(
        cls,
        source: str,
        symbol: str,
        target_language: str = "python",
    ) -> Dict[str, str]:
        """Return a native type for a top-level data constant named *symbol*."""
        try:
            tree = ast.parse(source)
        except SyntaxError:
            return {}
        for node in tree.body:
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == symbol:
                        native = cls._annotation_to_native_type(getattr(node, "annotation", None)) or cls._value_to_native_type(node.value)
                        if native:
                            return {symbol: native}
                        return {}
            if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.target.id == symbol:
                native = cls._annotation_to_native_type(node.annotation) or cls._value_to_native_type(node.value)
                if native:
                    return {symbol: native}
                return {}
        return {}

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
    def _verify_and_heal_attributes(
        cls,
        source: str,
        function_name: Optional[str] = None,
        artifact_path: Optional[str] = None,
    ) -> str:
        """Run the built-in attribute gate and apply deterministic boilerplate healing.

        If the gate rejects an attribute (e.g. ``__name__.eq``) the
        EntrypointBoilerplateNormalizer attempts to rewrite the malformed
        entrypoint idiom before the SMT solver is invoked.
        """
        label = artifact_path or function_name or "<module>"
        try:
            BuiltinAttributeGate.verify(source, artifact_path=label)
        except SMTSaturationError as exc:
            try:
                # Lazy import avoids a circular dependency with emitters/base.py.
                from aero_forge.builder.emitters.base import EntrypointBoilerplateNormalizer

                healed = EntrypointBoilerplateNormalizer.normalize(source)
            except Exception:
                raise exc
            if healed == source:
                raise exc
            try:
                BuiltinAttributeGate.verify(healed, artifact_path=label)
            except SMTSaturationError:
                raise exc
            _accel_log("info", f"Attribute Verification Healed for {label}")
            return healed
        _accel_log("info", f"Attribute Verification Passed for {label}")
        return source

    @classmethod
    def saturate(
        cls,
        source: str,
        function_name: Optional[str] = None,
        target_language: str = "rust",
        artifact_path: Optional[str] = None,
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

        source = cls._verify_and_heal_attributes(
            source,
            function_name=function_name,
            artifact_path=artifact_path,
        )

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

        # Data constants are not functions; infer their types from annotations/value shape.
        if function_name and not funcs:
            data_env = cls.infer_type_env_for_data_constant(
                source, function_name, target_language=target_language
            )
            if data_env:
                return data_env

        target = funcs[0] if funcs else tree

        required: set = set()
        for node in ast.walk(target):
            if isinstance(node, ast.arg):
                required.add(node.arg)
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
                required.add(node.id)

        missing = sorted(n for n in required if n not in env)
        if missing:
            recovered: Dict[str, str] = {}
            for name in missing:
                const_env = cls.infer_type_env_for_data_constant(
                    source, name, target_language=target_language
                )
                if const_env:
                    recovered.update(const_env)
            still_missing = [n for n in missing if n not in recovered and n not in env]
            if still_missing:
                raise SMTSaturationError(
                    f"SMT model is missing non-null type assignments for variables: {still_missing}"
                )
            env.update(recovered)

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

        # Heal malformed entry-point boilerplate such as ``if __name__.eq == '__main__':``.
        if kind in ("compare", "Compare"):
            left = node.get("left") or node.get("operands", [None])[0]
            ops = node.get("ops") or node.get("op")
            comparators = node.get("comparators") or node.get("operands", [None, None])[1:]
            if (
                left
                and isinstance(ops, list)
                and len(ops) == 1
                and self._is_eq_op(ops[0])
                and comparators
                and self._is_str_literal(comparators[0], "__main__")
                and self._is_name_attr(left, "__name__", {"eq", "equals", "equal"})
            ):
                node["left"] = {"type": "Name", "id": "__name__", "ctx": {"type": "Load"}}

        for key, value in list(node.items()):
            if isinstance(value, (dict, list)):
                node[key] = self._resolve(value)
        return node

    def _is_eq_op(self, node: Any) -> bool:
        if not isinstance(node, dict):
            return False
        return node.get("type") in ("Eq", "eq", "==")

    def _is_str_literal(self, node: Any, value: str) -> bool:
        if not isinstance(node, dict):
            return False
        return (
            node.get("type") in ("constant", "Constant", "literal", "Literal")
            and node.get("value") == value
        )

    def _is_name_attr(self, node: Any, name: str, attrs: Set[str]) -> bool:
        if not isinstance(node, dict):
            return False
        return (
            node.get("type") in ("attribute", "Attribute")
            and self._is_name(node.get("value"), name)
            and node.get("attr") in attrs
        )

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
