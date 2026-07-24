"""AST-based classification and routing engine for aero-forge.

Determines whether Python functions should be transpiled through the
zero-allocation UAST->HIN pipeline or kept in native Python execution.
"""

from __future__ import annotations

import ast
from typing import Any, Dict, List, Optional, Set

HIN_COMPUTE = "HIN_COMPUTE"
GENERAL_PURPOSE = "GENERAL_PURPOSE"

_SCALAR_TYPES = {"int", "float", "bool", "complex", "str", "bytes", "None"}

_ALLOWED_BUILTINS = {
    "abs",
    "round",
    "pow",
    "min",
    "max",
    "len",
    "range",
    "int",
    "float",
    "bool",
    "str",
    "isinstance",
    "sorted",
    "sum",
    "enumerate",
    "zip",
    "reversed",
    "all",
    "any",
    "complex",
}

_ALLOWED_MATH_MODULES = {"math", "numpy", "np"}
_TYPE_MODULES = {"typing", "collections.abc"}

# Map typing aliases to lowercase container names used by the HIN pipeline.
_TYPING_ALIASES = {"List": "list", "Tuple": "tuple", "Dict": "dict"}

_IO_BUILTINS = {
    "open",
    "print",
    "input",
    "eval",
    "exec",
    "compile",
    "getattr",
    "setattr",
    "hasattr",
    "super",
    "type",
    "vars",
    "locals",
    "globals",
}

_IO_METHODS = {
    "read",
    "write",
    "close",
    "flush",
    "open",
    "send",
    "recv",
    "connect",
    "bind",
    "listen",
    "accept",
    "format",
    "join",
    "split",
    "replace",
    "strip",
    "lower",
    "upper",
}

_LIST_MUTATORS = {"append", "extend", "pop"}
_COLLECTION_ACCESSORS = {"get", "items", "keys", "values"}


def _annotation_name(node: ast.AST) -> Optional[str]:
    """Return a simple name string for a Python type annotation."""
    if isinstance(node, ast.Name):
        return _TYPING_ALIASES.get(node.id, node.id)
    if isinstance(node, ast.Constant):
        if node.value is None:
            return "None"
        if isinstance(node.value, bool):
            return "bool"
        if isinstance(node.value, str):
            return node.value
    if isinstance(node, getattr(ast, "Str", ())):
        return node.s  # type: ignore[attr-defined]
    if isinstance(node, ast.Tuple):
        parts = [_annotation_name(elt) for elt in node.elts]
        if all(parts):
            return f"tuple[{', '.join(parts)}]"
        return None
    if isinstance(node, ast.Attribute):
        base = _base_name(node)
        if base in ("np", "numpy") and node.attr == "ndarray":
            # The HIN pipeline lowers NumPy 1-D arrays to Vec<f64>.
            return "list[float]"
        if base in _TYPE_MODULES and node.attr in _TYPING_ALIASES:
            return _TYPING_ALIASES[node.attr]
    if isinstance(node, ast.Subscript):
        base = _annotation_name(node.value)
        if base is None:
            return None
        if base in _TYPING_ALIASES:
            base = _TYPING_ALIASES[base]
        if isinstance(node.slice, ast.Name):
            return f"{base}[{node.slice.id}]"
        if isinstance(node.slice, ast.Tuple):
            parts = [_annotation_name(elt) for elt in node.slice.elts]
            return f"{base}[{', '.join(p for p in parts if p)}]"
        slice_name = _annotation_name(node.slice)
        if slice_name:
            return f"{base}[{slice_name}]"
    return None


def _split_type_args(s: str) -> List[str]:
    """Split a type argument string at top-level commas (ignoring nested brackets)."""
    parts: List[str] = []
    depth = 0
    current: List[str] = []
    for ch in s:
        if ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
        if ch == "," and depth == 0:
            parts.append("".join(current).strip())
            current = []
            continue
        current.append(ch)
    if current or not s:
        parts.append("".join(current).strip())
    return parts


def _base_name(node: ast.expr) -> Optional[str]:
    """Return the leftmost name in an attribute chain, e.g. ``math`` in ``math.sin``."""
    while isinstance(node, ast.Attribute):
        node = node.value
    return node.id if isinstance(node, ast.Name) else None


def _is_homogeneous_numeric_list(node: ast.List) -> bool:
    """Return True if the list literal is not obviously heterogeneous.

    Literals mixing string and numeric constants are rejected; lists composed
    of variables, subscripts, or homogeneous numeric constants are accepted.
    """
    if not node.elts:
        return True
    constant_types: Set[type] = set()
    for elt in node.elts:
        if isinstance(elt, ast.Constant) and isinstance(
            elt.value, (int, float, bool, complex)
        ):
            constant_types.add(type(elt.value))
        elif isinstance(elt, (ast.Name, ast.Subscript, ast.BinOp, ast.UnaryOp, ast.Call)):
            # Variable/expression elements are assumed homogeneous for the target type.
            pass
        else:
            return False
    return len(constant_types) <= 1


class _FunctionClassifier(ast.NodeVisitor):
    """Collect whether a single function is suitable for HIN extraction."""

    def __init__(self, function: ast.FunctionDef, local_functions: Set[str]) -> None:
        self.function = function
        self.local_functions = local_functions
        self.self_name = function.args.args[0].arg if function.args.args else None
        self.reasons: List[str] = []
        self.callees: Set[str] = set()
        self.is_hin = True

    def _reject(self, reason: str) -> None:
        self.is_hin = False
        if reason not in self.reasons:
            self.reasons.append(reason)

    def _is_strictly_typed(self) -> None:
        """Validate any present primitive type annotations.

        Missing annotations are allowed for pure numeric functions because the
        UAST/HIN pipeline can infer scalar types. Present annotations must be
        scalar or uniform annotated collections.
        """
        for arg in self.function.args.args:
            if arg.annotation is None:
                continue
            name = _annotation_name(arg.annotation)
            if name is None:
                self._reject(
                    f"Function '{self.function.name}' has an unsupported annotation for parameter '{arg.arg}'"
                )
                continue
            base = name.split("[", 1)[0]
            if base in _TYPING_ALIASES:
                base = _TYPING_ALIASES[base]
            if base not in _SCALAR_TYPES and base not in {"list", "tuple", "dict"}:
                self._reject(
                    f"Function '{self.function.name}' parameter '{arg.arg}' uses non-primitive type '{name}'"
                )
            self._check_iterable_annotation(arg.annotation, f"parameter '{arg.arg}'")
        if self.function.returns is not None:
            ret_name = _annotation_name(self.function.returns)
            if ret_name is None:
                self._reject(
                    f"Function '{self.function.name}' has an unsupported return annotation"
                )
            else:
                ret_base = ret_name.split("[", 1)[0]
                if ret_base in _TYPING_ALIASES:
                    ret_base = _TYPING_ALIASES[ret_base]
                if ret_base not in _SCALAR_TYPES and ret_base not in {"list", "tuple", "dict"}:
                    self._reject(
                        f"Function '{self.function.name}' return type '{ret_name}' is not primitive"
                    )
                self._check_iterable_annotation(
                    self.function.returns, "return annotation"
                )

    def _is_valid_type_name(self, name: str, context: str) -> bool:
        """Return True if ``name`` is a scalar or nested list/tuple/dict of scalars."""
        if not name:
            return False
        base = name.split("[", 1)[0].strip()
        if base in _SCALAR_TYPES:
            return True
        if base not in ("list", "tuple", "dict"):
            return False
        if "[" not in name or not name.endswith("]"):
            return False
        inner = name[name.find("[") + 1 : name.rfind("]")]
        parts = _split_type_args(inner)
        if not parts:
            return False
        for part in parts:
            if not self._is_valid_type_name(part.strip(), context):
                return False
        return True

    def _check_iterable_annotation(self, node: ast.AST, context: str) -> None:
        name = _annotation_name(node)
        if name is None:
            return
        base = name.split("[", 1)[0]
        if base not in ("list", "tuple", "dict"):
            return
        if "[" not in name or not name.endswith("]"):
            self._reject(
                f"Function '{self.function.name}' has unparameterized {base} annotation in {context}"
            )
            return
        if not self._is_valid_type_name(name, context):
            self._reject(
                f"Function '{self.function.name}' {base} annotation '{name}' is not a valid homogeneous/primitive collection"
            )

    def classify(self) -> Dict[str, Any]:
        """Return routing data for this function."""
        self._is_strictly_typed()
        self.visit(self.function)
        return {
            "is_hin": self.is_hin,
            "reasons": self.reasons,
            "callees": sorted(self.callees),
        }

    # ---- generic unsupported nodes ----
    def visit_Try(self, node: ast.Try) -> None:
        self._reject(f"Function '{self.function.name}' uses try/except exception handling")
        self.generic_visit(node)

    def visit_With(self, node: ast.With) -> None:
        self._reject(f"Function '{self.function.name}' uses with statements / context managers")
        self.generic_visit(node)

    def visit_AsyncWith(self, node: ast.AsyncWith) -> None:
        self._reject(f"Function '{self.function.name}' uses async with statements")
        self.generic_visit(node)

    def visit_Match(self, node: ast.Match) -> None:
        self._reject(f"Function '{self.function.name}' uses match/case pattern matching")
        self.generic_visit(node)

    def visit_Yield(self, node: ast.Yield) -> None:
        self._reject(f"Function '{self.function.name}' uses yield / generator expressions")
        self.generic_visit(node)

    def visit_YieldFrom(self, node: ast.YieldFrom) -> None:
        self._reject(f"Function '{self.function.name}' uses yield from")
        self.generic_visit(node)

    def visit_AsyncFor(self, node: ast.AsyncFor) -> None:
        self._reject(f"Function '{self.function.name}' uses async for")
        self.generic_visit(node)

    def visit_Await(self, node: ast.Await) -> None:
        self._reject(f"Function '{self.function.name}' uses await")
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._reject(f"Function '{self.function.name}' contains async/await syntax")
        # do not descend into nested functions

    def visit_Lambda(self, node: ast.Lambda) -> None:
        self._reject(f"Function '{self.function.name}' uses lambda expressions")
        self.generic_visit(node)

    def visit_NamedExpr(self, node: ast.NamedExpr) -> None:
        self._reject(f"Function '{self.function.name}' uses walrus operator")
        self.generic_visit(node)

    def visit_JoinedStr(self, node: ast.JoinedStr) -> None:
        self._reject(f"Function '{self.function.name}' uses f-string / string formatting")
        self.generic_visit(node)

    def visit_Dict(self, node: ast.Dict) -> None:
        # HIN-backed Rust generation now supports HashMap construction and
        # dict.{keys,values,items} iteration, so dictionary literals are allowed.
        self.generic_visit(node)

    def visit_Set(self, node: ast.Set) -> None:
        self._reject(f"Function '{self.function.name}' uses set literal")
        self.generic_visit(node)

    def visit_Starred(self, node: ast.Starred) -> None:
        self._reject(f"Function '{self.function.name}' uses starred unpacking")
        self.generic_visit(node)

    def visit_List(self, node: ast.List) -> None:
        if not _is_homogeneous_numeric_list(node):
            self._reject(
                f"Function '{self.function.name}' has a heterogeneous or non-numeric list literal"
            )
        self.generic_visit(node)

    def visit_BinOp(self, node: ast.BinOp) -> None:
        if isinstance(node.op, ast.Mod):
            left_is_str = isinstance(node.left, ast.Constant) and isinstance(
                node.left.value, str
            )
            if left_is_str:
                self._reject(
                    f"Function '{self.function.name}' uses '%' string formatting"
                )
        self.generic_visit(node)

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            top = alias.name.split(".")[0]
            if top not in _ALLOWED_MATH_MODULES and top not in _TYPE_MODULES:
                self._reject(
                    f"Function '{self.function.name}' imports non-math module '{alias.name}'"
                )
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        module = (node.module or "").split(".")[0]
        if module not in _ALLOWED_MATH_MODULES and module not in _TYPE_MODULES:
            self._reject(
                f"Function '{self.function.name}' imports from non-math module '{module}'"
            )
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        if isinstance(node.func, ast.Name):
            name = node.func.id
            if name in _IO_BUILTINS:
                self._reject(
                    f"Function '{self.function.name}' calls I/O/dynamic builtin '{name}'"
                )
            elif name not in _ALLOWED_BUILTINS and name not in self.local_functions:
                self._reject(
                    f"Function '{self.function.name}' calls unrecognized function '{name}'"
                )
            if name in self.local_functions:
                self.callees.add(name)
        elif isinstance(node.func, ast.Attribute):
            base = _base_name(node.func)
            attr = node.func.attr
            if base in _ALLOWED_MATH_MODULES:
                pass
            elif base == self.self_name:
                # Method calls on `self` are allowed for HIN-extractable classes.
                pass
            elif attr in _LIST_MUTATORS:
                # Local list/vector mutation methods are supported by the HIN pipeline.
                pass
            elif attr in _COLLECTION_ACCESSORS:
                # HashMap/Vec read accessors (get, keys, values, items) are supported.
                pass
            elif attr in _IO_METHODS:
                self._reject(
                    f"Function '{self.function.name}' calls I/O/dynamic method '{attr}'"
                )
            else:
                self._reject(
                    f"Function '{self.function.name}' calls unsupported method '{attr}'"
                )
        self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        if node is not self.function:
            self._reject(
                f"Function '{self.function.name}' contains nested function '{node.name}'"
            )
        # Descend only into the target function body, not into nested functions.
        if node is self.function:
            self.generic_visit(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self._reject(f"Function '{self.function.name}' defines a nested class")
        # do not descend


class CodeRouteClassifier:
    """AST classification visitor that decides how to route a Python module."""

    def __init__(
        self,
        source: str,
        function_names: Optional[List[str]] = None,
    ) -> None:
        self.source = source
        self.function_names = function_names

    def classify(self) -> Dict[str, Any]:
        """Return the routing decision payload for the provided source."""
        return classify(self.source, function_names=self.function_names)


def classify(source: str, function_names: Optional[List[str]] = None) -> Dict[str, Any]:
    """Inspect ``source`` and return a routing decision payload.

    The payload has the following shape:

        {
            "route": "HIN_COMPUTE" | "GENERAL_PURPOSE",
            "reasons": ["..."],
            "target_functions": ["func_a", "func_b"],
        }

    When ``function_names`` is provided, the top-level route is HIN only if every
    requested function appears in ``target_functions``.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        return {
            "route": GENERAL_PURPOSE,
            "reasons": [f"Could not parse source: {exc}"],
            "target_functions": [],
        }

    def _is_hin_class(node: ast.ClassDef) -> bool:
        if node.bases:
            return False
        for stmt in node.body:
            if (
                isinstance(stmt, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == "__slots__" for t in stmt.targets)
            ):
                return False
            if any(kw.arg == "__slots__" for kw in node.keywords):
                return False
        return True

    local_functions: Set[str] = {
        node.name
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and (isinstance(node, ast.FunctionDef) or _is_hin_class(node))
    }

    # First pass: classify each top-level function and eligible class independently.
    per_function: Dict[str, Dict[str, Any]] = {}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            classifier = _FunctionClassifier(node, local_functions)
            per_function[node.name] = classifier.classify()
        elif isinstance(node, ast.ClassDef) and _is_hin_class(node):
            per_function[node.name] = {
                "is_hin": True,
                "reasons": [],
                "callees": set(),
            }

    # Second pass: a HIN function may only call other HIN functions/classes.
    hin_set: Set[str] = {
        name
        for name, data in per_function.items()
        if data["is_hin"]
    }
    changed = True
    while changed:
        changed = False
        for name, data in per_function.items():
            if not data["is_hin"]:
                continue
            for callee in data["callees"]:
                callee_data = per_function.get(callee)
                if callee not in local_functions or (callee_data and not callee_data["is_hin"]):
                    data["is_hin"] = False
                    data["reasons"].append(
                        f"Function '{name}' calls '{callee}', which is not suitable for HIN extraction"
                    )
                    if name in hin_set:
                        hin_set.remove(name)
                    changed = True

    # Check module-level constructs that disqualify the whole module.
    reasons: List[str] = []
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.bases:
            reasons.append(
                f"Module contains class '{node.name}' with inheritance hierarchy"
            )

    target_functions = sorted(hin_set)

    def _lookup_node(name: str) -> Optional[ast.AST]:
        for node in tree.body:
            if (
                isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
                and node.name == name
            ):
                return node
        return None

    if function_names:
        requested = set(function_names)
        if target_functions and requested.issubset(set(target_functions)):
            route = HIN_COMPUTE
            reasons.insert(
                0,
                f"Requested function(s) can be extracted to HIN: {sorted(requested)}",
            )
        else:
            route = GENERAL_PURPOSE
            missing = sorted(requested - set(target_functions))
            if missing:
                reasons.append(
                    f"Requested function(s) not suitable for HIN: {missing}"
                )
                for name in missing:
                    data = per_function.get(name)
                    if data:
                        reasons.extend(data["reasons"])
                    else:
                        node = _lookup_node(name)
                        if isinstance(node, ast.AsyncFunctionDef):
                            reasons.append(
                                f"Function '{name}' uses async/await syntax"
                            )
                        elif isinstance(node, ast.ClassDef):
                            has_slots = any(
                                isinstance(kw.value, (ast.List, ast.Tuple))
                                and kw.arg == "__slots__"
                                for kw in node.keywords
                            )
                            if not has_slots:
                                for stmt in node.body:
                                    if (
                                        isinstance(stmt, ast.Assign)
                                        and any(
                                            isinstance(t, ast.Name)
                                            and t.id == "__slots__"
                                            for t in stmt.targets
                                        )
                                        and isinstance(
                                            stmt.value, (ast.List, ast.Tuple)
                                        )
                                    ):
                                        has_slots = True
                                        break
                            if has_slots:
                                reasons.append(
                                    f"Class '{name}' uses __slots__"
                                )
                            else:
                                reasons.append(
                                    f"Class '{name}' is not a HIN-suitable function"
                                )
                        elif node is None:
                            reasons.append(f"Function or class {name!r} not found")
                        else:
                            reasons.append(f"Function '{name}' is not suitable for HIN extraction")
            else:
                reasons.append("No HIN-suitable functions found in source")
    else:
        route = HIN_COMPUTE if target_functions else GENERAL_PURPOSE
        if route == HIN_COMPUTE:
            reasons.insert(
                0, f"Functions suitable for zero-allocation extraction: {target_functions}"
            )
        else:
            reasons.append("No functions matched HIN compute criteria")
            for name, data in sorted(per_function.items()):
                if not data["is_hin"]:
                    for reason in data["reasons"][:3]:
                        reasons.append(reason)

    return {
        "route": route,
        "reasons": reasons,
        "target_functions": target_functions,
    }


__all__ = ["classify", "HIN_COMPUTE", "GENERAL_PURPOSE", "CodeRouteClassifier"]
