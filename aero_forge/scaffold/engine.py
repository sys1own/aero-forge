"""Generate a self-contained Rust/PyO3 crate from an annotated HIN graph."""

from __future__ import annotations

import ast
import re
import logging
import sys
import tempfile
from importlib import resources
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from aero_forge._constants import IO_MODULES, IO_NAMES, MATH_ATTRS, MATH_CONSTANTS
from aero_forge.errors import UnsupportedError
from aero_forge.precision_shield.shield import _FLOAT_MATH_FUNCS

logger = logging.getLogger("aero_forge.scaffold.engine")


class Engine:
    """Write a Rust source crate for the functions described by ``annotated_graph``."""

    def generate(
        self,
        annotated_graph: Any,
        output_dir: Path,
        *,
        module_name: str,
        function_names: List[str],
        source: str,
    ) -> Path:
        """Create a temporary crate, write Cargo.toml and src/lib.rs, and return its path."""
        traits_by_name = self._traits(annotated_graph)
        crate_root = Path(tempfile.mkdtemp(prefix="accelerator-crate-"))
        src_dir = crate_root / "src"
        src_dir.mkdir(parents=True)

        tree = ast.parse(source)
        class_names = {
            node.name for node in tree.body if isinstance(node, ast.ClassDef)
        }
        function_blocks: List[str] = []
        module_init_lines: List[str] = []
        all_traits: Set[str] = set()

        for name in function_names:
            node, is_class = _find_top_level(tree, name)
            if node is None:
                raise UnsupportedError(
                    f"Function or class {name!r} not found in source"
                )
            traits = traits_by_name.get(name, {}) or {}
            if is_class:
                generator = ClassGenerator(node, module_name, traits, class_names)
                block = generator.emit()
                function_blocks.append(block)
                module_init_lines.append(
                    f"    m.add_class::<{_rust_identifier(node.name)}>()?;"
                )
                all_traits.update(generator.shield_traits())
            else:
                generator = RustGenerator(node, module_name, traits, class_names)
                block = generator.emit()
                function_blocks.append(block)
                module_init_lines.append(
                    f"    m.add_wrapped(wrap_pyfunction!({generator.rust_function_name}))?;"
                )
                all_traits.update(generator.shield_traits())

        cargo_template = (
            resources.files("aero_forge.templates").joinpath("Cargo.toml").read_text()
        )
        lib_template = (
            resources.files("aero_forge.templates").joinpath("lib.rs").read_text()
        )

        crate_name = _rust_identifier(module_name)
        extra_deps = (
            'rug = { version = "=1.24.0", features = ["integer"] }\n' 'az = "=1.2.1"'
            if all_traits
            else ""
        )
        cargo = cargo_template.format(crate_name=crate_name, extra_deps=extra_deps)
        lib = lib_template.format(
            shield_imports=_shield_imports(all_traits),
            functions="\n\n".join(function_blocks),
            module_init="\n".join(module_init_lines),
            module_name=crate_name,
        )

        (crate_root / "Cargo.toml").write_text(cargo, encoding="utf-8")
        (src_dir / "lib.rs").write_text(lib, encoding="utf-8")

        return crate_root

    @staticmethod
    def _traits(annotated_graph: Any) -> Dict[str, Any]:
        by_name = getattr(annotated_graph, "traits_by_name", None)
        if by_name is not None:
            return by_name
        single = getattr(annotated_graph, "traits", None)
        if isinstance(single, dict):
            return {single.get("function_name", ""): single}
        return {}


class RustGenerator:
    """Convert a Python numeric function into a Rust PyO3 extension."""

    IO_MODULES = IO_MODULES
    IO_NAMES = IO_NAMES
    MATH_ATTRS = MATH_ATTRS

    def __init__(
        self,
        func: ast.FunctionDef,
        module_name: str,
        traits: Dict[str, Any],
        class_names: Optional[Set[str]] = None,
    ):
        self.func = func
        self.orig_name = func.name
        self.safe_name = _rust_identifier(func.name)
        self.rust_function_name = f"_accel_{self.safe_name}"
        self.module_name = _rust_identifier(module_name)
        self.crate_name = self.module_name
        self.traits = traits
        self.function_type = traits.get("function_type", "i64")
        self.return_type = traits.get("return_type", self.function_type)
        self.class_names = class_names or set()

        arg_names = [a.arg for a in func.args.args]
        arg_types = self._annotated_arg_types(func, arg_names)
        if arg_types is None:
            arg_types = traits.get("arg_types") or [self.function_type] * len(arg_names)
        if len(arg_types) != len(arg_names):
            arg_types = [self.function_type] * len(arg_names)
        self.arg_names = arg_names
        self.arg_types = arg_types

        annotated_return = _annotation_to_rust_type(func.returns, self.class_names)
        self.annotated_return = bool(annotated_return)
        if annotated_return:
            self.return_type = annotated_return

        self.assigned = self._collect_assigned()
        self.type_env: Dict[str, str] = {}
        for name, typ in zip(self.arg_names, self.arg_types):
            self.type_env[name] = typ
        self._infer_all_types()
        self._tmp_counter = 0
        self.used_traits: Set[str] = set()

    def shield_traits(self) -> Set[str]:
        return self.used_traits

    def emit(self) -> str:
        return self._emit_function()

    def _annotated_arg_types(
        self, func: ast.FunctionDef, arg_names: List[str]
    ) -> Optional[List[str]]:
        """Return Rust types from parameter annotations, or None if none are present."""
        types: List[str] = []
        any_annotation = False
        for arg in func.args.args:
            typ = _annotation_to_rust_type(arg.annotation, self.class_names)
            if typ:
                any_annotation = True
            types.append(typ or self.function_type)
        return types if any_annotation else None

    def _infer_all_types(self) -> None:
        """Iteratively infer Rust types for arguments and locals from usage."""
        # Start from unknown so subscript/append-driven inference can override
        # the default scalar function_type. Annotated arguments are seeded with
        # their declared type because the source explicitly gives us that type.
        types: Dict[str, Optional[str]] = {}
        for i, name in enumerate(self.arg_names):
            if self.func.args.args[i].annotation is not None:
                types[name] = self.arg_types[i]
            else:
                types[name] = None

        for stmt in self.func.body:
            self._infer_annotated_local(stmt, types)

        for _ in range(20):
            new_types = dict(types)
            for stmt in self.func.body:
                self._infer_stmt_types(stmt, new_types)
            if new_types == types:
                break
            types = new_types

        # Fall back to default/annotated arg types for anything still unknown.
        for i, name in enumerate(self.arg_names):
            if types.get(name) is None:
                types[name] = self.arg_types[i]
            self.arg_types[i] = types.get(name, self.arg_types[i])

        # Replace any unresolved placeholder element types with the default.
        def resolve(t: Optional[str]) -> Optional[str]:
            if t is None:
                return None
            return t.replace("?", self.function_type)

        self.type_env = {k: resolve(v) for k, v in types.items() if v is not None}

        if not self.annotated_return:
            ret_type = resolve(types.get("__return__"))
            if ret_type:
                self.return_type = ret_type

    def _infer_annotated_local(self, stmt: ast.stmt, types: Dict[str, str]) -> None:
        if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
            typ = _annotation_to_rust_type(stmt.annotation, self.class_names)
            if typ:
                types[stmt.target.id] = typ
        for child in ("body", "orelse"):
            for child_stmt in getattr(stmt, child, []):
                self._infer_annotated_local(child_stmt, types)

    def _infer_stmt_types(self, stmt: ast.stmt, types: Dict[str, str]) -> None:
        if isinstance(stmt, ast.Assign):
            for target in stmt.targets:
                if isinstance(target, ast.Name):
                    rhs_type = self._infer_expr_type(stmt.value, types)
                    if rhs_type:
                        types[target.id] = self._unify(types.get(target.id), rhs_type)
                    self._propagate_subscript_types(stmt.value, rhs_type, types)
        elif isinstance(stmt, ast.AugAssign):
            target_type = None
            if isinstance(stmt.target, ast.Name):
                target_type = types.get(stmt.target.id)
            rhs_type = (
                target_type if target_type else self._infer_expr_type(stmt.value, types)
            ) or self.function_type
            if isinstance(stmt.target, ast.Name):
                types[stmt.target.id] = self._unify(types.get(stmt.target.id), rhs_type)
            self._propagate_subscript_types(stmt.value, rhs_type, types)
        elif isinstance(stmt, ast.Expr):
            if isinstance(stmt.value, ast.Call) and _call_name(stmt.value) == "append":
                target_name = _call_base(stmt.value)
                arg = stmt.value.args[0]
                arg_type = self._infer_expr_type(arg, types) or self.function_type
                if target_name:
                    types[target_name] = self._unify(
                        types.get(target_name), f"Vec<{arg_type}>"
                    )
                self._propagate_subscript_types(arg, arg_type, types)
        elif isinstance(stmt, ast.Return):
            if stmt.value is not None:
                ret_type = (
                    self._infer_expr_type(stmt.value, types) or self.function_type
                )
                types["__return__"] = self._unify(types.get("__return__"), ret_type)
                self._propagate_subscript_types(stmt.value, ret_type, types)
        elif isinstance(stmt, ast.For):
            self._infer_for_types(stmt, types)

        for child in ("body", "orelse"):
            for child_stmt in getattr(stmt, child, []):
                self._infer_stmt_types(child_stmt, types)

    def _infer_for_types(self, stmt: ast.For, types: Dict[str, str]) -> None:
        target = stmt.target
        iter_expr = stmt.iter
        if isinstance(iter_expr, ast.Call):
            name = _call_name(iter_expr)
            if name == "range" and isinstance(target, ast.Name) and target.id != "_":
                types[target.id] = "i64"
            elif name == "enumerate" and isinstance(target, ast.Tuple):
                if len(target.elts) >= 1 and isinstance(target.elts[0], ast.Name):
                    types[target.elts[0].id] = "i64"
                if len(target.elts) >= 2 and isinstance(target.elts[1], ast.Name):
                    iterable = iter_expr.args[0] if iter_expr.args else None
                    if iterable is not None:
                        types[target.elts[1].id] = self._infer_element_or_ref(
                            self._infer_expr_type(iterable, types)
                        )
            elif name == "zip" and isinstance(target, ast.Tuple):
                for elt, arg in zip(target.elts, iter_expr.args):
                    if isinstance(elt, ast.Name):
                        types[elt.id] = self._infer_element_or_ref(
                            self._infer_expr_type(arg, types)
                        )
            elif name not in ("range", "enumerate", "zip"):
                pass
        elif isinstance(iter_expr, ast.Name) and isinstance(target, ast.Name):
            types[target.id] = self._infer_element_or_ref(
                types.get(iter_expr.id, self.function_type)
            )

    def _infer_element_or_ref(self, container_type: Optional[str]) -> str:
        if not container_type or not container_type.startswith("Vec<"):
            return self.function_type
        element = _element_type(container_type)
        if element in ("i64", "f64", "bool"):
            return element
        return f"&{element}"

    def _infer_expr_type(self, expr: ast.expr, types: Dict[str, str]) -> Optional[str]:
        if isinstance(expr, ast.Constant):
            if isinstance(expr.value, bool):
                return "bool"
            if isinstance(expr.value, int):
                return "i64"
            if isinstance(expr.value, float):
                return "f64"
            if expr.value is None:
                return "()"
            return None
        if isinstance(expr, ast.Name):
            return types.get(expr.id)
        if isinstance(expr, ast.Attribute):
            return self._field_type(expr.value, expr.attr)
        if isinstance(expr, ast.Subscript):
            base_type = self._infer_expr_type(expr.value, types)
            if base_type and base_type.startswith("Vec<"):
                element_type = _element_type(base_type)
                if isinstance(expr.slice, ast.Slice):
                    return f"Vec<{element_type}>"
                return element_type
            return None
        if isinstance(expr, ast.List):
            if not expr.elts:
                return "Vec<?>"
            inner = self._infer_expr_type(expr.elts[0], types) or self.function_type
            return f"Vec<{inner}>"
        if isinstance(expr, ast.ListComp):
            inner = self._infer_expr_type(expr.elt, types) or self.function_type
            return f"Vec<{inner}>"
        if isinstance(expr, ast.Tuple):
            elts = [self._infer_expr_type(e, types) for e in expr.elts]
            if all(elts):
                return f"({', '.join(elts)})"
            return None
        if isinstance(expr, ast.BinOp):
            left = self._infer_expr_type(expr.left, types)
            right = self._infer_expr_type(expr.right, types)
            if isinstance(expr.op, ast.Mult):
                if left and left.startswith("Vec<"):
                    return left
                if right and right.startswith("Vec<"):
                    return right
            return self._unify(left, right)
        if isinstance(expr, ast.UnaryOp):
            return self._infer_expr_type(expr.operand, types)
        if isinstance(expr, (ast.Compare, ast.BoolOp)):
            return "bool"
        if isinstance(expr, ast.IfExp):
            return self._unify(
                self._infer_expr_type(expr.body, types),
                self._infer_expr_type(expr.orelse, types),
            )
        if isinstance(expr, ast.Call):
            name = _call_name(expr)
            base = _call_base(expr)
            if name == "len" and base is None:
                return "i64"
            if base == "math" and name in _FLOAT_MATH_FUNCS:
                return "f64"
            if name in ("sqrt", "sin", "cos", "tan", "exp", "log", "log10"):
                return "f64"
            if name == self.func.name:
                return self.return_type or self.function_type
            if name in self.class_names:
                return name
        return None

    def _propagate_subscript_types(
        self, expr: ast.expr, element_type: Optional[str], types: Dict[str, str]
    ) -> None:
        if element_type is None:
            return
        if isinstance(expr, ast.Subscript):
            base = expr.value
            if isinstance(base, ast.Name):
                current = types.get(base.id)
                types[base.id] = self._unify(current, f"Vec<{element_type}>")
            elif isinstance(base, ast.Subscript):
                self._propagate_subscript_types(base, f"Vec<{element_type}>", types)
            else:
                self._propagate_subscript_types(base, f"Vec<{element_type}>", types)
        elif isinstance(expr, ast.BinOp):
            self._propagate_subscript_types(expr.left, element_type, types)
            self._propagate_subscript_types(expr.right, element_type, types)
        elif isinstance(expr, ast.UnaryOp):
            self._propagate_subscript_types(expr.operand, element_type, types)
        elif isinstance(expr, ast.Compare):
            for op in expr.comparators:
                self._propagate_subscript_types(op, element_type, types)
        elif isinstance(expr, ast.BoolOp):
            for v in expr.values:
                self._propagate_subscript_types(v, element_type, types)
        elif isinstance(expr, ast.IfExp):
            self._propagate_subscript_types(expr.body, element_type, types)
            self._propagate_subscript_types(expr.orelse, element_type, types)
        elif isinstance(expr, ast.Tuple):
            for i, e in enumerate(expr.elts):
                # Each tuple element must match the corresponding element type.
                # element_type is itself a tuple type if we are propagating into a tuple.
                if element_type.startswith("(") and element_type.endswith(")"):
                    inner = self._tuple_element_at(element_type, i)
                    self._propagate_subscript_types(e, inner, types)
                else:
                    self._propagate_subscript_types(e, element_type, types)

    def _tuple_element_at(self, tuple_type: str, index: int) -> Optional[str]:
        """Return the i-th element type of a Rust tuple type string."""
        if not (tuple_type.startswith("(") and tuple_type.endswith(")")):
            return None
        inner = tuple_type[1:-1]
        if not inner:
            return None
        # Simple split on top-level commas.
        parts = []
        depth = 0
        current = ""
        for ch in inner:
            if ch in "(<":
                depth += 1
            elif ch in ")>":
                depth -= 1
            if ch == "," and depth == 0:
                parts.append(current.strip())
                current = ""
            else:
                current += ch
        parts.append(current.strip())
        if index < len(parts):
            return parts[index]
        return None

    def _unify(self, t1: Optional[str], t2: Optional[str]) -> Optional[str]:
        if t1 is None:
            return t2
        if t2 is None:
            return t1
        if t1 == t2:
            return t1
        # ``?`` is an unknown/placeholder element type from an empty list literal.
        if t1 == "?":
            return t2
        if t2 == "?":
            return t1
        if t1.startswith("Vec<") and t2.startswith("Vec<"):
            e1 = _element_type(t1)
            e2 = _element_type(t2)
            unified = self._unify(e1, e2)
            if unified:
                return f"Vec<{unified}>"
        if (
            t1.startswith("(")
            and t1.endswith(")")
            and t2.startswith("(")
            and t2.endswith(")")
        ):
            parts1 = self._tuple_element_types(t1)
            parts2 = self._tuple_element_types(t2)
            if len(parts1) == len(parts2):
                unified = [self._unify(a, b) for a, b in zip(parts1, parts2)]
                if all(unified):
                    return f"({', '.join(unified)})"
        if "f64" in (t1, t2):
            return "f64"
        if "i64" in (t1, t2):
            return "i64"
        if "bool" in (t1, t2):
            return "i64" if "i64" in (t1, t2) else "bool"
        return t1

    def _tuple_element_types(self, tuple_type: str) -> List[str]:
        if not (tuple_type.startswith("(") and tuple_type.endswith(")")):
            return []
        inner = tuple_type[1:-1]
        if not inner:
            return []
        parts = []
        depth = 0
        current = ""
        for ch in inner:
            if ch in "(<":
                depth += 1
            elif ch in ")>":
                depth -= 1
            if ch == "," and depth == 0:
                parts.append(current.strip())
                current = ""
            else:
                current += ch
        parts.append(current.strip())
        return parts

    def _type_of(self, expr: ast.expr) -> str:
        """Return the Rust type of an expression without emitting it."""
        if isinstance(expr, ast.Constant):
            if isinstance(expr.value, bool):
                return "bool"
            if isinstance(expr.value, int):
                return "i64"
            if isinstance(expr.value, float):
                return "f64"
            if expr.value is None:
                return "()"
        if isinstance(expr, ast.Name):
            return self.type_env.get(expr.id, self.function_type)
        if isinstance(expr, ast.Attribute):
            return self._field_type(expr.value, expr.attr)
        if isinstance(expr, ast.Subscript):
            base_type = self._type_of(expr.value)
            if not base_type.startswith("Vec<"):
                return self.function_type
            element_type = _element_type(base_type)
            if isinstance(expr.slice, ast.Slice):
                return f"Vec<{element_type}>"
            return element_type
        if isinstance(expr, ast.List):
            if not expr.elts:
                return "Vec<?>"
            return f"Vec<{self._type_of(expr.elts[0])}>"
        if isinstance(expr, ast.ListComp):
            return f"Vec<{self._type_of(expr.elt)}>"
        if isinstance(expr, ast.BinOp):
            if isinstance(expr.op, ast.Mult):
                left_type = self._type_of(expr.left)
                if left_type.startswith("Vec<") or isinstance(expr.left, ast.List):
                    return left_type
                right_type = self._type_of(expr.right)
                if right_type.startswith("Vec<") or isinstance(expr.right, ast.List):
                    return right_type
            left_type = self._type_of(expr.left)
            right_type = self._type_of(expr.right)
            if "f64" in (left_type, right_type):
                return "f64"
            if "i64" in (left_type, right_type):
                return "i64"
            return self.function_type
        if isinstance(expr, ast.UnaryOp):
            return self._type_of(expr.operand)
        if isinstance(expr, (ast.Compare, ast.BoolOp)):
            return "bool"
        if isinstance(expr, ast.IfExp):
            return self._type_of(expr.body)
        if isinstance(expr, ast.Call):
            name = _call_name(expr)
            base = _call_base(expr)
            if name == "len" and base is None:
                return "i64"
            if base == "math" and name in _FLOAT_MATH_FUNCS:
                return "f64"
            if name in ("sqrt", "sin", "cos", "tan", "exp", "log", "log10"):
                return "f64"
            if name in ("pow",):
                return self.function_type
            if base in ("np", "numpy"):
                return self._type_of_numpy_call(name, expr)
            if name == self.func.name and name not in self.class_names:
                return self.return_type or self.function_type
            if getattr(self, "class_name", None) and name == self.class_name:
                return "Self"
            if name in self.class_names:
                return name
        return self.function_type

    def _type_of_numpy_call(self, name: str, expr: ast.Call) -> str:
        if name in ("zeros", "ones"):
            if expr.args and isinstance(expr.args[0], ast.Tuple):
                dims = len(expr.args[0].elts)
                return ("Vec<" * dims) + "f64" + (">" * dims)
            return "Vec<f64>"
        if name == "array" and expr.args:
            return self._type_of(expr.args[0])
        if name == "dot" and len(expr.args) == 2:
            t1 = self._type_of(expr.args[0])
            t2 = self._type_of(expr.args[1])
            if t1 == "Vec<f64>" and t2 == "Vec<f64>":
                return "f64"
            if "Vec<Vec<f64>>" in (t1, t2):
                if t1 == t2:
                    return "Vec<Vec<f64>>"
                matrix = t1 if t1 == "Vec<Vec<f64>>" else t2
                vector = t2 if matrix == t1 else t1
                if vector == "Vec<f64>":
                    return "Vec<f64>"
            return "f64"
        if name == "sum" and expr.args:
            return "f64"
        return "Vec<f64>"

    def _field_type(self, value: ast.expr, attr: str) -> str:
        """Return the type of ``value.attr`` when ``value`` is a class instance."""
        base_type = self._type_of(value)
        class_names = self.class_names | {getattr(self, "class_name", "")}
        class_names.discard("")
        if base_type in class_names or base_type == "Self":
            fields = getattr(self, "fields", {})
            return fields.get(attr, self.function_type)
        if base_type.startswith("&") and base_type[1:] in class_names:
            fields = getattr(self, "fields", {})
            return fields.get(attr, self.function_type)
        return self.function_type

    def _coerce(self, expr: str, from_type: str, to_type: str) -> str:
        """Cast ``expr`` from ``from_type`` to ``to_type`` when necessary."""
        if from_type == to_type:
            return expr
        if from_type == "Self" and to_type == getattr(self, "class_name", ""):
            return expr
        if from_type == getattr(self, "class_name", "") and to_type == "Self":
            return expr
        if to_type.startswith("&") and from_type == to_type[1:]:
            return expr
        if from_type.startswith("&") and to_type == from_type[1:]:
            if to_type.startswith("Vec<"):
                return f"({expr}).clone()"
            return expr
        if from_type.startswith("Vec<") and to_type.startswith("Vec<"):
            return expr
        if from_type == "i64" and to_type == "f64":
            return f"({expr} as f64)"
        if from_type == "f64" and to_type == "i64":
            return f"({expr} as i64)"
        if from_type == "bool" and to_type in ("i64", "f64"):
            return f"({expr} as {to_type})"
        if from_type == "i64" and to_type == "bool":
            return f"({expr} != 0)"
        if from_type == "f64" and to_type == "bool":
            return f"({expr} != 0.0)"
        return expr

    # ------------------------------------------------------------------
    # Collection helpers
    # ------------------------------------------------------------------
    def _collect_assigned(self) -> set[str]:
        names: set[str] = set()
        for node in ast.walk(self.func):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    names.update(_names_in_target(target))
            elif isinstance(node, ast.AnnAssign):
                names.update(_names_in_target(node.target))
            elif isinstance(node, ast.AugAssign):
                names.update(_names_in_target(node.target))
        return names

    @staticmethod
    def _name_in_expr(expr: ast.expr, name: str) -> bool:
        for node in ast.walk(expr):
            if isinstance(node, ast.Name) and node.id == name:
                return True
        return False

    def _is_mutable(self, name: str) -> bool:
        """Return True if ``name`` is assigned more than once, in a loop, or appended to."""
        count = self._count_targets_in_body(name, self.func.body, in_loop=False)
        if self._has_append_call(name, self.func.body):
            return True
        if name in self.arg_names:
            # The parameter is the first binding; any target assignment is a
            # reassignment, so the local shadow needs to be mutable.
            return count > 0
        return count > 1

    def _has_append_call(self, name: str, stmts: List[ast.stmt]) -> bool:
        for stmt in stmts:
            if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
                call = stmt.value
                if (
                    isinstance(call.func, ast.Attribute)
                    and isinstance(call.func.value, ast.Name)
                    and call.func.value.id == name
                    and call.func.attr in ("append", "extend", "sort", "reverse")
                ):
                    return True
            for child in ("body", "orelse"):
                body = getattr(stmt, child, [])
                if body and self._has_append_call(name, body):
                    return True
        return False

    @staticmethod
    def _rhs_uses_only(expr: ast.expr, allowed: set[str]) -> bool:
        """Return True if ``expr`` references no names outside ``allowed``."""
        for node in ast.walk(expr):
            if isinstance(node, ast.Name) and node.id not in allowed:
                return False
        return True

    def _count_targets_in_body(
        self, name: str, stmts: List[ast.stmt], in_loop: bool
    ) -> int:
        return sum(self._count_targets(name, stmt, in_loop) for stmt in stmts)

    def _count_targets(self, name: str, stmt: ast.stmt, in_loop: bool) -> int:
        if isinstance(stmt, ast.Assign):
            for target in stmt.targets:
                if name in _names_in_target(target):
                    # An assignment inside a loop may execute multiple times.
                    return 2 if in_loop else 1
            return 0
        if isinstance(stmt, ast.AnnAssign):
            if stmt.value and name in _names_in_target(stmt.target):
                return 2 if in_loop else 1
            return 0
        if isinstance(stmt, ast.AugAssign):
            if name in _names_in_target(stmt.target):
                return 2 if in_loop else 1
            return 0
        if isinstance(stmt, ast.If):
            return self._count_targets_in_body(
                name, stmt.body, in_loop
            ) + self._count_targets_in_body(name, stmt.orelse, in_loop)
        if isinstance(stmt, (ast.For, ast.While)):
            return self._count_targets_in_body(
                name, stmt.body, in_loop=True
            ) + self._count_targets_in_body(name, stmt.orelse, in_loop=True)
        return 0

    def _initializers_and_body(self) -> Tuple[List[str], List[ast.stmt]]:
        """Return `let` declarations and the remaining body statements.

        Local variables are initialized with their first top-level assignment
        when the right-hand side only references already-declared names. The
        ``mut`` keyword is omitted when the variable is never assigned again. This
        preserves source order and avoids dummy zero values that are immediately
        overwritten.
        """
        defaults: List[str] = []
        body: List[ast.stmt] = []
        declared = set(self.arg_names)

        for stmt in self.func.body:
            if isinstance(stmt, ast.Assign) and all(
                isinstance(t, ast.Name) for t in stmt.targets
            ):
                target_names = [t.id for t in stmt.targets]
                new_targets = [
                    n
                    for n in target_names
                    if n in self.assigned
                    and n not in self.arg_names
                    and n not in declared
                ]
                if new_targets and all(
                    not self._name_in_expr(stmt.value, n) for n in new_targets
                ):
                    # Only hoist the initializer if all RHS references are already
                    # declared; otherwise fall back to normal body emission.
                    first_new = new_targets[0]
                    if (
                        isinstance(stmt.value, ast.List)
                        and not stmt.value.elts
                        and first_new in self.type_env
                    ):
                        rhs_type = self.type_env[first_new]
                    else:
                        rhs_type = self._type_of(stmt.value)
                    value = self._strip_outer_parens(
                        self._emit_expr(stmt.value, rhs_type)
                    )
                    for name in target_names:
                        if name in new_targets:
                            mutable = self._is_mutable(name)
                            mut = "mut " if mutable else ""
                            defaults.append(f"let {mut}{name} = {value};")
                            declared.add(name)
                    if not all(n in declared for n in target_names):
                        body.append(stmt)
                    continue
            body.append(stmt)

        # Reassigned arguments need a mutable shadow binding.
        for name in self.arg_names:
            if name in self.assigned:
                defaults.append(f"let mut {name} = {name};")

        # Any remaining assigned names are declared uninitialized; the body will
        # initialize them on first use.
        remaining = sorted(self.assigned - declared)
        for name in remaining:
            mutable = self._is_mutable(name)
            mut = "mut " if mutable else ""
            defaults.append(f"let {mut}{name};")

        # Argument shadows should appear before local declarations.
        return defaults, body

    def _zero(self) -> str:
        return self._zero_for_type(self.return_type)

    def _zero_for_type(self, typ: str) -> str:
        if typ == "f64":
            return "0.0_f64"
        if typ == "bool":
            return "false"
        if typ.startswith("Vec<"):
            inner = _element_type(typ)
            return f"Vec::<{inner}>::new()"
        if typ.startswith("(") and typ.endswith(")"):
            parts = self._tuple_element_types(typ)
            return f"({', '.join(self._zero_for_type(p) for p in parts)})"
        return "0_i64"

    def _return_type(self) -> str:
        """Derive the Rust return type from the function's return statements."""
        # If the function was explicitly annotated with a tuple type, trust it.
        if self.annotated_return and self.return_type.startswith("("):
            return self.return_type

        sizes: set[int] = set()
        return_values: List[ast.expr] = []
        for node in ast.walk(self.func):
            if isinstance(node, ast.Return) and node.value is not None:
                return_values.append(node.value)
                if isinstance(node.value, (ast.Tuple, ast.List)):
                    sizes.add(len(_elements(node.value)))
                else:
                    sizes.add(1)
        if not sizes or sizes == {1}:
            return self.return_type
        if len(sizes) != 1:
            raise UnsupportedError(
                "All return statements must return the same tuple size",
                node=self.func,
            )

        n = sizes.pop()
        element_types: List[Optional[str]] = [None] * n
        for rv in return_values:
            if isinstance(rv, (ast.Tuple, ast.List)):
                elts = _elements(rv)
            else:
                elts = [rv]
            for i, elt in enumerate(elts):
                typ = self._infer_expr_type(elt, self.type_env)
                element_types[i] = self._unify(element_types[i], typ)
        for i in range(n):
            if element_types[i] is None:
                element_types[i] = self.return_type
        return f"({', '.join(element_types)})"

    def _next_tmp(self) -> str:
        self._tmp_counter += 1
        return f"_accel_tmp{self._tmp_counter}"

    @staticmethod
    def _strip_outer_parens(expr: str) -> str:
        while len(expr) >= 2 and expr[0] == "(" and expr[-1] == ")":
            depth = 0
            match_index = -1
            for i, ch in enumerate(expr):
                if ch == "(":
                    depth += 1
                elif ch == ")":
                    depth -= 1
                    if depth == 0:
                        match_index = i
                        break
            if match_index == len(expr) - 1:
                expr = expr[1:-1]
            else:
                break
        return expr

    # ------------------------------------------------------------------
    # Statement emission
    # ------------------------------------------------------------------
    def _emit_function(self) -> str:
        args = ", ".join(
            f"{name}: {typ}" for name, typ in zip(self.arg_names, self.arg_types)
        )
        return_type = self._return_type()
        header = (
            f'#[pyfunction(name = "{self.orig_name}")]\n'
            f"fn {self.rust_function_name}({args}) -> {return_type} {{"
        )

        defaults, body_stmts = self._initializers_and_body()
        body_lines = [self._emit_stmt(stmt) for stmt in body_stmts]
        if (
            body_stmts
            and isinstance(body_stmts[-1], ast.If)
            and not body_stmts[-1].orelse
            and self._block_returns(body_stmts[-1].body)
        ):
            body_lines.append(f"return {self._zero()};")
        body = "\n".join(defaults + body_lines)
        indented = "\n".join("    " + line for line in body.splitlines())
        return f"{header}\n{indented}\n}}"

    def _emit_stmt(self, stmt: ast.stmt) -> str:
        if isinstance(stmt, ast.Return):
            if stmt.value is None:
                # Bare ``return`` compiles to the zero value for the declared return type.
                if self.return_type == "()":
                    return "return;"
                return f"return {self._zero()};"
            if isinstance(stmt.value, (ast.Tuple, ast.List)):
                value = self._emit_expr(stmt.value, self.return_type)
            else:
                value = self._strip_outer_parens(
                    self._emit_expr(stmt.value, self.return_type)
                )
            return f"return {value};"
        if isinstance(stmt, ast.Assign):
            return self._emit_assign(stmt)
        if isinstance(stmt, ast.AnnAssign):
            if stmt.value is None:
                raise UnsupportedError(
                    "Annotated assignments without a value are not supported", node=stmt
                )
            return self._emit_assign(
                ast.Assign(targets=[stmt.target], value=stmt.value, lineno=stmt.lineno)
            )
        if isinstance(stmt, (ast.Import, ast.ImportFrom)):
            return ""
        if isinstance(stmt, ast.AugAssign):
            return self._emit_augassign(stmt)
        if isinstance(stmt, ast.If):
            return self._emit_if(stmt)
        if isinstance(stmt, ast.While):
            return self._emit_while(stmt)
        if isinstance(stmt, ast.For):
            return self._emit_for(stmt)
        if isinstance(stmt, ast.Break):
            return "break;"
        if isinstance(stmt, ast.Continue):
            return "continue;"
        if isinstance(stmt, ast.Pass):
            return ""
        if isinstance(stmt, ast.Expr):
            # Docstrings and standalone string constants have no side effects;
            # skip them. Other expressions are validated to ensure I/O and
            # unsupported calls do not slip through as ignored statements.
            if isinstance(stmt.value, ast.Constant) and isinstance(
                stmt.value.value, str
            ):
                return ""
            if isinstance(stmt.value, ast.Call) and _call_name(stmt.value) in (
                "append",
                "extend",
            ):
                if _call_name(stmt.value) == "append":
                    return self._emit_append(stmt.value)
                return self._emit_call(stmt.value, self.function_type)
            self._emit_expr(stmt.value, self.function_type)
            return ""
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
        if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)) and isinstance(
            getattr(stmt, "returns", None), ast.Await
        ):
            # This branch is not normally reachable; async def is handled earlier.
            raise UnsupportedError("async/await is not supported", node=stmt)
        if isinstance(stmt, ast.Match):
            raise UnsupportedError("match/case is not supported", node=stmt)
        raise UnsupportedError(
            f"Unsupported statement: {type(stmt).__name__}", node=stmt
        )

    def _emit_assign(self, stmt: ast.Assign) -> str:
        if len(stmt.targets) == 1:
            target = stmt.targets[0]
            if isinstance(target, ast.Name):
                name = target.id
                if (
                    isinstance(stmt.value, ast.List)
                    and not stmt.value.elts
                    and name in self.type_env
                ):
                    rhs_type = self.type_env[name]
                else:
                    rhs_type = self._type_of(stmt.value)
                value = self._strip_outer_parens(self._emit_expr(stmt.value, rhs_type))
                return f"{name} = {value};"
            if isinstance(target, ast.Subscript):
                target_type = self._type_of(target)
                rhs_type = target_type
                if isinstance(target.slice, ast.Slice):
                    base_type = self._type_of(target.value)
                    base_expr = self._emit_expr(target.value, base_type)
                    lower = (
                        self._emit_expr(target.slice.lower, "i64")
                        if target.slice.lower is not None
                        else "0"
                    )
                    if target.slice.upper is None:
                        upper = f"({base_expr}).len()"
                    else:
                        upper = self._emit_expr(target.slice.upper, "i64")
                    value = self._strip_outer_parens(
                        self._emit_expr(stmt.value, rhs_type)
                    )
                    return f"{base_expr}.splice(({lower}) as usize..({upper}) as usize, {value});"
                lvalue = self._emit_lvalue(target, target_type)
                value = self._strip_outer_parens(self._emit_expr(stmt.value, rhs_type))
                return f"{lvalue} = {value};"
            if isinstance(target, (ast.Tuple, ast.List)):
                return self._emit_tuple_unpack(target, stmt.value)

        # Chain assignment: ``a = b = expr`` becomes a temporary plus assignments.
        if all(isinstance(t, ast.Name) for t in stmt.targets):
            first_name = stmt.targets[0].id
            if (
                isinstance(stmt.value, ast.List)
                and not stmt.value.elts
                and first_name in self.type_env
            ):
                rhs_type = self.type_env[first_name]
            else:
                rhs_type = self._type_of(stmt.value)
            value = self._strip_outer_parens(self._emit_expr(stmt.value, rhs_type))
            tmp = self._next_tmp()
            lines = [f"let {tmp} = {value};"]
            for target in stmt.targets:
                lines.append(f"{target.id} = {tmp};")
            return "\n".join(lines)

        raise UnsupportedError(
            "Only single-target, tuple unpacking, or chain assignments are supported",
            node=stmt,
        )

    def _emit_append(self, expr: ast.Call) -> str:
        if not isinstance(expr.func, ast.Attribute):
            raise UnsupportedError("append() must be a method call", node=expr)
        if len(expr.args) != 1:
            raise UnsupportedError("append() takes exactly one argument", node=expr)
        target = expr.func.value
        container_type = self._type_of(target)
        if not container_type.startswith("Vec<"):
            raise UnsupportedError(
                "append() is only supported on list/Vec types", node=expr
            )
        element_type = _element_type(container_type)
        target_str = self._emit_expr(target, container_type)
        arg_str = self._emit_expr(expr.args[0], element_type)
        return f"{target_str}.push({arg_str});"

    def _emit_tuple_unpack(self, target: ast.AST, value: ast.expr) -> str:
        names = _names_in_target(target)
        if len(names) != len(_elements(target)):
            raise UnsupportedError(
                "Only plain names may appear in a tuple unpack", node=target
            )
        if not isinstance(value, (ast.Tuple, ast.List)):
            raise UnsupportedError(
                "Tuple unpack requires a tuple/list on the right", node=value
            )

        value_elts = _elements(value)
        elements = [
            self._strip_outer_parens(self._emit_expr(e, self._type_of(e)))
            for e in value_elts
        ]
        tmp = self._next_tmp()
        # The parentheses here form a Rust tuple literal, so we keep them.
        lines = [f"let {tmp} = ({', '.join(elements)});"]
        for i, name in enumerate(names):
            lines.append(f"{name} = {tmp}.{i};")
        return "\n".join(lines)

    def _emit_augassign(self, stmt: ast.AugAssign) -> str:
        if isinstance(stmt.target, ast.Name):
            name = stmt.target.id
            target_type = self._type_of(stmt.target)
            fake = ast.BinOp(
                left=ast.Name(id=name, ctx=ast.Load()),
                op=stmt.op,
                right=stmt.value,
            )
            value = self._strip_outer_parens(self._emit_binop(fake, target_type))
            return f"{name} = {value};"
        target_type = self._type_of(stmt.target)
        lvalue = self._emit_lvalue(stmt.target, target_type)
        rhs = self._emit_expr(stmt.value, target_type)
        op = _augassign_op(stmt.op)
        return f"{lvalue} {op} {rhs};"

    def _emit_lvalue(self, target: ast.expr, ctx: str) -> str:
        """Emit a place expression, stripping outer parentheses when possible."""
        expr = self._emit_expr(target, ctx)
        while True:
            stripped = self._strip_outer_parens(expr)
            if stripped == expr:
                break
            expr = stripped
        return expr

    def _emit_if(self, stmt: ast.If) -> str:
        cond = self._strip_outer_parens(self._emit_expr(stmt.test, "bool"))
        then_body = self._emit_body(stmt.body)

        parts = [f"if {cond} {{\n{then_body}\n}}"]
        orelse = stmt.orelse
        while orelse and len(orelse) == 1 and isinstance(orelse[0], ast.If):
            inner = orelse[0]
            inner_cond = self._strip_outer_parens(self._emit_expr(inner.test, "bool"))
            inner_body = self._emit_body(inner.body)
            parts.append(f"else if {inner_cond} {{\n{inner_body}\n}}")
            orelse = inner.orelse

        if orelse:
            else_body = self._emit_body(orelse)
            parts.append(f"else {{\n{else_body}\n}}")

        return " ".join(parts)

    @staticmethod
    def _block_returns(stmts: List[ast.stmt]) -> bool:
        """Return True if the statement block contains an unconditional return."""
        for stmt in stmts:
            for node in ast.walk(stmt):
                if isinstance(node, ast.Return):
                    return True
        return False

    def _emit_while(self, stmt: ast.While) -> str:
        cond = self._strip_outer_parens(self._emit_expr(stmt.test, "bool"))
        body = self._emit_body(stmt.body)
        return f"while {cond} {{\n{body}\n}}"

    def _emit_for(self, stmt: ast.For) -> str:
        target = stmt.target
        iter_expr = stmt.iter
        if isinstance(iter_expr, ast.Call):
            name = _call_name(iter_expr)
            if name == "range":
                return self._emit_for_range(stmt)
            if name == "enumerate":
                return self._emit_for_enumerate(stmt)
            if name == "zip":
                return self._emit_for_zip(stmt)
            raise UnsupportedError(
                f"Unsupported for-loop iterator: {name}", node=iter_expr
            )
        if isinstance(iter_expr, ast.Name) and isinstance(target, ast.Name):
            return self._emit_for_list_variable(target.id, iter_expr, stmt)
        raise UnsupportedError(
            "Only range, enumerate, zip, and list variable loops are supported",
            node=stmt,
        )

    def _emit_for_range(self, stmt: ast.For) -> str:
        target = stmt.target
        if not isinstance(target, ast.Name):
            raise UnsupportedError(
                "Only a single loop variable is supported for range(...)", node=stmt
            )
        call = stmt.iter
        assert isinstance(call, ast.Call)
        if len(call.args) == 1:
            stop = self._emit_expr(call.args[0], self.function_type)
            if self.function_type == "f64":
                stop = f"({stop} as i64)"
                range_expr = f"0_i64..{stop}"
            else:
                range_expr = f"0..{stop}"
        elif len(call.args) == 2:
            start = self._emit_expr(call.args[0], self.function_type)
            stop = self._emit_expr(call.args[1], self.function_type)
            if self.function_type == "f64":
                start = f"({start} as i64)"
                stop = f"({stop} as i64)"
            range_expr = f"{start}..{stop}"
        elif len(call.args) == 3:
            start = self._emit_expr(call.args[0], self.function_type)
            stop = self._emit_expr(call.args[1], self.function_type)
            step = self._emit_expr(call.args[2], self.function_type)
            if self.function_type == "f64":
                start = f"({start} as i64)"
                stop = f"({stop} as i64)"
            step_usize = f"(({step} as i64) as usize)"
            range_expr = f"({start}..{stop}).step_by({step_usize})"
        else:
            raise UnsupportedError(
                "range(...) requires 1, 2, or 3 arguments", node=call
            )
        body = self._emit_body(stmt.body)
        return f"for {target.id} in {range_expr} {{\n{body}\n}}"

    def _emit_for_enumerate(self, stmt: ast.For) -> str:
        target = stmt.target
        if not isinstance(target, ast.Tuple) or len(target.elts) != 2:
            raise UnsupportedError(
                "enumerate() requires a two-element tuple target", node=stmt
            )
        idx_name, val_name = target.elts[0].id, target.elts[1].id
        call = stmt.iter
        assert isinstance(call, ast.Call)
        iterable = call.args[0]
        iterable_type = self._type_of(iterable)
        if not iterable_type.startswith("Vec<"):
            raise UnsupportedError(
                "enumerate() is only supported on list/Vec types", node=stmt
            )
        element_type = _element_type(iterable_type)
        iter_rust = self._emit_expr(iterable, iterable_type)
        if element_type in ("i64", "f64", "bool"):
            iterator = f"{iter_rust}.iter().copied().enumerate()"
        else:
            iterator = f"{iter_rust}.iter().enumerate()"
        body = self._emit_body(stmt.body)
        return (
            f"for ({idx_name}, {val_name}) in {iterator} {{\n"
            f"    let {idx_name} = {idx_name} as i64;\n"
            f"{body}\n"
            f"}}"
        )

    def _emit_for_zip(self, stmt: ast.For) -> str:
        target = stmt.target
        call = stmt.iter
        assert isinstance(call, ast.Call)
        if not isinstance(target, ast.Tuple) or len(target.elts) != len(call.args):
            raise UnsupportedError(
                "zip() target must match the number of iterables", node=stmt
            )
        names = [elt.id for elt in target.elts if isinstance(elt, ast.Name)]
        if len(names) != len(target.elts):
            raise UnsupportedError(
                "Only plain names are supported in a zip() target", node=target
            )
        iterators: List[str] = []
        for arg in call.args:
            arg_type = self._type_of(arg)
            if not arg_type.startswith("Vec<"):
                raise UnsupportedError(
                    "zip() is only supported on list/Vec types", node=stmt
                )
            element_type = _element_type(arg_type)
            arg_rust = self._emit_expr(arg, arg_type)
            if element_type in ("i64", "f64", "bool"):
                iterators.append(f"{arg_rust}.iter().copied()")
            else:
                iterators.append(f"{arg_rust}.iter()")
        if len(iterators) > 2:
            raise UnsupportedError(
                "zip() with more than two iterables is not supported", node=stmt
            )
        zip_expr = iterators[0]
        for it in iterators[1:]:
            zip_expr = f"{zip_expr}.zip({it})"
        body = self._emit_body(stmt.body)
        return f"for ({', '.join(names)}) in {zip_expr} {{\n{body}\n}}"

    def _emit_for_list_variable(
        self, target_name: str, iter_expr: ast.Name, stmt: ast.For
    ) -> str:
        iter_type = self._type_of(iter_expr)
        if not iter_type.startswith("Vec<"):
            raise UnsupportedError(
                "for ... in variable only supports list/Vec types", node=stmt
            )
        element_type = _element_type(iter_type)
        iter_rust = self._emit_expr(iter_expr, iter_type)
        if element_type in ("i64", "f64", "bool"):
            iterator = f"{iter_rust}.iter().copied()"
        else:
            iterator = f"{iter_rust}.iter()"
        body = self._emit_body(stmt.body)
        return f"for {target_name} in {iterator} {{\n{body}\n}}"

    def _emit_body(self, stmts: List[ast.stmt]) -> str:
        lines = [self._emit_stmt(s) for s in stmts]
        joined = "\n".join(line for line in lines if line)
        return "\n".join("    " + line for line in joined.splitlines())

    # ------------------------------------------------------------------
    # Expression emission
    # ------------------------------------------------------------------
    def _emit_expr(self, expr: ast.expr, ctx: str) -> str:
        if isinstance(expr, ast.Constant):
            return self._emit_constant(expr, ctx)
        if isinstance(expr, ast.Name):
            return self._coerce(expr.id, self._type_of(expr), ctx)
        if isinstance(expr, ast.BinOp):
            return self._emit_binop(expr, ctx)
        if isinstance(expr, ast.UnaryOp):
            return self._emit_unaryop(expr, ctx)
        if isinstance(expr, ast.Compare):
            return self._emit_compare(expr, ctx)
        if isinstance(expr, ast.BoolOp):
            return self._emit_boolop(expr, ctx)
        if isinstance(expr, ast.Call):
            return self._emit_call(expr, ctx)
        if isinstance(expr, ast.IfExp):
            return self._emit_ifexp(expr, ctx)
        if isinstance(expr, ast.List):
            if ctx and ctx.startswith("Vec<"):
                return self._emit_list_literal(expr, ctx)
            return f"({', '.join(self._emit_expr(e, ctx) for e in expr.elts)})"
        if isinstance(expr, ast.ListComp):
            return self._emit_listcomp(expr, ctx)
        if isinstance(expr, ast.Tuple):
            tuple_parts: List[str] = []
            if ctx and ctx.startswith("(") and ctx.endswith(")"):
                parts = self._tuple_element_types(ctx)
                tuple_parts = [
                    self._emit_expr(e, parts[i] if i < len(parts) else ctx)
                    for i, e in enumerate(expr.elts)
                ]
            else:
                tuple_parts = [self._emit_expr(e, ctx) for e in expr.elts]
            return f"({', '.join(tuple_parts)})"
        if isinstance(expr, ast.Attribute):
            return self._emit_attribute(expr, ctx)
        if isinstance(expr, ast.Subscript):
            return self._emit_subscript(expr, ctx)
        if isinstance(expr, ast.NamedExpr):
            raise UnsupportedError("walrus operator (:=) is not supported", node=expr)
        if isinstance(expr, (ast.Await, ast.Yield, ast.YieldFrom)):
            raise UnsupportedError(
                "async/await and yield expressions are not supported", node=expr
            )
        raise UnsupportedError(
            f"Unsupported expression: {type(expr).__name__}", node=expr
        )

    def _emit_attribute(self, expr: ast.Attribute, ctx: str) -> str:
        if (
            isinstance(expr.value, ast.Name)
            and expr.value.id == "math"
            and expr.attr in MATH_CONSTANTS
        ):
            constant = ast.Constant(
                value=MATH_CONSTANTS[expr.attr],
                lineno=getattr(expr, "lineno", None) or 0,
                col_offset=getattr(expr, "col_offset", None) or 0,
            )
            return self._emit_constant(constant, ctx)

        base_expr = self._emit_expr(expr.value, self._type_of(expr.value))
        field_type = self._field_type(expr.value, expr.attr)
        access = f"{base_expr}.{expr.attr}"
        return self._coerce(access, field_type, ctx)

    def _emit_constant(self, expr: ast.Constant, ctx: str) -> str:
        value = expr.value
        if isinstance(value, bool):
            return "true" if value else "false"
        if isinstance(value, (int, float)):
            if ctx == "f64":
                if isinstance(value, float):
                    return f"{value}_f64"
                return f"{value}_f64"
            if isinstance(value, float):
                raise UnsupportedError(
                    "Float literal in an integer-typed function", node=expr
                )
            return f"{value}_i64"
        if value is None:
            raise UnsupportedError("None is not a numeric value", node=expr)
        raise UnsupportedError(f"Unsupported literal: {value!r}", node=expr)

    def _emit_list_literal(self, expr: ast.List, ctx: str) -> str:
        """Emit a Python list literal as a Rust vec! macro."""
        element_type = _element_type(ctx)
        if element_type == "?":
            element_type = self.function_type
        if not expr.elts:
            return f"Vec::<{element_type}>::new()"
        elements = [self._emit_expr(e, element_type) for e in expr.elts]
        return f"vec![{', '.join(elements)}]"

    def _emit_listcomp(self, expr: ast.ListComp, ctx: str) -> str:
        """Emit a list comprehension as a Rust ``for`` loop block."""
        if len(expr.generators) != 1 or expr.generators[0].ifs:
            raise UnsupportedError(
                "Only simple list comprehensions are supported", node=expr
            )
        gen = expr.generators[0]
        element_type = _element_type(ctx)
        loop_vars, iterator, var_types = self._emit_listcomp_iterator(gen, expr)
        tmp = self._next_tmp()
        old_types: Dict[str, Optional[str]] = {}
        for name, typ in var_types:
            old_types[name] = self.type_env.get(name)
            self.type_env[name] = typ
        body_lines: List[str] = []
        if (
            isinstance(gen.iter, ast.Call)
            and _call_name(gen.iter) == "enumerate"
            and len(loop_vars) >= 2
        ):
            body_lines.append(f"let {loop_vars[0]} = {loop_vars[0]} as i64;")
        body_lines.append(f"{tmp}.push({self._emit_expr(expr.elt, element_type)});")
        body = "\n".join("    " + line for line in body_lines)
        for name, _ in var_types:
            if old_types[name] is not None:
                self.type_env[name] = old_types[name]  # type: ignore[assignment]
            else:
                self.type_env.pop(name, None)
        return (
            f"{{ let mut {tmp} = Vec::<{element_type}>::new(); "
            f"for ({', '.join(loop_vars)}) in {iterator} {{\n{body}\n}} "
            f"{tmp} }}"
        )

    def _emit_listcomp_iterator(
        self, gen: ast.comprehension, expr: ast.ListComp
    ) -> Tuple[List[str], str, List[Tuple[str, str]]]:
        """Return (loop variable names, Rust iterator expression, variable types)."""
        if isinstance(gen.target, ast.Name):
            name = gen.target.id
            if isinstance(gen.iter, ast.Call):
                call = gen.iter
                call_name = _call_name(call)
                if call_name == "range":
                    if len(call.args) == 1:
                        count = self._emit_expr(call.args[0], "i64")
                        return [name], f"0_i64..({count})", [(name, "i64")]
                    if len(call.args) == 2:
                        start = self._emit_expr(call.args[0], "i64")
                        stop = self._emit_expr(call.args[1], "i64")
                        return [name], f"({start})..({stop})", [(name, "i64")]
                    raise UnsupportedError(
                        "List comprehension range(...) with step is not supported",
                        node=expr,
                    )
                if call_name == "enumerate":
                    raise UnsupportedError(
                        "Use a tuple target for enumerate() in list comprehensions",
                        node=expr,
                    )
            if isinstance(gen.iter, ast.Name):
                iter_type = self._type_of(gen.iter)
                element_type = _element_type(iter_type)
                iter_rust = self._emit_expr(gen.iter, iter_type)
                if element_type in ("i64", "f64", "bool"):
                    return (
                        [name],
                        f"{iter_rust}.iter().copied()",
                        [(name, element_type)],
                    )
                return [name], f"{iter_rust}.iter()", [(name, f"&{element_type}")]
        if isinstance(gen.target, ast.Tuple):
            names = [elt.id for elt in gen.target.elts if isinstance(elt, ast.Name)]
            if len(names) != len(gen.target.elts):
                raise UnsupportedError(
                    "Only plain names are supported in a list comprehension target",
                    node=expr,
                )
            if isinstance(gen.iter, ast.Call):
                call = gen.iter
                call_name = _call_name(call)
                if call_name == "enumerate" and len(names) == 2:
                    iterable = call.args[0]
                    iterable_type = self._type_of(iterable)
                    element_type = _element_type(iterable_type)
                    iter_rust = self._emit_expr(iterable, iterable_type)
                    if element_type in ("i64", "f64", "bool"):
                        iterator = f"{iter_rust}.iter().copied().enumerate()"
                    else:
                        iterator = f"{iter_rust}.iter().enumerate()"
                    return (
                        names,
                        iterator,
                        [(names[0], "i64"), (names[1], element_type)],
                    )
                if call_name == "zip":
                    if len(names) != len(call.args) or len(names) > 2:
                        raise UnsupportedError(
                            "zip() in list comprehensions supports two iterables",
                            node=expr,
                        )
                    iterators: List[str] = []
                    var_types: List[Tuple[str, str]] = []
                    for elt, arg in zip(gen.target.elts, call.args):
                        if not isinstance(elt, ast.Name):
                            continue
                        arg_type = self._type_of(arg)
                        element_type = _element_type(arg_type)
                        arg_rust = self._emit_expr(arg, arg_type)
                        if element_type in ("i64", "f64", "bool"):
                            iterators.append(f"{arg_rust}.iter().copied()")
                        else:
                            iterators.append(f"{arg_rust}.iter()")
                        var_types.append((elt.id, element_type))
                    zip_expr = iterators[0]
                    for it in iterators[1:]:
                        zip_expr = f"{zip_expr}.zip({it})"
                    return names, zip_expr, var_types
        raise UnsupportedError(
            "Unsupported list comprehension iterator or target", node=expr
        )

    def _emit_subscript(self, expr: ast.Subscript, ctx: str) -> str:
        base_type = self._type_of(expr.value)
        if not base_type.startswith("Vec<"):
            raise UnsupportedError(
                "Subscript/indexing is only supported on Vec/list types", node=expr
            )
        element_type = _element_type(base_type)
        value_expr = self._emit_expr(expr.value, base_type)

        if isinstance(expr.slice, ast.Slice):
            lower = (
                self._emit_expr(expr.slice.lower, "i64")
                if expr.slice.lower is not None
                else "0"
            )
            if expr.slice.upper is None and expr.slice.lower is None:
                return self._coerce(f"{value_expr}.clone()", base_type, ctx)
            if expr.slice.upper is None:
                return self._coerce(
                    f"{value_expr}[({lower}) as usize..].to_vec()", base_type, ctx
                )
            upper = self._emit_expr(expr.slice.upper, "i64")
            if expr.slice.lower is None:
                return self._coerce(
                    f"{value_expr}[0..({upper}) as usize].to_vec()", base_type, ctx
                )
            return self._coerce(
                f"{value_expr}[({lower}) as usize..({upper}) as usize].to_vec()",
                base_type,
                ctx,
            )

        index_expr = self._emit_expr(expr.slice, "i64")
        access = f"{value_expr}[({index_expr}) as usize]"
        return self._coerce(access, element_type, ctx)

    def _emit_unaryop(self, expr: ast.UnaryOp, ctx: str) -> str:
        if isinstance(expr.op, ast.Invert):
            if self.function_type != "i64":
                raise UnsupportedError(
                    "Bitwise inversion is only supported on integer-typed values",
                    node=expr,
                )
            operand = self._emit_expr(expr.operand, "i64")
            result = f"!({operand})"
            if ctx == "bool":
                return f"({result} != 0)"
            return result

        if isinstance(expr.op, ast.UAdd):
            return self._emit_expr(expr.operand, ctx)
        if isinstance(expr.op, ast.USub):
            operand_type = self._type_of(expr.operand)
            operand = self._emit_expr(expr.operand, operand_type)
            return self._coerce(f"-({operand})", operand_type, ctx)
        if isinstance(expr.op, ast.Not):
            operand = self._emit_expr(expr.operand, "bool")
            return self._coerce(f"!({operand})", "bool", ctx)
        raise UnsupportedError(
            f"Unsupported unary operator: {type(expr.op).__name__}", node=expr
        )

    def _emit_binop(self, expr: ast.BinOp, ctx: str) -> str:
        op = expr.op

        # List replication: [value] * n -> vec![value; n as usize]
        if isinstance(op, ast.Mult) and isinstance(expr.left, ast.List):
            element_type = _element_type(ctx)
            if expr.left.elts:
                element = self._emit_expr(expr.left.elts[0], element_type)
            else:
                element = self._zero_for_type(element_type)
            count = self._emit_expr(expr.right, "i64")
            return f"vec![{element}; ({count}) as usize]"

        left_type = self._type_of(expr.left)
        right_type = self._type_of(expr.right)

        # NumPy-style elementwise vector <op> scalar.
        if _is_vec_type(left_type) and _is_numeric_scalar(right_type):
            return self._emit_elementwise_vec_scalar(
                expr.left, expr.right, op, left_type, ctx
            )
        if _is_numeric_scalar(left_type) and _is_vec_type(right_type):
            return self._emit_elementwise_scalar_vec(
                expr.left, expr.right, op, right_type, ctx
            )

        result_type = self._type_of(expr)
        left = self._emit_expr(expr.left, result_type)
        right = self._emit_expr(expr.right, result_type)

        if isinstance(op, ast.Add):
            return self._coerce(f"({left} + {right})", result_type, ctx)
        if isinstance(op, ast.Sub):
            return self._coerce(f"({left} - {right})", result_type, ctx)
        if isinstance(op, ast.Mult):
            return self._coerce(f"({left} * {right})", result_type, ctx)
        if isinstance(op, ast.Div):
            return self._coerce(f"({left} / {right})", "f64", ctx)
        if isinstance(op, ast.FloorDiv):
            if result_type == "f64":
                return self._coerce(f"(({left}) / ({right})).floor()", result_type, ctx)
            return self._coerce(f"({left}).div_euclid({right})", result_type, ctx)
        if isinstance(op, ast.Mod):
            if result_type == "f64":
                return self._coerce(f"(({left}) % ({right}))", result_type, ctx)
            return self._coerce(f"({left}).rem_euclid({right})", result_type, ctx)
        if isinstance(op, ast.Pow):
            if result_type == "f64":
                return self._coerce(f"({left}).powf({right})", result_type, ctx)
            return self._coerce(f"({left}).pow(({right}) as u32)", result_type, ctx)
        if isinstance(op, (ast.LShift, ast.RShift, ast.BitOr, ast.BitXor, ast.BitAnd)):
            if self.function_type != "i64":
                raise UnsupportedError(
                    "Bitwise operations are only supported on integer-typed values",
                    node=expr,
                )
            left = self._emit_expr(expr.left, "i64")
            right = self._emit_expr(expr.right, "i64")
            op_str = {
                ast.LShift: "<<",
                ast.RShift: ">>",
                ast.BitOr: "|",
                ast.BitXor: "^",
                ast.BitAnd: "&",
            }[type(op)]
            result = f"({left} {op_str} {right})"
            if ctx == "bool":
                return f"({result} != 0)"
            return result

        raise UnsupportedError(
            f"Unsupported binary operator: {type(op).__name__}", node=expr
        )

    def _emit_elementwise_vec_scalar(
        self,
        vec_expr: ast.expr,
        scalar_expr: ast.expr,
        op: ast.operator,
        vec_type: str,
        ctx: str,
    ) -> str:
        element_type = _element_type(vec_type)
        vec_code = self._strip_outer_parens(self._emit_expr(vec_expr, vec_type))
        scalar_code = self._strip_outer_parens(
            self._emit_expr(scalar_expr, element_type)
        )
        closure = self._build_elementwise_closure(op, "x", scalar_code)
        return f"({vec_code}).iter().map(|x| {closure}).collect::<{vec_type}>()"

    def _emit_elementwise_scalar_vec(
        self,
        scalar_expr: ast.expr,
        vec_expr: ast.expr,
        op: ast.operator,
        vec_type: str,
        ctx: str,
    ) -> str:
        element_type = _element_type(vec_type)
        vec_code = self._strip_outer_parens(self._emit_expr(vec_expr, vec_type))
        scalar_code = self._strip_outer_parens(
            self._emit_expr(scalar_expr, element_type)
        )
        closure = self._build_elementwise_closure(
            op, "x", scalar_code, left_scalar=True
        )
        return f"({vec_code}).iter().map(|x| {closure}).collect::<{vec_type}>()"

    def _build_elementwise_closure(
        self, op: ast.operator, var: str, scalar: str, left_scalar: bool = False
    ) -> str:
        op_map = {
            ast.Add: "+",
            ast.Sub: "-",
            ast.Mult: "*",
            ast.Div: "/",
            ast.FloorDiv: "/",
            ast.Mod: "%",
            ast.Pow: ".pow",
        }
        if type(op) not in op_map:
            raise UnsupportedError(
                f"Unsupported elementwise operator: {type(op).__name__}"
            )
        op_str = op_map[type(op)]
        if isinstance(op, ast.Pow):
            if left_scalar:
                return f"({scalar}).powf({var})"
            return f"({var}).powf({scalar})"
        if left_scalar:
            return f"({scalar} {op_str} {var})"
        return f"({var} {op_str} {scalar})"

    def _emit_compare(self, expr: ast.Compare, ctx: str) -> str:
        if len(expr.ops) != 1 or len(expr.comparators) != 1:
            raise UnsupportedError(
                "Only simple binary comparisons are supported", node=expr
            )
        # Comparison operands are always evaluated in the function's numeric
        # type, even when the comparison itself is used as a boolean (e.g. in
        # an `if` or `while` condition).
        numeric_ctx = self.function_type
        left = self._emit_expr(expr.left, numeric_ctx)
        right = self._emit_expr(expr.comparators[0], numeric_ctx)
        op = expr.ops[0]
        op_str = {
            ast.Eq: "==",
            ast.NotEq: "!=",
            ast.Lt: "<",
            ast.LtE: "<=",
            ast.Gt: ">",
            ast.GtE: ">=",
        }.get(type(op))
        if op_str is None:
            raise UnsupportedError(
                f"Unsupported comparison: {type(op).__name__}", node=expr
            )
        return f"({left} {op_str} {right})"

    def _emit_boolop(self, expr: ast.BoolOp, ctx: str) -> str:
        op = " && " if isinstance(expr.op, ast.And) else " || "
        parts = [self._emit_expr(v, ctx) for v in expr.values]
        return f"({op.join(parts)})"

    def _emit_ifexp(self, expr: ast.IfExp, ctx: str) -> str:
        cond = self._emit_expr(expr.test, "bool")
        body = self._emit_expr(expr.body, ctx)
        orelse = self._emit_expr(expr.orelse, ctx)
        return f"if {cond} {{ {body} }} else {{ {orelse} }}"

    def _emit_call(self, expr: ast.Call, ctx: str) -> str:
        name = _call_name(expr)
        base = _call_base(expr)
        args = [
            self._strip_outer_parens(self._emit_expr(a, self.function_type))
            for a in expr.args
        ]

        # Class constructor call or chained constructor method call, e.g.
        # ``Point(x, y).distance()`` -> ``Point::new(x, y).distance()``.
        if (
            isinstance(expr.func, ast.Attribute)
            and isinstance(expr.func.value, ast.Call)
            and _call_name(expr.func.value) in self.class_names
        ):
            class_name = _rust_identifier(_call_name(expr.func.value))
            ctor_args = [
                self._strip_outer_parens(self._emit_expr(a, self.function_type))
                for a in expr.func.value.args
            ]
            method_args = [
                self._strip_outer_parens(self._emit_expr(a, self.function_type))
                for a in expr.args
            ]
            call = f"{class_name}::new({', '.join(ctor_args)})"
            if expr.func.attr:
                rust_method = f"_accel_{_rust_identifier(expr.func.attr)}"
                call = f"{call}.{rust_method}({', '.join(method_args)})"
            return call

        if base is None and name in self.class_names:
            class_name = _rust_identifier(name)
            return f"{class_name}::new({', '.join(args)})"

        if base is None and name == self.func.name:
            if len(args) != len(self.arg_names):
                raise UnsupportedError(
                    f"Recursive call to {name} has wrong number of arguments",
                    node=expr,
                )
            return f"{self.rust_function_name}({', '.join(args)})"

        if base is None and name in {"abs", "round"}:
            if len(args) != 1:
                raise UnsupportedError(
                    f"{name}() takes exactly one argument", node=expr
                )
            arg = args[0]
            if name == "abs":
                return f"({arg}).abs()"
            if ctx == "f64":
                return f"({arg}).round()"
            return f"(({arg} as f64).round() as i64)"

        if base is None and name == "pow":
            if len(args) != 2:
                raise UnsupportedError("pow() takes exactly two arguments", node=expr)
            left, right = args
            if ctx == "f64":
                return f"({left}).powf({right})"
            return f"({left}).pow(({right}) as u32)"

        if base is None and name in {"min", "max"}:
            if not args:
                raise UnsupportedError(
                    f"{name}() requires at least one argument", node=expr
                )
            method = "min" if name == "min" else "max"
            result = f"({args[0]})"
            for a in args[1:]:
                result = f"({result}.{method}({a}))"
            return result

        if base is None and name == "len":
            if len(expr.args) != 1:
                raise UnsupportedError("len() takes exactly one argument", node=expr)
            arg_node = expr.args[0]
            container_type = self._type_of(arg_node)
            if isinstance(arg_node, ast.Subscript):
                container_type = self._type_of(arg_node.value)
            arg_str = self._emit_expr(arg_node, container_type)
            return f"(({arg_str}).len() as i64)"

        # list.extend(other)
        if base is not None and name == "extend":
            if len(expr.args) != 1:
                raise UnsupportedError("extend() takes exactly one argument", node=expr)
            target = expr.func.value
            target_type = self._type_of(target)
            element_type = _element_type(target_type)
            target_str = self._emit_expr(target, target_type)
            arg_type = f"Vec<{element_type}>"
            arg_str = self._emit_expr(expr.args[0], arg_type)
            return f"{target_str}.extend({arg_str}.iter().cloned());"

        if base in ("np", "numpy"):
            return self._emit_numpy_call(name, expr, ctx)

        if base == "math" and name in self.MATH_ATTRS:
            return self._emit_math_call(name, args, ctx)

        if base in self.IO_MODULES or name in self.IO_NAMES:
            raise UnsupportedError("io", node=expr)
        raise UnsupportedError(f"Unsupported call: {name}", node=expr)

    def _emit_math_call(self, name: str, args: List[str], ctx: str) -> str:
        if name == "pow":
            if len(args) != 2:
                raise UnsupportedError("math.pow() takes exactly two arguments")
            left, right = args
            if ctx == "f64":
                return f"({left}).powf({right})"
            return f"(({left} as f64).powf({right} as f64) as i64)"

        if name == "radians":
            arg = args[0]
            if ctx == "f64":
                return f"(({arg}) * std::f64::consts::PI / 180.0)"
            return f"((({arg}) as f64) * std::f64::consts::PI / 180.0) as i64"

        if name == "degrees":
            arg = args[0]
            if ctx == "f64":
                return f"(({arg}) * 180.0 / std::f64::consts::PI)"
            return f"((({arg}) as f64) * 180.0 / std::f64::consts::PI) as i64"

        if len(args) != 1:
            raise UnsupportedError(f"math.{name}() takes exactly one argument")
        arg = args[0]
        rust_method = {
            "sqrt": "sqrt",
            "sin": "sin",
            "cos": "cos",
            "tan": "tan",
            "exp": "exp",
            "log": "ln",
            "log10": "log10",
            "ceil": "ceil",
            "floor": "floor",
            "trunc": "trunc",
        }[name]

        if ctx == "f64":
            return f"({arg}).{rust_method}()"
        return f"(({arg} as f64).{rust_method}() as i64)"

    def _emit_numpy_call(self, name: str, expr: ast.Call, ctx: str) -> str:
        if name == "array" and expr.args:
            return self._emit_expr(expr.args[0], ctx)
        if name in ("zeros", "ones") and expr.args:
            fill = "0.0_f64" if name == "zeros" else "1.0_f64"
            arg = expr.args[0]
            if isinstance(arg, ast.Tuple) and len(arg.elts) == 2:
                rows = self._emit_expr(arg.elts[0], "i64")
                cols = self._emit_expr(arg.elts[1], "i64")
                return f"vec![vec![{fill}; ({cols}) as usize]; ({rows}) as usize]"
            count = self._emit_expr(arg, "i64")
            return f"vec![{fill}; ({count}) as usize]"
        if name == "dot" and len(expr.args) == 2:
            return self._emit_numpy_dot(expr.args[0], expr.args[1], ctx)
        if name == "sum" and expr.args:
            arr = self._strip_outer_parens(self._emit_expr(expr.args[0], ctx))
            arr_type = self._type_of(expr.args[0])
            if arr_type == "Vec<Vec<f64>>":
                return f"({arr}).iter().map(|row| row.iter().sum::<f64>()).sum::<f64>()"
            return f"({arr}).iter().sum::<f64>()"
        raise UnsupportedError(f"Unsupported NumPy call: np.{name}", node=expr)

    def _emit_numpy_dot(self, a: ast.expr, b: ast.expr, ctx: str) -> str:
        a_type = self._type_of(a)
        b_type = self._type_of(b)
        a_expr = self._strip_outer_parens(self._emit_expr(a, a_type))
        b_expr = self._strip_outer_parens(self._emit_expr(b, b_type))
        if a_type == "Vec<f64>" and b_type == "Vec<f64>":
            return (
                f"({a_expr}).iter().zip(({b_expr}).iter())"
                f".map(|(x, y)| x * y).sum::<f64>()"
            )
        if a_type == "Vec<Vec<f64>>" and b_type == "Vec<Vec<f64>>":
            tmp = self._next_tmp()
            return (
                f"{{ "
                f"let mut {tmp} = vec![vec![0.0_f64; ({b_expr})[0].len()]; ({a_expr}).len()]; "
                f"for _i in 0_i64..(({a_expr}).len() as i64) {{ "
                f"for _j in 0_i64..(({b_expr})[0].len() as i64) {{ "
                f"for _k in 0_i64..(({b_expr}).len() as i64) {{ "
                f"{tmp}[_i as usize][_j as usize] += ({a_expr})[_i as usize][_k as usize] * ({b_expr})[_k as usize][_j as usize]; "
                f"}} }} }} {tmp} }}"
            )
        raise UnsupportedError(
            f"np.dot not supported for types {a_type} and {b_type}", node=a
        )


# ---------------------------------------------------------------------------
# Class support
# ---------------------------------------------------------------------------
class ClassGenerator:
    """Generate a Rust PyO3 class (struct + #[pymethods] impl) from a Python class."""

    def __init__(
        self,
        class_node: ast.ClassDef,
        module_name: str,
        traits: Dict[str, Any],
        class_names: Optional[Set[str]] = None,
    ):
        self.class_node = class_node
        self.module_name = module_name
        self.traits = traits
        self.orig_name = class_node.name
        self.class_name = _rust_identifier(class_node.name)
        self.class_names = (class_names or set()) | {class_node.name}
        self._check_slots()
        self.methods: Dict[str, ast.FunctionDef] = {
            node.name: node
            for node in class_node.body
            if isinstance(node, ast.FunctionDef)
        }
        self.fields = self._collect_fields()
        self._used_traits: Set[str] = set()

    def _check_slots(self) -> None:
        """Reject ``__slots__`` early with a clear message."""
        for node in self.class_node.body:
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == "__slots__":
                        raise UnsupportedError(
                            "__slots__ is not supported in PyO3 classes; "
                            "declare class attributes as __init__ assignments instead",
                            node=node,
                        )
            if isinstance(node, ast.AnnAssign):
                if isinstance(node.target, ast.Name) and node.target.id == "__slots__":
                    raise UnsupportedError(
                        "__slots__ is not supported in PyO3 classes",
                        node=node,
                    )

    def shield_traits(self) -> Set[str]:
        return self._used_traits

    def emit(self) -> str:
        init = self.methods.get("__init__")
        if init is None:
            raise UnsupportedError(
                f"Class {self.orig_name!r} must define __init__", node=self.class_node
            )

        method_names = {m for m in self.methods if m != "__init__"}
        field_decls: List[str] = []
        for name, typ in self.fields.items():
            attrs = []
            if f"get_{name}" not in method_names:
                attrs.append("get")
            if f"set_{name}" not in method_names:
                attrs.append("set")
            attr = f"#[pyo3({', '.join(attrs)})]" if attrs else ""
            field_decls.append(
                f"    {attr}\n    pub {name}: {typ},"
                if attr
                else f"    pub {name}: {typ},"
            )
        struct_block = (
            f"#[pyclass]\n"
            f"struct {self.class_name} {{\n"
            f"{chr(10).join(field_decls)}\n"
            f"}}"
        )

        init_arg_types = [
            _annotation_to_rust_type(arg.annotation, self.class_names)
            or self.traits.get("function_type", "i64")
            for arg in init.args.args[1:]
        ]
        init_gen = ClassMethodGenerator(
            init,
            self.module_name,
            self.class_name,
            self.fields,
            self.traits,
            class_names=self.class_names,
            is_new=True,
            init_arg_types=init_arg_types,
        )
        blocks = [init_gen.emit()]
        self._used_traits.update(init_gen.shield_traits())

        for method_name, method in self.methods.items():
            if method_name == "__init__":
                continue
            gen = ClassMethodGenerator(
                method,
                self.module_name,
                self.class_name,
                self.fields,
                self.traits,
                class_names=self.class_names,
                init_arg_types=init_arg_types,
            )
            blocks.append(gen.emit())
            self._used_traits.update(gen.shield_traits())

        impl_body = "\n\n".join(blocks)
        impl_block = (
            f"#[pymethods]\n" f"impl {self.class_name} {{\n" f"{impl_body}\n" f"}}"
        )
        return f"{struct_block}\n\n{impl_block}"

    def _collect_fields(self) -> Dict[str, str]:
        """Map field names to Rust types inferred from __init__ assignments."""
        init = self.methods.get("__init__")
        if init is None:
            return {}
        fields: Dict[str, str] = {}
        function_type = self.traits.get("function_type", "i64")
        for stmt in init.body:
            targets: List[ast.expr] = []
            if isinstance(stmt, ast.Assign):
                targets = list(stmt.targets)
            elif isinstance(stmt, ast.AnnAssign):
                targets = [stmt.target]
            for target in targets:
                if (
                    isinstance(target, ast.Attribute)
                    and isinstance(target.value, ast.Name)
                    and target.value.id == "self"
                ):
                    field = target.attr
                    if isinstance(stmt, ast.AnnAssign):
                        typ = _annotation_to_rust_type(
                            stmt.annotation, self.class_names
                        )
                        if typ is None and stmt.value is not None:
                            typ = _infer_expr_type(
                                stmt.value, function_type, self.class_names
                            )
                        fields[field] = typ or function_type
                    else:
                        typ = _infer_expr_type(
                            stmt.value, function_type, self.class_names
                        )
                        fields[field] = typ or function_type
        return fields


class ClassMethodGenerator(RustGenerator):
    """Generate Rust code for a single Python method inside a #[pymethods] impl."""

    def __init__(
        self,
        func: ast.FunctionDef,
        module_name: str,
        class_name: str,
        fields: Dict[str, str],
        traits: Dict[str, Any],
        class_names: Optional[Set[str]] = None,
        is_new: bool = False,
        init_arg_types: Optional[List[str]] = None,
    ):
        self.class_name = class_name
        self.fields = fields
        self.is_new = is_new
        self.init_arg_types = init_arg_types or []
        self.is_staticmethod = self._has_decorator(func, "staticmethod")
        self.is_classmethod = self._has_decorator(func, "classmethod")
        self.mutates_self = (
            not self.is_staticmethod
            and not self.is_classmethod
            and self._method_mutates_self(func)
        )
        self.field_inits: Dict[str, str] = {}
        super().__init__(func, module_name, traits, class_names=class_names)
        # Methods always have `self`/`cls` as the first parameter unless they are
        # static methods.
        if self.is_staticmethod:
            arg_slice = func.args.args
        elif self.is_classmethod:
            arg_slice = func.args.args[1:]
        else:
            arg_slice = func.args.args[1:]
        self.arg_names = [a.arg for a in arg_slice]
        self.arg_types: List[str] = []
        any_annotation = False
        for arg in arg_slice:
            typ = _annotation_to_rust_type(arg.annotation, self.class_names)
            if typ:
                any_annotation = True
            self.arg_types.append(typ or self.function_type)
        if not any_annotation:
            self.arg_types = [self.function_type] * len(self.arg_names)
        if len(self.arg_types) != len(self.arg_names):
            self.arg_types = [self.function_type] * len(self.arg_names)

        # Update the type environment with the post-processed method argument types.
        for name, typ in zip(self.arg_names, self.arg_types):
            self.type_env[name] = typ

        # `self` is the receiver, not a local variable.
        self.assigned.discard("self")
        self.assigned.discard("cls")
        self.type_env["self"] = "Self"
        if self.is_classmethod:
            self.type_env["cls"] = "&PyType"

        # Convert same-class argument/return types to borrowed Rust forms.
        for i, typ in enumerate(self.arg_types):
            if typ == self.class_name:
                self.arg_types[i] = f"&{typ}"
        if self.return_type == self.class_name:
            self.return_type = "Self"

    @staticmethod
    def _has_decorator(func: ast.FunctionDef, name: str) -> bool:
        return any(
            isinstance(d, ast.Name) and d.id == name for d in func.decorator_list
        )

    def _method_mutates_self(self, func: ast.FunctionDef) -> bool:
        """Return True when the method body assigns to a self.* field."""
        for node in ast.walk(func):
            if isinstance(node, (ast.Assign, ast.AnnAssign)):
                for target in node.targets:
                    if self._is_self_attribute(target):
                        return True
            if isinstance(node, ast.AugAssign) and self._is_self_attribute(node.target):
                return True
        return False

    @staticmethod
    def _is_self_attribute(node: ast.AST) -> bool:
        return (
            isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and node.value.id == "self"
        )

    def shield_traits(self) -> Set[str]:
        return self.used_traits

    def _return_type(self) -> str:
        """Use ``Self`` as the return type for classmethods/constructors."""
        if self.is_classmethod or self.return_type in ("Self", self.class_name):
            return "Self"
        return super()._return_type()

    def _emit_function(self) -> str:
        args = ", ".join(
            f"{name}: {typ}" for name, typ in zip(self.arg_names, self.arg_types)
        )
        return_type = self._return_type()

        if self.is_new:
            header = f"#[new]\nfn new({args}) -> Self {{"
            defaults, body_stmts = self._initializers_and_body()
            for stmt in body_stmts:
                self._emit_stmt(stmt)
            field_lines = [
                f"        {name}: {self.field_inits.get(name, self._zero_for_type(typ))},"
                for name, typ in self.fields.items()
            ]
            body = "\n".join(["    Self {"] + field_lines + ["    }"])
            return f"{header}\n{body}\n}}"

        if self.is_staticmethod:
            header = (
                f'#[staticmethod]\n#[pyo3(name = "{self.orig_name}")]\n'
                f"fn {self.rust_function_name}({args}) -> {return_type} {{"
            )
        elif self.is_classmethod:
            if args:
                sig_args = f"cls: &PyType, {args}"
            else:
                sig_args = "cls: &PyType"
            header = (
                f'#[classmethod]\n#[pyo3(name = "{self.orig_name}")]\n'
                f"fn {self.rust_function_name}({sig_args}) -> {return_type} {{"
            )
        else:
            receiver = "&mut self" if self.mutates_self else "&self"
            if args:
                sig_args = f"{receiver}, {args}"
            else:
                sig_args = receiver
            header = (
                f'#[pyo3(name = "{self.orig_name}")]\n'
                f"fn {self.rust_function_name}({sig_args}) -> {return_type} {{"
            )
        defaults, body_stmts = self._initializers_and_body()
        body_lines = [self._emit_stmt(s) for s in body_stmts]
        if (
            body_stmts
            and isinstance(body_stmts[-1], ast.If)
            and not body_stmts[-1].orelse
            and self._block_returns(body_stmts[-1].body)
        ):
            body_lines.append(f"return {self._zero()};")
        body = "\n".join(defaults + body_lines)
        indented = "\n".join("    " + line for line in body.splitlines())
        return f"{header}\n{indented}\n}}"

    def _emit_expr(self, expr: ast.expr, ctx: str) -> str:
        return super()._emit_expr(expr, ctx)

    def _emit_assign(self, stmt: ast.Assign) -> str:
        if len(stmt.targets) == 1 and self._is_self_attribute(stmt.targets[0]):
            field = stmt.targets[0].attr
            field_type = self.fields.get(field, self.function_type)
            value = self._strip_outer_parens(self._emit_expr(stmt.value, field_type))
            if self.is_new:
                self.field_inits[field] = value
                return ""
            return f"self.{field} = {value};"
        return super()._emit_assign(stmt)

    def _emit_augassign(self, stmt: ast.AugAssign) -> str:
        if self._is_self_attribute(stmt.target):
            field = stmt.target.attr
            fake = ast.BinOp(
                left=ast.Attribute(
                    value=ast.Name(id="self", ctx=ast.Load()),
                    attr=field,
                    ctx=ast.Load(),
                ),
                op=stmt.op,
                right=stmt.value,
            )
            value = self._strip_outer_parens(self._emit_binop(fake, self.function_type))
            return f"self.{field} = {value};"
        return super()._emit_augassign(stmt)

    def _emit_call(self, expr: ast.Call, ctx: str) -> str:
        """Handle constructor calls like ``cls(...)`` or ``Matrix(...)`` inside methods."""
        name = _call_name(expr)
        if (name == "cls" and self.is_classmethod) or (
            name == self.class_name and self.class_name
        ):
            types = self.init_arg_types or [self.function_type] * len(expr.args)
            if len(types) != len(expr.args):
                types = [self.function_type] * len(expr.args)
            args = [
                self._strip_outer_parens(
                    self._emit_expr(
                        a, types[i] if i < len(types) else self.function_type
                    )
                )
                for i, a in enumerate(expr.args)
            ]
            return f"Self::new({', '.join(args)})"
        return super()._emit_call(expr, ctx)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _find_top_level(tree: ast.AST, name: str) -> Tuple[Optional[ast.AST], bool]:
    """Find a top-level class or function by name."""
    for node in getattr(tree, "body", []):
        if isinstance(node, ast.ClassDef) and node.name == name:
            return node, True
        if (
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == name
        ):
            if isinstance(node, ast.AsyncFunctionDef):
                raise UnsupportedError("async/await is not supported", node=node)
            return node, False
    # Fallback: search the whole tree, but prefer classes when both exist.
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == name:
            return node, True
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node, False
    return None, False


def _find_function(tree: ast.AST, name: str) -> Optional[ast.FunctionDef]:
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    return None


def _names_in_target(target: ast.expr) -> List[str]:
    names: List[str] = []
    if isinstance(target, ast.Name):
        names.append(target.id)
    elif isinstance(target, (ast.Tuple, ast.List)):
        for elt in target.elts:
            names.extend(_names_in_target(elt))
    elif isinstance(target, ast.Attribute):
        names.extend(_names_in_target(target.value))
    elif isinstance(target, ast.Subscript):
        names.extend(_names_in_target(target.value))
    return names


def _augassign_op(op: ast.operator) -> str:
    mapping = {
        ast.Add: "+=",
        ast.Sub: "-=",
        ast.Mult: "*=",
        ast.Div: "/=",
        ast.FloorDiv: "/=",
        ast.Mod: "%=",
        ast.Pow: "=",
    }
    if type(op) not in mapping:
        raise UnsupportedError(
            f"Unsupported augmented assignment operator: {type(op).__name__}"
        )
    return mapping[type(op)]


def _elements(container: ast.expr) -> List[ast.expr]:
    if isinstance(container, (ast.Tuple, ast.List)):
        return list(container.elts)
    return []


def _call_name(expr: ast.Call) -> str:
    if isinstance(expr.func, ast.Name):
        return expr.func.id
    if isinstance(expr.func, ast.Attribute):
        return expr.func.attr
    return ""


def _call_base(expr: ast.Call) -> Optional[str]:
    if isinstance(expr.func, ast.Attribute) and isinstance(expr.func.value, ast.Name):
        return expr.func.value.id
    return None


def _shield_imports(traits: Set[str]) -> str:
    if not traits:
        return ""
    lines = ["#[allow(unused_imports)]"]
    for t in sorted(traits):
        lines.append(f"use rug::{t};")
    return "\n".join(lines)


_RUST_KEYWORDS = {
    "as",
    "break",
    "const",
    "continue",
    "crate",
    "else",
    "enum",
    "extern",
    "false",
    "fn",
    "for",
    "if",
    "impl",
    "in",
    "let",
    "loop",
    "match",
    "mod",
    "move",
    "mut",
    "pub",
    "ref",
    "return",
    "self",
    "Self",
    "static",
    "struct",
    "super",
    "trait",
    "true",
    "type",
    "unsafe",
    "use",
    "where",
    "while",
    "dyn",
    "async",
    "await",
    "abstract",
    "become",
    "box",
    "do",
    "final",
    "macro",
    "override",
    "priv",
    "typeof",
    "unsized",
    "virtual",
    "yield",
}


def _annotation_to_rust_type(
    annotation: Optional[ast.expr], class_names: Optional[Set[str]] = None
) -> Optional[str]:
    """Convert a Python type annotation to a Rust type string.

    Supported forms: int, float, bool, list[<T>], List[<T>], and forward
    references to classes (e.g. ``"Matrix"``).
    """
    if annotation is None:
        return None
    class_names = class_names or set()

    if isinstance(annotation, ast.Constant) and isinstance(annotation.value, str):
        name = annotation.value
        if name in class_names:
            return name
        # Scalar aliases may also be wrapped in strings.
        return _SCALAR_TYPE_MAP.get(name)

    if isinstance(annotation, ast.Name):
        name = annotation.id
        if name in class_names:
            return name
        return _SCALAR_TYPE_MAP.get(name)

    if isinstance(annotation, ast.Subscript):
        base_name = ""
        if isinstance(annotation.value, ast.Name):
            base_name = annotation.value.id
        if base_name in ("list", "List"):
            inner = _annotation_to_rust_type(annotation.slice, class_names)
            if inner is None:
                return None
            return f"Vec<{inner}>"
        if base_name in ("tuple", "Tuple"):
            inner = _annotation_to_rust_type(annotation.slice, class_names)
            if inner and inner.startswith("(") and inner.endswith(")"):
                return inner
            # ``tuple[int, float]`` becomes ``(i64, f64)``.
            if isinstance(annotation.slice, ast.Tuple):
                parts = [
                    _annotation_to_rust_type(elt, class_names)
                    for elt in annotation.slice.elts
                ]
                if all(parts):
                    return f"({', '.join(parts)})"
            return None
        if base_name == "ndarray":
            inner = _annotation_to_rust_type(annotation.slice, class_names)
            if inner:
                return f"Vec<{inner}>"
            return "Vec<f64>"

    if isinstance(annotation, ast.Attribute):
        if isinstance(annotation.value, ast.Name) and annotation.value.id in (
            "np",
            "numpy",
        ):
            if annotation.attr == "ndarray":
                return "Vec<f64>"

    if isinstance(annotation, ast.Name) and annotation.id == "ndarray":
        return "Vec<f64>"

    return None


_SCALAR_TYPE_MAP = {
    "int": "i64",
    "float": "f64",
    "bool": "bool",
    "None": "()",
}


def _infer_expr_type(
    expr: ast.expr, function_type: str = "i64", class_names: Optional[Set[str]] = None
) -> Optional[str]:
    """Infer a Rust type from an expression used for an unannotated field."""
    class_names = class_names or set()
    if isinstance(expr, ast.Constant):
        if isinstance(expr.value, bool):
            return "bool"
        if isinstance(expr.value, int):
            return "i64"
        if isinstance(expr.value, float):
            return "f64"
    if isinstance(expr, ast.List):
        if not expr.elts:
            return f"Vec<{function_type}>"
        inner = _infer_expr_type(expr.elts[0], function_type, class_names)
        return f"Vec<{inner or function_type}>"
    if isinstance(expr, ast.ListComp):
        inner = _infer_expr_type(expr.elt, function_type, class_names)
        return f"Vec<{inner or function_type}>"
    if isinstance(expr, ast.BinOp) and isinstance(expr.op, ast.Mult):
        if isinstance(expr.left, ast.List):
            return _infer_expr_type(expr.left, function_type, class_names)
    if isinstance(expr, ast.Call):
        name = _call_name(expr)
        if name in class_names:
            return name
    return None


def _element_type(rust_type: str) -> str:
    """Return the element type of a Vec, or the type itself if not a Vec."""
    if rust_type.startswith("Vec<") and rust_type.endswith(">"):
        depth = 0
        for i in range(4, len(rust_type) - 1):
            ch = rust_type[i]
            if ch == "<":
                depth += 1
            elif ch == ">":
                depth -= 1
            elif ch == "," and depth == 0:
                # Not expected for Vec<T>, but keep parser robust.
                return rust_type[4:i].strip()
        return rust_type[4:-1]
    return rust_type


def _vec_depth(rust_type: str) -> int:
    depth = 0
    while rust_type.startswith("Vec<") and rust_type.endswith(">"):
        depth += 1
        rust_type = _element_type(rust_type)
    return depth


def _is_numeric_scalar(rust_type: str) -> bool:
    return rust_type in ("i64", "f64")


def _is_vec_type(rust_type: str) -> bool:
    return rust_type.startswith("Vec<")


def _is_class_type(rust_type: str, class_names: Set[str]) -> bool:
    refs = {"Self", "&Self"}
    refs.update(f"&{c}" for c in class_names)
    return rust_type in class_names or rust_type in refs


def _rust_identifier(name: str) -> str:
    sanitized = re.sub(r"[^0-9a-zA-Z_]", "_", name)
    if not sanitized:
        sanitized = "module"
    if sanitized[0].isdigit() or sanitized in _RUST_KEYWORDS:
        sanitized = "a_" + sanitized
    return sanitized


# ---------------------------------------------------------------------------
# Package scaffolding helpers
# ---------------------------------------------------------------------------
def find_project_root(start: Path) -> Path:
    """Locate the project root by searching for common markers.

    Falls back to the current working directory when no marker is found, but
    if ``start`` is outside the working directory the directory containing
    ``start`` is used so the source file is always inside the returned root.
    """
    start = Path(start).resolve()
    directory = start if start.is_dir() else start.parent
    for parent in [directory] + list(directory.parents):
        if (
            (parent / ".git").is_dir()
            or (parent / "pyproject.toml").is_file()
            or (parent / ".aero-forge-root").is_file()
        ):
            return parent

    cwd = Path.cwd().resolve()
    if cwd == directory or cwd in directory.parents:
        return cwd
    return directory


def ensure_init_files(target: Path, project_root: Optional[Path] = None) -> None:
    """Create ``__init__.py`` files from ``target``'s directory up to ``project_root``."""
    target = Path(target).resolve()
    if project_root is None:
        project_root = find_project_root(target.parent)
    project_root = project_root.resolve()

    current = target.parent
    while current != project_root and project_root in current.parents:
        init_file = current / "__init__.py"
        if not init_file.exists():
            init_file.write_text("# Created by aero-forge\n", encoding="utf-8")
            logger.info("Created %s", init_file)
        current = current.parent


def ensure_sys_path(root: Optional[Path] = None) -> None:
    """Insert ``root`` at the front of ``sys.path`` if it is not already present."""
    root = (root or find_project_root(Path.cwd())).resolve()
    str_root = str(root)
    if str_root not in sys.path:
        sys.path.insert(0, str_root)
        logger.info("Added %s to sys.path", str_root)
