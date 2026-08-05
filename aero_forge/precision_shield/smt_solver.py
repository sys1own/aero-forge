"""Neuro-Symbolic SMT constraint engine for AST sketch verification."""

from __future__ import annotations

import ast
from typing import Any, Dict, List, Optional, Tuple

import z3


class SMTASTEngine:
    """Z3-backed solver for typed AST sketches with cross-language holes.

    Combines type, FFI layout, and import-visibility constraints into a single
    first-order formula and returns concrete hole bindings when satisfiable.
    """

    def __init__(self):
        self._ctx = z3.Context()
        self.solver = z3.Solver(ctx=self._ctx)
        self.solver.set("unsat_core", True)
        self._track_counter = 0
        self.TypeSort, (
            self.RustType,
            self.CppType,
            self.PyType,
            self.UnknownType,
        ) = z3.EnumSort("TypeSort", ["RustType", "CppType", "PyType", "UnknownType"], ctx=self._ctx)

        self.NativeTypeSort, (
            self.I64,
            self.F64,
            self.Usize,
            self.Bool,
            self.String,
            self.VecI64,
            self.VecF64,
            self.VecUsize,
            self.Map,
            self.Set,
            self.UnknownNative,
        ) = z3.EnumSort(
            "NativeType",
            ["i64", "f64", "usize", "bool", "string", "vec_i64", "vec_f64", "vec_usize", "map", "set", "unknown"],
            ctx=self._ctx,
        )

    def _add(self, expr: z3.BoolRef, *, name: str | None = None) -> None:
        """Add a tracked assertion so unsat cores can be reported.

        The tracker symbol is always made unique to avoid collisions when the
        same constraint template is instantiated multiple times.
        """
        self._track_counter += 1
        tracker_name = f"track_{name or 'assert'}_{self._track_counter}"
        tracker = z3.Bool(tracker_name, ctx=self._ctx)
        self.solver.assert_and_track(expr, tracker)

    def solve_ast_sketch_holes(
        self, holes: List[str], constraints: List[Dict[str, Any]]
    ) -> Dict[str, str]:
        """Solve for hole variables under the combined type/FFI/import constraints.

        Args:
            holes: Names of typed holes to solve for.
            constraints: List of constraint dictionaries. Supported keys:
                - ``source_hole`` / ``target_hole``: type equality/subtyping.
                - ``target_language``: ``'rust' | 'cpp' | 'python'``.
                - ``ffi_layout``: dict with ``struct``, ``field``,
                  ``rust_offset``, ``cpp_offset``, ``rust_align``, ``cpp_align``.
                - ``imports``: dict with ``module`` and ``symbols`` (visible=True).

        Returns:
            A mapping from hole names to solved type strings.

        Raises:
            ValueError: If the constraints are unsatisfiable, including the
                unsat core when available.
        """
        self.solver.reset()

        z3_holes: Dict[str, z3.ExprRef] = {
            h: z3.Const(h, self.TypeSort) for h in holes
        }

        # No hole may remain UnknownType.
        for h_var in z3_holes.values():
            self._add(h_var != self.UnknownType, name="no_unknown_type")

        # Keep a cache of FFI/layout integer variables so repeated constraints
        # on the same (struct, field) share the same symbol and can conflict.
        layout_vars: Dict[Tuple[str, str, str], z3.ArithRef] = {}

        def _layout_var(kind: str, struct: str, field: str) -> z3.ArithRef:
            key = (kind, struct, field)
            if key not in layout_vars:
                layout_vars[key] = z3.Int(f"{kind}__{struct}__{field}", ctx=self._ctx)
            return layout_vars[key]

        for c in constraints:
            source = c.get("source_hole")
            target = c.get("target_hole")
            target_lang = c.get("target_language")
            ffi_layout = c.get("ffi_layout")
            imports = c.get("imports")

            source_var = z3_holes.get(source) if source else None
            target_var = z3_holes.get(target) if target else None

            if source_var is not None and target_lang is not None:
                if target_lang == "rust":
                    self._add(source_var == self.RustType, name=f"{source}_is_rust")
                elif target_lang == "cpp":
                    self._add(source_var == self.CppType, name=f"{source}_is_cpp")
                elif target_lang == "python":
                    self._add(source_var == self.PyType, name=f"{source}_is_python")

            if source_var is not None and target_var is not None:
                self._add(
                    source_var == target_var,
                    name=f"{source}_eq_{target}",
                )

            if ffi_layout is not None:
                struct = ffi_layout.get("struct", "_")
                field = ffi_layout.get("field", "_")
                rust_off = ffi_layout.get("rust_offset")
                cpp_off = ffi_layout.get("cpp_offset")
                rust_align = ffi_layout.get("rust_align")
                cpp_align = ffi_layout.get("cpp_align")

                if rust_off is not None and cpp_off is not None:
                    ro = _layout_var("rust_offset", struct, field)
                    co = _layout_var("cpp_offset", struct, field)
                    self._add(ro == rust_off, name=f"offset_rust_{struct}_{field}")
                    self._add(co == cpp_off, name=f"offset_cpp_{struct}_{field}")
                    self._add(ro == co, name=f"offset_eq_{struct}_{field}")

                if rust_align is not None and cpp_align is not None:
                    ra = _layout_var("rust_align", struct, field)
                    ca = _layout_var("cpp_align", struct, field)
                    self._add(ra == rust_align, name=f"align_rust_{struct}_{field}")
                    self._add(ca == cpp_align, name=f"align_cpp_{struct}_{field}")
                    self._add(ra == ca, name=f"align_eq_{struct}_{field}")

            if imports is not None:
                module = imports.get("module", "_")
                visible = bool(imports.get("visible", True))
                for symbol in imports.get("symbols", []):
                    sym_var = z3.Bool(f"visible__{module}__{symbol}", ctx=self._ctx)
                    self._add(sym_var == visible, name=f"import_{module}_{symbol}")

        result = self.solver.check()

        if result == z3.sat:
            model = self.solver.model()
            return {name: str(model[var]) for name, var in z3_holes.items()}

        core = self.solver.unsat_core()
        core_str = ", ".join(str(c) for c in core) if core else "unknown"
        raise ValueError(
            f"SMT Constraint Unsatisfiable: Invalid cross-language bindings. "
            f"Unsat core: [{core_str}]"
        )

    def infer_native_types(
        self,
        source: str,
        function_name: Optional[str] = None,
        target_language: str = "rust",
    ) -> Dict[str, str]:
        """Infer native types for dynamic Python variables using Z3.

        Collects SMT constraints from a function body: literal types, arithmetic
        and comparison operand equalities, subscript/container element relations,
        and conversion builtins.  Returns a mapping from variable name to a native
        type string such as ``"f64"`` or ``"Vec<f64>"``.
        """
        self.solver.reset()
        self._track_counter = 0

        tree = ast.parse(source)
        funcs = [
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef)
        ]
        if function_name:
            funcs = [f for f in funcs if f.name == function_name]
        if not funcs:
            return {}
        func = funcs[0]

        # Variables bound by tuple/list unpack should be inferred from the
        # unpack source (e.g., a tuple-returning call) rather than coerced to
        # an SMT-wide promoted type like ``f64``.
        tuple_unpack_names: set = set()
        for node in ast.walk(func):
            if isinstance(node, (ast.Assign, ast.AnnAssign)):
                for target in getattr(node, "targets", [getattr(node, "target", None)]):
                    if isinstance(target, (ast.Tuple, ast.List)):
                        for elt in ast.walk(target):
                            if isinstance(elt, ast.Name):
                                tuple_unpack_names.add(elt.id)
            if isinstance(node, ast.For):
                if isinstance(node.target, (ast.Tuple, ast.List)):
                    for elt in ast.walk(node.target):
                        if isinstance(elt, ast.Name):
                            tuple_unpack_names.add(elt.id)

        vars: Dict[str, z3.ExprRef] = {}
        fresh_counter = [0]

        def _fresh(prefix: str = "t") -> z3.ExprRef:
            fresh_counter[0] += 1
            return z3.Const(f"{prefix}_{fresh_counter[0]}", self.NativeTypeSort)

        def _var(name: str) -> z3.ExprRef:
            if name not in vars:
                v = z3.Const(f"ntype_{name}", self.NativeTypeSort)
                vars[name] = v
                self._add(v != self.UnknownNative, name=f"{name}_not_unknown")
            return vars[name]

        # Predicate helpers over the native type enum.
        def _is_scalar_numeric(v: z3.ExprRef) -> z3.BoolRef:
            return z3.Or(v == self.I64, v == self.F64, v == self.Usize)

        def _is_int(v: z3.ExprRef) -> z3.BoolRef:
            return z3.Or(v == self.I64, v == self.Usize)

        def _is_float(v: z3.ExprRef) -> z3.BoolRef:
            return v == self.F64

        def _is_vec(v: z3.ExprRef) -> z3.BoolRef:
            return z3.Or(v == self.VecI64, v == self.VecF64, v == self.VecUsize)

        def _is_map(v: z3.ExprRef) -> z3.BoolRef:
            return v == self.Map

        def _is_set(v: z3.ExprRef) -> z3.BoolRef:
            return v == self.Set

        def _element_constraint(container: z3.ExprRef, element: z3.ExprRef) -> z3.BoolRef:
            return z3.Or(
                z3.And(container == self.VecI64, element == self.I64),
                z3.And(container == self.VecF64, element == self.F64),
                z3.And(container == self.VecUsize, element == self.Usize),
            )

        # Type constants for variable-free expressions (used as uninterpreted seeds).
        def _annotation_to_native(ann: Optional[ast.expr]) -> Optional[z3.ExprRef]:
            if ann is None:
                return None
            text = ast.unparse(ann)
            mapping = {
                "int": self.I64,
                "float": self.F64,
                "bool": self.Bool,
                "str": self.String,
                "list": self.VecI64,
                "list[int]": self.VecI64,
                "list[float]": self.VecF64,
                "list[bool]": self.VecI64,
                "dict": self.Map,
                "set": self.Set,
            }
            return mapping.get(text.strip())

        # Parameter annotations seed the solver.
        for arg in func.args.args:
            ann = _annotation_to_native(arg.annotation)
            if ann is not None:
                self._add(_var(arg.arg) == ann, name=f"param_{arg.arg}")

        ret_ann = _annotation_to_native(func.returns)
        if ret_ann is not None:
            self._add(_var("__return__") == ret_ann, name="return_annotation")
        else:
            _var("__return__")

        def _declare_store(node: ast.AST) -> None:
            for child in ast.walk(node):
                if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Store):
                    _var(child.id)
                if isinstance(child, ast.arg):
                    _var(child.arg)
                if isinstance(child, (ast.For, ast.AsyncFor)):
                    for t in ast.walk(child.target):
                        if isinstance(t, ast.Name):
                            _var(t.id)

        _declare_store(func)

        def _collect_expr(expr: Optional[ast.expr]) -> z3.ExprRef:
            """Return a Z3 expression representing the type of ``expr``."""
            if expr is None:
                return self.UnknownNative

            if isinstance(expr, ast.Constant):
                if isinstance(expr.value, bool):
                    return self.Bool
                if isinstance(expr.value, int):
                    return self.I64
                if isinstance(expr.value, float):
                    return self.F64
                if isinstance(expr.value, str):
                    return self.String
                if expr.value is None:
                    return self.UnknownNative
                return self.UnknownNative

            if isinstance(expr, ast.Name):
                if expr.id in ("True", "False"):
                    return self.Bool
                if expr.id == "__return__":
                    return _var("__return__")
                return _var(expr.id)

            if isinstance(expr, ast.UnaryOp):
                operand_t = _collect_expr(expr.operand)
                if isinstance(expr.op, ast.Not):
                    self._add(operand_t == self.Bool, name="not_bool")
                    return self.Bool
                self._add(_is_scalar_numeric(operand_t), name="unary_numeric")
                return operand_t

            if isinstance(expr, ast.BinOp):
                left_t = _collect_expr(expr.left)
                right_t = _collect_expr(expr.right)
                result_t = _fresh()

                if isinstance(expr.op, ast.Div):
                    self._add(left_t == self.F64, name="div_left_f64")
                    self._add(right_t == self.F64, name="div_right_f64")
                    self._add(result_t == self.F64, name="div_result_f64")
                    return result_t

                # Prefer scalar arithmetic unless an operand is syntactically a
                # container (list/set/dict/subscript), which keeps unannotated
                # arguments like ``a * b`` from being solved as ``Vec<i64>``.
                def _may_be_vector(e: ast.expr) -> bool:
                    return isinstance(
                        e, (ast.List, ast.ListComp, ast.Set, ast.Dict, ast.Subscript)
                    )

                both_i64 = z3.And(left_t == self.I64, right_t == self.I64, result_t == self.I64)
                both_usize = z3.And(
                    left_t == self.Usize, right_t == self.Usize, result_t == self.Usize
                )
                both_f64 = z3.And(left_t == self.F64, right_t == self.F64, result_t == self.F64)
                mixed_int = z3.And(
                    _is_int(left_t),
                    _is_int(right_t),
                    z3.Or(left_t == self.I64, right_t == self.I64),
                    result_t == self.I64,
                )
                mixed_float = z3.And(
                    _is_scalar_numeric(left_t),
                    _is_scalar_numeric(right_t),
                    z3.Or(left_t == self.F64, right_t == self.F64),
                    result_t == self.F64,
                )
                scalar_case = z3.Or(both_i64, both_usize, both_f64, mixed_int, mixed_float)

                if _may_be_vector(expr.left) or _may_be_vector(expr.right):
                    vec_case = z3.Or(
                        z3.And(
                            _is_vec(left_t),
                            left_t == right_t,
                            result_t == left_t,
                        ),
                        # elementwise vector <op> scalar (e.g. ``[1.0] * n``)
                        z3.And(
                            _is_vec(left_t),
                            _is_scalar_numeric(right_t),
                            _element_constraint(left_t, right_t),
                            result_t == left_t,
                        ),
                        z3.And(
                            _is_scalar_numeric(left_t),
                            _is_vec(right_t),
                            _element_constraint(right_t, left_t),
                            result_t == right_t,
                        ),
                    )
                    self._add(z3.Or(vec_case, scalar_case), name="binop_vec_or_scalar")
                else:
                    self._add(scalar_case, name="binop_scalar")
                return result_t

            if isinstance(expr, ast.BoolOp):
                for value in expr.values:
                    self._add(_collect_expr(value) == self.Bool, name="boolop")
                return self.Bool

            if isinstance(expr, ast.Compare):
                left_t = _collect_expr(expr.left)
                for op, comp in zip(expr.ops, expr.comparators):
                    comp_t = _collect_expr(comp)
                    if isinstance(op, ast.In):
                        self._add(
                            z3.Or(
                                _element_constraint(comp_t, left_t),
                                z3.And(comp_t == self.Map, left_t == self.String),
                                z3.And(comp_t == self.Set, left_t == self.I64),
                            ),
                            name="in_membership",
                        )
                    else:
                        self._add(left_t == comp_t, name="compare_equality")
                return self.Bool

            if isinstance(expr, ast.IfExp):
                body_t = _collect_expr(expr.body)
                orelse_t = _collect_expr(expr.orelse)
                self._add(expr_t := _fresh() == body_t, name="ifexp_body")
                self._add(body_t == orelse_t, name="ifexp_branches")
                return body_t

            if isinstance(expr, ast.Subscript):
                base_t = _collect_expr(expr.value)
                result_t = _fresh()
                self._add(
                    z3.Or(
                        _element_constraint(base_t, result_t),
                        z3.And(base_t == self.Map, result_t == self.String),
                    ),
                    name="subscript",
                )
                return result_t

            if isinstance(expr, ast.List):
                if not expr.elts:
                    return self.VecI64
                element_t = _collect_expr(expr.elts[0])
                result_t = _fresh()
                self._add(
                    z3.Or(
                        z3.And(element_t == self.I64, result_t == self.VecI64),
                        z3.And(element_t == self.F64, result_t == self.VecF64),
                        z3.And(element_t == self.Usize, result_t == self.VecUsize),
                    ),
                    name="list_literal",
                )
                return result_t

            if isinstance(expr, ast.Dict):
                return self.Map

            if isinstance(expr, ast.Set):
                return self.Set

            if isinstance(expr, ast.ListComp):
                element_t = _collect_expr(expr.elt)
                result_t = _fresh()
                self._add(
                    z3.Or(
                        z3.And(element_t == self.I64, result_t == self.VecI64),
                        z3.And(element_t == self.F64, result_t == self.VecF64),
                        z3.And(element_t == self.Usize, result_t == self.VecUsize),
                    ),
                    name="listcomp",
                )
                return result_t

            if isinstance(expr, ast.Tuple):
                return self.UnknownNative

            if isinstance(expr, ast.Call):
                name = ""
                base = None
                if isinstance(expr.func, ast.Name):
                    name = expr.func.id
                elif isinstance(expr.func, ast.Attribute) and isinstance(expr.func.value, ast.Name):
                    name = expr.func.attr
                    base = expr.func.value.id

                if name == "range":
                    for a in expr.args:
                        self._add(_is_int(_collect_expr(a)), name="range_arg")
                    return self.VecUsize
                if name == "len":
                    return self.I64
                if name in ("int", "__int__"):
                    for a in expr.args:
                        self._add(_is_scalar_numeric(_collect_expr(a)), name="int_arg")
                    return self.I64
                if name in ("float", "__float__"):
                    for a in expr.args:
                        self._add(_is_scalar_numeric(_collect_expr(a)), name="float_arg")
                    return self.F64
                if name == "bool":
                    return self.Bool
                if name == "str":
                    return self.String
                if name == "list" and expr.args:
                    arg_t = _collect_expr(expr.args[0])
                    result_t = _fresh()
                    self._add(
                        z3.Or(
                            z3.And(arg_t == self.VecI64, result_t == self.VecI64),
                            z3.And(arg_t == self.VecF64, result_t == self.VecF64),
                            z3.And(arg_t == self.VecUsize, result_t == self.VecUsize),
                            z3.And(_is_scalar_numeric(arg_t), result_t == self.VecI64),
                        ),
                        name="list_cast",
                    )
                    return result_t
                if name == "sorted" and expr.args:
                    return _collect_expr(expr.args[0])
                if base == "math":
                    for a in expr.args:
                        self._add(_is_scalar_numeric(_collect_expr(a)), name="math_arg")
                    return self.F64
                if name == func.name:
                    return _var("__return__")
                return _fresh()

            return _fresh()

        def _collect_stmt(stmt: ast.stmt) -> None:
            if isinstance(stmt, (ast.Assign, ast.AnnAssign)):
                if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
                    ann = _annotation_to_native(stmt.annotation)
                    if ann is not None:
                        self._add(_var(stmt.target.id) == ann, name="ann_assign")
                    value_t = _collect_expr(stmt.value)
                    self._add(_var(stmt.target.id) == value_t, name="ann_assign_value")
                    return

                value_t = _collect_expr(stmt.value)
                for target in stmt.targets if isinstance(stmt, ast.Assign) else [stmt.target]:
                    if isinstance(target, ast.Name):
                        self._add(_var(target.id) == value_t, name="assign")
                    elif isinstance(target, (ast.Tuple, ast.List)):
                        # Tuple/list unpacking is not modelled at component level because
                        # the SMT solver cannot see inside an arbitrary call return type.
                        pass
            elif isinstance(stmt, ast.AugAssign):
                target_t = _collect_expr(stmt.target)
                value_t = _collect_expr(stmt.value)
                binop_t = _collect_expr(ast.BinOp(left=stmt.target, op=stmt.op, right=stmt.value))
                self._add(target_t == binop_t, name="augassign")
            elif isinstance(stmt, ast.Return):
                if stmt.value is not None:
                    ret_t = _collect_expr(stmt.value)
                    self._add(_var("__return__") == ret_t, name="return")
            elif isinstance(stmt, ast.For):
                iter_t = _collect_expr(stmt.iter)
                target_names = [
                    t.id for t in ast.walk(stmt.target) if isinstance(t, ast.Name)
                ]
                if isinstance(stmt.iter, ast.Call) and isinstance(stmt.iter.func, ast.Name) and stmt.iter.func.id == "range":
                    for n in target_names:
                        self._add(_var(n) == self.Usize, name="for_range_target")
                else:
                    for n in target_names:
                        self._add(_element_constraint(iter_t, _var(n)), name="for_iter")
                for body_stmt in stmt.body:
                    _collect_stmt(body_stmt)
            elif isinstance(stmt, (ast.If, ast.While)):
                if stmt.test is not None:
                    _collect_expr(stmt.test)
                for body_stmt in stmt.body:
                    _collect_stmt(body_stmt)
                for orelse_stmt in getattr(stmt, "orelse", []):
                    _collect_stmt(orelse_stmt)
            elif isinstance(stmt, ast.With):
                for body_stmt in stmt.body:
                    _collect_stmt(body_stmt)
                for orelse_stmt in getattr(stmt, "orelse", []):
                    _collect_stmt(orelse_stmt)

        for body_stmt in func.body:
            _collect_stmt(body_stmt)

        result = self.solver.check()
        if result != z3.sat:
            return {}

        model = self.solver.model()

        def _model_to_rust(v: z3.ExprRef) -> str:
            val = model[v]
            val_str = str(val)
            mapping = {
                "i64": "i64",
                "f64": "f64",
                "usize": "usize",
                "bool": "bool",
                "string": "String",
                "vec_i64": "Vec<i64>",
                "vec_f64": "Vec<f64>",
                "vec_usize": "Vec<usize>",
                "map": "BTreeMap<String, String>",
                "set": "HashSet<i64>",
            }
            return mapping.get(val_str, val_str)

        return {
            name: _model_to_rust(var)
            for name, var in vars.items()
            if str(model[var]) != "unknown" and name not in tuple_unpack_names
        }
