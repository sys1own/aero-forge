"""Neuro-Symbolic SMT constraint engine for AST sketch verification."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

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
