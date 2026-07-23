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
        function_blocks: List[str] = []
        module_init_lines: List[str] = []
        all_traits: Set[str] = set()

        for name in function_names:
            func = _find_function(tree, name)
            if func is None:
                raise UnsupportedError(f"Function {name!r} not found in source")
            traits = traits_by_name.get(name, {}) or {}
            generator = RustGenerator(func, module_name, traits)
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

        arg_names = [a.arg for a in func.args.args]
        arg_types = traits.get("arg_types") or [self.function_type] * len(arg_names)
        if len(arg_types) != len(arg_names):
            arg_types = [self.function_type] * len(arg_names)
        self.arg_names = arg_names
        self.arg_types = arg_types

        self.assigned = self._collect_assigned()
        self._tmp_counter = 0
        self.used_traits: Set[str] = set()

    def shield_traits(self) -> Set[str]:
        return self.used_traits

    def emit(self) -> str:
        return self._emit_function()

    # ------------------------------------------------------------------
    # Collection helpers
    # ------------------------------------------------------------------
    def _collect_assigned(self) -> set[str]:
        names: set[str] = set()
        for node in ast.walk(self.func):
            if isinstance(node, (ast.Assign, ast.AnnAssign)):
                for target in node.targets:
                    names.update(_names_in_target(target))
            elif isinstance(node, ast.AugAssign) and isinstance(node.target, ast.Name):
                names.add(node.target.id)
        return names

    @staticmethod
    def _name_in_expr(expr: ast.expr, name: str) -> bool:
        for node in ast.walk(expr):
            if isinstance(node, ast.Name) and node.id == name:
                return True
        return False

    def _is_mutable(self, name: str) -> bool:
        """Return True if ``name`` is assigned more than once or inside a loop."""
        count = self._count_targets_in_body(name, self.func.body, in_loop=False)
        if name in self.arg_names:
            # The parameter is the first binding; any target assignment is a
            # reassignment, so the local shadow needs to be mutable.
            return count > 0
        return count > 1

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
        if isinstance(stmt, (ast.Assign, ast.AnnAssign)):
            for target in stmt.targets:
                if name in _names_in_target(target):
                    # An assignment inside a loop may execute multiple times.
                    return 2 if in_loop else 1
            return 0
        if isinstance(stmt, ast.AugAssign) and isinstance(stmt.target, ast.Name):
            if stmt.target.id == name:
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
            if (
                isinstance(stmt, ast.Assign)
                and len(stmt.targets) == 1
                and isinstance(stmt.targets[0], ast.Name)
            ):
                name = stmt.targets[0].id
                if (
                    name in self.assigned
                    and name not in self.arg_names
                    and not self._name_in_expr(stmt.value, name)
                    and self._rhs_uses_only(stmt.value, declared)
                ):
                    mutable = self._is_mutable(name)
                    mut = "mut " if mutable else ""
                    value = self._strip_outer_parens(
                        self._emit_expr(stmt.value, self.function_type)
                    )
                    defaults.append(f"let {mut}{name} = {value};")
                    declared.add(name)
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
        return "0.0_f64" if self.function_type == "f64" else "0_i64"

    def _return_type(self) -> str:
        """Derive the Rust return type from the function's return statements."""
        sizes: set[int] = set()
        for node in ast.walk(self.func):
            if isinstance(node, ast.Return) and node.value is not None:
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
        return f"({', '.join([self.return_type] * sizes.pop())})"

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
        body = "\n".join(defaults + body_lines)
        indented = "\n".join("    " + line for line in body.splitlines())
        return f"{header}\n{indented}\n}}"

    def _emit_stmt(self, stmt: ast.stmt) -> str:
        if isinstance(stmt, ast.Return):
            if stmt.value is None:
                return "return;"
            if isinstance(stmt.value, (ast.Tuple, ast.List)):
                value = self._emit_expr(stmt.value, self.return_type)
            else:
                value = self._strip_outer_parens(
                    self._emit_expr(stmt.value, self.return_type)
                )
            return f"return {value};"
        if isinstance(stmt, ast.Assign):
            return self._emit_assign(stmt)
        if isinstance(stmt, ast.AugAssign):
            return self._emit_augassign(stmt)
        if isinstance(stmt, ast.If):
            return self._emit_if(stmt)
        if isinstance(stmt, ast.While):
            return self._emit_while(stmt)
        if isinstance(stmt, ast.For):
            return self._emit_for(stmt)
        if isinstance(stmt, ast.Pass):
            return ""
        if isinstance(stmt, ast.Expr):
            # Validate the expression so I/O and unsupported calls cannot
            # slip through as ignored statements.
            self._emit_expr(stmt.value, self.function_type)
            return ""
        if isinstance(stmt, (ast.With, ast.AsyncWith)):
            raise UnsupportedError("io", node=stmt)
        raise UnsupportedError(
            f"Unsupported statement: {type(stmt).__name__}", node=stmt
        )

    def _emit_assign(self, stmt: ast.Assign) -> str:
        if len(stmt.targets) == 1:
            target = stmt.targets[0]
            if isinstance(target, ast.Name):
                name = target.id
                value = self._strip_outer_parens(
                    self._emit_expr(stmt.value, self.function_type)
                )
                return f"{name} = {value};"
            if isinstance(target, (ast.Tuple, ast.List)):
                return self._emit_tuple_unpack(target, stmt.value)

        raise UnsupportedError(
            "Only single-target or tuple unpacking assignments are supported",
            node=stmt,
        )

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

        elements = [self._emit_expr(e, self.function_type) for e in _elements(value)]
        tmp = self._next_tmp()
        # The parentheses here form a Rust tuple literal, so we keep them.
        lines = [f"let {tmp} = ({', '.join(elements)});"]
        for i, name in enumerate(names):
            lines.append(f"{name} = {tmp}.{i};")
        return "\n".join(lines)

    def _emit_augassign(self, stmt: ast.AugAssign) -> str:
        if not isinstance(stmt.target, ast.Name):
            raise UnsupportedError(
                "Only simple names may be used in augmented assignment",
                node=stmt,
            )
        name = stmt.target.id
        fake = ast.BinOp(
            left=ast.Name(id=name, ctx=ast.Load()),
            op=stmt.op,
            right=stmt.value,
        )
        value = self._strip_outer_parens(self._emit_binop(fake, self.function_type))
        return f"{name} = {value};"

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

    def _emit_while(self, stmt: ast.While) -> str:
        cond = self._strip_outer_parens(self._emit_expr(stmt.test, "bool"))
        body = self._emit_body(stmt.body)
        return f"while {cond} {{\n{body}\n}}"

    def _emit_for(self, stmt: ast.For) -> str:
        if not isinstance(stmt.target, ast.Name):
            raise UnsupportedError(
                "Only a single loop variable is supported", node=stmt
            )
        if not isinstance(stmt.iter, ast.Call):
            raise UnsupportedError("Only range(...) loops are supported", node=stmt)
        call = stmt.iter
        if _call_name(call) != "range":
            raise UnsupportedError("Only range(...) loops are supported", node=call)
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
        else:
            raise UnsupportedError("range(...) with step is not supported", node=call)
        body = self._emit_body(stmt.body)
        if self.function_type == "f64" and stmt.target.id != "_":
            # The loop index is an integer, but the rest of the function expects
            # f64. Shadow it as f64 inside the body so `return i` works.
            body = f"    let {stmt.target.id} = {stmt.target.id} as f64;\n{body}"
        return f"for {stmt.target.id} in {range_expr} {{\n{body}\n}}"

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
            return expr.id
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
        if isinstance(expr, (ast.Tuple, ast.List)):
            return f"({', '.join(self._emit_expr(e, ctx) for e in expr.elts)})"
        if isinstance(expr, ast.Attribute):
            return self._emit_attribute(expr, ctx)
        if isinstance(expr, ast.Subscript):
            raise UnsupportedError("Subscript/indexing is not supported", node=expr)
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
        raise UnsupportedError(
            f"Unsupported expression: {type(expr).__name__}", node=expr
        )

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

        operand = self._emit_expr(expr.operand, ctx)
        if isinstance(expr.op, ast.UAdd):
            return operand
        if isinstance(expr.op, ast.USub):
            return f"-({operand})"
        if isinstance(expr.op, ast.Not):
            return f"!({operand})"
        raise UnsupportedError(
            f"Unsupported unary operator: {type(expr.op).__name__}", node=expr
        )

    def _emit_binop(self, expr: ast.BinOp, ctx: str) -> str:
        left = self._emit_expr(expr.left, ctx)
        right = self._emit_expr(expr.right, ctx)
        op = expr.op

        if isinstance(op, ast.Add):
            return f"({left} + {right})"
        if isinstance(op, ast.Sub):
            return f"({left} - {right})"
        if isinstance(op, ast.Mult):
            return f"({left} * {right})"
        if isinstance(op, ast.Div):
            return f"({left} / {right})"
        if isinstance(op, ast.FloorDiv):
            if ctx == "f64":
                return f"(({left}) / ({right})).floor()"
            return f"({left}).div_euclid({right})"
        if isinstance(op, ast.Mod):
            if ctx == "f64":
                return f"(({left}) % ({right}))"
            return f"({left}).rem_euclid({right})"
        if isinstance(op, ast.Pow):
            if ctx == "f64":
                return f"({left}).powf({right})"
            return f"({left}).pow(({right}) as u32)"
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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
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
    return names


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
