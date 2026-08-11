"""Core proactive synthesis verification and in-memory healing.

The ``CoreVerificationPipeline`` runs before source files are written to disk.
It uses the SMT engine to prove that declared return types are unifiable with
the types realized by function bodies, and it asks ``FallbackManager`` to
rewrite the return statement or signature when they are not.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

import z3


class ReturnTypeUnificationError(ValueError):
    """Raised when a declared return type cannot be unified with the body."""


class CoreVerificationPipeline:
    """Pre-materialization verification gate for return-type consistency.

    The pipeline treats each function as a pair of Z3 integer variables:
    ``declared`` and ``realized``.  Two types are unifiable when they are equal,
    both numeric (allowing an ``as`` cast), or one of them is ``void``.
    """

    _TYPE_IDS = {
        "void": 0,
        "()": 0,
        "": 0,
        "i8": 1,
        "i16": 2,
        "i32": 3,
        "i64": 4,
        "i128": 5,
        "u8": 6,
        "u16": 7,
        "u32": 8,
        "u64": 9,
        "u128": 10,
        "isize": 11,
        "usize": 12,
        "f32": 13,
        "f64": 14,
        "bool": 15,
        "string": 16,
        "str": 16,
        "String": 16,
    }

    @classmethod
    def _type_id(cls, t: Optional[str]) -> int:
        if t is None:
            return 0
        t = t.strip()
        if not t or t in ("void", "None", "()"):
            return 0
        # Strip PyResult/Result/Option wrappers for the unification check.
        for wrapper in ("PyResult<", "Result<", "Option<"):
            if t.startswith(wrapper) and t.endswith(">"):
                inner = t[len(wrapper) : -1].strip()
                return cls._type_id(inner)
        if t in cls._TYPE_IDS:
            return cls._TYPE_IDS[t]
        # Unknown opaque types are assigned a unique negative slot so they only
        # unify with themselves.
        return -(hash(t) % (2**31 - 1))

    @classmethod
    def _is_numeric(cls, t: Optional[str]) -> bool:
        tid = cls._type_id(t)
        return tid >= 1 and tid <= 15

    @classmethod
    def unifiable(cls, declared: Optional[str], realized: Optional[str]) -> bool:
        """Return ``True`` when Z3 can prove the two return types are compatible."""
        d_id = cls._type_id(declared)
        r_id = cls._type_id(realized)
        if d_id == r_id:
            return True
        if d_id == 0 or r_id == 0:
            return True
        if d_id < 0 or r_id < 0:
            # Opaque types only unify with themselves (checked above).
            return False
        # For primitive scalar types, numeric casts are allowed in Rust.
        if cls._is_numeric(declared) and cls._is_numeric(realized):
            return True
        s = z3.Solver()
        d = z3.Int("declared")
        r = z3.Int("realized")
        s.add(d == d_id, r == r_id)
        s.add(d == r)
        return s.check() == z3.sat

    @staticmethod
    def _rust_function_signature(
        source: str, symbol: str
    ) -> Tuple[Optional[str], Optional[str], int, int]:
        """Return ``(return_type, function_body, body_start_index, body_end_index)``."""
        # Match a Rust function header, possibly preceded by attributes.
        pattern = re.compile(
            r"(?:#\[[^\]]*\]\s*)*"  # optional attributes
            r"(?:pub\s+)?(?:extern\s+\"C\"\s+)?fn\s+" + re.escape(symbol) + r"\s*\([^)]*\)"
            r"(?:\s*->\s*([^\{\n]+))?\s*\{",
            re.DOTALL,
        )
        m = pattern.search(source)
        if not m:
            return None, None, 0, 0
        declared = (m.group(1) or "").strip() or None
        start = m.end() - 1  # index of the opening '{'
        # Extract balanced body.
        depth = 0
        end = start
        for i in range(start, len(source)):
            ch = source[i]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    end = i + 1
                    break
        return declared, source[start:end], start, end

    @staticmethod
    def _infer_body_return_type(body: str) -> Optional[str]:
        """Best-effort realized return type from a Rust function body."""
        # Prefer explicit `return <expr>;` statements.
        for m in re.finditer(r"return\s+([^;]+);", body):
            expr = m.group(1).strip()
            return CoreVerificationPipeline._infer_expr_type(expr)
        # Tail expression: last non-empty, non-comment line before the closing brace.
        lines = [l.strip() for l in body.splitlines() if l.strip() and not l.strip().startswith("//")]
        if lines:
            tail = lines[-1].rstrip(";")
            if tail and tail != "}":
                return CoreVerificationPipeline._infer_expr_type(tail)
        return None

    @staticmethod
    def _infer_expr_type(expr: str) -> Optional[str]:
        expr = expr.strip()
        if expr.startswith("\"") or expr.startswith("r\""):
            return "String"
        if expr in ("true", "false"):
            return "bool"
        if re.match(r"^-?\d+\.\d+([eE][+-]?\d+)?$", expr):
            return "f64"
        if re.match(r"^-?\d+(_\w+)?$", expr):
            suffix = expr.rsplit("_", 1)[-1]
            if suffix in ("i64", "i32", "u64", "u32", "f64", "f32"):
                return suffix
            return "i64"
        if re.match(r"^-?\d+\.\d+(_\w+)?$", expr):
            suffix = expr.rsplit("_", 1)[-1]
            if suffix in ("f64", "f32"):
                return suffix
            return "f64"
        if expr.startswith("vec!["):
            return "Vec<()>"
        if expr.startswith("["):
            return "Vec<()>"
        # Arithmetic expressions: i64 unless a float literal is present.
        if re.search(r"\bf64\b|\b\d+\.\d+\b", expr):
            return "f64"
        return "i64"

    @staticmethod
    def _is_void_type(t: Optional[str]) -> bool:
        return not t or t.strip() in ("", "void", "None", "()")

    @staticmethod
    def _is_complex_type(t: Optional[str]) -> bool:
        if not t:
            return False
        t = t.strip().lower().replace(" ", "")
        return t.startswith("(") or t.startswith("tuple[") or "vec<" in t or "pyresult" in t or "result<" in t

    @staticmethod
    def _is_pyoish_type(t: Optional[str]) -> bool:
        if not t:
            return False
        return "pyresult" in t.lower() or "pyobject" in t.lower()

    @classmethod
    def _rewrite_signature(cls, source: str, symbol: str, new_return_type: str) -> str:
        """Replace the declared return type of *symbol* with *new_return_type*."""
        pattern = re.compile(
            r"((?:#\[[^\]]*\]\s*)*(?:pub\s+)?(?:extern\s+\"C\"\s+)?fn\s+"
            + re.escape(symbol)
            + r"\s*\([^)]*\))"
            r"(?:\s*->\s*[^\{\n]+)?\s*\{",
            re.DOTALL,
        )
        m = pattern.search(source)
        if not m:
            return source
        header = m.group(1)
        # If an arrow already exists, drop the old return type; otherwise insert.
        return (
            source[: m.start()]
            + f"{header} -> {new_return_type} {source[m.end() - 1:]}"
        )

    @classmethod
    def _rewrite_return_cast(cls, source: str, symbol: str, to_type: str) -> str:
        """Insert ``as <to_type>`` casts into every ``return`` in *symbol*'s body."""
        def _cast_return(match: Any) -> str:
            expr = match.group(1).strip()
            if expr.endswith(f" as {to_type}"):
                return match.group(0)
            return f"return ({expr} as {to_type});"

        declared, body, start, end = cls._rust_function_signature(source, symbol)
        if body is None:
            return source
        body = re.sub(r"return\s+([^;]+);", _cast_return, body)
        return source[:start] + body + source[end:]

    @classmethod
    def _rewrite_void_returns(cls, source: str, symbol: str, declared: str) -> str:
        """Replace bare ``return;`` with a zero-value return of *declared*."""
        zero = "0_i64"
        if declared in ("f64", "f32"):
            zero = "0.0_f64" if declared == "f64" else "0.0_f32"
        if declared == "bool":
            zero = "false"
        if declared == "String":
            zero = '\"\"'

        declared_tp, body, start, end = cls._rust_function_signature(source, symbol)
        if body is None:
            return source
        body = re.sub(r"return\s*;", f"return {zero};", body)
        body = re.sub(r"^\s*\}$", "", body)  # leave balancing braces untouched
        return source[:start] + body + source[end:]

    @classmethod
    def verify_and_repair(
        cls,
        source: str,
        symbols: List[str],
        language: str = "rust",
    ) -> Tuple[str, List[str]]:
        """Return ``(repaired_source, diagnostics)`` after checking return types.

        For every *symbol* in *symbols*, compare the function's declared return
        type with its body's realized return type.  If the two are unifiable
        but differ as primitive numeric types, the body is rewritten with an
        ``as`` cast so the signature stays intact.  If the declared type is
        void/unknown but the body returns a concrete value, the signature is
        aligned to the realized type.  If the body is void but the signature is
        concrete, bare returns are rewritten to a zero value.  If they are not
        unifiable, a ``ReturnTypeUnificationError`` is raised so the caller can
        trigger a broader fallback.
        """
        if language != "rust":
            return source, []
        diagnostics: List[str] = []
        for symbol in symbols:
            declared, body, _, _ = cls._rust_function_signature(source, symbol)
            if body is None:
                continue
            realized = cls._infer_body_return_type(body)
            # Declared void + concrete body  -> align signature to body.
            if cls._is_void_type(declared) and realized and not cls._is_void_type(realized):
                source = cls._rewrite_signature(source, symbol, realized)
                diagnostics.append(
                    f"Signature aligned for {symbol}: {declared!r} -> {realized!r}"
                )
                continue
            # Concrete declared + void body  -> emit a zero-value return.
            if (
                not cls._is_void_type(declared)
                and (realized is None or cls._is_void_type(realized))
                and not cls._is_complex_type(declared)
                and not cls._is_pyoish_type(declared)
            ):
                source = cls._rewrite_void_returns(source, symbol, declared)
                diagnostics.append(
                    f"Void return patched for {symbol}: -> {declared!r}"
                )
                continue
            if realized is None or declared is None:
                continue
            if cls.unifiable(declared, realized):
                if cls._type_id(declared) != cls._type_id(realized) and cls._is_numeric(declared) and cls._is_numeric(realized):
                    source = cls._rewrite_return_cast(source, symbol, declared)
                    diagnostics.append(
                        f"Return-type cast inserted for {symbol}: {realized} -> {declared}"
                    )
            else:
                raise ReturnTypeUnificationError(
                    f"{symbol}: declared return {declared!r} is not unifiable with realized {realized!r}"
                )
        return source, diagnostics
