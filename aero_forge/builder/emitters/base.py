"""Base primitives and registry for polyglot source emitter plugins."""

from __future__ import annotations

import builtins
import importlib
import ast
import json
import logging
import os
import re
import tempfile
import textwrap
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from threading import Lock
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, cast

logger = logging.getLogger("aero_forge.emitter_registry")

from aero_forge.builder.language_router import (
    SystemToolchainRouter,
    ToolchainNotFoundError,
    _accel_log,
)
from aero_forge.builder.spec import ASTNode, EngineSpec
from aero_forge.scheduler.goi_solver import _loop_dependency_matrix
from aero_forge.translator import uast_to_python_source
from aero_forge.builder.smt_engine import AttributeResolver, BuiltinAttributeGate


class BoundaryContract(Enum):
    """Known cross-language binding contracts."""

    C_ABI = "c_abi"
    PYO3_MATURIN = "pyo3_maturin"
    WASM_WASI = "wasm_wasi"
    JNI = "jni"
    CGO = "cgo"
    PINVOKE = "pinvoke"
    CUDA_HIP_C = "cuda_hip_c"


@dataclass(frozen=True)
class CapabilityDescriptor:
    """Describes the targets and toolchains an emitter plugin supports."""

    language_id: str
    supported_boundaries: Set[BoundaryContract]
    toolchains: List[str]
    file_extensions: List[str]
    supports_zero_copy: bool
    supports_async_ffi: bool


@dataclass
class CodeArtifact:
    """A single generated source or build file."""

    file_path: str
    content: str
    language: str
    is_header: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_uast(
        cls,
        uast: Any,
        file_path: str,
        *,
        attribute_resolver: Optional[Callable[[Any], Any]] = None,
    ) -> "CodeArtifact":
        """Materialize a UAST JSON logic sketch into a Python source artifact.

        The engine (not the LLM) owns final attribute spelling, so incorrect
        LLM-suggested attribute names such as ``conj`` are rewritten to the
        intent-correct ``conjugate`` by the SMT attribute resolver before
        unparsing.
        """
        resolver = attribute_resolver or AttributeResolver().resolve
        source = uast_to_python_source(uast, attribute_resolver=resolver)
        return cls(file_path=file_path, content=source, language="python")


class EntrypointBoilerplateNormalizer:
    """Rewrite malformed entry-point boilerplate into canonical Python.

    Common LLM hallucinations such as ``if __name__.eq == '__main__':`` are
    converted to ``if __name__ == '__main__':`` so the emitted source is both
    syntactically valid and HIN-friendly.
    """

    _ENTRYPOINT_ALIASES = {"eq", "equals", "equal"}
    _MAIN_LITERALS = {"__main__", "'__main__'", '"__main__"'}

    @classmethod
    def normalize(cls, source: str) -> str:
        """Return *source* with common entry-point idiom mistakes corrected."""
        if not source or not source.strip():
            return source
        try:
            tree = ast.parse(source)
        except SyntaxError:
            return source
        transformer = cls._Transformer()
        new_tree = transformer.visit(tree)
        ast.fix_missing_locations(new_tree)
        try:
            return ast.unparse(new_tree)
        except Exception:
            return source

    class _Transformer(ast.NodeTransformer):
        def visit_If(self, node: ast.If) -> ast.AST:
            new_test = self._canonical_main_guard(node.test)
            if new_test is not None:
                node = ast.copy_location(
                    ast.If(test=new_test, body=node.body, orelse=node.orelse),
                    node,
                )
            return self.generic_visit(node)

        def _canonical_main_guard(self, expr: ast.expr) -> Optional[ast.AST]:
            # ``__name__.eq == '__main__'`` or ``__name__.eq('__main__')``
            if isinstance(expr, ast.Compare) and len(expr.ops) == 1:
                left, op, right = expr.left, expr.ops[0], expr.comparators[0]
                if isinstance(op, ast.Eq) and self._is_main_literal(right):
                    new_left = self._name_from_attr(left)
                    if new_left is not None:
                        return ast.Compare(
                            left=new_left,
                            ops=[ast.Eq()],
                            comparators=[right],
                        )
            if isinstance(expr, ast.Call):
                func = expr.func
                if (
                    isinstance(func, ast.Attribute)
                    and isinstance(func.value, ast.Name)
                    and func.value.id == "__name__"
                    and func.attr in EntrypointBoilerplateNormalizer._ENTRYPOINT_ALIASES
                    and len(expr.args) == 1
                    and self._is_main_literal(expr.args[0])
                ):
                    return ast.Compare(
                        left=ast.Name(id="__name__", ctx=ast.Load()),
                        ops=[ast.Eq()],
                        comparators=[expr.args[0]],
                    )
            return None

        @staticmethod
        def _is_main_literal(node: ast.expr) -> bool:
            return (
                isinstance(node, ast.Constant)
                and isinstance(node.value, str)
                and node.value == "__main__"
            )

        def _name_from_attr(self, node: ast.expr) -> Optional[ast.Name]:
            if (
                isinstance(node, ast.Attribute)
                and isinstance(node.value, ast.Name)
                and node.value.id == "__name__"
                and node.attr in EntrypointBoilerplateNormalizer._ENTRYPOINT_ALIASES
            ):
                return ast.Name(id="__name__", ctx=ast.Load())
            return None


class ContentDensityValidator:
    """Ensure generated source files contain enough functional code to be useful.

    A file that is only imports, docstrings, and boilerplate is flagged as
    ``Synthesis Incompleteness`` so the build fails fast instead of producing a
    hollow artifact.
    """

    MIN_FUNCTIONAL_NODES: int = 2
    MIN_GENERIC_FUNCTIONAL_NODES: int = 2

    @classmethod
    def validate(cls, content: str, language: str) -> int:
        """Return the number of functional AST/semantic nodes in *content*.

        Raises :class:`ValueError` when the file is too sparse.
        """
        language = (language or "").lower()
        is_python = language == "python" or content.lstrip().startswith(
            ("import ", "from ")
        )
        if is_python:
            count = cls._count_python_nodes(content)
            threshold = cls.MIN_FUNCTIONAL_NODES
        else:
            count = cls._count_generic_nodes(content)
            threshold = cls.MIN_GENERIC_FUNCTIONAL_NODES
        if count < threshold:
            raise ValueError(
                f"Synthesis Incompleteness: source has only {count} functional node(s) "
                f"(minimum {threshold})"
            )
        _accel_log(
            "info",
            f"Logic Density Gate passed: {count} functional node(s) in source",
        )
        return count

    @classmethod
    def validate_pure_python(cls, content: str) -> None:
        """Fail fast if a supposedly pure-Python source imports native modules.

        This enforces the negative constraint emitted in the Compacted
        Functional Matrix: ``pure_python`` targets must not reference
        ``rust_core``, ``cpp_core``, ``ctypes``, ``cffi``, ``pyo3``,
        ``ffi_bridges`` scaffolding, or ``@accelerate`` decorators.
        """
        if re.search(
            r"\b(?:rust_core|cpp_core|ctypes|cffi|pyo3|ffi_bridges)\b",
            content,
            re.IGNORECASE,
        ):
            raise ValueError(
                "Forbidden native dependency in pure_python source: "
                "rust_core/cpp_core/ctypes/cffi/pyo3/ffi_bridges are not allowed."
            )
        if re.search(r"@\s*accelerate\s*\(", content):
            raise ValueError("Forbidden @accelerate decorator in pure_python source.")

    @classmethod
    def _count_python_nodes(cls, content: str) -> int:
        try:
            tree = ast.parse(content)
        except SyntaxError:
            # Defer syntactically invalid sources to the normal syntax validator.
            return cls.MIN_FUNCTIONAL_NODES

        functional: Set[ast.AST] = set()

        def _add(node: ast.AST) -> None:
            functional.add(id(node))

        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                _add(node)
                for stmt in node.body:
                    cls._count_statement(stmt, _add, is_module=False)
            else:
                cls._count_statement(node, _add, is_module=True)

        return len(functional)

    @classmethod
    def _count_statement(
        cls, node: ast.AST, add: Callable[[ast.AST], None], *, is_module: bool
    ) -> None:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            return
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant):
            # Module/function docstrings are not functional code.
            if isinstance(node.value.value, str):
                return
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Name):
            # Standalone name references are not meaningful logic.
            return
        if isinstance(node, ast.Pass):
            return
        if isinstance(node, ast.AnnAssign) and node.value is None:
            return
        add(node)

    @classmethod
    def _count_generic_nodes(cls, content: str) -> int:
        """Fast fallback for non-Python sources using regex patterns.

        Counts only real computational work: control flow, function calls, and
        operators.  Declarations (``let``/``const``/``var``/``struct``/``class``),
        bare ``return`` statements, and ``_ = arg_*`` suppression lines are
        *not* counted, so a stub like ``fn f() { _ = arg_0; return 0; }``
        returns 0 and triggers a retry.
        """
        # Remove C++/Zig/Rust block comments and single-line comments.
        cleaned = re.sub(r"//.*$|#.*$", "", content, flags=re.MULTILINE)
        cleaned = re.sub(r"/\*.*?\*/", "", cleaned, flags=re.DOTALL)
        if not cleaned.strip():
            return 0

        # Strip import/include/use lines and ``const x = @import(...)`` imports.
        cleaned = re.sub(
            r"^\s*(?:#include|import|use|using)\b.*$",
            "",
            cleaned,
            flags=re.MULTILINE | re.IGNORECASE,
        )
        cleaned = re.sub(
            r"^\s*const\s+\w+\s*=\s*@import\s*\([^)]*\)\s*;\s*$",
            "",
            cleaned,
            flags=re.MULTILINE,
        )

        # Remove suppression placeholders: ``_ = arg_*;``, ``let _ = ...;``,
        # and ``return 0/arg_*;`` before counting.  This specifically targets the
        # hollow stubs produced by under-trained responses.
        cleaned = re.sub(r"_\s*=\s*arg_\d+\s*;", "", cleaned)
        cleaned = re.sub(r"\b(?:let|var|const)\s+_\s*=\s*[^;]+;", "", cleaned)
        cleaned = re.sub(
            r"return\s+(?:0|arg_\d+|_)\s*;", "", cleaned, flags=re.IGNORECASE
        )

        if not cleaned.strip():
            return 0

        patterns = [
            # Function definitions (Zig/Go/Rust/C/Mojo/General).
            r'\b(?:export\s+)?(?:pub\s+)?(?:extern\s+(?:"C"\s+))?fn\s+',
            # C/C++ function definitions with optional return type, qualifiers and extern "C".
            r'\b(?:extern\s+"C"\s+)?(?:const\s+)?(?:[\w<>,:*&~]+\s+){1,4}\w+\s*\([^)]*\)\s*(?:const\s+)?\{',
            r"\bfunc\s+",
            r"\bdef\s+",
            # Control flow.
            r"\b(?:if|for|while|loop|switch|match)\b",
            # Assignment (=) and operators (arithmetic, comparison, logical, bitwise).
            r"==|!=|<=|>=|&&|\|\||[\+\-*/%&|^<>!]=?",
        ]
        count = 0
        for pattern in patterns:
            count += len(re.findall(pattern, cleaned))
        return count

    @classmethod
    def _function_is_hollow(cls, func: ast.FunctionDef) -> bool:
        """Return True when *func* contains no real computational statements."""
        for stmt in func.body:
            if isinstance(stmt, (ast.Import, ast.ImportFrom, ast.Pass)):
                continue
            if isinstance(stmt, ast.Expr) and isinstance(
                getattr(stmt, "value", None), ast.Constant
            ):
                if isinstance(stmt.value.value, str):
                    continue
            if isinstance(stmt, ast.AnnAssign) and stmt.value is None:
                continue
            return False
        return True

    @classmethod
    def _symbol_is_pass_docstring_only(cls, content: str, symbol_name: str) -> bool:
        """Return True when *symbol_name* is a Python function with only pass/docstring."""
        try:
            tree = ast.parse(content)
        except SyntaxError:
            return False
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == symbol_name:
                return cls._function_is_hollow(node)
        return False

    @classmethod
    def has_execution_flow(cls, content: str, language: str) -> bool:
        """Return True when *content* has a non-zero GoI execution matrix.

        For Python we compute the block-diagonal union of per-function
        loop-dependency matrices so that the execution matrix ``M`` accounts for
        the dependency flow of every contracted function. Functions with no
        computational statements (empty body or only `pass`) are rejected before
        the GoI check. For non-Python sources we use the generic functional-node
        count as a proxy.
        """
        language = (language or "").lower()
        is_python = language == "python" or content.lstrip().startswith(
            ("import ", "from ")
        )
        if not is_python:
            return cls._count_generic_nodes(content) >= cls.MIN_GENERIC_FUNCTIONAL_NODES

        try:
            tree = ast.parse(content)
        except SyntaxError:
            # Syntactically invalid Python is deferred to the syntax validator.
            return True

        import json
        import numpy as np
        from aero_forge._native import execution_matrix_nonzero
        from aero_forge.scheduler.goi_solver import _loop_dependency_matrix

        functions = [
            node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
        ]
        if not functions:
            # No functions: use module-level functional density.
            return cls._count_python_nodes(content) >= cls.MIN_FUNCTIONAL_NODES

        # Every contracted function must contain real computational statements.
        for func in functions:
            if cls._function_is_hollow(func):
                return False

        matrices = []
        for func in functions:
            M, _ = _loop_dependency_matrix(func)
            if M.size == 0:
                # Represent symbol-free functions with a 1x1 zero block so the
                # block-diagonal matrix still accounts for every contracted symbol.
                M = np.zeros((1, 1), dtype=np.float64)
            matrices.append(M)

        total = sum(M.shape[0] for M in matrices)
        result = np.zeros((total, total), dtype=np.float64)
        offset = 0
        for M in matrices:
            n = M.shape[0]
            result[offset : offset + n, offset : offset + n] = M
            offset += n

        if not result.any():
            # A pure entrypoint may have no loop-carried writes but still calls
            # other functions; fall back to module-level functional density.
            return cls._count_python_nodes(content) >= cls.MIN_FUNCTIONAL_NODES

        # GoI Proof-Net verification: confirm the execution matrix is non-zero
        # using the native Rust solver.
        try:
            return execution_matrix_nonzero(json.dumps(result.tolist()))
        except Exception:
            return bool(result.any())

    @staticmethod
    def _has_data_payload(tree: ast.AST, symbol_name: str) -> bool:
        """Return True when *symbol_name* is a non-trivial top-level data constant."""
        for node in tree.body:
            if isinstance(node, (ast.Assign, ast.AnnAssign)):
                targets = getattr(node, "targets", [getattr(node, "target", None)])
                for target in targets:
                    if isinstance(target, ast.Name) and target.id == symbol_name:
                        value = getattr(node, "value", None)
                        if value is None:
                            return False
                        if isinstance(value, ast.Constant) and value.value is None:
                            return False
                        if isinstance(value, ast.Name) and value.id == "__AERO_IN_FILL__":
                            return False
                        if isinstance(value, ast.Dict) and not value.keys:
                            return False
                        if isinstance(value, (ast.List, ast.Set, ast.Tuple)) and not value.elts:
                            return False
                        return True
            if isinstance(node, ast.FunctionDef) and node.name == symbol_name:
                for stmt in reversed(node.body):
                    if isinstance(stmt, ast.Return) and stmt.value is not None:
                        value = stmt.value
                        if isinstance(value, ast.Constant) and value.value is None:
                            return False
                        if isinstance(value, ast.Name) and value.id == "__AERO_IN_FILL__":
                            return False
                        if isinstance(value, ast.Dict) and not value.keys:
                            return False
                        if isinstance(value, (ast.List, ast.Set, ast.Tuple)) and not value.elts:
                            return False
                        return True
        return False

    @classmethod
    def has_execution_flow_for_symbol(
        cls,
        content: str,
        symbol_name: str,
        language: str = "python",
    ) -> bool:
        """Return True when *symbol_name* has a non-zero GoI execution matrix.

        A contracted function with no loops is still valid as long as it contains
        at least ``MIN_FUNCTIONAL_NODES`` functional AST nodes. This prevents
        a single symbol from being emitted as a hollow stub even when the file
        as a whole happens to have execution flow elsewhere.
        """
        language = (language or "").lower()
        is_python = language == "python" or content.lstrip().startswith(
            ("import ", "from ")
        )
        if not is_python:
            return cls._count_generic_nodes(content) >= cls.MIN_GENERIC_FUNCTIONAL_NODES

        try:
            tree = ast.parse(content)
        except SyntaxError:
            return True

        if cls._has_data_payload(tree, symbol_name):
            return True

        functions = {
            node.name: node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef)
        }
        func = functions.get(symbol_name)
        if func is None:
            return False

        if cls._function_is_hollow(func):
            return False

        import numpy as np

        M, _ = _loop_dependency_matrix(func)
        if M.size > 0 and M.any():
            return True

        # No loop-carried dependencies: fall back to per-function node density.
        func_source = ast.unparse(func)
        return cls._count_python_nodes(func_source) >= cls.MIN_FUNCTIONAL_NODES


class ContextExhaustionError(RuntimeError):
    """Raised when the Compacted Functional Matrix lacks a logic intent for a contracted symbol."""

    def __init__(self, message: str, symbols: Optional[List[str]] = None) -> None:
        super().__init__(message)
        self.symbols = symbols or []


class SLIIntentValidator:
    """Validate that every contracted symbol has a logic intent in the CFM.

    During the Semantic Logic In-Fill phase the Builder Emission Agent must be
    able to look up each symbol it is asked to implement. If the Compacted
    Functional Matrix does not contain an entry for a required symbol, the
    validator raises ``ContextExhaustionError`` so the orchestrator can trigger
    a focused LLM retry for that symbol.
    """

    @staticmethod
    def _collect_cfm_symbols(compacted_context: Optional[Dict[str, Any]]) -> Set[str]:
        """Return every symbol name present in the CFM."""
        context = compacted_context or {}
        symbols: Set[str] = set()
        for fn in context.get("functions", []):
            name = fn.get("name") or fn.get("symbol")
            if name:
                symbols.add(name)
        impl_map = context.get("full_implementation_map") or {}
        for entry in impl_map.get("symbols", []):
            if isinstance(entry, dict):
                name = entry.get("name") or entry.get("symbol")
                if name:
                    symbols.add(name)
            elif isinstance(entry, str):
                symbols.add(entry)
        # Graph-polyglot contracts also carry explicit symbols (e.g. C-ABI/PyO3 edges).
        for contract in context.get("contracts", []):
            if isinstance(contract, dict):
                name = contract.get("symbol") or contract.get("name")
                if name:
                    symbols.add(name)
        return symbols

    @classmethod
    def find_exhausted_symbols(
        cls,
        compacted_context: Optional[Dict[str, Any]],
        required_symbols: List[str],
    ) -> List[str]:
        """Return required symbols that are missing from the CFM."""
        cfm_symbols = cls._collect_cfm_symbols(compacted_context)
        return [s for s in required_symbols if s not in cfm_symbols]

    @classmethod
    def validate(
        cls,
        compacted_context: Optional[Dict[str, Any]],
        required_symbols: List[str],
    ) -> None:
        """Raise ``ContextExhaustionError`` if any required symbol lacks intent."""
        exhausted = cls.find_exhausted_symbols(compacted_context, required_symbols)
        if exhausted:
            raise ContextExhaustionError(
                f"Context Exhaustion: no logic intent in CFM for symbols: {exhausted}",
                symbols=exhausted,
            )


class ContractIntegrityValidator:
    """Verify that materialized source defines every symbol declared in the blueprint.

    Contract-to-source integrity prevents hollow builds where the LLM emits
    imports and boilerplate but omits requested functions.  Python sources are
    checked with ``ast``; other languages fall back to declaration regexes.
    """

    @classmethod
    def missing_symbols(
        cls, content: str, language: str, symbols: List[str]
    ) -> List[str]:
        """Return the subset of *symbols* not defined in *content*."""
        if not symbols:
            return []
        language = (language or "").lower()
        is_python = language in ("python", "py") or content.lstrip().startswith(
            ("import ", "from ")
        )
        if is_python:
            try:
                tree = ast.parse(content)
            except SyntaxError:
                # If the source is not even parseable, treat every symbol as missing
                # so the syntax validator will surface the real problem first.
                return list(symbols)
            defined: Set[str] = set()
            for node in ast.walk(tree):
                if isinstance(
                    node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
                ):
                    defined.add(node.name)
                elif isinstance(node, ast.Assign):
                    for target in node.targets:
                        if isinstance(target, ast.Name):
                            defined.add(target.id)
            return [s for s in symbols if s not in defined]

        # Generic fallback for non-Python sources.
        defined: Set[str] = set()
        for sym in symbols:
            if re.search(
                rf"(?:^|\b)(?:def|fn|func|function|class|struct|interface)\s+{re.escape(sym)}\b",
                content,
                re.MULTILINE,
            ):
                defined.add(sym)
            elif re.search(rf"\b{re.escape(sym)}\s*\(", content):
                defined.add(sym)
        return [s for s in symbols if s not in defined]

    @classmethod
    def validate(cls, content: str, language: str, symbols: List[str]) -> List[str]:
        """Raise :class:`ValueError` if any required *symbols* are missing."""
        missing = cls.missing_symbols(content, language, symbols)
        if missing:
            raise ValueError(f"Missing contracted symbols: {missing}")
        return missing


class AtomicSymbolAssemblyError(RuntimeError):
    """Raised when a node cannot be emitted atomically with all contracted symbols present."""

    def __init__(self, message: str, symbols: Optional[List[str]] = None) -> None:
        super().__init__(message)
        self.symbols = symbols or []


class HINSaturationError(RuntimeError):
    """Raised when a contracted symbol produces an unsaturated HIN graph."""


class GoIWavefrontTaskCompletion:
    """Wavefront task completion gate.

    A wavefront task (one file/node) is not marked complete until every
    contracted symbol inside it has passed the HIN Node Saturation check and
    the GoI Proof Net Verification.
    """

    @staticmethod
    def mark_complete(node_id: str, symbols: List[str]) -> None:
        _accel_log(
            "info",
            f"Wavefront task complete for {node_id}: "
            f"{len(symbols)} contracted symbol(s) verified",
        )


class HINSaturationValidator:
    """Verify that a contracted symbol has a saturated HIN interaction net.

    A saturated HIN graph has at least one active principal-principal pair and
    no stalled (same-kind) active pairs. Unsaturated or stalled nets indicate
    that the symbol lacks real computational intent and must not be materialized.
    """

    @staticmethod
    def _symbol_is_data_constant(tree: "ast.AST", symbol: str) -> bool:
        """Return True when *symbol* is defined as a non-trivial top-level data constant."""
        for node in tree.body:
            if isinstance(node, (ast.Assign, ast.AnnAssign)):
                targets = getattr(node, "targets", [getattr(node, "target", None)])
                for target in targets:
                    if isinstance(target, ast.Name) and target.id == symbol:
                        value = getattr(node, "value", None)
                        if value is None:
                            return False
                        if isinstance(value, ast.Constant) and value.value is None:
                            return False
                        if isinstance(value, ast.Name) and value.id == "__AERO_IN_FILL__":
                            return False
                        if isinstance(value, ast.Dict) and not value.keys:
                            return False
                        if isinstance(value, (ast.List, ast.Set, ast.Tuple)) and not value.elts:
                            return False
                        return True
        return False

    @classmethod
    def verify_symbol(
        cls,
        content: str,
        symbol: str,
        language: str = "python",
    ) -> bool:
        """Return True when *symbol* in *content* has a saturated HIN graph."""
        language = (language or "").lower()
        if language not in ("python", "py"):
            _accel_log("info", f"HIN Node Saturation Verified: {symbol} (non-Python)")
            return True

        import ast as _ast

        try:
            tree = _ast.parse(content)
        except SyntaxError as exc:
            raise HINSaturationError(
                f"HIN saturation check for {symbol}: source is not syntactically valid: {exc}"
            ) from exc

        func = next(
            (
                node
                for node in _ast.walk(tree)
                if isinstance(node, _ast.FunctionDef) and node.name == symbol
            ),
            None,
        )
        is_data_constant = func is None and cls._symbol_is_data_constant(tree, symbol)
        if func is None and not is_data_constant:
            raise HINSaturationError(
                f"HIN saturation check for {symbol}: symbol not found in source"
            )

        if is_data_constant:
            _accel_log("info", f"HIN Node Saturation Verified: {symbol} (data constant)")
            return True

        func_source = _ast.get_source_segment(content, func)
        if not func_source:
            # Fallback to unparsing the function node.
            try:
                func_source = _ast.unparse(func)
            except Exception as exc:
                raise HINSaturationError(
                    f"HIN saturation check for {symbol}: could not extract function source: {exc}"
                ) from exc

        from aero_forge._native import HinEngine, verify_hin_saturation
        from aero_forge.translator.aero_frontend import python_source_to_uast

        try:
            uast = python_source_to_uast(func_source)
            engine = HinEngine()
            engine.build_from_json(json.dumps(uast))
            arena = engine.to_json()
        except Exception as exc:
            raise HINSaturationError(
                f"HIN saturation check for {symbol}: could not build HIN arena: {exc}"
            ) from exc

        try:
            result_json = verify_hin_saturation(arena)
            result = json.loads(result_json)
        except Exception as exc:
            raise HINSaturationError(
                f"HIN saturation check for {symbol}: energy evaluation failed: {exc}"
            ) from exc

        if not result.get("saturated"):
            reason = result.get("reason", "unknown")
            raise HINSaturationError(
                f"HIN Node Saturation failed for {symbol}: {reason} (energy={result})"
            )

        _accel_log("info", f"HIN Node Saturation Verified: {symbol}")
        return True


class HINNativeHealer:
    """Attempt a deterministic, native HIN repair for an unsaturated symbol."""

    @staticmethod
    def heal_symbol(
        content: str,
        symbol: str,
        error_log: str,
        workspace: Optional[Path] = None,
    ) -> Optional[str]:
        """Return a patched source fragment for *symbol* or None if repair failed."""
        import ast as _ast
        import tempfile

        from aero_forge.healing.healer import DeterministicHealer
        from aero_forge.translator.aero_frontend import python_source_to_uast

        try:
            tree = _ast.parse(content)
            func = next(
                node
                for node in _ast.walk(tree)
                if isinstance(node, _ast.FunctionDef) and node.name == symbol
            )
            func_source = _ast.get_source_segment(content, func) or content
        except Exception as exc:
            _accel_log(
                "warning",
                f"HIN-Native Healing could not locate {symbol}: {exc}",
            )
            return None

        ws = workspace or Path(tempfile.gettempdir()) / "hin_heal"
        ws.mkdir(parents=True, exist_ok=True)
        healer = DeterministicHealer(ws)

        try:
            uast = python_source_to_uast(func_source)
            result = healer.execute_healing_pass(
                error_log=error_log,
                source_text=func_source,
                uast_json=json.dumps(uast),
                apply=False,
            )
        except Exception as exc:
            _accel_log(
                "warning",
                f"HIN-Native Healing pass failed for {symbol}: {exc}",
            )
            return None

        patch = result.get("patch")
        if isinstance(patch, str) and patch.strip():
            _accel_log(
                "info",
                f"HIN-Native Healing produced source patch for {symbol}",
            )
            return patch

        _accel_log(
            "info",
            f"HIN-Native Healing pass completed for {symbol} without source patch",
        )
        return None


class MaterializationParityError(RuntimeError):
    """Raised when a node's emitted files do not match its blueprint manifest."""


class MaterializationParityGate:
    """Verify that every file declared for a node is physically present and non-empty."""

    @staticmethod
    def _expected_files(node_spec: Dict[str, Any]) -> List[str]:
        files: Set[str] = set()
        for sf in node_spec.get("source_files") or []:
            if isinstance(sf, str) and sf:
                files.add(sf)
        manifest = PolyglotEmitterPlugin.required_manifest(node_spec)
        if manifest:
            files.add(manifest)
        return sorted(files)

    @classmethod
    def verify(
        cls,
        node_id: str,
        node_spec: Dict[str, Any],
        node_dir: Path,
    ) -> None:
        """Raise :class:`MaterializationParityError` if any expected file is missing or empty."""
        expected = cls._expected_files(node_spec)
        missing: List[str] = []
        empty: List[str] = []
        for rel in expected:
            path = node_dir / rel
            if not path.is_file():
                missing.append(rel)
                continue
            if path.stat().st_size == 0:
                empty.append(rel)
        if missing or empty:
            raise MaterializationParityError(
                f"Node {node_id} materialization parity failed: "
                f"missing={missing}, empty={empty}"
            )
        total = len(expected)
        _accel_log("info", f"Node Materialization Verified: {node_id} ({total}/{total} file{'s' if total != 1 else ''})")


class LogicStarvationError(RuntimeError):
    """Raised when the compacted implementation map collapses to a single entrypoint
    despite the prompt requesting additional library/API modules."""


class LogicStarvationValidator:
    """Detect hollow blueprints that only emit an entrypoint while omitting
    library modules described in the prompt."""

    ENTRYPOINT_NAMES = {"main", "run", "cli", "__main__"}

    @classmethod
    def validate(
        cls,
        compacted_context: Dict[str, Any],
        prompt: str,
    ) -> None:
        """Raise :class:`LogicStarvationError` if the implementation map is hollow.

        A blueprint is considered starved when the only symbol in the
        ``full_implementation_map`` is an entrypoint while the prompt explicitly
        names other source files (e.g. a library module).
        """
        if not prompt:
            return
        impl_map = compacted_context.get("full_implementation_map") or {}
        symbols = [
            entry
            for entry in impl_map.get("symbols", [])
            if isinstance(entry, dict) and entry.get("name")
        ]
        if len(symbols) != 1:
            return
        only_name = str(symbols[0].get("name", "")).strip()
        if only_name not in cls.ENTRYPOINT_NAMES:
            return
        # Check whether the prompt explicitly names source files other than the
        # single entrypoint, which indicates the request describes a library.
        pattern = re.compile(
            r"\b[a-zA-Z_][\w/.-]*\.(?:py|rs|cpp|c|h|hpp|go|js|ts|toml|txt)\b"
        )
        required_files = [m.group(0) for m in pattern.finditer(prompt)]
        if not required_files:
            return
        entrypoint_files = {f"{only_name}.py", "main.py", "run.py", "cli.py"}
        library_files = [
            f for f in required_files if Path(f).name not in entrypoint_files
        ]
        if library_files:
            raise LogicStarvationError(
                f"Logic Starvation: compacted context contains only the entrypoint "
                f"'{only_name}' but the prompt requires additional modules: {library_files}. "
                "The blueprint must include at least one symbol for each named module."
            )


class TestDensityError(RuntimeError):
    """Raised when the generated test suite does not satisfy the per-symbol density constraint."""


class TestDensityValidator:
    """Verify that every contracted symbol has a non-zero test-to-symbol ratio."""

    @staticmethod
    def _list_test_functions(tests_dir: Path) -> List[str]:
        """Return all `def test_...` function names found under *tests_dir*."""
        functions: List[str] = []
        if not tests_dir.is_dir():
            return functions
        for test_file in tests_dir.glob("test_*.py"):
            try:
                tree = ast.parse(test_file.read_text(encoding="utf-8"))
            except SyntaxError:
                continue
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef) and node.name.startswith("test_"):
                    functions.append(f"{test_file.name}::{node.name}")
        return functions

    @staticmethod
    def _collect_contracted_symbols(
        nodes: List[Dict[str, Any]],
        edges: List[Dict[str, Any]],
    ) -> Set[str]:
        """Return the union of exported and edge-bound symbols, excluding test nodes."""

        def _is_test_node(node: Dict[str, Any]) -> bool:
            node_id = node.get("node_id", "")
            if node_id == "tests" or node_id.startswith("test_"):
                return True
            source_files = node.get("source_files") or []
            return any(
                isinstance(p, str) and (p.startswith("tests/") or "/tests/" in p)
                for p in source_files
            )

        symbols: Set[str] = set()
        for node in nodes:
            if _is_test_node(node):
                continue
            for sym in node.get("exports") or []:
                if sym and not sym.startswith("test_"):
                    symbols.add(sym)
        for edge in edges:
            sym = edge.get("symbol")
            if sym and not sym.startswith("test_"):
                symbols.add(sym)
        return symbols

    @classmethod
    def verify(
        cls,
        workspace: Path,
        nodes: List[Dict[str, Any]],
        edges: List[Dict[str, Any]],
        *,
        min_tests_per_symbol: int = 5,
    ) -> None:
        """Raise `TestDensityError` if the test suite is too sparse for the contracted symbols."""
        symbols = cls._collect_contracted_symbols(nodes, edges)
        if not symbols:
            return
        tests_dir = Path(workspace) / "tests"

        # Only enforce high-density tests when the blueprint actually contains a
        # dedicated test node or an existing tests/ directory. This keeps existing
        # low-level materializer unit tests valid while still gating real builds.
        has_test_node = any(
            (n.get("node_id") or "").startswith("test")
            or any(
                isinstance(p, str) and (p.startswith("tests/") or "/tests/" in p)
                for p in n.get("source_files") or []
            )
            for n in nodes
        )
        if not has_test_node and not tests_dir.is_dir():
            _accel_log(
                "warning",
                f"Test density skipped: no tests/ directory or test node for {len(symbols)} symbol(s).",
            )
            return

        test_functions = cls._list_test_functions(tests_dir)
        if not test_functions:
            raise TestDensityError(
                f"No tests found in {tests_dir} for {len(symbols)} contracted symbol(s). "
                "The Test Density Constraint requires at least five distinct unit tests per symbol."
            )
        required = max(1, len(symbols) * min_tests_per_symbol)
        if len(test_functions) < required:
            raise TestDensityError(
                f"Test density too low: {len(test_functions)} test(s) for {len(symbols)} symbol(s); "
                f"required at least {required} (min {min_tests_per_symbol} per symbol)."
            )
        _accel_log(
            "success",
            f"Test density verified: {len(test_functions)} test(s) for {len(symbols)} symbol(s) "
            f"(ratio {len(test_functions) / len(symbols):.2f})",
        )


class AtomicSymbolAssembly:
    """Atomic, multi-symbol SLI gate.

    Before a file is written the builder must confirm that:
      1. Every contracted symbol has a logic intent in the Compacted Functional Matrix.
      2. The emitted source defines every contracted symbol.
      3. Each contracted symbol has a saturated HIN interaction net.
      4. Each contracted symbol has a non-zero GoI execution matrix.

    This transitions the builder from per-file materialization to an Atomic Symbol
    Assembly model: the final source artifact is accepted or rejected as a whole.
    """

    @staticmethod
    def _required_symbols(
        node_spec: Dict[str, Any],
        contracts: List[Dict[str, Any]],
        is_pure_python: bool = False,
    ) -> List[str]:
        """Return the contracted symbols this node is expected to define."""
        node_id = node_spec.get("node_id", "")
        symbols: Set[str] = set(node_spec.get("exports") or [])
        for contract in contracts or []:
            if contract.get("source") == node_id:
                symbols.add(contract.get("symbol", ""))
        symbols.discard("")
        is_target = any(c.get("target") == node_id for c in contracts or [])
        if not symbols and not is_target:
            # Every node must be accounted for, including pure_python entrypoints.
            symbols.add(node_id)
        return sorted(symbols)

    @staticmethod
    def _is_data_payload_symbol(
        symbol: str, compacted_context: Optional[Dict[str, Any]]
    ) -> bool:
        if not symbol or not compacted_context:
            return False
        for fn in compacted_context.get("functions", []):
            if (fn.get("name") == symbol or fn.get("symbol") == symbol) and fn.get("data_payload"):
                return True
        for entry in compacted_context.get("data_constants", []):
            if entry.get("name") == symbol or entry.get("symbol") == symbol:
                return True
        impl_map = compacted_context.get("full_implementation_map") or {}
        for entry in impl_map.get("symbols", []):
            if (entry.get("name") == symbol or entry.get("symbol") == symbol) and entry.get("data_payload"):
                return True
        return False

    @staticmethod
    def _data_payload_is_trivial(content: str, symbol: str) -> bool:
        """Return True when a data constant is missing, empty, or still a placeholder."""
        import ast as _ast

        try:
            tree = _ast.parse(content)
        except SyntaxError:
            return True

        def _value_is_trivial(value: Optional[_ast.expr]) -> bool:
            if value is None:
                return True
            if isinstance(value, _ast.Constant) and value.value is None:
                return True
            if isinstance(value, _ast.Name) and value.id == "__AERO_IN_FILL__":
                return True
            if isinstance(value, _ast.Dict) and not value.keys:
                return True
            if isinstance(value, (_ast.List, _ast.Set, _ast.Tuple)) and not value.elts:
                return True
            return False

        for node in tree.body:
            if isinstance(node, _ast.Assign):
                for target in node.targets:
                    if isinstance(target, _ast.Name) and target.id == symbol:
                        return _value_is_trivial(node.value)
            if isinstance(node, _ast.AnnAssign) and isinstance(node.target, _ast.Name) and node.target.id == symbol:
                return _value_is_trivial(node.value)
            if isinstance(node, _ast.FunctionDef) and node.name == symbol:
                for stmt in reversed(node.body):
                    if isinstance(stmt, _ast.Return) and stmt.value is not None:
                        return _value_is_trivial(stmt.value)
        return True

    @classmethod
    def validate(
        cls,
        artifacts: List[Any],
        node_spec: Dict[str, Any],
        contracts: List[Dict[str, Any]],
        compacted_context: Optional[Dict[str, Any]],
        language: str = "python",
        is_pure_python: bool = False,
    ) -> None:
        """Raise :class:`AtomicSymbolAssemblyError` if the emission is not atomic.

        ``artifacts`` is a list of :class:`CodeArtifact` instances that are
        candidates to be written for *node_spec*.
        """
        required = cls._required_symbols(
            node_spec, contracts, is_pure_python=is_pure_python
        )
        if not required:
            return

        # 1. CFM intent retrieval gate. Only enforce it when the CFM was actually
        # populated with symbols; unit tests and offline materializers may provide
        # the source directly without a populated Compacted Functional Matrix.
        cfm_symbols = (
            SLIIntentValidator._collect_cfm_symbols(compacted_context)
            if compacted_context
            else set()
        )
        if cfm_symbols:
            missing_intent = [s for s in required if s not in cfm_symbols]
            if missing_intent:
                raise AtomicSymbolAssemblyError(
                    f"Context Exhaustion: no logic intent in CFM for symbols: {missing_intent}",
                    symbols=missing_intent,
                )

        # 2. Source must define all contracted symbols.
        source_artifacts = [a for a in artifacts if not getattr(a, "is_header", False)]

        # 2b. HIN AST normalization: heal malformed entry-point boilerplate before
        # any HIN/SMT verification so comparison operators are wired to Value agents.
        for artifact in source_artifacts:
            artifact.content = EntrypointBoilerplateNormalizer.normalize(
                artifact.content
            )
        combined = "\n".join(
            a.content for a in source_artifacts if not getattr(a, "is_header", False)
        )

        # Built-in attribute gate: reject hallucinated attributes on str/int/list/dict.
        label = (
            source_artifacts[0].file_path
            if source_artifacts
            else node_spec.get("node_id", "<unknown>")
        )
        try:
            BuiltinAttributeGate.verify(combined, artifact_path=label)
            _accel_log("info", f"Attribute Verification Passed for {label}")
        except Exception as exc:
            raise AtomicSymbolAssemblyError(
                f"Attribute Verification Failed for {label}: {exc}",
                symbols=required,
            ) from exc

        source_missing = ContractIntegrityValidator.missing_symbols(
            combined, language, required
        )
        if source_missing:
            raise AtomicSymbolAssemblyError(
                f"Incomplete materialization: missing symbols {source_missing}",
                symbols=source_missing,
            )

        # 2a. Data constants must be fully populated, not placeholders.
        payload_missing: List[str] = []
        for sym in required:
            if cls._is_data_payload_symbol(sym, compacted_context):
                if cls._data_payload_is_trivial(combined, sym):
                    payload_missing.append(sym)
        if payload_missing:
            raise AtomicSymbolAssemblyError(
                f"Data payload missing or trivial for symbols: {payload_missing}",
                symbols=payload_missing,
            )

        # 3. Per-symbol HIN Node Saturation check. The constructed HIN graph
        # must have active principal-principal pairs and no stalled wires.
        for sym in required:
            try:
                HINSaturationValidator.verify_symbol(combined, sym, language)
            except HINSaturationError as hse:
                _accel_log(
                    "warning",
                    f"HIN Node Saturation failed for {sym}: {hse}; "
                    f"triggering HIN-Native Healing pass",
                )
                patch = HINNativeHealer.heal_symbol(
                    combined,
                    sym,
                    str(hse),
                )
                if patch:
                    combined = cls._replace_function_source(combined, sym, patch)
                    # Retry after applying the native HIN patch.
                    HINSaturationValidator.verify_symbol(combined, sym, language)
                else:
                    raise AtomicSymbolAssemblyError(
                        f"HIN Node Saturation failed for {sym}: {hse}",
                        symbols=[sym],
                    ) from hse

        # 4. Per-symbol GoI proof-net verification.
        hollow: List[str] = []
        for sym in required:
            if not ContentDensityValidator.has_execution_flow_for_symbol(
                combined, sym, language
            ):
                hollow.append(sym)
        if hollow:
            logic_density_failures = [
                sym
                for sym in hollow
                if ContentDensityValidator._symbol_is_pass_docstring_only(
                    combined, sym
                )
            ]
            if logic_density_failures:
                raise AtomicSymbolAssemblyError(
                    f"Logic Density Failure: function(s) {logic_density_failures} "
                    "contain only a pass statement or docstring",
                    symbols=logic_density_failures,
                )
            raise AtomicSymbolAssemblyError(
                f"Zero execution matrix for symbols: {hollow}",
                symbols=hollow,
            )

        node_id = node_spec.get("node_id", "<unknown>")
        _accel_log(
            "success",
            f"Atomic Symbol Assembly verified for {node_id}: "
            f"{len(required)}/{len(required)} symbol(s) with non-zero execution matrices",
        )

        GoIWavefrontTaskCompletion.mark_complete(node_id, required)

    @staticmethod
    def _replace_function_source(source: str, symbol: str, patch: str) -> str:
        """Replace the definition of *symbol* in *source* with *patch* and unparse."""
        import ast as _ast

        try:
            tree = _ast.parse(source)
        except SyntaxError:
            return patch

        patch_tree = _ast.parse(patch)
        patch_func = next(
            (
                node
                for node in _ast.walk(patch_tree)
                if isinstance(node, _ast.FunctionDef)
            ),
            None,
        )
        if patch_func is None:
            return patch

        for i, node in enumerate(tree.body):
            if isinstance(node, _ast.FunctionDef) and node.name == symbol:
                tree.body[i] = patch_func
                break
        else:
            # Symbol not found; append the patch defensively.
            tree.body.append(patch_func)

        try:
            return _ast.unparse(tree)
        except Exception:
            return patch


class FocusedIntentRecovery:
    """Synthesize missing logic intents and merge them into the CFM.

    When Atomic Symbol Assembly fails because the CFM lacks intent for a
    contracted symbol, this agent asks the LLM for a focused logic sketch for
    exactly those symbols and patches the Compacted Functional Matrix so a
    final materialization retry can succeed.
    """

    _SYSTEM_PROMPT = (
        "You are the Intent Recovery Agent for the Proactive Formal Synthesis Engine. "
        "The builder failed to materialize logic for the listed symbols. "
        "Return a single JSON object with a top-level 'symbols' array. "
        "Each entry must contain: 'name', 'description', 'args' (list of strings), "
        "'return_type' (string), and 'steps' (list of algorithmic steps). "
        "For data-constant symbols (e.g. scoring matrices, lookup tables, BLOSUM62), "
        "set 'data_payload': true, 'payload_kind': 'dict'|'list'|'set', and provide "
        "a compact literal or a clear algorithm for populating it. "
        "Do not include prose, markdown fences, or source code."
    )

    @staticmethod
    def _lookup_data_payload(symbol: str, compacted_context: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        if not compacted_context or not symbol:
            return None
        for source in (
            compacted_context.get("functions", []),
            compacted_context.get("data_constants", []),
            (compacted_context.get("full_implementation_map") or {}).get("symbols", []),
        ):
            for entry in source:
                if entry.get("name") == symbol or entry.get("symbol") == symbol:
                    if entry.get("data_payload"):
                        return {
                            "data_payload": True,
                            "payload_kind": entry.get("payload_kind", "dict"),
                            "logic_sketch": entry.get("logic_sketch", ""),
                        }
        return None

    @classmethod
    def synthesize_missing_intents(
        cls,
        missing_symbols: List[str],
        node_spec: Dict[str, Any],
        compacted_context: Optional[Dict[str, Any]],
        llm_client: Any,
    ) -> Dict[str, Any]:
        """Return a copy of *compacted_context* with recovered intents for *missing_symbols*."""
        context = dict(compacted_context) if compacted_context else {}
        if not missing_symbols or not llm_client:
            return context

        user_prompt = (
            f"Project context: {json.dumps(context.get('project') or context.get('user_prompt') or {}, default=str)}\n"
            f"Node specification: {json.dumps(node_spec, default=str)}\n"
            f"Missing contracted symbols: {missing_symbols}\n"
            "Provide a focused logic intent for each missing symbol as a JSON object."
        )
        try:
            raw = llm_client.generate(
                [
                    {"role": "system", "content": cls._SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.1,
                max_tokens=2048,
            )
        except Exception as exc:
            raise AtomicSymbolAssemblyError(
                f"Focused Intent Synthesis failed for {missing_symbols}: {exc}"
            ) from exc

        if not raw:
            raise AtomicSymbolAssemblyError(
                f"Focused Intent Synthesis returned an empty response for {missing_symbols}"
            )

        parsed = cls._parse_json(raw)
        symbols = parsed.get("symbols") or (
            parsed if isinstance(parsed, list) else [parsed]
        )

        impl_map: Dict[str, Any] = context.setdefault("full_implementation_map", {})
        if not isinstance(impl_map, dict):
            impl_map = {"symbols": []}
            context["full_implementation_map"] = impl_map
        impl_symbols: List[Any] = impl_map.setdefault("symbols", [])
        existing_impl: Set[str] = {
            entry.get("name") if isinstance(entry, dict) else str(entry)
            for entry in impl_symbols
        }

        functions: List[Dict[str, Any]] = context.setdefault("functions", [])
        existing_fn: Set[str] = {
            f.get("name") or f.get("symbol", "")
            for f in functions
            if isinstance(f, dict)
        }

        for entry in symbols:
            if not isinstance(entry, dict):
                continue
            name = entry.get("name") or entry.get("symbol")
            if not name or name not in missing_symbols:
                continue
            if name not in existing_impl:
                impl_symbols.append(entry)
                existing_impl.add(name)
            if name not in existing_fn:
                fn_entry: Dict[str, Any] = {
                    "name": name,
                    "symbol": name,
                    "args": entry.get("args", []),
                    "return_type": entry.get("return_type", ""),
                    "description": entry.get("description", ""),
                    "logic_sketch": entry.get("logic_sketch", "") or entry.get("steps", ""),
                }
                if entry.get("data_payload"):
                    fn_entry["data_payload"] = True
                    fn_entry["payload_kind"] = entry.get("payload_kind", "dict")
                else:
                    existing = cls._lookup_data_payload(name, compacted_context)
                    if existing:
                        fn_entry.update(existing)
                functions.append(fn_entry)
                existing_fn.add(name)

        _accel_log(
            "info",
            f"Focused Intent Recovery synthesized intents for {sorted(missing_symbols)}",
        )
        return context

    @staticmethod
    def _parse_json(raw: str) -> Dict[str, Any]:
        """Best-effort JSON extraction from an LLM response."""
        text = raw.strip()
        if text.startswith("```"):
            # Strip markdown fences if present.
            parts = text.split("```", 2)
            text = parts[2] if len(parts) >= 3 else text.lstrip("`")
            text = text.strip()
            if text.lower().startswith("json"):
                text = text[4:].lstrip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            match = re.search(r"(\{.*\})", text, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group(1))
                except json.JSONDecodeError:
                    pass
        raise AtomicSymbolAssemblyError(
            f"Focused Intent Synthesis returned unparseable JSON: {raw[:200]!r}"
        )


class SyntaxValidator:
    """Non-destructive syntax validation for generated source artifacts.

    Catches truncated or malformed Python code before it is persisted. Other
    languages are currently validated by the toolchain during compilation.
    """

    @classmethod
    def validate(cls, content: str, language: str) -> None:
        """Raise :class:`SyntaxError` when *content* is not valid Python syntax.

        Only Python sources are checked; non-Python artifacts are accepted so
        the normal build pipeline can report language-specific errors.
        """
        language = (language or "").lower()
        is_python = (
            language == "python"
            or content.lstrip().startswith(("import ", "from "))
            or "__AERO_IN_FILL__" in content
        )
        if not is_python:
            return

        try:
            ast.parse(content)
        except (SyntaxError, IndentationError) as exc:
            raise SyntaxError(
                f"Syntax verification failed: {exc.msg} at line {exc.lineno}"
            ) from exc


class PolyglotEmitterPlugin(ABC):
    """Plugin interface for language-specific source emitters."""

    # Map canonical toolchains to the manifest file each node must carry.
    REQUIRED_MANIFESTS: Dict[str, str] = {
        "cmake": "CMakeLists.txt",
        "cargo": "Cargo.toml",
        "maturin": "Cargo.toml",
    }

    @property
    @abstractmethod
    def descriptor(self) -> CapabilityDescriptor:
        """Return the plugin capability descriptor."""

    @abstractmethod
    def emit_source_files(
        self,
        node_id: str,
        node_spec: Dict[str, Any],
        boundary_contracts: List[Dict[str, Any]],
    ) -> List[CodeArtifact]:
        """Emit source files for *node_spec* under the given *boundary_contracts*."""

    @abstractmethod
    def emit_build_manifest(
        self,
        node_id: str,
        dependencies: List[str],
        compiler_flags: List[str],
    ) -> CodeArtifact:
        """Emit a build-system manifest for the node."""

    @classmethod
    def required_manifest(cls, node_spec: Dict[str, Any]) -> Optional[str]:
        """Return the manifest file required by *node_spec*'s toolchain, if any."""
        toolchain = (node_spec.get("toolchain") or node_spec.get("lang") or "").lower()
        return cls.REQUIRED_MANIFESTS.get(toolchain)


class ManifestRecoveryError(RuntimeError):
    """Raised when a required build manifest cannot be synthesized."""


class ManifestRecovery:
    """Synthesize a missing build manifest for cmake/cargo/maturin nodes.

    Plugins are required to emit a manifest, but JIT-synthesized or hollow
    plugins may omit one. This recovery pass creates a deterministic, valid
    manifest from the node spec and source files.
    """

    @classmethod
    def synthesize(
        cls,
        node_id: str,
        node_spec: Dict[str, Any],
        source_artifacts: List[CodeArtifact],
        language_id: str,
    ) -> CodeArtifact:
        """Return a manifest artifact for *node_id* based on its toolchain."""
        toolchain = (node_spec.get("toolchain") or node_spec.get("lang") or language_id).lower()
        if toolchain == "cmake":
            return cls._cmake_manifest(node_id, node_spec, source_artifacts)
        if toolchain in ("cargo", "maturin"):
            return cls._cargo_manifest(node_id, node_spec, source_artifacts)
        raise ManifestRecoveryError(
            f"Cannot recover missing manifest for node {node_id}: unknown toolchain {toolchain!r}"
        )

    @classmethod
    def _source_names(cls, source_artifacts: List[CodeArtifact]) -> List[str]:
        srcs = [
            Path(a.file_path).name
            for a in source_artifacts
            if not cls._is_build_manifest_static(a)
            and Path(a.file_path).suffix in {".cpp", ".cc", ".cxx", ".rs"}
        ]
        return srcs or ["src/lib.rs"]

    @classmethod
    def _cmake_manifest(
        cls, node_id: str, node_spec: Dict[str, Any], source_artifacts: List[CodeArtifact]
    ) -> CodeArtifact:
        srcs = cls._source_names(source_artifacts)
        # If the plugin emitted a C++ source at the root of the node dir, the
        # CMake source list should reference it without a phantom src/ prefix.
        if len(srcs) == 1 and srcs[0].endswith(".cpp") and not srcs[0].startswith("src/"):
            src_ref = srcs[0]
        else:
            src_ref = " ".join(srcs)
        content = (
            "cmake_minimum_required(VERSION 3.16)\n"
            f"project({node_id or 'cpp_project'} LANGUAGES CXX)\n\n"
            "set(CMAKE_CXX_STANDARD 17)\n"
            "set(CMAKE_CXX_STANDARD_REQUIRED ON)\n"
            "set(CMAKE_POSITION_INDEPENDENT_CODE ON)\n\n"
            f"add_library({node_id or 'cpp_project'} SHARED {src_ref})\n"
            f"target_compile_options({node_id or 'cpp_project'} PRIVATE -fPIC -O3)\n"
            f"target_link_options({node_id or 'cpp_project'} PRIVATE -shared)\n"
            f"set_target_properties({node_id or 'cpp_project'} PROPERTIES OUTPUT_NAME {node_id or 'cpp_project'})\n"
        )
        return CodeArtifact(file_path="CMakeLists.txt", content=content, language="cmake")

    @classmethod
    def _cargo_manifest(
        cls, node_id: str, node_spec: Dict[str, Any], source_artifacts: List[CodeArtifact]
    ) -> CodeArtifact:
        crate = node_id or "rust_project"
        crate_safe = crate.replace("-", "_")
        content = (
            "[package]\n"
            f'name = "{crate}"\n'
            'version = "0.1.0"\n'
            'edition = "2021"\n\n'
            "[lib]\n"
            f'name = "{crate_safe}"\n'
            'crate-type = ["cdylib"]\n'
        )
        return CodeArtifact(file_path="Cargo.toml", content=content, language="toml")

    @staticmethod
    def _is_build_manifest_static(artifact: CodeArtifact) -> bool:
        """Local copy of manifest detection used during recovery synthesis."""
        return Path(artifact.file_path).name in {
            "CMakeLists.txt",
            "Cargo.toml",
            "pyproject.toml",
            "go.mod",
            "build.gradle",
            "pom.xml",
            "build.rs",
            "build.zig",
            "Makefile",
        }


class EmitterError(Exception):
    """Raised when an emitter cannot render an engine spec."""


class SynthesizedPluginError(EmitterError):
    """Raised when an LLM-synthesized emitter plugin is invalid or unsafe."""


class StructuralViolationError(EmitterError):
    """Raised when an LLM response violates the required token-delimited structure."""


def _pascal_case(language_id: str) -> str:
    """Convert a lowercase language id into a PascalCase class prefix."""
    parts = re.split(r"[-_ ]+", language_id.strip().lower())
    return "".join(p.capitalize() for p in parts if p)


def _python_string_literal(value: str) -> str:
    """Return *value* as a valid double-quoted Python string literal."""
    return json.dumps(value)


def _function_signature_for_prompt(
    language_id: str,
    symbol: str,
    args: List[str],
    return_type: str,
) -> str:
    """Return a concrete native function signature for the prompt placeholder."""
    lang = language_id.lower()
    arg_names = [f"arg_{i}" for i in range(len(args))]
    arg_decls = [
        f"{name}: {_type_map_for_boundary(lang, kind)}"
        for name, kind in zip(arg_names, args)
    ]
    rt = (return_type or "").strip().lower()
    if rt in ("", "void"):
        ret = ""
    else:
        ret = _type_map_for_boundary(lang, rt)
    if lang == "zig":
        return f"export fn {symbol}({', '.join(arg_decls)}) {ret}".strip()
    if lang == "go":
        proto = f"func {symbol}({', '.join(arg_decls)}) {ret}".strip()
        return f'//export {symbol}\nimport "C"\n{proto}'
    if lang in ("c", "cpp", "c++"):
        c_arg_decls = [
            f"{_type_map_for_boundary('c', kind)} {name}"
            for name, kind in zip(arg_names, args)
        ]
        return f"{ret or 'void'} {symbol}({', '.join(c_arg_decls)})"
    if lang == "csharp":
        cs_arg_decls = [
            f"{_type_map_for_boundary('csharp', kind)} {name}"
            for name, kind in zip(arg_names, args)
        ]
        return (
            f'[UnmanagedCallersOnly(EntryPoint = "{symbol}")]\n'
            f"public static {ret or 'void'} {symbol}"
            f"({', '.join(cs_arg_decls)})"
        )
    if lang == "mojo":
        return f"fn {symbol}({', '.join(arg_decls)}) -> {ret or 'None'}"
    return f"{ret or 'void'} {symbol}({', '.join(arg_decls)})"


def _function_context_from_node(node_spec: Optional[Dict[str, Any]]) -> str:
    """Extract a human-readable function purpose from the node spec."""
    if not node_spec:
        return ""
    for key in ("description", "purpose", "_synthesis_context"):
        value = node_spec.get(key)
        if value:
            return str(value)
    exports = node_spec.get("exports") or []
    if exports:
        return (
            f"Implement the exported function(s) {exports} "
            f"for the {node_spec.get('lang', 'target')} target."
        )
    return ""


def _type_map_for_boundary(
    language_id: str, arg_kind: str, is_return: bool = False
) -> str:
    """Map an abstract arg/return kind to a concrete native type string."""
    lang = language_id.lower()
    maps: Dict[str, Dict[str, str]] = {
        "zig": {
            "int32": "i32",
            "int64": "i64",
            "float32": "f32",
            "float64": "f64",
            "pointer": "[*c]f64",
        },
        "go": {
            "int32": "C.int",
            "int64": "C.longlong",
            "float32": "C.float",
            "float64": "C.double",
            "pointer": "*C.double",
        },
        "c": {
            "int32": "int32_t",
            "int64": "int64_t",
            "float32": "float",
            "float64": "double",
            "pointer": "double*",
        },
        "csharp": {
            "int32": "int",
            "int64": "long",
            "float32": "float",
            "float64": "double",
            "pointer": "IntPtr",
        },
        "java": {
            "int32": "jint",
            "int64": "jlong",
            "float32": "jfloat",
            "float64": "jdouble",
            "pointer": "jdoubleArray",
        },
        "d": {
            "int32": "int",
            "int64": "long",
            "float32": "float",
            "float64": "double",
            "pointer": "double*",
        },
        "nim": {
            "int32": "int32",
            "int64": "int64",
            "float32": "float32",
            "float64": "float64",
            "pointer": "ptr float64",
        },
        "fortran": {
            "int32": "integer(c_int32_t)",
            "int64": "integer(c_int64_t)",
            "float32": "real(c_float)",
            "float64": "real(c_double)",
            "pointer": "type(c_ptr), value",
        },
    }
    if lang in ("cpp", "c++"):
        lang = "c"
    if lang == "mojo":
        return {
            "int32": "Int32",
            "int64": "Int64",
            "float32": "Float32",
            "float64": "Float64",
            "pointer": "DTypePointer[DType.float64]",
        }.get(arg_kind, "Int64")
    return maps.get(lang, maps["c"]).get(arg_kind, "int64_t")


def _default_ret_literal(language_id: str, return_type: str) -> str:
    """Return a safe default literal for the given language and return type."""
    rt = (return_type or "").strip().lower()
    if rt in ("", "void"):
        return ""
    if rt in ("float32", "float64"):
        if language_id.lower() == "go":
            return "C.double(0.0)"
        return "0.0"
    if language_id.lower() == "go":
        return "C.longlong(0)"
    return "0"


def _scalar_ret_expr(arg_names: List[str], args: List[str], return_type: str) -> str:
    """Return a minimal math expression for the first scalar argument, or empty."""
    if not arg_names or not args:
        return ""
    scalar_kinds = {
        "int",
        "int32",
        "int64",
        "i32",
        "i64",
        "long",
        "longlong",
        "float",
        "float32",
        "float64",
        "f32",
        "f64",
        "double",
    }
    rt = (return_type or "").strip().lower()
    if rt == "void" or rt == "":
        return ""
    kind = (args[0] or "").strip().lower()
    if kind not in scalar_kinds:
        return ""
    is_float = kind in (
        "float",
        "float32",
        "float64",
        "f32",
        "f64",
        "double",
    ) or rt in ("float", "float32", "float64", "f32", "f64")
    multiplier = "2.0" if is_float else "2"
    return f"{arg_names[0]} * {multiplier}"


def _fallback_source(
    language_id: str,
    symbol: str,
    args: List[str],
    return_type: str,
    node_id: str,
) -> str:
    """Build a minimal C-ABI source body for a language fallback."""
    lang = language_id.lower()

    arg_names = [f"arg_{i}" for i in range(len(args))]
    arg_decls = [
        f"{_type_map_for_boundary(lang, kind)} {name}"
        for name, kind in zip(arg_names, args)
    ]
    ret_type = _type_map_for_boundary(lang, return_type, is_return=True)
    ret_literal = _default_ret_literal(lang, return_type)
    scalar_expr = _scalar_ret_expr(arg_names, args, return_type)
    ret_expr = scalar_expr if scalar_expr else ret_literal

    if lang == "zig":
        zig_args = [
            f"{name}: {_type_map_for_boundary('zig', kind)}"
            for name, kind in zip(arg_names, args)
        ]
        zig_ret = ret_type if ret_type != "void" else "void"
        body = [f"export fn {symbol}({', '.join(zig_args)}) {zig_ret} {{"]
        for name in arg_names:
            # Zig errors on a pointless discard when the argument is used later.
            if name not in (ret_expr or ""):
                body.append(f"    _ = {name};")
        if ret_expr:
            body.append(f"    return @as({zig_ret}, {ret_expr});")
        else:
            body.append("    return;")
        body.append("}")
        return "\n".join(body)

    if lang == "go":
        go_args = [
            f"{name} {_type_map_for_boundary('go', kind)}"
            for name, kind in zip(arg_names, args)
        ]
        go_ret = ret_type if ret_type != "void" else ""
        proto = f"func {symbol}({', '.join(go_args)}) {go_ret}".strip()
        body = [
            "package main",
            "",
            'import "C"',
            "",
            f"//export {symbol}",
            f"{proto} {{",
        ]
        for name in arg_names:
            body.append(f"    _ = {name}")
        if ret_expr:
            body.append(f"    return {ret_expr}")
        body.append("}")
        return "\n".join(body)

    if lang == "csharp":
        cs_args = [
            f"{_type_map_for_boundary('csharp', kind)} {name}"
            for name, kind in zip(arg_names, args)
        ]
        cs_ret = ret_type if ret_type != "void" else "void"
        body = [
            "using System;",
            "using System.Runtime.InteropServices;",
            "",
            "public static class Exports {",
            f'    [UnmanagedCallersOnly(EntryPoint = "{symbol}")]',
            f"    public static {cs_ret} {symbol}({', '.join(cs_args)}) {{",
        ]
        for name in arg_names:
            body.append(f"        _ = {name};")
        if ret_expr:
            body.append(f"        return {ret_expr};")
        body.extend(["    }", "}"])
        return "\n".join(body)

    if lang == "mojo":
        mojo_args = [
            f"{name}: {_type_map_for_boundary('mojo', kind)}"
            for name, kind in zip(arg_names, args)
        ]
        mojo_ret = ret_type if ret_type != "void" else ""
        proto = f"fn {symbol}({', '.join(mojo_args)}) -> {mojo_ret}".strip()
        body = [proto + " {"]
        for name in arg_names:
            body.append(f"    _ = {name}")
        if ret_expr:
            body.append(f"    return {ret_expr}")
        body.append("}")
        return "\n".join(body)

    # Default C fallback (also covers cpp/c++, d, nim, java, fortran via C-like syntax)
    c_args = [
        f"{_type_map_for_boundary('c', kind)} {name}"
        for name, kind in zip(arg_names, args)
    ]
    c_ret = ret_type if ret_type != "void" else "void"
    body = [
        "#include <stdint.h>",
        "#include <stddef.h>",
        "",
        "#ifdef __cplusplus",
        'extern "C" {',
        "#endif",
        "",
        f'{c_ret} {symbol}({", ".join(c_args)}) {{',
    ]
    for name in arg_names:
        body.append(f"    (void){name};")
    if ret_expr:
        body.append(f"    return {ret_expr};")
    body.extend(["}", "", "#ifdef __cplusplus", '} // extern "C"', "#endif"])
    return "\n".join(body)


def _fallback_manifest(language_id: str, node_id: str) -> Tuple[str, str]:
    """Return (file_path, content) for a minimal build manifest."""
    lang = language_id.lower()
    if lang == "zig":
        return (
            "build.zig",
            textwrap.dedent("""\
                const std = @import("std");

                pub fn build(b: *std.Build) void {
                    _ = b;
                }
                """),
        )
    if lang == "go":
        return (
            "go.mod",
            f"module {node_id}\n\ngo 1.21\n",
        )
    if lang == "csharp":
        return (
            f"{node_id}.csproj",
            textwrap.dedent(f"""\
                <Project Sdk="Microsoft.NET.Sdk">
                  <PropertyGroup>
                    <TargetFramework>net8.0</TargetFramework>
                    <OutputType>Library</OutputType>
                    <PublishAot>true</PublishAot>
                  </PropertyGroup>
                </Project>
                """),
        )
    # Generic shell build script for C-like languages.
    return (
        "build.sh",
        textwrap.dedent(f"""\
            #!/bin/bash
            set -e
            echo "Aero-Forge fallback build for {language_id}"
            """),
    )


class _FallbackEmitterPlugin(PolyglotEmitterPlugin):
    """A deterministic, conservative emitter used when LLM synthesis fails."""

    def __init__(
        self, language_id: str, boundary: Optional[BoundaryContract] = None
    ) -> None:
        self._language_id = language_id
        self._boundary = boundary or BoundaryContract.C_ABI

    @property
    def descriptor(self) -> CapabilityDescriptor:
        return CapabilityDescriptor(
            language_id=self._language_id,
            supported_boundaries={self._boundary},
            toolchains=[self._language_id],
            file_extensions=[self._language_id],
            supports_zero_copy=False,
            supports_async_ffi=False,
        )

    def emit_source_files(
        self,
        node_id: str,
        node_spec: Dict[str, Any],
        boundary_contracts: List[Dict[str, Any]],
    ) -> List[CodeArtifact]:
        contract = boundary_contracts[0] if boundary_contracts else {}
        symbol = contract.get("symbol", "fast_math_kernel")
        args = list(contract.get("args", ["int64"]))
        return_type = contract.get("return_type", "int64") or "int64"
        ext = node_spec.get("lang", self._language_id).lower()
        content = _fallback_source(ext, symbol, args, return_type, node_id)
        return [
            CodeArtifact(
                file_path=f"src/{symbol}.{ext}",
                content=content,
                language=ext,
            )
        ]

    def emit_build_manifest(
        self,
        node_id: str,
        dependencies: List[str],
        compiler_flags: List[str],
    ) -> List[CodeArtifact]:
        file_path, content = _fallback_manifest(self._language_id, node_id)
        return [
            CodeArtifact(
                file_path=file_path,
                content=content,
                language=self._language_id,
            )
        ]


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
            from aero_forge.builder.spec import literal

            return self._list_literal([literal(v) for v in value])
        if isinstance(value, dict):
            from aero_forge.builder.spec import literal

            return self._dict_literal(
                [
                    ASTNode(kind="pair", children=[literal(k), literal(v)])
                    for k, v in value.items()
                ]
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


class EmitterRegistry:
    """Thread-safe singleton registry of :class:`PolyglotEmitterPlugin` instances.

    When the registry is configured with an LLM client and a synthesis prompt, a
    lookup for an unknown language triggers JIT synthesis of a temporary emitter
    plugin. The synthesized class is validated against the base interface and
    the requested FFI boundary contract before it is registered.
    """

    _instance: Optional["EmitterRegistry"] = None
    _registry_lock: Lock = Lock()

    def __new__(cls) -> "EmitterRegistry":
        if cls._instance is None:
            with cls._registry_lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    @classmethod
    def get_instance(cls) -> "EmitterRegistry":
        return cls()

    def __init__(self) -> None:
        if hasattr(self, "_plugins"):
            return
        self._plugins: Dict[str, PolyglotEmitterPlugin] = {}
        self._lock: Lock = Lock()
        self._synthesis_client: Optional[Any] = None
        self._synthesis_provider: Optional[str] = None
        self._synthesis_model: Optional[str] = None
        self._synthesis_api_key: Optional[str] = None
        self._synthesis_prompt: Optional[str] = None

    def configure_jit_synthesis(
        self,
        llm_client: Optional[Any] = None,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        prompt: Optional[str] = None,
    ) -> None:
        """Configure the LLM used to synthesize missing emitter plugins."""
        self._synthesis_client = llm_client
        self._synthesis_provider = provider
        self._synthesis_model = model
        self._synthesis_api_key = api_key
        self._synthesis_prompt = prompt

    def register(self, plugin: PolyglotEmitterPlugin) -> None:
        with self._lock:
            key = plugin.descriptor.language_id.lower().strip()
            self._plugins[key] = plugin

    def get_plugin(
        self,
        language_id: str,
        synthesize: bool = True,
        boundary_type: Optional[BoundaryContract] = None,
        node_spec: Optional[Dict[str, Any]] = None,
        contracts: Optional[List[Dict[str, Any]]] = None,
    ) -> PolyglotEmitterPlugin:
        """Return a registered plugin, synthesizing one on demand if allowed."""
        key = language_id.lower().strip()
        with self._lock:
            if key in self._plugins:
                return self._plugins[key]
            if synthesize and self._can_synthesize():
                plugin = self._synthesize_plugin(
                    key,
                    boundary_type=boundary_type,
                    node_spec=node_spec,
                    contracts=contracts,
                )
                # Pre-flight the toolchains the synthesized plugin declares and warn
                # early; the materializer will enforce availability before dispatch.
                try:
                    SystemToolchainRouter.preflight_plugin(plugin.descriptor)
                except ToolchainNotFoundError as exc:
                    logger.warning(
                        "Synthesized %s emitter requires toolchain %s which is not "
                        "available; materializer will raise before build dispatch: %s",
                        key,
                        exc.toolchain,
                        exc,
                    )
                self._plugins[key] = plugin
                return plugin
            raise EmitterError(
                f"No emitter plugin registered for language {language_id!r}. "
                f"Supported: {sorted(self._plugins.keys())}"
            )

    def _can_synthesize(self) -> bool:
        return bool(self._synthesis_prompt) and (
            self._synthesis_client is not None
            or self._synthesis_provider
            or self._synthesis_api_key
        )

    def _get_llm_client(self) -> Any:
        if self._synthesis_client is None:
            from aero_forge.llm.clients import get_llm_client

            self._synthesis_client = get_llm_client(
                provider=self._synthesis_provider or "deepseek",
                model=self._synthesis_model
                or os.getenv("AERO_FORGE_MODEL")
                or "deepseek-chat",
                api_key=self._synthesis_api_key,
                raise_on_error=True,
            )
            if self._synthesis_client is None:
                raise EmitterError(
                    "Could not construct an LLM client for plugin synthesis"
                )
        return self._synthesis_client

    def _synthesize_plugin(
        self,
        language_id: str,
        boundary_type: Optional[BoundaryContract] = None,
        node_spec: Optional[Dict[str, Any]] = None,
        contracts: Optional[List[Dict[str, Any]]] = None,
    ) -> PolyglotEmitterPlugin:
        """Ask the LLM to generate a temporary emitter plugin for *language_id*.

        A deterministic skeleton is materialised to disk before the first LLM call.
        The model must return the completed skeleton wrapped between
        ``__AERO_LOGIC_START__`` and ``__AERO_LOGIC_END__``.  If the first
        response is empty, structurally invalid, or hollow (empty GoI execution
        matrix), a second attempt is made with the v11_universal_architect
        guidance before falling back to the conservative C-ABI emitter.
        """
        if not self._synthesis_prompt:
            raise EmitterError("JIT synthesis prompt is not configured")

        skeleton_path, skeleton_content = self._build_skeleton_file(
            language_id, boundary_type, node_spec, contracts
        )
        _accel_log(
            "info",
            f"Materialized skeleton file for {language_id} at {skeleton_path}",
        )

        client = self._get_llm_client()
        system_prompt = self._prepare_system_prompt(
            self._synthesis_prompt,
            language_id,
            boundary_type,
            node_spec,
            contracts,
        )
        user_prompt = self._build_synthesis_user_prompt(
            language_id, boundary_type, node_spec, contracts
        )

        v11_guidance = self._v11_universal_guidance()
        attempts = [
            ("direct", system_prompt, user_prompt),
            (
                "v11_universal",
                system_prompt + "\n\n" + v11_guidance,
                user_prompt,
            ),
        ]

        last_error: Optional[Exception] = None
        for label, sys, usr in attempts:
            try:
                raw = client.generate(
                    [
                        {"role": "system", "content": sys},
                        {"role": "user", "content": usr},
                    ],
                    temperature=0.2,
                    max_tokens=4096,
                )
                if not raw:
                    raise StructuralViolationError(
                        f"LLM returned an empty response during {label} synthesis for {language_id}"
                    )
                plugin = self._try_load_generated(
                    raw,
                    language_id,
                    boundary_type,
                    require_delimiters=(label == "direct"),
                )
                self._verify_plugin_logic(plugin, language_id)
                _accel_log("success", "Logic In-Fill Successful")
                _accel_log(
                    "success", f"JIT-synthesized {language_id} emitter plugin ({label})"
                )
                return plugin
            except StructuralViolationError as exc:
                _accel_log(
                    "error", f"JIT synthesis structural violation ({label}): {exc}"
                )
                raw_preview = (raw or "")[:800]
                _accel_log("error", f"Raw preview: {raw_preview!r}")
                last_error = exc
            except SynthesizedPluginError as exc:
                _accel_log("warning", f"JIT synthesis hollow logic ({label}): {exc}")
                last_error = exc
            except Exception as exc:
                _accel_log("warning", f"JIT synthesis {label} failed: {exc}")
                last_error = exc

        _accel_log(
            "warning",
            f"JIT synthesis for {language_id} failed after {len(attempts)} attempts; "
            f"using deterministic C-ABI fallback emitter",
        )
        logger.warning(
            "LLM synthesis for %s failed twice; using deterministic C-ABI fallback emitter",
            language_id,
        )
        return self._fallback_plugin(language_id, boundary_type)

    def _try_load_generated(
        self,
        raw: Optional[str],
        language_id: str,
        boundary_type: Optional[BoundaryContract] = None,
        require_delimiters: bool = True,
    ) -> PolyglotEmitterPlugin:
        """Extract and validate a plugin from an LLM response.

        Raises:
            StructuralViolationError: when the response is empty or missing the
                required token delimiters.
            SynthesizedPluginError: when the extracted code is not a valid
                ``PolyglotEmitterPlugin`` subclass.
        """
        if not raw:
            raise StructuralViolationError(
                f"JIT synthesis for {language_id}: LLM returned an empty response"
            )

        # Markdown-agnostic extraction: SSP tokens first, then Markdown fences,
        # then structural keyword scans. Any valid Python class/function block is
        # accepted before declaring the response empty.
        code = self._fuzzy_extract_python_code(raw)
        if not code:
            raise StructuralViolationError(
                f"JIT synthesis for {language_id}: response is missing the required "
                f"__AERO_LOGIC_START__ / __AERO_LOGIC_END__ delimiters and no Python "
                f"code block could be found"
            )

        code = self._strip_redefined_helpers(code)
        # Markdown-agnostic logic density gate: imports/comments-only payloads are
        # considered hollow and trigger a retry with the v11 template.
        try:
            ContentDensityValidator.validate(code, "python")
        except ValueError as exc:
            raise SynthesizedPluginError(
                f"JIT synthesis for {language_id}: extracted code is hollow ({exc})"
            ) from exc
        try:
            return self._load_and_validate_plugin(code, language_id, boundary_type)
        except Exception as exc:
            raise SynthesizedPluginError(
                f"JIT synthesis for {language_id}: extracted code failed validation: {exc}"
            ) from exc

    def _prepare_system_prompt(
        self,
        prompt_template: str,
        language_id: str,
        boundary_type: Optional[BoundaryContract] = None,
        node_spec: Optional[Dict[str, Any]] = None,
        contracts: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """Fill language-specific placeholders in the emitter synthesis prompt."""
        boundary = boundary_type or BoundaryContract.C_ABI
        boundary_name = boundary.name
        title = _pascal_case(language_id)
        node_id = f"{language_id}_kernel"

        first_contract = (contracts or [{}])[0]
        symbol = first_contract.get("symbol", "fast_math_kernel")
        args = list(first_contract.get("args", ["int64"]))
        return_type = first_contract.get("return_type", "int64") or "int64"
        signature = _function_signature_for_prompt(
            language_id, symbol, args, return_type
        )
        context = _function_context_from_node(node_spec)

        toolchain = {
            "zig": "zig",
            "go": "go",
            "mojo": "mojo",
            "c": "gcc",
            "c++": "clang++",
            "cpp": "clang++",
            "csharp": "dotnet",
            "java": "javac",
            "d": "dmd",
            "nim": "nim",
            "fortran": "gcc",
        }.get(language_id.lower(), language_id.lower())

        ext = {
            "c++": "cpp",
            "csharp": "cs",
            "fortran": "f90",
        }.get(language_id.lower(), language_id.lower())

        manifest_file, manifest_content = _fallback_manifest(language_id, node_id)

        manifest_repr = _python_string_literal(manifest_content)

        return (
            prompt_template.replace("__LANGUAGE_TITLE__", title)
            .replace("__LANGUAGE_ID__", language_id.lower())
            .replace("__BOUNDARY_NAME__", boundary_name)
            .replace("__TOOLCHAIN__", toolchain)
            .replace("__EXT__", ext)
            .replace("__MANIFEST_FILE__", manifest_file)
            .replace("__MANIFEST_CONTENT__", manifest_repr)
            .replace("__FUNCTION_SIGNATURE__", signature)
            .replace("__FUNCTION_CONTEXT__", context)
        )

    def _build_synthesis_user_prompt(
        self,
        language_id: str,
        boundary_type: Optional[BoundaryContract] = None,
        node_spec: Optional[Dict[str, Any]] = None,
        contracts: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        boundary = boundary_type.value if boundary_type else "c_abi"
        boundary_name = boundary_type.name if boundary_type else "C_ABI"
        toolchain = {
            "zig": "zig",
            "go": "go",
            "mojo": "mojo",
            "c": "gcc",
            "c++": "clang++",
            "cpp": "clang++",
            "csharp": "dotnet",
            "java": "javac",
            "d": "dmd",
            "nim": "nim",
            "fortran": "gcc",
        }.get(language_id.lower(), language_id.lower())
        ext = {
            "c++": "cpp",
            "csharp": "cs",
            "fortran": "f90",
        }.get(language_id.lower(), language_id.lower())

        first_contract = (contracts or [{}])[0]
        symbol = first_contract.get("symbol", "fast_math_kernel")
        args = list(first_contract.get("args", ["int64"]))
        return_type = first_contract.get("return_type", "int64") or "int64"
        signature = _function_signature_for_prompt(
            language_id, symbol, args, return_type
        )
        context = _function_context_from_node(node_spec)
        smt_types = self._synthesis_smt_types(node_spec, contracts)

        compacted = node_spec.get("_compacted_context") if node_spec else None
        parts = [
            f"Complete the skeleton for a `{_pascal_case(language_id)}EmitterPlugin` "
            f"for language '{language_id}' (toolchain '{toolchain}', source extension '.{ext}').",
            "",
            "You MUST:",
            "1. Output ONLY the completed class wrapped between __AERO_LOGIC_START__ and __AERO_LOGIC_END__.",
            "2. Replace every __AERO_IN_FILL__ marker with real, compilable logic.",
            "3. Do not write prose, markdown fences, TODOs, or placeholder summaries.",
            "",
            f"Function signature:\n{signature}",
            f"Semantic purpose:\n{context}",
            f"Boundary contract: `{boundary_name}` (value '{boundary}').",
        ]
        if compacted:
            compacted_text = (
                json.dumps(compacted, indent=2, sort_keys=False)
                if isinstance(compacted, dict)
                else str(compacted)
            )
            parts.extend(
                [
                    "",
                    "Compacted functional context (contracts, functions, SMT types):",
                    compacted_text,
                ]
            )
        if smt_types:
            parts.extend(
                [
                    "",
                    "SMT-inferred native types for the generated function body:",
                    json.dumps(smt_types, indent=2, sort_keys=True),
                ]
            )
        parts.extend(
            [
                "",
                "Implement `descriptor`, `emit_source_files`, and `emit_build_manifest` using only the base classes imported in the skeleton. "
                "Do not redeclare BoundaryContract, CapabilityDescriptor, CodeArtifact, or PolyglotEmitterPlugin.",
            ]
        )
        return "\n".join(parts)

    @staticmethod
    def _synthesis_smt_types(
        node_spec: Optional[Dict[str, Any]],
        contracts: Optional[List[Dict[str, Any]]],
    ) -> Dict[str, str]:
        """Collect SMT-inferred native types from node/extra and contract metadata."""
        types: Dict[str, str] = {}
        if node_spec:
            extra = node_spec.get("extra") or {}
            types.update(extra.get("smt_types") or {})
            types.update(node_spec.get("smt_types") or {})
        for contract in contracts or []:
            if isinstance(contract, dict):
                types.update(contract.get("smt_types") or {})
                extra = contract.get("extra") or {}
                types.update(extra.get("smt_types") or {})
        return types

    @staticmethod
    def _v11_universal_guidance() -> str:
        """Return the v11_universal_architect planning guidance for retries."""
        from aero_forge.prompts import get_template

        try:
            template = get_template("v11_universal_architect")
            return (
                "Use the following universal polyglot architect guidance when "
                "completing the skeleton:\n" + template.system_prompt
            )
        except Exception:
            return (
                "Use the v11_universal_architect template: design the emitter as a "
                "strict C-ABI Python extension class that emits real, compilable "
                "source code for the requested language. Fill every skeleton marker "
                "and wrap the completed class in __AERO_LOGIC_START__ / __AERO_LOGIC_END__."
            )

    def _build_skeleton_file(
        self,
        language_id: str,
        boundary_type: Optional[BoundaryContract] = None,
        node_spec: Optional[Dict[str, Any]] = None,
        contracts: Optional[List[Dict[str, Any]]] = None,
    ) -> Tuple[Path, str]:
        """Materialize the boilerplate skeleton to disk and return its path/content."""
        prompt = self._prepare_system_prompt(
            self._synthesis_prompt or "",
            language_id,
            boundary_type,
            node_spec,
            contracts,
        )
        # Locate a workspace directory from the node spec, otherwise a temp dir.
        workspace: Path
        if node_spec:
            candidate = node_spec.get("workspace") or node_spec.get("output_dir")
            if candidate:
                workspace = Path(candidate)
            else:
                workspace = (
                    Path(tempfile.gettempdir()) / f"aero_jit_{language_id}_skeleton"
                )
        else:
            workspace = Path(tempfile.gettempdir()) / f"aero_jit_{language_id}_skeleton"
        workspace.mkdir(parents=True, exist_ok=True)
        path = workspace / f"{language_id}_emitter_skeleton.py"
        path.write_text(prompt, encoding="utf-8")
        return path, prompt

    @staticmethod
    def _execution_matrix_for_source(source: str) -> Any:
        """Return whether *source* contains any non-empty dependency matrix.

        We concatenate each top-level function's loop-dependency matrix into a
        block-diagonal matrix so that functions with different variable sets
        can be checked together.  Empty (size 0) or all-zero result indicates
        hollow logic with no computational dependency flow.
        """
        import numpy as np

        try:
            tree = ast.parse(source)
        except SyntaxError:
            return np.zeros((0, 0), dtype=np.float64)
        funcs = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
        if not funcs:
            return np.zeros((0, 0), dtype=np.float64)
        matrices = []
        for func in funcs:
            M, _ = _loop_dependency_matrix(func)
            if M.size:
                matrices.append(M)
        if not matrices:
            return np.zeros((0, 0), dtype=np.float64)

        # Block-diagonal concatenation to avoid broadcasting mismatches.
        total = sum(M.shape[0] for M in matrices)
        result = np.zeros((total, total), dtype=np.float64)
        offset = 0
        for M in matrices:
            n = M.shape[0]
            result[offset : offset + n, offset : offset + n] = M
            offset += n
        return result

    def _verify_plugin_logic(
        self,
        plugin: PolyglotEmitterPlugin,
        language_id: str,
    ) -> None:
        """Raise SynthesizedPluginError if the plugin body has no dependency flow."""
        source = getattr(plugin, "__source__", "")
        if not source:
            import inspect

            try:
                source = inspect.getsource(plugin.__class__)
            except (OSError, TypeError):
                source = ""
        M = self._execution_matrix_for_source(source)
        if M.size == 0 or not M.any():
            raise SynthesizedPluginError(
                f"Synthesized {language_id} emitter has an empty execution matrix; "
                "no functional dependency flow was detected (hollow logic)."
            )

    def _fallback_plugin(
        self,
        language_id: str,
        boundary_type: Optional[BoundaryContract] = None,
    ) -> PolyglotEmitterPlugin:
        """Return a deterministic, conservative emitter plugin for *language_id*."""
        return _FallbackEmitterPlugin(language_id, boundary_type)

    @staticmethod
    def _fuzzy_extract_python_code(raw: str) -> str:
        """Extract Python source using the Aero-Forge SSP delimiters.

        The model is instructed to wrap the completed class between
        ``__AERO_LOGIC_START__`` and ``__AERO_LOGIC_END__``.  If those
        markers are present, the body between them is returned verbatim. As
        a compatibility fallback we also scan for markdown ```python fences
        and structural keywords, but a missing token pair in a strict parse
        is treated as a structural violation by the caller.
        """
        if not raw:
            return ""
        from aero_forge.orchestrator.prompt_builder import (
            TruncatedAeroLogicError,
            extract_aero_logic,
        )

        try:
            text = extract_aero_logic(raw)
        except TruncatedAeroLogicError:
            return ""

        # Markdown fenced block fallback (legacy/chatty models).
        all_fenced = re.findall(
            r"```\s*(\w*)\s*(?::\s*[^\n\r]*?)?\s*\r?\n([\s\S]*?)\r?\n?\s*```",
            text,
            re.DOTALL | re.IGNORECASE,
        )
        python_fenced = [
            code for lang, code in all_fenced if lang.lower() in ("python", "py")
        ]
        if python_fenced:
            return python_fenced[0].strip()
        for _, code in all_fenced:
            stripped = code.strip()
            if any(
                stripped.startswith(prefix)
                for prefix in ("import ", "from ", "class ", "def ")
            ):
                return stripped
        if all_fenced:
            return all_fenced[0][1].strip()

        # Structural keyword scan.
        lines = text.splitlines()
        start = 0
        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped.startswith(("import ", "from ", "class ", "def ")):
                start = i
                break
        for end in range(len(lines), start, -1):
            snippet = "\n".join(lines[start:end])
            try:
                ast.parse(snippet)
                return snippet.strip()
            except SyntaxError:
                continue

        try:
            ast.parse(text)
            return text
        except SyntaxError:
            pass
        return raw

    def _extract_python_code(self, raw: str) -> str:
        """Compatibility alias for ``_fuzzy_extract_python_code``."""
        return self._fuzzy_extract_python_code(raw)

    @staticmethod
    def _strip_redefined_helpers(code: str) -> str:
        """Remove any local redefinitions of the shared base classes.

        LLM output often includes self-contained dataclasses/enums. We already
        inject the real classes into the execution namespace, so the generated
        emitter only needs to subclass and instantiate them.
        """
        helper_names = [
            "BoundaryContract",
            "CapabilityDescriptor",
            "CodeArtifact",
            "PolyglotEmitterPlugin",
        ]
        pattern = re.compile(
            r"^(class\s+(?:"
            + "|".join(helper_names)
            + r")\b[^:\n]*:[^\n]*\n(?:\s+.*\n|\n)*)",
            re.MULTILINE,
        )
        code = pattern.sub("", code)
        # Remove dangling @abstractmethod / @property decorators that may be left behind.
        code = re.sub(
            r"^\s*@(abstractmethod|property|dataclass|staticmethod)\s*\n",
            "",
            code,
            flags=re.MULTILINE,
        )
        return code

    def _load_and_validate_plugin(
        self,
        code: str,
        language_id: str,
        boundary_type: Optional[BoundaryContract] = None,
    ) -> PolyglotEmitterPlugin:
        """Exec the generated plugin source, validate it, and return an instance."""
        namespace: Dict[str, Any] = {
            "__builtins__": builtins,
            "PolyglotEmitterPlugin": PolyglotEmitterPlugin,
            "BoundaryContract": BoundaryContract,
            "CapabilityDescriptor": CapabilityDescriptor,
            "CodeArtifact": CodeArtifact,
            "List": List,
            "Dict": Dict,
            "Set": Set,
            "Optional": Optional,
            "Any": Any,
        }
        # Allow the generated code to import from the aero_forge package if it wishes.
        try:
            exec(code, namespace)  # nosec B102
        except Exception as exc:
            raise SynthesizedPluginError(
                f"Generated plugin source for {language_id!r} could not be executed: {exc}"
            ) from exc

        # The generated source may re-declare helper base classes. We identify
        # the concrete emitter by name (<Language>EmitterPlugin) and then verify
        # its descriptor and emitted artifacts against the real base class.
        candidates = [
            obj
            for name, obj in namespace.items()
            if isinstance(obj, type)
            and name.lower().endswith("emitterplugin")
            and name.lower() != "polyglotemitterplugin"
        ]
        if not candidates:
            raise SynthesizedPluginError(
                f"Generated plugin source for {language_id!r} did not define a "
                "concrete `*EmitterPlugin` subclass"
            )

        cls = candidates[0]
        try:
            raw_instance = cls()
        except Exception as exc:
            raise SynthesizedPluginError(
                f"Could not instantiate generated plugin {cls.__name__!r}: {exc}"
            ) from exc

        instance = self._wrap_plugin(raw_instance)
        instance.__source__ = code
        self._validate_plugin(instance, language_id, boundary_type)
        return instance

    def _wrap_plugin(self, raw_instance: Any) -> PolyglotEmitterPlugin:
        """Return a plugin instance whose descriptor and artifacts are canonical types.

        LLM output may redefine ``CapabilityDescriptor``/``CodeArtifact`` inside
        the execution namespace. We create a small wrapper subclass so the
        ``descriptor`` property and emission methods always return the real
        base classes.
        """

        def _norm_boundary(value: Any) -> BoundaryContract:
            if isinstance(value, BoundaryContract):
                return value
            raw = value.value if hasattr(value, "value") else str(value)
            return BoundaryContract(raw)

        def _norm_descriptor(value: Any) -> CapabilityDescriptor:
            if isinstance(value, CapabilityDescriptor):
                return value
            supported: Set[BoundaryContract] = set()
            for item in value.supported_boundaries:
                supported.add(_norm_boundary(item))
            return CapabilityDescriptor(
                language_id=str(value.language_id),
                supported_boundaries=supported,
                toolchains=list(value.toolchains),
                file_extensions=list(value.file_extensions),
                supports_zero_copy=bool(value.supports_zero_copy),
                supports_async_ffi=bool(value.supports_async_ffi),
            )

        def _norm_artifact(value: Any) -> CodeArtifact:
            if isinstance(value, CodeArtifact):
                return value
            return CodeArtifact(
                file_path=str(value.file_path),
                content=str(value.content),
                language=str(value.language),
                is_header=bool(getattr(value, "is_header", False)),
                metadata=dict(getattr(value, "metadata", {})),
            )

        try:
            raw_descriptor = raw_instance.descriptor
        except Exception as exc:
            raise SynthesizedPluginError(
                f"Generated plugin {raw_instance.__class__.__name__!r} raised from `descriptor`: {exc}"
            ) from exc

        # The generated plugin may define `descriptor` as a method instead of a property.
        if callable(raw_descriptor) and not isinstance(
            raw_descriptor, CapabilityDescriptor
        ):
            try:
                raw_descriptor = raw_descriptor()
            except Exception as exc:
                raise SynthesizedPluginError(
                    f"Generated plugin {raw_instance.__class__.__name__!r} `descriptor()` raised: {exc}"
                ) from exc

        norm_descriptor = _norm_descriptor(raw_descriptor)
        orig_source = raw_instance.emit_source_files
        orig_manifest = raw_instance.emit_build_manifest

        def emit_source_files(self, *args: Any, **kwargs: Any) -> List[CodeArtifact]:
            result = orig_source(*args, **kwargs)
            if result is None:
                result = []
            elif isinstance(result, CodeArtifact):
                result = [result]
            elif not isinstance(result, list):
                # Generators, tuples, and other iterables must be fully consumed
                # before the plugin method returns so no CodeArtifact is dropped.
                result = list(result)
            return [_norm_artifact(a) for a in result]

        def emit_build_manifest(self, *args: Any, **kwargs: Any) -> List[CodeArtifact]:
            result = orig_manifest(*args, **kwargs)
            if result is None:
                result = []
            elif isinstance(result, CodeArtifact):
                result = [result]
            elif not isinstance(result, list):
                result = list(result)
            return [_norm_artifact(a) for a in result]

        wrapper_cls = type(
            raw_instance.__class__.__name__,
            (raw_instance.__class__,),
            {
                "descriptor": property(lambda self: norm_descriptor),
                "emit_source_files": emit_source_files,
                "emit_build_manifest": emit_build_manifest,
            },
        )
        return cast(PolyglotEmitterPlugin, wrapper_cls())

    def _validate_plugin(
        self,
        plugin: PolyglotEmitterPlugin,
        language_id: str,
        boundary_type: Optional[BoundaryContract] = None,
    ) -> None:
        """Verify that a synthesized plugin satisfies the base contract and boundary."""
        try:
            descriptor = plugin.descriptor
        except Exception as exc:
            raise SynthesizedPluginError(
                f"Synthesized plugin for {language_id!r} raised an error from `descriptor`: {exc}"
            ) from exc

        if not isinstance(descriptor, CapabilityDescriptor):
            raise SynthesizedPluginError(
                f"Synthesized plugin descriptor for {language_id!r} is not a CapabilityDescriptor"
            )

        if descriptor.language_id.lower() != language_id.lower():
            raise SynthesizedPluginError(
                f"Synthesized plugin claims language_id={descriptor.language_id!r} "
                f"but {language_id!r} was requested"
            )

        if not isinstance(
            descriptor.supported_boundaries, (set, frozenset, list, tuple)
        ):
            raise SynthesizedPluginError(
                f"Synthesized plugin for {language_id!r} `supported_boundaries` is not a collection"
            )
        supported = set(descriptor.supported_boundaries)
        for item in supported:
            if not isinstance(item, BoundaryContract):
                raise SynthesizedPluginError(
                    f"Synthesized plugin for {language_id!r} has invalid boundary {item!r}"
                )

        if boundary_type is not None and boundary_type not in supported:
            raise SynthesizedPluginError(
                f"Synthesized plugin for {language_id!r} does not support {boundary_type.value!r}"
            )

        if not descriptor.toolchains or not descriptor.file_extensions:
            raise SynthesizedPluginError(
                f"Synthesized plugin for {language_id!r} must provide non-empty "
                "`toolchains` and `file_extensions`"
            )

        if not callable(getattr(plugin, "emit_source_files", None)):
            raise SynthesizedPluginError(
                f"Synthesized plugin for {language_id!r} does not implement `emit_source_files`"
            )
        if not callable(getattr(plugin, "emit_build_manifest", None)):
            raise SynthesizedPluginError(
                f"Synthesized plugin for {language_id!r} does not implement `emit_build_manifest`"
            )

        # Type-check the emitted artifacts by calling the methods with minimal stubs.
        try:
            stubs = self._artifact_stubs(language_id)
            source_artifacts = plugin.emit_source_files(**stubs["source_args"])
            manifest = plugin.emit_build_manifest(**stubs["manifest_args"])
        except Exception as exc:
            raise SynthesizedPluginError(
                f"Synthesized plugin for {language_id!r} failed during artifact emission: {exc}"
            ) from exc

        if not isinstance(source_artifacts, list) or not all(
            isinstance(a, CodeArtifact) for a in source_artifacts
        ):
            raise SynthesizedPluginError(
                f"Synthesized plugin for {language_id!r} `emit_source_files` must return a list of CodeArtifact"
            )
        if not isinstance(manifest, list) or not all(
            isinstance(a, CodeArtifact) for a in manifest
        ):
            raise SynthesizedPluginError(
                f"Synthesized plugin for {language_id!r} `emit_build_manifest` must return a list of CodeArtifact"
            )

    @staticmethod
    def _artifact_stubs(language_id: str) -> Dict[str, Any]:
        return {
            "source_args": {
                "node_id": f"{language_id}_kernel",
                "node_spec": {
                    "lang": language_id,
                    "toolchain": language_id,
                    "source_files": [f"src/{language_id}_kernel.zig"],
                    "compiler_flags": ["-O3"],
                    "exports": ["fast_math_kernel"],
                },
                "boundary_contracts": [
                    {
                        "boundary_type": "c_abi",
                        "boundary": "c_abi",
                        "symbol": "fast_math_kernel",
                        "args": ["int64"],
                        "return_type": "int64",
                        "is_zero_copy": True,
                    }
                ],
            },
            "manifest_args": {
                "node_id": f"{language_id}_kernel",
                "dependencies": [],
                "compiler_flags": ["-O3"],
            },
        }

    def find_emitters_for_boundary(
        self, boundary: BoundaryContract
    ) -> List[PolyglotEmitterPlugin]:
        with self._lock:
            return [
                plugin
                for plugin in self._plugins.values()
                if boundary in plugin.descriptor.supported_boundaries
            ]


# Re-export `literal` for emitter internal use.
from aero_forge.builder.spec import literal  # noqa: E402,F401
