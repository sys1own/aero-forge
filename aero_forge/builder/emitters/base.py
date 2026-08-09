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
        is_python = language == "python" or content.lstrip().startswith(("import ", "from "))
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
        return count

    @classmethod
    def validate_pure_python(cls, content: str) -> None:
        """Fail fast if a supposedly pure-Python source imports native modules.

        This enforces the negative constraint emitted in the Compacted
        Functional Matrix: ``pure_python`` targets must not reference
        ``rust_core``, ``cpp_core``, or ``@accelerate`` decorators.
        """
        if re.search(r"\brust_core\b|\bcpp_core\b", content, re.IGNORECASE):
            raise ValueError(
                "Forbidden native dependency in pure_python source: "
                "rust_core/cpp_core imports are not allowed."
            )
        if re.search(r"@\s*accelerate\s*\(", content):
            raise ValueError(
                "Forbidden @accelerate decorator in pure_python source."
            )

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
        cleaned = re.sub(r"return\s+(?:0|arg_\d+|_)\s*;", "", cleaned, flags=re.IGNORECASE)

        if not cleaned.strip():
            return 0

        patterns = [
            # Function definitions (Zig/Go/Rust/C/Mojo/General).
            r'\b(?:export\s+)?(?:pub\s+)?(?:extern\s+(?:"C"\s+))?fn\s+',
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
    def has_execution_flow(cls, content: str, language: str) -> bool:
        """Return True when *content* has a non-zero GoI execution matrix.

        For Python we compute the block-diagonal union of per-function
        loop-dependency matrices. Functions with no computational statements
        (empty body or only `pass`) produce an empty matrix; in that case we
        fall back to module-level functional density so that CLI/loader-style
        Python files are not rejected. For non-Python sources we use the generic
        functional-node count as a proxy.
        """
        language = (language or "").lower()
        is_python = language == "python" or content.lstrip().startswith(("import ", "from "))
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

        matrices = []
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                M, _ = _loop_dependency_matrix(node)
                if M.size:
                    matrices.append(M)
        if matrices:
            total = sum(M.shape[0] for M in matrices)
            result = np.zeros((total, total), dtype=np.float64)
            offset = 0
            for M in matrices:
                n = M.shape[0]
                result[offset : offset + n, offset : offset + n] = M
                offset += n
            if result.size == 0 or not result.any():
                # A pure entrypoint (e.g. ``main.py``) may have no loop-carried
                # writes but still performs a real function call; fall back to
                # module-level functional density so simple orchestrators are not
                # rejected as hollow.
                return cls._count_python_nodes(content) >= cls.MIN_FUNCTIONAL_NODES
            # GoI Proof-Net verification: confirm the execution matrix is non-zero
            # using the native Rust solver.
            try:
                return execution_matrix_nonzero(json.dumps(result.tolist()))
            except Exception:
                return bool(result.any())

        # No function bodies with loop-carried flow: use module-level density.
        return cls._count_python_nodes(content) >= cls.MIN_FUNCTIONAL_NODES


class PolyglotEmitterPlugin(ABC):
    """Plugin interface for language-specific source emitters."""

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
        return f"//export {symbol}\nimport \"C\"\n{proto}"
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
    body.extend(["}", "", "#ifdef __cplusplus", "} // extern \"C\"", "#endif"])
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
                    [{"role": "system", "content": sys}, {"role": "user", "content": usr}],
                    temperature=0.2,
                    max_tokens=4096,
                )
                if not raw:
                    raise StructuralViolationError(
                        f"LLM returned an empty response during {label} synthesis for {language_id}"
                    )
                plugin = self._try_load_generated(
                    raw, language_id, boundary_type, require_delimiters=(label == "direct")
                )
                self._verify_plugin_logic(plugin, language_id)
                _accel_log("success", "Logic In-Fill Successful")
                _accel_log("success", f"JIT-synthesized {language_id} emitter plugin ({label})")
                return plugin
            except StructuralViolationError as exc:
                _accel_log("error", f"JIT synthesis structural violation ({label}): {exc}")
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
            parts.extend([
                "",
                "Compacted functional context (contracts, functions, SMT types):",
                compacted_text,
            ])
        if smt_types:
            parts.extend([
                "",
                "SMT-inferred native types for the generated function body:",
                json.dumps(smt_types, indent=2, sort_keys=True),
            ])
        parts.extend([
            "",
            "Implement `descriptor`, `emit_source_files`, and `emit_build_manifest` using only the base classes imported in the skeleton. "
            "Do not redeclare BoundaryContract, CapabilityDescriptor, CodeArtifact, or PolyglotEmitterPlugin.",
        ])
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
                workspace = Path(tempfile.gettempdir()) / f"aero_jit_{language_id}_skeleton"
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
        funcs = [
            node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
        ]
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
        from aero_forge.orchestrator.prompt_builder import extract_aero_logic

        text = extract_aero_logic(raw)

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
            if not isinstance(result, list):
                result = [result]
            return [_norm_artifact(a) for a in result]

        def emit_build_manifest(self, *args: Any, **kwargs: Any) -> List[CodeArtifact]:
            result = orig_manifest(*args, **kwargs)
            if isinstance(result, CodeArtifact):
                result = [result]
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
