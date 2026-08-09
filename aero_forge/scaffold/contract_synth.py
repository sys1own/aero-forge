"""Deterministic FFI morphism synthesis for missing blueprint contracts.

``ContractSynthesizer`` takes the ABI contracts declared in a blueprint and
emits canonical wrapper templates for the supported cross-language bindings:

* ``python/rust``  - PyO3 Python/Rust bridge.
* ``rust/cpp``     - Rust ``extern "C"`` wrapper with a C header and ctypes loader.
* ``rust/rust``    - Zero-copy in-process Rust trait transmute.

No LLM or heuristic text generation is used; every stub is derived directly from
the contract name and signature.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from aero_forge.blueprint import ContractEntry


FFI_TYPE_LAYOUTS: Dict[str, Dict[str, Any]] = {
    "int32": {
        "c_type": "int32_t",
        "rust_type": "i32",
        "python_ctype": "c_int32",
        "csharp_type": "int",
        "go_type": "C.int32_t",
        "size": 4,
        "alignment": 4,
    },
    "int64": {
        "c_type": "int64_t",
        "rust_type": "i64",
        "python_ctype": "c_int64",
        "csharp_type": "long",
        "go_type": "C.int64_t",
        "size": 8,
        "alignment": 8,
    },
    "float32": {
        "c_type": "float",
        "rust_type": "f32",
        "python_ctype": "c_float",
        "csharp_type": "float",
        "go_type": "C.float",
        "size": 4,
        "alignment": 4,
    },
    "float64": {
        "c_type": "double",
        "rust_type": "f64",
        "python_ctype": "c_double",
        "csharp_type": "double",
        "go_type": "C.double",
        "size": 8,
        "alignment": 8,
    },
    "pointer": {
        "c_type": "void*",
        "rust_type": "*mut c_void",
        "python_ctype": "c_void_p",
        "csharp_type": "IntPtr",
        "go_type": "unsafe.Pointer",
        "size": 8,
        "alignment": 8,
    },
}


# Simplified type maps used for canonical stub generation.  They are
# intentionally conservative (defaulting to float/double scalars) because the
# synthesized wrapper is a scaffold meant to be validated and refined.
_RUST_TYPE: Dict[str, str] = {
    "float": "f64",
    "double": "f64",
    "int": "i64",
    "int64": "i64",
    "i64": "i64",
    "bool": "bool",
    "str": "String",
    "string": "String",
    "list": "Vec<f64>",
    "list[float]": "Vec<f64>",
    "list[int]": "Vec<i64>",
}

_C_TYPE: Dict[str, str] = {
    "float": "double",
    "double": "double",
    "int": "int64_t",
    "int64": "int64_t",
    "i64": "int64_t",
    "bool": "bool",
    "str": "const char*",
    "string": "const char*",
    "list": "const double*",
    "list[float]": "const double*",
    "list[int]": "const int64_t*",
}

_PYTHON_CTYPE: Dict[str, str] = {
    "float": "c_double",
    "double": "c_double",
    "int": "c_int64",
    "int64": "c_int64",
    "i64": "c_int64",
    "bool": "c_bool",
    "str": "c_char_p",
    "string": "c_char_p",
    "list": "POINTER(c_double)",
    "list[float]": "POINTER(c_double)",
    "list[int]": "POINTER(c_int64)",
}


def _normalise_type(type_hint: str) -> str:
    t = (type_hint or "").strip().lower()
    t = re.sub(r"typing\.", "", t)
    t = re.sub(r"[\[\]]", "", t)
    if t.startswith("list["):
        inner = t[5:-1].strip()
        if inner in ("int", "i64", "int64"):
            return "list[int]"
        return "list[float]"
    return t


def _to_rust_type(type_hint: str) -> str:
    return _RUST_TYPE.get(_normalise_type(type_hint), "f64")


def _to_c_type(type_hint: str) -> str:
    return _C_TYPE.get(_normalise_type(type_hint), "double")


def _to_python_ctype(type_hint: str) -> str:
    return _PYTHON_CTYPE.get(_normalise_type(type_hint), "c_double")


def _parse_signature(signature: str) -> Tuple[str, List[Tuple[str, str]], str]:
    """Parse a contract signature string into (name, args, return_type).

    Supports forms such as ``add(a: float, b: float) -> float`` or
    ``fn add(a: float, b: float) -> float``.
    """
    sig = signature.strip() if signature else ""
    if not sig:
        return "", [], "float"

    sig = re.sub(r"^fn\s+", "", sig)
    m = re.match(r"(\w+)\s*\(([^)]*)\)\s*(?:->\s*(\S+))?", sig)
    if not m:
        return sig.split("(")[0].strip() or "entry", [], "float"

    name = m.group(1)
    args_str = m.group(2).strip()
    ret = (m.group(3) or "float").strip()

    args: List[Tuple[str, str]] = []
    if args_str:
        for part in args_str.split(","):
            part = part.strip()
            if not part:
                continue
            if ":" in part:
                arg_name, arg_type = part.split(":", 1)
                args.append((arg_name.strip(), arg_type.strip()))
            else:
                args.append((part, "float"))
    return name, args, ret


def _rust_arg_list(args: List[Tuple[str, str]]) -> str:
    return ", ".join(f"{name}: {_to_rust_type(t)}" for name, t in args)


def _c_arg_list(args: List[Tuple[str, str]]) -> str:
    return ", ".join(f"{_to_c_type(t)} {name}" for name, t in args)


def _python_ctypes_arg_list(args: List[Tuple[str, str]]) -> str:
    return ", ".join(_to_python_ctype(t) for _, t in args)


class ContractSynthesizer:
    """Synthesize canonical FFI wrapper templates from blueprint contracts."""

    def __init__(self, contracts: Optional[List[ContractEntry]] = None) -> None:
        self.contracts: Dict[str, ContractEntry] = {
            c.name: c for c in (contracts or []) if c.name
        }

    def _resolve_contract(self, symbol: str) -> Optional[ContractEntry]:
        return self.contracts.get(symbol)

    def synthesize_missing_morphism(
        self,
        symbol: str,
        language_pair: str,
        signature: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Emit a canonical wrapper template for *symbol* and *language_pair*.

        *language_pair* is one of ``python/rust``, ``rust/cpp``,
        ``c-abi``/``c_abi``, or ``rust/rust``.  If *signature* is omitted the
        matching blueprint contract's signature is used.
        """
        contract = self._resolve_contract(symbol)
        sig = signature or (contract.signature if contract else "")
        name, args, ret = _parse_signature(sig)
        name = name or symbol

        pair = language_pair.replace("-", "_").replace("/", "_").lower()

        if pair in ("python_rust", "pyo3"):
            return self._synthesize_pyo3(name, args, ret, symbol)
        if pair in ("rust_cpp", "c_abi", "cabi", "c"):
            return self._synthesize_c_abi(name, args, ret, symbol)
        if pair in ("rust_rust", "rust"):
            return self._synthesize_rust_rust(name, args, ret, symbol)

        raise ValueError(f"unsupported language pair: {language_pair}")

    def _synthesize_pyo3(
        self, name: str, args: List[Tuple[str, str]], ret: str, symbol: str
    ) -> Dict[str, Any]:
        rust_args = _rust_arg_list(args)
        rust_ret = _to_rust_type(ret)
        call_args = ", ".join(a for a, _ in args)
        rust_stub = (
            f"#[pyfunction]\n"
            f"fn {name}({rust_args}) -> {rust_ret} {{\n"
            f"    // TODO: wire to the native Rust implementation.\n"
            f"    aero_core::{symbol}({call_args})\n"
            f"}}\n"
        )
        py_args = ", ".join(f"{a}=_" for a, _ in args)
        python_stub = (
            f"from aero_forge_native import {name}\n\n"
            f"result = {name}({py_args})\n"
        )
        return {
            "language_pair": "python/rust",
            "symbol": name,
            "rust_stub": rust_stub,
            "python_stub": python_stub,
        }

    def _synthesize_c_abi(
        self, name: str, args: List[Tuple[str, str]], ret: str, symbol: str
    ) -> Dict[str, Any]:
        c_symbol = name
        c_args = _c_arg_list(args)
        c_ret = _to_c_type(ret)
        rust_args = _rust_arg_list(args)
        rust_ret = _to_rust_type(ret)
        call_args = ", ".join(a for a, _ in args)

        rust_stub = (
            f"#[no_mangle]\n"
            f"pub unsafe extern \"C\" fn {c_symbol}({rust_args}) -> {rust_ret} {{\n"
            f"    aero_core::{symbol}({call_args})\n"
            f"}}\n"
        )
        cpp_header = f"extern \"C\" {c_ret} {c_symbol}({c_args});\n"
        py_ctypes_args = _python_ctypes_arg_list(args)
        python_stub = (
            f"import ctypes\n"
            f"lib = ctypes.CDLL('./lib{symbol}.so')\n"
            f"lib.{c_symbol}.argtypes = [{py_ctypes_args}]\n"
            f"lib.{c_symbol}.restype = {_to_python_ctype(ret)}\n"
            f"result = lib.{c_symbol}({', '.join(a for a, _ in args)})\n"
        )
        return {
            "language_pair": "rust/cpp",
            "symbol": name,
            "rust_stub": rust_stub,
            "cpp_header": cpp_header,
            "python_stub": python_stub,
        }

    def _synthesize_rust_rust(
        self, name: str, args: List[Tuple[str, str]], ret: str, symbol: str
    ) -> Dict[str, Any]:
        rust_args = _rust_arg_list(args)
        rust_ret = _to_rust_type(ret)
        call_args = ", ".join(a for a, _ in args)
        rust_stub = (
            f"pub trait AeroZeroCopy<T> {{\n"
            f"    fn as_view(&self) -> &[T];\n"
            f"}}\n\n"
            f"pub fn {name}({rust_args}) -> {rust_ret} {{\n"
            f"    // Zero-copy in-process transmute: input and output share the\n"
            f"    // backing Rust memory; no cross-FFI allocation occurs.\n"
            f"    aero_core::{symbol}({call_args})\n"
            f"}}\n"
        )
        return {
            "language_pair": "rust/rust",
            "symbol": name,
            "rust_stub": rust_stub,
        }


@dataclass
class FFIBoundaryEdge:
    """A directed edge between two language nodes requiring an FFI bridge."""

    edge_id: str
    source_node: str
    source_lang: str
    target_node: str
    target_lang: str
    boundary_type: str
    symbol_name: str
    argument_types: List[str] = field(default_factory=list)
    return_type: str = "void"
    is_zero_copy: bool = False


@dataclass
class GeneratedFFIBridge:
    """Output of synthesizing one FFI boundary edge."""

    edge_id: str
    boundary_type: str
    header: str
    source: str
    python_loader: str
    csharp_stub: str
    build_manifest: Dict[str, Any] = field(default_factory=dict)


class DynamicContractSynthesizer:
    """Synthesize dynamic FFI bridge contracts across arbitrary polyglot edges."""

    def synthesize_boundary(self, edge: FFIBoundaryEdge) -> GeneratedFFIBridge:
        """Dispatch to the generator matching the edge boundary type."""
        boundary = edge.boundary_type.replace("-", "_").lower()
        if boundary in ("c_abi", "cabi", "c", "wasm_wasi", "wasm", "wasi"):
            # WASM modules are exposed through a C-ABI boundary; the Rust source
            # is compiled with the requested `--target` triple (e.g. wasm32-*).
            return self._synth_c_abi_boundary(edge)
        if boundary in ("pyo3", "pyo3_maturin", "python_rust"):
            return self._synth_pyo3_boundary(edge)
        if boundary in ("cgo", "go_c", "go"):
            return self._synth_cgo_boundary(edge)
        if boundary in ("pinvoke", "csharp", "cs"):
            return self._synth_pinvoke_boundary(edge)
        raise ValueError(f"unsupported boundary type: {edge.boundary_type}")

    def _synth_c_abi_boundary(self, edge: FFIBoundaryEdge) -> GeneratedFFIBridge:
        args = list(zip(edge.argument_types, [f"arg{i}" for i in range(len(edge.argument_types))]))
        c_args = ", ".join(
            f"{FFI_TYPE_LAYOUTS.get(t, {}).get('c_type', 'void*')} {name}"
            for t, name in args
        )
        c_ret = FFI_TYPE_LAYOUTS.get(edge.return_type, {}).get("c_type", "void")
        header_guard = f"AERO_{edge.symbol_name.upper()}_H"
        header = (
            f"#ifndef {header_guard}\n"
            f"#define {header_guard}\n\n"
            "#include <stdint.h>\n\n"
            f'#ifdef __cplusplus\nextern "C" {{\n#endif\n\n'
            f"{c_ret} {edge.symbol_name}({c_args});\n\n"
            f'#ifdef __cplusplus\n}}\n#endif\n\n'
            f"#endif /* {header_guard} */\n"
        )

        rust_args = ", ".join(
            f"{name}: {FFI_TYPE_LAYOUTS.get(t, {}).get('rust_type', '*mut c_void')}"
            for t, name in args
        )
        rust_ret = FFI_TYPE_LAYOUTS.get(edge.return_type, {}).get("rust_type", "()")
        source = (
            "use std::os::raw::c_void;\n\n"
            f"#[no_mangle]\n"
            f"pub extern \"C\" fn {edge.symbol_name}({rust_args}) -> {rust_ret} {{\n"
            "    // TODO: wire to implementation\n"
            f"    {'0' if rust_ret not in ('()', 'void') else ''}\n"
            "}\n"
        )

        py_args = ", ".join(
            FFI_TYPE_LAYOUTS.get(t, {}).get("python_ctype", "c_void_p")
            for t, _ in args
        )
        python_loader = (
            "import ctypes\n\n"
            f"lib = ctypes.CDLL('./lib{edge.symbol_name}.so')\n"
            f"lib.{edge.symbol_name}.argtypes = [{py_args}]\n"
            f"lib.{edge.symbol_name}.restype = {FFI_TYPE_LAYOUTS.get(edge.return_type, {}).get('python_ctype', 'None')}\n"
        )

        return GeneratedFFIBridge(
            edge_id=edge.edge_id,
            boundary_type="c_abi",
            header=header,
            source=source,
            python_loader=python_loader,
            csharp_stub="",
            build_manifest={"cbindgen": True, "crate_type": "cdylib"},
        )

    def _synth_pyo3_boundary(self, edge: FFIBoundaryEdge) -> GeneratedFFIBridge:
        rust_args = ", ".join(
            f"arg{i}: {FFI_TYPE_LAYOUTS.get(t, {}).get('rust_type', 'PyObject')}"
            for i, t in enumerate(edge.argument_types)
        )
        rust_ret = FFI_TYPE_LAYOUTS.get(edge.return_type, {}).get("rust_type", "PyObject")
        source = (
            "use pyo3::prelude::*;\n\n"
            f"#[pyfunction]\n"
            f"fn {edge.symbol_name}({rust_args}) -> {rust_ret} {{\n"
            "    // TODO: wire to implementation\n"
            f"    {'Default::default()' if rust_ret not in ('()', 'void') else ''}\n"
            "}\n"
        )
        header = f"// PyO3 bridge for {edge.symbol_name}\n"
        python_loader = (
            f"from {edge.symbol_name} import {edge.symbol_name}\n"
        )
        csharp_stub = ""
        return GeneratedFFIBridge(
            edge_id=edge.edge_id,
            boundary_type="pyo3_maturin",
            header=header,
            source=source,
            python_loader=python_loader,
            csharp_stub=csharp_stub,
            build_manifest={"maturin": True, "crate_type": "cdylib"},
        )

    def _synth_cgo_boundary(self, edge: FFIBoundaryEdge) -> GeneratedFFIBridge:
        go_args = ", ".join(
            f"arg{i} {FFI_TYPE_LAYOUTS.get(t, {}).get('go_type', 'C.int')}"
            for i, t in enumerate(edge.argument_types)
        )
        go_ret = FFI_TYPE_LAYOUTS.get(edge.return_type, {}).get("go_type", "")
        ret_sig = f" {go_ret}" if go_ret else ""
        header = f"// CGO header for {edge.symbol_name}\n"
        source = (
            "package main\n\n"
            '/*\n#include <stdint.h>\n*/\n'
            'import "C"\n\n'
            f"//export {edge.symbol_name}\n"
            f"func {edge.symbol_name}({go_args}){ret_sig} {{\n"
            "    // TODO: wire to implementation\n"
            "}\n\n"
            "func main() {}\n"
        )
        python_loader = (
            "import ctypes\n\n"
            f"lib = ctypes.CDLL('./{edge.symbol_name}.so')\n"
        )
        csharp_stub = ""
        return GeneratedFFIBridge(
            edge_id=edge.edge_id,
            boundary_type="cgo",
            header=header,
            source=source,
            python_loader=python_loader,
            csharp_stub=csharp_stub,
            build_manifest={"buildmode": "c-shared"},
        )

    def _synth_pinvoke_boundary(self, edge: FFIBoundaryEdge) -> GeneratedFFIBridge:
        cs_args = ", ".join(
            f"{FFI_TYPE_LAYOUTS.get(t, {}).get('csharp_type', 'IntPtr')} arg{i}"
            for i, t in enumerate(edge.argument_types)
        )
        cs_ret = FFI_TYPE_LAYOUTS.get(edge.return_type, {}).get("csharp_type", "void")
        csharp_stub = (
            "using System;\n"
            "using System.Runtime.InteropServices;\n\n"
            "public static partial class AeroNative\n{\n"
            f'    [LibraryImport("{edge.symbol_name}", EntryPoint = "{edge.symbol_name}")]\n'
            f"    public static partial {cs_ret} {edge.symbol_name}({cs_args});\n"
            "}\n"
        )
        header = f"// C# P/Invoke bridge for {edge.symbol_name}\n"
        source = (
            "#include <stdint.h>\n\n"
            f"__declspec(dllexport) {FFI_TYPE_LAYOUTS.get(edge.return_type, {}).get('c_type', 'void')} "
            f"{edge.symbol_name}(\n"
        )
        for i, t in enumerate(edge.argument_types):
            source += f"    {FFI_TYPE_LAYOUTS.get(t, {}).get('c_type', 'void*')} arg{i}{',' if i < len(edge.argument_types) - 1 else ''}\n"
        source += ") {\n    // TODO: wire to implementation\n}\n"
        python_loader = ""
        return GeneratedFFIBridge(
            edge_id=edge.edge_id,
            boundary_type="pinvoke",
            header=header,
            source=source,
            python_loader=python_loader,
            csharp_stub=csharp_stub,
            build_manifest={"aot": True, "allow_unsafe_blocks": True},
        )
