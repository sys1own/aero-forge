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
from typing import Any, Dict, List, Optional, Tuple

from aero_forge.blueprint import ContractEntry


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
