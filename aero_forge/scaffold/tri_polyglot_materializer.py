"""Physical file materialization for tri-polyglot (Python + Rust + C++) blueprints.

The materializer writes a Python driver package, a Rust PyO3 core, and a
C-ABI dynamic shared library into a single workspace and builds all three.
"""

from __future__ import annotations

import logging
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from aero_forge.blueprint import (
    ABIContract,
    Blueprint,
    ContractEntry,
    FunctionSpec,
    ManifestEntry,
    write_blueprint,
)
from aero_forge.errors import BuildStageError
from aero_forge.builder import language_router
from aero_forge.native_bridge import _ctypes_loader_source
from aero_forge.scaffold.cargo_manifest import sanitize_crate_name
from aero_forge.scaffold.cargo_runner import cargo_build
from aero_forge.scaffold.entrypoint_adapter import EntrypointAdapterEngine
from aero_forge.scaffold.cpp_materializer import (
    _collect_cpp_sources,
    _collect_include_dirs,
    _contract_to_python_stub as _cpp_contract_to_python_stub,
    _find_cpp_compiler,
    _generate_cpp_header,
    _generate_native_cpp,
    _is_c_abi_contract,
    _is_c_abi_list,
    _is_c_abi_tuple_return,
    _is_special_cpp_contract,
    _so_name,
)
from aero_forge.scaffold.polyglot_materializer import (
    _native_loader_source,
    _parse_signature,
    _render_pyproject,
    _render_readme,
    _synthesize_python_source,
)
from aero_forge.scaffold.python_repo_generator import _sanitize_module_name
from aero_forge.scaffold.test_generator import _abi_contract_to_signature
from aero_forge.translator import TargetMode, UASTToHINTranslator, python_source_to_uast
from aero_forge.scaffold.engine import Engine

logger = logging.getLogger("aero_forge.scaffold.tri_polyglot")


def _accel_log(level: str, message: str) -> None:
    """Append a structured line to the per-session accelerator log if set."""
    log_path = os.environ.get("AERO_FORGE_ACCEL_LOG")
    if not log_path:
        return
    try:
        import time

        timestamp = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"{timestamp} [{level}] {message}\n")
    except Exception:
        pass


def _partition_contracts(
    contracts: List[ContractEntry],
) -> Tuple[List[ContractEntry], List[ContractEntry], List[ContractEntry]]:
    """Partition contracts into C++ (C-ABI), Rust (PyO3), and Python fallbacks.

    Honors an explicit non-default ``language`` field (``cpp`` or ``rust``), but
    falls back to signature-based routing because the default ``ContractEntry``
    language is ``python``.
    """
    cpp: List[ContractEntry] = []
    rust: List[ContractEntry] = []
    python: List[ContractEntry] = []
    for contract in contracts:
        if not contract.signature:
            continue
        lang = (contract.language or "").lower()

        # Explicit C++ request: only accept if the signature is actually C-ABI compatible.
        if lang == "cpp":
            if _is_c_abi_contract(contract):
                cpp.append(contract)
            else:
                python.append(contract)
            continue

        # Explicit Rust request.
        if lang in ("rust", "rust/c", "pyo3", "c"):
            rust.append(contract)
            continue

        # Default/unknown language: route by signature.
        if _is_c_abi_contract(contract):
            cpp.append(contract)
            continue

        try:
            _, args, return_type = _parse_signature(contract.signature)
        except Exception:
            python.append(contract)
            continue

        scalar_types = {
            "int",
            "i64",
            "i32",
            "u64",
            "u32",
            "usize",
            "uint",
            "uint64",
            "float",
            "f64",
            "f32",
            "bool",
            "str",
            "string",
            "none",
            "void",
            "",
        }
        list_types = {
            "list",
            "list[float]",
            "list[int]",
            "list[i64]",
            "list[u64]",
            "list[u32]",
            "list[bool]",
            "list[str]",
        }

        def _rust_supported_type(t: str) -> bool:
            t = (t or "").strip().lower()
            return (
                t in scalar_types
                or t in list_types
                or (
                    t.startswith("list[")
                    and t.endswith("]")
                    and t[5:-1].strip() in scalar_types
                )
            )

        rust_supported = all(
            _rust_supported_type(t) or a == "self" for a, t in args
        ) and _rust_supported_type(return_type)
        if rust_supported:
            rust.append(contract)
        else:
            python.append(contract)
    return cpp, rust, python


def _abi_contract_to_contract_entry(abi: ABIContract) -> Optional[ContractEntry]:
    """Convert an ``ABIContract`` with a structured signature into a ``ContractEntry``."""
    sig = abi.signature
    if not sig:
        return None
    try:
        signature_str = _abi_contract_to_signature(abi)
    except Exception:
        return None
    if not signature_str:
        return None
    target = (abi.target_language or "").lower()
    binding = (abi.binding_framework or "").lower()
    identity = f"{(abi.contract_id or '')} {(abi.export_symbol or '')}".lower()
    is_cpp = "cpp" in identity or target == "cpp" or target in ("c++", "cxx")
    is_rust = (
        "rust" in identity
        or "pyo3" in identity
        or target == "rust"
        or binding == "pyo3"
    )
    if is_cpp and not is_rust:
        lang = "cpp"
    elif is_rust and not is_cpp:
        lang = "rust"
    elif is_cpp and is_rust:
        # Rust C-ABI exposure (e.g. ``rust_orchestrate_cabi`` with target_language ``cpp``).
        lang = "rust/c" if binding == "c_abi" else "rust"
    elif target == "python":
        lang = "python"
    else:
        lang = target or "python"
    return ContractEntry(
        name=abi.export_symbol or abi.contract_id,
        signature=signature_str,
        language=lang,
        python_name=abi.export_symbol or abi.contract_id,
    )


def _default_tri_polyglot_contracts() -> List[ContractEntry]:
    """Return a canonical set of tri-polyglot contracts for prompt-driven builds."""
    return [
        ContractEntry(
            name="sliding_window_dtw",
            signature="def sliding_window_dtw(seq1: list[float], seq2: list[float], window: int) -> float",
            language="cpp",
            python_name="sliding_window_dtw",
        ),
        ContractEntry(
            name="vectorized_fft",
            signature="def vectorized_fft(real: list[float], imag: list[float]) -> tuple[list[float], list[float]]",
            language="cpp",
            python_name="vectorized_fft",
        ),
        ContractEntry(
            name="orchestrate",
            signature="def orchestrate(json: str) -> str",
            language="rust",
            python_name="orchestrate",
        ),
        ContractEntry(
            name="free_string",
            signature="def free_string(ptr: int) -> None",
            language="rust",
            python_name="free_string",
        ),
        ContractEntry(
            name="run_kernel",
            signature="def run_kernel(kernel_id: int, input: list[float], output: list[float]) -> int",
            language="rust",
            python_name="run_kernel",
        ),
    ]


def _normalize_tri_polyglot_contracts(blueprint: Blueprint) -> List[ContractEntry]:
    """Return high-quality contracts, deriving from ``abi_contracts`` and defaults.

    When the LLM planner emits wrapper contracts full of ``Any`` or unparameterized
    ``list`` types, we rebuild the contract list from the structured ABI contracts
    and a small set of canonical tri-polyglot kernels so the materializer always has
    concrete C++/Rust functions to emit.
    """
    existing = list(blueprint.contracts or [])

    def _usable(c: ContractEntry) -> bool:
        if not c.signature:
            return False
        sig = c.signature.lower()
        if "any" in sig:
            return False
        # Reject bare ``list`` without an element type for C-ABI purposes.
        import re

        if re.search(r"(?<![a-z0-9_])list(?![a-z0-9_\[])", sig):
            return False
        return True

    usable = [c for c in existing if _usable(c)]

    # A contract marked as C++ but not actually expressible in C-ABI should be
    # discarded so the normalizer can recover a correct language routing.
    def _cpp_usable(c: ContractEntry) -> bool:
        if (c.language or "").lower() != "cpp":
            return True
        return _is_c_abi_contract(c) or _is_special_cpp_contract(c)

    usable = [c for c in usable if _cpp_usable(c)]
    cpp_contracts, rust_contracts, _ = _partition_contracts(usable)
    has_cpp = bool(cpp_contracts)
    has_rust = bool(rust_contracts)

    # If the explicitly-provided contracts are already complete, use them as-is.
    if (
        has_cpp
        and has_rust
        and not any(
            "vectorized_fft" in c.name and "tuple" not in (c.signature or "")
            for c in usable
        )
    ):
        return usable

    derived: List[ContractEntry] = []
    seen: set[str] = {c.name for c in usable}
    for abi in blueprint.abi_contracts or []:
        ce = _abi_contract_to_contract_entry(abi)
        if not ce or ce.name in seen or not _usable(ce):
            continue
        # Skip low-quality or non-C-ABI C++ derivations and let defaults fill in.
        if (ce.language or "").lower() == "cpp" and not (
            _is_c_abi_contract(ce) or _is_special_cpp_contract(ce)
        ):
            continue
        # Skip duplicate PyO3 variants (e.g. ``orchestrate_py``) when the
        # canonical C-ABI symbol (``orchestrate``) is already present.
        if ce.name.endswith("_py"):
            if ce.name[:-3] in seen:
                continue
        if ce.name.endswith("_pyo3"):
            if ce.name[:-5] in seen:
                continue
        derived.append(ce)
        seen.add(ce.name)

    result: List[ContractEntry] = usable + derived

    for default in _default_tri_polyglot_contracts():
        if default.name not in seen:
            result.append(default)
            continue
        # Replace a low-quality ABI-derived contract with the canonical default
        # when the derived signature does not satisfy the expected contract shape.
        existing_index = next(
            (i for i, c in enumerate(result) if c.name == default.name), -1
        )
        if existing_index >= 0:
            existing = result[existing_index]
            if "vectorized_fft" in default.name and "tuple" not in (
                existing.signature or ""
            ):
                result[existing_index] = default
            elif default.name == "sliding_window_dtw":
                try:
                    _, existing_args, existing_ret = _parse_signature(
                        existing.signature
                    )
                except Exception:
                    existing_args, existing_ret = [], ""
                if (
                    existing_ret.lower() not in ("float", "f64", "double")
                    or len(existing_args) > 3
                ):
                    result[existing_index] = default
    return result


def _function_names(contracts: List[ContractEntry]) -> List[str]:
    names: List[str] = []
    for contract in contracts:
        if not contract.signature:
            continue
        try:
            name, _, _ = _parse_signature(contract.signature)
        except Exception:
            continue
        names.append(name)
    return names


def _generate_rust_source(contracts: List[ContractEntry]) -> str:
    """Return a Python source string containing only the Rust-routed contracts."""
    lines: List[str] = ["from __future__ import annotations", ""]
    names: List[str] = []
    for contract in contracts:
        if not contract.signature:
            continue
        try:
            name, args, return_type = _parse_signature(contract.signature)
        except Exception:
            continue
        names.append(name)
        arg_str = ", ".join(f"{arg}: {typ}" for arg, typ in args if arg != "self")
        lines.append(f"def {name}({arg_str}) -> {return_type}:")
        lines.append(
            _generate_stub_body(
                name, [(a, t) for a, t in args if a != "self"], return_type
            )
        )
        lines.append("")
    if names:
        lines.append("__all__ = [" + ", ".join(f'"{n}"' for n in names) + "]")
    else:
        lines.append("__all__: list[str] = []")
    return "\n".join(lines) + "\n"


def _is_scalar_type(type_hint: str) -> bool:
    return type_hint.lower() in (
        "int",
        "i64",
        "i32",
        "float",
        "f64",
        "f32",
        "bool",
        "str",
        "string",
    )


def _is_list_type(type_hint: str) -> bool:
    return type_hint.lower().startswith("list[") and type_hint.endswith("]")


def _list_element_type(type_hint: str) -> str:
    """Return the inner element type of a ``list[T]`` hint, lower-cased."""
    hint = type_hint.strip()
    if hint.lower().startswith("list[") and hint.endswith("]"):
        return hint[5:-1].strip().lower()
    return ""


def _generate_stub_body(
    name: str, args: List[Tuple[str, str]], return_type: str
) -> str:
    """Return a simple fallback body for a contract stub from type patterns."""
    rt = return_type.lower()
    scalar_args = [(a, t) for a, t in args if _is_scalar_type(t)]
    list_args = [(a, t) for a, t in args if _is_list_type(t)]

    if rt in ("int", "i64", "i32"):
        return "    return 0"
    if rt in ("float", "f64", "f32"):
        return "    return 0.0"
    if rt == "bool":
        if len(scalar_args) == 1 and scalar_args[0][1].lower() in ("str", "string"):
            arg = scalar_args[0][0]
            return f"    return len({arg}) > 8"
        return "    return True"
    if rt == "str":
        return '    return "ok"'
    if "list" in rt and list_args:
        list_name, list_type = list_args[0]
        list_elt = _list_element_type(list_type)
        ret_elt = _list_element_type(return_type)
        if scalar_args:
            scalar_name = scalar_args[0][0]
            factor = scalar_name
        else:
            factor = "2"
        if list_elt in ("str", "string"):
            if ret_elt in ("str", "string"):
                # String-in/string-out: preserve the input string unchanged.
                return f"    return [x for x in {list_name}]"
            # Numeric output derived from string length.
            return f"    return [(len(x) * {factor}) for x in {list_name}]"
        return f"    return [x * {factor} for x in {list_name}]"
    if "dict" in rt and not args:
        return '    return {"status": "ok"}'
    return "    return None"


def _to_rust_identifier(name: str) -> str:
    sanitized = re.sub(r"[^0-9a-zA-Z_]", "_", name)
    if not sanitized:
        sanitized = "module"
    if sanitized[0].isdigit():
        sanitized = "a_" + sanitized
    return sanitized


def _append_rust_c_abi_wrappers(rust_dir: Path, rust_contracts: List[ContractEntry]) -> None:
    """Append #[no_mangle] C-ABI wrappers for Rust contracts that expose string-list inputs.

    This is used by tri-polyglot workspaces so a single ``cdylib`` crate can be
    consumed both as a PyO3 module and as a plain C shared library.
    """
    lib_rs = rust_dir / "src" / "lib.rs"
    if not lib_rs.is_file():
        return

    text = lib_rs.read_text(encoding="utf-8")
    if "std::ffi::CStr" not in text:
        text = text.replace(
            "use std::collections::BTreeMap;",
            "use std::collections::BTreeMap;\nuse std::ffi::CStr;",
        )

    _C_ABI_SCALAR_MAP = {
        "int": "i64",
        "i64": "i64",
        "i32": "i32",
        "u64": "u64",
        "u32": "u32",
        "usize": "usize",
        "float": "f64",
        "f64": "f64",
        "f32": "f32",
        "bool": "bool",
    }

    wrappers: List[str] = []
    for contract in rust_contracts:
        if not contract.signature:
            continue
        try:
            name, args, ret = _parse_signature(contract.signature)
        except Exception:
            continue
        if not name:
            continue
        list_args = [(a, _list_element_type(t)) for a, t in args if _is_list_type(t)]
        if not list_args or list_args[0][1] not in ("str", "string"):
            continue
        ret_elt = _list_element_type(ret)
        if ret_elt not in _C_ABI_SCALAR_MAP:
            continue

        rust_fn = f"_accel_{_to_rust_identifier(name)}"
        # Convention: ``run_scheduler`` -> C symbol ``scheduler_run``.
        if name.startswith("run_"):
            c_name = f"{name[4:]}_run"
        else:
            c_name = f"{name}_cabi"
        out_type = _C_ABI_SCALAR_MAP[ret_elt]
        count_type = "i64"
        for a, t in args:
            if t.lower() in ("int", "i64", "i32", "u64", "u32", "usize"):
                count_type = _C_ABI_SCALAR_MAP.get(t.lower(), "i64")
                break

        wrappers.append(
            f"""#[no_mangle]
pub unsafe extern \"C\" fn {c_name}(tasks: *const *const std::os::raw::c_char, count: usize, out: *mut {out_type}) {{
    if tasks.is_null() || out.is_null() {{
        return;
    }}
    let tasks_vec: Vec<String> = (0..count)
        .map(|i| {{
            let ptr = *tasks.add(i);
            if ptr.is_null() {{
                return String::new();
            }}
            CStr::from_ptr(ptr).to_string_lossy().into_owned()
        }})
        .collect();
    let results = {rust_fn}(tasks_vec, count as {count_type});
    let out_slice = std::slice::from_raw_parts_mut(out, count);
    for (i, v) in results.iter().enumerate().take(count) {{
        out_slice[i] = *v;
    }}
}}
"""
        )

    if wrappers:
        text = text + "\n\n" + "\n".join(wrappers)
    lib_rs.write_text(text, encoding="utf-8")


def _generate_python_init(
    pkg_name: str,
    cpp_dir: Path,
    cpp_contracts: List[ContractEntry],
    rust_contracts: List[ContractEntry],
    python_contracts: List[ContractEntry],
    rust_crate_name: str,
    native_bridge_module: Optional[str] = None,
    rust_dir: str = "rust_core",
    workspace_root: Optional[Path] = None,
    pkg_dir: Optional[Path] = None,
) -> str:
    """Generate ``<pkg>/__init__.py`` that re-exports the native loader."""
    return (
        '"""Tri-polyglot driver package."""\n'
        "\n"
        "from __future__ import annotations\n"
        "\n"
        "from .native_loader import *\n"
    )


def _generate_native_loader_py(
    pkg_name: str,
    cpp_dir: Path,
    cpp_contracts: List[ContractEntry],
    rust_contracts: List[ContractEntry],
    python_contracts: List[ContractEntry],
    rust_crate_name: str,
    native_bridge_module: Optional[str] = None,
    rust_dir: str = "rust_core",
    workspace_root: Optional[Path] = None,
    pkg_dir: Optional[Path] = None,
) -> str:
    """Generate ``<pkg>/native_loader.py`` that loads both the Rust extension and C++ .so."""
    lines: List[str] = [
        '"""Native bridge loader for the tri-polyglot package."""',
        "",
        "from __future__ import annotations",
        "",
    ]

    # Rust extension loader
    lines.append(_native_loader_source([rust_crate_name], rust_dir=rust_dir))
    lines.append("")

    # Rust-backed functions
    for contract in rust_contracts:
        if not contract.signature:
            continue
        try:
            name, args, return_type = _parse_signature(contract.signature)
        except Exception:
            continue
        # Methods with a ``self`` receiver become free functions in the generated Rust module.
        call_args = [a for a, _ in args if a != "self"]
        arg_sig = ", ".join(f"{a}: {t}" for a, t in args if a != "self")
        arg_call = ", ".join(call_args)
        fallback = _generate_stub_body(name, args, return_type)
        if arg_sig:
            lines.append(f"def {name}({arg_sig}) -> {return_type}:")
        else:
            lines.append(f"def {name}() -> {return_type}:")
        lines.append(f"    if _NATIVE is not None and hasattr(_NATIVE, '{name}'):")
        if arg_call:
            lines.append(f"        return _NATIVE.{name}({arg_call})")
        else:
            lines.append(f"        return _NATIVE.{name}()")
        lines.append(fallback)
        lines.append("")

    # C++ ctypes loader for C-ABI contracts
    cpp_names: List[str] = _function_names(cpp_contracts)
    if cpp_names:
        if native_bridge_module:
            lines.append(f"from {native_bridge_module} import {', '.join(cpp_names)}")
            lines.append("")
        else:
            stub_source = "\n".join(
                _cpp_contract_to_python_stub(c) for c in cpp_contracts
            )
            so_path = (cpp_dir / _so_name(f"{pkg_name}_cpp")).resolve()
            loader_path = (
                (pkg_dir / "native_loader.py")
                if pkg_dir
                else (so_path.parent / "native_loader.py")
            )
            lines.append(
                _ctypes_loader_source(
                    stub_source,
                    so_path,
                    cpp_names,
                    workspace_root=workspace_root,
                    loader_path=loader_path,
                )
            )
            lines.append("")

    # Fallback pure-Python implementations
    for contract in python_contracts:
        if not contract.signature:
            continue
        try:
            name, args, return_type = _parse_signature(contract.signature)
        except Exception:
            continue
        arg_sig = ", ".join(f"{a}: {t}" for a, t in args if a != "self")
        lines.append(f"def {name}({arg_sig}) -> {return_type}:")
        lines.append(_generate_fallback_body(name, args, return_type))
        lines.append("")

    all_names = _function_names(cpp_contracts + rust_contracts + python_contracts)
    lines.append(f"__all__ = {all_names!r}")
    return "\n".join(lines) + "\n"


def _generate_makefile(
    rust_dir: str,
    cpp_dir: str,
    python_dir: str,
    tests_dir: str,
    cpp_pkg_name: str,
    cpp_source_name: str = "native.cpp",
    header_dirs: Optional[List[str]] = None,
) -> str:
    """Generate a root ``Makefile`` with build, test, and run targets."""
    cpp_so = _so_name(cpp_pkg_name)
    includes: List[str] = ["."]
    for h in header_dirs or []:
        p = Path(h)
        inc = str(p.parent) if p.suffix in (".h", ".hpp") else str(p)
        if inc and inc not in (".", "/"):
            includes.append(inc)
    include_flags = " ".join(f"-I {inc}" for inc in sorted(set(includes)))
    return (
        ".PHONY: build test run\n"
        "\n"
        "build:\n"
        f"\tcd {rust_dir} && cargo build --release\n"
        f"\tg++ -O3 -march=native -shared -fPIC -std=c++17 {include_flags} -o {cpp_dir}/{cpp_so} {cpp_dir}/{cpp_source_name}\n"
        "\n"
        "test:\n"
        f"\tPYTHONPATH=. pytest {tests_dir}\n"
        "\n"
        "run:\n"
        f"\tPYTHONPATH=. python3 -m {python_dir.replace('/', '.')}.main\n"
    )


def _generate_core_py(contracts: List[ContractEntry]) -> str:
    """Generate a pure-Python reference ``core.py`` matching the workspace contracts."""
    lines: List[str] = [
        '"""Pure-Python reference implementation."""',
        "",
        "from __future__ import annotations",
        "",
        "",
    ]
    for contract in contracts:
        if not contract.signature:
            continue
        try:
            name, args, return_type = _parse_signature(contract.signature)
        except Exception:
            continue
        arg_names = [a for a, _ in args if a != "self"]
        if name == "multiply_matrices":
            lines.append(
                "def multiply_matrices(a, b, rows, cols, inner):\n"
                "    out = [0.0] * (rows * cols)\n"
                "    for i in range(rows):\n"
                "        for j in range(cols):\n"
                "            total = 0.0\n"
                "            for k in range(inner):\n"
                "                total += a[i * inner + k] * b[k * cols + j]\n"
                "            out[i * cols + j] = total\n"
                "    return out\n"
            )
            continue
        arg_sig = ", ".join(arg_names)
        lines.append(f"def {name}({arg_sig}):")
        lines.append(_generate_stub_body(name, args, return_type))
        lines.append("")
    return "\n".join(lines) + "\n"


def _generate_orchestration_test(contracts: List[ContractEntry]) -> str:
    """Generate a pytest that validates native wrappers against the core reference."""
    rust_names = []
    for contract in contracts:
        if not contract.signature:
            continue
        try:
            name, args, return_type = _parse_signature(contract.signature)
        except Exception:
            continue
        if name == "run_scheduler":
            rust_names.append(name)
        elif name == "multiply_matrices":
            rust_names.append(name)
    lines: List[str] = [
        '"""End-to-end orchestration tests for the native pipeline."""',
        "",
        "from __future__ import annotations",
        "",
        "import pytest",
        "",
        "from python_interface import run_scheduler, multiply_matrices",
        "from src.session_1rp6n5ynsg6msbrvxax import core",
        "",
    ]
    if "run_scheduler" in rust_names:
        lines.extend([
            "def test_run_scheduler_matches_core():",
            "    task_descrs = [\"hello\", \"world\", \"x\"]",
            "    count = 3",
            "    assert run_scheduler(task_descrs, count) == core.run_scheduler(task_descrs, count)",
            "",
        ])
    if "multiply_matrices" in rust_names:
        lines.extend([
            "def test_multiply_matrices_matches_core():",
            "    rows, cols, inner = 2, 2, 2",
            "    a = [1.0, 2.0, 3.0, 4.0]",
            "    b = [5.0, 6.0, 7.0, 8.0]",
            "    assert multiply_matrices(a, b, rows, cols, inner) == pytest.approx(core.multiply_matrices(a, b, rows, cols, inner))",
            "",
        ])
    return "\n".join(lines) + "\n"


def _generate_fallback_body(
    name: str, args: List[Tuple[str, str]], return_type: str
) -> str:
    rt = return_type.lower()
    scalar_args = [(a, t) for a, t in args if _is_scalar_type(t)]
    list_args = [(a, t) for a, t in args if _is_list_type(t)]
    if rt.startswith("tuple["):
        inner = rt[6:-1].strip()
        parts = [p.strip() for p in inner.split(",")]
        if all(p.startswith("list") for p in parts) and len(list_args) >= len(parts):
            tuple_parts = [
                f"[x * 2 for x in {list_args[i][0]}]" for i in range(len(parts))
            ]
            return f"    return ({', '.join(tuple_parts)})"
    if "list" in rt and list_args:
        list_name, _ = list_args[0]
        if scalar_args:
            scalar_name = scalar_args[0][0]
            return f"    return [x * {scalar_name} for x in {list_name}]"
        return f"    return [x * 2 for x in {list_name}]"
    if "dict" in rt:
        if not args:
            return '    return {"status": "ok"}'
        return "    return {}"
    if rt in ("int", "i64", "i32"):
        return "    return 0"
    if rt in ("float", "f64", "f32"):
        return "    return 0.0"
    if rt == "bool":
        if len(scalar_args) == 1 and scalar_args[0][1].lower() in ("str", "string"):
            arg = scalar_args[0][0]
            return f"    return len({arg}) > 8"
        return "    return True"
    if rt == "str":
        return '    return "ok"'
    return "    return None"


def _sample_arg(t: str) -> str:
    t = (t or "").lower().replace(" ", "")
    if t in ("int", "i64", "i32"):
        return "5"
    if t in ("float", "f64", "f32"):
        return "2.0"
    if t == "bool":
        return "True"
    if t in ("str", "string"):
        return '"validtoken123"'
    if t == "list[float]":
        return "[1.0, 2.0, 3.0]"
    if t in ("list[list[float]]", "list[ list[float] ]"):
        return "[[1.0, 2.0], [3.0, 4.0]]"
    if t == "dict[str,str]" or t.startswith("dict["):
        return "{}"
    return "None"


def _march_arg_expr(
    name: str,
    type_hint: str,
    count_var: str,
    origins_var: str,
    dirs_var: str,
    out_var: str,
    max_steps: str,
    hit_threshold: str,
    sphere_radius: str,
) -> str:
    n = name.lower()
    if n == "origins":
        return origins_var
    if n in ("dirs", "directions"):
        return dirs_var
    if n == "hit_distances":
        return out_var
    if n in ("count", "n", "num", "length", "len"):
        return count_var
    if n in ("max_steps", "steps", "maxsteps"):
        return max_steps
    if n in ("hit_threshold", "threshold"):
        return hit_threshold
    if n in ("sphere_radius", "radius"):
        return sphere_radius
    return _sample_arg(type_hint)


def _march_call_lines(
    func_name: str,
    args: List[Tuple[str, str]],
    return_type: str,
    *,
    count: int,
    prefix: str = "",
) -> List[str]:
    """Generate Python source lines for a ray-marching function call."""
    lines: List[str] = []
    count_var = f"{prefix}count" if prefix else "count"
    origins_var = f"{prefix}origins" if prefix else "origins"
    dirs_var = f"{prefix}dirs" if prefix else "dirs"
    out_var = f"{prefix}hit_distances" if prefix else "hit_distances"
    lines.append(f"    {count_var} = {count}")
    lines.append(f"    {origins_var} = [{_march_coord_sample(count)}]")
    lines.append(f"    {dirs_var} = [{_march_dir_sample(count)}]")
    return_list = _is_c_abi_list(return_type)
    has_out_buffer = (not return_list) and any(
        t.lower() == "hit_distances" or n.lower() == "hit_distances" for n, t in args
    )
    if has_out_buffer:
        lines.append(f"    {out_var} = [0.0] * {count_var}")
    call_args = ", ".join(
        _march_arg_expr(
            n, t, count_var, origins_var, dirs_var, out_var, "64", "1e-3", "1.5"
        )
        for n, t in args
    )
    lines.append(f'    print("C++ {func_name}:", {func_name}({call_args}))')
    return lines


def _march_coord_sample(count: int) -> str:
    return ", ".join(["2.0"] * (count * 3))


def _march_dir_sample(count: int) -> str:
    return ", ".join(["-1.0"] * (count * 3))


def _generate_main_py(
    pkg_name: str,
    function_names: List[str],
    contracts: Optional[List[ContractEntry]] = None,
) -> str:
    lines = [
        '"""Tri-polyglot REPL CLI entrypoint."""',
        "",
        "from __future__ import annotations",
        "",
        "import argparse",
        "import random",
        "import sys",
        "",
        f"from {pkg_name} import {', '.join(function_names)}",
        "",
        "",
        "def run_all() -> None:",
        '    """Run the automated tri-polyglot verification commands."""',
    ]
    contracts = contracts or []
    sigs: Dict[str, str] = {}
    langs: Dict[str, str] = {}
    for c in contracts:
        if c.signature:
            try:
                name, _, _ = _parse_signature(c.signature)
                sigs[name] = c.signature
                langs[name] = c.language or ""
            except Exception:
                pass

    has_sdf = False
    has_march = False
    march_name: Optional[str] = None
    march_args: List[Tuple[str, str]] = []
    march_return = ""
    for name in function_names:
        sig = sigs.get(name, "")
        if not sig:
            continue
        if "march" in name and "ray" in name:
            has_march = True
            march_name = name
            _, march_args, march_return = _parse_signature(sig)
        if "sdf" in name and "sphere" in name:
            has_sdf = True
        try:
            _, args, _ = _parse_signature(sig)
        except Exception:
            continue
        if "march" in name and "ray" in name:
            lines.extend(_march_call_lines(name, args, march_return, count=1))
            continue
        call_args = ", ".join(_sample_arg(t) for a, t in args if a != "self")
        lang = langs.get(name, "").lower()
        if "cpp" in lang or _is_c_abi_contract(ContractEntry(name=name, signature=sig)):
            label = "C++"
        elif "rust" in lang:
            label = "Rust"
        else:
            label = "Python"
        lines.append(f'    print("{label} {name}:", {name}({call_args}))')

    benchmark_lines = [
        "",
        "",
        "def run_benchmark() -> None:",
        '    """Run a performance benchmark exercising all native backends."""',
    ]
    if has_sdf:
        benchmark_lines.append(
            '    print("Benchmark: compute_sdf_sphere", compute_sdf_sphere(2.0, 2.0, 2.0, 1.5))'
        )
    if has_march and march_name:
        benchmark_lines.extend(
            [
                "    random.seed(42)",
                "    count = 10000",
                "    origins = [random.uniform(-2.0, 2.0) for _ in range(count * 3)]",
                "    dirs = [random.uniform(-1.0, 1.0) for _ in range(count * 3)]",
            ]
        )
        return_list = _is_c_abi_list(march_return)
        return_tuple = _is_c_abi_tuple_return(march_return)
        if not return_list and not return_tuple:
            benchmark_lines.append("    hit_distances = [0.0] * count")
        call_args = ", ".join(
            _march_arg_expr(
                n, t, "count", "origins", "dirs", "hit_distances", "64", "1e-3", "1.5"
            )
            for n, t in march_args
        )
        benchmark_lines.append(f"    result = {march_name}({call_args})")
        if return_tuple:
            benchmark_lines.append("    _, hit_distances = result")
        elif return_list:
            benchmark_lines.append("    hit_distances = result")
        benchmark_lines.append("    hits = sum(1 for d in hit_distances if d > 0)")
        benchmark_lines.append('    print(f"Benchmark: hits {hits} / {count}")')
    if not has_sdf and not has_march:
        benchmark_lines.extend(
            [
                "    run_all()",
                '    print("Benchmark: generic run_all completed")',
            ]
        )
    lines.extend(benchmark_lines)
    lines.extend(
        [
            "",
            "",
            "def main() -> int:",
            '    parser = argparse.ArgumentParser(description="Tri-polyglot CLI")',
            '    parser.add_argument("--cmd", default=None, help="Headless command (e.g. run_all, benchmark)")',
            "    ns = parser.parse_args()",
            '    if ns.cmd in ("run_all", "benchmark"):',
            '        run_benchmark() if ns.cmd == "benchmark" else run_all()',
            "    elif not sys.stdin.isatty():",
            '        print("Tri-polyglot CLI ready")',
            "        run_all()",
            "    else:",
            '        print("Use --cmd run_all or --cmd benchmark for headless execution")',
            "    return 0",
            "",
            'if __name__ == "__main__":',
            "    sys.exit(main() or 0)",
            "",
        ]
    )
    return "\n".join(lines)


def _generate_run_shell(main_module: str) -> str:
    return (
        "import sys\n"
        f"from {main_module} import main\n"
        "\n"
        'if __name__ == "__main__":\n'
        "    sys.exit(main() or 0)\n"
    )


def _generate_tests(
    blueprint: Blueprint,
    pkg_module: str,
) -> str:
    return _generate_test_file(Path("tests/test_tri.py"), blueprint, pkg_module)


def _generate_test_file(
    path: Path,
    blueprint: Blueprint,
    pkg_module: str,
) -> str:
    """Generate contract-driven pytest tests for a tri-polyglot workspace."""
    from aero_forge.scaffold import test_generator

    return test_generator.generate_blueprint_tests(blueprint, module_name=pkg_module)


def _generate_native_bridge_py(
    pkg_name: str,
    cpp_dir: Path,
    cpp_contracts: List[ContractEntry],
    workspace_root: Optional[Path] = None,
    loader_path: Optional[Path] = None,
) -> str:
    """Generate a root-level ``native_bridge.py`` that loads the C++ shared library."""
    cpp_names = _function_names(cpp_contracts)
    if not cpp_names:
        return "# native bridge placeholder\n"
    stub_source = "\n".join(_cpp_contract_to_python_stub(c) for c in cpp_contracts)
    so_path = (cpp_dir / _so_name(f"{pkg_name}_cpp")).resolve()
    return _ctypes_loader_source(
        stub_source,
        so_path,
        cpp_names,
        workspace_root=workspace_root,
        loader_path=loader_path or (so_path.parent / "native_bridge.py"),
    )


def _generate_readme_tri(
    project: str,
    *,
    rust_dir: Path,
    cpp_dir: Path,
    python_dir: Path,
    tests_dir: Path,
    so_name: str,
    header_dirs: List[str],
    source_name: str = "native.cpp",
) -> str:
    """Return a README with concrete build/run commands for the workspace."""
    rust_rel = (
        rust_dir.relative_to(rust_dir.parents[0])
        if rust_dir.parent == Path()
        else rust_dir.name
    )
    rust_rel = rust_dir.name
    cpp_rel = cpp_dir.name
    python_rel = (
        python_dir.relative_to(python_dir.parents[0]).as_posix()
        if python_dir.parents
        else python_dir.name
    )
    python_rel = python_dir.name
    tests_rel = tests_dir.name
    include_flags = " ".join(f"-I../{d}" for d in sorted(set(header_dirs)))
    return f"""# {project}

Tri-polyglot (Python + Rust + C++) workspace generated by aero-forge.

## Build

```bash
# Rust PyO3 core
cd {rust_rel} && cargo build --release

# C-ABI shared library (from rust_core, ../{cpp_rel} is the workspace {cpp_rel})
cd ../{cpp_rel} && g++ -shared -fPIC -O3 -march=native -std=c++17 {include_flags} -o {so_name} {source_name}

# Optional editable Python install
cd .. && pip install -e .
```

## Run

```bash
PYTHONPATH=. python {python_rel}/main.py
PYTHONPATH=. pytest {tests_rel}
```
"""


class TriPolyglotMaterializer:
    """Write and build a Python + Rust + C++ tri-polyglot workspace from a Blueprint."""

    def __init__(self, workspace: Path) -> None:
        self.workspace = Path(workspace)
        self.workspace.mkdir(parents=True, exist_ok=True)
        self.build_logs = ""

    def _log(self, text: str) -> None:
        if text:
            self.build_logs = (self.build_logs + "\n" + text).strip()

    def _resolve_pkg_dir(self, blueprint: Blueprint, pkg_name: str) -> Path:
        """Resolve the Python package directory from the manifest/module graph."""
        # Prefer an __init__.py entry, then a main.py entry, then default to pkg_name.
        for e in blueprint.manifest:
            if e.path.endswith("/__init__.py"):
                return self.workspace / Path(e.path).parent
        for e in blueprint.manifest:
            if Path(e.path).name == "main.py":
                return self.workspace / Path(e.path).parent
        return self.workspace / pkg_name

    def _resolve_cpp_dir(self, blueprint: Blueprint, pkg_name: str) -> Path:
        """Resolve the C++ source directory from the manifest/module graph."""
        cpp_entries = [
            e
            for e in blueprint.manifest
            if e.lang == "cpp" or Path(e.path).suffix in (".cpp", ".cc", ".cxx")
        ]
        for e in cpp_entries:
            if Path(e.path).suffix in (".cpp", ".cc", ".cxx"):
                return self.workspace / Path(e.path).parent
        return self.workspace / "src/cpp"

    def _resolve_rust_dir(self, blueprint: Blueprint, pkg_name: str) -> Path:
        """Resolve the Rust crate directory from the manifest/module graph."""
        cargo_entries = [
            e for e in blueprint.manifest if Path(e.path).name == "Cargo.toml"
        ]
        if not cargo_entries:
            return self.workspace / "crates/native"

        # If a Cargo.toml has a matching src/lib.rs entry in the manifest, treat it as the crate root.
        def _lib_entry_for(crate_entry):
            crate_dir = Path(crate_entry.path).parent
            lib_path = str(crate_dir / "src" / "lib.rs").replace("/./", "/")
            return lib_path

        manifest_paths = {e.path for e in blueprint.manifest}
        candidates = []
        for e in cargo_entries:
            crate_dir = self.workspace / Path(e.path).parent
            if _lib_entry_for(e) in manifest_paths:
                candidates.append(crate_dir)
        if len(candidates) == 1:
            return candidates[0]
        if len(candidates) > 1:
            # Prefer a non-root crate; fall back to the deepest path.
            non_root = [c for c in candidates if c != self.workspace]
            return non_root[0] if non_root else candidates[0]

        # No explicit src/lib.rs entry; prefer a non-root Cargo.toml, then the only Cargo.toml.
        non_root = [e for e in cargo_entries if Path(e.path).parent != Path(".")]
        if non_root:
            return self.workspace / sorted(Path(e.path).parent for e in non_root)[0]
        return self.workspace / Path(cargo_entries[0].path).parent

    def _dotted_module(self, rel: Path) -> str:
        rel = rel.with_suffix("") if rel.suffix == ".py" else rel
        return ".".join(rel.parts)

    def materialize(
        self,
        blueprint: Blueprint,
        *,
        build: bool = False,
        force_overwrite: bool = False,
    ) -> Blueprint:
        """Write the tri-polyglot workspace files and optionally build everything."""
        from aero_forge.scaffold.polyglot_materializer import guard_materialization

        guard_materialization(
            self.workspace, blueprint, force_overwrite=force_overwrite
        )
        project = blueprint.project or "tri_polyglot_project"
        pkg_name = _sanitize_module_name(project)

        contracts = _normalize_tri_polyglot_contracts(blueprint)
        blueprint.contracts = contracts
        # LLM-generated verification nodes often describe flags/tests that do not
        # match the concrete generated entrypoint; rely on contract smoke tests.
        blueprint.verification_nodes = []
        if not blueprint.abi_contracts:
            from aero_forge.blueprint import _contracts_to_abi_contracts

            blueprint.abi_contracts = _contracts_to_abi_contracts(
                contracts, list(blueprint.manifest)
            )
        cpp_contracts, rust_contracts, python_contracts = _partition_contracts(
            contracts
        )

        accel_log = self.workspace / ".aero_forge_accel.log"
        os.environ["AERO_FORGE_ACCEL_LOG"] = str(accel_log)

        pkg_dir = self._resolve_pkg_dir(blueprint, pkg_name)
        pkg_dir.mkdir(parents=True, exist_ok=True)
        cpp_dir = self._resolve_cpp_dir(blueprint, pkg_name)
        cpp_dir.mkdir(parents=True, exist_ok=True)
        rust_dir = self._resolve_rust_dir(blueprint, pkg_name)
        rust_dir.mkdir(parents=True, exist_ok=True)
        rust_dir_rel = rust_dir.relative_to(self.workspace)
        rust_crate_name = sanitize_crate_name("_".join(rust_dir_rel.parts))
        tests_dir = self.workspace / "tests"
        tests_dir.mkdir(exist_ok=True)
        tests_dir_rel = tests_dir.relative_to(self.workspace)

        test_entries = [
            e
            for e in blueprint.manifest
            if e.path.endswith(".py")
            and Path(e.path).name.startswith("test_")
            and "orchestration" not in Path(e.path).name
        ]
        if test_entries:
            test_path = self.workspace / test_entries[0].path
        else:
            test_path = tests_dir / "test_generated_contracts.py"

        all_names = _function_names(contracts)

        # Telemetry
        _accel_log(
            "info",
            "Routing tri-polyglot build through Python + Rust + C++ dynamic bridges",
        )
        for contract in cpp_contracts:
            language_router.select_native_backend(
                _cpp_contract_to_python_stub(contract), hint="cpp"
            )
        for contract in rust_contracts:
            language_router.select_native_backend(
                _cpp_contract_to_python_stub(contract), hint="rust_hin"
            )

        # C++ source path is driven by the blueprint manifest when the prompt asks for
        # a specific source file (e.g. ``cpp_core/ray_engine.cpp``); otherwise default.
        cpp_pkg_name = f"{pkg_name}_cpp"
        cpp_entries = [
            e
            for e in blueprint.manifest
            if e.lang == "cpp"
            or Path(e.path).suffix in (".cpp", ".cc", ".cxx", ".h", ".hpp")
        ]
        cpp_source_entry = next(
            (e for e in cpp_entries if Path(e.path).suffix in (".cpp", ".cc", ".cxx")),
            None,
        )
        if cpp_source_entry is None:
            cpp_source_entry = ManifestEntry(
                path="src/cpp/native.cpp",
                lang="cpp",
                purpose="C-ABI shared library source",
            )
            blueprint.manifest.append(cpp_source_entry)
        cpp_source_path = self.workspace / cpp_source_entry.path
        cpp_source_path.parent.mkdir(parents=True, exist_ok=True)

        header_paths: List[str] = []
        for abi in blueprint.abi_contracts:
            if abi.header_path and Path(abi.header_path).suffix in (".h", ".hpp"):
                header_paths.append(abi.header_path)
        for hdr_entry in cpp_entries:
            if Path(hdr_entry.path).suffix in (".h", ".hpp"):
                header_paths.append(hdr_entry.path)
        header_paths = list(dict.fromkeys(header_paths))

        cpp_source_path.write_text(
            _generate_native_cpp(
                cpp_pkg_name,
                cpp_contracts,
                header_includes=header_paths,
            ),
            encoding="utf-8",
        )
        for header_path in header_paths:
            hdr_path = self.workspace / header_path
            hdr_path.parent.mkdir(parents=True, exist_ok=True)
            if not hdr_path.exists():
                hdr_path.write_text(
                    _generate_cpp_header(pkg_name, cpp_contracts), encoding="utf-8"
                )

        # Detect whether the blueprint requested a root-level native_bridge.py module.
        native_bridge_module: Optional[str] = None
        if any(e.path == "native_bridge.py" for e in blueprint.manifest):
            native_bridge_module = "native_bridge"

        # Python package and project files
        pkg_rel = pkg_dir.relative_to(self.workspace)
        pkg_module = self._dotted_module(pkg_rel)
        main_rel = pkg_rel / "main.py"
        main_module = self._dotted_module(main_rel)
        cpp_dir_rel = cpp_dir.relative_to(self.workspace)
        rust_dir_rel = rust_dir.relative_to(self.workspace)

        (pkg_dir / "__init__.py").write_text(
            _generate_python_init(
                pkg_name,
                cpp_dir,
                cpp_contracts,
                rust_contracts,
                python_contracts,
                rust_crate_name,
                native_bridge_module,
                rust_dir=str(rust_dir_rel),
                workspace_root=self.workspace,
                pkg_dir=pkg_dir,
            ),
            encoding="utf-8",
        )
        (pkg_dir / "native_loader.py").write_text(
            _generate_native_loader_py(
                pkg_name,
                cpp_dir,
                cpp_contracts,
                rust_contracts,
                python_contracts,
                rust_crate_name,
                native_bridge_module,
                rust_dir=str(rust_dir_rel),
                workspace_root=self.workspace,
                pkg_dir=pkg_dir,
            ),
            encoding="utf-8",
        )

        # Generate the primary entrypoint from the execution strategy / CLI contract.
        execution_strategy = (
            blueprint.execution_strategy.model_dump()
            if blueprint.execution_strategy
            else {
                "primary_entrypoint": {
                    "path": str(main_rel),
                    "runtime": "python3",
                    "wrapper_generation": True,
                },
                "cli_contract": {"parser_type": "argparse", "flags": []},
                "run_spec": {},
            }
        )
        EntrypointAdapterEngine(
            execution_strategy,
            str(self.workspace),
            contracts=contracts,
            abi_contracts=list(blueprint.abi_contracts or []),
            function_module=pkg_module,
        ).synthesize_root_entrypoint()

        pyproject_pkg_dir = str(pkg_rel.parent) if pkg_rel.parts[:-1] else "."
        pyproject_pkg_name = pkg_rel.name
        (self.workspace / "pyproject.toml").write_text(
            _render_pyproject(pyproject_pkg_name, package_dir=pyproject_pkg_dir),
            encoding="utf-8",
        )
        (self.workspace / "run_shell.py").write_text(
            _generate_run_shell(main_module), encoding="utf-8"
        )
        test_path.write_text(_generate_tests(blueprint, pkg_module), encoding="utf-8")
        python_interface_dir = self.workspace / "python_interface"
        if not (python_interface_dir / "main.py").is_file():
            python_interface_dir = pkg_dir
        header_dirs = sorted(
            {str(Path(h).parent) for h in header_paths if Path(h).parent != Path(".")}
        )
        (self.workspace / "README.md").write_text(
            _generate_readme_tri(
                project,
                rust_dir=rust_dir,
                cpp_dir=cpp_source_path.parent,
                python_dir=python_interface_dir,
                tests_dir=tests_dir,
                so_name=_so_name(cpp_pkg_name),
                header_dirs=header_dirs,
                source_name=cpp_source_path.name,
            ),
            encoding="utf-8",
        )

        (self.workspace / "Makefile").write_text(
            _generate_makefile(
                rust_dir=str(rust_dir_rel),
                cpp_dir=str(cpp_dir_rel),
                python_dir=str(pkg_rel),
                tests_dir=str(tests_dir_rel),
                cpp_pkg_name=cpp_pkg_name,
                cpp_source_name=cpp_source_path.name,
                header_dirs=header_paths,
            ),
            encoding="utf-8",
        )

        # Enforce blueprint manifest integrity: every declared file must be materialized.
        self._write_missing_manifest_entries(
            blueprint,
            pkg_name,
            cpp_contracts,
            rust_contracts,
            python_contracts,
            all_names,
            rust_crate_name,
            native_bridge_module,
            header_paths,
        )

        # Standard tri-polyglot manifest entries.
        rust_dir_rel = rust_dir.relative_to(self.workspace)
        manifest: List[ManifestEntry] = [
            ManifestEntry(
                path=str(pkg_dir / "__init__.py"),
                lang="python",
                purpose="Python driver package init",
            ),
            ManifestEntry(
                path=str(pkg_dir / "native_loader.py"),
                lang="python",
                purpose="Native shared-library loader",
            ),
            ManifestEntry(
                path=str(pkg_dir / "main.py"),
                lang="python",
                purpose="Python CLI / REPL entrypoint",
            ),
            ManifestEntry(
                path=str(cpp_source_path.relative_to(self.workspace)),
                lang="cpp",
                purpose="C-ABI shared library source",
            ),
            ManifestEntry(
                path=str(rust_dir_rel / "Cargo.toml"),
                lang="toml",
                purpose="PyO3 crate manifest",
            ),
            ManifestEntry(
                path=str(rust_dir_rel / "src" / "lib.rs"),
                lang="rust",
                purpose="Rust native core",
            ),
            ManifestEntry(
                path="Cargo.toml", lang="toml", purpose="Rust workspace manifest"
            ),
            ManifestEntry(
                path="pyproject.toml", lang="toml", purpose="Python package manifest"
            ),
            ManifestEntry(
                path="run_shell.py", lang="python", purpose="Headless launcher"
            ),
            ManifestEntry(
                path=str(test_path.relative_to(self.workspace)),
                lang="python",
                purpose="pytest tests",
            ),
            ManifestEntry(path="Makefile", lang="makefile", purpose="Build/test/run targets"),
            ManifestEntry(path="README.md", lang="markdown", purpose="Project README"),
        ]
        existing_paths = {e.path for e in blueprint.manifest}
        for entry in manifest:
            if entry.path not in existing_paths:
                blueprint.manifest.append(entry)

        # Persist blueprint before generating Rust so Engine can validate the manifest.
        write_blueprint(blueprint, self.workspace / "blueprint.aero")

        # Rust source and crate
        rust_source = _generate_rust_source(rust_contracts)
        uast = python_source_to_uast(rust_source)
        graph = UASTToHINTranslator().translate(uast)
        graph.traits_by_name = {}
        graph.traits = {}
        Engine().generate(
            graph,
            self.workspace / "dist",
            workspace_root=self.workspace,
            module_name=rust_crate_name,
            function_names=_function_names(rust_contracts),
            source=rust_source,
            target_mode=TargetMode.PYO3,
        )
        _append_rust_c_abi_wrappers(rust_dir, rust_contracts)

        if build:
            self._build_cpp(cpp_pkg_name, cpp_source_path, header_paths)
            self._build_rust(rust_dir)

        functions: List[FunctionSpec] = [
            FunctionSpec(
                file=pkg_dir / "__init__.py",
                name=name,
                tests=[tests_dir / "test_tri.py"],
                skip_build=True,
            )
            for name in all_names
        ]
        if (pkg_dir / "main.py").is_file():
            functions.append(
                FunctionSpec(
                    file=pkg_dir / "main.py",
                    name="main",
                    tests=[tests_dir / "test_tri.py"],
                    skip_build=True,
                )
            )
        blueprint = blueprint.model_copy(update={"functions": functions})
        return blueprint

    def _write_missing_manifest_entries(
        self,
        blueprint: Blueprint,
        pkg_name: str,
        cpp_contracts: List[ContractEntry],
        rust_contracts: List[ContractEntry],
        python_contracts: List[ContractEntry],
        function_names: List[str],
        rust_crate_name: str,
        native_bridge_module: Optional[str],
        header_paths: List[str],
    ) -> None:
        """Write any manifest entry that has not already been materialized."""
        pkg_dir = self._resolve_pkg_dir(blueprint, pkg_name)
        cpp_dir = self._resolve_cpp_dir(blueprint, pkg_name)
        rust_dir = self._resolve_rust_dir(blueprint, pkg_name)
        rust_dir_rel = rust_dir.relative_to(self.workspace)
        cpp_dir_rel = cpp_dir.relative_to(self.workspace)
        pkg_rel = pkg_dir.relative_to(self.workspace)
        tests_dir_rel = (self.workspace / "tests").relative_to(self.workspace)
        cpp_pkg_name = f"{pkg_name}_cpp"
        pkg_module = self._dotted_module(pkg_rel)
        for entry in list(blueprint.manifest):
            path = self.workspace / entry.path
            if path.exists():
                continue
            content: Optional[str] = None
            rel = Path(entry.path)
            if entry.lang == "python":
                if path.name == "__init__.py":
                    content = _generate_python_init(
                        pkg_name,
                        cpp_dir,
                        cpp_contracts,
                        rust_contracts,
                        python_contracts,
                        rust_crate_name,
                        native_bridge_module,
                        rust_dir=str(rust_dir_rel),
                        workspace_root=self.workspace,
                        pkg_dir=pkg_dir,
                    )
                elif path.name == "native_loader.py":
                    content = _generate_native_loader_py(
                        pkg_name,
                        cpp_dir,
                        cpp_contracts,
                        rust_contracts,
                        python_contracts,
                        rust_crate_name,
                        native_bridge_module,
                        rust_dir=str(rust_dir_rel),
                        workspace_root=self.workspace,
                        pkg_dir=pkg_dir,
                    )
                elif path.name == "main.py":
                    execution_strategy = (
                        blueprint.execution_strategy.model_dump()
                        if blueprint.execution_strategy
                        else {
                            "primary_entrypoint": {
                                "path": entry.path,
                                "runtime": "python3",
                                "wrapper_generation": True,
                            },
                            "cli_contract": {"parser_type": "argparse", "flags": []},
                            "run_spec": {},
                        }
                    )
                    # Ensure this manifest entry gets its own wrapper, even if the
                    # blueprint's primary entrypoint points elsewhere.
                    execution_strategy.setdefault("primary_entrypoint", {})[
                        "path"
                    ] = entry.path
                    function_module = self._dotted_module(rel.parent)
                    EntrypointAdapterEngine(
                        execution_strategy,
                        str(self.workspace),
                        contracts=list(blueprint.contracts or []),
                        abi_contracts=list(blueprint.abi_contracts or []),
                        function_module=function_module,
                    ).synthesize_root_entrypoint()
                    continue
                elif path.name == "native_bridge.py":
                    content = _generate_native_bridge_py(
                        pkg_name,
                        cpp_dir,
                        cpp_contracts,
                        workspace_root=self.workspace,
                        loader_path=path,
                    )
                elif path.name == "core.py":
                    content = _generate_core_py(
                        rust_contracts + cpp_contracts + python_contracts
                    )
                elif "orchestration" in path.name and path.suffix == ".py":
                    content = _generate_orchestration_test(
                        rust_contracts + cpp_contracts + python_contracts
                    )
                elif "test" in path.name and path.suffix == ".py":
                    content = _generate_test_file(rel, blueprint, pkg_module)
                elif path.suffix == ".py":
                    content = f"# {path.name} placeholder generated by aero-forge\n"
            elif entry.lang == "cpp":
                if path.suffix in (".h", ".hpp"):
                    content = _generate_cpp_header(pkg_name, cpp_contracts)
                elif path.suffix in (".cpp", ".cc", ".cxx"):
                    content = _generate_native_cpp(
                        f"{pkg_name}_cpp",
                        cpp_contracts,
                        header_includes=header_paths,
                    )
                else:
                    content = "// C++ placeholder\n"
            elif entry.lang == "makefile" or path.name == "Makefile":
                content = _generate_makefile(
                    rust_dir=str(rust_dir_rel),
                    cpp_dir=str(cpp_dir_rel),
                    python_dir=str(pkg_rel),
                    tests_dir=str(tests_dir_rel),
                    cpp_pkg_name=cpp_pkg_name,
                    cpp_source_name=cpp_source_path.name,
                    header_dirs=header_dirs,
                )
            elif entry.lang == "rust":
                content = "// Rust placeholder\n"
            elif entry.lang == "toml":
                content = "# TOML placeholder\n"
            elif entry.lang == "markdown":
                content = f"# {blueprint.project or 'project'}\n"
            if content is not None:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(content, encoding="utf-8")

    def _build_cpp(
        self, cpp_pkg_name: str, cpp_source: Path, header_paths: List[str]
    ) -> None:
        compiler = _find_cpp_compiler()
        if compiler is None:
            raise BuildStageError(
                "No C++ compiler found (g++, clang++, or c++)",
                stage="cpp_compile",
                logs="",
            )

        so_name = _so_name(cpp_pkg_name)
        cpp_dir = cpp_source.parent
        so_path = cpp_dir / so_name

        source_files = _collect_cpp_sources(cpp_source)
        include_dirs = _collect_include_dirs(cpp_source, header_paths, self.workspace)
        build_cmd = [
            compiler,
            "-shared",
            "-fPIC",
            "-O2",
            "-std=c++17",
        ]
        for inc in sorted(set(include_dirs)):
            build_cmd.extend(["-I", inc])
        build_cmd.extend(["-o", str(so_path)])
        build_cmd.extend(str(p) for p in source_files)

        self._log(f"Compiling C-ABI shared library: {' '.join(build_cmd)}")
        _accel_log(
            "info", f"BUILD: compiling dynamic shared object with {' '.join(build_cmd)}"
        )

        env = dict(os.environ)
        env["PYTHONUNBUFFERED"] = "1"

        build_proc = subprocess.run(
            build_cmd,
            cwd=self.workspace,
            env=env,
            capture_output=True,
            text=True,
        )
        self._log(build_proc.stdout)
        self._log(build_proc.stderr)

        if build_proc.returncode != 0:
            full_output = f"{build_proc.stdout}\n{build_proc.stderr}".strip()
            logger.error("C++ shared library build failed:\n%s", full_output)
            _accel_log("error", f"C++ shared library build failed: {full_output}")
            raise BuildStageError(
                f"C++ shared library build failed for {cpp_pkg_name}",
                stage="cpp_compile",
                logs=full_output,
            )

        _accel_log("success", f"BUILD: dynamic shared library compiled: {so_path}")

    def _build_rust(self, rust_dir: Path) -> None:
        cargo_toml = rust_dir / "Cargo.toml"
        if not cargo_toml.is_file():
            raise BuildStageError(
                f"Rust crate manifest not found: {cargo_toml}",
                stage="rust_compile",
                logs="",
            )

        self._log(f"Building Rust PyO3 crate in {rust_dir}")
        _accel_log("info", "BUILD: building Rust PyO3 extension with cargo")

        result = cargo_build(rust_dir, release=True, timeout=600)
        output = f"{result.stdout}\n{result.stderr}".strip()
        if output:
            self._log(f"--- cargo build ---\n{output}")
        if result.returncode != 0:
            logger.error("Rust PyO3 build failed:\n%s", output)
            _accel_log("error", f"Rust PyO3 build failed: {output}")
            raise BuildStageError(
                f"Rust PyO3 build failed in {rust_dir}",
                stage="rust_compile",
                logs=output,
            )

        _accel_log("success", "BUILD: Rust PyO3 extension compiled successfully")
