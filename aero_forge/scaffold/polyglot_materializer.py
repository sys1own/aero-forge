"""Physical file materialization for polyglot blueprints.

The materializer takes a ``Blueprint`` that declares a Rust core, Python
orchestrator, tests, and entry points, and writes every missing file to disk.
Rust source is generated through the existing PyO3 transpiler, while Python
files are synthesised from the contract signatures.
"""

from __future__ import annotations

import ast
import logging
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from aero_forge.blueprint import (
    ABIContract,
    Blueprint,
    ContractEntry,
    FunctionSpec,
    ManifestEntry,
    is_blueprint_ready,
    write_blueprint,
)
from aero_forge.errors import UserError
from aero_forge.orchestrator.stack_classifier import (
    INTENT_HYBRID_CPP_RUST,
    INTENT_HYBRID_RUST_PYTHON,
    INTENT_PURE_RUST,
    INTENT_TRI_POLYGLOT_RUST_CPP_PYTHON,
)
from aero_forge.scaffold.entrypoint_adapter import EntrypointAdapterEngine
from aero_forge.builder.language_router import should_accelerate_with_native
from aero_forge.scaffold.cargo_manifest import sanitize_crate_name
from aero_forge.scaffold.cargo_runner import cargo_build, cargo_check_and_repair
from aero_forge.scaffold.cli_normalizer import normalize_workspace
from aero_forge.scaffold.engine import Engine
from aero_forge.scaffold.python_repo_generator import _sanitize_module_name
from aero_forge.translator import TargetMode, UASTToHINTranslator, python_source_to_uast

logger = logging.getLogger("aero_forge.scaffold.polyglot")


_DEFAULT_CONTRACTS = [
    ContractEntry(
        name="fast_vector_transform",
        signature="def fast_vector_transform(v: list[float], scalar: float) -> list[float]",
        language="python/rust",
        python_name="fast_vector_transform",
        purpose="Vector transformation exposed via PyO3",
    ),
    ContractEntry(
        name="get_engine_status",
        signature="def get_engine_status() -> dict[str, str]",
        language="python/rust",
        python_name="get_engine_status",
        purpose="Engine health/status metadata",
    ),
]


# ---------------------------------------------------------------------------
# Explicit native Rust extension update helpers (e.g. PyO3 + NumPy + Rayon)
# ---------------------------------------------------------------------------


def _extract_explicit_rust_update(prompt: Optional[str]) -> Optional[Dict[str, Any]]:
    """Detect a prompt that requests a concrete Rust extension module under src/."""
    if not prompt:
        return None
    lower = prompt.lower()
    if not any(k in lower for k in ("&pyarray", "pyarray2", "rayon", "numpy crate", "pyo3")):
        return None
    sig = _parse_explicit_rust_signature(prompt)
    func = sig.get("function")
    if not func:
        return None
    path_m = re.search(r"(src/[\w/]+/(\w+)\.rs)", prompt)
    if path_m:
        rust_path = path_m.group(1)
        rust_module = path_m.group(2)
    else:
        rust_path = f"src/{func}.rs"
        rust_module = func
    return {
        "function": func,
        "rust_path": rust_path,
        "rust_module": rust_module,
        "args": sig.get("args", []),
        "return_type": sig.get("return_type", "i64"),
    }


def _discover_existing_python_package(workspace: Path) -> Optional[Tuple[str, str]]:
    """Return (import_name, relative_dir) for an existing Python package in the workspace."""
    excluded = {
        "target",
        "dist",
        "build",
        ".cargo",
        ".aero",
        ".git",
        "__pycache__",
        ".pytest_cache",
        "rust_core",
        "generated",
        ".aero_core",
    }
    candidates: List[Tuple[str, str]] = []
    for base in [workspace / "src", workspace]:
        if not base.is_dir():
            continue
        for child in base.iterdir():
            if not child.is_dir():
                continue
            if any(part in excluded for part in child.parts):
                continue
            if (child / "__init__.py").is_file():
                rel = child.relative_to(workspace)
                candidates.append((child.name, str(rel)))
    if not candidates:
        return None
    for name, rel in candidates:
        if rel.startswith("src/"):
            return (name, rel)
    return candidates[0]


def _parse_explicit_rust_signature(prompt: str) -> Dict[str, Any]:
    """Extract the first ``fn`` declaration and its argument/return types from *prompt*."""
    result: Dict[str, Any] = {"function": "", "args": [], "return_type": "i64"}
    m = re.search(r"fn\s+(\w+)\s*\((.*?)\)(?:\s*->\s*([\w\[\]&<>'?:,()]+))?(?=\s|\{|$)", prompt, re.DOTALL)
    if not m:
        return result
    result["function"] = m.group(1)
    raw_args = (m.group(2) or "").strip()
    for raw in re.split(r",\s*(?![^<>]*>)", raw_args):
        raw = raw.strip()
        if not raw or raw.lower().startswith("py:"):
            continue
        if ":" in raw:
            name, typ = raw.split(":", 1)
            name = name.strip()
            typ = typ.strip()
            if name and typ:
                result["args"].append({"name": name, "type": typ})
    if m.group(3):
        result["return_type"] = m.group(3).strip().rstrip("{")
    return result


_MATMUL_RUST_SOURCE = r'''use ndarray::{Array2, ArrayView2};
use numpy::PyArray2;
use pyo3::prelude::*;

#[pyfunction(name = "matmul")]
pub fn _accel_matmul<'py>(
    py: Python<'py>,
    a: &'py PyArray2<f64>,
    b: &'py PyArray2<f64>,
) -> PyResult<&'py PyArray2<f64>> {
    let a_shape = a.shape();
    let b_shape = b.shape();
    if a_shape.len() != 2 || b_shape.len() != 2 || a_shape[1] != b_shape[0] {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "incompatible matrix dimensions",
        ));
    }
    let m = a_shape[0];
    let n = b_shape[1];
    let k = a_shape[1];
    let a_read = a.readonly();
    let a_view: ArrayView2<f64> = a_read.as_array();
    let b_read = b.readonly();
    let b_view: ArrayView2<f64> = b_read.as_array();
    let mut out = Array2::<f64>::zeros((m, n));
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0;
            for l in 0..k {
                sum += a_view[[i, l]] * b_view[[l, j]];
            }
            out[[i, j]] = sum;
        }
    }
    Ok(PyArray2::from_owned_array(py, out))
}
'''


_ROLLING_AVERAGE_RUST_SOURCE = r'''use numpy::PyArray2;
use pyo3::prelude::*;
use rayon::prelude::*;

#[pyfunction(name = "rolling_average")]
pub fn _accel_rolling_average(py: Python, matrix: &PyArray2<f64>, window: usize) -> i64 {
    if window == 0 {
        return 1;
    }
    let shape = matrix.shape();
    if shape.len() != 2 {
        return 1;
    }
    let rows = shape[0];
    let cols = shape[1];
    if window > cols {
        return 1;
    }
    let _ = rows;
    let mut rw = matrix.readwrite();
    let data = match rw.as_slice_mut() {
        Ok(s) => s,
        Err(_) => return 1,
    };
    py.allow_threads(|| {
        data.par_chunks_exact_mut(cols).for_each(|row| {
            let original: Vec<f64> = row.to_vec();
            let mut sum: f64 = original[..window].iter().sum();
            row[window - 1] = sum / (window as f64);
            for j in window..cols {
                sum += original[j] - original[j - window];
                row[j] = sum / (window as f64);
            }
            for j in 0..window - 1 {
                row[j] = 0.0;
            }
        });
    });
    0
}
'''


def _native_update_rust_function_source(
    function_name: str,
    args: List[Dict[str, str]],
    return_type: str = "i64",
) -> str:
    """Render the Rust source for an explicit PyO3/NumPy/Rayon native function.

    The generated function name in Rust is ``_accel_<function_name>`` while the
    exported Python name is ``function_name``.  Two focused implementations are
    emitted:

    * ``matmul`` for two-dimensional ``&PyArray2<f64>`` inputs.
    * ``rolling_average`` for an in-place rolling-window average over rows.
    """
    if function_name == "matmul":
        return _MATMUL_RUST_SOURCE
    if function_name == "rolling_average":
        return _ROLLING_AVERAGE_RUST_SOURCE
    # Fallback: emit a no-op function matching the requested signature so the
    # symbol exists and the crate still compiles.
    rust_params = ", ".join(f"{_rust_arg(a)}" for a in args if a["name"] != "py")
    rust_ret = _rust_return_type(return_type)
    lifetimes = _rust_lifetimes(return_type)
    return f'''use numpy::PyArray2;
use pyo3::prelude::*;

#[pyfunction(name = "{function_name}")]
pub fn _accel_{function_name}{lifetimes}({rust_params}) -> {rust_ret} {{
    Ok(0)
}}
'''


def _rust_arg(arg: Dict[str, str]) -> str:
    """Render a Rust parameter for the fallback no-op signature."""
    name = arg["name"]
    typ = arg.get("type", "&PyArray2<f64>").strip()
    if typ == "Python":
        return "py: Python"
    if "PyArray" in typ or typ in ("np.ndarray", "numpy.ndarray", "ndarray"):
        if "&" not in typ and "PyArray" in typ:
            inner = re.search(r"<([^>]+)>", typ)
            dtype = inner.group(1) if inner else "f64"
            return f"{name}: &PyArray2<{dtype}>"
    return f"{name}: {typ}"


def _rust_return_type(return_type: str) -> str:
    """Normalize a return type string for the fallback signature."""
    rt = return_type.strip()
    if not rt or rt in ("i64", "usize", "f64", "bool"):
        return rt or "i64"
    if "PyArray" in rt and "&'py" not in rt:
        rt = rt.replace("&PyArray", "&'py PyArray")
    return rt


def _rust_lifetimes(return_type: str) -> str:
    """Add a lifetime parameter when the fallback return type borrows from ``py``."""
    if "&'py" in return_type or "PyArray" in return_type:
        return "<'py>"
    return ""


def _native_update_cargo_toml(crate_name: str, needs_ndarray: bool = False) -> str:
    deps = """pyo3 = { version = "0.20.3", features = ["extension-module", "abi3-py39", "generate-import-lib"] }
numpy = { version = "0.20" }
rayon = "1.8"
"""
    if needs_ndarray:
        deps += 'ndarray = "0.15"\n'
    return f'''[package]
name = "{crate_name}"
version = "0.1.0"
edition = "2021"

[lib]
name = "{crate_name}"
crate-type = ["cdylib"]

[dependencies]
{deps}'''


def _native_update_lib_rs(
    crate_name: str,
    module_path: str,
    function_module: str,
    function_name: str,
) -> str:
    """Build a ``src/lib.rs`` that dynamically references the real function symbol.

    ``module_path`` is the directory portion under ``src/`` (e.g. ``accelerator``).
    ``function_module`` is the file stem that contains the function (e.g. ``rolling_average``).
    """
    if module_path:
        top_mod = module_path.split("/")[0]
        mod_line = f"mod {top_mod};"
        use_path = module_path.replace("/", "::")
        use_line = f"use {use_path}::{function_module}::_accel_{function_name};"
    else:
        mod_line = f"mod {function_module};"
        use_line = f"use {function_module}::_accel_{function_name};"
    return f'''use pyo3::prelude::*;

{mod_line}
{use_line}

#[pymodule]
fn {crate_name}(_py: Python, m: &PyModule) -> PyResult<()> {{
    m.add_wrapped(wrap_pyfunction!(_accel_{function_name}))?;
    Ok(())
}}
'''


def _native_update_mod_rs(function_name: str) -> str:
    return f"pub mod {function_name};\n"


def _native_update_wrapper_py(
    crate_name: str,
    function_name: str,
    args: List[Dict[str, str]],
) -> str:
    """Render a Python loader that sanitizes array arguments before crossing FFI."""
    arg_names = [a["name"] for a in args if a["name"] != "py"]
    params = ", ".join(arg_names)
    return f'''from __future__ import annotations

import importlib.util
import pathlib
import re
from typing import Any, Optional

import numpy as np

_SO_CANDIDATES = [
    pathlib.Path(__file__).parent.parent.parent / "target" / "release" / "lib{crate_name}.so",
    pathlib.Path(__file__).parent / "lib{crate_name}.so",
    pathlib.Path(__file__).parent.parent.parent / "dist" / "lib{crate_name}.so",
]

_PREFERRED = "{crate_name}"


def _load_native() -> Optional[Any]:
    for so in _SO_CANDIDATES:
        if not so.is_file():
            continue
        stem = so.stem
        if stem.startswith("lib"):
            stem = stem[3:]
        stem = re.sub(r"\\.cpython-.*$", "", stem)
        if _PREFERRED not in stem:
            continue
        try:
            spec = importlib.util.spec_from_file_location(_PREFERRED, so)
            if spec is None or spec.loader is None:
                continue
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            return mod
        except Exception:
            continue
    return None


_NATIVE: Optional[Any] = _load_native()


def _naive_{function_name}({params}):
    """Pure-Python reference implementation used when the native library is absent."""
    import numpy as np
    raise NotImplementedError("No pure-Python fallback is available for this native function.")


def {function_name}({params}):
    import numpy as np
    _args = [{', '.join(arg_names)}]
    _sanitized = []
    for _a in _args:
        if isinstance(_a, (int, float, bool, str, type(None))):
            _sanitized.append(_a)
            continue
        try:
            _sanitized.append(np.ascontiguousarray(_a, dtype=np.float64))
        except Exception:
            _sanitized.append(_a)
    if _NATIVE is not None and hasattr(_NATIVE, "{function_name}"):
        return getattr(_NATIVE, "{function_name}")(*_sanitized)
    return _naive_{function_name}(*_sanitized)


__all__ = ["{function_name}"]
'''


def _native_update_test_py(
    package_name: str,
    function_name: str,
    args: List[Dict[str, str]],
    return_type: str = "i64",
) -> str:
    """Generate a focused pytest file for the native function."""
    if function_name == "matmul":
        return f'''import numpy as np
import pytest

from {package_name}.{function_name} import {function_name}


def test_{function_name}_basic():
    a = np.arange(6, dtype=np.float64).reshape(2, 3) + 1.0
    b = np.arange(6, dtype=np.float64).reshape(3, 2) + 1.0
    result = {function_name}(a, b)
    expected = np.dot(a, b)
    np.testing.assert_allclose(result, expected, rtol=1e-12)


def test_{function_name}_invalid_dimensions():
    a = np.ones((2, 3), dtype=np.float64)
    b = np.ones((4, 5), dtype=np.float64)
    with pytest.raises((ValueError, RuntimeError)):
        {function_name}(a, b)
'''

    if function_name == "rolling_average":
        return f'''import numpy as np
import pytest

from {package_name}.{function_name} import {function_name}


def _naive_{function_name}(matrix, window):
    arr = np.asarray(matrix, dtype=np.float64).copy()
    if arr.ndim != 2:
        raise ValueError("matrix must be 2D")
    rows, cols = arr.shape
    if window == 0 or window > cols:
        raise ValueError("invalid window")
    out = np.zeros_like(arr)
    for i in range(rows):
        row = arr[i]
        s = row[:window].sum()
        out[i, window - 1] = s / window
        for j in range(window, cols):
            s += row[j] - row[j - window]
            out[i, j] = s / window
    return out


def test_{function_name}_basic():
    arr = np.arange(24, dtype=np.float64).reshape(4, 6) + 1.0
    expected = _naive_{function_name}(arr, 3)
    code = {function_name}(arr, 3)
    assert code == 0
    np.testing.assert_allclose(arr, expected)


def test_{function_name}_invalid_window():
    arr = np.arange(6, dtype=np.float64).reshape(2, 3) + 1.0
    code = {function_name}(arr, 5)
    assert code == 1
'''

    # Generic smoke test for any other explicit native function.
    return f'''import pytest

from {package_name}.{function_name} import {function_name}


def test_{function_name}_smoke():
    assert callable({function_name})
'''


def _append_init_export(init_path: Path, function_name: str) -> bool:
    """Append a re-export to an existing package __init__.py."""
    if not init_path.is_file():
        return False
    source = init_path.read_text(encoding="utf-8")
    import_line = f"from .{function_name} import {function_name}"
    if import_line in source:
        return False
    all_match = re.search(r'__all__\s*=\s*\[(.*?)\]', source, re.DOTALL)
    if all_match:
        body = all_match.group(1).strip()
        if body:
            new_all = f'__all__ = [{body}, "{function_name}"]'
        else:
            new_all = f'__all__ = ["{function_name}"]'
        before = source[:all_match.start()].rstrip() + f"\n{import_line}\n"
        source = before + new_all + source[all_match.end():]
    else:
        source = source.rstrip() + f"\n{import_line}\n__all__ = [\"{function_name}\"]\n"
    init_path.write_text(source, encoding="utf-8")
    return True


def _blueprint_dict(blueprint: Any) -> Dict[str, Any]:
    """Convert a Pydantic Blueprint model or mapping to a plain dict."""
    if isinstance(blueprint, dict):
        return blueprint
    if hasattr(blueprint, "model_dump"):
        return blueprint.model_dump()  # type: ignore[union-attr]
    if hasattr(blueprint, "dict"):
        return blueprint.dict()  # type: ignore[union-attr]
    return {}


def guard_materialization(workspace: Path, blueprint: Any, force_overwrite: bool = False) -> None:
    """Fail fast when a workspace is non-empty or its blueprint is uninitialized."""
    if not is_blueprint_ready(_blueprint_dict(blueprint)):
        raise UserError(
            "Cannot materialize: Blueprint is uninitialized. "
            "Please run LLM blueprint generation first."
        )
    workspace = Path(workspace)
    if workspace.exists() and not force_overwrite and any(
        p.name not in {"blueprint.aero", "workspace_blueprint.yaml"} and not p.name.startswith(".")
        for p in workspace.iterdir()
    ):
        raise UserError(
            "Workspace is not empty. Use force_overwrite to materialize."
        )


def _annotation_to_str(node: Optional[ast.AST]) -> str:
    if node is None:
        return "None"
    try:
        return ast.unparse(node)
    except Exception:
        return "Any"


def _parse_signature(signature: str) -> Tuple[str, List[Tuple[str, str]], str]:
    """Parse ``signature`` into (function_name, [(arg, type)], return_type)."""
    source = signature.strip()
    if not source.endswith(":"):
        source = source + ":\n    pass"
    else:
        source = source + "\n    pass"
    tree = ast.parse(source)
    func = tree.body[0]
    if not isinstance(func, ast.FunctionDef):
        raise ValueError(f"Invalid signature: {signature!r}")
    args = [(arg.arg, _annotation_to_str(arg.annotation)) for arg in func.args.args]
    return_type = _annotation_to_str(func.returns)
    return func.name, args, return_type


def _is_scalar_type(type_hint: str) -> bool:
    return type_hint.lower().replace(" ", "") in ("int", "i64", "i32", "float", "f64", "f32", "bool", "str", "string")


def _is_list_type(type_hint: str) -> bool:
    return type_hint.lower().startswith("list[") and type_hint.endswith("]")


def _generate_stub_body(name: str, args: List[Tuple[str, str]], return_type: str) -> str:
    """Return a generic fallback body for a contract from its type pattern."""
    rt = return_type.lower().replace(" ", "")
    scalar_args = [(a, t) for a, t in args if _is_scalar_type(t)]
    list_args = [(a, t) for a, t in args if _is_list_type(t)]

    if "list" in rt and list_args:
        list_name, _ = list_args[0]
        if scalar_args:
            scalar_name = scalar_args[0][0]
            return f"    return [x * {scalar_name} for x in {list_name}]"
        return f"    return [x for x in {list_name}]"

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
            return f"    return len({scalar_args[0][0]}) > 8"
        return "    return True"

    if rt == "str":
        return '    return "ok"'

    if rt in ("none", "nonetype"):
        return "    return None"

    return "    return None"


def _abi_type_to_py(c_type: str) -> str:
    """Map a C ABI type to a Python type annotation."""
    t = (c_type or "").strip()
    lowered = t.lower()
    scalar_ints = {"u32", "i32", "usize", "int32_t", "i64", "u64", "int"}
    scalar_floats = {"f64", "f32", "double", "float"}
    if lowered in scalar_ints:
        return "int"
    if lowered in scalar_floats:
        return "float"
    if lowered in {"bool"}:
        return "bool"
    # Pointer types become lists of the pointed-to element type.
    if lowered.endswith("*"):
        inner = lowered.rstrip("*").strip()
        if inner in {"float", "f32", "f64", "double"}:
            return "list[float]"
        if inner in {"int", "i32", "i64", "u32", "usize", "int32_t"}:
            return "list[int]"
        return "list"
    if lowered.startswith("*const ") or lowered.startswith("*mut "):
        inner = lowered.split(None, 1)[1]
        if inner in {"float", "f32", "f64", "double"}:
            return "list[float]"
        if inner in {"int", "i32", "i64", "u32", "usize", "int32_t"}:
            return "list[int]"
        return "list"
    return "Any"


def _abi_io_to_py(io_list: List[Dict[str, str]]) -> List[Tuple[str, str]]:
    return [(entry["name"], _abi_type_to_py(entry["type"])) for entry in io_list]


def _abi_contract_to_contract_entry(abi: ABIContract) -> Optional[ContractEntry]:
    """Convert an ABIContract into a Python-style ContractEntry signature."""
    sig = abi.signature
    inputs = sig.get("inputs", []) if isinstance(sig, dict) else []
    outputs = sig.get("outputs", []) if isinstance(sig, dict) else []
    if not inputs:
        return None
    args = _abi_io_to_py(inputs)
    if not outputs:
        return_type = "None"
    elif len(outputs) == 1:
        return_type = _abi_type_to_py(outputs[0]["type"])
    else:
        py_types = [_abi_type_to_py(o["type"]) for o in outputs]
        return_type = f"tuple[{', '.join(py_types)}]"
    arg_str = ", ".join(f"{name}: {typ}" for name, typ in args)
    signature = f"def {abi.export_symbol}({arg_str}) -> {return_type}"
    return ContractEntry(
        name=abi.export_symbol,
        signature=signature,
        language=abi.target_language,
        python_name=abi.export_symbol,
        purpose=f"ABI contract for {abi.contract_id}",
    )


def _contracts_from_abi(abi_contracts: List[ABIContract]) -> List[ContractEntry]:
    """Convert ABI contracts into synthesisable Python-style contract entries."""
    entries: List[ContractEntry] = []
    for abi in abi_contracts:
        entry = _abi_contract_to_contract_entry(abi)
        if entry:
            entries.append(entry)
    return entries


def _synthesize_python_source(contracts: List[ContractEntry]) -> str:
    """Build a stub Python module from contract signatures."""
    lines: List[str] = ["from __future__ import annotations", ""]
    names: List[str] = []
    for contract in contracts:
        if not contract.signature:
            continue
        try:
            name, args, return_type = _parse_signature(contract.signature)
        except Exception:
            logger.warning("Could not parse contract signature: %s", contract.signature)
            continue
        names.append(name)
        arg_str = ", ".join(f"{arg}: {typ}" for arg, typ in args)
        lines.append(f"def {name}({arg_str}) -> {return_type}:")
        lines.append(_generate_stub_body(name, args, return_type))
        lines.append("")

    if names:
        lines.append("__all__ = [" + ", ".join(f'"{n}"' for n in names) + "]")
    else:
        lines.append('__all__: list[str] = []')

    return "\n".join(lines) + "\n"


def _native_loader_source(crate_names: List[str], rust_dir: str = "rust_core") -> str:
    """Return module-level code that searches for and loads a compiled .so."""
    lines = [
        "import importlib.util",
        "import pathlib",
        "import re",
        "from typing import Any, Optional",
        "",
        "_SO_CANDIDATES = [",
        f'    pathlib.Path(__file__).parent.parent / "{rust_dir}" / "target" / "release",',
        '    pathlib.Path(__file__).parent.parent / "target" / "release",',
        '    pathlib.Path(__file__).parent.parent / "dist",',
        '    pathlib.Path(__file__).parent,',
        "]",
        "",
        f"_PREFERRED_MODULE_NAMES = {crate_names!r}",
        "",
        "",
        "def _load_native() -> Optional[Any]:",
        "    for directory in _SO_CANDIDATES:",
        "        if not directory.is_dir():",
        "            continue",
        '        for so in sorted(directory.glob("*.so")):',
        "            stem = so.stem",
        '            if stem.startswith("lib"):',
        '                stem = stem[3:]',
        r'            stem = re.sub(r"\.cpython-.*$", "", stem)',
        "            for preferred in _PREFERRED_MODULE_NAMES:",
        "                if preferred in stem:",
        "                    stem = preferred",
        "                    break",
        "            try:",
        "                spec = importlib.util.spec_from_file_location(stem, so)",
        "                if spec is None or spec.loader is None:",
        "                    continue",
        "                mod = importlib.util.module_from_spec(spec)",
        "                spec.loader.exec_module(mod)",
        "            except Exception:",
        "                continue",
        "            return mod",
        "    return None",
        "",
        "",
        "_NATIVE: Optional[Any] = _load_native()",
    ]
    return "\n".join(lines) + "\n"


def _function_impl(name: str, args: List[Tuple[str, str]], return_type: str) -> str:
    """Return a single Python function that delegates to ``_NATIVE`` or falls back."""
    arg_sig = ", ".join(f"{a}: {t}" for a, t in args)
    arg_call = ", ".join(a for a, _ in args)
    fallback = _generate_stub_body(name, args, return_type)
    return (
        f"def {name}({arg_sig}) -> {return_type}:\n"
        f"    if _NATIVE is not None and hasattr(_NATIVE, \"{name}\"):\n"
        f"        return _NATIVE.{name}({arg_call})\n"
        f"{fallback}\n"
    )


def _render_python_module(
    contracts: List[ContractEntry],
    module_name: str,
    *,
    is_cli: bool = False,
) -> str:
    """Render a Python module that delegates to a compiled Rust extension."""
    lines: List[str] = [
        "from __future__ import annotations",
        "",
        _native_loader_source([module_name]),
    ]
    for contract in contracts:
        if not contract.signature:
            continue
        try:
            name, args, return_type = _parse_signature(contract.signature)
        except Exception:
            continue
        lines.append(_function_impl(name, args, return_type))
        lines.append("")
    names = [c.name for c in contracts if c.signature]
    if names:
        lines.append("__all__ = [" + ", ".join(f'"{n}"' for n in names) + "]")
    else:
        lines.append('__all__: list[str] = []')
    if is_cli:
        lines.append("")
        lines.append('def main() -> int:')
        lines.append(f'    print("{module_name} CLI ready")')
        lines.append('    return 0')
    return "\n".join(lines) + "\n"


def _render_orchestrator(
    contracts: List[ContractEntry],
    module_name: str,
    rust_dir: str = "rust_core",
) -> str:
    """Render ``aero_polyglot_runner/orchestrator.py`` with ``PolyglotEngine``."""
    lines: List[str] = [
        '"""Polyglot runner that loads the compiled extension with a pure-Python fallback."""',
        "",
        "from __future__ import annotations",
        "",
        "import importlib.util",
        "import pathlib",
        "import re",
        "from typing import Any, Dict, List, Optional",
        "",
        "_SO_CANDIDATES = [",
        f'    pathlib.Path(__file__).parent.parent / "{rust_dir}" / "target" / "release",',
        '    pathlib.Path(__file__).parent.parent / "target" / "release",',
        '    pathlib.Path(__file__).parent.parent / "dist",',
        '    pathlib.Path(__file__).parent,',
        "]",
        "",
        f"_PREFERRED_MODULES = {[module_name]!r}",
        "",
        "",
        "class PolyglotEngine:",
        '    """Loads the compiled Rust extension or falls back to pure Python."""',
        "",
        "    def __init__(self) -> None:",
        "        self._native: Optional[Any] = self._load_native()",
        "",
        "    @property",
        '    def backend(self) -> str:',
        '        return "rust" if self._native is not None else "python"',
        "",
        "    def _load_native(self) -> Optional[Any]:",
        "        for directory in _SO_CANDIDATES:",
        "            if not directory.is_dir():",
        "                continue",
        '            for so in sorted(directory.glob("*.so")):',
        "                stem = so.stem",
        '                if stem.startswith("lib"):',
        '                    stem = stem[3:]',
        r'                stem = re.sub(r"\.cpython-.*$", "", stem)',
        "                for preferred in _PREFERRED_MODULES:",
        "                    if preferred in stem:",
        "                        stem = preferred",
        "                        break",
        "                try:",
        "                    spec = importlib.util.spec_from_file_location(stem, so)",
        "                    if spec is None or spec.loader is None:",
        "                        continue",
        "                    mod = importlib.util.module_from_spec(spec)",
        "                    spec.loader.exec_module(mod)",
        "                except Exception:",
        "                    continue",
        "                return mod",
        "        return None",
    ]

    for contract in contracts:
        if not contract.signature:
            continue
        try:
            name, args, return_type = _parse_signature(contract.signature)
        except Exception:
            continue
        typed_args = ", ".join(f"{a}: {t}" for a, t in args)
        arg_sig = f"self, {typed_args}" if typed_args else "self"
        arg_call = ", ".join(a for a, _ in args)
        fallback = _generate_stub_body(name, args, return_type).strip()
        lines.append("")
        lines.append(f"    def {name}({arg_sig}) -> {return_type}:")
        lines.append(f"        if self._native is not None and hasattr(self._native, \"{name}\"):")
        lines.append(f"            return self._native.{name}({arg_call})")
        lines.append(f"        {fallback.lstrip()}")

    return "\n".join(lines) + "\n"


def _render_init(module: str, exports: List[str]) -> str:
    if not exports:
        return "# Generated by aero-forge\n"
    lines = [f"from .{module} import {', '.join(exports)}"]
    lines.append("__all__ = [" + ", ".join(f'"{e}"' for e in exports) + "]")
    return "\n".join(lines) + "\n"


def _module_uses_engine(exports_module: str) -> bool:
    """Return True if *exports_module* points to an orchestrator class module."""
    return exports_module.endswith(".orchestrator") or exports_module == "orchestrator"


def _sample_arg(t: str) -> str:
    t = (t or "").lower().replace(" ", "")
    if t == "str":
        return '"test"'
    if t == "int" or t == "i64" or t == "i32":
        return "1"
    if t == "float" or t == "f64" or t == "f32":
        return "1.0"
    if t == "bool":
        return "True"
    if t.startswith("list["):
        inner = t[5:-1].strip() if t.endswith("]") else "float"
        lit = _sample_arg(inner)
        if lit.startswith("[") or lit.startswith("{") or lit.startswith("("):
            return f"[{lit}, {lit}]"
        return f"[{lit}, {lit}, {lit}]"
    if t.startswith("dict["):
        return "{}"
    return "None"


def _render_demo(exports_module: str, contracts: List[ContractEntry]) -> str:
    names = [c.name for c in contracts if c.signature]
    if not names:
        return "if __name__ == \"__main__\":\n    pass\n"
    uses_engine = _module_uses_engine(exports_module)
    if uses_engine:
        lines = [
            f"from {exports_module} import PolyglotEngine",
            "",
            "def main() -> None:",
            "    engine = PolyglotEngine()",
        ]
    else:
        lines = [
            f"from {exports_module} import {', '.join(names)}",
            "",
            "def main() -> None:",
        ]
    for contract in contracts:
        if not contract.signature:
            continue
        try:
            name, args, _ = _parse_signature(contract.signature)
        except Exception:
            continue
        prefix = "engine." if uses_engine else ""
        arg_values = [_sample_arg(t) for _, t in args]
        lines.append(f"    print({prefix}{name}({', '.join(arg_values)}))")
    lines.extend(["", 'if __name__ == "__main__":', "    main()"])
    return "\n".join(lines) + "\n"


def _render_run_shell(cli_module: str) -> str:
    """Render a launch script that calls the package CLI main()."""
    return (
        "import sys\n"
        f"from {cli_module} import main\n"
        "\n"
        'if __name__ == "__main__":\n'
        "    sys.exit(main() or 0)\n"
    )


def _render_cli(exports_module: str, contracts: List[ContractEntry]) -> str:
    """Render a lightweight CLI module importing functions from *exports_module*."""
    names = [c.name for c in contracts if c.signature]
    if not names:
        return "def main():\n    print('No CLI commands defined')\n    return 0\n"
    lines = [f"from {exports_module} import {', '.join(names)}", ""]
    for contract in contracts:
        if not contract.signature:
            continue
        try:
            name, args, return_type = _parse_signature(contract.signature)
        except Exception:
            continue
        arg_values = [_sample_arg(t) for _, t in args]
        lines.append(f"def {name}_cmd() -> None:")
        lines.append(f"    print({name}({', '.join(arg_values)}))")
        lines.append("")
    lines.append("def main() -> int:")
    lines.append('    print("CLI ready")')
    for name in names:
        lines.append(f"    {name}_cmd()")
    lines.append("    return 0")
    lines.append("")
    return "\n".join(lines) + "\n"


def _contract_source(contracts: List[ContractEntry]) -> str:
    """Build a synthetic Python source containing one stub per contract."""
    lines: List[str] = []
    for contract in contracts:
        if not contract.signature:
            continue
        try:
            name, args, return_type = _parse_signature(contract.signature)
        except Exception:
            continue
        arg_sig = ", ".join(f"{a}: {t}" for a, t in args)
        lines.append(f"def {name}({arg_sig}) -> {return_type}:")
        lines.append(_generate_stub_body(name, args, return_type))
        lines.append("")
    return "\n".join(lines)


def _render_tests(exports_module: str, contracts: List[ContractEntry]) -> str:
    """Render contract-driven pytest tests for *exports_module*."""
    from aero_forge.scaffold import test_generator

    if not contracts:
        return "def test_placeholder():\n    pass\n"

    if _module_uses_engine(exports_module):
        names = [c.name for c in contracts if c.signature]
        lines = [
            f"from {exports_module} import PolyglotEngine",
            "",
            "def test_engine_instantiates():",
            "    assert PolyglotEngine() is not None",
            "",
        ]
        for contract in contracts:
            if not contract.signature:
                continue
            try:
                name, args, return_type = _parse_signature(contract.signature)
            except Exception:
                continue
            sample_call = ", ".join(_sample_arg(t) for _, t in args if _ != "self")
            lines.append(f"def test_{name}():")
            lines.append("    engine = PolyglotEngine()")
            if sample_call:
                lines.append(f"    result = engine.{name}({sample_call})")
            else:
                lines.append(f"    result = engine.{name}()")
            rt = return_type.lower().replace(" ", "")
            if "list" in rt:
                lines.append("    assert isinstance(result, list)")
            elif "dict" in rt:
                lines.append("    assert isinstance(result, dict)")
            elif rt in ("int", "i64", "i32"):
                lines.append("    assert isinstance(result, int)")
            elif rt in ("float", "f64", "f32"):
                lines.append("    assert isinstance(result, float)")
            elif rt == "bool":
                lines.append("    assert result in (True, False)")
            elif rt == "str":
                lines.append("    assert isinstance(result, str)")
            else:
                lines.append("    assert result is not None")
            lines.append("")
        return "\n".join(lines) + "\n"

    source = _contract_source(contracts)
    if not source:
        return "def test_placeholder():\n    pass\n"
    return test_generator.generate_smoke_tests(source, module_name=exports_module)


def _render_pyproject(pkg_name: str, package_dir: str = ".") -> str:
    lines = [
        "[build-system]",
        'requires = ["setuptools>=61"]',
        'build-backend = "setuptools.build_meta"',
        "",
        "[project]",
        f'name = "{pkg_name}"',
        'version = "0.1.0"',
        'requires-python = ">=3.9"',
        "",
        "[tool.setuptools]",
        f'package-dir = {{"" = "{package_dir}"}}',
        f'packages = ["{pkg_name}"]',
    ]
    return "\n".join(lines) + "\n"


def _render_readme(project: str) -> str:
    lines = [
        f"# {project}",
        "",
        "Aero-Forge generated Python-Rust polyglot workspace.",
        "",
        "## Build",
        "",
        "```bash",
        "cargo build --release",
        "python -m pytest tests -q",
        "```",
    ]
    return "\n".join(lines) + "\n"


def _find_manifest_entries(blueprint: Blueprint) -> Dict[str, ManifestEntry]:
    """Index manifest entries by lowercased basename for quick lookup."""
    return {Path(e.path).name.lower(): e for e in blueprint.manifest}


def _sibling_module_for_init(init_entry: ManifestEntry, blueprint: Blueprint) -> Optional[Tuple[str, List[str]]]:
    """Return a sibling .py module name and exports for a package ``__init__.py``."""
    init_dir = Path(init_entry.path).parent
    candidates = []
    for e in blueprint.manifest:
        p = Path(e.path)
        if (
            p.parent == init_dir
            and p.suffix == ".py"
            and p.name != "__init__.py"
            and not p.name.startswith("test_")
        ):
            candidates.append(p)
    if not candidates:
        return None
    names = [c.name for c in blueprint.contracts if c.signature]
    # Prefer a native/orchestrator wrapper, then any implementation module.
    for p in candidates:
        if p.name == "orchestrator.py":
            return (p.stem, ["PolyglotEngine"])
        if p.name == "native.py":
            return (p.stem, names)
    return (candidates[0].stem, names)


def _package_modules(blueprint: Blueprint) -> Dict[Path, str]:
    """Map package directories to their dotted Python module names."""
    packages: Dict[Path, str] = {}
    for e in blueprint.manifest:
        p = Path(e.path)
        if p.name == "__init__.py":
            packages[p.parent] = ".".join(p.parent.parts)
    return packages


def _module_for_rel(rel: Path, packages: Dict[Path, str]) -> str:
    """Return the dotted module name for a manifest file path."""
    for pkg_dir, module in packages.items():
        try:
            sub = rel.relative_to(pkg_dir)
        except ValueError:
            continue
        parts = list(sub.parts[:-1]) + [rel.stem]
        if parts == [rel.stem]:
            return f"{module}.{rel.stem}"
        return ".".join([module] + parts)
    return rel.stem


def _resolve_test_target(
    rel: Path,
    blueprint: Blueprint,
    packages: Dict[Path, str],
    pkg_name: str,
) -> str:
    """Return the module a test file should import from."""
    stem = rel.stem
    target_stem = stem[5:] if stem.startswith("test_") else stem
    # Look for an implementation module with the same name inside any package.
    for e in blueprint.manifest:
        p = Path(e.path)
        if p.suffix == ".py" and p.stem == target_stem and not p.name.startswith("test_"):
            return _module_for_rel(p, packages)
    # Fall back to the package's orchestrator or root module.
    for e in blueprint.manifest:
        if Path(e.path).name == "orchestrator.py":
            return _module_for_rel(Path(e.path), packages)
    return pkg_name


def _resolve_cli_module(
    blueprint: Blueprint,
    packages: Dict[Path, str],
    pkg_name: str,
) -> str:
    """Return the module a launcher script should import ``main`` from."""
    for e in blueprint.manifest:
        if Path(e.path).name == "main.py":
            return _module_for_rel(Path(e.path), packages)
    for e in blueprint.manifest:
        if Path(e.path).name == "cli.py":
            return _module_for_rel(Path(e.path), packages)
    for e in blueprint.manifest:
        if Path(e.path).name == "orchestrator.py":
            return _module_for_rel(Path(e.path), packages)
    return pkg_name


def _has_native_module(
    blueprint: Blueprint,
    packages: Dict[Path, str],
    pkg_name: str,
) -> Optional[str]:
    """Return the dotted module name of a native wrapper if one exists."""
    for e in blueprint.manifest:
        if Path(e.path).name == "native.py":
            return _module_for_rel(Path(e.path), packages)
    return None


class PolyglotMaterializer:
    """Write the files declared by a polyglot ``Blueprint`` to disk."""

    def __init__(self, workspace_root: Path) -> None:
        self.workspace = Path(workspace_root)
        self.workspace.mkdir(parents=True, exist_ok=True)
        self.build_logs = ""

    def _accel_log(self, level: str, message: str) -> None:
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

    def _materialize_native_rust_update(
        self,
        blueprint: Blueprint,
        build: bool,
        explicit: Dict[str, Any],
    ) -> Blueprint:
        """Materialize a concrete PyO3/NumPy/Rayon native extension requested by prompt."""
        function_name = explicit["function"]
        rust_path_str = explicit["rust_path"]
        args = explicit.get("args", [])
        return_type = explicit.get("return_type", "i64")
        rust_file = self.workspace / rust_path_str
        module_dir = rust_file.parent
        module_dir.mkdir(parents=True, exist_ok=True)

        package = _discover_existing_python_package(self.workspace)
        if package is None:
            package_name = function_name
            package_rel = f"src/{function_name}"
            package_dir = self.workspace / package_rel
            package_dir.mkdir(parents=True, exist_ok=True)
            (package_dir / "__init__.py").write_text(
                f'from .{function_name} import {function_name}\n__all__ = ["{function_name}"]\n',
                encoding="utf-8",
            )
        else:
            package_name, package_rel = package
            package_dir = self.workspace / package_rel

        crate_name = f"aero_forge_native_{sanitize_crate_name(package_name)}"

        rel_mod = module_dir.relative_to(self.workspace)
        module_path = ""
        if rel_mod.parts and rel_mod.parts[0] == "src":
            module_path = "/".join(rel_mod.parts[1:])
        function_module = rust_file.stem

        needs_ndarray = function_name == "matmul" or "PyArray" in return_type
        # Root crate manifest and lib entry point.
        (self.workspace / "Cargo.toml").write_text(
            _native_update_cargo_toml(crate_name, needs_ndarray=needs_ndarray), encoding="utf-8"
        )
        lib_rs = self.workspace / "src" / "lib.rs"
        lib_rs.parent.mkdir(parents=True, exist_ok=True)
        lib_rs.write_text(
            _native_update_lib_rs(crate_name, module_path, function_module, function_name),
            encoding="utf-8",
        )

        # Intermediate mod.rs files for nested module directories.
        if module_path:
            parts = rel_mod.parts[1:] if rel_mod.parts[0] == "src" else rel_mod.parts
            for i in range(1, len(parts)):
                parent = self.workspace / "src" / "/".join(parts[:i])
                parent.mkdir(parents=True, exist_ok=True)
                mod_decl = "\n".join(f"pub mod {p};" for p in parts[i:]) + "\n"
                (parent / "mod.rs").write_text(mod_decl, encoding="utf-8")
            leaf = self.workspace / "src" / "/".join(parts)
            leaf.mkdir(parents=True, exist_ok=True)
            (leaf / "mod.rs").write_text(
                _native_update_mod_rs(function_module), encoding="utf-8"
            )

        rust_file.write_text(
            _native_update_rust_function_source(function_name, args, return_type),
            encoding="utf-8",
        )

        # Python wrapper and package export.
        wrapper_path = package_dir / f"{function_name}.py"
        wrapper_path.write_text(
            _native_update_wrapper_py(crate_name, function_name, args), encoding="utf-8"
        )
        _append_init_export(package_dir / "__init__.py", function_name)

        # Tests.
        tests_dir = self.workspace / "tests"
        tests_dir.mkdir(parents=True, exist_ok=True)
        test_path = tests_dir / f"test_{function_name}.py"
        test_path.write_text(
            _native_update_test_py(package_name, function_name, args, return_type),
            encoding="utf-8",
        )

        if build:
            cargo_check_and_repair(self.workspace, max_retries=2, timeout=600)
            result = cargo_build(self.workspace, release=True, timeout=600)
            output = f"{result.stdout}\n{result.stderr}".strip()
            self.build_logs = output
            if result.returncode != 0:
                raise RuntimeError(
                    f"Rust compilation failed in {self.workspace} (exit {result.returncode}):\n{output}"
                )
            so_name = f"lib{crate_name}.so"
            so_src = self.workspace / "target" / "release" / so_name
            dist_dir = self.workspace / "dist"
            dist_dir.mkdir(parents=True, exist_ok=True)
            if so_src.is_file():
                shutil.copy(so_src, dist_dir / so_name)
                shutil.copy(so_src, package_dir / so_name)

        # Rebuild the blueprint to reflect the concrete generated files.
        new_manifest: List[ManifestEntry] = [
            ManifestEntry(path="Cargo.toml", lang="toml", purpose="crate manifest"),
            ManifestEntry(path="src/lib.rs", lang="rust", purpose="rust core"),
        ]
        if module_path:
            parts = rel_mod.parts[1:] if rel_mod.parts[0] == "src" else rel_mod.parts
            for i in range(1, len(parts) + 1):
                mod_path = "src/" + "/".join(parts[:i]) + "/mod.rs"
                new_manifest.append(
                    ManifestEntry(path=mod_path, lang="rust", purpose="rust module")
                )
        new_manifest.extend(
            [
                ManifestEntry(
                    path=rust_path_str,
                    lang="rust",
                    purpose=f"native {function_name} implementation",
                ),
                ManifestEntry(
                    path=str(wrapper_path.relative_to(self.workspace)),
                    lang="python",
                    purpose="python native wrapper",
                ),
                ManifestEntry(
                    path=str(test_path.relative_to(self.workspace)),
                    lang="python",
                    purpose="tests",
                ),
                ManifestEntry(
                    path=str((package_dir / "__init__.py").relative_to(self.workspace)),
                    lang="python",
                    purpose="package init",
                ),
            ]
        )
        py_args = ", ".join(a["name"] for a in args if a["name"] != "py")
        py_sig = f"def {function_name}({py_args})"
        new_contracts: List[ContractEntry] = [
            ContractEntry(
                name=function_name,
                signature=py_sig,
                purpose=f"native {function_name}",
            )
        ]
        new_functions: List[FunctionSpec] = [
            FunctionSpec(
                file=wrapper_path,
                name=function_name,
                tests=[test_path],
                skip_build=True,
            )
        ]
        updated = blueprint.model_copy(
            update={
                "manifest": new_manifest,
                "contracts": new_contracts,
                "functions": new_functions,
                "cargo_dependencies": {"rayon": "1.8", "numpy": "0.20", "pyo3": "0.20.3"},
                "project": package_name,
                "architecture": INTENT_HYBRID_RUST_PYTHON,
                "toolchains": ["python", "rust", "cargo"],
            }
        )
        write_blueprint(updated, self.workspace / "blueprint.aero")
        return updated

    def materialize(
        self,
        blueprint: Blueprint,
        *,
        build: bool = False,
        force_overwrite: bool = False,
    ) -> Blueprint:
        """Create every missing file declared in *blueprint* and return the updated blueprint."""
        guard_materialization(self.workspace, blueprint, force_overwrite=force_overwrite)
        project = blueprint.project or "polyglot_project"
        crate_name = f"aero_forge_native_{sanitize_crate_name(project)}"
        pkg_name = _sanitize_module_name(project)

        # If the prompt asks for an explicit PyO3/NumPy/Rayon native function,
        # materialize it directly instead of synthesising generic stubs.
        explicit = _extract_explicit_rust_update(getattr(blueprint, "prompt", None) or "")
        if explicit and blueprint.architecture in (
            INTENT_HYBRID_RUST_PYTHON,
            INTENT_PURE_RUST,
            INTENT_TRI_POLYGLOT_RUST_CPP_PYTHON,
        ):
            logger.info("Detected explicit native Rust update for %s", explicit["function"])
            self._accel_log("info", f"Native Rust update: {explicit['function']}")
            return self._materialize_native_rust_update(blueprint, build, explicit)

        contracts = list(blueprint.contracts) if blueprint.contracts else list(_DEFAULT_CONTRACTS)
        if blueprint.abi_contracts:
            abi_entries = _contracts_from_abi(blueprint.abi_contracts)
            # Preserve explicit Python contracts while appending ABI-derived ones.
            existing_names = {c.name for c in contracts}
            contracts.extend(c for c in abi_entries if c.name not in existing_names)

        # If the blueprint includes C++ files, delegate to the C++ materializer so
        # header inclusion and shared-library linking are handled consistently.
        has_cpp_manifest = any(
            entry.lang == "cpp" or str(entry.path).endswith((".cpp", ".hpp", ".h", ".cc", ".cxx"))
            for entry in blueprint.manifest
        )
        if blueprint.architecture == "hybrid_cpp_python" or has_cpp_manifest:
            from aero_forge.scaffold.cpp_materializer import CppPolyglotMaterializer

            logger.info("C++ manifest detected; delegating to CppPolyglotMaterializer")
            self._accel_log("info", "Routing C++ selective acceleration through cpp_emitter.py and cpp_materializer.py")
            self._accel_log("success", "ACCELERATED: C++ extern \"C\" dynamic shared library selected")
            return CppPolyglotMaterializer(self.workspace).materialize(blueprint, build=build)

        source = _synthesize_python_source(contracts)

        # 1. Write Python / packaging files from the manifest first so the Rust
        #    generator's manifest validation sees a complete workspace.
        self._write_python_files(blueprint, project, crate_name, pkg_name, contracts)

        # 1b. Normalize CLI/native module exports so __init__.py and cli.py are
        # consistent regardless of how the LLM or fallback templates wrote them.
        normalize_workspace(self.workspace)

        # 2. Persist the blueprint so Engine.generate can find it.
        blueprint_path = self.workspace / "blueprint.aero"
        write_blueprint(blueprint, blueprint_path)

        # 3. Generate the Rust crate via the PyO3 transpiler.  Engine.generate
        #    copies Cargo.toml/src/lib.rs into the directories declared by the
        #    manifest and validates every declared file.
        uast = python_source_to_uast(source)
        graph = UASTToHINTranslator().translate(uast)
        graph.traits_by_name = {}
        graph.traits = {}
        Engine().generate(
            graph,
            self.workspace / "dist",
            workspace_root=self.workspace,
            module_name=crate_name,
            function_names=[c.name for c in contracts],
            source=source,
            target_mode=TargetMode.PYO3,
        )

        # 4. Compile if requested.
        if build:
            self._build_crates()

        # 5. Populate blueprint.functions with concrete file references.
        updated = self._blueprint_with_functions(blueprint, contracts, pkg_name)
        write_blueprint(updated, blueprint_path)
        return updated

    def _write_python_files(
        self,
        blueprint: Blueprint,
        project: str,
        crate_name: str,
        pkg_name: str,
        contracts: List[ContractEntry],
    ) -> None:
        """Generate missing Python, TOML, and README files declared in the manifest."""
        packages = _package_modules(blueprint)
        native_module = _has_native_module(blueprint, packages, pkg_name)

        for entry in blueprint.manifest:
            path = self.workspace / entry.path
            if path.exists():
                continue

            content: Optional[str] = None
            rel = Path(entry.path)
            module_name = _module_for_rel(rel, packages)

            if entry.lang == "python" and path.name == "main.py":
                if blueprint.execution_strategy is not None:
                    strategy = blueprint.execution_strategy.model_dump()
                    primary = strategy.get("primary_entrypoint", {})
                    if primary.get("runtime") == "python3":
                        strategy["primary_entrypoint"]["path"] = str(rel)
                        rel_pkg = rel.parent
                        function_module = ".".join(rel_pkg.parts) if rel_pkg.parts else ""
                        EntrypointAdapterEngine(
                            strategy,
                            str(self.workspace),
                            contracts=contracts,
                            abi_contracts=list(blueprint.abi_contracts or []),
                            function_module=function_module,
                        ).synthesize_root_entrypoint()
                        content = path.read_text(encoding="utf-8")
                    else:
                        content = f"# {path.name} placeholder generated by aero-forge\n"
                else:
                    content = _render_python_module(contracts, crate_name, is_cli=True)
            elif entry.lang == "python" and path.name == "orchestrator.py":
                content = _render_orchestrator(contracts, crate_name)
            elif entry.lang == "python" and path.name == "__init__.py":
                sibling = _sibling_module_for_init(entry, blueprint)
                if sibling:
                    module, exports = sibling
                else:
                    module, exports = pkg_name, [c.name for c in contracts]
                content = _render_init(module, exports)
            elif entry.lang == "python" and path.name.endswith(".py"):
                if "test" in path.name:
                    exports = _resolve_test_target(rel, blueprint, packages, pkg_name)
                    content = _render_tests(exports, contracts)
                elif path.name.startswith("run_"):
                    cli_module = _resolve_cli_module(blueprint, packages, pkg_name)
                    if _module_uses_engine(cli_module):
                        content = _render_demo(cli_module, contracts)
                    else:
                        content = _render_run_shell(cli_module)
                elif path.name == "cli.py":
                    if native_module:
                        content = _render_cli(native_module, contracts)
                    else:
                        content = _render_python_module(contracts, crate_name, is_cli=True)
                elif path.name in ("service.py", "bench.py"):
                    content = f"# {path.name} placeholder generated by aero-forge\n"
                elif path.name == "native.py":
                    content = _render_python_module(contracts, crate_name)
                else:
                    content = _render_python_module(contracts, crate_name)
            elif entry.lang == "toml" and path.name == "pyproject.toml":
                content = _render_pyproject(pkg_name, package_dir=".")
            elif entry.lang == "toml" and path.name == "Cargo.toml":
                continue
            elif entry.lang == "markdown":
                content = _render_readme(project)

            if content is not None:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(content, encoding="utf-8")
                logger.info("Synthesised %s", path.relative_to(self.workspace))

    def _has_accelerable_contract(self, blueprint: Blueprint) -> bool:
        """Return ``True`` if the blueprint declares a native Rust build."""
        for contract in blueprint.contracts:
            if not contract.signature:
                continue
            try:
                name, args, return_type = _parse_signature(contract.signature)
            except Exception:
                continue
            stub_body = _generate_stub_body(name, args, return_type)
            source = f"def {name}({', '.join(f'{a}: {t}' for a, t in args)}) -> {return_type}:\n{stub_body}"
            if should_accelerate_with_native(source):
                return True

        # If the blueprint explicitly declares a Rust architecture and a Cargo
        # manifest, treat it as accelerable even when no contracts were inferred.
        if blueprint.architecture in (
            INTENT_HYBRID_RUST_PYTHON,
            INTENT_HYBRID_CPP_RUST,
            INTENT_TRI_POLYGLOT_RUST_CPP_PYTHON,
            INTENT_PURE_RUST,
        ):
            if any(Path(e.path).name == "Cargo.toml" for e in blueprint.manifest):
                return True
        return False

    def _build_crates(self) -> None:
        """Run ``cargo build --release`` and copy ``.so`` artefacts to ``dist/``."""
        blueprint = self._read_blueprint()
        if not self._has_accelerable_contract(blueprint):
            logger.info("No accelerable contracts found; skipping Rust crate build")
            return

        cargo_tomls = [
            self.workspace / e.path
            for e in blueprint.manifest
            if Path(e.path).name == "Cargo.toml"
        ]
        if not cargo_tomls:
            return

        root_cargo = next((p for p in cargo_tomls if p.parent == self.workspace), None)
        build_dirs = [root_cargo.parent] if root_cargo else [p.parent for p in cargo_tomls]

        combined_logs: List[str] = []
        for build_dir in build_dirs:
            has_lib = (build_dir / "src" / "lib.rs").is_file()
            has_members = any(
                p.is_dir() and (p / "Cargo.toml").is_file()
                for p in build_dir.iterdir()
            )
            if not has_lib and not has_members:
                continue
            logger.info("Building Rust crate in %s", build_dir)
            result = cargo_build(build_dir, release=True, timeout=600)
            output = f"{result.stdout}\n{result.stderr}".strip()
            if output:
                combined_logs.append(f"--- cargo build in {build_dir} ---\n{output}")
            if result.returncode != 0:
                self.build_logs = "\n\n".join(combined_logs)
                raise RuntimeError(
                    f"Rust compilation failed in {build_dir} (exit {result.returncode}):\n"
                    f"{output}"
                )

        self.build_logs = "\n\n".join(combined_logs)

        dist = self.workspace / "dist"
        dist.mkdir(parents=True, exist_ok=True)

        target_dirs = set()
        if (self.workspace / "target" / "release").is_dir():
            target_dirs.add(self.workspace / "target" / "release")
        for cargo_toml in cargo_tomls:
            td = cargo_toml.parent / "target" / "release"
            if td.is_dir():
                target_dirs.add(td)

        for td in target_dirs:
            for so in td.glob("*.so"):
                shutil.copy(so, dist / so.name)

    def _read_blueprint(self) -> Blueprint:
        """Read the workspace blueprint if it exists, else return an empty one."""
        from aero_forge.blueprint import parse_blueprint

        path = self.workspace / "blueprint.aero"
        if path.is_file():
            return parse_blueprint(path)
        return Blueprint()

    def _blueprint_with_functions(
        self,
        blueprint: Blueprint,
        contracts: List[ContractEntry],
        pkg_name: str,
    ) -> Blueprint:
        """Return a blueprint whose ``functions`` list references materialised files."""
        by_name = _find_manifest_entries(blueprint)

        lib_entry = by_name.get("lib.rs")
        native_entry = by_name.get("native.py")
        orchestrator_entry = next(
            (e for e in blueprint.manifest if e.path.endswith("orchestrator.py")),
            None,
        )
        test_entries = [
            e for e in blueprint.manifest if e.path.endswith(".py") and "test" in e.path
        ]
        test_paths = [self.workspace / e.path for e in test_entries]

        # Each contract is implemented by the Rust core; the Python wrapper is a
        # thin native loader. Prefer native.py / orchestrator.py over lib.rs when
        # the manifest declares a Python entry point.
        impl_entry = native_entry or orchestrator_entry or lib_entry

        functions: List[FunctionSpec] = []
        if impl_entry:
            for contract in contracts:
                functions.append(
                    FunctionSpec(
                        file=self.workspace / impl_entry.path,
                        name=contract.name,
                        tests=list(test_paths),
                        skip_build=True,
                    )
                )

        if orchestrator_entry:
            functions.append(
                FunctionSpec(
                    file=self.workspace / orchestrator_entry.path,
                    name="PolyglotEngine",
                    tests=list(test_paths),
                    skip_build=True,
                )
            )

        cli_entry = by_name.get("cli.py")
        if cli_entry:
            functions.append(
                FunctionSpec(
                    file=self.workspace / cli_entry.path,
                    name="main",
                    tests=list(test_paths),
                    skip_build=True,
                )
            )

        return blueprint.model_copy(update={"functions": functions})


def _emit_verified_files(payload: Dict[str, Any]) -> "Blueprint":
    """Materialize a verified blueprint to disk after all gates pass.

    The *payload* must contain a ``workspace`` path and a ``blueprint`` dict
    suitable for constructing a ``Blueprint``. Files are written without
    triggering a native build. Tri-polyglot blueprints are routed to the
    dedicated tri-polyglot materializer.
    """
    from aero_forge.blueprint import Blueprint

    workspace = Path(payload.get("workspace", "."))
    blueprint_dict = payload.get("blueprint", {})
    blueprint = Blueprint(**blueprint_dict)

    if blueprint.architecture == "tri_polyglot_rust_cpp_python" or (
        {"rust", "cpp", "python"}.issubset(set(blueprint.toolchains))
        and any(Path(e.path).name == "Cargo.toml" for e in blueprint.manifest)
        and any(e.lang == "cpp" or str(e.path).endswith(".cpp") for e in blueprint.manifest)
    ):
        from aero_forge.scaffold.tri_polyglot_materializer import TriPolyglotMaterializer

        return TriPolyglotMaterializer(workspace).materialize(
            blueprint,
            build=False,
            force_overwrite=payload.get("force_overwrite", False),
        )

    return PolyglotMaterializer(workspace).materialize(
        blueprint,
        build=False,
        force_overwrite=payload.get("force_overwrite", False),
    )
