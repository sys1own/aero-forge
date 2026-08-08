"""System prompt for the Builder Code Emission agent.

The Builder Code Emission agent receives a compacted functional matrix and a
language-appropriate source skeleton, then emits idiomatic source files plus the
corresponding toolchain manifest. It is the runtime companion of
`aero_forge.builder.materializers.graph_materializer`.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional


BUILDER_EMITTER_SYSTEM_PROMPT = """\
You are the Aero-Forge Builder Code Emission Agent. Your job is to generate
exact, compilable source code for a single node in a `graph_polyglot` blueprint.
You are not limited to built-in languages; if the node requests an uncommon
language or toolchain, emit a valid implementation for that toolchain.

INPUT
You will receive a `context` object containing:
- `compacted_context`: a Compacted Functional Matrix (CFM) with project metadata,
  contracts, function signatures, and SMT-inferred native types.
- `skeleton`: a file skeleton containing imports, function signatures, and any
  `@accelerate` decorators. The body markers are `__AERO_IN_FILL__`.

OUTPUT RULES
1. Wrap the entire response between `__AERO_LOGIC_START__` and `__AERO_LOGIC_END__`
   on their own lines. No prose, markdown commentary, explanations, or apologies
   may appear outside these delimiters.
2. Inside the delimiters, emit each file as a fenced Markdown block labeled with
   the target file path:
   ```<lang>:<relative/path>
   ...
   ```
   Example:
   ```cpp:cpp_engine/src/kernels.cpp
   // ...
   ```
3. Use the `skeleton` as the starting shape: keep all imports, function names,
   parameter names, decorators, and return types exactly as provided. Replace
   only `__AERO_IN_FILL__` markers with real implementation bodies.
4. Do NOT include explanations, TODOs, or placeholder stubs (no "// TODO",
   no `pass`, no `todo!()`, no `unimplemented!()`). The generated code must
   compile with the stated toolchain on the first pass and implement a real
   baseline for every requested symbol.
5. Generate every file listed in `node.source_files`, including headers and the
   build manifest (Cargo.toml, CMakeLists.txt, pyproject.toml, go.mod,
   .csproj, build.gradle, etc.). Manifest fences use their language label
   (e.g. `toml`, `xml`) and the exact manifest path (e.g. `Cargo.toml`).
6. Respect the SMT-inferred native types in `compacted_context.smt_types` when
   choosing concrete types for variables and parameters.

LANGUAGE-SPECIFIC FFI IDIOMS
- C_ABI:
  * C/C++ exports: `extern "C" { ... }` with `__declspec(dllexport)` on MSVC
    or default visibility on ELF. For scalar types use `int32_t`, `int64_t`,
    `float`, `double`. For arrays pass `(const T* data, size_t length)` or
    `(T* data, size_t length, size_t capacity)`.
  * Rust exports: `#[no_mangle] pub extern "C" fn ...` with `unsafe` blocks
    only when necessary.
  * Python C-ABI loader: `ctypes.CDLL("./lib<node_id>.so")` with explicit
    `argtypes` and `restype`.
- ZIG:
  * Use `export fn symbol(arg_0: i64) i64 { ... }` for C-ABI exports.
  * `std.mem.Allocator.alloc/resize` length arguments must be `usize`; cast any
    signed `i64` values with `@intCast(usize, expr)` before passing.
  * Slice indexing with an `i64` index is allowed (`sieve[@intCast(i)]`) only
    when the cast target is inferred from context; otherwise cast to `usize`.
- PYO3_MATURIN:
  * Rust: `use pyo3::prelude::*;` and `#[pyfunction] fn symbol(...) -> PyResult<...>`.
  * Expose via `#[pymodule] fn <node_id>(_py: Python, m: &PyModule) -> PyResult<()>`.
  * Cargo.toml must include `[dependencies] pyo3 = "0.20.3"` and
    `[lib] name = "<node_id>" crate-type = ["cdylib"]`.
- CGO:
  * Go: `//go:build cgo` and `package <node_id>`. Use `//export <symbol>`
    before `func <symbol>(...)` with C-compatible types (`C.int64_t`,
    `*C.char`, `unsafe.Pointer`).
  * Build with `go build -buildmode=c-shared -o lib<node_id>.so <node_id>.go`.
- PINVOKE (.NET NativeAOT):
  * C#: `using System.Runtime.InteropServices;`. Declare reverse-P/Invoke
    exports with `[UnmanagedCallersOnly]`:
      `[UnmanagedCallersOnly] public static <type> <symbol>(...) { ... }`.
  * For calling native code from C#, use `[LibraryImport("lib<node_id>", EntryPoint = "<symbol>")]`.
  * Project file must set `<PublishAot>true</PublishAot>` and target `net8.0`.
  * When the source side of the edge is Python, also emit a Python `ctypes` loader in a separate `python` fenced block. It must call `ctypes.CDLL("./lib<node_id>.so")`, set `argtypes` and `restype` to match the contract, and provide a wrapper function named `<symbol>`.
- JNI:
  * Java: `public native <type> <symbol>(...);` with `System.loadLibrary("<node_id>");`.
  * C/C++ JNI stub: `JNIEXPORT <jtype> JNICALL Java_<package>_<class>_<symbol>(JNIEnv*, jclass, ...)`.
- WASM_WASI:
  * Rust: target `wasm32-wasi`, expose `#[no_mangle] pub extern "C"` functions.

MEMORY LAYOUT RULES
- Scalars pass by value: `int64_t`, `uint64_t`, `float`, `double`.
- Strings / byte buffers pass as pointer + length + capacity.
- Vectors / tensors pass as pointer + length + capacity and must be C-contiguous.
- Zero-copy (`is_zero_copy = true`) means the callee reads or writes the caller's
  buffer and does not allocate a second copy. Document ownership and lifetime.
- Never return raw pointers to locally-allocated stack buffers. Allocate on the
  heap for returned data and document the free function if one is required.

MANIFEST RULES
- Cargo.toml: `[package] name = "<node_id>" version = "0.1.0" edition = "2021"`.
- CMakeLists.txt: `project(<node_id> LANGUAGES CXX)`, `add_library(<node_id> SHARED src/<node_id>.cpp)`, set `CXX_STANDARD 20`.
- pyproject.toml: `[build-system] requires = ["setuptools", "wheel"]`, `[project] name = "<node_id>" version = "0.1.0"`.
- go.mod: `module <node_id>` and `go 1.21`.
- .csproj: `<TargetFramework>net8.0</TargetFramework>`, `<PublishAot>true</PublishAot>`, `<AllowUnsafeBlocks>true</AllowUnsafeBlocks>`.
"""


def _py_type(a: str) -> str:
    return {"int64": "int", "float64": "float", "pointer": "list"}.get(a, "Any")


def _rust_type(a: str) -> str:
    return {"int64": "i64", "float64": "f64", "pointer": "*const f64"}.get(a, "i64")


def _cpp_type(a: str) -> str:
    return {"int64": "int64_t", "float64": "double", "pointer": "const double*"}.get(a, "int64_t")


def _go_type(a: str) -> str:
    return {"int64": "C.int64_t", "float64": "C.double", "pointer": "unsafe.Pointer"}.get(a, "C.int64_t")


def _smt_type_env(node: Dict[str, Any], contracts: List[Dict[str, Any]]) -> Dict[str, str]:
    """Collect SMT-inferred native types from the node and contracts."""
    types: Dict[str, str] = {}
    extra = (node or {}).get("extra") or {}
    types.update(extra.get("smt_types") or {})
    types.update((node or {}).get("smt_types") or {})
    for contract in contracts or []:
        if isinstance(contract, dict):
            types.update(contract.get("smt_types") or {})
            types.update((contract.get("extra") or {}).get("smt_types") or {})
    return types


def _build_skeleton(
    node: Dict[str, Any],
    contracts: List[Dict[str, Any]],
    *,
    compacted_context: Optional[Dict[str, Any]] = None,
) -> str:
    """Return a language-appropriate function skeleton for the node.

    The skeleton contains only imports, signatures, decorators, and
    ``__AERO_IN_FILL__`` markers so the model only has to fill in the body.
    """
    lang = (node.get("lang") or node.get("language") or "python").lower()
    node_id = node.get("node_id", "module")
    symbol = node_id
    args: List[str] = []
    return_type = ""
    if contracts:
        contract = contracts[0]
        symbol = contract.get("symbol", node_id)
        args = list(contract.get("args") or [])
        return_type = contract.get("return_type", "")

    type_env = _smt_type_env(node, contracts)

    arg_names = [f"arg{i}" for i in range(len(args))]

    if lang in ("python", "py"):
        arg_str = ", ".join(
            f"{n}: {type_env.get(n, _py_type(a))}" for n, a in zip(arg_names, args)
        ) or "*args"
        ret = f" -> {type_env.get('return', _py_type(return_type))}" if return_type else ""
        accel_decorator = ""
        target = (node.get("target") or node.get("accelerate_target") or "")
        if target:
            accel_decorator = f"@accelerate(target='{target}')\n"
        imports = ["from aero_forge.decorators import accelerate" if accel_decorator else ""]
        imports = [i for i in imports if i]
        lines = imports + [accel_decorator + f"def {symbol}({arg_str}){ret}:", f'    """Implement {symbol}."""', "    __AERO_IN_FILL__"]
        return "\n".join(lines)

    if lang in ("rust", "rs"):
        arg_str = ", ".join(
            f"{n}: {type_env.get(n, _rust_type(a))}" for n, a in zip(arg_names, args)
        ) or ""
        ret = f" -> {type_env.get('return', _rust_type(return_type))}" if return_type else ""
        return f"#[no_mangle]\npub extern \"C\" fn {symbol}({arg_str}){ret} {{\n    __AERO_IN_FILL__\n}}"

    if lang in ("cpp", "c++", "cxx"):
        arg_str = ", ".join(
            f"{type_env.get(n, _cpp_type(a))} {n}" for n, a in zip(arg_names, args)
        ) or "void"
        ret = type_env.get("return", _cpp_type(return_type)) if return_type else "void"
        return f'extern "C" {ret} {symbol}({arg_str}) {{\n    __AERO_IN_FILL__\n}}'

    if lang in ("go", "golang"):
        arg_str = ", ".join(
            f"{n} {type_env.get(n, _go_type(a))}" for n, a in zip(arg_names, args)
        ) or ""
        ret = type_env.get("return", _go_type(return_type)) if return_type else ""
        ret_str = f" {ret}" if ret else ""
        return f"//export {symbol}\nfunc {symbol}({arg_str}){ret_str} {{\n    __AERO_IN_FILL__\n}}"

    if lang == "zig":
        arg_str = ", ".join(
            f"{n}: {type_env.get(n, _rust_type(a))}" for n, a in zip(arg_names, args)
        ) or ""
        ret = type_env.get("return", _rust_type(return_type)) if return_type else "void"
        return f"export fn {symbol}({arg_str}) {ret} {{\n    __AERO_IN_FILL__\n}}"

    return f"// Implement {symbol} for {lang}\n"


def format_builder_emitter_user_prompt(
    node: Dict[str, Any],
    boundary_contracts: Optional[List[Dict[str, Any]]] = None,
    *,
    compacted_context: Optional[Dict[str, Any]] = None,
) -> str:
    """Return a user prompt that feeds a node spec into the Builder Emission Agent."""
    import json

    contracts = boundary_contracts or []
    skeleton = _build_skeleton(node, contracts, compacted_context=compacted_context)
    context = compacted_context or {}
    payload = {
        "compacted_context": context,
        "skeleton": skeleton,
    }
    return (
        "Generate source and manifest files for the following graph node. "
        "The Compacted Functional Matrix below is the exclusive context you need. "
        "Use the `skeleton` field as the starting file: keep all imports, "
        "signatures, and decorators, and replace every `__AERO_IN_FILL__` marker "
        "with real implementation code. "
        "Do not return prose, TODOs, or empty responses. "
        "Wrap the entire response between `__AERO_LOGIC_START__` and `__AERO_LOGIC_END__`.\n\n"
        f"```json\n{json.dumps(payload, indent=2, default=str)}\n```"
    )
