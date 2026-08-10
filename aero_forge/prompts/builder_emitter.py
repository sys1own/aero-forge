"""System prompt for the Builder Code Emission agent.

The Builder Code Emission agent receives a compacted functional matrix and a
language-appropriate source skeleton, then emits idiomatic source files plus the
corresponding toolchain manifest. It is the runtime companion of
`aero_forge.builder.materializers.graph_materializer`.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from aero_forge.builder.smt_engine import SkeletonTypeInjector


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
   the target file path. For Python targets the preferred format is a UAST JSON
   sketch; the engine will lower it to source and resolve attributes like
   ``conj`` -> ``conjugate``. For all other languages emit the final source.
   ```uast:main.py
   {"type": "Module", "body": [{"type": "FunctionDef", "name": "fft", ...}]}
   ```
   Other language example:
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
5. Full Implementation Map: the response must define **every** symbol listed in
   `context.required_symbols`.  Do not omit, truncate, or skip any requested
   function.  If `context.missing_symbols` is non-empty, those functions were
   absent from a previous attempt and must be included now.
5a. DATA CONSTANT RULE: Any symbol flagged with `data_payload: true` (e.g.
   scoring matrices, lookup tables, or constants such as `blosum62`) must be
   emitted as a fully populated top-level assignment or return value. Replace
   the placeholder `__AERO_IN_FILL__` with a complete dict/list/set literal; do
   not emit `pass`, `None`, or an empty container as the final payload.
6. Generate every file listed in `node.source_files`, including headers and the
   build manifest (Cargo.toml, CMakeLists.txt, pyproject.toml, go.mod,
   .csproj, build.gradle, etc.). Manifest fences use their language label
   (e.g. `toml`, `xml`) and the exact manifest path (e.g. `Cargo.toml`).
7. Respect the SMT-inferred native types in `compacted_context.smt_types` when
   choosing concrete types for variables and parameters.
8. UAST sketches must use Python `ast` node names (`Module`, `FunctionDef`,
   `Call`, `Attribute`, `BinOp`, `Compare`, etc.). Attribute names should follow
   the intent (e.g. `conj` on a complex value); the engine will rewrite them to
   the correct Python spelling (`conjugate`) via SMT verification.
9. TEST DENSITY CONSTRAINT: if the node is a `tests/` artifact, emit at least
   five (5) distinct `def test_<symbol>_<case>():` functions for every contracted
   symbol. Cover success paths, edge cases (empty inputs, large buffers), and
   error handling. Do not group multiple scenarios under a single test name.

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


def _looks_like_bytes(symbol: str) -> bool:
    """Heuristic: symbols/ids with these tokens usually denote byte buffers."""
    return any(tok in symbol.lower() for tok in (
        "key", "buf", "byte", "aes", "sbox", "iv", "nonce", "schedule", "hash",
        "digest", "cipher", "block", "round", "seed", "entropy", "secret",
    ))


def _py_type(a: str, symbol: str = "") -> str:
    return {"int64": "int", "float64": "float", "pointer": "list"}.get(a, "Any")


def _rust_type(a: str, symbol: str = "") -> str:
    if a == "pointer":
        # Mutable pointers let the generated body write output buffers without
        # fighting Rust's ownership rules in ``from_raw_parts_mut``.
        return "*mut u8" if _looks_like_bytes(symbol) else "*mut f64"
    return {"int64": "i64", "float64": "f64"}.get(a, "i64")


def _rust_pyo3_type(a: str, symbol: str = "") -> str:
    """PyO3-friendly Rust types for ``#[pyfunction]`` signatures."""
    if a == "pointer":
        return "Vec<u8>" if _looks_like_bytes(symbol) else "Vec<f64>"
    return {"int64": "i64", "float64": "f64"}.get(a, "i64")


def _cpp_type(a: str, symbol: str = "") -> str:
    if a == "pointer":
        return "uint8_t*" if _looks_like_bytes(symbol) else "double*"
    return {"int64": "int64_t", "float64": "double"}.get(a, "int64_t")


def _go_type(a: str, symbol: str = "") -> str:
    return {"int64": "C.int64_t", "float64": "C.double", "pointer": "unsafe.Pointer"}.get(a, "C.int64_t")


def _zig_type(a: str, symbol: str = "") -> str:
    """Zig-compatible type for C-ABI pointers and primitives."""
    if a == "pointer":
        return "[*c]u8" if _looks_like_bytes(symbol) else "[*c]f64"
    return {"int64": "i64", "float64": "f64"}.get(a, "i64")


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


def _symbol_specs(
    node: Dict[str, Any], contracts: List[Dict[str, Any]]
) -> List[Tuple[str, List[str], str]]:
    """Return (symbol, args, return_type) specs for every contract/export.

    The skeleton must expose every function the blueprint asks for so the LLM
    fills in a complete implementation map rather than a single symbol.
    """
    specs: List[Tuple[str, List[str], str]] = []
    for c in contracts or []:
        sym = c.get("symbol") or node.get("node_id", "module")
        specs.append((sym, list(c.get("args") or []), c.get("return_type", "")))
    if not specs:
        for sym in node.get("exports") or []:
            specs.append((sym, [], ""))
    if not specs:
        specs.append((node.get("node_id", "module"), [], ""))
    return specs


def _lookup_payload_kind(
    symbol: str, compacted_context: Optional[Dict[str, Any]]
) -> Optional[str]:
    if not compacted_context:
        return None
    for fn in compacted_context.get("functions", []):
        if (fn.get("name") == symbol or fn.get("symbol") == symbol) and fn.get("data_payload"):
            return fn.get("payload_kind") or "dict"
    for entry in compacted_context.get("data_constants", []):
        if entry.get("name") == symbol or entry.get("symbol") == symbol:
            return entry.get("payload_kind") or "dict"
    impl_map = compacted_context.get("full_implementation_map") or {}
    for entry in impl_map.get("symbols", []):
        if (entry.get("name") == symbol or entry.get("symbol") == symbol) and entry.get("data_payload"):
            return entry.get("payload_kind") or "dict"
    return None


def _build_skeleton(
    node: Dict[str, Any],
    contracts: List[Dict[str, Any]],
    *,
    compacted_context: Optional[Dict[str, Any]] = None,
) -> str:
    """Return a language-appropriate function skeleton for the node.

    The skeleton contains only imports, signatures, decorators, and
    ``__AERO_IN_FILL__`` markers so the model only has to fill in the body.
    Every contracted/exported symbol gets its own signature block.
    Data constants are emitted as top-level assignments rather than functions.
    """
    lang = (node.get("lang") or node.get("language") or "python").lower()
    node_id = node.get("node_id", "module")
    specs = _symbol_specs(node, contracts)
    type_env = _smt_type_env(node, contracts)

    is_pyo3 = any(
        (c.get("boundary_type") or "").lower().replace("-", "_") == "pyo3_maturin"
        for c in contracts
    )

    if is_pyo3 and lang in ("rust", "rs"):
        pyo3_type_map = {
            "&mut [f64]": "Vec<f64>",
            "&[f64]": "Vec<f64>",
            "*mut f64": "Vec<f64>",
            "*const f64": "Vec<f64>",
            "&mut [u8]": "Vec<u8>",
            "&[u8]": "Vec<u8>",
            "*mut u8": "Vec<u8>",
            "*const u8": "Vec<u8>",
            "double*": "Vec<f64>",
            "float*": "Vec<f64>",
            "uint8_t*": "Vec<u8>",
        }
        type_env = {k: pyo3_type_map.get(v.strip(), v) for k, v in type_env.items()}

    if lang in ("python", "py"):
        target = (node.get("target") or node.get("accelerate_target") or "")
        accel_decorator = f"@accelerate(target='{target}')\n" if target else ""
        imports = ["from typing import Any", "from aero_forge.decorators import accelerate" if accel_decorator else ""]
        imports = [i for i in imports if i]
        lines = imports + [""]
        for symbol, args, return_type in specs:
            payload_kind = _lookup_payload_kind(symbol, compacted_context)
            if payload_kind and not accel_decorator:
                py_type = {"dict": "dict", "list": "list", "set": "set"}.get(payload_kind, "Any")
                lines.append(f"{symbol}: {py_type} = __AERO_IN_FILL__")
                lines.append("")
                continue
            arg_names = [f"arg{i}" for i in range(len(args))]
            # First, infer per-symbol SMT types from an annotated stub so each
            # function's signature can be typed independently.
            stub_arg_str = ", ".join(
                f"{n}: {_py_type(a, symbol)}" for n, a in zip(arg_names, args)
            ) or "*args"
            stub_ret = f" -> {_py_type(return_type, symbol)}" if return_type else ""
            stub = f"def {symbol}({stub_arg_str}){stub_ret}:\n    pass"
            sym_env: Dict[str, str] = {}
            try:
                sym_env = SkeletonTypeInjector.infer_type_env_for_symbol(stub, symbol)
            except Exception:
                pass
            sym_type_env = {**type_env, **sym_env}
            arg_str = ", ".join(
                f"{n}: {sym_type_env.get(n, _py_type(a, symbol))}" for n, a in zip(arg_names, args)
            ) or "*args"
            ret = f" -> {sym_type_env.get('return', _py_type(return_type, symbol))}" if return_type else ""
            if accel_decorator:
                lines.append(accel_decorator.rstrip())
            lines.append(f"def {symbol}({arg_str}){ret}:")
            lines.append(f'    """Implement {symbol}."""')
            lines.append("    __AERO_IN_FILL__")
            lines.append("")
        return "\n".join(lines).rstrip()

    if lang in ("rust", "rs"):
        parts: List[str] = []
        for symbol, args, return_type in specs:
            arg_names = [f"arg{i}" for i in range(len(args))]
            rust_arg = lambda a, n, sym=symbol: type_env.get(n, _rust_pyo3_type(a, sym) if is_pyo3 else _rust_type(a, sym))
            rust_ret = lambda a, sym=symbol: type_env.get("return", _rust_pyo3_type(a, sym) if is_pyo3 else _rust_type(a, sym))
            arg_str = ", ".join(f"{n}: {rust_arg(a, n)}" for n, a in zip(arg_names, args)) or ""
            ret = f" -> {rust_ret(return_type)}" if return_type else ""
            if is_pyo3:
                parts.append(f"#[pyfunction]\npub fn {symbol}({arg_str}){ret} {{\n    __AERO_IN_FILL__\n}}")
            else:
                parts.append(f"#[no_mangle]\npub extern \"C\" fn {symbol}({arg_str}){ret} {{\n    __AERO_IN_FILL__\n}}")
            parts.append("")
        return "\n".join(parts).rstrip()

    if lang in ("cpp", "c++", "cxx"):
        parts: List[str] = []
        for symbol, args, return_type in specs:
            arg_names = [f"arg{i}" for i in range(len(args))]
            arg_str = ", ".join(
                f"{type_env.get(n, _cpp_type(a, symbol))} {n}" for n, a in zip(arg_names, args)
            ) or "void"
            ret = type_env.get("return", _cpp_type(return_type, symbol)) if return_type else "void"
            parts.append(f'extern "C" {ret} {symbol}({arg_str}) {{\n    __AERO_IN_FILL__\n}}')
            parts.append("")
        return "\n".join(parts).rstrip()

    if lang in ("go", "golang"):
        parts: List[str] = []
        for symbol, args, return_type in specs:
            arg_names = [f"arg{i}" for i in range(len(args))]
            arg_str = ", ".join(f"{n} {type_env.get(n, _go_type(a, symbol))}" for n, a in zip(arg_names, args)) or ""
            ret = type_env.get("return", _go_type(return_type, symbol)) if return_type else ""
            ret_str = f" {ret}" if ret else ""
            parts.append(f"//export {symbol}\nfunc {symbol}({arg_str}){ret_str} {{\n    __AERO_IN_FILL__\n}}")
            parts.append("")
        return "\n".join(parts).rstrip()

    if lang == "zig":
        parts: List[str] = []
        for symbol, args, return_type in specs:
            arg_names = [f"arg{i}" for i in range(len(args))]
            arg_str = ", ".join(f"{n}: {type_env.get(n, _zig_type(a, symbol))}" for n, a in zip(arg_names, args)) or ""
            ret = type_env.get("return", _zig_type(return_type, symbol)) if return_type else "void"
            parts.append(f"export fn {symbol}({arg_str}) {ret} {{\n    __AERO_IN_FILL__\n}}")
            parts.append("")
        return "\n".join(parts).rstrip()

    return f"// Implement {specs[0][0]} for {lang}\n"


def format_builder_emitter_user_prompt(
    node: Dict[str, Any],
    boundary_contracts: Optional[List[Dict[str, Any]]] = None,
    *,
    compacted_context: Optional[Dict[str, Any]] = None,
    user_prompt: str = "",
    missing_symbols: Optional[List[str]] = None,
) -> str:
    """Return a user prompt that feeds a node spec into the Builder Emission Agent."""
    import json

    contracts = boundary_contracts or []
    skeleton = _build_skeleton(node, contracts, compacted_context=compacted_context)
    context = compacted_context or {}
    required_symbols = sorted(
        {c.get("symbol", "") for c in contracts if c.get("symbol")}
        | set(node.get("exports") or [])
    )
    payload = {
        "user_prompt": user_prompt,
        "compacted_context": context,
        "skeleton": skeleton,
        "required_symbols": required_symbols,
        "missing_symbols": list(missing_symbols or []),
        "full_implementation_map": (
            "Implement every symbol in `required_symbols`. Do not omit, "
            "truncate, or skip any requested function."
        ),
    }
    missing_note = ""
    if missing_symbols:
        missing_note = (
            "INCOMPLETE MATERIALIZATION RETRY: the following symbols were "
            f"missing from the previous attempt and MUST be included now: {missing_symbols}. "
        )
    return (
        "Generate source and manifest files for the following graph node. "
        "The user request and the Compacted Functional Matrix below are the "
        "exclusive context you need. Use the `skeleton` field as the starting "
        "file: keep all imports, signatures, and decorators, and replace every "
        "`__AERO_IN_FILL__` marker with real implementation code. "
        "You MUST implement ALL symbols listed in `required_symbols` as a "
        "Full Implementation Map. Do not omit any function from the blueprint. "
        + missing_note +
        "For Python targets, prefer emitting a UAST JSON sketch inside a "
        "```uast:<path> fence instead of raw source; the engine will lower it, "
        "run HIN verification, and resolve attribute names like `conj` to "
        "`conjugate`. "
        "Do not return prose, TODOs, or empty responses. "
        "Wrap the entire response between `__AERO_LOGIC_START__` and `__AERO_LOGIC_END__`.\n\n"
        f"```json\n{json.dumps(payload, indent=2, default=str)}\n```"
    )
