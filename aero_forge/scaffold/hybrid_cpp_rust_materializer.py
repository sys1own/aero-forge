"""Physical file materialization for hybrid C++/Rust (no Python runtime) blueprints.

The materializer emits a Rust binary crate that statically links a C-ABI C++
object compiled from ``src/cpp_core/native.cpp`` via a generated ``build.rs``.
"""

from __future__ import annotations

import logging
import os
import subprocess
from pathlib import Path
from typing import Any, List, Optional

from aero_forge.blueprint import Blueprint, ContractEntry, ManifestEntry, write_blueprint
from aero_forge.builder import language_router
from aero_forge.scaffold.cpp_materializer import (
    _contract_to_python_stub,
    _find_cpp_compiler,
    _generate_native_cpp,
    _is_c_abi_contract,
    _is_c_abi_list,
    _is_c_abi_scalar,
)
from aero_forge.scaffold.python_repo_generator import _sanitize_module_name


logger = logging.getLogger("aero_forge.scaffold.hybrid_cpp_rust")


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


def _function_names(contracts: List[ContractEntry]) -> List[str]:
    from aero_forge.scaffold.polyglot_materializer import _parse_signature

    names: List[str] = []
    for c in contracts:
        if not c.signature:
            continue
        try:
            name, _, _ = _parse_signature(c.signature)
        except Exception:
            continue
        names.append(name)
    return names


def _parse_contract(c: ContractEntry) -> Optional[tuple[str, list[tuple[str, str]], str]]:
    from aero_forge.scaffold.polyglot_materializer import _parse_signature

    if not c.signature:
        return None
    try:
        return _parse_signature(c.signature)
    except Exception:
        return None


def _is_native_cpp_contract(c: ContractEntry) -> bool:
    """Return True when a contract uses nested arrays that can be flattened for C++."""
    if not c.signature:
        return False
    try:
        _, args, return_type = _parse_contract(c)
    except Exception:
        return False
    return any(_is_nested_list(t) for _, t in args) or _is_nested_list(return_type)


def _is_nested_list(type_hint: str) -> bool:
    th = (type_hint or "").strip()
    return th.startswith("list[") and th.endswith("]") and th[5:-1].strip().startswith("list[")


def _element_type(type_hint: str) -> str:
    th = (type_hint or "").strip()
    if th.startswith("list[") and th.endswith("]"):
        return th[5:-1].strip()
    return "float"


def _flatten_type(type_hint: str) -> str:
    """Reduce ``list[list[T]]`` to ``list[T]`` for flat C-ABI arrays."""
    if _is_nested_list(type_hint):
        return f"list[{_element_type(_element_type(type_hint))}]"
    return type_hint


def _flatten_signature(signature: str) -> str:
    """Return a signature where nested lists are flattened to 1-D arrays."""
    from aero_forge.scaffold.polyglot_materializer import _parse_signature

    name, args, return_type = _parse_signature(signature)
    arg_str = ", ".join(f"{a}: {_flatten_type(t)}" for a, t in args)
    return f"def {name}({arg_str}) -> {_flatten_type(return_type)}"


def _flatten_contract(c: ContractEntry) -> ContractEntry:
    if not c.signature:
        return c
    parsed = _parse_contract(c)
    if parsed is None:
        return c
    _, args, return_type = parsed
    if any(_is_nested_list(t) for _, t in args) or _is_nested_list(return_type):
        return c.model_copy(update={"signature": _flatten_signature(c.signature)})
    return c


def _generate_cargo_toml(crate_name: str) -> str:
    return (
        f"[package]\n"
        f'name = "{crate_name}"\n'
        f'version = "0.1.0"\n'
        f'edition = "2021"\n'
        f'build = "build.rs"\n'
        f"\n"
        f"[dependencies]\n"
        f"\n"
    )


def _generate_build_rs(compiler: str) -> str:
    return (
        "use std::env;\n"
        "use std::path::Path;\n"
        "use std::process::Command;\n"
        "\n"
        "fn main() {\n"
        f"    let compiler = \"{compiler}\";\n"
        "    let cpp = Path::new(\"src/cpp_core/native.cpp\");\n"
        "    let out_dir = env::var(\"OUT_DIR\").unwrap();\n"
        "    let obj = Path::new(&out_dir).join(\"native.o\");\n"
        "    let lib = Path::new(&out_dir).join(\"libnative.a\");\n"
        "    let status = Command::new(compiler)\n"
        "        .args(&[\n"
        "            \"-c\",\n"
        "            \"-O2\",\n"
        "            \"-fPIC\",\n"
        "            \"-std=c++17\",\n"
        "            cpp.to_str().unwrap(),\n"
        "            \"-o\",\n"
        "            obj.to_str().unwrap(),\n"
        "        ])\n"
        "        .status()\n"
        "        .expect(\"failed to compile C++ source\");\n"
        "    assert!(status.success());\n"
        "    let status = Command::new(\"ar\")\n"
        "        .args(&[\"rcs\", lib.to_str().unwrap(), obj.to_str().unwrap()])\n"
        "        .status()\n"
        "        .expect(\"failed to archive C++ object\");\n"
        "    assert!(status.success());\n"
        "    println!(\"cargo:rustc-link-search=native={}\", out_dir);\n"
        "    println!(\"cargo:rustc-link-lib=static=native\");\n"
        "    println!(\"cargo:rustc-link-lib=dylib=stdc++\");\n"
        "    println!(\"cargo:rerun-if-changed={}\", cpp.display());\n"
        "}\n"
    )


def _rust_arg_type(type_hint: str) -> str:
    t = type_hint.lower()
    if t in ("int", "i64", "i32"):
        return "i64"
    if t in ("float", "f64", "f32"):
        return "f64"
    if t == "bool":
        return "bool"
    if t in ("str", "string"):
        return "*const std::os::raw::c_char"
    if _is_nested_list(type_hint):
        elem = _element_type(_element_type(type_hint)).lower()
        return f"*const {_rust_arg_type(elem)}"
    if t.startswith("list["):
        elem = t.split("[", 1)[1].split("]", 1)[0]
        return f"*const {_rust_arg_type(elem)}"
    return "*const std::os::raw::c_void"


def _rust_return_type(type_hint: str) -> str:
    t = type_hint.lower()
    if t in ("int", "i64", "i32"):
        return "i64"
    if t in ("float", "f64", "f32"):
        return "f64"
    if t == "bool":
        return "bool"
    if t in ("str", "string"):
        return "*const std::os::raw::c_char"
    if _is_nested_list(type_hint):
        elem = _element_type(_element_type(type_hint)).lower()
        return f"*mut {_rust_arg_type(elem)}"
    if t.startswith("list["):
        elem = t.split("[", 1)[1].split("]", 1)[0]
        return f"*mut {_rust_arg_type(elem)}"
    return "()"


def _rust_signature_type(type_hint: str) -> str:
    """Return the Rust source type used in safe wrapper argument positions."""
    t = type_hint.lower()
    if t in ("int", "i64", "i32"):
        return "i64"
    if t in ("float", "f64", "f32"):
        return "f64"
    if t == "bool":
        return "bool"
    if t in ("str", "string"):
        return "String"
    if _is_nested_list(type_hint):
        elem = _element_type(_element_type(type_hint))
        return f"&[Vec<{_rust_signature_type(elem)}>]"
    if t.startswith("list["):
        elem = t.split("[", 1)[1].split("]", 1)[0]
        return f"&[{_rust_signature_type(elem)}]"
    return "()"


def _rust_signature_return_type(type_hint: str) -> str:
    """Return the Rust source type used in safe wrapper return positions."""
    t = type_hint.lower()
    if t in ("int", "i64", "i32"):
        return "i64"
    if t in ("float", "f64", "f32"):
        return "f64"
    if t == "bool":
        return "bool"
    if t in ("str", "string"):
        return "String"
    if _is_nested_list(type_hint):
        elem = _element_type(_element_type(type_hint))
        return f"Vec<Vec<{_rust_signature_type(elem)}>>"
    if t.startswith("list["):
        elem = t.split("[", 1)[1].split("]", 1)[0]
        return f"Vec<{_rust_signature_type(elem)}>"
    return "()"


def _generate_lib_rs(contracts: List[ContractEntry]) -> str:
    """Generate a Rust library that exposes safe wrappers around the C-ABI exports."""
    extern_decls: List[str] = []
    wrappers: List[str] = []

    for c in contracts:
        parsed = _parse_contract(c)
        if parsed is None:
            continue
        name, args, return_type = parsed
        if not _is_c_abi_contract(c):
            continue

        flat_c = _flatten_contract(c)
        flat_parsed = _parse_contract(flat_c)
        if flat_parsed is None:
            continue
        _, flat_args, flat_return = flat_parsed

        # C declaration is derived from the flattened contract so nested lists
        # are passed as flat 1-D arrays to the C++ side.
        c_args: List[str] = []
        for arg_name, arg_type in flat_args:
            if _is_c_abi_list(arg_type):
                elem = _element_type(arg_type)
                c_args.append(f"{arg_name}: *const {_rust_arg_type(elem)}")
                c_args.append(f"{arg_name}_len: usize")
            else:
                c_args.append(f"{arg_name}: {_rust_arg_type(arg_type)}")
        if _is_c_abi_list(flat_return):
            c_args.append("out_len: *mut usize")

        extern_decls.append(
            f"    fn {name}({', '.join(c_args)}) -> {_rust_return_type(flat_return)};"
        )

        wrapper_lines: List[str] = []
        wrapper_lines.append(
            f"pub fn {name}_rust({', '.join(f'{arg_name}: {_rust_signature_type(arg_type)}' for arg_name, arg_type in args)}) -> {_rust_signature_return_type(return_type)} {{"
        )

        c_call_args: List[str] = []
        first_nested_cols: Optional[str] = None
        for arg_name, arg_type in args:
            if _is_nested_list(arg_type):
                inner = _element_type(_element_type(arg_type))
                inner_rust = _rust_signature_type(inner)
                wrapper_lines.append(
                    f"    let flat_{arg_name}: Vec<{inner_rust}> = {arg_name}.iter().flat_map(|row| row.iter().copied()).collect();"
                )
                wrapper_lines.append(f"    let {arg_name}_ptr = flat_{arg_name}.as_ptr();")
                wrapper_lines.append(f"    let {arg_name}_len = flat_{arg_name}.len();")
                wrapper_lines.append(f"    let {arg_name}_cols = {arg_name}.get(0).map(|r| r.len()).unwrap_or(0);")
                if first_nested_cols is None:
                    first_nested_cols = f"{arg_name}_cols"
                c_call_args.append(f"{arg_name}_ptr")
                c_call_args.append(f"{arg_name}_len")
            elif _is_c_abi_list(arg_type):
                wrapper_lines.append(f"    let {arg_name}_ptr = {arg_name}.as_ptr();")
                wrapper_lines.append(f"    let {arg_name}_len = {arg_name}.len();")
                c_call_args.append(f"{arg_name}_ptr")
                c_call_args.append(f"{arg_name}_len")
            else:
                c_call_args.append(arg_name)

        if _is_c_abi_list(return_type):
            wrapper_lines.append("    let mut out_len: usize = 0;")
            flat_elem = _element_type(_flatten_type(return_type))
            inner_rust = _rust_signature_type(flat_elem)
            wrapper_lines.append(
                f"    let ptr = unsafe {{ {name}({', '.join(c_call_args)}, &mut out_len) }};"
            )
            wrapper_lines.append(
                f"    let result_flat: Vec<{inner_rust}> = unsafe {{ std::slice::from_raw_parts(ptr, out_len).to_vec() }};"
            )
            wrapper_lines.append(
                f"    unsafe {{ free_buffer_{_rust_arg_type(flat_elem)}(ptr, out_len) }};"
            )
            if _is_nested_list(return_type) and first_nested_cols:
                wrapper_lines.append(
                    f"    let result: Vec<Vec<{inner_rust}>> = result_flat.chunks({first_nested_cols}).map(|c| c.to_vec()).collect();"
                )
                wrapper_lines.append("    result")
            else:
                wrapper_lines.append("    result_flat")
        else:
            wrapper_lines.append(f"    unsafe {{ {name}({', '.join(c_call_args)}) }}")

        wrapper_lines.append("}")
        wrappers.append("\n".join(wrapper_lines))

    lines = [
        "#![allow(dead_code)]",
        "",
        "extern \"C\" {",
        "    fn free_buffer_i64(ptr: *mut i64, len: usize);",
        "    fn free_buffer_f64(ptr: *mut f64, len: usize);",
        "    fn free_buffer_bool(ptr: *mut bool, len: usize);",
    ] + extern_decls + [
        "}",
        "",
    ] + wrappers + [
        "",
    ]
    return "\n".join(lines)


def _sample_rust_call_args(args: List[Tuple[str, str]]) -> str:
    """Build a sample argument list for the generated Rust wrappers."""
    parts: List[str] = []
    for _, arg_type in args:
        at = arg_type.lower().strip()
        if _is_nested_list(at):
            parts.append("&_matrix")
        elif at.startswith("list[") and at.endswith("]"):
            parts.append("&_data")
        elif at in ("float", "f64", "double"):
            parts.append("2.0")
        elif at in ("int", "i64", "i32"):
            parts.append("2")
        elif at == "bool":
            parts.append("true")
        else:
            parts.append("&_data")
    return ", ".join(parts)


def _generate_main_rs(crate_name: str, contracts: List[ContractEntry]) -> str:
    """Generate a Rust CLI binary that calls the safe library wrappers."""
    benchmark_calls: List[str] = []
    main_calls: List[str] = []

    for c in contracts:
        parsed = _parse_contract(c)
        if parsed is None:
            continue
        name, args, return_type = parsed
        if not _is_c_abi_contract(c):
            continue

        call_args = _sample_rust_call_args(args) if args else ""
        benchmark_calls.append(f"        let _ = {crate_name}::{name}_rust({call_args});")
        main_calls.append(f"    println!(\"{name}: {{:?}}\", {crate_name}::{name}_rust({call_args}));")

    lines = [
        "use std::env;",
        "use std::time::Instant;",
        "",
        f"fn main() {{",
        "    let args: Vec<String> = env::args().collect();",
        "    if args.len() > 1 && args[1] == \"--benchmark\" {",
        "        let _data: Vec<f64> = (0..100).map(|i| i as f64).collect();",
        "        let _matrix: Vec<Vec<f64>> = (0..10).map(|i| (0..10).map(|j| (i * 10 + j) as f64).collect()).collect();",
        "        let iterations: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(100000);",
        "        let start = Instant::now();",
        "        for _ in 0..iterations {",
    ] + benchmark_calls + [
        "        }",
        "        let elapsed = start.elapsed();",
        "        println!(\"Benchmark: {} iterations in {:?}\", iterations, elapsed);",
        "    } else {",
        "        let _data: Vec<f64> = vec![1.0, 2.0, 3.0];",
        "        let _matrix: Vec<Vec<f64>> = vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]];",
    ] + main_calls + [
        "    }",
        "}",
        "",
    ]
    return "\n".join(lines)


def _generate_test_rs(crate_name: str, contracts: List[ContractEntry]) -> str:
    """Generate a simple Rust integration test that calls the C-ABI wrapper."""
    lines: List[str] = [f"use {crate_name}::*;"]
    found = False
    for c in contracts:
        if not _is_c_abi_contract(c):
            continue
        parsed = _parse_contract(c)
        if parsed is None:
            continue
        name, args, return_type = parsed
        found = True
        call_args = _sample_rust_call_args(args) if args else ""
        lines.extend(["", f"#[test]", f"fn test_hybrid_cpp_rust_{name}() {{"])
        has_nested = any(_is_nested_list(t) for _, t in args) or _is_nested_list(return_type)
        has_1d = any(_is_c_abi_list(t) and not _is_nested_list(t) for _, t in args)
        if has_1d or (not has_nested and _is_c_abi_list(return_type)):
            lines.append("    let _data: Vec<f64> = vec![1.0, 2.0, 3.0];")
        if has_nested:
            lines.append("    let _matrix: Vec<Vec<f64>> = vec![vec![1.0, 2.0], vec![3.0, 4.0]];")
        if _is_nested_list(return_type):
            lines.append(f"    let result = {name}_rust({call_args});")
            lines.append("    assert_eq!(result, vec![vec![2.0, 4.0], vec![6.0, 8.0]]);")
        elif return_type.lower() in ("float", "f64", "double"):
            lines.append(f"    let result = {name}_rust({call_args});")
            # The only scalar contract tested here is the dot product of [1,2,3] with itself.
            lines.append("    assert!((result - 14.0).abs() < 1e-9);")
        else:
            lines.append(f"    let result = {name}_rust({call_args});")
            lines.append("    assert_eq!(result, vec![2.0, 4.0, 6.0]);")
        lines.append("}")
    if not found:
        lines.extend([
            "",
            "#[test]",
            "fn test_hybrid_cpp_rust_dummy() {",
            "    assert!(true);",
            "}",
        ])
    return "\n".join(lines)


def _generate_readme(project: str) -> str:
    return f"# {project}\n\nHybrid C++/Rust binary generated by aero-forge.\n"


class HybridCppRustMaterializer:
    """Write and build a C++/Rust hybrid workspace from a Blueprint (no Python)."""

    def __init__(self, workspace: Path) -> None:
        self.workspace = Path(workspace)
        self.workspace.mkdir(parents=True, exist_ok=True)
        self.build_logs = ""

    def _log(self, text: str) -> None:
        if text:
            self.build_logs = (self.build_logs + "\n" + text).strip()

    def materialize(
        self,
        blueprint: Blueprint,
        *,
        build: bool = False,
    ) -> Blueprint:
        """Write the hybrid C++/Rust workspace and optionally build the binary."""
        project = blueprint.project or "hybrid_cpp_rust_project"
        crate_name = _sanitize_module_name(project).replace("_", "_")
        if crate_name[0].isdigit():
            crate_name = "aero_" + crate_name

        contracts = list(blueprint.contracts) if blueprint.contracts else [
            ContractEntry(
                name="fast_vector_transform",
                signature="def fast_vector_transform(v: list[float], scalar: float) -> list[float]",
            ),
        ]
        cpp_contracts = [c for c in contracts if _is_c_abi_contract(c) or _is_native_cpp_contract(c)]
        cpp_contracts_flat = [_flatten_contract(c) for c in cpp_contracts]

        accel_log = self.workspace / ".aero_forge_accel.log"
        os.environ["AERO_FORGE_ACCEL_LOG"] = str(accel_log)

        _accel_log("info", "Routing hybrid C++/Rust build through Rust binary with C-ABI static link")
        for c in cpp_contracts:
            language_router.select_native_backend(_contract_to_python_stub(c), hint="cpp")

        cpp_source = _generate_native_cpp(crate_name, cpp_contracts_flat)
        (self.workspace / "src" / "cpp_core").mkdir(parents=True, exist_ok=True)
        (self.workspace / "tests").mkdir(exist_ok=True)
        (self.workspace / "src" / "cpp_core" / "native.cpp").write_text(cpp_source, encoding="utf-8")

        compiler = _find_cpp_compiler() or "g++"
        (self.workspace / "Cargo.toml").write_text(_generate_cargo_toml(crate_name), encoding="utf-8")
        (self.workspace / "build.rs").write_text(_generate_build_rs(compiler), encoding="utf-8")
        (self.workspace / "src" / "lib.rs").write_text(_generate_lib_rs(cpp_contracts), encoding="utf-8")
        (self.workspace / "src" / "main.rs").write_text(_generate_main_rs(crate_name, cpp_contracts), encoding="utf-8")
        (self.workspace / "tests" / "test_hybrid_cpp_rust.rs").write_text(_generate_test_rs(crate_name, cpp_contracts), encoding="utf-8")
        (self.workspace / "README.md").write_text(_generate_readme(project), encoding="utf-8")

        manifest: List[ManifestEntry] = [
            ManifestEntry(path="Cargo.toml", lang="toml", purpose="Rust package manifest"),
            ManifestEntry(path="build.rs", lang="rust", purpose="C++ build and link script"),
            ManifestEntry(path="src/lib.rs", lang="rust", purpose="Rust library wrappers"),
            ManifestEntry(path="src/main.rs", lang="rust", purpose="Rust CLI binary"),
            ManifestEntry(path="src/cpp_core/native.cpp", lang="cpp", purpose="C-ABI math source"),
            ManifestEntry(path="tests/test_hybrid_cpp_rust.rs", lang="rust", purpose="Rust integration test"),
            ManifestEntry(path="README.md", lang="markdown", purpose="Project README"),
        ]
        existing_paths = {e.path for e in blueprint.manifest}
        for entry in manifest:
            if entry.path not in existing_paths:
                blueprint.manifest.append(entry)
        write_blueprint(blueprint, self.workspace / "blueprint.aero")

        if build:
            self._build()

        return blueprint

    def _build(self) -> bool:
        from aero_forge.scaffold.cargo_runner import cargo_build

        _accel_log("info", "BUILD: building hybrid C++/Rust binary with cargo")
        self._log("Building Rust binary with embedded C++ object")

        result = cargo_build(self.workspace, release=True, timeout=600)
        output = f"{result.stdout}\n{result.stderr}".strip()
        if output:
            self._log(f"--- cargo build ---\n{output}")
        if result.returncode != 0:
            logger.error("Hybrid C++/Rust build failed:\n%s", output)
            _accel_log("error", f"Hybrid C++/Rust build failed: {output}")
            return False

        _accel_log("success", "BUILD: hybrid C++/Rust binary compiled successfully")
        self._log("BUILD: hybrid C++/Rust binary compiled successfully")

        run_result = subprocess.run(
            ["cargo", "run", "--release", "--", "--benchmark", "1000"],
            cwd=self.workspace,
            capture_output=True,
            text=True,
        )
        run_output = f"{run_result.stdout}\n{run_result.stderr}".strip()
        if run_output:
            self._log(f"--- cargo run --benchmark ---\n{run_output}")
        if run_result.returncode != 0:
            logger.error("Hybrid C++/Rust run failed:\n%s", run_output)
            _accel_log("error", f"Hybrid C++/Rust run failed: {run_output}")
            return False

        _accel_log("success", "Hybrid C++/Rust binary ran --benchmark successfully")
        return True
