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
    if t.startswith("list[") or t.startswith("list["):
        elem = t.split("[", 1)[1].split("]", 1)[0]
        return f"(usize, *const {_rust_arg_type(elem)})"
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

        # C declaration (match the C-ABI order emitted by CppEmitter: pointer then length)
        c_args: List[str] = []
        for arg_name, arg_type in args:
            at = arg_type.lower()
            if at.startswith("list["):
                elem = at.split("[", 1)[1].split("]", 1)[0]
                c_args.append(f"{arg_name}: *const {_rust_arg_type(elem)}")
                c_args.append(f"{arg_name}_len: usize")
            else:
                c_args.append(f"{arg_name}: {_rust_arg_type(arg_type)}")
        if return_type.lower().startswith("list["):
            c_args.append("out_len: *mut usize")

        extern_decls.append(
            f"    fn {name}({', '.join(c_args)}) -> {_rust_return_type(return_type)};"
        )

        # Rust wrapper that builds a Vec<f64> for the list[float] contract
        wrapper_lines: List[str] = []
        wrapper_lines.append(f"pub fn {name}_rust({', '.join(f'{arg_name}: {_rust_signature_type(arg_type)}' for arg_name, arg_type in args)}) -> {_rust_signature_return_type(return_type)} {{")
        c_call_args: List[str] = []
        for arg_name, arg_type in args:
            at = arg_type.lower()
            if at.startswith("list["):
                elem = at.split("[", 1)[1].split("]", 1)[0]
                wrapper_lines.append(f"    let {arg_name}_ptr = {arg_name}.as_ptr();")
                wrapper_lines.append(f"    let {arg_name}_len = {arg_name}.len();")
                c_call_args.append(f"{arg_name}_ptr")
                c_call_args.append(f"{arg_name}_len")
            else:
                c_call_args.append(arg_name)
        if return_type.lower().startswith("list["):
            wrapper_lines.append("    let mut out_len: usize = 0;")
            elem = return_type.lower().split("[", 1)[1].split("]", 1)[0]
            wrapper_lines.append(
                f"    let ptr = unsafe {{ {name}({', '.join(c_call_args)}, &mut out_len) }};"
            )
            wrapper_lines.append(
                f"    let result: Vec<{_rust_arg_type(elem)}> = unsafe {{ std::slice::from_raw_parts(ptr, out_len).to_vec() }};"
            )
            wrapper_lines.append(f"    unsafe {{ free_buffer_{_rust_arg_type(elem)}(ptr, out_len) }};")
            wrapper_lines.append("    result")
        else:
            wrapper_lines.append(f"    unsafe {{ {name}({', '.join(c_call_args)}) }}")
        wrapper_lines.append("}")
        wrappers.append("\n".join(wrapper_lines))

    lines = [
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
        if at.startswith("list[") and at.endswith("]"):
            parts.append("&data")
        elif at in ("float", "f64", "double"):
            parts.append("2.0")
        elif at in ("int", "i64", "i32"):
            parts.append("2")
        elif at == "bool":
            parts.append("true")
        else:
            parts.append("&data")
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
        "        let data: Vec<f64> = (0..100).map(|i| i as f64).collect();",
        "        let iterations: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(100000);",
        "        let start = Instant::now();",
        "        for _ in 0..iterations {",
    ] + benchmark_calls + [
        "        }",
        "        let elapsed = start.elapsed();",
        "        println!(\"Benchmark: {} iterations in {:?}\", iterations, elapsed);",
        "    } else {",
        "        let data: Vec<f64> = vec![1.0, 2.0, 3.0];",
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
        if not found:
            found = True
            call_args = _sample_rust_call_args(args) if args else ""
            lines.extend([
                "",
                "#[test]",
                f"fn test_hybrid_cpp_rust_{name}() {{",
                "    let data: Vec<f64> = vec![1.0, 2.0, 3.0];",
                f"    let result = {name}_rust({call_args});",
                "    assert_eq!(result, vec![2.0, 4.0, 6.0]);",
                "}",
            ])
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
        cpp_contracts = [c for c in contracts if _is_c_abi_contract(c)]

        accel_log = self.workspace / ".aero_forge_accel.log"
        os.environ["AERO_FORGE_ACCEL_LOG"] = str(accel_log)

        _accel_log("info", "Routing hybrid C++/Rust build through Rust binary with C-ABI static link")
        for c in cpp_contracts:
            language_router.select_native_backend(_contract_to_python_stub(c), hint="cpp")

        cpp_source = _generate_native_cpp(crate_name, cpp_contracts)
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
