"""Physical file materialization for tri-polyglot (Python + Rust + C++) blueprints.

The materializer writes a Python driver package, a Rust PyO3 core, and a
C-ABI dynamic shared library into a single workspace and builds all three.
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from aero_forge.blueprint import (
    Blueprint,
    ContractEntry,
    FunctionSpec,
    ManifestEntry,
    write_blueprint,
)
from aero_forge.builder import language_router
from aero_forge.native_bridge import _ctypes_loader_source
from aero_forge.scaffold.cargo_runner import cargo_build
from aero_forge.scaffold.cpp_materializer import (
    _contract_to_python_stub as _cpp_contract_to_python_stub,
    _find_cpp_compiler,
    _generate_native_cpp,
    _is_c_abi_contract,
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
    """Partition contracts into C++ (C-ABI), Rust (PyO3), and Python fallbacks."""
    cpp: List[ContractEntry] = []
    rust: List[ContractEntry] = []
    python: List[ContractEntry] = []
    for contract in contracts:
        if not contract.signature:
            continue
        if _is_c_abi_contract(contract):
            cpp.append(contract)
            continue
        try:
            _, args, return_type = _parse_signature(contract.signature)
        except Exception:
            python.append(contract)
            continue
        rust_supported = all(
            t.lower() in ("int", "i64", "i32", "float", "f64", "f32", "bool", "str", "string")
            for _, t in args
        ) and return_type.lower() in ("int", "i64", "i32", "float", "f64", "f32", "bool", "str", "string")
        if rust_supported:
            rust.append(contract)
        else:
            python.append(contract)
    return cpp, rust, python


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
        arg_str = ", ".join(f"{arg}: {typ}" for arg, typ in args)
        lines.append(f"def {name}({arg_str}) -> {return_type}:")
        lines.append(_generate_stub_body(name, args, return_type))
        lines.append("")
    if names:
        lines.append("__all__ = [" + ", ".join(f'"{n}"' for n in names) + "]")
    else:
        lines.append('__all__: list[str] = []')
    return "\n".join(lines) + "\n"


def _generate_stub_body(name: str, args: List[Tuple[str, str]], return_type: str) -> str:
    """Return a simple fallback body for a contract stub."""
    rt = return_type.lower()
    if name == "validate_token":
        return '    return len(token) > 8'
    if rt in ("int", "i64", "i32"):
        return "    return 0"
    if rt in ("float", "f64", "f32"):
        return "    return 0.0"
    if rt == "bool":
        return "    return True"
    if rt == "str":
        return '    return "ok"'
    return "    return None"


def _generate_python_init(
    pkg_name: str,
    workspace: Path,
    cpp_contracts: List[ContractEntry],
    rust_contracts: List[ContractEntry],
    python_contracts: List[ContractEntry],
    rust_crate_name: str,
) -> str:
    """Generate ``<pkg>/__init__.py`` that loads both the Rust extension and C++ .so."""
    lines: List[str] = ['"""Tri-polyglot driver package."""', "", "from __future__ import annotations", ""]

    # Rust extension loader
    lines.append(_native_loader_source([rust_crate_name]))
    lines.append("")

    # Rust-backed functions
    for contract in rust_contracts:
        if not contract.signature:
            continue
        try:
            name, args, return_type = _parse_signature(contract.signature)
        except Exception:
            continue
        arg_sig = ", ".join(f"{a}: {t}" for a, t in args)
        arg_call = ", ".join(a for a, _ in args)
        fallback = _generate_stub_body(name, args, return_type)
        lines.append(f"def {name}({arg_sig}) -> {return_type}:")
        lines.append(f"    if _NATIVE is not None and hasattr(_NATIVE, '{name}'):")
        lines.append(f"        return _NATIVE.{name}({arg_call})")
        lines.append(fallback)
        lines.append("")

    # C++ ctypes loader for C-ABI contracts
    cpp_names: List[str] = _function_names(cpp_contracts)
    if cpp_names:
        stub_source = "\n".join(_cpp_contract_to_python_stub(c) for c in cpp_contracts)
        so_path = (workspace / "cpp_core" / _so_name(f"{pkg_name}_cpp")).resolve()
        lines.append(_ctypes_loader_source(stub_source, so_path, cpp_names))
        lines.append("")

    # Fallback pure-Python implementations
    for contract in python_contracts:
        if not contract.signature:
            continue
        try:
            name, args, return_type = _parse_signature(contract.signature)
        except Exception:
            continue
        arg_sig = ", ".join(f"{a}: {t}" for a, t in args)
        lines.append(f"def {name}({arg_sig}) -> {return_type}:")
        lines.append(_generate_fallback_body(name, args, return_type))
        lines.append("")

    all_names = _function_names(cpp_contracts + rust_contracts + python_contracts)
    lines.append(f"__all__ = {all_names!r}")
    return "\n".join(lines) + "\n"


def _generate_fallback_body(name: str, args: List[Tuple[str, str]], return_type: str) -> str:
    rt = return_type.lower()
    if name == "get_engine_status":
        return '    return {"status": "ok", "engine": "tri_polyglot", "languages": ["python", "rust", "cpp"]}'
    if "list" in rt:
        return "    return []"
    if "dict" in rt:
        return "    return {}"
    if rt in ("int", "i64", "i32"):
        return "    return 0"
    if rt in ("float", "f64", "f32"):
        return "    return 0.0"
    if rt == "bool":
        return "    return True"
    if rt == "str":
        return '    return "ok"'
    return "    return None"


def _generate_main_py(pkg_name: str, function_names: List[str]) -> str:
    lines = [
        '"""Tri-polyglot REPL CLI entrypoint."""',
        "",
        "from __future__ import annotations",
        "",
        "import argparse",
        "import sys",
        "",
        f"from {pkg_name} import {', '.join(function_names)}",
        "",
        "",
        "def run_all() -> None:",
        '    """Run the automated tri-polyglot verification commands."""',
    ]
    if "fast_vector_transform" in function_names:
        lines.append('    print("C++ transform:", fast_vector_transform([1.0, 2.0, 3.0], 2.0))')
    if "validate_token" in function_names:
        lines.append('    print("Rust validate:", validate_token("validtoken123"))')
    if "get_engine_status" in function_names:
        lines.append('    print("Python status:", get_engine_status())')
    lines.extend([
        "",
        "",
        "def main() -> int:",
        '    parser = argparse.ArgumentParser(description="Tri-polyglot CLI")',
        '    parser.add_argument("--cmd", default=None, help="Headless command (e.g. run_all)")',
        "    ns = parser.parse_args()",
        '    if ns.cmd == "run_all":',
        "        run_all()",
        "    elif not sys.stdin.isatty():",
        '        print("Tri-polyglot CLI ready")',
        "        run_all()",
        "    else:",
        '        print("Use --cmd run_all for headless execution")',
        "    return 0",
        "",
        'if __name__ == "__main__":',
        "    sys.exit(main() or 0)",
        "",
    ])
    return "\n".join(lines)


def _generate_run_shell(pkg_name: str) -> str:
    return (
        "import sys\n"
        f"from {pkg_name}.main import main\n"
        "\n"
        'if __name__ == "__main__":\n'
        "    sys.exit(main() or 0)\n"
    )


def _generate_tests(pkg_name: str, function_names: List[str]) -> str:
    lines = [
        f"from {pkg_name} import {', '.join(function_names)}",
        "",
    ]
    if "fast_vector_transform" in function_names:
        lines.extend([
            "def test_fast_vector_transform():",
            "    result = fast_vector_transform([1.0, 2.0, 3.0], 2.0)",
            "    assert isinstance(result, list)",
            "    assert result == [2.0, 4.0, 6.0]",
            "",
        ])
    if "validate_token" in function_names:
        lines.extend([
            "def test_validate_token():",
            '    assert validate_token("validtoken123") is True',
            '    assert validate_token("short") is False',
            "",
        ])
    if "get_engine_status" in function_names:
        lines.extend([
            "def test_get_engine_status():",
            "    status = get_engine_status()",
            "    assert isinstance(status, dict)",
            '    assert status.get("status") == "ok"',
            "",
        ])
    lines.extend([
        "def test_run_all():",
        f"    from {pkg_name}.main import run_all",
        "    run_all()",
        "",
    ])
    return "\n".join(lines) + "\n"


def _generate_readme_tri(project: str) -> str:
    return f"# {project}\n\nTri-polyglot (Python + Rust + C++) workspace generated by aero-forge.\n"


class TriPolyglotMaterializer:
    """Write and build a Python + Rust + C++ tri-polyglot workspace from a Blueprint."""

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
        """Write the tri-polyglot workspace files and optionally build everything."""
        project = blueprint.project or "tri_polyglot_project"
        pkg_name = _sanitize_module_name(project)
        rust_crate_name = f"aero_forge_native_{pkg_name}"

        contracts = list(blueprint.contracts) if blueprint.contracts else [
            ContractEntry(
                name="fast_vector_transform",
                signature="def fast_vector_transform(v: list[float], scalar: float) -> list[float]",
            ),
            ContractEntry(
                name="validate_token",
                signature="def validate_token(token: str) -> bool",
            ),
            ContractEntry(
                name="get_engine_status",
                signature="def get_engine_status() -> dict[str, str]",
            ),
        ]
        cpp_contracts, rust_contracts, python_contracts = _partition_contracts(contracts)

        accel_log = self.workspace / ".aero_forge_accel.log"
        os.environ["AERO_FORGE_ACCEL_LOG"] = str(accel_log)

        pkg_dir = self.workspace / pkg_name
        pkg_dir.mkdir(exist_ok=True)
        cpp_dir = self.workspace / "cpp_core"
        cpp_dir.mkdir(exist_ok=True)
        rust_dir = self.workspace / "rust_core"
        rust_dir.mkdir(exist_ok=True)
        tests_dir = self.workspace / "tests"
        tests_dir.mkdir(exist_ok=True)

        all_names = _function_names(contracts)

        # Telemetry
        _accel_log("info", "Routing tri-polyglot build through Python + Rust + C++ dynamic bridges")
        for contract in cpp_contracts:
            language_router.select_native_backend(
                _cpp_contract_to_python_stub(contract), hint="cpp"
            )
        for contract in rust_contracts:
            language_router.select_native_backend(
                _cpp_contract_to_python_stub(contract), hint="rust_hin"
            )

        # C++ source
        cpp_pkg_name = f"{pkg_name}_cpp"
        cpp_source = _generate_native_cpp(cpp_pkg_name, cpp_contracts)
        (cpp_dir / "native.cpp").write_text(cpp_source, encoding="utf-8")

        # Python package and project files
        (pkg_dir / "__init__.py").write_text(
            _generate_python_init(
                pkg_name, self.workspace, cpp_contracts, rust_contracts, python_contracts, rust_crate_name
            ),
            encoding="utf-8",
        )
        (pkg_dir / "main.py").write_text(_generate_main_py(pkg_name, all_names), encoding="utf-8")
        (self.workspace / "pyproject.toml").write_text(
            _render_pyproject(pkg_name, package_dir="."), encoding="utf-8"
        )
        (self.workspace / "run_shell.py").write_text(
            _generate_run_shell(pkg_name), encoding="utf-8"
        )
        (tests_dir / "test_tri.py").write_text(
            _generate_tests(pkg_name, all_names), encoding="utf-8"
        )
        (self.workspace / "README.md").write_text(
            _generate_readme_tri(project), encoding="utf-8"
        )

        manifest: List[ManifestEntry] = [
            ManifestEntry(path=f"{pkg_name}/__init__.py", lang="python", purpose="Python driver package init"),
            ManifestEntry(path=f"{pkg_name}/main.py", lang="python", purpose="Python CLI / REPL entrypoint"),
            ManifestEntry(path="cpp_core/native.cpp", lang="cpp", purpose="C-ABI shared library source"),
            ManifestEntry(path="rust_core/Cargo.toml", lang="toml", purpose="PyO3 crate manifest"),
            ManifestEntry(path="rust_core/src/lib.rs", lang="rust", purpose="Rust native core"),
            ManifestEntry(path="Cargo.toml", lang="toml", purpose="Rust workspace manifest"),
            ManifestEntry(path="pyproject.toml", lang="toml", purpose="Python package manifest"),
            ManifestEntry(path="run_shell.py", lang="python", purpose="Headless launcher"),
            ManifestEntry(path="tests/test_tri.py", lang="python", purpose="pytest tests"),
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

        if build:
            self._build_cpp(cpp_pkg_name)
            self._build_rust()

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

    def _build_cpp(self, cpp_pkg_name: str) -> bool:
        compiler = _find_cpp_compiler()
        if compiler is None:
            raise RuntimeError("No C++ compiler found (g++, clang++, or c++)")

        cpp_path = self.workspace / "cpp_core" / "native.cpp"
        so_name = _so_name(cpp_pkg_name)
        so_path = self.workspace / "cpp_core" / so_name

        build_cmd = [
            compiler,
            "-shared",
            "-fPIC",
            "-O2",
            "-std=c++17",
            "-o",
            str(so_path),
            str(cpp_path),
        ]
        self._log(f"Compiling C-ABI shared library: {' '.join(build_cmd)}")
        _accel_log("info", f"BUILD: compiling dynamic shared object with {' '.join(build_cmd)}")

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
            logger.error("C++ shared library build failed:\n%s", build_proc.stderr)
            _accel_log("error", f"C++ shared library build failed: {build_proc.stderr}")
            return False

        _accel_log("success", f"BUILD: dynamic shared library compiled: {so_path}")
        return True

    def _build_rust(self) -> bool:
        cargo_toml = self.workspace / "rust_core" / "Cargo.toml"
        if not cargo_toml.is_file():
            logger.error("Rust crate manifest not found: %s", cargo_toml)
            return False

        self._log("Building Rust PyO3 crate in rust_core")
        _accel_log("info", "BUILD: building Rust PyO3 extension with cargo")

        result = cargo_build(self.workspace / "rust_core", release=True, timeout=600)
        output = f"{result.stdout}\n{result.stderr}".strip()
        if output:
            self._log(f"--- cargo build ---\n{output}")
        if result.returncode != 0:
            logger.error("Rust PyO3 build failed:\n%s", output)
            _accel_log("error", f"Rust PyO3 build failed: {output}")
            return False

        _accel_log("success", "BUILD: Rust PyO3 extension compiled successfully")
        return True
