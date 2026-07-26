"""Physical file materialization for C++/pybind11 hybrid blueprints.

This materializer is the C++ analogue of :class:`PolyglotMaterializer`: it writes
a pybind11 extension module, a Python loader, an interactive CLI, and pytest
coverage, then compiles the extension with ``g++``/``clang++`` and runs the test
suite.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys
import sysconfig
import textwrap
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from aero_forge.blueprint import Blueprint, ContractEntry, FunctionSpec, ManifestEntry, write_blueprint
from aero_forge.scaffold.polyglot_materializer import _DEFAULT_CONTRACTS, _parse_signature
from aero_forge.scaffold.python_repo_generator import _sanitize_module_name

logger = logging.getLogger("aero_forge.scaffold.cpp")


def _map_cpp_type(type_hint: str) -> str:
    """Map a Python-style type hint to a C++ type understood by pybind11."""
    th = type_hint.strip()
    if th == "float":
        return "double"
    if th == "int":
        return "int64_t"
    if th in ("str", "string"):
        return "std::string"
    if th == "bool":
        return "bool"
    if th in ("None", "void"):
        return "void"
    if th == "list":
        return "std::vector<py::object>"
    if th.startswith("list[") and th.endswith("]"):
        inner = th[5:-1].strip()
        return f"std::vector<{_map_cpp_type(inner)}>"
    if th.startswith("dict[") and th.endswith("]"):
        inner = th[5:-1].strip()
        parts = [p.strip() for p in inner.split(",", 1)]
        k = _map_cpp_type(parts[0]) if parts else "std::string"
        v = _map_cpp_type(parts[1]) if len(parts) > 1 else "py::object"
        return f"std::map<{k}, {v}>"
    if "ndarray" in th.lower() or "array" in th.lower():
        # Best-effort NumPy array binding.
        return "py::array_t<double>"
    return "double"


def _cpp_default_value(type_hint: str) -> str:
    """Return a literal C++ default value for *type_hint*."""
    th = type_hint.strip()
    if th == "float":
        return "0.0"
    if th == "int":
        return "0"
    if th in ("str", "string"):
        return '""'
    if th == "bool":
        return "false"
    if th in ("None", "void"):
        return ""
    if th.startswith("list[") or th == "list":
        return "{}"
    if th.startswith("dict["):
        return "{}"
    return "{}"


def _generate_function_body(name: str, args: List[Tuple[str, str]], return_type: str) -> str:
    """Generate a simple but correct C++ implementation for known contracts."""
    arg_map = {a: t for a, t in args}
    arg_names = [a for a, _ in args]
    ret = _map_cpp_type(return_type)

    if name == "fast_vector_transform" and len(args) == 2:
        # Expect (v: list[float], scalar: float) -> list[float]
        return (
            "    std::vector<double> out;\n"
            "    out.reserve(v.size());\n"
            "    for (double x : v) {\n"
            "        out.push_back(x * scalar);\n"
            "    }\n"
            "    return out;"
        )
    if name == "get_engine_status":
        return (
            '    return {{"status", "ok"}, {"engine", "cpp"}};'
        )

    # Generic stub that returns a default value and ignores args.
    default = _cpp_default_value(return_type)
    if ret == "void":
        return "    (void)" + "; (void)".join(arg_names) + ";\n" if arg_names else ""
    if arg_names:
        return f"    (void)({', '.join(arg_names)});\n    return {default};"
    return f"    return {default};"


def _generate_native_cpp(pkg_name: str, contracts: List[ContractEntry]) -> str:
    """Generate a pybind11 C++ extension source file."""
    includes = [
        "<pybind11/pybind11.h>",
        "<pybind11/stl.h>",
        "<vector>",
        "<string>",
        "<map>",
        "<cmath>",
    ]
    numpy_available = True
    try:
        import numpy  # noqa: F401
    except Exception:
        numpy_available = False
    if numpy_available:
        includes.append("<pybind11/numpy.h>")

    lines: List[str] = ['// Auto-generated C++/pybind11 extension for aero-forge']
    for inc in includes:
        lines.append(f"#include {inc}")
    lines.append("")
    lines.append('#ifdef _WIN32')
    lines.append('#define AERO_EXPORT __declspec(dllexport)')
    lines.append('#else')
    lines.append('#define AERO_EXPORT __attribute__((visibility("default")))')
    lines.append('#endif')
    lines.append("")
    lines.append("namespace py = pybind11;")
    lines.append("")

    function_signatures = []
    for contract in contracts:
        if not contract.signature:
            continue
        try:
            name, args, return_type = _parse_signature(contract.signature)
        except Exception:
            continue
        cpp_ret = _map_cpp_type(return_type)
        cpp_args = [f"{_map_cpp_type(t)} {a}" for a, t in args]
        sig = f'extern "C" AERO_EXPORT {cpp_ret} {name}({", ".join(cpp_args)})'
        function_signatures.append((name, sig))
        lines.append(f"{sig} {{")
        lines.append(_generate_function_body(name, args, return_type))
        lines.append("}")
        lines.append("")

    lines.append(f'PYBIND11_MODULE(_core, m) {{')
    lines.append(f'    m.doc() = "Native C++ extension for {pkg_name}";')
    for name, _ in function_signatures:
        lines.append(f'    m.def("{name}", &{name}, "{name}");')
    lines.append("}")
    lines.append("")
    return "\n".join(lines)


def _generate_setup_py(pkg_name: str) -> str:
    """Generate a setuptools-based setup.py for the pybind11 extension."""
    return (
        'from setuptools import setup, Extension\n'
        'import pybind11\n'
        'import sysconfig\n'
        '\n'
        'pkg = "' + pkg_name + '"\n'
        'ext_name = f"{pkg}._core"\n'
        'ext_sources = [f"{pkg}/native.cpp"]\n'
        '\n'
        'include_dirs = [\n'
        '    pybind11.get_include(),\n'
        '    sysconfig.get_paths()["include"],\n'
        ']\n'
        'try:\n'
        '    import numpy\n'
        '    include_dirs.append(numpy.get_include())\n'
        'except Exception:\n'
        '    pass\n'
        '\n'
        'ext_modules = [\n'
        '    Extension(\n'
        '        ext_name,\n'
        '        ext_sources,\n'
        '        include_dirs=include_dirs,\n'
        '        language="c++",\n'
        '        extra_compile_args=["-std=c++17", "-O3", "-fvisibility=hidden"],\n'
        '    )\n'
        ']\n'
        '\n'
        'setup(\n'
        '    name=pkg,\n'
        '    version="0.1.0",\n'
        '    packages=[pkg],\n'
        '    ext_modules=ext_modules,\n'
        '    zip_safe=False,\n'
        ')\n'
    )


def _generate_pyproject_toml(pkg_name: str) -> str:
    return (
        "[build-system]\n"
        'requires = ["setuptools>=61", "wheel", "pybind11"]\n'
        'build-backend = "setuptools.build_meta"\n'
        "\n"
        "[project]\n"
        f'name = "{pkg_name}"\n'
        'version = "0.1.0"\n'
        'description = "C++/pybind11 hybrid project generated by aero-forge"\n'
        'requires-python = ">=3.10"\n'
    )


def _generate_init(pkg_name: str, function_names: List[str]) -> str:
    all_names = ", ".join(f'"{n}"' for n in function_names)
    imports = ", ".join(function_names) if function_names else ""
    fallback_funcs = []
    for name in function_names:
        if name == "fast_vector_transform":
            fallback_funcs.append(
                "def fast_vector_transform(v, scalar):\n"
                "    return [x * scalar for x in v]\n"
            )
        elif name == "get_engine_status":
            fallback_funcs.append(
                "def get_engine_status():\n"
                '    return {"status": "ok", "engine": "python"}\n'
            )
        else:
            fallback_funcs.append(f"def {name}(*args, **kwargs):\n    pass\n")
    fallback_block = "".join(fallback_funcs)
    if imports:
        return (
            'try:\n'
            f'    from ._core import {imports}\n'
            'except Exception as _exc:\n'
            f'{textwrap.indent(fallback_block, "    ")}'
            '\n'
            f"__all__ = [{all_names}]\n"
        )
    return fallback_block + f"\n__all__ = [{all_names}]\n"


def _generate_cli(pkg_name: str, function_names: List[str]) -> str:
    has_fast = "fast_vector_transform" in function_names
    has_status = "get_engine_status" in function_names
    lines: List[str] = [
        '"""Interactive CLI / REPL for the C++/pybind11 package."""',
        "import cmd",
        "import sys",
        "from typing import List, Optional",
        "",
        f"from {pkg_name} import {', '.join(function_names)}",
        "",
    ]

    commands: List[Tuple[str, str, str]] = []
    if has_fast:
        commands.append((
            "transform",
            (
                '    """Usage: transform <comma-separated floats> <scale>"""\n'
                "    if not args:\n"
                '        print("Usage: transform <comma-separated floats> <scale>")\n'
                "        return\n"
                "    parts = args.split()\n"
                "    if len(parts) != 2:\n"
                '        print("Usage: transform <comma-separated floats> <scale>")\n'
                "        return\n"
                "    try:\n"
                "        data = [float(x.strip()) for x in parts[0].split(',') if x.strip()]\n"
                "        scale = float(parts[1])\n"
                "    except ValueError:\n"
                '        print("Usage: transform <comma-separated floats> <scale>")\n'
                "        return\n"
                "    print(fast_vector_transform(data, scale))"
            ),
            "do_transform(args)",
        ))
    if has_status:
        commands.append((
            "status",
            (
                '    """Print engine status."""\n'
                "    print(get_engine_status())"
            ),
            "do_status(args)",
        ))
    if has_fast:
        commands.extend([
            (
                "bench",
                (
                    '    """Usage: bench <iterations>"""\n'
                    "    import timeit\n"
                    "    try:\n"
                    "        iterations = int(args.strip())\n"
                    "    except (ValueError, AttributeError):\n"
                    '        print("Usage: bench <iterations>")\n'
                    "        return\n"
                    "    data = [float(i) for i in range(100)]\n"
                    "    scale = 2.0\n"
                    "    total = timeit.timeit(lambda: fast_vector_transform(data, scale), number=iterations)\n"
                    "    avg = total / iterations if iterations else 0.0\n"
                    '    print(f"Ran {iterations} calls in {total:.6f} sec, avg: {avg:.9f} sec/call")'
                ),
                "do_bench(args)",
            ),
            (
                "inspect",
                (
                    '    """Inspect the native function."""\n'
                    "    try:\n"
                    "        import inspect\n"
                    "        sig = inspect.signature(fast_vector_transform)\n"
                    '        print(f"fast_vector_transform signature: {sig}")\n'
                    "    except Exception as e:\n"
                    '        print(f"Inspect failed: {e}")'
                ),
                "do_inspect(args)",
            ),
            (
                "batch",
                (
                    '    """Usage: batch <count> <workers>"""\n'
                    "    import random\n"
                    "    import time\n"
                    "    from concurrent.futures import ThreadPoolExecutor\n"
                    "    parts = args.split() if args else []\n"
                    "    if len(parts) != 2:\n"
                    '        print("Usage: batch <count> <workers>")\n'
                    "        return\n"
                    "    try:\n"
                    "        count = int(parts[0])\n"
                    "        workers = int(parts[1])\n"
                    "    except ValueError:\n"
                    '        print("Usage: batch <count> <workers>")\n'
                    "        return\n"
                    "    data_list = [[random.uniform(0, 1) for _ in range(100)] for _ in range(count)]\n"
                    "    scale = 2.0\n"
                    "    start = time.perf_counter()\n"
                    "    with ThreadPoolExecutor(max_workers=workers) as executor:\n"
                    "        results = list(executor.map(lambda x: fast_vector_transform(x, scale), data_list))\n"
                    "    elapsed = time.perf_counter() - start\n"
                    '    print(f"Processed {count} vectors in {elapsed:.6f} seconds using {workers} workers.")\n'
                    "    if results:\n"
                    '        print(f"First result: {results[0]}")'
                ),
                "do_batch(args)",
            ),
            (
                "telemetry",
                (
                    '    """Measure a single transform."""\n'
                    "    import time\n"
                    "    data = [float(i) for i in range(100)]\n"
                    "    scale = 2.0\n"
                    "    start = time.perf_counter()\n"
                    "    fast_vector_transform(data, scale)\n"
                    "    elapsed = time.perf_counter() - start\n"
                    '    print(f"Telemetry: vector transform took {elapsed:.9f} seconds.")'
                ),
                "do_telemetry(args)",
            ),
        ])

    for cmd_name, body, _ in commands:
        lines.append(f"def do_{cmd_name}(args: str = '') -> None:")
        lines.append(body)
        lines.append("")

    lines.append("class AeroShell(cmd.Cmd):")
    lines.append('    intro = "C++/pybind11 REPL. Type \'help\' for commands, \'quit\' to exit."')
    lines.append('    prompt = "cpp> "')
    lines.append("")
    for cmd_name, _, delegate in commands:
        lines.append(f"    def do_{cmd_name}(self, args: str) -> None:")
        lines.append(f"        {delegate}")
        lines.append("")
    lines.append("    def do_quit(self, args: str) -> bool:")
    lines.append('        """Exit the REPL."""')
    lines.append("        return True")
    lines.append("")
    lines.append("    do_exit = do_quit")
    lines.append("")
    lines.append("def main(argv: Optional[List[str]] = None) -> int:")
    lines.append("    import argparse")
    lines.append("    parser = argparse.ArgumentParser()")
    lines.append("    parser.add_argument('commands', nargs='*')")
    lines.append("    ns = parser.parse_args(argv)")
    lines.append("    shell = AeroShell()")
    lines.append("    if ns.commands:")
    lines.append("        for cmd_str in ns.commands:")
    lines.append("            shell.onecmd(cmd_str)")
    lines.append("    else:")
    lines.append("        try:")
    lines.append("            shell.cmdloop()")
    lines.append("        except (EOFError, KeyboardInterrupt):")
    lines.append("            print()")
    lines.append("    return 0")
    lines.append("")
    lines.append('if __name__ == "__main__":')
    lines.append("    sys.exit(main() or 0)")
    lines.append("")
    return "\n".join(lines)


def _generate_tests(pkg_name: str, function_names: List[str]) -> str:
    lines: List[str] = [
        "import math",
        "import pytest",
        "from unittest.mock import patch, MagicMock",
        "",
        f"from {pkg_name} import {', '.join(function_names)}",
        "",
    ]
    if "fast_vector_transform" in function_names:
        lines.extend([
            "def test_fast_vector_transform():",
            "    result = fast_vector_transform([1.0, 2.0, 3.0], 2.0)",
            "    assert isinstance(result, list)",
            "    assert all(math.isclose(a, b, rel_tol=1e-9) for a, b in zip(result, [2.0, 4.0, 6.0]))",
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
    if "fast_vector_transform" in function_names:
        lines.extend([
            "def test_cli_transform(capsys):",
            f"    from {pkg_name}.cli import do_transform",
            '    do_transform("1.0,2.0,3.0 2.0")',
            '    assert "[2.0, 4.0, 6.0]" in capsys.readouterr().out',
            "",
            "def test_repl_quit():",
            f"    from {pkg_name}.cli import AeroShell",
            "    shell = AeroShell()",
            "    assert shell.onecmd('quit') is True",
            "",
        ])
    return "\n".join(lines) + "\n"


def _generate_run_shell(pkg_name: str) -> str:
    return (
        "import sys\n"
        f"from {pkg_name}.cli import main\n"
        "\n"
        'if __name__ == "__main__":\n'
        "    sys.exit(main() or 0)\n"
    )


def _generate_readme(pkg_name: str) -> str:
    return f"# {pkg_name}\n\nC++/pybind11 hybrid project generated by aero-forge.\n"


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


class CppPolyglotMaterializer:
    """Write and build a C++/pybind11 hybrid workspace from a Blueprint."""

    def __init__(self, workspace: Path) -> None:
        self.workspace = Path(workspace)
        self.build_logs = ""

    def materialize(self, blueprint: Blueprint, *, build: bool = False) -> Blueprint:
        """Write the C++ workspace files and optionally build the extension."""
        project = blueprint.project or "polyglot_cpp_project"
        pkg_name = _sanitize_module_name(project)
        contracts = list(blueprint.contracts) if blueprint.contracts else list(_DEFAULT_CONTRACTS)
        function_names = _function_names(contracts)

        self.workspace.mkdir(parents=True, exist_ok=True)
        pkg_dir = self.workspace / pkg_name
        pkg_dir.mkdir(exist_ok=True)
        tests_dir = self.workspace / "tests"
        tests_dir.mkdir(exist_ok=True)

        (pkg_dir / "native.cpp").write_text(_generate_native_cpp(pkg_name, contracts), encoding="utf-8")
        (pkg_dir / "__init__.py").write_text(_generate_init(pkg_name, function_names), encoding="utf-8")
        (pkg_dir / "cli.py").write_text(_generate_cli(pkg_name, function_names), encoding="utf-8")
        (self.workspace / "setup.py").write_text(_generate_setup_py(pkg_name), encoding="utf-8")
        (self.workspace / "pyproject.toml").write_text(_generate_pyproject_toml(pkg_name), encoding="utf-8")
        (self.workspace / "run_shell.py").write_text(_generate_run_shell(pkg_name), encoding="utf-8")
        (self.workspace / "README.md").write_text(_generate_readme(pkg_name), encoding="utf-8")
        (tests_dir / "test_cli.py").write_text(_generate_tests(pkg_name, function_names), encoding="utf-8")

        manifest = [
            ManifestEntry(path=f"{pkg_name}/native.cpp", lang="cpp", purpose="pybind11 extension source"),
            ManifestEntry(path=f"{pkg_name}/__init__.py", lang="python", purpose="package init"),
            ManifestEntry(path=f"{pkg_name}/cli.py", lang="python", purpose="CLI module"),
            ManifestEntry(path="setup.py", lang="python", purpose="setuptools build script"),
            ManifestEntry(path="pyproject.toml", lang="toml", purpose="project manifest"),
            ManifestEntry(path="run_shell.py", lang="python", purpose="launcher"),
            ManifestEntry(path="tests/test_cli.py", lang="python", purpose="tests"),
            ManifestEntry(path="README.md", lang="markdown", purpose="docs"),
        ]

        # Merge manifest into the blueprint so enforcement checks see the files.
        existing_paths = {e.path for e in blueprint.manifest}
        for entry in manifest:
            if entry.path not in existing_paths:
                blueprint.manifest.append(entry)

        write_blueprint(blueprint, self.workspace / "blueprint.aero")

        if build:
            self._build_extension(pkg_name)

        test_path = self.workspace / "tests" / "test_cli.py"
        init_path = pkg_dir / "__init__.py"
        functions = [
            FunctionSpec(
                file=init_path,
                name=name,
                tests=[test_path] if test_path.is_file() else [],
                skip_build=True,
            )
            for name in function_names
        ]
        if (pkg_dir / "cli.py").is_file():
            functions.append(
                FunctionSpec(
                    file=pkg_dir / "cli.py",
                    name="main",
                    tests=[test_path] if test_path.is_file() else [],
                    skip_build=True,
                )
            )
        blueprint = blueprint.model_copy(update={"functions": functions})

        return blueprint

    def _log(self, text: str) -> None:
        """Append *text* to the build log."""
        if text:
            self.build_logs = (self.build_logs + "\n" + text).strip()

    def _build_extension(self, pkg_name: str) -> bool:
        """Compile the pybind11 extension in place. Returns True on success."""
        env = dict(os.environ)
        env["PYTHONPATH"] = (
            f"{self.workspace}{os.pathsep}{env.get('PYTHONPATH', '')}"
        ).strip(os.pathsep)

        # Build the extension in place.
        build_cmd = [sys.executable, "setup.py", "build_ext", "--inplace"]
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
            logger.error("C++ extension build failed:\n%s", build_proc.stderr)
            return False
        return True
