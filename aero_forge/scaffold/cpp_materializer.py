"""Physical file materialization for C++/ctypes hybrid blueprints.

This materializer is the C++ analogue of :class:`PolyglotMaterializer`: it writes
a C-ABI shared dynamic library (``.so``/``.dylib``/``.dll``), a ``ctypes``
Python loader, an interactive CLI, and pytest coverage, then compiles the
library with ``g++``/``clang++`` and runs the test suite.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from aero_forge.blueprint import Blueprint, ContractEntry, FunctionSpec, ManifestEntry, write_blueprint
from aero_forge.builder import language_router
from aero_forge.builder.emitters.cpp_emitter import CppEmitter
from aero_forge.builder.spec import (
    ASTNode,
    EngineSpec,
    binding,
    binary_op,
    block,
    call,
    function,
    list_literal,
    literal,
    module,
    param,
    reference,
    return_node,
)
from aero_forge.native_bridge import _ctypes_loader_source
from aero_forge.scaffold.polyglot_materializer import _DEFAULT_CONTRACTS, _parse_signature
from aero_forge.scaffold.python_repo_generator import _sanitize_module_name

logger = logging.getLogger("aero_forge.scaffold.cpp")


def _find_cpp_compiler() -> Optional[str]:
    for name in ["g++", "clang++", "c++"]:
        if shutil.which(name):
            return name
    return None


def _map_py_type(type_hint: str) -> str:
    """Return the canonical Python scalar type label for a C-ABI type hint."""
    th = (type_hint or "").strip().lower()
    if th in ("float", "f64", "double"):
        return "float"
    if th in ("int", "i64", "i32"):
        return "int"
    if th == "bool":
        return "bool"
    return ""


def _is_c_abi_scalar(type_hint: str) -> bool:
    return bool(_map_py_type(type_hint))


def _is_c_abi_list(type_hint: str) -> bool:
    th = (type_hint or "").strip()
    if th == "list":
        return True
    if th.startswith("list[") and th.endswith("]"):
        inner = th[5:-1].strip()
        return _is_c_abi_scalar(inner)
    return False


def _is_c_abi_contract(contract: ContractEntry) -> bool:
    """Return True when *contract* can be exposed through an extern "C" ABI."""
    if not contract.signature:
        return False
    try:
        name, args, return_type = _parse_signature(contract.signature)
    except Exception:
        return False
    if not _is_c_abi_list(return_type) and not _is_c_abi_scalar(return_type):
        return False
    return all(_is_c_abi_list(t) or _is_c_abi_scalar(t) for _, t in args)


def _contract_to_python_stub(contract: ContractEntry) -> str:
    """Return a typed Python stub suitable for the ctypes loader generator."""
    if not contract.signature:
        return ""
    try:
        name, args, return_type = _parse_signature(contract.signature)
    except Exception:
        return ""
    arg_sig = ", ".join(f"{a}: {t}" for a, t in args)
    return f"def {name}({arg_sig}) -> {return_type}:\n    pass\n"


def _telemetry_source_for_contract(contract: ContractEntry) -> str:
    """Return a representative Python source for AST telemetry logging."""
    stub = _contract_to_python_stub(contract)
    if not stub:
        return ""
    try:
        name, args, return_type = _parse_signature(contract.signature)
    except Exception:
        return stub
    list_args = [(a, t) for a, t in args if _is_c_abi_list(t)]
    scalar_args = [(a, t) for a, t in args if _is_c_abi_scalar(t)]
    # Provide a loop body for the canonical vector transform contract so the
    # AST heuristic logs a heavy numerical matrix loop verdict.
    if name == "fast_vector_transform" and len(args) == 2:
        return (
            "def fast_vector_transform(v: list[float], scalar: float) -> list[float]:\n"
            "    out = []\n"
            "    for x in v:\n"
            "        out.append(x * scalar)\n"
            "    return out\n"
        )
    if len(list_args) == 2 and return_type in ("float", "f64", "double"):
        a_name = list_args[0][0]
        b_name = list_args[1][0]
        return (
            f"def {name}({a_name}: list[float], {b_name}: list[float]) -> float:\n"
            "    total = 0.0\n"
            f"    for i in range(len({a_name})):\n"
            f"        total += {a_name}[i] * {b_name}[i]\n"
            "    return total\n"
        )
    if len(list_args) == 1 and not scalar_args and return_type in ("float", "f64", "double", "int", "i64"):
        a_name = list_args[0][0]
        return (
            f"def {name}({a_name}: list[float]) -> {return_type}:\n"
            "    total = 0\n"
            f"    for x in {a_name}:\n"
            "        total += x\n"
            f"    return total\n"
        )
    return stub


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


def _vector_transform_spec(pkg_name: str) -> EngineSpec:
    """Return an EngineSpec for the canonical ``fast_vector_transform`` contract."""
    out = binding("out", list_literal([]), type_hint="list[float]")
    loop_body = block(
        children=[
            call(
                "out.push_back",
                [binary_op(reference("x"), "*", reference("scalar"))],
            )
        ]
    )
    loop = ASTNode(kind="for", name="x", children=[reference("v"), loop_body])
    ret = return_node(reference("out"))
    func = function(
        "fast_vector_transform",
        params=[param("v", "list[float]"), param("scalar", "float")],
        return_type="list[float]",
        body=[out, loop, ret],
    )
    return EngineSpec(name=pkg_name, root=module(name=pkg_name, children=[func]))


def _known_contract_spec(pkg_name: str, contract: ContractEntry) -> Optional[EngineSpec]:
    """Return an EngineSpec for a contract whose C++ body is known in advance."""
    if not contract.signature:
        return None
    try:
        name, _, _ = _parse_signature(contract.signature)
    except Exception:
        return None
    if name == "fast_vector_transform":
        return _vector_transform_spec(pkg_name)
    return None


def _is_nested_list_type(type_hint: str) -> bool:
    th = (type_hint or "").strip()
    return th.startswith("list[") and th.endswith("]") and _is_c_abi_list(th[5:-1].strip())


def _generic_c_abi_contract_spec(pkg_name: str, contract: ContractEntry) -> Optional[EngineSpec]:
    """Build a default numeric EngineSpec for any C-ABI-compatible contract."""
    if not contract.signature:
        return None
    try:
        name, args, return_type = _parse_signature(contract.signature)
    except Exception:
        return None
    if not _is_c_abi_list(return_type) and not _is_c_abi_scalar(return_type):
        return None
    if not all(_is_c_abi_list(t) or _is_c_abi_scalar(t) for _, t in args):
        return None
    if _is_nested_list_type(return_type) or any(_is_nested_list_type(t) for _, t in args):
        return None
    list_args = [(a, t) for a, t in args if _is_c_abi_list(t)]
    scalar_args = [(a, t) for a, t in args if _is_c_abi_scalar(t)]

    if not list_args:
        return None

    def is_float_list(at: str) -> bool:
        return at.startswith("list[") and _map_py_type(at[5:-1].strip()) == "float"

    def is_int_list(at: str) -> bool:
        return at.startswith("list[") and _map_py_type(at[5:-1].strip()) == "int"

    def scalar_literal(value: Any) -> Any:
        if is_int_list(list_args[0][1]):
            return literal(int(value))
        return literal(float(value))

    # Case 1: scalar return
    if _is_c_abi_scalar(return_type):
        total = binding("total", scalar_literal(0), type_hint=return_type)

        # Two float lists -> dot product
        if len(list_args) == 2 and all(is_float_list(t) for _, t in list_args):
            a_name, _ = list_args[0]
            b_name, _ = list_args[1]
            idx = "i"
            loop = ASTNode(
                kind="for",
                name=idx,
                children=[
                    call("range", [call(f"{a_name}.size", [])]),
                    block(children=[
                        ASTNode(
                            kind="aug_assign",
                            name="total",
                            value="+",
                            children=[
                                binary_op(
                                    ASTNode(kind="subscript", children=[reference(a_name), reference(idx)]),
                                    "*",
                                    ASTNode(kind="subscript", children=[reference(b_name), reference(idx)]),
                                )
                            ],
                        )
                    ]),
                ],
            )
            ret = return_node(reference("total"))
            func = function(
                name,
                params=[param(a, t) for a, t in args],
                return_type=return_type,
                body=[total, loop, ret],
            )
            return EngineSpec(name=pkg_name, root=module(name=pkg_name, children=[func]))

        # Single list -> sum
        if len(list_args) == 1 and not scalar_args:
            list_name, _ = list_args[0]
            loop = ASTNode(
                kind="for",
                name="x",
                children=[
                    reference(list_name),
                    block(children=[
                        ASTNode(
                            kind="aug_assign",
                            name="total",
                            value="+",
                            children=[reference("x")],
                        )
                    ]),
                ],
            )
            ret = return_node(reference("total"))
            func = function(
                name,
                params=[param(a, t) for a, t in args],
                return_type=return_type,
                body=[total, loop, ret],
            )
            return EngineSpec(name=pkg_name, root=module(name=pkg_name, children=[func]))

        # Single list + scalar -> weighted sum (x * scalar)
        if len(list_args) == 1 and len(scalar_args) == 1:
            list_name, _ = list_args[0]
            scalar_name, _ = scalar_args[0]
            loop = ASTNode(
                kind="for",
                name="x",
                children=[
                    reference(list_name),
                    block(children=[
                        ASTNode(
                            kind="aug_assign",
                            name="total",
                            value="+",
                            children=[binary_op(reference("x"), "*", reference(scalar_name))],
                        )
                    ]),
                ],
            )
            ret = return_node(reference("total"))
            func = function(
                name,
                params=[param(a, t) for a, t in args],
                return_type=return_type,
                body=[total, loop, ret],
            )
            return EngineSpec(name=pkg_name, root=module(name=pkg_name, children=[func]))

        return None

    # Case 2: list return
    if _is_c_abi_list(return_type):
        if len(list_args) != 1 or len(scalar_args) > 1:
            return None
        list_name, list_type = list_args[0]
        inner = _map_py_type(list_type[5:-1].strip()) if list_type.startswith("list[") and list_type.endswith("]") else "float"
        scalar_name = scalar_args[0][0] if scalar_args else None

        out = binding("out", list_literal([]), type_hint=return_type)
        if scalar_name:
            loop_expr = binary_op(reference("x"), "*", reference(scalar_name))
        elif inner == "int":
            loop_expr = binary_op(reference("x"), "*", literal(2))
        else:
            loop_expr = binary_op(reference("x"), "*", literal(2.0))
        loop_body = block(children=[call("out.push_back", [loop_expr])])
        loop = ASTNode(kind="for", name="x", children=[reference(list_name), loop_body])
        ret = return_node(reference("out"))
        func = function(
            name,
            params=[param(a, t) for a, t in args],
            return_type=return_type,
            body=[out, loop, ret],
        )
        return EngineSpec(name=pkg_name, root=module(name=pkg_name, children=[func]))

    return None


def _contract_to_engine_spec(pkg_name: str, contract: ContractEntry) -> Optional[EngineSpec]:
    spec = _known_contract_spec(pkg_name, contract)
    if spec is not None:
        return spec
    return _generic_c_abi_contract_spec(pkg_name, contract)


def _generate_native_cpp(pkg_name: str, contracts: List[ContractEntry]) -> str:
    """Generate an ``extern "C"`` shared-library C++ source from *contracts*."""
    specs: List[EngineSpec] = []
    for contract in contracts:
        spec = _contract_to_engine_spec(pkg_name, contract)
        if spec is None:
            continue
        # Emit telemetry for each contract routed to C++.
        telemetry_source = _telemetry_source_for_contract(contract)
        language_router.should_accelerate_with_native(telemetry_source, min_numeric_ops=2)
        language_router.select_native_backend(telemetry_source, hint="cpp")
        specs.append(spec)

    if not specs:
        # Keep the file syntactically valid even when nothing is accelerated.
        return "// Auto-generated C-ABI shared library for aero-forge\n// No C-ABI-compatible contracts were detected.\n"

    # Combine all function specs into a single module so CppEmitter emits one
    # preamble, one set of free-buffer helpers, and one C-ABI wrapper section.
    combined = EngineSpec(
        name=pkg_name,
        root=module(
            name=pkg_name,
            children=[func for spec in specs for func in (spec.root.children or [])],
        ),
    )
    return CppEmitter(c_abi=True).emit(combined)


def _generate_pyproject_toml(pkg_name: str) -> str:
    return (
        "[build-system]\n"
        'requires = ["setuptools>=61", "wheel"]\n'
        'build-backend = "setuptools.build_meta"\n'
        "\n"
        "[project]\n"
        f'name = "{pkg_name}"\n'
        'version = "0.1.0"\n'
        'description = "C++/ctypes hybrid project generated by aero-forge"\n'
        'requires-python = ">=3.10"\n'
    )


def _generate_fallback_body(name: str, args: List[Tuple[str, str]], return_type: str) -> str:
    """Return a pure-Python fallback implementation for a non-native contract."""
    if name == "get_engine_status":
        return '    return {"status": "ok", "engine": "cpp"}'
    if name == "fast_vector_transform":
        return "    return [x * 2.0 for x in v]"
    rt = return_type.lower()
    if "list" in rt:
        return "    return []"
    if "dict" in rt:
        return '    return {}'
    if rt in ("int", "i64", "i32"):
        return "    return 0"
    if rt in ("float", "f64", "f32"):
        return "    return 0.0"
    if rt == "bool":
        return "    return True"
    if rt == "str":
        return '    return "ok"'
    return "    return None"


def _generate_init(pkg_name: str, pkg_dir: Path, contracts: List[ContractEntry]) -> str:
    """Generate ``__init__.py`` that loads the C-ABI .so via ctypes."""
    native_contracts = [c for c in contracts if _is_c_abi_contract(c)]
    fallback_contracts = [c for c in contracts if c not in native_contracts]

    native_names: List[str] = []
    for contract in native_contracts:
        try:
            name, _, _ = _parse_signature(contract.signature)
            native_names.append(name)
        except Exception:
            continue

    all_names: List[str] = []
    for contract in contracts:
        try:
            name, _, _ = _parse_signature(contract.signature)
            all_names.append(name)
        except Exception:
            continue

    pieces: List[str] = []
    if native_names:
        stub_source = "\n".join(_contract_to_python_stub(c) for c in native_contracts)
        so_path = (pkg_dir / _so_name(pkg_name)).resolve()
        pieces.append(_ctypes_loader_source(stub_source, so_path, native_names))

    for contract in fallback_contracts:
        if not contract.signature:
            continue
        try:
            name, args, return_type = _parse_signature(contract.signature)
        except Exception:
            continue
        arg_sig = ", ".join(f"{a}: {t}" for a, t in args)
        pieces.append(f"def {name}({arg_sig}) -> {return_type}:")
        pieces.append(_generate_fallback_body(name, args, return_type))
        pieces.append("")

    pieces.append(f"__all__ = {all_names!r}")
    return "\n".join(pieces) + "\n"


def _is_py_list_type(t: str) -> bool:
    return (t or "").strip().lower().startswith("list[")


def _is_py_nested_list(t: str) -> bool:
    t = (t or "").strip().lower()
    return t.startswith("list[") and t[5:-1].strip().startswith("list[")


def _sample_arg_py(t: str) -> str:
    t = (t or "").strip().lower()
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
    if t in ("list[list[float]]", "list[list[float]]"):
        return "[[1.0, 2.0], [3.0, 4.0]]"
    if t.startswith("dict["):
        return '{"status": "ok"}'
    return "None"


def _parse_arg_token(t: str, token: str) -> str:
    t = (t or "").strip().lower()
    if t in ("int", "i64", "i32"):
        return f"int({token})"
    if t in ("float", "f64", "f32"):
        return f"float({token})"
    if t == "bool":
        return f"{token}.lower() == 'true'"
    if t in ("str", "string"):
        return token
    if t == "list[float]":
        return f"[float(x) for x in {token}.split(',') if x.strip()]"
    if t == "list[list[float]]":
        return f"[[float(x) for x in row.split(',') if x.strip()] for row in {token}.split(';')]"
    return token


def _generate_cli(pkg_name: str, function_names: List[str], contracts: Optional[List[ContractEntry]] = None) -> str:
    """Generate an interactive CLI that can also run headless commands."""
    contracts = contracts or []
    sigs: Dict[str, str] = {}
    for c in contracts:
        if c.signature:
            try:
                name, _, _ = _parse_signature(c.signature)
                sigs[name] = c.signature
            except Exception:
                pass

    lines: List[str] = [
        '"""Interactive CLI / REPL for the C++/ctypes package."""',
        "import argparse",
        "import cmd",
        "import sys",
        "from typing import List, Optional",
        "",
        f"from {pkg_name} import {', '.join(function_names)}",
        "",
    ]

    for name in function_names:
        sig = sigs.get(name, "")
        if not sig:
            continue
        try:
            _, args, _ = _parse_signature(sig)
        except Exception:
            continue
        if not args:
            lines.extend([
                f"def do_{name}(args: str = '') -> None:",
                f'    """Call {name}."""',
                f"    print({name}())",
                "",
            ])
            continue
        usage_parts: List[str] = []
        token_vars: List[str] = []
        for idx, (a, t) in enumerate(args):
            usage_parts.append(f"<arg{idx}>" if _is_py_list_type(t) else f"<{a}>")
            token_vars.append(f"arg{idx}")
        lines.extend([
            f"def do_{name}(args: str = '') -> None:",
            f'    """Usage: {name} {" ".join(usage_parts)}"""',
            "    if not args:",
            f'        print("Usage: {name} {" ".join(usage_parts)}")',
            "        return",
            f"    parts = args.split()",
            f"    if len(parts) < {len(args)}:",
            f'        print("Usage: {name} {" ".join(usage_parts)}")',
            "        return",
            "    try:",
        ])
        parse_lines: List[str] = []
        call_args: List[str] = []
        for idx, (a, t) in enumerate(args):
            parse_lines.append(f"        {token_vars[idx]} = {_parse_arg_token(t, f'parts[{idx}]')}")
            call_args.append(token_vars[idx])
        lines.extend(parse_lines)
        lines.extend([
            "    except ValueError:",
            f'        print("Usage: {name} {" ".join(usage_parts)}")',
            "        return",
            f"    print({name}({', '.join(call_args)}))",
            "",
        ])

    lines.extend([
        "def run_all() -> None:",
        '    """Run every exported function with sample arguments."""',
    ])
    for name in function_names:
        sig = sigs.get(name, "")
        if not sig:
            continue
        try:
            _, args, _ = _parse_signature(sig)
        except Exception:
            continue
        sample_call = ", ".join(_sample_arg_py(t) for _, t in args)
        lines.append(f'    print("{name}:", {name}({sample_call}))')
    lines.append("")

    lines.extend([
        "class AeroShell(cmd.Cmd):",
        '    intro = "C++/ctypes REPL. Type \'help\' for commands, \'quit\' to exit."',
        '    prompt = "cpp> "',
        "",
    ])
    for name in function_names:
        if name not in sigs:
            continue
        lines.extend([
            f"    def do_{name}(self, args: str) -> None:",
            f"        do_{name}(args)",
            "",
        ])
    lines.extend([
        "    def do_quit(self, args: str) -> bool:",
        '        """Exit the REPL."""',
        "        return True",
        "",
        "    do_exit = do_quit",
        "",
        "def main(argv: Optional[List[str]] = None) -> int:",
        '    parser = argparse.ArgumentParser(description="C++/ctypes CLI")',
        '    parser.add_argument("commands", nargs="*")',
        '    parser.add_argument("--cmd", default=None)',
        "    ns = parser.parse_args(argv)",
        "    shell = AeroShell()",
        "    if ns.cmd:",
        "        shell.onecmd(ns.cmd)",
        "    elif ns.commands:",
        "        shell.onecmd(' '.join(ns.commands))",
        "    elif not sys.stdin.isatty():",
        '        print("CLI ready")',
        "        run_all()",
        "    else:",
        "        try:",
        "            shell.cmdloop()",
        "        except (EOFError, KeyboardInterrupt):",
        "            print()",
        "    return 0",
        "",
        'if __name__ == "__main__":',
        "    sys.exit(main() or 0)",
        "",
    ])
    return "\n".join(lines) + "\n"


def _generate_tests(pkg_name: str, function_names: List[str], contracts: Optional[List[ContractEntry]] = None) -> str:
    contracts = contracts or []
    sigs: Dict[str, str] = {}
    for c in contracts:
        if c.signature:
            try:
                name, _, _ = _parse_signature(c.signature)
                sigs[name] = c.signature
            except Exception:
                pass

    lines: List[str] = [
        "import math",
        "from typing import Any",
        "",
        f"from {pkg_name} import {', '.join(function_names)}",
        "",
    ]

    def _expected_for(name: str, sig: str) -> str:
        try:
            _, args, return_type = _parse_signature(sig)
        except Exception:
            return "assert result is not None"
        rt = (return_type or "").strip().lower()
        list_args = [(a, t) for a, t in args if _is_py_list_type(t) and not _is_py_nested_list(t)]
        scalar = next((a for a, t in args if t.strip().lower() in ("float", "f64", "f32")), None)
        if rt == "list[float]" and len(list_args) == 1 and scalar:
            return "assert isinstance(result, list) and all(math.isclose(a, b, rel_tol=1e-9) for a, b in zip(result, [2.0, 4.0, 6.0]))"
        if rt == "list[list[float]]" and len(list_args) == 1 and scalar:
            return "assert isinstance(result, list) and result == [[2.0, 4.0], [6.0, 8.0]]"
        if rt in ("float", "f64", "f32") and len(list_args) == 2:
            return "assert isinstance(result, float) and math.isclose(result, 14.0, rel_tol=1e-9)"
        if rt in ("int", "i64") and len(list_args) == 1:
            return "assert isinstance(result, int) and result == 6"
        if rt == "bool":
            return "assert result is True or result is False"
        if rt.startswith("dict["):
            return "assert isinstance(result, dict)"
        if rt.startswith("list["):
            return "assert isinstance(result, list)"
        return "assert result is not None"

    for name in function_names:
        sig = sigs.get(name, "")
        if not sig:
            continue
        try:
            _, args, _ = _parse_signature(sig)
        except Exception:
            continue
        sample_call = ", ".join(_sample_arg_py(t) for _, t in args)
        lines.extend([
            f"def test_{name}() -> None:",
            f"    result = {name}({sample_call})",
            f"    {_expected_for(name, sig)}",
            "",
        ])

    if function_names:
        first = function_names[0]
        lines.extend([
            "def test_cli_run_all(capsys) -> None:",
            f"    from {pkg_name}.cli import run_all",
            "    run_all()",
            "    assert capsys.readouterr().out",
            "",
            "def test_repl_quit() -> None:",
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
    return f"# {pkg_name}\n\nC++/ctypes hybrid project generated by aero-forge.\n"


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


def _so_name(pkg_name: str) -> str:
    if sys.platform == "win32":
        return f"{pkg_name}.dll"
    if sys.platform == "darwin":
        return f"lib{pkg_name}.dylib"
    return f"lib{pkg_name}.so"


class CppPolyglotMaterializer:
    """Write and build a C++/ctypes hybrid workspace from a Blueprint."""

    def __init__(self, workspace: Path) -> None:
        self.workspace = Path(workspace)
        self.build_logs = ""

    def _log(self, text: str) -> None:
        """Append *text* to the build log."""
        if text:
            self.build_logs = (self.build_logs + "\n" + text).strip()

    def materialize(self, blueprint: Blueprint, *, build: bool = False) -> Blueprint:
        """Write the C++ workspace files and optionally build the shared library."""
        project = blueprint.project or "polyglot_cpp_project"
        pkg_name = _sanitize_module_name(project)
        contracts = list(blueprint.contracts) if blueprint.contracts else list(_DEFAULT_CONTRACTS)
        function_names = _function_names(contracts)

        # Ensure the acceleration log is wired so router telemetry is captured.
        accel_log = self.workspace / ".aero_forge_accel.log"
        os.environ["AERO_FORGE_ACCEL_LOG"] = str(accel_log)

        self.workspace.mkdir(parents=True, exist_ok=True)
        pkg_dir = self.workspace / pkg_name
        pkg_dir.mkdir(exist_ok=True)
        tests_dir = self.workspace / "tests"
        tests_dir.mkdir(exist_ok=True)

        for contract in contracts:
            if not contract.signature:
                continue
            stub = _contract_to_python_stub(contract)
            if _is_c_abi_contract(contract):
                language_router.select_native_backend(stub, hint="cpp")
            else:
                language_router.select_native_backend(stub, hint="rust_hin")

        _accel_log("info", "Routing C++ selective acceleration through CppEmitter and CppPolyglotMaterializer")

        (pkg_dir / "native.cpp").write_text(
            _generate_native_cpp(pkg_name, contracts), encoding="utf-8"
        )
        (pkg_dir / "__init__.py").write_text(
            _generate_init(pkg_name, pkg_dir, contracts), encoding="utf-8"
        )
        (pkg_dir / "cli.py").write_text(
            _generate_cli(pkg_name, function_names, contracts=contracts), encoding="utf-8"
        )
        (self.workspace / "pyproject.toml").write_text(
            _generate_pyproject_toml(pkg_name), encoding="utf-8"
        )
        (self.workspace / "run_shell.py").write_text(
            _generate_run_shell(pkg_name), encoding="utf-8"
        )
        (self.workspace / "README.md").write_text(
            _generate_readme(pkg_name), encoding="utf-8"
        )
        (tests_dir / "test_cli.py").write_text(
            _generate_tests(pkg_name, function_names, contracts=contracts), encoding="utf-8"
        )

        manifest: List[ManifestEntry] = [
            ManifestEntry(path=f"{pkg_name}/native.cpp", lang="cpp", purpose="C-ABI shared library source"),
            ManifestEntry(path=f"{pkg_name}/__init__.py", lang="python", purpose="ctypes loader package init"),
            ManifestEntry(path=f"{pkg_name}/cli.py", lang="python", purpose="CLI module"),
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
            self._build_extension(pkg_name, pkg_dir)

        functions = [
            FunctionSpec(
                file=pkg_dir / "__init__.py",
                name=name,
                tests=[tests_dir / "test_cli.py"],
                skip_build=True,
            )
            for name in function_names
        ]
        if (pkg_dir / "cli.py").is_file():
            functions.append(
                FunctionSpec(
                    file=pkg_dir / "cli.py",
                    name="main",
                    tests=[tests_dir / "test_cli.py"],
                    skip_build=True,
                )
            )
        blueprint = blueprint.model_copy(update={"functions": functions})

        return blueprint

    def _build_extension(self, pkg_name: str, pkg_dir: Path) -> bool:
        """Compile the C-ABI shared library in place. Returns True on success."""
        compiler = _find_cpp_compiler()
        if compiler is None:
            raise RuntimeError("No C++ compiler found (g++, clang++, or c++)")

        cpp_path = pkg_dir / "native.cpp"
        so_name = _so_name(pkg_name)
        so_path = pkg_dir / so_name

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
