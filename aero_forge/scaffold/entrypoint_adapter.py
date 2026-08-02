"""Entrypoint adapter engine for generating root-level CLI wrappers."""

import ast
import os
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


class EntrypointAdapterEngine:
    """Generates CLI parsing code and entrypoint wrappers for root project execution.

    The wrapper is driven entirely by ``execution_strategy.primary_entrypoint`` and
    ``execution_strategy.cli_contract``.  When contract/ABI metadata is supplied,
    a matching ``engine.py`` dispatch module is also emitted so that the generated
    ``main.py`` is immediately executable.
    """

    def __init__(
        self,
        execution_strategy: Dict[str, Any],
        output_dir: str,
        contracts: Optional[List[Any]] = None,
        abi_contracts: Optional[List[Any]] = None,
        function_module: Optional[str] = None,
    ):
        self.strategy = execution_strategy
        self.output_dir = Path(output_dir)
        self.contracts = contracts or []
        self.abi_contracts = abi_contracts or []
        self.function_module = function_module

    def synthesize_root_entrypoint(self) -> str:
        """Write the primary entrypoint file and companion engine module."""
        primary = self.strategy.get("primary_entrypoint") or {}
        cli_contract = self.strategy.get("cli_contract") or {}

        runtime = primary.get("runtime", "python3")
        if runtime != "python3":
            # Non-Python primary entrypoints (e.g. Rust native binaries) must be
            # built by their own toolchain; no Python CLI wrapper is required.
            return ""

        entrypoint_path = Path(primary.get("path", "main.py"))
        if entrypoint_path.is_absolute():
            target_path = entrypoint_path
        else:
            target_path = self.output_dir / entrypoint_path
        target_path.parent.mkdir(parents=True, exist_ok=True)

        engine_path, engine_module = self._prepare_engine_module(target_path)
        code = self._generate_python_cli_wrapper(target_path.name, cli_contract, engine_module)
        target_path.write_text(code, encoding="utf-8")

        if engine_path is not None and self._should_write_engine(engine_path):
            engine_code = self._generate_engine_py(engine_module)
            engine_path.write_text(engine_code, encoding="utf-8")

        return str(target_path)

    # --------------------------------------------------------------------- #
    # Flag normalisation helpers
    # --------------------------------------------------------------------- #
    @staticmethod
    def _flag_get(flag: Dict[str, Any], *keys: str, default: Any = None) -> Any:
        for key in keys:
            if key in flag:
                return flag[key]
        return default

    def _normalise_flag(self, flag: Dict[str, Any]) -> Dict[str, Any]:
        """Accept a range of LLM field names and produce a canonical flag dict."""
        name = (
            self._flag_get(flag, "name", "long_flag", "long", default="").strip().lstrip("-")
        )
        short = self._flag_get(flag, "short", "short_flag", default="").strip().lstrip("-")
        dest_var = (
            self._flag_get(flag, "dest_var", "dest", "destination", default="").strip()
        )
        if not dest_var:
            dest_var = name.replace("-", "_")
        ftype = self._flag_get(flag, "type", "dtype", default="string").strip().lower()
        if ftype in ("json_str", "json", "json_string"):
            # These are string-typed flags that accept JSON text.
            ftype = "string"
        return {
            "name": name,
            "short": short,
            "dest_var": dest_var,
            "type": ftype,
            "required": self._flag_get(flag, "required", default=False),
            "default": self._flag_get(flag, "default", "default_value", default=None),
            "choices": self._flag_get(flag, "choices", "options", default=None) or [],
            "help": self._flag_get(flag, "help", "description", default=""),
        }

    @staticmethod
    def _is_valid_identifier(s: str) -> bool:
        if not s:
            return False
        return s.isidentifier()

    # --------------------------------------------------------------------- #
    # Engine module discovery / emission
    # --------------------------------------------------------------------- #
    def _prepare_engine_module(
        self, target_path: Path
    ) -> Tuple[Optional[Path], str]:
        """Return the engine file path and the import string for ``main.py``."""
        target_parent = target_path.parent
        sibling = target_parent / "engine.py"

        # Prefer an existing engine.py (sibling first), otherwise the sibling path
        # where we will generate a placeholder.
        existing = sorted(p for p in self.output_dir.rglob("engine.py") if p.is_file())
        if sibling in existing:
            engine_path = sibling
        elif existing:
            engine_path = existing[0]
        else:
            engine_path = sibling

        if (
            engine_path.parent == target_parent
            and (target_parent / "__init__.py").exists()
        ):
            # Main and engine are in the same package; use a relative import so
            # ``python -m package.main`` works.
            engine_module = ".engine"
        elif (
            engine_path.parent == self.output_dir
            and not (self.output_dir / "__init__.py").exists()
        ):
            # Both files in a non-package root; plain ``engine`` is importable.
            engine_module = "engine"
        else:
            engine_module = self._dotted_module(self.output_dir, engine_path)

        return engine_path, engine_module

    def _should_write_engine(self, engine_path: Path) -> bool:
        """Write the engine module unless a real one already exists."""
        if not engine_path.exists():
            return True
        # If an engine exists but is empty/placeholder, overwrite it.
        text = engine_path.read_text(encoding="utf-8")
        stripped = text.strip()
        if not stripped or "pass" in stripped.splitlines()[0] or "placeholder" in stripped.lower():
            return True
        return False

    @staticmethod
    def _dotted_module(base_dir: Path, module_path: Path) -> str:
        rel = module_path.with_suffix("").relative_to(base_dir)
        return ".".join(rel.parts)

    # --------------------------------------------------------------------- #
    # main.py generation
    # --------------------------------------------------------------------- #
    def _generate_python_cli_wrapper(
        self,
        entrypoint_filename: str,
        cli_contract: Dict[str, Any],
        engine_module: str,
    ) -> str:
        raw_flags = cli_contract.get("flags", [])
        flags = [self._normalise_flag(f) for f in raw_flags]
        flags = [f for f in flags if self._is_valid_identifier(f["name"])]

        # Ensure there is always a dispatchable --cmd flag.
        if not any(f["name"] == "cmd" for f in flags):
            func_names = self._function_names()
            cmd_choices = ["run_all", "benchmark"] + func_names
            flags.insert(
                0,
                {
                    "name": "cmd",
                    "short": "c",
                    "dest_var": "cmd",
                    "type": "string",
                    "required": False,
                    "default": "run_all" if func_names else None,
                    "choices": cmd_choices,
                    "help": "Command to execute",
                },
            )

        lines = [
            "# Auto-generated entrypoint wrapper produced by Aero-Forge Engine Core",
            "import argparse",
            "import sys",
            f"from {engine_module} import run_domain_task",
            "",
            "def main(*main_args, **main_kwargs):",
            "    parser = argparse.ArgumentParser(description='Aero-Forge Executable Pipeline')",
        ]

        for flag in flags:
            short_flag = f"-{flag['short']}" if flag["short"] else ""
            long_flag = f"--{flag['name']}"
            flag_args = [f"{long_flag!r}"]
            if short_flag:
                flag_args.insert(0, f"{short_flag!r}")
            flag_str = ", ".join(flag_args)

            if flag["type"] == "bool":
                lines.append(
                    f"    parser.add_argument({flag_str}, dest={flag['dest_var']!r}, "
                    f"action='store_true', help={flag['help']!r})"
                )
            else:
                arg_type = {
                    "int": "int",
                    "i32": "int",
                    "i64": "int",
                    "u32": "int",
                    "usize": "int",
                    "float": "float",
                    "f32": "float",
                    "f64": "float",
                    "double": "float",
                }.get(flag["type"], "str")
                extras = []
                if flag["choices"]:
                    extras.append(f"choices={flag['choices']!r}")
                if flag["default"] is not None:
                    extras.append(f"default={flag['default']!r}")
                if flag["required"]:
                    extras.append("required=True")
                extras.append(f"help={flag['help']!r}")
                extras.append(f"type={arg_type}")
                lines.append(
                    f"    parser.add_argument({flag_str}, dest={flag['dest_var']!r}, "
                    f"{', '.join(extras)})"
                )

        lines.extend(
            [
                "    args = parser.parse_args()",
                "    status = run_domain_task(args, *main_args, **main_kwargs)",
                "    sys.exit(status if isinstance(status, int) else 0)",
                "",
                "if __name__ == '__main__':",
                "    main()",
                "",
            ]
        )

        return "\n".join(lines)

    # --------------------------------------------------------------------- #
    # engine.py generation
    # --------------------------------------------------------------------- #
    @staticmethod
    def _contract_field(contract: Any, *names: str) -> Any:
        if isinstance(contract, dict):
            for n in names:
                if n in contract:
                    return contract[n]
            return None
        for n in names:
            val = getattr(contract, n, None)
            if val is not None:
                return val
        return None

    def _function_names(self) -> List[str]:
        names = []
        for contract in self.contracts:
            sig = self._contract_field(contract, "signature")
            if sig:
                try:
                    name, _, _ = _parse_signature(sig)
                    names.append(name)
                except Exception:
                    pass
        for abi in self.abi_contracts:
            symbol = self._contract_field(abi, "export_symbol", "contract_id")
            if symbol:
                names.append(symbol)
        return list(dict.fromkeys(names))

    def _default_function_module(self, target_path: Path) -> str:
        if self.function_module:
            return self.function_module
        # Derive the package module from the main.py path.
        parent = target_path.parent
        if parent == self.output_dir:
            return ""
        rel = parent.relative_to(self.output_dir)
        return ".".join(rel.parts)

    def _generate_engine_py(self, engine_module: str) -> str:
        funcs = self._function_names()
        func_specs = []
        for contract in self.contracts:
            sig = self._contract_field(contract, "signature")
            if not sig:
                continue
            try:
                name, args, return_type = _parse_signature(sig)
                func_specs.append((name, args, return_type))
            except Exception:
                pass
        if not func_specs:
            func_specs = [(name, [], "None") for name in funcs]

        # Build a kwargs lookup for each function based on its typed arguments.
        dispatch: List[List[str]] = []
        for name, args, _ in func_specs:
            arg_lines = []
            for arg_name, arg_type in args:
                if arg_name == "self":
                    continue
                sample = _sample_arg(arg_type)
                arg_lines.append(
                    f"    '{arg_name}': getattr(args, '{arg_name}', None) if "
                    f"getattr(args, '{arg_name}', None) is not None else {sample},"
                )
            if arg_lines:
                dispatch.append([
                    f"if cmd == {name!r}:",
                    "    kwargs = {",
                    *arg_lines,
                    "    }",
                    f"    print(funcs['{name}'](**kwargs))",
                    "    return 0",
                ])
            else:
                dispatch.append([
                    f"if cmd == {name!r}:",
                    f"    print(funcs['{name}']())",
                    "    return 0",
                ])

        run_all_calls: List[List[str]] = []
        for name, args, _ in func_specs:
            arg_names = [a for a, _ in args if a != "self"]
            call_args = ", ".join(
                f"{a}=getattr(args, {a!r}, None) if getattr(args, {a!r}, None) is not None else {_sample_arg(t)}"
                for a, t in args if a != "self"
            )
            if arg_names:
                call = f"funcs['{name}']({call_args})"
            else:
                call = f"funcs['{name}']()"
            run_all_calls.append([
                f"try:",
                f"    print('{name}:', {call})",
                f"except Exception as _e:",
                f"    print('{name} failed:', _e)",
            ])

        benchmark_calls: List[List[str]] = []
        for name, args, _ in func_specs:
            arg_names = [a for a, _ in args if a != "self"]
            sample_args = ", ".join(
                f"{a}={_sample_arg(t)}" for a, t in args if a != "self"
            )
            if arg_names:
                benchmark_calls.append([
                    f"for _ in range(count):",
                    f"    funcs['{name}']({sample_args})",
                    f"print('{name}: benchmarked', count, 'iterations')",
                ])
            else:
                benchmark_calls.append([
                    f"for _ in range(count):",
                    f"    funcs['{name}']()",
                    f"print('{name}: benchmarked', count, 'iterations')",
                ])

        function_module = self.function_module or ""
        func_names = [f[0] for f in func_specs]

        lines = [
            "# Auto-generated engine module produced by Aero-Forge Engine Core",
            "import importlib",
            "import sys",
            f"_FUNCTION_MODULE = {function_module!r}",
            "",
            "def _load_functions():",
            f"    names = {func_names!r}",
            "    if not _FUNCTION_MODULE:",
            "        return {}",
            "    try:",
            "        mod = importlib.import_module(_FUNCTION_MODULE)",
            "    except Exception as exc:",
            '        print(f"Could not import functions module {_FUNCTION_MODULE}: {exc}", file=sys.stderr)',
            "        return {}",
            "    return {n: getattr(mod, n) for n in names if hasattr(mod, n)}",
            "",
            "def run_domain_task(args, *extra_args, **kwargs):",
            "    funcs = _load_functions()",
            '    cmd = getattr(args, "cmd", None)',
        ]

        if dispatch:
            lines.append("    if cmd and cmd in funcs:")
            for block in dispatch:
                for line in block:
                    lines.append("        " + line)

        lines.extend([
            '    if cmd == "benchmark":',
            '        count = getattr(args, "iterations", None) or getattr(args, "simulations", None) or 1000',
            "        if not funcs:",
            '            print("No functions available for benchmark")',
            "            return 0",
        ])
        if benchmark_calls:
            for block in benchmark_calls:
                for line in block:
                    lines.append("        " + line)
        else:
            lines.append("        pass")

        lines.extend([
            "",
            "    # Default: run all available functions",
            "    if not funcs:",
            '        print(f"Aero-Forge CLI ready: cmd={cmd} args={args}")',
            "        return 0",
        ])
        for block in run_all_calls:
            for line in block:
                lines.append("    " + line)
        if not run_all_calls:
            lines.append('    print("No functions defined")')
        lines.append("    return 0")

        return "\n".join(lines) + "\n"


# ------------------------------------------------------------------------- #
# Signature / argument helpers (mirroring polyglot materializer utilities)
# ------------------------------------------------------------------------- #
def _annotation_to_str(node: Optional[ast.AST]) -> str:
    if node is None:
        return "None"
    try:
        return ast.unparse(node)
    except Exception:
        return "Any"


def _parse_signature(signature: str) -> Tuple[str, List[Tuple[str, str]], str]:
    """Parse a Python-style signature into (function_name, [(arg, type)], return_type)."""
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


def _sample_arg(py_type: str) -> str:
    t = py_type.lower().replace(" ", "")
    if "list[list" in t or "list[list[float]]" in t:
        return "[[1.0, 2.0], [3.0, 4.0]]"
    if "list" in t and "float" in t:
        return "[1.0, 2.0, 3.0]"
    if "list" in t and ("int" in t or "i64" in t or "i32" in t):
        return "[1, 2, 3]"
    if "list" in t and "str" in t:
        return '["a", "b"]'
    if "list" in t:
        return "[1.0, 2.0]"
    if "dict" in t:
        return '{"status": "ok"}'
    if t in ("int", "i64", "i32", "u32", "usize"):
        return "1"
    if t in ("float", "f64", "f32", "double"):
        return "1.0"
    if t == "bool":
        return "True"
    if t == "str":
        return '"x"'
    return "None"
