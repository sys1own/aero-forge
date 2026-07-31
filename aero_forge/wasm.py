"""WASM target backend for numeric Python functions.

Compiles scalar Python functions to a ``wasm32-unknown-unknown`` shared
library with C ABI exports and generates a small JavaScript loader.
"""

from __future__ import annotations

import ast
import json
import logging
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from aero_forge.errors import UnsupportedError
from aero_forge.scaffold.cargo_runner import cargo_build, write_cargo_config
from aero_forge.scaffold.engine import (
    RustGenerator,
    _find_top_level,
    _rust_identifier,
)

logger = logging.getLogger("aero_forge.wasm")


class WasmGenerator(RustGenerator):
    """Emit a Rust function suitable for ``wasm32-unknown-unknown`` C ABI."""

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        # Use the original function name as the exported symbol.
        self.rust_function_name = self.safe_name

    def emit(self) -> str:
        block = super().emit()
        # Remove #[pyfunction(name = "...")] attribute.
        block = re.sub(
            r'^#\[pyfunction\(name = "[^"]+"\)\]\n', "", block, flags=re.MULTILINE
        )
        # Replace `fn name` with `#[no_mangle]\npub extern "C" fn name`.
        block = re.sub(
            r"^fn ", '#[no_mangle]\npub extern "C" fn ', block, flags=re.MULTILINE
        )
        return block


class WasmEngine:
    """Generate a temporary Rust crate targeting ``wasm32-unknown-unknown``."""

    def generate(
        self,
        source: str,
        function_names: List[str],
        *,
        module_name: str,
    ) -> Tuple[Path, List[WasmGenerator]]:
        """Create a temporary crate and return its root plus the generators used."""
        tree = ast.parse(source)
        class_names = {
            node.name for node in tree.body if isinstance(node, ast.ClassDef)
        }

        blocks: List[str] = []
        generators: List[WasmGenerator] = []
        for name in function_names:
            node, is_class = _find_top_level(tree, name)
            if node is None:
                raise UnsupportedError(f"Function {name!r} not found in source")
            if is_class:
                raise UnsupportedError(
                    "WASM target does not yet support Python classes"
                )
            # Traits are not wired for WASM; pass an empty traits dict.
            generator = WasmGenerator(node, module_name, {}, class_names)
            blocks.append(generator.emit())
            generators.append(generator)

        crate_name = _rust_identifier(module_name)
        crate_root = Path(tempfile.mkdtemp(prefix="aero-wasm-crate-"))
        src_dir = crate_root / "src"
        src_dir.mkdir(parents=True)

        cargo = _wasm_cargo_toml(crate_name)
        lib = "\n\n".join(blocks)

        (crate_root / "Cargo.toml").write_text(cargo, encoding="utf-8")
        (src_dir / "lib.rs").write_text(lib, encoding="utf-8")
        write_cargo_config(crate_root)

        return crate_root, generators


def _wasm_cargo_toml(crate_name: str) -> str:
    return (
        f"[package]\n"
        f'name = "{crate_name}"\n'
        f'version = "0.1.0"\n'
        f'edition = "2021"\n\n'
        f"[lib]\n"
        f'name = "{crate_name}"\n'
        f'crate-type = ["cdylib"]\n'
    )


def build_wasm_module(
    source_path: Path,
    function_names: List[str],
    output_dir: Path,
    *,
    module_name: Optional[str] = None,
    compiler_flags: Optional[List[str]] = None,
) -> Optional[Path]:
    """Compile ``function_names`` from ``source_path`` to a ``.wasm`` bundle.

    Returns the path to the generated ``.wasm`` file, or ``None`` on failure.
    """
    module_name = module_name or source_path.stem
    source = source_path.read_text(encoding="utf-8")

    crate_root, generators = WasmEngine().generate(
        source, function_names, module_name=module_name
    )

    # Add the wasm target if missing.
    _ensure_wasm_target()

    env = os.environ.copy()
    if compiler_flags:
        env["RUSTFLAGS"] = " ".join(compiler_flags)

    result = cargo_build(
        crate_root,
        release=True,
        target="wasm32-unknown-unknown",
        env=env,
    )
    if result.returncode != 0:
        logger.error("WASM build failed:\n%s", result.stderr)
        raise UnsupportedError(f"WASM build failed for {source_path}: {result.stderr}")

    wasm_artifact = (
        crate_root
        / "target"
        / "wasm32-unknown-unknown"
        / "release"
        / f"{_rust_identifier(module_name)}.wasm"
    )
    if not wasm_artifact.is_file():
        raise UnsupportedError(f"WASM artifact not found after build: {wasm_artifact}")

    output_dir.mkdir(parents=True, exist_ok=True)
    dest = output_dir / f"{module_name}.wasm"
    shutil.copy(wasm_artifact, dest)

    _write_js_loader(output_dir, module_name, generators)

    return dest


def _ensure_wasm_target() -> None:
    """Ensure the ``wasm32-unknown-unknown`` target is installed."""
    result = subprocess.run(
        ["rustup", "target", "list", "--installed"],
        capture_output=True,
        text=True,
        check=False,
    )
    if "wasm32-unknown-unknown" in result.stdout:
        return
    logger.info("Installing wasm32-unknown-unknown target")
    install = subprocess.run(
        ["rustup", "target", "add", "wasm32-unknown-unknown"],
        capture_output=True,
        text=True,
        check=False,
    )
    if install.returncode != 0:
        raise UnsupportedError(
            f"Could not install wasm32-unknown-unknown target: {install.stderr}"
        )


def _write_js_loader(
    output_dir: Path, module_name: str, generators: List[WasmGenerator]
) -> None:
    """Generate a Node-compatible CommonJS loader for the wasm module."""
    wasm_file = f"{module_name}.wasm"
    lines: List[str] = [
        "const fs = require('fs');",
        "const path = require('path');",
        "",
        "let _instance = null;",
        "",
        "async function _loadInstance() {",
        "  if (_instance) return _instance;",
        f"  const wasmPath = path.join(__dirname, '{wasm_file}');",
        "  const bytes = fs.readFileSync(wasmPath);",
        "  const result = await WebAssembly.instantiate(bytes, { env: {} });",
        "  _instance = result.instance;",
        "  return _instance;",
        "}",
        "",
        "function _toWasm(value, type) {",
        "  if (type === 'i64') return BigInt(value);",
        "  return value;",
        "}",
        "",
        "function _fromWasm(value, type) {",
        "  if (type === 'i64') return Number(value);",
        "  if (type === 'bool') return Boolean(value);",
        "  return value;",
        "}",
        "",
    ]

    for generator in generators:
        func_name = generator.orig_name
        safe_name = _rust_identifier(func_name)
        arg_list = ", ".join(generator.arg_names) if generator.arg_names else ""
        arg_conversions = (
            ", ".join(
                f"_toWasm({name}, {json.dumps(typ)})"
                for name, typ in zip(generator.arg_names, generator.arg_types)
            )
            if generator.arg_names
            else ""
        )
        return_type = generator.return_type
        lines.append(f"exports.{func_name} = async function({arg_list}) {{")
        lines.append("  const instance = await _loadInstance();")
        lines.append(f"  const raw = instance.exports.{safe_name}({arg_conversions});")
        lines.append(f"  return _fromWasm(raw, {json.dumps(return_type)});")
        lines.append("};")
        lines.append("")

    lines.append("exports.loadModule = _loadInstance;")
    (output_dir / f"{module_name}.js").write_text("\n".join(lines), encoding="utf-8")


def is_wasm_supported_function(source: str, name: str) -> bool:
    """Return True if ``name`` in ``source`` is a scalar function (not a class)."""
    tree = ast.parse(source)
    node, is_class = _find_top_level(tree, name)
    return node is not None and not is_class and _is_arithmetic_scalar_function(node)


def _is_arithmetic_scalar_function(node: ast.AST) -> bool:
    """Check whether a function only contains arithmetic on scalar arguments."""
    if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return False
    body = [s for s in node.body if not isinstance(s, ast.Expr) or not isinstance(s.value, ast.Constant)]
    if len(body) != 1 or not isinstance(body[0], ast.Return):
        return False
    return _is_arithmetic_expr(body[0].value)


def _is_arithmetic_expr(expr: ast.AST) -> bool:
    if isinstance(expr, ast.BinOp):
        return type(expr.op) in (ast.Add, ast.Sub, ast.Mult, ast.Div) and _is_arithmetic_expr(expr.left) and _is_arithmetic_expr(expr.right)
    if isinstance(expr, ast.UnaryOp):
        return type(expr.op) in (ast.UAdd, ast.USub) and _is_arithmetic_expr(expr.operand)
    if isinstance(expr, ast.Name):
        return True
    if isinstance(expr, ast.Constant):
        return isinstance(expr.value, (int, float))
    return False


_WASM_OP = {
    ast.Add: "f64.add",
    ast.Sub: "f64.sub",
    ast.Mult: "f64.mul",
    ast.Div: "f64.div",
    ast.UAdd: "",
    ast.USub: "f64.neg",
}


def _emit_wasm_expr(expr: ast.AST, locals: Dict[str, int]) -> List[str]:
    """Emit WAT instructions for an arithmetic expression."""
    if isinstance(expr, ast.Constant):
        return [f"    f64.const {float(expr.value)}"]
    if isinstance(expr, ast.Name):
        return [f"    local.get {locals[expr.id]}"]
    if isinstance(expr, ast.UnaryOp) and type(expr.op) is ast.USub:
        return _emit_wasm_expr(expr.operand, locals) + ["    f64.neg"]
    if isinstance(expr, ast.UnaryOp) and type(expr.op) is ast.UAdd:
        return _emit_wasm_expr(expr.operand, locals)
    if isinstance(expr, ast.BinOp):
        op = _WASM_OP.get(type(expr.op))
        if op is None:
            raise UnsupportedError(f"Unsupported WASM operator {type(expr.op).__name__}")
        left = _emit_wasm_expr(expr.left, locals)
        right = _emit_wasm_expr(expr.right, locals)
        return left + right + [f"    {op}"]
    raise UnsupportedError(f"Unsupported WASM expression {type(expr).__name__}")


def lower_hin_to_wat(source: str, function_name: str, *, use_f64: bool = True) -> str:
    """Lower a scalar numeric Python function to WebAssembly Text (WAT).

    The function must contain a single ``return`` statement with arithmetic on
    its parameters.  This is the WASM acceleration path for HIN numeric kernels.
    """
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == function_name:
            func = node
            break
    else:
        raise UnsupportedError(f"Function {function_name!r} not found in source")

    if not _is_arithmetic_scalar_function(func):
        raise UnsupportedError(
            f"Function {function_name!r} is not a scalar arithmetic kernel"
        )

    params = func.args.args
    param_names = [a.arg for a in params]
    local_map = {name: i for i, name in enumerate(param_names)}
    return_node = [s for s in func.body if isinstance(s, ast.Return)][0]

    param_decls = " ".join(f"(param f64)" for _ in param_names)
    expr = "\n".join(_emit_wasm_expr(return_node.value, local_map))
    return f"""(module
  (func (export "{function_name}") {param_decls} (result f64)
{expr}
    return)
)"""


def lower_hin_to_wasm(source: str, function_name: str, output_path: Path) -> Path:
    """Compile a scalar numeric Python function to a ``.wasm`` binary via WAT.

    Requires ``wat2wasm`` from the WebAssembly Binary Toolkit (wabt) to be on PATH.
    """
    wat = lower_hin_to_wat(source, function_name)
    wat2wasm = shutil.which("wat2wasm")
    if wat2wasm is None:
        raise UnsupportedError(
            "wat2wasm not found; install wabt to compile WAT to wasm"
        )
    wat_path = output_path.with_suffix(".wat")
    wat_path.write_text(wat, encoding="utf-8")
    result = subprocess.run(
        [wat2wasm, str(wat_path), "-o", str(output_path)],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise UnsupportedError(f"wat2wasm failed: {result.stderr}")
    return output_path
