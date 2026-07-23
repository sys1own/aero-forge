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
from typing import Any, Dict, List, Optional, Set, Tuple

from aero_forge.errors import UnsupportedError
from aero_forge.scaffold.engine import (
    Engine,
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

    result = subprocess.run(
        ["cargo", "build", "--target", "wasm32-unknown-unknown", "--release"],
        cwd=crate_root,
        capture_output=True,
        text=True,
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
    return node is not None and not is_class
