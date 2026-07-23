"""Tests for the WASM target backend."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from aero_forge.blueprint import parse_blueprint
from aero_forge.build_runner import BuildRunner
from aero_forge.wasm import build_wasm_module


@pytest.mark.skipif(
    not shutil.which("cargo") or not shutil.which("rustc"),
    reason="Rust toolchain not installed",
)
def test_build_wasm_module_scalar_function(tmp_path):
    source = tmp_path / "math.py"
    source.write_text(
        "def square(n):\n"
        "    return n * n\n"
        "\n"
        "def double(x):\n"
        "    return x * 2\n"
    )
    output_dir = tmp_path / "dist"

    wasm_path = build_wasm_module(
        source, ["square", "double"], output_dir, module_name="math"
    )

    assert wasm_path is not None
    assert wasm_path.name == "math.wasm"
    js_path = output_dir / "math.js"
    assert js_path.is_file()

    # Node can load the generated CommonJS module and call the wasm exports.
    node_script = f"""
const math = require('{js_path}');
Promise.all([math.square(7n), math.double(5n)]).then(([a, b]) => {{
  if (a !== 49) throw new Error(`square(7) = ${{a}}`);
  if (b !== 10) throw new Error(`double(5) = ${{b}}`);
  console.log('ok');
}}).catch(err => {{ console.error(err); process.exit(1); }});
"""
    result = subprocess.run(
        [shutil.which("node") or "node", "-e", node_script],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout


@pytest.mark.skipif(
    not shutil.which("cargo") or not shutil.which("rustc"),
    reason="Rust toolchain not installed",
)
def test_build_runner_wasm_target(tmp_path):
    source = tmp_path / "fib.py"
    source.write_text(
        "def fib(n):\n"
        "    if n <= 1:\n"
        "        return n\n"
        "    return fib(n - 1) + fib(n - 2)\n"
    )
    blueprint_path = tmp_path / "blueprint.aero"
    blueprint_path.write_text(
        "project: wasm_test\n"
        "functions:\n"
        "  - file: fib.py\n"
        "    name: fib\n"
        "llm:\n"
        "  provider: none\n"
        "output_dir: ./dist\n"
    )

    bp = parse_blueprint(blueprint_path)
    runner = BuildRunner(bp, max_workers=1, target="wasm32-unknown-unknown")
    result = runner.build()

    assert result["success"] is True
    assert result["passed"] == 1
    wasm_file = tmp_path / "dist" / "fib.wasm"
    assert wasm_file.is_file()
    js_file = tmp_path / "dist" / "fib.js"
    assert js_file.is_file()

    # Verify the wasm loads and computes fib(10) == 55.
    node_script = f"""
const {{ fib }} = require('{js_file}');
fib(10n).then(v => {{
  if (v !== 55) throw new Error(`fib(10) = ${{v}}`);
  console.log('ok');
}}).catch(err => {{ console.error(err); process.exit(1); }});
"""
    res = subprocess.run(
        [shutil.which("node") or "node", "-e", node_script],
        capture_output=True,
        text=True,
        check=False,
    )
    assert res.returncode == 0, res.stderr
    assert "ok" in res.stdout
