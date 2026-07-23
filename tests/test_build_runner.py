"""Integration tests for the blueprint-driven BuildRunner."""

import importlib
import shutil
import sys
from pathlib import Path

import pytest

from aero_forge.blueprint import parse_blueprint
from aero_forge.build_runner import BuildRunner
from aero_forge.cache.build_cache import BuildCache


@pytest.mark.skipif(
    not shutil.which("cargo") or not shutil.which("rustc"),
    reason="Rust toolchain not installed",
)
def test_build_runner_compiles_multiple_functions(tmp_path):
    source = tmp_path / "compute.py"
    test = tmp_path / "test_compute.py"
    source.write_text(
        "def factorial(n):\n"
        "    if n <= 1:\n"
        "        return 1\n"
        "    result = 1\n"
        "    for i in range(2, n + 1):\n"
        "        result *= i\n"
        "    return result\n"
        "\n"
        "def double(x):\n"
        "    return x * 2\n"
    )
    test.write_text(
        "from compute import factorial, double\n"
        "\n"
        "def test_compute():\n"
        "    assert factorial(5) == 120\n"
        "    assert double(7) == 14\n"
    )
    blueprint_path = tmp_path / "blueprint.aero"
    blueprint_path.write_text(
        "project: test_build\n"
        "functions:\n"
        "  - file: compute.py\n"
        "    name: factorial\n"
        "    tests: [test_compute.py]\n"
        "  - file: compute.py\n"
        "    name: double\n"
        "    tests: [test_compute.py]\n"
        "llm:\n"
        "  provider: none\n"
        "output_dir: ./dist\n"
    )

    bp = parse_blueprint(blueprint_path)
    runner = BuildRunner(bp, max_workers=1, cache_enabled=False)
    result = runner.build()

    assert result["success"] is True
    assert result["total"] == 2
    assert result["passed"] == 2
    so_files = list((tmp_path / "dist").glob("*.so"))
    assert so_files
    assert (tmp_path / "dist" / "compute.py").is_file()

    sys.path.insert(0, str(tmp_path / "dist"))
    try:
        mod = importlib.import_module("compute")
        assert mod.factorial(5) == 120
        assert mod.double(7) == 14
    finally:
        sys.path.remove(str(tmp_path / "dist"))


@pytest.mark.skipif(
    not shutil.which("cargo") or not shutil.which("rustc"),
    reason="Rust toolchain not installed",
)
def test_build_runner_caches_results(tmp_path):
    source = tmp_path / "calc.py"
    test = tmp_path / "test_calc.py"
    marker = tmp_path.name
    source.write_text(
        f"# {marker}\n"
        "def square(n):\n"
        "    return n * n\n"
    )
    test.write_text(
        "from calc import square\n"
        "def test_square():\n"
        "    assert square(4) == 16\n"
    )
    blueprint_path = tmp_path / "blueprint.aero"
    blueprint_path.write_text(
        "project: cache_test\n"
        "functions:\n"
        "  - file: calc.py\n"
        "    name: square\n"
        "    tests: [test_calc.py]\n"
        "llm:\n"
        "  provider: none\n"
        "output_dir: ./dist\n"
    )

    BuildCache().clear()

    bp = parse_blueprint(blueprint_path)
    runner1 = BuildRunner(bp, max_workers=1, cache_enabled=True)
    result1 = runner1.build()
    assert result1["success"] is True
    assert result1["results"][0]["iterations"] > 0

    runner2 = BuildRunner(bp, max_workers=1, cache_enabled=True)
    result2 = runner2.build()
    assert result2["success"] is True
    assert result2["results"][0]["iterations"] == 0


@pytest.mark.skipif(
    not shutil.which("cargo") or not shutil.which("rustc"),
    reason="Rust toolchain not installed",
)
def test_build_runner_compile_all(tmp_path):
    source = tmp_path / "utils.py"
    test = tmp_path / "test_utils.py"
    source.write_text(
        "def add(a, b):\n"
        "    return a + b\n"
        "def mul(a, b):\n"
        "    return a * b\n"
        "def _private():\n"
        "    pass\n"
    )
    test.write_text(
        "from utils import add, mul\n"
        "def test_utils():\n"
        "    assert add(2, 3) == 5\n"
        "    assert mul(4, 5) == 20\n"
    )
    blueprint_path = tmp_path / "blueprint.aero"
    blueprint_path.write_text(
        "project: all_test\n"
        "functions:\n"
        "  - file: utils.py\n"
        '    name: "*"\n'
        "    tests: [test_utils.py]\n"
        "llm:\n"
        "  provider: none\n"
        "output_dir: ./dist\n"
    )

    bp = parse_blueprint(blueprint_path)
    runner = BuildRunner(bp, max_workers=1, cache_enabled=False)
    result = runner.build()

    assert result["success"] is True
    assert result["total"] == 2
    assert result["passed"] == 2


def test_build_runner_dry_run(tmp_path):
    source = tmp_path / "utils.py"
    source.write_text(
        "def add(a, b):\n"
        "    return a + b\n"
        "def _private():\n"
        "    pass\n"
    )
    blueprint_path = tmp_path / "blueprint.aero"
    blueprint_path.write_text(
        "project: dry\n"
        "functions:\n"
        "  - file: utils.py\n"
        "    compile_all: true\n"
        "output_dir: ./dist\n"
    )

    bp = parse_blueprint(blueprint_path)
    runner = BuildRunner(bp, max_workers=1, dry_run=True)
    result = runner.build()

    assert result["dry_run"] is True
    assert result["total"] == 1
    assert result["results"][0]["functions"] == ["add"]
    assert not (tmp_path / "dist").exists()
