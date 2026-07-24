"""Integration tests for the blueprint-driven BuildRunner."""

import importlib
import shutil
import sys
from pathlib import Path
from typing import List

import pytest

from aero_forge.blueprint import parse_blueprint
from aero_forge.build_runner import BuildRunner, BuildTaskDAG
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
    source.write_text(f"# {marker}\n" "def square(n):\n" "    return n * n\n")
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


@pytest.mark.skipif(
    not shutil.which("cargo") or not shutil.which("rustc"),
    reason="Rust toolchain not installed",
)
def test_build_runner_compile_all_mixed_types(tmp_path):
    """Regression test for f64 loops, bool returns, and underscore loop vars."""
    source = tmp_path / "math_ops.py"
    test = tmp_path / "test_math_ops.py"
    source.write_text(
        "def factorial(n):\n"
        "    if n <= 1:\n"
        "        return 1\n"
        "    result = 1\n"
        "    for i in range(2, n + 1):\n"
        "        result *= i\n"
        "    return result\n"
        "\n"
        "def power(base, exp):\n"
        "    result = 1.0\n"
        "    for _ in range(exp):\n"
        "        result *= base\n"
        "    return result\n"
        "\n"
        "def is_prime(n):\n"
        "    if n < 2:\n"
        "        return False\n"
        "    i = 2\n"
        "    while i * i <= n:\n"
        "        if n % i == 0:\n"
        "            return False\n"
        "        i += 1\n"
        "    return True\n"
    )
    test.write_text(
        "from math_ops import factorial, power, is_prime\n"
        "\n"
        "def test_factorial():\n"
        "    assert factorial(0) == 1\n"
        "    assert factorial(5) == 120\n"
        "\n"
        "def test_power():\n"
        "    assert power(2, 3) == 8.0\n"
        "    assert power(3, 0) == 1.0\n"
        "\n"
        "def test_is_prime():\n"
        "    assert is_prime(2) == True\n"
        "    assert is_prime(4) == False\n"
        "    assert is_prime(17) == True\n"
    )
    blueprint_path = tmp_path / "blueprint.aero"
    blueprint_path.write_text(
        "project: mixed_types\n"
        "functions:\n"
        "  - file: math_ops.py\n"
        "    compile_all: true\n"
        "    tests: [test_math_ops.py]\n"
        "llm:\n"
        "  provider: none\n"
        "output_dir: ./dist\n"
    )

    bp = parse_blueprint(blueprint_path)
    runner = BuildRunner(bp, max_workers=1, cache_enabled=False)
    result = runner.build()

    assert result["success"] is True
    assert result["total"] == 3
    assert result["passed"] == 3

    sys.path.insert(0, str(tmp_path / "dist"))
    try:
        mod = importlib.import_module("math_ops")
        assert mod.factorial(5) == 120
        assert mod.power(2, 3) == 8.0
        assert mod.is_prime(17) is True
    finally:
        sys.path.remove(str(tmp_path / "dist"))


def test_build_runner_dry_run(tmp_path):
    source = tmp_path / "utils.py"
    source.write_text(
        "def add(a, b):\n" "    return a + b\n" "def _private():\n" "    pass\n"
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


@pytest.mark.skipif(
    not shutil.which("cargo") or not shutil.which("rustc"),
    reason="Rust toolchain not installed",
)
def test_build_runner_force_and_cache_dir(tmp_path):
    source = tmp_path / "calc.py"
    test = tmp_path / "test_calc.py"
    source.write_text("def square(n):\n    return n * n\n")
    test.write_text(
        "from calc import square\n"
        "def test_square():\n"
        "    assert square(4) == 16\n"
    )
    blueprint_path = tmp_path / "blueprint.aero"
    blueprint_path.write_text(
        "project: cache_dir_test\n"
        "functions:\n"
        "  - file: calc.py\n"
        "    name: square\n"
        "    tests: [test_calc.py]\n"
        "llm:\n"
        "  provider: none\n"
        "output_dir: ./dist\n"
    )
    cache_dir = tmp_path / "cache"

    bp = parse_blueprint(blueprint_path)
    runner1 = BuildRunner(bp, max_workers=1, cache_dir=cache_dir)
    result1 = runner1.build()
    assert result1["success"] is True
    assert result1["results"][0]["iterations"] > 0

    runner2 = BuildRunner(bp, max_workers=1, cache_dir=cache_dir)
    result2 = runner2.build()
    assert result2["results"][0]["iterations"] == 0

    runner3 = BuildRunner(bp, max_workers=1, cache_dir=cache_dir, force=True)
    result3 = runner3.build()
    assert result3["results"][0]["iterations"] > 0


@pytest.mark.skipif(
    not shutil.which("cargo") or not shutil.which("rustc"),
    reason="Rust toolchain not installed",
)
def test_build_runner_gpu_fallback_to_cpu(tmp_path):
    source = tmp_path / "gpu.py"
    test = tmp_path / "test_gpu.py"
    source.write_text("# @accelerate gpu\n" "def increment(n):\n" "    return n + 1\n")
    test.write_text(
        "from gpu import increment\n"
        "def test_increment():\n"
        "    assert increment(5) == 6\n"
    )
    blueprint_path = tmp_path / "blueprint.aero"
    blueprint_path.write_text(
        "project: gpu_fallback\n"
        "functions:\n"
        "  - file: gpu.py\n"
        "    name: increment\n"
        "    tests: [test_gpu.py]\n"
        "llm:\n"
        "  provider: none\n"
        "output_dir: ./dist\n"
    )

    bp = parse_blueprint(blueprint_path)
    runner = BuildRunner(bp, max_workers=1, gpu=True)
    result = runner.build()
    assert result["success"] is True
    assert result["passed"] == 1


@pytest.mark.skipif(
    not shutil.which("cargo") or not shutil.which("rustc"),
    reason="Rust toolchain not installed",
)
def test_build_runner_distributed(tmp_path):
    source1 = tmp_path / "a.py"
    source2 = tmp_path / "b.py"
    test1 = tmp_path / "test_a.py"
    test2 = tmp_path / "test_b.py"
    source1.write_text("def add(a, b):\n    return a + b\n")
    source2.write_text("def mul(a, b):\n    return a * b\n")
    test1.write_text("from a import add\ndef test_add():\n    assert add(2, 3) == 5\n")
    test2.write_text("from b import mul\ndef test_mul():\n    assert mul(4, 5) == 20\n")
    blueprint_path = tmp_path / "blueprint.aero"
    blueprint_path.write_text(
        "project: dist_test\n"
        "functions:\n"
        "  - file: a.py\n"
        "    name: add\n"
        "    tests: [test_a.py]\n"
        "  - file: b.py\n"
        "    name: mul\n"
        "    tests: [test_b.py]\n"
        "llm:\n"
        "  provider: none\n"
        "output_dir: ./dist\n"
    )

    bp = parse_blueprint(blueprint_path)
    runner = BuildRunner(bp, max_workers=2, distributed=True, cache_enabled=False)
    result = runner.build()

    assert result["success"] is True
    assert result["total"] == 2
    assert result["passed"] == 2


@pytest.mark.skipif(
    not shutil.which("cargo") or not shutil.which("rustc"),
    reason="Rust toolchain not installed",
)
def test_build_runner_reports_transpiler_error_details(tmp_path):
    source = tmp_path / "bad.py"
    test = tmp_path / "test_bad.py"
    source.write_text(
        "def bad():\n"
        "    with open(\"x\") as f:\n"
        "        return f.read()\n"
    )
    test.write_text("from bad import bad\n\ndef test_bad():\n    assert bad()\n")
    blueprint_path = tmp_path / "blueprint.aero"
    blueprint_path.write_text(
        "project: transpiler_error\n"
        "functions:\n"
        "  - file: bad.py\n"
        "    name: bad\n"
        "    tests: [test_bad.py]\n"
        "llm:\n"
        "  provider: none\n"
        "output_dir: ./dist\n"
    )

    bp = parse_blueprint(blueprint_path)
    runner = BuildRunner(bp, max_workers=1, cache_enabled=False)
    result = runner.build()

    assert result["success"] is False
    assert result["total"] == 1
    assert result["passed"] == 0
    assert result["failed"] == 1
    assert result["error"]
    assert "with statements" in result["error"]
    assert "general-purpose" in result["error"]
    assert result["logs"]
    assert "FileNotFoundError" in result["logs"]


def test_build_task_dag_topological_order(tmp_path: Path):
    """BuildTaskDAG should execute tasks in dependency order."""
    cache = BuildCache(root=tmp_path / "cache", enabled=False)
    dag = BuildTaskDAG(cache)
    order: List[str] = []

    def make_task(name: str):
        def task():
            order.append(name)
            return {"result": name}
        return task

    dag.add_task("a", make_task("a"), inputs=[tmp_path / "a.in"], deps=[])
    dag.add_task("b", make_task("b"), inputs=[tmp_path / "b.in"], deps=["a"])
    dag.add_task("c", make_task("c"), inputs=[tmp_path / "c.in"], deps=["a"])
    dag.add_task("d", make_task("d"), inputs=[tmp_path / "d.in"], deps=["b", "c"])

    results = dag.run()
    assert results == {"a": "a", "b": "b", "c": "c", "d": "d"}
    # Verify d comes after b and c, and b/c after a.
    assert order.index("a") < order.index("b")
    assert order.index("a") < order.index("c")
    assert order.index("b") < order.index("d")
    assert order.index("c") < order.index("d")


@pytest.mark.skipif(
    not shutil.which("cargo") or not shutil.which("rustc"),
    reason="Rust toolchain not installed",
)
def test_build_runner_dag_skips_unchanged_files(tmp_path: Path):
    """The DAG runner should bypass recompilation when source inputs are unchanged."""
    source = tmp_path / "calc.py"
    test = tmp_path / "test_calc.py"
    source.write_text("def square(n: int) -> int:\n    return n * n\n")
    test.write_text(
        "from calc import square\n"
        "def test_square():\n"
        "    assert square(4) == 16\n"
    )
    blueprint_path = tmp_path / "blueprint.aero"
    blueprint_path.write_text(
        "project: dag_cache_test\n"
        "functions:\n"
        "  - file: calc.py\n"
        "    name: square\n"
        "    tests: [test_calc.py]\n"
        "llm:\n"
        "  provider: none\n"
        "output_dir: ./dist\n"
    )

    cache_dir = tmp_path / "cache"
    bp = parse_blueprint(blueprint_path)
    runner1 = BuildRunner(bp, max_workers=1, cache_dir=cache_dir)
    result1 = runner1.build()
    assert result1["success"] is True
    assert result1["results"][0]["iterations"] > 0

    runner2 = BuildRunner(bp, max_workers=1, cache_dir=cache_dir)
    result2 = runner2.build()
    assert result2["success"] is True
    assert result2["results"][0]["iterations"] == 0
    assert result2["results"][0]["logs"] == "DAG cache hit"
