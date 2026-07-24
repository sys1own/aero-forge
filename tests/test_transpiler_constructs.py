"""Tests for newly supported transpiler constructs."""

import shutil
from pathlib import Path

import pytest

from aero_forge.orchestrator.orchestrator import Orchestrator


@pytest.mark.skipif(
    not shutil.which("cargo") or not shutil.which("rustc"),
    reason="Rust toolchain not installed",
)
def _run_case(tmp_path: Path, name: str, source: str, test_code: str) -> dict:
    src = tmp_path / f"{name}.py"
    src.write_text(source)
    test = tmp_path / f"test_{name}.py"
    test.write_text(f"from {name} import {name}\n\n{test_code}")
    orchestrator = Orchestrator(
        src,
        function_name=name,
        test_path=test,
        max_iterations=2,
        use_llm=False,
    )
    result = orchestrator.run()
    assert result["success"], result.get("error", "")
    return result


@pytest.mark.skipif(
    not shutil.which("cargo") or not shutil.which("rustc"),
    reason="Rust toolchain not installed",
)
def test_list_comprehension_over_range(tmp_path: Path) -> None:
    source = (
        "def list_comp(n: int) -> list[int]:\n" "    return [x * x for x in range(n)]\n"
    )
    _run_case(
        tmp_path,
        "list_comp",
        source,
        "def test_list_comp():\n    assert list_comp(5) == [0, 1, 4, 9, 16]\n",
    )


@pytest.mark.skipif(
    not shutil.which("cargo") or not shutil.which("rustc"),
    reason="Rust toolchain not installed",
)
def test_list_comprehension_over_list(tmp_path: Path) -> None:
    source = (
        "def double_list(arr: list[int]) -> list[int]:\n"
        "    return [x * 2 for x in arr]\n"
    )
    _run_case(
        tmp_path,
        "double_list",
        source,
        "def test_double_list():\n    assert double_list([1, 2, 3]) == [2, 4, 6]\n",
    )


@pytest.mark.skipif(
    not shutil.which("cargo") or not shutil.which("rustc"),
    reason="Rust toolchain not installed",
)
def test_nested_list_comprehension_matrix(tmp_path: Path) -> None:
    source = (
        "def zero_matrix(rows: int, cols: int) -> list[list[int]]:\n"
        "    return [[0 for _ in range(cols)] for _ in range(rows)]\n"
    )
    _run_case(
        tmp_path,
        "zero_matrix",
        source,
        "def test_zero_matrix():\n    assert zero_matrix(2, 3) == [[0, 0, 0], [0, 0, 0]]\n",
    )


@pytest.mark.skipif(
    not shutil.which("cargo") or not shutil.which("rustc"),
    reason="Rust toolchain not installed",
)
def test_tuple_unpacking_in_while(tmp_path: Path) -> None:
    source = (
        "def fib(n: int) -> int:\n"
        "    a, b = 0, 1\n"
        "    while n > 0:\n"
        "        a, b = b, a + b\n"
        "        n -= 1\n"
        "    return a\n"
    )
    _run_case(
        tmp_path,
        "fib",
        source,
        "def test_fib():\n    assert fib(10) == 55\n",
    )


@pytest.mark.skipif(
    not shutil.which("cargo") or not shutil.which("rustc"),
    reason="Rust toolchain not installed",
)
def test_enumerate_loop(tmp_path: Path) -> None:
    source = (
        "def sum_enum(arr: list[int]) -> int:\n"
        "    total = 0\n"
        "    for i, x in enumerate(arr):\n"
        "        total += i + x\n"
        "    return total\n"
    )
    _run_case(
        tmp_path,
        "sum_enum",
        source,
        "def test_sum_enum():\n    assert sum_enum([10, 20, 30]) == 63\n",
    )


@pytest.mark.skipif(
    not shutil.which("cargo") or not shutil.which("rustc"),
    reason="Rust toolchain not installed",
)
def test_zip_loop(tmp_path: Path) -> None:
    source = (
        "def sum_pairs(a: list[int], b: list[int]) -> int:\n"
        "    total = 0\n"
        "    for x, y in zip(a, b):\n"
        "        total += x * y\n"
        "    return total\n"
    )
    _run_case(
        tmp_path,
        "sum_pairs",
        source,
        "def test_sum_pairs():\n    assert sum_pairs([1, 2, 3], [4, 5, 6]) == 32\n",
    )


@pytest.mark.skipif(
    not shutil.which("cargo") or not shutil.which("rustc"),
    reason="Rust toolchain not installed",
)
def test_list_concatenation(tmp_path: Path) -> None:
    source = (
        "def concat(a: list[int], b: list[int]) -> list[int]:\n" "    return a + b\n"
    )
    _run_case(
        tmp_path,
        "concat",
        source,
        "def test_concat():\n    assert concat([1, 2], [3, 4]) == [1, 2, 3, 4]\n",
    )


@pytest.mark.skipif(
    not shutil.which("cargo") or not shutil.which("rustc"),
    reason="Rust toolchain not installed",
)
def test_complex_literal_rejected(tmp_path: Path) -> None:
    source = "def bad(x: int) -> int:\n    return x + 1j\n"
    src = tmp_path / "bad.py"
    src.write_text(source)
    test = tmp_path / "test_bad.py"
    test.write_text("from bad import bad\ndef test_bad():\n    assert bad(0) == 1\n")
    orchestrator = Orchestrator(
        src,
        function_name="bad",
        test_path=test,
        max_iterations=1,
        use_llm=False,
    )
    result = orchestrator.run()
    assert not result["success"]
    assert "Complex numbers are not supported" in result.get("error", "")
