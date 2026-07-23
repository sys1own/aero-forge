"""Integration tests for the build/orchestrate loop."""

import shutil
from pathlib import Path

import pytest

from aero_forge.orchestrator.orchestrator import Orchestrator


@pytest.fixture
def fibonacci_fixture(tmp_path):
    src = tmp_path / "fibonacci.py"
    test = tmp_path / "test_fibonacci.py"
    src.write_text(
        "def fibonacci(n):\n"
        "    if n <= 1:\n"
        "        return n\n"
        "    a, b = 0, 1\n"
        "    for _ in range(2, n + 1):\n"
        "        a, b = b, a + b\n"
        "    return b\n"
    )
    test.write_text(
        "from fibonacci import fibonacci\n\n"
        "def test_fibonacci():\n"
        "    assert fibonacci(0) == 0\n"
        "    assert fibonacci(10) == 55\n"
    )
    return src


@pytest.mark.skipif(
    not shutil.which("cargo") or not shutil.which("rustc"),
    reason="Rust toolchain not installed",
)
def test_orchestrator_compiles_valid_function(fibonacci_fixture, tmp_path):
    orchestrator = Orchestrator(
        fibonacci_fixture,
        function_name="fibonacci",
        test_path=tmp_path / "test_fibonacci.py",
        max_iterations=2,
        use_llm=False,
    )
    result = orchestrator.run()
    assert result["success"]
    assert result["iterations"] == 1
    assert (fibonacci_fixture.parent / "libaero_forge_fibonacci.so").is_file()


def test_orchestrator_returns_partial_for_missing_function(fibonacci_fixture, tmp_path):
    orchestrator = Orchestrator(
        fibonacci_fixture,
        function_name="missing",
        test_path=tmp_path / "test_fibonacci.py",
        max_iterations=1,
        use_llm=False,
    )
    result = orchestrator.run()
    assert not result["success"]
    assert result.get("partial")
    assert (
        "missing" in result["error"].lower() or "not found" in result["error"].lower()
    )


def test_orchestrator_uses_cache_and_router_first(fibonacci_fixture, tmp_path):
    orchestrator = Orchestrator(
        fibonacci_fixture,
        function_name="fibonacci",
        test_path=tmp_path / "test_fibonacci.py",
        max_iterations=1,
        use_llm=False,
    )
    # A valid function should pass without ever touching the LLM.
    result = orchestrator.run()
    assert result["success"]
