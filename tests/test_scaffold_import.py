"""Tests for scaffold package initialization and importability."""

import importlib
import shutil
import sys
from pathlib import Path

import pytest

from aero_forge.orchestrator.orchestrator import Orchestrator


@pytest.fixture
def nested_fibonacci_project(tmp_path):
    """Create a nested package with a Fibonacci module and tests."""
    project_root = tmp_path / "project"
    project_root.mkdir()
    (project_root / "pyproject.toml").write_text(
        "[tool.aero-forge]\n", encoding="utf-8"
    )
    pkg = project_root / "pkg"
    pkg.mkdir()
    src = pkg / "fibonacci.py"
    src.write_text(
        "def fibonacci(n):\n"
        "    if n <= 1:\n"
        "        return n\n"
        "    a, b = 0, 1\n"
        "    for _ in range(2, n + 1):\n"
        "        a, b = b, a + b\n"
        "    return b\n",
        encoding="utf-8",
    )
    test = pkg / "test_fibonacci.py"
    test.write_text(
        "from fibonacci import fibonacci\n\n"
        "def test_fibonacci():\n"
        "    assert fibonacci(0) == 0\n"
        "    assert fibonacci(10) == 55\n",
        encoding="utf-8",
    )
    return project_root, src, test


@pytest.mark.skipif(
    not shutil.which("cargo") or not shutil.which("rustc"),
    reason="Rust toolchain not installed",
)
def test_fix_creates_init_files_and_imports_nested_module(nested_fibonacci_project):
    project_root, src, test = nested_fibonacci_project
    orchestrator = Orchestrator(
        source_path=src,
        function_name="fibonacci",
        test_path=test,
        max_iterations=2,
        use_llm=False,
    )
    result = orchestrator.run()
    assert result["success"], result.get("logs", "")

    # __init__.py should be created automatically so the package is importable.
    init_file = src.parent / "__init__.py"
    assert (
        init_file.is_file()
    ), "Expected __init__.py to be created in package directory"

    # The module should be importable as a normal Python package.
    str_root = str(project_root)
    if str_root not in sys.path:
        sys.path.insert(0, str_root)
    sys.modules.pop("pkg.fibonacci", None)
    sys.modules.pop("pkg", None)

    module = importlib.import_module("pkg.fibonacci")
    assert module.fibonacci(10) == 55
    assert hasattr(module, "fibonacci")
