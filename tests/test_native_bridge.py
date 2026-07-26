"""Tests for the native accelerator bridge and UAST node cache."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import pytest

from aero_forge import accelerate
from aero_forge.cache.build_cache import BuildCache
from aero_forge.native_bridge import NativeAccelerator, _extract_function_uast, _source_for_function
from aero_forge.translator import python_source_to_uast


def _uast_node_for(source: str, name: str) -> Dict[str, Any]:
    uast = python_source_to_uast(source)
    for child in uast.get("children", []):
        if child.get("type") == "function_declaration" and child.get("name") == name:
            return child
    raise ValueError(f"Function {name} not found in UAST")


def test_extract_function_uast() -> None:
    source = "def add(a: float, b: float) -> float:\n    return a + b\n"
    node = _extract_function_uast(source, "add")
    assert node is not None
    assert node["name"] == "add"
    assert node["type"] == "function_declaration"


def test_node_cache_put_and_get(tmp_path: Path) -> None:
    cache = BuildCache(root=tmp_path, enabled=True)
    node = _uast_node_for("def add(a: float, b: float) -> float:\n    return a + b\n", "add")
    artifact = tmp_path / "libadd.so"
    artifact.write_bytes(b"dummy")

    stored = cache.put_node(node, "add", artifact)
    assert stored.is_file()
    assert stored.name.endswith("_libadd.so")

    found = cache.get_node(node, "add")
    assert found == stored


def test_node_cache_disabled_returns_none(tmp_path: Path) -> None:
    cache = BuildCache(root=tmp_path, enabled=False)
    node = _uast_node_for("def add(a: float, b: float) -> float:\n    return a + b\n", "add")
    assert cache.get_node(node, "add") is None


def test_accelerator_returns_wrapped_callable() -> None:
    def my_add(a: float, b: float) -> float:
        return a + b

    wrapped = accelerate()(my_add)
    assert callable(wrapped)
    assert isinstance(wrapped.__aero_accelerator__, NativeAccelerator)
    # Unsupported constructs or missing toolchains fall back to the original function.
    assert wrapped(2.0, 3.0) == 5.0


def test_accelerator_passes_through_kwargs() -> None:
    def scale(x: float, factor: float = 2.0) -> float:
        return x * factor

    wrapped = accelerate()(scale)
    assert wrapped(5.0, factor=3.0) == 15.0


@pytest.mark.slow
def test_accelerator_compiles_numeric_function(tmp_path: Path, monkeypatch) -> None:
    """End-to-end: compile a simple numeric function through the native bridge."""
    import os

    cache_root = tmp_path / "cache"
    cache_root.mkdir()
    monkeypatch.setenv("AERO_FORGE_CACHE_DIR", str(cache_root))

    @accelerate()
    def add(a: float, b: float) -> float:
        return a + b

    result = add(1.0, 2.0)
    assert result == pytest.approx(3.0)
    # A second call should hit the cache rather than recompile.
    result2 = add(10.0, 20.0)
    assert result2 == pytest.approx(30.0)


@pytest.mark.slow
def test_accelerator_cpp_compiles_and_runs(tmp_path: Path, monkeypatch) -> None:
    """End-to-end: compile a numeric loop function to C++ and call it."""
    import os

    cache_root = tmp_path / "cache"
    cache_root.mkdir()
    monkeypatch.setenv("AERO_FORGE_CACHE_DIR", str(cache_root))

    @accelerate(target="cpp")
    def sum_even(n: int) -> int:
        total = 0
        for i in range(n + 1):
            if i % 2 == 0:
                total += i
        return total

    assert sum_even(10) == 30
    # Second call should be served from cache.
    assert sum_even(100) == 2550


@pytest.mark.slow
def test_accelerator_cpp_mandelbrot(tmp_path: Path, monkeypatch) -> None:
    """A float loop with multiple assignments should compile to C++ and run."""
    import os

    cache_root = tmp_path / "cache"
    cache_root.mkdir()
    monkeypatch.setenv("AERO_FORGE_CACHE_DIR", str(cache_root))

    @accelerate(target="cpp")
    def mandelbrot(c_re: float, c_im: float, max_iter: int) -> int:
        z_re = 0.0
        z_im = 0.0
        for i in range(max_iter):
            if z_re * z_re + z_im * z_im > 4.0:
                return i
            new_re = z_re * z_re - z_im * z_im + c_re
            z_im = 2.0 * z_re * z_im + c_im
            z_re = new_re
        return max_iter

    # (-2, 0) stays bounded.
    assert mandelbrot(-2.0, 0.0, 100) == 100


def test_accelerator_cpp_falls_back_for_numpy(tmp_path: Path, monkeypatch) -> None:
    """Functions that depend on NumPy should gracefully fall back to Python."""
    import os

    cache_root = tmp_path / "cache"
    cache_root.mkdir()
    monkeypatch.setenv("AERO_FORGE_CACHE_DIR", str(cache_root))

    numpy = pytest.importorskip("numpy")

    @accelerate(target="auto")
    def vector_dot(a: numpy.ndarray, b: numpy.ndarray) -> float:
        return float(numpy.dot(a, b))

    a = numpy.array([1.0, 2.0, 3.0])
    b = numpy.array([4.0, 5.0, 6.0])
    assert vector_dot(a, b) == pytest.approx(32.0)
