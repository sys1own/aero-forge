"""Regression tests for the stream_stats transpiler failures."""

from __future__ import annotations

import importlib.util
import subprocess
from pathlib import Path

import pytest

from aero_forge.scaffold.engine import Engine
from aero_forge.translator import TargetMode, UASTToHINTranslator, python_source_to_uast


SOURCE = """\
def stream_stats(data: list[float]) -> dict[str, float]:
    if len(data) == 0:
        return {'mean': 0.0, 'variance': 0.0, 'min': 0.0, 'max': 0.0}
    n = 0
    mean = 0.0
    M2 = 0.0
    min_val = data[0]
    max_val = data[0]
    for x in data:
        n += 1
        delta = x - mean
        mean += delta / n
        delta2 = x - mean
        M2 += delta * delta2
        if x < min_val:
            min_val = x
        if x > max_val:
            max_val = x
    variance = M2 / (n - 1) if n > 1 else 0.0
    return {'mean': mean, 'variance': variance, 'min': min_val, 'max': max_val}
"""


SOURCE_ENUMERATE = """\
def stream_stats(data: list[float]) -> dict[str, float]:
    if len(data) == 0:
        return {'mean': 0.0, 'variance': 0.0, 'min': 0.0, 'max': 0.0}
    n = 0
    mean = 0.0
    M2 = 0.0
    min_val = data[0]
    max_val = data[0]
    for i, x in enumerate(data, start=1):
        n = i
        delta = x - mean
        mean += delta / n
        delta2 = x - mean
        M2 += delta * delta2
        if x < min_val:
            min_val = x
        if x > max_val:
            max_val = x
    variance = M2 / (n - 1) if n > 1 else 0.0
    return {'mean': mean, 'variance': variance, 'min': min_val, 'max': max_val}
"""


def _compile_and_call(tmp_path: Path, source: str) -> None:
    output_dir = tmp_path / "stream_stats"
    output_dir.mkdir(parents=True, exist_ok=True)
    uast = python_source_to_uast(source)
    graph = UASTToHINTranslator().translate(uast)
    graph.traits_by_name = {}
    graph.traits = {}

    crate_root = Engine().generate(
        graph,
        output_dir / "dist",
        workspace_root=None,
        module_name="stream_stats",
        function_names=["stream_stats"],
        source=source,
        target_mode=TargetMode.PYO3,
    )

    build = subprocess.run(
        ["cargo", "build", "--release"],
        cwd=crate_root,
        capture_output=True,
        text=True,
    )
    assert build.returncode == 0, build.stderr

    so_files = list((crate_root / "target" / "release").glob("*.so"))
    assert so_files, "No .so produced by cargo build"

    spec = importlib.util.spec_from_file_location("stream_stats", so_files[0])
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    result = mod.stream_stats([1.0, 2.0, 3.0, 4.0, 5.0])
    assert result["mean"] == pytest.approx(3.0)
    assert result["variance"] == pytest.approx(2.5)
    assert result["min"] == pytest.approx(1.0)
    assert result["max"] == pytest.approx(5.0)

    empty = mod.stream_stats([])
    assert empty == {
        "mean": 0.0,
        "variance": 0.0,
        "min": 0.0,
        "max": 0.0,
    }


@pytest.mark.integration
def test_stream_stats_variance_after_loop(tmp_path: Path) -> None:
    """Variance must be computed after the loop, not hoisted to the top."""
    _compile_and_call(tmp_path, SOURCE)


@pytest.mark.integration
def test_stream_stats_enumerate_start_one(tmp_path: Path) -> None:
    """enumerate(data, start=1) must yield 1-based indices in Rust."""
    _compile_and_call(tmp_path, SOURCE_ENUMERATE)
