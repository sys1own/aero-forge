"""Tests for multi-variant generation and Pareto selection."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import pytest

from aero_forge.variants import pareto_frontier, select_best_variant


def _make_variant(
    variant: int, passed: int, total: int, elapsed: float
) -> Dict[str, Any]:
    return {
        "variant": variant,
        "source_path": "/tmp/src/generated.py",
        "test_path": "/tmp/tests/test_generated.py",
        "blueprint_path": "/tmp/blueprint.aero",
        "implementation": "def f(): pass",
        "tests": "def test_f(): pass",
        "elapsed_seconds": elapsed,
        "build": {
            "success": passed == total and total > 0,
            "passed": passed,
            "total": total,
        },
    }


def test_pareto_frontier_prefers_faster_equal_accuracy():
    results = [
        _make_variant(0, 1, 1, 2.0),
        _make_variant(1, 1, 1, 1.0),
        _make_variant(2, 1, 1, 3.0),
    ]
    front = pareto_frontier(results)
    assert len(front) == 1
    assert front[0]["variant"] == 1


def test_pareto_keeps_higher_accuracy_slower():
    results = [
        _make_variant(0, 1, 2, 1.0),  # lower accuracy but faster
        _make_variant(1, 2, 2, 2.0),  # higher accuracy, slower
    ]
    front = pareto_frontier(results)
    # Neither dominates the other (accuracy vs. speed tradeoff).
    assert len(front) == 2
    assert {r["variant"] for r in front} == {0, 1}


def test_select_best_variant_copies_files(tmp_path):
    src_dir = tmp_path / ".variant_0" / "src"
    test_dir = tmp_path / ".variant_0" / "tests"
    src_dir.mkdir(parents=True)
    test_dir.mkdir(parents=True)
    src_file = src_dir / "generated.py"
    test_file = test_dir / "test_generated.py"
    src_file.write_text("def f(): return 1")
    test_file.write_text("def test_f(): pass")

    results = [
        {
            "variant": 0,
            "source_path": str(src_file),
            "test_path": str(test_file),
            "blueprint_path": str(tmp_path / "blueprint.aero"),
            "implementation": "def f(): return 1",
            "tests": "def test_f(): pass",
            "elapsed_seconds": 1.0,
            "build": {"success": True, "passed": 1, "total": 1},
        }
    ]
    best = select_best_variant(results, output_dir=tmp_path)
    assert best["variant"] == 0
    assert (tmp_path / "src" / "generated.py").is_file()
    assert (tmp_path / "tests" / "test_generated.py").is_file()
