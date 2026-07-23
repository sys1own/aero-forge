"""Tests for project auto-detection."""

from __future__ import annotations

from pathlib import Path

from aero_forge.blueprint import discover_project


def test_discover_project(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    (src / "math.py").write_text("def add(a, b):\n    return a + b\n")
    tests = tmp_path / "tests"
    tests.mkdir()
    (tests / "test_add.py").write_text(
        "from math import add\n\ndef test_add():\n    assert add(2, 3) == 5\n"
    )

    functions = discover_project(tmp_path)
    assert len(functions) == 1
    assert functions[0].name == "add"
    assert len(functions[0].tests) >= 1


def test_discover_project_respects_aeroignore(tmp_path):
    (tmp_path / ".aeroignore").write_text("skip.py\n")
    (tmp_path / "keep.py").write_text("def keep():\n    return 1\n")
    (tmp_path / "skip.py").write_text("def skip():\n    return 2\n")

    functions = discover_project(tmp_path)
    names = {f.name for f in functions}
    assert "keep" in names
    assert "skip" not in names
