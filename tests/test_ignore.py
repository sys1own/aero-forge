"""Tests for .aeroignore pattern matching."""

from __future__ import annotations

from pathlib import Path

from aero_forge.ignore import is_ignored, parse_aeroignore


def test_parse_aeroignore(tmp_path):
    ignore_file = tmp_path / ".aeroignore"
    ignore_file.write_text("# comment\ntests/\n*.tmp\n__pycache__/\n")
    patterns = parse_aeroignore(ignore_file)
    assert patterns == ["tests/", "*.tmp", "__pycache__/"]


def test_missing_aeroignore(tmp_path):
    assert parse_aeroignore(tmp_path / ".aeroignore") == []


def test_is_ignored(tmp_path):
    root = tmp_path
    patterns = ["tests/", "*.tmp", "__pycache__/"]
    assert is_ignored(root / "tests" / "test_foo.py", patterns, root)
    assert is_ignored(root / "foo.tmp", patterns, root)
    assert is_ignored(root / "__pycache__" / "bar.pyc", patterns, root)
    assert not is_ignored(root / "src" / "foo.py", patterns, root)
