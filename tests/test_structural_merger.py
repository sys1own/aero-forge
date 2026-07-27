"""Tests for aero_forge.healing.structural_merger."""

import pytest

from aero_forge.healing.structural_merger import MergeConflictError, apply_overlay, structural_merge


def test_python_function_overlay():
    base = (
        "def add(a, b):\n"
        "    return a + b\n"
        "\n"
        "def mul(a, b):\n"
        "    return a * b\n"
    )
    overlay = (
        "def add(a, b):\n"
        "    return a - b\n"
    )
    result = apply_overlay(base, overlay, language="python")
    assert "def add(a, b):" in result
    assert "return a - b" in result
    assert "def mul(a, b):" in result
    assert "return a * b" in result


def test_python_structural_merge_adds_new_function():
    base = (
        "def add(a, b):\n"
        "    return a + b\n"
    )
    left = (
        "def add(a, b):\n"
        "    return a + b\n"
        "\n"
        "def sub(a, b):\n"
        "    return a - b\n"
    )
    right = base
    overlay = structural_merge(base, left, right, language="python")
    assert overlay.conflicts == []
    assert "def sub(a, b):" in overlay.source


def test_python_merge_detects_conflict():
    base = (
        "def add(a, b):\n"
        "    return a + b\n"
    )
    left = (
        "def add(a, b):\n"
        "    return a - b\n"
    )
    right = (
        "def add(a, b):\n"
        "    return a * b\n"
    )
    overlay = structural_merge(base, left, right, language="python")
    assert overlay.conflicts


def test_python_syntax_error_rejected():
    base = "def add(a, b):\n    return a + b\n"
    overlay = "def broken(\n"  # invalid
    with pytest.raises(MergeConflictError):
        apply_overlay(base, overlay, language="python")


def test_rust_function_overlay():
    base = (
        "fn add(a: i64, b: i64) -> i64 {\n"
        "    a + b\n"
        "}\n"
        "\n"
        "fn mul(a: i64, b: i64) -> i64 {\n"
        "    a * b\n"
        "}\n"
    )
    overlay = (
        "fn add(a: i64, b: i64) -> i64 {\n"
        "    a - b\n"
        "}\n"
    )
    result = apply_overlay(base, overlay, language="rust")
    assert "fn add" in result
    assert "a - b" in result
    assert "fn mul" in result


def test_rust_merge_adds_struct():
    base = "fn add(a: i64, b: i64) -> i64 { a + b }\n"
    left = (
        "fn add(a: i64, b: i64) -> i64 { a + b }\n"
        "\n"
        "struct Point { x: i64, y: i64 }\n"
    )
    right = base
    overlay = structural_merge(base, left, right, language="rust")
    assert overlay.conflicts == []
    assert "struct Point" in overlay.source
