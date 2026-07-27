"""Tests for the strict LogEvaluator classifier."""

import pytest

from aero_forge.healing.evaluator import LogEvaluator


@pytest.fixture
def evaluator() -> LogEvaluator:
    return LogEvaluator()


def test_rust_unexpected_closing_delimiter_is_healable(evaluator):
    log = """
error: unexpected closing delimiter: `}`
   --> src/lib.rs:42:1
    |
42  | }
    | ^
"""
    result = evaluator.evaluate_log("cargo test", 101, log)
    assert result["healable"] is True
    assert result["error_type"] == "rust_syntax"
    assert result["target_file"] == "src/lib.rs"
    assert result["line_number"] == 42


def test_rust_e0308_type_mismatch_is_not_healable(evaluator):
    log = """
error[E0308]: mismatched types
   --> native_core/src/lib.rs:55:9
    |
55  |     let x: i64 = 1.0;
    |                  ^^^ expected `i64`, found `f64`
"""
    result = evaluator.evaluate_log("cargo test -p native_core", 101, log)
    assert result["healable"] is False
    assert result["error_type"] == "type_mismatch"
    assert result["code"] == "E0308"
    assert "type refactoring" in result["reason"].lower()


def test_rust_e0432_unresolved_import_is_not_healable(evaluator):
    log = """
error[E0432]: unresolved import `pyo3`
   --> native_core/src/lib.rs:3:5
    |
3   | use pyo3::prelude::*;
    |     ^^^
"""
    result = evaluator.evaluate_log("cargo build -p native_core", 101, log)
    assert result["healable"] is False
    assert result["error_type"] == "unresolved_import"
    assert result["code"] == "E0432"


def test_cargo_dependency_conflict_is_not_healable(evaluator):
    log = """
error: failed to select a version for `cc`.
    ... required by package `blake3 v1.5.4`
    ... which satisfies dependency `blake3 = "=1.5.4"` of package `native_core`
versions that meet the requirements `^1.1.12` are: 1.4.0, 1.3.0

all possible versions conflict with previously selected packages.
"""
    result = evaluator.evaluate_log("cargo test -p native_core", 101, log)
    assert result["healable"] is False
    assert result["error_type"] == "cargo_dependency_conflict"
    assert "Cargo dependency version conflict" in result["summary"]
    assert result["target_file"] == "Cargo.toml"


def test_python_syntax_error_is_healable(evaluator):
    log = """
  File "/tmp/proj/main.py", line 7, in <module>
    if x == 1
             ^
SyntaxError: expected ':'
"""
    result = evaluator.evaluate_log("python main.py", 1, log)
    assert result["healable"] is True
    assert result["error_type"] == "python_syntax"
    assert result["target_file"] == "/tmp/proj/main.py"
    assert result["line_number"] == 7


def test_python_module_not_found_is_not_healable(evaluator):
    log = "ModuleNotFoundError: No module named 'numpy'"
    result = evaluator.evaluate_log("python main.py", 1, log)
    assert result["healable"] is False
    assert result["error_type"] == "python_missing_module"
    assert "Missing Python dependency" in result["summary"]


def test_python_name_error_is_healable(evaluator):
    log = (
        "Traceback (most recent call last):\n"
        '  File "/tmp/proj/main.py", line 4, in <module>\n'
        "    print(math.sqrt(16))\n"
        "NameError: name 'math' is not defined\n"
    )
    result = evaluator.evaluate_log("python main.py", 1, log)
    assert result["healable"] is True
    assert result["error_type"] == "python_name_error"
    assert result["target_file"] == "/tmp/proj/main.py"


def test_missing_toolchain_is_not_healable(evaluator):
    log = "/bin/sh: 1: maturin: not found"
    result = evaluator.evaluate_log("maturin develop", 127, log)
    assert result["healable"] is False
    assert result["error_type"] == "missing_toolchain"
