"""Tests for aero_forge.healing.evaluator.LogEvaluator."""

import pytest

from aero_forge.healing.evaluator import LogEvaluator


@pytest.fixture
def evaluator():
    return LogEvaluator()


def test_python_syntax_error_is_healable(evaluator):
    log = (
        "Traceback (most recent call last):\n"
        '  File "/tmp/main.py", line 7, in <module>\n'
        "    if x ==\n"
        "           ^\n"
        "SyntaxError: invalid syntax\n"
    )
    result = evaluator.evaluate_log("python main.py", 1, log)
    assert result["healable"] is True
    assert result["error_type"] == "python_syntax"
    assert result["target_file"].endswith("main.py")
    assert result["line_number"] == 7


def test_python_name_error_is_healable(evaluator):
    log = (
        "Traceback (most recent call last):\n"
        '  File "src/main.py", line 12, in run\n'
        "    value = math.sqrt(x)\n"
        "NameError: name 'math' is not defined\n"
    )
    result = evaluator.evaluate_log("python src/main.py", 1, log)
    assert result["healable"] is True
    assert result["error_type"] == "python_name_error"
    assert result["target_file"].endswith("src/main.py")


def test_rust_unexpected_closing_delimiter_is_healable(evaluator):
    log = (
        "error: unexpected closing delimiter: `}`\n"
        " --> src/lib.rs:42:17\n"
        "   |\n"
        "42 | }\n"
        "   |                 ^ unexpected closing delimiter\n"
    )
    result = evaluator.evaluate_log("cargo build", 101, log)
    assert result["healable"] is True
    assert result["error_type"] == "rust_compile"
    assert result["target_file"] == "src/lib.rs"
    assert result["line_number"] == 42


def test_missing_toolchain_not_healable(evaluator):
    result = evaluator.evaluate_log("maturin develop", 127, "/bin/sh: 1: maturin: not found")
    assert result["healable"] is False
    assert result["error_type"] == "missing_toolchain"


def test_successful_exit_returns_not_healable(evaluator):
    result = evaluator.evaluate_log("python main.py", 0, "")
    assert result["healable"] is False
    assert result["summary"] == "No error detected."
