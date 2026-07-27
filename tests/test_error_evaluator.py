"""Tests for the smart error evaluator and healing strategy selection."""

import pytest

from aero_forge.healing.evaluator import HealingStrategy, LogEvaluator


@pytest.fixture
def evaluator() -> LogEvaluator:
    return LogEvaluator()


def test_python_syntax_error_selects_ast(evaluator: LogEvaluator) -> None:
    log = (
        "Traceback (most recent call last):\n"
        '  File "main.py", line 2, in <module>\n'
        "    def foo(\n"
        "SyntaxError: unexpected EOF while parsing\n"
    )
    assert evaluator.evaluate_error(log) == HealingStrategy.AST


def test_python_name_error_selects_ast(evaluator: LogEvaluator) -> None:
    log = "Traceback (most recent call last):\nNameError: name 'math' is not defined\n"
    assert evaluator.evaluate_error(log) == HealingStrategy.AST


def test_rust_type_mismatch_selects_llm(evaluator: LogEvaluator) -> None:
    log = (
        "error[E0308]: mismatched types\n"
        "  --> native_core/src/lib.rs:12:23\n"
    )
    assert evaluator.evaluate_error(log) == HealingStrategy.LLM


def test_rust_trait_error_selects_llm(evaluator: LogEvaluator) -> None:
    log = "error[E0277]: the trait bound `i64: std::ops::Add` is not satisfied\n"
    assert evaluator.evaluate_error(log) == HealingStrategy.LLM


def test_cargo_dependency_conflict_selects_llm(evaluator: LogEvaluator) -> None:
    log = "error: failed to select a version for `blake3`\n"
    assert evaluator.evaluate_error(log) == HealingStrategy.LLM


def test_missing_toolchain_selects_manual(evaluator: LogEvaluator) -> None:
    log = "bash: cargo: command not found\n"
    assert evaluator.evaluate_error(log) == HealingStrategy.MANUAL


def test_unknown_error_selects_manual(evaluator: LogEvaluator) -> None:
    log = "some random failure with no recognizable pattern\n"
    assert evaluator.evaluate_error(log) == HealingStrategy.MANUAL
