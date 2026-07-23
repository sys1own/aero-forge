"""Tests for the error explainer."""

from __future__ import annotations

from aero_forge.error_explainer import explain_error


def test_local_type_mismatch_explanation():
    log = "error[E0308]: mismatched types\nexpected `i64`, found `f64`"
    explanation = explain_error(log, source="def f() -> int:\n    return 1.0")
    assert (
        "type mismatch" in explanation.lower()
        or "mismatched types" in explanation.lower()
    )
    assert "Suggestion" in explanation


def test_local_unsupported_explanation():
    log = "UnsupportedError: with statements / context managers are not supported"
    explanation = explain_error(log)
    assert "with" in explanation.lower()
