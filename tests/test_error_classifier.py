"""Tests for error classification."""

import pytest

from aero_forge.orchestrator.error_classifier import (
    ErrorClass,
    classify,
    classify_exception,
    is_fatal,
    is_transient,
)


@pytest.mark.parametrize(
    "text,expected",
    [
        ("Rate limit exceeded", ErrorClass.TRANSIENT),
        ("APIConnectionError: unreachable", ErrorClass.TRANSIENT),
        ("timeout connecting to api", ErrorClass.TRANSIENT),
        ("No linker found", ErrorClass.FATAL),
        ("missing Rust toolchain: cargo", ErrorClass.FATAL),
        ("out of memory", ErrorClass.FATAL),
        ("expected i64, found f64", ErrorClass.RECOVERABLE),
        ("assert fibonacci(10) == 55", ErrorClass.RECOVERABLE),
        # Cargo progress lines must not be flagged as errors.
        (
            "Compiling pyo3 v0.20.3\nFinished release [optimized] target(s)",
            ErrorClass.RECOVERABLE,
        ),
        ("Updating crates.io index\nDownloading foo v1.0", ErrorClass.RECOVERABLE),
    ],
)
def test_classify(text, expected):
    assert classify(text) == expected


def test_fatal_and_transient_helpers():
    assert is_fatal("No linker found")
    assert is_transient("Rate limit exceeded")
    assert not is_fatal("expected i64")
    # Cargo stderr progress should not be considered fatal or transient.
    cargo_log = "Compiling pyo3 v0.20.3\nFinished release [optimized] target(s) in 1.2s"
    assert not is_fatal(cargo_log)
    assert not is_transient(cargo_log)


class FakeTransient(Exception):
    pass


def test_classify_exception():
    from unittest.mock import MagicMock

    try:
        from openai import RateLimitError
    except ImportError:
        pytest.skip("openai not installed")
    response = MagicMock()
    response.request = MagicMock()
    assert (
        classify_exception(RateLimitError("x", response=response, body=None))
        == ErrorClass.TRANSIENT
    )
