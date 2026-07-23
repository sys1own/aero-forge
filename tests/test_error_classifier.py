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
    ],
)
def test_classify(text, expected):
    assert classify(text) == expected


def test_fatal_and_transient_helpers():
    assert is_fatal("No linker found")
    assert is_transient("Rate limit exceeded")
    assert not is_fatal("expected i64")


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
