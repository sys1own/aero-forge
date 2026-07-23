"""Tests for LLM self-review on generated code."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from aero_forge.generate import _review_code


def test_review_code_returns_corrected_version():
    response = (
        "The original implementation is slow; use an iterative loop.\n\n"
        "```python\ndef factorial(n: int) -> int:\n"
        "    result = 1\n"
        "    for i in range(2, n + 1):\n"
        "        result *= i\n"
        "    return result\n```"
    )
    client = MagicMock()
    client.generate.return_value = response

    original = "def factorial(n: int) -> int:\n    return n * factorial(n - 1)"
    with patch("aero_forge.generate.get_llm_client", return_value=client):
        corrected = _review_code(original, "factorial", None, "openai", None, 3)

    assert "iterative" not in corrected  # corrected code only, no notes
    assert "result = 1" in corrected


def test_review_code_no_client_returns_original():
    original = "def f(): pass"
    with patch("aero_forge.generate.get_llm_client", return_value=None):
        assert _review_code(original, "f", None, "none", None, 3) == original
