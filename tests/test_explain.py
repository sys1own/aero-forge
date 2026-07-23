"""Tests for explainable builds."""

from __future__ import annotations

from aero_forge.generate import extract_explanation


def test_extract_explanation_markdown_section():
    text = (
        "```python\ndef f():\n    pass\n```\n\n"
        "## Explanation\nThis uses O(n log n) divide and conquer.\n"
        "## Tradeoffs\nSpace is O(n).\n"
    )
    explanation = extract_explanation(text)
    assert "O(n log n)" in explanation
    assert "divide and conquer" in explanation
    assert "Space" not in explanation


def test_extract_explanation_marker():
    text = "EXPLANATION: uses dynamic programming for optimal substructure.\n"
    assert "dynamic programming" in extract_explanation(text)


def test_extract_explanation_missing():
    assert extract_explanation("just code without explanation") == ""
