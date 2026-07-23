"""Tests for the examples gallery."""

from __future__ import annotations

from pathlib import Path

import pytest

from aero_forge.examples import list_examples, run_example


def test_list_examples():
    examples = list_examples()
    names = {e["name"] for e in examples}
    assert "fibonacci" in names
    assert "factorial" in names


def test_run_example(tmp_path):
    """A curated example should build and pass tests."""
    result = run_example("fibonacci", build_kwargs={"cache_enabled": False})
    assert result["success"] is True
    assert result["passed"] >= 1


def test_run_unknown_example():
    with pytest.raises(ValueError):
        run_example("does_not_exist")
