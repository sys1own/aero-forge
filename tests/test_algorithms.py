"""Tests for the algorithm library and selection."""

from __future__ import annotations

from aero_forge.algorithms import (
    find_algorithm,
    get_algorithm,
    list_algorithms,
    list_categories,
    select_algorithm,
)


def test_library_contains_expected_algorithms():
    names = list_algorithms()
    assert "quicksort" in names
    assert "mergesort" in names
    assert "is_prime" in names
    assert len(names) >= 10


def test_categories():
    categories = list_categories()
    assert "sorting" in categories
    assert "math" in categories
    assert "matrix" in categories


def test_find_algorithm_keyword():
    algo = find_algorithm("Build a fast sorting function")
    assert algo is not None
    assert algo.name == "quicksort"


def test_find_algorithm_prime():
    algo = find_algorithm("prime number test")
    assert algo is not None
    assert algo.name == "is_prime"


def test_select_algorithm_fallback_to_keywords():
    """Without an LLM provider, selection falls back to keyword matching."""
    algo = select_algorithm("sort a list quickly")
    assert algo is not None
    assert algo.category == "sorting"


def test_get_algorithm_metadata():
    algo = get_algorithm("quicksort")
    assert algo is not None
    assert algo.metadata["category"] == "sorting"
    assert "complexity" in algo.metadata
    assert "time" in algo.metadata["complexity"]
