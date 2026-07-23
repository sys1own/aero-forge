"""Tests for the built-in algorithm library."""

from __future__ import annotations

import pytest

from aero_forge.algorithms import find_algorithm, get_algorithm, list_algorithms


@pytest.mark.parametrize(
    "name",
    ["fibonacci", "gcd", "is_prime", "matrix_multiply", "quicksort"],
)
def test_algorithm_library_contains_common_algorithms(name):
    assert name in list_algorithms()
    assert get_algorithm(name) is not None


def test_find_algorithm_matches_keywords():
    assert "def fibonacci" in find_algorithm("compute the fibonacci number")
    assert "def gcd" in find_algorithm("greatest common divisor")
    assert "def is_prime" in find_algorithm("check if a number is prime")
    assert "def matrix_multiply" in find_algorithm("multiply two matrices")
    assert "def quicksort" in find_algorithm("sort an array with quicksort")
