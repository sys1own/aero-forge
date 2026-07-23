"""Unit tests for the self-healing router."""

from aero_forge.healing.router import try_auto_fix


def test_adds_missing_math_import():
    code = "def sqrt_x(x):\n    return math.sqrt(x)\n"
    error = "NameError: name 'math' is not defined"
    fixed = try_auto_fix(error, code)
    assert fixed is not None
    assert "import math" in fixed


def test_no_fix_for_unknown_error():
    code = "def f(x):\n    return x\n"
    assert try_auto_fix("some random error", code) is None
