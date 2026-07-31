"""Tests for HIN-to-WASM lowering."""

import pytest

from aero_forge.errors import UnsupportedError
from aero_forge.wasm import lower_hin_to_wat


def test_lower_hin_to_wat_arithmetic():
    wat = lower_hin_to_wat("def f(x, y):\n    return x + y * 2.0\n", "f")
    assert '(export "f")' in wat
    assert "(param f64)" in wat
    assert "(result f64)" in wat
    assert "f64.mul" in wat
    assert "f64.add" in wat


def test_lower_hin_to_wat_unary():
    wat = lower_hin_to_wat("def g(x):\n    return -x + 3.0\n", "g")
    assert "f64.neg" in wat
    assert "f64.const 3.0" in wat
    assert "f64.add" in wat


def test_lower_hin_to_wat_unsupported_body():
    with pytest.raises(UnsupportedError):
        lower_hin_to_wat("def h(x):\n    if x > 0:\n        return x\n    return 0\n", "h")
