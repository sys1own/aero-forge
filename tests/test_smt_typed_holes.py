"""Tests for SMTASTEngine native type inference and typed-hole logging."""

from aero_forge.precision_shield.smt_solver import SMTASTEngine


def test_infer_native_types_simple_arithmetic():
    """Ambiguous scalar arithmetic is resolved to f64 by usage."""
    engine = SMTASTEngine()
    source = (
        "def transform(a, b):\n"
        "    c = a + b\n"
        "    return c * 2.0\n"
    )
    result = engine.infer_native_types(source)
    assert result["a"] == "f64"
    assert result["b"] == "f64"
    assert result["c"] == "f64"
    assert result["__return__"] == "f64"


def test_infer_native_types_vector_reduction():
    """Loop over a list of floats resolves the iterable to Vec<f64>."""
    engine = SMTASTEngine()
    source = (
        "def process(items):\n"
        "    s = 0.0\n"
        "    for x in items:\n"
        "        s = s + x * 2.0\n"
        "    return s\n"
    )
    result = engine.infer_native_types(source)
    assert result["items"] == "Vec<f64>"
    assert result["x"] == "f64"
    assert result["s"] == "f64"
    assert result["__return__"] == "f64"


def test_infer_native_types_integer_only():
    """Pure integer arithmetic stays integer (i64 or usize)."""
    engine = SMTASTEngine()
    source = (
        "def total(n):\n"
        "    s = 0\n"
        "    for i in range(n):\n"
        "        s = s + i\n"
        "    return s\n"
    )
    result = engine.infer_native_types(source)
    assert result["n"] in ("i64", "usize")
    assert result["i"] == "usize"
    assert result["s"] == "i64"
    assert result["__return__"] == "i64"
