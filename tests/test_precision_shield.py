"""Tests for precision shield `src/precision.rs` injection and SMT sketch solver."""

import pytest

from aero_forge.precision_shield import ensure_precision_traits


def test_ensure_precision_traits_creates_file(tmp_path):
    precision_rs, created = ensure_precision_traits(tmp_path)
    assert created
    assert precision_rs == tmp_path / "src" / "precision.rs"
    assert precision_rs.is_file()
    text = precision_rs.read_text(encoding="utf-8")
    assert "AeroNegMutExt" in text
    assert "impl AeroNegMutExt for rug::Float" in text
    assert "impl AeroNegMutExt for rug::Complex" in text


def test_ensure_precision_traits_is_idempotent(tmp_path):
    ensure_precision_traits(tmp_path)
    precision_rs, created = ensure_precision_traits(tmp_path)
    assert not created
    count = precision_rs.read_text(encoding="utf-8").count("AeroNegMutExt")
    # trait declaration plus one impl per type
    assert count == 3


def test_smt_ast_engine_solves_typed_holes():
    """SAT: hole constraints resolve to concrete target-language types."""
    from aero_forge.precision_shield import SMTASTEngine

    engine = SMTASTEngine()
    holes = ["h1", "h2"]
    constraints = [
        {"source_hole": "h1", "target_language": "rust"},
        {"source_hole": "h2", "target_language": "python"},
        {"source_hole": "h1", "target_hole": "h2"},
    ]
    # h1 forced Rust and equal to h2, h2 forced Python -> h2 must be both Rust
    # and Python, which is impossible. Make h2 unconstrained instead.
    constraints[2] = {"source_hole": "h1", "target_hole": "h1"}
    result = engine.solve_ast_sketch_holes(holes, constraints)
    assert result["h1"] == "RustType"
    assert result["h2"] == "PyType"


def test_smt_ast_engine_ffi_layout_sat():
    """SAT: matching FFI layout constraints are satisfiable."""
    from aero_forge.precision_shield import SMTASTEngine

    engine = SMTASTEngine()
    constraints = [
        {"source_hole": "h1", "target_language": "rust"},
        {
            "ffi_layout": {
                "struct": "Session",
                "field": "count",
                "rust_offset": 0,
                "cpp_offset": 0,
                "rust_align": 8,
                "cpp_align": 8,
            }
        },
    ]
    result = engine.solve_ast_sketch_holes(["h1"], constraints)
    assert result["h1"] == "RustType"


def test_smt_ast_engine_unsat_on_conflicting_types():
    """UNSAT: a hole cannot be both Rust and Python."""
    from aero_forge.precision_shield import SMTASTEngine

    engine = SMTASTEngine()
    constraints = [
        {"source_hole": "h1", "target_language": "rust"},
        {"source_hole": "h1", "target_language": "python"},
    ]
    with pytest.raises(ValueError, match="Unsatisfiable"):
        engine.solve_ast_sketch_holes(["h1"], constraints)


def test_smt_ast_engine_unsat_on_ffi_layout_mismatch():
    """UNSAT: mismatched FFI offsets make the layout constraints impossible."""
    from aero_forge.precision_shield import SMTASTEngine

    engine = SMTASTEngine()
    constraints = [
        {
            "ffi_layout": {
                "struct": "Session",
                "field": "count",
                "rust_offset": 0,
                "cpp_offset": 4,
                "rust_align": 8,
                "cpp_align": 8,
            }
        }
    ]
    with pytest.raises(ValueError, match="Unsatisfiable"):
        engine.solve_ast_sketch_holes([], constraints)


def test_smt_ast_engine_import_visibility():
    """SAT: reachable imports can be enforced as boolean predicates."""
    from aero_forge.precision_shield import SMTASTEngine

    engine = SMTASTEngine()
    constraints = [
        {"source_hole": "h1", "target_language": "cpp"},
        {
            "imports": {
                "module": "native_loader",
                "symbols": ["load_rust", "load_cpp"],
                "visible": True,
            }
        },
    ]
    result = engine.solve_ast_sketch_holes(["h1"], constraints)
    assert result["h1"] == "CppType"
