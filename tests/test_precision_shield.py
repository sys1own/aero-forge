"""Tests for precision shield `src/precision.rs` injection."""

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
