"""Tests for the Rust/C/C++ syntax truncation guard."""

from __future__ import annotations

from pathlib import Path

from aero_forge.scaffold.syntax_guard import repair_file, repair_source, repair_workspace


def test_repair_missing_closing_brace() -> None:
    source = "fn main() {\n    let x = 1;\n"
    repaired = repair_source(source)
    assert "}\n" in repaired
    assert repaired.count("{") == repaired.count("}")


def test_repair_unbalanced_parens_and_brackets() -> None:
    source = "fn foo(a: [i32; 3] {\n    bar(1, 2\n"
    repaired = repair_source(source)
    assert repaired.count("(") == repaired.count(")")
    assert repaired.count("[") == repaired.count("]")


def test_strip_trailing_doc_comments() -> None:
    source = "fn main() {}\n///\n/// dangling doc\n"
    repaired = repair_source(source)
    assert "///" not in repaired
    assert "fn main() {}" in repaired


def test_strip_unclosed_block_comment() -> None:
    source = "fn main() {}\n/* dangling block comment\n"
    repaired = repair_source(source)
    assert "/*" not in repaired
    assert "fn main() {}" in repaired


def test_no_change_for_valid_file(tmp_path: Path) -> None:
    path = tmp_path / "valid.rs"
    path.write_text("fn main() {\n    let x = 1;\n}\n", encoding="utf-8")
    assert repair_file(path) is False
    assert path.read_text(encoding="utf-8") == "fn main() {\n    let x = 1;\n}\n"


def test_repair_workspace(tmp_path: Path) -> None:
    (tmp_path / "a.rs").write_text("fn a() {\n", encoding="utf-8")
    (tmp_path / "b.cpp").write_text("int main() {\n", encoding="utf-8")
    (tmp_path / "readme.txt").write_text("hello", encoding="utf-8")
    changed = repair_workspace(tmp_path)
    assert len(changed) == 2
    assert all(p.suffix in {".rs", ".cpp"} for p in changed)
