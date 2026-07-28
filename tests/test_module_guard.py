"""Tests for the module tree guard."""

from pathlib import Path

from aero_forge.scaffold.module_guard import reify_missing_modules


def test_reifies_missing_rust_module(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir(parents=True)
    lib = tmp_path / "src" / "lib.rs"
    lib.write_text("pub mod diagnostics;\n", encoding="utf-8")
    created = reify_missing_modules(tmp_path)
    assert len(created) == 1
    assert created[0] == tmp_path / "src" / "diagnostics.rs"
    assert created[0].read_text(encoding="utf-8").startswith("// Auto-generated module stub")


def test_reifies_missing_python_module(tmp_path: Path) -> None:
    main = tmp_path / "main.py"
    main.write_text("import helpers\n", encoding="utf-8")
    created = reify_missing_modules(tmp_path)
    assert len(created) == 1
    assert created[0] == tmp_path / "helpers.py"


def test_no_stub_when_module_exists(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir(parents=True)
    lib = tmp_path / "src" / "lib.rs"
    lib.write_text("pub mod existing;\n", encoding="utf-8")
    (tmp_path / "src" / "existing.rs").write_text("pub fn x() {}\n")
    created = reify_missing_modules(tmp_path)
    assert not created
