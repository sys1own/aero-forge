"""Tests for automatic blueprint generation from uploaded repositories."""

from __future__ import annotations

from pathlib import Path

import pytest

from aero_forge.blueprint import (
    generate_blueprint_from_uploaded_repo,
    parse_blueprint,
)


def test_generates_blueprint_for_rust_python_repo(tmp_path: Path) -> None:
    repo = tmp_path / "my_repo"
    repo.mkdir()
    (repo / "Cargo.toml").write_text(
        '[package]\nname = "my_repo"\nversion = "0.1.0"\nedition = "2021"\n',
        encoding="utf-8",
    )
    (repo / "rust_core").mkdir()
    (repo / "rust_core" / "Cargo.toml").write_text(
        '[package]\nname = "rust_core"\nversion = "0.1.0"\n',
        encoding="utf-8",
    )
    (repo / "rust_core" / "src").mkdir(parents=True)
    (repo / "rust_core" / "src" / "lib.rs").write_text("pub fn add(a: i32, b: i32) -> i32 { a + b }\n", encoding="utf-8")
    (repo / "pyproject.toml").write_text('[project]\nname = "my_repo"\nversion = "0.1.0"\n', encoding="utf-8")
    (repo / "src").mkdir()
    (repo / "src" / "engine.py").write_text("def run():\n    pass\n", encoding="utf-8")

    blueprint_path = generate_blueprint_from_uploaded_repo(repo)
    assert blueprint_path == repo / "blueprint.aero"
    assert blueprint_path.is_file()

    bp = parse_blueprint(blueprint_path)
    assert bp.architecture == "hybrid_rust_python"
    assert "cargo" in bp.toolchains
    assert "python" in bp.toolchains
    paths = {m.path for m in bp.manifest}
    assert "Cargo.toml" in paths
    assert "pyproject.toml" in paths
    assert "src/engine.py" in paths or "rust_core/src/lib.rs" in paths


def test_returns_existing_blueprint_without_overwrite(tmp_path: Path) -> None:
    repo = tmp_path / "existing"
    repo.mkdir()
    existing = repo / "blueprint.aero"
    existing.write_text("project: existing\n", encoding="utf-8")

    result = generate_blueprint_from_uploaded_repo(repo)
    assert result == existing
    assert existing.read_text(encoding="utf-8") == "project: existing\n"


def test_generates_blueprint_for_pure_python_repo(tmp_path: Path) -> None:
    repo = tmp_path / "py_repo"
    repo.mkdir()
    (repo / "pyproject.toml").write_text('[project]\nname = "py_repo"\n', encoding="utf-8")
    (repo / "app.py").write_text("def main():\n    print('hi')\n", encoding="utf-8")

    generate_blueprint_from_uploaded_repo(repo)
    bp = parse_blueprint(repo / "blueprint.aero")
    assert bp.architecture == "pure_python"
    assert "python" in bp.languages
    assert any(m.path == "app.py" for m in bp.manifest)
