"""Integration tests for the workspace.aeroc pack/unpack round-trip."""

from __future__ import annotations

import shutil
from pathlib import Path

from aero_forge.builder.aeroc_compiler import compile_directory_to_aeroc
from aero_forge.materializer import auto_materialize, unpack_aeroc_file, workspace_requires_materialization


def test_compile_and_unpack_roundtrip(tmp_path: Path) -> None:
    """A workspace tree compiled to .aeroc is reconstructed byte-for-byte."""
    src = tmp_path / "src"
    src.mkdir(parents=True)
    (src / "main.rs").write_text("fn main() {}\n")
    (src / "lib.rs").write_text("pub fn add(a: i32, b: i32) -> i32 { a + b }\n")
    (tmp_path / "Cargo.toml").write_text('[package]\nname = "demo"\n')
    (tmp_path / "blueprint.aero").write_text("metadata:\n  schema_version: 3.0.0\n")

    aeroc = tmp_path / "workspace.aeroc"
    compile_directory_to_aeroc(tmp_path, aeroc)

    empty = tmp_path / "extracted"
    empty.mkdir()
    unpack_aeroc_file(aeroc, empty)

    assert (empty / "src" / "main.rs").read_text() == (src / "main.rs").read_text()
    assert (empty / "src" / "lib.rs").read_text() == (src / "lib.rs").read_text()
    assert (empty / "Cargo.toml").read_text() == (tmp_path / "Cargo.toml").read_text()
    assert (empty / "blueprint.aero").read_text() == (tmp_path / "blueprint.aero").read_text()


def test_auto_materialize_extracts_aeroc(tmp_path: Path) -> None:
    """An empty directory containing only workspace.aeroc is auto-materialized."""
    original = tmp_path / "original"
    (original / "src").mkdir(parents=True)
    (original / "Cargo.toml").write_text('[package]\nname = "demo"\n')
    (original / "src" / "main.rs").write_text("fn main() {}\n")

    aeroc = original / "workspace.aeroc"
    compile_directory_to_aeroc(original, aeroc)

    empty = tmp_path / "empty"
    empty.mkdir()
    shutil.copy(aeroc, empty / "workspace.aeroc")

    assert workspace_requires_materialization(empty)
    assert auto_materialize(empty)
    assert (empty / "Cargo.toml").read_text() == (original / "Cargo.toml").read_text()
    assert (empty / "src" / "main.rs").read_text() == (original / "src" / "main.rs").read_text()
    assert not workspace_requires_materialization(empty)
