"""Integration tests for the workspace.aeroc pack/unpack round-trip."""

from __future__ import annotations

import shutil
from pathlib import Path

from aero_forge.builder.aeroc_compiler import compile_directory_to_aeroc
from aero_forge.materializer import auto_materialize, unpack_aeroc_file, workspace_requires_materialization
from aero_forge.scaffold.aeroc_export import compile_hybrid_aeroc


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


def _dummy_elf() -> bytes:
    """Return the smallest byte sequence that looks like a valid ELF shared object."""
    return b"\x7fELF" + b"\x00" * 16


def test_hybrid_aeroc_packs_and_unpacks_matching_native_binary(tmp_path: Path) -> None:
    """A hybrid .aeroc stores source and a matching native binary."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "main.py").write_text("print('hello')\n")
    (workspace / "libnative.so").write_bytes(_dummy_elf())

    aeroc = tmp_path / "workspace.aeroc"
    compile_hybrid_aeroc(workspace, aeroc)

    out = tmp_path / "extracted"
    out.mkdir()
    unpack_aeroc_file(aeroc, out)

    assert (out / "main.py").read_text() == "print('hello')\n"
    assert (out / "libnative.so").is_file()
    assert (out / "libnative.so").read_bytes() == _dummy_elf()
    assert (out / "environment.lock").is_file()
    assert not (out / ".aeroc_fallback").exists()


def test_hybrid_aeroc_skips_mismatched_native_binary_and_marks_fallback(tmp_path: Path) -> None:
    """A binary built for a different platform is skipped and a fallback marker is written."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "main.py").write_text("print('hello')\n")
    (workspace / "libnative.so").write_bytes(_dummy_elf())

    aeroc = tmp_path / "workspace.aeroc"
    # Package the binary for a platform that is not the current host.
    compile_hybrid_aeroc(workspace, aeroc, platform_tag="macos_aarch64")

    out = tmp_path / "extracted"
    out.mkdir()
    unpack_aeroc_file(aeroc, out)

    assert (out / "main.py").is_file()
    assert not (out / "libnative.so").exists()
    assert (out / ".aeroc_fallback").is_file()
    assert "no matching native binary" in (out / ".aeroc_fallback").read_text()


def test_hybrid_aeroc_preserves_non_hybrid_src_prefix(tmp_path: Path) -> None:
    """Legacy .aeroc files without the hybrid marker keep their ``src/`` directories intact."""
    src = tmp_path / "src"
    src.mkdir()
    (src / "main.rs").write_text("fn main() {}\n")
    (tmp_path / "Cargo.toml").write_text('[package]\nname = "legacy"\n')

    aeroc = tmp_path / "workspace.aeroc"
    compile_directory_to_aeroc(tmp_path, aeroc)

    out = tmp_path / "extracted"
    out.mkdir()
    unpack_aeroc_file(aeroc, out)

    # Legacy layout must keep the original ``src/`` directory.
    assert (out / "src" / "main.rs").is_file()
    assert (out / "Cargo.toml").is_file()
