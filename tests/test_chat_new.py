"""Tests for recent chat/build pipeline additions."""

from pathlib import Path

import pytest

from aero_forge.chat import (
    ChatSession,
    _parse_multifile_response,
    _split_by_file_markers,
)


def test_split_by_file_markers():
    response = (
        "# file: Cargo.toml\n[workspace]\n\n"
        "// file: src/lib.rs\nfn main() {}\n\n"
        "# file: tests/test.py\ndef test(): pass\n"
    )
    sections = _split_by_file_markers(response)
    assert [p for p, _ in sections] == ["Cargo.toml", "src/lib.rs", "tests/test.py"]
    assert "[workspace]" in sections[0][1]
    assert "fn main()" in sections[1][1]


def test_parse_multifile_response_marker_fallback(tmp_path):
    response = (
        '# file: Cargo.toml\n[workspace]\nresolver = "2"\n'
        "// file: rust_core/src/lib.rs\nfn rust() {}\n"
    )
    files = _parse_multifile_response(response, tmp_path)
    for target, content in files.items():
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
    assert (tmp_path / "Cargo.toml").exists()
    assert (tmp_path / "rust_core" / "src" / "lib.rs").exists()
    assert "resolver" in (tmp_path / "Cargo.toml").read_text()


def test_ensure_package_inits_creates_init(tmp_path):
    session = ChatSession(tmp_path)
    pkg_dir = tmp_path / "cpp_engine"
    pkg_dir.mkdir()
    (pkg_dir / "native_bridge.py").write_text("x = 1\n", encoding="utf-8")
    created = session._ensure_package_inits()
    assert any(p.name == "__init__.py" for p in created)
    assert (pkg_dir / "__init__.py").is_file()


def test_build_cpp_extensions_runs_makefiles(tmp_path):
    session = ChatSession(tmp_path)
    cpp_dir = tmp_path / "cpp_engine"
    cpp_dir.mkdir()
    (cpp_dir / "Makefile").write_text(
        "all:\n\t@echo built > marker.txt\n", encoding="utf-8"
    )
    results = session._build_cpp_extensions()
    assert len(results) == 1
    assert results[0].returncode == 0
    assert (cpp_dir / "marker.txt").is_file()
