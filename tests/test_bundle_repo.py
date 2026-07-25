"""Tests for ``aero_forge.bundle_repo``."""

import json
from pathlib import Path

import pytest

from aero_forge.bundle_repo import (
    bundle_to_json,
    bundle_to_xml,
    bundle_workspace,
    format_context_block,
)


def test_bundle_workspace_includes_source_files(tmp_path: Path) -> None:
    (tmp_path / "Cargo.toml").write_text("[package]\nname = 'demo'\n")
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "lib.rs").write_text("pub fn add(a: i32, b: i32) -> i32 { a + b }\n")
    (tmp_path / "python_engine").mkdir()
    (tmp_path / "python_engine" / "core.py").write_text("def run(): pass\n")
    (tmp_path / "blueprint.aero").write_text("project: demo\n")
    (tmp_path / "README.md").write_text("# Demo\n")

    bundle = bundle_workspace(tmp_path, max_file_size_kb=100)

    assert bundle["files"]["Cargo.toml"] == "[package]\nname = 'demo'\n"
    assert "src/lib.rs" in bundle["files"]
    assert "python_engine/core.py" in bundle["files"]
    assert bundle["blueprint"] == "project: demo\n"
    assert bundle["files"]["README.md"] == "# Demo\n"


def test_bundle_workspace_excludes_build_directories(tmp_path: Path) -> None:
    (tmp_path / "target" / "release").mkdir(parents=True)
    (tmp_path / "target" / "release" / "libdemo.so").write_bytes(b"binary")
    (tmp_path / "dist" / "artifact.so").parent.mkdir(parents=True, exist_ok=True)
    (tmp_path / "dist" / "artifact.so").write_bytes(b"binary")
    (tmp_path / "rust_core" / "target" / "debug").mkdir(parents=True)
    (tmp_path / "rust_core" / "target" / "debug" / "libdemo.rlib").write_bytes(b"binary")
    (tmp_path / "__pycache__").mkdir()
    (tmp_path / "__pycache__" / "foo.cpython-310.pyc").write_bytes(b"binary")
    (tmp_path / ".pytest_cache").mkdir()
    (tmp_path / ".cargo").mkdir()
    (tmp_path / ".git").mkdir()
    (tmp_path / "src" / "main.rs").parent.mkdir(parents=True, exist_ok=True)
    (tmp_path / "src" / "main.rs").write_text("fn main() {}\n")

    bundle = bundle_workspace(tmp_path, max_file_size_kb=100)

    assert "target/release/libdemo.so" not in bundle["files"]
    assert "dist/artifact.so" not in bundle["files"]
    assert "rust_core/target/debug/libdemo.rlib" not in bundle["files"]
    assert "__pycache__/foo.cpython-310.pyc" not in bundle["files"]
    assert ".pytest_cache" not in str(bundle["files"])
    assert ".cargo" not in str(bundle["files"])
    assert ".git" not in str(bundle["files"])
    assert bundle["files"]["src/main.rs"] == "fn main() {}\n"


def test_bundle_workspace_excludes_binary_artifacts(tmp_path: Path) -> None:
    for name in ["lib.so", "lib.pyd", "lib.dll", "lib.dylib", "app.wasm", "out.zip", "img.png"]:
        (tmp_path / name).write_bytes(b"binary")
    (tmp_path / "src" / "lib.rs").parent.mkdir(parents=True, exist_ok=True)
    (tmp_path / "src" / "lib.rs").write_text("pub fn ok() {}\n")

    bundle = bundle_workspace(tmp_path, max_file_size_kb=100)

    for name in ["lib.so", "lib.pyd", "lib.dll", "lib.dylib", "app.wasm", "out.zip", "img.png"]:
        assert name not in bundle["files"]
    assert bundle["files"]["src/lib.rs"] == "pub fn ok() {}\n"


def test_bundle_workspace_respects_size_limit(tmp_path: Path) -> None:
    (tmp_path / "small.py").write_text("x = 1\n")
    (tmp_path / "huge.py").write_text("x\n" * 100_000)

    bundle = bundle_workspace(tmp_path, max_file_size_kb=1)

    assert "small.py" in bundle["files"]
    assert "huge.py" not in bundle["files"]


def test_bundle_workspace_compacts_whitespace(tmp_path: Path) -> None:
    (tmp_path / "spaced.py").write_text("def f():\n    pass\n\n\n\n\nx = 1\n")

    bundle = bundle_workspace(tmp_path, max_file_size_kb=100)

    # Runs of more than two blank lines are collapsed to two.
    assert bundle["files"]["spaced.py"] == "def f():\n    pass\n\n\nx = 1\n"


def test_bundle_workspace_reads_test_status(tmp_path: Path) -> None:
    (tmp_path / "tests" / "test_demo.py").parent.mkdir(parents=True, exist_ok=True)
    (tmp_path / "tests" / "test_demo.py").write_text("def test_ok(): pass\n")

    bundle = bundle_workspace(tmp_path, max_file_size_kb=100)

    assert bundle["test_status"] is not None
    assert bundle["test_status"]["status"] == "unknown"
    assert "tests/test_demo.py" in bundle["test_status"]["test_files"]


def test_bundle_workspace_reads_lastfailed_status(tmp_path: Path) -> None:
    lastfailed = tmp_path / ".pytest_cache" / "v" / "cache" / "lastfailed"
    lastfailed.parent.mkdir(parents=True)
    lastfailed.write_text("tests/test_demo.py::test_bad\n")

    bundle = bundle_workspace(tmp_path, max_file_size_kb=100)

    assert bundle["test_status"] == {
        "status": "failed",
        "failed_tests": ["tests/test_demo.py::test_bad"],
    }


def test_bundle_to_xml_serializes_paths(tmp_path: Path) -> None:
    (tmp_path / "blueprint.aero").write_text("project: x\n")
    (tmp_path / "run.py").write_text("print(hello)\n")

    bundle = bundle_workspace(tmp_path, max_file_size_kb=100)
    xml = bundle_to_xml(bundle)

    assert '<workspace path="' in xml
    assert '<blueprint>' in xml
    assert '<file path="blueprint.aero">' in xml
    assert '<file path="run.py">' in xml
    assert "print(hello)" in xml


def test_bundle_to_json_is_round_trip_safe(tmp_path: Path) -> None:
    (tmp_path / "config.toml").write_text("[tool]\nname = 'demo'\n")
    bundle = bundle_workspace(tmp_path, max_file_size_kb=100)
    json_str = bundle_to_json(bundle)
    parsed = json.loads(json_str)
    assert parsed["workspace"] == bundle["workspace"]
    assert parsed["files"] == bundle["files"]


def test_format_context_block_contains_marker(tmp_path: Path) -> None:
    (tmp_path / "main.py").write_text("pass\n")
    bundle = bundle_workspace(tmp_path, max_file_size_kb=100)
    block = format_context_block(bundle, fmt="xml")
    assert "CURRENT_PROJECT_CONTEXT" in block
    assert "<file path=\"main.py\">" in block

    block_json = format_context_block(bundle, fmt="json")
    assert "CURRENT_PROJECT_CONTEXT" in block_json
    assert '"main.py"' in block_json
