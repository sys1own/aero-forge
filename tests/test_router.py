"""Tests for the build-intent router."""

import pytest

from aero_forge.orchestrator.router import (
    BUILD_INTENT_HYBRID_RUST_PYTHON,
    BUILD_INTENT_PURE_PYTHON,
    BUILD_INTENT_PURE_RUST,
    classify_build_intent,
    required_manifest_for_intent,
    toolchains_for_intent,
)


@pytest.mark.parametrize(
    "prompt, expected",
    [
        ("Implement a Python service", BUILD_INTENT_PURE_PYTHON),
        ("Sort a list of integers in Python", BUILD_INTENT_PURE_PYTHON),
        ("Implement a Rust crate for parsing", BUILD_INTENT_PURE_RUST),
        ("cargo build a Rust parser", BUILD_INTENT_PURE_RUST),
        ("Build a Python Rust hybrid with PyO3", BUILD_INTENT_HYBRID_RUST_PYTHON),
        ("Create a polyglot orchestration in Python and Rust", BUILD_INTENT_HYBRID_RUST_PYTHON),
        ("Use maturin to expose a Rust core to Python", BUILD_INTENT_HYBRID_RUST_PYTHON),
        ("Build a Python-Rust FFI bridge", BUILD_INTENT_HYBRID_RUST_PYTHON),
        ("Native core Python engine using cargo", BUILD_INTENT_HYBRID_RUST_PYTHON),
        ("Python Rust orchestration build for batch processing", BUILD_INTENT_HYBRID_RUST_PYTHON),
    ],
)
def test_classify_build_intent(prompt: str, expected: str) -> None:
    assert classify_build_intent(prompt) == expected


def test_toolchains_for_intent() -> None:
    assert toolchains_for_intent(BUILD_INTENT_PURE_PYTHON) == ["python"]
    assert toolchains_for_intent(BUILD_INTENT_PURE_RUST) == ["rust", "cargo"]
    assert toolchains_for_intent(BUILD_INTENT_HYBRID_RUST_PYTHON) == ["python", "rust", "cargo"]


def test_required_manifest_for_hybrid() -> None:
    manifest = required_manifest_for_intent(BUILD_INTENT_HYBRID_RUST_PYTHON, "batch_processor")
    paths = {entry["path"] for entry in manifest}
    assert "Cargo.toml" in paths
    assert "rust_core/Cargo.toml" in paths
    assert "rust_core/src/lib.rs" in paths
    assert "python_engine/pyproject.toml" in paths
    assert "python_engine/src/batch_processor/__init__.py" in paths


def test_required_manifest_for_pure_python_is_empty() -> None:
    assert required_manifest_for_intent(BUILD_INTENT_PURE_PYTHON, "foo") == []


def test_required_manifest_for_pure_rust() -> None:
    manifest = required_manifest_for_intent(BUILD_INTENT_PURE_RUST, "foo")
    paths = {entry["path"] for entry in manifest}
    assert "Cargo.toml" in paths
    assert "src/lib.rs" in paths
