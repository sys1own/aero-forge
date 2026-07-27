"""Adversarial healing stress tests against a temporary copy of aero-mitosis.

These tests inject Class A-D errors into the scaffolded `crates/native_core`
workspace, mock an LLM client that returns repair directives, and assert that
``LLMHealer`` applies the directives and restores a passing ``cargo test -p
native_core`` run.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

from aero_forge.healing.llm_healer import LLMHealer, run_command
from aero_forge.healing.context_builder import ContextBuilder


class MockLLMClient:
    """OpenAI-compatible mock client that returns deterministic directives."""

    def __init__(self, directives: List[Dict[str, Any]]) -> None:
        self.directives = directives

    def generate(
        self,
        prompt: Any,
        temperature: float = 0.2,
        **kwargs: Any,
    ) -> Optional[str]:
        return json.dumps(
            {
                "diagnosis": "Mock directive from adversarial test harness.",
                "directives": self.directives,
            }
        )


@pytest.fixture(scope="session")
def aero_mitosis_source(tmp_path_factory: Any) -> Path:
    """Return a pre-scaffolded aero-mitosis workspace with native_core built.

    The fixture clones the repo once per session into a shared directory so each
    test can copy it cheaply.
    """
    base = tmp_path_factory.mktemp("aero_mitosis_source")
    target = base / "test-aero-mitosis"
    if not (target / "Cargo.toml").is_file():
        subprocess.run(
            ["git", "clone", "https://github.com/sys1own/aero-mitosis.git", str(target)],
            check=True,
            timeout=120,
        )
        from aero_forge import inspector

        inspector.scaffold_pyo3_workspace(target)
    return target


def _copy_workspace(source: Path, dest: Path) -> None:
    """Copy source workspace while skipping large build artifacts."""
    shutil.copytree(
        source,
        dest,
        ignore=shutil.ignore_patterns(
            ".git",
            "target",
            ".venv",
            "*.egg-info",
            "__pycache__",
        ),
    )


def _read_original(target_file: Path) -> str:
    return target_file.read_text(encoding="utf-8")


def _run_cargo_test(workspace: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["cargo", "test", "-p", "native_core"],
        cwd=str(workspace),
        capture_output=True,
        text=True,
        timeout=300,
    )


@pytest.mark.slow
@pytest.mark.skipif(not shutil.which("cargo"), reason="cargo not installed")
def test_class_a_type_mismatch_e0308(aero_mitosis_source: Path, tmp_path: Path) -> None:
    """Class A: inject an E0308 type mismatch and repair via LLM directive."""
    workspace = tmp_path / "class_a"
    _copy_workspace(aero_mitosis_source, workspace)
    target = workspace / "crates" / "native_core" / "src" / "lib.rs"
    original = _read_original(target)

    # Inject a type mismatch: finalize returns an integer instead of String.
    corrupted = original.replace(
        "slf.inner.finalize().to_hex().to_string()",
        "42",
    )
    target.write_text(corrupted, encoding="utf-8")

    proc = _run_cargo_test(workspace)
    assert proc.returncode != 0, "expected build failure after corruption"
    assert "E0308" in proc.stderr

    healer = LLMHealer(client=MockLLMClient([{
        "target_file": "crates/native_core/src/lib.rs",
        "action": "rewrite",
        "reason": "Revert E0308 type mismatch.",
        "instructions": "Restore original finalize signature and return String.",
        "content": original,
    }]))
    result = healer.generate_and_apply_fix(
        workspace,
        {
            "command": "cargo test -p native_core",
            "exit_code": proc.returncode,
            "log_text": proc.stdout + proc.stderr,
        },
    )
    assert result["status"] == "success"
    assert target.read_text(encoding="utf-8") == original

    rerun = run_command("cargo test -p native_core", workspace)
    assert rerun["exit_code"] == 0, rerun["output"]


@pytest.mark.slow
@pytest.mark.skipif(not shutil.which("cargo"), reason="cargo not installed")
def test_class_b_missing_trait_e0277(aero_mitosis_source: Path, tmp_path: Path) -> None:
    """Class B: remove a trait-derived operation that produces a trait bound error."""
    workspace = tmp_path / "class_b"
    _copy_workspace(aero_mitosis_source, workspace)
    target = workspace / "crates" / "native_core" / "src" / "lib.rs"
    original = _read_original(target)

    # Inject a generic helper that requires a `Sync` bound the PyO3 PyRef does not satisfy.
    corrupted = original.replace(
        "use std::collections::{HashMap, HashSet, VecDeque};",
        "use std::collections::{HashMap, HashSet, VecDeque};\n\nfn assert_sync<T: Sync>(_: &T) {}",
    ).replace(
        "fn finalize<'py>(slf: PyRef<'py, Self>) -> String {",
        "fn finalize<'py>(slf: PyRef<'py, Self>) -> String {\n        assert_sync(&slf);",
    )
    target.write_text(corrupted, encoding="utf-8")

    proc = _run_cargo_test(workspace)
    assert proc.returncode != 0, "expected build failure after corruption"
    assert "E0277" in proc.stderr or "trait" in proc.stderr.lower()

    healer = LLMHealer(client=MockLLMClient([{
        "target_file": "crates/native_core/src/lib.rs",
        "action": "rewrite",
        "reason": "Restore trait-correct finalize implementation.",
        "instructions": "Remove the invalid assert_sync call and helper to restore compilation.",
        "content": original,
    }]))
    result = healer.generate_and_apply_fix(
        workspace,
        {
            "command": "cargo test -p native_core",
            "exit_code": proc.returncode,
            "log_text": proc.stdout + proc.stderr,
        },
    )
    assert result["status"] == "success"
    rerun = run_command("cargo test -p native_core", workspace)
    assert rerun["exit_code"] == 0, rerun["output"]


@pytest.mark.slow
@pytest.mark.skipif(not shutil.which("cargo"), reason="cargo not installed")
def test_class_c_pyo3_binding_mismatch(aero_mitosis_source: Path, tmp_path: Path) -> None:
    """Class C: corrupt a PyO3 module export registration."""
    workspace = tmp_path / "class_c"
    _copy_workspace(aero_mitosis_source, workspace)
    target = workspace / "crates" / "native_core" / "src" / "lib.rs"
    original = _read_original(target)

    # Reference a non-existent PyO3 function so module registration fails.
    corrupted = original.replace(
        "m.add_function(wrap_pyfunction!(hash_bytes, m)?)?;",
        "m.add_function(wrap_pyfunction!(hash_by, m)?)?;",
    )
    target.write_text(corrupted, encoding="utf-8")

    proc = _run_cargo_test(workspace)
    assert proc.returncode != 0

    healer = LLMHealer(client=MockLLMClient([{
        "target_file": "crates/native_core/src/lib.rs",
        "action": "rewrite",
        "reason": "Restore PyO3 module function export.",
        "instructions": "Use the correct hash_bytes function name in wrap_pyfunction!.",
        "content": original,
    }]))
    result = healer.generate_and_apply_fix(
        workspace,
        {
            "command": "cargo test -p native_core",
            "exit_code": proc.returncode,
            "log_text": proc.stdout + proc.stderr,
        },
    )
    assert result["status"] == "success"
    rerun = run_command("cargo test -p native_core", workspace)
    assert rerun["exit_code"] == 0, rerun["output"]


@pytest.mark.slow
@pytest.mark.skipif(not shutil.which("cargo"), reason="cargo not installed")
def test_class_d_dependency_conflict(aero_mitosis_source: Path, tmp_path: Path) -> None:
    """Class D: introduce a conflicting exact blake3 version in Cargo.toml."""
    workspace = tmp_path / "class_d"
    _copy_workspace(aero_mitosis_source, workspace)
    cargo_toml = workspace / "crates" / "native_core" / "Cargo.toml"
    original_cargo = _read_original(cargo_toml)
    corrupted = original_cargo.replace(
        'blake3 = { version = "=1.5.3", features = ["rayon", "mmap"] }',
        'blake3 = { version = "=1.5.4", features = ["rayon", "mmap"] }',
    )
    cargo_toml.write_text(corrupted, encoding="utf-8")

    proc = _run_cargo_test(workspace)
    assert proc.returncode != 0
    assert "failed to select a version" in proc.stderr or "cc" in proc.stderr

    lib_rs = workspace / "crates" / "native_core" / "src" / "lib.rs"
    original_lib = _read_original(lib_rs)

    healer = LLMHealer(client=MockLLMClient([
        {
            "target_file": "crates/native_core/Cargo.toml",
            "action": "update_manifest",
            "reason": "Pin blake3 to a toolchain-compatible exact version.",
            "instructions": "Set blake3 version to '=1.5.3'.",
            "content": original_cargo,
        },
        {
            "target_file": "crates/native_core/src/lib.rs",
            "action": "rewrite",
            "reason": "Ensure source file is restored.",
            "instructions": "Write original lib.rs content.",
            "content": original_lib,
        },
    ]))
    result = healer.generate_and_apply_fix(
        workspace,
        {
            "command": "cargo test -p native_core",
            "exit_code": proc.returncode,
            "log_text": proc.stdout + proc.stderr,
        },
    )
    assert result["status"] == "success"
    rerun = run_command("cargo test -p native_core", workspace)
    assert rerun["exit_code"] == 0, rerun["output"]


def test_context_builder_includes_failure_and_workspace() -> None:
    """Unit test: ContextBuilder aggregates references and affected files."""
    from aero_forge.healing.evaluator import LogEvaluator

    evaluator = LogEvaluator()
    log = """
error[E0308]: mismatched types
   --> crates/native_core/src/lib.rs:55:9
    |
55  |     let x: i64 = 1.0;
    |                  ^^^ expected `i64`, found `f64`
"""
    diagnosis = evaluator.evaluate_log("cargo test -p native_core", 101, log)
    ctx = ContextBuilder(Path("/tmp/fake_workspace")).build_failure_context(
        "cargo test -p native_core", 101, log, diagnosis
    )
    assert ctx["command"] == "cargo test -p native_core"
    assert ctx["exit_code"] == 101
    assert any(ref.get("file") == "crates/native_core/src/lib.rs" for ref in ctx["references"])
    assert "crates/native_core/src/lib.rs" in ctx["affected_files"]


def test_llm_healer_validates_directive_schema() -> None:
    """Unit test: LLMHealer rejects malformed directives from the LLM."""
    class BadClient:
        def generate(self, prompt: Any, **kwargs: Any) -> str:
            return json.dumps({"directives": [{"target_file": "x"}]})

    healer = LLMHealer(client=BadClient())
    result = healer.generate_and_apply_fix(
        Path("/tmp/fake_workspace_does_not_exist"),
        {"command": "cargo test", "exit_code": 1, "log_text": "error"},
    )
    assert result["status"] == "failed"
    assert "directive" in result["reason"].lower() or "schema" in result["reason"].lower()
