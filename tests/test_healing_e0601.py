"""Tests for Cargo E0601 (missing build.rs fn main) self-healing."""

from pathlib import Path

import pytest

from aero_forge.healing.evaluator import LogEvaluator
from aero_forge.healing.healer import DeterministicHealer
from aero_forge.healing.router import try_auto_fix


_ERROR_LOG = """error[E0601]: main function not found in crate build_script_build
  --> build.rs:1:1
   |
1  | // Rust placeholder
   | ^^^^^^^^^^^^^^^^^^^ consider adding a main function

error: could not compile `rust_core` (build script)
"""


def test_log_evaluator_classifies_e0601_as_ast_healable() -> None:
    """LogEvaluator reports E0601 as healable by AST rewrite."""
    diagnosis = LogEvaluator().evaluate_log("cargo build", 1, _ERROR_LOG)
    assert diagnosis["healable"] is True
    assert diagnosis["ast_healable"] is True
    assert diagnosis["target_file"] == "build.rs"


def test_try_auto_fix_rewrites_placeholder_build_rs() -> None:
    """try_auto_fix replaces a placeholder build.rs with a valid fn main."""
    broken_build_rs = "// Rust placeholder\n"
    patched = try_auto_fix(_ERROR_LOG, broken_build_rs)
    assert patched is not None
    assert "fn main()" in patched


def test_deterministic_healer_repairs_build_rs(tmp_path: Path) -> None:
    """DeterministicHealer applies the E0601 patch to disk when apply=True."""
    build_rs = tmp_path / "build.rs"
    build_rs.write_text("// Rust placeholder\n", encoding="utf-8")
    healer = DeterministicHealer(tmp_path)
    result = healer.heal(_ERROR_LOG, target_file="build.rs")
    assert result["status"] == "success"
    assert result["strategy_used"] == "ast"
    assert "build.rs" in result["target_file"]
    assert "fn main()" in build_rs.read_text(encoding="utf-8")
