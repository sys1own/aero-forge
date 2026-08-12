"""End-to-end PyO3 Rust matmul synthesis and verification."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from aero_forge.blueprint import Blueprint, ContractEntry, FunctionalIntent
from aero_forge.orchestrator.stack_classifier import INTENT_HYBRID_RUST_PYTHON
from aero_forge.scaffold.polyglot_materializer import PolyglotMaterializer

PROMPT = (
    "Add a Rust extension module src/matmul.rs. "
    "Implement a PyO3 fn matmul(a: &PyArray2<f64>, b: &PyArray2<f64>) "
    "-> PyResult<&PyArray2<f64>> using the numpy crate."
)


def test_pyo3_matmul_compiles_and_matches_numpy_dot(tmp_path: Path) -> None:
    """The materializer builds a PyO3 matmul module that matches numpy.dot."""
    workspace = tmp_path / "workspace"
    blueprint = Blueprint(
        project="matmul_test",
        architecture=INTENT_HYBRID_RUST_PYTHON,
        toolchains=["python", "rust"],
        prompt=PROMPT,
        metadata={"llm_initialized": "true"},
        functional_intent=[FunctionalIntent(symbol_name="matmul", type="function")],
        contracts=[ContractEntry(name="matmul", signature="", language="rust")],
    )

    materializer = PolyglotMaterializer(workspace)
    updated = materializer.materialize(blueprint, build=True, force_overwrite=True)

    assert (workspace / "Cargo.toml").is_file()
    assert (workspace / "src" / "lib.rs").is_file()
    assert (workspace / "src" / "matmul.rs").is_file()
    assert (workspace / "tests" / "test_matmul.py").is_file()
    lib_rs = (workspace / "src" / "lib.rs").read_text()
    assert "mod matmul;" in lib_rs
    assert "use matmul::_accel_matmul;" in lib_rs
    assert "m.add_wrapped(wrap_pyfunction!(_accel_matmul))" in lib_rs

    # Run the generated test suite against numpy.dot in an isolated process.
    env = os.environ.copy()
    env["PYTHONPATH"] = str(workspace / "src")
    env["TMPDIR"] = os.environ.get("TMPDIR", "/var/tmp")
    result = subprocess.run(
        [sys.executable, "-m", "pytest", str(workspace / "tests"), "-q"],
        env=env,
        capture_output=True,
        text=True,
    )
    assert (
        result.returncode == 0
    ), f"Generated tests failed:\n{result.stdout}\n{result.stderr}"
    assert "passed" in result.stdout
