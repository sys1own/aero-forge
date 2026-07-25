"""Tests for ``aero_forge.scaffold.rust_merge``."""

import subprocess
from pathlib import Path
from typing import List

import pytest

from aero_forge.scaffold.rust_merge import (
    fix_rust_core_impls,
    merge_rust_impl_blocks,
)


def _write_minimal_workspace(tmp_path: Path, lib_rs: str) -> Path:
    rust_core = tmp_path / "rust_core"
    src = rust_core / "src"
    src.mkdir(parents=True)
    cargo_toml = rust_core / "Cargo.toml"
    cargo_toml.write_text(
        "[package]\n"
        "name = \"rust_core\"\n"
        "version = \"0.1.0\"\n"
        "edition = \"2021\"\n\n"
        "[lib]\n"
        'crate-type = ["cdylib"]\n'
        "name = \"rust_core\"\n\n"
        "[dependencies]\n"
        'pyo3 = { version = "0.20.3", features = ["extension-module", "abi3-py39", "generate-import-lib"] }\n'
    )
    (rust_core / ".cargo").mkdir(exist_ok=True)
    (rust_core / ".cargo" / "config.toml").write_text(
        "[net]\nretry = 5\n"
    )
    src.joinpath("lib.rs").write_text(lib_rs)
    return rust_core


def test_merge_duplicate_impl_blocks(tmp_path: Path) -> None:
    source = """use pyo3::prelude::*;

#[pyclass]
pub struct Counter {
    value: i64,
}

impl Counter {
    fn new(value: i64) -> Self {
        Counter { value }
    }

    fn inc(&mut self) {
        self.value += 1;
    }
}

#[pymethods]
impl Counter {
    #[new]
    fn py_new(value: i64) -> Self {
        Counter::new(value)
    }

    fn inc(&mut self) {
        self.inc();
    }
}

#[pymodule]
fn rust_core(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<Counter>()?;
    Ok(())
}
"""
    fixed = merge_rust_impl_blocks(source)
    assert "fn _py_inc" in fixed
    assert '#[pyo3(name = "inc")]' in fixed
    assert "fn inc(&mut self)" in fixed
    # The duplicate method name should no longer appear twice in the same file.
    assert fixed.count("fn inc(&mut self)") == 1


def test_fix_rust_core_impls_compiles(tmp_path: Path) -> None:
    source = """use pyo3::prelude::*;
use std::collections::VecDeque;

#[pyclass]
pub struct RollingBuffer {
    data: VecDeque<(f64, f64)>,
    capacity: usize,
}

impl RollingBuffer {
    fn new(capacity: usize) -> Self {
        RollingBuffer { data: VecDeque::with_capacity(capacity), capacity }
    }

    fn update(&mut self, _timestamp: f64, _value: f64) {
        self.data.push_back((_timestamp, _value));
    }
}

#[pymethods]
impl RollingBuffer {
    #[new]
    fn py_new(capacity: usize) -> Self {
        RollingBuffer::new(capacity)
    }

    fn update(&mut self, timestamp: f64, value: f64) {
        self.update(timestamp, value);
    }
}

#[pymodule]
fn rust_core(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<RollingBuffer>()?;
    Ok(())
}
"""
    rust_core = _write_minimal_workspace(tmp_path, source)
    assert fix_rust_core_impls(tmp_path)
    result = subprocess.run(
        ["cargo", "build", "--release"],
        cwd=rust_core,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
