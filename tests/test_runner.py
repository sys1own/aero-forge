"""Tests for aero_forge.runner toolchain resolution."""

import asyncio
import os
import shutil

import pytest

from aero_forge import runner


def test_ensure_cargo_finds_system_cargo():
    if shutil.which("cargo"):
        assert runner.ensure_cargo() is True


def test_resolve_command_leaves_non_toolchain_command_unchanged():
    env = os.environ.copy()
    resolved = asyncio.run(runner.resolve_command("pytest -q", env))
    assert "pytest" in resolved


def test_resolve_command_detects_missing_cargo_for_maturin(monkeypatch):
    """If neither cargo nor a cargo shim is available, maturin commands fail fast."""
    monkeypatch.setattr("shutil.which", lambda name: None)
    monkeypatch.setattr("aero_forge.runner.ensure_cargo", lambda: False)
    with pytest.raises(RuntimeError, match="Rust toolchain"):
        asyncio.run(runner.resolve_command("maturin develop"))
