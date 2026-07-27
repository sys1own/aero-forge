"""Tests for aero_forge.runner toolchain resolution."""

import asyncio
import os
import shutil
from pathlib import Path

import pytest

from aero_forge import runner


def test_ensure_cargo_finds_system_cargo():
    if shutil.which("cargo"):
        assert runner.ensure_cargo() is True


def test_resolve_command_leaves_non_toolchain_command_unchanged(tmp_path: Path):
    env = os.environ.copy()
    resolved, out_env, logs = asyncio.run(
        runner.resolve_command("pytest -q", env, tmp_path)
    )
    assert "pytest" in resolved
    assert "VIRTUAL_ENV" in out_env
    assert any(prefix == "ENV" for _level, prefix, _msg in logs)


def test_resolve_command_falls_back_when_maturin_unavailable(monkeypatch, tmp_path: Path):
    """If maturin cannot be provisioned, rewrite to a cargo build fallback."""
    monkeypatch.setattr("shutil.which", lambda name, path=None: None)
    monkeypatch.setattr("aero_forge.toolchain.ensure_cargo_in_path", lambda: None)
    resolved, out_env, logs = asyncio.run(
        runner.resolve_command("maturin develop", os.environ.copy(), tmp_path)
    )
    assert resolved.startswith("cargo")
    assert any("maturin" in msg or "cargo" in msg for _level, prefix, msg in logs)
