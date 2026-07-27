"""Tests for ToolchainManager resilience and maturin fallback paths."""

import os
import shutil
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

from aero_forge.toolchain import ToolchainManager


def test_prefers_host_maturin_over_pip(tmp_path, monkeypatch):
    """When a host maturin binary exists, it should be linked into the venv."""
    fake_bin = tmp_path / "fake_bin"
    fake_bin.mkdir()
    host_maturin = fake_bin / "maturin"
    host_maturin.write_text("#!/bin/sh\necho 'maturin 1.0.0'\n", encoding="utf-8")
    host_maturin.chmod(0o755)
    monkeypatch.setenv("PATH", str(fake_bin) + os.pathsep + os.environ.get("PATH", ""))

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    manager = ToolchainManager(workspace)
    manager.ensure_maturin()

    venv_maturin = workspace / ".venv" / "bin" / "maturin"
    assert venv_maturin.exists()
    assert venv_maturin.is_symlink() or venv_maturin.is_file()


@patch("aero_forge.toolchain.shutil.which")
@patch("aero_forge.toolchain.subprocess.run")
def test_pip_failure_streams_diagnostics(mock_run, mock_which, tmp_path):
    """A failing pip install should log stderr and fall back without an unhandled exception."""
    mock_which.return_value = None  # no host maturin, no cargo

    def fake_run(cmd, **kwargs):
        class FakeProc:
            returncode = 1
            stdout = ""
            stderr = "ERROR: Could not find a version that satisfies the requirement maturin"
        is_venv = len(cmd) >= 3 and cmd[1] == "-m" and cmd[2] == "venv"
        is_bootstrap = "pip" in str(cmd) and "--upgrade" in str(cmd)
        if is_venv or is_bootstrap:
            FakeProc.returncode = 0
            FakeProc.stderr = ""
        return FakeProc()

    mock_run.side_effect = fake_run

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    logs = []
    manager = ToolchainManager(workspace, log_callback=lambda level, prefix, msg: logs.append((level, prefix, msg)))

    # ensure_maturin should return False, not raise, after logging the diagnostic.
    result = manager.ensure_maturin()
    assert result is False

    assert any(
        "pip install maturin failed" in msg or "Could not find a version" in msg
        for level, prefix, msg in logs
        if level == "error"
    )


def test_resolve_command_falls_back_to_cargo_when_maturin_unavailable(tmp_path, monkeypatch):
    """If maturin cannot be provisioned, a maturin command is rewritten to cargo."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    manager = ToolchainManager(workspace)

    # Force a situation where maturin is unavailable.
    manager._maturin_available = False
    manager._prepared = True

    resolved = manager.resolve_command("maturin develop")
    assert resolved.startswith("cargo")

    resolved_test = manager.resolve_command("maturin develop --test")
    assert "cargo test" in resolved_test


@patch("aero_forge.toolchain.shutil.which")
@patch("aero_forge.toolchain.subprocess.run")
def test_cargo_install_fallback_for_maturin(mock_run, mock_which, tmp_path):
    """If pip install maturin fails but cargo is present, try cargo install."""
    mock_which.side_effect = lambda name, path=None: (name == "cargo")

    calls = []

    def fake_run(cmd, **kwargs):
        class FakeProc:
            returncode = 0
            stdout = ""
            stderr = ""
        calls.append(str(cmd))
        if "pip" in str(cmd) and "maturin" in str(cmd):
            FakeProc.returncode = 1
            FakeProc.stderr = "Network is unreachable"
        return FakeProc()

    mock_run.side_effect = fake_run

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    manager = ToolchainManager(workspace)
    result = manager.ensure_maturin()

    assert result is True
    assert any("cargo" in c and "install" in c and "maturin" in c for c in calls)
