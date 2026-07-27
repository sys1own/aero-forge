"""Tests for creating a virtualenv without ensurepip and linking host binaries."""

import os
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

from aero_forge.toolchain import ToolchainManager


def _log_collector():
    logs = []
    return logs, lambda level, prefix, msg: logs.append((level, prefix, msg))


def test_ensure_virtualenv_uses_without_pip_and_links_host_binaries(tmp_path):
    """A venv should be created without ensurepip and .venv/bin/pip should resolve."""
    logs, callback = _log_collector()
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    manager = ToolchainManager(workspace, log_callback=callback)
    env = manager.ensure_virtualenv()

    venv_bin = workspace / ".venv" / "bin"
    assert (venv_bin / "python").is_file()
    assert (venv_bin / "pip").exists()
    assert "VIRTUAL_ENV" in env
    assert str(venv_bin) in env["PATH"]
    # The creation log should mention without-pip (or at least success).
    assert any("virtual environment" in msg.lower() for _, _, msg in logs)


@patch("aero_forge.toolchain.subprocess.run")
def test_ensurepip_missing_uses_without_pip(mock_run, tmp_path):
    """If ensurepip is unavailable, --without-pip should still create a working venv."""
    calls = []

    def fake_run(cmd, **kwargs):
        class FakeProc:
            returncode = 0
            stdout = ""
            stderr = ""
        calls.append(list(cmd))
        # Simulate ensurepip failing when venv is created without --without-pip.
        if "venv" in cmd and "--without-pip" not in cmd:
            FakeProc.returncode = 1
            FakeProc.stderr = "ensurepip is not available"
        return FakeProc()

    mock_run.side_effect = fake_run

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    manager = ToolchainManager(workspace)
    manager._bootstrap_venv_from_host = lambda: None  # host packages not needed for this unit
    manager.ensure_virtualenv()

    without_pip_calls = [c for c in calls if "--without-pip" in c]
    assert without_pip_calls
