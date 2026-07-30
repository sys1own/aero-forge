"""Tests for aero_forge.environment.env_manager workspace pip operations."""

import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from aero_forge.environment import env_manager


def test_install_workspace_editable_runs_pip_in_workspace(tmp_path: Path):
    """Editable install is executed with cwd set to the workspace root."""
    workspace = tmp_path / "session"
    workspace.mkdir()
    (workspace / "pyproject.toml").write_text(
        '[build-system]\nrequires = ["setuptools", "wheel"]\n'
        'build-backend = "setuptools.build_meta"\n\n'
        '[project]\nname = "demo"\nversion = "0.1.0"\n',
        encoding="utf-8",
    )
    (workspace / "src" / "demo").mkdir(parents=True)
    (workspace / "src" / "demo" / "__init__.py").write_text("", encoding="utf-8")

    recorded = {}

    def fake_run(cmd, **kwargs):
        recorded["cmd"] = cmd
        recorded["cwd"] = kwargs.get("cwd")
        recorded["env"] = kwargs.get("env")
        class FakeProc:
            returncode = 0
            stdout = "Successfully installed demo-0.1.0"
            stderr = ""
        return FakeProc()

    with patch("aero_forge.environment.env_manager.subprocess.run", side_effect=fake_run):
        proc = env_manager.install_workspace_editable(
            workspace, sys.executable, env={"PYTHONPATH": "/tmp"}
        )

    assert proc.returncode == 0
    assert recorded["cmd"][:3] == [sys.executable, "-m", "pip"]
    assert "install" in recorded["cmd"]
    assert "-e" in recorded["cmd"]
    assert Path(recorded["cwd"]) == workspace
    assert recorded["env"]["PYTHONPATH"] == "/tmp"


def test_install_workspace_editable_skips_without_manifest(tmp_path: Path):
    """When no packaging manifest exists, no pip invocation is attempted."""
    workspace = tmp_path / "session"
    workspace.mkdir()

    with patch("aero_forge.environment.env_manager.subprocess.run") as mock_run:
        proc = env_manager.install_workspace_editable(workspace, sys.executable)

    mock_run.assert_not_called()
    assert proc.returncode == 0
    assert "No packaging manifest" in proc.stderr


def test_run_pip_in_workspace_uses_cwd(tmp_path: Path):
    """``run_pip_in_workspace`` always uses the workspace as cwd."""
    workspace = tmp_path / "session"
    workspace.mkdir()

    captured = {}

    def fake_run(cmd, **kwargs):
        captured["cwd"] = kwargs.get("cwd")
        class FakeProc:
            returncode = 0
            stdout = ""
            stderr = ""
        return FakeProc()

    with patch("aero_forge.environment.env_manager.subprocess.run", side_effect=fake_run):
        env_manager.run_pip_in_workspace(
            ["install", "pytest"], workspace, sys.executable
        )

    assert Path(captured["cwd"]) == workspace
