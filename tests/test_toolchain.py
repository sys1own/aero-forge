"""Tests for aero_forge.toolchain.ToolchainManager."""

import os
import shutil
from pathlib import Path

import pytest

from aero_forge.toolchain import ToolchainManager


def test_ensure_virtualenv_creates_venv(tmp_path: Path) -> None:
    manager = ToolchainManager(tmp_path)
    env = manager.ensure_virtualenv()
    assert "VIRTUAL_ENV" in env
    assert "PATH" in env
    assert (tmp_path / ".venv").is_dir()
    python_bin = manager._venv_python()
    assert Path(python_bin).is_file()


def test_ensure_python_packages_installs_missing(tmp_path: Path) -> None:
    manager = ToolchainManager(tmp_path)
    # wheel is small and likely not installed in a fresh venv.
    manager.ensure_python_packages(["wheel"])
    result = manager._run_venv_python(["-m", "pip", "show", "wheel"])
    assert result.returncode == 0


def test_environment_variable_injection(tmp_path: Path) -> None:
    manager = ToolchainManager(tmp_path)
    manager.ensure_virtualenv()
    assert manager.env["VIRTUAL_ENV"] == str(tmp_path / ".venv")
    venv_bin = str(manager._venv_bin())
    assert manager.env["PATH"].startswith(venv_bin + os.pathsep)


def test_ensure_rust_toolchain_reports_status(monkeypatch, tmp_path: Path) -> None:
    logs = []
    manager = ToolchainManager(tmp_path, log_callback=lambda level, prefix, msg: logs.append((level, prefix, msg)))

    def fake_which(name):
        if name == "cargo":
            return "/usr/bin/cargo"
        return shutil.which(name)

    monkeypatch.setattr("shutil.which", fake_which)
    assert manager.ensure_rust_toolchain() is True
    assert any(prefix == "TOOLCHAIN" and "Cargo" in msg for _level, prefix, msg in logs)
