"""Tests for Cargo workspace manifest registration and native cargo commands."""

import shutil
import subprocess

import pytest

from aero_forge.bundle_repo import scaffold_native_crate
from aero_forge.inspector import inspect_workspace, scaffold_pyo3_workspace
from aero_forge.scaffold.cargo_manifest import ensure_workspace_cargo_toml


def test_ensure_workspace_cargo_toml_creates_workspace(tmp_path):
    ensure_workspace_cargo_toml(tmp_path)
    manifest = tmp_path / "Cargo.toml"
    assert manifest.is_file()
    text = manifest.read_text(encoding="utf-8")
    assert "[workspace]" in text
    assert '"crates/native_core"' in text
    assert 'resolver = "2"' in text


def test_ensure_workspace_cargo_toml_appends_member(tmp_path):
    (tmp_path / "Cargo.toml").write_text('[workspace]\nmembers = ["crates/old"]\n', encoding="utf-8")
    ensure_workspace_cargo_toml(tmp_path)
    text = (tmp_path / "Cargo.toml").read_text(encoding="utf-8")
    assert '"crates/old"' in text
    assert '"crates/native_core"' in text


def test_scaffold_pyo3_workspace_registers_workspace(tmp_path):
    scaffold_pyo3_workspace(tmp_path, project_name="test-native")
    assert (tmp_path / "Cargo.toml").is_file()
    assert "[workspace]" in (tmp_path / "Cargo.toml").read_text(encoding="utf-8")
    assert (tmp_path / "crates" / "native_core" / "Cargo.toml").is_file()


@pytest.mark.slow
@pytest.mark.skipif(not shutil.which("cargo"), reason="cargo not installed")
def test_cargo_test_workspace_succeeds(tmp_path):
    scaffold_native_crate(tmp_path, project_name="test-native")
    ensure_workspace_cargo_toml(tmp_path)
    proc = subprocess.run(
        ["cargo", "test", "-p", "native_core"],
        cwd=str(tmp_path),
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert proc.returncode == 0, f"cargo test failed:\n{proc.stdout}\n{proc.stderr}"


def test_inspector_uses_workspace_package_commands(tmp_path):
    scaffold_pyo3_workspace(tmp_path, project_name="test-native")
    commands = inspect_workspace(tmp_path)
    cmds = [c["cmd"] for c in commands]
    assert "cargo test -p native_core" in cmds
    assert "cargo build -p native_core" in cmds
    assert "cargo test --workspace" in cmds
    assert "cargo build --workspace" in cmds
    assert "cargo test --manifest-path crates/native_core/Cargo.toml" not in cmds
