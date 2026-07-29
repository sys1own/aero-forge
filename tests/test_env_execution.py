"""Tests for workspace command execution environment preparation."""

import pytest
from aero_forge import runner


def _write_python_package(workspace: object, layout: str) -> None:
    """Create a minimal ``accelerator`` package in *workspace* using *layout*."""
    if layout == "src":
        pkg_root = workspace / "src" / "accelerator"
    else:
        pkg_root = workspace / "accelerator"
    pkg_root.mkdir(parents=True)
    (pkg_root / "__init__.py").write_text("", encoding="utf-8")
    (pkg_root / "cli.py").write_text(
        'def main():\n    print("accelerator ok")\n\nif __name__ == "__main__":\n    main()\n',
        encoding="utf-8",
    )


def _write_pyproject(workspace: object) -> None:
    """Create a minimal ``pyproject.toml`` with setuptools."""
    (workspace / "pyproject.toml").write_text(
        '[build-system]\n'
        'requires = ["setuptools", "wheel"]\n'
        'build-backend = "setuptools.build_meta"\n\n'
        '[project]\n'
        'name = "accelerator"\n'
        'version = "0.1.0"\n\n'
        '[tool.setuptools.packages.find]\n'
        'where = ["src"]\n',
        encoding="utf-8",
    )


@pytest.mark.parametrize("layout", ["src", "flat"])
def test_python_module_execution(tmp_path, layout):
    """``python -m accelerator.cli`` runs without ``ModuleNotFoundError``."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    _write_python_package(workspace, layout)
    _write_pyproject(workspace)

    result = runner.execute("python -m accelerator.cli", sandbox_dir=workspace)
    combined = (result.get("stdout") or "") + (result.get("stderr") or "")
    assert result["returncode"] == 0, combined
    assert "ModuleNotFoundError" not in combined
    assert "accelerator ok" in combined


def test_python_module_without_manifest(tmp_path):
    """``python -m accelerator.cli`` works without a packaging manifest via PYTHONPATH."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    _write_python_package(workspace, "src")

    result = runner.execute("python -m accelerator.cli", sandbox_dir=workspace)
    combined = (result.get("stdout") or "") + (result.get("stderr") or "")
    assert result["returncode"] == 0, combined
    assert "ModuleNotFoundError" not in combined
    assert "accelerator ok" in combined
