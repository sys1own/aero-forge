"""Tests for BlueprintRegenerator workspace rebuild."""

from __future__ import annotations

import ast
import tempfile
from pathlib import Path

import pytest

from aero_forge.errors import UserError
from aero_forge.scaffold.workspace import BlueprintRegenerator


def _make_workspace(tmp_path: Path) -> Path:
    (tmp_path / "src" / "testproj").mkdir(parents=True)
    (tmp_path / "tests").mkdir(parents=True)
    (tmp_path / "src" / "testproj" / "core.py").write_text(
        "def add(a, b)\n    return a + b\n", encoding="utf-8"
    )
    (tmp_path / "tests" / "test_core.py").write_text(
        "def test_add():\n    assert add(1, 2) == 3\n", encoding="utf-8"
    )
    (tmp_path / "blueprint.aero").write_text(
        """project: testproj
architecture: pure_python
manifest:
  - path: pyproject.toml
    lang: toml
    purpose: Python package manifest
  - path: src/testproj/__init__.py
    lang: python
    purpose: Package init
  - path: src/testproj/core.py
    lang: python
    purpose: Core implementation
  - path: tests/test_core.py
    lang: python
    purpose: pytest tests
  - path: README.md
    lang: markdown
contracts:
  - name: add
    signature: "def add(a: float, b: float) -> float:"
    language: python
    python_name: testproj.core.add
""",
        encoding="utf-8",
    )
    return tmp_path


def test_regenerator_requires_blueprint(tmp_path: Path) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    reg = BlueprintRegenerator(empty)
    with pytest.raises(FileNotFoundError):
        reg.run()


def test_regenerator_re_scaffolds_python_workspace(tmp_path: Path) -> None:
    workspace = _make_workspace(tmp_path)
    reg = BlueprintRegenerator(workspace, keep_backup=True, run_build=False, force_overwrite=True)
    result = reg.run()

    assert result["status"] == "success"
    assert result["errors"] == []

    core = workspace / "src" / "testproj" / "core.py"
    init = workspace / "src" / "testproj" / "__init__.py"
    test_file = workspace / "tests" / "test_core.py"
    readme = workspace / "README.md"

    assert core.exists()
    assert init.exists()
    assert test_file.exists()
    assert readme.exists()

    # Generated Python stubs must be syntactically valid.
    for py_file in [core, init, test_file]:
        text = py_file.read_text(encoding="utf-8")
        ast.parse(text)

    # The corrupted core.py was replaced by a stub from the contract.
    assert "def add(a: float, b: float) -> float:" in core.read_text(encoding="utf-8")


def test_regenerator_creates_backup(tmp_path: Path) -> None:
    workspace = _make_workspace(tmp_path)
    original_core = (workspace / "src" / "testproj" / "core.py").read_text(encoding="utf-8")

    reg = BlueprintRegenerator(workspace, keep_backup=True, run_build=False, force_overwrite=True)
    reg.run()

    backup = workspace / ".aero_backup" / "src" / "testproj" / "core.py"
    assert backup.exists()
    assert backup.read_text(encoding="utf-8") == original_core


def _make_uninitialized_workspace(tmp_path: Path) -> Path:
    (tmp_path / "src" / "testproj").mkdir(parents=True)
    (tmp_path / "tests").mkdir(parents=True)
    source = tmp_path / "src" / "testproj" / "core.py"
    source.write_text("def add(a, b):\n    return a + b\n", encoding="utf-8")
    (tmp_path / "blueprint.aero").write_text(
        "metadata:\n  status: draft\n  schema_version: 3.0.0\nmanifest: []\n",
        encoding="utf-8",
    )
    return tmp_path


def test_regenerator_blocks_uninitialized_blueprint(tmp_path: Path) -> None:
    workspace = _make_uninitialized_workspace(tmp_path)
    reg = BlueprintRegenerator(workspace, force_overwrite=True)
    with pytest.raises(UserError):
        reg.run()
    assert (workspace / "src" / "testproj" / "core.py").exists()


def test_regenerator_blocks_uninitialized_without_force_overwrite(tmp_path: Path) -> None:
    workspace = _make_uninitialized_workspace(tmp_path)
    (workspace / "blueprint.aero").write_text(
        "metadata:\n  status: finalized\n  schema_version: 3.0.0\n  generation_method: llm_synthesized\n"
        "manifest:\n  - path: src/testproj/core.py\n    lang: python\n"
        "contracts:\n  - name: add\n    language: python\n    signature: 'def add(a, b):'\n",
        encoding="utf-8",
    )
    reg = BlueprintRegenerator(workspace, force_overwrite=False)
    with pytest.raises(UserError):
        reg.run()
    assert (workspace / "src" / "testproj" / "core.py").exists()
