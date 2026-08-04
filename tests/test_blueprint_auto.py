"""Tests for automatic blueprint generation and missing-blueprint warning suppression."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from aero_forge.blueprint import ensure_workspace_blueprint, is_blueprint_ready
from aero_forge.blueprint.schema import BlueprintV3
from aero_forge.blueprint_templates import list_templates


def test_ensure_workspace_blueprint_creates_file_for_empty_directory() -> None:
    """An empty workspace should receive a minimal v3 blueprint from templates."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        path = ensure_workspace_blueprint(root)
        assert path == root / "blueprint.aero"
        assert path.is_file()
        data = BlueprintV3.load(path).model_dump(mode="json")
        assert data["metadata"]["schema_version"] == "3.0.0"
        assert data["metadata"]["llm_initialized"] is False
        assert data["metadata"]["auto_generated"] is True
        assert data["metadata"]["status"] == "draft"


def test_ensure_workspace_blueprint_does_not_overwrite_existing() -> None:
    """If ``blueprint.aero`` already exists, the helper should leave it untouched."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        existing = root / "blueprint.aero"
        existing.write_text("existing\n", encoding="utf-8")
        result = ensure_workspace_blueprint(root)
        assert result == existing
        assert existing.read_text(encoding="utf-8") == "existing\n"


def test_ensure_workspace_blueprint_uses_pure_python_template() -> None:
    """The generated description should be seeded from the pure_python template."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        path = ensure_workspace_blueprint(root)
        bp = BlueprintV3.load(path)
        assert "pure Python" in bp.llm_context.repository_summary.lower() or "numeric" in bp.llm_context.repository_summary.lower()


def test_ensure_workspace_blueprint_detects_rust_workspace() -> None:
    """A workspace with ``Cargo.toml`` should get a rust-oriented blueprint."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "Cargo.toml").write_text("[package]\nname = 'demo'\n", encoding="utf-8")
        path = ensure_workspace_blueprint(root)
        bp = BlueprintV3.load(path)
        assert bp.metadata.project_name
        assert bp.metadata.status.value == "draft"
        assert bp.metadata.auto_generated is True
        assert bp.metadata.llm_initialized is False
        assert bp.metadata.transferable is False
        assert bp.llm_context.state.value == "raw"
        assert bp.build_pipeline
        assert bp.build_pipeline[0].type.value == "cargo_cdylib"
        assert any("src/lib.rs" in sf for sf in bp.build_pipeline[0].source_files)


def test_ensure_workspace_blueprint_detects_python_rust_hybrid() -> None:
    """A workspace with both Cargo.toml and Python files should be a hybrid."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "Cargo.toml").write_text("[package]\nname = 'demo'\n", encoding="utf-8")
        (root / "main.py").write_text("print('ok')\n", encoding="utf-8")
        path = ensure_workspace_blueprint(root)
        bp = BlueprintV3.load(path)
        assert bp.metadata.project_name
        assert bp.metadata.status.value == "draft"
        assert bp.metadata.auto_generated is True
        assert bp.metadata.llm_initialized is False
        assert bp.llm_context.state.value == "raw"
        assert any(t.name.lower() == "rust" for t in bp.toolchains)
        assert any(t.name.lower() in {"python", "cpython"} for t in bp.toolchains)


def test_is_blueprint_ready_for_auto_generated_draft() -> None:
    """An auto-generated empty-workspace draft is not ready until synthesized."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        path = ensure_workspace_blueprint(root)
        assert is_blueprint_ready(path) is False


def test_all_blueprint_templates_are_draft_and_auto_generated() -> None:
    """Starter templates must be valid v3 blueprints marked as auto-generated drafts."""
    for name, path in list_templates().items():
        bp = BlueprintV3.load(path)
        assert bp.metadata.llm_initialized is False, name
        assert bp.metadata.auto_generated is True, name
        assert bp.metadata.status.value == "draft", name
        assert bp.llm_context.state.value == "raw", name
