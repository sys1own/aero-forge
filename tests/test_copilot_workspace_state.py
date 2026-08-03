"""Tests for Copilot workspace-state detection (draft vs active)."""

from pathlib import Path

import pytest

from aero_forge.chat import ChatSession
from aero_forge.context_bundler import get_blueprint_status


# Minimal v3 blueprint helpers.
def _draft_blueprint() -> str:
    return """metadata:
  schema_version: 3.0.0
  project_name: empty_project
  status: draft
  generation_method: static_heuristic
  transferable: true
  llm_initialized: false
  auto_generated: true
  description: Auto-generated minimal blueprint for empty workspace.
llm_context:
  state: raw
  repository_summary: ''
  dependency_graph: {}
  exported_api_signatures: {}
  polyglot_boundaries: []
  compute_hotspots: []
toolchains: []
build_pipeline: []
abi_contracts: []
execution_strategy:
  primary_entrypoint: ''
  runtime: python3
"""


def _active_blueprint() -> str:
    return """metadata:
  schema_version: 3.0.0
  project_name: existing_project
  status: finalized
  generation_method: manual
  transferable: true
  llm_initialized: true
  auto_generated: false
  description: User-uploaded active project.
llm_context:
  state: synthesized
  repository_summary: ''
  dependency_graph: {}
  exported_api_signatures: {}
  polyglot_boundaries: []
  compute_hotspots: []
toolchains:
- name: python
build_pipeline:
- id: core
  type: python_extension
  source_files:
  - src/core.py
abi_contracts: []
execution_strategy:
  primary_entrypoint: src/core.py
  runtime: python3
"""


def test_auto_generated_draft_is_not_llm_initialized(tmp_path: Path) -> None:
    """Auto-generated placeholder blueprints are reported as draft, not active."""
    blueprint_path = tmp_path / "blueprint.aero"
    blueprint_path.write_text(_draft_blueprint(), encoding="utf-8")
    status = get_blueprint_status(tmp_path)
    assert status["exists"] is True
    assert status["auto_generated"] is True
    assert status["llm_initialized"] is False
    assert status["source_count"] == 0


def test_copilot_prompt_blank_workspace_is_draft(tmp_path: Path) -> None:
    """The copilot system prompt marks a blank workspace as draft."""
    (tmp_path / "blueprint.aero").write_text(_draft_blueprint(), encoding="utf-8")
    session = ChatSession(output_dir=tmp_path)
    prompt = session._copilot_system_prompt()
    assert "[WORKSPACE STATE]" in prompt
    assert "draft" in prompt
    assert "fresh project architecture" in prompt or "from scratch" in prompt


def test_copilot_prompt_active_workspace_is_active(tmp_path: Path) -> None:
    """The copilot system prompt marks an existing source tree as active."""
    (tmp_path / "blueprint.aero").write_text(_active_blueprint(), encoding="utf-8")
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "core.py").write_text("def run(): pass\n", encoding="utf-8")
    session = ChatSession(output_dir=tmp_path)
    prompt = session._copilot_system_prompt()
    assert "[WORKSPACE STATE]" in prompt
    assert "active" in prompt
    assert "existing architecture" in prompt
