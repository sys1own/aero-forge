"""Tests for JIT blueprint synthesis in the chat endpoint."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import yaml

from aero_forge.blueprint.schema import (
    BlueprintStatus,
    BlueprintV3,
    ContextState,
    GenerationMethod,
    LLMContext,
    Metadata,
    write_v3_blueprint,
)
from aero_forge.chat import ChatSession, WorkspaceContextHarvester
from aero_forge.server import AeroForgeHandler


@pytest.fixture
def raw_workspace(tmp_path: Path) -> Path:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "main.py").write_text(
        "def compute(x):\n    return x * x\n", encoding="utf-8"
    )
    bp = BlueprintV3(
        metadata=Metadata(
            schema_version="3.0.0",
            project_name="raw_demo",
            status=BlueprintStatus.draft,
            generation_method=GenerationMethod.static_heuristic,
            transferable=False,
        ),
        llm_context=LLMContext(
            state=ContextState.raw,
            repository_summary="",
            dependency_graph={},
            compute_hotspots=[],
        ),
    )
    (tmp_path / "blueprint.aero").write_text(
        yaml.safe_dump(bp.model_dump(mode="json"), sort_keys=False), encoding="utf-8"
    )
    return tmp_path


def test_workspace_context_harvester_reads_llm_context(tmp_path: Path) -> None:
    (tmp_path / "app.py").write_text("def main(): pass\n", encoding="utf-8")
    bp = BlueprintV3(
        metadata=Metadata(project_name="ctx_test"),
        llm_context=LLMContext(
            state=ContextState.synthesized,
            repository_summary="A demo project",
            dependency_graph={"app.py": ["utils.py"]},
            compute_hotspots=[{"name": "main", "file": "app.py", "complexity": "O(n)", "reason": "loops"}],
        ),
    )
    (tmp_path / "blueprint.aero").write_text(
        yaml.safe_dump(bp.model_dump(mode="json"), sort_keys=False), encoding="utf-8"
    )

    harvester = WorkspaceContextHarvester(tmp_path)
    prompt = harvester.to_prompt()
    assert "CURRENT_PROJECT_CONTEXT" in prompt
    assert "Repository summary: A demo project" in prompt
    assert "app.py -> utils.py" in prompt
    assert "main (app.py) complexity=O(n)" in prompt
    assert '<file path="app.py">' in prompt


def test_chat_session_uses_harvester_context(tmp_path: Path) -> None:
    (tmp_path / "app.py").write_text("def main(): pass\n", encoding="utf-8")
    session = ChatSession(tmp_path)
    assert session.project_context is not None
    assert "CURRENT_PROJECT_CONTEXT" in session.system_prompt
    assert "app.py" in session.system_prompt


def test_synthesize_if_raw_triggers_and_updates_blueprint(raw_workspace: Path) -> None:
    blueprint_path = raw_workspace / "blueprint.aero"
    fake_blueprint = BlueprintV3(
        metadata=Metadata(
            schema_version="3.0.0",
            project_name="raw_demo",
            status=BlueprintStatus.finalized,
            generation_method=GenerationMethod.llm_synthesized,
            transferable=True,
        ),
        llm_context=LLMContext(
            state=ContextState.synthesized,
            repository_summary="Synthesized summary",
            dependency_graph={"src/main.py": []},
            compute_hotspots=[],
        ),
    )

    def _fake_synthesize(workspace, draft=None, output_path=None, spec=None):
        if output_path:
            write_v3_blueprint(fake_blueprint, output_path)
        return fake_blueprint

    fake_synthesizer = MagicMock()
    fake_synthesizer.return_value.synthesize.side_effect = _fake_synthesize

    handler = SimpleNamespace()

    config = SimpleNamespace(
        llm_provider=None,
        api_key=None,
        model=None,
        max_retries=3,
    )

    with patch("aero_forge.server.LLMBlueprintSynthesizer", fake_synthesizer):
        result = AeroForgeHandler._synthesize_if_raw(handler, raw_workspace, config)

    assert result is True
    assert fake_synthesizer.return_value.synthesize.called
    updated = BlueprintV3.load(blueprint_path)
    assert updated.llm_context.state == ContextState.synthesized
    assert updated.llm_context.repository_summary == "Synthesized summary"
