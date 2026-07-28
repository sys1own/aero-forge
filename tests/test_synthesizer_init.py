"""Tests for LLMBlueprintSynthesizer client initialization and fallback."""

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from aero_forge.blueprint.schema import BlueprintStatus, BlueprintV3, write_v3_blueprint
from aero_forge.blueprint.synthesizer import LLMBlueprintSynthesizer, synthesize_v3_blueprint


def test_synthesizer_falls_back_to_default_provider(monkeypatch: Any) -> None:
    """When no LLM instance is passed, the synthesizer resolves a client via the factory."""
    fake_client = MagicMock()
    fake_client.generate.return_value = '{"metadata": {"schema_version": "3.0.0", "project_name": "x", "status": "finalized", "generation_method": "llm_synthesized", "transferable": true}, "toolchains": [], "build_pipeline": [], "abi_contracts": [], "execution_strategy": {}, "verification_nodes": []}'

    calls = []

    def fake_get_llm_client(provider, **kwargs):
        calls.append((provider, kwargs.get("model")))
        return fake_client

    monkeypatch.setattr(
        "aero_forge.blueprint.synthesizer.get_llm_client", fake_get_llm_client
    )

    synthesizer = LLMBlueprintSynthesizer()
    assert synthesizer.provider == "deepseek"

    workspace = Path("/tmp/test_workspace")
    workspace.mkdir(parents=True, exist_ok=True)
    try:
        result = synthesizer.synthesize(workspace)
    finally:
        import shutil
        shutil.rmtree(workspace, ignore_errors=True)

    assert result.metadata.status == BlueprintStatus.finalized
    assert result.metadata.transferable is True
    assert calls


def test_synthesizer_raises_when_no_provider_configured(monkeypatch: Any) -> None:
    """synthesize() raises a clear error if no LLM provider can be initialized."""
    monkeypatch.setattr(
        "aero_forge.blueprint.synthesizer.get_llm_client", lambda *a, **k: None
    )

    synthesizer = LLMBlueprintSynthesizer(provider="none")
    workspace = Path("/tmp/test_workspace2")
    workspace.mkdir(parents=True, exist_ok=True)
    try:
        with pytest.raises(ValueError, match="No active LLM provider configured"):
            synthesizer.synthesize(workspace)
    finally:
        import shutil
        shutil.rmtree(workspace, ignore_errors=True)


def test_synthesizer_uses_provided_llm_instance(tmp_path: Path) -> None:
    """Passing an explicit LLM client skips factory resolution."""
    fake_client = MagicMock()
    fake_client.generate.return_value = '{"metadata": {"schema_version": "3.0.0", "project_name": "provided", "status": "finalized", "generation_method": "llm_synthesized", "transferable": true}, "toolchains": [], "build_pipeline": [], "abi_contracts": [], "execution_strategy": {}, "verification_nodes": []}'

    synthesizer = LLMBlueprintSynthesizer(llm=fake_client)
    result = synthesizer.synthesize(tmp_path)

    fake_client.generate.assert_called_once()
    assert result.metadata.status == BlueprintStatus.finalized


def test_synthesize_v3_blueprint_passes_api_key_and_override(monkeypatch: Any, tmp_path: Path) -> None:
    """The convenience function forwards API key and config override to the synthesizer."""
    fake_client = MagicMock()
    fake_client.generate.return_value = '{"metadata": {"schema_version": "3.0.0", "project_name": "x", "status": "finalized", "generation_method": "llm_synthesized", "transferable": true}, "toolchains": [], "build_pipeline": [], "abi_contracts": [], "execution_strategy": {}, "verification_nodes": []}'

    captured = {}

    def fake_get_llm_client(provider, **kwargs):
        captured["provider"] = provider
        captured["api_key"] = kwargs.get("api_key")
        captured["config_override"] = kwargs.get("config_override")
        return fake_client

    monkeypatch.setattr(
        "aero_forge.blueprint.synthesizer.get_llm_client", fake_get_llm_client
    )

    draft = tmp_path / "blueprint.aero"
    write_v3_blueprint(
        BlueprintV3(
            metadata={
                "schema_version": "3.0.0",
                "project_name": "draft",
                "status": BlueprintStatus.draft,
                "generation_method": "static_heuristic",
                "transferable": False,
            }
        ),
        draft,
    )

    output = tmp_path / "final.aero"
    synthesize_v3_blueprint(
        tmp_path,
        output,
        provider="openai",
        model="gpt-4",
        draft_path=draft,
        api_key="sk-test",
        config_override={"dummy": "value"},
    )

    assert captured["provider"] == "openai"
    assert captured["api_key"] == "sk-test"
    assert captured["config_override"] == {"dummy": "value"}
    assert output.is_file()
