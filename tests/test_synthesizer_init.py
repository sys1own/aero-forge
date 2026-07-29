"""Tests for LLMBlueprintSynthesizer client initialization and fallback."""

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from aero_forge.blueprint.schema import BlueprintStatus, BlueprintV3, write_v3_blueprint
from aero_forge.blueprint.synthesizer import (
    LLMBlueprintSynthesizer,
    sanitize_llm_blueprint_output,
    synthesize_v3_blueprint,
)


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


def test_synthesizer_parses_toml_output(tmp_path: Path) -> None:
    """Synthesize a blueprint from TOML-formatted LLM output."""
    fake_client = MagicMock()
    fake_client.generate.return_value = (
        'metadata = { schema_version = "3.0.0", project_name = "toml_project", '
        'status = "finalized", generation_method = "llm_synthesized", transferable = true }\n'
        "toolchains = []\n"
        "build_pipeline = []\n"
        "abi_contracts = []\n"
        "execution_strategy = {}\n"
        "verification_nodes = []\n"
    )

    (tmp_path / "main.py").write_text("print('hello')\n")
    synthesizer = LLMBlueprintSynthesizer(llm=fake_client)
    result = synthesizer.synthesize(tmp_path, output_path=tmp_path / "blueprint.aero")

    assert result.metadata.project_name == "toml_project"
    assert result.metadata.status == BlueprintStatus.finalized
    assert (tmp_path / "blueprint.aero").is_file()


def test_synthesizer_parses_markdown_yaml_output(tmp_path: Path) -> None:
    """Synthesize a blueprint from markdown-wrapped YAML output with scalar lines."""
    raw = (
        "Here is the blueprint you requested:\n"
        "```yaml\n"
        "metadata:\n"
        '  schema_version: "3.0.0"\n'
        '  project_name: "yaml_project"\n'
        '  status: "finalized"\n'
        '  generation_method: "llm_synthesized"\n'
        "  transferable: true\n"
        "toolchains: []\n"
        "build_pipeline: []\n"
        "abi_contracts: []\n"
        "execution_strategy: {}\n"
        "verification_nodes: []\n"
        "```\n"
        "Hope this helps!\n"
    )

    fake_client = MagicMock()
    fake_client.generate.return_value = raw

    (tmp_path / "main.py").write_text("print('hello')\n")
    synthesizer = LLMBlueprintSynthesizer(llm=fake_client)
    result = synthesizer.synthesize(tmp_path, output_path=tmp_path / "blueprint.aero")

    assert result.metadata.project_name == "yaml_project"
    assert result.metadata.status == BlueprintStatus.finalized
    assert (tmp_path / "blueprint.aero").is_file()


def test_synthesizer_fallback_on_garbage_output(tmp_path: Path) -> None:
    """Fallback to a workspace-scanned blueprint when the LLM returns garbage."""
    fake_client = MagicMock()
    fake_client.generate.return_value = "This is complete garbage, not a blueprint."

    (tmp_path / "main.py").write_text("print('hello')\n")
    synthesizer = LLMBlueprintSynthesizer(llm=fake_client)
    result = synthesizer.synthesize(tmp_path, output_path=tmp_path / "blueprint.aero")

    assert (tmp_path / "blueprint.aero").is_file()
    assert result.metadata.status == BlueprintStatus.finalized
    assert result.metadata.transferable is True
    assert result.build_pipeline


def test_sanitize_llm_blueprint_output_parses_toml_and_yaml() -> None:
    """The sanitizer strips fences and parses TOML and YAML variants."""
    toml_text = (
        'project_name = "sanitized"\n'
        "toolchains = []\n"
    )
    assert sanitize_llm_blueprint_output(toml_text)["project_name"] == "sanitized"

    yaml_text = "```yaml\nproject_name: fenced\n```\n"
    assert sanitize_llm_blueprint_output(yaml_text)["project_name"] == "fenced"
