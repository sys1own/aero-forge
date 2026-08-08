"""Tests for tier-aware LLM model routing and configuration."""

import json

import pytest

from aero_forge.config import DEFAULT_TIER_MODELS, Tier, resolve_tier_model
from aero_forge.llm import OpenAIClient, get_llm_client


@pytest.mark.parametrize(
    "provider,fast_model,reasoning_model",
    [
        ("deepseek", "deepseek-v4-flash", "deepseek-chat"),
        ("openai", "gpt-4o-mini", "gpt-4o"),
        ("gemini", "gemini-2.5-flash", "gemini-2.5-pro"),
        ("openrouter", "anthropic/claude-3-haiku", "anthropic/claude-3.5-sonnet"),
    ],
)
def test_default_tier_models(provider: str, fast_model: str, reasoning_model: str) -> None:
    assert DEFAULT_TIER_MODELS[provider]["fast"] == fast_model
    assert DEFAULT_TIER_MODELS[provider]["reasoning"] == reasoning_model
    assert resolve_tier_model(provider, Tier.FAST) == fast_model
    assert resolve_tier_model(provider, Tier.REASONING) == reasoning_model


def test_aero_llm_tier_fast_env_json_override(monkeypatch):
    monkeypatch.setenv("AERO_LLM_TIER_FAST", json.dumps({"deepseek": "deepseek-chat"}))
    assert resolve_tier_model("deepseek", "fast") == "deepseek-chat"
    # Reasoning tier stays on its default.
    assert resolve_tier_model("deepseek", "reasoning") == "deepseek-chat"


def test_aero_llm_tier_fast_env_raw_model_for_provider(monkeypatch):
    monkeypatch.setenv(
        "AERO_LLM_TIER_FAST", json.dumps({"openai": "gpt-4o-mini-custom"})
    )
    assert resolve_tier_model("openai", "fast") == "gpt-4o-mini-custom"
    assert resolve_tier_model("deepseek", "fast") == "deepseek-v4-flash"


def test_aero_llm_tier_reasoning_env_json_override(monkeypatch):
    monkeypatch.setenv(
        "AERO_LLM_TIER_REASONING",
        json.dumps({"deepseek": {"reasoning": "deepseek-reasoner"}}),
    )
    assert resolve_tier_model("deepseek", "reasoning") == "deepseek-reasoner"


def test_config_file_tier_models_override(tmp_path):
    cfg_path = tmp_path / "accelerate.toml"
    cfg_path.write_text('[tier_models]\nopenai_fast = "gpt-4o-mini-custom"\n')
    from aero_forge.config import load_config

    file_config = load_config(cfg_path)
    assert resolve_tier_model("openai", "fast", file_config=file_config) == "gpt-4o-mini-custom"


def test_get_llm_client_uses_fast_default(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai")
    monkeypatch.delenv("AERO_FORGE_MODEL", raising=False)
    monkeypatch.delenv("AERO_LLM_TIER_FAST", raising=False)
    monkeypatch.delenv("AERO_LLM_TIER_REASONING", raising=False)
    client = get_llm_client("openai")
    assert isinstance(client, OpenAIClient)
    assert client.model == "gpt-4o-mini"


def test_get_llm_client_uses_reasoning_tier(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai")
    monkeypatch.delenv("AERO_FORGE_MODEL", raising=False)
    monkeypatch.delenv("AERO_LLM_TIER_FAST", raising=False)
    monkeypatch.delenv("AERO_LLM_TIER_REASONING", raising=False)
    client = get_llm_client("openai", tier="reasoning")
    assert client.model == "gpt-4o"


def test_get_llm_client_explicit_model_pinned(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai")
    monkeypatch.setenv("AERO_FORGE_MODEL", "gpt-4-turbo")
    monkeypatch.delenv("AERO_LLM_TIER_FAST", raising=False)
    monkeypatch.delenv("AERO_LLM_TIER_REASONING", raising=False)
    client = get_llm_client("openai")
    assert client.model == "gpt-4-turbo"
    # Tier switches should not override an explicit (pinned) model.
    assert client._resolve_model("fast") == "gpt-4-turbo"
    assert client._resolve_model("reasoning") == "gpt-4-turbo"
