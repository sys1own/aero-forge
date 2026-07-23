"""Tests for the provider-agnostic LLM client module."""

from unittest.mock import MagicMock, patch

import pytest

from aero_forge.llm import (
    GeminiClient,
    LLMError,
    OpenAIClient,
    OpenRouterClient,
    get_llm_client,
)


class TestGetLLMClient:
    def test_none_provider_returns_none(self):
        assert get_llm_client("none") is None
        assert get_llm_client(None) is None
        assert get_llm_client("") is None

    def test_unknown_provider_returns_none(self, monkeypatch, caplog):
        import logging

        with caplog.at_level(logging.ERROR):
            assert get_llm_client("unknown") is None
        assert "Unknown LLM provider" in caplog.text

    def test_openai_missing_key_returns_none(self, monkeypatch, caplog):
        import logging

        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("AERO_FORGE_API_KEY", raising=False)
        with caplog.at_level(logging.ERROR):
            assert get_llm_client("openai") is None
        assert "OPENAI_API_KEY or AERO_FORGE_API_KEY is not set" in caplog.text

    def test_openrouter_missing_key_returns_none(self, monkeypatch, caplog):
        import logging

        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        monkeypatch.delenv("AERO_FORGE_API_KEY", raising=False)
        with caplog.at_level(logging.ERROR):
            assert get_llm_client("openrouter") is None
        assert "OPENROUTER_API_KEY or AERO_FORGE_API_KEY is not set" in caplog.text

    def test_gemini_missing_package_raises_import_error(self, monkeypatch, caplog):
        import logging

        # No google-generativeai in test environment, so this will first fail on import.
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        monkeypatch.delenv("AERO_FORGE_API_KEY", raising=False)
        with pytest.raises(ImportError, match="google-generativeai"):
            get_llm_client("gemini")

    def test_openai_uses_provider_key(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-openai")
        client = get_llm_client("openai", model="gpt-4o")
        assert isinstance(client, OpenAIClient)
        assert client.model == "gpt-4o"
        assert client.api_key == "sk-openai"

    def test_openai_falls_back_to_aero_forge_key(self, monkeypatch):
        monkeypatch.setenv("AERO_FORGE_API_KEY", "sk-generic")
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        client = get_llm_client("openai")
        assert isinstance(client, OpenAIClient)
        assert client.api_key == "sk-generic"

    def test_openrouter_defaults_and_key(self, monkeypatch):
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or")
        client = get_llm_client("openrouter")
        assert isinstance(client, OpenRouterClient)
        assert client.model == "openrouter/free"
        assert client.api_key == "sk-or"

    def test_openrouter_falls_back_to_aero_forge_key(self, monkeypatch):
        monkeypatch.setenv("AERO_FORGE_API_KEY", "sk-generic")
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        client = get_llm_client("openrouter", model="openai/gpt-4")
        assert isinstance(client, OpenRouterClient)
        assert client.model == "openai/gpt-4"
        assert client.api_key == "sk-generic"

    def test_model_env_override(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-openai")
        monkeypatch.setenv("AERO_FORGE_MODEL", "gpt-3.5-turbo")
        client = get_llm_client("openai")
        assert client.model == "gpt-3.5-turbo"

    def test_model_argument_overrides_env(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-openai")
        monkeypatch.setenv("AERO_FORGE_MODEL", "gpt-3.5-turbo")
        client = get_llm_client("openai", model="gpt-4o")
        assert client.model == "gpt-4o"


class TestOpenAIClient:
    def _make_response(self, content: str):
        response = MagicMock()
        response.choices = [MagicMock(message=MagicMock(content=content))]
        return response

    def test_generate_returns_content(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-openai")
        client = OpenAIClient(model="gpt-4", api_key="sk-openai")
        with patch("openai.OpenAI") as mock_openai:
            mock_openai.return_value.chat.completions.create.return_value = self._make_response(
                "fixed"
            )
            result = client.generate("fix this")
        assert result == "fixed"

    def test_generate_retries_on_rate_limit(self, monkeypatch):
        from openai import RateLimitError

        monkeypatch.setenv("OPENAI_API_KEY", "sk-openai")
        client = OpenAIClient(model="gpt-4", api_key="sk-openai", max_retries=2)
        with patch("openai.OpenAI") as mock_openai:
            create = mock_openai.return_value.chat.completions.create
            create.side_effect = [
                RateLimitError(
                    "rate limited",
                    response=MagicMock(request=MagicMock()),
                    body=None,
                ),
                self._make_response("ok"),
            ]
            with patch("aero_forge.llm.clients.time.sleep"):
                result = client.generate([{"role": "user", "content": "fix"}])
        assert result == "ok"
        assert create.call_count == 2

    def test_generate_missing_key_raises(self):
        client = OpenAIClient(model="gpt-4", api_key=None)
        with pytest.raises(LLMError, match="OpenAI API key not found"):
            client.generate("prompt")


class TestOpenRouterClient:
    def test_generate_uses_openrouter_base_url_and_key(self, monkeypatch):
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or")
        client = OpenRouterClient(model="openrouter/free")
        with patch("openai.OpenAI") as mock_openai:
            response = MagicMock()
            response.choices = [MagicMock(message=MagicMock(content="fallback"))]
            create = mock_openai.return_value.chat.completions.create
            create.return_value = response
            result = client.generate("fix")
        assert result == "fallback"
        assert mock_openai.call_args.kwargs["base_url"] == "https://openrouter.ai/api/v1"
        assert mock_openai.call_args.kwargs["api_key"] == "sk-or"


class TestGeminiClient:
    def test_missing_package_raises_clear_error(self, monkeypatch):
        # Ensure the package is not importable.
        monkeypatch.setenv("GEMINI_API_KEY", "sk-gemini")
        client = GeminiClient(model="gemini-2.0-flash", api_key="sk-gemini")
        with patch(
            "aero_forge.llm.clients.importlib.import_module",
            side_effect=ImportError("No module named 'google'"),
        ):
            with pytest.raises(LLMError, match="google-generativeai"):
                client.generate("prompt")

    def test_generate_converts_messages_to_string(self, monkeypatch):
        monkeypatch.setenv("GEMINI_API_KEY", "sk-gemini")
        client = GeminiClient(model="gemini-2.0-flash", api_key="sk-gemini")

        mock_genai = MagicMock()
        mock_model = MagicMock()
        mock_model.generate_content.return_value.text = "gemini result"
        mock_genai.GenerativeModel.return_value = mock_model
        mock_genai.configure = MagicMock()

        with patch("aero_forge.llm.clients.importlib.import_module", return_value=mock_genai):
            result = client.generate(
                [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "Fix this function."},
                ]
            )

        assert result == "gemini result"
        mock_genai.configure.assert_called_once_with(api_key="sk-gemini")
        mock_genai.GenerativeModel.assert_called_once_with("gemini-2.0-flash")
        prompt = mock_model.generate_content.call_args.args[0]
        assert "System instruction:" in prompt
        assert "Fix this function." in prompt
