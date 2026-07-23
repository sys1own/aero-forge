"""LLM rate-limit and API-failure stress tests.

These tests mock the provider SDKs to verify that Aero-Forge retries with
exponential backoff and degrades gracefully when an LLM provider is unavailable.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from openai import RateLimitError

from aero_forge.llm.clients import (
    GeminiClient,
    OpenRouterClient,
    get_llm_client,
)
from aero_forge.orchestrator.orchestrator import Orchestrator


class TestRetryDelayExtraction:
    def test_extract_retry_delay_from_openai_response_header(self):
        class FakeResponse:
            headers = {"Retry-After": "18"}

        class FakeError(Exception):
            response = FakeResponse()

        delay = OpenRouterClient._extract_retry_delay(FakeError())
        assert delay == 18.0

    def test_extract_retry_delay_from_google_rpc_string(self):
        class FakeError(Exception):
            pass

        exc = FakeError(
            "ResourceExhausted 429 Quota exceeded\nretry_delay {\n  seconds: 57\n}"
        )
        delay = OpenRouterClient._extract_retry_delay(exc)
        assert delay == 57.0

    def test_extract_retry_delay_returns_none_for_unknown_errors(self):
        assert OpenRouterClient._extract_retry_delay(ValueError("boom")) is None


class TestOpenRouterRateLimitRetry:
    """OpenRouter is OpenAI-compatible, so rate limits surface as openai.RateLimitError."""

    def test_openrouter_retries_on_rate_limit(self, monkeypatch):
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or")
        client = OpenRouterClient(model="openrouter/free", max_retries=2)

        with patch("openai.OpenAI") as mock_openai:
            create = mock_openai.return_value.chat.completions.create
            create.side_effect = [
                RateLimitError(
                    "rate limited",
                    response=MagicMock(request=MagicMock()),
                    body=None,
                ),
                MagicMock(choices=[MagicMock(message=MagicMock(content="fixed"))]),
            ]
            with patch.object(client, "generate", wraps=client.generate) as _:
                with patch("aero_forge.llm.clients.time.sleep"):
                    result = client.generate("fix this")

        assert result == "fixed"
        assert create.call_count == 2


class TestGeminiRateLimitRetry:
    def test_gemini_retries_on_resource_exhausted(self, monkeypatch):
        monkeypatch.setenv("GEMINI_API_KEY", "sk-gemini")
        client = GeminiClient(model="gemini-2.0-flash", max_retries=2)

        class FakeResourceExhausted(Exception):
            pass

        # Force the Gemini client to treat FakeResourceExhausted as retryable.
        with patch.object(
            client, "_retryable_exceptions", return_value=(FakeResourceExhausted,)
        ):
            with patch("aero_forge.llm.clients.importlib.import_module") as mock_import:
                mock_genai = MagicMock()
                mock_model = MagicMock()
                mock_model.generate_content.side_effect = [
                    FakeResourceExhausted("rate limited"),
                    MagicMock(text="gemini fix"),
                ]
                mock_genai.GenerativeModel.return_value = mock_model
                mock_import.return_value = mock_genai

                with patch("aero_forge.llm.clients.time.sleep"):
                    result = client.generate("fix this")

        assert result == "gemini fix"
        assert mock_model.generate_content.call_count == 2


class TestMissingKeyFallback:
    def test_get_llm_client_returns_none_without_key(self, monkeypatch):
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        monkeypatch.delenv("AERO_FORGE_API_KEY", raising=False)
        assert get_llm_client("openrouter") is None

    def test_orchestrator_runs_router_only_when_llm_unavailable(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        monkeypatch.delenv("AERO_FORGE_API_KEY", raising=False)

        source = tmp_path / "bad.py"
        source.write_text("def f():\n    return 1\n")
        test = tmp_path / "test_bad.py"
        test.write_text("from bad import f\ndef test_f():\n    assert f() == 1\n")

        # No API key and provider openrouter should force fallback to none.
        orch = Orchestrator(
            source_path=source,
            function_name="f",
            test_paths=[test],
            llm_provider="openrouter",
        )
        # Without an API key, get_llm_client returns None and use_llm becomes False.
        assert orch.use_llm is False
        result = orch.run()
        assert result["success"] is True


class TestAPIFailureDoesNotCrash:
    def test_api_failure_returns_partial_result(self, tmp_path):
        source = tmp_path / "bad.py"
        source.write_text("def broken():\n    return 1/\n")
        test = tmp_path / "test_bad.py"
        test.write_text("from bad import broken\ndef test_broken(): pass\n")

        with patch(
            "aero_forge.orchestrator.orchestrator.get_llm_client"
        ) as mock_get_client:
            mock_client = MagicMock()
            mock_client.generate.return_value = None
            mock_get_client.return_value = mock_client

            orch = Orchestrator(
                source_path=source,
                function_name="broken",
                test_paths=[test],
                llm_provider="openrouter",
                max_iterations=2,
            )
            result = orch.run()

        assert result.get("success") is False
        assert result.get("partial") is True
        assert "could not be fixed" in result.get("error", "").lower()
