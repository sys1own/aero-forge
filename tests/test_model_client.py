"""Tests for the multi-provider LLM client with retry/fallback."""

from unittest.mock import MagicMock, patch

import pytest

from aero_forge.orchestrator.model_client import ModelClient


@pytest.fixture
def client():
    return ModelClient(models=["gpt-4", "openrouter/free"], max_retries=2)


def test_complete_returns_first_success(client):
    with patch("openai.OpenAI") as mock_openai:
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="fixed"))]
        mock_openai.return_value.chat.completions.create.return_value = mock_response

        result = client.complete([{"role": "user", "content": "fix"}])
        assert result == "fixed"
        assert mock_openai.return_value.chat.completions.create.call_count == 1


def _make_response():
    resp = MagicMock()
    resp.request = MagicMock()
    return resp


def test_complete_retries_on_rate_limit_then_succeeds(client):
    from openai import RateLimitError

    with patch("openai.OpenAI") as mock_openai:
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="ok"))]
        create = mock_openai.return_value.chat.completions.create
        create.side_effect = [
            RateLimitError("rate limited", response=_make_response(), body=None),
            mock_response,
        ]

        with patch("aero_forge.orchestrator.model_client.time.sleep"):
            result = client.complete([{"role": "user", "content": "fix"}])
        assert result == "ok"
        assert create.call_count == 2


def test_complete_falls_back_to_next_model(client):
    from openai import AuthenticationError

    with patch("openai.OpenAI") as mock_openai:
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="fallback"))]
        create = mock_openai.return_value.chat.completions.create
        create.side_effect = [
            AuthenticationError("bad key", response=_make_response(), body=None),
            mock_response,
        ]

        result = client.complete([{"role": "user", "content": "fix"}])
        assert result == "fallback"
        assert create.call_count == 2


def test_complete_returns_none_when_all_models_fail(client):
    from openai import APIConnectionError

    with patch("openai.OpenAI") as mock_openai:
        mock_openai.return_value.chat.completions.create.side_effect = (
            APIConnectionError(message="down", request=MagicMock())
        )

        with patch("aero_forge.orchestrator.model_client.time.sleep"):
            result = client.complete([{"role": "user", "content": "fix"}])
        assert result is None
