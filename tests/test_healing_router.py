"""Tests that healing and generation code routes LLM tasks to the correct tier."""

from unittest.mock import MagicMock, patch

import pytest

from aero_forge.config import Tier
from aero_forge.error_explainer import explain_error
from aero_forge.healing.evaluator import LogEvaluator
from aero_forge.healing.llm_healer import LLMHealer


def test_evaluator_marks_python_syntax_error_as_ast_healable() -> None:
    log = (
        "Traceback (most recent call last):\n"
        '  File "main.py", line 3, in <module>\n'
        "    def foo(\n"
        "SyntaxError: unexpected EOF while parsing\n"
    )
    result = LogEvaluator().evaluate_log("python main.py", 1, log)
    assert result["ast_healable"] is True
    assert result["llm_healable"] is False
    assert "syntax" in result["error_type"].lower()


def test_evaluator_marks_rust_type_error_as_llm_healable() -> None:
    log = (
        "error[E0308]: mismatched types\n"
        "  --> native_core/src/lib.rs:12:23\n"
        "   |\n"
        "12 |     let x: i64 = 1.0;\n"
        "   |                       expected `i64`, found `f64`\n"
    )
    result = LogEvaluator().evaluate_log("cargo test -p native_core", 101, log)
    assert result["ast_healable"] is False
    assert result["llm_healable"] is True
    assert result["code"] == "E0308"


def test_error_explainer_requests_fast_tier(monkeypatch):
    with patch("aero_forge.error_explainer.get_llm_client") as mock_get_client:
        mock_client = MagicMock()
        mock_client.generate.return_value = "You are missing a semicolon."
        mock_get_client.return_value = mock_client
        explain_error("error[E0308]: mismatched types", llm_provider="openai")
    assert mock_get_client.called
    _, kwargs = mock_get_client.call_args
    assert kwargs["tier"] == Tier.FAST


def test_llm_healer_requests_reasoning_tier():
    with patch("aero_forge.healing.llm_healer.get_llm_client") as mock_get_client:
        mock_client = MagicMock()
        mock_client.generate.return_value = "{}"
        mock_get_client.return_value = mock_client
        healer = LLMHealer(provider="deepseek")
        client = healer._get_client()
        assert client is mock_client
    assert mock_get_client.called
    _, kwargs = mock_get_client.call_args
    assert kwargs["tier"] == Tier.REASONING
