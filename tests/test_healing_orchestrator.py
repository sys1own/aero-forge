"""Tests for the smart healing orchestrator's AST-to-LLM fallback chain."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from aero_forge.healing.evaluator import HealingStrategy
from aero_forge.healing.orchestrator import HealingOrchestrator


def test_ast_failure_escalates_to_llm_healer(tmp_path: Path) -> None:
    """If the deterministic AST patch cannot fix the error, LLM healer is invoked."""
    source = tmp_path / "main.py"
    source.write_text("x = unknown_func()\n", encoding="utf-8")
    # NameError for a non-stdlib name -> AST selected but try_auto_fix returns None.
    log = "Traceback (most recent call last):\nNameError: name 'unknown_func' is not defined\n"

    with patch("aero_forge.healing.orchestrator.LLMHealer") as mock_healer_cls:
        mock_healer = MagicMock()
        mock_healer.heal.return_value = {
            "status": "success",
            "applied": ["main.py"],
        }
        mock_healer_cls.return_value = mock_healer

        orchestrator = HealingOrchestrator(tmp_path)
        result = orchestrator.heal(log, command="python main.py", exit_code=1)

    assert result["status"] == "success"
    assert result["strategy_used"] == HealingStrategy.LLM.value
    assert result["patched_files"] == ["main.py"]
    assert mock_healer.heal.called
    # Ensure the call was made with reasoning-tier and full-workspace flags.
    _, kwargs = mock_healer.heal.call_args
    assert kwargs.get("tier") == "reasoning"
    assert kwargs.get("full_workspace") is True
    assert kwargs.get("force_full_rewrite") is True


def test_force_llm_bypasses_ast_and_calls_llm_healer(tmp_path: Path) -> None:
    """force_llm=True skips AST and routes directly to the full-workspace LLM healer."""
    source = tmp_path / "main.py"
    source.write_text("x = 1\n", encoding="utf-8")
    log = "NameError: name 'math' is not defined\n"

    with patch("aero_forge.healing.orchestrator.LLMHealer") as mock_healer_cls:
        mock_healer = MagicMock()
        mock_healer.heal.return_value = {
            "status": "success",
            "applied": ["main.py"],
        }
        mock_healer_cls.return_value = mock_healer

        orchestrator = HealingOrchestrator(tmp_path)
        result = orchestrator.heal(log, command="python main.py", exit_code=1, force_llm=True)

    assert result["status"] == "success"
    assert result["strategy_used"] == HealingStrategy.LLM.value
    assert mock_healer.heal.called


def test_orchestrator_returns_failure_when_both_ast_and_llm_fail(tmp_path: Path) -> None:
    """If neither AST nor LLM can repair, return a coherent failure (not a dead-end)."""
    source = tmp_path / "main.py"
    source.write_text("x = unknown()\n", encoding="utf-8")
    log = "NameError: name 'unknown' is not defined\n"

    with patch("aero_forge.healing.orchestrator.LLMHealer") as mock_healer_cls:
        mock_healer = MagicMock()
        mock_healer.heal.return_value = {
            "status": "failed",
            "reason": "No directives generated.",
        }
        mock_healer_cls.return_value = mock_healer

        orchestrator = HealingOrchestrator(tmp_path)
        result = orchestrator.heal(log, command="python main.py", exit_code=1)

    assert result["status"] == "failed"
    assert result["strategy_used"] == HealingStrategy.LLM.value
    assert "Both AST and Full-Workspace LLM" not in result["error_message"]
    assert "No directives generated" in result["error_message"]


def test_llm_strategy_errors_skip_ast_and_call_llm_healer(tmp_path: Path) -> None:
    """Errors classified as LLM-healable go directly to the LLM healer."""
    log = (
        "error[E0308]: mismatched types\n"
        "  --> crates/native_core/src/lib.rs:12:23\n"
    )

    with patch("aero_forge.healing.orchestrator.LLMHealer") as mock_healer_cls:
        mock_healer = MagicMock()
        mock_healer.heal.return_value = {
            "status": "success",
            "applied": ["crates/native_core/src/lib.rs"],
        }
        mock_healer_cls.return_value = mock_healer

        orchestrator = HealingOrchestrator(tmp_path)
        result = orchestrator.heal(log, command="cargo test -p native_core", exit_code=101)

    assert result["status"] == "success"
    assert result["strategy_used"] == HealingStrategy.LLM.value
    assert mock_healer.heal.called
