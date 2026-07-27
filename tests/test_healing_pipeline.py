"""Integration tests for the smart healing pipeline and orchestrator."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from aero_forge.healing.evaluator import HealingStrategy
from aero_forge.healing.orchestrator import HealingOrchestrator


def test_orchestrator_applies_ast_patch_for_python_name_error(tmp_path: Path) -> None:
    source = tmp_path / "main.py"
    source.write_text("print(sqrt(2))\n", encoding="utf-8")
    log = "Traceback (most recent call last):\nNameError: name 'math' is not defined\n"

    orchestrator = HealingOrchestrator(tmp_path)
    result = orchestrator.heal(log, command="python main.py", exit_code=1)

    assert result["status"] == "success"
    assert result["strategy_used"] == HealingStrategy.AST.value
    assert "main.py" in result["patched_files"]
    assert result["error_message"] is None
    assert "import math" in source.read_text(encoding="utf-8")


def test_orchestrator_escalates_to_llm_when_ast_patch_not_applicable(tmp_path: Path) -> None:
    source = tmp_path / "main.py"
    source.write_text("x = 1 / 2\n", encoding="utf-8")
    # A type/semantic error that try_auto_fix cannot patch.
    log = (
        "error[E0308]: mismatched types\n"
        "  --> native_core/src/lib.rs:12:23\n"
        "   |\n"
        "12 |     let x: i64 = 1.0;\n"
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
    assert result["patched_files"] == ["crates/native_core/src/lib.rs"]
    assert result["error_message"] is None
    mock_healer.heal.assert_called_once()


def test_orchestrator_returns_failure_when_llm_produces_no_directives(tmp_path: Path) -> None:
    source = tmp_path / "main.py"
    source.write_text("print('ok')\n", encoding="utf-8")
    log = "error[E0308]: mismatched types\n"

    with patch("aero_forge.healing.orchestrator.LLMHealer") as mock_healer_cls:
        mock_healer = MagicMock()
        mock_healer.generate_and_apply_fix.return_value = {
            "status": "failed",
            "reason": "No repair directives generated.",
        }
        mock_healer_cls.return_value = mock_healer

        orchestrator = HealingOrchestrator(tmp_path)
        result = orchestrator.heal(log, command="cargo test", exit_code=101)

    assert result["status"] == "failed"
    assert result["strategy_used"] == HealingStrategy.LLM.value
    assert result["patched_files"] == []
    assert result["error_message"] is not None


def test_orchestrator_does_not_raise_on_missing_target_file(tmp_path: Path) -> None:
    log = "NameError: name 'math' is not defined\n"
    orchestrator = HealingOrchestrator(tmp_path)
    result = orchestrator.heal(log, command="python main.py", exit_code=1)

    # No exception raised, but AST cannot find the file so it escalates to LLM.
    assert result["status"] in ("success", "failed")
    assert "strategy_used" in result
