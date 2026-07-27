"""Smart healing orchestrator: AST-first, then full-workspace LLM fallback."""

from __future__ import annotations

import difflib
import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from aero_forge.healing.evaluator import HealingStrategy, LogEvaluator
from aero_forge.healing.llm_healer import LLMHealer
from aero_forge.healing.router import try_auto_fix
from aero_forge.healing.structural_merger import MergeConflictError, apply_overlay

logger = logging.getLogger("aero_forge.healing.orchestrator")


class HealingOrchestrator:
    """Route a build/test failure to the right healing strategy and execute it.

    The orchestrator tries a cheap deterministic AST patch first. If the AST
    patch is not applicable or fails, it automatically escalates to a
    full-workspace LLM-driven repair.
    """

    def __init__(
        self,
        workspace: Path,
        llm_provider: str = "deepseek",
        llm_model: Optional[str] = None,
        log_callback: Optional[Callable[[str, str, str], None]] = None,
    ) -> None:
        self.workspace = Path(workspace).resolve()
        self.llm_provider = llm_provider
        self.llm_model = llm_model
        self.log_callback = log_callback or self._default_log

    @staticmethod
    def _default_log(level: str, prefix: str, message: str) -> None:
        log_line = f"[{level.upper()}] {prefix}: {message}"
        if level == "error":
            logger.error(log_line)
        elif level == "warning":
            logger.warning(log_line)
        else:
            logger.info(log_line)

    def heal(
        self,
        error_logs: str,
        command: str = "",
        exit_code: int = 1,
        target_file: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Evaluate *error_logs* and apply the appropriate repair strategy.

        Returns a structured payload with ``status``, ``strategy_used``,
        ``patched_files``, and ``error_message``.
        """
        evaluator = LogEvaluator()
        diagnosis = evaluator.evaluate_log(command, exit_code, error_logs)
        strategy = evaluator.evaluate_error(error_logs, command=command, exit_code=exit_code)

        self.log_callback("info", "HEAL", f"Selected strategy: {strategy.value}")

        if strategy == HealingStrategy.AST:
            ast_result = self._try_ast_heal(error_logs, diagnosis, target_file)
            if ast_result.get("status") == "success":
                return ast_result
            self.log_callback(
                "info",
                "HEAL",
                "AST patch attempt unsuccessful. Escalating to Full-Workspace LLM Heal.",
            )

        if strategy in (HealingStrategy.LLM, HealingStrategy.MANUAL):
            # Even MANUAL cases get an LLM attempt; the model may see patterns
            # the deterministic classifier missed.
            return self._llm_heal(error_logs, command, exit_code, diagnosis)

        return {
            "status": "failed",
            "strategy_used": strategy.value,
            "patched_files": [],
            "error_message": "No repair strategy could be applied.",
        }

    def _try_ast_heal(
        self,
        error_logs: str,
        diagnosis: Dict[str, Any],
        target_file: Optional[str],
    ) -> Dict[str, Any]:
        target = target_file or diagnosis.get("target_file") or "main.py"
        target_path = (self.workspace / target).resolve()
        if not str(target_path).startswith(str(self.workspace)):
            return {
                "status": "failed",
                "strategy_used": HealingStrategy.AST.value,
                "patched_files": [],
                "error_message": "Target file escapes workspace.",
            }
        if not target_path.is_file():
            return {
                "status": "failed",
                "strategy_used": HealingStrategy.AST.value,
                "patched_files": [],
                "error_message": f"Target file not found: {target}",
            }

        original = target_path.read_text(encoding="utf-8")
        try:
            patched = try_auto_fix(error_logs, original)
        except Exception as exc:  # pragma: no cover - defensive
            return {
                "status": "failed",
                "strategy_used": HealingStrategy.AST.value,
                "patched_files": [],
                "error_message": f"AST healer encountered exception: {exc}",
            }

        if patched is None or patched == original:
            return {
                "status": "failed",
                "strategy_used": HealingStrategy.AST.value,
                "patched_files": [],
                "error_message": "AST patch could not be applied cleanly.",
            }

        language = "rust" if target_path.suffix == ".rs" else "python"
        try:
            merged = apply_overlay(original, patched, language=language)
        except (MergeConflictError, SyntaxError) as exc:
            return {
                "status": "failed",
                "strategy_used": HealingStrategy.AST.value,
                "patched_files": [],
                "error_message": f"AST overlay merge failed: {exc}",
            }

        if merged == original:
            return {
                "status": "failed",
                "strategy_used": HealingStrategy.AST.value,
                "patched_files": [],
                "error_message": "AST overlay produced no changes.",
            }

        target_path.write_text(merged, encoding="utf-8")
        self.log_callback("info", "HEAL", f"Applied AST patch to {target}")
        diff = "\n".join(
            difflib.unified_diff(
                original.splitlines(keepends=True),
                merged.splitlines(keepends=True),
                fromfile=target,
                tofile=target,
            )
        )
        return {
            "status": "success",
            "strategy_used": HealingStrategy.AST.value,
            "patched_files": [target],
            "error_message": None,
            "target_file": target,
            "diff": diff,
            "diagnosis": diagnosis,
        }

    def _llm_heal(
        self,
        error_logs: str,
        command: str,
        exit_code: int,
        diagnosis: Dict[str, Any],
    ) -> Dict[str, Any]:
        self.log_callback("info", "HEAL", "Building full-workspace context for LLM healing...")
        failure_context = {
            "command": command,
            "exit_code": exit_code,
            "log_text": error_logs,
            "diagnosis": diagnosis,
        }
        healer = LLMHealer(
            provider=self.llm_provider,
            model=self.llm_model,
            log_callback=self.log_callback,
        )
        result = healer.generate_and_apply_fix(self.workspace, failure_context)

        if result.get("status") == "success":
            return {
                "status": "success",
                "strategy_used": HealingStrategy.LLM.value,
                "patched_files": result.get("applied", []),
                "error_message": None,
                "diagnosis": diagnosis,
            }

        return {
            "status": "failed",
            "strategy_used": HealingStrategy.LLM.value,
            "patched_files": [],
            "error_message": result.get("reason", "Full-workspace LLM healing failed."),
            "diagnosis": diagnosis,
        }
