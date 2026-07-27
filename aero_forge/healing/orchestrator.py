"""Smart healing orchestrator: AST-first, then full-workspace LLM fallback."""

from __future__ import annotations

import difflib
import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from aero_forge.bundle_repo import bundle_workspace
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

    def _attempts_path(self) -> Path:
        return self.workspace / ".aero" / "healing_attempts.json"

    def _load_attempts(self) -> List[Dict[str, Any]]:
        try:
            data = self._attempts_path().read_text(encoding="utf-8")
        except FileNotFoundError:
            return []
        try:
            attempts = json.loads(data)
            if isinstance(attempts, list):
                return attempts
        except json.JSONDecodeError:
            pass
        return []

    def _record_attempt(self, error_digest: str, strategy: str, success: bool) -> None:
        path = self._attempts_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        attempts = self._load_attempts()
        attempts.append({
            "error_digest": error_digest,
            "strategy": strategy,
            "success": success,
        })
        # Keep the last 50 attempts to bound file growth.
        attempts = attempts[-50:]
        try:
            path.write_text(json.dumps(attempts, indent=2), encoding="utf-8")
        except OSError as exc:
            logger.warning("Could not persist healing attempts: %s", exc)

    def _error_digest(self, error_logs: str, command: str, exit_code: int) -> str:
        return hashlib.sha256(
            f"{command}:{exit_code}:{error_logs}".encode("utf-8")
        ).hexdigest()

    def _has_failed_attempt(self, error_digest: str, strategy: str) -> bool:
        return any(
            a.get("error_digest") == error_digest
            and a.get("strategy") == strategy
            and a.get("success") is False
            for a in self._load_attempts()
        )

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
        force_llm: bool = False,
    ) -> Dict[str, Any]:
        """Evaluate *error_logs* and apply the appropriate repair strategy.

        Returns a structured payload with ``status``, ``strategy_used``,
        ``patched_files``, and ``error_message``.

        AST-only errors are patched with a deterministic overlay. If the AST
        patch fails or ``force_llm`` is set, the orchestrator escalates to a
        full-workspace LLM healer using bundled project context. Failed
        attempts are recorded so the exact same strategy is not retried
        indefinitely for an unchanged error.
        """
        evaluator = LogEvaluator()
        diagnosis = evaluator.evaluate_log(command, exit_code, error_logs)
        strategy = evaluator.evaluate_error(error_logs, command=command, exit_code=exit_code)

        digest = self._error_digest(error_logs, command, exit_code)
        self.log_callback("info", "HEAL", f"Selected strategy: {strategy.value}; digest={digest[:16]}")

        if not force_llm and strategy == HealingStrategy.AST and not self._has_failed_attempt(digest, "ast"):
            ast_result = self._try_ast_heal(error_logs, diagnosis, target_file)
            if ast_result.get("status") == "success":
                return ast_result
            self._record_attempt(digest, "ast", success=False)
            self.log_callback(
                "info",
                "HEAL",
                "AST patch attempt unsuccessful. Escalating to Full-Workspace LLM Heal.",
            )

        if force_llm and self._has_failed_attempt(digest, "llm"):
            return {
                "status": "failed",
                "strategy_used": None,
                "patched_files": [],
                "error_message": "LLM healing was already attempted and failed for this error. Manual fix required.",
                "attempts_exhausted": True,
            }

        if not self._has_failed_attempt(digest, "llm"):
            llm_result = self._llm_heal(error_logs, command, exit_code, diagnosis)
            self._record_attempt(digest, "llm", success=llm_result.get("status") == "success")
            return llm_result

        return {
            "status": "failed",
            "strategy_used": None,
            "patched_files": [],
            "error_message": "Both AST and full-workspace LLM healing were already attempted and failed. Manual fix required.",
            "attempts_exhausted": True,
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
        workspace_context = bundle_workspace(self.workspace)
        healer = LLMHealer(
            provider=self.llm_provider,
            model=self.llm_model,
            log_callback=self.log_callback,
        )
        result = healer.heal(
            self.workspace,
            error_logs,
            command=command,
            exit_code=exit_code,
            diagnosis=diagnosis,
            tier="reasoning",
            full_workspace=True,
            workspace_context=workspace_context,
            force_full_rewrite=True,
        )

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
