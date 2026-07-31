"""Deterministic self-healing orchestrator backed by native proof-theoretic engines.

``DeterministicHealer.execute_healing_pass`` runs a strictly deterministic
repair pipeline:

1. HIN graph energy evaluation of the failing source.
2. Static AST rewrites (``aero_forge.healing.router``).
3. E-graph equality-saturation rewriting of UAST expressions.
4. FFI morphism synthesis fallback for missing cross-language contracts.
5. Geometry-of-Interaction matrix perturbation validation.

No LLM API calls are made inside this loop.
"""

from __future__ import annotations

import difflib
import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from aero_forge._native import (
    HinEngine,
    enforce_repair_isolation_py,
    evaluate_hin_energy,
    repair_uast_expression,
)
from aero_forge.healing.router import try_auto_fix
from aero_forge.scaffold.contract_synth import ContractSynthesizer
from aero_forge.translator.aero_frontend import python_source_to_uast

logger = logging.getLogger("aero_forge.healing.healer")


def _recursively_repair_expressions(uast: Any) -> Any:
    """Walk a UAST dict/list and apply e-graph rewriting to every expression node."""
    if isinstance(uast, list):
        return [_recursively_repair_expressions(item) for item in uast]
    if not isinstance(uast, dict):
        return uast

    node_type = uast.get("type", "")
    if node_type in {"literal", "reference", "call", "binop", "unaryop"}:
        original = json.dumps(uast)
        try:
            rewritten = repair_uast_expression(original)
        except Exception:
            return uast
        try:
            parsed = json.loads(rewritten)
        except json.JSONDecodeError:
            return uast
        if parsed != uast:
            return parsed
        return uast

    return {k: _recursively_repair_expressions(v) for k, v in uast.items()}


def _build_hin_arena(source_text: str) -> Optional[str]:
    """Lower Python source to a HIN arena JSON description."""
    try:
        uast = python_source_to_uast(source_text)
    except Exception as exc:
        logger.debug("python_source_to_uast failed: %s", exc)
        return None
    try:
        engine = HinEngine()
        engine.build_from_json(json.dumps(uast))
        return engine.to_json()
    except Exception as exc:
        logger.debug("HinEngine build failed: %s", exc)
        return None


def _extract_target_file(error_log: str, workspace: Path) -> Optional[str]:
    """Return a likely target file path extracted from a compiler/test traceback."""
    # Python traceback file paths.
    m = re.search(r'File "([^"]+)", line', error_log)
    if m:
        raw = Path(m.group(1))
        candidates = [raw, workspace / raw]
        for path in candidates:
            if path.is_file():
                try:
                    return str(path.relative_to(workspace))
                except ValueError:
                    return str(path)
    # Rust/Cargo error paths: "--> path:line:col"
    m = re.search(r"-->\s+(\S+):\d+:\d+", error_log)
    if m:
        raw = Path(m.group(1))
        candidates = [raw, workspace / raw]
        for path in candidates:
            if path.is_file():
                try:
                    return str(path.relative_to(workspace))
                except ValueError:
                    return str(path)
    return None


def _extract_missing_symbol(error_log: str) -> Optional[str]:
    """Return the name referenced in a NameError/undefined reference."""
    patterns = [
        r"NameError: name ['\"](\w+)['\"] is not defined",
        r"undefined reference to [`\"]?(\w+)[`\"]?",
        r"cannot find (?:function|value|symbol) ['\"]?(\w+)['\"]?",
    ]
    for pat in patterns:
        m = re.search(pat, error_log)
        if m:
            return m.group(1)
    return None


class DeterministicHealer:
    """Proof-theoretic build/test repair orchestrator.

    The healer operates on source text, UAST expressions, and blueprint contracts
    without ever calling an LLM.
    """

    def __init__(
        self,
        workspace: Path,
        contract_synthesizer: Optional[ContractSynthesizer] = None,
        log_callback: Optional[Any] = None,
    ) -> None:
        self.workspace = Path(workspace).resolve()
        self.contract_synthesizer = contract_synthesizer
        self.log_callback = log_callback

    def _log(self, level: str, prefix: str, message: str) -> None:
        if self.log_callback:
            self.log_callback(level, prefix, message)
        getattr(logger, level.lower(), logger.info)("[%s] %s", prefix, message)

    def execute_healing_pass(
        self,
        error_log: str,
        source_text: Optional[str] = None,
        source_path: Optional[Path] = None,
        command: str = "",
        exit_code: int = 1,
        uast_json: Optional[str] = None,
        expression_json: Optional[str] = None,
        base_matrix: Optional[Dict[str, Any]] = None,
        delta_matrix: Optional[Dict[str, Any]] = None,
        apply: bool = False,
    ) -> Dict[str, Any]:
        """Run the full deterministic repair pipeline and return a structured result.

        The result contains at least ``status``, ``strategy_used``,
        ``patched_files``, and ``error_message``.  Additional keys include
        ``energy``, ``expression_patch``, ``fallback_wrappers``, and ``goi_result``.
        """
        result: Dict[str, Any] = {
            "status": "failed",
            "strategy_used": None,
            "patched_files": [],
            "error_message": "No deterministic repair applicable.",
        }

        # ------------------------------------------------------------------
        # (d) GoI boundary perturbation validation.  Run first because it may veto
        # an unsafe repair plan before any source mutation.
        # ------------------------------------------------------------------
        if base_matrix is not None and delta_matrix is not None:
            try:
                goi_result = json.loads(
                    enforce_repair_isolation_py(
                        json.dumps(base_matrix), json.dumps(delta_matrix)
                    )
                )
                result["goi_result"] = goi_result
                if not goi_result.get("isolated", True):
                    result["error_message"] = (
                        f"GoI repair boundary violated (radius={goi_result.get('radius')} "
                        f">= bound={goi_result.get('bound')})."
                    )
                    return result
            except Exception as exc:
                self._log("warning", "HEAL", f"GoI validation skipped: {exc}")

        # ------------------------------------------------------------------
        # (a) HIN graph energy evaluation.
        # ------------------------------------------------------------------
        arena: Optional[str] = None
        if uast_json:
            try:
                engine = HinEngine()
                engine.build_from_json(uast_json)
                arena = engine.to_json()
            except Exception as exc:
                self._log("debug", "HEAL", f"HIN build failed: {exc}")
        elif source_text:
            arena = _build_hin_arena(source_text)

        if arena:
            try:
                energy = json.loads(evaluate_hin_energy(arena))
                result["energy"] = energy
                self._log(
                    "info",
                    "HEAL",
                    f"HIN energy: stalled={energy.get('stalled')}, "
                    f"wires={energy.get('wires')}, dangling={energy.get('dangling')}, "
                    f"total={energy.get('total')}",
                )
            except Exception as exc:
                self._log("debug", "HEAL", f"HIN energy evaluation failed: {exc}")

        # ------------------------------------------------------------------
        # (b) Static AST rewrites.
        # ------------------------------------------------------------------
        if source_text:
            try:
                patch = try_auto_fix(error_log, source_text)
                if patch is not None and patch != source_text:
                    target = str(source_path or "source.py")
                    result["status"] = "success"
                    result["strategy_used"] = "ast"
                    result["patch"] = patch
                    result["target_file"] = target
                    result["diff"] = "".join(
                        difflib.unified_diff(
                            source_text.splitlines(keepends=True),
                            patch.splitlines(keepends=True),
                            fromfile=target,
                            tofile=target,
                        )
                    )
                    result["error_message"] = None
                    if apply and source_path:
                        resolved = self.workspace / source_path
                        resolved.write_text(patch, encoding="utf-8")
                        result["patched_files"] = [target]
                    else:
                        result["patched_files"] = [target]
                    return result
            except Exception as exc:
                self._log("warning", "HEAL", f"AST rewrite failed: {exc}")

        # ------------------------------------------------------------------
        # (b2) E-Graph equality saturation rewriting of a single UAST expression.
        # ------------------------------------------------------------------
        if expression_json:
            try:
                rewritten = repair_uast_expression(expression_json)
                parsed = json.loads(rewritten)
                original = json.loads(expression_json)
                if parsed != original:
                    result["status"] = "success"
                    result["strategy_used"] = "egraph_rewrite"
                    result["expression_patch"] = parsed
                    result["error_message"] = None
                    return result
            except Exception as exc:
                self._log("debug", "HEAL", f"E-graph rewrite failed: {exc}")

        # Walk the whole module UAST and rewrite any expressions inside.
        if uast_json:
            try:
                uast = json.loads(uast_json)
                repaired = _recursively_repair_expressions(uast)
                if repaired != uast:
                    result["uast_repaired"] = repaired
                    result["status"] = "success"
                    result["strategy_used"] = "egraph_rewrite"
                    result["error_message"] = None
                    # No source patch available from UAST alone; caller must materialise.
                    return result
            except Exception as exc:
                self._log("debug", "HEAL", f"UAST e-graph rewrite failed: {exc}")

        # ------------------------------------------------------------------
        # (c) FFI morphism synthesis fallback.
        # ------------------------------------------------------------------
        missing_symbol = _extract_missing_symbol(error_log)
        if missing_symbol and self.contract_synthesizer:
            try:
                wrappers: Dict[str, Any] = {}
                contract = self.contract_synthesizer.contracts.get(missing_symbol)
                if contract:
                    # Emit all supported bindings for the missing symbol.
                    for pair in ("python/rust", "rust/cpp", "rust/rust"):
                        wrappers[pair] = self.contract_synthesizer.synthesize_missing_morphism(
                            missing_symbol, pair
                        )
                    result["status"] = "success"
                    result["strategy_used"] = "ffi_synth"
                    result["fallback_wrappers"] = wrappers
                    result["error_message"] = None
                    return result
            except Exception as exc:
                self._log("debug", "HEAL", f"FFI synthesis failed: {exc}")

        target_file = _extract_target_file(error_log, self.workspace)
        if target_file:
            result["target_file"] = target_file

        return result

    def heal(
        self,
        error_logs: str,
        command: str = "",
        exit_code: int = 1,
        target_file: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Convenience wrapper compatible with the legacy HealingOrchestrator API.

        Reads the source from *target_file* (or extracts it from the error log)
        under the workspace and dispatches to ``execute_healing_pass``.
        """
        target_file = target_file or _extract_target_file(error_logs, self.workspace)
        if target_file:
            path = self.workspace / target_file
            try:
                source = path.read_text(encoding="utf-8")
            except OSError:
                source = None
            return self.execute_healing_pass(
                error_log=error_logs,
                source_text=source,
                source_path=Path(target_file),
                command=command,
                exit_code=exit_code,
                apply=True,
            )

        return self.execute_healing_pass(
            error_log=error_logs,
            command=command,
            exit_code=exit_code,
        )
