"""Three-tier fallback remediation for the proactive polyglot build pipeline."""

from __future__ import annotations

import ast
import copy
import json
import re
from typing import Any, Dict, List, Optional, Tuple

from aero_forge.errors import HeuristicWarning

__all__ = ["FallbackManager", "HeuristicWarning"]


class _ReturnTypeRepairer:
    """Rewrite Rust return statements so the body's realized type matches the signature."""

    @staticmethod
    def repair(source: str, symbol: str, declared: str) -> str:
        """Cast every return expression in *symbol* to *declared*."""
        from aero_forge.builder.proactive_synthesis import CoreVerificationPipeline

        return CoreVerificationPipeline._rewrite_return_cast(source, symbol, declared)


class _CollectionAstRepairer(ast.NodeTransformer):
    """Rewrite common Python collection idioms into HIN-compatible forms.

    Transformations performed:

    * ``dict_obj.get(key)`` -> ``dict_obj[key]`` (maps to ``dict_lookup`` UAST node).
    * ``dict_obj.get(key, default)`` -> ``dict_obj[key]`` with a diagnostic note;
      the HIN ``dict_lookup`` agent returns ``Null`` for missing keys, which callers
      can treat as a sentinel value.
    * ``dict(k=v, ...)`` -> ``{"k": v, ...}`` (maps to the ``dict`` UAST constructor).
    * ``dict()`` -> ``{}``.
    """

    def __init__(self) -> None:
        self.diagnostics: List[str] = []

    def visit_Call(self, node: ast.Call) -> ast.AST:
        self.generic_visit(node)
        if isinstance(node.func, ast.Attribute) and node.func.attr == "get":
            base = node.func.value
            args = node.args
            if len(args) == 1:
                return ast.Subscript(value=base, slice=self._slice(args[0]), ctx=ast.Load())
            if len(args) == 2:
                self.diagnostics.append(
                    "dict.get(key, default) default argument dropped; "
                    "HIN dict_lookup returns Null for missing keys."
                )
                return ast.Subscript(value=base, slice=self._slice(args[0]), ctx=ast.Load())
        if isinstance(node.func, ast.Name) and node.func.id == "dict":
            if not node.args and not node.keywords:
                return ast.Dict(keys=[], values=[])
            if node.keywords and not node.args:
                keys = [ast.Constant(k.arg) for k in node.keywords]
                values = [k.value for k in node.keywords]
                return ast.Dict(keys=keys, values=values)
        return node

    @staticmethod
    def _slice(expr: ast.expr) -> ast.expr:
        # Python 3.9+ uses ast.Index internally; unwrap if necessary.
        if isinstance(expr, ast.Index):  # type: ignore[attr-defined]
            return expr.value  # type: ignore[attr-defined]
        return expr


class FallbackManager:
    """Apply tiered remediation before materializing files to disk.

    The three tiers are:

    1. Safe Type Degradation: SMT UNSAT caused by raw pointer memory
       alignment/layout is resolved by degrading the offending FFI contract to
       a wrapped, safe byte-buffer transfer (``Vec<u8>`` / ``SerializationBuffer``).
    2. Structural Sub-Graph Pruning: GoI non-nilpotency is resolved by
       identifying cyclic proof-net edges and replacing the corresponding async
       channels with thread-safe blocking channels (resetting those ``sigma``
       entries so the resolvent becomes nilpotent).
    3. Interactive Blueprint Clarification: SMT UNSAT on fundamental business
       logic or import visibility is not auto-repairable; materialization is
       aborted and a precise diagnostic report is emitted.

    In addition, the manager performs *proactive* AST healing for unsupported
    Python idioms such as ``dict.get()`` so that Tier-1 ``rust_hin`` routing is
    preferred over a CPython fallback.
    """

    def __init__(self) -> None:
        self.patches: List[Dict[str, Any]] = []
        self.diagnostics: List[str] = []
        self.last_level: int | None = None

    def remediate_collection_ast(
        self, source_text: str
    ) -> Tuple[bool, str, List[str]]:
        """Repair collection idioms in Python source and return the healed text.

        Returns ``(changed, new_source, diagnostics)``.
        """
        try:
            tree = ast.parse(source_text)
        except SyntaxError:
            return False, source_text, ["Source text is not valid Python; AST healing skipped."]

        repairer = _CollectionAstRepairer()
        new_tree = ast.fix_missing_locations(repairer.visit(tree))
        new_source = ast.unparse(new_tree)
        changed = new_source != source_text
        if changed:
            self.patches.append(
                {
                    "target_node_id": "python_ast",
                    "replacement_type": "hin_collection_subscript",
                    "purpose": "dict.get() -> dict[key] for HIN KeyLookup",
                }
            )
        self.diagnostics.extend(repairer.diagnostics)
        return changed, new_source, repairer.diagnostics

    def remediate_uast_expressions(
        self, uast: Dict[str, Any]
    ) -> Tuple[bool, Dict[str, Any], List[str]]:
        """Apply e-graph equality saturation to UAST expression nodes.

        This wraps the native ``repair_uast_expression`` proof engine so that
        arithmetic and algebraic sub-expressions are minimized before lowering
        to the HIN arena.  Unsupported expression fragments are left unchanged.
        """
        try:
            from aero_forge._native import repair_uast_expression
        except Exception as exc:
            return False, uast, [f"Native repair engine unavailable: {exc}"]

        def _repair(node: Any) -> Any:
            if isinstance(node, list):
                return [_repair(item) for item in node]
            if not isinstance(node, dict):
                return node
            node_type = node.get("type", "")
            if node_type in {"literal", "reference", "call", "binop", "unaryop"}:
                try:
                    rewritten = repair_uast_expression(json.dumps(node))
                    return json.loads(rewritten)
                except Exception:
                    pass
            return {k: _repair(v) for k, v in node.items()}

        new_uast = _repair(copy.deepcopy(uast))
        changed = new_uast != uast
        if changed:
            self.patches.append(
                {
                    "target_node_id": "uast",
                    "replacement_type": "egraph_minimized_expression",
                    "purpose": "equality saturation rewrite",
                }
            )
        return changed, new_uast, []

    def remediate_smt_unsat(
        self,
        payload: Dict[str, Any],
        trace: str,
    ) -> Tuple[bool, Dict[str, Any]]:
        """Attempt to remediate an SMT UNSAT failure.

        Returns ``(success, remediated_payload)``. ``success`` is ``True`` when
        the payload was modified so it has a chance of becoming SAT on the next
        SMT call. ``False`` indicates a Level-3 (unrecoverable) diagnostic.
        """
        lower = trace.lower()
        payload = copy.deepcopy(payload)

        # Level 1: raw pointer / FFI layout alignment failures.
        if any(k in lower for k in ("align", "offset", "layout", "raw pointer", "ffi")):
            self.last_level = 1
            remediated = self._degrade_raw_pointer_ffi(payload)
            return True, remediated

        # Level 3: fundamental business logic / import visibility failures.
        if any(k in lower for k in ("import", "business", "symbol", "visibility", "unreachable")):
            self.last_level = 3
            self.diagnostics.append(
                f"Level-3 abort: SMT UNSAT core indicates a fundamental "
                f"constraint violation that cannot be auto-remediated. Trace: {trace!r}"
            )
            return False, payload

        # No recognized remediation path; treat as Level-3.
        self.last_level = 3
        self.diagnostics.append(f"Level-3 abort: unrecognized SMT UNSAT trace: {trace!r}")
        return False, payload

    def _degrade_raw_pointer_ffi(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Replace raw pointer FFI layout constraints with safe byte-buffer wrappers."""
        original_constraints = payload.get("constraints", [])
        holes_with_type_constraint: set = set()
        for c in original_constraints:
            if "target_language" in c and c.get("source_hole"):
                holes_with_type_constraint.add(c["source_hole"])

        new_constraints: List[Dict[str, Any]] = []
        for c in original_constraints:
            ffi_layout = c.get("ffi_layout")
            if ffi_layout is not None:
                struct = ffi_layout.get("struct", "_")
                field = ffi_layout.get("field", "_")
                self.patches.append(
                    {
                        "target_node_id": f"{struct}_{field}",
                        "replacement_type": "Vec<u8>",
                        "wrapped_type": "SerializationBuffer",
                        "purpose": "safe byte-buffer transfer",
                    }
                )
                # Drop the offending FFI layout constraint.  Only force the hole to
                # the Python byte-buffer domain if there is no existing type
                # constraint, avoiding a fresh SAT conflict.
                if c.get("source_hole") and c["source_hole"] not in holes_with_type_constraint:
                    new_constraints.append(
                        {
                            "source_hole": c["source_hole"],
                            "target_language": "python",
                        }
                    )
            else:
                new_constraints.append(c)
        payload["constraints"] = new_constraints
        payload["fallback_patches"] = self.patches
        return payload

    def remediate_goi_non_nilpotent(
        self,
        payload: Dict[str, Any],
    ) -> Tuple[bool, Dict[str, Any]]:
        """Attempt to break cycles in the GoI proof net.

        Identifies cyclic off-diagonal M pairs and cuts the corresponding sigma
        entries, conceptually replacing cyclic async channels with blocking
        channels. Returns ``(success, remediated_payload)``.
        """
        payload = copy.deepcopy(payload)
        dim = payload.get("goi_dim", 2)
        sigma = list(payload.get("goi_sigma", [0.0] * (dim * dim)))
        m = payload.get("goi_m", [0.0] * (dim * dim))

        changed = False
        for i in range(dim):
            for j in range(dim):
                idx = i * dim + j
                # Self-loop in M: remove the node's cut entry.
                if i == j and m[idx] != 0.0:
                    sigma[idx] = 0.0
                    changed = True
                    self.patches.append(
                        {
                            "target_node_id": f"goi_node_{i}",
                            "replacement_type": "std::sync::mpsc::channel",
                            "wrapped_type": "DeadlockFreeChannel",
                            "purpose": "replace cyclic async channel with blocking channel",
                        }
                    )
                # Symmetric off-diagonal M pair indicates a 2-cycle. Remove both
                # the cross-cut entries and the diagonal cut entries that feed it.
                elif i < j and m[idx] != 0.0 and m[j * dim + i] != 0.0:
                    sigma[idx] = 0.0
                    sigma[j * dim + i] = 0.0
                    sigma[i * dim + i] = 0.0
                    sigma[j * dim + j] = 0.0
                    changed = True
                    self.patches.append(
                        {
                            "target_node_id": f"goi_edge_{i}_{j}",
                            "replacement_type": "std::sync::mpsc::channel",
                            "wrapped_type": "DeadlockFreeChannel",
                            "purpose": "replace cyclic async channel with blocking channel",
                        }
                    )

        if changed:
            self.last_level = 2
            payload["goi_sigma"] = sigma
            payload["fallback_patches"] = self.patches
            return True, payload

        self.last_level = 3
        self.diagnostics.append(
            "Level-3 abort: GoI proof net remains non-nilpotent and no cyclic "
            "edges could be pruned."
        )
        return False, payload

    def remediate_return_type_mismatch(
        self,
        source: str,
        symbol: str,
        declared: Optional[str],
        realized: Optional[str],
    ) -> str:
        """Trigger a return-statement rewrite when SMT proves a numeric cast is sufficient.

        Falls back to a signature rewrite (returning the body type as the new
        declared type) when the two types are not unifiable via a cast.
        """
        from aero_forge.builder.proactive_synthesis import (
            CoreVerificationPipeline,
            ReturnTypeUnificationError,
        )

        if declared is None or realized is None:
            return source
        try:
            if CoreVerificationPipeline.unifiable(declared, realized):
                repaired = _ReturnTypeRepairer.repair(source, symbol, declared)
                if repaired != source:
                    self.patches.append(
                        {
                            "target_node_id": symbol,
                            "replacement_type": "return_cast",
                            "declared": declared,
                            "realized": realized,
                            "purpose": "cast return expression to declared return type",
                        }
                    )
                return repaired
        except ReturnTypeUnificationError:
            pass
        # Non-unifiable: rewrite the function signature to the realized type.
        if CoreVerificationPipeline._rust_function_signature(source, symbol)[1] is not None:
            return self._rewrite_signature_to_realized(source, symbol, realized)
        return source

    def _rewrite_signature_to_realized(self, source: str, symbol: str, realized: str) -> str:
        """Replace a Rust function's declared return type with *realized*."""
        from aero_forge.builder.proactive_synthesis import CoreVerificationPipeline

        declared, body, start, end = CoreVerificationPipeline._rust_function_signature(source, symbol)
        if body is None:
            return source
        # Find the return-type arrow in the header and overwrite it.
        header = source[:start]
        new_header = re.sub(
            r"(fn\s+" + re.escape(symbol) + r"\s*\([^)]*\)\s*)->\s*[^\{\n]+",
            r"\1-> " + realized,
            header,
            count=1,
        )
        if new_header != header:
            self.patches.append(
                {
                    "target_node_id": symbol,
                    "replacement_type": "signature_alignment",
                    "declared": declared,
                    "realized": realized,
                    "purpose": "align function signature with realized return type",
                }
            )
            return new_header + body + source[end:]
        return source

    def diagnostic_report(self) -> str:
        """Return a formatted diagnostic report of the last remediation attempt."""
        lines = [f"Fallback level: {self.last_level}"]
        if self.patches:
            lines.append("Applied patches:")
            for p in self.patches:
                lines.append(f"  - {p}")
        if self.diagnostics:
            lines.append("Diagnostics:")
            for d in self.diagnostics:
                lines.append(f"  - {d}")
        return "\n".join(lines)

    def clear(self) -> None:
        """Reset the manager state."""
        self.patches = []
        self.diagnostics = []
        self.last_level = None
