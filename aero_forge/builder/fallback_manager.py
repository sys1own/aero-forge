"""Three-tier fallback remediation for the proactive polyglot build pipeline."""

from __future__ import annotations

import copy
from typing import Any, Dict, List, Tuple


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
    """

    def __init__(self) -> None:
        self.patches: List[Dict[str, Any]] = []
        self.diagnostics: List[str] = []
        self.last_level: int | None = None

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
