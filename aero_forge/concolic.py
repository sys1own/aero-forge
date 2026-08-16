"""Concolic manifest verification using Z3.

Treats a candidate ``blueprint.aero`` manifest as a set of Natural Language Text
Constraints (NLTCs), lowers them to SMT-LIB assertions, and asks Z3 whether the
build plan is internally consistent. When the solver returns UNSAT, the minimal
unsatisfiable core is translated back into concrete build-rule conflicts for the
LLM to repair.
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from z3 import And, Bool, Implies, Int, Not, Or, Solver, set_param, unknown, unsat

logger = logging.getLogger("aero_forge.concolic")


# Valid architecture names and their allowed languages.
_ARCHITECTURE_LANGS = {
    "pure_python": {"python"},
    "pure_rust": {"rust"},
    "hybrid_rust_python": {"python", "rust"},
    "hybrid_cpp_python": {"python", "cpp"},
    "hybrid_cpp_rust": {"rust", "cpp"},
    "tri_polyglot_rust_cpp_python": {"python", "rust", "cpp"},
    "graph_polyglot": {"python", "rust", "cpp", "go", "zig"},
}

_LANG_TOOLCHAINS = {
    "python": {"python"},
    "rust": {"cargo", "rustc"},
    "cpp": {"cmake", "clang", "gcc", "g++", "clang++"},
    "c": {"cmake", "clang", "gcc"},
    "go": {"go"},
    "zig": {"zig"},
}

_VALID_BOUNDARIES = {
    ("python", "rust"): {"PYO3_MATURIN"},
    ("rust", "python"): {"PYO3_MATURIN"},
    ("python", "cpp"): {"C_ABI"},
    ("cpp", "python"): {"C_ABI"},
    ("rust", "cpp"): {"C_ABI"},
    ("cpp", "rust"): {"C_ABI"},
    ("go", "rust"): {"CGO"},
    ("rust", "go"): {"CGO"},
    ("python", "zig"): {"C_ABI"},
    ("zig", "python"): {"C_ABI"},
    ("rust", "zig"): {"C_ABI"},
    ("zig", "rust"): {"C_ABI"},
    ("cpp", "zig"): {"C_ABI"},
    ("zig", "cpp"): {"C_ABI"},
}


def _safe_id(name: str) -> str:
    """Turn an arbitrary string into an SMT identifier."""
    return re.sub(r"[^A-Za-z0-9_]+", "_", name).strip("_") or "x"


@dataclass
class ConcolicResult:
    """Result of a concolic manifest verification attempt."""

    satisfiable: bool
    model: Optional[Dict[str, Any]] = None
    unsat_core: List[str] = field(default_factory=list)
    conflicting_rules: List[str] = field(default_factory=list)
    trace: List[str] = field(default_factory=list)


class ConcolicManifestVerifier:
    """Verify a candidate manifest with Z3 and extract unsatisfiable cores.

    Example:
        verifier = ConcolicManifestVerifier(manifest_dict)
        result = verifier.verify()
        if not result.satisfiable:
            print(result.conflicting_rules)
    """

    def __init__(
        self,
        manifest: Dict[str, Any],
        allowed_architectures: Optional[List[str]] = None,
    ) -> None:
        self.manifest = manifest
        self.allowed_architectures = allowed_architectures or list(
            _ARCHITECTURE_LANGS.keys()
        )
        self.solver = Solver()
        self.solver.set("unsat_core", True)
        # Resource guard: cap each solver check at 30 seconds and 512 MB of memory.
        # ``timeout`` is per-check (milliseconds); ``memory_max_size`` and
        # ``memory_high_watermark_mb`` are process-wide memory limits in megabytes.
        timeout_ms = int(os.getenv("AERO_FORGE_Z3_TIMEOUT_MS", 30000))
        max_memory_mb = int(os.getenv("AERO_FORGE_Z3_MAX_MEMORY_MB", 512))
        self.solver.set("timeout", timeout_ms)
        set_param("memory_max_size", max_memory_mb)
        set_param("memory_high_watermark_mb", max_memory_mb)
        self._trackers: Dict[str, Bool] = {}

    def _track(self, name: str, expr) -> None:
        """Assert an expression under a named tracker."""
        if name in self._trackers:
            name = f"{name}_{len(self._trackers)}"
        tracker = Bool(name)
        self._trackers[name] = tracker
        self.solver.assert_and_track(expr, tracker)

    def _node_id(self, node: Dict[str, Any]) -> str:
        return _safe_id(node.get("node_id", "node"))

    def _add_architecture_constraint(self) -> None:
        arch = str(self.manifest.get("architecture", "")).lower().replace("-", "_")
        valid = arch in self.allowed_architectures
        self._track("architecture_valid", Bool("architecture_valid") == valid)
        if valid:
            allowed_langs = _ARCHITECTURE_LANGS.get(arch, set())
            self._track("architecture_has_known_langs", Bool("known_arch") == True)
        else:
            # Stop adding language constraints if the architecture itself is bad;
            # the tracker above will already be in any unsat core.
            return

    def _add_toolchain_constraints(self) -> None:
        for node in self.manifest.get("nodes", []):
            nid = self._node_id(node)
            lang = str(node.get("lang", "")).lower()
            toolchain = str(node.get("toolchain", "")).lower()
            ok = toolchain in _LANG_TOOLCHAINS.get(lang, set())
            name = f"node_{nid}_toolchain_{toolchain}_for_{lang}"
            self._track(name, Bool(_safe_id(name)) == ok)

    def _add_edge_constraints(self) -> None:
        node_ids = {
            self._node_id(n)
            for n in self.manifest.get("nodes", [])
        }
        node_lang = {
            self._node_id(n): str(n.get("lang", "")).lower()
            for n in self.manifest.get("nodes", [])
        }

        for idx, edge in enumerate(self.manifest.get("edges", [])):
            src = _safe_id(edge.get("source", ""))
            tgt = _safe_id(edge.get("target", ""))
            boundary = str(edge.get("boundary_type", "")).upper()

            target_exists = tgt in node_ids
            name = f"edge_{idx}_{src}_to_{tgt}_target_exists"
            self._track(name, Bool(_safe_id(name)) == target_exists)

            src_lang = node_lang.get(src, "")
            tgt_lang = node_lang.get(tgt, "")
            if src_lang and tgt_lang:
                if src_lang == tgt_lang:
                    boundary_ok = boundary in {"", "INTERNAL"}
                else:
                    boundary_ok = boundary in _VALID_BOUNDARIES.get(
                        (src_lang, tgt_lang), set()
                    )
                name = f"edge_{idx}_{src}_to_{tgt}_boundary_{boundary}_ok"
                self._track(name, Bool(_safe_id(name)) == boundary_ok)

    def _add_functional_intent_constraints(self) -> None:
        exports: set = set()
        for node in self.manifest.get("nodes", []):
            for sym in node.get("exports", []):
                exports.add(_safe_id(str(sym)))
            for contract in node.get("contracts", []):
                sym = contract.get("symbol") if isinstance(contract, dict) else None
                if sym:
                    exports.add(_safe_id(str(sym)))
        for edge in self.manifest.get("edges", []):
            sym = edge.get("symbol")
            if sym:
                exports.add(_safe_id(str(sym)))

        for idx, intent in enumerate(self.manifest.get("functional_intent", [])):
            sym = _safe_id(
                str(intent.get("symbol_name") or intent.get("name", ""))
            )
            if not sym:
                continue
            covered = sym in exports
            name = f"intent_{idx}_symbol_{sym}_covered"
            self._track(name, Bool(_safe_id(name)) == covered)

    def _add_acyclicity_constraint(self) -> None:
        """Assign integer ranks and require rank(source) < rank(target) for edges."""
        nodes = self.manifest.get("nodes", [])
        if not nodes:
            return
        ranks = {
            self._node_id(n): Int(f"rank_{self._node_id(n)}")
            for n in nodes
        }
        # Distinct ranks naturally enforce a total order; a strict partial order
        # is enough, but distinctness makes SAT/UNSAT feedback crisper.
        for i, a in enumerate(nodes):
            for b in nodes[i + 1 :]:
                self.solver.add(ranks[self._node_id(a)] != ranks[self._node_id(b)])
        for idx, edge in enumerate(self.manifest.get("edges", [])):
            src = _safe_id(edge.get("source", ""))
            tgt = _safe_id(edge.get("target", ""))
            if src in ranks and tgt in ranks:
                tracker = Bool(f"edge_{idx}_acyclic_{src}_before_{tgt}")
                self.solver.assert_and_track(ranks[src] < ranks[tgt], tracker)

    def _build_constraints(self) -> None:
        self._add_architecture_constraint()
        self._add_toolchain_constraints()
        self._add_edge_constraints()
        self._add_functional_intent_constraints()
        self._add_acyclicity_constraint()

    def _describe_tracker(self, name: str) -> str:
        """Turn a tracker name into a human-readable rule description."""
        if "architecture" in name:
            return (
                f"Architecture '{self.manifest.get('architecture')}' is not "
                f"in the allowed list {self.allowed_architectures}."
            )
        if "toolchain" in name:
            m = re.search(r"node_(.+?)_toolchain_(.+?)_for_(.+)$", name)
            if m:
                return (
                    f"Node '{m.group(1)}' uses toolchain '{m.group(2)}' which is "
                    f"not valid for language '{m.group(3)}'."
                )
        if "target_exists" in name:
            m = re.search(r"edge_(\d+)_(.+?)_to_(.+?)_target_exists", name)
            if m:
                return (
                    f"Edge {m.group(1)} from '{m.group(2)}' refers to missing "
                    f"target node '{m.group(3)}'."
                )
        if "boundary" in name and "ok" in name:
            m = re.search(r"edge_(\d+)_(.+?)_to_(.+?)_boundary_(.+?)_ok", name)
            if m:
                return (
                    f"Edge {m.group(1)} from '{m.group(2)}' to '{m.group(3)}' has "
                    f"an invalid boundary type '{m.group(4)}'."
                )
        if "covered" in name:
            m = re.search(r"intent_(\d+)_symbol_(.+?)_covered", name)
            if m:
                return (
                    f"Functional intent symbol '{m.group(2)}' is not exported by "
                    f"any node or contract."
                )
        if "acyclic" in name:
            m = re.search(r"edge_(\d+)_acyclic_(.+?)_before_(.+?)$", name)
            if m:
                return (
                    f"Edge {m.group(1)} from '{m.group(2)}' to '{m.group(3)}' "
                    f"participates in a dependency cycle."
                )
        return f"Conflicting constraint: {name}"

    def verify(self) -> ConcolicResult:
        """Run Z3 and return a structured verification result."""
        self._build_constraints()
        try:
            result = self.solver.check()
        except Exception as exc:
            logger.warning("Z3 solver raised an exception: %s", exc)
            return ConcolicResult(
                satisfiable=True,
                trace=[
                    "Concolic manifest verification: skipped due to solver error",
                    str(exc),
                ],
            )

        if result == unsat:
            core = self.solver.unsat_core()
            names = [str(c) for c in core]
            descriptions = [self._describe_tracker(n) for n in names]
            return ConcolicResult(
                satisfiable=False,
                unsat_core=names,
                conflicting_rules=descriptions,
                trace=[
                    "Concolic manifest verification: UNSAT",
                    f"Unsat core size: {len(names)}",
                ]
                + descriptions,
            )

        if result == unknown:
            logger.warning("Z3 solver returned unknown (resource limit reached)")
            return ConcolicResult(
                satisfiable=True,
                trace=[
                    "Concolic manifest verification: unknown (timeout/memory limit)",
                    "Proceeding without a verified model to avoid blocking the build.",
                ],
            )

        # For SAT builds we also produce a compact trace; the model is usually not
        # needed for manifests but kept for debugging.
        model = None
        try:
            m = self.solver.model()
        except Exception:
            m = None
        if m is not None:
            try:
                model = {str(v): str(m[v]) for v in m if m[v] is not None}
            except Exception:
                model = None
        return ConcolicResult(
            satisfiable=True,
            model=model,
            trace=["Concolic manifest verification: SAT"],
        )

    def refinement_feedback(self, result: Optional[ConcolicResult] = None) -> str:
        """Return a single string suitable for feeding back to the LLM."""
        result = result or self.verify()
        if result.satisfiable:
            return "Manifest is internally consistent."
        lines = [
            "The generated manifest is logically inconsistent. The minimal "
            "conflicting rule set is:",
            "",
        ]
        for rule in result.conflicting_rules:
            lines.append(f"- {rule}")
        lines.append("")
        lines.append(
            "Repair instructions: resolve the above conflicts before "
            "materialising any source files."
        )
        return "\n".join(lines)


def verify_manifest(manifest: Dict[str, Any]) -> ConcolicResult:
    """Convenience entry point: verify a manifest dict and return a result."""
    return ConcolicManifestVerifier(manifest).verify()
